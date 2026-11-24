#include "cli.hpp"
#include "dataset.hpp"
#include "gpu.hpp"
#include "model.hpp"

#ifdef RWKV_HAVE_NCCL
#include <nccl.h>
#endif
#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <thread>

namespace rwkv {
namespace {
using Index = std::uint64_t;

struct Config {
  std::string input, dataset, mask, tokenizer, output;
  std::vector<int> devices;
  int context, batch, accum, head_chunk;
  Index steps, save_every, warmup, start_step, start_window, seed;
  bool optimized_wkv, checkpoint_activations;
  Adam adam;
  float final_lr;
};

std::vector<int> parse_devices(const std::string &text) {
  std::vector<int> result;
  for (std::size_t start = 0;;) {
    const auto end = text.find(',', start);
    const int id = integer<int>(text.substr(start, end - start));
    if (id < 0 || std::find(result.begin(), result.end(), id) != result.end())
      throw std::invalid_argument("device IDs must be non-negative and unique");
    result.push_back(id);
    if (end == std::string::npos)
      return result;
    start = end + 1;
  }
}

bool boolean(Options &options, const std::string &name, bool fallback) {
  const int i = options.number(name, int(fallback));
  if (i < 0 || i > 1)
    throw std::invalid_argument(name + " must be 0 or 1");
  return i != 0;
}

Config config(int argc, char **argv) {
  Options options(argc, argv, 2);
  Config c{};
  c.input = argv[1];
  c.dataset = options.take("--dataset");
  c.mask = options.take("--mask");
  c.tokenizer =
      options.take("--tokenizer", "data/tokenizer/rwkv_vocab_v20230424.txt");
  c.output = options.take("--out");
  c.devices = parse_devices(options.take("--devices", "0"));
  c.context = options.number("--ctx", 1024);
  c.batch = options.number("--batch", 1);
  c.accum = options.number("--accum", 1);
  c.head_chunk = options.number("--head-chunk", 128);
  c.steps = options.number<Index>("--steps", 0);
  c.save_every = options.number<Index>("--save-every", 0);
  c.warmup = options.number<Index>("--warmup", 0);
  c.start_step = options.number<Index>("--start-step", 0);
  c.start_window = options.number<Index>("--start-window", 0);
  c.seed = options.number<Index>("--seed", 1234);
  c.optimized_wkv = boolean(options, "--wkv-v3", false);
  c.checkpoint_activations = boolean(options, "--activation-checkpoint", true);
  c.adam = {float(options.real("--lr", 6e-4)),
            float(options.real("--beta1", .9)),
            float(options.real("--beta2", .99)),
            float(options.real("--adam-eps", 1e-18)),
            float(options.real("--weight-decay", 0)),
            float(options.real("--clip", 1))};
  c.final_lr = options.real("--lr-final", c.adam.lr);
  options.finish();
  if (c.dataset.empty() || c.output.empty() || !c.steps)
    throw std::invalid_argument(
        "--dataset, --out and a positive --steps are required");
  if (c.context < 16 || c.context % 16 || c.batch < 1 || c.batch > 65535 ||
      c.accum < 1 || c.head_chunk < 1 || c.start_step > UINT64_MAX - c.steps)
    throw std::invalid_argument(
        "invalid context, batch size, accumulation, head chunk or step range");
  const auto &a = c.adam;
  if (!std::isfinite(a.lr) || !std::isfinite(c.final_lr) ||
      !std::isfinite(a.eps) || !std::isfinite(a.decay) ||
      !std::isfinite(a.clip) || a.lr <= 0 || c.final_lr < 0 || a.beta1 < 0 ||
      a.beta1 >= 1 || a.beta2 < 0 || a.beta2 >= 1 || a.eps <= 0 ||
      a.decay < 0 || a.clip < 0)
    throw std::invalid_argument("invalid AdamW or learning-rate option");
  return c;
}

class Sampler {
public:
  Sampler(Index count, Index seed, Index position)
      : count_(count), seed_(seed), position_(position) {}
  Index next() {
    if (position_ == UINT64_MAX)
      throw std::runtime_error("dataset cursor overflow");
    const auto epoch = position_ / count_;
    if (epoch != epoch_) {
      epoch_ = epoch;
      std::mt19937_64 rng(seed_ + epoch_);
      shift_ = rng() % count_;
      stride_ = 1 + rng() % count_;
      while (std::gcd(count_, stride_) != 1)
        stride_ = stride_ == count_ ? 1 : stride_ + 1;
    }
    return (static_cast<unsigned __int128>(position_++ % count_) * stride_ +
            shift_) %
           count_;
  }
  Index position() const { return position_; }

private:
  Index count_, seed_, position_;
  Index epoch_ = UINT64_MAX, stride_ = 1, shift_ = 0;
};

#ifdef RWKV_HAVE_NCCL
void nccl_ok(ncclResult_t status) {
  if (status != ncclSuccess)
    throw std::runtime_error(std::string("NCCL: ") +
                             ncclGetErrorString(status));
}
#endif

class DataParallel {
public:
  explicit DataParallel(const std::vector<int> &devices)
#ifdef RWKV_HAVE_NCCL
      : devices_(devices)
#endif
  {
    if (devices.size() == 1)
      return;
#ifdef RWKV_HAVE_NCCL
    comms_.resize(devices.size(), nullptr);
    const auto result =
        ncclCommInitAll(comms_.data(), devices.size(), devices.data());
    if (result != ncclSuccess) {
      abort();
      nccl_ok(result);
    }
#else
    throw std::runtime_error("multi-GPU training requires a build with NCCL");
#endif
  }

  ~DataParallel() {
#ifdef RWKV_HAVE_NCCL
    if (std::uncaught_exceptions())
      abort();
    else
      for (auto comm : comms_)
        ncclCommDestroy(comm);
#endif
  }

  void sum(const std::vector<std::unique_ptr<Model>> &models) {
#ifdef RWKV_HAVE_NCCL
    if (comms_.empty())
      return;
    nccl_ok(ncclGroupStart());
    ncclResult_t first_error = ncclSuccess;
    cudaError_t cuda_status = cudaSuccess;
    for (std::size_t i = 0; i < models.size(); ++i) {
      cuda_status = cudaSetDevice(devices_[i]);
      if (cuda_status != cudaSuccess)
        break;
      auto &m = *models[i];
      const auto status =
          ncclAllReduce(m.gradients(), m.gradients(), m.parameter_count(),
                        ncclFloat32, ncclSum, comms_[i], m.stream());
      if (first_error == ncclSuccess)
        first_error = status;
    }
    const auto group_status = ncclGroupEnd();
    CUDA(cuda_status);
    nccl_ok(first_error);
    nccl_ok(group_status);
#else
    (void)models;
#endif
  }

  void synchronize(const std::vector<std::unique_ptr<Model>> &models) {
#ifdef RWKV_HAVE_NCCL
    if (!comms_.empty()) {
      bool pending;
      do {
        pending = false;
        for (std::size_t i = 0; i < models.size(); ++i) {
          ncclResult_t status;
          nccl_ok(ncclCommGetAsyncError(comms_[i], &status));
          nccl_ok(status);
          models[i]->activate();
          const auto ready = cudaStreamQuery(models[i]->stream());
          if (ready == cudaErrorNotReady)
            pending = true;
          else
            CUDA(ready);
        }
        if (pending)
          std::this_thread::sleep_for(std::chrono::microseconds(100));
      } while (pending);
      return;
    }
#endif
    models.front()->synchronize();
  }

private:
#ifdef RWKV_HAVE_NCCL
  void abort() {
    for (auto comm : comms_)
      if (comm)
        ncclCommAbort(comm);
  }
  std::vector<int> devices_;
  std::vector<ncclComm_t> comms_;
#endif
};

void fill_batch(MicroBatch &batch, std::vector<std::int64_t> &labels,
                const Dataset &data, Sampler &sampler, const Config &c) {
  const int rows = c.batch * c.context;
  batch.input.resize(rows);
  batch.lanes = 0;
  batch.time = 0;
  batch.active_rows.clear();
  batch.labels.clear();
  batch.losses.clear();
  batch.active_rows.reserve(rows);
  batch.labels.reserve(rows);
  batch.losses.reserve(rows);
  for (int lane = 0; lane < c.batch; ++lane) {
    const int off = batch.lanes * c.context;
    data.sample(sampler.next(), batch.input.data() + off, labels.data());
    int last = 0;
    for (int t = 0; t < c.context; ++t) {
      if (labels[t] >= 0) {
        batch.active_rows.push_back(off + t);
        batch.labels.push_back(labels[t]);
        last = t + 1;
      }
    }
    if (last) {
      ++batch.lanes;
      batch.time = std::max(batch.time, last);
    }
  }
  batch.time = (batch.time + 15) / 16 * 16;
  // Causality allows us to omit the suffix after every lane's last target.
  if (batch.time < c.context) {
    for (int lane = 1; lane < batch.lanes; ++lane)
      std::memmove(batch.input.data() + lane * batch.time,
                   batch.input.data() + lane * c.context,
                   batch.time * sizeof(int));
    for (int &row : batch.active_rows)
      row -= (row / c.context) * (c.context - batch.time);
  }
}

float learning_rate(const Config &c, Index i) {
  if (i < c.warmup)
    return c.adam.lr * (0.01 + 0.99 * double(i) / c.warmup);
  const auto end = c.start_step + c.steps - 1;
  if (end <= c.warmup)
    return c.adam.lr;
  const double progress = double(i - c.warmup) / (end - c.warmup);
  return c.final_lr + (c.adam.lr - c.final_lr) *
                          (1 + std::cos(std::acos(-1.) * progress)) / 2;
}

void run(const Config &c) {
  const Checkpoint input(c.input);
  const auto shape = input.shape();
  const auto layout = weight_layout(shape);
  const auto rows = std::uint64_t(c.batch) * c.context;
  // WKV uses int indices. TimeMix backward tiles eight rows per grid.y block.
  if (rows > INT32_MAX / std::uint64_t(shape.channels) || rows > 65535 * 8ULL)
    throw std::invalid_argument(
        "batch and context exceed the CUDA kernel indexing range");
  const Dataset data(c.dataset, c.mask, c.tokenizer, c.context, shape.vocab);
  int device_count;
  CUDA(cudaGetDeviceCount(&device_count));
  for (int id : c.devices) {
    if (id >= device_count)
      throw std::invalid_argument("CUDA device is not visible: " +
                                  std::to_string(id));
    cudaDeviceProp prop;
    CUDA(cudaGetDeviceProperties(&prop, id));
    if (prop.major < 8)
      throw std::invalid_argument(
          "BF16 training requires CUDA compute capability 8.0 or newer");
  }
  std::filesystem::create_directories(c.output);
  const auto output_path = [&](Index step) {
    return std::filesystem::path(c.output) /
           ("rwkv-" + std::to_string(step) + ".pth");
  };
  const auto validate_output = [&](Index step) {
    if (std::filesystem::exists(output_path(step)))
      throw std::runtime_error("checkpoint already exists: " +
                               output_path(step).string());
  };
  validate_output(c.start_step + c.steps);

  std::vector<MicroBatch> micro(std::size_t(c.accum) * c.devices.size());
  std::vector<std::unique_ptr<Model>> models;
  DataParallel dp(c.devices);
  for (int id : c.devices)
    models.push_back(std::make_unique<Model>(id, input, shape, layout,
                                             c.head_chunk, c.optimized_wkv,
                                             c.checkpoint_activations));
  Sampler sampler(data.size(), c.seed, c.start_window);
  std::cout << "parameters=" << models.front()->parameter_count()
            << " windows=" << data.size() << " devices=" << models.size()
            << '\n';
  std::vector<std::int64_t> labels(c.context);
  for (Index completed = 0; completed < c.steps;) {
    const auto began = std::chrono::steady_clock::now();
    const auto step = c.start_step + completed + 1;
    const bool save =
        (c.save_every && step % c.save_every == 0) || completed + 1 == c.steps;
    if (save)
      validate_output(step);
    const auto initial_window = sampler.position();
    std::size_t valid = 0;
    do {
      for (auto &batch : micro) {
        fill_batch(batch, labels, data, sampler, c);
        valid += batch.labels.size();
      }
      // Cover a full epoch even when this search starts partway through an
      // epoch.
      if (!valid && (sampler.position() - initial_window) / data.size() >= 2)
        throw std::runtime_error("dataset has no unmasked next-token targets");
    } while (!valid);

    for (auto &model : models)
      model->clear_gradients();
    const float inv_tokens = 1.f / valid;
    for (std::size_t i = 0; i < micro.size(); ++i)
      models[i % models.size()]->accumulate(micro[i], inv_tokens);
    dp.sum(models);
    dp.synchronize(models);

    double loss = 0;
    for (const auto &batch : micro)
      loss = std::accumulate(batch.losses.begin(), batch.losses.end(), loss);
    loss /= valid;
    if (!std::isfinite(loss))
      throw std::runtime_error("non-finite masked loss");

    auto opt = c.adam;
    opt.lr = learning_rate(c, step - 1);
    const float scale = models.front()->gradient_scale(opt.clip);
    for (auto &model : models)
      model->update(opt, completed + 1, scale);
    dp.synchronize(models);
    ++completed;
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - began)
            .count();
    std::cout << std::setprecision(7) << "step=" << step << " loss=" << loss
              << " lr=" << opt.lr << " targets=" << valid
              << " targets/s=" << valid / seconds
              << " next_window=" << sampler.position() << std::endl;
    if (save) {
      const auto path = output_path(step).string();
      models.front()->save(path);
      std::cout << "Wrote " << path << std::endl;
    }
  }
}
} // namespace
} // namespace rwkv

int main(int argc, char **argv) try {
  if (argc == 1 || std::string(argv[1]) == "--help") {
    std::cout << "Usage: rwkv7_train INPUT.pth --dataset PATH --out DIR "
                 "--steps N [options]\n"
                 "  Data: --mask PATH --tokenizer PATH --ctx N --seed N "
                 "--start-window N\n"
                 "  CUDA: --devices 0,1 --batch N --accum N --head-chunk N\n"
                 "        --activation-checkpoint 0|1 --wkv-v3 0|1\n"
                 "  AdamW: --lr X --lr-final X --warmup N --beta1 X --beta2 X\n"
                 "         --weight-decay X --adam-eps X --clip X\n"
                 "  Weights: --save-every N --start-step N\n";
    return argc == 1 ? 1 : 0;
  }
  rwkv::run(rwkv::config(argc, argv));
  return 0;
} catch (const std::exception &error) {
  std::cerr << "rwkv7_train: " << error.what() << '\n';
  return 1;
}
