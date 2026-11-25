#include "cli.hpp"
#include "dataset.hpp"
#include "gpu.hpp"
#include "model.hpp"

#ifdef RWKV_HAVE_NCCL
#include <nccl.h>
#endif
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <sys/ioctl.h>
#include <thread>
#include <type_traits>
#include <unistd.h>

namespace rwkv {
namespace {
using Index = std::uint64_t;

struct Config {
  std::string input, dataset, mask, tokenizer, output;
  int device;
  int context, batch, accum, head_chunk;
  Index steps, epoch_save, bucket_mib, warmup, start_step, start_window, seed;
  bool optimized_wkv, checkpoint_activations;
  Adam adam;
  double final_lr;
};

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
  c.device = options.number("--devices", 0);
  c.context = options.number("--ctx", 1024);
  c.batch = options.number("--batch", 1);
  c.accum = options.number("--accum", 1);
  c.head_chunk = options.number("--head-chunk", 128);
  const auto steps = options.take("--steps");
  c.steps = steps.empty() ? 0 : integer<Index>(steps);
  c.epoch_save = options.number<Index>("--epoch-save", 1);
  c.bucket_mib = options.number<Index>("--nccl-bucket-mb", 25);
  if (!c.bucket_mib || c.bucket_mib > SIZE_MAX / (1024 * 1024))
    throw std::invalid_argument("invalid NCCL bucket size");
  c.warmup = options.number<Index>("--warmup", 0);
  c.start_step = options.number<Index>("--start-step", 0);
  c.start_window = options.number<Index>("--start-window", 0);
  c.seed = options.number<Index>("--seed", 1234);
  c.optimized_wkv = boolean(options, "--wkv-v3", false);
  c.checkpoint_activations = boolean(options, "--activation-checkpoint", true);
  c.adam = {options.real("--lr", 6e-4),
            options.real("--beta1", .9),
            options.real("--beta2", .99),
            options.real("--adam-eps", 1e-18),
            options.real("--weight-decay", 0),
            float(options.real("--clip", 1))};
  c.final_lr = options.real("--lr-final", c.adam.lr);
  options.finish();
  if (c.dataset.empty() || c.output.empty())
    throw std::invalid_argument("--dataset and --out are required");
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

int env_rank(const char *name, int fallback) {
  const char *value = std::getenv(name);
  return value ? integer<int>(value) : fallback;
}

class DataParallel {
public:
  DataParallel(int rank, int world, cudaStream_t compute) : compute_(compute) {
    int least, highest;
    CUDA(cudaDeviceGetStreamPriorityRange(&least, &highest));
    CUDA(cudaStreamCreateWithPriority(&comm_stream_, cudaStreamNonBlocking,
                                      highest));
    CUDA(cudaEventCreateWithFlags(&ready_, cudaEventDisableTiming));
    CUDA(cudaEventCreateWithFlags(&done_, cudaEventDisableTiming));
    if (world == 1)
      return;
#ifdef RWKV_HAVE_NCCL
    const auto dir = std::getenv("RWKV_RENDEZVOUS");
    if (!dir)
      throw std::runtime_error("missing RWKV_RENDEZVOUS");
    const auto path = std::filesystem::path(dir) / "comm.id";
    ncclUniqueId id;
    if (!rank) {
      nccl_ok(ncclGetUniqueId(&id));
      std::ofstream out(path.string() + ".tmp", std::ios::binary);
      out.write(reinterpret_cast<const char *>(&id), sizeof(id));
      out.close();
      if (!out)
        throw std::runtime_error("cannot publish communicator ID");
      std::filesystem::rename(path.string() + ".tmp", path);
    } else {
      const auto deadline =
          std::chrono::steady_clock::now() + std::chrono::minutes(10);
      while (!std::filesystem::exists(path)) {
        if (std::chrono::steady_clock::now() > deadline)
          throw std::runtime_error("communicator rendezvous timed out");
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
      }
      std::ifstream in(path, std::ios::binary);
      if (!in.read(reinterpret_cast<char *>(&id), sizeof(id)))
        throw std::runtime_error("cannot read communicator ID");
    }
    nccl_ok(ncclCommInitRank(&comm_, world, id, rank));
#else
    throw std::runtime_error("distributed training requires NCCL or RCCL");
#endif
  }

  ~DataParallel() {
#ifdef RWKV_HAVE_NCCL
    if (comm_) {
      if (std::uncaught_exceptions())
        ncclCommAbort(comm_);
      else
        ncclCommDestroy(comm_);
    }
#endif
    (void)cudaEventDestroy(done_);
    (void)cudaEventDestroy(ready_);
    (void)cudaStreamDestroy(comm_stream_);
  }

  void sum(Model &model, std::size_t bucket) {
#ifdef RWKV_HAVE_NCCL
    if (!comm_)
      return;
    begin();
    for (std::size_t off = 0; off < model.parameter_count();) {
      const auto n = std::min(bucket, model.parameter_count() - off);
      float *g = model.gradients() + off;
      nccl_ok(
          ncclAllReduce(g, g, n, ncclFloat32, ncclSum, comm_, comm_stream_));
      off += n;
    }
    end();
#endif
  }

  // Called at step boundaries for the global target count, loss and VRAM
  // checks.
  enum class Reduction { sum, min, max };
  template <class T> T scalar(T value, Reduction op = Reduction::sum) {
#ifdef RWKV_HAVE_NCCL
    if (comm_) {
      static_assert(sizeof(T) == 8);
      auto p = scalar_.get();
      std::memcpy(host_scalar_.data(), &value, sizeof(T));
      CUDA(cudaMemcpyAsync(p, host_scalar_.data(), sizeof(T),
                           cudaMemcpyHostToDevice, compute_));
      begin();
      nccl_ok(ncclAllReduce(
          p, p, 1, std::is_same_v<T, double> ? ncclFloat64 : ncclUint64,
          op == Reduction::sum   ? ncclSum
          : op == Reduction::min ? ncclMin
                                 : ncclMax,
          comm_, comm_stream_));
      end();
      CUDA(cudaMemcpyAsync(host_scalar_.data(), p, sizeof(T),
                           cudaMemcpyDeviceToHost, compute_));
      CUDA(cudaStreamSynchronize(compute_));
      std::memcpy(&value, host_scalar_.data(), sizeof(T));
    }
#endif
    return value;
  }

private:
  void begin() {
    CUDA(cudaEventRecord(ready_, compute_));
    CUDA(cudaStreamWaitEvent(comm_stream_, ready_, 0));
  }
  void end() {
    CUDA(cudaEventRecord(done_, comm_stream_));
    CUDA(cudaStreamWaitEvent(compute_, done_, 0));
  }
  cudaStream_t compute_, comm_stream_{};
  cudaEvent_t ready_{}, done_{};
  Buffer<Index> scalar_{1};
  std::vector<Index, PinnedAllocator<Index>> host_scalar_{0};
#ifdef RWKV_HAVE_NCCL
  ncclComm_t comm_ = nullptr;
#endif
};

void fill_local(MicroBatch &batch, const Dataset &data, Sampler &sampler,
                const Config &c, Index count, int rank, int world,
                std::vector<std::int64_t> &labels) {
  batch.labels.clear();
  batch.active_rows.clear();
  batch.losses.clear();
  std::fill(batch.input.begin(), batch.input.end(), 0);
  // Every process advances the same global cursor. Only the owner reads the
  // window.
  for (Index i = 0; i < count; ++i) {
    const Index window = sampler.next();
    if (i % world != Index(rank))
      continue;
    const int offset = int(i / world) * c.context;
    data.sample(window, batch.input.data() + offset, labels.data());
    for (int t = 0; t < c.context; ++t) {
      if (labels[t] < 0)
        continue;
      batch.active_rows.push_back(offset + t);
      batch.labels.push_back(labels[t]);
    }
  }
}

double learning_rate(const Config &c, Index i) {
  if (i < c.warmup)
    return c.adam.lr * (0.01 + 0.99 * double(i) / c.warmup);
  const auto end = c.start_step + c.steps - 1;
  if (end <= c.warmup)
    return c.adam.lr;
  const double progress = double(i - c.warmup) / (end - c.warmup);
  return c.final_lr + (c.adam.lr - c.final_lr) *
                          (1 + std::cos(std::acos(-1.) * progress)) / 2;
}

void run(Config c) {
  const int rank = env_rank("LOCAL_RANK", 0), world = env_rank("WORLD_SIZE", 1);
  if (world < 1 || rank < 0 || rank >= world || (world > 1 && c.device != 0) ||
      c.device < 0)
    throw std::invalid_argument("use the platform training script for one process per visible GPU");
  const int device = world == 1 ? c.device : rank;
  int device_count;
  CUDA(cudaGetDeviceCount(&device_count));
  if (world > device_count || device >= device_count)
    throw std::invalid_argument("requested GCDs exceed visible devices");
  CUDA(cudaSetDevice(device));
#ifndef RWKV_ROCM
  cudaDeviceProp prop;
  CUDA(cudaGetDeviceProperties(&prop, device));
  if (prop.major < 8)
    throw std::invalid_argument(
        "BF16 training requires CUDA compute capability 8.0 or newer");
#endif
  const Checkpoint input(c.input);
  const auto shape = input.shape();
  const auto layout = weight_layout(shape);
  const auto rows = std::uint64_t(c.batch) * c.context;
  // WKV uses int indices. TimeMix backward tiles eight rows per grid.y block.
  if (rows > INT32_MAX / std::uint64_t(shape.channels) || rows > 65535 * 8ULL)
    throw std::invalid_argument(
        "batch and context exceed the GPU kernel indexing range");
  const Dataset data(c.dataset, c.mask, c.tokenizer, c.context, shape.vocab);
  constexpr Index epoch_windows = 40320;
  const Index global_batch = Index(c.batch) * world * c.accum;
  const bool full_pass = c.steps == 0;
  // Resume an automatic run at its absolute window cursor within this pass.
  if (full_pass && c.start_window >= data.size()) {
    std::cout << "Dataset pass complete at window " << c.start_window << '\n';
    return;
  }
  const auto ceil_div = [](Index n, Index d) { return n / d + (n % d != 0); };
  if (full_pass) {
    const Index remaining = data.size() - c.start_window;
    const Index first =
        std::min(remaining, epoch_windows - c.start_window % epoch_windows);
    const Index rest = remaining - first;
    c.steps = ceil_div(first, global_batch) +
              (rest / epoch_windows) * ceil_div(epoch_windows, global_batch) +
              ceil_div(rest % epoch_windows, global_batch);
    if (c.steps > UINT64_MAX - c.start_step)
      throw std::invalid_argument("step range overflow");
  }
  // Reserve a fresh output directory before training so older checkpoints
  // cannot collide with a save several hundred steps later.
  std::filesystem::path output = c.output;
  if (!rank && !std::filesystem::create_directories(output)) {
    for (Index i = 1;; ++i) {
      output = std::filesystem::path(c.output) / ("run-" + std::to_string(i));
      if (std::filesystem::create_directory(output))
        break;
    }
  }
  const Index first_epoch = c.start_window / epoch_windows;
  const Index offset_in_first = c.start_window % epoch_windows;
  const Index first_steps =
      ceil_div(epoch_windows - offset_in_first, global_batch);
  const Index normal_steps = ceil_div(epoch_windows, global_batch);
  const Index epochs =
      full_pass ? ceil_div(data.size(), epoch_windows) - first_epoch
                : 1 + (c.steps > first_steps
                           ? ceil_div(c.steps - first_steps, normal_steps)
                           : 0);

  std::vector<MicroBatch> micro(c.accum);
  std::vector<std::int64_t> labels(c.context);
  for (auto &b : micro) {
    b.lanes = c.batch;
    b.time = c.context;
    b.input.resize(rows);
    b.active_rows.reserve(rows);
    b.labels.reserve(rows);
    b.losses.reserve(rows);
  }
  Model model(device, input, shape, layout, c.head_chunk, c.optimized_wkv,
              c.checkpoint_activations, rows, c.accum > 1, c.adam.clip > 0);
  DataParallel dp(rank, world, model.stream());
  const Index bucket = c.bucket_mib * (1024 * 1024 / sizeof(float));
  // Exercise maximum shapes and all collective types before the memory
  // baseline.
  model.clear_gradients();
  for (auto &warm : micro) {
    warm.labels.resize(rows, 0);
    warm.active_rows.resize(rows);
    std::iota(warm.active_rows.begin(), warm.active_rows.end(), 0);
    model.accumulate(warm, rows * c.accum);
  }
  model.clear_gradients();
  dp.sum(model, bucket);
  model.gradient_scale(c.adam.clip);
  model.warm_optimizer();
  dp.scalar(0.0);
  dp.scalar(Index(0));
  dp.scalar(Index(0), DataParallel::Reduction::min);
  dp.scalar(Index(0), DataParallel::Reduction::max);
  model.synchronize();
  const bool log_vram_samples = !isatty(STDERR_FILENO);
  Index baseline = 0;
  auto check_memory = [&](Index step) {
    std::size_t free_b, total_b;
    CUDA(cudaMemGetInfo(&free_b, &total_b));
    if (!step)
      baseline = free_b;
    const bool sample = !step || step == 1 || step == 5 || step == 20;
    if ((sample && log_vram_samples) || free_b != baseline) {
      std::ostringstream log;
      log << "vram rank=" << rank << " step=" << step << " free_b=" << free_b
          << " total_b=" << total_b << '\n';
      std::cerr << log.str();
    }
    if (step && free_b != baseline)
      throw std::runtime_error("VRAM drift from initialization baseline " +
                               std::to_string(baseline));
    if (sample) {
      const Index used = total_b - free_b;
      const Index lo = dp.scalar(used, DataParallel::Reduction::min);
      const Index hi = dp.scalar(used, DataParallel::Reduction::max);
      if (hi - lo > 64 * 1024 * 1024)
        throw std::runtime_error("rank memory spread exceeds 64 MiB");
    }
  };
  check_memory(0);
  Sampler sampler(data.size(), c.seed, c.start_window);
  if (!rank)
    std::cout << "parameters=" << model.parameter_count()
              << " windows=" << data.size() << " global_batch=" << global_batch
              << " ctx=" << c.context << " micro_batch=" << c.batch
              << " accum=" << c.accum << " planned_steps=" << c.steps
              << " epoch_save=" << c.epoch_save << " mini_epochs=" << epochs
              << " steps_per_full_mini_epoch=" << normal_steps
              << " start_window=" << c.start_window
              << " output=" << output.string()
              << " mini_epoch_windows=" << epoch_windows
              << " nccl_bucket_mib=" << c.bucket_mib << " devices=" << world
              << '\n';
  const auto rank_capacity = Index(world) * c.batch;
  const bool interactive = isatty(STDOUT_FILENO);
  Index updates = 0;
  double epoch_seconds = 0;
  Index timed_windows = 0;
  for (Index completed = 0; completed < c.steps;) {
    const auto began = std::chrono::steady_clock::now();
    const auto step = c.start_step + completed + 1;
    const auto initial_window = sampler.position();
    std::size_t valid = 0;
    const Index epoch = initial_window / epoch_windows;
    const Index epoch_start = initial_window - initial_window % epoch_windows;
    Index length = epoch_windows;
    const Index offset = initial_window - epoch_start;
    if (full_pass)
      length = std::min(length, data.size() - epoch_start);
    else if (c.steps - completed < ceil_div(length - offset, global_batch))
      length = offset + (c.steps - completed) * global_batch;
    Index pending =
        std::min(global_batch, length - (initial_window - epoch_start));
    for (auto &batch : micro) {
      const auto count = std::min<Index>(rank_capacity, pending);
      fill_local(batch, data, sampler, c, count, rank, world, labels);
      valid += batch.labels.size();
      pending -= count;
    }
    valid = dp.scalar(Index(valid));
    double loss = 0;
    auto opt = c.adam;
    opt.lr = learning_rate(c, step - 1);
    if (valid) {
      model.clear_gradients();
      for (auto &batch : micro)
        model.accumulate(batch, valid);
      dp.sum(model, bucket);
      model.synchronize();

      for (const auto &batch : micro)
        loss = std::accumulate(batch.losses.begin(), batch.losses.end(), loss);
      loss = dp.scalar(loss) / valid;
      if (!std::isfinite(loss))
        throw std::runtime_error("non-finite masked loss");

      ++updates;
      const float scale = model.gradient_scale(opt.clip);
      model.update(opt, updates, scale);
      model.synchronize();
    }
    ++completed;
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - began)
            .count();
    const Index progress = sampler.position() - epoch_start;
    const bool finished = progress == length || completed == c.steps;
    epoch_seconds += seconds;
    timed_windows += sampler.position() - initial_window;
    if (completed == c.steps && !updates)
      throw std::runtime_error("no unmasked targets in remaining dataset");
    if (!rank && (interactive || finished)) {
      const int filled = static_cast<int>(24.0 * progress / length);
      std::ostringstream line;
      line << "mini_epoch=" << epoch - first_epoch + 1 << '/' << epochs << " ["
           << std::string(filled, '=') << std::string(24 - filled, ' ') << "] "
           << progress << '/' << length << " step=" << step
           << (valid ? " loss=" : " skipped_all_masked=") << (valid ? loss : 1)
           << " ETA="
           << std::llround(epoch_seconds / timed_windows * (length - progress))
           << "s lr=" << opt.lr << " targets=" << valid
           << " tokens/s=" << valid / seconds;
      auto message = line.str();
      if (interactive) {
        struct winsize size{};
        const unsigned columns =
            ioctl(STDOUT_FILENO, TIOCGWINSZ, &size) == 0 && size.ws_col
                ? size.ws_col
                : 80;
        message.resize(std::min(message.size(), std::size_t(columns - 1)));
        std::cout << "\r\033[2K";
      }
      std::cout << message;
      if (finished)
        std::cout << '\n';
      std::cout << std::flush;
    }
    const bool save =
        completed == c.steps ||
        (finished && c.epoch_save && (epoch - first_epoch) % c.epoch_save == 0);
    if (save && !rank) {
      const auto path = output / ("rwkv-" + std::to_string(step) + ".pth");
      model.save(path.string());
      std::cout << "saved=" << path.string() << " step=" << step
                << " next_window=" << sampler.position() << std::endl;
    }
    if (save)
      dp.scalar(Index(0));
    check_memory(completed);
    if (completed == c.steps)
      dp.scalar(Index(0));
    if (finished) {
      epoch_seconds = 0;
      timed_windows = 0;
    }
  }
}
} // namespace
} // namespace rwkv

int main(int argc, char **argv) try {
  if (argc == 1 || std::string(argv[1]) == "--help") {
    std::cout << "Usage: rwkv7_train INPUT.pth --dataset PATH --out DIR "
                 "[--steps N] [--nccl-bucket-mb MiB] [options]\n"
                 "  Data: --mask PATH --tokenizer PATH --ctx N --seed N "
                 "--start-window N\n"
                 "  GPU: --devices ID --batch N --accum N --head-chunk N\n"
                 "        --activation-checkpoint 0|1 --wkv-v3 0|1\n"
                 "  AdamW: --lr X --lr-final X --warmup N --beta1 X --beta2 X\n"
                 "         --weight-decay X --adam-eps X --clip X\n"
                 "  Weights: --start-step N --epoch-save N\n";
    return argc == 1 ? 1 : 0;
  }
  rwkv::run(rwkv::config(argc, argv));
  return 0;
} catch (const std::exception &error) {
  std::cerr << "rwkv7_train: " << error.what() << '\n';
  return 1;
}
