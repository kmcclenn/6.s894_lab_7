#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>

////////////////////////////////////////////////////////////////////////////////
// Utility Functions

void cuda_check(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": "
                  << cudaGetErrorString(code) << std::endl;
        exit(1);
    }
}

#define CUDA_CHECK(x) \
    do { \
        cuda_check((x), __FILE__, __LINE__); \
    } while (0)

template <typename Op>
void print_array(
    size_t n,
    typename Op::Data const *x // allowed to be either a CPU or GPU pointer
);

////////////////////////////////////////////////////////////////////////////////
// CPU Reference Implementation (Already Written)

template <typename Op>
void scan_cpu(size_t n, typename Op::Data const *x, typename Op::Data *out) {
    using Data = typename Op::Data;
    Data accumulator = Op::identity();
    for (size_t i = 0; i < n; i++) {
        accumulator = Op::combine(accumulator, x[i]);
        out[i] = accumulator;
    }
}

/// <--- your code here --->

////////////////////////////////////////////////////////////////////////////////
// Optimized GPU Implementation

namespace scan_gpu {
// #define THREAD_X 32
// #define THREAD_Y 32
#define THREADS (4 * 32)
// #define BLOCK (1 << 12)
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
// #define BASE (BLOCK / (THREAD_X * THREAD_Y))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
// #define MIDDLE_THREADS (16 * 32)

#define PAD 32
#define SHMEM_PADDING(idx) ((idx) + ((idx) / PAD))

/* TODO: your GPU kernels here... */

template <typename Op>
__global__ void reduce(size_t n, typename Op::Data const *x, typename Op::Data *out) {
    using Data = typename Op::Data;
    Data accumulator = Op::identity();

    Data const *x_block = x + blockIdx.x * blockDim.x;

    extern __shared__ __align__(16) char shmem_raw[];
    Data *shmem = reinterpret_cast<Data *>(shmem_raw);

    shmem[SHMEM_PADDING(threadIdx.x)] = x_block[threadIdx.x];

    __syncthreads();

    for (int i = 1; i < THREADS; i <<= 1) {
        Data add =
            (threadIdx.x >= i) ? shmem[SHMEM_PADDING(threadIdx.x - i)] : Op::identity();
        __syncthreads();
        shmem[SHMEM_PADDING(threadIdx.x)] =
            Op::combine(add, shmem[SHMEM_PADDING(threadIdx.x)]);
        __syncthreads();
    }

    // TODO: account for blocks with not full THREADS length

    int last = MIN((int)n - (int)(blockIdx.x * blockDim.x), (int)blockDim.x) - 1;
    out[blockIdx.x] = shmem[SHMEM_PADDING(last)];
}

template <typename Op>
__global__ void scan_block(size_t n, typename Op::Data const *x, typename Op::Data *out) {
    using Data = typename Op::Data;
    Data accumulator = Op::identity();

    extern __shared__ __align__(16) char shmem_raw[];
    Data *shmem = reinterpret_cast<Data *>(shmem_raw);

    shmem[SHMEM_PADDING(threadIdx.x)] = x[threadIdx.x];

    __syncthreads();

    for (int i = 1; i < THREADS; i <<= 1) {
        Data add =
            (threadIdx.x >= i) ? shmem[SHMEM_PADDING(threadIdx.x - i)] : Op::identity();
        __syncthreads();
        shmem[SHMEM_PADDING(threadIdx.x)] =
            Op::combine(add, shmem[SHMEM_PADDING(threadIdx.x)]);
        __syncthreads();
    }
    if (threadIdx.x < n)
        out[threadIdx.x] = shmem[SHMEM_PADDING(threadIdx.x)];
}

template <typename Op>
__global__ void scan(
    size_t n,
    typename Op::Data const *x,
    typename Op::Data const *end_points,
    typename Op::Data *out) {

    using Data = typename Op::Data;
    Data accumulator = Op::identity();

    extern __shared__ __align__(16) char shmem_raw[];
    Data *shmem = reinterpret_cast<Data *>(shmem_raw);

    Data const *x_block = x + blockIdx.x * blockDim.x;
    Data *out_block = out + blockIdx.x * blockDim.x;

    shmem[SHMEM_PADDING(threadIdx.x)] = x_block[threadIdx.x];

    __syncthreads();

    for (int i = 1; i < THREADS; i <<= 1) {
        Data add =
            (threadIdx.x >= i) ? shmem[SHMEM_PADDING(threadIdx.x - i)] : Op::identity();
        __syncthreads();
        shmem[SHMEM_PADDING(threadIdx.x)] =
            Op::combine(add, shmem[SHMEM_PADDING(threadIdx.x)]);
        __syncthreads();
    }
    Data block_carry = (blockIdx.x == 0) ? Op::identity() : end_points[blockIdx.x - 1];
    out_block[threadIdx.x] = Op::combine(block_carry, shmem[SHMEM_PADDING(threadIdx.x)]);
}

// Returns desired size of scratch buffer in bytes.
template <typename Op> size_t get_workspace_size(size_t n) {
    using Data = typename Op::Data;
    /* TODO: your CPU code here... */
    size_t total = n;
    size_t size = n;
    while (size > THREADS) {
        total += size;
        size = CEIL_DIV(size, THREADS);
    }

    return 2 * total * sizeof(Data);
}

// 'launch_scan'
//
// Input:
//
//   'n': Number of elements in the input array 'x'.
//
//   'x': Input array in GPU memory. The 'launch_scan' function is allowed to
//   overwrite the contents of this buffer.
//
//   'workspace': Scratch buffer in GPU memory. The size of the scratch buffer
//   in bytes is determined by 'get_workspace_size<Op>(n)'.
//
// Output:
//
//   Returns a pointer to GPU memory which will contain the results of the scan
//   after all launched kernels have completed. Must be either a pointer to the
//   'x' buffer or to an offset within the 'workspace' buffer.
//
//   The contents of the output array should be "partial reductions" of the
//   input; each element 'i' of the output array should be given by:
//
//     output[i] = Op::combine(x[0], x[1], ..., x[i])
//
//   where 'Op::combine(...)' of more than two arguments is defined in terms of
//   repeatedly combining pairs of arguments. Note that 'Op::combine' is
//   guaranteed to be associative, but not necessarily commutative, so
//
//        Op::combine(a, b, c)              // conceptual notation; not real C++
//     == Op::combine(a, Op::combine(b, c)) // real C++
//     == Op::combine(Op::combine(a, b), c) // real C++
//
//  but we don't necessarily have
//
//    Op::combine(a, b) == Op::combine(b, a) // not true in general!
//

template <typename Op>
typename Op::Data *launch_scan(
    size_t n,
    typename Op::Data *x, // pointer to GPU memory
    void *workspace       // pointer to GPU memory
) {
    using Data = typename Op::Data;
    /* TODO: your CPU code here... */
    Data *arr = reinterpret_cast<Data *>(workspace); // size n
    Data *data = x;

    // compute sums per block

    size_t size = n;
    size_t offsets[8];
    offsets[0] = 0;
    int iter = 1;

    while (size > THREADS) {
        size_t blocks = CEIL_DIV(size, THREADS);
        reduce<Op><<<blocks, THREADS, (THREADS + PAD) * sizeof(Data)>>>(
            size,
            data,
            arr + offsets[iter - 1]);

        size = blocks;
        data = arr + offsets[iter - 1];

        offsets[iter] = offsets[iter - 1] + size;
        iter++;
    }
    iter--;

    Data *final_block = (iter == 0) ? data : arr + offsets[iter];
    size_t threads = MIN(THREADS, n);

    Data *base_out = (iter == 0) ? arr : final_block + threads;

    scan_block<Op>
        <<<1, threads, (threads + PAD) * sizeof(Data)>>>(size, final_block, base_out);

    size_t larger_size = threads;
    base_out += larger_size;
    Data *end_points = arr + offsets[iter];
    if (n > THREADS) {
        while (iter >= 0) {

            larger_size = (iter == 0) ? n : (offsets[iter] - offsets[iter - 1]);
            size_t blocks = CEIL_DIV(larger_size, THREADS);

            Data *data_ptr = (iter == 0) ? x : arr + offsets[iter - 1];
            scan<Op><<<blocks, THREADS, (THREADS + PAD) * sizeof(Data)>>>(
                larger_size,
                data_ptr,
                end_points,
                base_out);

            size = larger_size;
            iter--;
            end_points = base_out;

            base_out += larger_size;
        }
    }

    return base_out - larger_size; // replace with an appropriate pointer
}

} // namespace scan_gpu

/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

struct DebugRange {
    uint32_t lo;
    uint32_t hi;

    static constexpr uint32_t INVALID = 0xffffffff;

    static __host__ __device__ __forceinline__ DebugRange invalid() {
        return {INVALID, INVALID};
    }

    __host__ __device__ __forceinline__ bool operator==(const DebugRange &other) const {
        return lo == other.lo && hi == other.hi;
    }

    __host__ __device__ __forceinline__ bool operator!=(const DebugRange &other) const {
        return !(*this == other);
    }

    __host__ __device__ bool is_empty() const { return lo == hi; }

    __host__ __device__ bool is_valid() const { return lo != INVALID; }

    std::string to_string() const {
        if (lo == INVALID) {
            return "INVALID";
        } else {
            return std::to_string(lo) + ":" + std::to_string(hi);
        }
    }
};

struct DebugRangeConcatOp {
    using Data = DebugRange;

    static __host__ __device__ __forceinline__ Data identity() { return {0, 0}; }

    static __host__ __device__ __forceinline__ Data combine(Data a, Data b) {
        if (a.is_empty()) {
            return b;
        } else if (b.is_empty()) {
            return a;
        } else if (a.is_valid() && b.is_valid() && a.hi == b.lo) {
            return {a.lo, b.hi};
        } else {
            return Data::invalid();
        }
    }

    static std::string to_string(Data d) { return d.to_string(); }
};

struct SumOp {
    using Data = uint32_t;

    static __host__ __device__ __forceinline__ Data identity() { return 0; }

    static __host__ __device__ __forceinline__ Data combine(Data a, Data b) {
        return a + b;
    }

    static std::string to_string(Data d) { return std::to_string(d); }
};

constexpr size_t max_print_array_output = 1025;
static thread_local size_t total_print_array_output = 0;

template <typename Op> void print_array(size_t n, typename Op::Data const *x) {
    using Data = typename Op::Data;

    // copy 'x' from device to host if necessary
    cudaPointerAttributes attr;
    CUDA_CHECK(cudaPointerGetAttributes(&attr, x));
    auto x_host_buf = std::vector<Data>();
    Data const *x_host_ptr = nullptr;
    if (attr.type == cudaMemoryTypeDevice) {
        x_host_buf.resize(n);
        x_host_ptr = x_host_buf.data();
        CUDA_CHECK(
            cudaMemcpy(x_host_buf.data(), x, n * sizeof(Data), cudaMemcpyDeviceToHost));
    } else {
        x_host_ptr = x;
    }

    if (total_print_array_output >= max_print_array_output) {
        return;
    }

    printf("[\n");
    for (size_t i = 0; i < n; i++) {
        auto s = Op::to_string(x_host_ptr[i]);
        printf("  [%zu] = %s,\n", i, s.c_str());
        total_print_array_output++;
        if (total_print_array_output > max_print_array_output) {
            printf("  ... (output truncated)\n");
            break;
        }
    }
    printf("]\n");

    if (total_print_array_output >= max_print_array_output) {
        printf(
            "(Reached maximum limit on 'print_array' output; skipping further calls "
            "to 'print_array')\n");
    }

    total_print_array_output++;
}

template <typename Reset, typename F>
double benchmark_ms(double target_time_ms, Reset &&reset, F &&f) {
    double best_time_ms = std::numeric_limits<double>::infinity();
    double elapsed_ms = 0.0;
    while (elapsed_ms < target_time_ms) {
        reset();
        CUDA_CHECK(cudaDeviceSynchronize());
        auto start = std::chrono::high_resolution_clock::now();
        f();
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        double this_ms = std::chrono::duration<double, std::milli>(end - start).count();
        elapsed_ms += this_ms;
        best_time_ms = std::min(best_time_ms, this_ms);
    }
    return best_time_ms;
}

struct Results {
    double time_ms;
    double bandwidth_gb_per_sec;
};

enum class Mode {
    TEST,
    BENCHMARK,
};

template <typename Op>
Results run_config(Mode mode, std::vector<typename Op::Data> const &x) {
    // Allocate buffers
    using Data = typename Op::Data;
    size_t n = x.size();
    size_t workspace_size = scan_gpu::get_workspace_size<Op>(n);
    Data *x_gpu;
    Data *workspace_gpu;
    CUDA_CHECK(cudaMalloc(&x_gpu, n * sizeof(Data)));
    CUDA_CHECK(cudaMalloc(&workspace_gpu, workspace_size));
    CUDA_CHECK(cudaMemcpy(x_gpu, x.data(), n * sizeof(Data), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(workspace_gpu, 0, workspace_size));

    // Test correctness
    auto expected = std::vector<Data>(n);
    scan_cpu<Op>(n, x.data(), expected.data());
    auto out_gpu = scan_gpu::launch_scan<Op>(n, x_gpu, workspace_gpu);
    if (out_gpu == nullptr) {
        printf("'launch_scan' function not yet implemented (returned nullptr)\n");
        exit(1);
    }
    auto actual = std::vector<Data>(n);
    CUDA_CHECK(
        cudaMemcpy(actual.data(), out_gpu, n * sizeof(Data), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n; ++i) {
        if (actual.at(i) != expected.at(i)) {
            auto actual_str = Op::to_string(actual.at(i));
            auto expected_str = Op::to_string(expected.at(i));
            printf(
                "Mismatch at position %zu: %s != %s\n",
                i,
                actual_str.c_str(),
                expected_str.c_str());
            if (n <= 128) {
                printf("Input:\n");
                print_array<Op>(n, x.data());
                printf("\nExpected:\n");
                print_array<Op>(n, expected.data());
                printf("\nActual:\n");
                print_array<Op>(n, actual.data());
            }
            exit(1);
        }
    }
    if (mode == Mode::TEST) {
        return {0.0, 0.0};
    }

    // Benchmark
    double target_time_ms = 200.0;
    double time_ms = benchmark_ms(
        target_time_ms,
        [&]() {
            CUDA_CHECK(
                cudaMemcpy(x_gpu, x.data(), n * sizeof(Data), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(workspace_gpu, 0, workspace_size));
        },
        [&]() { scan_gpu::launch_scan<Op>(n, x_gpu, workspace_gpu); });
    double bytes_processed = n * sizeof(Data) * 2;
    double bandwidth_gb_per_sec = bytes_processed / time_ms / 1e6;

    // Cleanup
    CUDA_CHECK(cudaFree(x_gpu));
    CUDA_CHECK(cudaFree(workspace_gpu));

    return {time_ms, bandwidth_gb_per_sec};
}

std::vector<DebugRange> gen_debug_ranges(uint32_t n) {
    auto ranges = std::vector<DebugRange>();
    for (uint32_t i = 0; i < n; ++i) {
        ranges.push_back({i, i + 1});
    }
    return ranges;
}

template <typename Rng> std::vector<uint32_t> gen_random_data(Rng &rng, uint32_t n) {
    auto uniform = std::uniform_int_distribution<uint32_t>(0, 100);
    auto data = std::vector<uint32_t>();
    for (uint32_t i = 0; i < n; ++i) {
        data.push_back(uniform(rng));
    }
    return data;
}

template <typename Op, typename GenData>
void run_tests(std::vector<uint32_t> const &sizes, GenData &&gen_data) {
    for (auto size : sizes) {
        auto data = gen_data(size);
        printf("  Testing size %8u\n", size);
        run_config<Op>(Mode::TEST, data);
        printf("  OK\n\n");
    }
}

int main(int argc, char const *const *argv) {
    auto correctness_sizes = std::vector<uint32_t>{
        16,
        10,
        128,
        100,
        1024,
        1000,
        1 << 20,
        1'000'000,
        16 << 20,
        64 << 20,
    };

    auto rng = std::mt19937(0xCA7CAFE);

    printf("Correctness:\n\n");
    printf("Testing scan operation: debug range concatenation\n\n");
    run_tests<DebugRangeConcatOp>(correctness_sizes, gen_debug_ranges);
    printf("Testing scan operation: integer sum\n\n");
    run_tests<SumOp>(correctness_sizes, [&](uint32_t n) {
        return gen_random_data(rng, n);
    });

    printf("Performance:\n\n");

    size_t n = 64 << 20;
    auto data = gen_random_data(rng, n);

    printf("Benchmarking scan operation: integer sum, size %zu\n\n", n);

    // Warmup
    run_config<SumOp>(Mode::BENCHMARK, data);
    // Benchmark
    auto results = run_config<SumOp>(Mode::BENCHMARK, data);
    printf("  Time: %.2f ms\n", results.time_ms);
    printf("  Throughput: %.2f GB/s\n", results.bandwidth_gb_per_sec);

    return 0;
}
