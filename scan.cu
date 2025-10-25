#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
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
#define THREAD_X 32
#define THREAD_Y 32
#define BLOCK (1 << 12)
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define BASE (BLOCK / (THREAD_X * THREAD_Y))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MIDDLE_THREADS (32 * 32)

#define PAD 32
#define SHMEM_PADDING(idx) ((idx) + ((idx) / PAD))

/* TODO: your GPU kernels here... */
template <typename Op>
__global__ void compute_sums(
    size_t n,
    typename Op::Data const *x,
    typename Op::Data *out,
    typename Op::Data *end_sums,
    size_t total_n) {

    using Data = typename Op::Data;
    Data accumulator = Op::identity();

    size_t offset = blockIdx.x * BLOCK + threadIdx.x * BASE;

    Data const *x_block = x + offset;
    Data *out_block = out + offset;

    size_t remaining = total_n - offset;
    int less_than_base = remaining < BASE;
    size_t loop_bound = less_than_base * remaining + (1 - less_than_base) * BASE;

    if (offset >= total_n)
        return;

    extern __shared__ __align__(16) char shmem_raw[];
    Data *shmem = reinterpret_cast<Data *>(shmem_raw);

    for (size_t i = 0; i < loop_bound; i++) { // BASE
        accumulator = Op::combine(accumulator, x_block[i]);
        out_block[i] = accumulator;
    }
    shmem[0] = Op::identity();
    shmem[SHMEM_PADDING(threadIdx.x + 1)] = accumulator;
    //(threadIdx.x >= n) ? Op::identity() : accumulator;

    __syncthreads();

    // prefix sum on
    int num_threads = CEIL_DIV(MIN(BLOCK, total_n), BASE);

#pragma unroll
    for (int i = 1; i < num_threads; i <<= 1) {

        Data add =
            (threadIdx.x >= i) ? shmem[SHMEM_PADDING(threadIdx.x - i)] : Op::identity();

        __syncthreads();

        shmem[SHMEM_PADDING(threadIdx.x)] =
            Op::combine(add, shmem[SHMEM_PADDING(threadIdx.x)]);
        // }
        __syncthreads();
    }

    Data to_add = shmem[SHMEM_PADDING(threadIdx.x)]; //

    for (int i = 0; i < loop_bound; i++) {

        // if (offset + i < total_n)
        out_block[i] = Op::combine(to_add, out_block[i]); //
    }
    // __syncthreads();
    if (threadIdx.x == blockDim.x - 1) {

        end_sums[blockIdx.x] = out_block[BASE - 1];
    }
}

// TODO: fix hardcoding
template <typename Op>
__global__ void
compute_middle_sums(size_t n, typename Op::Data const *x, typename Op::Data *out) {
    using Data = typename Op::Data;
    Data accumulator = Op::identity();

    int base_case = CEIL_DIV(n, MIDDLE_THREADS);

    int offset = threadIdx.x * base_case; // hardcoded to 4

    Data const *x_block = x + offset;
    Data *out_block = out + offset;

    // if (offset >= n)
    //     return;

    extern __shared__ __align__(16) char shmem_raw[];
    Data *shmem = reinterpret_cast<Data *>(shmem_raw);

    // partial sum from threadIdx block

    for (int i = 0; i < base_case; i++) {
        accumulator = Op::combine(accumulator, x_block[i]);
        out_block[i] = accumulator;
    }
    shmem[0] = Op::identity();
    shmem[SHMEM_PADDING(threadIdx.x + 1)] = accumulator;
    //(threadIdx.x >= n) ? Op::identity() : accumulator;

    __syncthreads();

// prefix sum on
#pragma unroll
    for (int i = 1; i < MIDDLE_THREADS; i <<= 1) {

        Data add =
            (threadIdx.x >= i) ? shmem[SHMEM_PADDING(threadIdx.x - i)] : Op::identity();

        __syncthreads();
        shmem[SHMEM_PADDING(threadIdx.x)] =
            Op::combine(add, shmem[SHMEM_PADDING(threadIdx.x)]);

        __syncthreads();
    }

    Data to_add = shmem[SHMEM_PADDING(threadIdx.x)];
    //(threadIdx.x == 0) ? Op::identity() : shmem[threadIdx.x - 1]; //

    for (int i = 0; i < base_case; i++) {
        out_block[i] = Op::combine(to_add, out_block[i]); //
    }
}

// template <typename Op>
// __global__ void combine_sums(int num_blocks, typename Op::Data *arr) {
//     using Data = typename Op::Data;
//     Data total = arr[BLOCK - 1];
//     for (int i = 2; i < num_blocks; i++) {
//         total = Op::combine(total, arr[i * BLOCK - 1]);
//         // std::cout << i << std::endl;
//         arr[i * BLOCK - 1] = total;
//     }
// }

template <typename Op>
__global__ void fill_blocks(
    size_t n,
    typename Op::Data *workspace,
    size_t total_n,
    typename Op::Data *end_sums) {
    using Data = typename Op::Data;
    Data *arr = workspace + (blockIdx.x + 1) * BLOCK;

    Data base = end_sums[blockIdx.x]; //*(arr - 1);
    int end = (blockIdx.x == gridDim.x - 1) ? ((total_n - 1) % BLOCK) + 1 : n; //

    for (size_t i = threadIdx.x; i < end; i += blockDim.x) {

        arr[i] = Op::combine(base, arr[i]);
    }
}

// Returns desired size of scratch buffer in bytes.
template <typename Op> size_t get_workspace_size(size_t n) {
    using Data = typename Op::Data;
    /* TODO: your CPU code here... */
    return n * sizeof(Data) + 2 * CEIL_DIV(n, BLOCK) * sizeof(Data);
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

    // compute sums per block
    int num_blocks = CEIL_DIV(n, BLOCK);
    Data *end_points = arr + n;
    Data *end_points_write = end_points + num_blocks;
    // std::cout << "num blocks: " << num_blocks << std::endl;
    int shmem_bytes = sizeof(Data) * (THREAD_X * THREAD_Y + PAD);

    int num_threads = CEIL_DIV(MIN(BLOCK, n), BASE); // should be 1024 at max sizes

    compute_sums<Op>
        <<<num_blocks, num_threads, shmem_bytes>>>(BLOCK, x, arr, end_points, n);

    compute_middle_sums<Op><<<1, MIDDLE_THREADS, (MIDDLE_THREADS + PAD) * sizeof(Data)>>>(
        num_blocks,
        end_points,
        end_points_write);

    // fill in intermediate

    fill_blocks<Op><<<num_blocks - 1, num_threads>>>(BLOCK, arr, n, end_points_write);
    // }

    return arr; // replace with an appropriate pointer
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
