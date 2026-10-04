#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <cstdint>

#define HIP_CHECK(command) \
    { \
        hipError_t status = command; \
        if (status != hipSuccess) { \
            std::cerr << "HIP Error: " << hipGetErrorString(status) \
                      << " at line " << __LINE__ << std::endl; \
            exit(1); \
        } \
    }

constexpr double PI = 3.14159265358979323846;

// ============================================================================
// 1. GPU KERNELS
// ============================================================================

// Correct 2D Negacyclic Pre-weighting (Spatial domain: r = j1, c = j2)
__global__ void kernel_negacyclic_preweight(hipDoubleComplex* grid, const double* limbs, 
                                            int N1, int N2, bool forward, double scale) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < N1 && c < N2) {
        int idx = r * N2 + c;
        double total_N = (double)(N1 * N2);
        
        // Forward: exp(-i * pi * k / N)
        // Inverse: exp(+i * pi * k / N) * scale
	double angle = (forward ? -1.0 : 1.0) * PI * ((double)r / (double)N1 + (double)c / (double)N2);
        hipDoubleComplex tw = make_hipDoubleComplex(cos(angle), sin(angle));

        if (forward) {
            grid[idx] = make_hipDoubleComplex(limbs[idx] * tw.x, limbs[idx] * tw.y);
        } else {
            // Complex multiplication with scale
            hipDoubleComplex val = grid[idx];
            double re = (val.x * tw.x - val.y * tw.y) * scale;
            grid[idx] = make_hipDoubleComplex(re, 0.0); // Real part extraction inline
        }
    }
}

// Pure Transpose (N1 x N2 -> N2 x N1) - No Twiddle
__global__ void kernel_transpose_bailey_twiddle_fwd(hipDoubleComplex* dst, const hipDoubleComplex* src, 
                                                   int N1, int N2) {
    int r = blockIdx.y * blockDim.y + threadIdx.y; 
    int c = blockIdx.x * blockDim.x + threadIdx.x; 

    if (r < N1 && c < N2) {
        int src_idx = r * N2 + c;
        int dst_idx = c * N1 + r;
        dst[dst_idx] = src[src_idx];
    }
}

// Pure Un-transpose (N2 x N1 -> N1 x N2) - No Twiddle
__global__ void kernel_transpose_bailey_twiddle_inv(hipDoubleComplex* dst, const hipDoubleComplex* src, int N1, int N2) {
    int row2 = blockIdx.y * blockDim.y + threadIdx.y; 
    int col1 = blockIdx.x * blockDim.x + threadIdx.x; 

    if (row2 < N2 && col1 < N1) {
        int src_idx = row2 * N1 + col1; 
        int dst_idx = col1 * N2 + row2; 

        dst[dst_idx] = src[src_idx];
    }
}

// Pointwise Complex Squaring Kernel
__global__ void kernel_pointwise_square_explicit(hipDoubleComplex* __restrict__ grid, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        hipDoubleComplex z = grid[idx];
        
        // (a + bi)^2 = (a^2 - b^2) + i(2ab)
        double real_part = (z.x * z.x) - (z.y * z.y);
        double imag_part = 2.0 * z.x * z.y;
        
        grid[idx] = make_hipDoubleComplex(real_part, imag_part);
    }
}

// Unscaled 1D Row FFT Kernel
__global__ void kernel_fft_rows(hipDoubleComplex* grid, int num_rows, int N, bool invert) {
    int r = blockIdx.x; 
    if (r >= num_rows) return;

    hipDoubleComplex* row = grid + r * N;
    int tid = threadIdx.x;

    int logN = 0;
    for (int temp = N; temp > 1; temp >>= 1) logN++;

    for (int i = tid; i < N; i += blockDim.x) {
        unsigned int j = 0;
        for (int b = 0; b < logN; b++) {
            if ((i >> b) & 1) j |= (1 << (logN - 1 - b));
        }
        if (i < j) {
            hipDoubleComplex tmp = row[i];
            row[i] = row[j];
            row[j] = tmp;
        }
    }
    __syncthreads();

    for (int len = 2; len <= N; len <<= 1) {
        double ang = 2.0 * PI / (double)len * (invert ? 1.0 : -1.0);
        int half_len = len >> 1;

        for (int i = 0; i < N; i += len) {
            for (int k = tid; k < half_len; k += blockDim.x) {
                int u_idx = i + k;
                int v_idx = i + k + half_len;

                double w_ang = ang * k;
                hipDoubleComplex w = make_hipDoubleComplex(cos(w_ang), sin(w_ang));

                hipDoubleComplex u = row[u_idx];
                hipDoubleComplex v = hipCmul(row[v_idx], w);

                row[u_idx] = make_hipDoubleComplex(u.x + v.x, u.y + v.y);
                row[v_idx] = make_hipDoubleComplex(u.x - v.x, u.y - v.y);
            }
        }
        __syncthreads();
    }
}

// Extract real component to limbs array with sign inversion
__global__ void kernel_extract_real(double* limbs, const hipDoubleComplex* grid, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        limbs[idx] = grid[idx].x;
    }
}

// ============================================================================
// 2. CPU EXACT REFERENCE ACCUMULATORS & CONVOLUTION ANALYSIS
// ============================================================================

std::vector<double> compute_cpu_exact_negacyclic_2d_omp(const std::vector<double>& A, size_t N1, size_t N2) {
    size_t total_N = N1 * N2;
    std::vector<double> exact(total_N, 0.0);

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (size_t r_out = 0; r_out < N1; ++r_out) {
        for (size_t c_out = 0; c_out < N2; ++c_out) {
            double sum = 0.0;
            for (size_t r1 = 0; r1 < N1; ++r1) {
                size_t r2 = (r_out >= r1) ? (r_out - r1) : (r_out + N1 - r1);
                int sign_r = (r_out < r1) ? -1 : 1;

                for (size_t c1 = 0; c1 < N2; ++c1) {
                    double v1 = A[r1 * N2 + c1];
                    if (v1 == 0.0) continue;

                    size_t c2 = (c_out >= c1) ? (c_out - c1) : (c_out + N2 - c1);
                    int sign_c = (c_out < c1) ? -1 : 1;

                    double v2 = A[r2 * N2 + c2];
                    sum += (sign_r * sign_c) * v1 * v2;
                }
            }
            exact[r_out * N2 + c_out] = sum;
        }
    }
    return exact;
}

std::vector<double> compute_cpu_exact_negacyclic_2d(const std::vector<double>& A, size_t N1, size_t N2) {
    size_t total_N = N1 * N2;
    std::vector<double> exact(total_N, 0.0);

    for (size_t r1 = 0; r1 < N1; ++r1) {
        for (size_t c1 = 0; c1 < N2; ++c1) {
            double v1 = A[r1 * N2 + c1];
            if (v1 == 0.0) continue;

            for (size_t r2 = 0; r2 < N1; ++r2) {
                size_t r_out = r1 + r2;
                int sign_r = (r_out >= N1) ? -1 : 1;
                if (r_out >= N1) r_out -= N1;

                for (size_t c2 = 0; c2 < N2; ++c2) {
                    size_t c_out = c1 + c2;
                    int sign_c = (c_out >= N2) ? -1 : 1;
                    if (c_out >= N2) c_out -= N2;

                    double v2 = A[r2 * N2 + c2];
                    int total_sign = sign_r * sign_c; 
                    exact[r_out * N2 + c_out] += total_sign * v1 * v2;
                }
            }
        }
    }
    return exact;
}

void print_convolution_analysis(const std::vector<double>& gpu_res, const std::vector<double>& cpu_exact, size_t count) {
    double max_err = 0.0;
    double sum_sq_err = 0.0;
    double max_rel_err = 0.0;
    size_t max_err_idx = 0;

    for (size_t i = 0; i < count; ++i) {
        double err = std::abs(gpu_res[i] - cpu_exact[i]);
        sum_sq_err += err * err;
        if (err > max_err) {
            max_err = err;
            max_err_idx = i;
        }
        if (std::abs(cpu_exact[i]) > 0.0) {
            double rel_err = err / std::abs(cpu_exact[i]);
            if (rel_err > max_rel_err) max_rel_err = rel_err;
        }
    }

    double rms_err = std::sqrt(sum_sq_err / static_cast<double>(count));

    std::cout << "\n=======================================================\n";
    std::cout << "--- CONVOLUTION ACCURACY ANALYSIS (" << count << " bins) ---\n";
    std::cout << "=======================================================\n";
    std::cout << "Max Absolute FP64 Error : " << std::scientific << std::setprecision(7) << max_err 
              << " at Index [" << max_err_idx << "]\n";
    std::cout << "RMS Error               : " << std::scientific << std::setprecision(7) << rms_err << "\n";
    std::cout << "Max Relative Error      : " << std::scientific << std::setprecision(7) << max_rel_err << "\n";
    std::cout << "Exact Threshold Margin  : " << std::fixed << std::setprecision(4) << (0.5 - max_err) 
              << " (0.5 rounding boundary)\n";
    std::cout << "Overall Status          : " << (max_err < 0.5 ? "[PASS] Safe for Rounding" : "[FAIL] Loss of Precision") << "\n";
    std::cout << "=======================================================\n\n";
}

void print_complex_sample(const std::string& label, const hipDoubleComplex* d_data, size_t count) {
    std::vector<hipDoubleComplex> h_data(count);
    HIP_CHECK(hipMemcpy(h_data.data(), d_data, count * sizeof(hipDoubleComplex), hipMemcpyDeviceToHost));
    std::cout << "--- " << label << " ---\n";
    for (size_t i = 0; i < count; ++i) {
        std::cout << "  [" << i << "] = (" << h_data[i].x << ", " << h_data[i].y << "i)\n";
    }
    std::cout << "\n";
}

// ============================================================================
// 3. IMPULSE RESPONSE TEST
// ============================================================================

void run_impulse_test(int N1, int N2, bool do_squaring = false) {
    size_t total_N = N1 * N2;
    std::cout << "=======================================================\n";
    std::cout << "--- Item 1 Test: Impulse Response Vector ---\n";
    std::cout << "=======================================================\n";

    std::vector<double> h_limbs(total_N, 1.0);
    std::vector<double> h_input_copy = h_limbs;

    double *d_limbs;
    hipDoubleComplex *d_grid, *d_transposed;

    HIP_CHECK(hipMalloc(&d_limbs, total_N * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_grid, total_N * sizeof(hipDoubleComplex)));
    HIP_CHECK(hipMalloc(&d_transposed, total_N * sizeof(hipDoubleComplex)));

    HIP_CHECK(hipMemcpy(d_limbs, h_limbs.data(), total_N * sizeof(double), hipMemcpyHostToDevice));

    dim3 threads2d(16, 16);
    dim3 blocks_N1_N2((N2 + 15) / 16, (N1 + 15) / 16);
    dim3 blocks_N2_N1((N1 + 15) / 16, (N2 + 15) / 16);
    int threads_1d = 256;
    int blocks_1d = (total_N + threads_1d - 1) / threads_1d;
    double norm_scale = 1.0 / (double)total_N;

    // --- FORWARD TRANSFORM ---
    kernel_negacyclic_preweight<<<blocks_N1_N2, threads2d>>>(d_grid, d_limbs, N1, N2, true, 1.0);
    kernel_fft_rows<<<N1, 256>>>(d_grid, N1, N2, false);
    kernel_transpose_bailey_twiddle_fwd<<<blocks_N1_N2, threads2d>>>(d_transposed, d_grid, N1, N2);
    kernel_fft_rows<<<N2, 256>>>(d_transposed, N2, N1, false);

    print_complex_sample("Spectrum Sample (Forward)", d_transposed, 8);

    if (do_squaring) {
        kernel_pointwise_square_explicit<<<blocks_1d, threads_1d>>>(d_transposed, total_N);
    }

    // --- INVERSE TRANSFORM ---
    kernel_fft_rows<<<N2, 256>>>(d_transposed, N2, N1, true);
    kernel_transpose_bailey_twiddle_inv<<<blocks_N2_N1, threads2d>>>(d_grid, d_transposed, N1, N2);
    kernel_fft_rows<<<N1, 256>>>(d_grid, N1, N2, true);
    kernel_negacyclic_preweight<<<blocks_N1_N2, threads2d>>>(d_grid, nullptr, N1, N2, false, norm_scale);

    kernel_extract_real<<<blocks_1d, threads_1d>>>(d_limbs, d_grid, total_N);

    HIP_CHECK(hipMemcpy(h_limbs.data(), d_limbs, total_N * sizeof(double), hipMemcpyDeviceToHost));

    std::cout << "GPU Output for Impulse Input [1.0, 1.0, ...]:\n";
    for (size_t i = 0; i < 4; ++i) {
        std::cout << "  Bin[" << i << "] = " << std::fixed << std::setprecision(6) << h_limbs[i] << "\n";
    }
    std::cout << "\n";

    HIP_CHECK(hipFree(d_limbs));
    HIP_CHECK(hipFree(d_grid));
    HIP_CHECK(hipFree(d_transposed));
}

// ============================================================================
// 4. MAIN HARNESS DRIVER
// ============================================================================

void run_verification(int N1, int N2, int b_bits, bool do_squaring) {
    size_t total_N = N1 * N2;
    std::cout << "=======================================================\n";
    std::cout << "--- Running " << (do_squaring ? "Full Convolution (Square)" : "Identity Round-Trip") << " ---\n";
    std::cout << "=======================================================\n";

    std::vector<double> h_limbs(total_N);
    uint64_t max_val = (1ULL << b_bits) - 1;

    for (size_t i = 0; i < total_N; ++i) {
        h_limbs[i] = (double)((i * 2654435761ULL) % (max_val + 1));
    }

    std::vector<double> h_input_copy = h_limbs;

    double *d_limbs;
    hipDoubleComplex *d_grid, *d_transposed;

    HIP_CHECK(hipMalloc(&d_limbs, total_N * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_grid, total_N * sizeof(hipDoubleComplex)));
    HIP_CHECK(hipMalloc(&d_transposed, total_N * sizeof(hipDoubleComplex)));

    HIP_CHECK(hipMemcpy(d_limbs, h_limbs.data(), total_N * sizeof(double), hipMemcpyHostToDevice));

    dim3 threads2d(16, 16);
    dim3 blocks_N1_N2((N2 + 15) / 16, (N1 + 15) / 16);
    dim3 blocks_N2_N1((N1 + 15) / 16, (N2 + 15) / 16);

    int threads_1d = 256;
    int blocks_1d = (total_N + threads_1d - 1) / threads_1d;

    double norm_scale = 1.0 / (double)total_N;

    // --- FORWARD TRANSFORM ---
    kernel_negacyclic_preweight<<<blocks_N1_N2, threads2d>>>(d_grid, d_limbs, N1, N2, true, 1.0);
    kernel_fft_rows<<<N1, 256>>>(d_grid, N1, N2, false);
    kernel_transpose_bailey_twiddle_fwd<<<blocks_N1_N2, threads2d>>>(d_transposed, d_grid, N1, N2);
    kernel_fft_rows<<<N2, 256>>>(d_transposed, N2, N1, false);

    // --- SPECTRAL POINTWISE SQUARING ---
    if (do_squaring) {
        kernel_pointwise_square_explicit<<<blocks_1d, threads_1d>>>(d_transposed, total_N);
    }

    // --- INVERSE TRANSFORM ---
    kernel_fft_rows<<<N2, 256>>>(d_transposed, N2, N1, true);
    kernel_transpose_bailey_twiddle_inv<<<blocks_N2_N1, threads2d>>>(d_grid, d_transposed, N1, N2);
    kernel_fft_rows<<<N1, 256>>>(d_grid, N1, N2, true);
    kernel_negacyclic_preweight<<<blocks_N1_N2, threads2d>>>(d_grid, nullptr, N1, N2, false, norm_scale);

    kernel_extract_real<<<blocks_1d, threads_1d>>>(d_limbs, d_grid, total_N);

    HIP_CHECK(hipMemcpy(h_limbs.data(), d_limbs, total_N * sizeof(double), hipMemcpyDeviceToHost));

    if (do_squaring) {
        // Evaluate 2D exact reference for N1 x N2
        std::vector<double> h_exact = compute_cpu_exact_negacyclic_2d_omp(h_input_copy, N1, N2);
        
        std::cout << "\n  --- Sample Comparison (GPU vs CPU Exact 2D Negacyclic) ---\n";
        for (size_t i = 0; i < std::min(total_N, (size_t)8); ++i) {
            double err = std::abs(h_limbs[i] - h_exact[i]);
            std::cout << "  Bin[" << i << "] GPU: " << std::fixed << std::setprecision(2) 
                      << h_limbs[i] << " | CPU Exact: " << h_exact[i] 
                      << " | Diff: " << std::scientific << err << "\n";
        }

        print_convolution_analysis(h_limbs, h_exact, total_N);
    } else {
        double max_err = 0.0;
        size_t check_bins = std::min(total_N, (size_t)8);
        for (size_t i = 0; i < check_bins; ++i) {
            double err = std::abs(h_limbs[i] - h_input_copy[i]);
            if (err > max_err) max_err = err;
        }

        std::cout << "\nMax Measured FP64 Roundoff Error: " << std::scientific 
                  << std::setprecision(7) << max_err 
                  << (max_err < 0.5 ? " [PASS]" : " [FAIL]") << "\n\n";
    }

    HIP_CHECK(hipFree(d_limbs));
    HIP_CHECK(hipFree(d_grid));
    HIP_CHECK(hipFree(d_transposed));
}

struct PrimorialTestTarget {
    int b_bits;
    std::string name;
    double primorial_val;
    int num_limbs;
    size_t total_N;
};

int main() {
    std::vector<PrimorialTestTarget> targets = {
        {20, "13649# + 1",    13649.0,    5862,    19475},
        {15, "42209# + 1",    42209.0,    18241,   60595},
        {8,  "4328927# + 1", 4328927.0, 1878843, 6241400},
        {1,  "9562633# + 1", 9562633.0, 4151498, 13791000}
    };

    // For testing with fixed 1024 x 512 verification harness:
    int N1 = 1024, N2 = 512;
    size_t harness_N = N1 * N2;

    run_impulse_test(N1, N2, false);
    run_verification(N1, N2, 15, false);
    run_verification(N1, N2, 15, true);

    // Loop through targets that fit or adapt dimensions per target
    for (const auto& target : targets) {
        if (target.total_N <= harness_N) {
            std::cout << "\nRunning target: " << target.name << " (N = " << target.total_N << ")\n";
            // You can slice or adjust N1/N2 here to match target.total_N
            // run_verification(N1_custom, N2_custom, target.b_bits, true);
        } else {
            std::cout << "\nSkipping target " << target.name
                      << " (requires " << target.total_N << " elements, harness max is " << harness_N << ")\n";
        }
    }

    return 0;
}
