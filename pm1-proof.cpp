#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <gmp.h>
#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>

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

struct PrimorialTestTarget {
    int b_bits;
    std::string name;
    unsigned long prime_arg;
    int num_limbs;
    size_t total_N;
};

// ============================================================================
// GPU KERNELS (2D Negacyclic IBDWT Engine)
// ============================================================================

__global__ void kernel_negacyclic_preweight(hipDoubleComplex* grid, const double* limbs, 
                                            int N1, int N2, bool forward, double scale) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < N1 && c < N2) {
        int idx = r * N2 + c;
        double k_global = (double)r * N2 + (double)c;
        double angle = (forward ? -1.0 : 1.0) * PI * k_global / (double)(N1 * N2);
        hipDoubleComplex tw = make_hipDoubleComplex(cos(angle), sin(angle));

        if (forward) {
            grid[idx] = make_hipDoubleComplex(limbs[idx] * tw.x, limbs[idx] * tw.y);
        } else {
            hipDoubleComplex val = grid[idx];
            double re = (val.x * tw.x - val.y * tw.y) * scale;
            grid[idx] = make_hipDoubleComplex(re, 0.0);
        }
    }
}

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

__global__ void kernel_transpose_bailey_twiddle_inv(hipDoubleComplex* dst, const hipDoubleComplex* src, int N1, int N2) {
    int row2 = blockIdx.y * blockDim.y + threadIdx.y; 
    int col1 = blockIdx.x * blockDim.x + threadIdx.x; 

    if (row2 < N2 && col1 < N1) {
        int src_idx = row2 * N1 + col1; 
        int dst_idx = col1 * N2 + row2; 
        dst[dst_idx] = src[src_idx];
    }
}

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

__global__ void kernel_pointwise_square(hipDoubleComplex* __restrict__ grid, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        hipDoubleComplex z = grid[idx];
        double real_part = (z.x * z.x) - (z.y * z.y);
        double imag_part = 2.0 * z.x * z.y;
        grid[idx] = make_hipDoubleComplex(real_part, imag_part);
    }
}

__global__ void kernel_pointwise_mul(hipDoubleComplex* __restrict__ dst, const hipDoubleComplex* src1, const hipDoubleComplex* src2, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        hipDoubleComplex a = src1[idx];
        hipDoubleComplex b = src2[idx];
        double real = a.x * b.x - a.y * b.y;
        double imag = a.x * b.y + a.y * b.x;
        dst[idx] = make_hipDoubleComplex(real, imag);
    }
}

__global__ void kernel_extract_real(double* limbs, const hipDoubleComplex* grid, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        limbs[idx] = grid[idx].x;
    }
}

// ============================================================================
// GPU BIG INT MODULAR CLASS
// ============================================================================

class GpuBigIntMod {
public:
    size_t N;
    int N1, N2;
    hipDoubleComplex* d_buffer;
    hipDoubleComplex* d_transposed;
    double* d_temp_real;

    GpuBigIntMod(size_t size) : N(size) {
        N1 = 64;
        N2 = 2048;
        while ((size_t)(N1 * N2) < N) {
            N2 *= 2;
        }
        size_t total_size = (size_t)N1 * N2;
        
        HIP_CHECK(hipMalloc(&d_buffer, total_size * sizeof(hipDoubleComplex)));
        HIP_CHECK(hipMalloc(&d_transposed, total_size * sizeof(hipDoubleComplex)));
        HIP_CHECK(hipMalloc(&d_temp_real, total_size * sizeof(double)));
    }

    ~GpuBigIntMod() {
        hipFree(d_buffer);
        hipFree(d_transposed);
        hipFree(d_temp_real);
    }

    void set_int(uint64_t val) {
        size_t total_size = (size_t)N1 * N2;
        std::vector<double> h_limbs(total_size, 0.0);
        h_limbs[0] = (double)val;
        HIP_CHECK(hipMemcpy(d_temp_real, h_limbs.data(), total_size * sizeof(double), hipMemcpyHostToDevice));
        
        dim3 threads2d(16, 16);
        dim3 blocks_N1_N2((N2 + 15) / 16, (N1 + 15) / 16);
        kernel_negacyclic_preweight<<<blocks_N1_N2, threads2d>>>(d_buffer, d_temp_real, N1, N2, true, 1.0);
        hipDeviceSynchronize();
    }

    void set_mpz(const mpz_t val) {
        size_t total_size = (size_t)N1 * N2;
        std::vector<double> h_limbs(total_size, 0.0);

        mpz_t temp;
        mpz_init_set(temp, val);
        size_t limb_idx = 0;
        while (mpz_sgn(temp) > 0 && limb_idx < total_size) {
            h_limbs[limb_idx++] = (double)mpz_tstbit(temp, 0);
            mpz_fdiv_q_2exp(temp, temp, 1);
        }
        mpz_clear(temp);

        HIP_CHECK(hipMemcpy(d_temp_real, h_limbs.data(), total_size * sizeof(double), hipMemcpyHostToDevice));
        
        dim3 threads2d(16, 16);
        dim3 blocks_N1_N2((N2 + 15) / 16, (N1 + 15) / 16);
        kernel_negacyclic_preweight<<<blocks_N1_N2, threads2d>>>(d_buffer, d_temp_real, N1, N2, true, 1.0);
        hipDeviceSynchronize();
    }


void get_mpz(mpz_t val, long bit_idx) const {
        size_t total_size = (size_t)N1 * N2;
        dim3 threads2d(16, 16);
        dim3 blocks_N1_N2((N2 + 15) / 16, (N1 + 15) / 16);
        dim3 blocks_N2_N1((N1 + 15) / 16, (N2 + 15) / 16);
        int threads_1d = 256;
        int blocks_1d = (total_size + threads_1d - 1) / threads_1d;
        double norm_scale = 1.0 / (double)total_size;

        // Inverse 2D Negacyclic Transform
        kernel_fft_rows<<<N2, 256>>>(d_buffer, N2, N1, true);
        kernel_transpose_bailey_twiddle_inv<<<blocks_N2_N1, threads2d>>>(d_transposed, d_buffer, N1, N2);
        kernel_fft_rows<<<N1, 256>>>(d_transposed, N1, N2, true);
        kernel_negacyclic_preweight<<<blocks_N1_N2, threads2d>>>(d_transposed, nullptr, N1, N2, false, norm_scale);
        kernel_extract_real<<<blocks_1d, threads_1d>>>(d_temp_real, d_transposed, total_size);

        std::vector<double> h_limbs(total_size);
        HIP_CHECK(hipMemcpy(h_limbs.data(), d_temp_real, total_size * sizeof(double), hipMemcpyDeviceToHost));

        mpz_set_ui(val, 0);
        mpz_t term;
        mpz_init(term);

        // --- CARRY PROPAGATION PASS ---
        int64_t carry = 0;
        for (size_t i = 0; i < total_size; ++i) {
            int64_t lval = (int64_t)std::round(h_limbs[i]) + carry;
            carry = lval >> 1; // Extract base-2 carry
            lval &= 1;         // Keep normalized bit (0 or 1)

            if (lval != 0) {
                mpz_set_ui(term, lval);
                mpz_mul_2exp(term, term, i);
                mpz_add(val, val, term);
            }
        }
        
	if (carry != 0) {
    // In negacyclic convolution (mod 2^N + 1), 2^N = -1
    mpz_set_si(term, -carry);
    mpz_add(val, val, term);
}

        mpz_clear(term);

        if (bit_idx <= 2) {
            char* str_val = mpz_get_str(NULL, 10, val);
            std::cout << "\n================ DEBUG FULL VALUE ================\n";
            std::cout << "Reconstructed bit length: " << mpz_sizeinbase(val, 2) << "\n";
            std::cout << "Full decimal value (copyable into PARI/GP):\n" << str_val << "\n";
            std::cout << "==================================================\n\n";
            free(str_val);
        }
    }

    void run_forward_transform() {
        dim3 threads2d(16, 16);
        dim3 blocks_N1_N2((N2 + 15) / 16, (N1 + 15) / 16);

        kernel_fft_rows<<<N1, 256>>>(d_buffer, N1, N2, false);
        kernel_transpose_bailey_twiddle_fwd<<<blocks_N1_N2, threads2d>>>(d_transposed, d_buffer, N1, N2);
        kernel_fft_rows<<<N2, 256>>>(d_transposed, N2, N1, false);
        
        hipDoubleComplex* tmp = d_buffer;
        d_buffer = d_transposed;
        d_transposed = tmp;
    }

void square() {
        run_forward_transform();
        size_t total_size = (size_t)N1 * N2;
        int threads_1d = 256;
        int blocks_1d = (total_size + threads_1d - 1) / threads_1d;

        // Use d_transposed as a temporary complex buffer
        HIP_CHECK(hipMemcpy(d_transposed, d_buffer, total_size * sizeof(hipDoubleComplex), hipMemcpyDeviceToDevice));
        
        // Multiply d_buffer by the temporary copy
        kernel_pointwise_mul<<<blocks_1d, threads_1d>>>(d_buffer, d_buffer, d_transposed, total_size);
        hipDeviceSynchronize();
    }

    void multiply_transformed(const GpuBigIntMod& transformed_other) {
        size_t total_size = (size_t)N1 * N2;
        int threads_1d = 256;
        int blocks_1d = (total_size + threads_1d - 1) / threads_1d;

        run_forward_transform();
        kernel_pointwise_mul<<<blocks_1d, threads_1d>>>(d_buffer, d_buffer, transformed_other.d_buffer, total_size);
        hipDeviceSynchronize();
    }
};

void gpu_binary_exponentiation(GpuBigIntMod& base_val, unsigned long prime_arg, GpuBigIntMod& result) {
    mpz_t p_primorial, M, current_val;
    mpz_init(p_primorial); mpz_init(M); mpz_init(current_val);
    
    mpz_primorial_ui(p_primorial, prime_arg);
    mpz_add_ui(M, p_primorial, 1);
    
    size_t bit_count = mpz_sizeinbase(p_primorial, 2);
    std::cout << "[GMP] Computed " << prime_arg << "# (Bit length: " << bit_count << " bits)\n";
    std::cout << "[GPU] Executing corrected binary exponentiation loop...\n";

    auto start_time = std::chrono::high_resolution_clock::now();

    // Initialize result with base a = 3 (MSB = 1)
    result.set_int(3);

    GpuBigIntMod transformed_base = base_val;
    transformed_base.run_forward_transform();

    // Loop from MSB-1 down to 0
    for (long i = (long)bit_count - 2; i >= 0; --i) {
        result.square();
        result.get_mpz(current_val, i);
        mpz_mod(current_val, current_val, M);
        result.set_mpz(current_val);

        int bit_val = mpz_tstbit(p_primorial, i) ? 1 : 0;
        if (bit_val) {
            result.multiply_transformed(transformed_base);
            result.get_mpz(current_val, i);
            mpz_mod(current_val, current_val, M);
            result.set_mpz(current_val);
        }

        if (i % 1000 == 0 || i <= 5) {
            char* str = mpz_get_str(NULL, 10, current_val);
            std::string snippet(str);
            if (snippet.length() > 20) {
                snippet = snippet.substr(0, 20) + "...";
            }
            std::cout << "  [PROGRESS] Bit index " << i << " / " << bit_count 
                      << " (bit=" << bit_val << ") -> Current residue snippet: " << snippet << "\n";
            free(str);
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cout << "[GPU] Exponentiation completed in " << duration_ms << " ms.\n";

    mpz_clear(p_primorial);
    mpz_clear(M);
    mpz_clear(current_val);
}

bool verify_p_minus_1_proof(const PrimorialTestTarget& target) {
    auto total_start = std::chrono::high_resolution_clock::now();

    std::cout << "\n=======================================================\n";
    std::cout << "--- Running P-1 Primality Test for " << target.name << " ---\n";
    std::cout << "=======================================================\n";

    GpuBigIntMod base(target.total_N);
    base.set_int(3);

    GpuBigIntMod result(target.total_N);

    gpu_binary_exponentiation(base, target.prime_arg, result);

    mpz_t p_primorial, M, final_val, residue;
    mpz_init(p_primorial); mpz_init(M); mpz_init(final_val); mpz_init(residue);
    
    mpz_primorial_ui(p_primorial, target.prime_arg);
    mpz_add_ui(M, p_primorial, 1);

    result.get_mpz(final_val, 0);
    mpz_mod(residue, final_val, M);

    bool fermat_passed = (mpz_cmp_ui(residue, 1) == 0);

    if (!fermat_passed) {
        char* res_str = mpz_get_str(NULL, 10, residue);
        std::cout << "   [DEBUG] Final computed residue: " << std::string(res_str) << "\n";
        free(res_str);
    }

    mpz_clear(p_primorial);
    mpz_clear(M);
    mpz_clear(final_val);
    mpz_clear(residue);

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();

    if (fermat_passed) {
        std::cout << "   [PASS] Fermat residue test passed (residue ≡ 1).\n";
        std::cout << "Status: [PRIME] (Total time: " << total_ms << " ms)\n";
        return true;
    } else {
        std::cout << "   [FAIL] Fermat residue test failed (residue != 1).\n";
        std::cout << "Status: [COMPOSITE] (Total time: " << total_ms << " ms)\n";
        return false;
    }
}

int main() {
    std::vector<PrimorialTestTarget> targets = {
        {20, "13649# + 1",  13649,  5862,  19475}
    };

    std::cout << "=======================================================\n";
    std::cout << "--- GPU Primorial IBDWT P-1 Proof Harness (Snippet) ---\n";
    std::cout << "=======================================================\n";

    for (const auto& target : targets) {
        verify_p_minus_1_proof(target);
    }

    return 0;
}
