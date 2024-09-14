#include <cstdlib>
#include <cstdio>

#include <chrono>

#include <curand.h>
#include <curand_kernel.h>

__global__ void vec_add(float *A, float *B, float *result, int N) {
   asm(
        ".reg .pred p;\n\t"
        ".reg .f32 temp_val, a_val, b_val;\n\t"
        ".reg .u64 a, b, res;\n\t"
        ".reg .u32 tx, bx, bs, ti;\n\t"
        ".reg .u64 tia;\n\t"
        "\n\t"
        "mov.u32 tx, %tid.x;\n\t"
        "mov.u32 bx, %ctaid.x;\n\t"
        "mov.u32 bs, %ntid.x;\n\t"
        "mad.lo.u32  ti, bs, bx, tx;\n\t"
        "setp.lt.u32 p, ti, %3;\n\t"
        "cvt.u64.u32 tia, ti;\n\t"
        "@!p bra end_if;\n\t"
        "{mad.lo.u64 a, 4, tia, %0; ld.global.f32 a_val, [a]; mad.lo.u64 b, 4, tia, %1; ld.global.f32 b_val, [b]; add.f32 temp_val, a_val, b_val; mad.lo.u64 res, 4, tia, %2; st.f32 [res], temp_val;}\n\t"
        "end_if:"
   :
   : "l"(A), "l"(B), "l"(result), "r"(N)
   );
}

int main(int argc, char **argv) {
    int N = 10000;

    curandGenerator_t rng;
    curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);

    float *d_A, *d_B, *d_result;
    cudaMalloc((void **)&d_A, N * sizeof(float));
    cudaMalloc((void **)&d_B, N * sizeof(float));
    cudaMalloc((void **)&d_result, N * sizeof(float));

    curandGenerateUniform(rng, d_A, N);
    curandGenerateUniform(rng, d_B, N);

    int grid_dim = (N + 1023) / 1024;
    vec_add <<<grid_dim, 1024>>> (d_A, d_B, d_result, N);

    float *result, *A, *B;
    result = new float[N];
    A = new float[N];
    B = new float[N];
    cudaMemcpy(result, d_result, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemcpy(A, d_A, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(B, d_B, N * sizeof(float), cudaMemcpyDeviceToHost);
    float t;
    bool success = true;
    for (int i = 0; i < N && success; ++i) {
	t = A[i] + B[i];
        if (t - result[i] > 1e-4) {
		printf("Test failed! on entry(%d)\n", i);
                printf("%f - %f\n", result[i], t);
                success = false;
	}
    }
    if (success) {
        printf("Test Successful!\n");
    }
    
    return 0;
}

