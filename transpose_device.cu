
#include <cassert> 
#include <cuda_runtime.h>
#include "transpose_device.cuh"
#include <stdio.h>
#include <cuda.h>

/*
 * TODO for all kernels (including naive):
 * Leave a comment above all non-coalesced memory accesses and bank conflicts.
 * Make it clear if the suboptimal access is a read or write. If an access is
 * non-coalesced, specify how many cache lines it touches, and if an access
 * causes bank conflicts, say if its a 2-way bank conflict, 4-way bank
 * conflict, etc.
 *
 * Comment all of your kernels.
 */



#define BS 32
__global__
void naiveTransposeKernel(const float *input, float *output, int n) {
    // TODO: do not modify code, just comment on suboptimal accesses
    // Each thread loads one element from input and writes one element to output.
    // But memory accesses are bad non-coalesced.
    // - Reads: Threads in a warp access different rows → slow global memory access.
    // - Writes: Threads write scattered values in output → bad for memory efficiency.

    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;


    output[j + n * i] = input[i + n * j];

}


__global__
void shmemTransposeKernel(const float *input, float *output, int n) {
    // TODO: Modify transpose kernel to use shared memory. 
    // All global memory reads and writes should be coalesced. 
    // Minimize the number of shared memory bank conflicts 
    // (0 bank conflicts should be possible using padding).
    
    __shared__ float tile[BS][BS + 1]; // Adding padding to avoid bank conflicts

    int x = threadIdx.x;
    int y = threadIdx.y;
    int i = x + blockDim.x * blockIdx.x;
    int j = y + blockDim.y * blockIdx.y;

    // Load data into shared memory
    if (i < n && j < n) {
        tile[y][x] = input[j * n + i]; 
    }

    __syncthreads(); // Sync before writing back

    // Transpose and write back
    int transposed_i = y + blockDim.y * blockIdx.y;
    int transposed_j = x + blockDim.x * blockIdx.x;

    if (transposed_i < n && transposed_j < n) {
        output[transposed_j * n + transposed_i] = tile[y][x];
    }
}



__global__
void optimalTransposeKernel(const float *input, float *output, int n) {
    // TODO: This should be based off of your shmemTransposeKernel.
    // Use any optimization tricks discussed so far to improve performance.
    // Consider ILP and loop unrolling (thread coarsening)

    __shared__ float tile[BS][BS + 1]; // Shared memory to reduce global memory access

    int x = threadIdx.x;
    int y = threadIdx.y;
    int i = x + blockDim.x * blockIdx.x;
    int j = y + blockDim.y * blockIdx.y;

    
    if (i < n && j < n) {
        tile[y][x] = input[j * n + i]; // Coalesced read into shared memory
    }
    __syncthreads();

    // Transpose the indices within the shared memory
    int transposed_i = blockIdx.y * blockDim.y + threadIdx.x;
    int transposed_j = blockIdx.x * blockDim.x + threadIdx.y;

    
    if (transposed_i < n && transposed_j < n) {
        output[transposed_j * n + transposed_i] = tile[x][y];
    }
}




void cudaTranspose(
    const float *d_input,
    float *d_output,
    int n,
    TransposeImplementation type)
{
    // TODO: you can change the block dims

    dim3 blockSize(32, 32);
    dim3 gridSize((n + 31) / 32, (n + 31) / 32);

    if (type == NAIVE) {
        naiveTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    else if (type == SHMEM) {
        shmemTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    else if (type == OPTIMAL) {
        optimalTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    else
        assert(false);
}


