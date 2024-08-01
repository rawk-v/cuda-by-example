#include <cuda_runtime.h>
#include <cstdio>

__global__ void doubleElements(int* d_data, int size) {
    uint idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        d_data[idx] *= 2;
    }
}

int main() {
    const int size = 256;
    int h_data[size];

    // Initialize host data
    for (int i = 0; i < size; ++i) {
        h_data[i] = i;
    }

    // Allocate device memory
    int* d_data;
    cudaMalloc(&d_data, size * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel with 1 block of 256 threads
    doubleElements<<<1, 256>>>(d_data, size);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy data back from device to host
    cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Print results
    for (int i = 0; i < size; ++i) {
        printf("%d ", h_data[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(d_data);

    return 0;
}
