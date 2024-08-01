#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

void checkCudaError(cudaError_t status, const char* msg) {
    if (status != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(status) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void checkCublasError(cublasStatus_t status, const char* msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << msg << ": " << status << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    const int N = 3; // Size of the matrix (NxN)
    float alpha = 1.0f;
    float beta = 0.0f;
    float h_A[N * N] = {1, 2, 3, 4, 5, 6, 7, 8, 9}; // Host matrix A
    float h_B[N * N] = {9, 8, 7, 6, 5, 4, 3, 2, 1}; // Host matrix B
    float h_C[N * N] = {0};                        // Host matrix C

    float *d_A, *d_B, *d_C;
    cublasHandle_t handle;

    // Allocate device memory
    checkCudaError(cudaMalloc((void**)&d_A, N * N * sizeof(float)), "Failed to allocate device memory for A");
    checkCudaError(cudaMalloc((void**)&d_B, N * N * sizeof(float)), "Failed to allocate device memory for B");
    checkCudaError(cudaMalloc((void**)&d_C, N * N * sizeof(float)), "Failed to allocate device memory for C");

    // Create cuBLAS handle
    checkCublasError(cublasCreate(&handle), "Failed to create cuBLAS handle");

    // Copy matrices from host to device
    checkCublasError(cublasSetMatrix(N, N, sizeof(float), h_A, N, d_A, N), "Failed to copy matrix A from host to device");
    checkCublasError(cublasSetMatrix(N, N, sizeof(float), h_B, N, d_B, N), "Failed to copy matrix B from host to device");
    checkCublasError(cublasSetMatrix(N, N, sizeof(float), h_C, N, d_C, N), "Failed to copy matrix C from host to device");

    // Perform matrix multiplication: C = αAB + βC
    checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N),
                     "Failed to perform matrix multiplication");

    // Copy the result matrix from device to host
    checkCublasError(cublasGetMatrix(N, N, sizeof(float), d_C, N, h_C, N), "Failed to copy result matrix C from device to host");

    // Print the result
    std::cout << "Result matrix C:" << std::endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << h_C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // Clean up
    checkCudaError(cudaFree(d_A), "Failed to free device memory for A");
    checkCudaError(cudaFree(d_B), "Failed to free device memory for B");
    checkCudaError(cudaFree(d_C), "Failed to free device memory for C");
    checkCublasError(cublasDestroy(handle), "Failed to destroy cuBLAS handle");

    return 0;
}
