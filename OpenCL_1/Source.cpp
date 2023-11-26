#include <CL/opencl.h>
#include <iostream>

const char* programSource =
"__kernel void vectorAdd(__global const float* a, __global const float* b, __global float* result, const unsigned int dataSize) {"
"    int i = get_global_id(0);"
"    if (i < dataSize) {"
"        result[i] = a[i] + b[i];"
"    }"
"}";

int main() {
    const int dataSize = 10;
    float a[dataSize] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    float b[dataSize] = { 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };
    float result[dataSize];

    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, NULL);

    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * dataSize, NULL, NULL);
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * dataSize, NULL, NULL);
    cl_mem bufferResult = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * dataSize, NULL, NULL);

    cl_program program = clCreateProgramWithSource(context, 1, &programSource, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "vectorAdd", NULL);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferResult);
    clSetKernelArg(kernel, 3, sizeof(int), &dataSize);

    clEnqueueWriteBuffer(queue, bufferA, CL_TRUE, 0, sizeof(float) * dataSize, a, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufferB, CL_TRUE, 0, sizeof(float) * dataSize, b, 0, NULL, NULL);

    size_t globalSize = dataSize;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);

    clEnqueueReadBuffer(queue, bufferResult, CL_TRUE, 0, sizeof(float) * dataSize, result, 0, NULL, NULL);

    for (int i = 0; i < dataSize; i++) {
        std::cout << result[i] << " ";
    }

    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferResult);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}