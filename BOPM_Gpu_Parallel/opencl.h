
#ifndef MODULE_OPENCL_H__
#define MODULE_OPENCL_H__

#define CL_USE_DEPRECATED_OPENCL_2_0_APIS

#include <CL/cl.h>

#include <iostream>
#include <vector>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define MEM_READ_FLAGS (CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR)
#define MEM_READ_WRITE_FLAGS (CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR)

#define CHECK_STATUS {\
    if (status != CL_SUCCESS) {\
        fprintf(stderr, "[FATAL]Error at line: %d code: %d message: \"%s\"\n", \
                __LINE__, \
                status, \
                status_string()); \
        exit(1); \
    }\
}

#define QUERY_DEVICE_INFO(device, type, param) clGetDeviceInfo(device, type, sizeof(param), &param, NULL)
#define QUERY_PLATFORM_INFO(platform, type, param) clGetPlatformInfo(platform, type, sizeof(param), &param, NULL)

double ExecutionTime(cl_event &event);
char *deviceTypeToString(cl_device_type deviceType);

struct OclKernel {
    cl_kernel           kernel;
    cl_int              num_args;
    cl_uint             work_dim;
    size_t*             global_work_size;
    size_t*             local_work_size;

    void SetKernel(size_t* global_size, size_t* local_size, cl_uint dim);
};
//
// single platform, single device
//
class OclModule
{
public:
    OclModule() {}
    ~OclModule() {}

    void InitModule(int platform_id, int device_id, cl_device_type device_type);

    void InitPlatform(int platform_id);
    void InitDevice(int device_id, cl_device_type device_type);
    void InitContext();
    void InitCommandQueue();

    bool LoadSource(const char* prog_fname, char **source, int *nsize);
    void InitProgram(const char* fname);
    template<typename T>
    cl_mem AllocBuffer(T* hostBuf, size_t mem_size, cl_mem_flags flags);
    template<typename T>
    void ReadBuffer(T* host_buf, size_t mem_size, cl_mem dev_mem);
    template<typename T>
    void SetKernelArgs(int kernel_idx, T* arg);
    template<typename T>
    void SetKernelLastArg(int kernel_idx, T* arg);
    void InitKernel(const char *func_name);
    cl_event RunKernel(int kernel_idx);

    // utils
    char *status_string();
    void ShowPlatform(cl_platform_id platform);
    void show_device();
    void show_devices();
    void show_device(cl_device_id device);


    std::vector<OclKernel>   kernels;
    cl_command_queue    command_queue;

private:
    cl_int status;

    cl_uint             num_platforms;
    cl_platform_id*     platforms;
    cl_platform_id      platform;

    cl_uint             num_devices;
    cl_device_type      device_type;
    cl_device_id*       devices;
    cl_device_id        device;

    cl_context          context;

    cl_program          program;
};

template<typename T>
cl_mem OclModule::AllocBuffer(T* hostBuf, size_t mem_size, cl_mem_flags flags)
{
    cl_mem deviceBuf;
    deviceBuf = clCreateBuffer(context, flags, mem_size, hostBuf, &status);
    CHECK_STATUS;

    status = clEnqueueWriteBuffer(command_queue,
                                  deviceBuf,
                                  CL_TRUE,
                                  0,
                                  mem_size,
                                  hostBuf,
                                  0,
                                  NULL,
                                  NULL);
    CHECK_STATUS;

    return deviceBuf;
}
template<typename T>
void OclModule::ReadBuffer(T* host_buf, size_t mem_size, cl_mem dev_mem) {
    status = clEnqueueReadBuffer(command_queue, dev_mem, CL_TRUE, 0, mem_size, host_buf, 0, NULL, NULL);
    CHECK_STATUS;
}

template<typename T>
void OclModule::SetKernelArgs(int kernel_idx, T* arg)
{
    status = clSetKernelArg(kernels[kernel_idx].kernel, kernels[kernel_idx].num_args, sizeof(T), arg);
    CHECK_STATUS;

    kernels[kernel_idx].num_args ++;
#ifdef DEBUG
    printf("[OpenCL] Setup kernel #%d with #%d Argument\n", kernel_idx, kernels[kernel_idx].num_args);
#endif
}

template<typename T>
void OclModule::SetKernelLastArg(int kernel_idx, T* arg)
{
    status = clSetKernelArg(kernels[kernel_idx].kernel, kernels[kernel_idx].num_args-1, sizeof(T), arg);
    CHECK_STATUS;
#ifdef DEBUG
    printf("[OpenCL] Setup kernel #%d with #%d Argument\n", kernel_idx, kernels[kernel_idx].num_args);
#endif
}

#endif
