
#include "opencl.h"
// About kernel operations

// return millisecond 
double ExecutionTime(cl_event &event)
{
    cl_ulong start, end;

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    return (double)(end - start) * 1e-6f;
}

void OclKernel::SetKernel(size_t* global_size, size_t* local_size, cl_uint dim) {
    global_work_size = global_size;
    local_work_size = local_size;
    work_dim = dim;
}

void OclModule::InitKernel(const char *func_name) {
    OclKernel ocl_kernel;
    ocl_kernel.kernel = clCreateKernel(program, func_name, &status);
    CHECK_STATUS;
    ocl_kernel.num_args = 0;
    kernels.push_back(ocl_kernel);
#ifdef DEBUG
    printf("[OpenCL] Kernel '%s' Built ...\n", func_name);
#endif
}

cl_event OclModule::RunKernel(int kernel_idx) {
    cl_event event;
    status = clEnqueueNDRangeKernel(command_queue,
                                    kernels[kernel_idx].kernel,
                                    kernels[kernel_idx].work_dim,
                                    NULL,
                                    kernels[kernel_idx].global_work_size,
                                    kernels[kernel_idx].local_work_size,
                                    0,
                                    NULL,
                                    &event);
    CHECK_STATUS;

    status = clFinish(command_queue);
    CHECK_STATUS;

#ifdef DEBUG
    //puts("[OpenCL] Finished Running Kernel");
#endif
    return event;
}

void OclModule::InitModule(int platform_id, int device_id, cl_device_type device_type) {
    status = CL_SUCCESS;

    printf("\e[1;31mInitializing OpenCL ...\e[m\n");
    InitPlatform(platform_id);
    InitDevice(device_id, device_type);
    InitContext();
    InitCommandQueue();
}

void OclModule::InitPlatform(int platform_id) {
    cl_uint num_platforms = 0;

    status = clGetPlatformIDs(0, NULL, &num_platforms);

    if (num_platforms == 0) {
        puts("There's no existing OpenCL platform.");
        exit(1);
    }
    cl_platform_id * platforms = (cl_platform_id*) malloc(num_platforms * sizeof(cl_platform_id));

    status = clGetPlatformIDs(num_platforms, platforms, NULL); CHECK_STATUS;

    if (platform_id >= num_platforms || platform_id < 0) {
        printf("Platform Id illegal, please select one of the following platforms:\n");
        for (int i = 0; i < num_platforms; i++) {
            printf("\e[1;32mPlatform id\e[m: %d\n", i);
            ShowPlatform(platforms[i]);
        }
        exit(1);
    }

    platform = platforms[platform_id];
    printf("\e[1;32mPlatform id\e[m: %d\n", platform_id);
    ShowPlatform(platform);
    free(platforms);
    platforms = NULL;

#ifdef DEBUG
    puts("[OpenCL] Platform Initialized ...");
#endif
}

void OclModule::InitDevice(int device_id, cl_device_type device_type) {
    if (device_type == 10) {
        puts("Device type illegal, please select one of the following devices:");
        puts("\e[1;32mall\e[m");
        puts("\e[1;32maccelerator\e[m");
        puts("\e[1;32mcpu\e[m");
        puts("\e[1;32mgpu\e[m");
        exit(1);
    }

    printf("\e[1;32mDevice Type:\e[m %s\n", deviceTypeToString(device_type));
    this->device_type = device_type;

    cl_uint num_devices = 0;

    status = clGetDeviceIDs(platform, device_type, 0, NULL, &num_devices);
    //CHECK_STATUS;

    if (num_devices == 0) {
        printf("Device with type %s not found.\n", deviceTypeToString(device_type));
        exit(1);
    }

    cl_device_id *devices = (cl_device_id *) malloc(num_devices * sizeof(cl_device_id));

    status = clGetDeviceIDs(platform, device_type, num_devices, devices, NULL);
    CHECK_STATUS;
    if (device_id >= num_devices || device_id < 0) {
        puts("Device Id illegal, please select:");
        for (int i = 0; i < num_devices; i++) {
            printf("\e[1;32mDevice id\e[m: %d\n", i);
            show_device(devices[i]);
        }
        exit(1);
    }

    // show device info
    printf("\e[1;32mDevice id\e[m: %d\n", device_id);
    show_device(devices[device_id]);
    device = devices[device_id];

    free(devices);
    devices = NULL;

#ifdef DEBUG
    puts("[OpenCL] Device Initialized ...");

#endif
}

void OclModule::InitContext() {
	cl_context_properties properties[3] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platform,
        0
    };
    context = clCreateContextFromType(properties,
                                      device_type,
                                      NULL,
                                      NULL,
                                      &status);
    CHECK_STATUS;

#ifdef DEBUG
    puts("[OpenCL] Context Initialized ...");
#endif
}

void OclModule::InitCommandQueue() {
    command_queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
    CHECK_STATUS;

#ifdef DEBUG
    puts("[OpenCL] Command Queue Initialized ...");
#endif
}

bool OclModule::LoadSource(const char *prog_fname, char **source, int *nsize) {
    bool error = false;
    FILE *fp = NULL;
    *nsize = 0;

    fp = fopen(prog_fname, "rb");
    if (!fp)
        error = true;
    else
    {
        fseek(fp, 0, SEEK_END);
        *nsize = ftell(fp);
        fseek(fp, 0, SEEK_SET);

        *source = new char [*nsize+1];
        if (*source)
        {
            fread(*source, 1, *nsize, fp);
            (*source)[*nsize] = 0;
        }
        else
            error = true;
        fclose(fp);
    }
    return error;
}

void OclModule::InitProgram(const char *fname) {
    char *programSource = NULL;
    int programSize = 0;

    if (LoadSource(fname, &programSource, &programSize)) {
	    printf("Error: Couldn't load kernel source from file '%s'.\n", fname);
	    status = CL_INVALID_OPERATION;
    }
    CHECK_STATUS;

#ifdef DEBUG
    printf("[OpenCL] Program Source '%s' Loaded ...\n", fname);
#endif

    // CreateProgram
    program = clCreateProgramWithSource(context,
                                        1,
                                        (const char **)&programSource,
                                        NULL,
                                        &status);
    CHECK_STATUS;

    // Build program
    status = clBuildProgram(program, 1, &device, "-I ./kernel/", NULL, NULL);

    //CHECK_STATUS;
    //puts("[OpenCL] Program Built");
#ifdef PROGDEBUG
    puts("[OpenCL] Program Compilation ...");
    puts("---");
    size_t buildLogSize = 0;
    clGetProgramBuildInfo(program,
                          device,
                          CL_PROGRAM_BUILD_LOG,
                          0,
                          NULL,
                          &buildLogSize);

    cl_char *buildLog = (cl_char*) malloc(sizeof(cl_char) * buildLogSize);
    if (buildLog) {
        clGetProgramBuildInfo(program,
                              device,
                              CL_PROGRAM_BUILD_LOG,
                              buildLogSize,
                              buildLog,
                              NULL);
        printf("%s\n", buildLog);
    }
    puts("---");
#endif
    CHECK_STATUS;
}

char *deviceTypeToString(cl_device_type deviceType) {
    switch (deviceType)
    {
        case CL_DEVICE_TYPE_DEFAULT: return "Default";        
        case CL_DEVICE_TYPE_CPU: return "Cpu";
        case CL_DEVICE_TYPE_GPU: return "Gpu";
        case CL_DEVICE_TYPE_ACCELERATOR: return "Accelerator";
        default: return "***UNKNOWN***";
    }
}

void OclModule::ShowPlatform(cl_platform_id platform) {
    char platformOpenCLVersion[100];
    char platformName[100];
    char platformVendor[100];
    char platformExtensions[1000];

    status |= QUERY_PLATFORM_INFO(platform, CL_PLATFORM_NAME, platformName);
    status |= QUERY_PLATFORM_INFO(platform, CL_PLATFORM_VENDOR, platformVendor);
    status |= QUERY_PLATFORM_INFO(platform, CL_PLATFORM_EXTENSIONS, platformExtensions);
    status |= QUERY_PLATFORM_INFO(platform, CL_PLATFORM_VERSION, platformOpenCLVersion);

    printf("Name:\t\t%s\n", platformName);
    printf("Vendor:\t\t%s\n", platformVendor);
    printf("Version:\t%s\n", platformOpenCLVersion);
    printf("Extensions:\t%s\n", platformExtensions);
    puts("");
}

void OclModule::show_device() { show_device(device); }

void OclModule::show_device(cl_device_id device) {
    cl_device_type  deviceType;
    char            deviceName[100];
    char            deviceVendor[100];
    cl_uint         deviceMaxComputeUnits;
    cl_uint         deviceMaxWorkItemDimensions;
    size_t          deviceMaxWorkItemSizes[16];
    size_t          deviceMaxWorkGroupSize;
    cl_uint         deviceMaxClockFrequency;
    cl_bool         deviceImageSupport;
    size_t	        deviceImageMaxWidth;
    size_t	        deviceImageMaxHeight;
    cl_ulong	    deviceMaxMemAlloc;

    status |= QUERY_DEVICE_INFO(device, CL_DEVICE_NAME, deviceName);
	status |= QUERY_DEVICE_INFO(device, CL_DEVICE_TYPE, deviceType);
	status |= QUERY_DEVICE_INFO(device, CL_DEVICE_VENDOR, deviceVendor);
	status |= QUERY_DEVICE_INFO(device, CL_DEVICE_MAX_COMPUTE_UNITS, deviceMaxComputeUnits);
    status |= QUERY_DEVICE_INFO(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, deviceMaxWorkItemDimensions);
    status |= QUERY_DEVICE_INFO(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, deviceMaxWorkItemSizes);
    status |= QUERY_DEVICE_INFO(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, deviceMaxWorkGroupSize);
    status |= QUERY_DEVICE_INFO(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, deviceMaxClockFrequency);
    status |= QUERY_DEVICE_INFO(device, CL_DEVICE_IMAGE_SUPPORT, deviceImageSupport);
    status |= QUERY_DEVICE_INFO(device, CL_DEVICE_IMAGE2D_MAX_WIDTH, deviceImageMaxWidth);
    status |= QUERY_DEVICE_INFO(device, CL_DEVICE_IMAGE2D_MAX_HEIGHT, deviceImageMaxHeight);
    status |= QUERY_DEVICE_INFO(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, deviceMaxMemAlloc);
    CHECK_STATUS;

    printf("Name: \t\t\t\t%s\n", deviceName);
    printf("Type: \t\t\t\t%s\n", deviceTypeToString(deviceType));
    printf("Vendor: \t\t\t%s\n", deviceVendor);
    printf("Max Compute Units: \t\t%d\n", deviceMaxComputeUnits);
    printf("Max Work Item Dimensions: \t%d\n", deviceMaxWorkItemDimensions);

    for (int j = 0; j < deviceMaxWorkItemDimensions; j++)
        printf("Max Work Item Sizes[%d]: \t%lu\n", j, deviceMaxWorkItemSizes[j]);

    printf("Max Work Group Size: \t\t%lu\n", deviceMaxWorkGroupSize);
    printf("Max Clock Frequency: \t\t%d\n", deviceMaxClockFrequency);
    printf("Image Support?  \t\t%s\n", deviceImageSupport ? "true" : "false");
    printf("Image Max 2D Width  \t\t%lu\n", deviceImageMaxWidth);
    printf("Image Max 2D Height \t\t%lu\n", deviceImageMaxHeight);
    printf("Max Mem Alloc Size \t\t%llu \t%llu MB\n", deviceMaxMemAlloc, deviceMaxMemAlloc/(1024*1024) );
    puts("");
}

// dictionary-like function
char * OclModule::status_string() {
    switch (status) {
        case CL_SUCCESS: 
            return strdup("Success!");
        case CL_DEVICE_NOT_FOUND:
            return strdup("Device not found.");
        case CL_DEVICE_NOT_AVAILABLE:
            return strdup("Device not available");
        case CL_COMPILER_NOT_AVAILABLE:
            return strdup("Compiler not available");
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
            return strdup("Memory object allocation failure");
        case CL_OUT_OF_RESOURCES:
            return strdup("Out of resources");
        case CL_OUT_OF_HOST_MEMORY:
            return strdup("Out of host memory");
        case CL_PROFILING_INFO_NOT_AVAILABLE:
            return strdup("Profiling information not available");
        case CL_MEM_COPY_OVERLAP:
            return strdup("Memory copy overlap");
        case CL_IMAGE_FORMAT_MISMATCH:
            return strdup("Image format mismatch");
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:
            return strdup("Image format not supported");
        case CL_BUILD_PROGRAM_FAILURE:
            return strdup("Program build failure");
        case CL_MAP_FAILURE:
            return strdup("Map failure");
        case CL_INVALID_VALUE:
            return strdup("Invalid value");
        case CL_INVALID_DEVICE_TYPE:
            return strdup("Invalid device type");
        case CL_INVALID_PLATFORM:
            return strdup("Invalid platform");
        case CL_INVALID_DEVICE:
            return strdup("Invalid device");
        case CL_INVALID_CONTEXT:
            return strdup("Invalid context");
        case CL_INVALID_QUEUE_PROPERTIES:
            return strdup("Invalid queue properties");
        case CL_INVALID_COMMAND_QUEUE:
            return strdup("Invalid command queue");
        case CL_INVALID_HOST_PTR:
            return strdup("Invalid host pointer");
        case CL_INVALID_MEM_OBJECT:
            return strdup("Invalid memory object");
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
            return strdup("Invalid image format descriptor");
        case CL_INVALID_IMAGE_SIZE:
            return strdup("Invalid image size");
        case CL_INVALID_SAMPLER:
            return strdup("Invalid sampler");
        case CL_INVALID_BINARY:
            return strdup("Invalid binary");
        case CL_INVALID_BUILD_OPTIONS:
            return strdup("Invalid build options");
        case CL_INVALID_PROGRAM:
            return strdup("Invalid program");
        case CL_INVALID_PROGRAM_EXECUTABLE:
            return strdup("Invalid program executable");
        case CL_INVALID_KERNEL_NAME:
            return strdup("Invalid kernel name");
        case CL_INVALID_KERNEL_DEFINITION:
            return strdup("Invalid kernel definition");
        case CL_INVALID_KERNEL:
            return strdup("Invalid kernel");
        case CL_INVALID_ARG_INDEX:
            return strdup("Invalid argument index");
        case CL_INVALID_ARG_VALUE:
            return strdup("Invalid argument value");
        case CL_INVALID_ARG_SIZE:
            return strdup("Invalid argument size");
        case CL_INVALID_KERNEL_ARGS:
            return strdup("Invalid kernel arguments");
        case CL_INVALID_WORK_DIMENSION:
            return strdup("Invalid work dimension");
        case CL_INVALID_WORK_GROUP_SIZE:
            return strdup("Invalid work group size");
        case CL_INVALID_WORK_ITEM_SIZE:
            return strdup("Invalid work item size");
        case CL_INVALID_GLOBAL_OFFSET:
            return strdup("Invalid global offset");
        case CL_INVALID_EVENT_WAIT_LIST:
            return strdup("Invalid event wait list");
        case CL_INVALID_EVENT:
            return strdup("Invalid event");
        case CL_INVALID_OPERATION:
            return strdup("Invalid operation");
        case CL_INVALID_GL_OBJECT:
            return strdup("Invalid OpenGL object");
        case CL_INVALID_BUFFER_SIZE:
            return strdup("Invalid buffer size");
        case CL_INVALID_MIP_LEVEL:
            return strdup("Invalid mip-map level");
        default:
            return strdup("Unknown");
    }
}
