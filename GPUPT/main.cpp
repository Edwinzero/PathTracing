#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <boost/array.hpp>
#include <boost/assert.hpp>
#include <CL/cl.h>


const char* cSourceFile = "kernel.cl";

// OpenCL Vars
cl_context cxGPUContext;        // OpenCL context
cl_command_queue cqCommandQue;  // OpenCL command que
cl_device_id* cdDevices;        // OpenCL device list    
cl_program cpProgram;           // OpenCL program
cl_kernel ckKernel;             // OpenCL kernel
size_t szGlobalWorkSize;        // 1D var for Total # of work items
size_t szLocalWorkSize;		    // 1D var for # of work items in the work group	
size_t szParmDataBytes;			// Byte size of context information

// demo config vars
cl_int width = 800;
cl_int height = 600;
int iNumElements = width * height;	// Length of float arrays to process (odd # for illustration)

// Forward Declarations
// *********************************************************************
void VectorAddHost(const float* pfData1, const float* pfData2, float* pfResult, int iNumElements);
void Cleanup (int iExitCode);

#define CL_VERIFY(expr) \
    do { cl_int XXXXerr = expr; if (XXXXerr != CL_SUCCESS) { std::cerr << #expr << " at " << __LINE__ << " failed! Error: " << XXXXerr << std::endl; exit(1); } } while (0)

std::string readFile(std::string name)
{
    std::ifstream f(name);
    f.seekg(0, std::ios::end);
    std::vector<char> buf(f.tellg());
    f.seekg(0, std::ios::beg);
    f.read(&buf[0], buf.size());

    return std::string(buf.begin(), buf.end());
}

void printError(const std::string& ctx, int n)
{
    std::cout << ctx << " error: ";

#define CASE(xxx) case xxx: std::cout << xxx << std::endl; break

    switch (n)
    {
        CASE(CL_INVALID_PROGRAM);
        CASE(CL_INVALID_PROGRAM_EXECUTABLE);
        CASE(CL_INVALID_KERNEL_NAME);
    }

#undef CASE
}

typedef boost::array<float, 4> float4;

struct Sphere
{
    float radius;
    float4 position;
    float4 emission;
    float4 color;
    short reflType;
};

const cl_short DIFF = 0;
const cl_short SPEC = 1;
const cl_short REFR = 2;

Sphere spheres[] = {//Scene: radius, position, emission, color, material
    { 1e5f, { 1e5f + 1, 40.8f, 81.6f }, { 0, 0, 0, 0 }, { .75, .25, .25 }, DIFF }, //Left
    { 1e5f, {-1e5f + 99.0f, 40.8f, 81.6f }, { 0, 0, 0, 0 }, { .25, .25, .75 }, DIFF }, //Rght
    { 1e5f, { 50.0f, 40.8f, 1e5f }, { 0, 0, 0, 0 }, { .75, .75, .75 }, DIFF }, //Back
    { 1e5f, { 50.0f, 40.8f, -1e5f + 170.0f }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, DIFF }, //Frnt
    { 1e5f, { 50.0f, 1e5, 81.6f }, { 0, 0, 0, 0 }, { .75, .75, .75 }, DIFF }, //Botm
    { 1e5f, { 50.0f, -1e5 + 81.6f, 81.6f }, { 0, 0, 0, 0 }, { .75, .75, .75 }, DIFF }, //Top
    { 16.5f, { 27.0f, 16.5f, 47 }, { 0, 0, 0, 0 }, { 0.999f, 0.999f, 0.999f }, SPEC }, //Mirr
    { 16.5f, { 73.0f, 16.5f, 78 }, { 0, 0, 0, 0 }, { 0.999f, 0.999f, 0.999f }, REFR }, //Glas
    { 600.0f, { 50.0f, 681.6f - .27f, 81.6f }, { 12, 12, 12 },  { 0, 0, 0, 0 }, DIFF } //Lite
};

template <int N>
int parallelize(const Sphere (&spheres)[N], std::vector<float>& radius, std::vector<float4>& position, std::vector<float4>& emission, std::vector<float4>& color, std::vector<cl_short>& reflType)
{
    for (int i = 0; i != N; ++i)
    {
        radius.push_back(spheres[i].radius);
        position.push_back(spheres[i].position);
        position.back()[3] = 0;
        emission.push_back(spheres[i].emission);
        emission.back()[3] = 0;
        color.push_back(spheres[i].color);
        color.back()[3] = 0;
        reflType.push_back(spheres[i].reflType);
    }

    return N;
}

template <typename T>
cl_mem createBuffer(cl_context ctx, const std::vector<T>& data)
{
    cl_int err;
    return clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(data[0]) * data.size(), (void *) &data[0], &err);
    BOOST_ASSERT(err == CL_SUCCESS);
}


inline float clamp(float x)
{
     return x < 0 ? 0 : x > 1 ? 1 : x;
}
inline int toInt(float x)
{
     return int(pow(clamp(x), 1 / 2.2f) * 255 + .5f);
}

void writeRaw(const std::vector<float4>& image)
{
    std::ofstream f("gpuimage.raw");

    for (int i = 0; i < width * height; ++i)
    {
        f << image[i][0] << " " << image[i][1] << " " << image[i][2] << " ";
    }
}

void writePPM(const std::vector<float4>& image)
{
    std::ofstream f("gpuimage.ppm");

    f << "P3\n" << width << " " << height << '\n' << 255 << '\n';

    for (int i = 0; i < width * height; ++i)
    {
        f << toInt(image[i][0]) << " " << toInt(image[i][1]) << " " << toInt(image[i][2]) << " ";
    }
}

// Main function 
// *********************************************************************
int main(int argc, char **argv)
{
    // set and log Global and Local work size dimensions
    szLocalWorkSize = 16;
    szGlobalWorkSize = (iNumElements + szLocalWorkSize - 1) / szLocalWorkSize * szLocalWorkSize;  // rounded up to the nearest multiple of the LocalWorkSize

    // Allocate and initialize host arrays 
    std::vector<float> radius;
    std::vector<float4> position;
    std::vector<float4> emission;
    std::vector<float4> color;
    std::vector<cl_short> reflType;
    std::vector<float4> output;
    cl_int numSpheres = parallelize(spheres, radius, position, emission, color, reflType);

    output.resize(szGlobalWorkSize);

    // Create the OpenCL context on a GPU device
    cl_int err = 0;
    cxGPUContext = clCreateContextFromType(0, CL_DEVICE_TYPE_GPU, NULL, NULL, &err);
    CL_VERIFY(err);

    // Get the list of GPU devices associated with context
    clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, 0, NULL, &szParmDataBytes);
    cdDevices = (cl_device_id*)malloc(szParmDataBytes);
    clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, szParmDataBytes, cdDevices, NULL);

    cqCommandQue = clCreateCommandQueue(cxGPUContext, cdDevices[0], 0, 0);

    // Allocate the OpenCL buffer memory objects for source and result on the device GMEM
    cl_mem radiusBuf = createBuffer(cxGPUContext, radius);
    cl_mem positionBuf = createBuffer(cxGPUContext, position);
    cl_mem emissionBuf = createBuffer(cxGPUContext, emission);
    cl_mem colorBuf = createBuffer(cxGPUContext, color);
    cl_mem reflTypeBuf = createBuffer(cxGPUContext, reflType);
    cl_mem outputBuf = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, sizeof(output[0]) * output.size(), NULL, 0);
    
    std::string source = readFile(cSourceFile);
    const char *cSource = source.c_str();

    // Create the program
    cpProgram = clCreateProgramWithSource(cxGPUContext, 1, &cSource, 0, 0);

    // Build the program
    err = clBuildProgram(cpProgram, 0, NULL, NULL, NULL, NULL);
    CL_VERIFY(err);

    // Create the kernel
    ckKernel = clCreateKernel(cpProgram, "radiance", &err);
    CL_VERIFY(err);

    // Set the Argument values
    CL_VERIFY(clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void*)&radiusBuf));
    CL_VERIFY(clSetKernelArg(ckKernel, 1, sizeof(cl_mem), (void*)&positionBuf));
    CL_VERIFY(clSetKernelArg(ckKernel, 2, sizeof(cl_mem), (void*)&emissionBuf));
    CL_VERIFY(clSetKernelArg(ckKernel, 3, sizeof(cl_mem), (void*)&colorBuf));
    CL_VERIFY(clSetKernelArg(ckKernel, 4, sizeof(cl_mem), (void*)&reflTypeBuf));
    CL_VERIFY(clSetKernelArg(ckKernel, 5, sizeof(cl_int), (void*)&numSpheres));
    CL_VERIFY(clSetKernelArg(ckKernel, 6, sizeof(cl_mem), (void*)&outputBuf));
    CL_VERIFY(clSetKernelArg(ckKernel, 7, sizeof(cl_int), (void*)&width));
    CL_VERIFY(clSetKernelArg(ckKernel, 8, sizeof(cl_int), (void*)&height));

    // --------------------------------------------------------
    // Start Core sequence... copy input data to GPU, compute, copy results back

    // Launch kernel
    CL_VERIFY(clEnqueueNDRangeKernel(cqCommandQue, ckKernel, 1, NULL, &szGlobalWorkSize, &szLocalWorkSize, 0, NULL, NULL));

    // Synchronous/blocking read of results, and check accumulated errors
    CL_VERIFY(clEnqueueReadBuffer(cqCommandQue, outputBuf, CL_TRUE, 0, sizeof(output[0]) * output.size(), &output.front(), 0, NULL, NULL));

    writePPM(output);
    writeRaw(output);

    // Cleanup and leave
    Cleanup (EXIT_SUCCESS);
}

void Cleanup (int iExitCode)
{
    // Cleanup allocated objects
    if(cdDevices)free(cdDevices);
	if(ckKernel)clReleaseKernel(ckKernel);  
    if(cpProgram)clReleaseProgram(cpProgram);
    if(cqCommandQue)clReleaseCommandQueue(cqCommandQue);
    if(cxGPUContext)clReleaseContext(cxGPUContext);

    exit (iExitCode);
}

