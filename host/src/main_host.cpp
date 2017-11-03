/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Host program for Matrix Multiplication Kernel
//
// Copyright (C) 2013-2016 Altera Corporation, San Jose, California, USA. All rights reserved.
// Reference: https://www.altera.com/support/support-resources/design-examples/design-software/opencl/vector-addition.html
//
// Host Modification by Sukrut Kelkar
//
// Date: 5th March 2017
//
// Project: Hardware Acceleration using OpenCL on FPGA 
//
// Description:
// ------------
// This host program executes a Matrix Multiplication kernel to perform:
// C = A * B
// where A, B and C are Matrices with row x column elements.
//
// This host program supports partitioning the problem across multiple OpenCL
// devices if available. If there are M available devices, the problem is
// divided so that each device operates on N/M points. The host program
// assumes that all devices are of the same type (that is, the same binary can
// be used), but the code can be generalized to support different device types
// easily.
//
// Verification is performed against the same computation on the host CPU.
///////////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"

//Matrix 1 dimensions
#define rowSizeA 2000
#define colSizeA 2000

//Matrix 1 dimensions
#define rowSizeB 2000
#define colSizeB 2000

double start_time_cpu;
double end_time_cpu;
  
using namespace aocl_utils;

// OpenCL runtime configuration
cl_platform_id platform = NULL;
unsigned num_devices = 0;
scoped_array<cl_device_id> device; // num_devices elements
cl_context context = NULL;
scoped_array<cl_command_queue> queue; // num_devices elements
cl_program program = NULL;
scoped_array<cl_kernel> kernel; // num_devices elements

scoped_array<cl_mem> input_a_buf; // num_devices elements
scoped_array<cl_mem> input_b_buf; // num_devices elements
scoped_array<cl_mem> output_buf; // num_devices elements


// Problem data.
unsigned N = 1000000; // problem size

scoped_array<scoped_aligned_ptr<int> > input_a, input_b; // num_devices elements
scoped_array<scoped_aligned_ptr<int> > output; // num_devices elements

scoped_array<scoped_array<int> > ref_output; // num_devices elements
scoped_array<unsigned> n_per_device; // num_devices elements

// Function prototypes
float rand_float();
bool init_opencl();
void init_problem();
void run();
void cleanup();

//Fill in the Matrix 1 and Matrix 2 from buffers
//int Data2Matrix (int *p, int *q);

//Multiply both the Matrices
//int MatrixMultiply();

//Display Matrix 1, 2 and its Multiplication result
//int displayMatrix();

//Global declaration of all the 2D Matrices
//int Matrix1[rowSizeM1][columnSizeM1];
//int Matrix2[rowSizeM2][columnSizeM2];
//int MultResult [rowSizeM1][columnSizeM2];

//int *A;
//int *B;
//int *C;
FILE *FP1,*FP2;

// Entry point.
int main(int argc, char **argv) {
  Options options(argc, argv);

  // Optional argument to specify the problem size.
  if(options.has("n")) {
    N = options.get<unsigned>("n");
  }
printf("\nInitializing OpenCL\n");
  // Initialize OpenCL.
  if(!init_opencl()) {
    return -1;
  }

  // Initialize the problem data.
  // Requires the number of devices to be known.
  printf("\nInitializing Problem\n");
  init_problem();

  // Run the kernel.
   printf("\nExecuting Kernel\n");
  run();

  // Free the resources allocated
  cleanup();

  return 0;
}

/////// HELPER FUNCTIONS ///////

// Randomly generate a floating-point number between -10 and 10.
int rand_int() {
  return int(rand()) / int(RAND_MAX) * 20 - 10;
}

// Initializes the OpenCL objects.
bool init_opencl() {
  cl_int status;

  printf("Initializing OpenCL\n");

  if(!setCwdToExeDir()) {
    return false;
  }

  // Get the OpenCL platform.
  platform = findPlatform("Intel(R) FPGA");
  if(platform == NULL) {
    printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform.\n");
    return false;
  }

  // Query the available OpenCL device.
  device.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));
  printf("Platform: %s\n", getPlatformName(platform).c_str());
  printf("Using %d device(s)\n", num_devices);
  for(unsigned i = 0; i < num_devices; ++i) {
    printf("  %s\n", getDeviceName(device[i]).c_str());
  }

  // Create the context.
  context = clCreateContext(NULL, num_devices, device, &oclContextCallback, NULL, &status);
  checkError(status, "Failed to create context");

  // Create the program for all device. Use the first device as the
  // representative device (assuming all device are of the same type).
  std::string binary_file = getBoardBinaryFile("matrix_mul", device[0]);
  printf("Using AOCX: %s\n", binary_file.c_str());
  program = createProgramFromBinary(context, binary_file.c_str(), device, num_devices);

  // Build the program that was just created.
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  checkError(status, "Failed to build program");

  // Create per-device objects.
  queue.reset(num_devices);
  kernel.reset(num_devices);
  n_per_device.reset(num_devices);

  input_a_buf.reset(num_devices);
  input_b_buf.reset(num_devices);
  output_buf.reset(num_devices);


  for(unsigned i = 0; i < num_devices; ++i) {
    // Command queue.
    queue[i] = clCreateCommandQueue(context, device[i], CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status, "Failed to create command queue");

    // Kernel.
    const char *kernel_name = "matrix_mul";
    kernel[i] = clCreateKernel(program, kernel_name, &status);
    checkError(status, "Failed to create kernel");

    // Determine the number of elements processed by this device.
   // n_per_device[i] = N / num_devices; // number of elements handled by this device
   n_per_device[i] = rowSizeA * colSizeB;

    // Spread out the remainder of the elements over the first
    // N % num_devices.
//    if(i < (N % num_devices)) {
//      n_per_device[i]++;
//    }


    // Input buffers.
    input_a_buf[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, 
        rowSizeA*colSizeA*sizeof(int), NULL, &status);	
    checkError(status, "Failed to create buffer for input A");

    input_b_buf[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, 
        rowSizeB*colSizeB*sizeof(int), NULL, &status);
    checkError(status, "Failed to create buffer for input B");

    // Output buffer.
    output_buf[i] = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
        rowSizeA * colSizeB * sizeof(int), NULL, &status);
    checkError(status, "Failed to create buffer for output");

  }

  return true;
}

// Initialize the data for the problem. Requires num_devices to be known.
void init_problem() {
  if(num_devices == 0) {
    checkError(-1, "No devices");
  }

  input_a.reset(num_devices);
  input_b.reset(num_devices);
  output.reset(num_devices);
  ref_output.reset(num_devices);
  
  printf("\nInitializing Input matrix\n");

  // Generate input vectors A and B and the reference output consisting
  // of a total of N elements.
  // We create separate arrays for each device so that each device has an
  // aligned buffer.
  for(unsigned i = 0; i < num_devices; ++i) {

    input_a[i].reset(rowSizeA*colSizeA);
    input_b[i].reset(rowSizeB*colSizeB);
    output[i].reset(rowSizeA*colSizeB);
    ref_output[i].reset(rowSizeA*colSizeB);
/*
    for(unsigned j = 0; j < n_per_device[i]; ++j) {
      input_a[i][j] = rand_float();
      input_b[i][j] = rand_float();
      ref_output[i][j] = input_a[i][j] + input_b[i][j];
    }
*/ 
//Put Matrix multiplication code here
//*******************/
start_time_cpu = getCurrentTimestamp();	
printf("\nFile Input to matrix\n");
    FP1 = fopen("../Matrix1.txt", "r");
    FP2 = fopen("../Matrix2.txt", "r");
    for (int l = 0; l < (rowSizeA*colSizeA); l++)
    {
       fscanf(FP1, "%d", &input_a[i][l]);
	   //input_a[i][l] = rand_int() + l;
	  // printf("\t%d",input_a[i][l]);
	   
    }
    
    //Matrix 2
    for (int l = 0; l < (rowSizeB*colSizeB); l++)
    {
        fscanf(FP2, "%d", &input_b[i][l]);
	//	input_b[i][l] = rand_int() - l;
		// printf("\t%d",input_b[i][l]);
    }	

	printf("\nmatrix multiplication on CPU\n");
	
	
	
	//Calculate multiplication
	for(int l=0;l<rowSizeA;l++)
	{
		for(int m=0; m<colSizeB ; m++)
		{
			int temp = 0;
			for(int k=0; k<colSizeA ; k++)
				{
					temp = temp + input_a[i][l*rowSizeA + k] * input_b[i][k*colSizeB + m];
				}
			ref_output[i][l*rowSizeA + m] = temp;
			//printf("\t%d",ref_output[i][l*rowSizeA + m]);
		}
	}
	
	end_time_cpu = getCurrentTimestamp();
 // printf("\nTime: %0.3f ms\n", (end_time_cpu - start_time_cpu) * 1e3);
  
     fclose(FP1);
    fclose(FP2);

  }
}

void run() {
  cl_int status;
  int r_A = rowSizeA;
  int c_B = colSizeB;

  const double start_time = getCurrentTimestamp();

  // Launch the problem for each device.
  scoped_array<cl_event> kernel_event(num_devices);
  scoped_array<cl_event> finish_event(num_devices);

  for(unsigned i = 0; i < num_devices; ++i) {


    // Transfer inputs to each device. Each of the host buffers supplied to
    // clEnqueueWriteBuffer here is already aligned to ensure that DMA is used
    // for the host-to-device transfer.
    cl_event write_event[2];
    status = clEnqueueWriteBuffer(queue[i], input_a_buf[i], CL_FALSE,
        0, (rowSizeA*colSizeA) * sizeof(int), input_a[i], 0, NULL, &write_event[0]);
    checkError(status, "Failed to transfer input A");

    status = clEnqueueWriteBuffer(queue[i], input_b_buf[i], CL_FALSE,
        0, (rowSizeB*colSizeB) * sizeof(int), input_b[i], 0, NULL, &write_event[1]);
    checkError(status, "Failed to transfer input B");


    // Set kernel arguments.
    unsigned argi = 0;
double start_fpga = getCurrentTimestamp();


    status = clSetKernelArg(kernel[i], argi++, sizeof(int), (void *)&r_A);
    checkError(status, "Failed to set argument %d", argi - 1);

    status = clSetKernelArg(kernel[i], argi++, sizeof(int), (void *)&c_B);
    checkError(status, "Failed to set argument %d", argi - 1);
	
	status = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem), &input_a_buf[i]);
    checkError(status, "Failed to set argument %d", argi - 1);
	
	status = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem), &input_b_buf[i]);
    checkError(status, "Failed to set argument %d", argi - 1);

    status = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem), &output_buf[i]);
    checkError(status, "Failed to set argument %d", argi - 1);


    // Enqueue kernel.
    // Use a global work size corresponding to the number of elements to add
    // for this device.
    //
    // We don't specify a local work size and let the runtime choose
    // (it'll choose to use one work-group with the same size as the global
    // work-size).
    //
    // Events are used to ensure that the kernel is not launched until
    // the writes to the input buffers have completed.
    const size_t global_work_size[2] = {rowSizeA,colSizeB};
	const size_t local_work_size[2] = {2,2};
    printf("Launching for device %d (%zd elements)\n", i, global_work_size);
//double start_fpga = getCurrentTimestamp();

cl_ulong time_start, time_end;
double total_time;

clGetEventProfilingInfo(kernel_event[i], CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);



    status = clEnqueueNDRangeKernel(queue[i], kernel[i], 2, NULL,
        global_work_size, local_work_size, 2, write_event, &kernel_event[i]);

    checkError(status, "Failed to launch kernel");

clFinish(queue[i]);

double	end_fpga = getCurrentTimestamp();
 printf("\nFPGA Time: %0.3f ms\n", (end_fpga - start_fpga) * 1e3);

clGetEventProfilingInfo(kernel_event[i], CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
total_time = time_end - time_start;
printf("\nProfiling Execution time in milliseconds = %0.3f ms\n", total_time * 1000000 );




    // Read the result. This the final operation.
    status = clEnqueueReadBuffer(queue[i], output_buf[i], CL_FALSE,
        0, n_per_device[i] * sizeof(int), output[i], 1, &kernel_event[i], &finish_event[i]);

    // Release local events.
    clReleaseEvent(write_event[0]);
    clReleaseEvent(write_event[1]);

  }

  // Wait for all devices to finish.
  clWaitForEvents(num_devices, finish_event);

  const double end_time = getCurrentTimestamp();

  // Wall-clock time taken.
  printf("\nTime: %0.3f ms\n", (end_time - start_time) * 1e3);





  // Get kernel times using the OpenCL event profiling API.
  for(unsigned i = 0; i < num_devices; ++i) {
    cl_ulong time_ns = getStartEndTime(kernel_event[i]);
    printf("Kernel time (device %d): %0.3f ms\n", i, double(time_ns) * 1e-6);
  }
 printf("\nCPU Time: %0.3f ms\n", (end_time_cpu - start_time_cpu) * 1e3);




  // Release all events.
  for(unsigned i = 0; i < num_devices; ++i) {
    clReleaseEvent(kernel_event[i]);
    clReleaseEvent(finish_event[i]);
  }

  // Verify results.
  bool pass = true;
  for(unsigned i = 0; i < num_devices && pass; ++i) {
    for(unsigned j = 0; j < n_per_device[i] && pass; ++j) {
      if(fabsf(output[i][j] - ref_output[i][j]) > 1.0e-5f) {
        printf("Failed verification @ device %d, index %d\nOutput: %f\nReference: %f\n",
            i, j, output[i][j], ref_output[i][j]);
        pass = false;
      }
    }
  }


  printf("\nVerification: %s\n", pass ? "PASS" : "FAIL");
}

// Free the resources allocated during initialization
void cleanup() {
  for(unsigned i = 0; i < num_devices; ++i) {
    if(kernel && kernel[i]) {
      clReleaseKernel(kernel[i]);
    }
    if(queue && queue[i]) {
      clReleaseCommandQueue(queue[i]);
    }

    if(input_a_buf && input_a_buf[i]) {
      clReleaseMemObject(input_a_buf[i]);
    }
    if(input_b_buf && input_b_buf[i]) {
      clReleaseMemObject(input_b_buf[i]);
    }
    if(output_buf && output_buf[i]) {
      clReleaseMemObject(output_buf[i]);
    }

  }

  if(program) {
    clReleaseProgram(program);
  }
  if(context) {
    clReleaseContext(context);
  }
}

