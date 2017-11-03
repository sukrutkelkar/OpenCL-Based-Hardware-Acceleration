/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Matrix Multiplication Kernel
//
// Copyright (C) 2013-2016 Altera Corporation, San Jose, California, USA. All rights reserved.
// Reference: https://www.altera.com/support/support-resources/design-examples/design-software/opencl/vector-addition.html
//
// Kernel Modification by Sukrut Kelkar
//
// Date: 5th March 2017
//
// Project: Hardware Acceleration using OpenCL on FPGA 
//
// Description:
// ------------
// Multiplying one row of matrix A with one column of Matrix B 
// to calculate a single element of Matrix C
// 
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


 // ACL kernel for Matrix Multiplication
__kernel void matrix_mul(int rA,
						 int cB,	
						 __global const int *A, 
                         __global const int *B, 
                         __global int *restrict C)
{
    // get index of the work item
    //int index = get_global_id(0); Gives the unique index of the workItem
	int globalRow = get_global_id(0); // Row ID of C (0..M)  //i
    int globalCol = get_global_id(1); // Col ID of C (0..N)		//j
	
	// Compute a single element (loop over K)
    int temp = 0;
    for (int k=0; k<cB; k++) {
        temp += A[globalRow*rA + k] * B[k*cB + globalCol];
    }
 
    // Store the result
    C[globalRow*rA + globalCol] = temp;
	//printf("\t%d",C[globalRow*rA + globalCol]);
}
