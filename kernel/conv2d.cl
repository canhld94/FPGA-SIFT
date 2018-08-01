/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//
// Copyright (C) 2018, Network and Computing Laboratory, KAIST, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

/**********************************************************************************************
 Implementation of Gaussian kernels is based on the Xilinx convolution kernel, below is
 its license
 **********************************************************************************************/ 

/**********
Copyright (c) 2018, Xilinx, Inc.
All rights reserved.
Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/


#define FLEN0 5
#define FLEN1 5
#define FLEN2 7
#define FLEN3 9
#define FLEN4 13



#define COLS 1024
#define ROWS 1024

#define B (8)

#define M(x) (((x)-1)/(B) + 1)
#define REG_WIDTH0 (M(FLEN0+B-1)*B)
#define REG_WIDTH1 (M(FLEN1+B-1)*B)
#define REG_WIDTH2 (M(FLEN2+B-1)*B)
#define REG_WIDTH3 (M(FLEN3+B-1)*B)
#define REG_WIDTH4 (M(FLEN4+B-1)*B)

#if(B == 32)
typedef uint16 bus_t;
#elif(B == 16)
typedef uint8 bus_t;
#elif(B == 8)
typedef uint4 bus_t;
#elif(B == 4)
typedef uint2 bus_t;
#elif(B == 2)
typedef uint bus_t;
#endif

typedef short pixel_t;


typedef union {
	bus_t b;
	pixel_t s[B];
} bus_to_pixel_t_t;

void bus_to_pixel_t(bus_t in, pixel_t out[B]) {
	bus_to_pixel_t_t val;

	val.b = in;

	for(uint i = 0; i < B; i++) {
		out[i] = val.s[i];
	}
}

bus_t pixel_t_to_bus(pixel_t in[B]) {
	bus_to_pixel_t_t val;

	for(uint i = 0; i < B; i++) {
		val.s[i] = in[i];
	}

	return val.b;
}

__attribute__((reqd_work_group_size(1,1,1)))
__kernel void gaussian_s0(
   __global bus_t *input,
   __global bus_t *output,
   uint rows,
   uint cols,
   uint oct
) {
	pixel_t coef_buf[FLEN0*FLEN0]
#ifdef __xilinx__
		__attribute__((xcl_array_partition(complete, 1)))
#endif
		= { 3 , 30 , 59 , 30 , 3 ,
        30 , 228 , 448 , 228 , 30 ,
        59 , 448 , 880 , 448 , 59 ,
        30 , 228 , 448 , 228 , 30 ,
        3 , 30 , 59 , 30 , 3};

	/* Pad registers to align line_buf read/write */
	pixel_t line_reg[FLEN0][REG_WIDTH0]
#ifdef __xilinx__
		__attribute__((xcl_array_partition(complete,1)))
		__attribute__((xcl_array_partition(complete,2)))
#endif
		;
	/* Line buffers to store values */
	pixel_t line_buf[FLEN0-1][M(COLS-REG_WIDTH0)*B]
#ifdef __xilinx__
		__attribute__((xcl_array_partition(complete, 1)))
		__attribute__((xcl_array_partition(cyclic, B, 2)))
#endif
		;

	uint w;
	switch (oct) {
		case 0: w = COLS;
			break;
		case 1: w = COLS >> 1;
			break;
		case 2: w = COLS >> 2;
			break;
		case 3: w = COLS >> 3;
			break;
		case 4: w = COLS >> 4;
			break;
		default: w = COLS;
			break;
	}

#ifdef __xilinx__
	__attribute__((xcl_pipeline_loop))
#endif
	for(uint i = 0; i < M(rows*cols); i++) {
		pixel_t input_buf[B];

		/* Read pixels from the input image */
		bus_to_pixel_t(input[i], input_buf);

		/* Rotate Buffers */
		for(uint y = 0; y < FLEN0-1; y++) {
			/* Move the line reg B pixels at a time */
			for(uint x = 0; x < REG_WIDTH0 - B; x++) {
				line_reg[y][x] = line_reg[y][x+B];
			}
			/* Add values from line_buf to end of regs */
			for(uint j = 0; j < B; j++) {
				line_reg[y][(REG_WIDTH0 - B) + j] = line_buf[y][j + B*(i % (M(w-REG_WIDTH0)))];
			}
			/* Write values from the start of the next line to the line_buf */
			for(uint j = 0; j < B; j++) {
				line_buf[y][j + B*(i % (M(w-REG_WIDTH0)))] = line_reg[y+1][j];
			}
		}
		/* On last line rotate regs */
		for(uint x = 0; x < REG_WIDTH0 - B; x++) {
			line_reg[FLEN0-1][x] = line_reg[FLEN0-1][x+B];
		}
		/* Add the new input data to the end */
		for(uint j = 0; j < B; j++) {
			line_reg[FLEN0-1][(REG_WIDTH0 - B) + j] = input_buf[j];
		}

		pixel_t filter_sums[B];

		for(uint j = 0; j < B; j++) {
			int sum = 0;
			for(uint y = 0; y < FLEN0; y++) {
				for(uint x = 0; x  < FLEN0; x++) {
					const uint offset = REG_WIDTH0 - FLEN0 - B + 1;
					pixel_t val = line_reg[y][offset + x + j];
					sum +=  coef_buf[y*FLEN0 + x] * val;
				}
			}

			/* Handle Saturation */
			if (sum > INT_MAX) {
				sum = INT_MAX;
			} else if (sum < INT_MIN) {
				sum = INT_MIN;
			}

			filter_sums[j] = sum >> 13;
		}
		output[i] = pixel_t_to_bus(filter_sums);
//		input[i] = pixel_t_to_bus(filter_sums);
	}
}

__attribute__((reqd_work_group_size(1,1,1)))
__kernel void gaussian_s1(
   __global bus_t *input,
   __global bus_t *output,
   uint rows,
   uint cols,
   uint oct
) {
	pixel_t coef_buf[FLEN1*FLEN1]
#ifdef __xilinx__
		__attribute__((xcl_array_partition(complete, 1)))
#endif
		= { 0 , 8 , 22 , 8 , 0 ,
        8 , 172 , 479 , 172 , 8 ,
        22 , 479 , 1330 , 479 , 22 ,
        8 , 172 , 479 , 172 , 8 ,
        0 , 8 , 22 , 8 , 0};

	/* Pad registers to align line_buf read/write */
	pixel_t line_reg[FLEN1][REG_WIDTH1]
#ifdef __xilinx__
		__attribute__((xcl_array_partition(complete,1)))
		__attribute__((xcl_array_partition(complete,2)))
#endif
		;
	/* Line buffers to store values */
	pixel_t line_buf[FLEN1-1][M(COLS-REG_WIDTH1)*B]
#ifdef __xilinx__
		__attribute__((xcl_array_partition(complete, 1)))
		__attribute__((xcl_array_partition(cyclic, B, 2)))
#endif
		;

	uint w;
	switch (oct) {
		case 0: w = COLS;
			break;
		case 1: w = COLS >> 1;
			break;
		case 2: w = COLS >> 2;
			break;
		case 3: w = COLS >> 3;
			break;
		case 4: w = COLS >> 4;
			break;
		default: w = COLS;
			break;
	}

#ifdef __xilinx__
	__attribute__((xcl_pipeline_loop))
#endif
	for(uint i = 0; i < M(rows*cols); i++) {
		pixel_t input_buf[B];

		/* Read pixels from the input image */
		bus_to_pixel_t(input[i], input_buf);

		/* Rotate Buffers */
		for(uint y = 0; y < FLEN1-1; y++) {
			/* Move the line reg B pixels at a time */
			for(uint x = 0; x < REG_WIDTH1 - B; x++) {
				line_reg[y][x] = line_reg[y][x+B];
			}
			/* Add values from line_buf to end of regs */
			for(uint j = 0; j < B; j++) {
				line_reg[y][(REG_WIDTH1 - B) + j] = line_buf[y][j + B*(i % (M(w-REG_WIDTH1)))];
			}
			/* Write values from the start of the next line to the line_buf */
			for(uint j = 0; j < B; j++) {
				line_buf[y][j + B*(i % (M(w-REG_WIDTH1)))] = line_reg[y+1][j];
			}
		}
		/* On last line rotate regs */
		for(uint x = 0; x < REG_WIDTH1 - B; x++) {
			line_reg[FLEN1-1][x] = line_reg[FLEN1-1][x+B];
		}
		/* Add the new input data to the end */
		for(uint j = 0; j < B; j++) {
			line_reg[FLEN1-1][(REG_WIDTH1 - B) + j] = input_buf[j];
		}

		pixel_t filter_sums[B];

		for(uint j = 0; j < B; j++) {
			int sum = 0;
			for(uint y = 0; y < FLEN1; y++) {
				for(uint x = 0; x < FLEN1; x++) {
					const uint offset = REG_WIDTH1 - FLEN1 - B + 1;
					pixel_t val = line_reg[y][offset + x + j];
					sum +=  coef_buf[y*FLEN1 + x] * val;
				}
			}

			/* Handle Saturation */
			if (sum > INT_MAX) {
				sum = INT_MAX;
			} else if (sum < INT_MIN) {
				sum = INT_MIN;
			}

			filter_sums[j] = sum >> 13;
		}

		output[i] = pixel_t_to_bus(filter_sums);
//		input[i] = pixel_t_to_bus(filter_sums);
	}
}

__attribute__((reqd_work_group_size(1,1,1)))
__kernel void gaussian_s2(
   __global bus_t *input,
   __global bus_t *output,
   uint rows,
   uint cols,
   uint oct
) {
	pixel_t coef_buf[FLEN2*FLEN2]
#ifdef __xilinx__
		__attribute__((xcl_array_partition(complete, 1)))
#endif
		= { 0 , 2 , 9 , 14 , 9 , 2 , 0 ,
        2 , 21 , 70 , 105 , 70 , 21 , 2 ,
        9 , 70 , 234 , 348 , 234 , 70 , 9 ,
        14 , 105 , 348 , 518 , 348 , 105 , 14 ,
        9 , 70 , 234 , 348 , 234 , 70 , 9 ,
        2 , 21 , 70 , 105 , 70 , 21 , 2 ,
        0 , 2 , 9 , 14 , 9 , 2 , 0};

	/* Pad registers to align line_buf read/write */
	pixel_t line_reg[FLEN2][REG_WIDTH2]
#ifdef __xilinx__
		__attribute__((xcl_array_partition(complete,1)))
		__attribute__((xcl_array_partition(complete,2)))
#endif
		;
	/* Line buffers to store values */
	pixel_t line_buf[FLEN2-1][M(COLS-REG_WIDTH2)*B]
#ifdef __xilinx__
		__attribute__((xcl_array_partition(complete, 1)))
		__attribute__((xcl_array_partition(cyclic, B, 2)))
#endif
		;

	uint w;
	switch (oct) {
		case 0: w = COLS;
			break;
		case 1: w = COLS >> 1;
			break;
		case 2: w = COLS >> 2;
			break;
		case 3: w = COLS >> 3;
			break;
		case 4: w = COLS >> 4;
			break;
		default: w = COLS;
			break;
	}

#ifdef __xilinx__
	__attribute__((xcl_pipeline_loop))
#endif
	for(uint i = 0; i < M(rows*cols); i++) {
		pixel_t input_buf[B];

		/* Read pixels from the input image */
		bus_to_pixel_t(input[i], input_buf);

		/* Rotate Buffers */
		for(uint y = 0; y < FLEN2-1; y++) {
			/* Move the line reg B pixels at a time */
			for(uint x = 0; x < REG_WIDTH2 - B; x++) {
				line_reg[y][x] = line_reg[y][x+B];
			}
			/* Add values from line_buf to end of regs */
			for(uint j = 0; j < B; j++) {
				line_reg[y][(REG_WIDTH2 - B) + j] = line_buf[y][j + B*(i % (M(w-REG_WIDTH2)))];
			}
			/* Write values from the start of the next line to the line_buf */
			for(uint j = 0; j < B; j++) {
				line_buf[y][j + B*(i % (M(w-REG_WIDTH2)))] = line_reg[y+1][j];
			}
		}
		/* On last line rotate regs */
		for(uint x = 0; x < REG_WIDTH2 - B; x++) {
			line_reg[FLEN2-1][x] = line_reg[FLEN2-1][x+B];
		}
		/* Add the new input data to the end */
		for(uint j = 0; j < B; j++) {
			line_reg[FLEN2-1][(REG_WIDTH2 - B) + j] = input_buf[j];
		}

		pixel_t filter_sums[B];

		for(uint j = 0; j < B; j++) {
			int sum = 0;
			for(uint y = 0; y < FLEN2; y++) {
				for(uint x = 0; x < FLEN2; x++) {
					const uint offset = REG_WIDTH2 - FLEN2 - B + 1;
					pixel_t val = line_reg[y][offset + x + j];
					sum +=  coef_buf[y*FLEN2 + x] * val;
				}
			}

			/* Handle Saturation */
			if (sum > INT_MAX) {
				sum = INT_MAX;
			} else if (sum < INT_MIN) {
				sum = INT_MIN;
			}

			filter_sums[j] = sum >> 13;
		}

		output[i] = pixel_t_to_bus(filter_sums);
//		input[i] = pixel_t_to_bus(filter_sums);
	}
}

__attribute__((reqd_work_group_size(1,1,1)))
__kernel void gaussian_s3(
   __global bus_t *input,
   __global bus_t *output,
   uint rows,
   uint cols,
   uint oct
) {
	pixel_t coef_buf[FLEN3*FLEN3]
#ifdef __xilinx__
		__attribute__((xcl_array_partition(complete, 1)))
#endif
		= { 1 , 4 , 10 , 15 , 18 , 15 , 10 , 4 , 1 ,
        4 , 13 , 28 , 44 , 51 , 44 , 28 , 13 , 4 ,
        10 , 28 , 59 , 91 , 106 , 91 , 59 , 28 , 10 ,
        15 , 44 , 91 , 141 , 164 , 141 , 91 , 44 , 15 ,
        18 , 51 , 106 , 164 , 190 , 164 , 106 , 51 , 18 ,
        15 , 44 , 91 , 141 , 164 , 141 , 91 , 44 , 15 ,
        10 , 28 , 59 , 91 , 106 , 91 , 59 , 28 , 10 ,
        4 , 13 , 28 , 44 , 51 , 44 , 28 , 13 , 4 ,
        1 , 4 , 10 , 15 , 18 , 15 , 10 , 4 , 1};

	/* Pad registers to align line_buf read/write */
	pixel_t line_reg[FLEN3][REG_WIDTH3]
#ifdef __xilinx__
		__attribute__((xcl_array_partition(complete,1)))
		__attribute__((xcl_array_partition(complete,2)))
#endif
		;
	/* Line buffers to store values */
	pixel_t line_buf[FLEN3-1][M(COLS-REG_WIDTH3)*B]
#ifdef __xilinx__
		__attribute__((xcl_array_partition(complete, 1)))
		__attribute__((xcl_array_partition(cyclic, B, 2)))
#endif
		;

	uint w;
	switch (oct) {
		case 0: w = COLS;
			break;
		case 1: w = COLS >> 1;
			break;
		case 2: w = COLS >> 2;
			break;
		case 3: w = COLS >> 3;
			break;
		case 4: w = COLS >> 4;
			break;
		default: w = COLS;
			break;
	}

#ifdef __xilinx__
	__attribute__((xcl_pipeline_loop))
#endif
	for(uint i = 0; i < M(rows*cols); i++) {
		pixel_t input_buf[B];

		/* Read pixels from the input image */
		bus_to_pixel_t(input[i], input_buf);

		/* Rotate Buffers */
		for(uint y = 0; y < FLEN3-1; y++) {
			/* Move the line reg B pixels at a time */
			for(uint x = 0; x < REG_WIDTH3 - B; x++) {
				line_reg[y][x] = line_reg[y][x+B];
			}
			/* Add values from line_buf to end of regs */
			for(uint j = 0; j < B; j++) {
				line_reg[y][(REG_WIDTH3 - B) + j] = line_buf[y][j + B*(i % (M(w-REG_WIDTH3)))];
			}
			/* Write values from the start of the next line to the line_buf */
			for(uint j = 0; j < B; j++) {
				line_buf[y][j + B*(i % (M(w-REG_WIDTH3)))] = line_reg[y+1][j];
			}
		}
		/* On last line rotate regs */
		for(uint x = 0; x < REG_WIDTH3 - B; x++) {
			line_reg[FLEN3-1][x] = line_reg[FLEN3-1][x+B];
		}
		/* Add the new input data to the end */
		for(uint j = 0; j < B; j++) {
			line_reg[FLEN3-1][(REG_WIDTH3 - B) + j] = input_buf[j];
		}

		pixel_t filter_sums[B];

		for(uint j = 0; j < B; j++) {
			int sum = 0;
			for(uint y = 0; y < FLEN3; y++) {
				for(uint x = 0; x < FLEN3; x++) {
					const uint offset = REG_WIDTH3 - FLEN3 - B + 1;
					pixel_t val = line_reg[y][offset + x + j];
					sum +=  coef_buf[y*FLEN3 + x] * val;
				}
			}

			/* Handle Saturation */
			if (sum > INT_MAX) {
				sum = INT_MAX;
			} else if (sum < INT_MIN) {
				sum = INT_MIN;
			}

			filter_sums[j] = sum >> 13;
		}

		output[i] = pixel_t_to_bus(filter_sums);
//		input[i] = pixel_t_to_bus(filter_sums);
	}
}


__attribute__((reqd_work_group_size(1,1,1)))
__kernel void gaussian_s4(
   __global bus_t *input,
   __global bus_t *output,
   uint rows,
   uint cols,
   uint oct
) {
	pixel_t coef_buf[FLEN4*FLEN4]
#ifdef __xilinx__
		__attribute__((xcl_array_partition(complete, 1)))
#endif
		= { 0 , 1 , 2 , 4 , 5 , 7 , 7 , 7 , 5 , 4 , 2 , 1 , 0 ,
        1 , 2 , 5 , 8 , 12 , 15 , 16 , 15 , 12 , 8 , 5 , 2 , 1 ,
        2 , 5 , 10 , 16 , 22 , 27 , 29 , 27 , 22 , 16 , 10 , 5 , 2 ,
        4 , 8 , 16 , 26 , 36 , 44 , 48 , 44 , 36 , 26 , 16 , 8 , 4 ,
        5 , 12 , 22 , 36 , 51 , 63 , 67 , 63 , 51 , 36 , 22 , 12 , 5 ,
        7 , 15 , 27 , 44 , 63 , 77 , 82 , 77 , 63 , 44 , 27 , 15 , 7 ,
        7 , 16 , 29 , 48 , 67 , 82 , 88 , 82 , 67 , 48 , 29 , 16 , 7 ,
        7 , 15 , 27 , 44 , 63 , 77 , 82 , 77 , 63 , 44 , 27 , 15 , 7 ,
        5 , 12 , 22 , 36 , 51 , 63 , 67 , 63 , 51 , 36 , 22 , 12 , 5 ,
        4 , 8 , 16 , 26 , 36 , 44 , 48 , 44 , 36 , 26 , 16 , 8 , 4 ,
        2 , 5 , 10 , 16 , 22 , 27 , 29 , 27 , 22 , 16 , 10 , 5 , 2 ,
        1 , 2 , 5 , 8 , 12 , 15 , 16 , 15 , 12 , 8 , 5 , 2 , 1 ,
        0 , 1 , 2 , 4 , 5 , 7 , 7 , 7 , 5 , 4 , 2 , 1 , 0};

	/* Pad registers to align line_buf read/write */
	pixel_t line_reg[FLEN4][REG_WIDTH4]
#ifdef __xilinx__
		__attribute__((xcl_array_partition(complete,1)))
		__attribute__((xcl_array_partition(complete,2)))
#endif
		;
	/* Line buffers to store values */
	pixel_t line_buf[FLEN4-1][M(COLS-REG_WIDTH4)*B]
#ifdef __xilinx__
		__attribute__((xcl_array_partition(complete, 1)))
		__attribute__((xcl_array_partition(cyclic, B, 2)))
#endif
		;

	uint w;
	switch (oct) {
		case 0: w = COLS;
			break;
		case 1: w = COLS >> 1;
			break;
		case 2: w = COLS >> 2;
			break;
		case 3: w = COLS >> 3;
			break;
		case 4: w = COLS >> 4;
			break;
		default: w = COLS;
			break;
	}

#ifdef __xilinx__
	__attribute__((xcl_pipeline_loop))
#endif
	for(uint i = 0; i < M(rows*cols); i++) {
		pixel_t input_buf[B];

		/* Read pixels from the input image */
		bus_to_pixel_t(input[i], input_buf);

		/* Rotate Buffers */
		for(uint y = 0; y < FLEN4-1; y++) {
			/* Move the line reg B pixels at a time */
			for(uint x = 0; x < REG_WIDTH4 - B; x++) {
				line_reg[y][x] = line_reg[y][x+B];
			}
			/* Add values from line_buf to end of regs */
			for(uint j = 0; j < B; j++) {
				line_reg[y][(REG_WIDTH4 - B) + j] = line_buf[y][j + B*(i % (M(w-REG_WIDTH4)))];
			}
			/* Write values from the start of the next line to the line_buf */
			for(uint j = 0; j < B; j++) {
				line_buf[y][j + B*(i % (M(w-REG_WIDTH4)))] = line_reg[y+1][j];
			}
		}
		/* On last line rotate regs */
		for(uint x = 0; x < REG_WIDTH4 - B; x++) {
			line_reg[FLEN4-1][x] = line_reg[FLEN4-1][x+B];
		}
		/* Add the new input data to the end */
		for(uint j = 0; j < B; j++) {
			line_reg[FLEN4-1][(REG_WIDTH4 - B) + j] = input_buf[j];
		}

		pixel_t filter_sums[B];

		for(uint j = 0; j < B; j++) {
			int sum = 0;
			for(uint y = 0; y < FLEN4; y++) {
				for(uint x = 0; x < FLEN4; x++) {
					const uint offset = REG_WIDTH4 - FLEN4 - B + 1;
//					const uint offset = 0;
					pixel_t val = line_reg[y][offset + x + j];
					sum +=  coef_buf[y*FLEN4 + x] * val;
				}
			}

			/* Handle Saturation */
			if (sum > INT_MAX) {
				sum = INT_MAX;
			} else if (sum < INT_MIN) {
				sum = INT_MIN;
			}

			filter_sums[j] = sum >> 13;
		}

		output[i] = pixel_t_to_bus(filter_sums);
//		input[i] = pixel_t_to_bus(filter_sums);
	}
}





