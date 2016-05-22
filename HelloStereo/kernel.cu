#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.hpp"

#include <stdio.h>
#include <typeinfo>

#include "opencv2/opencv.hpp"

using namespace cv;

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);


__global__ void negateKernel(unsigned char* img, int N)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i<N)
		img[i] = 255 - img[i];
}

__global__ void erodeKernel(unsigned char* img, unsigned char* imgout)
{

	unsigned char *in, *out;

	int i = blockIdx.x*blockDim.x + threadIdx.x;

	in = img + i;
	out = imgout + i;

	*out = *in;

	if ((blockIdx.x != 0) && (blockIdx.x != gridDim.x - 1) && (threadIdx.x != 0) && (threadIdx.x != blockDim.x - 1))
	{
		for (int k = -1; k <= 1; k++)
		{
			for (int l = -1; l <= 1; l++)
			{
				if (*(in + k*blockDim.x + l) < *out)
				{
					*out = *(in + k*blockDim.x + l);
				}
			}
		}
	}

}

__global__ void dilateKernel(unsigned char* img, unsigned char* imgout)
{

	unsigned char *in, *out;

	int i = blockIdx.x*blockDim.x + threadIdx.x;

	in = img + i;
	out = imgout + i;

	*out = *in;

	if ((blockIdx.x != 0) && (blockIdx.x != gridDim.x - 1) && (threadIdx.x != 0) && (threadIdx.x != blockDim.x - 1))
	{
		for (int k = -1; k <= 1; k++)
		{
			for (int l = -1; l <= 1; l++)
			{
				if (*(in + k*blockDim.x + l) > *out)
				{
					*out = *(in + k*blockDim.x + l);
				}
			}
		}
	}

}


__global__ void ct11Kernel(unsigned char* img, unsigned long* imgout)
{

	unsigned char *in;
	unsigned long * out;

	int i = blockIdx.x*blockDim.x + threadIdx.x,BitCnt = 0;

	in = img + i;
	out = imgout + i * 2;

	if ((blockIdx.x > 4) && (blockIdx.x < gridDim.x - 5) && (threadIdx.x > 4) && (threadIdx.x < blockDim.x - 5))
	{
		for (int k = -5; k <= 5; k++)
		{
			for (int l = -5; l <= 5; l++)
			{
				if (~(k == 0 && l == 0))
				{
					*out = *out << 1;

					if (*(in + k*blockDim.x + l) < *in)
					{
						*out = *out + 1;
					}

					BitCnt++;

					if (BitCnt % 64 == 0)
						out = out + 1;
					
				}
			}
		}
	}

}

__global__ void hammKernel(unsigned short* hamm)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	hamm[i] = 28561;
}

__global__ void xorKernel(unsigned long* img1, unsigned long* img2, unsigned char* xorsum, int Dvalue)
{
	unsigned long xor;
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (threadIdx.x < blockDim.x - Dvalue)
	{
		xor = img1[(i+Dvalue) * 2] ^ img2[i * 2];
		xorsum[i] = __popcll(xor);
		xor = img1[(i + Dvalue) * 2 + 1] ^ img2[i * 2 + 1];
		xorsum[i] += __popcll(xor);
	}

}

__global__ void yaddKernel(unsigned char* xorsum, unsigned short* hammy, int Dvalue)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned short *hammyc; 
	hammyc = hammy + i;
	*hammyc = 0;

	if ((blockIdx.x > 5) && (blockIdx.x < gridDim.x - 7) && (threadIdx.x < blockDim.x - Dvalue))
	{
		for (int k = -6; k <= 6; k++)
		{
			*hammyc += *(xorsum + (i + k)*blockDim.x);
		}

	}

}

__global__ void dmKernel(unsigned short* hammy, unsigned short* hamm, unsigned char* dm, int Dvalue)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned short hammc = 0;

	if ((blockIdx.x > 5) && (blockIdx.x < gridDim.x - 7) && (threadIdx.x > 5) && (threadIdx.x < blockDim.x - Dvalue))
	{
		for (int k = -6; k <= 6; k++)
		{
			hammc += *(hammy + i + k);
		}

		if (hammc < hamm[i])
		{
			hamm[i] = hammc;
			dm[i] = Dvalue;
		}
	}

}


int main()
{
		VideoCapture cap(1),cap2(2); // open the default camera
	if (!cap.isOpened() || !cap2.isOpened())  // check if we succeeded
		return -1;

	namedWindow("Camera 1", 1);
	namedWindow("Camera 2", 1);
	namedWindow("DM", 1);

	int N, row, column;
	
	row = cap.get(CAP_PROP_FRAME_HEIGHT);
	column = cap.get(CAP_PROP_FRAME_WIDTH);
	N = row*column;

	std::cout << row << std::endl<< column << std::endl;

	unsigned char *d_x, *d_y,*x;
	unsigned short *hammy, *hamm;
	unsigned long *ct_1, *ct_2;

	x = (unsigned char*)malloc(N*sizeof(unsigned char));

	cudaMalloc(&d_x, N*sizeof(unsigned char));
	cudaMalloc(&d_x, N*sizeof(unsigned char));
	cudaMalloc(&hammy, N*sizeof(unsigned short));
	cudaMalloc(&hamm, N*sizeof(unsigned short));
	cudaMalloc(&ct_1, N*2*sizeof(unsigned long));
	cudaMalloc(&ct_2, N*2*sizeof(unsigned long));

	Mat frame,gray;
	for (;;)
	{
		cap >> frame; // get a new frame from camera
		cvtColor(frame, gray, CV_BGR2GRAY);

		x = gray.data;
		
		cudaMemcpy(d_x, x, N*sizeof(unsigned char), cudaMemcpyHostToDevice);
		negateKernel << <(N + 255) / 256, 256 >> >(d_x, N);

/*		erodeKernel <<<row, column >>>(d_x, d_y);
		dilateKernel << <row, column >> >(d_y, d_x);
		ct11Kernel << <row, column >> >(d_x, ct_1);
*/
		cudaMemcpy(x, d_x, N*sizeof(unsigned char), cudaMemcpyDeviceToHost);

		imshow("Camera 1",gray);


/*		cap2 >> frame; // get a new frame from camera
		cvtColor(frame, gray, CV_BGR2GRAY);

		cudaMemcpy(d_x, gray.data, N*sizeof(unsigned char), cudaMemcpyHostToDevice);
		erodeKernel << <row, column >> >(d_x, d_y);
		dilateKernel << <row, column >> >(d_y, d_x);
		ct11Kernel << <row, column >> >(d_x, ct_2);
		cudaMemcpy(gray.data, d_x, N*sizeof(unsigned char), cudaMemcpyDeviceToHost);

		imshow("Camera 2", gray);

		hammKernel << <row, column >> >(hamm);

		for (int Dvalue = 100; Dvalue >= 0; Dvalue--)
		{
			xorKernel << <row, column >> >(ct_1, ct_2, d_x, Dvalue);
			yaddKernel << <row, column >> >(d_x, hammy, Dvalue);
			dmKernel << <row, column >> >(hammy, hamm, d_y, Dvalue);
		}

		cudaMemcpy(x, d_y, N*sizeof(unsigned char), cudaMemcpyDeviceToHost);

		gray = Mat(row, column, CV_8UC1, x);

		imshow("DM", gray);
*/

		if (waitKey(1) >= 0) break;
	}

	return 0;
}