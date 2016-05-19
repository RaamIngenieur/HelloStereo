#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

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

/*
__global__ void ct11Kernel(unsigned char* img, unsigned long* imgout)
{

	unsigned char *in, *out;

	int i = blockIdx.x*blockDim.x + threadIdx.x;

	in = img + i;
	out = imgout + i;

	*out = *in;

	if ((blockIdx.x > 4) && (blockIdx.x < gridDim.x - 6) && (threadIdx.x > 4) && (threadIdx.x < blockDim.x - 6))
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

void Census11(unsigned long * OPic, UCarray &IPic, int row, int column)
{
	int BitCnt;
	unsigned long * OPicc;

	for (int i = 5; i <= row - 6; i++)
	{
		for (int j = 5; j <= column - 6; j++)
		{
			BitCnt = 0;
			OPicc = OPic + i*column * 2 + j * 2;

			for (int k = -5; k <= 5; k++)
			{
				for (int l = -5; l <= 5; l++)
				{
					if (~(k == 0 && l == 0))
					{
						*OPicc = *OPicc << 1;

						if (*(IPic.Pix + (i + k)*column + j + l) < *(IPic.Pix + i*column + j))
						{
							*OPicc = *OPicc + 1;
						}

						BitCnt++;

						if (BitCnt % 64 == 0)
							OPicc = OPicc + 1;

					}
				}
			}
		}
	}
}
*/


int main()
{
		VideoCapture cap(1),cap2(2); // open the default camera
	if (!cap.isOpened() || !cap2.isOpened())  // check if we succeeded
		return -1;

	namedWindow("Camera 1", 1);
	namedWindow("Camera 2", 1);

	int N, row, column;
	
	row = cap.get(CAP_PROP_FRAME_HEIGHT);
	column = cap.get(CAP_PROP_FRAME_WIDTH);
	N = row*column;

	std::cout << row << std::endl<< column << std::endl;

	unsigned char *x, *y;

	unsigned char *d_x, *d_y;
	unsigned long *ct_1, *ct_2;

	cudaMalloc(&d_x, N*sizeof(unsigned char));
	cudaMalloc(&d_x, N*sizeof(unsigned char));
	cudaMalloc(&ct_1, N*2*sizeof(unsigned long));
	cudaMalloc(&ct_2, N*2*sizeof(unsigned long));

	Mat frame,gray;
	for (;;)
	{
		cap >> frame; // get a new frame from camera
		cvtColor(frame, gray, CV_BGR2GRAY);
		
		cudaMemcpy(d_x, gray.data, N*sizeof(unsigned char), cudaMemcpyHostToDevice);
		erodeKernel <<<row, column >>>(d_x, d_y);
		dilateKernel << <row, column >> >(d_y, d_x);
		cudaMemcpy(gray.data, d_x, N*sizeof(unsigned char), cudaMemcpyDeviceToHost);

		imshow("Camera 1", gray);


		cap2 >> frame; // get a new frame from camera
		cvtColor(frame, gray, CV_BGR2GRAY);

		cudaMemcpy(d_x, gray.data, N*sizeof(unsigned char), cudaMemcpyHostToDevice);
		erodeKernel << <row, column >> >(d_x, d_y);
		dilateKernel << <row, column >> >(d_y, d_x);
		cudaMemcpy(gray.data, d_x, N*sizeof(unsigned char), cudaMemcpyDeviceToHost);

		imshow("Camera 2", gray);


		if (waitKey(30) >= 0) break;
	}

	return 0;
}