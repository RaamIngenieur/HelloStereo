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

	int row = gridDim.x, column = blockDim.x, r = blockIdx.x, c = threadIdx.x;

	int i = r*column + c;

	in = img + i;
	out = imgout + i;

	*out = *in;

	if ((r > 1) && (r < (row - 2)) && (c > 1) && (c < (column - 2)))
	{
		for (int k = -2; k <= 2; k++)
		{
			for (int l = -2; l <= 2; l++)
			{
				if (*(in + k*column + l) < *out)
				{
					*out = *(in + k*column + l);
				}
			}
		}
	}


}

__global__ void dilateKernel(unsigned char* img, unsigned char* imgout)
{

	unsigned char *in, *out;

	int row = gridDim.x, column = blockDim.x, r = blockIdx.x, c = threadIdx.x;

	int i = r*column + c;

	in = img + i;
	out = imgout + i;

	*out = *in;


	if ((r > 1) && (r < (row - 2)) && (c > 1) && (c < (column - 2)))
	{
		for (int k = -2; k <= 2; k++)
		{
			for (int l = -2; l <= 2; l++)
			{
				if (*(in + k*column + l) > *out)
				{
					*out = *(in + k*column + l);
				}
			}
		}
	}


}


__global__ void ct11Kernel(unsigned char* img, unsigned long* imgout)
{

	unsigned char *in;
	unsigned long * out;

	int row = gridDim.x, column = blockDim.x, r = blockIdx.x, c = threadIdx.x;

	int i = r*column + c, BitCnt=0;

	in = img + i;
	out = imgout + i * 2;

	if ((r > 4) && (r < row - 5) && (c > 4) && (c < column - 5))
	{
		for (int k = -5; k <= 5; k++)
		{
			for (int l = -5; l <= 5; l++)
			{
				if (~(k == 0 && l == 0))
				{
					*out = *out << 1;

					if (*(in + k*column + l) < *in)
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
	int r = blockIdx.x,column = blockDim.x, c = threadIdx.x;

	int i = r*column + c;

	if (c < column - Dvalue)
	{
		xor = img1[(i+Dvalue) * 2] ^ img2[i * 2];
		xorsum[i] = __popcll(xor);
		xor = img1[(i + Dvalue) * 2 + 1] ^ img2[i * 2 + 1];
		xorsum[i] += __popcll(xor);
	}

}

__global__ void yaddKernel(unsigned char* xorsum, unsigned short* hammy, int Dvalue)
{
	int row = gridDim.x, column = blockDim.x, r = blockIdx.x, c = threadIdx.x;

	int i = r*column + c;

	unsigned short *hammyc;
	unsigned char *xorc;
	hammyc = hammy + i;
	xorc = xorsum + i;
	*hammyc = 0;

	if ((r > 5) && (r < row - 7) && (c < column - Dvalue))
	{
		for (int k = -6; k <= 6; k++)
		{
			*hammyc += *(xorc + k*column);
		}
		
	}

}

__global__ void dmKernel(unsigned short* hammy, unsigned short* hamm, unsigned char* dm, int Dvalue)
{
	int row = gridDim.x, column = blockDim.x, r = blockIdx.x, c = threadIdx.x;

	int i = r*column + c;

	unsigned short hammc = 0,*hammyc = hammy +i, *hammmc = hamm +i;
	unsigned char* dmc = dm + i;

	if ((r > 5) && (r < row - 7) && (c > 5) && (c < column - Dvalue))
	{
		for (int k = -6; k <= 6; k++)
		{
			hammc += *(hammyc + k);
		}

		if (hammc < *hammmc)
		{
			*hammmc = hammc;
			*dmc = Dvalue;
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
	std::cout << cap2.get(CAP_PROP_FRAME_HEIGHT) << std::endl << cap2.get(CAP_PROP_FRAME_WIDTH) << std::endl;

	unsigned char *d_x1, *d_y1, *d_z1, *d_x2, *d_y2, *d_z2, *xor, *dm;
	unsigned short *hammy, *hamm;
	unsigned long *ct_1, *ct_2;

	cudaMalloc(&d_x1, N*sizeof(unsigned char));
	cudaMalloc(&d_y1, N*sizeof(unsigned char));
	cudaMalloc(&d_z1, N*sizeof(unsigned char));
	cudaMalloc(&d_x2, N*sizeof(unsigned char));
	cudaMalloc(&d_y2, N*sizeof(unsigned char));
	cudaMalloc(&d_z2, N*sizeof(unsigned char));
	cudaMalloc(&xor, N*sizeof(unsigned char));
	cudaMalloc(&dm, N*sizeof(unsigned char));
	cudaMalloc(&hammy, N*sizeof(unsigned short));
	cudaMalloc(&hamm, N*sizeof(unsigned short));
	cudaMalloc(&ct_1, N*2*sizeof(unsigned long));
	cudaMalloc(&ct_2, N*2*sizeof(unsigned long));

	Mat frame, gray, out = Mat(row, column, CV_8UC1, Scalar(255));

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		cvtColor(frame, gray, CV_BGR2GRAY);

		
		cudaMemcpy(d_x1, gray.data, N*sizeof(unsigned char), cudaMemcpyHostToDevice);;

		erodeKernel << <row, column >> >(d_x1, d_y1);
		cudaDeviceSynchronize();
 	    dilateKernel << <row, column >> >(d_y1, d_z1);
		cudaDeviceSynchronize();
		ct11Kernel << <row, column >> >(d_z1, ct_1);

		cudaMemcpy(out.data, d_z1, N*sizeof(unsigned char), cudaMemcpyDeviceToHost);

		imshow("Camera 1",out);


		cap2 >> frame; // get a new frame from camera
		cvtColor(frame, gray, CV_BGR2GRAY);

		cudaMemcpy(d_x2, gray.data, N*sizeof(unsigned char), cudaMemcpyHostToDevice);
		erodeKernel << <row, column >> >(d_x2, d_y2);
		cudaDeviceSynchronize();
		dilateKernel << <row, column >> >(d_y2, d_z2);
		cudaDeviceSynchronize();
		ct11Kernel << <row, column >> >(d_z2, ct_2);
		cudaMemcpy(out.data, d_z2, N*sizeof(unsigned char), cudaMemcpyDeviceToHost);

		imshow("Camera 2", out);

		hammKernel << <row, column >> >(hamm);

		for (int Dvalue = 200; Dvalue >= 0; Dvalue--)
		{
			cudaDeviceSynchronize();
			xorKernel << <row, column >> >(ct_1, ct_2, xor, Dvalue);
			cudaDeviceSynchronize();
			yaddKernel << <row, column >> >(xor, hammy, Dvalue);
			cudaDeviceSynchronize();
			dmKernel << <row, column >> >(hammy, hamm, dm, Dvalue);
		}

		cudaMemcpy(gray.data, dm, N*sizeof(unsigned char), cudaMemcpyDeviceToHost);

		imshow("DM", gray);


		if (waitKey(1) >= 0) break;
	}

	return 0;
}