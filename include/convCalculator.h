#pragma once

#include <iostream>
#include <string.h>
#include "Kernel_GPU.h"
// #include "Kernel.h"

/*
卷积层的计算
*/

template<typename type>
class convCalculator
{
public:
    // Conv2D使用GPU加速的卷积计算
	void Conv2D(type* ImageData, int ImageWidth, int ImageHeight, int ImageChannel, int BatchSize, ConvKernel<type>* CK);
	//  void Conv2D(ConvKernel* CK);

	// CPU版本的卷积计算，用于对比GPU版本的计算速度和计算准确性
	void Conv2D_CPU(type* ImageData, int ImageWidth, int ImageHeight, int ImageChannel, int BatchSize, ConvKernel<type>* CK);

	// 一维卷积计算，使用GPU加速
	void Conv1D(type* TextData,int TextLength,int TextChannel, int BatchSize,ConvKernel<type>* CK);

	// 一维卷积计算，使用CPU计算
	void Conv1D_CPU(type* TextData,int TextLength,int TextChannel,int BatchSize,ConvKernel<type>* CK);

	// 三维卷积计算，使用GPU加速
	void Conv3D(type* VideoData,int VideoWidth,int VideoHeight,int VideoDepth,int VideoChannel,int BatchSize,ConvKernel<type>* CK);

	void Conv3D_CPU(type* VideoData,int VideoWidth,int VideoHeight,int VideoDepth,int VideoChannel,int BatchSize,ConvKernel<type>* CK);

	// 将上面三种卷积函数的模板函数实例化为int，float，double三种类型
	void Conv2D_G_int(int* ImageData, int ImageWidth, int ImageHeight, int ImageChannel, int BatchSize, ConvKernel<int>* CK)
	{
		Conv2D((type*)ImageData, ImageWidth, ImageHeight, ImageChannel, BatchSize, (ConvKernel<type>*)CK);
	}

	void Conv2D_G_float(float* ImageData, int ImageWidth, int ImageHeight, int ImageChannel, int BatchSize, ConvKernel<float>* CK)
	{
		Conv2D((type*)ImageData, ImageWidth, ImageHeight, ImageChannel, BatchSize, (ConvKernel<type>*)CK);
	}

	void Conv2D_G_double(double* ImageData, int ImageWidth, int ImageHeight, int ImageChannel, int BatchSize, ConvKernel<double>* CK)
	{
		Conv2D((type*)ImageData, ImageWidth, ImageHeight, ImageChannel, BatchSize, (ConvKernel<type>*)CK);
	}

	void Conv2D_C_int(int* ImageData, int ImageWidth, int ImageHeight, int ImageChannel, int BatchSize, ConvKernel<int>* CK)
	{
		Conv2D_CPU((type*)ImageData, ImageWidth, ImageHeight, ImageChannel, BatchSize, (ConvKernel<type>*)CK);
	}

	void Conv2D_C_float(float* ImageData, int ImageWidth, int ImageHeight, int ImageChannel, int BatchSize, ConvKernel<float>* CK)
	{
		Conv2D_CPU((type*)ImageData, ImageWidth, ImageHeight, ImageChannel, BatchSize, (ConvKernel<type>*)CK);
	}

	void Conv2D_C_double(double* ImageData, int ImageWidth, int ImageHeight, int ImageChannel, int BatchSize, ConvKernel<double>* CK)
	{
		Conv2D_CPU((type*)ImageData, ImageWidth, ImageHeight, ImageChannel, BatchSize, (ConvKernel<type>*)CK);
	}

	void Conv1D_G_int(int* TextData,int TextLength,int TextChannel, int BatchSize,ConvKernel<int>* CK)
	{
		Conv1D((type*)TextData, TextLength, TextChannel, BatchSize, (ConvKernel<type>*)CK);
	}

	void Conv1D_G_float(float* TextData,int TextLength,int TextChannel, int BatchSize,ConvKernel<float>* CK)
	{
		Conv1D((type*)TextData, TextLength, TextChannel, BatchSize, (ConvKernel<type>*)CK);
	}

	void Conv1D_G_double(double* TextData,int TextLength,int TextChannel, int BatchSize,ConvKernel<double>* CK)
	{
		Conv1D((type*)TextData, TextLength, TextChannel, BatchSize, (ConvKernel<type>*)CK);
	}

	void Conv1D_C_int(int* TextData,int TextLength,int TextChannel,int BatchSize,ConvKernel<int>* CK)
	{
		Conv1D_CPU((type*)TextData, TextLength, TextChannel, BatchSize, (ConvKernel<type>*)CK);
	}

	void Conv1D_C_float(float* TextData,int TextLength,int TextChannel,int BatchSize,ConvKernel<float>* CK)
	{
		Conv1D_CPU((type*)TextData, TextLength, TextChannel, BatchSize, (ConvKernel<type>*)CK);
	}

	void Conv1D_C_double(double* TextData,int TextLength,int TextChannel,int BatchSize,ConvKernel<double>* CK)
	{
		Conv1D_CPU((type*)TextData, TextLength, TextChannel, BatchSize, (ConvKernel<type>*)CK);
	}

	void Conv3D_G_int(int* VideoData,int VideoWidth,int VideoHeight,int VideoDepth,int VideoChannel,int BatchSize,ConvKernel<int>* CK)
	{
		Conv3D((type*)VideoData, VideoWidth, VideoHeight, VideoDepth, VideoChannel, BatchSize, (ConvKernel<type>*)CK);
	}

	void Conv3D_G_float(float* VideoData,int VideoWidth,int VideoHeight,int VideoDepth,int VideoChannel,int BatchSize,ConvKernel<float>* CK)
	{
		Conv3D((type*)VideoData, VideoWidth, VideoHeight, VideoDepth, VideoChannel, BatchSize, (ConvKernel<type>*)CK);
	}

	void Conv3D_G_double(double* VideoData,int VideoWidth,int VideoHeight,int VideoDepth,int VideoChannel,int BatchSize,ConvKernel<double>* CK)
	{
		Conv3D((type*)VideoData, VideoWidth, VideoHeight, VideoDepth, VideoChannel, BatchSize, (ConvKernel<type>*)CK);
	}

	void Conv3D_C_int(int* VideoData,int VideoWidth,int VideoHeight,int VideoDepth,int VideoChannel,int BatchSize,ConvKernel<int>* CK)
	{
		Conv3D_CPU((type*)VideoData, VideoWidth, VideoHeight, VideoDepth, VideoChannel, BatchSize, (ConvKernel<type>*)CK);
	}

	void Conv3D_C_float(float* VideoData,int VideoWidth,int VideoHeight,int VideoDepth,int VideoChannel,int BatchSize,ConvKernel<float>* CK)
	{
		Conv3D_CPU((type*)VideoData, VideoWidth, VideoHeight, VideoDepth, VideoChannel, BatchSize, (ConvKernel<type>*)CK);
	}

	void Conv3D_C_double(double* VideoData,int VideoWidth,int VideoHeight,int VideoDepth,int VideoChannel,int BatchSize,ConvKernel<double>* CK)
	{
		Conv3D_CPU((type*)VideoData, VideoWidth, VideoHeight, VideoDepth, VideoChannel, BatchSize, (ConvKernel<type>*)CK);
	}

	// 构造函数和析构函数
	convCalculator()
	{
		mInWidth = 0;
		mInHeight = 0;
		mInDepth = 0;
		mInChannel = 0;
		mOutWidth = 0;
		mOutHeight = 0;
		mOutDepth = 0;
		mOutChannel = 0;
		mPadding = 1;
		mBatchSize = 0;
        OutData = nullptr;
	}
	~convCalculator() 
    {
        SAFE_DELETE_ARRAY(OutData);
    };

	int mInWidth;
	int mInHeight;
	int mInDepth;   //三维数据的深度
	int mInChannel;
	int mOutWidth;
	int mOutHeight;
	int mOutDepth;	//三维数据的深度
	int mOutChannel;

	int mPadding;
	int mBatchSize;
    // 图片输出信息。
	type* OutData;

	void PrintOutData();
};

template <typename type>
void convCalculator<type>::PrintOutData()
{
	for(int i = 0;i < mBatchSize;i++)
	{
		printf("Batch %d:\n", i);
		// 三维数据的输出
		for(int j = 0;j < mOutChannel;j++)
		{
			for(int k = 0;k < mOutDepth;k++)
			{
				for(int l = 0;l < mOutHeight;l++)
				{
					for(int m = 0;m < mOutWidth;m++)
					{
						printf("%f\t", OutData[i * mOutChannel * mOutDepth * mOutHeight * mOutWidth + j * mOutDepth * mOutHeight * mOutWidth + k * mOutHeight * mOutWidth + l * mOutWidth + m]);
					}
					printf("\n");
				}
				printf("\n");
			}
		}
	}
}


// 用于计算卷积的函数

template <typename type>
void Conv3DCUDA(type* d_Indata, ConvKernelGPU<type> * d_CKC, type* OutData,
	int VideoWidth, int VideoHeight, int VideoDepth, type* ReluOutData, int BatchSize,
	int InChannel, int OutChannel);
template <typename type>
void Conv2DCUDA_3(type* d_Indata, ConvKernelGPU<type> * d_CKC, type* ImageOutData,
	int ImageWidth, int ImageHeight, type* ReluOutData, int BatchSize,
	int InChannel, int OutChannel);

template <typename type>
void Conv1DCUDA(type* d_Indata, ConvKernelGPU<type> * d_CKC, type* OutData,
	int TextWidth, type* ReluOutData, int BatchSize,
	int InChannel, int OutChannel);

template <typename type>
void convCalculator<type>::Conv3D(type* VideoData,int VideoWidth,int VideoHeight,int VideoDepth,int VideoChannel,int BatchSize,ConvKernel<type>* CK)
{
	if(CK->ConvKernelDepth > VideoDepth)
	{
		std::cout << "Conv3D Error: ConvKernelDepth < VideoDepth" << std::endl;
		return;
	}
	mInWidth = mOutWidth = VideoWidth;
	mInHeight = mOutHeight = VideoHeight;
	mInDepth = mOutDepth = VideoDepth;
	mInChannel = VideoChannel;

	mOutChannel = CK->ConvKernelCount;
	mBatchSize = BatchSize;
	mPadding = 1;
	int mPaddingW = (CK->ConvKernelWidth - 1) >> 1;
	int mPaddingH = (CK->ConvKernelHeight - 1) >> 1;
	int mPaddingD = (CK->ConvKernelDepth - 1) >> 1;

	SAFE_DELETE_ARRAY(OutData);
	OutData = new type[BatchSize * VideoWidth * VideoHeight * VideoDepth * CK->ConvKernelCount];

	memset(OutData, 0, sizeof(type) * BatchSize * VideoWidth * VideoHeight * VideoDepth * CK->ConvKernelCount);

	type* ReluGradientData = new type[mInWidth * mInHeight * mInDepth * BatchSize * CK->ConvKernelCount];

	memset(ReluGradientData, 0, sizeof(type) * mInWidth * mInHeight * mInDepth * mBatchSize * CK->ConvKernelCount);

	// 开始计算三维卷积
	cudaError_t Err;
	int InChannel = CK->ConvKernelChannel, OutChannel = CK->ConvKernelCount;
	int InSize = mBatchSize * InChannel * VideoWidth * VideoHeight * VideoDepth * sizeof(type), OutSize = mBatchSize * OutChannel * VideoWidth * VideoHeight * VideoDepth * sizeof(type);
	int ConvLen = CK->ConvKernelWidth * CK->ConvKernelHeight * CK->ConvKernelDepth * InChannel;
	// if (CK->CB->ConvArray == nullptr) {
	// 	CK->CB->ConvArray = new type[ImageSize2D * mBatchSize * InChannel];		//输入数据总量
	// 	CK->CB->ArrayLen = InChannel;							//输入数据通道数
	// }
	ConvKernelGPU<type>* CKC = new ConvKernelGPU<type>(CK), * d_CKC;

	Err = CUDA_MALLOC(d_CKC, sizeof(ConvKernelGPU<type>));
	HANDLE_ERROR(Err);
	Err = cudaMemcpy(d_CKC, CKC, sizeof(ConvKernelGPU<type>), cudaMemcpyHostToDevice);
	HANDLE_ERROR(Err);

	type* d_InData;
	Err = CUDA_MALLOC(d_InData, InSize);
	HANDLE_ERROR(Err);
	Err = cudaMemcpy(d_InData, VideoData, InSize, cudaMemcpyHostToDevice);
	HANDLE_ERROR(Err);

	Conv3DCUDA<type>(d_InData, d_CKC, OutData, VideoWidth, VideoHeight, VideoDepth, ReluGradientData, mBatchSize,
					InChannel, OutChannel);
	
	HANDLE_ERROR(Err);
	Err = cudaFree(d_CKC);
	HANDLE_ERROR(Err);
	Err = cudaFree(d_InData);
	HANDLE_ERROR(Err);
	SAFE_DELETE(CKC);
	// CK->ApplyImageReluGradientData(ReluGradientData, mInWidth * mInHeight * mBatchSize * CK->ConvKernelCount);
	// CK->ApplyImageInputWidthAndHeight(mInWidth, mInHeight);
	SAFE_DELETE_ARRAY(ReluGradientData);
}

template <typename type>
void convCalculator<type>::Conv3D_CPU(type* VideoData,int VideoWidth,int VideoHeight,int VideoDepth,int VideoChannel,int BatchSize,ConvKernel<type>* CK)
{
	mInWidth = mOutWidth = VideoWidth;
	mInHeight = mOutHeight = VideoHeight;
	mInDepth = mOutDepth = VideoDepth;
	mInChannel = VideoChannel;

	mOutChannel = CK->ConvKernelCount;
	mBatchSize = BatchSize;
	mPadding = 1;
	int mPaddingW = (CK->ConvKernelWidth - 1) >> 1;
	int mPaddingH = (CK->ConvKernelHeight - 1) >> 1;
	int mPaddingD = (CK->ConvKernelDepth - 1) >> 1;

	SAFE_DELETE_ARRAY(OutData);
	OutData = new type[BatchSize * VideoWidth * VideoHeight * VideoDepth * CK->ConvKernelCount];

	memset(OutData, 0, sizeof(type) * BatchSize * VideoWidth * VideoHeight * VideoDepth * CK->ConvKernelCount);

	type* ReluGradientData = new type[mInWidth * mInHeight * mInDepth * BatchSize * CK->ConvKernelCount];

	memset(ReluGradientData, 1, sizeof(type) * mInWidth * mInHeight * mInDepth * mBatchSize * CK->ConvKernelCount);

	// 开始计算三维卷积
	for (int i = 0; i < BatchSize; i++)
	{
		for(int j = 0;j < CK->ConvKernelCount;j++)
		{
			for(int k = 0;k < VideoChannel;k++)
			{
				for(int l = 0;l < VideoDepth;l++)
				{
					for(int m = 0;m < VideoHeight;m++)
					{
						for(int n = 0;n < VideoWidth;n++)
						{
							for(int o = 0;o < CK->ConvKernelDepth;o++)
							{
								for(int p = 0;p < CK->ConvKernelHeight;p++)
								{
									for(int q = 0;q < CK->ConvKernelWidth;q++)
									{
										int x = n - mPaddingW + q;
										int y = m - mPaddingH + p;
										int z = l - mPaddingD + o;
										if(x >= 0 && x < VideoWidth && y >= 0 && y < VideoHeight && z >= 0 && z < VideoDepth)			// 零填充
										{
											OutData[i * VideoWidth * VideoHeight * VideoDepth * CK->ConvKernelCount + j * VideoWidth * VideoHeight * VideoDepth + l * VideoWidth * VideoHeight + m * VideoWidth + n] +=
												VideoData[i * VideoWidth * VideoHeight * VideoDepth * VideoChannel + k * VideoWidth * VideoHeight * VideoDepth + z * VideoWidth * VideoHeight + y * VideoWidth + x] *
												CK->W[j * VideoChannel * CK->ConvKernelDepth * CK->ConvKernelHeight * CK->ConvKernelWidth + k * CK->ConvKernelDepth * CK->ConvKernelHeight * CK->ConvKernelWidth + o * CK->ConvKernelHeight * CK->ConvKernelWidth + p * CK->ConvKernelWidth + q];
										}
									}
								}
							}
						}
					}
				}
			}
			for(int l =0;l < VideoDepth;l++)
			{
				for(int m = 0;m < VideoHeight;m++)
				{
					for(int n = 0;n < VideoWidth;n++)
					{
						OutData[i * VideoWidth * VideoHeight * VideoDepth * CK->ConvKernelCount + j * VideoWidth * VideoHeight * VideoDepth + l * VideoWidth * VideoHeight + m * VideoWidth + n] += CK->B[j];
						// if(OutData[i * VideoWidth * VideoHeight * VideoDepth * CK->ConvKernelCount + j * VideoWidth * VideoHeight * VideoDepth + l * VideoWidth * VideoHeight + m * VideoWidth + n] < 0)
						// {
						// 	OutData[i * VideoWidth * VideoHeight * VideoDepth * CK->ConvKernelCount + j * VideoWidth * VideoHeight * VideoDepth + l * VideoWidth * VideoHeight + m * VideoWidth + n] = 0;		//relu激活函数
						// 	ReluGradientData[i * VideoWidth * VideoHeight * VideoDepth * CK->ConvKernelCount + j * VideoWidth * VideoHeight * VideoDepth + l * VideoWidth * VideoHeight + m * VideoWidth + n] = 0;
						// }
					}
				}
			}
		}
	}
	
	SAFE_DELETE_ARRAY(ReluGradientData);
}

template <typename type>
void convCalculator<type>::Conv2D(type* ImageData, int ImageWidth, int ImageHeight, int ImageChannel, int BatchSize, ConvKernel<type>* CK)
{
	if(CK->ConvKernelDepth != 1)
	{
		std::cout << "Conv2D Error: ConvKernelDepth != 1" << std::endl;
		return;
	}
	mInWidth = mOutWidth = ImageWidth;
	mInHeight = mOutHeight = ImageHeight;
	mInDepth = mOutDepth = 1;
	mInChannel = ImageChannel;

	mOutChannel = CK->ConvKernelCount;
	mBatchSize = BatchSize;
	mPadding = 1;		//是否进行零填充
	int mPaddingW = (CK->ConvKernelWidth - 1) >> 1;
	int mPaddingH = (CK->ConvKernelHeight - 1) >> 1;

	// 输出数据
	// 这里要乘以CK->ConvKernelCount，一张图其实是分开了CK->ConvKernelCount那么多张特征图
	SAFE_DELETE_ARRAY(OutData);
	OutData = new type[BatchSize * ImageWidth * ImageHeight * CK->ConvKernelCount];

	memset(OutData, 0, sizeof(type) * BatchSize * ImageWidth * ImageHeight * CK->ConvKernelCount);

	type* ReluGradientData = new type[mInWidth * mInHeight * BatchSize * CK->ConvKernelCount];

	memset(ReluGradientData, 0, sizeof(type) * mInWidth * mInHeight * mBatchSize * CK->ConvKernelCount);

	// // 开始计算卷积
	// if (CK->CB == nullptr)
	// {
	// 	CK->CB = new ConvBlocks;
	// }

	int ImageSize2D = ImageWidth * ImageHeight;			//每张特征图尺寸
	cudaError_t Err;
	int InChannel = CK->ConvKernelChannel, OutChannel = CK->ConvKernelCount;
	int InSize = mBatchSize * InChannel * ImageSize2D * sizeof(type), OutSize = mBatchSize * OutChannel * ImageSize2D * sizeof(type);
	int ConvLen = CK->ConvKernelWidth * CK->ConvKernelHeight * InChannel;
	// if (CK->CB->ConvArray == nullptr) {
	// 	CK->CB->ConvArray = new type[ImageSize2D * mBatchSize * InChannel];		//输入数据总量
	// 	CK->CB->ArrayLen = InChannel;							//输入数据通道数
	// }
	ConvKernelGPU<type>* CKC = new ConvKernelGPU<type>(CK), * d_CKC;

	Err = CUDA_MALLOC(d_CKC, sizeof(ConvKernelGPU<type>));
	HANDLE_ERROR(Err);
	Err = cudaMemcpy(d_CKC, CKC, sizeof(ConvKernelGPU<type>), cudaMemcpyHostToDevice);
	HANDLE_ERROR(Err);

	type* d_InData;
	Err = CUDA_MALLOC(d_InData, InSize);
	HANDLE_ERROR(Err);
	Err = cudaMemcpy(d_InData, ImageData, InSize, cudaMemcpyHostToDevice);
	HANDLE_ERROR(Err);

	Conv2DCUDA_3<type>(d_InData, d_CKC, OutData, ImageWidth, ImageHeight, ReluGradientData, mBatchSize,
					InChannel, OutChannel);

	HANDLE_ERROR(Err);
	Err = cudaFree(d_CKC);
	HANDLE_ERROR(Err);
	Err = cudaFree(d_InData);
	HANDLE_ERROR(Err);
	SAFE_DELETE(CKC);
	// CK->ApplyImageReluGradientData(ReluGradientData, mInWidth * mInHeight * mBatchSize * CK->ConvKernelCount);
	// CK->ApplyImageInputWidthAndHeight(mInWidth, mInHeight);
	SAFE_DELETE_ARRAY(ReluGradientData);
}

template <typename type>
void convCalculator<type>::Conv2D_CPU(type* ImageData, int ImageWidth, int ImageHeight, int ImageChannel, int BatchSize, ConvKernel<type>* CK)
{
	if(CK->ConvKernelDepth != 1)
	{
		std::cout << "Conv2D Error: ConvKernelDepth != 1" << std::endl;
		return;
	}
	mInWidth = mOutWidth = ImageWidth;
	mInHeight = mOutHeight = ImageHeight;
	mInDepth = mOutDepth = 1;
	mInChannel = ImageChannel;

	mOutChannel = CK->ConvKernelCount;
	mBatchSize = BatchSize;
	mPadding = 1;
	int mPaddingW = (CK->ConvKernelWidth - 1) >> 1;
	int mPaddingH = (CK->ConvKernelHeight - 1) >> 1;

	SAFE_DELETE_ARRAY(OutData);
	OutData = new type[BatchSize * ImageWidth * ImageHeight * CK->ConvKernelCount];

	memset(OutData, 0, sizeof(type) * BatchSize * ImageWidth * ImageHeight * CK->ConvKernelCount);

	type* ReluGradientData = new type[mInWidth * mInHeight * BatchSize * CK->ConvKernelCount];

	memset(ReluGradientData, 1, sizeof(type) * mInWidth * mInHeight * mBatchSize * CK->ConvKernelCount);

	// 开始计算卷积
	for(int i = 0;i < BatchSize;i++)
	{
		for(int j = 0;j < CK->ConvKernelCount;j++)
		{
			for(int k = 0;k < ImageChannel;k++)
			{
				for(int l = 0;l < ImageHeight;l++)
				{
					for(int m = 0;m < ImageWidth;m++)
					{
						for(int n = 0;n < CK->ConvKernelHeight;n++)
						{
							for(int o = 0;o < CK->ConvKernelWidth;o++)
							{
								int x = m - mPaddingW + o;
								int y = l - mPaddingH + n;
								if(x >= 0 && x < ImageWidth && y >= 0 && y < ImageHeight)			// 零填充
								{
									OutData[i * ImageWidth * ImageHeight * CK->ConvKernelCount + j * ImageWidth * ImageHeight + l * ImageWidth + m] +=
										ImageData[i * ImageWidth * ImageHeight * ImageChannel + k * ImageWidth * ImageHeight + y * ImageWidth + x] *
										CK->W[j * ImageChannel * CK->ConvKernelHeight * CK->ConvKernelWidth + k * CK->ConvKernelHeight * CK->ConvKernelWidth + n * CK->ConvKernelWidth + o];
								}
							}
						}
					}
				}
			}
			for(int l = 0;l < ImageHeight;l++)
			{
				for(int m = 0;m < ImageWidth;m++)
				{
					OutData[i * ImageWidth * ImageHeight * CK->ConvKernelCount + j * ImageWidth * ImageHeight + l * ImageWidth + m] += CK->B[j];
					// if(OutData[i * ImageWidth * ImageHeight * CK->ConvKernelCount + j * ImageWidth * ImageHeight + l * ImageWidth + m] < 0)
					// {
					// 	OutData[i * ImageWidth * ImageHeight * CK->ConvKernelCount + j * ImageWidth * ImageHeight + l * ImageWidth + m] = 0;		//relu激活函数
					//  ReluGradientData[i * ImageWidth * ImageHeight * CK->ConvKernelCount + j * ImageWidth * ImageHeight + l * ImageWidth + m] = 0;
					// }
				}
			}
		}
	}

	SAFE_DELETE_ARRAY(ReluGradientData);
}

template <typename type>
void convCalculator<type>::Conv1D(type* TextData,int TextLength,int TextChannel, int BatchSize,ConvKernel<type>* CK)
{
	if(CK->ConvKernelHeight != 1 || CK->ConvKernelDepth != 1)
	{
		std::cout << "Conv1D Error: ConvKernelHeight != 1 or ConvKernelDepth != 1" << std::endl;
		return;
	}

	mInWidth = mOutWidth = TextLength;
	mInHeight = mOutHeight = 1;
	mInDepth = mOutDepth = 1;
	mInChannel = TextChannel;

	mOutChannel = CK->ConvKernelCount;
	mBatchSize = BatchSize;
	mPadding = 1;
	int mPaddingW = (CK->ConvKernelWidth - 1) >> 1;

	SAFE_DELETE_ARRAY(OutData);
	OutData = new type[BatchSize * TextLength * CK->ConvKernelCount];

	memset(OutData, 0, sizeof(type) * BatchSize * TextLength * CK->ConvKernelCount);

	type* ReluGradientData = new type[mInWidth * mInHeight * BatchSize * CK->ConvKernelCount];

	memset(ReluGradientData, 1, sizeof(type) * mInWidth * mInHeight * mBatchSize * CK->ConvKernelCount);

	// 开始计算卷积
	cudaError_t Err;
	int InChannel = CK->ConvKernelChannel, OutChannel = CK->ConvKernelCount;
	int InSize = mBatchSize * InChannel * TextLength * sizeof(type), OutSize = mBatchSize * OutChannel * TextLength * sizeof(type);
	int ConvLen = CK->ConvKernelWidth * CK->ConvKernelHeight * InChannel;
	// if (CK->CB->ConvArray == nullptr) {
	// 	CK->CB->ConvArray = new type[ImageSize2D * mBatchSize * InChannel];		//输入数据总量
	// 	CK->CB->ArrayLen = InChannel;							//输入数据通道数
	// }
	ConvKernelGPU<type>* CKC = new ConvKernelGPU<type>(CK), * d_CKC;

	Err = CUDA_MALLOC(d_CKC, sizeof(ConvKernelGPU<type>));
	HANDLE_ERROR(Err);
	Err = cudaMemcpy(d_CKC, CKC, sizeof(ConvKernelGPU<type>), cudaMemcpyHostToDevice);
	HANDLE_ERROR(Err);

	type* d_InData;
	Err = CUDA_MALLOC(d_InData, InSize);
	HANDLE_ERROR(Err);
	Err = cudaMemcpy(d_InData, TextData, InSize, cudaMemcpyHostToDevice);
	HANDLE_ERROR(Err);

	Conv1DCUDA<type>(d_InData, d_CKC, OutData, TextLength, ReluGradientData, mBatchSize,
					InChannel, OutChannel);

	HANDLE_ERROR(Err);
	Err = cudaFree(d_CKC);
	HANDLE_ERROR(Err);
	Err = cudaFree(d_InData);
	HANDLE_ERROR(Err);
	SAFE_DELETE(CKC);
	// CK->ApplyImageReluGradientData(ReluGradientData, mInWidth * mInHeight * mBatchSize * CK->ConvKernelCount);
	// CK->ApplyImageInputWidthAndHeight(mInWidth, mInHeight);
	SAFE_DELETE_ARRAY(ReluGradientData);
}

template <typename type>
void convCalculator<type>::Conv1D_CPU(type* TextData,int TextLength,int TextChannel,int BatchSize,ConvKernel<type>* CK)
{
	if(CK->ConvKernelHeight != 1 || CK->ConvKernelDepth != 1)
	{
		std::cout << "Conv1D Error: ConvKernelHeight != 1 or ConvKernelDepth != 1" << std::endl;
		return;
	}

	mInWidth = mOutWidth = TextLength;
	mInHeight = mOutHeight = 1;
	mInDepth = mOutDepth = 1;
	mInChannel = TextChannel;

	mOutChannel = CK->ConvKernelCount;
	mBatchSize = BatchSize;
	mPadding = 1;
	int mPaddingW = (CK->ConvKernelWidth - 1) >> 1;

	SAFE_DELETE_ARRAY(OutData);
	OutData = new type[BatchSize * TextLength * CK->ConvKernelCount];

	memset(OutData, 0, sizeof(type) * BatchSize * TextLength * CK->ConvKernelCount);

	type* ReluGradientData = new type[mInWidth * mInHeight * BatchSize * CK->ConvKernelCount];

	memset(ReluGradientData, 1, sizeof(type) * mInWidth * mInHeight * mBatchSize * CK->ConvKernelCount);

	// 开始计算卷积
	for(int i = 0;i < BatchSize;i++)
	{
		for(int j = 0;j < CK->ConvKernelCount;j++)
		{
			for(int k = 0;k < TextChannel;k++)
			{
				for(int l = 0;l < TextLength;l++)
				{
					for(int m = 0;m < CK->ConvKernelWidth;m++)
					{
						int x = l - mPaddingW + m;
						if(x >= 0 && x < TextLength)			// 零填充
						{
							OutData[i * TextLength * CK->ConvKernelCount + j * TextLength + l] +=
								TextData[i * TextLength * TextChannel + k * TextLength + x] *
								CK->W[j * TextChannel * CK->ConvKernelWidth + k * CK->ConvKernelWidth + m];
						}
					}
				}
			}
			for(int l = 0;l < TextLength;l++)
			{
				OutData[i * TextLength * CK->ConvKernelCount + j * TextLength + l] += CK->B[j];
				// if(OutData[i * TextLength * CK->ConvKernelCount + j * TextLength + l] < 0)
				// {
				// 	OutData[i * TextLength * CK->ConvKernelCount + j * TextLength + l] = 0;		//relu激活函数
				// 	ReluGradientData[i * TextLength * CK->ConvKernelCount + j * TextLength + l] = 0;
				// }
			}
		}
	}

	SAFE_DELETE_ARRAY(ReluGradientData);
}
