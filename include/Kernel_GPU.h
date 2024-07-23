#include "Kernel_CPU.h"
#include <cuda_runtime_api.h>

/*
在gpu计算时卷积核数据需要从内存中拷贝到显存中，计算完成后再拷贝回内存中
*/

#define CUDA_MALLOC(dp, size)	cudaMalloc((void**)&dp, size)
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

static void HandleError(cudaError_t err, const char* file, int line)
{
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}

template<typename type>
struct ConvBlocksGPU
{
	type* ConvArray;
	int ArrayLen;
	int cBatchSize;
	int cImgSize;
	ConvBlocks<type>* CB;
	ConvBlocksGPU(ConvBlocks<type>* CBs, int BatchSize, int ImgSize)
	{
		cudaError_t Err; 
		Err = CUDA_MALLOC(ConvArray, CBs->ArrayLen * BatchSize * ImgSize *  sizeof(type));
		HANDLE_ERROR(Err);
		Err = cudaMemcpy(ConvArray, CBs->ConvArray, CBs->ArrayLen * BatchSize * ImgSize * sizeof(type), cudaMemcpyHostToDevice);
		HANDLE_ERROR(Err);
		ArrayLen = CBs->ArrayLen;
		CB = CBs;
		cBatchSize = BatchSize;
		cImgSize = ImgSize;
	}
	~ConvBlocksGPU()
	{
		cudaError_t Err;
		Err = cudaMemcpy(CB->ConvArray, ConvArray, cBatchSize * ArrayLen * cImgSize * sizeof(type), cudaMemcpyDeviceToHost);
		HANDLE_ERROR(Err);
		Err = cudaFree(ConvArray);
		HANDLE_ERROR(Err);
		CB = nullptr;
		ArrayLen = 0;
		cBatchSize = 0;
	}
};

template<typename type>
struct ConvKernelGPU
{
	int ConvKernelWidth;
	int ConvKernelHeight;
	int ConvKernelDepth;
	int ConvKernelChannel;
	int ConvKernelCount;  // kernel 个数即为 输出 OutChannels 的值

	type* W;
	type* B;


    // 反向传播中会用到
	// int ImageInputWidth;
	// int ImageInputHeight;
	// ConvBlocks* CB;
	// ...
	// type* WRotate180;				// 反向求导的时候用到，旋转180度
	// type* DropOutGradient;
	// type* ImageMaxPoolGradientData;
	// type* ImageReluGradientData;
	// ConvKernel* cCK;

	ConvKernelGPU(ConvKernel<type>* CK)
	{
		cudaError_t Err;
		ConvKernelWidth = CK->ConvKernelWidth;
		ConvKernelHeight = CK->ConvKernelHeight;
		ConvKernelDepth = CK->ConvKernelDepth;
		ConvKernelChannel = CK->ConvKernelChannel;
		ConvKernelCount = CK->ConvKernelCount;
		// W = CK->W;
		// WRotate180 = CK->WRotate180;
		// B = CK->B;
		// CB = CK->CB;
		// ImageInputWidth = CK->ImageInputWidth;
		// ImageInputHeight = CK->ImageInputHeight;
		// DropOutGradient = CK->DropOutGradient;
		// ImageMaxPoolGradientData = CK->ImageMaxPoolGradientData;
		// ImageReluGradientData = CK->ImageReluGradientData;
		// cCK = CK;
		int W_Size = ConvKernelWidth * ConvKernelHeight * ConvKernelDepth * ConvKernelChannel * ConvKernelCount * sizeof(type);
		int B_Size = ConvKernelCount * sizeof(type);
		Err = CUDA_MALLOC(W, W_Size);
		HANDLE_ERROR(Err);
		Err = cudaMemset(W, 0, W_Size);
		HANDLE_ERROR(Err);
		Err = CUDA_MALLOC(B, B_Size);
		HANDLE_ERROR(Err);
		// Err = CUDA_MALLOC(WRotate180, W_Size);
		// HANDLE_ERROR(Err);
		Err = cudaMemcpy(W, CK->W, W_Size, cudaMemcpyHostToDevice);
		HANDLE_ERROR(Err);
		Err = cudaMemcpy(B, CK->B, B_Size, cudaMemcpyHostToDevice);
		HANDLE_ERROR(Err);
		// Err = cudaMemcpy(WRotate180, CK->WRotate180, W_Size, cudaMemcpyHostToDevice);
		// HANDLE_ERROR(Err);
	}
	~ConvKernelGPU()
	{
		cudaError_t Err;
		// ConvKernel* CK = cCK;
		// CK->ConvKernelWidth = ConvKernelWidth;
		// CK->ConvKernelHeight = ConvKernelHeight;
		// CK->ConvKernelChannel = ConvKernelChannel;
		// CK->ConvKernelCount = ConvKernelCount;
		int W_Size = ConvKernelWidth * ConvKernelHeight * ConvKernelDepth * ConvKernelChannel * ConvKernelCount * sizeof(type);
		int B_Size = ConvKernelCount * sizeof(type);
		// CK->W = W;
		// CK->WRotate180 = WRotate180;
		// CK->B = B;
		// Err = cudaMemcpy(CK->W, W, W_Size, cudaMemcpyDeviceToHost);
		// HANDLE_ERROR(Err);
		Err = cudaFree(W);
		HANDLE_ERROR(Err);
		// Err = cudaMemcpy(CK->B, B, B_Size, cudaMemcpyDeviceToHost);
		// HANDLE_ERROR(Err);
		Err = cudaFree(B);
		HANDLE_ERROR(Err);
		// Err = cudaMemcpy(CK->WRotate180, WRotate180, W_Size, cudaMemcpyDeviceToHost);
		// HANDLE_ERROR(Err);
		// Err = cudaFree(WRotate180);
		// HANDLE_ERROR(Err);
		// CK->CB = CB;
		// CK->ImageInputWidth = ImageInputWidth;
		// CK->ImageInputHeight = ImageInputHeight;
		// CK->DropOutGradient = DropOutGradient;
		// CK->ImageMaxPoolGradientData = ImageMaxPoolGradientData;
		// CK->ImageReluGradientData = ImageReluGradientData;
		// cCK = nullptr;
	}
};