#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "Kernel_GPU.h"

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))
#define BLOCK_SIZE 16

// 卷积的结果已经加上偏差，如果要使用relu激活函数进行激活。添加判断条件 sum > 0.0f

template <typename type>
__global__ void _Conv3D(type* InData, ConvKernelGPU<type>* CKC, type* OutData,type* ReluData,int BatchSize,int InChannel,int OutChannel,int VideoWidth,int VideoHeight,int VideoDepth)
{   
    // 输出数据每个元素的坐标，储存方式为NCDHW
    int dep = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int row = blockIdx.z * blockDim.z + threadIdx.z;

    while(col < VideoWidth && row < VideoHeight && dep < OutChannel * BatchSize * VideoDepth){
        int ChannelIdx = (dep % (OutChannel * VideoDepth) )/ VideoDepth, BatchIdx = dep / (OutChannel * VideoDepth);   // ChannelIdx是输出数据的通道索引，BatchIdx是批次索引
        int ImgSize3D = VideoWidth * VideoHeight * VideoDepth;
        int ConvWidth = CKC->ConvKernelWidth, ConvHeight = CKC->ConvKernelHeight, ConvDepth = CKC->ConvKernelDepth;
        int ConvLen = CKC->ConvKernelWidth * CKC->ConvKernelHeight * CKC->ConvKernelDepth * CKC->ConvKernelChannel;
        int HalfConvW = (ConvWidth - 1)>> 1;
        int HalfConvH = (ConvHeight - 1)>> 1;
        int HalfConvD = (ConvDepth - 1)>> 1;
        type sum = 0;
        for (int k = 0; k < InChannel; k++) {
            for (int l = 0; l < ConvDepth; l++) {
                for (int m = 0; m < ConvHeight; m++) {
                    for (int n = 0; n < ConvWidth; n++) {
                        int x_ = n - HalfConvW;             
                        int y_ = m - HalfConvH;
                        int z_ = l - HalfConvD;
                        int x = col + x_;           // x,y,z是每个输入数据在每个通道上的坐标
                        int y = row + y_;
                        int z = dep % VideoDepth + z_;
                        if (!(x < 0 || x >= VideoWidth || y < 0 || y >= VideoHeight || z < 0 || z >= VideoDepth)) {             // 实现零填充的作用
                            int ConvPos = k * ConvDepth * ConvHeight * ConvWidth + l * ConvHeight * ConvWidth + m * ConvWidth + n;
                            int Pos = BatchIdx * ImgSize3D * InChannel + k * ImgSize3D + z * VideoWidth * VideoHeight + y * VideoWidth + x;
                            int ConvKernelPos = ChannelIdx * ConvLen + ConvPos;
                            sum += InData[Pos] * CKC->W[ConvKernelPos];
                        }
                    }
                }
            }
        }
        sum += CKC->B[ChannelIdx];
        __syncthreads();
        int OutPos = BatchIdx * ImgSize3D * OutChannel + ChannelIdx * ImgSize3D + dep % VideoDepth * VideoWidth * VideoHeight
                    + row * VideoWidth + col;
        if (1) {                            // relu激活函数  sum > 0.0f
            OutData[OutPos] = sum;
            ReluData[OutPos] = 1;
        }

        dep += blockDim.x * gridDim.x;
        col += blockDim.y * gridDim.y;
        row += blockDim.z * gridDim.z;
    }
}

template __global__ void _Conv3D<int>(int* InData, ConvKernelGPU<int>* CKC, int* OutData, int* ReluData, int BatchSize, int InChannel, int OutChannel, int VideoWidth, int VideoHeight, int VideoDepth);
template __global__ void _Conv3D<float>(float* InData, ConvKernelGPU<float>* CKC, float* OutData, float* ReluData, int BatchSize, int InChannel, int OutChannel, int VideoWidth, int VideoHeight, int VideoDepth);
template __global__ void _Conv3D<double>(double* InData, ConvKernelGPU<double>* CKC, double* OutData, double* ReluData, int BatchSize, int InChannel, int OutChannel, int VideoWidth, int VideoHeight, int VideoDepth);


template<typename type>
__global__ void _Conv2D_Both(type* InData, ConvKernelGPU<type>* CKC, type* OutData, type* ReluData, int BatchSize, int InChannel, int OutChannel, int ImageWidth, int ImageHeight)
{
    int dep = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int row = blockIdx.z * blockDim.z + threadIdx.z;

    if (col < ImageWidth && row < ImageHeight && dep < OutChannel * BatchSize) {
        int ChannelIdx = dep % OutChannel, BatchIdx = dep / OutChannel;
        int ImgSize2D = ImageWidth * ImageHeight;
        int ConvWidth = CKC->ConvKernelWidth, ConvHeight = CKC->ConvKernelHeight;
        int ConvLen = CKC->ConvKernelWidth * CKC->ConvKernelHeight * CKC->ConvKernelChannel;
        int HalfConvW = (ConvWidth - 1)>> 1;
        int HalfConvH = (ConvHeight - 1)>> 1;
        type sum = 0;
        for (int k = 0; k < InChannel; k++) {
            for (int m = 0; m < ConvHeight; m++) {
                for (int n = 0; n < ConvWidth; n++) {
                    int x_ = n - HalfConvW;
                    int y_ = m - HalfConvH;
                    int x = col + x_;
                    int y = row + y_;
                    if (!(x < 0 || x >= ImageWidth || y < 0 || y >= ImageHeight)) {             // 实现零填充的作用
                        int ConvPos = k * ConvHeight * ConvWidth + m * ConvWidth + n;
                        int Pos = BatchIdx * ImgSize2D * InChannel + k * ImgSize2D + y * ImageWidth + x;
                        int ConvKernelPos = ChannelIdx * ConvLen + ConvPos;
                        sum += InData[Pos] * CKC->W[ConvKernelPos];
                    }
                }
            }
        }
        sum += CKC->B[ChannelIdx];
        __syncthreads();
        int OutPos = BatchIdx * ImgSize2D * OutChannel + ChannelIdx * ImgSize2D
                    + row * ImageWidth + col;
        if (1) {                            // relu激活函数  sum > 0.0f
            OutData[OutPos] = sum;
            ReluData[OutPos] = 1;
        }

    }
}

template __global__ void _Conv2D_Both<int>(int* InData, ConvKernelGPU<int>* CKC, int* OutData, int* ReluData, int BatchSize, int InChannel, int OutChannel, int ImageWidth, int ImageHeight);
template __global__ void _Conv2D_Both<float>(float* InData, ConvKernelGPU<float>* CKC, float* OutData, float* ReluData, int BatchSize, int InChannel, int OutChannel, int ImageWidth, int ImageHeight);
template  __global__ void _Conv2D_Both<double>(double* InData, ConvKernelGPU<double>* CKC, double* OutData, double* ReluData, int BatchSize, int InChannel, int OutChannel, int ImageWidth, int ImageHeight);

template<typename type>
__global__ void _Conv2D_Opt(type* InData, ConvKernelGPU<type>* CKC, type* OutData, type* ReluData, int BatchSize, int InChannel, int OutChannel, int ImageWidth, int ImageHeight)
{
    int dep = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int row = blockIdx.z * blockDim.z + threadIdx.z;

    if (col < ImageWidth && row < ImageHeight && dep < OutChannel * BatchSize) {
        __shared__ type Data_Share[BLOCK_SIZE * BLOCK_SIZE];
        int ChannelIdx = dep % OutChannel, BatchIdx = dep / OutChannel;
        int ImgSize2D = ImageWidth * ImageHeight;
        int ConvWidth = CKC->ConvKernelWidth, ConvHeight = CKC->ConvKernelHeight;
        int ConvLen = CKC->ConvKernelWidth * CKC->ConvKernelHeight * CKC->ConvKernelChannel;
        int HalfConvW = (ConvWidth - 1)>> 1;
        int HalfConvH = (ConvHeight - 1)>> 1;
        type sum = 0;
        for (int k = 0; k < InChannel; k++) {
            int InPos = BatchIdx * ImgSize2D * InChannel + k * ImgSize2D + row * ImageWidth + col;
            Data_Share[threadIdx.z * blockDim.y + threadIdx.y] = InData[InPos];
            __syncthreads();
            int This_tile_start_point_x = blockIdx.y * blockDim.y;
            int Next_tile_start_point_x = (blockIdx.y + 1) * blockDim.y;
            int This_tile_start_point_y = blockIdx.z * blockDim.z;
            int Next_tile_start_point_y = (blockIdx.z + 1) * blockDim.z;
            int N_start_point_x = col - HalfConvW;
            int N_start_point_y = row - HalfConvH;
            for (int m = 0; m < ConvHeight; m++) {
                for (int n = 0; n < ConvWidth; n++) {
                    int x_index = N_start_point_x + n;
                    int y_index = N_start_point_y + m;
                    int ConvPos = k * ConvHeight * ConvWidth + m * ConvWidth + n;
                    int ConvKernelPos = ChannelIdx * ConvLen + ConvPos;
                    if (x_index >= 0 && x_index < ImageWidth && y_index >= 0 && y_index < ImageHeight) {
                        if (x_index >= This_tile_start_point_x && x_index < Next_tile_start_point_x &&
                            y_index >= This_tile_start_point_y && y_index < Next_tile_start_point_y) {
                            sum += Data_Share[(threadIdx.z + m - HalfConvW) * BLOCK_SIZE + (threadIdx.y + n - HalfConvH)] * CKC->W[ConvKernelPos];
                        }
                        else {
                            int Pos = BatchIdx* ImgSize2D* InChannel + k * ImgSize2D + y_index * ImageWidth + x_index;
                            sum += InData[Pos] * CKC->W[ConvKernelPos];
                        }

                    }
                }
            }
        }
        sum += CKC->B[ChannelIdx];
        __syncthreads();
        int OutPos = BatchIdx * ImgSize2D * OutChannel + ChannelIdx * ImgSize2D
            + row * ImageWidth + col;
        if (1) {               // relu激活函数  sum > 0.0f
            OutData[OutPos] = sum;
            ReluData[OutPos] = 1;
        }

    }
}

template __global__ void _Conv2D_Opt<int>(int* InData, ConvKernelGPU<int>* CKC, int* OutData, int* ReluData, int BatchSize, int InChannel, int OutChannel, int ImageWidth, int ImageHeight);
template __global__ void _Conv2D_Opt<float>(float* InData, ConvKernelGPU<float>* CKC, float* OutData, float* ReluData, int BatchSize, int InChannel, int OutChannel, int ImageWidth, int ImageHeight);
template __global__ void _Conv2D_Opt<double>(double* InData, ConvKernelGPU<double>* CKC, double* OutData, double* ReluData, int BatchSize, int InChannel, int OutChannel, int ImageWidth, int ImageHeight);

template<typename type>
__global__ void _Conv1D(type* InData, ConvKernelGPU<type>* CKC, type* OutData, type* ReluData, int BatchSize, int InChannel, int OutChannel, int TextWidth)
{
    int dep = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < TextWidth && dep < OutChannel * BatchSize) {
        int ChannelIdx = dep % OutChannel, BatchIdx = dep / OutChannel;
        int ConvWidth = CKC->ConvKernelWidth;
        int ConvLen = CKC->ConvKernelWidth * CKC->ConvKernelChannel;
        int HalfConv = (ConvWidth - 1) >> 1;
        type sum = 0;
        for (int k = 0; k < InChannel; k++) {
            for (int n = 0; n < ConvWidth; n++) {
                int x_ = n - HalfConv;
                int x = col + x_;
                if (!(x < 0 || x >= TextWidth)) {             // 实现零填充的作用
                    int ConvPos = k * ConvWidth + n;
                    int Pos = BatchIdx * TextWidth * InChannel + k * TextWidth + x;
                    int ConvKernelPos = ChannelIdx * ConvLen + ConvPos;
                    sum += InData[Pos] * CKC->W[ConvKernelPos];
                }
            }
        }
        sum += CKC->B[ChannelIdx];
        __syncthreads();
        int OutPos = BatchIdx * TextWidth * OutChannel + ChannelIdx * TextWidth + col;
        if (1) {                            // relu激活函数  sum > 0.0f
            OutData[OutPos] = sum;
            ReluData[OutPos] = 1;
        }

    }
}

template __global__ void _Conv1D<int>(int* InData, ConvKernelGPU<int>* CKC, int* OutData, int* ReluData, int BatchSize, int InChannel, int OutChannel, int TextWidth);
template __global__ void _Conv1D<float>(float* InData, ConvKernelGPU<float>* CKC, float* OutData, float* ReluData, int BatchSize, int InChannel, int OutChannel, int TextWidth);
template __global__ void _Conv1D<double>(double* InData, ConvKernelGPU<double>* CKC, double* OutData, double* ReluData, int BatchSize, int InChannel, int OutChannel, int TextWidth);

template <typename type>
void Conv3DCUDA(type* d_Indata, ConvKernelGPU<type> * d_CKC, type* OutData,
	int VideoWidth, int VideoHeight, int VideoDepth, type* ReluOutData, int BatchSize,
	int InChannel, int OutChannel)
{
    cudaError_t Err;
    int VideoSize3D = VideoWidth * VideoHeight * VideoDepth;
    int InSize = BatchSize * InChannel * VideoSize3D * sizeof(type), OutSize = BatchSize * OutChannel * VideoSize3D * sizeof(type);

    // type* OutData = nullptr; //new type[OutSize / sizeof(type)];
    type* d_OutData, * d_ReluData;
    Err = CUDA_MALLOC(d_OutData, OutSize);
    HANDLE_ERROR(Err);
    Err = CUDA_MALLOC(d_ReluData, OutSize);
    HANDLE_ERROR(Err);
    dim3 Block(1, BLOCK_SIZE, BLOCK_SIZE);
    dim3 Grid(BatchSize * OutChannel * (VideoDepth - 1 + BLOCK_SIZE),(VideoWidth - 1 + BLOCK_SIZE) / BLOCK_SIZE, (VideoHeight - 1 + BLOCK_SIZE) / BLOCK_SIZE);

    _Conv3D<type> <<< Grid, Block >>> (d_Indata, d_CKC, d_OutData, d_ReluData, BatchSize, InChannel, OutChannel, VideoWidth, VideoHeight, VideoDepth);

    //Err = cudaMemcpy(OutData, d_OutData, OutSize, cudaMemcpyDeviceToHost);
    //PRINT(OutData, 100);
    Err = cudaMemcpy(OutData, d_OutData, OutSize, cudaMemcpyDeviceToHost);
    HANDLE_ERROR(Err);
    Err = cudaMemcpy(ReluOutData, d_ReluData, OutSize, cudaMemcpyDeviceToHost);
    HANDLE_ERROR(Err);
    Err = cudaFree(d_OutData);
    HANDLE_ERROR(Err);
    Err = cudaFree(d_ReluData);
    HANDLE_ERROR(Err);

}

template void Conv3DCUDA<int>(int* d_Indata, ConvKernelGPU<int> * d_CKC, int* OutData,
    int VideoWidth, int VideoHeight, int VideoDepth, int* ReluOutData, int BatchSize,
    int InChannel, int OutChannel);
template void Conv3DCUDA<float>(float* d_Indata, ConvKernelGPU<float> * d_CKC, float* OutData,
    int VideoWidth, int VideoHeight, int VideoDepth, float* ReluOutData, int BatchSize,
    int InChannel, int OutChannel);
template void Conv3DCUDA<double>(double* d_Indata, ConvKernelGPU<double> * d_CKC, double* OutData,
    int VideoWidth, int VideoHeight, int VideoDepth, double* ReluOutData, int BatchSize,
    int InChannel, int OutChannel);


template<typename type>
void Conv2DCUDA_3(type* d_Indata, ConvKernelGPU<type> * d_CKC, type* ImageOutData,
    int ImageWidth, int ImageHeight, type* ReluOutData, int BatchSize,
    int InChannel, int OutChannel)
{
    //test_CB(CB, CK, BatchSize, nullptr, 0);
    cudaError_t Err;
    int ImageSize2D = ImageWidth * ImageHeight;
    int InSize = BatchSize * InChannel * ImageSize2D * sizeof(type), OutSize = BatchSize * OutChannel * ImageSize2D * sizeof(type);

    // type* OutData = nullptr; //new type[OutSize / sizeof(type)];
    type* d_OutData, * d_ReluData;
    Err = CUDA_MALLOC(d_OutData, OutSize);
    HANDLE_ERROR(Err);
    Err = CUDA_MALLOC(d_ReluData, OutSize);
    HANDLE_ERROR(Err);
    dim3 Block(1,BLOCK_SIZE, BLOCK_SIZE);
    dim3 Grid(BatchSize * OutChannel,(ImageWidth - 1 + BLOCK_SIZE) / BLOCK_SIZE, (ImageHeight - 1 + BLOCK_SIZE) / BLOCK_SIZE );

	_Conv2D_Both<type> <<< Grid, Block >>> (d_Indata, d_CKC, d_OutData, d_ReluData, BatchSize, InChannel, OutChannel, ImageWidth, ImageHeight);
    // _Conv2D_Opt<type>  <<< Grid, Block >>> (d_Indata, d_CKC, d_OutData, d_ReluData, BatchSize, InChannel, OutChannel, ImageWidth, ImageHeight);

    //Err = cudaMemcpy(OutData, d_OutData, OutSize, cudaMemcpyDeviceToHost);
    //PRINT(OutData, 100);
    Err = cudaMemcpy(ImageOutData, d_OutData, OutSize, cudaMemcpyDeviceToHost);
    HANDLE_ERROR(Err);
    Err = cudaMemcpy(ReluOutData, d_ReluData, OutSize, cudaMemcpyDeviceToHost);
    HANDLE_ERROR(Err);
    Err = cudaFree(d_OutData);
    HANDLE_ERROR(Err);
    Err = cudaFree(d_ReluData);
    HANDLE_ERROR(Err);

    // return OutData;
}

template void Conv2DCUDA_3<int>(int* d_Indata, ConvKernelGPU<int> * d_CKC, int* ImageOutData,
    int ImageWidth, int ImageHeight, int* ReluOutData, int BatchSize,
    int InChannel, int OutChannel);
template void Conv2DCUDA_3<float>(float* d_Indata, ConvKernelGPU<float> * d_CKC, float* ImageOutData,
    int ImageWidth, int ImageHeight, float* ReluOutData, int BatchSize,
    int InChannel, int OutChannel);
template void Conv2DCUDA_3<double>(double* d_Indata, ConvKernelGPU<double> * d_CKC, double* ImageOutData,
    int ImageWidth, int ImageHeight, double* ReluOutData, int BatchSize,
    int InChannel, int OutChannel);

template <typename type>
void Conv1DCUDA(type* d_Indata, ConvKernelGPU<type> * d_CKC, type* OutData,
	int TextWidth, type* ReluOutData, int BatchSize,
	int InChannel, int OutChannel)
{
    cudaError_t Err;
    int TextSize = TextWidth * sizeof(type);
    int InSize = BatchSize * InChannel * TextSize, OutSize = BatchSize * OutChannel * TextSize;

    type* d_OutData, * d_ReluData;
    Err = CUDA_MALLOC(d_OutData, OutSize);
    HANDLE_ERROR(Err);
    Err = CUDA_MALLOC(d_ReluData, OutSize);
    HANDLE_ERROR(Err);
    dim3 Block(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 Grid( BatchSize * OutChannel,(TextWidth - 1 + BLOCK_SIZE) / BLOCK_SIZE);

    _Conv1D<type> <<< Grid, Block >>> (d_Indata, d_CKC, d_OutData, d_ReluData, BatchSize, InChannel, OutChannel, TextWidth);

    Err = cudaMemcpy(OutData, d_OutData, OutSize, cudaMemcpyDeviceToHost);
    HANDLE_ERROR(Err);
    Err = cudaMemcpy(ReluOutData, d_ReluData, OutSize, cudaMemcpyDeviceToHost);
    HANDLE_ERROR(Err);
    Err = cudaFree(d_OutData);
    HANDLE_ERROR(Err);
    Err = cudaFree(d_ReluData);
    HANDLE_ERROR(Err);

}

template void Conv1DCUDA<int>(int* d_Indata, ConvKernelGPU<int> * d_CKC, int* OutData,
    int TextWidth, int* ReluOutData, int BatchSize,
    int InChannel, int OutChannel);
template void Conv1DCUDA<float>(float* d_Indata, ConvKernelGPU<float> * d_CKC, float* OutData,
    int TextWidth, float* ReluOutData, int BatchSize,
    int InChannel, int OutChannel);
template void Conv1DCUDA<double>(double* d_Indata, ConvKernelGPU<double> * d_CKC, double* OutData,
    int TextWidth, double* ReluOutData, int BatchSize,
    int InChannel, int OutChannel);
