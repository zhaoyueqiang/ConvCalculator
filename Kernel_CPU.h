#include <stdio.h>
#include <stdlib.h>

/*
卷积层中的卷积核，每个卷积核的参数包括：
1. 卷积核的宽度
2. 卷积核的高度
3. 卷积核的深度
4. 卷积核的个数
5. 卷积核的权重
6. 卷积核的偏置
*/

#define SAFE_DELETE_ARRAY(p) { if(p) { delete[] (p);   (p)=NULL; } }
#define SAFE_DELETE(p)       { if(p) { delete (p);     (p)=NULL; } }

template <typename type>
inline type UnitRandom()          //单位随机数
{
    static bool FirstRun = true;
    if (FirstRun) {
        srand((unsigned)time(NULL));
        FirstRun = false;
    }
    return (type)(rand()) / type(RAND_MAX);
}

template <typename type>
struct ConvBlocks
{
	type* ConvArray;
	int ArrayLen;
	ConvBlocks()
	{
		ConvArray = nullptr;
		ArrayLen = 0;
	}
	~ConvBlocks()
	{
		Reset();
	}
	void Reset()
	{
		SAFE_DELETE_ARRAY(ConvArray);
		ArrayLen = 0;
	}
};

template<typename type>
struct ConvKernel
{
	int ConvKernelWidth;
	int ConvKernelHeight;
	int ConvKernelDepth;
	int ConvKernelChannel;
	int ConvKernelCount;  // kernel 个数即为 输出 OutChannels 的值

	type* W;
	type* B;

    
    //下面的参数是在二维卷积反向传播的时候用到的
	// int ImageInputWidth;
	// int ImageInputHeight;
	// ConvBlocks* CB;

	// type* WRotate180;				// 反向传播的时候用到，旋转180度
	// type* DropOutGradient;             // 反向传播时计算DropOut的梯度
	// type* ImageMaxPoolGradientData;    //  反向传播时计算最大池化的梯度
	// type* ImageReluGradientData;       // 反向传播时计算激活函数的梯度

	ConvKernel(int Width, int Height, int Channel, int Count)
	{
		ConvKernelWidth = Width;
		ConvKernelHeight = Height;
		ConvKernelDepth = 1;
		ConvKernelChannel = Channel;
		ConvKernelCount = Count;
		W = new type[Width * Height * Channel * Count];
		// WRotate180 = new type[Width * Height * Channel * Count];
		B = new type[Count];
		// CB = nullptr;
		// ImageInputWidth = 0;
		// ImageInputHeight = 0;
		// DropOutGradient = nullptr;
		// ImageMaxPoolGradientData = nullptr;
		// ImageReluGradientData = nullptr;

	}
	ConvKernel(int Width, int Height,int Depth, int Channel, int Count)
	{
		ConvKernelWidth = Width;
		ConvKernelHeight = Height;
		ConvKernelDepth = Depth;
		ConvKernelChannel = Channel;
		ConvKernelCount = Count;
		W = new type[Width * Height * Depth * Channel * Count];
		// WRotate180 = new type[Width * Height * Depth * Channel * Count];
		B = new type[Count];
		// CB = nullptr;
		// ImageInputWidth = 0;
		// ImageInputHeight = 0;
		// DropOutGradient = nullptr;
		// ImageMaxPoolGradientData = nullptr;
		// ImageReluGradientData = nullptr;
	}
	~ConvKernel()
	{
		SAFE_DELETE_ARRAY(W);
		// SAFE_DELETE_ARRAY(WRotate180);
		SAFE_DELETE_ARRAY(B);
		// SAFE_DELETE_ARRAY(DropOutGradient);
		// SAFE_DELETE_ARRAY(ImageMaxPoolGradientData);
		// SAFE_DELETE_ARRAY(ImageReluGradientData);

		// SAFE_DELETE(CB);
	}
    
    // 初始化卷积核数据
    void InitializeDataToRandom(type fMin, type fMax)
	{
		int Len = ConvKernelWidth * ConvKernelHeight * ConvKernelDepth * ConvKernelChannel * ConvKernelCount;
		for (int i = 0; i < Len; i++) {
			W[i] = fMin + (fMax - fMin) * UnitRandom<type>();
			//  W[i] = 1;
		}
		for (int i = 0; i < ConvKernelCount; i++) {
			B[i] = fMin + (fMax - fMin) * UnitRandom<type>();
			//  B[i] = 1;
		}
	}

	void PrintKernel()
	{
		printf("ConvKernelWidth = %d\n", ConvKernelWidth);
		printf("ConvKernelHeight = %d\n", ConvKernelHeight);
		printf("ConvKernelDepth = %d\n", ConvKernelDepth);
		printf("ConvKernelChannel = %d\n", ConvKernelChannel);
		printf("ConvKernelCount = %d\n", ConvKernelCount);
		for(int i=0;i <  ConvKernelCount;i++)
		{
			printf("COnvKernel: %d\n", i);
			for(int j=0;j < ConvKernelChannel;j++)
			{
				// 输出三维卷积核的数据
				for(int k=0;k < ConvKernelDepth;k++)
				{
					for(int l=0;l < ConvKernelHeight;l++)
					{
						for(int m=0;m < ConvKernelWidth;m++)
						{
							printf("%f\t ", W[i * ConvKernelChannel * ConvKernelDepth * ConvKernelHeight * ConvKernelWidth + j * ConvKernelDepth * ConvKernelHeight * ConvKernelWidth + k * ConvKernelHeight * ConvKernelWidth + l * ConvKernelWidth + m]);
						}
						printf("\n");
					}
					printf("\n");
				}
			}
			printf("bias: %f\n", B[i]);
			printf("\n");
		}
		// printf("CB = %p\n", CB);
		// printf("ImageInputWidth = %d\n", ImageInputWidth);
		// printf("ImageInputHeight = %d\n", ImageInputHeight);
		// printf("WRotate180 = %p\n", WRotate180);
		// printf("DropOutGradient = %p\n", DropOutGradient);
		// printf("ImageMaxPoolGradientData = %p\n", ImageMaxPoolGradientData);
		// printf("ImageReluGradientData = %p\n", ImageReluGradientData);
	}
};
