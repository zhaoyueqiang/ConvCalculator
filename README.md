# 卷积模块
## 1. 数据结构说明
### 1.1 ConvKernel结构体
ConvKernel模板结构体中包含了卷积核的数据和参数。
#### 1.1.1 数据
| 数据类型 | 数据名称 | 说明 |
| :----: | :----: | :----: |
| int | ConvKernelWidth | 卷积核的宽度 |
| int | ConvKernelHeight | 卷积核的高度 |
| int | ConvKernelDepth | 卷积核的深度 |
| int | ConvKernelChannel | 卷积核的通道数 |
| int | ConvKernelCount | 卷积核的数量 |
| type* | W | 卷积核的数据，可实例化为float、double、int类型 |
| type* | B | 卷积核的偏置，可实例化为float、double、int类型 |

#### 1.1.2 函数
| 函数原型 | 说明 |
| ConvKernel(int Width, int Height, int Channel, int Count) | 二维卷积核构造函数，初始化二维卷积核数据 |
| ConvKernel(int Width, int Height,int Depth, int Channel, int Count) | 三维卷积核构造函数，初始化三维卷积核数据 |
| void InitializeDataToRandom(type fMin, type fMax) | 随机初始化卷积核数据和偏置，fMin为随机数最小值，fMax为随机数最大值 |
| void PrintKernel() | 打印卷积核数据 |
### 1.2 convCalculator类
convCalculator模板类中包含了卷积计算的数据和函数。  
#### 1.2.1 数据
| 数据类型 | 数据名称 | 说明 |
| :----: | :----: | :----: |
| int | mInWidth | 输入数据宽度 |
| int | mInHeight | 输入数据高度 |
| int | mInDepth | 输入三维数据的深度 |
| int | mInChannel | 输入数据通道数 |
| int | mBatchSize | 输入数据的数量 |
| int | mOutWidth | 输出数据宽度 |
| int | mOutHeight | 输出数据高度 |
| int | mOutDepth | 输出三维数据的深度 |
| int | mOutChannel | 输出数据通道数 |
| int | mPadding | 零填充，0为不填充，1为填充，默认为1，填充之后输入与输入数据大小一致 |
| type* | OutData | 输出数据，可实例化为float、double、int类型 |
#### 1.2.2 函数
| 函数原型 | 说明 |
| :----: | :----: |
| convCalculator() | 构造函数，初始化数据 |
| void Conv2D(type* ImageData, int ImageWidth, int ImageHeight, int ImageChannel, int BatchSize, ConvKernel<type>* CK) | 二维卷积函数，使用GPU加速计算 |
| void Conv2D_CPU(type* ImageData, int ImageWidth, int ImageHeight, int ImageChannel, int BatchSize, ConvKernel<type>* CK) | 二维卷积函数，使用CPU计算 |
| void Conv1D(type* TextData,int TextLength,int TextChannel, int BatchSize,ConvKernel<type>* CK) | 一维卷积函数，使用GPU加速计算 |
| void Conv1D_CPU(type* TextData,int TextLength,int TextChannel, int BatchSize,ConvKernel<type>* CK) | 一维卷积函数，使用CPU计算 |
| void Conv3D(type* VideoData,int VideoWidth,int VideoHeight,int VideoChannel,int BatchSize,ConvKernel<type>* CK) | 三维卷积函数，使用GPU加速计算 |
| void Conv3D_CPU(type* VideoData,int VideoWidth,int VideoHeight,int VideoChannel,int BatchSize,ConvKernel<type>* CK) | 三维卷积函数，使用CPU计算 |
| void PrintOutData() | 打印输出数据 |

## 2. 使用方法
在需要调用卷积模块的文件中包含头文件convCalculator.h，实例化convCalculator类的对象，实例化卷积核，设置卷积核的大小和通道数，设置卷积核的参数，目前只有随机设置卷积核数据的函数。然后调用对应卷积函数进行卷积计算。
* 本函数使用的卷积核数据是随机生成的，可以使用随机生成的卷积核数据进行测试，也可以使用自己的卷积核数据进行测试。
### 2.1 函数调用
#### 2.1.1 卷积核初始化
```c++
ConvKernel(int Width, int Height, int Channel, int Count) // 二维卷积核，一维卷积时，Height设置为1
ConvKernel(int Width, int Height,int Depth, int Channel, int Count) // 三维卷积核
```
参数说明：
* Width：卷积核宽度。
* Height：卷积核高度。
* Depth：卷积核深度。
* Channel：卷积核通道数。
* Count：卷积核数量。也是输出数据通道数。
#### 2.1.2 二维卷积
```c++
template<typename type>
void Conv2D(type* ImageData, int ImageWidth, int ImageHeight, int ImageChannel, int BatchSize, ConvKernel<type>* CK)
template<typename type>
void Conv2D_CPU(type* ImageData, int ImageWidth, int ImageHeight, int ImageChannel, int BatchSize, ConvKernel<type>* CK)
```
参数说明：
* ImageData：输入数据，一维数组，大小为ImageWidth * ImageHeight * ImageChannel * BatchSize，按照NCHW方式保存，N为BatchSize，C为ImageChannel，H为ImageHeight，W为ImageWidth。可实例化为double、float、int类型。
* ImageWidth：输入数据宽度。
* ImageHeight：输入数据高度。
* ImageChannel：输入数据通道数。
* BatchSize：输入数据数量。
* CK：卷积核，ConvKernel<type>类型，可实例化为double、float、int类型。  

函数功能说明：  
* 输入数据和卷积核进行二维卷积计算，步长为一，进行零填充保证输出数据与输入数据尺寸相同。输出数据保存在convCalculator类的OutData中。
* Conv2D函数使用GPU加速计算，Conv2D_CPU函数使用CPU计算。
#### 2.1.3 一维卷积
```c++
template<typename type>
void Conv1D(type* TextData,int TextLength,int TextChannel, int BatchSize,ConvKernel<type>* CK)
template<typename type>
void Conv1D_CPU(type* TextData,int TextLength,int TextChannel, int BatchSize,ConvKernel<type>* CK)
```
参数说明：
* TextData：输入数据，一维数组，大小为TextLength * TextChannel * BatchSize，按照NCW方式保存，N为BatchSize，C为TextChannel，W为TextLength。可实例化为double、float、int类型。
* TextLength：输入数据长度。
* TextChannel：输入数据通道数。
* BatchSize：输入数据数量。
* CK：卷积核，ConvKernel<type>类型，可实例化为double、float、int类型。

函数功能说明：
* 输入数据和卷积核进行一维卷积计算，步长为一，进行零填充保证输出数据与输入数据尺寸相同。输出数据保存在convCalculator类的OutData中。
* Conv1D函数使用GPU加速计算，Conv1D_CPU函数使用CPU计算。

#### 2.1.4 三维卷积
```c++
template<typename type>
void Conv3D(type* VideoData,int VideoWidth,int VideoHeight,int VideoDepth,int VideoChannel,int BatchSize,ConvKernel<type>* CK)
template<typename type>
void Conv3D_CPU(type* VideoData,int VideoWidth,int VideoHeight,int VideoDepth,int VideoChannel,int BatchSize,ConvKernel<type>* CK)
```

参数说明：
* VideoData：输入数据，一维数组，大小为VideoWidth * VideoHeight * VideoDepth * VideoChannel * BatchSize，按照NCDHW方式保存，N为BatchSize，C为VideoChannel，D为VideoDepth，H为VideoHeight，W为VideoWidth。可实例化为double、float、int类型。
* VideoWidth：输入数据宽度。
* VideoHeight：输入数据高度。
* VideoDepth：输入数据深度。
* VideoChannel：输入数据通道数。
* BatchSize：输入数据数量。
* CK：卷积核，ConvKernel<type>类型，可实例化为double、float、int类型。

函数功能说明：
* 输入数据和卷积核进行三维卷积计算，步长为一，进行零填充保证输出数据与输入数据尺寸相同。输出数据保存在convCalculator类的OutData中。
* Conv3D函数使用GPU加速计算，Conv3D_CPU函数使用CPU计算。

## 3.函数实现
### 使用GPU加速计算的函数
将输入数据视为图片宽度、图片高度、输出通道数乘以图片数量的三个维度来分配GPU线程，每个线程计算输出特征图的一个像素点。
## 其他说明
卷积步长为一。
在卷积时使用了零填充保证特征图和原图大小一致。  
卷积的结果已经加上了偏置b。
这个是一个基本的example，并没有采用一些效率很高的优化的方法，优化效果不是特别好，比如使用多个线程计算一个像素点，使用多个线程计算一个卷积核，使用多个线程计算一个通道等等。具体优化措施还需要进一步研究。

## 函数接口
``` c++
void Conv2D_G_int(int* ImageData, int ImageWidth, int ImageHeight, int ImageChannel, int BatchSize, ConvKernel<int>* CK)

void Conv2D_G_float(float* ImageData, int ImageWidth, int ImageHeight, int ImageChannel, int BatchSize, ConvKernel<float>* CK)

void Conv2D_G_double(double* ImageData, int ImageWidth, int ImageHeight, int ImageChannel, int BatchSize, ConvKernel<double>* CK)

void Conv2D_C_int(int* ImageData, int ImageWidth, int ImageHeight, int ImageChannel, int BatchSize, ConvKernel<int>* CK)

void Conv2D_C_float(float* ImageData, int ImageWidth, int ImageHeight, int ImageChannel, int BatchSize, ConvKernel<float>* CK)

void Conv2D_C_double(double* ImageData, int ImageWidth, int ImageHeight, int ImageChannel, int BatchSize, ConvKernel<double>* CK)

void Conv1D_G_int(int* TextData,int TextLength,int TextChannel, int BatchSize,ConvKernel<int>* CK)

void Conv1D_G_float(float* TextData,int TextLength,int TextChannel, int BatchSize,ConvKernel<float>* CK)

void Conv1D_G_double(double* TextData,int TextLength,int TextChannel, int BatchSize,ConvKernel<double>* CK)

void Conv1D_C_int(int* TextData,int TextLength,int TextChannel,int BatchSize,ConvKernel<int>* CK)

void Conv1D_C_float(float* TextData,int TextLength,int TextChannel,int BatchSize,ConvKernel<float>* CK)

void Conv1D_C_double(double* TextData,int TextLength,int TextChannel,int BatchSize,ConvKernel<double>* CK)

void Conv3D_G_int(int* VideoData,int VideoWidth,int VideoHeight,int VideoDepth,int VideoChannel,int BatchSize,ConvKernel<int>* CK)

void Conv3D_G_float(float* VideoData,int VideoWidth,int VideoHeight,int VideoDepth,int VideoChannel,int BatchSize,ConvKernel<float>* CK)

void Conv3D_G_double(double* VideoData,int VideoWidth,int VideoHeight,int VideoDepth,int VideoChannel,int BatchSize,ConvKernel<double>* CK)

void Conv3D_C_int(int* VideoData,int VideoWidth,int VideoHeight,int VideoDepth,int VideoChannel,int BatchSize,ConvKernel<int>* CK)

void Conv3D_C_float(float* VideoData,int VideoWidth,int VideoHeight,int VideoDepth,int VideoChannel,int BatchSize,ConvKernel<float>* CK)

void Conv3D_C_double(double* VideoData,int VideoWidth,int VideoHeight,int VideoDepth,int VideoChannel,int BatchSize,ConvKernel<double>* CK)
```

### 编译
采用Makefile编译，卷积数据采用mnistData数据集的train-images.idx3-ubyte数据。