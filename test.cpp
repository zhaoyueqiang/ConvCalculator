#include <iostream>
#include <string>
#include "convCalculator.h"
#include <time.h>
#include <cmath>
// #include <torch/csrc/api/include/torch/torch.h>

// 测试函数 2D卷积： 2D卷积的底层实现是使用滑动窗口的方式，在图像的宽度和高度上滑动卷积核，然后在每个位置上进行元素点积运算，得到输出图像的对应位置的值。


//mnist数据集的数据
const int batch_size = 1000;//取batch_size张图片
const int classes_count = 10;//输出的label种类有0,1,2,3... 十种
const int width  = 28;
const int height = 28;
const int ImageChannel = 1;

//大端小端转换
int translateEndian_32(int i){
    return ((i & 0x000000FF) << 24 | (i & 0x0000FF00) << 8 | (i & 0x00FF0000) >> 8 | (i & 0xFF000000) >> 24);
}

// type为inmt、float和double类型
template<typename type>
void load_mnist_data(type* &ImageData, const char * file_name){
    FILE *fp = NULL;
    fp = fopen(file_name, "rb");
    if(fp == NULL) {
        std::cout << "load mnist data failed." << std::endl;
        return;
    }
    int magic_number = 0;
    int sample_number = 0;
    int n_rows = 0, n_cols = 0;
    fread((int *)&magic_number, sizeof(magic_number), 1, fp);       //magic number应该占用四个字节，sizeof(magic_number)输出可能不是四，因为sizeof是在编译时计算的
    //文件存储格式为大端，Intel CPU架构存储为小端，所以需要将字节序反转
    magic_number = translateEndian_32(magic_number);
//    cout << "magic number = " << magic_number << endl;

    fread((int *)&sample_number, sizeof(sample_number), 1, fp);     //图片张数
    sample_number = translateEndian_32(sample_number);
//    cout << "sample number = " << sample_number << endl;

    fread((int *)&n_rows, sizeof(n_rows), 1, fp);       //图片宽度
    n_rows = translateEndian_32(n_rows);
//    cout << "n_rows = " << n_rows << endl;

    fread((int *)&n_cols, sizeof(n_cols), 1, fp);       //图片高度
    n_cols = translateEndian_32(n_cols);
//    cout << "n_cols = " << n_cols << endl;

    unsigned char temp;
    type normalize_max = 1, normalize_min = -1;
//    type normalize_max = 1.175, normalize_min = -0.1;
    ImageData = (type *)malloc(batch_size * n_rows * n_cols * sizeof(type));
    memset(ImageData, 0, batch_size * n_rows * n_cols * sizeof(type));
    if(typeid(ImageData).name() == "int")
    {
        for(int k = 0; k < batch_size; k++){            //读取batch_size张图片 实际迭代时需要每次读取不同的batch_size张图片
            for(int i = 0; i < n_rows; i++){
                for(int j = 0; j < n_cols; j++){
                    fread(&temp, 1, 1, fp);
                    ImageData[i * n_cols + j] = (type)temp;
                }
            }
        }
    }
    else
    {
        for(int k = 0; k < batch_size; k++){            //读取batch_size张图片 实际迭代时需要每次读取不同的batch_size张图片
            for(int i = 0; i < n_rows; i++){
                for(int j = 0; j < n_cols; j++){
                    fread(&temp, 1, 1, fp);
                    ImageData[i * n_cols + j] = (type)temp/255 * (normalize_max - normalize_min) + normalize_min;//灰度归一化 
                }
            }
        }
    }
    
    fclose(fp);
    fp = NULL;
}

template<typename type>
bool ResultCheck(type *result_gpu,int size, type* result_cpu) {
    printf("Checking computed result for correctness: ");
    bool correct = true;
    
    // test relative error by the formula
    //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
    double eps = 1.e-6; // machine zero

    for (int i = 0; i < size; i++) {
        double abs_err = fabs(result_gpu[i] - result_cpu[i]);
        double abs_rel = fabs(result_cpu[i]);
        double abs_val = fabs(result_gpu[i]);
        double rel_err = abs_err / abs_val / abs_rel;

        if (rel_err > eps) {
            // printf("Error! Matrix[%05d]=%.8f, result_cpu=%.8f error term is > %E\n", i, result_gpu[i], result_cpu[i], eps);
            correct = false;
        }
    }
    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");
    return correct;
}

int main()
{
    clock_t start, end;
    std::cout << "--------------------------------------------" << std::endl;
    std::cout << "data type = \"double\"" << std::endl;
    // 设置卷积核的数据
    ConvKernel<double>* kernel = new ConvKernel<double>(5, 5, 1, 32);
    kernel->InitializeDataToRandom(-0.1d, 0.1d);

    // kernel->PrintKernel();

    // 设置输入数据
    double *imageData ;
    load_mnist_data<double>(imageData, "train-images.idx3-ubyte");

    convCalculator<double> conv;
    start = clock();
    conv.Conv2D_G_double(imageData,width,height,ImageChannel,batch_size,kernel);
    end = clock();
    std::cout << "Conv2D with GPU time: " << (double)(end - start) / CLOCKS_PER_SEC << "s" << std::endl;

    convCalculator<double> conv2;
    start = clock();
    conv2.Conv2D_C_double(imageData,width,height,ImageChannel,batch_size,kernel);
    end = clock();
    std::cout << "Conv2D CPU time: " << (float)(end - start) / CLOCKS_PER_SEC << "s" << std::endl;

    ResultCheck(conv.OutData, batch_size * width * height * kernel->ConvKernelCount, conv2.OutData);

    std::cout << "--------------------------------------------" << std::endl;
    std::cout << "data type = \"float\"" << std::endl;

    // 设置卷积核的数据
    ConvKernel<float>* kernel2 = new ConvKernel<float>(5, 5, 1, 32);
    kernel2->InitializeDataToRandom(-0.1f, 0.1f);

    // 设置输入数据
    float *imageData2 ;
    load_mnist_data<float>(imageData2, "train-images.idx3-ubyte");

    convCalculator<float> conv3;
    start = clock();
    conv3.Conv2D_G_float(imageData2,width,height,ImageChannel,batch_size,kernel2);
    end = clock();
    std::cout << "Conv2D with GPU time: " << (float)(end - start) / CLOCKS_PER_SEC << "s" << std::endl;

    convCalculator<float> conv4;
    start = clock();
    conv4.Conv2D_C_float(imageData2,width,height,ImageChannel,batch_size,kernel2);
    end = clock();
    std::cout << "Conv2D CPU time: " << (float)(end - start) / CLOCKS_PER_SEC << "s" << std::endl;

    ResultCheck(conv3.OutData, batch_size * width * height * kernel2->ConvKernelCount, conv4.OutData);

    std::cout << "--------------------------------------------" << std::endl;
    std::cout << "data type = \"int\"" << std::endl;

    // 设置卷积核的数据
    ConvKernel<int>* kernel3 = new ConvKernel<int>(5, 5, 1, 32);
    kernel3->InitializeDataToRandom(-1, 1);

    int *imageData3 ;
    load_mnist_data<int>(imageData3, "train-images.idx3-ubyte");

    convCalculator<int> conv5;
    start = clock();
    conv5.Conv2D_G_int(imageData3,width,height,ImageChannel,batch_size,kernel3);
    end = clock();
    std::cout << "Conv2D with GPU time: " << (float)(end - start) / CLOCKS_PER_SEC << "s" << std::endl;

    convCalculator<int> conv6;
    start = clock();
    conv6.Conv2D_C_int(imageData3,width,height,ImageChannel,batch_size,kernel3);
    end = clock();
    std::cout << "Conv2D CPU time: " << (float)(end - start) / CLOCKS_PER_SEC << "s" << std::endl;
 
    ResultCheck(conv5.OutData, batch_size * width * height * kernel3->ConvKernelCount, conv6.OutData);

    std::cout << "--------------------------------------------" << std::endl;
    std::cout << "data type = \"double\"" << std::endl;
    // 设置卷积核的数据
    ConvKernel<double>* kernel4 = new ConvKernel<double>(5, 1, 1, 32);
    kernel4->InitializeDataToRandom(-0.1d, 0.1d);

    // 设置输入数据
    double *TextData ;
    load_mnist_data<double>(TextData, "train-images.idx3-ubyte");

    convCalculator<double> conv7;
    start = clock();
    conv7.Conv1D_G_double(TextData,width,ImageChannel,batch_size,kernel4);
    end = clock();
    std::cout << "Conv1D with GPU time: " << (double)(end - start) / CLOCKS_PER_SEC << "s" << std::endl;

    convCalculator<double> conv8;
    start = clock();
    conv8.Conv1D_C_double(TextData,width,ImageChannel,batch_size,kernel4);
    end = clock();
    std::cout << "Conv1D CPU time: " << (float)(end - start) / CLOCKS_PER_SEC << "s" << std::endl;

    ResultCheck(conv7.OutData, batch_size * width * kernel4->ConvKernelCount, conv8.OutData);

    std::cout << "--------------------------------------------" << std::endl;
    std::cout << "data type = \"float\"" << std::endl;

    // 设置卷积核的数据
    ConvKernel<float>* kernel5 = new ConvKernel<float>(5, 1, 1, 32);
    kernel5->InitializeDataToRandom(-0.1f, 0.1f);

    // 设置输入数据
    float *TextData2 ;
    load_mnist_data<float>(TextData2, "train-images.idx3-ubyte");

    convCalculator<float> conv9;
    start = clock();
    conv9.Conv1D_G_float(TextData2,width,ImageChannel,batch_size,kernel5);
    end = clock();
    std::cout << "Conv1D with GPU time: " << (float)(end - start) / CLOCKS_PER_SEC << "s" << std::endl;

    convCalculator<float> conv10;
    start = clock();
    conv10.Conv1D_C_float(TextData2,width,ImageChannel,batch_size,kernel5);
    end = clock();
    std::cout << "Conv1D CPU time: " << (float)(end - start) / CLOCKS_PER_SEC << "s" << std::endl;

    ResultCheck(conv9.OutData, batch_size * width * kernel5->ConvKernelCount, conv10.OutData);

    std::cout << "--------------------------------------------" << std::endl;
    std::cout << "data type = \"int\"" << std::endl;

    // 设置卷积核的数据
    ConvKernel<int>* kernel6 = new ConvKernel<int>(5, 1, 1, 32);
    kernel6->InitializeDataToRandom(-1, 1);

    // 设置输入数据
    int *TextData3 ;
    load_mnist_data<int>(TextData3, "train-images.idx3-ubyte");

    convCalculator<int> conv11;
    start = clock();
    conv11.Conv1D_G_int(TextData3,width,ImageChannel,batch_size,kernel6);
    end = clock();
    std::cout << "Conv1D with GPU time: " << (float)(end - start) / CLOCKS_PER_SEC << "s" << std::endl;

    convCalculator<int> conv12;
    start = clock();
    conv12.Conv1D_C_int(TextData3,width,ImageChannel,batch_size,kernel6);
    end = clock();
    std::cout << "Conv1D CPU time: " << (float)(end - start) / CLOCKS_PER_SEC << "s" << std::endl;

    ResultCheck(conv11.OutData, batch_size * width * kernel6->ConvKernelCount, conv12.OutData);

    std::cout << "--------------------------------------------" << std::endl;

    std::cout << "data type = \"double\"" << std::endl;
    // 设置卷积核的数据
    ConvKernel<double>* kernel7 = new ConvKernel<double>(3, 3, 3, 1, 32);
    kernel7->InitializeDataToRandom(-0.1d, 0.1d);

    // kernel7->PrintKernel();

    // 设置输入数据
    double *imageData4 ;
    int VideoDepth = 3;
    load_mnist_data<double>(imageData4, "train-images.idx3-ubyte");


    convCalculator<double> conv13;
    start = clock();
    conv13.Conv3D_G_double(imageData4,width,height,VideoDepth,ImageChannel,batch_size/3,kernel7);
    end = clock();
    std::cout << "Conv3D with GPU time: " << (double)(end - start) / CLOCKS_PER_SEC << "s" << std::endl;

    convCalculator<double> conv14;
    start = clock();
    conv14.Conv3D_C_double(imageData4,width,height,VideoDepth,ImageChannel,batch_size/3,kernel7);
    end = clock();
    std::cout << "Conv3D CPU time: " << (float)(end - start) / CLOCKS_PER_SEC << "s" << std::endl;

    ResultCheck(conv13.OutData, batch_size/3 * width * height * VideoDepth * kernel7->ConvKernelCount, conv14.OutData);

    std::cout << "--------------------------------------------" << std::endl;

    std::cout << "data type = \"float\"" << std::endl;
    // 设置卷积核的数据
    ConvKernel<float>* kernel8 = new ConvKernel<float>(3, 3, 3, 1, 2);
    kernel8->InitializeDataToRandom(-0.1f, 0.1f);

    // 设置输入数据
    float *imageData5 ;
    load_mnist_data<float>(imageData5, "train-images.idx3-ubyte");

    convCalculator<float> conv15;
    start = clock();
    conv15.Conv3D_G_float(imageData5,width,height,VideoDepth,ImageChannel,batch_size/3,kernel8);
    end = clock();
    std::cout << "Conv3D with GPU time: " << (float)(end - start) / CLOCKS_PER_SEC << "s" << std::endl;

    convCalculator<float> conv16;
    start = clock();
    conv16.Conv3D_C_float(imageData5,width,height,VideoDepth,ImageChannel,batch_size/3,kernel8);
    end = clock();
    std::cout << "Conv3D CPU time: " << (float)(end - start) / CLOCKS_PER_SEC << "s" << std::endl;

    ResultCheck(conv15.OutData, batch_size/3 * width * height * VideoDepth * kernel8->ConvKernelCount, conv16.OutData);

    std::cout << "--------------------------------------------" << std::endl;

    std::cout << "data type = \"int\"" << std::endl;
    // 设置卷积核的数据
    ConvKernel<int>* kernel9 = new ConvKernel<int>(3, 3, 3, 1, 2);
    kernel9->InitializeDataToRandom(-1, 1);

    // 设置输入数据
    int *imageData6 ;
    load_mnist_data<int>(imageData6, "train-images.idx3-ubyte");

    convCalculator<int> conv17;
    start = clock();
    conv17.Conv3D_G_int(imageData6,width,height,VideoDepth,ImageChannel,batch_size/3,kernel9);
    end = clock();
    std::cout << "Conv3D with GPU time: " << (float)(end - start) / CLOCKS_PER_SEC << "s" << std::endl;

    convCalculator<int> conv18;
    start = clock();
    conv18.Conv3D_C_int(imageData6,width,height,VideoDepth,ImageChannel,batch_size/3,kernel9);
    end = clock();
    std::cout << "Conv3D CPU time: " << (float)(end - start) / CLOCKS_PER_SEC << "s" << std::endl;

    ResultCheck(conv17.OutData, batch_size/3 * width * height * VideoDepth * kernel9->ConvKernelCount, conv18.OutData);

    std::cout << "--------------------------------------------" << std::endl;


    return 0;
}