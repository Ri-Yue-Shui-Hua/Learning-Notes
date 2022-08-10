
# 记录Tensorrt的一些使用



## onnx文件转tensorrt文件

之前有写过C++ 调用Engine模型的内容。实际使用中更偏向于在线转换onnx模型到Engine模型。因为onnx模型是通用格式，可以在线转换成Engine模型。

Engine模型是对特定显卡优化过的模型，如果提供的是Engine模型，则不能将代码在不同的平台上运行。

```cpp
#include <iostream>
#include <fstream>
#include "NvInfer.h"
#include "NvOnnxParser.h"

// 实例化记录器界面。捕获所有警告消息，但忽略信息性消息
class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;


void ONNX2TensorRT(const char* ONNX_file, std::string save_ngine)
{
    // 1.创建构建器的实例
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);

    // 2.创建网络定义
    uint32_t flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(flag);

    // 3.创建一个 ONNX 解析器来填充网络
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);

    // 4.读取模型文件并处理任何错误
    parser->parseFromFile(ONNX_file, static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING));
    for (int32_t i = 0; i < parser->getNbErrors(); ++i)
    {
        std::cout << parser->getError(i)->desc() << std::endl;
    }

    // 5.创建一个构建配置，指定 TensorRT 应该如何优化模型
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();

    // 6.设置属性来控制 TensorRT 如何优化网络
    // 设置内存池的空间
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 16 * (1 << 20));
    // 设置低精度   注释掉为FP32
    if (builder->platformHasFastFp16())
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    // 7.指定配置后，构建引擎
    nvinfer1::IHostMemory* serializedModel = builder->buildSerializedNetwork(*network, *config);

    // 8.保存TensorRT模型
    std::ofstream p(save_ngine, std::ios::binary);
    p.write(reinterpret_cast<const char*>(serializedModel->data()), serializedModel->size());

    // 9.序列化引擎包含权重的必要副本，因此不再需要解析器、网络定义、构建器配置和构建器，可以安全地删除
    delete parser;
    delete network;
    delete config;
    delete builder;

    // 10.将引擎保存到磁盘，并且可以删除它被序列化到的缓冲区
    delete serializedModel;
}


void exportONNX(const char* ONNX_file, std::string save_ngine)
{
    std::ifstream file(ONNX_file, std::ios::binary);
    if (!file.good())
    {
        std::cout << "Load ONNX file failed! No file found from:" << ONNX_file << std::endl;
        return ;
    }

    std::cout << "Load ONNX file from: " << ONNX_file << std::endl;
    std::cout << "Starting export ..." << std::endl;

    ONNX2TensorRT(ONNX_file, save_ngine);

    std::cout << "Export success, saved as: " << save_ngine << std::endl;

}


int main(int argc, char** argv)
{
    // 输入信息
    const char* ONNX_file  = "../weights/test.onnx";
    std::string save_ngine = "../weights/test.engine";

    exportONNX(ONNX_file, save_ngine);

    return 0;
}


```

