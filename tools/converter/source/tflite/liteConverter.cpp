//
//  liteConverter.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <iostream>

#include "logkit.h"

#include "liteConverter.hpp"
#include "liteOpConverter.hpp"

static MNN::DataType _dataTypeMap(tflite::TensorType type) {
    switch (type) {
        case tflite::TensorType_FLOAT32:
            return MNN::DataType_DT_FLOAT;
            break;
        case tflite::TensorType_INT32:
            return MNN::DataType_DT_INT32;
            break;
        case tflite::TensorType_UINT8:
            return MNN::DataType_DT_UINT8;
            break;
        default:
            return MNN::DataType_DT_FLOAT;
            break;
    }
}

static void _converteConstantDataToMNNConstantNode(
    int tensorIndex, const tflite::TensorT * tensor,
    const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffers, std::unique_ptr<MNN::NetT>& MNNNetT) {
    // check whether buffer data size is greater than zero,
    // if size > 0, then this tensor is Constant, convete this tensor to be MNN Constant node
    const uint32_t bufferIndex = tensor->buffer;
    const auto tensorBuffer    = tfliteModelBuffers[bufferIndex]->data;
    const auto bufferSize      = tensorBuffer.size();
    if (bufferSize == 0)
        return;

    // this is Constant data
    std::unique_ptr<MNN::OpT> mnnConstantOp(new MNN::OpT);
    mnnConstantOp->name      = tensor->name;
    mnnConstantOp->type      = MNN::OpType_Const;
    mnnConstantOp->main.type = MNN::OpParameter_Blob;
    mnnConstantOp->outputIndexes.push_back(tensorIndex);

    std::unique_ptr<MNN::BlobT> mnnBlob(new MNN::BlobT);
    // TODO, map tflite data type to mnn data type
    mnnBlob->dataType   = _dataTypeMap(tensor->type);
    mnnBlob->dataFormat = MNN::MNN_DATA_FORMAT_NHWC;
    mnnBlob->dims       = tensor->shape;

    if (mnnBlob->dataType == MNN::DataType_DT_FLOAT) {
        mnnBlob->float32s.resize(bufferSize / 4);
        memcpy(mnnBlob->float32s.data(), tensorBuffer.data(), bufferSize);
    } else if (mnnBlob->dataType == MNN::DataType_DT_INT32) {
        mnnBlob->int32s.resize(bufferSize / 4);
        memcpy(mnnBlob->int32s.data(), tensorBuffer.data(), bufferSize);
    } else {
        DCHECK(false) << "TODO support other data type! dataType = " << mnnBlob->dataType << ", tensorType == " << tensor->type;
    }
    mnnConstantOp->main.value = mnnBlob.release();

    MNNNetT->tensorName.emplace_back(mnnConstantOp->name);
    MNNNetT->oplists.emplace_back(std::move(mnnConstantOp));
}

static MNN::DataType _convertType(tflite::TensorType type) {
    if (type == tflite::TensorType_FLOAT32) {
        return MNN::DataType_DT_FLOAT;
    }
    if (type == tflite::TensorType_INT8) {
        return MNN::DataType_DT_INT8;
    }
    if (type == tflite::TensorType_UINT8) {
        return MNN::DataType_DT_UINT8;
    }
    if (type == tflite::TensorType_INT32) {
        return MNN::DataType_DT_INT32;
    }
    return MNN::DataType_DT_INVALID;
}
static bool needExtractInput(uint32_t opCode) {
#define NONEED(x) if (x == opCode) return false;
    NONEED(tflite::BuiltinOperator_CONV_2D);
    NONEED(tflite::BuiltinOperator_DEPTHWISE_CONV_2D);
    NONEED(tflite::BuiltinOperator_SPLIT);
    NONEED(tflite::BuiltinOperator_CONCATENATION);
    NONEED(tflite::BuiltinOperator_CONV_2D);
    NONEED(tflite::BuiltinOperator_RESHAPE);
    NONEED(tflite::BuiltinOperator_RESIZE_BILINEAR);
    NONEED(tflite::BuiltinOperator_RESIZE_NEAREST_NEIGHBOR);
    NONEED(tflite::BuiltinOperator_SOFTMAX);
    NONEED(tflite::BuiltinOperator_CUSTOM);


    return true;
}

int tflite2MNNNet(const std::string inputModel, const std::string bizCode, std::unique_ptr<MNN::NetT>& MNNNetT) {
    const std::string model_name = inputModel;
    auto model                   = std::shared_ptr<TfliteModel>(new TfliteModel(model_name));
    model->readModel();
    auto& tfliteModel = model->get();

    const auto& tfliteOpSet = tfliteModel->operator_codes;
    // const auto operatorCodesSize = tfliteOpSet.size();
    const auto subGraphsSize      = tfliteModel->subgraphs.size();
    const auto& tfliteModelBuffer = tfliteModel->buffers;

    // check whether this tflite model is quantization model
    // use the weight's data type of Conv2D|DepthwiseConv2D to decide quantizedModel mode
    bool quantizedModel = true;
    for (int i = 0; i < subGraphsSize; ++i) {
        const auto& ops     = tfliteModel->subgraphs[i]->operators;
        const auto& tensors = tfliteModel->subgraphs[i]->tensors;
        const int opNums    = ops.size();
        for (int j = 0; j < opNums; ++j) {
            const int opcodeIndex = ops[j]->opcode_index;
            const auto opCode     = tfliteOpSet[opcodeIndex]->builtin_code;
            if (opCode == tflite::BuiltinOperator_CONV_2D || opCode == tflite::BuiltinOperator_DEPTHWISE_CONV_2D) {
                const int weightIndex    = ops[j]->inputs[1];
                const auto& weightTensor = tensors[weightIndex];
                quantizedModel           = weightTensor->type == tflite::TensorType_UINT8;
                if(weightTensor->type == tflite::TensorType_INT8){
                    DLOG(ERROR) << "***MNN DO NOT SUPPORT Tflite [INT8] quantized model, please use MNN quantization tool to quantize model***";
                    return -1;
                }
                if (!quantizedModel)
                    break;
            }
        }
    }
    auto& buffers = tfliteModel->buffers;

    MNN::OpT * pDetectorMnnOp = nullptr;
    std::vector<const tflite::TensorT*> detectorTensors;
    int maxTensorIndex = -1;

    for (int i = 0; i < subGraphsSize; ++i) {
        const auto& ops     = tfliteModel->subgraphs[i]->operators;
        const auto& tensors = tfliteModel->subgraphs[i]->tensors;

        // set const
        std::vector<bool> extractedTensors(tfliteModel->subgraphs[i]->tensors.size(), false);

        // set input
        for (const auto index : tfliteModel->subgraphs[i]->inputs) {
            MNN::OpT* inputOp       = new MNN::OpT;
            const auto& inputTensor = tensors[index];
            inputOp->name           = inputTensor->name;
            inputOp->type           = MNN::OpType_Input;
            inputOp->main.type      = MNN::OpParameter_Input;

            auto inputParam     = new MNN::InputT;
            inputParam->dformat = MNN::MNN_DATA_FORMAT_NHWC;
            inputParam->dims = inputTensor->shape;
            inputParam->dtype = _convertType(inputTensor->type);
            inputOp->main.value = inputParam;
            inputOp->outputIndexes.push_back(index);
            MNNNetT->oplists.emplace_back(inputOp);
        }
        if (maxTensorIndex < (int)(tensors.size() - 1))
            maxTensorIndex = tensors.size() - 1;
        // set output names
        for (int k = 0; k < tfliteModel->subgraphs[i]->outputs.size(); ++k) {
            MNNNetT->outputName.push_back(tensors[tfliteModel->subgraphs[i]->outputs[k]]->name);
        }
        // tensor names
        for (const auto& tensor : tensors) {
            MNNNetT->tensorName.push_back(tensor->name);
        }

        const int opNums = ops.size();
        for (int j = 0; j < opNums; ++j) {
            const int opcodeIndex = ops[j]->opcode_index;
            const auto opCode     = tfliteOpSet[opcodeIndex]->builtin_code;
            if (needExtractInput(opCode)) {
                for (auto input : ops[j]->inputs) {
                    if (extractedTensors[input]) {
                        continue;
                    }
                    extractedTensors[input] = true;
                    auto& tensor = tfliteModel->subgraphs[i]->tensors[input];
                    auto& buffer = buffers[tensor->buffer];
                    if (buffer->data.empty()) {
                        continue;
                    }
                    std::unique_ptr<MNN::OpT> newOp(new MNN::OpT);
                    newOp->type = MNN::OpType_Const;
                    newOp->name = tensor->name;
                    newOp->outputIndexes = {input};
                    newOp->main.type = MNN::OpParameter_Blob;
                    newOp->main.value = new MNN::BlobT;
                    auto blob = newOp->main.AsBlob();
                    blob->dims = tensor->shape;
                    blob->dataFormat = MNN::MNN_DATA_FORMAT_NHWC;
                    blob->dataType = _convertType(tensor->type);
                    int size = 1;
                    for (auto s : blob->dims) {
                        size *= s;
                    }
                    void* dst = nullptr;
                    switch (blob->dataType) {
                        case MNN::DataType_DT_FLOAT:
                            blob->float32s.resize(size);
                            dst = blob->float32s.data();
                            break;
                        case MNN::DataType_DT_INT32:
                            blob->int32s.resize(size);
                            dst = blob->int32s.data();
                            break;
                        case MNN::DataType_DT_INT8:
                            blob->int8s.resize(size);
                            dst = blob->int8s.data();
                            break;
                        case MNN::DataType_DT_UINT8:
                            blob->uint8s.resize(size);
                            dst = blob->uint8s.data();
                            break;
                        default:
                            break;
                    }
                    ::memcpy(dst, buffer->data.data(), buffer->data.size());
                    MNNNetT->oplists.emplace_back(std::move(newOp));
                }
            }

            MNN::OpT* op = new MNN::OpT;
            auto creator = liteOpConverterSuit::get()->search(opCode);
            DCHECK(creator) << "NOT_SUPPORTED_OP: [ " << tflite::EnumNameBuiltinOperator(opCode) << " ]";

            // tflite op to MNN op
            op->name      = tensors[ops[j]->outputs[0]]->name;
            op->type      = creator->opType(quantizedModel);
            op->main.type = creator->type(quantizedModel);
            // set default input output index
            op->inputIndexes.resize(ops[j]->inputs.size());
            op->outputIndexes.resize(ops[j]->outputs.size());
            for (int i = 0; i < ops[j]->inputs.size(); i++) {
                op->inputIndexes[i] = ops[j]->inputs[i];
            }
            for (int i = 0; i < ops[j]->outputs.size(); i++) {
                op->outputIndexes[i] = ops[j]->outputs[i];
            }

            if (quantizedModel && opCode == tflite::BuiltinOperator_CUSTOM && tfliteOpSet[opcodeIndex]->custom_code == "TFLite_Detection_PostProcess")
            {
                DCHECK(pDetectorMnnOp == nullptr) << "Only one PostProcess supported";
                pDetectorMnnOp = op;
                const int inputSize = ops[j]->inputs.size();
                for (int k = 0; k < inputSize; ++k)
                    detectorTensors.push_back(tensors[ops[j]->inputs[k]].get());
            }
            else  if (opCode == tflite::BuiltinOperator_CUSTOM) {
                const int inputSize = ops[j]->inputs.size();
                for (int k = 0; k < inputSize; ++k) {
                    _converteConstantDataToMNNConstantNode(ops[j]->inputs[k], tensors[ops[j]->inputs[k]].get(), tfliteModelBuffer, MNNNetT);
                }
            }
            // Run actual conversion
            creator->run(op, ops[j], tensors, tfliteModelBuffer, tfliteOpSet, quantizedModel);
            MNNNetT->oplists.emplace_back(op);
        }
    }
    if (!!pDetectorMnnOp)
    {
        DCHECK(MNNNetT->tensorName.size() == (maxTensorIndex + 1)) << MNNNetT->tensorName.size() << "names, " << (maxTensorIndex+1) << "indices";
        int cnt = pDetectorMnnOp->inputIndexes.size();
        for (int k = 0; k < cnt; ++k)
        {
            int idx = pDetectorMnnOp->inputIndexes[k];
            const auto tensor = detectorTensors[k];
            const auto tensorBuffer = tfliteModelBuffer[tensor->buffer]->data;
            const auto bufferSize = tensorBuffer.size();
            DCHECK(tensor->quantization != nullptr) << "PostProcess tensor quantization missing";
            DCHECK(tensor->quantization->zero_point.size() == 1);
            DCHECK(tensor->quantization->scale.size() == 1);
            float qZero = tensor->quantization->zero_point[0];
            float qScale = tensor->quantization->scale[0];
            if (bufferSize == 0)
            {
                LOG(INFO) << "Adding dequantize for tensor " << tensor->name;
                std::unique_ptr<MNN::OpT> mnnConstantOp1(new MNN::OpT);
                std::unique_ptr<MNN::OpT> mnnConstantOp2(new MNN::OpT);
                mnnConstantOp1->name      = tensor->name + "_DQMIN";
                mnnConstantOp1->type      = MNN::OpType_Const;
                mnnConstantOp1->main.type = MNN::OpParameter_Blob;
                mnnConstantOp1->outputIndexes.push_back(++maxTensorIndex);
                mnnConstantOp2->name      = tensor->name + "_DQMAX";
                mnnConstantOp2->type      = MNN::OpType_Const;
                mnnConstantOp2->main.type = MNN::OpParameter_Blob;
                mnnConstantOp2->outputIndexes.push_back(++maxTensorIndex);
                std::unique_ptr<MNN::BlobT> mnnBlob1(new MNN::BlobT);
                std::unique_ptr<MNN::BlobT> mnnBlob2(new MNN::BlobT);
                mnnBlob1->dataType   = MNN::DataType_DT_FLOAT;
                mnnBlob1->dataFormat = MNN::MNN_DATA_FORMAT_NHWC;
                mnnBlob1->dims.push_back(1);
                mnnBlob1->dims.push_back(1);
                mnnBlob1->dims.push_back(1);
                mnnBlob1->dims.push_back(1);
                mnnBlob1->float32s.push_back((0 - qZero) * qScale);
                mnnBlob2->dataType   = MNN::DataType_DT_FLOAT;
                mnnBlob2->dataFormat = MNN::MNN_DATA_FORMAT_NHWC;
                mnnBlob2->dims.push_back(1);
                mnnBlob2->dims.push_back(1);
                mnnBlob2->dims.push_back(1);
                mnnBlob2->dims.push_back(1);
                mnnBlob2->float32s.push_back((255 - qZero) * qScale);
                mnnConstantOp1->main.value = mnnBlob1.release();
                mnnConstantOp2->main.value = mnnBlob2.release();
                MNNNetT->tensorName.emplace_back(mnnConstantOp1->name);
                MNNNetT->tensorName.emplace_back(mnnConstantOp2->name);
                MNNNetT->oplists.emplace_back(std::move(mnnConstantOp1));
                MNNNetT->oplists.emplace_back(std::move(mnnConstantOp2));

                std::unique_ptr<MNN::OpT> mnnDequantOp(new MNN::OpT);
                mnnDequantOp->name      = tensor->name + "_DQ";
                mnnDequantOp->type      = MNN::OpType_Dequantize;
                mnnDequantOp->main.type = MNN::OpParameter_Dequantize;
                mnnDequantOp->outputIndexes.push_back(++maxTensorIndex);
                mnnDequantOp->inputIndexes.push_back(idx);
                mnnDequantOp->inputIndexes.push_back(maxTensorIndex-2);
                mnnDequantOp->inputIndexes.push_back(maxTensorIndex-1);
                std::unique_ptr<MNN::DequantizeT> mnnDeqPrm(new MNN::DequantizeT);
                mnnDeqPrm->mode = MNN::QuantizeMode_MIN_COMBINED;
                mnnDeqPrm->modelFormat = MNN::ModeFormat_TENSORFLOW;
                mnnDeqPrm->type = MNN::DataType_DT_QUINT8;
                mnnDeqPrm->inputQuantizedParam = std::unique_ptr<MNN::QuantizedParamT>(new MNN::QuantizedParamT);
                mnnDeqPrm->inputQuantizedParam->zeroPoint = tensor->quantization->zero_point[0];
                mnnDeqPrm->inputQuantizedParam->scale     = tensor->quantization->scale[0];

                mnnDequantOp->main.value = mnnDeqPrm.release();
                MNNNetT->tensorName.emplace_back(mnnDequantOp->name);
                MNNNetT->oplists.emplace_back(std::move(mnnDequantOp));
                pDetectorMnnOp->inputIndexes[k] = maxTensorIndex;
            }
            else
            {
                LOG(INFO) << "Adding constant operator for tensor " << tensor->name;
                DCHECK(tensor->type == tflite::TensorType_UINT8);
                std::unique_ptr<MNN::OpT> mnnConstantOp(new MNN::OpT);
                mnnConstantOp->name      = tensor->name;
                mnnConstantOp->type      = MNN::OpType_Const;
                mnnConstantOp->main.type = MNN::OpParameter_Blob;
                mnnConstantOp->outputIndexes.push_back(idx);
                std::unique_ptr<MNN::BlobT> mnnBlob(new MNN::BlobT);
                mnnBlob->dataType   = MNN::DataType_DT_FLOAT;
                mnnBlob->dataFormat = MNN::MNN_DATA_FORMAT_NHWC;
                mnnBlob->dims = tensor->shape;
                for (unsigned int v : tensorBuffer)
                    mnnBlob->float32s.push_back(((float)v - qZero) * qScale);
                mnnConstantOp->main.value = mnnBlob.release();
                MNNNetT->oplists.emplace_back(std::move(mnnConstantOp));
            }
        }
    }


    MNNNetT->sourceType = MNN::NetSource_TFLITE;
    MNNNetT->bizCode    = bizCode;

    return 0;
}

TfliteModel::TfliteModel(const std::string fileName) : _modelName(fileName) {
}

TfliteModel::~TfliteModel() {
}

void TfliteModel::readModel() {
    std::ifstream inputFile(_modelName, std::ios::binary);
    inputFile.seekg(0, std::ios::end);
    const auto size = inputFile.tellg();
    inputFile.seekg(0, std::ios::beg);

    char* buffer = new char[size];
    inputFile.read(buffer, size);
    inputFile.close();

    // verify model
    flatbuffers::Verifier verify((uint8_t*)buffer, size);
    if (!tflite::VerifyModelBuffer(verify)) {
        LOG(FATAL) << "TFlite model version ERROR!";
    }

    _tfliteModel = tflite::UnPackModel(buffer);
    delete[] buffer;
}

std::unique_ptr<tflite::ModelT>& TfliteModel::get() {
    return _tfliteModel;
}
