// header/segmentation/ITissueSegmenter.h
#pragma once
#include "SegmentationTypes.h"
#include <memory>
#include <string>

class ITissueSegmenter {
public:
    virtual ~ITissueSegmenter() = default;
    virtual SegmentationResult segment(const MriVolume& mri) const = 0;
    virtual std::string name() const = 0;
};

enum class SegmenterBackend {
    GmmSpatial,   // Lightweight pure-C++ EM-GMM + MRF/Atlas
    OnnxRuntime   // Future deep learning inference backend
};

std::unique_ptr<ITissueSegmenter> createSegmenter(
    SegmenterBackend backend,
    const std::string& resource_path // Atlas path for GMM, or .onnx model path for DL
);