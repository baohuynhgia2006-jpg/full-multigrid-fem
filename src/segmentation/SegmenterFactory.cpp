#include "segmentation/ITissueSegmenter.h"
#include "segmentation/GmmSpatialSegmenter.h"
#include <stdexcept>

#ifdef ENABLE_ONNXRUNTIME
#include "segmentation/OnnxSegmenter.h"
#endif

std::unique_ptr<ITissueSegmenter> createSegmenter(
    SegmenterBackend backend,
    const std::string& resource_path
) {
    switch (backend) {
        case SegmenterBackend::GmmSpatial:
            return std::make_unique<GmmSpatialSegmenter>(resource_path);

        case SegmenterBackend::OnnxRuntime:
#ifdef ENABLE_ONNXRUNTIME
            return std::make_unique<OnnxSegmenter>(resource_path);
#else
            throw std::runtime_error(
                "OnnxRuntime backend requested, but project was built with -DENABLE_ONNXRUNTIME=OFF"
            );
#endif
        default:
            throw std::invalid_argument("Unknown SegmenterBackend enum value");
    }
}