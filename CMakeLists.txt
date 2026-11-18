cmake_minimum_required(VERSION 3.16)
project(TmsMultigridFem VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ENABLE_OPENMP "Enable OpenMP parallelization for multigrid kernels" ON)
option(ENABLE_ZLIB "Enable zlib support for compressed .nii.gz files" ON)
option(ENABLE_ONNXRUNTIME "Enable ONNX Runtime deep learning segmentation backend" OFF)

# Core FEM, Multigrid & NIfTI I/O static library
add_library(tms_mg_fem STATIC
    src/grid/MultigridHierarchy.cpp
    src/io/NiftiLoader.cpp
    src/io/VoxelGridResampler.cpp
    src/solver/ConductivityBuilder.cpp
    src/solver/CoilModel.cpp
    src/solver/FemStencil.cpp
    src/solver/MultigridSolver.cpp
    src/solver/FieldPostProcessor.cpp
)

target_include_directories(tms_mg_fem PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/header
)

# Release optimization flags
if(MSVC)
    target_compile_options(tms_mg_fem PRIVATE $<$<CONFIG:Release>:/O2 /fp:fast>)
else()
    target_compile_options(tms_mg_fem PRIVATE $<$<CONFIG:Release>:-O3 -march=native -ffast-math>)
endif()

# OpenMP parallelization
if(ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(tms_mg_fem PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(tms_mg_fem PUBLIC TMS_USE_OPENMP)
    if(MSVC)
        target_compile_options(tms_mg_fem PUBLIC /openmp:llvm)
    endif()
endif()

# Optional zlib support for .nii.gz / .img.gz
if(ENABLE_ZLIB)
    find_package(ZLIB QUIET)
    if(ZLIB_FOUND)
        target_link_libraries(tms_mg_fem PUBLIC ZLIB::ZLIB)
        target_compile_definitions(tms_mg_fem PUBLIC TMS_USE_ZLIB)
    else()
        message(STATUS "zlib not found: uncompressed .nii supported, .nii.gz disabled")
    endif()
endif()

# Reserved hook for future ONNX Runtime segmentation module
if(ENABLE_ONNXRUNTIME)
    target_compile_definitions(tms_mg_fem PUBLIC ENABLE_ONNXRUNTIME)
endif()

# Unit & integration tests
enable_testing()

add_executable(test_multigrid_solver test/test_multigrid_solver.cpp)
target_link_libraries(test_multigrid_solver PRIVATE tms_mg_fem)
add_test(NAME LayeredSphereBenchmark COMMAND test_multigrid_solver)

add_executable(test_nifti_io test/test_nifti_io.cpp)
target_link_libraries(test_nifti_io PRIVATE tms_mg_fem)
add_test(NAME NiftiRoundTripAndResampler COMMAND test_nifti_io)