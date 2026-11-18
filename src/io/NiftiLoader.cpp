#include "io/NiftiLoader.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

#ifdef TMS_USE_ZLIB
#include <zlib.h>
#endif

namespace {

#pragma pack(push, 1)
struct Nifti1Header {
    int32_t sizeof_hdr;
    char data_type[10];
    char db_name[18];
    int32_t extents;
    int16_t session_error;
    char regular;
    char dim_info;
    int16_t dim[8];
    float intent_p1;
    float intent_p2;
    float intent_p3;
    int16_t intent_code;
    int16_t datatype;
    int16_t bitpix;
    int16_t slice_start;
    float pixdim[8];
    float vox_offset;
    float scl_slope;
    float scl_inter;
    int16_t slice_end;
    char slice_code;
    char xyzt_units;
    float cal_max;
    float cal_min;
    float slice_duration;
    float toffset;
    int32_t glmax;
    int32_t glmin;
    char descrip[80];
    char aux_file[24];
    int16_t qform_code;
    int16_t sform_code;
    float quatern_b;
    float quatern_c;
    float quatern_d;
    float qoffset_x;
    float qoffset_y;
    float qoffset_z;
    float srow_x[4];
    float srow_y[4];
    float srow_z[4];
    char intent_name[16];
    char magic[4];
};

struct Nifti2Header {
    int32_t sizeof_hdr;
    char magic[8];
    int16_t datatype;
    int16_t bitpix;
    int64_t dim[8];
    double intent_p1;
    double intent_p2;
    double intent_p3;
    double pixdim[8];
    int64_t vox_offset;
    double scl_slope;
    double scl_inter;
    double cal_max;
    double cal_min;
    double slice_duration;
    double toffset;
    int64_t slice_start;
    int64_t slice_end;
    char descrip[80];
    char aux_file[24];
    int32_t qform_code;
    int32_t sform_code;
    double quatern_b;
    double quatern_c;
    double quatern_d;
    double qoffset_x;
    double qoffset_y;
    double qoffset_z;
    double srow_x[4];
    double srow_y[4];
    double srow_z[4];
    int32_t slice_code;
    int32_t xyzt_units;
    int32_t intent_code;
    char intent_name[16];
    char dim_info;
    char unused_str[15];
};
#pragma pack(pop)

static_assert(sizeof(Nifti1Header) == 348, "Nifti1Header must be 348 bytes");
static_assert(sizeof(Nifti2Header) == 540, "Nifti2Header must be 540 bytes");

struct NormalizedNiftiMeta {
    int version = 1;
    bool swapped = false;
    int16_t datatype = 0;
    int16_t bitpix = 0;
    int64_t dim[8]{};
    double pixdim[8]{};
    int64_t vox_offset = 0;
    double scl_slope = 1.0;
    double scl_inter = 0.0;
    int32_t qform_code = 0;
    int32_t sform_code = 0;
    double quatern_b = 0.0;
    double quatern_c = 0.0;
    double quatern_d = 0.0;
    double qoffset_x = 0.0;
    double qoffset_y = 0.0;
    double qoffset_z = 0.0;
    double srow_x[4]{};
    double srow_y[4]{};
    double srow_z[4]{};
    int32_t xyzt_units = 0;
    char magic[8]{};
};

template <typename T>
T byteSwap(T val) {
    T out{};
    const auto* src = reinterpret_cast<const uint8_t*>(&val);
    auto* dst = reinterpret_cast<uint8_t*>(&out);
    for (size_t i = 0; i < sizeof(T); ++i) {
        dst[i] = src[sizeof(T) - 1 - i];
    }
    return out;
}

void swapNifti1Endianness(Nifti1Header& h) {
    h.sizeof_hdr = byteSwap(h.sizeof_hdr);
    h.extents = byteSwap(h.extents);
    h.session_error = byteSwap(h.session_error);
    for (int i = 0; i < 8; ++i) h.dim[i] = byteSwap(h.dim[i]);
    h.intent_p1 = byteSwap(h.intent_p1);
    h.intent_p2 = byteSwap(h.intent_p2);
    h.intent_p3 = byteSwap(h.intent_p3);
    h.intent_code = byteSwap(h.intent_code);
    h.datatype = byteSwap(h.datatype);
    h.bitpix = byteSwap(h.bitpix);
    h.slice_start = byteSwap(h.slice_start);
    for (int i = 0; i < 8; ++i) h.pixdim[i] = byteSwap(h.pixdim[i]);
    h.vox_offset = byteSwap(h.vox_offset);
    h.scl_slope = byteSwap(h.scl_slope);
    h.scl_inter = byteSwap(h.scl_inter);
    h.slice_end = byteSwap(h.slice_end);
    h.cal_max = byteSwap(h.cal_max);
    h.cal_min = byteSwap(h.cal_min);
    h.slice_duration = byteSwap(h.slice_duration);
    h.toffset = byteSwap(h.toffset);
    h.glmax = byteSwap(h.glmax);
    h.glmin = byteSwap(h.glmin);
    h.qform_code = byteSwap(h.qform_code);
    h.sform_code = byteSwap(h.sform_code);
    h.quatern_b = byteSwap(h.quatern_b);
    h.quatern_c = byteSwap(h.quatern_c);
    h.quatern_d = byteSwap(h.quatern_d);
    h.qoffset_x = byteSwap(h.qoffset_x);
    h.qoffset_y = byteSwap(h.qoffset_y);
    h.qoffset_z = byteSwap(h.qoffset_z);
    for (int i = 0; i < 4; ++i) {
        h.srow_x[i] = byteSwap(h.srow_x[i]);
        h.srow_y[i] = byteSwap(h.srow_y[i]);
        h.srow_z[i] = byteSwap(h.srow_z[i]);
    }
}

void swapNifti2Endianness(Nifti2Header& h) {
    h.sizeof_hdr = byteSwap(h.sizeof_hdr);
    h.datatype = byteSwap(h.datatype);
    h.bitpix = byteSwap(h.bitpix);
    for (int i = 0; i < 8; ++i) h.dim[i] = byteSwap(h.dim[i]);
    h.intent_p1 = byteSwap(h.intent_p1);
    h.intent_p2 = byteSwap(h.intent_p2);
    h.intent_p3 = byteSwap(h.intent_p3);
    for (int i = 0; i < 8; ++i) h.pixdim[i] = byteSwap(h.pixdim[i]);
    h.vox_offset = byteSwap(h.vox_offset);
    h.scl_slope = byteSwap(h.scl_slope);
    h.scl_inter = byteSwap(h.scl_inter);
    h.cal_max = byteSwap(h.cal_max);
    h.cal_min = byteSwap(h.cal_min);
    h.slice_duration = byteSwap(h.slice_duration);
    h.toffset = byteSwap(h.toffset);
    h.slice_start = byteSwap(h.slice_start);
    h.slice_end = byteSwap(h.slice_end);
    h.qform_code = byteSwap(h.qform_code);
    h.sform_code = byteSwap(h.sform_code);
    h.quatern_b = byteSwap(h.quatern_b);
    h.quatern_c = byteSwap(h.quatern_c);
    h.quatern_d = byteSwap(h.quatern_d);
    h.qoffset_x = byteSwap(h.qoffset_x);
    h.qoffset_y = byteSwap(h.qoffset_y);
    h.qoffset_z = byteSwap(h.qoffset_z);
    for (int i = 0; i < 4; ++i) {
        h.srow_x[i] = byteSwap(h.srow_x[i]);
        h.srow_y[i] = byteSwap(h.srow_y[i]);
        h.srow_z[i] = byteSwap(h.srow_z[i]);
    }
    h.slice_code = byteSwap(h.slice_code);
    h.xyzt_units = byteSwap(h.xyzt_units);
    h.intent_code = byteSwap(h.intent_code);
}

bool endsWith(const std::string& str, const std::string& suffix) {
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string resolveSiblingImgPath(const std::string& hdr_path) {
    if (endsWith(hdr_path, ".hdr.gz")) {
        return hdr_path.substr(0, hdr_path.size() - 7) + ".img.gz";
    }
    if (endsWith(hdr_path, ".hdr")) {
        return hdr_path.substr(0, hdr_path.size() - 4) + ".img";
    }
    throw std::runtime_error("Two-file NIfTI magic ('ni1'/'ni2') requires a .hdr/.hdr.gz path: " + hdr_path);
}

std::vector<uint8_t> readFileBytes(const std::string& filepath) {
    if (endsWith(filepath, ".gz")) {
#ifdef TMS_USE_ZLIB
        gzFile gz = gzopen(filepath.c_str(), "rb");
        if (!gz) {
            throw std::runtime_error("Failed to open gzipped NIfTI file: " + filepath);
        }
        std::vector<uint8_t> buffer;
        uint8_t chunk[65536];
        int bytes_read = 0;
        while ((bytes_read = gzread(gz, chunk, sizeof(chunk))) > 0) {
            buffer.insert(buffer.end(), chunk, chunk + bytes_read);
        }
        int err = 0;
        const char* err_msg = gzerror(gz, &err);
        gzclose(gz);
        if (err != Z_OK && err != Z_STREAM_END) {
            throw std::runtime_error(
                std::string("zlib decompression error in ") + filepath + ": " + err_msg
            );
        }
        return buffer;
#else
        throw std::runtime_error(
            "Cannot read .nii.gz without zlib (rebuild with -DENABLE_ZLIB=ON): " + filepath
        );
#endif
    }

    std::ifstream ifs(filepath, std::ios::binary | std::ios::ate);
    if (!ifs) {
        throw std::runtime_error("Failed to open NIfTI file: " + filepath);
    }
    const std::streamsize size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(static_cast<size_t>(size));
    if (!ifs.read(reinterpret_cast<char*>(buffer.data()), size)) {
        throw std::runtime_error("Failed to read bytes from NIfTI file: " + filepath);
    }
    return buffer;
}

void writeFileBytes(const std::string& filepath, const uint8_t* data, size_t size) {
    if (endsWith(filepath, ".gz")) {
#ifdef TMS_USE_ZLIB
        gzFile gz = gzopen(filepath.c_str(), "wb6");
        if (!gz) {
            throw std::runtime_error("Failed to open output .nii.gz file: " + filepath);
        }
        size_t offset = 0;
        while (offset < size) {
            const unsigned int chunk = static_cast<unsigned int>(
                std::min<size_t>(size - offset, 1u << 30)
            );
            const int written = gzwrite(gz, data + offset, chunk);
            if (written <= 0) {
                gzclose(gz);
                throw std::runtime_error("Failed writing compressed NIfTI data: " + filepath);
            }
            offset += static_cast<size_t>(written);
        }
        gzclose(gz);
        return;
#else
        throw std::runtime_error(
            "Cannot write .nii.gz without zlib (rebuild with -DENABLE_ZLIB=ON): " + filepath
        );
#endif
    }

    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("Failed to open output NIfTI file: " + filepath);
    }
    ofs.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
    if (!ofs) {
        throw std::runtime_error("Failed writing NIfTI bytes: " + filepath);
    }
}

double spatialUnitScaleToMm(int32_t xyzt_units) {
    const int space_unit = xyzt_units & 0x07;
    if (space_unit == 1) return 1000.0; // meters -> mm
    if (space_unit == 3) return 1e-3;   // micrometers -> mm
    return 1.0;                         // default: millimeters
}

std::array<double, 16> extractAffineMm(const NormalizedNiftiMeta& m, double unit_scale) {
    std::array<double, 16> aff = {
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    };

    if (m.sform_code > 0) {
        for (int c = 0; c < 4; ++c) {
            aff[0 + c] = m.srow_x[c] * unit_scale;
            aff[4 + c] = m.srow_y[c] * unit_scale;
            aff[8 + c] = m.srow_z[c] * unit_scale;
        }
        return aff;
    }

    const double dx = std::abs(m.pixdim[1]) * unit_scale;
    const double dy = std::abs(m.pixdim[2]) * unit_scale;
    const double dz = std::abs(m.pixdim[3]) * unit_scale;

    if (m.qform_code > 0) {
        const double b = m.quatern_b;
        const double c = m.quatern_c;
        const double d = m.quatern_d;
        const double a = std::sqrt(std::max(0.0, 1.0 - (b * b + c * c + d * d)));
        const double qfac = (m.pixdim[0] < 0.0) ? -1.0 : 1.0;

        const double r00 = a * a + b * b - c * c - d * d;
        const double r01 = 2.0 * (b * c - a * d);
        const double r02 = 2.0 * (b * d + a * c);

        const double r10 = 2.0 * (b * c + a * d);
        const double r11 = a * a + c * c - b * b - d * d;
        const double r12 = 2.0 * (c * d - a * b);

        const double r20 = 2.0 * (b * d - a * c);
        const double r21 = 2.0 * (c * d + a * b);
        const double r22 = a * a + d * d - b * b - c * c;

        aff[0] = r00 * dx; aff[1] = r01 * dy; aff[2]  = r02 * dz * qfac; aff[3]  = m.qoffset_x * unit_scale;
        aff[4] = r10 * dx; aff[5] = r11 * dy; aff[6]  = r12 * dz * qfac; aff[7]  = m.qoffset_y * unit_scale;
        aff[8] = r20 * dx; aff[9] = r21 * dy; aff[10] = r22 * dz * qfac; aff[11] = m.qoffset_z * unit_scale;
        return aff;
    }

    aff[0]  = (dx > 0.0) ? dx : 1.0;
    aff[5]  = (dy > 0.0) ? dy : 1.0;
    aff[10] = (dz > 0.0) ? dz : 1.0;
    return aff;
}

template <typename RawT>
void decodeVoxels(
    const uint8_t* src,
    size_t count,
    bool swapped,
    double slope,
    double inter,
    std::vector<float>& out
) {
    out.resize(count);
    for (size_t i = 0; i < count; ++i) {
        RawT val{};
        std::memcpy(&val, src + i * sizeof(RawT), sizeof(RawT));
        if constexpr (sizeof(RawT) > 1) {
            if (swapped) val = byteSwap(val);
        }
        out[i] = static_cast<float>(static_cast<double>(val) * slope + inter);
    }
}

void parseNiftiMetaAndPayload(
    const std::string& filepath,
    NormalizedNiftiMeta& meta,
    std::vector<uint8_t>& payload_bytes,
    size_t& payload_offset
) {
    std::vector<uint8_t> bytes = readFileBytes(filepath);
    if (bytes.size() < 4) {
        throw std::runtime_error("File too small to be a NIfTI file: " + filepath);
    }

    int32_t sizeof_hdr_raw = 0;
    std::memcpy(&sizeof_hdr_raw, bytes.data(), sizeof(int32_t));

    constexpr int32_t kSwapped540 = 469893120;  // byteSwap<int32_t>(540) -> NIfTI-2
    constexpr int32_t kSwapped348 = 1543569408; // byteSwap<int32_t>(348) -> NIfTI-1

    if (sizeof_hdr_raw == 540 || sizeof_hdr_raw == kSwapped540) {
        // Primary path: NIfTI-2
        if (bytes.size() < sizeof(Nifti2Header)) {
            throw std::runtime_error("File smaller than 540-byte NIfTI-2 header: " + filepath);
        }
        Nifti2Header h{};
        std::memcpy(&h, bytes.data(), sizeof(Nifti2Header));
        meta.version = 2;
        meta.swapped = (sizeof_hdr_raw == kSwapped540);
        if (meta.swapped) swapNifti2Endianness(h);

        meta.datatype = h.datatype;
        meta.bitpix = h.bitpix;
        for (int i = 0; i < 8; ++i) {
            meta.dim[i] = h.dim[i];
            meta.pixdim[i] = h.pixdim[i];
        }
        meta.vox_offset = h.vox_offset;
        meta.scl_slope = h.scl_slope;
        meta.scl_inter = h.scl_inter;
        meta.qform_code = h.qform_code;
        meta.sform_code = h.sform_code;
        meta.quatern_b = h.quatern_b;
        meta.quatern_c = h.quatern_c;
        meta.quatern_d = h.quatern_d;
        meta.qoffset_x = h.qoffset_x;
        meta.qoffset_y = h.qoffset_y;
        meta.qoffset_z = h.qoffset_z;
        for (int c = 0; c < 4; ++c) {
            meta.srow_x[c] = h.srow_x[c];
            meta.srow_y[c] = h.srow_y[c];
            meta.srow_z[c] = h.srow_z[c];
        }
        meta.xyzt_units = h.xyzt_units;
        std::memcpy(meta.magic, h.magic, 8);

    } else if (sizeof_hdr_raw == 348 || sizeof_hdr_raw == kSwapped348) {
        // Fallback path: NIfTI-1
        if (bytes.size() < sizeof(Nifti1Header)) {
            throw std::runtime_error("File smaller than 348-byte NIfTI-1 header: " + filepath);
        }
        Nifti1Header h{};
        std::memcpy(&h, bytes.data(), sizeof(Nifti1Header));
        meta.version = 1;
        meta.swapped = (sizeof_hdr_raw == kSwapped348);
        if (meta.swapped) swapNifti1Endianness(h);

        meta.datatype = h.datatype;
        meta.bitpix = h.bitpix;
        for (int i = 0; i < 8; ++i) {
            meta.dim[i] = h.dim[i];
            meta.pixdim[i] = static_cast<double>(h.pixdim[i]);
        }
        meta.vox_offset = static_cast<int64_t>(h.vox_offset);
        meta.scl_slope = static_cast<double>(h.scl_slope);
        meta.scl_inter = static_cast<double>(h.scl_inter);
        meta.qform_code = h.qform_code;
        meta.sform_code = h.sform_code;
        meta.quatern_b = static_cast<double>(h.quatern_b);
        meta.quatern_c = static_cast<double>(h.quatern_c);
        meta.quatern_d = static_cast<double>(h.quatern_d);
        meta.qoffset_x = static_cast<double>(h.qoffset_x);
        meta.qoffset_y = static_cast<double>(h.qoffset_y);
        meta.qoffset_z = static_cast<double>(h.qoffset_z);
        for (int c = 0; c < 4; ++c) {
            meta.srow_x[c] = static_cast<double>(h.srow_x[c]);
            meta.srow_y[c] = static_cast<double>(h.srow_y[c]);
            meta.srow_z[c] = static_cast<double>(h.srow_z[c]);
        }
        meta.xyzt_units = static_cast<uint8_t>(h.xyzt_units);
        std::memcpy(meta.magic, h.magic, 4);

    } else {
        throw std::runtime_error(
            "Unrecognized NIfTI sizeof_hdr (expected 540 for NIfTI-2 or 348 for NIfTI-1): " + filepath
        );
    }

    if (meta.dim[0] < 3 || meta.dim[1] <= 0 || meta.dim[2] <= 0 || meta.dim[3] <= 0) {
        throw std::runtime_error("NIfTI volume must have at least 3 valid spatial dimensions");
    }

    const bool is_two_file =
        (std::strncmp(meta.magic, "ni2", 3) == 0) ||
        (std::strncmp(meta.magic, "ni1", 3) == 0);

    if (is_two_file) {
        const std::string img_path = resolveSiblingImgPath(filepath);
        payload_bytes = readFileBytes(img_path);
        payload_offset = static_cast<size_t>(std::max<int64_t>(0, meta.vox_offset));
    } else {
        const int64_t min_offset = (meta.version == 2) ? 544 : 352;
        payload_bytes = std::move(bytes);
        payload_offset = static_cast<size_t>(std::max<int64_t>(min_offset, meta.vox_offset));
    }
}

void writeNiftiBuffer(
    const std::string& filepath,
    const uint8_t* payload,
    size_t payload_bytes,
    int16_t datatype,
    int16_t bitpix,
    int nx, int ny, int nz,
    const std::array<double, 3>& spacing_mm,
    const std::array<double, 16>& affine_mm,
    NiftiVersion version
) {
    constexpr int kMaxInt16 = std::numeric_limits<int16_t>::max();
    if (version == NiftiVersion::Nifti1 &&
        (nx > kMaxInt16 || ny > kMaxInt16 || nz > kMaxInt16)) {
        throw std::invalid_argument(
            "Grid dimensions exceed int16_t limit (32767) required by NIfTI-1 fallback"
        );
    }

    if (version == NiftiVersion::Nifti2) {
        // Default path: NIfTI-2
        Nifti2Header hdr{};
        hdr.sizeof_hdr = 540;
        constexpr char kNifti2Magic[8] = {'n', '+', '2', '\0', '\r', '\n', '\032', '\n'};
        std::memcpy(hdr.magic, kNifti2Magic, 8);
        hdr.datatype = datatype;
        hdr.bitpix = bitpix;
        hdr.dim[0] = 3;
        hdr.dim[1] = nx;
        hdr.dim[2] = ny;
        hdr.dim[3] = nz;
        hdr.pixdim[0] = 1.0;
        hdr.pixdim[1] = spacing_mm[0];
        hdr.pixdim[2] = spacing_mm[1];
        hdr.pixdim[3] = spacing_mm[2];
        hdr.vox_offset = 544;
        hdr.scl_slope = 1.0;
        hdr.scl_inter = 0.0;
        hdr.xyzt_units = 2; // NIFTI_UNITS_MM
        hdr.sform_code = 1; // NIFTI_XFORM_SCANNER_ANAT
        for (int c = 0; c < 4; ++c) {
            hdr.srow_x[c] = affine_mm[0 + c];
            hdr.srow_y[c] = affine_mm[4 + c];
            hdr.srow_z[c] = affine_mm[8 + c];
        }

        std::vector<uint8_t> out(544 + payload_bytes, 0);
        std::memcpy(out.data(), &hdr, sizeof(Nifti2Header));
        std::memcpy(out.data() + 544, payload, payload_bytes);
        writeFileBytes(filepath, out.data(), out.size());
    } else {
        // Fallback path: NIfTI-1
        Nifti1Header hdr{};
        hdr.sizeof_hdr = 348;
        hdr.dim[0] = 3;
        hdr.dim[1] = static_cast<int16_t>(nx);
        hdr.dim[2] = static_cast<int16_t>(ny);
        hdr.dim[3] = static_cast<int16_t>(nz);
        hdr.datatype = datatype;
        hdr.bitpix = bitpix;
        hdr.pixdim[0] = 1.0f;
        hdr.pixdim[1] = static_cast<float>(spacing_mm[0]);
        hdr.pixdim[2] = static_cast<float>(spacing_mm[1]);
        hdr.pixdim[3] = static_cast<float>(spacing_mm[2]);
        hdr.vox_offset = 352.0f;
        hdr.scl_slope = 1.0f;
        hdr.scl_inter = 0.0f;
        hdr.xyzt_units = 2; // NIFTI_UNITS_MM
        hdr.sform_code = 1; // NIFTI_XFORM_SCANNER_ANAT
        for (int c = 0; c < 4; ++c) {
            hdr.srow_x[c] = static_cast<float>(affine_mm[0 + c]);
            hdr.srow_y[c] = static_cast<float>(affine_mm[4 + c]);
            hdr.srow_z[c] = static_cast<float>(affine_mm[8 + c]);
        }
        std::memcpy(hdr.magic, "n+1\0", 4);

        std::vector<uint8_t> out(352 + payload_bytes, 0);
        std::memcpy(out.data(), &hdr, sizeof(Nifti1Header));
        std::memcpy(out.data() + 352, payload, payload_bytes);
        writeFileBytes(filepath, out.data(), out.size());
    }
}

} // namespace

MriVolume NiftiLoader::loadMri(const std::string& filepath) {
    NormalizedNiftiMeta meta{};
    std::vector<uint8_t> bytes;
    size_t offset = 0;
    parseNiftiMetaAndPayload(filepath, meta, bytes, offset);

    if (meta.dim[1] > std::numeric_limits<int>::max() ||
        meta.dim[2] > std::numeric_limits<int>::max() ||
        meta.dim[3] > std::numeric_limits<int>::max()) {
        throw std::runtime_error("NIfTI dimension exceeds 32-bit int index limit");
    }

    const int nx = static_cast<int>(meta.dim[1]);
    const int ny = static_cast<int>(meta.dim[2]);
    const int nz = static_cast<int>(meta.dim[3]);
    const size_t voxel_count = static_cast<size_t>(nx) * ny * nz;
    const size_t bytes_per_voxel = static_cast<size_t>(meta.bitpix / 8);

    if (bytes_per_voxel == 0 || offset + voxel_count * bytes_per_voxel > bytes.size()) {
        throw std::runtime_error("Truncated or invalid NIfTI voxel payload in: " + filepath);
    }

    const double slope = (meta.scl_slope != 0.0) ? meta.scl_slope : 1.0;
    const double inter = (meta.scl_slope != 0.0) ? meta.scl_inter : 0.0;
    const uint8_t* payload = bytes.data() + offset;

    MriVolume vol;
    vol.nx = nx;
    vol.ny = ny;
    vol.nz = nz;

    const double unit_scale = spatialUnitScaleToMm(meta.xyzt_units);
    vol.spacing_mm = {
        std::abs(meta.pixdim[1]) * unit_scale,
        std::abs(meta.pixdim[2]) * unit_scale,
        std::abs(meta.pixdim[3]) * unit_scale
    };
    vol.affine = extractAffineMm(meta, unit_scale);

    switch (meta.datatype) {
        case 2:    decodeVoxels<uint8_t>(payload, voxel_count, meta.swapped, slope, inter, vol.intensities); break;
        case 4:    decodeVoxels<int16_t>(payload, voxel_count, meta.swapped, slope, inter, vol.intensities); break;
        case 8:    decodeVoxels<int32_t>(payload, voxel_count, meta.swapped, slope, inter, vol.intensities); break;
        case 16:   decodeVoxels<float>(payload, voxel_count, meta.swapped, slope, inter, vol.intensities); break;
        case 64:   decodeVoxels<double>(payload, voxel_count, meta.swapped, slope, inter, vol.intensities); break;
        case 256:  decodeVoxels<int8_t>(payload, voxel_count, meta.swapped, slope, inter, vol.intensities); break;
        case 512:  decodeVoxels<uint16_t>(payload, voxel_count, meta.swapped, slope, inter, vol.intensities); break;
        case 768:  decodeVoxels<uint32_t>(payload, voxel_count, meta.swapped, slope, inter, vol.intensities); break;
        case 1024: decodeVoxels<int64_t>(payload, voxel_count, meta.swapped, slope, inter, vol.intensities); break;
        case 1280: decodeVoxels<uint64_t>(payload, voxel_count, meta.swapped, slope, inter, vol.intensities); break;
        default:
            throw std::runtime_error("Unsupported NIfTI datatype code: " + std::to_string(meta.datatype));
    }

    return vol;
}

SegmentationResult NiftiLoader::loadLabels(const std::string& filepath) {
    MriVolume tmp = loadMri(filepath);
    SegmentationResult res;
    res.nx = tmp.nx;
    res.ny = tmp.ny;
    res.nz = tmp.nz;
    res.spacing_mm = tmp.spacing_mm;
    res.labels.resize(tmp.intensities.size());

    for (size_t i = 0; i < tmp.intensities.size(); ++i) {
        const int rounded = static_cast<int>(std::lround(tmp.intensities[i]));
        res.labels[i] = static_cast<uint8_t>(std::clamp(rounded, 0, 255));
    }
    return res;
}

void NiftiLoader::saveLabels(
    const std::string& filepath,
    const std::vector<uint8_t>& labels,
    int nx, int ny, int nz,
    const std::array<double, 3>& spacing_mm,
    const std::array<double, 16>& affine_mm,
    NiftiVersion version
) {
    const size_t n = static_cast<size_t>(nx) * ny * nz;
    if (labels.size() != n) {
        throw std::invalid_argument("Label count does not match dimensions in saveLabels");
    }

    writeNiftiBuffer(
        filepath,
        labels.data(),
        n,
        /*datatype=*/2, // DT_UINT8
        /*bitpix=*/8,
        nx, ny, nz,
        spacing_mm,
        affine_mm,
        version
    );
}

void NiftiLoader::saveScalarGrid(
    const std::string& filepath,
    const Grid3D& grid,
    const std::array<double, 16>& affine_mm,
    NiftiVersion version
) {
    const size_t n = grid.size();
    const double h_mm = grid.h() * 1000.0;

    std::vector<float> float_payload(n);
    const double* src = grid.raw();
    for (size_t i = 0; i < n; ++i) {
        float_payload[i] = static_cast<float>(src[i]);
    }

    writeNiftiBuffer(
        filepath,
        reinterpret_cast<const uint8_t*>(float_payload.data()),
        n * sizeof(float),
        /*datatype=*/16, // DT_FLOAT32
        /*bitpix=*/32,
        grid.nx(), grid.ny(), grid.nz(),
        {h_mm, h_mm, h_mm},
        affine_mm,
        version
    );
}