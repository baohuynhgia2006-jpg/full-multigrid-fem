#include "grid/MultigridHierarchy.h"
#include "io/NiftiLoader.h"
#include "io/VoxelGridResampler.h"
#include "segmentation/ITissueSegmenter.h"
#include "segmentation/SegmentationTypes.h"
#include "solver/CoilModel.h"
#include "solver/ConductivityBuilder.h"
#include "solver/FemStencil.h"
#include "solver/FieldPostProcessor.h"
#include "solver/MultigridSolver.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct CliOptions {
    std::string input_mri_path;
    std::string input_labels_path;
    std::string output_dir = "output";
    NiftiVersion output_version = NiftiVersion::Nifti1;
    bool save_e_components = false;

    // Segmentation
    SegmenterBackend seg_backend = SegmenterBackend::GmmSpatial;
    std::string seg_model_path;

    // Grid & Multigrid Solver
    double target_h_mm = 1.0;
    int num_levels = 0; // 0 = auto (4 for >=1mm, 5 for 0.5mm, 6 for <=0.25mm)
    double rel_tol = 1e-5;
    int max_iter = 80;
    int num_threads = 0;
    int warm_start_mode = -1; // -1 = auto (enabled when res < 1.8 mm), 0 = off, 1 = on

    // Coil Model
    std::string coil_spec = "figure8"; // "figure8", "magstim70", "circular", "dipole", or path to .ccd
    double coil_radius_mm = -1.0;      // < 0 uses preset default
    double dI_dt = 1.0e8;              // A/s (100 A/us)

    // Coil Placement Mode A: Explicit world position (mm) & Euler angles (deg)
    bool has_explicit_pos = false;
    std::array<double, 3> coil_pos_mm{0.0, 0.0, 0.0};
    std::array<double, 3> coil_rot_deg{0.0, 0.0, 0.0};

    // Coil Placement Mode B: Auto-snap to scalp near target point (mm)
    bool has_scalp_target = false;
    std::array<double, 3> scalp_target_mm{0.0, 0.0, 0.0};
    double scalp_offset_mm = 5.0;
    double handle_yaw_deg = 45.0;
};

void printUsage(const char* prog) {
    std::cout
        << "TMS Fast Multigrid FEM Solver CLI\n\n"
        << "Usage:\n"
        << "  " << prog << " --input <t1.nii[.gz]> [options]\n"
        << "  " << prog << " --labels <seg.nii[.gz]> [options]\n\n"
        << "Input / Output:\n"
        << "  -i, --input <path>          Input T1-weighted MRI (.nii or .nii.gz)\n"
        << "      --labels <path>         Skip segmentation and load pre-segmented tissue labels\n"
        << "  -o, --output-dir <dir>      Output directory (default: ./output)\n"
        << "      --save-components       Also save Ex, Ey, Ez vector field NIfTI volumes\n"
        << "      --nifti1                Save outputs in NIfTI-1 (default, 3D Slicer compatible)\n"
        << "      --nifti2                Save outputs in NIfTI-2 (64-bit header)\n\n"
        << "Segmentation & Solver:\n"
        << "      --seg-backend <type>    'gmm' (default) or 'onnx'\n"
        << "      --seg-model <path>      Optional atlas path (gmm) or .onnx model path (onnx)\n"
        << "      --res <mm>              Isotropic FEM grid resolution in mm (default: 1.0)\n"
        << "      --levels <int>          Multigrid V-cycle levels (default: auto)\n"
        << "      --tol <float>           Relative residual stopping tolerance (default: 1e-5)\n"
        << "      --max-iter <int>        Maximum MGCG iterations (default: 80)\n"
        << "      --threads <int>         OpenMP thread count (default: 0 = auto)\n"
        << "      --warm-start            Force 2.0 mm coarse-to-fine initial guess\n"
        << "      --no-warm-start         Disable 2.0 mm coarse-to-fine initial guess\n\n"
        << "Coil Selection:\n"
        << "      --coil <name|file.ccd>  'figure8' (default), 'magstim70', 'circular', 'dipole',\n"
        << "                              or path to a .ccd magnetic dipole file\n"
        << "      --coil-radius <mm>      Override loop radius in mm for figure8/circular coils\n"
        << "      --didt <A/s>            Coil current rate of change dI/dt (default: 1e8)\n\n"
        << "Coil Positioning (choose one mode):\n"
        << "      --coil-pos <x,y,z>      Explicit coil center in scanner mm (e.g. -30,0,95)\n"
        << "      --coil-rot <rx,ry,rz>   Explicit Euler angles in degrees (default: 0,0,0)\n"
        << "      --scalp-target <x,y,z>  Target point in scanner mm; snaps coil onto nearest\n"
        << "                              scalp surface along outward normal\n"
        << "      --scalp-offset <mm>     Stand-off distance above scalp (default: 5.0)\n"
        << "      --handle-yaw <deg>      In-plane coil handle rotation in degrees (default: 45)\n";
}

std::array<double, 3> parseVec3(const std::string& str, const std::string& flag_name) {
    std::array<double, 3> out{};
    std::string cleaned = str;
    std::replace(cleaned.begin(), cleaned.end(), ',', ' ');
    std::istringstream iss(cleaned);
    if (!(iss >> out[0] >> out[1] >> out[2])) {
        throw std::invalid_argument("Invalid 3D vector for " + flag_name + ": '" + str + "'");
    }
    return out;
}

CliOptions parseArgs(int argc, char* argv[]) {
    CliOptions opts;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto nextArg = [&](const std::string& flag) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument("Missing value after " + flag);
            }
            return argv[++i];
        };

        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            std::exit(0);
        } else if (arg == "-i" || arg == "--input") {
            opts.input_mri_path = nextArg(arg);
        } else if (arg == "--labels") {
            opts.input_labels_path = nextArg(arg);
        } else if (arg == "-o" || arg == "--output-dir") {
            opts.output_dir = nextArg(arg);
        } else if (arg == "--save-components") {
            opts.save_e_components = true;
        } else if (arg == "--nifti1") {
            opts.output_version = NiftiVersion::Nifti1;
        } else if (arg == "--nifti2") {
            opts.output_version = NiftiVersion::Nifti2;
        } else if (arg == "--seg-backend") {
            const std::string b = nextArg(arg);
            if (b == "gmm") opts.seg_backend = SegmenterBackend::GmmSpatial;
            else if (b == "onnx") opts.seg_backend = SegmenterBackend::OnnxRuntime;
            else throw std::invalid_argument("Unknown --seg-backend: " + b);
        } else if (arg == "--seg-model") {
            opts.seg_model_path = nextArg(arg);
        } else if (arg == "--res") {
            opts.target_h_mm = std::stod(nextArg(arg));
        } else if (arg == "--levels") {
            opts.num_levels = std::stoi(nextArg(arg));
        } else if (arg == "--tol") {
            opts.rel_tol = std::stod(nextArg(arg));
        } else if (arg == "--max-iter") {
            opts.max_iter = std::stoi(nextArg(arg));
        } else if (arg == "--threads") {
            opts.num_threads = std::stoi(nextArg(arg));
        } else if (arg == "--warm-start") {
            opts.warm_start_mode = 1;
        } else if (arg == "--no-warm-start") {
            opts.warm_start_mode = 0;
        } else if (arg == "--coil") {
            opts.coil_spec = nextArg(arg);
        } else if (arg == "--coil-radius") {
            opts.coil_radius_mm = std::stod(nextArg(arg));
        } else if (arg == "--didt") {
            opts.dI_dt = std::stod(nextArg(arg));
        } else if (arg == "--coil-pos") {
            opts.has_explicit_pos = true;
            opts.coil_pos_mm = parseVec3(nextArg(arg), arg);
        } else if (arg == "--coil-rot") {
            opts.coil_rot_deg = parseVec3(nextArg(arg), arg);
        } else if (arg == "--scalp-target") {
            opts.has_scalp_target = true;
            opts.scalp_target_mm = parseVec3(nextArg(arg), arg);
        } else if (arg == "--scalp-offset") {
            opts.scalp_offset_mm = std::stod(nextArg(arg));
        } else if (arg == "--handle-yaw") {
            opts.handle_yaw_deg = std::stod(nextArg(arg));
        } else {
            throw std::invalid_argument("Unknown CLI option: " + arg);
        }
    }

    if (opts.input_mri_path.empty() && opts.input_labels_path.empty()) {
        throw std::invalid_argument("Must specify either --input <t1.nii> or --labels <seg.nii>");
    }
    return opts;
}

CoilModel buildSelectedCoil(const CliOptions& opts) {
    if (opts.coil_spec == "figure8") {
        const double r_m = (opts.coil_radius_mm > 0.0) ? (opts.coil_radius_mm * 1e-3) : 0.025;
        return CoilModel::createFigure8Coil(r_m, 314);
    }
    if (opts.coil_spec == "magstim70") {
        return CoilModel::createMagstim70Coil();
    }
    if (opts.coil_spec == "circular") {
        const double r_m = (opts.coil_radius_mm > 0.0) ? (opts.coil_radius_mm * 1e-3) : 0.045;
        return CoilModel::createCircularCoil(r_m, 314);
    }
    if (opts.coil_spec == "dipole") {
        return CoilModel::createSingleDipole({0.0, 0.0, 1.0});
    }
    return CoilModel::loadFromCcdFile(opts.coil_spec);
}

// Finds the nearest non-air surface voxel to target_mm, estimates the outward
// surface normal from the local air/tissue gradient, and computes coil pose.
void computeScalpSurfacePose(
    const SolverGridVolume& grid,
    const std::array<double, 3>& target_mm,
    double offset_mm,
    double yaw_deg,
    std::array<double, 3>& out_coil_pos_m,
    std::array<double, 9>& out_coil_rot
) {
    const int nx = grid.nx;
    const int ny = grid.ny;
    const int nz = grid.nz;
    const double h_mm = grid.h_m * 1000.0;

    auto isHead = [&](int i, int j, int k) -> bool {
        if (i < 0 || i >= nx || j < 0 || j >= ny || k < 0 || k >= nz) return false;
        const size_t idx = static_cast<size_t>(i) + nx * (static_cast<size_t>(j) + ny * k);
        return grid.labels[idx] != static_cast<uint8_t>(TissueClass::Air);
    };

    double best_d2 = 1e30;
    int best_i = nx / 2, best_j = ny / 2, best_k = nz / 2;
    bool found_surface = false;

    for (int k = 1; k < nz - 1; ++k) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                if (!isHead(i, j, k)) continue;

                const bool on_surface =
                    !isHead(i - 1, j, k) || !isHead(i + 1, j, k) ||
                    !isHead(i, j - 1, k) || !isHead(i, j + 1, k) ||
                    !isHead(i, j, k - 1) || !isHead(i, j, k + 1);
                if (!on_surface) continue;

                const double wx = grid.affine_mm[0] * i + grid.affine_mm[1] * j + grid.affine_mm[2] * k + grid.affine_mm[3];
                const double wy = grid.affine_mm[4] * i + grid.affine_mm[5] * j + grid.affine_mm[6] * k + grid.affine_mm[7];
                const double wz = grid.affine_mm[8] * i + grid.affine_mm[9] * j + grid.affine_mm[10] * k + grid.affine_mm[11];

                const double dx = wx - target_mm[0];
                const double dy = wy - target_mm[1];
                const double dz = wz - target_mm[2];
                const double d2 = dx * dx + dy * dy + dz * dz;

                if (d2 < best_d2) {
                    best_d2 = d2;
                    best_i = i;
                    best_j = j;
                    best_k = k;
                    found_surface = true;
                }
            }
        }
    }

    if (!found_surface) {
        throw std::runtime_error("No non-air head surface voxels found for scalp placement");
    }

    std::array<double, 3> n_vox{0.0, 0.0, 0.0};
    for (int dz = -2; dz <= 2; ++dz) {
        for (int dy = -2; dy <= 2; ++dy) {
            for (int dx = -2; dx <= 2; ++dx) {
                if (!isHead(best_i + dx, best_j + dy, best_k + dz)) {
                    n_vox[0] += dx;
                    n_vox[1] += dy;
                    n_vox[2] += dz;
                }
            }
        }
    }

    std::array<double, 3> n_world = {
        grid.affine_mm[0] * n_vox[0] + grid.affine_mm[1] * n_vox[1] + grid.affine_mm[2] * n_vox[2],
        grid.affine_mm[4] * n_vox[0] + grid.affine_mm[5] * n_vox[1] + grid.affine_mm[6] * n_vox[2],
        grid.affine_mm[8] * n_vox[0] + grid.affine_mm[9] * n_vox[1] + grid.affine_mm[10] * n_vox[2]
    };

    const double n_len = std::sqrt(
        n_world[0] * n_world[0] + n_world[1] * n_world[1] + n_world[2] * n_world[2]
    );
    if (n_len < 1e-9) {
        n_world = {0.0, 0.0, 1.0};
    } else {
        n_world[0] /= n_len;
        n_world[1] /= n_len;
        n_world[2] /= n_len;
    }

    const double surf_x = grid.affine_mm[0] * best_i + grid.affine_mm[1] * best_j + grid.affine_mm[2] * best_k + grid.affine_mm[3];
    const double surf_y = grid.affine_mm[4] * best_i + grid.affine_mm[5] * best_j + grid.affine_mm[6] * best_k + grid.affine_mm[7];
    const double surf_z = grid.affine_mm[8] * best_i + grid.affine_mm[9] * best_j + grid.affine_mm[10] * best_k + grid.affine_mm[11];

    const double total_shift_mm = 0.5 * h_mm + offset_mm;
    out_coil_pos_m = {
        (surf_x + total_shift_mm * n_world[0]) * 1e-3,
        (surf_y + total_shift_mm * n_world[1]) * 1e-3,
        (surf_z + total_shift_mm * n_world[2]) * 1e-3
    };

    out_coil_rot = CoilModel::rotationFromNormalAndYaw(n_world, yaw_deg);
}

void printCorticalDosimetrySummary(
    const Vec3Grid& E,
    const std::vector<uint8_t>& labels,
    double h_m
) {
    const double voxel_vol_cm3 = (h_m * h_m * h_m) * 1e6;
    std::vector<double> gm_mags;
    gm_mags.reserve(labels.size() / 8);

    const double* ex = E.x.raw();
    const double* ey = E.y.raw();
    const double* ez = E.z.raw();

    for (size_t idx = 0; idx < labels.size(); ++idx) {
        if (labels[idx] == static_cast<uint8_t>(TissueClass::GrayMatter)) {
            const double mag = std::sqrt(ex[idx] * ex[idx] + ey[idx] * ey[idx] + ez[idx] * ez[idx]);
            gm_mags.push_back(mag);
        }
    }

    if (gm_mags.empty()) {
        std::cout << "  [Warning] No GrayMatter voxels found for cortical dosimetry summary.\n";
        return;
    }

    std::sort(gm_mags.begin(), gm_mags.end(), std::greater<double>());
    const size_t idx_99 = std::min(gm_mags.size() - 1, static_cast<size_t>(0.01 * gm_mags.size()));

    std::cout << "\n--- Cortical Gray Matter Dosimetry ---\n"
              << "  GM Volume:  " << (gm_mags.size() * voxel_vol_cm3) << " cm^3\n"
              << "  E_max:      " << gm_mags.front() << " V/m\n"
              << "  E_99:       " << gm_mags[idx_99] << " V/m\n";

    for (double target_cm3 : {0.04, 0.2, 1.0, 5.0}) {
        const size_t vox_idx = static_cast<size_t>(target_cm3 / voxel_vol_cm3);
        if (vox_idx < gm_mags.size()) {
            std::cout << "  E_thresh (" << target_cm3 << " cm^3): " << gm_mags[vox_idx] << " V/m\n";
        }
    }
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        const CliOptions opts = parseArgs(argc, argv);
        fs::create_directories(opts.output_dir);

        const int num_levels = (opts.num_levels > 0)
            ? opts.num_levels
            : ((opts.target_h_mm >= 1.0) ? 4 : ((opts.target_h_mm >= 0.5) ? 5 : 6));

        // 1. Load MRI or Pre-Segmented Tissue Labels
        SegmentationResult seg;
        std::array<double, 16> src_affine_mm{};

        if (!opts.input_labels_path.empty()) {
            std::cout << "[1/6] Loading pre-segmented labels: " << opts.input_labels_path << "\n";
            MriVolume raw_lbl = NiftiLoader::loadMri(opts.input_labels_path);
            src_affine_mm = raw_lbl.affine;
            seg = NiftiLoader::loadLabels(opts.input_labels_path);
        } else {
            std::cout << "[1/6] Loading T1-weighted MRI: " << opts.input_mri_path << "\n";
            MriVolume mri = NiftiLoader::loadMri(opts.input_mri_path);
            src_affine_mm = mri.affine;

            const auto t_seg0 = std::chrono::steady_clock::now();
            auto segmenter = createSegmenter(opts.seg_backend, opts.seg_model_path);
            std::cout << "[2/6] Segmenting tissues using " << segmenter->name() << "...\n";
            seg = segmenter->segment(mri);
            const auto t_seg1 = std::chrono::steady_clock::now();
            const double seg_s = std::chrono::duration<double>(t_seg1 - t_seg0).count();
            std::cout << "      Segmentation finished in " << seg_s << " s\n";
        }

        // 2. Resample onto Uniform Cubical FEM Grid & Pad for Multigrid
        std::cout << "[3/6] Resampling onto " << opts.target_h_mm << " mm isotropic grid ("
                  << num_levels << " multigrid levels)...\n";
        SolverGridVolume fem_vol = VoxelGridResampler::resampleLabelsForSolver(
            seg, src_affine_mm, opts.target_h_mm, num_levels, /*min_air_pad_voxels=*/2
        );
        std::cout << "      Padded grid dimensions: "
                  << fem_vol.nx << " x " << fem_vol.ny << " x " << fem_vol.nz << "\n";

        // 3. Build Multigrid Hierarchy & Smooth Conductivities
        MultigridHierarchy hierarchy(
            fem_vol.nx, fem_vol.ny, fem_vol.nz, fem_vol.h_m, num_levels
        );
        ConductivityBuilder cond_builder;
        cond_builder.assignFromLabels(fem_vol.labels, hierarchy.level(0).sigma);
        ConductivityBuilder::applyStaircaseSmoothing(hierarchy.level(0).sigma);
        hierarchy.buildCoarseConductivities();

        // 4. Configure Coil & Evaluate Primary Vector Potential A0'
        CoilModel coil = buildSelectedCoil(opts);
        std::array<double, 3> coil_pos_m{};
        std::array<double, 9> coil_rot{};

        if (opts.has_explicit_pos) {
            coil_pos_m = {
                opts.coil_pos_mm[0] * 1e-3,
                opts.coil_pos_mm[1] * 1e-3,
                opts.coil_pos_mm[2] * 1e-3
            };
            coil_rot = CoilModel::rotationFromEulerDeg(
                opts.coil_rot_deg[0], opts.coil_rot_deg[1], opts.coil_rot_deg[2]
            );
        } else if (opts.has_scalp_target) {
            computeScalpSurfacePose(
                fem_vol, opts.scalp_target_mm, opts.scalp_offset_mm, opts.handle_yaw_deg,
                coil_pos_m, coil_rot
            );
        } else {
            const std::array<double, 3> default_target_mm = {
                fem_vol.affine_mm[3] + fem_vol.affine_mm[0] * (0.5 * fem_vol.nx) - 30.0,
                fem_vol.affine_mm[7] + fem_vol.affine_mm[5] * (0.5 * fem_vol.ny),
                fem_vol.affine_mm[11] + fem_vol.affine_mm[10] * (0.85 * fem_vol.nz)
            };
            computeScalpSurfacePose(
                fem_vol, default_target_mm, opts.scalp_offset_mm, opts.handle_yaw_deg,
                coil_pos_m, coil_rot
            );
        }

        std::cout << "[4/6] Evaluating primary vector potential A0' ("
                  << coil.dipoles().size() << " dipoles) at coil center ["
                  << coil_pos_m[0] * 1000.0 << ", "
                  << coil_pos_m[1] * 1000.0 << ", "
                  << coil_pos_m[2] * 1000.0 << "] mm...\n";

        Vec3Grid A0_prime(fem_vol.nx, fem_vol.ny, fem_vol.nz, fem_vol.h_m);
        coil.evaluateA0Prime(A0_prime, opts.dI_dt, fem_vol.origin_m, coil_pos_m, coil_rot);

        // Assemble FEM RHS f^h
        GridLevel& finest = hierarchy.level(0);
#ifdef TMS_USE_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int k = 1; k < fem_vol.nz; ++k) {
            for (int j = 1; j < fem_vol.ny; ++j) {
                for (int i = 1; i < fem_vol.nx; ++i) {
                    finest.f(i, j, k) = FemStencil::assembleNodalSource(
                        finest.sigma, A0_prime, i, j, k
                    );
                }
            }
        }

        // 4b. Optional Coarse-to-Fine Warm-Start (2.0 mm -> fine grid)
        const bool use_warm_start =
            (opts.warm_start_mode == 1) ||
            (opts.warm_start_mode == -1 && opts.target_h_mm < 1.8);

        if (use_warm_start) {
            constexpr double kCoarseH_mm = 2.0;
            constexpr int kCoarseLevels = 4;
            std::cout << "[4b/6] Running 2.0 mm coarse warm-start solve...\n";

            SolverGridVolume coarse_vol = VoxelGridResampler::resampleLabelsForSolver(
                seg, src_affine_mm, kCoarseH_mm, kCoarseLevels, /*min_air_pad_voxels=*/2
            );

            MultigridHierarchy coarse_hierarchy(
                coarse_vol.nx, coarse_vol.ny, coarse_vol.nz, coarse_vol.h_m, kCoarseLevels
            );
            cond_builder.assignFromLabels(coarse_vol.labels, coarse_hierarchy.level(0).sigma);
            ConductivityBuilder::applyStaircaseSmoothing(coarse_hierarchy.level(0).sigma);
            coarse_hierarchy.buildCoarseConductivities();

            Vec3Grid coarse_A0(coarse_vol.nx, coarse_vol.ny, coarse_vol.nz, coarse_vol.h_m);
            coil.evaluateA0Prime(coarse_A0, opts.dI_dt, coarse_vol.origin_m, coil_pos_m, coil_rot);

            GridLevel& coarse_finest = coarse_hierarchy.level(0);
#ifdef TMS_USE_OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (int k = 1; k < coarse_vol.nz; ++k) {
                for (int j = 1; j < coarse_vol.ny; ++j) {
                    for (int i = 1; i < coarse_vol.nx; ++i) {
                        coarse_finest.f(i, j, k) = FemStencil::assembleNodalSource(
                            coarse_finest.sigma, coarse_A0, i, j, k
                        );
                    }
                }
            }

            SolverConfig coarse_cfg;
            coarse_cfg.rel_residual_tol = opts.rel_tol;
            coarse_cfg.max_vcycles = std::max(80, opts.max_iter);
            coarse_cfg.num_threads = opts.num_threads;
            coarse_cfg.verbose = false;

            const auto t_ws0 = std::chrono::steady_clock::now();
            MultigridSolver coarse_solver(coarse_cfg);
            SolverStats ws_stats = coarse_solver.solve(coarse_hierarchy);
            const auto t_ws1 = std::chrono::steady_clock::now();
            const double ws_ms = std::chrono::duration<double, std::milli>(t_ws1 - t_ws0).count();

            VoxelGridResampler::interpolateNodalField(
                coarse_finest.phi, coarse_vol.origin_m,
                finest.phi, fem_vol.origin_m
            );

            std::cout << "       Warm-start finished in " << ws_ms << " ms ("
                      << ws_stats.iterations << " iters, rel_res = "
                      << ws_stats.final_rel_residual << ")\n";
        }

        // 5. Solve Scalar Potential Equation via OpenMP MGCG
        std::cout << "[5/6] Solving FEM system with Multigrid-Preconditioned CG...\n";
        SolverConfig solver_cfg;
        solver_cfg.use_initial_guess = use_warm_start;
        solver_cfg.rel_residual_tol = opts.rel_tol;
        solver_cfg.max_vcycles = opts.max_iter;
        solver_cfg.num_threads = opts.num_threads;
        solver_cfg.verbose = true;

        MultigridSolver solver(solver_cfg);
        const auto t_sol0 = std::chrono::steady_clock::now();
        SolverStats stats = solver.solve(hierarchy);
        const auto t_sol1 = std::chrono::steady_clock::now();
        const double sol_ms = std::chrono::duration<double, std::milli>(t_sol1 - t_sol0).count();

        std::cout << "      Converged: " << (stats.converged ? "YES" : "NO")
                  << " | Iterations: " << stats.iterations
                  << " | Rel Res: " << stats.final_rel_residual
                  << " | Solve Time: " << sol_ms << " ms\n";

        // 6. Compute Electric Field E = -grad(phi) - A0' and Export NIfTI Volumes
        std::cout << "[6/6] Computing induced electric field E and saving outputs to "
                  << opts.output_dir << "...\n";
        Vec3Grid E = FieldPostProcessor::computeElectricField(
            finest.phi, finest.sigma, A0_prime
        );

        Grid3D E_mag(fem_vol.nx, fem_vol.ny, fem_vol.nz, fem_vol.h_m, 0.0);
        const size_t total_vox = E_mag.size();
        for (size_t idx = 0; idx < total_vox; ++idx) {
            const double ex = E.x.raw()[idx];
            const double ey = E.y.raw()[idx];
            const double ez = E.z.raw()[idx];
            E_mag.raw()[idx] = std::sqrt(ex * ex + ey * ey + ez * ez);
        }

        const fs::path out_dir(opts.output_dir);
#ifdef TMS_USE_ZLIB
        const std::string ext = ".nii.gz";
#else
        const std::string ext = ".nii";
#endif
        NiftiLoader::saveLabels(
            (out_dir / ("tissue_labels" + ext)).string(),
            fem_vol.labels, fem_vol.nx, fem_vol.ny, fem_vol.nz,
            {opts.target_h_mm, opts.target_h_mm, opts.target_h_mm},
            fem_vol.affine_mm, opts.output_version
        );
        NiftiLoader::saveScalarGrid(
            (out_dir / ("efield_mag" + ext)).string(),
            E_mag, fem_vol.affine_mm, opts.output_version
        );

        if (opts.save_e_components) {
            NiftiLoader::saveScalarGrid((out_dir / ("efield_x" + ext)).string(), E.x, fem_vol.affine_mm, opts.output_version);
            NiftiLoader::saveScalarGrid((out_dir / ("efield_y" + ext)).string(), E.y, fem_vol.affine_mm, opts.output_version);
            NiftiLoader::saveScalarGrid((out_dir / ("efield_z" + ext)).string(), E.z, fem_vol.affine_mm, opts.output_version);
        }

        printCorticalDosimetrySummary(E, fem_vol.labels, fem_vol.h_m);
        return stats.converged ? 0 : 1;

    } catch (const std::exception& ex) {
        std::cerr << "[Error] " << ex.what() << "\n";
        return 1;
    }
}