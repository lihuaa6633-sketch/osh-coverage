#include <pcl/ModelCoefficients.h>
#include <pcl/PointIndices.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/filter.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

using Point = pcl::PointXYZ;
using Cloud = pcl::PointCloud<Point>;
using CloudPtr = Cloud::Ptr;

struct Options {
    fs::path input_directory;
    fs::path output_directory = "sphere_result";
    std::optional<fs::path> poses_file;

    bool use_roi = false;
    Eigen::Vector3d roi_center = Eigen::Vector3d::Zero();
    double roi_radius = 0.15;

    double voxel_size = 0.004;
    double ransac_threshold = 0.008;
    double final_threshold = 0.004;
    double radius_min = 0.0625;
    double radius_max = 0.0825;
    double fixed_radius = 0.0725;
    double huber_delta = 0.003;

    int max_iterations = 10000;
    double probability = 0.999;
    std::size_t min_inliers = 30;
    std::size_t min_supporting_frames = 0;
    std::size_t min_points_per_frame = 4;
};

struct FrameData {
    std::string name;
    CloudPtr roi_cloud;
    std::size_t loaded_points = 0;
    std::size_t valid_points = 0;
};

struct SphereModel {
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    double radius = 0.0;
    std::vector<int> inliers;
};

struct ResidualStats {
    std::size_t count = 0;
    double rmse = 0.0;
    double mae = 0.0;
    double mean_signed = 0.0;
    double median_absolute = 0.0;
    double p95_absolute = 0.0;
    double max_absolute = 0.0;
};

struct Classification {
    CloudPtr inliers{new Cloud};
    CloudPtr outliers{new Cloud};
    std::vector<int> inlier_indices;
    std::vector<double> signed_residuals;
};

struct SubsetEstimate {
    bool valid = false;
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    std::size_t points = 0;
    std::size_t inliers = 0;
};

struct VoxelKey {
    std::int64_t x = 0;
    std::int64_t y = 0;
    std::int64_t z = 0;

    bool operator==(const VoxelKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct VoxelKeyHash {
    std::size_t operator()(const VoxelKey& key) const {
        std::size_t seed = 0;
        const auto combine = [&seed](std::int64_t value) {
            const std::size_t h = std::hash<std::int64_t>{}(value);
            seed ^= h + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U);
        };
        combine(key.x);
        combine(key.y);
        combine(key.z);
        return seed;
    }
};

struct VoxelBucket {
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> z;
};

[[noreturn]] void usageError(const std::string& message);

void printUsage(const char* program) {
    std::cout
        << "Usage:\n"
        << "  " << program << " <pcd_directory> [options]\n\n"
        << "Options:\n"
        << "  --roi cx cy cz radius       Spherical ROI in the common coordinate frame.\n"
        << "  --poses poses.txt           Optional per-frame poses for a moving sensor.\n"
        << "  --voxel meters              Median voxel size; 0 disables it (default 0.004).\n"
        << "  --ransac-threshold meters   Coarse RANSAC shell threshold (default 0.008).\n"
        << "  --final-threshold meters    Final inlier shell threshold (default 0.004).\n"
        << "  --radius-min meters         Coarse minimum radius (default 0.0625).\n"
        << "  --radius-max meters         Coarse maximum radius (default 0.0825).\n"
        << "  --fixed-radius meters       Known radius used for final fit (default 0.0725).\n"
        << "  --huber-delta meters        Huber robust-loss transition (default 0.003).\n"
        << "  --max-iterations n          RANSAC iteration cap (default 10000).\n"
        << "  --probability p             RANSAC probability (default 0.999).\n"
        << "  --min-inliers n             Minimum final median-voxel inliers (default 30).\n"
        << "  --min-frames n              Minimum supporting frames; 0 means 60% (default 0).\n"
        << "  --min-points-per-frame n    Inliers needed for a frame to support (default 4).\n"
        << "  --output directory          Output directory (default sphere_result).\n"
        << "  --help                      Show this help.\n\n"
        << "Pose file format, one line per PCD:\n"
        << "  filename.pcd tx ty tz qx qy qz qw\n"
        << "The transform convention is p_common = T_common_lidar * p_lidar.\n";
}

[[noreturn]] void usageError(const std::string& message) {
    throw std::runtime_error(message);
}

double parseDouble(const std::string& text, const std::string& option) {
    std::size_t parsed = 0;
    double value = 0.0;
    try {
        value = std::stod(text, &parsed);
    } catch (const std::exception&) {
        usageError("Invalid number for " + option + ": " + text);
    }
    if (parsed != text.size() || !std::isfinite(value)) {
        usageError("Invalid number for " + option + ": " + text);
    }
    return value;
}

std::size_t parseSize(const std::string& text, const std::string& option) {
    std::size_t parsed = 0;
    unsigned long long value = 0;
    try {
        value = std::stoull(text, &parsed);
    } catch (const std::exception&) {
        usageError("Invalid integer for " + option + ": " + text);
    }
    if (parsed != text.size()) {
        usageError("Invalid integer for " + option + ": " + text);
    }
    return static_cast<std::size_t>(value);
}

int parseInt(const std::string& text, const std::string& option) {
    const std::size_t value = parseSize(text, option);
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        usageError("Integer is too large for " + option + ": " + text);
    }
    return static_cast<int>(value);
}

void requireValues(int index, int needed, int argc, const std::string& option) {
    if (index + needed >= argc) {
        usageError("Missing value(s) after " + option);
    }
}

Options parseArguments(int argc, char** argv) {
    if (argc < 2) {
        printUsage(argv[0]);
        usageError("No PCD directory was supplied.");
    }

    if (std::string(argv[1]) == "--help") {
        printUsage(argv[0]);
        std::exit(0);
    }

    Options options;
    options.input_directory = argv[1];

    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") {
            printUsage(argv[0]);
            std::exit(0);
        } else if (arg == "--roi") {
            requireValues(i, 4, argc, arg);
            options.use_roi = true;
            options.roi_center.x() = parseDouble(argv[++i], arg);
            options.roi_center.y() = parseDouble(argv[++i], arg);
            options.roi_center.z() = parseDouble(argv[++i], arg);
            options.roi_radius = parseDouble(argv[++i], arg);
        } else if (arg == "--poses") {
            requireValues(i, 1, argc, arg);
            options.poses_file = fs::path(argv[++i]);
        } else if (arg == "--voxel") {
            requireValues(i, 1, argc, arg);
            options.voxel_size = parseDouble(argv[++i], arg);
        } else if (arg == "--ransac-threshold") {
            requireValues(i, 1, argc, arg);
            options.ransac_threshold = parseDouble(argv[++i], arg);
        } else if (arg == "--final-threshold") {
            requireValues(i, 1, argc, arg);
            options.final_threshold = parseDouble(argv[++i], arg);
        } else if (arg == "--radius-min") {
            requireValues(i, 1, argc, arg);
            options.radius_min = parseDouble(argv[++i], arg);
        } else if (arg == "--radius-max") {
            requireValues(i, 1, argc, arg);
            options.radius_max = parseDouble(argv[++i], arg);
        } else if (arg == "--fixed-radius") {
            requireValues(i, 1, argc, arg);
            options.fixed_radius = parseDouble(argv[++i], arg);
        } else if (arg == "--huber-delta") {
            requireValues(i, 1, argc, arg);
            options.huber_delta = parseDouble(argv[++i], arg);
        } else if (arg == "--max-iterations") {
            requireValues(i, 1, argc, arg);
            options.max_iterations = parseInt(argv[++i], arg);
        } else if (arg == "--probability") {
            requireValues(i, 1, argc, arg);
            options.probability = parseDouble(argv[++i], arg);
        } else if (arg == "--min-inliers") {
            requireValues(i, 1, argc, arg);
            options.min_inliers = parseSize(argv[++i], arg);
        } else if (arg == "--min-frames") {
            requireValues(i, 1, argc, arg);
            options.min_supporting_frames = parseSize(argv[++i], arg);
        } else if (arg == "--min-points-per-frame") {
            requireValues(i, 1, argc, arg);
            options.min_points_per_frame = parseSize(argv[++i], arg);
        } else if (arg == "--output") {
            requireValues(i, 1, argc, arg);
            options.output_directory = fs::path(argv[++i]);
        } else {
            usageError("Unknown option: " + arg);
        }
    }

    if (!fs::exists(options.input_directory) ||
        !fs::is_directory(options.input_directory)) {
        usageError("PCD directory does not exist: " +
                   options.input_directory.string());
    }
    if (options.use_roi && options.roi_radius <= 0.0) {
        usageError("--roi radius must be positive.");
    }
    if (options.voxel_size < 0.0) {
        usageError("--voxel must be zero or positive.");
    }
    if (options.ransac_threshold <= 0.0 ||
        options.final_threshold <= 0.0 ||
        options.huber_delta <= 0.0) {
        usageError("RANSAC, final, and Huber thresholds must be positive.");
    }
    if (options.radius_min <= 0.0 ||
        options.radius_max <= options.radius_min ||
        options.fixed_radius <= 0.0) {
        usageError("Sphere radius settings are invalid.");
    }
    if (options.max_iterations <= 0) {
        usageError("--max-iterations must be positive.");
    }
    if (options.probability <= 0.0 || options.probability >= 1.0) {
        usageError("--probability must be between 0 and 1.");
    }
    if (options.min_inliers < 4 ||
        options.min_points_per_frame < 1) {
        usageError("Minimum count settings are invalid.");
    }

    const fs::path input_absolute =
        fs::absolute(options.input_directory).lexically_normal();
    const fs::path output_absolute =
        fs::absolute(options.output_directory).lexically_normal();
    if (input_absolute == output_absolute) {
        usageError("The output directory must differ from the input directory.");
    }

    return options;
}

std::vector<fs::path> listPcdFiles(const fs::path& directory) {
    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        std::string extension = entry.path().extension().string();
        std::transform(
            extension.begin(), extension.end(), extension.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (extension == ".pcd") {
            files.push_back(entry.path());
        }
    }

    std::sort(files.begin(), files.end(),
              [](const fs::path& a, const fs::path& b) {
                  return a.filename().string() < b.filename().string();
              });
    return files;
}

std::unordered_map<std::string, Eigen::Matrix4f> loadPoses(
    const fs::path& pose_file) {
    std::ifstream input(pose_file);
    if (!input) {
        throw std::runtime_error("Cannot open pose file: " +
                                 pose_file.string());
    }

    std::unordered_map<std::string, Eigen::Matrix4f> poses;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        const auto first_non_space = line.find_first_not_of(" \t\r\n");
        if (first_non_space == std::string::npos ||
            line[first_non_space] == '#') {
            continue;
        }

        std::istringstream stream(line);
        std::string filename;
        double tx = 0.0;
        double ty = 0.0;
        double tz = 0.0;
        double qx = 0.0;
        double qy = 0.0;
        double qz = 0.0;
        double qw = 1.0;
        if (!(stream >> filename >> tx >> ty >> tz >> qx >> qy >> qz >> qw)) {
            throw std::runtime_error(
                "Invalid pose at line " + std::to_string(line_number) +
                ". Expected: filename tx ty tz qx qy qz qw");
        }

        const std::string key = fs::path(filename).filename().string();
        if (poses.count(key) != 0U) {
            throw std::runtime_error("Duplicate pose for: " + key);
        }

        Eigen::Quaterniond quaternion(qw, qx, qy, qz);
        if (!quaternion.coeffs().allFinite() ||
            quaternion.norm() < 1e-12) {
            throw std::runtime_error("Invalid quaternion for: " + key);
        }
        quaternion.normalize();

        Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
        transform.block<3, 3>(0, 0) = quaternion.toRotationMatrix();
        transform.block<3, 1>(0, 3) = Eigen::Vector3d(tx, ty, tz);
        poses.emplace(key, transform.cast<float>());
    }

    return poses;
}

CloudPtr cropSphericalRoi(
    const Cloud& cloud,
    bool use_roi,
    const Eigen::Vector3d& center,
    double radius) {
    CloudPtr output(new Cloud);
    output->points.reserve(cloud.size());
    const double radius_squared = radius * radius;

    for (const Point& point : cloud.points) {
        if (!use_roi) {
            output->points.push_back(point);
            continue;
        }
        const double dx = static_cast<double>(point.x) - center.x();
        const double dy = static_cast<double>(point.y) - center.y();
        const double dz = static_cast<double>(point.z) - center.z();
        if (dx * dx + dy * dy + dz * dz <= radius_squared) {
            output->points.push_back(point);
        }
    }

    output->width = static_cast<std::uint32_t>(output->size());
    output->height = 1;
    output->is_dense = true;
    return output;
}

float median(std::vector<float>& values) {
    const std::size_t middle = values.size() / 2U;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    const float upper = values[middle];
    if ((values.size() % 2U) != 0U) {
        return upper;
    }
    const float lower =
        *std::max_element(values.begin(), values.begin() + middle);
    return 0.5F * (lower + upper);
}

CloudPtr medianVoxelDownsample(const Cloud& cloud, double leaf_size) {
    if (leaf_size <= 0.0 || cloud.empty()) {
        CloudPtr copy(new Cloud(cloud));
        return copy;
    }

    std::unordered_map<VoxelKey, VoxelBucket, VoxelKeyHash> buckets;
    buckets.reserve(cloud.size());

    for (const Point& point : cloud.points) {
        VoxelKey key;
        key.x = static_cast<std::int64_t>(
            std::floor(static_cast<double>(point.x) / leaf_size));
        key.y = static_cast<std::int64_t>(
            std::floor(static_cast<double>(point.y) / leaf_size));
        key.z = static_cast<std::int64_t>(
            std::floor(static_cast<double>(point.z) / leaf_size));
        VoxelBucket& bucket = buckets[key];
        bucket.x.push_back(point.x);
        bucket.y.push_back(point.y);
        bucket.z.push_back(point.z);
    }

    CloudPtr output(new Cloud);
    output->points.reserve(buckets.size());
    for (auto& item : buckets) {
        VoxelBucket& bucket = item.second;
        Point point;
        point.x = median(bucket.x);
        point.y = median(bucket.y);
        point.z = median(bucket.z);
        output->points.push_back(point);
    }

    output->width = static_cast<std::uint32_t>(output->size());
    output->height = 1;
    output->is_dense = true;
    return output;
}

SphereModel runSphereRansac(const CloudPtr& cloud, const Options& options) {
    pcl::SACSegmentation<Point> segmentation;
    segmentation.setOptimizeCoefficients(true);
    segmentation.setModelType(pcl::SACMODEL_SPHERE);
    segmentation.setMethodType(pcl::SAC_RANSAC);
    segmentation.setDistanceThreshold(options.ransac_threshold);
    segmentation.setRadiusLimits(options.radius_min, options.radius_max);
    segmentation.setMaxIterations(options.max_iterations);
    segmentation.setProbability(options.probability);
    segmentation.setInputCloud(cloud);

    pcl::PointIndices pcl_inliers;
    pcl::ModelCoefficients coefficients;
    segmentation.segment(pcl_inliers, coefficients);

    if (pcl_inliers.indices.empty() || coefficients.values.size() < 4U) {
        throw std::runtime_error(
            "RANSAC did not find a sphere. Tighten the ROI or relax only the "
            "coarse threshold/radius range.");
    }

    SphereModel model;
    model.center =
        Eigen::Vector3d(coefficients.values[0],
                        coefficients.values[1],
                        coefficients.values[2]);
    model.radius = coefficients.values[3];
    model.inliers = std::move(pcl_inliers.indices);
    return model;
}

std::vector<int> selectInliers(
    const Cloud& cloud,
    const Eigen::Vector3d& center,
    double radius,
    double threshold) {
    std::vector<int> indices;
    indices.reserve(cloud.size());
    for (std::size_t i = 0; i < cloud.size(); ++i) {
        const Point& point = cloud.points[i];
        const Eigen::Vector3d position(point.x, point.y, point.z);
        const double residual =
            std::abs((position - center).norm() - radius);
        if (residual <= threshold) {
            indices.push_back(static_cast<int>(i));
        }
    }
    return indices;
}

bool refineFixedRadius(
    const Cloud& cloud,
    Eigen::Vector3d& center,
    double radius,
    double gate,
    double huber_delta,
    const std::vector<int>* initial_candidates = nullptr,
    std::vector<int>* final_candidates = nullptr) {
    std::vector<int> candidates;
    if (initial_candidates != nullptr) {
        candidates = *initial_candidates;
        candidates.erase(
            std::remove_if(
                candidates.begin(),
                candidates.end(),
                [&cloud](int index) {
                    return index < 0 ||
                           static_cast<std::size_t>(index) >= cloud.size();
                }),
            candidates.end());
    } else {
        candidates = selectInliers(cloud, center, radius, gate);
    }
    if (candidates.size() < 4U) {
        return false;
    }

    for (int outer = 0; outer < 8; ++outer) {
        for (int inner = 0; inner < 30; ++inner) {
            Eigen::Matrix3d normal_matrix = Eigen::Matrix3d::Zero();
            Eigen::Vector3d gradient = Eigen::Vector3d::Zero();
            std::size_t usable = 0;

            for (const int index : candidates) {
                const Point& point =
                    cloud.points[static_cast<std::size_t>(index)];
                const Eigen::Vector3d position(point.x, point.y, point.z);
                const Eigen::Vector3d difference = center - position;
                const double distance = difference.norm();
                if (distance < 1e-12) {
                    continue;
                }

                const double residual = distance - radius;
                const double absolute_residual = std::abs(residual);
                const double weight =
                    absolute_residual <= huber_delta
                        ? 1.0
                        : huber_delta / absolute_residual;
                const Eigen::Vector3d jacobian = difference / distance;
                normal_matrix.noalias() +=
                    weight * jacobian * jacobian.transpose();
                gradient.noalias() += weight * jacobian * residual;
                ++usable;
            }

            if (usable < 4U) {
                return false;
            }

            normal_matrix.diagonal().array() += 1e-12;
            const Eigen::LDLT<Eigen::Matrix3d> decomposition(normal_matrix);
            if (decomposition.info() != Eigen::Success) {
                return false;
            }

            Eigen::Vector3d step = -decomposition.solve(gradient);
            if (!step.allFinite()) {
                return false;
            }

            const double step_norm = step.norm();
            if (step_norm > 0.01) {
                step *= 0.01 / step_norm;
            }
            center += step;
            if (step.norm() < 1e-8) {
                break;
            }
        }

        std::vector<int> updated =
            selectInliers(cloud, center, radius, gate);
        if (updated.size() < 4U) {
            return false;
        }
        if (updated == candidates) {
            candidates = std::move(updated);
            break;
        }
        candidates = std::move(updated);
    }

    if (final_candidates != nullptr) {
        *final_candidates = std::move(candidates);
    }
    return true;
}

Classification classifyCloud(
    const Cloud& cloud,
    const Eigen::Vector3d& center,
    double radius,
    double threshold) {
    Classification result;
    result.inliers->points.reserve(cloud.size());
    result.outliers->points.reserve(cloud.size());
    result.inlier_indices.reserve(cloud.size());
    result.signed_residuals.reserve(cloud.size());

    for (std::size_t i = 0; i < cloud.size(); ++i) {
        const Point& point = cloud.points[i];
        const Eigen::Vector3d position(point.x, point.y, point.z);
        const double signed_residual =
            (position - center).norm() - radius;
        if (std::abs(signed_residual) <= threshold) {
            result.inliers->points.push_back(point);
            result.inlier_indices.push_back(static_cast<int>(i));
            result.signed_residuals.push_back(signed_residual);
        } else {
            result.outliers->points.push_back(point);
        }
    }

    result.inliers->width =
        static_cast<std::uint32_t>(result.inliers->size());
    result.inliers->height = 1;
    result.inliers->is_dense = true;
    result.outliers->width =
        static_cast<std::uint32_t>(result.outliers->size());
    result.outliers->height = 1;
    result.outliers->is_dense = true;
    return result;
}

ResidualStats computeResidualStats(
    const std::vector<double>& signed_residuals) {
    ResidualStats stats;
    stats.count = signed_residuals.size();
    if (signed_residuals.empty()) {
        return stats;
    }

    std::vector<double> absolute;
    absolute.reserve(signed_residuals.size());
    double squared_sum = 0.0;
    double absolute_sum = 0.0;
    double signed_sum = 0.0;
    for (const double residual : signed_residuals) {
        const double abs_residual = std::abs(residual);
        squared_sum += residual * residual;
        absolute_sum += abs_residual;
        signed_sum += residual;
        absolute.push_back(abs_residual);
    }
    std::sort(absolute.begin(), absolute.end());

    const double count = static_cast<double>(signed_residuals.size());
    stats.rmse = std::sqrt(squared_sum / count);
    stats.mae = absolute_sum / count;
    stats.mean_signed = signed_sum / count;
    const std::size_t median_index = absolute.size() / 2U;
    stats.median_absolute = absolute[median_index];
    const std::size_t p95_index = static_cast<std::size_t>(
        std::ceil(0.95 * count) - 1.0);
    stats.p95_absolute =
        absolute[std::min(p95_index, absolute.size() - 1U)];
    stats.max_absolute = absolute.back();
    return stats;
}

Eigen::Vector3d directionEigenvalues(
    const Cloud& cloud,
    const std::vector<int>& indices,
    const Eigen::Vector3d& center) {
    Eigen::Matrix3d information = Eigen::Matrix3d::Zero();
    std::size_t usable = 0;
    for (const int index : indices) {
        const Point& point =
            cloud.points[static_cast<std::size_t>(index)];
        Eigen::Vector3d direction =
            Eigen::Vector3d(point.x, point.y, point.z) - center;
        const double norm = direction.norm();
        if (norm < 1e-12) {
            continue;
        }
        direction /= norm;
        information.noalias() += direction * direction.transpose();
        ++usable;
    }
    if (usable == 0U) {
        return Eigen::Vector3d::Zero();
    }
    information /= static_cast<double>(usable);
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(information);
    if (solver.info() != Eigen::Success) {
        return Eigen::Vector3d::Zero();
    }
    return solver.eigenvalues();
}

SubsetEstimate estimateSubset(
    const std::vector<FrameData>& frames,
    std::size_t begin,
    std::size_t end,
    const Options& options,
    const Eigen::Vector3d& initial_center) {
    SubsetEstimate estimate;
    Cloud merged;
    for (std::size_t i = begin; i < end; ++i) {
        const Cloud& frame = *frames[i].roi_cloud;
        merged.points.insert(
            merged.points.end(), frame.points.begin(), frame.points.end());
    }
    merged.width = static_cast<std::uint32_t>(merged.size());
    merged.height = 1;
    merged.is_dense = true;

    CloudPtr reduced =
        medianVoxelDownsample(merged, options.voxel_size);
    estimate.points = reduced->size();
    if (reduced->size() < 4U) {
        return estimate;
    }

    estimate.center = initial_center;
    if (!refineFixedRadius(
            *reduced,
            estimate.center,
            options.fixed_radius,
            options.ransac_threshold,
            options.huber_delta)) {
        return estimate;
    }
    estimate.inliers =
        selectInliers(
            *reduced,
            estimate.center,
            options.fixed_radius,
            options.final_threshold)
            .size();
    estimate.valid = estimate.inliers >= 4U;
    return estimate;
}

template <typename PointT>
void savePcd(
    const fs::path& path,
    const pcl::PointCloud<PointT>& cloud) {
    if (pcl::io::savePCDFileBinary(path.string(), cloud) < 0) {
        throw std::runtime_error("Failed to save: " + path.string());
    }
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr makeColoredCloud(
    const Cloud& cloud,
    const Eigen::Vector3d& center,
    double radius,
    double threshold) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored(
        new pcl::PointCloud<pcl::PointXYZRGB>);
    colored->points.reserve(cloud.size() + 33U);

    for (const Point& point : cloud.points) {
        pcl::PointXYZRGB output;
        output.x = point.x;
        output.y = point.y;
        output.z = point.z;
        const Eigen::Vector3d position(point.x, point.y, point.z);
        const bool is_inlier =
            std::abs((position - center).norm() - radius) <= threshold;
        if (is_inlier) {
            output.r = 0;
            output.g = 255;
            output.b = 0;
        } else {
            output.r = 120;
            output.g = 120;
            output.b = 120;
        }
        output.a = 255;
        colored->points.push_back(output);
    }

    for (int axis = 0; axis < 3; ++axis) {
        for (int step = -5; step <= 5; ++step) {
            Eigen::Vector3d marker = center;
            marker[axis] += 0.002 * static_cast<double>(step);
            pcl::PointXYZRGB point;
            point.x = static_cast<float>(marker.x());
            point.y = static_cast<float>(marker.y());
            point.z = static_cast<float>(marker.z());
            point.r = 255;
            point.g = 0;
            point.b = 0;
            point.a = 255;
            colored->points.push_back(point);
        }
    }

    colored->width = static_cast<std::uint32_t>(colored->size());
    colored->height = 1;
    colored->is_dense = true;
    return colored;
}

int main(int argc, char** argv) {
    try {
        const Options options = parseArguments(argc, argv);
        const std::vector<fs::path> pcd_files =
            listPcdFiles(options.input_directory);
        if (pcd_files.size() < 2U) {
            throw std::runtime_error(
                "At least two PCD frames are required in: " +
                options.input_directory.string());
        }

        std::unordered_map<std::string, Eigen::Matrix4f> poses;
        if (options.poses_file.has_value()) {
            poses = loadPoses(*options.poses_file);
        }

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Found " << pcd_files.size() << " PCD frames.\n";
        if (!options.poses_file.has_value()) {
            std::cout
                << "No pose file: assuming every PCD is already in the same "
                << "coordinate frame.\n";
        }
        if (!options.use_roi) {
            std::cerr
                << "WARNING: no ROI was supplied. Full-scene RANSAC is often "
                << "unreliable for a small sphere.\n";
        }

        std::vector<FrameData> frames;
        frames.reserve(pcd_files.size());
        CloudPtr merged_roi(new Cloud);

        for (const fs::path& pcd_path : pcd_files) {
            CloudPtr loaded(new Cloud);
            if (pcl::io::loadPCDFile<Point>(
                    pcd_path.string(), *loaded) < 0) {
                throw std::runtime_error(
                    "Failed to read PCD: " + pcd_path.string());
            }

            FrameData frame;
            frame.name = pcd_path.filename().string();
            frame.loaded_points = loaded->size();

            CloudPtr valid(new Cloud);
            std::vector<int> valid_indices;
            pcl::removeNaNFromPointCloud(
                *loaded, *valid, valid_indices);
            frame.valid_points = valid->size();

            CloudPtr common_frame(new Cloud);
            if (options.poses_file.has_value()) {
                const auto pose = poses.find(frame.name);
                if (pose == poses.end()) {
                    throw std::runtime_error(
                        "Pose is missing for PCD: " + frame.name);
                }
                pcl::transformPointCloud(
                    *valid, *common_frame, pose->second);
            } else {
                *common_frame = *valid;
            }

            frame.roi_cloud = cropSphericalRoi(
                *common_frame,
                options.use_roi,
                options.roi_center,
                options.roi_radius);
            merged_roi->points.insert(
                merged_roi->points.end(),
                frame.roi_cloud->points.begin(),
                frame.roi_cloud->points.end());

            std::cout
                << "  " << frame.name
                << ": loaded=" << frame.loaded_points
                << ", valid=" << frame.valid_points
                << ", roi=" << frame.roi_cloud->size() << '\n';
            frames.push_back(std::move(frame));
        }

        merged_roi->width =
            static_cast<std::uint32_t>(merged_roi->size());
        merged_roi->height = 1;
        merged_roi->is_dense = true;
        if (merged_roi->size() < 4U) {
            throw std::runtime_error(
                "Fewer than four points remain after ROI cropping.");
        }

        CloudPtr median_cloud =
            medianVoxelDownsample(*merged_roi, options.voxel_size);
        if (median_cloud->size() < 4U) {
            throw std::runtime_error(
                "Fewer than four points remain after voxel filtering.");
        }

        std::cout
            << "Merged ROI points: " << merged_roi->size() << '\n'
            << "Median-voxel points: " << median_cloud->size() << '\n';

        const SphereModel coarse =
            runSphereRansac(median_cloud, options);
        Eigen::Vector3d final_center = coarse.center;
        std::vector<int> refine_candidates;
        if (!refineFixedRadius(
                *median_cloud,
                final_center,
                options.fixed_radius,
                options.ransac_threshold,
                options.huber_delta,
                &coarse.inliers,
                &refine_candidates)) {
            throw std::runtime_error(
                "Known-radius center refinement failed.");
        }

        const Classification voxel_classification =
            classifyCloud(
                *median_cloud,
                final_center,
                options.fixed_radius,
                options.final_threshold);
        const Classification raw_classification =
            classifyCloud(
                *merged_roi,
                final_center,
                options.fixed_radius,
                options.final_threshold);
        if (voxel_classification.inliers->empty()) {
            throw std::runtime_error(
                "No final inliers satisfy the strict shell threshold.");
        }

        const ResidualStats stats =
            computeResidualStats(
                voxel_classification.signed_residuals);
        const Eigen::Vector3d direction_eigenvalues =
            directionEigenvalues(
                *median_cloud,
                voxel_classification.inlier_indices,
                final_center);
        const double direction_ratio =
            direction_eigenvalues.z() > 0.0
                ? direction_eigenvalues.x() /
                      direction_eigenvalues.z()
                : 0.0;

        std::vector<std::pair<std::string, std::size_t>>
            frame_inlier_counts;
        frame_inlier_counts.reserve(frames.size());
        std::size_t supporting_frames = 0;
        for (const FrameData& frame : frames) {
            const std::size_t count =
                selectInliers(
                    *frame.roi_cloud,
                    final_center,
                    options.fixed_radius,
                    options.final_threshold)
                    .size();
            frame_inlier_counts.emplace_back(frame.name, count);
            if (count >= options.min_points_per_frame) {
                ++supporting_frames;
            }
        }

        SubsetEstimate first_half;
        SubsetEstimate second_half;
        double half_center_delta =
            std::numeric_limits<double>::quiet_NaN();
        const std::size_t split = frames.size() / 2U;
        if (split > 0U && split < frames.size()) {
            first_half = estimateSubset(
                frames, 0U, split, options, final_center);
            second_half = estimateSubset(
                frames, split, frames.size(), options, final_center);
            if (first_half.valid && second_half.valid) {
                half_center_delta =
                    (first_half.center - second_half.center).norm();
            }
        }

        fs::create_directories(options.output_directory);
        savePcd(
            options.output_directory / "merged_roi_raw.pcd",
            *merged_roi);
        savePcd(
            options.output_directory / "merged_voxel_median.pcd",
            *median_cloud);
        savePcd(
            options.output_directory / "sphere_inliers.pcd",
            *raw_classification.inliers);
        savePcd(
            options.output_directory / "sphere_outliers.pcd",
            *raw_classification.outliers);

        Cloud center_cloud;
        Point center_point;
        center_point.x = static_cast<float>(final_center.x());
        center_point.y = static_cast<float>(final_center.y());
        center_point.z = static_cast<float>(final_center.z());
        center_cloud.points.push_back(center_point);
        center_cloud.width = 1;
        center_cloud.height = 1;
        center_cloud.is_dense = true;
        savePcd(
            options.output_directory / "sphere_center.pcd",
            center_cloud);

        const auto colored = makeColoredCloud(
            *merged_roi,
            final_center,
            options.fixed_radius,
            options.final_threshold);
        savePcd(
            options.output_directory / "sphere_fit_colored.pcd",
            *colored);

        const bool enough_inliers =
            voxel_classification.inliers->size() >=
            options.min_inliers;
        const std::size_t required_supporting_frames =
            options.min_supporting_frames > 0U
                ? options.min_supporting_frames
                : static_cast<std::size_t>(
                      std::ceil(0.60 *
                                static_cast<double>(frames.size())));
        const bool enough_frames =
            supporting_frames >= required_supporting_frames;
        const bool quality_pass = enough_inliers && enough_frames;

        std::ofstream report(
            options.output_directory / "sphere_report.txt");
        if (!report) {
            throw std::runtime_error(
                "Failed to create sphere_report.txt.");
        }
        report << std::fixed << std::setprecision(10);
        report
            << "coordinate_frame: "
            << (options.poses_file.has_value()
                    ? "common frame defined by poses"
                    : "input PCD frame (assumed common)")
            << '\n'
            << "frames_total: " << frames.size() << '\n'
            << "frames_supporting: " << supporting_frames << '\n'
            << "frames_supporting_required: "
            << required_supporting_frames << '\n'
            << "min_points_per_supporting_frame: "
            << options.min_points_per_frame << '\n'
            << "merged_roi_raw_points: " << merged_roi->size() << '\n'
            << "merged_voxel_median_points: "
            << median_cloud->size() << '\n'
            << "coarse_center_m: "
            << coarse.center.x() << ' '
            << coarse.center.y() << ' '
            << coarse.center.z() << '\n'
            << "coarse_radius_m: " << coarse.radius << '\n'
            << "coarse_diameter_m: " << 2.0 * coarse.radius << '\n'
            << "coarse_ransac_inliers: "
            << coarse.inliers.size() << '\n'
            << "final_center_m: "
            << final_center.x() << ' '
            << final_center.y() << ' '
            << final_center.z() << '\n'
            << "final_fixed_radius_m: "
            << options.fixed_radius << '\n'
            << "final_fixed_diameter_m: "
            << 2.0 * options.fixed_radius << '\n'
            << "final_threshold_m: "
            << options.final_threshold << '\n'
            << "final_voxel_inliers: "
            << voxel_classification.inliers->size() << '\n'
            << "final_raw_inliers: "
            << raw_classification.inliers->size() << '\n'
            << "final_voxel_inlier_ratio_percent: "
            << 100.0 *
                   static_cast<double>(
                       voxel_classification.inliers->size()) /
                   static_cast<double>(median_cloud->size())
            << '\n'
            << "rmse_m: " << stats.rmse << '\n'
            << "mae_m: " << stats.mae << '\n'
            << "mean_signed_residual_m: "
            << stats.mean_signed << '\n'
            << "median_absolute_residual_m: "
            << stats.median_absolute << '\n'
            << "p95_absolute_residual_m: "
            << stats.p95_absolute << '\n'
            << "max_absolute_residual_m: "
            << stats.max_absolute << '\n'
            << "direction_information_eigenvalues: "
            << direction_eigenvalues.x() << ' '
            << direction_eigenvalues.y() << ' '
            << direction_eigenvalues.z() << '\n'
            << "direction_min_max_ratio: "
            << direction_ratio << '\n';

        if (first_half.valid && second_half.valid) {
            report
                << "first_half_center_m: "
                << first_half.center.x() << ' '
                << first_half.center.y() << ' '
                << first_half.center.z() << '\n'
                << "first_half_inliers: "
                << first_half.inliers << '\n'
                << "second_half_center_m: "
                << second_half.center.x() << ' '
                << second_half.center.y() << ' '
                << second_half.center.z() << '\n'
                << "second_half_inliers: "
                << second_half.inliers << '\n'
                << "half_center_delta_m: "
                << half_center_delta << '\n';
        } else {
            report << "half_center_delta_m: unavailable\n";
        }

        report
            << "quality_enough_inliers: "
            << (enough_inliers ? "true" : "false") << '\n'
            << "quality_enough_frames: "
            << (enough_frames ? "true" : "false") << '\n'
            << "quality_pass: "
            << (quality_pass ? "true" : "false") << '\n'
            << "\nper_frame_final_inliers:\n";
        for (const auto& item : frame_inlier_counts) {
            report << item.first << ' ' << item.second << '\n';
        }

        std::cout
            << "\nCoarse RANSAC center [m]: "
            << coarse.center.transpose() << '\n'
            << "Coarse RANSAC radius [m]: "
            << coarse.radius << '\n'
            << "Final fixed-radius center [m]: "
            << final_center.transpose() << '\n'
            << "Final fixed radius [m]: "
            << options.fixed_radius << '\n'
            << "Final voxel inliers: "
            << voxel_classification.inliers->size()
            << " / " << median_cloud->size() << '\n'
            << "Supporting frames: "
            << supporting_frames << " / " << frames.size()
            << " (required " << required_supporting_frames << ")\n"
            << "RMSE / P95 [mm]: "
            << 1000.0 * stats.rmse << " / "
            << 1000.0 * stats.p95_absolute << '\n';
        if (std::isfinite(half_center_delta)) {
            std::cout
                << "First-half vs second-half center delta [mm]: "
                << 1000.0 * half_center_delta << '\n';
        }
        if (direction_ratio < 0.01) {
            std::cerr
                << "WARNING: sphere points have weak 3-D directional "
                << "coverage; the center may be poorly constrained.\n";
        }
        if (!quality_pass) {
            std::cerr
                << "WARNING: quality gate failed. Inspect sphere_report.txt "
                << "and sphere_fit_colored.pcd.\n";
        }
        std::cout
            << "Results saved to: "
            << options.output_directory << '\n';

        return quality_pass ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
