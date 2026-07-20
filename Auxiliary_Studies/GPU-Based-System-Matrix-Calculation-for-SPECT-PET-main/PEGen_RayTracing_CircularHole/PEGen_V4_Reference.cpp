#define _CRT_SECURE_NO_WARNINGS

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../common/first_interaction.h"
#include "../common/pe_v4_reference.h"

namespace
{
struct Options
{
    int detector_index = -1;
    long long voxel_index = -1;
    int rotation_index = 0;
    int face_subdivisions = 16;
    int depth_subdivisions = 8;
    std::string surface_rule = "halton";
    std::string v3_matrix_path;
    std::string output_path;
};

struct Vec3
{
    double x;
    double y;
    double z;
};

struct DetectorRecord
{
    double x;
    double y;
    double z;
    double width;
    double thickness;
    double height;
    double mu_total;
    double mu_photoelectric;
    double mu_compton;
    double rotation_y;
    int flag;
};

void printUsage(const char* executable)
{
    std::cout
        << "Usage: " << executable << " --detector INDEX [options]\n"
        << "Options:\n"
        << "  --voxel INDEX          0-based voxel index; default is grid center\n"
        << "  --rotation INDEX       0-based rotation index (default 0)\n"
        << "  --face-subdiv N        subdivisions per visible face axis (default 16)\n"
        << "  --depth-subdiv N       conditional first-depth samples (default 8)\n"
        << "  --surface-rule RULE    halton (default) or gauss\n"
        << "  --v3 PATH              optional raw v3 PE matrix for direct comparison\n"
        << "  --output PATH          output CSV path\n"
        << "\nThe reference evaluates exactly one detector/voxel pair. It supports the\n"
        << "current JSCC vacuum/no-hole collimator and rejects physical holes.\n";
}

long long parseInteger(const char* text, const char* name)
{
    char* end = NULL;
    const long long value = std::strtoll(text, &end, 10);
    if (end == text || *end != '\0')
        throw std::runtime_error(std::string("Invalid ") + name + ": " + text);
    return value;
}

Options parseOptions(int argc, char** argv)
{
    Options options;
    for (int index = 1; index < argc; ++index)
    {
        const std::string argument = argv[index];
        if (argument == "-h" || argument == "--help")
        {
            printUsage(argv[0]);
            std::exit(EXIT_SUCCESS);
        }
        if (index + 1 >= argc)
            throw std::runtime_error("Missing value after " + argument);
        const char* value = argv[++index];
        if (argument == "--detector")
            options.detector_index = static_cast<int>(parseInteger(value, "detector"));
        else if (argument == "--voxel")
            options.voxel_index = parseInteger(value, "voxel");
        else if (argument == "--rotation")
            options.rotation_index = static_cast<int>(parseInteger(value, "rotation"));
        else if (argument == "--face-subdiv")
            options.face_subdivisions = static_cast<int>(parseInteger(value, "face-subdiv"));
        else if (argument == "--depth-subdiv")
            options.depth_subdivisions = static_cast<int>(parseInteger(value, "depth-subdiv"));
        else if (argument == "--surface-rule")
            options.surface_rule = value;
        else if (argument == "--v3")
            options.v3_matrix_path = value;
        else if (argument == "--output")
            options.output_path = value;
        else
            throw std::runtime_error("Unknown option: " + argument);
    }
    if (options.detector_index < 0)
        throw std::runtime_error("--detector is required and must be nonnegative");
    if (options.rotation_index < 0 || options.face_subdivisions < 1
        || options.depth_subdivisions < 1)
        throw std::runtime_error("rotation and subdivision arguments are out of range");
    if (options.surface_rule != "halton" && options.surface_rule != "gauss")
        throw std::runtime_error("--surface-rule must be halton or gauss");
    return options;
}

std::vector<float> readFloatFile(const std::string& path)
{
    std::ifstream stream(path.c_str(), std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("Cannot open " + path);
    const std::streamoff byte_count = stream.tellg();
    if (byte_count < 0 || byte_count % static_cast<std::streamoff>(sizeof(float)) != 0)
        throw std::runtime_error("Malformed float32 file: " + path);
    stream.seekg(0, std::ios::beg);
    std::vector<float> values(static_cast<std::size_t>(byte_count / sizeof(float)));
    if (!values.empty())
        stream.read(reinterpret_cast<char*>(&values[0]), byte_count);
    if (!stream) throw std::runtime_error("Failed to read " + path);
    return values;
}

DetectorRecord detectorRecord(
    const std::vector<float>& detector,
    int index,
    double fov_to_detector)
{
    const std::size_t base = 1 + static_cast<std::size_t>(index) * 12;
    if (base + 11 >= detector.size())
        throw std::runtime_error("Detector record exceeds Params_Detector.dat");
    DetectorRecord record;
    record.x = detector[base + 0];
    record.y = detector[base + 1] + fov_to_detector;
    record.z = detector[base + 2];
    record.width = detector[base + 3];
    record.thickness = detector[base + 4];
    record.height = detector[base + 5];
    record.mu_total = detector[base + 6];
    record.mu_photoelectric = detector[base + 7];
    record.mu_compton = detector[base + 8];
    record.rotation_y = detector[base + 10];
    record.flag = static_cast<int>(std::floor(detector[base + 11] + 0.5f));
    return record;
}

Vec3 worldToLocal(const Vec3& point, const DetectorRecord& detector)
{
    const double cosine = std::cos(-detector.rotation_y);
    const double sine = std::sin(-detector.rotation_y);
    const double dx = point.x - detector.x;
    const double dz = point.z - detector.z;
    Vec3 local;
    local.x = dx * cosine - dz * sine;
    local.y = point.y - detector.y;
    local.z = dx * sine + dz * cosine;
    return local;
}

Vec3 localToWorld(const Vec3& point, const DetectorRecord& detector)
{
    const double cosine = std::cos(detector.rotation_y);
    const double sine = std::sin(detector.rotation_y);
    Vec3 world;
    world.x = detector.x + point.x * cosine - point.z * sine;
    world.y = detector.y + point.y;
    world.z = detector.z + point.x * sine + point.z * cosine;
    return world;
}

Vec3 voxelCenter(
    const std::vector<float>& image,
    long long voxel_index,
    int rotation_index)
{
    const int count_x = static_cast<int>(std::floor(image[0] + 0.5f));
    const int count_y = static_cast<int>(std::floor(image[1] + 0.5f));
    const int count_z = static_cast<int>(std::floor(image[2] + 0.5f));
    const long long in_slice = voxel_index % (static_cast<long long>(count_x) * count_y);
    const int index_z = static_cast<int>(voxel_index
        / (static_cast<long long>(count_x) * count_y));
    const int index_y = static_cast<int>(in_slice / count_x);
    const int index_x = static_cast<int>(in_slice % count_x);
    const double unrotated_x = (index_x - count_x / 2.0 + 0.5) * image[3] + image[8];
    const double unrotated_y = (index_y - count_y / 2.0 + 0.5) * image[4] + image[9];
    const double angle = rotation_index * image[7];
    Vec3 center;
    center.x = unrotated_x * std::cos(angle) - unrotated_y * std::sin(angle);
    center.y = unrotated_x * std::sin(angle) + unrotated_y * std::cos(angle);
    center.z = (index_z - count_z / 2.0 + 0.5) * image[5] + image[10];
    return center;
}

double segmentChord(const Vec3& start, const Vec3& end, const DetectorRecord& detector)
{
    const Vec3 local_start = worldToLocal(start, detector);
    const Vec3 local_end = worldToLocal(end, detector);
    return detector_segment_box_chord(
        local_start.x, local_start.y, local_start.z,
        local_end.x, local_end.y, local_end.z,
        detector.width, detector.thickness, detector.height);
}

double collimatorAttenuation(
    const Vec3& source,
    const Vec3& entry,
    const std::vector<float>& collimator,
    double fov_to_collimator)
{
    const int layer_count = static_cast<int>(std::floor(collimator[0] + 0.5f));
    double attenuation = 0.0;
    for (int layer = 0; layer < layer_count; ++layer)
    {
        const std::size_t base = static_cast<std::size_t>(layer + 1) * 10;
        const int hole_count = static_cast<int>(std::floor(collimator[base] + 0.5f));
        if (hole_count != 0)
            throw std::runtime_error(
                "PE v4 reference currently requires zero physical collimator holes");
        const double width = collimator[base + 1];
        const double thickness = collimator[base + 2];
        const double height = collimator[base + 3];
        const double center_y = fov_to_collimator + collimator[base + 4];
        const double mu_total = collimator[base + 5];
        if (!(mu_total > 0.0)) continue;
        const double chord = detector_segment_box_chord(
            source.x, source.y - center_y, source.z,
            entry.x, entry.y - center_y, entry.z,
            width, thickness, height);
        attenuation += chord * mu_total;
    }
    return attenuation;
}

float readV3Element(
    const std::string& path,
    std::size_t detector_count,
    std::size_t voxel_count,
    int rotation_index,
    int detector_index,
    long long voxel_index)
{
    const std::size_t element_index
        = (static_cast<std::size_t>(rotation_index) * detector_count
            + static_cast<std::size_t>(detector_index)) * voxel_count
        + static_cast<std::size_t>(voxel_index);
    std::ifstream stream(path.c_str(), std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot open v3 matrix: " + path);
    stream.seekg(static_cast<std::streamoff>(element_index * sizeof(float)), std::ios::beg);
    float value = 0.0f;
    stream.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!stream) throw std::runtime_error("Cannot read requested v3 matrix element");
    return value;
}
}

int main(int argc, char** argv)
{
    try
    {
        Options options = parseOptions(argc, argv);
        const std::vector<float> collimator = readFloatFile("Params_Collimator.dat");
        const std::vector<float> detector = readFloatFile("Params_Detector.dat");
        std::vector<float> image = readFloatFile("Params_Image.dat");
        if (collimator.empty() || detector.empty() || image.size() < 12)
            throw std::runtime_error("Parameter files are empty or incomplete");
        if (image.size() < 100) image.resize(100, 0.0f);

        const int detector_count = static_cast<int>(std::floor(detector[0] + 0.5f));
        const int count_x = static_cast<int>(std::floor(image[0] + 0.5f));
        const int count_y = static_cast<int>(std::floor(image[1] + 0.5f));
        const int count_z = static_cast<int>(std::floor(image[2] + 0.5f));
        const int rotation_count = static_cast<int>(std::floor(image[6] + 0.5f));
        const long long voxel_count = static_cast<long long>(count_x) * count_y * count_z;
        if (options.detector_index >= detector_count)
            throw std::runtime_error("Detector index exceeds detector count");
        if (options.rotation_index >= rotation_count)
            throw std::runtime_error("Rotation index exceeds rotation count");
        if (options.voxel_index < 0) options.voxel_index = voxel_count / 2;
        if (options.voxel_index >= voxel_count)
            throw std::runtime_error("Voxel index exceeds voxel count");

        const double fov_to_detector = image[11];
        const DetectorRecord target = detectorRecord(
            detector, options.detector_index, fov_to_detector);
        const Vec3 source_world = voxelCenter(
            image, options.voxel_index, options.rotation_index);
        const Vec3 source_local = worldToLocal(source_world, target);
        if (std::fabs(source_local.x) <= 0.5 * target.width
            && std::fabs(source_local.y) <= 0.5 * target.thickness
            && std::fabs(source_local.z) <= 0.5 * target.height)
            throw std::runtime_error("Selected source voxel lies inside target detector");

        double cached_entry_x = std::numeric_limits<double>::quiet_NaN();
        double cached_entry_y = std::numeric_limits<double>::quiet_NaN();
        double cached_entry_z = std::numeric_limits<double>::quiet_NaN();
        double cached_survival = 0.0;
        const PEV4ReferenceResult result = integrate_pe_v4_point_source_reference(
            source_local.x, source_local.y, source_local.z,
            target.width, target.thickness, target.height,
            target.mu_photoelectric, target.mu_compton,
            options.face_subdivisions, options.depth_subdivisions,
            [&](const FirstInteractionState& state)
            {
                if (state.entry_x == cached_entry_x
                    && state.entry_y == cached_entry_y
                    && state.entry_z == cached_entry_z)
                    return cached_survival;
                cached_entry_x = state.entry_x;
                cached_entry_y = state.entry_y;
                cached_entry_z = state.entry_z;
                const Vec3 entry_local = {
                    state.entry_x, state.entry_y, state.entry_z
                };
                const Vec3 entry_world = localToWorld(entry_local, target);
                double attenuation = collimatorAttenuation(
                    source_world, entry_world, collimator, image[11]);
                for (int index = 0; index < detector_count; ++index)
                {
                    if (index == options.detector_index) continue;
                    const DetectorRecord other = detectorRecord(
                        detector, index, fov_to_detector);
                    if (!(other.mu_total > 0.0)) continue;
                    attenuation += segmentChord(source_world, entry_world, other)
                        * other.mu_total;
                }
                cached_survival = attenuation < 745.0 ? std::exp(-attenuation) : 0.0;
                return cached_survival;
            }, options.surface_rule == "halton");

        const double closure_error = result.first_interaction_probability
            - result.photoelectric_probability - result.compton_probability;
        bool has_v3 = !options.v3_matrix_path.empty();
        float v3_value = std::numeric_limits<float>::quiet_NaN();
        double v4_over_v3 = std::numeric_limits<double>::quiet_NaN();
        if (has_v3)
        {
            v3_value = readV3Element(
                options.v3_matrix_path,
                static_cast<std::size_t>(detector_count),
                static_cast<std::size_t>(voxel_count),
                options.rotation_index,
                options.detector_index,
                options.voxel_index);
            if (v3_value > 0.0f)
                v4_over_v3 = result.photoelectric_probability / v3_value;
        }

        if (options.output_path.empty())
        {
            std::ostringstream path;
            path << "PE_V4_Reference_detector_" << options.detector_index
                << "_voxel_" << options.voxel_index << ".csv";
            options.output_path = path.str();
        }
        std::ofstream output(options.output_path.c_str());
        if (!output) throw std::runtime_error("Cannot create " + options.output_path);
        output << "detector_index,voxel_index,rotation_index,surface_rule,face_subdivisions,"
            << "depth_subdivisions,source_x_mm,source_y_mm,source_z_mm,"
            << "detector_x_mm,detector_y_mm,detector_z_mm,detector_width_mm,"
            << "detector_thickness_mm,detector_height_mm,mu_total_per_mm,"
            << "mu_photoelectric_per_mm,mu_compton_per_mm,"
            << "attenuated_solid_angle_fraction,first_interaction_probability,"
            << "photoelectric_probability,compton_probability,closure_error,"
            << "mean_depth_mm,mean_position_local_x_mm,mean_position_local_y_mm,"
            << "mean_position_local_z_mm,pe_entry_x_minus,pe_entry_x_plus,"
            << "pe_entry_y_minus,pe_entry_y_plus,pe_entry_z_minus,pe_entry_z_plus,"
            << "state_count,v3_photoelectric_probability,"
            << "v4_over_v3\n";
        output << std::setprecision(17)
            << options.detector_index << ',' << options.voxel_index << ','
            << options.rotation_index << ',' << options.surface_rule << ','
            << options.face_subdivisions << ','
            << options.depth_subdivisions << ','
            << source_world.x << ',' << source_world.y << ',' << source_world.z << ','
            << target.x << ',' << target.y << ',' << target.z << ','
            << target.width << ',' << target.thickness << ',' << target.height << ','
            << target.mu_total << ',' << target.mu_photoelectric << ','
            << target.mu_compton << ','
            << result.attenuated_solid_angle_fraction << ','
            << result.first_interaction_probability << ','
            << result.photoelectric_probability << ','
            << result.compton_probability << ',' << closure_error << ','
            << result.mean_depth_mm << ',' << result.mean_position_x << ','
            << result.mean_position_y << ',' << result.mean_position_z << ',';
        for (int face = 0; face < 6; ++face)
            output << result.photoelectric_probability_by_entry_face[face] << ',';
        output
            << result.state_count << ',';
        if (has_v3) output << v3_value << ',' << v4_over_v3;
        else output << "nan,nan";
        output << '\n';

        std::cout << std::setprecision(12)
            << "PE v4 reference detector=" << options.detector_index
            << " voxel=" << options.voxel_index
            << " rotation=" << options.rotation_index
            << " surface_rule=" << options.surface_rule << '\n'
            << "photoelectric_probability=" << result.photoelectric_probability << '\n'
            << "compton_probability=" << result.compton_probability << '\n'
            << "first_interaction_probability="
            << result.first_interaction_probability << '\n'
            << "closure_error=" << closure_error << '\n'
            << "mean_depth_mm=" << result.mean_depth_mm << '\n';
        std::cout << "pe_by_entry_face=[";
        for (int face = 0; face < 6; ++face)
        {
            if (face > 0) std::cout << ',';
            std::cout << result.photoelectric_probability_by_entry_face[face];
        }
        std::cout << "]\n";
        if (has_v3)
            std::cout << "v3_photoelectric_probability=" << v3_value << '\n'
                << "v4_over_v3=" << v4_over_v3 << '\n';
        std::cout << "output=" << options.output_path << '\n';
        return EXIT_SUCCESS;
    }
    catch (const std::exception& error)
    {
        std::cerr << "PE v4 reference error: " << error.what() << std::endl;
        return EXIT_FAILURE;
    }
}
