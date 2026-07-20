#define _CRT_SECURE_NO_WARNINGS

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "../common/energy_window.h"

namespace fs = std::filesystem;

namespace
{
constexpr float kPi = 3.14159265358979323846f;

struct Options
{
    int cuda_id = 0;
    int face_subdivisions = 16;
    int rows_per_chunk = 4;
    int samples_per_launch = 32;
    int detector_start = 0;
    int detector_count = -1;
    bool resume = false;
    bool overwrite = false;
    std::string output_unwindowed;
    std::string output_windowed;
    std::string progress_path = "PE_v4_progress.json";
    std::string log_path = "PE_v4_progress.tsv";
    std::string manifest_path = "PE_v4_manifest.json";
};

struct DetectorGpu
{
    float center_x;
    float center_y;
    float center_z;
    float half_x;
    float half_y;
    float half_z;
    float cosine;
    float sine;
    float mu_total;
    float mu_photoelectric;
    int flag;
    int layer;
};

struct LayerGpu
{
    float center_y;
    float half_y;
    float maximum_aabb_half_x;
    float maximum_aabb_half_z;
};

struct CollimatorLayerGpu
{
    float half_x;
    float center_y;
    float half_y;
    float half_z;
    float mu_total;
};

struct ImageGpu
{
    int count_x;
    int count_y;
    int count_z;
    float width_x;
    float width_y;
    float width_z;
    float angle_per_rotation;
    float shift_x;
    float shift_y;
    float shift_z;
};

struct SpatialGrid
{
    float origin_x;
    float origin_z;
    float cell_size;
    int count_x;
    int count_z;
    std::vector<int> offsets;
    std::vector<int> detector_ids;
};

struct RunProgress
{
    std::string status;
    std::string message;
    long long completed_rows = 0;
    long long total_rows = 0;
    long long completed_elements = 0;
    long long total_elements = 0;
    double elapsed_seconds = 0.0;
    double elements_per_second = 0.0;
    double eta_seconds = 0.0;
    double unwindowed_sum = 0.0;
    double windowed_sum = 0.0;
    long long nonzero_elements = 0;
    int current_rotation = 0;
    int current_detector = 0;
};

void cudaCheck(cudaError_t error, const char* operation)
{
    if (error == cudaSuccess) return;
    std::ostringstream message;
    message << operation << ": " << cudaGetErrorString(error);
    throw std::runtime_error(message.str());
}

long long parseInteger(const char* text, const char* name)
{
    char* end = NULL;
    const long long value = std::strtoll(text, &end, 10);
    if (end == text || *end != '\0')
        throw std::runtime_error(std::string("Invalid ") + name + ": " + text);
    return value;
}

void printUsage(const char* executable)
{
    std::cout
        << "Usage: " << executable << " [options]\n"
        << "  --cuda ID                    CUDA device (default 0)\n"
        << "  --face-subdiv N              Halton samples per face are N*N (default 16)\n"
        << "  --rows-per-chunk N           Detector rows written per checkpoint (default 4)\n"
        << "  --samples-per-launch N       Surface samples per short CUDA launch (default 32)\n"
        << "  --detector-start N           First detector row, zero based (default 0)\n"
        << "  --detector-count N           Number of rows (default all remaining rows)\n"
        << "  --output-unwindowed PATH     Raw PE output (default *_v4.sysmat)\n"
        << "  --output-windowed PATH       Energy-windowed PE output\n"
        << "  --progress PATH              Atomic JSON progress file\n"
        << "  --log PATH                   Append-only TSV progress log\n"
        << "  --manifest PATH              Completed-run metadata JSON\n"
        << "  --resume                     Continue matching .partial files\n"
        << "  --overwrite                  Replace matching outputs and partial files\n";
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
        if (argument == "--resume")
        {
            options.resume = true;
            continue;
        }
        if (argument == "--overwrite")
        {
            options.overwrite = true;
            continue;
        }
        if (index + 1 >= argc)
            throw std::runtime_error("Missing value after " + argument);
        const char* value = argv[++index];
        if (argument == "--cuda")
            options.cuda_id = static_cast<int>(parseInteger(value, "cuda"));
        else if (argument == "--face-subdiv")
            options.face_subdivisions = static_cast<int>(parseInteger(value, "face-subdiv"));
        else if (argument == "--rows-per-chunk")
            options.rows_per_chunk = static_cast<int>(parseInteger(value, "rows-per-chunk"));
        else if (argument == "--samples-per-launch")
            options.samples_per_launch = static_cast<int>(parseInteger(value, "samples-per-launch"));
        else if (argument == "--detector-start")
            options.detector_start = static_cast<int>(parseInteger(value, "detector-start"));
        else if (argument == "--detector-count")
            options.detector_count = static_cast<int>(parseInteger(value, "detector-count"));
        else if (argument == "--output-unwindowed")
            options.output_unwindowed = value;
        else if (argument == "--output-windowed")
            options.output_windowed = value;
        else if (argument == "--progress")
            options.progress_path = value;
        else if (argument == "--log")
            options.log_path = value;
        else if (argument == "--manifest")
            options.manifest_path = value;
        else
            throw std::runtime_error("Unknown option: " + argument);
    }
    if (options.cuda_id < 0 || options.face_subdivisions < 1
        || options.rows_per_chunk < 1 || options.samples_per_launch < 1
        || options.detector_start < 0 || options.detector_count == 0
        || options.detector_count < -1)
        throw std::runtime_error("Numeric arguments are out of range");
    if (options.resume && options.overwrite)
        throw std::runtime_error("--resume and --overwrite are mutually exclusive");
    return options;
}

std::vector<float> readFloatFile(const fs::path& path)
{
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("Cannot open " + path.string());
    const std::streamoff bytes = stream.tellg();
    if (bytes < 0 || bytes % static_cast<std::streamoff>(sizeof(float)) != 0)
        throw std::runtime_error("Malformed float32 file: " + path.string());
    stream.seekg(0, std::ios::beg);
    std::vector<float> values(static_cast<std::size_t>(bytes / sizeof(float)));
    if (!values.empty())
        stream.read(reinterpret_cast<char*>(values.data()), bytes);
    if (!stream) throw std::runtime_error("Failed to read " + path.string());
    return values;
}

std::string jsonEscape(const std::string& value)
{
    std::ostringstream output;
    for (std::string::const_iterator character = value.begin(); character != value.end(); ++character)
    {
        if (*character == '\\' || *character == '"') output << '\\' << *character;
        else if (*character == '\n') output << "\\n";
        else if (*character == '\r') output << "\\r";
        else if (*character == '\t') output << "\\t";
        else output << *character;
    }
    return output.str();
}

std::string isoTimestamp()
{
    const std::time_t now = std::time(NULL);
    std::tm local = {};
#ifdef _WIN32
    localtime_s(&local, &now);
#else
    localtime_r(&now, &local);
#endif
    char buffer[64];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%S", &local);
    return buffer;
}

void writeProgress(const fs::path& path, const RunProgress& progress)
{
    const fs::path temporary = path.string() + ".tmp";
    std::ofstream output(temporary, std::ios::trunc);
    if (!output) throw std::runtime_error("Cannot write " + temporary.string());
    output << std::setprecision(17)
        << "{\n"
        << "  \"schema_version\": 1,\n"
        << "  \"model\": \"PE_v4_visible_surface_symmetric_halton_layer_grid\",\n"
        << "  \"status\": \"" << jsonEscape(progress.status) << "\",\n"
        << "  \"message\": \"" << jsonEscape(progress.message) << "\",\n"
        << "  \"last_update\": \"" << isoTimestamp() << "\",\n"
        << "  \"completed_rows\": " << progress.completed_rows << ",\n"
        << "  \"total_rows\": " << progress.total_rows << ",\n"
        << "  \"completed_elements\": " << progress.completed_elements << ",\n"
        << "  \"total_elements\": " << progress.total_elements << ",\n"
        << "  \"elapsed_seconds\": " << progress.elapsed_seconds << ",\n"
        << "  \"elements_per_second\": " << progress.elements_per_second << ",\n"
        << "  \"eta_seconds\": " << progress.eta_seconds << ",\n"
        << "  \"unwindowed_sum\": " << progress.unwindowed_sum << ",\n"
        << "  \"windowed_sum\": " << progress.windowed_sum << ",\n"
        << "  \"nonzero_elements\": " << progress.nonzero_elements << ",\n"
        << "  \"current_rotation\": " << progress.current_rotation << ",\n"
        << "  \"current_detector\": " << progress.current_detector << "\n"
        << "}\n";
    output.close();
    std::error_code last_error;
    for (int attempt = 0; attempt < 100; ++attempt)
    {
        last_error.clear();
        if (fs::exists(path, last_error))
        {
            last_error.clear();
            fs::remove(path, last_error);
            if (last_error)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                continue;
            }
        }
        last_error.clear();
        fs::rename(temporary, path, last_error);
        if (!last_error) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    throw std::runtime_error(
        "Cannot replace progress file after retries: " + last_error.message());
}

void removeIfExists(const fs::path& path)
{
    if (fs::exists(path)) fs::remove(path);
}

double radicalInverse(unsigned long long index, unsigned int base)
{
    const double inverse_base = 1.0 / base;
    double factor = inverse_base;
    double value = 0.0;
    while (index > 0)
    {
        value += factor * (index % base);
        index /= base;
        factor *= inverse_base;
    }
    return value;
}

int findLayer(const std::vector<LayerGpu>& layers, float center_y)
{
    for (std::size_t index = 0; index < layers.size(); ++index)
        if (std::fabs(layers[index].center_y - center_y) < 1e-3f)
            return static_cast<int>(index);
    return -1;
}

void buildGeometry(
    const std::vector<float>& detector_values,
    const std::vector<float>& image_values,
    std::vector<DetectorGpu>* detectors,
    std::vector<LayerGpu>* layers)
{
    const int detector_count = static_cast<int>(std::floor(detector_values[0] + 0.5f));
    const double fov_to_detector = image_values[11];
    std::vector<float> layer_centers;
    for (int index = 0; index < detector_count; ++index)
    {
        const std::size_t base = 1 + static_cast<std::size_t>(index) * 12;
        const float center_y = detector_values[base + 1] + static_cast<float>(fov_to_detector);
        bool found = false;
        for (std::size_t layer = 0; layer < layer_centers.size(); ++layer)
            if (std::fabs(layer_centers[layer] - center_y) < 1e-3f) found = true;
        if (!found) layer_centers.push_back(center_y);
    }
    std::sort(layer_centers.begin(), layer_centers.end());
    layers->resize(layer_centers.size());
    for (std::size_t layer = 0; layer < layer_centers.size(); ++layer)
    {
        (*layers)[layer].center_y = layer_centers[layer];
        (*layers)[layer].half_y = 0.0f;
        (*layers)[layer].maximum_aabb_half_x = 0.0f;
        (*layers)[layer].maximum_aabb_half_z = 0.0f;
    }

    detectors->resize(detector_count);
    for (int index = 0; index < detector_count; ++index)
    {
        const std::size_t base = 1 + static_cast<std::size_t>(index) * 12;
        DetectorGpu detector = {};
        detector.center_x = detector_values[base + 0];
        detector.center_y = detector_values[base + 1] + static_cast<float>(fov_to_detector);
        detector.center_z = detector_values[base + 2];
        detector.half_x = 0.5f * detector_values[base + 3];
        detector.half_y = 0.5f * detector_values[base + 4];
        detector.half_z = 0.5f * detector_values[base + 5];
        detector.mu_total = detector_values[base + 6];
        detector.mu_photoelectric = detector_values[base + 7];
        const float rotation = detector_values[base + 10];
        detector.cosine = std::cos(rotation);
        detector.sine = std::sin(rotation);
        detector.flag = static_cast<int>(std::floor(detector_values[base + 11] + 0.5f));
        detector.layer = findLayer(*layers, detector.center_y);
        if (detector.layer < 0) throw std::runtime_error("Cannot classify detector layer");
        const float aabb_half_x = std::fabs(detector.cosine) * detector.half_x
            + std::fabs(detector.sine) * detector.half_z;
        const float aabb_half_z = std::fabs(detector.sine) * detector.half_x
            + std::fabs(detector.cosine) * detector.half_z;
        LayerGpu& layer = (*layers)[detector.layer];
        layer.half_y = std::max(layer.half_y, detector.half_y);
        layer.maximum_aabb_half_x = std::max(layer.maximum_aabb_half_x, aabb_half_x);
        layer.maximum_aabb_half_z = std::max(layer.maximum_aabb_half_z, aabb_half_z);
        (*detectors)[index] = detector;
    }
}

SpatialGrid buildSpatialGrid(
    const std::vector<DetectorGpu>& detectors,
    const std::vector<LayerGpu>& layers)
{
    SpatialGrid grid = {};
    grid.cell_size = 4.2f;
    float minimum_x = std::numeric_limits<float>::infinity();
    float maximum_x = -std::numeric_limits<float>::infinity();
    float minimum_z = std::numeric_limits<float>::infinity();
    float maximum_z = -std::numeric_limits<float>::infinity();
    for (std::size_t index = 0; index < detectors.size(); ++index)
    {
        minimum_x = std::min(minimum_x, detectors[index].center_x);
        maximum_x = std::max(maximum_x, detectors[index].center_x);
        minimum_z = std::min(minimum_z, detectors[index].center_z);
        maximum_z = std::max(maximum_z, detectors[index].center_z);
    }
    grid.origin_x = minimum_x - grid.cell_size;
    grid.origin_z = minimum_z - grid.cell_size;
    grid.count_x = static_cast<int>(std::ceil(
        (maximum_x - grid.origin_x + grid.cell_size) / grid.cell_size)) + 1;
    grid.count_z = static_cast<int>(std::ceil(
        (maximum_z - grid.origin_z + grid.cell_size) / grid.cell_size)) + 1;
    const int cells_per_layer = grid.count_x * grid.count_z;
    std::vector<std::vector<int> > buckets(layers.size() * cells_per_layer);
    for (std::size_t index = 0; index < detectors.size(); ++index)
    {
        const DetectorGpu& detector = detectors[index];
        int cell_x = static_cast<int>(std::floor(
            (detector.center_x - grid.origin_x) / grid.cell_size));
        int cell_z = static_cast<int>(std::floor(
            (detector.center_z - grid.origin_z) / grid.cell_size));
        cell_x = std::max(0, std::min(grid.count_x - 1, cell_x));
        cell_z = std::max(0, std::min(grid.count_z - 1, cell_z));
        buckets[detector.layer * cells_per_layer + cell_x * grid.count_z + cell_z]
            .push_back(static_cast<int>(index));
    }
    grid.offsets.resize(buckets.size() + 1, 0);
    for (std::size_t cell = 0; cell < buckets.size(); ++cell)
    {
        grid.offsets[cell + 1] = grid.offsets[cell]
            + static_cast<int>(buckets[cell].size());
        grid.detector_ids.insert(
            grid.detector_ids.end(), buckets[cell].begin(), buckets[cell].end());
    }
    return grid;
}

std::vector<CollimatorLayerGpu> buildCollimator(
    const std::vector<float>& values,
    const std::vector<float>& image)
{
    const int count = static_cast<int>(std::floor(values[0] + 0.5f));
    std::vector<CollimatorLayerGpu> layers;
    for (int layer = 0; layer < count; ++layer)
    {
        const std::size_t base = static_cast<std::size_t>(layer + 1) * 10;
        const int hole_count = static_cast<int>(std::floor(values[base] + 0.5f));
        if (hole_count != 0)
            throw std::runtime_error(
                "PE v4 production currently supports zero-hole collimators only");
        CollimatorLayerGpu output = {};
        output.half_x = 0.5f * values[base + 1];
        output.half_y = 0.5f * values[base + 2];
        output.half_z = 0.5f * values[base + 3];
        output.center_y = image[11] + values[base + 4];
        output.mu_total = values[base + 5];
        layers.push_back(output);
    }
    return layers;
}

__device__ bool segmentBoxInterval(
    const float start[3],
    const float end[3],
    const float half_extent[3],
    float* lower,
    float* upper)
{
    float minimum = 0.0f;
    float maximum = 1.0f;
    for (int axis = 0; axis < 3; ++axis)
    {
        const float delta = end[axis] - start[axis];
        if (fabsf(delta) <= 1e-12f)
        {
            if (start[axis] < -half_extent[axis]
                || start[axis] > half_extent[axis]) return false;
            continue;
        }
        float first = (-half_extent[axis] - start[axis]) / delta;
        float second = (half_extent[axis] - start[axis]) / delta;
        if (first > second)
        {
            const float temporary = first;
            first = second;
            second = temporary;
        }
        minimum = fmaxf(minimum, first);
        maximum = fminf(maximum, second);
        if (minimum >= maximum) return false;
    }
    *lower = minimum;
    *upper = maximum;
    return maximum > minimum;
}

__device__ float detectorSegmentChord(
    const float start_world[3],
    const float end_world[3],
    const DetectorGpu& detector)
{
    float start[3];
    float end[3];
    const float start_dx = start_world[0] - detector.center_x;
    const float start_dz = start_world[2] - detector.center_z;
    const float end_dx = end_world[0] - detector.center_x;
    const float end_dz = end_world[2] - detector.center_z;
    start[0] = start_dx * detector.cosine + start_dz * detector.sine;
    start[1] = start_world[1] - detector.center_y;
    start[2] = -start_dx * detector.sine + start_dz * detector.cosine;
    end[0] = end_dx * detector.cosine + end_dz * detector.sine;
    end[1] = end_world[1] - detector.center_y;
    end[2] = -end_dx * detector.sine + end_dz * detector.cosine;
    const float half_extent[3] = {
        detector.half_x, detector.half_y, detector.half_z
    };
    float lower = 0.0f;
    float upper = 0.0f;
    if (!segmentBoxInterval(start, end, half_extent, &lower, &upper)) return 0.0f;
    const float dx = end_world[0] - start_world[0];
    const float dy = end_world[1] - start_world[1];
    const float dz = end_world[2] - start_world[2];
    return (upper - lower) * sqrtf(dx * dx + dy * dy + dz * dz);
}

__device__ float axisAlignedSegmentChord(
    const float start[3],
    const float end[3],
    float center_y,
    float half_x,
    float half_y,
    float half_z)
{
    const float local_start[3] = {start[0], start[1] - center_y, start[2]};
    const float local_end[3] = {end[0], end[1] - center_y, end[2]};
    const float half_extent[3] = {half_x, half_y, half_z};
    float lower = 0.0f;
    float upper = 0.0f;
    if (!segmentBoxInterval(local_start, local_end, half_extent, &lower, &upper))
        return 0.0f;
    const float dx = end[0] - start[0];
    const float dy = end[1] - start[1];
    const float dz = end[2] - start[2];
    return (upper - lower) * sqrtf(dx * dx + dy * dy + dz * dz);
}

__device__ float targetExitDistance(
    const float position[3],
    const float direction[3],
    const float half_extent[3])
{
    float distance = FLT_MAX;
    for (int axis = 0; axis < 3; ++axis)
    {
        if (fabsf(position[axis]) > half_extent[axis] + 2e-4f) return 0.0f;
        if (direction[axis] > 1e-12f)
            distance = fminf(distance,
                (half_extent[axis] - position[axis]) / direction[axis]);
        else if (direction[axis] < -1e-12f)
            distance = fminf(distance,
                (-half_extent[axis] - position[axis]) / direction[axis]);
    }
    return distance < FLT_MAX && distance > 0.0f ? distance : 0.0f;
}

__device__ float detectorAttenuation(
    const float source[3],
    const float entry[3],
    int target_id,
    const DetectorGpu* detectors,
    const LayerGpu* layers,
    int layer_count,
    const int* grid_offsets,
    const int* grid_detector_ids,
    float grid_origin_x,
    float grid_origin_z,
    float grid_cell_size,
    int grid_count_x,
    int grid_count_z)
{
    const float delta_y = entry[1] - source[1];
    float attenuation = 0.0f;
    const int cells_per_layer = grid_count_x * grid_count_z;
    for (int layer_index = 0; layer_index < layer_count; ++layer_index)
    {
        const LayerGpu layer = layers[layer_index];
        const float lower_y = layer.center_y - layer.half_y;
        const float upper_y = layer.center_y + layer.half_y;
        float lower_t = 0.0f;
        float upper_t = 1.0f;
        if (fabsf(delta_y) <= 1e-12f)
        {
            if (source[1] < lower_y || source[1] > upper_y) continue;
        }
        else
        {
            float first = (lower_y - source[1]) / delta_y;
            float second = (upper_y - source[1]) / delta_y;
            if (first > second)
            {
                const float temporary = first;
                first = second;
                second = temporary;
            }
            lower_t = fmaxf(0.0f, first);
            upper_t = fminf(1.0f, second);
            if (lower_t > upper_t || upper_t < 0.0f || lower_t > 1.0f) continue;
        }

        const float x_first = source[0] + lower_t * (entry[0] - source[0]);
        const float x_second = source[0] + upper_t * (entry[0] - source[0]);
        const float z_first = source[2] + lower_t * (entry[2] - source[2]);
        const float z_second = source[2] + upper_t * (entry[2] - source[2]);
        int minimum_x = static_cast<int>(floorf(
            (fminf(x_first, x_second) - layer.maximum_aabb_half_x
                - grid_origin_x) / grid_cell_size));
        int maximum_x = static_cast<int>(floorf(
            (fmaxf(x_first, x_second) + layer.maximum_aabb_half_x
                - grid_origin_x) / grid_cell_size));
        int minimum_z = static_cast<int>(floorf(
            (fminf(z_first, z_second) - layer.maximum_aabb_half_z
                - grid_origin_z) / grid_cell_size));
        int maximum_z = static_cast<int>(floorf(
            (fmaxf(z_first, z_second) + layer.maximum_aabb_half_z
                - grid_origin_z) / grid_cell_size));
        minimum_x = max(0, minimum_x);
        maximum_x = min(grid_count_x - 1, maximum_x);
        minimum_z = max(0, minimum_z);
        maximum_z = min(grid_count_z - 1, maximum_z);
        if (minimum_x > maximum_x || minimum_z > maximum_z) continue;

        for (int cell_x = minimum_x; cell_x <= maximum_x; ++cell_x)
        {
            for (int cell_z = minimum_z; cell_z <= maximum_z; ++cell_z)
            {
                const int cell = layer_index * cells_per_layer
                    + cell_x * grid_count_z + cell_z;
                for (int offset = grid_offsets[cell];
                    offset < grid_offsets[cell + 1]; ++offset)
                {
                    const int detector_id = grid_detector_ids[offset];
                    if (detector_id == target_id) continue;
                    const DetectorGpu other = detectors[detector_id];
                    if (!(other.mu_total > 0.0f)) continue;
                    const float chord = detectorSegmentChord(source, entry, other);
                    attenuation += chord * other.mu_total;
                }
            }
        }
    }
    return attenuation;
}

__global__ void peV4SurfaceKernel(
    float* output,
    int output_rows,
    int detector_start,
    int voxel_count,
    int rotation_index,
    ImageGpu image,
    const DetectorGpu* detectors,
    const LayerGpu* layers,
    int layer_count,
    const int* grid_offsets,
    const int* grid_detector_ids,
    float grid_origin_x,
    float grid_origin_z,
    float grid_cell_size,
    int grid_count_x,
    int grid_count_z,
    const CollimatorLayerGpu* collimators,
    int collimator_count,
    const float* sample_u,
    const float* sample_v,
    int sample_start,
    int sample_stop,
    int total_samples)
{
    const long long pair = static_cast<long long>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    const long long pair_count = static_cast<long long>(output_rows) * voxel_count;
    if (pair >= pair_count) return;
    const int local_row = static_cast<int>(pair / voxel_count);
    const int voxel = static_cast<int>(pair % voxel_count);
    const int detector_id = detector_start + local_row;
    const DetectorGpu target = detectors[detector_id];
    // Flag 2 rows are tungsten blocks. They are not exported as detector bins,
    // but ScatterGen needs their first-interaction matrix as a scatter source.
    if (target.flag <= 0 || !(target.mu_total > 0.0f)
        || !(target.mu_photoelectric > 0.0f)) return;

    const int in_slice = voxel % (image.count_x * image.count_y);
    const int index_z = voxel / (image.count_x * image.count_y);
    const int index_y = in_slice / image.count_x;
    const int index_x = in_slice % image.count_x;
    const float unrotated_x = (index_x - image.count_x / 2.0f + 0.5f)
        * image.width_x + image.shift_x;
    const float unrotated_y = (index_y - image.count_y / 2.0f + 0.5f)
        * image.width_y + image.shift_y;
    const float rotation = rotation_index * image.angle_per_rotation;
    const float cosine_rotation = cosf(rotation);
    const float sine_rotation = sinf(rotation);
    const float source[3] = {
        unrotated_x * cosine_rotation - unrotated_y * sine_rotation,
        unrotated_x * sine_rotation + unrotated_y * cosine_rotation,
        (index_z - image.count_z / 2.0f + 0.5f)
            * image.width_z + image.shift_z
    };

    const float source_dx = source[0] - target.center_x;
    const float source_dz = source[2] - target.center_z;
    const float source_local[3] = {
        source_dx * target.cosine + source_dz * target.sine,
        source[1] - target.center_y,
        -source_dx * target.sine + source_dz * target.cosine
    };
    const float half_extent[3] = {target.half_x, target.half_y, target.half_z};
    const float axis_length[3] = {
        2.0f * target.half_x, 2.0f * target.half_y, 2.0f * target.half_z
    };
    float contribution = 0.0f;

    for (int normal_axis = 0; normal_axis < 3; ++normal_axis)
    {
        float face_sign = 0.0f;
        if (source_local[normal_axis] < -half_extent[normal_axis]) face_sign = -1.0f;
        else if (source_local[normal_axis] > half_extent[normal_axis]) face_sign = 1.0f;
        else continue;
        const int first_axis = (normal_axis + 1) % 3;
        const int second_axis = (normal_axis + 2) % 3;
        const float area_per_sample = axis_length[first_axis]
            * axis_length[second_axis] / total_samples;

        for (int sample = sample_start; sample < sample_stop; ++sample)
        {
            float entry_local[3] = {0.0f, 0.0f, 0.0f};
            entry_local[normal_axis] = face_sign * half_extent[normal_axis];
            entry_local[first_axis] = axis_length[first_axis]
                * (sample_u[sample] - 0.5f);
            entry_local[second_axis] = axis_length[second_axis]
                * (sample_v[sample] - 0.5f);
            float incoming[3] = {
                entry_local[0] - source_local[0],
                entry_local[1] - source_local[1],
                entry_local[2] - source_local[2]
            };
            const float distance_squared = incoming[0] * incoming[0]
                + incoming[1] * incoming[1] + incoming[2] * incoming[2];
            if (!(distance_squared > 0.0f)) continue;
            const float inverse_distance = rsqrtf(distance_squared);
            incoming[0] *= inverse_distance;
            incoming[1] *= inverse_distance;
            incoming[2] *= inverse_distance;
            const float projected_cosine = -face_sign * incoming[normal_axis];
            if (!(projected_cosine > 0.0f)) continue;
            const float chord = targetExitDistance(
                entry_local, incoming, half_extent);
            if (!(chord > 0.0f)) continue;

            const float entry_world[3] = {
                target.center_x + entry_local[0] * target.cosine
                    - entry_local[2] * target.sine,
                target.center_y + entry_local[1],
                target.center_z + entry_local[0] * target.sine
                    + entry_local[2] * target.cosine
            };
            float attenuation = detectorAttenuation(
                source, entry_world, detector_id,
                detectors, layers, layer_count,
                grid_offsets, grid_detector_ids,
                grid_origin_x, grid_origin_z, grid_cell_size,
                grid_count_x, grid_count_z);
            for (int layer = 0; layer < collimator_count; ++layer)
            {
                const CollimatorLayerGpu collimator = collimators[layer];
                if (!(collimator.mu_total > 0.0f)) continue;
                attenuation += collimator.mu_total * axisAlignedSegmentChord(
                    source, entry_world, collimator.center_y,
                    collimator.half_x, collimator.half_y, collimator.half_z);
            }
            if (attenuation >= 80.0f) continue;
            const float solid_angle = projected_cosine * area_per_sample
                / (4.0f * kPi * distance_squared);
            const float first_interaction = -expm1f(-target.mu_total * chord);
            contribution += expf(-attenuation) * solid_angle
                * first_interaction * target.mu_photoelectric / target.mu_total;
        }
    }
    output[pair] += contribution;
}

template <typename T>
void allocateAndCopy(T** device, const std::vector<T>& host, const char* name)
{
    const std::size_t bytes = host.size() * sizeof(T);
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(device), bytes), name);
    if (bytes > 0)
        cudaCheck(cudaMemcpy(*device, host.data(), bytes, cudaMemcpyHostToDevice), name);
}

void writeManifest(
    const fs::path& path,
    const Options& options,
    int detector_total,
    int detector_count,
    int voxel_count,
    int rotation_count,
    const SpatialGrid& grid,
    const RunProgress& progress,
    const fs::path& unwindowed,
    const fs::path& windowed)
{
    std::ofstream output(path, std::ios::trunc);
    if (!output) throw std::runtime_error("Cannot write " + path.string());
    output << std::setprecision(17)
        << "{\n"
        << "  \"format_version\": 1,\n"
        << "  \"model\": \"PE_v4_visible_surface_symmetric_halton_layer_grid\",\n"
        << "  \"intrinsic_response_applied\": false,\n"
        << "  \"detector_total\": " << detector_total << ",\n"
        << "  \"detector_start\": " << options.detector_start << ",\n"
        << "  \"detector_count\": " << detector_count << ",\n"
        << "  \"voxel_count\": " << voxel_count << ",\n"
        << "  \"rotation_count\": " << rotation_count << ",\n"
        << "  \"face_subdivisions\": " << options.face_subdivisions << ",\n"
        << "  \"samples_per_visible_face\": "
        << options.face_subdivisions * options.face_subdivisions << ",\n"
        << "  \"samples_per_cuda_launch\": " << options.samples_per_launch << ",\n"
        << "  \"rows_per_chunk\": " << options.rows_per_chunk << ",\n"
        << "  \"grid_cell_size_mm\": " << grid.cell_size << ",\n"
        << "  \"grid_count_x\": " << grid.count_x << ",\n"
        << "  \"grid_count_z\": " << grid.count_z << ",\n"
        << "  \"unwindowed_file\": \"" << jsonEscape(unwindowed.string()) << "\",\n"
        << "  \"windowed_file\": \"" << jsonEscape(windowed.string()) << "\",\n"
        << "  \"unwindowed_sum\": " << progress.unwindowed_sum << ",\n"
        << "  \"windowed_sum\": " << progress.windowed_sum << ",\n"
        << "  \"nonzero_elements\": " << progress.nonzero_elements << ",\n"
        << "  \"elapsed_seconds\": " << progress.elapsed_seconds << ",\n"
        << "  \"completed_at\": \"" << isoTimestamp() << "\"\n"
        << "}\n";
}

void recoverPartialStatistics(
    const fs::path& unwindowed,
    const fs::path& windowed,
    RunProgress* progress)
{
    std::ifstream raw(unwindowed, std::ios::binary);
    std::ifstream accepted(windowed, std::ios::binary);
    if (!raw || !accepted)
        throw std::runtime_error("Cannot read partial files while resuming");
    const std::size_t chunk_elements = 1 << 20;
    std::vector<float> raw_chunk(chunk_elements);
    std::vector<float> accepted_chunk(chunk_elements);
    while (raw)
    {
        raw.read(reinterpret_cast<char*>(raw_chunk.data()),
            chunk_elements * sizeof(float));
        const std::streamsize raw_bytes = raw.gcount();
        if (raw_bytes == 0) break;
        accepted.read(reinterpret_cast<char*>(accepted_chunk.data()), raw_bytes);
        if (accepted.gcount() != raw_bytes
            || raw_bytes % static_cast<std::streamsize>(sizeof(float)) != 0)
            throw std::runtime_error("Partial files changed while resuming");
        const std::size_t count = static_cast<std::size_t>(raw_bytes) / sizeof(float);
        for (std::size_t index = 0; index < count; ++index)
        {
            const float raw_value = raw_chunk[index];
            const float accepted_value = accepted_chunk[index];
            if (!std::isfinite(raw_value) || raw_value < 0.0f
                || !std::isfinite(accepted_value) || accepted_value < 0.0f)
                throw std::runtime_error("Partial files contain invalid values");
            progress->unwindowed_sum += raw_value;
            progress->windowed_sum += accepted_value;
            if (raw_value > 0.0f) ++progress->nonzero_elements;
        }
    }
}
}

int main(int argc, char** argv)
{
    Options options;
    RunProgress progress;
    fs::path progress_path;
    try
    {
        options = parseOptions(argc, argv);
        progress_path = options.progress_path;
        std::vector<float> collimator_values = readFloatFile("Params_Collimator.dat");
        std::vector<float> detector_values = readFloatFile("Params_Detector.dat");
        std::vector<float> image_values = readFloatFile("Params_Image.dat");
        std::vector<float> physics_values = readFloatFile("Params_Physics.dat");
        if (collimator_values.empty() || detector_values.empty()
            || image_values.size() < 12 || physics_values.size() < 12)
            throw std::runtime_error("Parameter files are empty or incomplete");
        image_values.resize(std::max<std::size_t>(image_values.size(), 100), 0.0f);
        physics_values.resize(std::max<std::size_t>(physics_values.size(), 100), 0.0f);

        const int detector_total = static_cast<int>(std::floor(detector_values[0] + 0.5f));
        const int voxel_count = static_cast<int>(std::floor(image_values[0] + 0.5f))
            * static_cast<int>(std::floor(image_values[1] + 0.5f))
            * static_cast<int>(std::floor(image_values[2] + 0.5f));
        const int rotation_count = static_cast<int>(std::floor(image_values[6] + 0.5f));
        if (options.detector_start >= detector_total)
            throw std::runtime_error("--detector-start exceeds detector count");
        const int detector_count = options.detector_count < 0
            ? detector_total - options.detector_start
            : options.detector_count;
        if (options.detector_start + detector_count > detector_total)
            throw std::runtime_error("Requested detector range exceeds detector count");
        if (voxel_count <= 0 || rotation_count <= 0)
            throw std::runtime_error("Invalid image dimensions");

        char default_unwindowed[256];
        char default_windowed[256];
        std::snprintf(default_unwindowed, sizeof(default_unwindowed),
            "PE_SysMat_shift_%f_%f_%f_v4.sysmat",
            image_values[8], image_values[9], image_values[10]);
        std::snprintf(default_windowed, sizeof(default_windowed),
            "PE_Windowed_SysMat_shift_%f_%f_%f_v4.sysmat",
            image_values[8], image_values[9], image_values[10]);
        const fs::path unwindowed = options.output_unwindowed.empty()
            ? fs::path(default_unwindowed) : fs::path(options.output_unwindowed);
        const fs::path windowed = options.output_windowed.empty()
            ? fs::path(default_windowed) : fs::path(options.output_windowed);
        const fs::path partial_unwindowed = unwindowed.string() + ".partial";
        const fs::path partial_windowed = windowed.string() + ".partial";
        const fs::path log_path = options.log_path;
        const fs::path manifest_path = options.manifest_path;

        if (options.overwrite)
        {
            removeIfExists(unwindowed);
            removeIfExists(windowed);
            removeIfExists(partial_unwindowed);
            removeIfExists(partial_windowed);
            removeIfExists(progress_path);
            removeIfExists(log_path);
            removeIfExists(manifest_path);
        }
        if (!options.resume && (fs::exists(unwindowed) || fs::exists(windowed)
            || fs::exists(partial_unwindowed) || fs::exists(partial_windowed)))
            throw std::runtime_error(
                "Output or partial files already exist; use --resume or --overwrite");

        std::vector<DetectorGpu> detectors;
        std::vector<LayerGpu> layers;
        buildGeometry(detector_values, image_values, &detectors, &layers);
        const SpatialGrid grid = buildSpatialGrid(detectors, layers);
        const std::vector<CollimatorLayerGpu> collimators = buildCollimator(
            collimator_values, image_values);

        ImageGpu image = {};
        image.count_x = static_cast<int>(std::floor(image_values[0] + 0.5f));
        image.count_y = static_cast<int>(std::floor(image_values[1] + 0.5f));
        image.count_z = static_cast<int>(std::floor(image_values[2] + 0.5f));
        image.width_x = image_values[3];
        image.width_y = image_values[4];
        image.width_z = image_values[5];
        image.angle_per_rotation = image_values[7];
        image.shift_x = image_values[8];
        image.shift_y = image_values[9];
        image.shift_z = image_values[10];

        const int total_surface_samples = options.face_subdivisions
            * options.face_subdivisions;
        std::vector<float> sample_u;
        std::vector<float> sample_v;
        sample_u.reserve(total_surface_samples);
        sample_v.reserve(total_surface_samples);
        const int symmetric_groups = total_surface_samples / 4;
        for (int group = 0; group < symmetric_groups; ++group)
        {
            const float u = static_cast<float>(radicalInverse(group + 1, 2));
            const float v = static_cast<float>(radicalInverse(group + 1, 3));
            sample_u.push_back(u);
            sample_v.push_back(v);
            sample_u.push_back(1.0f - u);
            sample_v.push_back(v);
            sample_u.push_back(u);
            sample_v.push_back(1.0f - v);
            sample_u.push_back(1.0f - u);
            sample_v.push_back(1.0f - v);
        }
        if (total_surface_samples % 4 == 1)
        {
            sample_u.push_back(0.5f);
            sample_v.push_back(0.5f);
        }
        if (static_cast<int>(sample_u.size()) != total_surface_samples)
            throw std::runtime_error("Internal error constructing symmetric Halton samples");

        int device_count = 0;
        cudaCheck(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        if (options.cuda_id >= device_count)
            throw std::runtime_error("Requested CUDA device does not exist");
        cudaCheck(cudaSetDevice(options.cuda_id), "cudaSetDevice");
        cudaDeviceProp device_properties = {};
        cudaCheck(cudaGetDeviceProperties(&device_properties, options.cuda_id),
            "cudaGetDeviceProperties");

        DetectorGpu* device_detectors = NULL;
        LayerGpu* device_layers = NULL;
        int* device_grid_offsets = NULL;
        int* device_grid_ids = NULL;
        CollimatorLayerGpu* device_collimators = NULL;
        float* device_sample_u = NULL;
        float* device_sample_v = NULL;
        float* device_output = NULL;
        allocateAndCopy(&device_detectors, detectors, "detectors");
        allocateAndCopy(&device_layers, layers, "layers");
        allocateAndCopy(&device_grid_offsets, grid.offsets, "grid offsets");
        allocateAndCopy(&device_grid_ids, grid.detector_ids, "grid ids");
        allocateAndCopy(&device_collimators, collimators, "collimators");
        allocateAndCopy(&device_sample_u, sample_u, "sample u");
        allocateAndCopy(&device_sample_v, sample_v, "sample v");
        const std::size_t maximum_chunk_elements = static_cast<std::size_t>(
            options.rows_per_chunk) * voxel_count;
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&device_output),
            maximum_chunk_elements * sizeof(float)), "chunk output");

        const long long row_bytes = static_cast<long long>(voxel_count) * sizeof(float);
        long long completed_rows = 0;
        if (options.resume)
        {
            if (!fs::exists(partial_unwindowed) || !fs::exists(partial_windowed))
                throw std::runtime_error("Both .partial files are required for --resume");
            const long long first_bytes = static_cast<long long>(fs::file_size(partial_unwindowed));
            const long long second_bytes = static_cast<long long>(fs::file_size(partial_windowed));
            if (first_bytes != second_bytes || first_bytes % row_bytes != 0)
                throw std::runtime_error("Partial files are inconsistent or not row aligned");
            completed_rows = first_bytes / row_bytes;
            recoverPartialStatistics(
                partial_unwindowed, partial_windowed, &progress);
        }
        const long long total_rows = static_cast<long long>(rotation_count) * detector_count;
        if (completed_rows > total_rows)
            throw std::runtime_error("Partial files exceed requested output shape");

        std::ofstream raw_output(partial_unwindowed,
            std::ios::binary | std::ios::app);
        std::ofstream accepted_output(partial_windowed,
            std::ios::binary | std::ios::app);
        if (!raw_output || !accepted_output)
            throw std::runtime_error("Cannot open partial output files");
        std::ofstream progress_log(log_path, std::ios::app);
        if (!progress_log) throw std::runtime_error("Cannot open progress log");
        if (fs::file_size(log_path) == 0)
            progress_log << "timestamp\tstatus\tcompleted_rows\ttotal_rows"
                "\telapsed_seconds\telements_per_second\teta_seconds"
                "\tcurrent_rotation\tcurrent_detector\tunwindowed_sum"
                "\twindowed_sum\tnonzero_elements\n";

        std::vector<float> host_output(maximum_chunk_elements);
        std::vector<float> host_windowed(maximum_chunk_elements);
        std::vector<float> acceptance(detector_total, 0.0f);
        for (int row = 0; row < detector_total; ++row)
            acceptance[row] = photopeak_energy_window_acceptance(
                physics_values.data(), detector_values.data() + 1 + row * 12);

        progress.status = "running";
        progress.message = "PE v4 production matrix generation is running";
        progress.completed_rows = completed_rows;
        progress.total_rows = total_rows;
        progress.completed_elements = completed_rows * voxel_count;
        progress.total_elements = total_rows * voxel_count;
        writeProgress(progress_path, progress);

        std::cout << "PE v4 production model\n"
            << "CUDA device: " << device_properties.name << '\n'
            << "Detector range: " << options.detector_start << " .. "
            << options.detector_start + detector_count - 1 << '\n'
            << "Rotations: " << rotation_count << ", voxels: " << voxel_count << '\n'
            << "Surface samples per visible face: " << total_surface_samples << '\n'
            << "Layer grid: " << layers.size() << " layers, "
            << grid.count_x << " x " << grid.count_z << " cells\n"
            << "Resume row: " << completed_rows << " / " << total_rows << std::endl;

        const std::chrono::steady_clock::time_point start_time
            = std::chrono::steady_clock::now();
        const long long initial_completed_rows = completed_rows;
        while (completed_rows < total_rows)
        {
            const int rotation_index = static_cast<int>(completed_rows / detector_count);
            const int relative_row = static_cast<int>(completed_rows % detector_count);
            const int chunk_rows = std::min(options.rows_per_chunk,
                detector_count - relative_row);
            const int detector_start = options.detector_start + relative_row;
            const std::size_t chunk_elements = static_cast<std::size_t>(chunk_rows)
                * voxel_count;
            cudaCheck(cudaMemset(device_output, 0,
                chunk_elements * sizeof(float)), "clear chunk output");

            const int threads = 128;
            const int blocks = static_cast<int>((chunk_elements + threads - 1) / threads);
            for (int sample_start = 0; sample_start < total_surface_samples;
                sample_start += options.samples_per_launch)
            {
                const int sample_stop = std::min(total_surface_samples,
                    sample_start + options.samples_per_launch);
                peV4SurfaceKernel<<<blocks, threads>>>(
                    device_output, chunk_rows, detector_start, voxel_count,
                    rotation_index, image,
                    device_detectors, device_layers, static_cast<int>(layers.size()),
                    device_grid_offsets, device_grid_ids,
                    grid.origin_x, grid.origin_z, grid.cell_size,
                    grid.count_x, grid.count_z,
                    device_collimators, static_cast<int>(collimators.size()),
                    device_sample_u, device_sample_v,
                    sample_start, sample_stop, total_surface_samples);
                cudaCheck(cudaGetLastError(), "PE v4 kernel launch");
                cudaCheck(cudaDeviceSynchronize(), "PE v4 kernel execution");
            }
            cudaCheck(cudaMemcpy(host_output.data(), device_output,
                chunk_elements * sizeof(float), cudaMemcpyDeviceToHost),
                "copy PE v4 chunk");

            for (int local_row = 0; local_row < chunk_rows; ++local_row)
            {
                const int row = detector_start + local_row;
                const float row_acceptance = acceptance[row];
                const std::size_t row_offset = static_cast<std::size_t>(local_row)
                    * voxel_count;
                for (int voxel = 0; voxel < voxel_count; ++voxel)
                {
                    const std::size_t index = row_offset + voxel;
                    const float value = host_output[index];
                    if (!std::isfinite(value) || value < 0.0f)
                        throw std::runtime_error("GPU produced an invalid PE value");
                    host_windowed[index] = value * row_acceptance;
                    progress.unwindowed_sum += value;
                    progress.windowed_sum += host_windowed[index];
                    if (value > 0.0f) ++progress.nonzero_elements;
                }
            }
            raw_output.write(reinterpret_cast<const char*>(host_output.data()),
                chunk_elements * sizeof(float));
            accepted_output.write(reinterpret_cast<const char*>(host_windowed.data()),
                chunk_elements * sizeof(float));
            raw_output.flush();
            accepted_output.flush();
            if (!raw_output || !accepted_output)
                throw std::runtime_error("Failed while writing PE v4 output");

            completed_rows += chunk_rows;
            const double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start_time).count();
            const long long session_rows = completed_rows - initial_completed_rows;
            const double rate = elapsed > 0.0
                ? session_rows * static_cast<double>(voxel_count) / elapsed : 0.0;
            progress.completed_rows = completed_rows;
            progress.completed_elements = completed_rows * voxel_count;
            progress.elapsed_seconds = elapsed;
            progress.elements_per_second = rate;
            progress.eta_seconds = rate > 0.0
                ? (progress.total_elements - progress.completed_elements) / rate : 0.0;
            progress.current_rotation = rotation_index;
            progress.current_detector = detector_start + chunk_rows - 1;
            writeProgress(progress_path, progress);
            progress_log << isoTimestamp() << '\t' << progress.status << '\t'
                << progress.completed_rows << '\t' << progress.total_rows << '\t'
                << progress.elapsed_seconds << '\t' << progress.elements_per_second
                << '\t' << progress.eta_seconds << '\t' << progress.current_rotation
                << '\t' << progress.current_detector << '\t'
                << std::setprecision(17) << progress.unwindowed_sum << '\t'
                << progress.windowed_sum << '\t' << progress.nonzero_elements << '\n';
            progress_log.flush();
            std::cout << "Rows " << progress.completed_rows << '/' << progress.total_rows
                << "  " << std::fixed << std::setprecision(1)
                << progress.elements_per_second / 1e6 << " M elements/s"
                << "  ETA " << progress.eta_seconds / 60.0 << " min" << std::endl;
        }

        raw_output.close();
        accepted_output.close();
        removeIfExists(unwindowed);
        removeIfExists(windowed);
        fs::rename(partial_unwindowed, unwindowed);
        fs::rename(partial_windowed, windowed);
        progress.status = "complete";
        progress.message = "PE v4 production matrices completed";
        progress.eta_seconds = 0.0;
        writeProgress(progress_path, progress);
        writeManifest(manifest_path, options, detector_total, detector_count,
            voxel_count, rotation_count, grid, progress, unwindowed, windowed);
        progress_log << isoTimestamp() << "\tcomplete\t" << progress.completed_rows
            << '\t' << progress.total_rows << '\t' << progress.elapsed_seconds << '\t'
            << progress.elements_per_second << "\t0\t" << progress.current_rotation
            << '\t' << progress.current_detector << '\t' << progress.unwindowed_sum
            << '\t' << progress.windowed_sum << '\t' << progress.nonzero_elements << '\n';

        cudaFree(device_output);
        cudaFree(device_sample_v);
        cudaFree(device_sample_u);
        cudaFree(device_collimators);
        cudaFree(device_grid_ids);
        cudaFree(device_grid_offsets);
        cudaFree(device_layers);
        cudaFree(device_detectors);
        std::cout << "PE v4 matrices completed: " << unwindowed << " and "
            << windowed << std::endl;
        return EXIT_SUCCESS;
    }
    catch (const std::exception& exception)
    {
        std::cerr << "PE v4 production failed: " << exception.what() << std::endl;
        if (!progress_path.empty())
        {
            try
            {
                progress.status = "failed";
                progress.message = exception.what();
                writeProgress(progress_path, progress);
            }
            catch (...) {}
        }
        return EXIT_FAILURE;
    }
}
