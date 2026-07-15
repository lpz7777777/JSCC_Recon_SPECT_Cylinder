// Generate System Matrix on GPU with primary compton scatter
// author: xingchun zheng @ tsinghua university
// last modified: 2024/12/21
// version: 1.0

#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include<cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <cstdint>
#include <vector>
#include <chrono> 
#include <limits>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include "scatter.h"
#include "../common/detector_local_scatter.h"
#include "../physics_data/nist_xcom_materials_1_1000keV.h"
using namespace std;


#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

#define max(a,b) ((a>=b)?a:b) 
#define min(a,b) ((a<=b)?a:b)

__constant__ float deviceXcomMuPhotoelectric[kXcomMaterialCount * kXcomEnergyCount];
__constant__ float deviceXcomMuCompton[kXcomMaterialCount * kXcomEnergyCount];
__device__ float deviceComptonNormalization;

static constexpr float kComptonIntegralStep = 0.01f;
static constexpr float kComptonPhaseStep = 0.00001f;
static constexpr int kComptonPhasePrefixCount = 315162;

enum CrystalPairFlags
{
	kCrystalPairKinematicallyAllowed = 1U
};

struct CrystalPairPath
{
	float4 material_lengths;
	float4 direction_distance;
	float source_exit_length;
	float target_absorption_length;
	unsigned int flags;
};

struct AxisAlignedLayerGrid
{
	float y_min;
	float y_max;
	float x_boundary_min;
	float z_boundary_min;
	float pitch_x;
	float pitch_z;
	int count_x;
	int count_z;
	int map_offset;
};

static vector<float> sortedUniqueCoordinates(vector<float> values)
{
	sort(values.begin(), values.end());
	vector<float> unique;
	for (float value : values)
	{
		if (unique.empty() || fabsf(value - unique.back()) > 1e-4f)
			unique.push_back(value);
	}
	return unique;
}

static bool inferUniformCoordinateGrid(
	const vector<float>& coordinates,
	float box_size,
	float* minimum_center,
	float* pitch,
	int* count)
{
	if (coordinates.empty()) return false;
	*minimum_center = coordinates.front();
	if (coordinates.size() == 1)
	{
		*pitch = box_size;
		*count = 1;
		return box_size > 0.0f;
	}
	float minimum_difference = numeric_limits<float>::infinity();
	for (size_t index = 1; index < coordinates.size(); ++index)
	{
		float difference = coordinates[index] - coordinates[index - 1];
		if (difference > 1e-4f && difference < minimum_difference)
			minimum_difference = difference;
	}
	if (!isfinite(minimum_difference) || !(minimum_difference > 0.0f)) return false;
	*pitch = minimum_difference;
	*count = static_cast<int>(floorf(
		(coordinates.back() - coordinates.front()) / *pitch + 0.5f)) + 1;
	if (box_size > *pitch + 1e-4f) return false;
	for (float coordinate : coordinates)
	{
		float position = (coordinate - *minimum_center) / *pitch;
		if (fabsf(position - floorf(position + 0.5f)) > 1e-3f) return false;
	}
	return true;
}

static bool buildAxisAlignedLayerGrids(
	const float* detector,
	int detector_count,
	vector<AxisAlignedLayerGrid>* layers,
	vector<int>* cell_to_detector)
{
	layers->clear();
	cell_to_detector->clear();
	vector<float> all_y;
	all_y.reserve(detector_count);
	for (int index = 0; index < detector_count; ++index)
	{
		if (fabsf(detector[index * 12 + 11]) > 1e-7f) return false;
		all_y.push_back(detector[index * 12 + 2]);
	}
	vector<float> layer_centers = sortedUniqueCoordinates(all_y);
	if (layer_centers.empty()) return false;

	for (float layer_center : layer_centers)
	{
		vector<int> indices;
		vector<float> x_values;
		vector<float> z_values;
		for (int index = 0; index < detector_count; ++index)
		{
			if (fabsf(detector[index * 12 + 2] - layer_center) <= 1e-4f)
			{
				indices.push_back(index);
				x_values.push_back(detector[index * 12 + 1]);
				z_values.push_back(detector[index * 12 + 3]);
			}
		}
		if (indices.empty()) return false;
		float width = detector[indices.front() * 12 + 4];
		float thickness = detector[indices.front() * 12 + 5];
		float height = detector[indices.front() * 12 + 6];
		for (int index : indices)
		{
			if (fabsf(detector[index * 12 + 4] - width) > 1e-4f
				|| fabsf(detector[index * 12 + 5] - thickness) > 1e-4f
				|| fabsf(detector[index * 12 + 6] - height) > 1e-4f)
				return false;
		}
		vector<float> unique_x = sortedUniqueCoordinates(x_values);
		vector<float> unique_z = sortedUniqueCoordinates(z_values);
		float minimum_x = 0.0f;
		float minimum_z = 0.0f;
		float pitch_x = 0.0f;
		float pitch_z = 0.0f;
		int count_x = 0;
		int count_z = 0;
		if (!inferUniformCoordinateGrid(
			unique_x, width, &minimum_x, &pitch_x, &count_x)
			|| !inferUniformCoordinateGrid(
				unique_z, height, &minimum_z, &pitch_z, &count_z))
			return false;

		AxisAlignedLayerGrid layer;
		layer.y_min = layer_center - thickness * 0.5f;
		layer.y_max = layer_center + thickness * 0.5f;
		layer.x_boundary_min = minimum_x - pitch_x * 0.5f;
		layer.z_boundary_min = minimum_z - pitch_z * 0.5f;
		layer.pitch_x = pitch_x;
		layer.pitch_z = pitch_z;
		layer.count_x = count_x;
		layer.count_z = count_z;
		layer.map_offset = static_cast<int>(cell_to_detector->size());
		cell_to_detector->resize(
			cell_to_detector->size() + static_cast<size_t>(count_x) * count_z, -1);
		for (int index : indices)
		{
			int x_index = static_cast<int>(floorf(
				(detector[index * 12 + 1] - minimum_x) / pitch_x + 0.5f));
			int z_index = static_cast<int>(floorf(
				(detector[index * 12 + 3] - minimum_z) / pitch_z + 0.5f));
			if (x_index < 0 || x_index >= count_x
				|| z_index < 0 || z_index >= count_z)
				return false;
			int map_index = layer.map_offset + z_index * count_x + x_index;
			if ((*cell_to_detector)[map_index] >= 0) return false;
			(*cell_to_detector)[map_index] = index;
		}
		layers->push_back(layer);
	}
	return true;
}

struct PairLengthCacheHeader
{
	char magic[8];
	uint32_t version;
	uint32_t detector_count;
	uint64_t detector_hash;
	uint64_t data_offset;
};

struct PairLengthCache
{
	int file_descriptor;
	uint64_t data_offset;
	int detector_count;
};

static uint64_t fnv1aBytes(uint64_t hash, const void* data, size_t size)
{
	const unsigned char* bytes = static_cast<const unsigned char*>(data);
	for (size_t index = 0; index < size; ++index)
	{
		hash ^= bytes[index];
		hash *= 1099511628211ULL;
	}
	return hash;
}

static uint64_t detectorGeometryHash(
	const float* detector,
	const vector<int>& materials,
	int detector_count)
{
	uint64_t hash = 1469598103934665603ULL;
	hash = fnv1aBytes(hash, &detector_count, sizeof(detector_count));
	const int geometry_fields[] = {1, 2, 3, 4, 5, 6, 11};
	for (int index = 0; index < detector_count; ++index)
	{
		for (int field : geometry_fields)
			hash = fnv1aBytes(hash, &detector[index * 12 + field], sizeof(float));
		hash = fnv1aBytes(hash, &materials[index], sizeof(int));
	}
	return hash;
}

static bool positionalReadAll(int descriptor, void* data, size_t size, off_t offset)
{
	unsigned char* bytes = static_cast<unsigned char*>(data);
	size_t completed = 0;
	while (completed < size)
	{
		ssize_t result = pread(descriptor, bytes + completed,
			size - completed, offset + completed);
		if (result <= 0) return false;
		completed += static_cast<size_t>(result);
	}
	return true;
}

static bool positionalWriteAll(int descriptor, const void* data, size_t size, off_t offset)
{
	const unsigned char* bytes = static_cast<const unsigned char*>(data);
	size_t completed = 0;
	while (completed < size)
	{
		ssize_t result = pwrite(descriptor, bytes + completed,
			size - completed, offset + completed);
		if (result <= 0) return false;
		completed += static_cast<size_t>(result);
	}
	return true;
}

static PairLengthCache openPairLengthCache(
	const char* filename,
	int detector_count,
	uint64_t detector_hash)
{
	PairLengthCache cache = {-1, 0, detector_count};
	if (filename == NULL || filename[0] == '\0') return cache;
	int descriptor = open(filename, O_RDWR | O_CREAT, 0664);
	if (descriptor < 0)
	{
		perror("Cannot open crystal-pair length cache");
		exit(EXIT_FAILURE);
	}
	if (flock(descriptor, LOCK_EX) != 0)
	{
		perror("Cannot lock crystal-pair length cache");
		exit(EXIT_FAILURE);
	}
	struct stat status;
	if (fstat(descriptor, &status) != 0)
	{
		perror("Cannot stat crystal-pair length cache");
		exit(EXIT_FAILURE);
	}
	PairLengthCacheHeader header = {};
	const char expected_magic[8] = {'S', 'P', 'A', 'I', 'R', '0', '1', '\0'};
	if (status.st_size == 0)
	{
		memcpy(header.magic, expected_magic, sizeof(expected_magic));
		header.version = 1;
		header.detector_count = detector_count;
		header.detector_hash = detector_hash;
		header.data_offset = sizeof(PairLengthCacheHeader) + detector_count;
		off_t total_size = static_cast<off_t>(header.data_offset)
			+ static_cast<off_t>(detector_count) * detector_count * sizeof(float4);
		if (!positionalWriteAll(descriptor, &header, sizeof(header), 0)
			|| ftruncate(descriptor, total_size) != 0)
		{
			perror("Cannot initialize crystal-pair length cache");
			exit(EXIT_FAILURE);
		}
	}
	else
	{
		if (!positionalReadAll(descriptor, &header, sizeof(header), 0)
			|| memcmp(header.magic, expected_magic, sizeof(expected_magic)) != 0
			|| header.version != 1
			|| header.detector_count != static_cast<uint32_t>(detector_count)
			|| header.detector_hash != detector_hash)
		{
			fprintf(stderr,
				"Crystal-pair length cache does not match current detector geometry: %s\n",
				filename);
			exit(EXIT_FAILURE);
		}
	}
	flock(descriptor, LOCK_UN);
	cache.file_descriptor = descriptor;
	cache.data_offset = header.data_offset;
	cout << "Crystal-pair material-length cache: " << filename
		<< " geometry_hash=0x" << hex << detector_hash << dec << endl;
	return cache;
}

static bool readPairLengthCacheRows(
	const PairLengthCache& cache,
	int start,
	int count,
	float4* output)
{
	if (cache.file_descriptor < 0) return false;
	vector<unsigned char> valid(count, 0);
	if (!positionalReadAll(cache.file_descriptor, valid.data(), valid.size(),
		sizeof(PairLengthCacheHeader) + start)) return false;
	for (unsigned char value : valid)
		if (value != 1U) return false;
	size_t pair_count = static_cast<size_t>(count) * cache.detector_count;
	off_t data_offset = static_cast<off_t>(cache.data_offset)
		+ static_cast<off_t>(start) * cache.detector_count * sizeof(float4);
	return positionalReadAll(cache.file_descriptor, output,
		pair_count * sizeof(float4), data_offset);
}

static void writePairLengthCacheRows(
	const PairLengthCache& cache,
	int start,
	int count,
	const float4* input)
{
	if (cache.file_descriptor < 0) return;
	size_t pair_count = static_cast<size_t>(count) * cache.detector_count;
	off_t data_offset = static_cast<off_t>(cache.data_offset)
		+ static_cast<off_t>(start) * cache.detector_count * sizeof(float4);
	if (!positionalWriteAll(cache.file_descriptor, input,
		pair_count * sizeof(float4), data_offset)
		|| fdatasync(cache.file_descriptor) != 0)
	{
		perror("Cannot write crystal-pair length cache data");
		exit(EXIT_FAILURE);
	}
	vector<unsigned char> valid(count, 1U);
	if (!positionalWriteAll(cache.file_descriptor, valid.data(), valid.size(),
		sizeof(PairLengthCacheHeader) + start)
		|| fdatasync(cache.file_descriptor) != 0)
	{
		perror("Cannot commit crystal-pair length cache rows");
		exit(EXIT_FAILURE);
	}
}

static float interpolateXcomHost(const float* table, int material_id, float energy_keV)
{
	if (material_id < 0) return 0.0f;
	if (energy_keV <= kXcomEnergyMinKeV)
		return table[material_id * kXcomEnergyCount];
	if (energy_keV >= kXcomEnergyMaxKeV)
		return table[(material_id + 1) * kXcomEnergyCount - 1];

	int lower_energy = static_cast<int>(floorf(energy_keV));
	float fraction = energy_keV - lower_energy;
	int lower_index = material_id * kXcomEnergyCount + lower_energy - kXcomEnergyMinKeV;
	return table[lower_index] + fraction * (table[lower_index + 1] - table[lower_index]);
}

static int identifyXcomMaterial(float mu_pe, float mu_compton, float energy_keV)
{
	if (mu_pe + mu_compton <= 1e-8f) return kMaterialVacuum;

	int best_material = kMaterialVacuum;
	float best_score = numeric_limits<float>::infinity();
	for (int material = 0; material < kXcomMaterialCount; ++material)
	{
		float expected_pe = interpolateXcomHost(kXcomMuPhotoelectric, material, energy_keV);
		float expected_compton = interpolateXcomHost(kXcomMuCompton, material, energy_keV);
		float pe_scale = fmaxf(expected_pe, 1e-8f);
		float compton_scale = fmaxf(expected_compton, 1e-8f);
		float score = fabsf(mu_pe - expected_pe) / pe_scale
			+ fabsf(mu_compton - expected_compton) / compton_scale;
		if (score < best_score)
		{
			best_score = score;
			best_material = material;
		}
	}
	if (best_score > 0.10f)
	{
		fprintf(stderr,
			"Cannot identify XCOM material at %.6g keV: mu_pe=%.9g mu_compton=%.9g best_relative_score=%.6g\n",
			energy_keV, mu_pe, mu_compton, best_score);
		exit(EXIT_FAILURE);
	}
	return best_material;
}

static const char* xcomMaterialName(int material_id)
{
	switch (material_id)
	{
	case kMaterialNaI: return "NaI";
	case kMaterialGAGG: return "GAGG";
	case kMaterialPb: return "Pb";
	case kMaterialW: return "W";
	default: return "Vacuum";
	}
}

struct DetectorLocalScatterType
{
	float width;
	float thickness;
	float height;
	float relative_fwhm;
	float window_lower;
	float window_upper;
	int material_id;
};

static bool sameLocalScatterType(
	const DetectorLocalScatterType& lhs,
	const DetectorLocalScatterType& rhs)
{
	return lhs.material_id == rhs.material_id
		&& fabsf(lhs.width - rhs.width) < 1e-5f
		&& fabsf(lhs.thickness - rhs.thickness) < 1e-5f
		&& fabsf(lhs.height - rhs.height) < 1e-5f
		&& fabsf(lhs.relative_fwhm - rhs.relative_fwhm) < 1e-6f
		&& fabsf(lhs.window_lower - rhs.window_lower) < 1e-4f
		&& fabsf(lhs.window_upper - rhs.window_upper) < 1e-4f;
}

static int positiveEnvironmentInteger(const char* name, int fallback)
{
	const char* text = getenv(name);
	if (text == NULL) return fallback;
	int value = atoi(text);
	return value > 0 ? value : fallback;
}

static float positiveEnvironmentFloat(const char* name, float fallback)
{
	const char* text = getenv(name);
	if (text == NULL) return fallback;
	float value = static_cast<float>(atof(text));
	return value > 0.0f ? value : fallback;
}

static void writeScatterComponentSlice(
	const char* filename,
	const float* values,
	size_t count,
	int rotation_index)
{
	FILE* file = fopen(filename, rotation_index == 0 ? "wb" : "ab");
	if (file == NULL)
	{
		perror(filename);
		exit(EXIT_FAILURE);
	}
	if (fwrite(values, sizeof(float), count, file) != count)
	{
		fprintf(stderr, "Cannot write scatter component %s.\n", filename);
		fclose(file);
		exit(EXIT_FAILURE);
	}
	fclose(file);
}

static void buildDetectorLocalScatterLookup(
	const float* detector,
	const float* physics,
	const vector<int>& detector_materials,
	int detector_count,
	vector<int>* detector_type_ids,
	vector<float2>* lookup,
	int* orientation_bins)
{
	detector_type_ids->assign(detector_count, -1);
	lookup->clear();
	*orientation_bins = positiveEnvironmentInteger(
		"DETECTOR_LOCAL_SCATTER_ORIENTATION_BINS", 17);
	if (*orientation_bins < 2) *orientation_bins = 2;

	const bool enable_compton = floorf(physics[0] + 0.5f) > 0.0f;
	const bool enable_recoil = floorf(physics[10] + 0.5f) > 0.0f;
	const bool enable_self_photoelectric = floorf(physics[11] + 0.5f) > 0.0f;
	cout << "Detector local scatter switches: compton=" << enable_compton
		<< " recoil_escape=" << enable_recoil
		<< " self_compton_photoelectric=" << enable_self_photoelectric << endl;
	if (!enable_compton || (!enable_recoil && !enable_self_photoelectric)) return;

	const float source_energy = physics[7];
	vector<DetectorLocalScatterType> types;
	for (int index = 0; index < detector_count; ++index)
	{
		const int flag = static_cast<int>(floorf(detector[index * 12 + 12] + 0.5f));
		if (flag != 1 || detector_materials[index] < 0) continue;

		DetectorLocalScatterType candidate;
		candidate.width = detector[index * 12 + 4];
		candidate.thickness = detector[index * 12 + 5];
		candidate.height = detector[index * 12 + 6];
		candidate.relative_fwhm = detector[index * 12 + 10];
		candidate.material_id = detector_materials[index];
		if (floorf(physics[4] + 0.5f) > 0.0f)
		{
			candidate.window_lower = physics[5];
			candidate.window_upper = physics[6];
		}
		else
		{
			candidate.window_lower = (1.0f - candidate.relative_fwhm / 2.0f)
				* source_energy;
			candidate.window_upper = (1.0f + candidate.relative_fwhm / 2.0f)
				* source_energy;
		}

		int type_id = -1;
		for (int existing = 0; existing < static_cast<int>(types.size()); ++existing)
		{
			if (sameLocalScatterType(types[existing], candidate))
			{
				type_id = existing;
				break;
			}
		}
		if (type_id < 0)
		{
			type_id = static_cast<int>(types.size());
			types.push_back(candidate);
		}
		(*detector_type_ids)[index] = type_id;
	}

	const int cosine_samples = positiveEnvironmentInteger(
		"DETECTOR_LOCAL_SCATTER_COSINE_SAMPLES", 64);
	const int azimuth_samples = positiveEnvironmentInteger(
		"DETECTOR_LOCAL_SCATTER_AZIMUTH_SAMPLES", 64);
	const int position_samples_per_axis = positiveEnvironmentInteger(
		"DETECTOR_LOCAL_SCATTER_POSITION_SAMPLES_PER_AXIS", 4);
	const int bins = *orientation_bins;
	lookup->resize(types.size() * bins * bins);
	const double half_pi = 1.57079632679489661923;

	for (int type_id = 0; type_id < static_cast<int>(types.size()); ++type_id)
	{
		const DetectorLocalScatterType& type = types[type_id];
		double minimum_escape = 1.0;
		double maximum_escape = 0.0;
		double maximum_partition_error = 0.0;
		for (int polar_index = 0; polar_index < bins; ++polar_index)
		{
			const double polar = half_pi * polar_index / (bins - 1);
			for (int azimuth_index = 0; azimuth_index < bins; ++azimuth_index)
			{
				const double azimuth = half_pi * azimuth_index / (bins - 1);
				const double incoming_x = sin(polar) * cos(azimuth);
				const double incoming_y = cos(polar);
				const double incoming_z = sin(polar) * sin(azimuth);
				DetectorLocalScatterResponse response
					= integrate_detector_local_scatter_response(
						incoming_x, incoming_y, incoming_z,
						type.width, type.thickness, type.height,
						type.material_id, source_energy, type.relative_fwhm,
						type.window_lower, type.window_upper,
						cosine_samples, azimuth_samples,
						position_samples_per_axis);
				const double partition_sum = response.escape_probability
					+ response.second_photoelectric_probability
					+ response.second_compton_probability;
				const double partition_error = fabs(partition_sum - 1.0);
				if (partition_error > maximum_partition_error)
					maximum_partition_error = partition_error;
				if (response.escape_probability < minimum_escape)
					minimum_escape = response.escape_probability;
				if (response.escape_probability > maximum_escape)
					maximum_escape = response.escape_probability;
				(*lookup)[(type_id * bins + polar_index) * bins + azimuth_index]
					= make_float2(
						static_cast<float>(response.recoil_windowed),
						static_cast<float>(response.self_photoelectric_windowed));
			}
		}
		cout << "Detector local scatter type " << type_id
			<< " material=" << xcomMaterialName(type.material_id)
			<< " size=" << type.width << "x" << type.thickness << "x"
			<< type.height << " mm window=[" << type.window_lower << ","
			<< type.window_upper << "] keV lookup=" << bins << "x" << bins
			<< " angular_samples=" << cosine_samples << "x" << azimuth_samples
			<< " position_samples=" << position_samples_per_axis << "^3"
			<< " escape_range=[" << minimum_escape << "," << maximum_escape << "]"
			<< " max_partition_error=" << maximum_partition_error << endl;
		if (maximum_partition_error > 1e-8)
		{
			fprintf(stderr, "Detector local scatter probability partition failed.\n");
			exit(EXIT_FAILURE);
		}
	}
}

struct CollimatorScatterSample
{
	float x;
	float y_center;
	float z;
	float thickness;
	float lead_area;
	int material_id;
};

static vector<CollimatorScatterSample> buildCollimatorScatterSamples(
	const float* collimator,
	float source_energy_keV)
{
	vector<CollimatorScatterSample> samples;
	int layer_count = static_cast<int>(floorf(collimator[0] + 0.001f));
	int hole_offset = 0;
	int requested_samples = 0;
	const char* sample_env = getenv("COLLIMATOR_SCATTER_SAMPLES_PER_LAYER");
	if (sample_env != NULL) requested_samples = atoi(sample_env);
	int area_subdivisions = 8;
	const char* subdivision_env = getenv("COLLIMATOR_SCATTER_AREA_SUBDIV");
	if (subdivision_env != NULL && atoi(subdivision_env) > 0)
		area_subdivisions = atoi(subdivision_env);

	for (int layer = 0; layer < layer_count; ++layer)
	{
		int header = (layer + 1) * 10;
		int hole_count = static_cast<int>(floorf(collimator[header] + 0.001f));
		float width = collimator[header + 1];
		float thickness = collimator[header + 2];
		float height = collimator[header + 3];
		float y_center = collimator[header + 4];
		int material_id = identifyXcomMaterial(
			collimator[header + 6], collimator[header + 7], source_energy_keV);
		if (material_id < 0 || !(width > 0.0f) || !(height > 0.0f) || !(thickness > 0.0f))
		{
			hole_offset += hole_count;
			continue;
		}

		int target_samples = requested_samples > 0 ? requested_samples : hole_count;
		if (target_samples <= 0) target_samples = 1024;
		int nx = static_cast<int>(floorf(sqrtf(target_samples * width / height) + 0.5f));
		if (nx < 1) nx = 1;
		int nz = (target_samples + nx - 1) / nx;
		float dx = width / nx;
		float dz = height / nz;
		double represented_area = 0.0;

		for (int iz = 0; iz < nz; ++iz)
		{
			for (int ix = 0; ix < nx; ++ix)
			{
				float cell_x = -width / 2.0f + (ix + 0.5f) * dx;
				float cell_z = -height / 2.0f + (iz + 0.5f) * dz;
				int lead_points = 0;
				float representative_x = cell_x;
				float representative_z = cell_z;
				float best_distance2 = numeric_limits<float>::infinity();

				for (int sz = 0; sz < area_subdivisions; ++sz)
				{
					for (int sx = 0; sx < area_subdivisions; ++sx)
					{
						float x = cell_x + ((sx + 0.5f) / area_subdivisions - 0.5f) * dx;
						float z = cell_z + ((sz + 0.5f) / area_subdivisions - 0.5f) * dz;
						bool in_hole = false;
						for (int hole = 0; hole < hole_count; ++hole)
						{
							int record = (hole_offset + hole) * 9 + 100;
							float hx = collimator[record];
							float hz = collimator[record + 3];
							float radius = collimator[record + 4];
							float ddx = x - hx;
							float ddz = z - hz;
							if (ddx * ddx + ddz * ddz <= radius * radius)
							{
								in_hole = true;
								break;
							}
						}
						if (!in_hole)
						{
							++lead_points;
							float ddx = x - cell_x;
							float ddz = z - cell_z;
							float distance2 = ddx * ddx + ddz * ddz;
							if (distance2 < best_distance2)
							{
								best_distance2 = distance2;
								representative_x = x;
								representative_z = z;
							}
						}
					}
				}

				if (lead_points == 0) continue;
				float lead_fraction = static_cast<float>(lead_points)
					/ static_cast<float>(area_subdivisions * area_subdivisions);
				float lead_area = dx * dz * lead_fraction;
			represented_area += lead_area;
			samples.push_back({representative_x, y_center, representative_z,
				thickness, lead_area, material_id});
			}
		}

		cout << "Collimator layer " << layer << " XCOM material="
			<< xcomMaterialName(material_id) << " volume samples=" << (nx * nz)
			<< " represented Pb/high-Z area=" << represented_area << " mm^2" << endl;
		hole_offset += hole_count;
	}
	return samples;
}

__device__ inline void interpolateXcomDevice(
	int material_id,
	float energy_keV,
	float* mu_pe,
	float* mu_compton)
{
	if (material_id < 0)
	{
		*mu_pe = 0.0f;
		*mu_compton = 0.0f;
		return;
	}

	if (energy_keV <= kXcomEnergyMinKeV)
	{
		int index = material_id * kXcomEnergyCount;
		*mu_pe = deviceXcomMuPhotoelectric[index];
		*mu_compton = deviceXcomMuCompton[index];
		return;
	}
	if (energy_keV >= kXcomEnergyMaxKeV)
	{
		int index = (material_id + 1) * kXcomEnergyCount - 1;
		*mu_pe = deviceXcomMuPhotoelectric[index];
		*mu_compton = deviceXcomMuCompton[index];
		return;
	}

	int lower_energy = static_cast<int>(floorf(energy_keV));
	float fraction = energy_keV - lower_energy;
	int lower_index = material_id * kXcomEnergyCount + lower_energy - kXcomEnergyMinKeV;
	*mu_pe = deviceXcomMuPhotoelectric[lower_index]
		+ fraction * (deviceXcomMuPhotoelectric[lower_index + 1] - deviceXcomMuPhotoelectric[lower_index]);
	*mu_compton = deviceXcomMuCompton[lower_index]
		+ fraction * (deviceXcomMuCompton[lower_index + 1] - deviceXcomMuCompton[lower_index]);
}

__device__ inline float detectorCenterExitDistance(
	float direction_x,
	float direction_y,
	float direction_z,
	float width,
	float thickness,
	float height)
{
	float distance = 1.0e30f;
	if (fabsf(direction_x) > 1e-8f)
		distance = fminf(distance, 0.5f * width / fabsf(direction_x));
	if (fabsf(direction_y) > 1e-8f)
		distance = fminf(distance, 0.5f * thickness / fabsf(direction_y));
	if (fabsf(direction_z) > 1e-8f)
		distance = fminf(distance, 0.5f * height / fabsf(direction_z));
	return distance < 1.0e29f ? distance : 0.0f;
}

__device__ inline float2 interpolateDetectorLocalScatterLookup(
	const float2* lookup,
	int type_id,
	int bins,
	float incoming_x,
	float incoming_y,
	float incoming_z)
{
	const float norm = sqrtf(incoming_x * incoming_x
		+ incoming_y * incoming_y + incoming_z * incoming_z);
	if (!(norm > 0.0f) || type_id < 0 || bins < 2)
		return make_float2(0.0f, 0.0f);
	incoming_x = fabsf(incoming_x / norm);
	incoming_y = fabsf(incoming_y / norm);
	incoming_z = fabsf(incoming_z / norm);

	const float half_pi = 1.57079632679489661923f;
	const float polar = acosf(fminf(incoming_y, 1.0f));
	const float azimuth = atan2f(incoming_z, incoming_x);
	const float polar_position = polar / half_pi * (bins - 1);
	const float azimuth_position = azimuth / half_pi * (bins - 1);
	const int polar0 = static_cast<int>(floorf(polar_position));
	const int azimuth0 = static_cast<int>(floorf(azimuth_position));
	const int polar1 = polar0 + 1 < bins ? polar0 + 1 : polar0;
	const int azimuth1 = azimuth0 + 1 < bins ? azimuth0 + 1 : azimuth0;
	const float polar_fraction = polar_position - polar0;
	const float azimuth_fraction = azimuth_position - azimuth0;
	const int base = type_id * bins * bins;
	const float2 value00 = lookup[base + polar0 * bins + azimuth0];
	const float2 value01 = lookup[base + polar0 * bins + azimuth1];
	const float2 value10 = lookup[base + polar1 * bins + azimuth0];
	const float2 value11 = lookup[base + polar1 * bins + azimuth1];
	const float2 value0 = make_float2(
		value00.x + azimuth_fraction * (value01.x - value00.x),
		value00.y + azimuth_fraction * (value01.y - value00.y));
	const float2 value1 = make_float2(
		value10.x + azimuth_fraction * (value11.x - value10.x),
		value10.y + azimuth_fraction * (value11.y - value10.y));
	return make_float2(
		value0.x + polar_fraction * (value1.x - value0.x),
		value0.y + polar_fraction * (value1.y - value0.y));
}

__device__ inline float attenuatedSlabDepthIntegral(float attenuation_in, float attenuation_out, float thickness)
{
	if (!(thickness > 0.0f)) return 0.0f;
	float difference = attenuation_in - attenuation_out;
	if (fabsf(difference) < 1e-5f)
		return thickness * expf(-attenuation_in * thickness);
	float value = (expf(-attenuation_out * thickness) - expf(-attenuation_in * thickness)) / difference;
	return value > 0.0f && isfinite(value) ? value : 0.0f;
}

__device__ float length_box_ray(float x_in, float y_in, float z_in, float x_out, float y_out, float z_out, float x1_box, float y1_box, float z1_box, float x2_box, float y2_box, float z2_box)
{
	// Incident ray position: (x_in, y_in, z_in)
	// Outgoing ray position: (x_out, y_out, z_out)
	// Left-Down box position: (x1_box, y1_box, z1_box)
	// Right-Up box position: (x2_box, y2_box, z2_box)
	float eps = 0.001;

	if (fabs(x_out - x_in) < eps & fabs(y_out - y_in) < eps & fabs(z_out - z_in) < eps)
	{
		return 0.000;
	}

	if (fabs(x_out - x_in) < eps & fabs(y_out - y_in) < eps)
	{
		if ((x_in >= x1_box & x_in <= x2_box) & (y_in >= y1_box & y_in <= y2_box))
			if ((z_in <= z1_box & z_out >= z2_box) || (z_out <= z1_box & z_in >= z2_box))
				return fabs(z2_box - z1_box);
			else
				return 0.000;
		else
			return 0.000;
	}

	if (fabs(z_out - z_in) < eps & fabs(y_out - y_in) < eps)
	{
		if ((z_in >= z1_box & z_in <= z2_box) & (y_in >= y1_box & y_in <= y2_box))
			if ((x_in <= x1_box & x_out >= x2_box) || (x_out <= x1_box & x_in >= x2_box))
				return fabs(x2_box -x1_box);
			else
				return 0.000;
		else
			return 0.000;
	}

	if (fabs(x_out - x_in) < eps & fabs(z_out - z_in) < eps)
	{
		if ((x_in >= x1_box & x_in <= x2_box) & (z_in >= z1_box & z_in <= z2_box))
		{
			if ((y_in <= y1_box & y_out >= y2_box) || (y_out <= y1_box & y_in >= y2_box))
				return fabs(y2_box - y1_box);
			else
				return 0.000;
		}
		else
			return 0.000;
	}


	if (fabs(x_out - x_in) < eps & (x_in >= x2_box || x_in <= x1_box))
	{
		return 0.000;
	}
	else if (fabs(x_out - x_in) < eps)
	{
		float tmin, tmax, tzmin, tzmax, t_inout;
		t_inout = sqrt((y_out - y_in) * (y_out - y_in) + (z_out - z_in) * (z_out - z_in));


		float inv_direction_y = t_inout / (y_out - y_in);
		float inv_direction_z = t_inout / (z_out - z_in);

		if (inv_direction_y < 0)
		{
			tmin = (y2_box - y_in) * inv_direction_y;
			tmax = (y1_box - y_in) * inv_direction_y;
		}
		else
		{
			tmax = (y2_box - y_in) * inv_direction_y;
			tmin = (y1_box - y_in) * inv_direction_y;
		}


		if (inv_direction_z < 0)
		{
			tzmin = (z2_box - z_in) * inv_direction_z;
			tzmax = (z1_box - z_in) * inv_direction_z;
		}
		else
		{
			tzmax = (z2_box - z_in) * inv_direction_z;
			tzmin = (z1_box - z_in) * inv_direction_z;
		}

		if ((tmin > tzmax) || (tzmin > tmax))
			return 0.0;

		if (tzmin > tmin)
			tmin = tzmin;

		if (tzmax < tmax)
			tmax = tzmax;

		if ((tmax - tmin) < eps)
			return 0.000;
		else if (tmin >= t_inout || tmax >= t_inout)
			return 0.000;
		else if (tmin <= eps || tmax <= eps)
			return 0.000;
		else
			return (tmax - tmin);

	}

	if (fabs(y_out - y_in) < eps & (y_in >= y2_box || y_in <= y1_box))
	{
		return 0.000;
	}
	else if (fabs(y_out - y_in) < eps)
	{
		float tmin, tmax, tzmin, tzmax, t_inout;
		t_inout = sqrt((x_out - x_in) * (x_out - x_in) + (z_out - z_in) * (z_out - z_in));


		float inv_direction_x = t_inout / (x_out - x_in);
		float inv_direction_z = t_inout / (z_out - z_in);

		if (inv_direction_x < 0)
		{
			tmin = (x2_box - x_in) * inv_direction_x;
			tmax = (x1_box - x_in) * inv_direction_x;
		}
		else
		{
			tmax = (x2_box - x_in) * inv_direction_x;
			tmin = (x1_box - x_in) * inv_direction_x;
		}


		if (inv_direction_z < 0)
		{
			tzmin = (z2_box - z_in) * inv_direction_z;
			tzmax = (z1_box - z_in) * inv_direction_z;
		}
		else
		{
			tzmax = (z2_box - z_in) * inv_direction_z;
			tzmin = (z1_box - z_in) * inv_direction_z;
		}

		if ((tmin > tzmax) || (tzmin > tmax))
			return 0.0;

		if (tzmin > tmin)
			tmin = tzmin;

		if (tzmax < tmax)
			tmax = tzmax;

		if ((tmax - tmin) < eps)
			return 0.000;
		else if (tmin >= t_inout || tmax >= t_inout)
			return 0.000;
		else if (tmin <= eps || tmax <= eps)
			return 0.000;
		else
			return (tmax - tmin);
	}

	if (fabs(z_out - z_in) < eps & (z_in >= z2_box || z_in <= z1_box))
	{
		return 0.000;
	}
	else if (fabs(z_out - z_in) < eps)
	{
		float tmin, tmax, tymin, tymax, t_inout;
		t_inout = sqrt((x_out - x_in) * (x_out - x_in) + (y_out - y_in) * (y_out - y_in));

		float inv_direction_x = t_inout / (x_out - x_in);
		float inv_direction_y = t_inout / (y_out - y_in);


		if (inv_direction_x < 0)
		{
			tmin = (x2_box - x_in) * inv_direction_x;
			tmax = (x1_box - x_in) * inv_direction_x;
		}
		else
		{
			tmax = (x2_box - x_in) * inv_direction_x;
			tmin = (x1_box - x_in) * inv_direction_x;
		}


		if (inv_direction_y < 0)
		{
			tymin = (y2_box - y_in) * inv_direction_y;
			tymax = (y1_box - y_in) * inv_direction_y;
		}
		else
		{
			tymax = (y2_box - y_in) * inv_direction_y;
			tymin = (y1_box - y_in) * inv_direction_y;
		}

		if ((tmin > tymax) || (tymin > tmax))
			return 0.0;

		if (tymin > tmin)
			tmin = tymin;

		if (tymax < tmax)
			tmax = tymax;
		if ((tmax - tmin) < eps)
			return 0.000;
		else if (tmin >= t_inout || tmax >= t_inout)
			return 0.000;
		else if (tmin <= eps || tmax <= eps)
			return 0.000;
		else
			return (tmax - tmin);
	}



	float tmin, tmax, tymin, tymax, tzmin, tzmax, t_inout;
	t_inout = sqrt((x_out - x_in) * (x_out - x_in) + (y_out - y_in) * (y_out - y_in) + (z_out - z_in) * (z_out - z_in));

	float inv_direction_x = t_inout / (x_out - x_in);
	float inv_direction_y = t_inout / (y_out - y_in);
	float inv_direction_z = t_inout / (z_out - z_in);


	if (inv_direction_x < 0)
	{
		tmin = (x2_box - x_in) * inv_direction_x;
		tmax = (x1_box - x_in) * inv_direction_x;
	}
	else
	{
		tmax = (x2_box - x_in) * inv_direction_x;
		tmin = (x1_box - x_in) * inv_direction_x;
	}


	if (inv_direction_y < 0)
	{
		tymin = (y2_box - y_in) * inv_direction_y;
		tymax = (y1_box - y_in) * inv_direction_y;
	}
	else
	{
		tymax = (y2_box - y_in) * inv_direction_y;
		tymin = (y1_box - y_in) * inv_direction_y;
	}

	if ((tmin > tymax) || (tymin > tmax))
		return 0.0;

	if (tymin > tmin)
		tmin = tymin;

	if (tymax < tmax)
		tmax = tymax;

	if (inv_direction_z < 0)
	{
		tzmin = (z2_box - z_in) * inv_direction_z;
		tzmax = (z1_box - z_in) * inv_direction_z;
	}
	else
	{
		tzmax = (z2_box - z_in) * inv_direction_z;
		tzmin = (z1_box - z_in) * inv_direction_z;
	}

	if ((tmin > tzmax) || (tzmin > tmax))
		return 0.0;

	if (tzmin > tmin)
		tmin = tzmin;

	if (tzmax < tmax)
		tmax = tzmax;


	if ((tmax - tmin) < eps)
		return 0.000;
	else if (tmin >= t_inout || tmax >= t_inout)
		return 0.000;
	else if (tmin <= eps || tmax <= eps)
		return 0.000;
	else
		return (tmax - tmin);
	// post condition:
	// if tmin > tmax (in the code above this is represented by a return value of INFINITY)
	//     no intersection
	// else
	//     front intersection point = ray.origin + ray.direction * tmin (normally only this point matters)
	//     back intersection point  = ray.origin + ray.direction * tmax

}

__device__ float length_box_ray_inside(float x_in, float y_in, float z_in, float x_out, float y_out, float z_out, float x1_box, float y1_box, float z1_box, float x2_box, float y2_box, float z2_box)
{
	// Incident ray position: (x_in, y_in, z_in)
	// Outgoing ray position: (x_out, y_out, z_out)
	// Left-Down box position: (x1_box, y1_box, z1_box)
	// Right-Up box position: (x2_box, y2_box, z2_box)
	// Outgoing position is inside the box!!!!
	if (fabs(y_out - y_in) < 0.001)
	{
		return 0.000;
	}

	float eps = 0.001;

	if (fabs(x_out - x_in) < eps & fabs(y_out - y_in) < eps & fabs(z_out - z_in) < eps)
	{
		return 0.000;
	}

	if (fabs(x_out - x_in) < eps & fabs(y_out - y_in) < eps)
	{
		if ((x_in >= x1_box & x_in <= x2_box) & (y_in >= y1_box & y_in <= y2_box))
		{
			if (z_in < z_out)
			{
				return (z_out - z1_box);
			}
			else if (z_in > z_out)
			{
				return (z2_box - z_out);
			}
		}
		else
			return 0.000;
	}

	if (fabs(z_out - z_in) < eps & fabs(y_out - y_in) < eps)
	{
		if ((z_in >= z1_box & z_in <= z2_box) & (y_in >= y1_box & y_in <= y2_box))
		{
			if (x_in < x_out)
			{
				return (x_out - x1_box);
			}
			else if (x_in > x_out)
			{
				return (x2_box - x_out);
			}
		}
		else
			return 0.000;
	}

	if (fabs(x_out - x_in) < eps & fabs(z_out - z_in) < eps)
	{
		if ((x_in >= x1_box & x_in <= x2_box) & (z_in >= z1_box & z_in <= z2_box))
		{
			if (y_in < y_out)
			{
				return (y_out - y1_box);
			}
			else if (y_in > y_out)
			{
				return (y2_box - y_out);
			}
		}
		else
			return 0.000;
	}


	if (fabs(x_out - x_in) < eps & (x_in >= x2_box || x_in <= x1_box))
	{
		return 0.000;
	}
	else if (fabs(x_out - x_in) < eps)
	{
		float tmin, tmax, tzmin, tzmax, t_inout;
		t_inout = sqrt((y_out - y_in) * (y_out - y_in) + (z_out - z_in) * (z_out - z_in));


		float inv_direction_y = t_inout / (y_out - y_in);
		float inv_direction_z = t_inout / (z_out - z_in);

		if (inv_direction_y < 0)
		{
			tmin = (y2_box - y_in) * inv_direction_y;
			tmax = (y1_box - y_in) * inv_direction_y;
		}
		else
		{
			tmax = (y2_box - y_in) * inv_direction_y;
			tmin = (y1_box - y_in) * inv_direction_y;
		}


		if (inv_direction_z < 0)
		{
			tzmin = (z2_box - z_in) * inv_direction_z;
			tzmax = (z1_box - z_in) * inv_direction_z;
		}
		else
		{
			tzmax = (z2_box - z_in) * inv_direction_z;
			tzmin = (z1_box - z_in) * inv_direction_z;
		}

		if ((tmin > tzmax) || (tzmin > tmax))
			return 0.0;

		if (tzmin > tmin)
			tmin = tzmin;

		if (tzmax < tmax)
			tmax = tzmax;



		if ((tmax - tmin) < eps)
			return 0.0;

		else if (tmin >= t_inout)
			return 0.0;

		else if (tmax <= eps)
			return 0.0;

		else if ((tmax >= t_inout) && (tmin > eps))
			return (t_inout - tmin);

		else if ((tmin <= eps) && (tmax <= t_inout))
			return tmax;

		else if ((tmin <= eps) && (tmax >= t_inout))
			return t_inout;

		else
			return (tmax - tmin);

	}

	if (fabs(y_out - y_in) < eps & (y_in >= y2_box || y_in <= y1_box))
	{
		return 0.000;
	}
	else if (fabs(y_out - y_in) < eps)
	{
		float tmin, tmax, tzmin, tzmax, t_inout;
		t_inout = sqrt((x_out - x_in) * (x_out - x_in) + (z_out - z_in) * (z_out - z_in));


		float inv_direction_x = t_inout / (x_out - x_in);
		float inv_direction_z = t_inout / (z_out - z_in);

		if (inv_direction_x < 0)
		{
			tmin = (x2_box - x_in) * inv_direction_x;
			tmax = (x1_box - x_in) * inv_direction_x;
		}
		else
		{
			tmax = (x2_box - x_in) * inv_direction_x;
			tmin = (x1_box - x_in) * inv_direction_x;
		}


		if (inv_direction_z < 0)
		{
			tzmin = (z2_box - z_in) * inv_direction_z;
			tzmax = (z1_box - z_in) * inv_direction_z;
		}
		else
		{
			tzmax = (z2_box - z_in) * inv_direction_z;
			tzmin = (z1_box - z_in) * inv_direction_z;
		}

		if ((tmin > tzmax) || (tzmin > tmax))
			return 0.0;

		if (tzmin > tmin)
			tmin = tzmin;

		if (tzmax < tmax)
			tmax = tzmax;


		if ((tmax - tmin) < eps)
			return 0.0;

		else if (tmin >= t_inout)
			return 0.0;

		else if (tmax <= eps)
			return 0.0;

		else if ((tmax >= t_inout) && (tmin > eps))
			return (t_inout - tmin);

		else if ((tmin <= eps) && (tmax <= t_inout))
			return tmax;

		else if ((tmin <= eps) && (tmax >= t_inout))
			return t_inout;

		else
			return (tmax - tmin);
	}

	if (fabs(z_out - z_in) < eps & (z_in >= z2_box || z_in <= z1_box))
	{
		return 0.000;
	}
	else if (fabs(z_out - z_in) < eps)
	{
		float tmin, tmax, tymin, tymax, t_inout;
		t_inout = sqrt((x_out - x_in) * (x_out - x_in) + (y_out - y_in) * (y_out - y_in));

		float inv_direction_x = t_inout / (x_out - x_in);
		float inv_direction_y = t_inout / (y_out - y_in);


		if (inv_direction_x < 0)
		{
			tmin = (x2_box - x_in) * inv_direction_x;
			tmax = (x1_box - x_in) * inv_direction_x;
		}
		else
		{
			tmax = (x2_box - x_in) * inv_direction_x;
			tmin = (x1_box - x_in) * inv_direction_x;
		}


		if (inv_direction_y < 0)
		{
			tymin = (y2_box - y_in) * inv_direction_y;
			tymax = (y1_box - y_in) * inv_direction_y;
		}
		else
		{
			tymax = (y2_box - y_in) * inv_direction_y;
			tymin = (y1_box - y_in) * inv_direction_y;
		}

		if ((tmin > tymax) || (tymin > tmax))
			return 0.0;

		if (tymin > tmin)
			tmin = tymin;

		if (tymax < tmax)
			tmax = tymax;


		if ((tmax - tmin) < eps)
			return 0.0;

		else if (tmin >= t_inout)
			return 0.0;

		else if (tmax <= eps)
			return 0.0;

		else if ((tmax >= t_inout) && (tmin > eps))
			return (t_inout - tmin);

		else if ((tmin <= eps) && (tmax <= t_inout))
			return tmax;

		else if ((tmin <= eps) && (tmax >= t_inout))
			return t_inout;

		else
			return (tmax - tmin);
	}






	float tmin, tmax, tymin, tymax, tzmin, tzmax, t_inout;
	t_inout = sqrt((x_out - x_in) * (x_out - x_in) + (y_out - y_in) * (y_out - y_in) + (z_out - z_in) * (z_out - z_in));

	float inv_direction_x = t_inout / (x_out - x_in);
	float inv_direction_y = t_inout / (y_out - y_in);
	float inv_direction_z = t_inout / (z_out - z_in);


	if (inv_direction_x < 0)
	{
		tmin = (x2_box - x_in) * inv_direction_x;
		tmax = (x1_box - x_in) * inv_direction_x;
	}
	else
	{
		tmax = (x2_box - x_in) * inv_direction_x;
		tmin = (x1_box - x_in) * inv_direction_x;
	}


	if (inv_direction_y < 0)
	{
		tymin = (y2_box - y_in) * inv_direction_y;
		tymax = (y1_box - y_in) * inv_direction_y;
	}
	else
	{
		tymax = (y2_box - y_in) * inv_direction_y;
		tymin = (y1_box - y_in) * inv_direction_y;
	}

	if ((tmin > tymax) || (tymin > tmax))
		return 0.0;

	if (tymin > tmin)
		tmin = tymin;

	if (tymax < tmax)
		tmax = tymax;

	if (inv_direction_z < 0)
	{
		tzmin = (z2_box - z_in) * inv_direction_z;
		tzmax = (z1_box - z_in) * inv_direction_z;
	}
	else
	{
		tzmax = (z2_box - z_in) * inv_direction_z;
		tzmin = (z1_box - z_in) * inv_direction_z;
	}

	if ((tmin > tzmax) || (tzmin > tmax))
		return 0.0;

	if (tzmin > tmin)
		tmin = tzmin;

	if (tzmax < tmax)
		tmax = tzmax;


	if ((tmax - tmin) < eps)
		return 0.0;

	else if (tmin >= t_inout)
		return 0.0;

	else if (tmax <= eps)
		return 0.0;

	else if ((tmax >= t_inout) && (tmin > eps))
		return (t_inout - tmin);

	else if ((tmin <= eps) && (tmax <= t_inout))
		return tmax;

	else if ((tmin <= eps) && (tmax >= t_inout))
		return t_inout;

	else
		return (tmax - tmin);


	// post condition:
	// if tmin > tmax (in the code above this is represented by a return value of INFINITY)
	//     no intersection
	// else
	//     front intersection point = ray.origin + ray.direction * tmin (normally only this point matters)
	//     back intersection point  = ray.origin + ray.direction * tmax

}

__device__ float length_cylinder_ray(float x_in, float y_in, float z_in, float x_out, float y_out, float z_out, float x_cylinder, float y1_cylinder, float y2_cylinder, float z_cylinder, float radius)
{
	// Incident ray position: (x_in, y_in, z_in)
	// Outgoing ray position: (x_out, y_out, z_out)
	// Left-Plane position: (y==y1)
	// Right-Plane position: (y==y2)
	// (x-x_cylinder)^2+(z-z_cylinder)^2=radius^2
	if (fabs(y1_cylinder - y2_cylinder) < 0.001)
	{
		return 0.000;
	}
	float t_inout = sqrt((x_out - x_in) * (x_out - x_in) + (y_out - y_in) * (y_out - y_in) + (z_out - z_in) * (z_out - z_in));
	float k_x = (x_out - x_in) / t_inout;
	float k_y = (y_out - y_in) / t_inout;
	float k_z = (z_out - z_in) / t_inout;

	float x_leftPlane = x_in + k_x / k_y * (y1_cylinder - y_in);
	float x_rightPlane = x_in + k_x / k_y * (y2_cylinder - y_in);

	float z_leftPlane = z_in + k_z / k_y * (y1_cylinder - y_in);
	float z_rightPlane = z_in + k_z / k_y * (y2_cylinder - y_in);

	float tmin = (y1_cylinder - y_in) / k_y;
	float tmax = (y2_cylinder - y_in) / k_y;


	int flag_leftPlane = 0;
	int flag_rightPlane = 0;

	float t1 = 0;
	float t2 = 0;

	if ((x_leftPlane - x_cylinder) * (x_leftPlane - x_cylinder) + (z_leftPlane - z_cylinder) * (z_leftPlane - z_cylinder) <= radius * radius)
	{
		flag_leftPlane = 1;
	}
	if ((x_rightPlane - x_cylinder) * (x_rightPlane - x_cylinder) + (z_rightPlane - z_cylinder) * (z_rightPlane - z_cylinder) <= radius * radius)
	{
		flag_rightPlane = 1;
	}


	if ((flag_rightPlane == 1) && (flag_leftPlane == 1))
	{
		if (tmin <= 0.0001 || tmax <= 0.0001)
			return 0.0;
		else if (tmin >= t_inout || tmax >= t_inout)
			return 0.0;
		else
			return fabs(tmax - tmin);
	}
	else
	{
		float x_ = x_in - x_cylinder;
		float z_ = z_in - z_cylinder;
		float Delta_ = (k_x * k_x + k_z * k_z) * radius * radius - (k_x * z_ - k_z * x_) * (k_x * z_ - k_z * x_);

		if (Delta_ <= 0.00001)
		{
			return 0.00000000;
		}
		else
		{
			t1 = (-(k_x * x_ + k_z * z_) - sqrt(Delta_)) / (k_x * k_x + k_z * k_z);
			t2 = (-(k_x * x_ + k_z * z_) + sqrt(Delta_)) / (k_x * k_x + k_z * k_z);
		}

		if ((flag_rightPlane == 0) && (flag_leftPlane == 0))
		{
			if ((t1 >= tmin) && (t2 <= tmax))
			{
				if (t2 <= 0.0001 || t1 <= 0.0001)
					return 0.0;
				else if (t2 >= t_inout || t1 >= t_inout)
					return 0.0;
				else
					return (t2 - t1);

			}
			else
			{
				return 0.000000;
			}

		}
		else if ((flag_leftPlane == 1) && (flag_rightPlane == 0))
		{
			if (t2 <= 0.0001 || tmin <= 0.0001)
				return 0.0;
			else if (t2 >= t_inout || tmin >= t_inout)
				return 0.0;
			else
				return (t2 - tmin);			

		}
		else if ((flag_leftPlane == 0) && (flag_rightPlane == 1))
		{
			if (tmax <= 0.0001 || t1 <= 0.0001)
				return 0.0;
			else if (tmax >= t_inout || t1 >= t_inout)
				return 0.0;
			else
				return (tmax - t1);
			
		}
		else
		{
			return 0.000000;
		}
	}

}

__device__ float length_ellipticalcylinder_ray(float x_in, float y_in, float z_in, float x_out, float y_out, float z_out, float x_cylinder, float y1_cylinder, float y2_cylinder, float z_cylinder, float x_rho, float z_rho, float theta)
{
	// Incident ray position: (x_in, y_in, z_in)
	// Outgoing ray position: (x_out, y_out, z_out)
	// Left-Plane position: (y==y1)
	// Right-Plane position: (y==y2)
	// [(x-x_cylinder)/x_rho]^2+[(z-z_cylinder)/z_rho]^2=1
	// with rotation angle theta, defined inverse conunter-clock wise from x axis
	//                        ^ z
	//                        .
	//                        .
	//                        .
	//                        .
	//                        .        . x'
	//                        .       .
	//                        .      .
	//                        .     .
	//                        .    .
	//                        .   .
	//                        .  .
	//                        . .  theta
	//........................................................>x
	//
	//

	float x_in_ = x_in - x_cylinder;
	float z_in_ = z_in - z_cylinder;

	float x_out_ = x_out - x_cylinder;
	float z_out_ = z_out - z_cylinder;

	x_in = x_cylinder + x_in_ * cos(theta) + z_in_ * sin(theta);
	z_in = z_cylinder - x_in_ * sin(theta) + z_in_ * cos(theta);

	x_out = x_cylinder + x_out_ * cos(theta) + z_out_ * sin(theta);
	z_out = z_cylinder - x_out_ * sin(theta) + z_out_ * cos(theta);

	if (fabs(y1_cylinder - y2_cylinder) < 0.001)
	{
		return 0.000;
	}
	float t_inout = sqrt((x_out - x_in) * (x_out - x_in) + (y_out - y_in) * (y_out - y_in) + (z_out - z_in) * (z_out - z_in));
	float k_x = (x_out - x_in) / t_inout;
	float k_y = (y_out - y_in) / t_inout;
	float k_z = (z_out - z_in) / t_inout;

	float x_leftPlane = x_in + k_x / k_y * (y1_cylinder - y_in);
	float x_rightPlane = x_in + k_x / k_y * (y2_cylinder - y_in);

	float z_leftPlane = z_in + k_z / k_y * (y1_cylinder - y_in);
	float z_rightPlane = z_in + k_z / k_y * (y2_cylinder - y_in);

	float tmin = (y1_cylinder - y_in) / k_y;
	float tmax = (y2_cylinder - y_in) / k_y;


	int flag_leftPlane = 0;
	int flag_rightPlane = 0;

	float t1 = 0;
	float t2 = 0;

	if (((x_leftPlane - x_cylinder) * (x_leftPlane - x_cylinder) / x_rho / x_rho + (z_leftPlane - z_cylinder) * (z_leftPlane - z_cylinder) / z_rho / z_rho) <= 1)
	{
		flag_leftPlane = 1;
	}
	if (((x_rightPlane - x_cylinder) * (x_rightPlane - x_cylinder) / x_rho / x_rho + (z_rightPlane - z_cylinder) * (z_rightPlane - z_cylinder) / z_rho / z_rho) <= 1)
	{
		flag_rightPlane = 1;
	}


	if ((flag_rightPlane == 1) && (flag_leftPlane == 1))
	{
		if (tmin <= 0.0001 || tmax <= 0.0001)
			return 0.0;
		else if (tmin >= t_inout || tmax >= t_inout)
			return 0.0;
		else
			return fabs(tmax - tmin);
	}
	else
	{
		float x_ = x_in - x_cylinder;
		float z_ = z_in - z_cylinder;

		float a = (k_x * k_x * z_rho * z_rho + k_z * k_z * x_rho * x_rho);
		float b = k_x * x_ * z_rho * z_rho + k_z * z_ * x_rho * x_rho;
		float c = z_rho * z_rho * x_ * x_ + x_rho * x_rho * z_ * z_ - x_rho * x_rho * z_rho * z_rho;

		float Delta_ = (b * b - a * c);

		if (Delta_ <= 0.00001)
		{
			return 0.00000000;
		}
		else
		{
			t1 = (-b - sqrt(Delta_)) / a;
			t2 = (-b + sqrt(Delta_)) / a;
		}

		if ((flag_rightPlane == 0) && (flag_leftPlane == 0))
		{
			if ((t1 >= tmin) && (t2 <= tmax))
			{
				if (t2 <= 0.0001 || t1 <= 0.0001)
					return 0.0;
				else if (t2 >= t_inout || t1 >= t_inout)
					return 0.0;
				else
					return (t2 - t1);

			}
			else
			{
				return 0.000000;
			}

		}
		else if ((flag_leftPlane == 1) && (flag_rightPlane == 0))
		{
			if (t2 <= 0.0001 || tmin <= 0.0001)
				return 0.0;
			else if (t2 >= t_inout || tmin >= t_inout)
				return 0.0;
			else
				return (t2 - tmin);

		}
		else if ((flag_leftPlane == 0) && (flag_rightPlane == 1))
		{
			if (tmax <= 0.0001 || t1 <= 0.0001)
				return 0.0;
			else if (tmax >= t_inout || t1 >= t_inout)
				return 0.0;
			else
				return (tmax - t1);

		}
		else
		{
			return 0.000000;
		}
	}

}

// Device function to compute the differential Compton cross section
__device__ float diffComptonSection(float theta, float E0)
{
	// Calculate the cosine of the angle, in [0,2pi]
	float cos_theta = cos(theta);

	// Calculate the normalized energy alpha
	float alpha = E0 / 511.0f;

	// Factor1 and Factor2 computations
	float factor1 = alpha * (1.0f - cos_theta);
	float factor2 = 1.0f + cos_theta * cos_theta;

	// Calculate the result
	float result = factor2 / (1.0f + factor1) / (1.0f + factor1) * (1.0f + factor1 * factor1 / factor2 / (1.0f + factor1));

	return result;
}

// Compute the differential cross section integral
__device__ float computeComptonIntegral(float E0, float theta_low, float theta_high, float DeltaTheta)
{

	float total_cross_section = 0.0f;
	int numSteps = (int)((theta_high - theta_low) / DeltaTheta);

	for (int i = 0; i <= numSteps; ++i) {
		float theta = theta_low + i * DeltaTheta; 
		total_cross_section += diffComptonSection(theta, E0) * sin(theta) * DeltaTheta;  
	}

	return total_cross_section;

}

__global__ void initializeComptonNormalization(float source_energy)
{
	if (blockIdx.x == 0 && threadIdx.x == 0)
	{
		deviceComptonNormalization = computeComptonIntegral(
			source_energy, 0.0f, static_cast<float>(M_PI), 0.01f);
	}
}

__global__ void initializeComptonPhasePrefix(
	float* integrand_lut,
	float source_energy)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= kComptonPhasePrefixCount) return;
	float theta = index * kComptonPhaseStep;
	integrand_lut[index] = diffComptonSection(theta, source_energy) * sinf(theta)
		* kComptonIntegralStep;
}

__device__ inline float computeComptonIntegralPhasePrefix(
	const float* integrand_lut,
	float theta_low,
	float theta_high)
{
	int num_steps = static_cast<int>(
		(theta_high - theta_low) / kComptonIntegralStep);
	float total = 0.0f;
	for (int step = 0; step <= num_steps; ++step)
	{
		float theta = theta_low + step * kComptonIntegralStep;
		float position = theta / kComptonPhaseStep;
		int lower_index = static_cast<int>(floorf(position));
		if (lower_index < 0) lower_index = 0;
		if (lower_index >= kComptonPhasePrefixCount - 1)
			lower_index = kComptonPhasePrefixCount - 2;
		float fraction = position - lower_index;
		float lower = __ldg(&integrand_lut[lower_index]);
		float upper = __ldg(&integrand_lut[lower_index + 1]);
		total += lower + fraction * (upper - lower);
	}
	return total;
}

__global__ void validateComptonPhasePrefix(
	const float* prefix,
	float source_energy,
	float2* errors,
	int sample_count)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= sample_count) return;
	unsigned int low_code = static_cast<unsigned int>(index) * 73U % 1021U;
	float theta_low = static_cast<float>(M_PI) * low_code / 1021.0f;
	unsigned int width_code = static_cast<unsigned int>(index) * 193U % 1023U + 1U;
	float theta_high = theta_low + (static_cast<float>(M_PI) - theta_low)
		* width_code / 1024.0f;
	float reference = computeComptonIntegral(
		source_energy, theta_low, theta_high, kComptonIntegralStep);
	float candidate = computeComptonIntegralPhasePrefix(
		prefix, theta_low, theta_high);
	float absolute = fabsf(candidate - reference);
	float relative = absolute / fmaxf(fabsf(reference), 1e-12f);
	errors[index] = make_float2(absolute, relative);
}

// Calculate the cone angle theta
__device__ float calculateConeAngle(float xSource, float ySource, float zSource, float xA, float yA, float zA, float xDetector, float yDetector, float zDetector)
{
	// Source------> Point A (Compton Scatter) --------> Detector
	float vSourceA[3] = { xA - xSource, yA - ySource, zA - zSource };	
	float vAD[3] = { xDetector - xA, yDetector - yA, zDetector - zA };

	float dotProduct = vAD[0] * vSourceA[0] + vAD[1] * vSourceA[1] + vAD[2] * vSourceA[2];

	float magnitude_vSourceA = sqrt(vSourceA[0] * vSourceA[0] + vSourceA[1] * vSourceA[1] + vSourceA[2] * vSourceA[2]);
	float magnitude_vAD = sqrt(vAD[0] * vAD[0] + vAD[1] * vAD[1] + vAD[2] * vAD[2]);

	if (magnitude_vSourceA < 1e-9f || magnitude_vAD < 1e-9f)
	{
		return 0.0f;
	}

	float cosTheta = dotProduct / (magnitude_vSourceA * magnitude_vAD);

	if (cosTheta > 1.0f) cosTheta = 1.0f;
	if (cosTheta < -1.0f) cosTheta = -1.0f;

	return acos(cosTheta);
}

// Calculate the energy of scattered photon
__device__ float calculateScatterEnergy(float theta, float E0)
{

	float cosTheta = cos(theta);
	float E_scatter = E0 / (1.0f + (E0 / 511.0f) * (1.0f - cosTheta));

	return E_scatter;
}

__device__ float calculategaussianIntegral(float scatterEnergy, float energy_resolution_scatterphoton, float lowerThresholdofEnergyWindow, float upperThresholdofEnergyWindow)
{
	float sigma = energy_resolution_scatterphoton / 2.35482f * scatterEnergy;
	float sqrt2sigma = sigma * sqrt(2.0);
	float z1 = (lowerThresholdofEnergyWindow - scatterEnergy) / sqrt2sigma;
	float z2 = (upperThresholdofEnergyWindow - scatterEnergy) / sqrt2sigma;
	float probability = 0.5f * (erf(z2) - erf(z1));
	return probability;
}

__device__ inline float calculateDist(float x1, float y1, float z1, float x2, float y2, float z2)
{
	return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));
}

__device__ inline bool clipRayToBoxAxis(
	float origin,
	float direction,
	float half_extent,
	float* entry,
	float* exit)
{
	if (fabsf(direction) < 1e-8f)
		return origin >= -half_extent && origin <= half_extent;
	float first = (-half_extent - origin) / direction;
	float second = (half_extent - origin) / direction;
	if (first > second)
	{
		float temporary = first;
		first = second;
		second = temporary;
	}
	*entry = fmaxf(*entry, first);
	*exit = fminf(*exit, second);
	return *entry <= *exit;
}

__device__ inline float rayBoxChordLength(
	float origin_x,
	float origin_y,
	float origin_z,
	float direction_x,
	float direction_y,
	float direction_z,
	float half_width,
	float half_thickness,
	float half_height)
{
	float entry = -1.0e30f;
	float exit = 1.0e30f;
	if (!clipRayToBoxAxis(origin_x, direction_x, half_width, &entry, &exit)
		|| !clipRayToBoxAxis(origin_y, direction_y, half_thickness, &entry, &exit)
		|| !clipRayToBoxAxis(origin_z, direction_z, half_height, &entry, &exit))
		return 0.0f;
	entry = fmaxf(entry, 0.0f);
	return exit > entry ? exit - entry : 0.0f;
}

__device__ float integrateIntercrystalTargetSurface(
	float x_image,
	float y_image,
	float z_image,
	float fov_to_collimator,
	const float* detector,
	const int* detector_material,
	int scatter_index,
	int target_index,
	float source_energy,
	float target_relative_fwhm,
	float window_lower,
	float window_upper,
	float compton_normalization,
	const CrystalPairPath& pair_path,
	int face_subdivisions)
{
	if (face_subdivisions < 1 || !(compton_normalization > 0.0f)) return 0.0f;

	const float scatter_x = detector[scatter_index * 12 + 1];
	const float scatter_y = detector[scatter_index * 12 + 2] + fov_to_collimator;
	const float scatter_z = detector[scatter_index * 12 + 3];
	const float target_x = detector[target_index * 12 + 1];
	const float target_y = detector[target_index * 12 + 2] + fov_to_collimator;
	const float target_z = detector[target_index * 12 + 3];

	float incoming_x = scatter_x - x_image;
	float incoming_y = scatter_y - y_image;
	float incoming_z = scatter_z - z_image;
	const float incoming_distance = sqrtf(incoming_x * incoming_x
		+ incoming_y * incoming_y + incoming_z * incoming_z);
	if (!(incoming_distance > 0.0f)) return 0.0f;
	incoming_x /= incoming_distance;
	incoming_y /= incoming_distance;
	incoming_z /= incoming_distance;

	const float target_rotation = detector[target_index * 12 + 11];
	const float target_cosine = cosf(-target_rotation);
	const float target_sine = sinf(-target_rotation);
	const float scatter_target_x = (scatter_x - target_x) * target_cosine
		- (scatter_z - target_z) * target_sine;
	const float scatter_target_y = scatter_y - target_y;
	const float scatter_target_z = (scatter_x - target_x) * target_sine
		+ (scatter_z - target_z) * target_cosine;
	const float target_half_extent[3] = {
		0.5f * detector[target_index * 12 + 4],
		0.5f * detector[target_index * 12 + 5],
		0.5f * detector[target_index * 12 + 6]
	};
	const float scatter_target[3] = {
		scatter_target_x, scatter_target_y, scatter_target_z
	};

	const float scatter_rotation = detector[scatter_index * 12 + 11];
	const float scatter_cosine = cosf(-scatter_rotation);
	const float scatter_sine = sinf(-scatter_rotation);
	const float scatter_width = detector[scatter_index * 12 + 4];
	const float scatter_thickness = detector[scatter_index * 12 + 5];
	const float scatter_height = detector[scatter_index * 12 + 6];
	const int scatter_material = detector_material[scatter_index];
	const int target_material = detector_material[target_index];
	if (scatter_material < 0 || target_material < 0) return 0.0f;

	const float cell_fraction = 1.0f
		/ static_cast<float>(face_subdivisions * face_subdivisions);
	const float inverse_compton_norm = 1.0f
		/ (2.0f * static_cast<float>(M_PI) * compton_normalization);
	float contribution = 0.0f;

	// For a convex box viewed from an external point, at most one face per
	// axis is visible. Their projected solid angles tile the target's exact
	// solid angle without the enclosing-sphere azimuth approximation.
	for (int normal_axis = 0; normal_axis < 3; ++normal_axis)
	{
		if (fabsf(scatter_target[normal_axis])
			<= target_half_extent[normal_axis] + 1e-6f)
			continue;
		const float normal_sign = scatter_target[normal_axis] > 0.0f ? 1.0f : -1.0f;
		const int first_axis = (normal_axis + 1) % 3;
		const int second_axis = (normal_axis + 2) % 3;
		const float face_area = 4.0f * target_half_extent[first_axis]
			* target_half_extent[second_axis];
		const float cell_area = face_area * cell_fraction;

		for (int first_index = 0; first_index < face_subdivisions; ++first_index)
		{
			for (int second_index = 0; second_index < face_subdivisions; ++second_index)
			{
				float sample_local[3] = {0.0f, 0.0f, 0.0f};
				sample_local[normal_axis] = normal_sign
					* target_half_extent[normal_axis];
				sample_local[first_axis] = -target_half_extent[first_axis]
					+ (first_index + 0.5f) * 2.0f
					* target_half_extent[first_axis] / face_subdivisions;
				sample_local[second_axis] = -target_half_extent[second_axis]
					+ (second_index + 0.5f) * 2.0f
					* target_half_extent[second_axis] / face_subdivisions;

				const float ray_local_x = sample_local[0] - scatter_target_x;
				const float ray_local_y = sample_local[1] - scatter_target_y;
				const float ray_local_z = sample_local[2] - scatter_target_z;
				const float distance_squared = ray_local_x * ray_local_x
					+ ray_local_y * ray_local_y + ray_local_z * ray_local_z;
				if (!(distance_squared > 0.0f)) continue;
				const float distance = sqrtf(distance_squared);
				const float direction_local_x = ray_local_x / distance;
				const float direction_local_y = ray_local_y / distance;
				const float direction_local_z = ray_local_z / distance;

				const float direction_x = direction_local_x * cosf(target_rotation)
					- direction_local_z * sinf(target_rotation);
				const float direction_y = direction_local_y;
				const float direction_z = direction_local_x * sinf(target_rotation)
					+ direction_local_z * cosf(target_rotation);
				const float projected_cosine = -(normal_axis == 0
					? normal_sign * direction_local_x
					: (normal_axis == 1 ? normal_sign * direction_local_y
						: normal_sign * direction_local_z));
				if (!(projected_cosine > 0.0f)) continue;
				const float solid_angle = projected_cosine * cell_area / distance_squared;

				float cosine_theta = incoming_x * direction_x
					+ incoming_y * direction_y + incoming_z * direction_z;
				cosine_theta = fminf(1.0f, fmaxf(-1.0f, cosine_theta));
				const float theta = acosf(cosine_theta);
				const float scattered_energy = calculateScatterEnergy(theta, source_energy);
				const float scattered_relative_fwhm = target_relative_fwhm
					* sqrtf(source_energy / scattered_energy);
				const float window_acceptance = calculategaussianIntegral(
					scattered_energy, scattered_relative_fwhm,
					window_lower, window_upper);
				if (!(window_acceptance > 0.0f)) continue;

				float source_mu_pe = 0.0f;
				float source_mu_compton = 0.0f;
				interpolateXcomDevice(scatter_material, scattered_energy,
					&source_mu_pe, &source_mu_compton);
				const float source_direction_x = direction_x * scatter_cosine
					- direction_z * scatter_sine;
				const float source_direction_z = direction_x * scatter_sine
					+ direction_z * scatter_cosine;
				const float source_exit_length = detectorCenterExitDistance(
					source_direction_x, direction_y, source_direction_z,
					scatter_width, scatter_thickness, scatter_height);
				float attenuation = source_exit_length
					* (source_mu_pe + source_mu_compton);

				float material_mu_pe = 0.0f;
				float material_mu_compton = 0.0f;
				if (pair_path.material_lengths.x > 0.0f)
				{
					interpolateXcomDevice(kMaterialNaI, scattered_energy,
						&material_mu_pe, &material_mu_compton);
					attenuation += pair_path.material_lengths.x
						* (material_mu_pe + material_mu_compton);
				}
				if (pair_path.material_lengths.y > 0.0f)
				{
					interpolateXcomDevice(kMaterialGAGG, scattered_energy,
						&material_mu_pe, &material_mu_compton);
					attenuation += pair_path.material_lengths.y
						* (material_mu_pe + material_mu_compton);
				}
				if (pair_path.material_lengths.z > 0.0f)
				{
					interpolateXcomDevice(kMaterialPb, scattered_energy,
						&material_mu_pe, &material_mu_compton);
					attenuation += pair_path.material_lengths.z
						* (material_mu_pe + material_mu_compton);
				}
				if (pair_path.material_lengths.w > 0.0f)
				{
					interpolateXcomDevice(kMaterialW, scattered_energy,
						&material_mu_pe, &material_mu_compton);
					attenuation += pair_path.material_lengths.w
						* (material_mu_pe + material_mu_compton);
				}

				float target_mu_pe = 0.0f;
				float target_mu_compton = 0.0f;
				interpolateXcomDevice(target_material, scattered_energy,
					&target_mu_pe, &target_mu_compton);
				const float target_mu_total = target_mu_pe + target_mu_compton;
				if (!(target_mu_total > 0.0f)) continue;
				const float target_chord = rayBoxChordLength(
					scatter_target_x, scatter_target_y, scatter_target_z,
					direction_local_x, direction_local_y, direction_local_z,
					target_half_extent[0], target_half_extent[1],
					target_half_extent[2]);
				if (!(target_chord > 0.0f)) continue;
				const float target_photoelectric = (1.0f - expf(-target_mu_total
					* target_chord)) * target_mu_pe / target_mu_total;

				const float angular_density = diffComptonSection(theta, source_energy)
					* inverse_compton_norm;
				const float sample_contribution = angular_density * solid_angle
					* window_acceptance * expf(-attenuation) * target_photoelectric;
				if (isfinite(sample_contribution) && sample_contribution > 0.0f)
					contribution += sample_contribution;
			}
		}
	}
	return contribution;
}

__device__ int indexFrombitmap_crystal(int i, int j, int k, unsigned int* d_bit_array, int numProjectionSingle, int bits_per_word)
{	
	if (i > j) 
	{
		int temp = i;
		i = j;
		j = temp;
	}
	
	long long pair_idx = static_cast<long long>(i) * numProjectionSingle - (static_cast<long long>(i) * (i + 1)) / 2 + j;
	long long bit_idx = pair_idx * numProjectionSingle + k;
	
	long long word_idx = bit_idx / bits_per_word;
	int bit_offset = bit_idx % bits_per_word;

	unsigned int word = d_bit_array[word_idx];
	int flag = 0;
	flag = (word >> bit_offset) & 1;
	return flag;
}

__device__ int indexFrombitmap_crystal_chunk(int local_i, int j, int k, unsigned int* d_bit_array, int numProjectionSingle, int bits_per_word)
{
	long long bit_idx = (static_cast<long long>(local_i) * numProjectionSingle + j) * numProjectionSingle + k;
	long long word_idx = bit_idx / bits_per_word;
	int bit_offset = bit_idx % bits_per_word;

	unsigned int word = d_bit_array[word_idx];
	return (word >> bit_offset) & 1;
}


__device__ int indexFrombitmap_collimator(int i, int j, int k, unsigned int* d_bit_array, int numProjectionSingle, int bits_per_word)
{

	long long bit_idx = static_cast<long long>(i) * static_cast<long long>(numProjectionSingle) * static_cast<long long>(numProjectionSingle) + static_cast<long long>(j) * static_cast<long long>(numProjectionSingle) + static_cast<long long>(k);

	long long word_idx = bit_idx / bits_per_word;
	int bit_offset = bit_idx % bits_per_word;

	unsigned int word = d_bit_array[word_idx];
	int flag = 0;
	flag = (word >> bit_offset) & 1;
	return flag;
}

__global__ void detectorLocalScatterSysMatCuda(
	float* dst,
	const float* deviceparameter_Detector,
	const float* deviceparameter_Image,
	const float* deviceparameter_Physics,
	const float* devicePESysMat,
	const int* deviceLocalScatterType,
	const float2* deviceLocalScatterLookup,
	int localScatterOrientationBins,
	int numProjectionSingle,
	int numImagebin,
	int componentMode)
{
	const int detector_index = blockIdx.x * blockDim.x + threadIdx.x;
	const int image_index = blockIdx.y * blockDim.y + threadIdx.y;
	if (detector_index >= numProjectionSingle || image_index >= numImagebin) return;
	const int type_id = deviceLocalScatterType[detector_index];
	if (type_id < 0) return;

	const bool enable_compton
		= floorf(deviceparameter_Physics[0] + 0.5f) > 0.0f;
	const bool physics_recoil = floorf(deviceparameter_Physics[10] + 0.5f) > 0.0f;
	const bool physics_self_photoelectric
		= floorf(deviceparameter_Physics[11] + 0.5f) > 0.0f;
	const bool enable_recoil = physics_recoil && componentMode != 2;
	const bool enable_self_photoelectric
		= physics_self_photoelectric && componentMode != 1;
	if (!enable_compton || (!enable_recoil && !enable_self_photoelectric)) return;

	const float mu_photoelectric_source
		= deviceparameter_Detector[detector_index * 12 + 8];
	const float mu_compton_source
		= deviceparameter_Detector[detector_index * 12 + 9];
	if (!(mu_photoelectric_source > 0.0f) || !(mu_compton_source > 0.0f)) return;
	const int matrix_index = detector_index * numImagebin + image_index;
	const float first_compton_probability = devicePESysMat[matrix_index]
		* mu_compton_source / mu_photoelectric_source;
	if (!(first_compton_probability > 0.0f)) return;

	const int image_x_count = static_cast<int>(floorf(deviceparameter_Image[0]));
	const int image_y_count = static_cast<int>(floorf(deviceparameter_Image[1]));
	const int image_z_index = image_index / (image_y_count * image_x_count);
	const int image_remainder = image_index % (image_y_count * image_x_count);
	const int image_y_index = image_remainder / image_x_count;
	const int image_x_index = image_remainder % image_x_count;
	float image_x = (image_x_index - image_x_count / 2.0f + 0.5f)
		* deviceparameter_Image[3] + deviceparameter_Image[8];
	float image_y = (image_y_index - image_y_count / 2.0f + 0.5f)
		* deviceparameter_Image[4] + deviceparameter_Image[9];
	float image_z = (image_z_index - floorf(deviceparameter_Image[2]) / 2.0f + 0.5f)
		* deviceparameter_Image[5] + deviceparameter_Image[10];
	const float rotation_angle = deviceparameter_Image[20] * deviceparameter_Image[7];
	const float rotated_image_x = image_x * cosf(rotation_angle)
		- image_y * sinf(rotation_angle);
	const float rotated_image_y = image_x * sinf(rotation_angle)
		+ image_y * cosf(rotation_angle);
	image_x = rotated_image_x;
	image_y = rotated_image_y;

	const float fov_to_detector = deviceparameter_Image[11];
	const float detector_x = deviceparameter_Detector[detector_index * 12 + 1];
	const float detector_y = deviceparameter_Detector[detector_index * 12 + 2]
		+ fov_to_detector;
	const float detector_z = deviceparameter_Detector[detector_index * 12 + 3];
	const float detector_rotation
		= deviceparameter_Detector[detector_index * 12 + 11];
	const float incoming_world_x = detector_x - image_x;
	const float incoming_world_y = detector_y - image_y;
	const float incoming_world_z = detector_z - image_z;
	const float incoming_local_x = incoming_world_x * cosf(-detector_rotation)
		- incoming_world_z * sinf(-detector_rotation);
	const float incoming_local_y = incoming_world_y;
	const float incoming_local_z = incoming_world_x * sinf(-detector_rotation)
		+ incoming_world_z * cosf(-detector_rotation);
	const float2 response = interpolateDetectorLocalScatterLookup(
		deviceLocalScatterLookup, type_id, localScatterOrientationBins,
		incoming_local_x, incoming_local_y, incoming_local_z);
	float contribution = 0.0f;
	if (enable_recoil) contribution += response.x;
	if (enable_self_photoelectric) contribution += response.y;
	contribution *= first_compton_probability;
	if (isfinite(contribution) && contribution > 0.0f)
		dst[matrix_index] += contribution;
}


// Retained only as a numerical-reference implementation. Production launches
// crystalScatterSurfaceSysMatCuda below and never calls this bounding-sphere
// approximation.
__global__ void crystalScatterBoundingSphereLegacyCuda(float* dst,
		float* deviceparameter_Detector,
		const int* deviceDetectorMaterial,
		float* deviceparameter_Image,
	float* deviceparameter_Physics,
	float* devicePESysMat,
	const float* deviceComptonPhasePrefix,
	const CrystalPairPath* deviceCrystalPairPaths,
	int numProjectionSingle,
	int numImagebin,
	int scatterStart,
	int scatterCount)

{
	// Calculate the primary compton scatter between crystals
	// Image---->Scatter crystal------>Detector crystal

		float _float_FOV2Collimator = deviceparameter_Image[11];
	
	//////////////////////////////////////////// Image Parameters ////////////////////////////////////////////

	float _float_widthImageVoxelX = deviceparameter_Image[3];
	float _float_widthImageVoxelY = deviceparameter_Image[4];
	float _float_widthImageVoxelZ = deviceparameter_Image[5];

	//float _float_numRotation = deviceparameter_Image[6];//numRotation;
	float _float_angelPerRotation = deviceparameter_Image[7];//Angel per Rotation;
	float _float_idxrotation = deviceparameter_Image[20];//idxRotation
	//float RotationAngle = _float_idxrotation / _float_numRotation * (2 * M_PI);
	float RotationAngle = _float_idxrotation * _float_angelPerRotation;
	float shiftFOVX_physics = deviceparameter_Image[8];
	float shiftFOVY_physics = deviceparameter_Image[9];
	float shiftFOVZ_physics = deviceparameter_Image[10];

	int numImageVoxelX = (int)floor(deviceparameter_Image[0]);
	int numImageVoxelY = (int)floor(deviceparameter_Image[1]);
	int numImageVoxelZ = (int)floor(deviceparameter_Image[2]);
	
	//////////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////// Threads Allocation ///////////////////////////////////////////

	long long int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < 0 || row > numProjectionSingle - 1) { return; }
	long long int col = blockIdx.y * blockDim.y + threadIdx.y;
	if (col < 0 || col > numImagebin - 1) { return; }
		long long int dstIndex = row * numImagebin + col;

		unsigned int idxDetector = row; // index of detector

	//Image Domain demensions=(z->y->x)
	int idxImageVoxelZ = col / (numImageVoxelY * numImageVoxelX);
	col = col % (numImageVoxelY * numImageVoxelX);
	int idxImageVoxelY = col / numImageVoxelX;
	int idxImageVoxelX = col % numImageVoxelX;


	// Finite Divisions of Detector Crystal
	const unsigned int divideX = 1, divideY = 1, divideZ = 1;


	///////////////////////////////////////// Physic Progress Parameters /////////////////////////////////////
	int flagUsingCompton = (int)floor(deviceparameter_Physics[0] + 0.5f);
	int flagUsingSameEnergyWindow = (int)floor(deviceparameter_Physics[4] + 0.5f);

	float lowerThresholdofEnergyWindow = deviceparameter_Physics[5];
	float upperThresholdofEnergyWindow = deviceparameter_Physics[6];

	float target_PE_Energy = deviceparameter_Physics[7];
	float energy_resolution_detector_targetPE = deviceparameter_Detector[idxDetector * 12 + 10];

	// Energy Window of detector crystal
	if (flagUsingSameEnergyWindow > 0)
	{
		lowerThresholdofEnergyWindow = deviceparameter_Physics[5];
		upperThresholdofEnergyWindow = deviceparameter_Physics[6];
	}
	else
	{
		lowerThresholdofEnergyWindow = (1 - energy_resolution_detector_targetPE / 2.0f) * target_PE_Energy;
		upperThresholdofEnergyWindow = (1 + energy_resolution_detector_targetPE / 2.0f) * target_PE_Energy;
	}

	float coeff_detector_total = deviceparameter_Detector[idxDetector * 12 + 7];

	float integration_Compton = deviceComptonNormalization;
	//////////////////////////////////////////////////////////////////////////////////////////////////////////


	///////////////////////////////////////// Image Rotation Shift Parameters /////////////////////////////////////
	float xImage = (idxImageVoxelX - numImageVoxelX / 2.0f + 0.5f) * _float_widthImageVoxelX;
	float yImage = (idxImageVoxelY - numImageVoxelY / 2.0f + 0.5f) * _float_widthImageVoxelY;
	float zImage = (idxImageVoxelZ - numImageVoxelZ / 2.0f + 0.5f) * _float_widthImageVoxelZ;

	xImage = xImage + shiftFOVX_physics;
	yImage = yImage + shiftFOVY_physics;
	zImage = zImage + shiftFOVZ_physics;

	float xImage_rot = xImage * cos(RotationAngle) - yImage * sin(RotationAngle);
	float yImage_rot = xImage * sin(RotationAngle) + yImage * cos(RotationAngle);
	float zImage_rot = zImage;
	xImage = xImage_rot;
	yImage = yImage_rot;
	zImage = zImage_rot;

	int ImageVoxel_index = idxImageVoxelX + idxImageVoxelY * numImageVoxelX + idxImageVoxelZ * numImageVoxelY * numImageVoxelX;

	// All variables without a suffix are in the real-world physical coordinate system
	// All parameters with 'self' suffix are in the detector crystal coordinate system
	float xDetectorCrystalCenter = deviceparameter_Detector[12 * idxDetector + 1];
	float yDetectorCrystalCenter = deviceparameter_Detector[12 * idxDetector + 2] + _float_FOV2Collimator;
	float zDetectorCrystalCenter = deviceparameter_Detector[12 * idxDetector + 3];
	
	float widthDetectorCrystal = deviceparameter_Detector[12 * idxDetector + 4];
	float heightDetectorCrystal = deviceparameter_Detector[12 * idxDetector + 6];
	float thicknessDetectorCrystal = deviceparameter_Detector[12 * idxDetector + 5];

	float rotationAngel_DetectorCrystal = deviceparameter_Detector[12 * idxDetector + 11];

	float xImage_self = (xImage - xDetectorCrystalCenter) * cos(-rotationAngel_DetectorCrystal) - (zImage - zDetectorCrystalCenter) * sin(-rotationAngel_DetectorCrystal);
	float yImage_self = yImage - yDetectorCrystalCenter;
	float zImage_self = (xImage - xDetectorCrystalCenter) * sin(-rotationAngel_DetectorCrystal) + (zImage - zDetectorCrystalCenter) * cos(-rotationAngel_DetectorCrystal);

		float chunk_contribution = 0.0f;
		for (int slice = 0; slice < scatterCount; ++slice)
		{
			unsigned int id_Detector = scatterStart + slice;
			if (idxDetector == id_Detector) continue;
			const CrystalPairPath pair_path
				= deviceCrystalPairPaths[slice * numProjectionSingle + idxDetector];
			if ((pair_path.flags & kCrystalPairKinematicallyAllowed) == 0U) continue;

		///////////////  Probability of Compton Scatter Happened on scatter crystal id_Detector //////////////////
		int PESysMat_index = numImagebin * id_Detector + ImageVoxel_index;
	float scatter_coeff_pe = deviceparameter_Detector[id_Detector * 12 + 8];
	float scatter_coeff_compton = deviceparameter_Detector[id_Detector * 12 + 9];
		if (!(scatter_coeff_pe > 0.0f))
		{
			continue;
	}
	float prob_Compton_othercrystal = devicePESysMat[PESysMat_index]
		* scatter_coeff_compton / scatter_coeff_pe;
		if (prob_Compton_othercrystal <= 0.0f)
		{
			continue;
	}

	float x_scatter = deviceparameter_Detector[id_Detector * 12 + 1];
	float y_scatter = deviceparameter_Detector[id_Detector * 12 + 2] + _float_FOV2Collimator;
	float z_scatter = deviceparameter_Detector[id_Detector * 12 + 3];
	///////////////////////////////////////// CalCulation Starts Below ///////////////////////////////////////
	if (flagUsingCompton == 1)
	{
		if (coeff_detector_total > 0.01f)
		{
			//float prob_Compton_to_detectionCrystal = 0.000;
			
			for (int NumZ = 0; NumZ < divideZ; NumZ++)
			{
				for (int NumX = 0; NumX < divideX; NumX++)
				{
					for (int NumY = 0; NumY < divideY; NumY++)
					{

						/////////////////////////////////  Parameters of the detector unit ////////////////////////////////
						// All variables without a suffix are in the real-world physical coordinate system
						// All parameters with 'self' suffix are in the detector crystal coordinate system
						float xDetector_self = -widthDetectorCrystal / 2.0f + (float)(NumX + 0.5f) / (float)divideX * widthDetectorCrystal;
						float zDetector_self = -heightDetectorCrystal / 2.0f + (float)(NumZ + 0.5f) / (float)divideZ * heightDetectorCrystal;
						float yDetector_self =  - thicknessDetectorCrystal / 2.0f + (float)(NumY + 0.5f) / (float)divideY * thicknessDetectorCrystal;

						float xDetector_rot = xDetector_self * cos(rotationAngel_DetectorCrystal) - zDetector_self * sin(rotationAngel_DetectorCrystal);
						float zDetector_rot = xDetector_self * sin(rotationAngel_DetectorCrystal) + zDetector_self * cos(rotationAngel_DetectorCrystal);
						float yDetector_rot = yDetector_self;

						float xDetector = xDetectorCrystalCenter + xDetector_rot;
						float zDetector = zDetectorCrystalCenter + zDetector_rot;
						float yDetector = yDetectorCrystalCenter + yDetector_rot;


						/////////////////////////////////  Compton scatter probability from the other crystal to detection crystal /////////////////////////////
						float length = 0;
						float comptonConeAngle = calculateConeAngle(xImage, yImage, zImage, x_scatter, y_scatter, z_scatter, xDetector, yDetector, zDetector);
						float scatterEnergy = calculateScatterEnergy(comptonConeAngle, target_PE_Energy);
						// Detector energy resolution is stored as relative FWHM at target_PE_Energy.
						// For scintillation statistics, relative FWHM scales as 1/sqrt(E).
						float energy_resolution_detector_scatterphoton = energy_resolution_detector_targetPE * sqrt(target_PE_Energy / scatterEnergy);

						//  The probability that a Compton scatterred photon being detected within the energy window of detector unit

						if (((scatterEnergy * (1 + 2 * energy_resolution_detector_scatterphoton / 2.35482f)) <= lowerThresholdofEnergyWindow) || (scatterEnergy * (1 - 2 * energy_resolution_detector_scatterphoton / 2.35482f) >= upperThresholdofEnergyWindow))
						{
							continue; 
							// The energy of the scattered photon detected within a detector element follows a Gaussian distribution. 
							// If the 2 sigma range of this Gaussian does not overlap with the full energy peak window of the detector element, 
							// then it is considered that the scattering does not affect the result.
						}
						float energyDetected_probability = calculategaussianIntegral(scatterEnergy, energy_resolution_detector_scatterphoton, lowerThresholdofEnergyWindow, upperThresholdofEnergyWindow);
								
								
						// The probability that a Compton scattered photon, among all the photons scattered at the scattering point, 
						// is scattered towards the direction of the detector element.
							float L_comptonAngle = pair_path.direction_distance.w;
							if (!(L_comptonAngle > 0.0f)) continue;
							float scatter_exit_mu_pe = 0.0f;
							float scatter_exit_mu_compton = 0.0f;
							interpolateXcomDevice(deviceDetectorMaterial[id_Detector], scatterEnergy,
								&scatter_exit_mu_pe, &scatter_exit_mu_compton);
							float scatter_crystal_exit_attenuation = pair_path.source_exit_length
								* (scatter_exit_mu_pe + scatter_exit_mu_compton);

						// Calculate the phi range, using the detector unit's minimum enclosing sphere as an approximation.
						float R_detector = sqrt(widthDetectorCrystal * widthDetectorCrystal / (float)divideX / (float)divideX + heightDetectorCrystal * heightDetectorCrystal / (float)divideZ / (float)divideZ + thicknessDetectorCrystal * thicknessDetectorCrystal / (float)divideY / (float)divideY) / 2.0f;
						float Range_Phi = 0.000f;
						if (L_comptonAngle * sin(comptonConeAngle) * 2.0f <= R_detector)
						{
							Range_Phi = 2.0f * M_PI;
						}
						else
						{
							Range_Phi = 4.0f * asin(min(R_detector / 2.0f / L_comptonAngle / sin(comptonConeAngle), 1.0f));
						}
								
						// Calculate the theta range
						float x_scatter_self = (x_scatter - xDetectorCrystalCenter) * cos(-rotationAngel_DetectorCrystal) - (z_scatter - zDetectorCrystalCenter) * sin(-rotationAngel_DetectorCrystal);
						float y_scatter_self = y_scatter - yDetectorCrystalCenter;
						float z_scatter_self = (x_scatter - xDetectorCrystalCenter) * sin(-rotationAngel_DetectorCrystal) + (z_scatter - zDetectorCrystalCenter) * cos(-rotationAngel_DetectorCrystal);
								
						float x1_detectorunit_self = ((float)NumX / (float)divideX - 0.5f) * widthDetectorCrystal;
						float x2_detectorunit_self = (((float)NumX + 1.0f) / (float)divideX - 0.5f) * widthDetectorCrystal;

						float y1_detectorunit_self = ((float)NumY / (float)divideY - 0.5f) * thicknessDetectorCrystal;
						float y2_detectorunit_self = (((float)NumY + 1.0f) / (float)divideY - 0.5f) * thicknessDetectorCrystal;

						float z1_detectorunit_self = ((float)NumZ / (float)divideZ - 0.5f) * heightDetectorCrystal;
						float z2_detectorunit_self = (((float)NumZ + 1.0f) / (float)divideZ - 0.5f) * heightDetectorCrystal;


						float dist_extend = 1000.0f;
						float dist_Image_scatterer = calculateDist(x_scatter_self, y_scatter_self, z_scatter_self, xImage_self, yImage_self, zImage_self);
						float x_tmp = x_scatter_self + dist_extend * (x_scatter_self - xImage_self) / dist_Image_scatterer;
						float y_tmp = y_scatter_self + dist_extend * (y_scatter_self - yImage_self) / dist_Image_scatterer;
						float z_tmp = z_scatter_self + dist_extend * (z_scatter_self - zImage_self) / dist_Image_scatterer;

						length = length_box_ray(xImage_self, yImage_self, zImage_self, x_tmp, y_tmp, z_tmp, x1_detectorunit_self, y1_detectorunit_self, z1_detectorunit_self, x2_detectorunit_self, y2_detectorunit_self, z2_detectorunit_self);
									
						float theta[8];
						theta[0] = calculateConeAngle(xImage_self, yImage_self, zImage_self, x_scatter_self, y_scatter_self, z_scatter_self, x1_detectorunit_self, y1_detectorunit_self, z1_detectorunit_self);
						theta[1] = calculateConeAngle(xImage_self, yImage_self, zImage_self, x_scatter_self, y_scatter_self, z_scatter_self, x2_detectorunit_self, y1_detectorunit_self, z1_detectorunit_self);
						theta[2] = calculateConeAngle(xImage_self, yImage_self, zImage_self, x_scatter_self, y_scatter_self, z_scatter_self, x1_detectorunit_self, y2_detectorunit_self, z1_detectorunit_self);
						theta[3] = calculateConeAngle(xImage_self, yImage_self, zImage_self, x_scatter_self, y_scatter_self, z_scatter_self, x1_detectorunit_self, y1_detectorunit_self, z2_detectorunit_self);
						theta[4] = calculateConeAngle(xImage_self, yImage_self, zImage_self, x_scatter_self, y_scatter_self, z_scatter_self, x2_detectorunit_self, y2_detectorunit_self, z1_detectorunit_self);
						theta[5] = calculateConeAngle(xImage_self, yImage_self, zImage_self, x_scatter_self, y_scatter_self, z_scatter_self, x2_detectorunit_self, y1_detectorunit_self, z2_detectorunit_self);
						theta[6] = calculateConeAngle(xImage_self, yImage_self, zImage_self, x_scatter_self, y_scatter_self, z_scatter_self, x1_detectorunit_self, y2_detectorunit_self, z2_detectorunit_self);
						theta[7] = calculateConeAngle(xImage_self, yImage_self, zImage_self, x_scatter_self, y_scatter_self, z_scatter_self, x2_detectorunit_self, y2_detectorunit_self, z2_detectorunit_self);

						float theta_min = theta[0];
						float theta_max = theta[0];
						for (int i = 1; i < 8; i++)
						{
							if (theta[i] > theta_max)
								theta_max = theta[i];
							if (theta[i] < theta_min)
								theta_min = theta[i];
						}
									
						if (length > 0.001f) 
						{
							theta_min=0.000; // If the extension of the line from the image to the scatterer passes through the detector unit, then theta_min=0
						}
									
						// Range_Theta = 2.0f * asin(min(1.0f, R_detector / R_comptonAngle));
							float interval_compton = deviceComptonPhasePrefix != NULL
								? computeComptonIntegralPhasePrefix(
									deviceComptonPhasePrefix, theta_min, theta_max)
								: computeComptonIntegral(
									target_PE_Energy, theta_min, theta_max, kComptonIntegralStep);
							float comptonAngleRatio = interval_compton / integration_Compton;

						/////////////////////////////////  Attenuation from the scatterer crystal to detector unit /////////////////////////////
						float attenuation_dist_crystal_crystal
							= scatter_crystal_exit_attenuation;
								
							float material_mu_pe = 0.0f;
							float material_mu_compton = 0.0f;
							if (pair_path.material_lengths.x > 0.0f)
							{
								interpolateXcomDevice(kMaterialNaI, scatterEnergy,
									&material_mu_pe, &material_mu_compton);
								attenuation_dist_crystal_crystal += pair_path.material_lengths.x
									* (material_mu_pe + material_mu_compton);
							}
							if (pair_path.material_lengths.y > 0.0f)
							{
								interpolateXcomDevice(kMaterialGAGG, scatterEnergy,
									&material_mu_pe, &material_mu_compton);
								attenuation_dist_crystal_crystal += pair_path.material_lengths.y
									* (material_mu_pe + material_mu_compton);
							}
							if (pair_path.material_lengths.z > 0.0f)
							{
								interpolateXcomDevice(kMaterialPb, scatterEnergy,
									&material_mu_pe, &material_mu_compton);
								attenuation_dist_crystal_crystal += pair_path.material_lengths.z
									* (material_mu_pe + material_mu_compton);
							}
							if (pair_path.material_lengths.w > 0.0f)
							{
								interpolateXcomDevice(kMaterialW, scatterEnergy,
									&material_mu_pe, &material_mu_compton);
								attenuation_dist_crystal_crystal += pair_path.material_lengths.w
									* (material_mu_pe + material_mu_compton);
							}


							float absorp_coeff_detector_pe = 0.0f;
							float absorp_coeff_detector_compton = 0.0f;
							interpolateXcomDevice(deviceDetectorMaterial[idxDetector], scatterEnergy,
								&absorp_coeff_detector_pe, &absorp_coeff_detector_compton);
							float absorp_coeff_detector_total = absorp_coeff_detector_pe + absorp_coeff_detector_compton;
							float length_absorp = pair_path.target_absorption_length;

						//prob_Compton_to_detectionCrystal += prob_Compton_othercrystal * Range_Phi / 2.0f / M_PI * comptonAngleRatio * energyDetected_probability * exp(-attenuation_dist_crystal_crystal) * (1.0f-exp(-length_absorp* absorp_coeff_detector_total))* absorp_coeff_detector_pe/ absorp_coeff_detector_total;																		
						float contrib = prob_Compton_othercrystal* Range_Phi / 2.0f / M_PI * comptonAngleRatio * energyDetected_probability * exp(-attenuation_dist_crystal_crystal) * (1.0f - exp(-length_absorp * absorp_coeff_detector_total)) * absorp_coeff_detector_pe / absorp_coeff_detector_total;
						if (isfinite(contrib) && contrib > 0.0f)
						{
								chunk_contribution += contrib;
						}
					}
				}
			}
			//atomicAdd(&dst[dstIndex], prob_Compton_to_detectionCrystal);
		}

		}
		}
		if (chunk_contribution > 0.0f) dst[dstIndex] += chunk_contribution;


	}


__global__ void crystalScatterSurfaceSysMatCuda(
	float* dst,
	float* activeIntercrystalComponent,
	const float* deviceparameter_Detector,
	const int* deviceDetectorMaterial,
	const float* deviceparameter_Image,
	const float* deviceparameter_Physics,
	const float* devicePESysMat,
	const CrystalPairPath* deviceCrystalPairPaths,
	int numProjectionSingle,
	int numImagebin,
	int scatterStart,
	int scatterCount,
	int targetFaceSubdivisions,
	int nearTargetFaceSubdivisions,
	float nearTargetDistanceFactor)
{
	// Image -> first-Compton crystal -> target crystal. Every target surface
	// quadrature point carries its own theta, E'(theta), window acceptance,
	// source exit length, and target ray-box absorption length.
	const int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < 0 || row >= numProjectionSingle) return;
	const int image_index = blockIdx.y * blockDim.y + threadIdx.y;
	if (image_index < 0 || image_index >= numImagebin) return;
	if (static_cast<int>(floorf(deviceparameter_Physics[0] + 0.5f)) != 1) return;
	if (!(deviceparameter_Detector[row * 12 + 7] > 0.01f)) return;

	const int num_x = static_cast<int>(floorf(deviceparameter_Image[0]));
	const int num_y = static_cast<int>(floorf(deviceparameter_Image[1]));
	const int index_z = image_index / (num_y * num_x);
	const int in_slice = image_index % (num_y * num_x);
	const int index_y = in_slice / num_x;
	const int index_x = in_slice % num_x;
	float image_x = (index_x - num_x / 2.0f + 0.5f) * deviceparameter_Image[3]
		+ deviceparameter_Image[8];
	float image_y = (index_y - num_y / 2.0f + 0.5f) * deviceparameter_Image[4]
		+ deviceparameter_Image[9];
	float image_z = (index_z - floorf(deviceparameter_Image[2]) / 2.0f + 0.5f)
		* deviceparameter_Image[5] + deviceparameter_Image[10];
	const float rotation = deviceparameter_Image[20] * deviceparameter_Image[7];
	const float rotated_x = image_x * cosf(rotation) - image_y * sinf(rotation);
	const float rotated_y = image_x * sinf(rotation) + image_y * cosf(rotation);
	image_x = rotated_x;
	image_y = rotated_y;

	const float source_energy = deviceparameter_Physics[7];
	const float target_relative_fwhm = deviceparameter_Detector[row * 12 + 10];
	float window_lower = deviceparameter_Physics[5];
	float window_upper = deviceparameter_Physics[6];
	if (static_cast<int>(floorf(deviceparameter_Physics[4] + 0.5f)) == 0)
	{
		window_lower = (1.0f - target_relative_fwhm / 2.0f) * source_energy;
		window_upper = (1.0f + target_relative_fwhm / 2.0f) * source_energy;
	}

	float chunk_contribution = 0.0f;
	float active_intercrystal_contribution = 0.0f;
	for (int slice = 0; slice < scatterCount; ++slice)
	{
		const int scatter_index = scatterStart + slice;
		if (row == scatter_index) continue;
		const CrystalPairPath pair_path
			= deviceCrystalPairPaths[slice * numProjectionSingle + row];
		if ((pair_path.flags & kCrystalPairKinematicallyAllowed) == 0U) continue;

		const float scatter_mu_pe = deviceparameter_Detector[scatter_index * 12 + 8];
		const float scatter_mu_compton = deviceparameter_Detector[scatter_index * 12 + 9];
		if (!(scatter_mu_pe > 0.0f) || !(scatter_mu_compton > 0.0f)) continue;
		const float first_compton_probability
			= devicePESysMat[scatter_index * numImagebin + image_index]
			* scatter_mu_compton / scatter_mu_pe;
		if (!(first_compton_probability > 0.0f)) continue;

		float maximum_dimension = deviceparameter_Detector[row * 12 + 4];
		maximum_dimension = fmaxf(maximum_dimension,
			deviceparameter_Detector[row * 12 + 5]);
		maximum_dimension = fmaxf(maximum_dimension,
			deviceparameter_Detector[row * 12 + 6]);
		maximum_dimension = fmaxf(maximum_dimension,
			deviceparameter_Detector[scatter_index * 12 + 4]);
		maximum_dimension = fmaxf(maximum_dimension,
			deviceparameter_Detector[scatter_index * 12 + 5]);
		maximum_dimension = fmaxf(maximum_dimension,
			deviceparameter_Detector[scatter_index * 12 + 6]);
		const bool near_target = pair_path.direction_distance.w
			<= nearTargetDistanceFactor * maximum_dimension;
		const int subdivisions = near_target
			? nearTargetFaceSubdivisions : targetFaceSubdivisions;
		const float conditional_response = integrateIntercrystalTargetSurface(
			image_x, image_y, image_z, deviceparameter_Image[11],
			deviceparameter_Detector, deviceDetectorMaterial,
			scatter_index, row, source_energy, target_relative_fwhm,
			window_lower, window_upper, deviceComptonNormalization,
			pair_path, subdivisions);
		const float pair_contribution
			= first_compton_probability * conditional_response;
		chunk_contribution += pair_contribution;
		const int scatter_flag = static_cast<int>(floorf(
			deviceparameter_Detector[scatter_index * 12 + 12] + 0.5f));
		if (scatter_flag == 1) active_intercrystal_contribution += pair_contribution;
	}
	const long long output_index
		= static_cast<long long>(row) * numImagebin + image_index;
	if (isfinite(chunk_contribution) && chunk_contribution > 0.0f)
		dst[output_index] += chunk_contribution;
	if (activeIntercrystalComponent != NULL
		&& isfinite(active_intercrystal_contribution)
		&& active_intercrystal_contribution > 0.0f)
		activeIntercrystalComponent[output_index]
			+= active_intercrystal_contribution;
}

__global__ void geometryRelationShip_Crystal2Crystal(unsigned int * dst_relation_crystal2crystal, float* deviceparameter_Detector)
{
	// clculate global index
	int numProjectionSingle =(int) deviceparameter_Detector[0];
	long long idx = static_cast<long long>(blockIdx.x) * static_cast<long long>(blockDim.x) + static_cast<long long>(threadIdx.x);
	long long total_threads = static_cast<long long>(numProjectionSingle) * static_cast<long long>(numProjectionSingle) * static_cast<long long>(numProjectionSingle);

	if (idx >= total_threads) return;

	int k = idx % numProjectionSingle;
	int j = (idx / numProjectionSingle) % numProjectionSingle;
	int i = idx / (numProjectionSingle * numProjectionSingle);

	if (i >= j) return;
	if (k == i) return;
	if (k == j) return;

	long long pair_idx = static_cast<long long>(i) * static_cast<long long>(numProjectionSingle) - (static_cast<long long>(i) * static_cast<long long>((i + 1))) / 2 + static_cast<long long>(j);

	long long bit_idx = pair_idx * static_cast<long long>(numProjectionSingle) + static_cast<long long>(k);

	int bits_per_word = 32;
	long long word_idx = bit_idx / bits_per_word;
	int bit_offset = bit_idx % bits_per_word;

	float xDetector_i = deviceparameter_Detector[12 * i + 1];
	float yDetector_i = deviceparameter_Detector[12 * i + 2];
	float zDetector_i = deviceparameter_Detector[12 * i + 3];
	float widthDetector_i = deviceparameter_Detector[12 * i + 4];
	float heightDetector_i = deviceparameter_Detector[12 * i + 6];
	float thicknessDetector_i = deviceparameter_Detector[12 * i + 5];
	float R_detector_i = sqrt(widthDetector_i * widthDetector_i + heightDetector_i * heightDetector_i + thicknessDetector_i * thicknessDetector_i) / 2.0f;
	
	float xDetector_j = deviceparameter_Detector[12 * j + 1];
	float yDetector_j = deviceparameter_Detector[12 * j + 2];
	float zDetector_j = deviceparameter_Detector[12 * j + 3];
	float widthDetector_j = deviceparameter_Detector[12 * j + 4];
	float heightDetector_j = deviceparameter_Detector[12 * j + 6];
	float thicknessDetector_j  = deviceparameter_Detector[12 * j + 5];
	float R_detector_j = sqrt(widthDetector_j * widthDetector_j + heightDetector_j * heightDetector_j + thicknessDetector_j * thicknessDetector_j) / 2.0f;

	float L_ij = calculateDist(xDetector_i, yDetector_i, zDetector_i, xDetector_j, yDetector_j, zDetector_j);

	float crit_i_j = R_detector_i / L_ij;
	float crit_j_i = R_detector_j / L_ij;

	float x_projectionOnUnitSphere_i_j = (xDetector_i - xDetector_j) / L_ij;
	float y_projectionOnUnitSphere_i_j = (yDetector_i - yDetector_j) / L_ij;
	float z_projectionOnUnitSphere_i_j = (zDetector_i - zDetector_j) / L_ij;

	float x_projectionOnUnitSphere_j_i = (xDetector_j - xDetector_i) / L_ij;
	float y_projectionOnUnitSphere_j_i = (yDetector_j - yDetector_i) / L_ij;
	float z_projectionOnUnitSphere_j_i = (zDetector_j - zDetector_i) / L_ij;

	
	float xDetector_k = deviceparameter_Detector[12 * k + 1];
	float yDetector_k = deviceparameter_Detector[12 * k + 2];
	float zDetector_k = deviceparameter_Detector[12 * k + 3];
	float widthDetector_k = deviceparameter_Detector[12 * k + 4];
	float heightDetector_k = deviceparameter_Detector[12 * k + 6];
	float thicknessDetector_k = deviceparameter_Detector[12 * k + 5];
	float R_detector_k = sqrt(widthDetector_k * widthDetector_k + heightDetector_k * heightDetector_k + thicknessDetector_k * thicknessDetector_k) / 2.0f;

	
	float L_ik = calculateDist(xDetector_k, yDetector_k, zDetector_k, xDetector_i, yDetector_i, zDetector_i);
	float crit_k_i = R_detector_k / L_ik;

	float L_jk = calculateDist(xDetector_k, yDetector_k, zDetector_k, xDetector_j, yDetector_j, zDetector_j);
	float crit_k_j = R_detector_k / L_jk;

	float x_projectionOnUnitSphere_k_i = (xDetector_k - xDetector_i) / L_ik;
	float y_projectionOnUnitSphere_k_i = (yDetector_k - yDetector_i) / L_ik;
	float z_projectionOnUnitSphere_k_i = (zDetector_k - zDetector_i) / L_ik;

	float x_projectionOnUnitSphere_k_j = (xDetector_k - xDetector_j) / L_jk;
	float y_projectionOnUnitSphere_k_j = (yDetector_k - yDetector_j) / L_jk;
	float z_projectionOnUnitSphere_k_j = (zDetector_k - zDetector_j) / L_jk;
	
	
	int flagcross = 0;
	// Whether the cover sphere of detector k is cross with the line between i and j, centered at i 
	float distOnUnitSphere_i = calculateDist(x_projectionOnUnitSphere_k_i, y_projectionOnUnitSphere_k_i, z_projectionOnUnitSphere_k_i, x_projectionOnUnitSphere_j_i, y_projectionOnUnitSphere_j_i, z_projectionOnUnitSphere_j_i);
	if (distOnUnitSphere_i <= crit_k_i + crit_j_i)
	{
		flagcross = 1;
	}
	else
	{	
		// Whether the cover sphere of detector k is cross with the line between i and j, centered at j 
		float distOnUnitSphere_j = calculateDist(x_projectionOnUnitSphere_k_j, y_projectionOnUnitSphere_k_j, z_projectionOnUnitSphere_k_j, x_projectionOnUnitSphere_i_j, y_projectionOnUnitSphere_i_j, z_projectionOnUnitSphere_i_j);
		if (distOnUnitSphere_j <= crit_k_j + crit_i_j)
		{
			flagcross = 1;
		}
	}

	if (flagcross==1)
	{		
		atomicOr(&dst_relation_crystal2crystal[word_idx], 1U << bit_offset);
	}
}

__global__ void geometryRelationShip_Crystal2Crystal_Chunk(
	unsigned int* dst_relation_crystal2crystal,
	float* deviceparameter_Detector,
	const CrystalPairPath* pair_paths,
	int scatterStart,
	int scatterCount)
{
	int numProjectionSingle = (int)deviceparameter_Detector[0];
	long long idx = static_cast<long long>(blockIdx.x) * static_cast<long long>(blockDim.x) + static_cast<long long>(threadIdx.x);
	long long total_threads = static_cast<long long>(scatterCount) * static_cast<long long>(numProjectionSingle) * static_cast<long long>(numProjectionSingle);
	bool flagcross = false;
	if (idx < total_threads)
	{
		int k = idx % numProjectionSingle;
		int j = (idx / numProjectionSingle) % numProjectionSingle;
		int local_i = idx / (static_cast<long long>(numProjectionSingle)
			* static_cast<long long>(numProjectionSingle));
		int i = scatterStart + local_i;
		long long pair_index = static_cast<long long>(local_i) * numProjectionSingle + j;
		bool eligible = i < numProjectionSingle && i != j && k != i && k != j
			&& deviceparameter_Detector[12 * i + 7] > 0.01f
			&& (pair_paths[pair_index].flags
				& kCrystalPairKinematicallyAllowed) != 0U;
		if (eligible)
		{
			float x_i = deviceparameter_Detector[12 * i + 1];
			float y_i = deviceparameter_Detector[12 * i + 2];
			float z_i = deviceparameter_Detector[12 * i + 3];
			float width_i = deviceparameter_Detector[12 * i + 4];
			float thickness_i = deviceparameter_Detector[12 * i + 5];
			float height_i = deviceparameter_Detector[12 * i + 6];
			float radius_i = sqrt(width_i * width_i + height_i * height_i
				+ thickness_i * thickness_i) / 2.0f;

			float x_j = deviceparameter_Detector[12 * j + 1];
			float y_j = deviceparameter_Detector[12 * j + 2];
			float z_j = deviceparameter_Detector[12 * j + 3];
			float width_j = deviceparameter_Detector[12 * j + 4];
			float thickness_j = deviceparameter_Detector[12 * j + 5];
			float height_j = deviceparameter_Detector[12 * j + 6];
			float radius_j = sqrt(width_j * width_j + height_j * height_j
				+ thickness_j * thickness_j) / 2.0f;
			float distance_ij = calculateDist(x_i, y_i, z_i, x_j, y_j, z_j);
			if (distance_ij > 0.0f)
			{
				float crit_i_j = radius_i / distance_ij;
				float crit_j_i = radius_j / distance_ij;
				float projection_i_j_x = (x_i - x_j) / distance_ij;
				float projection_i_j_y = (y_i - y_j) / distance_ij;
				float projection_i_j_z = (z_i - z_j) / distance_ij;
				float projection_j_i_x = -projection_i_j_x;
				float projection_j_i_y = -projection_i_j_y;
				float projection_j_i_z = -projection_i_j_z;

				float x_k = deviceparameter_Detector[12 * k + 1];
				float y_k = deviceparameter_Detector[12 * k + 2];
				float z_k = deviceparameter_Detector[12 * k + 3];
				float width_k = deviceparameter_Detector[12 * k + 4];
				float thickness_k = deviceparameter_Detector[12 * k + 5];
				float height_k = deviceparameter_Detector[12 * k + 6];
				float radius_k = sqrt(width_k * width_k + height_k * height_k
					+ thickness_k * thickness_k) / 2.0f;
				float distance_ik = calculateDist(x_k, y_k, z_k, x_i, y_i, z_i);
				float distance_jk = calculateDist(x_k, y_k, z_k, x_j, y_j, z_j);
				if (distance_ik > 0.0f && distance_jk > 0.0f)
				{
					float projection_k_i_x = (x_k - x_i) / distance_ik;
					float projection_k_i_y = (y_k - y_i) / distance_ik;
					float projection_k_i_z = (z_k - z_i) / distance_ik;
					float projection_k_j_x = (x_k - x_j) / distance_jk;
					float projection_k_j_y = (y_k - y_j) / distance_jk;
					float projection_k_j_z = (z_k - z_j) / distance_jk;
					float distance_on_i = calculateDist(
						projection_k_i_x, projection_k_i_y, projection_k_i_z,
						projection_j_i_x, projection_j_i_y, projection_j_i_z);
					flagcross = distance_on_i <= radius_k / distance_ik + crit_j_i;
					if (!flagcross)
					{
						float distance_on_j = calculateDist(
							projection_k_j_x, projection_k_j_y, projection_k_j_z,
							projection_i_j_x, projection_i_j_y, projection_i_j_z);
						flagcross = distance_on_j <= radius_k / distance_jk + crit_i_j;
					}
				}
			}
		}
	}

	unsigned int active = __activemask();
	unsigned int word = __ballot_sync(active, flagcross);
	if ((threadIdx.x & 31) == 0 && idx < total_threads)
	{
		dst_relation_crystal2crystal[idx >> 5] = word;
	}
}

__global__ void initializeCrystalPairPaths(
	CrystalPairPath* pair_paths,
	const float* deviceparameter_Detector,
	const float* deviceparameter_Image,
	const float* deviceparameter_Physics,
	int numProjectionSingle,
	int scatterStart,
	int scatterCount,
	int enable_kinematic_pruning)
{
	long long index = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
	long long pair_count = static_cast<long long>(scatterCount) * numProjectionSingle;
	if (index >= pair_count) return;

	int local_scatter = static_cast<int>(index / numProjectionSingle);
	int target = static_cast<int>(index % numProjectionSingle);
	int scatter = scatterStart + local_scatter;
	CrystalPairPath path = {};
	if (scatter == target || scatter >= numProjectionSingle)
	{
		pair_paths[index] = path;
		return;
	}

	float scatter_x = deviceparameter_Detector[scatter * 12 + 1];
	float scatter_y = deviceparameter_Detector[scatter * 12 + 2];
	float scatter_z = deviceparameter_Detector[scatter * 12 + 3];
	float target_x = deviceparameter_Detector[target * 12 + 1];
	float target_y = deviceparameter_Detector[target * 12 + 2];
	float target_z = deviceparameter_Detector[target * 12 + 3];
	float dx = target_x - scatter_x;
	float dy = target_y - scatter_y;
	float dz = target_z - scatter_z;
	float distance = sqrtf(dx * dx + dy * dy + dz * dz);
	if (!(distance > 0.0f))
	{
		pair_paths[index] = path;
		return;
	}

	float outgoing_x = dx / distance;
	float outgoing_y = dy / distance;
	float outgoing_z = dz / distance;
	path.direction_distance = make_float4(
		outgoing_x, outgoing_y, outgoing_z, distance);
	bool kinematically_allowed = true;
	if (enable_kinematic_pruning != 0)
	{
		float rotation = deviceparameter_Image[20] * deviceparameter_Image[7];
		float center_x = deviceparameter_Image[8] * cosf(rotation)
			- deviceparameter_Image[9] * sinf(rotation);
		float center_y = deviceparameter_Image[8] * sinf(rotation)
			+ deviceparameter_Image[9] * cosf(rotation);
		float center_z = deviceparameter_Image[10];
		float scatter_world_y = scatter_y + deviceparameter_Image[11];
		float incoming_x = scatter_x - center_x;
		float incoming_y = scatter_world_y - center_y;
		float incoming_z = scatter_z - center_z;
		float incoming_distance = sqrtf(incoming_x * incoming_x
			+ incoming_y * incoming_y + incoming_z * incoming_z);
		float extent_x = 0.5f * (deviceparameter_Image[0] - 1.0f)
			* deviceparameter_Image[3];
		float extent_y = 0.5f * (deviceparameter_Image[1] - 1.0f)
			* deviceparameter_Image[4];
		float extent_z = 0.5f * (deviceparameter_Image[2] - 1.0f)
			* deviceparameter_Image[5];
		float fov_radius = sqrtf(extent_x * extent_x + extent_y * extent_y
			+ extent_z * extent_z);
		float theta_min = 0.0f;
		float theta_max = static_cast<float>(M_PI);
		if (incoming_distance > fov_radius && incoming_distance > 0.0f)
		{
			incoming_x /= incoming_distance;
			incoming_y /= incoming_distance;
			incoming_z /= incoming_distance;
			float center_cosine = incoming_x * outgoing_x
				+ incoming_y * outgoing_y + incoming_z * outgoing_z;
			center_cosine = fminf(1.0f, fmaxf(-1.0f, center_cosine));
			float center_angle = acosf(center_cosine);
			const float target_half_width
				= 0.5f * deviceparameter_Detector[target * 12 + 4];
			const float target_half_thickness
				= 0.5f * deviceparameter_Detector[target * 12 + 5];
			const float target_half_height
				= 0.5f * deviceparameter_Detector[target * 12 + 6];
			const float target_radius = sqrtf(target_half_width * target_half_width
				+ target_half_thickness * target_half_thickness
				+ target_half_height * target_half_height);
			float half_angle = asinf(fminf(1.0f, fov_radius / incoming_distance));
			if (distance > target_radius)
				half_angle += asinf(fminf(1.0f, target_radius / distance));
			else
				half_angle = static_cast<float>(M_PI);
			half_angle += 1e-5f;
			theta_min = fmaxf(0.0f, center_angle - half_angle);
			theta_max = fminf(static_cast<float>(M_PI), center_angle + half_angle);
		}

		float source_energy = deviceparameter_Physics[7];
		float maximum_energy = calculateScatterEnergy(theta_min, source_energy);
		float minimum_energy = calculateScatterEnergy(theta_max, source_energy);
		float relative_fwhm = deviceparameter_Detector[target * 12 + 10];
		float window_lower = deviceparameter_Physics[5];
		float window_upper = deviceparameter_Physics[6];
		if (static_cast<int>(floorf(deviceparameter_Physics[4] + 0.5f)) == 0)
		{
			window_lower = (1.0f - relative_fwhm / 2.0f) * source_energy;
			window_upper = (1.0f + relative_fwhm / 2.0f) * source_energy;
		}
		float maximum_resolution = relative_fwhm
			* sqrtf(source_energy / maximum_energy);
		float minimum_resolution = relative_fwhm
			* sqrtf(source_energy / minimum_energy);
		// Five sigma keeps pruning conservative while the production surface
		// integral itself evaluates the full Gaussian CDF without hard cutoff.
		float support_upper = maximum_energy
			* (1.0f + 5.0f * maximum_resolution / 2.35482f);
		float support_lower = minimum_energy
			* (1.0f - 5.0f * minimum_resolution / 2.35482f);
		if (support_upper + 1e-3f <= window_lower
			|| support_lower - 1e-3f >= window_upper)
		{
			kinematically_allowed = false;
		}
	}

	float scatter_rotation = deviceparameter_Detector[scatter * 12 + 11];
	float scatter_cos = cosf(-scatter_rotation);
	float scatter_sin = sinf(-scatter_rotation);
	float outgoing_local_x = outgoing_x * scatter_cos - outgoing_z * scatter_sin;
	float outgoing_local_z = outgoing_x * scatter_sin + outgoing_z * scatter_cos;
	path.source_exit_length = detectorCenterExitDistance(
		outgoing_local_x, outgoing_y, outgoing_local_z,
		deviceparameter_Detector[scatter * 12 + 4],
		deviceparameter_Detector[scatter * 12 + 5],
		deviceparameter_Detector[scatter * 12 + 6]);

	float target_rotation = deviceparameter_Detector[target * 12 + 11];
	float target_cos = cosf(-target_rotation);
	float target_sin = sinf(-target_rotation);
	float scatter_local_x = (scatter_x - target_x) * target_cos
		- (scatter_z - target_z) * target_sin;
	float scatter_local_y = scatter_y - target_y;
	float scatter_local_z = (scatter_x - target_x) * target_sin
		+ (scatter_z - target_z) * target_cos;
	float target_width = deviceparameter_Detector[target * 12 + 4];
	float target_thickness = deviceparameter_Detector[target * 12 + 5];
	float target_height = deviceparameter_Detector[target * 12 + 6];
	float extended_x = -1000.0f * scatter_local_x / distance;
	float extended_y = -1000.0f * scatter_local_y / distance;
	float extended_z = -1000.0f * scatter_local_z / distance;
	path.target_absorption_length = length_box_ray(
		scatter_local_x, scatter_local_y, scatter_local_z,
		extended_x, extended_y, extended_z,
		-target_width * 0.5f, -target_thickness * 0.5f, -target_height * 0.5f,
		target_width * 0.5f, target_thickness * 0.5f, target_height * 0.5f);
	path.flags = kinematically_allowed ? kCrystalPairKinematicallyAllowed : 0U;
	pair_paths[index] = path;
}

__device__ inline bool clipStructuredSegmentAxis(
	float origin,
	float direction,
	float minimum_value,
	float maximum_value,
	float* lower,
	float* upper)
{
	if (fabsf(direction) < 1e-8f)
		return origin >= minimum_value && origin <= maximum_value;
	float first = (minimum_value - origin) / direction;
	float second = (maximum_value - origin) / direction;
	if (first > second)
	{
		float temporary = first;
		first = second;
		second = temporary;
	}
	*lower = fmaxf(*lower, first);
	*upper = fminf(*upper, second);
	return *lower <= *upper;
}

__global__ void gatherCrystalPairMaterialLengths(
	float4* material_lengths,
	const CrystalPairPath* pair_paths,
	long long pair_count)
{
	long long index = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
	if (index < pair_count)
		material_lengths[index] = pair_paths[index].material_lengths;
}

__global__ void applyCrystalPairMaterialLengths(
	CrystalPairPath* pair_paths,
	const float4* material_lengths,
	long long pair_count)
{
	long long index = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
	if (index < pair_count)
		pair_paths[index].material_lengths = material_lengths[index];
}

__global__ void buildStructuredCrystalPairMaterialPaths(
	CrystalPairPath* pair_paths,
	const float* deviceparameter_Detector,
	const int* deviceDetectorMaterial,
	const AxisAlignedLayerGrid* layers,
	const int* cell_to_detector,
	int layer_count,
	int numProjectionSingle,
	int scatterStart,
	int scatterCount,
	int process_pruned_pairs)
{
	long long pair_index = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
	long long pair_count = static_cast<long long>(scatterCount) * numProjectionSingle;
	if (pair_index >= pair_count) return;
	CrystalPairPath path = pair_paths[pair_index];
	if ((path.flags & kCrystalPairKinematicallyAllowed) == 0U
		&& process_pruned_pairs == 0) return;
	if (!(path.direction_distance.w > 0.0f)) return;

	int local_scatter = static_cast<int>(pair_index / numProjectionSingle);
	int target = static_cast<int>(pair_index % numProjectionSingle);
	int scatter = scatterStart + local_scatter;
	float scatter_x = deviceparameter_Detector[scatter * 12 + 1];
	float scatter_y = deviceparameter_Detector[scatter * 12 + 2];
	float scatter_z = deviceparameter_Detector[scatter * 12 + 3];
	float target_x = deviceparameter_Detector[target * 12 + 1];
	float target_y = deviceparameter_Detector[target * 12 + 2];
	float target_z = deviceparameter_Detector[target * 12 + 3];
	float full_dx = target_x - scatter_x;
	float full_dy = target_y - scatter_y;
	float full_dz = target_z - scatter_z;
	double material_lengths[kXcomMaterialCount] = {0.0, 0.0, 0.0, 0.0};

	for (int layer_index = 0; layer_index < layer_count; ++layer_index)
	{
		AxisAlignedLayerGrid layer = layers[layer_index];
		float segment_lower = 0.0f;
		float segment_upper = 1.0f;
		if (fabsf(full_dy) < 1e-8f)
		{
			if (scatter_y < layer.y_min || scatter_y > layer.y_max) continue;
		}
		else
		{
			float first = (layer.y_min - scatter_y) / full_dy;
			float second = (layer.y_max - scatter_y) / full_dy;
			if (first > second)
			{
				float temporary = first;
				first = second;
				second = temporary;
			}
			segment_lower = fmaxf(0.0f, first);
			segment_upper = fminf(1.0f, second);
			if (segment_lower > segment_upper) continue;
		}

		float layer_x0 = scatter_x + segment_lower * full_dx;
		float layer_z0 = scatter_z + segment_lower * full_dz;
		float layer_x1 = scatter_x + segment_upper * full_dx;
		float layer_z1 = scatter_z + segment_upper * full_dz;
		float layer_dx = layer_x1 - layer_x0;
		float layer_dz = layer_z1 - layer_z0;
		float grid_lower = 0.0f;
		float grid_upper = 1.0f;
		float grid_x_max = layer.x_boundary_min + layer.count_x * layer.pitch_x;
		float grid_z_max = layer.z_boundary_min + layer.count_z * layer.pitch_z;
		if (!clipStructuredSegmentAxis(
			layer_x0, layer_dx, layer.x_boundary_min, grid_x_max,
			&grid_lower, &grid_upper)
			|| !clipStructuredSegmentAxis(
				layer_z0, layer_dz, layer.z_boundary_min, grid_z_max,
				&grid_lower, &grid_upper))
			continue;

		float start_x = layer_x0 + grid_lower * layer_dx;
		float start_z = layer_z0 + grid_lower * layer_dz;
		float clipped_dx = (grid_upper - grid_lower) * layer_dx;
		float clipped_dz = (grid_upper - grid_lower) * layer_dz;
		float x_position = (start_x - layer.x_boundary_min) / layer.pitch_x;
		float z_position = (start_z - layer.z_boundary_min) / layer.pitch_z;
		int x_index = static_cast<int>(floorf(x_position));
		int z_index = static_cast<int>(floorf(z_position));
		if (clipped_dx < 0.0f && fabsf(x_position - floorf(x_position)) < 1e-5f)
			--x_index;
		if (clipped_dz < 0.0f && fabsf(z_position - floorf(z_position)) < 1e-5f)
			--z_index;
		x_index = x_index < 0 ? 0 : (x_index >= layer.count_x ? layer.count_x - 1 : x_index);
		z_index = z_index < 0 ? 0 : (z_index >= layer.count_z ? layer.count_z - 1 : z_index);
		int step_x = clipped_dx > 0.0f ? 1 : (clipped_dx < 0.0f ? -1 : 0);
		int step_z = clipped_dz > 0.0f ? 1 : (clipped_dz < 0.0f ? -1 : 0);
		float next_x = step_x == 0 ? 1e30f
			: (layer.x_boundary_min
				+ (x_index + (step_x > 0 ? 1 : 0)) * layer.pitch_x - start_x)
				/ clipped_dx;
		float next_z = step_z == 0 ? 1e30f
			: (layer.z_boundary_min
				+ (z_index + (step_z > 0 ? 1 : 0)) * layer.pitch_z - start_z)
				/ clipped_dz;
		float delta_x = step_x == 0 ? 1e30f
			: layer.pitch_x / fabsf(clipped_dx);
		float delta_z = step_z == 0 ? 1e30f
			: layer.pitch_z / fabsf(clipped_dz);
		int maximum_steps = layer.count_x + layer.count_z + 4;
		for (int traversal_step = 0; traversal_step < maximum_steps; ++traversal_step)
		{
			if (x_index < 0 || x_index >= layer.count_x
				|| z_index < 0 || z_index >= layer.count_z)
				break;
			int intermediate = cell_to_detector[
				layer.map_offset + z_index * layer.count_x + x_index];
			if (intermediate >= 0 && intermediate != scatter && intermediate != target)
			{
				int material = deviceDetectorMaterial[intermediate];
				if (material >= 0 && material < kXcomMaterialCount)
				{
					float center_x = deviceparameter_Detector[intermediate * 12 + 1];
					float center_y = deviceparameter_Detector[intermediate * 12 + 2];
					float center_z = deviceparameter_Detector[intermediate * 12 + 3];
					float rotation = deviceparameter_Detector[intermediate * 12 + 11];
					float cosine = cos(-rotation);
					float sine = sin(-rotation);
					float scatter_local_x = (scatter_x - center_x) * cosine
						- (scatter_z - center_z) * sine;
					float scatter_local_y = scatter_y - center_y;
					float scatter_local_z = (scatter_x - center_x) * sine
						+ (scatter_z - center_z) * cosine;
					float target_local_x = (target_x - center_x) * cosine
						- (target_z - center_z) * sine;
					float target_local_y = target_y - center_y;
					float target_local_z = (target_x - center_x) * sine
						+ (target_z - center_z) * cosine;
					float half_width = 0.5f
						* deviceparameter_Detector[intermediate * 12 + 4];
					float half_thickness = 0.5f
						* deviceparameter_Detector[intermediate * 12 + 5];
					float half_height = 0.5f
						* deviceparameter_Detector[intermediate * 12 + 6];
					float length = length_box_ray(
						scatter_local_x, scatter_local_y, scatter_local_z,
						target_local_x, target_local_y, target_local_z,
						-half_width, -half_thickness, -half_height,
						half_width, half_thickness, half_height);
					material_lengths[material] += length;
				}
			}
			float next_boundary = fminf(next_x, next_z);
			if (next_boundary > 1.0f) break;
			float previous_next_x = next_x;
			float previous_next_z = next_z;
			if (previous_next_x <= previous_next_z)
			{
				x_index += step_x;
				next_x += delta_x;
			}
			if (previous_next_z <= previous_next_x)
			{
				z_index += step_z;
				next_z += delta_z;
			}
		}
	}

	path.material_lengths = make_float4(
		static_cast<float>(material_lengths[kMaterialNaI]),
		static_cast<float>(material_lengths[kMaterialGAGG]),
		static_cast<float>(material_lengths[kMaterialPb]),
		static_cast<float>(material_lengths[kMaterialW]));
	pair_paths[pair_index] = path;
}

__global__ void reduceCrystalPairMaterialPaths(
	CrystalPairPath* pair_paths,
	const float* deviceparameter_Detector,
	const int* deviceDetectorMaterial,
	const unsigned int* relationship_bitmap,
	int numProjectionSingle,
	int scatterStart,
	int scatterCount)
{
	long long pair_index = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
	long long pair_count = static_cast<long long>(scatterCount) * numProjectionSingle;
	if (pair_index >= pair_count) return;
	CrystalPairPath path = pair_paths[pair_index];
	if ((path.flags & kCrystalPairKinematicallyAllowed) == 0U) return;

	int local_scatter = static_cast<int>(pair_index / numProjectionSingle);
	int target = static_cast<int>(pair_index % numProjectionSingle);
	int scatter = scatterStart + local_scatter;
	float scatter_x = deviceparameter_Detector[scatter * 12 + 1];
	float scatter_y = deviceparameter_Detector[scatter * 12 + 2];
	float scatter_z = deviceparameter_Detector[scatter * 12 + 3];
	float target_x = deviceparameter_Detector[target * 12 + 1];
	float target_y = deviceparameter_Detector[target * 12 + 2];
	float target_z = deviceparameter_Detector[target * 12 + 3];
	double material_lengths[kXcomMaterialCount] = {0.0, 0.0, 0.0, 0.0};

	long long pair_bit_start = pair_index * numProjectionSingle;
	int first_k = 0;
	while (first_k < numProjectionSingle)
	{
		long long bit_index = pair_bit_start + first_k;
		int bit_offset = static_cast<int>(bit_index & 31LL);
		int bit_count = min(32 - bit_offset, numProjectionSingle - first_k);
		unsigned int mask = relationship_bitmap[bit_index >> 5] >> bit_offset;
		if (bit_count < 32) mask &= (1U << bit_count) - 1U;

		while (mask != 0U)
		{
			int local_bit = __ffs(mask) - 1;
			int intermediate = first_k + local_bit;
			mask &= mask - 1U;
			if (intermediate == scatter || intermediate == target) continue;

			int material = deviceDetectorMaterial[intermediate];
			if (material < 0 || material >= kXcomMaterialCount) continue;
			float center_x = deviceparameter_Detector[intermediate * 12 + 1];
			float center_y = deviceparameter_Detector[intermediate * 12 + 2];
			float center_z = deviceparameter_Detector[intermediate * 12 + 3];
			float rotation = deviceparameter_Detector[intermediate * 12 + 11];
			float cosine = cos(-rotation);
			float sine = sin(-rotation);
			float scatter_local_x = (scatter_x - center_x) * cosine
				- (scatter_z - center_z) * sine;
			float scatter_local_y = scatter_y - center_y;
			float scatter_local_z = (scatter_x - center_x) * sine
				+ (scatter_z - center_z) * cosine;
			float target_local_x = (target_x - center_x) * cosine
				- (target_z - center_z) * sine;
			float target_local_y = target_y - center_y;
			float target_local_z = (target_x - center_x) * sine
				+ (target_z - center_z) * cosine;
			float half_width = 0.5f * deviceparameter_Detector[intermediate * 12 + 4];
			float half_thickness = 0.5f * deviceparameter_Detector[intermediate * 12 + 5];
			float half_height = 0.5f * deviceparameter_Detector[intermediate * 12 + 6];
			float length = length_box_ray(
				scatter_local_x, scatter_local_y, scatter_local_z,
				target_local_x, target_local_y, target_local_z,
				-half_width, -half_thickness, -half_height,
				half_width, half_thickness, half_height);
			material_lengths[material] += length;
		}
		first_k += bit_count;
	}

	path.material_lengths = make_float4(
		static_cast<float>(material_lengths[kMaterialNaI]),
		static_cast<float>(material_lengths[kMaterialGAGG]),
		static_cast<float>(material_lengths[kMaterialPb]),
		static_cast<float>(material_lengths[kMaterialW]));
	pair_paths[pair_index] = path;
}



__global__ void collimatorScatterSysMatCuda(float* dst,
		const CollimatorScatterSample* samples,
		int numCollimatorSamples,
		float* deviceparameter_Detector,
		const int* deviceDetectorMaterial,
		float* deviceparameter_Image,
	float* deviceparameter_Physics,
	const float* deviceComptonPhasePrefix,
	unsigned int* deviceGeometryRelationShip_Collimator2Crystal,
	int numProjectionSingle,
	int numImagebin)

{

	int numDetectorbins = numProjectionSingle;

	float _float_FOV2Collimator = deviceparameter_Image[11];

	////////////////////////////////////////// Collimator Parameters //////////////////////////////////////////
	/*
	float _float_numCollimatorHoles[5];
	int numCollimatorHoles[5];
	float _float_widthCollimatorLayers[5];
	float _float_heightCollimatorLayers[5];
	float _float_thicknessCollimatorLayers[5];
	float _float_coeffCollimatorLayers[5];

	int numCollimatorHoles_tot = 0;
	for (unsigned int id_CollimatorLayer = 0; id_CollimatorLayer < numCollimatorLayer; id_CollimatorLayer++)
	{
		_float_numCollimatorHoles[id_CollimatorLayer] = deviceparameter_Collimator[(id_CollimatorLayer + 1) * 10 + 0];
		numCollimatorHoles[id_CollimatorLayer] = (int)floor(_float_numCollimatorHoles[id_CollimatorLayer]);
		_float_widthCollimatorLayers[id_CollimatorLayer] = deviceparameter_Collimator[(id_CollimatorLayer + 1) * 10 + 1];
		_float_thicknessCollimatorLayers[id_CollimatorLayer] = deviceparameter_Collimator[(id_CollimatorLayer + 1) * 10 + 2];
		_float_heightCollimatorLayers[id_CollimatorLayer] = deviceparameter_Collimator[(id_CollimatorLayer + 1) * 10 + 3];
		_float_coeffCollimatorLayers[id_CollimatorLayer] = deviceparameter_Collimator[(id_CollimatorLayer + 1) * 10 + 5];
		numCollimatorHoles_tot = numCollimatorHoles_tot + numCollimatorHoles[id_CollimatorLayer];
	}
	*/
	//////////////////////////////////////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////// Image Parameters ////////////////////////////////////////////

	float _float_widthImageVoxelX = deviceparameter_Image[3];
	float _float_widthImageVoxelY = deviceparameter_Image[4];
	float _float_widthImageVoxelZ = deviceparameter_Image[5];

	//float _float_numRotation = deviceparameter_Image[6];//numRotation;
	float _float_angelPerRotation = deviceparameter_Image[7];//Angel per Rotation;
	float _float_idxrotation = deviceparameter_Image[20];//idxRotation
	//float RotationAngle = _float_idxrotation / _float_numRotation * (2 * M_PI);
	float RotationAngle = _float_idxrotation * _float_angelPerRotation;
	float shiftFOVX_physics = deviceparameter_Image[8];
	float shiftFOVY_physics = deviceparameter_Image[9];
	float shiftFOVZ_physics = deviceparameter_Image[10];

	int numImageVoxelX = (int)floor(deviceparameter_Image[0] + 0.001f);
	int numImageVoxelY = (int)floor(deviceparameter_Image[1] + 0.001f);
	int numImageVoxelZ = (int)floor(deviceparameter_Image[2] + 0.001f);

	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	long long int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < 0 || row > numProjectionSingle - 1) { return; }
	long long int col = blockIdx.y * blockDim.y + threadIdx.y;
	if (col < 0 || col > numImagebin - 1) { return; }
	long long int slice = blockIdx.z * blockDim.z + threadIdx.z;
	if (slice < 0 || slice > numCollimatorSamples - 1) { return; }

	long long int dstIndex = row * numImagebin + col;

	/*
	if (row < 5 && col < 5 && slice < 1) {
		printf("Thread (%lld, %lld, %lld) set dst[%lld] = %d\n", row, col, slice, dstIndex, numCollimator_Holes);
		printf("NumCollimator Holes = %d\n", numCollimator_Holes);
		printf("numImageVoxelX = %d\n", numImageVoxelX);
		
	}
	*/


	unsigned int idxDetector = row; // index of detector
	unsigned int id_CollimatorSample = slice;

	int idxImageVoxelZ = col / (numImageVoxelY * numImageVoxelX);
	col = col % (numImageVoxelY * numImageVoxelX);
	int idxImageVoxelY = col / numImageVoxelX;
	int idxImageVoxelX = col % numImageVoxelX;


	const unsigned int divideX = 1, divideY = 1, divideZ = 1;

	///////////////////////////////////////// Physic Progress Parameters /////////////////////////////////////
	int flagUsingCompton = (int)floor(deviceparameter_Physics[0] + 0.5f);
	int flagUsingSameEnergyWindow = (int)floor(deviceparameter_Physics[4] + 0.5f);

	float lowerThresholdofEnergyWindow = deviceparameter_Physics[5];
	float upperThresholdofEnergyWindow = deviceparameter_Physics[6];

	float target_PE_Energy = deviceparameter_Physics[7];
	float energy_resolution_detector_targetPE = deviceparameter_Detector[idxDetector * 12 + 10];

	// Energy Window of detector crystal
	if (flagUsingSameEnergyWindow > 0)
	{
		lowerThresholdofEnergyWindow = deviceparameter_Physics[5];
		upperThresholdofEnergyWindow = deviceparameter_Physics[6];
	}
	else
	{
		lowerThresholdofEnergyWindow = (1 - energy_resolution_detector_targetPE / 2.0f) * target_PE_Energy;
		upperThresholdofEnergyWindow = (1 + energy_resolution_detector_targetPE / 2.0f) * target_PE_Energy;
	}

	float coeff_detector_total = deviceparameter_Detector[idxDetector * 12 + 7];

	float integration_Compton = deviceComptonNormalization;
	//////////////////////////////////////////////////////////////////////////////////////////////////////////


	///////////////////////////////////////// Image Rotation Shift Parameters /////////////////////////////////////
	float xImage = (idxImageVoxelX - numImageVoxelX / 2.0f + 0.5f) * _float_widthImageVoxelX;
	float yImage = (idxImageVoxelY - numImageVoxelY / 2.0f + 0.5f) * _float_widthImageVoxelY;
	float zImage = (idxImageVoxelZ - numImageVoxelZ / 2.0f + 0.5f) * _float_widthImageVoxelZ;

	xImage = xImage + shiftFOVX_physics;
	yImage = yImage + shiftFOVY_physics;
	zImage = zImage + shiftFOVZ_physics;

	float xImage_rot = xImage * cos(RotationAngle) - yImage * sin(RotationAngle);
	float yImage_rot = xImage * sin(RotationAngle) + yImage * cos(RotationAngle);
	float zImage_rot = zImage;
	xImage = xImage_rot;
	yImage = yImage_rot;
	zImage = zImage_rot;


	// All variables without a suffix are in the real-world physical coordinate system
	// All parameters with 'self' suffix are in the detector crystal coordinate system
	float xDetectorCrystalCenter = deviceparameter_Detector[12 * idxDetector + 1];
	float yDetectorCrystalCenter = deviceparameter_Detector[12 * idxDetector + 2] + _float_FOV2Collimator;
	float zDetectorCrystalCenter = deviceparameter_Detector[12 * idxDetector + 3];

	float widthDetectorCrystal = deviceparameter_Detector[12 * idxDetector + 4];
	float heightDetectorCrystal = deviceparameter_Detector[12 * idxDetector + 6];
	float thicknessDetectorCrystal = deviceparameter_Detector[12 * idxDetector + 5];

	float rotationAngel_DetectorCrystal = deviceparameter_Detector[12 * idxDetector + 11];

	float xImage_self = (xImage - xDetectorCrystalCenter) * cos(-rotationAngel_DetectorCrystal) - (zImage - zDetectorCrystalCenter) * sin(-rotationAngel_DetectorCrystal);
	float yImage_self = yImage - yDetectorCrystalCenter;
	float zImage_self = (xImage - xDetectorCrystalCenter) * sin(-rotationAngel_DetectorCrystal) + (zImage - zDetectorCrystalCenter) * cos(-rotationAngel_DetectorCrystal);

	float x1_detectorcrystal_self = -widthDetectorCrystal / 2.0f;
	float x2_detectorcrystal_self = widthDetectorCrystal / 2.0f;
	float y1_detectorcrystal_self = -thicknessDetectorCrystal / 2.0f;
	float y2_detectorcrystal_self = thicknessDetectorCrystal / 2.0f;
	float z1_detectorcrystal_self = -heightDetectorCrystal / 2.0f;
	float z2_detectorcrystal_self = heightDetectorCrystal / 2.0f;

	//////////////////////////////////////////////////////////////////////////////////////////////////////////

	//////////////////// Compton scatter integrated over a physical collimator volume cell ////////////////////
	CollimatorScatterSample sample = samples[id_CollimatorSample];
	float x_scatter = sample.x;
	float y_scatter = sample.y_center + _float_FOV2Collimator;
	float z_scatter = sample.z;
	float L_image_hole = calculateDist(xImage, yImage, zImage, x_scatter, y_scatter, z_scatter);
	float coeff_collimator_pe_source = 0.0f;
	float coeff_collimator_compton_source = 0.0f;
	interpolateXcomDevice(sample.material_id, target_PE_Energy,
		&coeff_collimator_pe_source, &coeff_collimator_compton_source);
	float coeff_collimator_total_source = coeff_collimator_pe_source + coeff_collimator_compton_source;
	if (!(L_image_hole > 0.0f) || !(coeff_collimator_compton_source > 0.0f))
	{
		return;
	}
	float length = 0.0f;
	//////////////////////////////////////////////////////////////////////////////////////////////////////////

	
	///////////////////////////////////////// CalCulation Starts Below ///////////////////////////////////////
	if (flagUsingCompton == 1)
	{
		if (coeff_detector_total > 0.01f)
		{
			//float prob_Compton_to_detectionCrystal = 0.000;

			for (int NumZ = 0; NumZ < divideZ; NumZ++)
			{
				for (int NumX = 0; NumX < divideX; NumX++)
				{
					for (int NumY = 0; NumY < divideY; NumY++)
					{

						/////////////////////////////////  Parameters of the detector unit ////////////////////////////////
						// All variables without a suffix are in the real-world physical coordinate system
						// All parameters with 'self' suffix are in the detector crystal coordinate system
						float xDetector_self = -widthDetectorCrystal / 2.0f + (float)(NumX + 0.5f) / (float)divideX * widthDetectorCrystal;
						float zDetector_self = -heightDetectorCrystal / 2.0f + (float)(NumZ + 0.5f) / (float)divideZ * heightDetectorCrystal;
						float yDetector_self = -thicknessDetectorCrystal / 2.0f + (float)(NumY + 0.5f) / (float)divideY * thicknessDetectorCrystal;

						float xDetector_rot = xDetector_self * cos(rotationAngel_DetectorCrystal) - zDetector_self * sin(rotationAngel_DetectorCrystal);
						float zDetector_rot = xDetector_self * sin(rotationAngel_DetectorCrystal) + zDetector_self * cos(rotationAngel_DetectorCrystal);
						float yDetector_rot = yDetector_self;

						float xDetector = xDetectorCrystalCenter + xDetector_rot;
						float zDetector = zDetectorCrystalCenter + zDetector_rot;
						float yDetector = yDetectorCrystalCenter + yDetector_rot;


						/////////////////////////////////  Compton scatter probability from the hole to detection crystal /////////////////////////////

						float comptonConeAngle = calculateConeAngle(xImage, yImage, zImage, x_scatter, y_scatter, z_scatter, xDetector, yDetector, zDetector);
						float scatterEnergy = calculateScatterEnergy(comptonConeAngle, target_PE_Energy);
						// Detector energy resolution is stored as relative FWHM at target_PE_Energy.
						// For scintillation statistics, relative FWHM scales as 1/sqrt(E).
						float energy_resolution_detector_scatterphoton = energy_resolution_detector_targetPE * sqrt(target_PE_Energy / scatterEnergy);

						//  The probability that a Compton scatterred photon being detected within the energy window of detector unit

						if (((scatterEnergy * (1 + 2 * energy_resolution_detector_scatterphoton / 2.35482f)) <= lowerThresholdofEnergyWindow) || (scatterEnergy * (1 - 2 * energy_resolution_detector_scatterphoton / 2.35482f) >= upperThresholdofEnergyWindow))
						{
							continue;
							// The energy of the scattered photon detected within a detector element follows a Gaussian distribution. 
							// If the 2 sigma range of this Gaussian does not overlap with the full energy peak window of the detector element, 
							// then it is considered that the scattering does not affect the result.
						}
						float energyDetected_probability = calculategaussianIntegral(scatterEnergy, energy_resolution_detector_scatterphoton, lowerThresholdofEnergyWindow, upperThresholdofEnergyWindow);


							// The probability that a Compton scattered photon, among all the photons scattered at the scattering point, 
							// is scattered towards the direction of the detector element.
							float L_comptonAngle = calculateDist(x_scatter, y_scatter, z_scatter, xDetector, yDetector, zDetector);
							if (!(L_comptonAngle > 0.0f)) continue;
							float coeff_collimator_pe_scatter = 0.0f;
							float coeff_collimator_compton_scatter = 0.0f;
							interpolateXcomDevice(sample.material_id, scatterEnergy,
								&coeff_collimator_pe_scatter, &coeff_collimator_compton_scatter);
							float coeff_collimator_total_scatter =
								coeff_collimator_pe_scatter + coeff_collimator_compton_scatter;
							float cosine_in = fabsf(y_scatter - yImage) / L_image_hole;
							float cosine_out = fabsf(yDetector - y_scatter) / L_comptonAngle;
							if (cosine_in < 1e-4f) cosine_in = 1e-4f;
							if (cosine_out < 1e-4f) cosine_out = 1e-4f;
							float depth_integral = attenuatedSlabDepthIntegral(
								coeff_collimator_total_source / cosine_in,
								coeff_collimator_total_scatter / cosine_out,
								sample.thickness);
							float prob_Compton_otherCollimatorSample = sample.lead_area
								* coeff_collimator_compton_source * depth_integral
								/ (4.0f * M_PI * L_image_hole * L_image_hole);
							if (!(prob_Compton_otherCollimatorSample > 0.0f)) continue;

							// Calculate the phi range, using the detector unit's minimum enclosing sphere as an approximation.
						float R_detector = sqrt(widthDetectorCrystal * widthDetectorCrystal / (float)divideX / (float)divideX + heightDetectorCrystal * heightDetectorCrystal / (float)divideZ / (float)divideZ + thicknessDetectorCrystal * thicknessDetectorCrystal / (float)divideY / (float)divideY) / 2.0f;
						float Range_Phi = 0.000f;
						if (L_comptonAngle * sin(comptonConeAngle) * 2.0f <= R_detector)
						{
							Range_Phi = 2.0f * M_PI;
						}
						else
						{
							Range_Phi = 4.0f * asin(min(R_detector / 2.0f / L_comptonAngle / sin(comptonConeAngle), 1.0f));
						}

						// Calculate the theta range
						float x_scatter_self = (x_scatter - xDetectorCrystalCenter) * cos(-rotationAngel_DetectorCrystal) - (z_scatter - zDetectorCrystalCenter) * sin(-rotationAngel_DetectorCrystal);
						float y_scatter_self = y_scatter - yDetectorCrystalCenter;
						float z_scatter_self = (x_scatter - xDetectorCrystalCenter) * sin(-rotationAngel_DetectorCrystal) + (z_scatter - zDetectorCrystalCenter) * cos(-rotationAngel_DetectorCrystal);

						float x1_detectorunit_self = ((float)NumX / (float)divideX - 0.5f) * widthDetectorCrystal;
						float x2_detectorunit_self = (((float)NumX + 1.0f) / (float)divideX - 0.5f) * widthDetectorCrystal;

						float y1_detectorunit_self = ((float)NumY / (float)divideY - 0.5f) * thicknessDetectorCrystal;
						float y2_detectorunit_self = (((float)NumY + 1.0f) / (float)divideY - 0.5f) * thicknessDetectorCrystal;

						float z1_detectorunit_self = ((float)NumZ / (float)divideZ - 0.5f) * heightDetectorCrystal;
						float z2_detectorunit_self = (((float)NumZ + 1.0f) / (float)divideZ - 0.5f) * heightDetectorCrystal;


						float dist_extend = 1000.0f;
						float dist_Image_scatterer = calculateDist(x_scatter_self, y_scatter_self, z_scatter_self, xImage_self, yImage_self, zImage_self);
						float x_tmp = x_scatter_self + dist_extend * (x_scatter_self - xImage_self) / dist_Image_scatterer;
						float y_tmp = y_scatter_self + dist_extend * (y_scatter_self - yImage_self) / dist_Image_scatterer;
						float z_tmp = z_scatter_self + dist_extend * (z_scatter_self - zImage_self) / dist_Image_scatterer;

						length = length_box_ray(xImage_self, yImage_self, zImage_self, x_tmp, y_tmp, z_tmp, x1_detectorunit_self, y1_detectorunit_self, z1_detectorunit_self, x2_detectorunit_self, y2_detectorunit_self, z2_detectorunit_self);

						float theta[8];
						theta[0] = calculateConeAngle(xImage_self, yImage_self, zImage_self, x_scatter_self, y_scatter_self, z_scatter_self, x1_detectorunit_self, y1_detectorunit_self, z1_detectorunit_self);
						theta[1] = calculateConeAngle(xImage_self, yImage_self, zImage_self, x_scatter_self, y_scatter_self, z_scatter_self, x2_detectorunit_self, y1_detectorunit_self, z1_detectorunit_self);
						theta[2] = calculateConeAngle(xImage_self, yImage_self, zImage_self, x_scatter_self, y_scatter_self, z_scatter_self, x1_detectorunit_self, y2_detectorunit_self, z1_detectorunit_self);
						theta[3] = calculateConeAngle(xImage_self, yImage_self, zImage_self, x_scatter_self, y_scatter_self, z_scatter_self, x1_detectorunit_self, y1_detectorunit_self, z2_detectorunit_self);
						theta[4] = calculateConeAngle(xImage_self, yImage_self, zImage_self, x_scatter_self, y_scatter_self, z_scatter_self, x2_detectorunit_self, y2_detectorunit_self, z1_detectorunit_self);
						theta[5] = calculateConeAngle(xImage_self, yImage_self, zImage_self, x_scatter_self, y_scatter_self, z_scatter_self, x2_detectorunit_self, y1_detectorunit_self, z2_detectorunit_self);
						theta[6] = calculateConeAngle(xImage_self, yImage_self, zImage_self, x_scatter_self, y_scatter_self, z_scatter_self, x1_detectorunit_self, y2_detectorunit_self, z2_detectorunit_self);
						theta[7] = calculateConeAngle(xImage_self, yImage_self, zImage_self, x_scatter_self, y_scatter_self, z_scatter_self, x2_detectorunit_self, y2_detectorunit_self, z2_detectorunit_self);

						float theta_min = theta[0];
						float theta_max = theta[0];
						for (int i = 1; i < 8; i++)
						{
							if (theta[i] > theta_max)
								theta_max = theta[i];
							if (theta[i] < theta_min)
								theta_min = theta[i];
						}

						if (length > 0.001f)
						{
							theta_min = 0.000; // If the extension of the line from the image to the scatterer passes through the detector unit, then theta_min=0
						}

						// Range_Theta = 2.0f * asin(min(1.0f, R_detector / R_comptonAngle));
						float interval_compton = deviceComptonPhasePrefix != NULL
							? computeComptonIntegralPhasePrefix(
								deviceComptonPhasePrefix, theta_min, theta_max)
							: computeComptonIntegral(
								target_PE_Energy, theta_min, theta_max, kComptonIntegralStep);
						float comptonAngleRatio = interval_compton / integration_Compton;

						/////////////////////////////////  Attenuation from the scatterer hole to detector unit /////////////////////////////
						float attenuation_dist_crystal_crystal = 0.000f;

						for (int id_Detector_att = 0; id_Detector_att < numDetectorbins; id_Detector_att++)
						{
							if (id_Detector_att != idxDetector)
							{
								int bits_per_word = 32;
									int flagCross = indexFrombitmap_collimator(id_CollimatorSample, idxDetector, id_Detector_att, deviceGeometryRelationShip_Collimator2Crystal, numProjectionSingle, bits_per_word);
								if (flagCross == 0)
								{
									continue;
								}
								else
								{
									float length_att = 0;

									float x_AttCrystalCenter = deviceparameter_Detector[12 * id_Detector_att + 1];
									float y_AttCrystalCenter = deviceparameter_Detector[12 * id_Detector_att + 2] + _float_FOV2Collimator;
									float z_AttCrystalCenter = deviceparameter_Detector[12 * id_Detector_att + 3];

									float width_AttCrystal = deviceparameter_Detector[12 * id_Detector_att + 4];
									float height_AttCrystal = deviceparameter_Detector[12 * id_Detector_att + 6];
									float thickness_AttCrystal = deviceparameter_Detector[12 * id_Detector_att + 5];

									float rotationAngel_AttCrystal = deviceparameter_Detector[12 * id_Detector_att + 11];

										float coeff_pe_att = 0.0f;
										float coeff_compton_att = 0.0f;
										interpolateXcomDevice(deviceDetectorMaterial[id_Detector_att], scatterEnergy,
											&coeff_pe_att, &coeff_compton_att);
										float coeff_total_att = coeff_pe_att + coeff_compton_att;

									float x_scatter_Att = (x_scatter - x_AttCrystalCenter) * cos(-rotationAngel_AttCrystal) - (z_scatter - z_AttCrystalCenter) * sin(-rotationAngel_AttCrystal);
									float y_scatter_Att = y_scatter - y_AttCrystalCenter;
									float z_scatter_Att = (x_scatter - x_AttCrystalCenter) * sin(-rotationAngel_AttCrystal) + (z_scatter - z_AttCrystalCenter) * cos(-rotationAngel_AttCrystal);

									float x1_Att = -0.5f * width_AttCrystal;
									float x2_Att = 0.5f * width_AttCrystal;

									float y1_Att = -0.5f * thickness_AttCrystal;
									float y2_Att = 0.5f * thickness_AttCrystal;

									float z1_Att = -0.5f * height_AttCrystal;
									float z2_Att = 0.5f * height_AttCrystal;

									float xDetector_Att = (xDetector - x_AttCrystalCenter) * cos(-rotationAngel_AttCrystal) - (zDetector - z_AttCrystalCenter) * sin(-rotationAngel_AttCrystal);
									float yDetector_Att = yDetector - y_AttCrystalCenter;
									float zDetector_Att = (xDetector - x_AttCrystalCenter) * sin(-rotationAngel_AttCrystal) + (zDetector - z_AttCrystalCenter) * cos(-rotationAngel_AttCrystal);

									length_att = length_box_ray(x_scatter_Att, y_scatter_Att, z_scatter_Att, xDetector_Att, yDetector_Att, zDetector_Att, x1_Att, y1_Att, z1_Att, x2_Att, y2_Att, z2_Att);
									attenuation_dist_crystal_crystal = attenuation_dist_crystal_crystal + length_att * coeff_total_att;

								}

							}

						}


						///////////////  Attenuation from the scatterer hole to detector unit  within the detection crystal /////////////////
						float length_crystalself_att1 = 0.0000f;
						float length_crystalself_att2 = 0.0000f;

						length_crystalself_att1 = length_box_ray_inside(x_scatter_self, y_scatter_self, z_scatter_self, xDetector_self, yDetector_self, zDetector_self, x1_detectorcrystal_self, y1_detectorcrystal_self, z1_detectorcrystal_self, x2_detectorcrystal_self, y2_detectorcrystal_self, z2_detectorcrystal_self);
						length_crystalself_att2 = length_box_ray_inside(x_scatter_self, y_scatter_self, z_scatter_self, xDetector_self, yDetector_self, zDetector_self, x1_detectorunit_self, y1_detectorunit_self, z1_detectorunit_self, x2_detectorunit_self, y2_detectorunit_self, z2_detectorunit_self);

							float absorp_coeff_detector_pe = 0.0f;
							float absorp_coeff_detector_compton = 0.0f;
							interpolateXcomDevice(deviceDetectorMaterial[idxDetector], scatterEnergy,
								&absorp_coeff_detector_pe, &absorp_coeff_detector_compton);
							float absorp_coeff_detector_total = absorp_coeff_detector_pe + absorp_coeff_detector_compton;
						attenuation_dist_crystal_crystal = attenuation_dist_crystal_crystal + (length_crystalself_att1 - length_crystalself_att2) * absorp_coeff_detector_total;

						/////////////////////////////////  Absoption of scattered photons within the detector unit /////////////////////////////
						x_tmp = xDetector_self + dist_extend * (xDetector_self - x_scatter_self) / L_comptonAngle;
						y_tmp = yDetector_self + dist_extend * (yDetector_self - y_scatter_self) / L_comptonAngle;
						z_tmp = zDetector_self + dist_extend * (zDetector_self - z_scatter_self) / L_comptonAngle;
						float length_absorp = 0.000f;
						length_absorp = length_box_ray(x_scatter_self, y_scatter_self, z_scatter_self, x_tmp, y_tmp, z_tmp, x1_detectorunit_self, y1_detectorunit_self, z1_detectorunit_self, x2_detectorunit_self, y2_detectorunit_self, z2_detectorunit_self);

						//prob_Compton_to_detectionCrystal += prob_Compton_othercrystal * Range_Phi / 2.0f / M_PI * comptonAngleRatio * energyDetected_probability * exp(-attenuation_dist_crystal_crystal) * (1.0f-exp(-length_absorp* absorp_coeff_detector_total))* absorp_coeff_detector_pe/ absorp_coeff_detector_total;
						
							float contrib = prob_Compton_otherCollimatorSample * Range_Phi / 2.0f / M_PI * comptonAngleRatio * energyDetected_probability * exp(-attenuation_dist_crystal_crystal) * (1.0f - exp(-length_absorp * absorp_coeff_detector_total)) * absorp_coeff_detector_pe / absorp_coeff_detector_total;
						if (isfinite(contrib) && contrib > 0.0f)
						{
							atomicAdd(&dst[dstIndex], contrib);
						}


					}
				}
			}
			//atomicAdd(&dst[dstIndex], prob_Compton_to_detectionCrystal);
		}

	}


}


__global__ void geometryRelationShip_Collimator2Crystal(
	unsigned int* dst_relation_collimator2crystal,
	float* deviceparameter_Detector,
	const CollimatorScatterSample* samples,
	int numSamples)
{
	
	int numProjectionSingle = deviceparameter_Detector[0];
	long long idx = static_cast<long long>(blockIdx.x) * static_cast<long long>(blockDim.x) + static_cast<long long>(threadIdx.x);
	long long total_threads = static_cast<long long>(numSamples) * static_cast<long long>(numProjectionSingle) * static_cast<long long>(numProjectionSingle);

	if (idx >= total_threads) return;

	
	int k = idx % numProjectionSingle;
	int j = (idx / numProjectionSingle) % numProjectionSingle;
	int i = idx / (numProjectionSingle * numProjectionSingle);

	
	long long bit_idx = static_cast<long long>(i) * static_cast<long long>(numProjectionSingle) * static_cast<long long>(numProjectionSingle) + static_cast<long long>(j) * static_cast<long long>(numProjectionSingle) + static_cast<long long>(k);

	
	int bits_per_word = 32;
	long long word_idx = bit_idx / bits_per_word;
	int bit_offset = bit_idx % bits_per_word;

	
	float xCollimator_i = samples[i].x;
	float yCollimator_i = samples[i].y_center;
	float zCollimator_i = samples[i].z;
	//float rCollimator_i = deviceparameter_Collimator[9 * i + 104];
	//float thicknessCollimator_i = deviceparameter_Collimator[9 * i + 102] - deviceparameter_Collimator[9 * i + 101];

	float xDetector_j = deviceparameter_Detector[12 * j + 1];
	float yDetector_j = deviceparameter_Detector[12 * j + 2];
	float zDetector_j = deviceparameter_Detector[12 * j + 3];
	float widthDetector_j = deviceparameter_Detector[12 * j + 4];
	float heightDetector_j = deviceparameter_Detector[12 * j + 6];
	float thicknessDetector_j = deviceparameter_Detector[12 * j + 5];
	float R_detector_j = sqrt(widthDetector_j * widthDetector_j + heightDetector_j * heightDetector_j + thicknessDetector_j * thicknessDetector_j) / 2.0f;

	float L_ij = calculateDist(xCollimator_i, yCollimator_i, zCollimator_i, xDetector_j, yDetector_j, zDetector_j);

	float crit_j_i = R_detector_j / L_ij;

	float x_projectionOnUnitSphere_j_i = (xDetector_j - xCollimator_i) / L_ij;
	float y_projectionOnUnitSphere_j_i = (yDetector_j - yCollimator_i) / L_ij;
	float z_projectionOnUnitSphere_j_i = (zDetector_j - zCollimator_i) / L_ij;

	float xDetector_k = deviceparameter_Detector[12 * k + 1];
	float yDetector_k = deviceparameter_Detector[12 * k + 2];
	float zDetector_k = deviceparameter_Detector[12 * k + 3];
	float widthDetector_k = deviceparameter_Detector[12 * k + 4];
	float heightDetector_k = deviceparameter_Detector[12 * k + 6];
	float thicknessDetector_k = deviceparameter_Detector[12 * k + 5];
	float R_detector_k = sqrt(widthDetector_k * widthDetector_k + heightDetector_k * heightDetector_k + thicknessDetector_k * thicknessDetector_k) / 2.0f;

	float L_ik = calculateDist(xDetector_k, yDetector_k, zDetector_k, xCollimator_i, yCollimator_i, zCollimator_i);
	float crit_k_i = R_detector_k / L_ik;

	float x_projectionOnUnitSphere_k_i = (xDetector_k - xCollimator_i) / L_ik;
	float y_projectionOnUnitSphere_k_i = (yDetector_k - yCollimator_i) / L_ik;
	float z_projectionOnUnitSphere_k_i = (zDetector_k - zCollimator_i) / L_ik;


	// Whether the cover sphere of detector k is cross with the line between i and j, centered at i 
	float distOnUnitSphere_i = calculateDist(x_projectionOnUnitSphere_k_i, y_projectionOnUnitSphere_k_i, z_projectionOnUnitSphere_k_i, x_projectionOnUnitSphere_j_i, y_projectionOnUnitSphere_j_i, z_projectionOnUnitSphere_j_i);
	if (distOnUnitSphere_i <= crit_k_i + crit_j_i)
	{
		atomicOr(&dst_relation_collimator2crystal[word_idx], 1U << bit_offset);
	}


}



int scatter(float* parameter_Collimator, float* parameter_Detector, float* parameter_Image, float* parameter_Physics,float* PE_SysMat,const char* FnameGeoCrystal, const char* FnameGeoCollimator, float* dst, int cuda_id)
{

	cout << "Get into scatter function" << endl;

	int numPSFImageVoxelX = (int)floor(parameter_Image[0] + 0.001f);
	int numPSFImageVoxelY = (int)floor(parameter_Image[1] + 0.001f);
	int numPSFImageVoxelZ = (int)floor(parameter_Image[2] + 0.001f);

	int numProjectionSingle = (int)floor(parameter_Detector[0]+0.0001f);
	int numImagebin = numPSFImageVoxelX * numPSFImageVoxelY * numPSFImageVoxelZ;
	int numRotation = (int)floor(parameter_Image[6] + 0.001f);
	
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	int device;
	for (device = 0; device < deviceCount; ++device) {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, device);
		printf("Device %d has compute capability %d.%d.\n", device, deviceProp.major, deviceProp.minor);
	}
	if (cuda_id>=deviceCount)
	{
		cout << "cuda_id > the number of GPUs on the host! Set Device to Device 0!" << endl;
	}
	else
	{
		cudaSetDevice(cuda_id);
		cout << "Set Device to Device " <<cuda_id<< endl;
	}
	cudaCheckError(cudaMemcpyToSymbol(deviceXcomMuPhotoelectric,
		kXcomMuPhotoelectric, sizeof(kXcomMuPhotoelectric)));
	cudaCheckError(cudaMemcpyToSymbol(deviceXcomMuCompton,
		kXcomMuCompton, sizeof(kXcomMuCompton)));

	vector<int> detector_materials(numProjectionSingle, kMaterialVacuum);
	int material_counts[kXcomMaterialCount] = {0, 0, 0, 0};
	for (int detector = 0; detector < numProjectionSingle; ++detector)
	{
		detector_materials[detector] = identifyXcomMaterial(
			parameter_Detector[detector * 12 + 8],
			parameter_Detector[detector * 12 + 9],
			parameter_Physics[7]);
		if (detector_materials[detector] >= 0)
			++material_counts[detector_materials[detector]];
	}
	cout << "Detector XCOM materials: NaI=" << material_counts[kMaterialNaI]
		<< " GAGG=" << material_counts[kMaterialGAGG]
		<< " Pb=" << material_counts[kMaterialPb]
		<< " W=" << material_counts[kMaterialW] << endl;
	vector<AxisAlignedLayerGrid> structured_layers;
	vector<int> structured_cell_map;
	const char* structured_environment = getenv("SCATTER_STRUCTURED_TRAVERSAL");
	bool request_structured_traversal = structured_environment == NULL
		|| atoi(structured_environment) != 0;
	bool use_structured_traversal = request_structured_traversal
		&& buildAxisAlignedLayerGrids(
			parameter_Detector, numProjectionSingle,
			&structured_layers, &structured_cell_map);
	cout << "Axis-aligned layer-grid traversal: "
		<< (use_structured_traversal ? "enabled" : "generic bitmap fallback")
		<< " layers=" << structured_layers.size()
		<< " cells=" << structured_cell_map.size() << endl;
	const char* pair_cache_filename = getenv("SCATTER_PAIR_LENGTH_CACHE");
	if (pair_cache_filename != NULL && !use_structured_traversal)
	{
		fprintf(stderr,
			"SCATTER_PAIR_LENGTH_CACHE requires validated axis-aligned layer-grid traversal.\n");
		exit(EXIT_FAILURE);
	}
	PairLengthCache pair_length_cache = openPairLengthCache(
		pair_cache_filename, numProjectionSingle,
		detectorGeometryHash(
			parameter_Detector, detector_materials, numProjectionSingle));

	vector<int> detector_local_scatter_types;
	vector<float2> detector_local_scatter_lookup;
	int detector_local_scatter_orientation_bins = 0;
	int requested_start = getenv("SCATTER_CRYSTAL_START") != NULL
		? atoi(getenv("SCATTER_CRYSTAL_START")) : 0;
	int requested_end = getenv("SCATTER_CRYSTAL_END") != NULL
		? atoi(getenv("SCATTER_CRYSTAL_END")) : numProjectionSingle;
	bool requested_partial = requested_start > 0
		|| requested_end < numProjectionSingle;
	bool prepare_global_components = !requested_partial;
	if (getenv("SCATTER_INCLUDE_GLOBAL_COMPONENTS") != NULL)
		prepare_global_components
			= atoi(getenv("SCATTER_INCLUDE_GLOBAL_COMPONENTS")) != 0;
	if (prepare_global_components)
	{
		buildDetectorLocalScatterLookup(
			parameter_Detector,
			parameter_Physics,
			detector_materials,
			numProjectionSingle,
			&detector_local_scatter_types,
			&detector_local_scatter_lookup,
			&detector_local_scatter_orientation_bins);
	}
	else
	{
		cout << "Detector-local lookup skipped for crystal-range partial." << endl;
	}

	cudaStream_t stream;
	cudaStreamCreate(&stream);
	initializeComptonNormalization<<<1, 1, 0, stream>>>(parameter_Physics[7]);
	cudaCheckError(cudaGetLastError());
	cudaCheckError(cudaStreamSynchronize(stream));
	cout << "Compton normalization computed once for source energy "
		<< parameter_Physics[7] << " keV" << endl;
	float* deviceComptonPhasePrefix = NULL;
	const char* phase_lut_environment = getenv("SCATTER_COMPTON_INTEGRAND_LUT");
	if (phase_lut_environment == NULL)
		phase_lut_environment = getenv("SCATTER_COMPTON_PHASE_LUT");
	bool enable_phase_lut = phase_lut_environment == NULL
		|| atoi(phase_lut_environment) != 0;
	if (enable_phase_lut)
	{
		cudaCheckError(cudaMalloc(&deviceComptonPhasePrefix,
			kComptonPhasePrefixCount * sizeof(float)));
		initializeComptonPhasePrefix<<<
			(kComptonPhasePrefixCount + 255) / 256, 256, 0, stream>>>(
			deviceComptonPhasePrefix, parameter_Physics[7]);
		cudaCheckError(cudaGetLastError());

		const int validation_samples = 2048;
		float2* device_validation_errors = NULL;
		cudaCheckError(cudaMalloc(&device_validation_errors,
			validation_samples * sizeof(float2)));
		validateComptonPhasePrefix<<<8, 256, 0, stream>>>(
			deviceComptonPhasePrefix, parameter_Physics[7],
			device_validation_errors, validation_samples);
		cudaCheckError(cudaGetLastError());
		vector<float2> validation_errors(validation_samples);
		cudaCheckError(cudaMemcpyAsync(validation_errors.data(),
			device_validation_errors, validation_samples * sizeof(float2),
			cudaMemcpyDeviceToHost, stream));
		cudaCheckError(cudaStreamSynchronize(stream));
		cudaFree(device_validation_errors);
		float maximum_absolute = 0.0f;
		float maximum_relative = 0.0f;
		for (int index = 0; index < validation_samples; ++index)
		{
			maximum_absolute = fmaxf(maximum_absolute, validation_errors[index].x);
			maximum_relative = fmaxf(maximum_relative, validation_errors[index].y);
		}
		cout << "Compton integrand-LUT validation: samples=" << validation_samples
			<< " max_abs=" << maximum_absolute
			<< " max_rel=" << maximum_relative << endl;
		if (maximum_absolute > 2e-5f || maximum_relative > 2e-4f)
		{
			fprintf(stderr, "Compton integrand-LUT validation failed.\n");
			exit(EXIT_FAILURE);
		}
	}
	else
	{
		cout << "Compton integrand LUT disabled; using legacy per-thread integration."
			<< endl;
	}
	//////////////////////////// Promary Compton Scatter between Crystals ////////////////////////////

	////------------------------- Geometry RelationShip between Crystals -------------------------////
	// Calculate geometry relationship between each crystal, if the crystal_k is cross the line of 
	// crystal_i and crystal_j, then deviceGeometryRelationShip_Crystal[i,j,k]=1,else =0.
	// The array deviceGeometryRelationShip_Crystal is calculated in scatter-crystal chunks to avoid
	// allocating the full Ndet*Ndet*Ndet bitmap at once.
	cout << "Start Calculate or IO Geometry RelationShip between Crystals "<< endl;
	int bits_per_word = 32;
	int crystal_chunk_size = 16;
	const char* env_chunk = getenv("SCATTER_CRYSTAL_CHUNK");
	if (env_chunk != NULL)
	{
		int parsed_chunk = atoi(env_chunk);
		if (parsed_chunk > 0)
		{
			crystal_chunk_size = parsed_chunk;
		}
	}
	if (crystal_chunk_size > numProjectionSingle)
	{
		crystal_chunk_size = numProjectionSingle;
	}
	cout << "Crystal relationship chunk size: " << crystal_chunk_size << endl;
	int crystal_range_start = 0;
	int crystal_range_end = numProjectionSingle;
	const char* range_start_environment = getenv("SCATTER_CRYSTAL_START");
	const char* range_end_environment = getenv("SCATTER_CRYSTAL_END");
	if (range_start_environment != NULL)
		crystal_range_start = atoi(range_start_environment);
	if (range_end_environment != NULL)
		crystal_range_end = atoi(range_end_environment);
	crystal_range_start = crystal_range_start < 0 ? 0
		: (crystal_range_start > numProjectionSingle
			? numProjectionSingle : crystal_range_start);
	crystal_range_end = crystal_range_end < 0 ? 0
		: (crystal_range_end > numProjectionSingle
			? numProjectionSingle : crystal_range_end);
	if (crystal_range_start >= crystal_range_end)
	{
		fprintf(stderr, "Invalid scatter crystal range [%d,%d).\n",
			crystal_range_start, crystal_range_end);
		exit(EXIT_FAILURE);
	}
	bool partial_crystal_range = crystal_range_start != 0
		|| crystal_range_end != numProjectionSingle;
	const char* global_components_environment
		= getenv("SCATTER_INCLUDE_GLOBAL_COMPONENTS");
	bool include_global_components = !partial_crystal_range;
	if (global_components_environment != NULL)
		include_global_components = atoi(global_components_environment) != 0;
	const char* component_output_environment = getenv("SCATTER_WRITE_COMPONENTS");
	const bool write_scatter_components = component_output_environment != NULL
		&& atoi(component_output_environment) != 0;
	cout << "Scatter crystal source range: [" << crystal_range_start << ","
		<< crystal_range_end << ") of " << numProjectionSingle
		<< "; detector-local/collimator components="
		<< (include_global_components ? "included" : "excluded") << endl;
	cout << "Diagnostic component matrices: "
		<< (write_scatter_components ? "enabled" : "disabled") << endl;
	const char* pruning_environment = getenv("SCATTER_KINEMATIC_PRUNING");
	bool enable_kinematic_pruning = pruning_environment == NULL
		|| atoi(pruning_environment) != 0;
	cout << "Conservative crystal-pair energy/FOV pruning: "
		<< (enable_kinematic_pruning ? "enabled" : "disabled") << endl;
	int target_face_subdivisions = positiveEnvironmentInteger(
		"SCATTER_TARGET_FACE_SUBDIV", 1);
	int near_target_face_subdivisions = positiveEnvironmentInteger(
		"SCATTER_NEAR_TARGET_FACE_SUBDIV", 8);
	float near_target_distance_factor = positiveEnvironmentFloat(
		"SCATTER_NEAR_TARGET_DISTANCE_FACTOR", 2.0f);
	target_face_subdivisions = min(target_face_subdivisions, 16);
	near_target_face_subdivisions = min(near_target_face_subdivisions, 16);
	cout << "Inter-crystal target surface quadrature: far="
		<< target_face_subdivisions << "x" << target_face_subdivisions
		<< " near=" << near_target_face_subdivisions << "x"
		<< near_target_face_subdivisions
		<< " near_distance_factor=" << near_target_distance_factor << endl;
	cout << "Per full bitmap would require "
		<< (static_cast<double>(numProjectionSingle) * numProjectionSingle * numProjectionSingle / 8.0 / 1024.0 / 1024.0 / 1024.0)
		<< " GiB; chunked mode recomputes per block instead of saving/loading " << FnameGeoCrystal << endl;


	float* h_parameter_Collimator;
	cudaMallocHost(&h_parameter_Collimator, sizeof(float) * 200000);
	memcpy(h_parameter_Collimator, parameter_Collimator, sizeof(float) * 200000);

	float* h_parameter_Detector;
	cudaMallocHost(&h_parameter_Detector, sizeof(float) * 200000);
	memcpy(h_parameter_Detector, parameter_Detector, sizeof(float) * 200000);

	float* h_parameter_Image;
	cudaMallocHost(&h_parameter_Image, sizeof(float) * 100);
	memcpy(h_parameter_Image, parameter_Image, sizeof(float) * 100);

	float* h_parameter_Physics;
	cudaMallocHost(&h_parameter_Physics, sizeof(float) * 100);
	memcpy(h_parameter_Physics, parameter_Physics, sizeof(float) * 100);

	float* h_PE_SysMat;
	cudaMallocHost(&h_PE_SysMat, sizeof(float) * numProjectionSingle * numImagebin);
	memcpy(h_PE_SysMat, PE_SysMat, sizeof(float) * numProjectionSingle * numImagebin);

	const size_t matrix_element_count
		= static_cast<size_t>(numProjectionSingle) * numImagebin;
	const size_t matrix_byte_count = matrix_element_count * sizeof(float);

	// Allocate memory on device
	float* deviceMatrix_crystal;
	cudaMalloc(&deviceMatrix_crystal, matrix_byte_count);
	cudaMemset(deviceMatrix_crystal, 0, matrix_byte_count);

	float* deviceMatrix_component = NULL;
	if (write_scatter_components)
	{
		cudaCheckError(cudaMalloc(&deviceMatrix_component, matrix_byte_count));
		cudaCheckError(cudaMemset(deviceMatrix_component, 0, matrix_byte_count));
	}

	float* deviceMatrix_collimator;
	cudaMalloc(&deviceMatrix_collimator, sizeof(float) * numProjectionSingle * numImagebin);
	cudaMemset(deviceMatrix_collimator, 0, sizeof(float) * numProjectionSingle * numImagebin);

	float* devicePEMatrix;
	cudaMalloc(&devicePEMatrix, sizeof(float) * numProjectionSingle * numImagebin);
	cudaMemcpyAsync(devicePEMatrix, h_PE_SysMat, sizeof(float) * numProjectionSingle * numImagebin, cudaMemcpyHostToDevice, stream);

	float* deviceparameter_Collimator;
	cudaMalloc(&deviceparameter_Collimator, sizeof(float) * 200000);
	cudaMemcpyAsync(deviceparameter_Collimator, h_parameter_Collimator, sizeof(float) * 200000, cudaMemcpyHostToDevice, stream);

	float* deviceparameter_Detector;
	cudaMalloc(&deviceparameter_Detector, sizeof(float) * 200000);
	cudaMemcpyAsync(deviceparameter_Detector, h_parameter_Detector, sizeof(float) * 200000, cudaMemcpyHostToDevice, stream);

	float* deviceparameter_Image;
	cudaMalloc(&deviceparameter_Image, sizeof(float) * 100);
	cudaMemcpyAsync(deviceparameter_Image, h_parameter_Image, sizeof(float) * 100, cudaMemcpyHostToDevice, stream);

	float* deviceparameter_Physics;
	cudaMalloc(&deviceparameter_Physics, sizeof(float) * 100);
	cudaMemcpyAsync(deviceparameter_Physics, h_parameter_Physics, sizeof(float) * 100, cudaMemcpyHostToDevice, stream);

	int* deviceDetectorMaterial;
	cudaMalloc(&deviceDetectorMaterial, sizeof(int) * numProjectionSingle);
	cudaMemcpyAsync(deviceDetectorMaterial, detector_materials.data(),
		sizeof(int) * numProjectionSingle, cudaMemcpyHostToDevice, stream);
	AxisAlignedLayerGrid* deviceStructuredLayers = NULL;
	int* deviceStructuredCellMap = NULL;
	if (use_structured_traversal)
	{
		cudaCheckError(cudaMalloc(&deviceStructuredLayers,
			structured_layers.size() * sizeof(AxisAlignedLayerGrid)));
		cudaCheckError(cudaMemcpyAsync(deviceStructuredLayers,
			structured_layers.data(),
			structured_layers.size() * sizeof(AxisAlignedLayerGrid),
			cudaMemcpyHostToDevice, stream));
		cudaCheckError(cudaMalloc(&deviceStructuredCellMap,
			structured_cell_map.size() * sizeof(int)));
		cudaCheckError(cudaMemcpyAsync(deviceStructuredCellMap,
			structured_cell_map.data(),
			structured_cell_map.size() * sizeof(int),
			cudaMemcpyHostToDevice, stream));
	}

	int* deviceLocalScatterType = NULL;
	float2* deviceLocalScatterLookup = NULL;
	if (!detector_local_scatter_lookup.empty())
	{
		cudaCheckError(cudaMalloc(&deviceLocalScatterType,
			sizeof(int) * numProjectionSingle));
		cudaCheckError(cudaMemcpyAsync(deviceLocalScatterType,
			detector_local_scatter_types.data(), sizeof(int) * numProjectionSingle,
			cudaMemcpyHostToDevice, stream));
		cudaCheckError(cudaMalloc(&deviceLocalScatterLookup,
			sizeof(float2) * detector_local_scatter_lookup.size()));
		cudaCheckError(cudaMemcpyAsync(deviceLocalScatterLookup,
			detector_local_scatter_lookup.data(),
			sizeof(float2) * detector_local_scatter_lookup.size(),
			cudaMemcpyHostToDevice, stream));
	}

	auto start_crystalScatterSysMatCuda = std::chrono::high_resolution_clock::now();
	dim3 blockSize(16, 16, 1);
	cout << "Kernel crystalScatterSysMatCuda Launched in chunks" << endl;
	long long maximum_chunk_bits = static_cast<long long>(crystal_chunk_size)
		* numProjectionSingle * numProjectionSingle;
	long long maximum_chunk_words = (maximum_chunk_bits + bits_per_word - 1)
		/ bits_per_word;
	long long maximum_pair_count = static_cast<long long>(crystal_chunk_size)
		* numProjectionSingle;
	unsigned int* deviceGeometryRelationShip_Crystal2Crystal = NULL;
	CrystalPairPath* deviceCrystalPairPaths = NULL;
	float4* deviceCachedPairLengths = NULL;
	vector<float4> hostCachedPairLengths;
	if (!use_structured_traversal)
	{
		cudaCheckError(cudaMalloc(&deviceGeometryRelationShip_Crystal2Crystal,
			maximum_chunk_words * sizeof(unsigned int)));
	}
	cudaCheckError(cudaMalloc(&deviceCrystalPairPaths,
		maximum_pair_count * sizeof(CrystalPairPath)));
	if (pair_length_cache.file_descriptor >= 0)
	{
		cudaCheckError(cudaMalloc(&deviceCachedPairLengths,
			maximum_pair_count * sizeof(float4)));
		hostCachedPairLengths.resize(maximum_pair_count);
	}
	for (int scatterStart = crystal_range_start;
		scatterStart < crystal_range_end;
		scatterStart += crystal_chunk_size)
	{
		int scatterCount = crystal_chunk_size;
		if (scatterStart + scatterCount > crystal_range_end)
		{
			scatterCount = crystal_range_end - scatterStart;
		}

		long long chunk_total_bits = static_cast<long long>(scatterCount) * static_cast<long long>(numProjectionSingle) * static_cast<long long>(numProjectionSingle);
		long long chunk_array_size = (chunk_total_bits + bits_per_word - 1) / bits_per_word;
		double chunk_gib = static_cast<double>(chunk_array_size) * sizeof(unsigned int) / 1024.0 / 1024.0 / 1024.0;
		cout << "Crystal chunk scatterStart=" << scatterStart
			<< " scatterCount=" << scatterCount
			<< " bitmap=" << (use_structured_traversal ? 0 : chunk_array_size)
			<< " uint32 (" << (use_structured_traversal ? 0.0 : chunk_gib)
			<< " GiB)" << endl;

		long long pair_count = static_cast<long long>(scatterCount) * numProjectionSingle;
		int pair_blocks = static_cast<int>((pair_count + 255) / 256);
		initializeCrystalPairPaths<<<pair_blocks, 256, 0, stream>>>(
			deviceCrystalPairPaths, deviceparameter_Detector,
			deviceparameter_Image, deviceparameter_Physics,
			numProjectionSingle, scatterStart, scatterCount,
			enable_kinematic_pruning ? 1 : 0);
		cudaCheckError(cudaGetLastError());
		bool pair_cache_hit = readPairLengthCacheRows(
			pair_length_cache, scatterStart, scatterCount,
			hostCachedPairLengths.empty() ? NULL : hostCachedPairLengths.data());
		if (pair_cache_hit)
		{
			cudaCheckError(cudaMemcpyAsync(deviceCachedPairLengths,
				hostCachedPairLengths.data(), pair_count * sizeof(float4),
				cudaMemcpyHostToDevice, stream));
			applyCrystalPairMaterialLengths<<<pair_blocks, 256, 0, stream>>>(
				deviceCrystalPairPaths, deviceCachedPairLengths, pair_count);
			cudaCheckError(cudaGetLastError());
		}
		auto start_pair_reduction = std::chrono::high_resolution_clock::now();
		if (pair_cache_hit)
		{
			cout << "Pair material-path cache hit for A=[" << scatterStart << ","
				<< scatterStart + scatterCount << ")" << endl;
		}
		else if (use_structured_traversal)
		{
			buildStructuredCrystalPairMaterialPaths<<<pair_blocks, 256, 0, stream>>>(
				deviceCrystalPairPaths,
				deviceparameter_Detector,
				deviceDetectorMaterial,
				deviceStructuredLayers,
				deviceStructuredCellMap,
				static_cast<int>(structured_layers.size()),
				numProjectionSingle,
				scatterStart,
				scatterCount,
				pair_length_cache.file_descriptor >= 0 ? 1 : 0);
		}
		else
		{
			cudaCheckError(cudaMemset(deviceGeometryRelationShip_Crystal2Crystal,
				0, chunk_array_size * sizeof(unsigned int)));
			long long total_threads = static_cast<long long>(scatterCount)
				* numProjectionSingle * numProjectionSingle;
			int threads_per_block = 256;
			long long blocks_per_grid = (total_threads + threads_per_block - 1)
				/ threads_per_block;
			cout << "Launching geometryRelationShip_Crystal2Crystal_Chunk with "
				<< blocks_per_grid << " blocks of " << threads_per_block
				<< " threads each." << endl;
			geometryRelationShip_Crystal2Crystal_Chunk<<<
				blocks_per_grid, threads_per_block, 0, stream>>>(
					deviceGeometryRelationShip_Crystal2Crystal,
					deviceparameter_Detector,
					deviceCrystalPairPaths,
					scatterStart,
					scatterCount);
			cudaCheckError(cudaGetLastError());
			reduceCrystalPairMaterialPaths<<<pair_blocks, 256, 0, stream>>>(
				deviceCrystalPairPaths,
				deviceparameter_Detector,
				deviceDetectorMaterial,
				deviceGeometryRelationShip_Crystal2Crystal,
				numProjectionSingle,
				scatterStart,
				scatterCount);
		}
		cudaCheckError(cudaGetLastError());
		cudaCheckError(cudaStreamSynchronize(stream));
		if (!pair_cache_hit && pair_length_cache.file_descriptor >= 0)
		{
			gatherCrystalPairMaterialLengths<<<pair_blocks, 256, 0, stream>>>(
				deviceCachedPairLengths, deviceCrystalPairPaths, pair_count);
			cudaCheckError(cudaGetLastError());
			cudaCheckError(cudaMemcpyAsync(hostCachedPairLengths.data(),
				deviceCachedPairLengths, pair_count * sizeof(float4),
				cudaMemcpyDeviceToHost, stream));
			cudaCheckError(cudaStreamSynchronize(stream));
			writePairLengthCacheRows(pair_length_cache,
				scatterStart, scatterCount, hostCachedPairLengths.data());
		}
		auto end_pair_reduction = std::chrono::high_resolution_clock::now();
		cout << "Pair material-path generation ("
			<< (use_structured_traversal ? "layer-grid" : "generic bitmap") << "): "
			<< std::chrono::duration_cast<std::chrono::milliseconds>(
				end_pair_reduction - start_pair_reduction).count() / 1000.0
			<< " s" << endl;
		const char* debug_source_text = getenv("SCATTER_DEBUG_PAIR_SOURCE");
		const char* debug_target_text = getenv("SCATTER_DEBUG_PAIR_TARGET");
		if (debug_source_text != NULL && debug_target_text != NULL)
		{
			int debug_source = atoi(debug_source_text);
			int debug_target = atoi(debug_target_text);
			if (debug_source >= scatterStart
				&& debug_source < scatterStart + scatterCount
				&& debug_target >= 0 && debug_target < numProjectionSingle)
			{
				long long debug_index = static_cast<long long>(
					debug_source - scatterStart) * numProjectionSingle + debug_target;
				CrystalPairPath debug_path;
				cudaCheckError(cudaMemcpy(&debug_path,
					deviceCrystalPairPaths + debug_index,
					sizeof(CrystalPairPath), cudaMemcpyDeviceToHost));
				cout << "Debug crystal pair A=" << debug_source
					<< " B=" << debug_target
					<< " material_lengths=[" << debug_path.material_lengths.x
					<< "," << debug_path.material_lengths.y
					<< "," << debug_path.material_lengths.z
					<< "," << debug_path.material_lengths.w << "]"
					<< " source_exit=" << debug_path.source_exit_length
					<< " target_absorption=" << debug_path.target_absorption_length
					<< " distance=" << debug_path.direction_distance.w << endl;
			}
		}

		dim3 gridSize(
			(numProjectionSingle + 15) / 16,
			(numImagebin + 15) / 16,
			1
		);

		cout << "Launching crystalScatterSurfaceSysMatCuda chunk grid "
			<< gridSize.x << " x " << gridSize.y << " x " << gridSize.z << endl;
		crystalScatterSurfaceSysMatCuda<<<gridSize, blockSize, 0, stream>>>(
			deviceMatrix_crystal,
			deviceMatrix_component,
			deviceparameter_Detector,
			deviceDetectorMaterial,
			deviceparameter_Image,
			deviceparameter_Physics,
			devicePEMatrix,
			deviceCrystalPairPaths,
			numProjectionSingle,
			numImagebin,
			scatterStart,
			scatterCount,
			target_face_subdivisions,
			near_target_face_subdivisions,
			near_target_distance_factor);
		cudaCheckError(cudaGetLastError());
		cudaCheckError(cudaStreamSynchronize(stream));
	}
	cudaFree(deviceCrystalPairPaths);
	if (deviceCachedPairLengths != NULL) cudaFree(deviceCachedPairLengths);
	if (deviceGeometryRelationShip_Crystal2Crystal != NULL)
		cudaFree(deviceGeometryRelationShip_Crystal2Crystal);
	if (deviceStructuredLayers != NULL) cudaFree(deviceStructuredLayers);
	if (deviceStructuredCellMap != NULL) cudaFree(deviceStructuredCellMap);
	if (pair_length_cache.file_descriptor >= 0)
		close(pair_length_cache.file_descriptor);

	float* h_Crystal_SysMat;
	cudaMallocHost(&h_Crystal_SysMat, matrix_byte_count);
	float* h_Component_SysMat = NULL;
	if (write_scatter_components)
	{
		cudaMallocHost(&h_Component_SysMat, matrix_byte_count);
		cudaCheckError(cudaMemcpyAsync(h_Crystal_SysMat,
			deviceMatrix_crystal, matrix_byte_count, cudaMemcpyDeviceToHost, stream));
		cudaCheckError(cudaMemcpyAsync(h_Component_SysMat,
			deviceMatrix_component, matrix_byte_count, cudaMemcpyDeviceToHost, stream));
		cudaCheckError(cudaStreamSynchronize(stream));
		const int rotation_index = static_cast<int>(floorf(parameter_Image[20] + 0.5f));
		writeScatterComponentSlice("C_intercrystal.sysmat", h_Component_SysMat,
			matrix_element_count, rotation_index);
		for (size_t index = 0; index < matrix_element_count; ++index)
			h_Component_SysMat[index] = fmaxf(0.0f,
				h_Crystal_SysMat[index] - h_Component_SysMat[index]);
		writeScatterComponentSlice("C_highZ_to_crystal.sysmat", h_Component_SysMat,
			matrix_element_count, rotation_index);
	}

	if (deviceLocalScatterLookup != NULL && include_global_components)
	{
		auto start_local_scatter = std::chrono::high_resolution_clock::now();
		dim3 local_block(16, 16, 1);
		dim3 local_grid(
			(numProjectionSingle + local_block.x - 1) / local_block.x,
			(numImagebin + local_block.y - 1) / local_block.y,
			1);
		cout << "Launching detectorLocalScatterSysMatCuda grid "
			<< local_grid.x << " x " << local_grid.y
			<< " for recoil-escape and same-crystal Compton+PE responses." << endl;
		detectorLocalScatterSysMatCuda<<<local_grid, local_block, 0, stream>>>(
			deviceMatrix_crystal,
			deviceparameter_Detector,
			deviceparameter_Image,
			deviceparameter_Physics,
			devicePEMatrix,
			deviceLocalScatterType,
			deviceLocalScatterLookup,
			detector_local_scatter_orientation_bins,
			numProjectionSingle,
			numImagebin,
			0);
		cudaCheckError(cudaGetLastError());
		cudaCheckError(cudaStreamSynchronize(stream));
		if (write_scatter_components)
		{
			const int rotation_index
				= static_cast<int>(floorf(parameter_Image[20] + 0.5f));
			cudaCheckError(cudaMemset(deviceMatrix_component, 0, matrix_byte_count));
			detectorLocalScatterSysMatCuda<<<local_grid, local_block, 0, stream>>>(
				deviceMatrix_component,
				deviceparameter_Detector,
				deviceparameter_Image,
				deviceparameter_Physics,
				devicePEMatrix,
				deviceLocalScatterType,
				deviceLocalScatterLookup,
				detector_local_scatter_orientation_bins,
				numProjectionSingle,
				numImagebin,
				1);
			cudaCheckError(cudaGetLastError());
			cudaCheckError(cudaMemcpyAsync(h_Component_SysMat,
				deviceMatrix_component, matrix_byte_count,
				cudaMemcpyDeviceToHost, stream));
			cudaCheckError(cudaStreamSynchronize(stream));
			writeScatterComponentSlice("C_local_recoil.sysmat", h_Component_SysMat,
				matrix_element_count, rotation_index);

			cudaCheckError(cudaMemset(deviceMatrix_component, 0, matrix_byte_count));
			detectorLocalScatterSysMatCuda<<<local_grid, local_block, 0, stream>>>(
				deviceMatrix_component,
				deviceparameter_Detector,
				deviceparameter_Image,
				deviceparameter_Physics,
				devicePEMatrix,
				deviceLocalScatterType,
				deviceLocalScatterLookup,
				detector_local_scatter_orientation_bins,
				numProjectionSingle,
				numImagebin,
				2);
			cudaCheckError(cudaGetLastError());
			cudaCheckError(cudaMemcpyAsync(h_Component_SysMat,
				deviceMatrix_component, matrix_byte_count,
				cudaMemcpyDeviceToHost, stream));
			cudaCheckError(cudaStreamSynchronize(stream));
			writeScatterComponentSlice("C_local_self_photoelectric.sysmat",
				h_Component_SysMat, matrix_element_count, rotation_index);
		}
		auto end_local_scatter = std::chrono::high_resolution_clock::now();
		auto local_scatter_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
			end_local_scatter - start_local_scatter).count();
		cout << "Time of detector local scatter response: "
			<< local_scatter_ms / 1000.0 / 60.0 << " min" << endl;
	}
	else if (write_scatter_components)
	{
		memset(h_Component_SysMat, 0, matrix_byte_count);
		const int rotation_index
			= static_cast<int>(floorf(parameter_Image[20] + 0.5f));
		writeScatterComponentSlice("C_local_recoil.sysmat", h_Component_SysMat,
			matrix_element_count, rotation_index);
		writeScatterComponentSlice("C_local_self_photoelectric.sysmat",
			h_Component_SysMat, matrix_element_count, rotation_index);
	}
	
	cudaMemcpyAsync(h_Crystal_SysMat, deviceMatrix_crystal,
		matrix_byte_count, cudaMemcpyDeviceToHost, stream);
	cudaCheckError(cudaGetLastError());
	cudaStreamSynchronize(stream);
	
	/* // Write CrystalScatter_SysMat.sysmat for debug
	char Fnametmp[2048];
	sprintf(Fnametmp, "CrystalScatter_SysMat.sysmat");
	FILE* fptmp;
	fptmp = fopen(Fnametmp, "wb+");
	if (fptmp == 0) { puts("error"); exit(0); }
	fwrite(h_Crystal_SysMat, sizeof(float), numProjectionSingle * numImagebin, fptmp);
	fclose(fptmp);
	cout << "Write Results of crystalScatterSysMatCuda function finished " << endl;
	*/


	auto end_crystalScatterSysMatCuda = std::chrono::high_resolution_clock::now();
	auto duration_crystalScatterSysMatCuda = std::chrono::duration_cast<std::chrono::milliseconds>(end_crystalScatterSysMatCuda - start_crystalScatterSysMatCuda);
	cout << "Time of crystalScatterSysMatCuda function: " << duration_crystalScatterSysMatCuda.count()/1000.0/60.0 << " min" << endl;
	cudaStreamSynchronize(stream);
	cudaFreeHost(h_PE_SysMat);
	cudaFree(devicePEMatrix);
	cudaFree(deviceMatrix_crystal);
	if (deviceMatrix_component != NULL) cudaFree(deviceMatrix_component);
	if (deviceLocalScatterType != NULL) cudaFree(deviceLocalScatterType);
	if (deviceLocalScatterLookup != NULL) cudaFree(deviceLocalScatterLookup);
	cout << "########################" << endl;

	///////////////////////////////////////////////////////////////////////////////////////////////////

	///////////////// Promary Compton Scatter between Collimator and Detector Crystal /////////////////

	vector<CollimatorScatterSample> collimator_samples;
	if (include_global_components)
		collimator_samples = buildCollimatorScatterSamples(
			parameter_Collimator, parameter_Physics[7]);
	int numCollimatorSamples = static_cast<int>(collimator_samples.size());
	if (numCollimatorSamples <= 0)
	{
		cout << "No attenuating collimator volume configured; skipping collimator scatter." << endl;
		memcpy(dst, h_Crystal_SysMat, sizeof(float) * numProjectionSingle * numImagebin);
		if (write_scatter_components)
		{
			const int rotation_index
				= static_cast<int>(floorf(parameter_Image[20] + 0.5f));
			memset(h_Component_SysMat, 0, matrix_byte_count);
			writeScatterComponentSlice("C_collimator_to_crystal.sysmat",
				h_Component_SysMat, matrix_element_count, rotation_index);
			writeScatterComponentSlice("C_total.sysmat", dst,
				matrix_element_count, rotation_index);
		}

		cudaFreeHost(h_parameter_Collimator);
		cudaFreeHost(h_parameter_Detector);
		cudaFreeHost(h_parameter_Image);
		cudaFreeHost(h_parameter_Physics);
		cudaFreeHost(h_Crystal_SysMat);
		if (h_Component_SysMat != NULL) cudaFreeHost(h_Component_SysMat);

		cudaFree(deviceparameter_Collimator);
		cudaFree(deviceparameter_Detector);
		cudaFree(deviceparameter_Image);
			cudaFree(deviceparameter_Physics);
			cudaFree(deviceDetectorMaterial);
			cudaFree(deviceMatrix_collimator);
			if (deviceComptonPhasePrefix != NULL) cudaFree(deviceComptonPhasePrefix);
			cudaStreamDestroy(stream);

		return numImagebin;
	}

	long long total_bits = static_cast<long long>(numProjectionSingle)
		* static_cast<long long>(numProjectionSingle)
		* static_cast<long long>(numCollimatorSamples);
	long long array_size = (total_bits + bits_per_word - 1) / bits_per_word;


	cout << "Total bits: " << total_bits << endl;
	cout << "Array size (unsigned int): " << array_size << endl;

	CollimatorScatterSample* deviceCollimatorSamples;
	cudaCheckError(cudaMalloc(&deviceCollimatorSamples,
		sizeof(CollimatorScatterSample) * numCollimatorSamples));
	cudaCheckError(cudaMemcpyAsync(deviceCollimatorSamples, collimator_samples.data(),
		sizeof(CollimatorScatterSample) * numCollimatorSamples, cudaMemcpyHostToDevice, stream));

	unsigned int* deviceGeometryRelationShip_Collimator2Crystal;
	cudaCheckError(cudaMalloc(&deviceGeometryRelationShip_Collimator2Crystal,
		array_size * sizeof(unsigned int)));
	cudaCheckError(cudaMemset(deviceGeometryRelationShip_Collimator2Crystal, 0,
		array_size * sizeof(unsigned int)));
	long long total_threads = static_cast<long long>(numCollimatorSamples)
		* static_cast<long long>(numProjectionSingle) * static_cast<long long>(numProjectionSingle);
	int threads_per_block = 256;
	long long blocks_per_grid = (total_threads + threads_per_block - 1) / threads_per_block;
	cout << "Initializing physical collimator-volume relationship with numProjectionSingle="
		<< numProjectionSingle << " and numSamples=" << numCollimatorSamples << endl;
	geometryRelationShip_Collimator2Crystal<<<blocks_per_grid, threads_per_block, 0, stream>>>(
		deviceGeometryRelationShip_Collimator2Crystal,
		deviceparameter_Detector,
		deviceCollimatorSamples,
		numCollimatorSamples);
	cudaCheckError(cudaGetLastError());
	cudaCheckError(cudaStreamSynchronize(stream));

	dim3 blockSize_collimator(16, 16, 1);
	dim3 gridSize_collimator(
		(numProjectionSingle + 15) / 16,
		(numImagebin + 15) / 16,
		numCollimatorSamples
	);

	cout << "########################" << endl;
	cout << "numProjectionSingle = " << numProjectionSingle << endl;
	cout << "numImagebin = " << numImagebin << endl;
	cout << "gridSize.x = " << gridSize_collimator.x << endl;
	cout << "gridSize.y = " << gridSize_collimator.y << endl;
	cout << "gridSize.z = " << gridSize_collimator.z << endl;
	cout << "########################" << endl;

	auto start_CollimatorScatterSysMatCuda = std::chrono::high_resolution_clock::now();
	cout << "Kernel collimatorScatterSysMatCuda Launched " << endl;
	
	collimatorScatterSysMatCuda <<<gridSize_collimator, blockSize_collimator >>> (
		deviceMatrix_collimator,
		deviceCollimatorSamples,
		numCollimatorSamples,
		deviceparameter_Detector,
		deviceDetectorMaterial,
			deviceparameter_Image,
			deviceparameter_Physics,
			deviceComptonPhasePrefix,
			deviceGeometryRelationShip_Collimator2Crystal,
		numProjectionSingle,
		numImagebin);
	
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cerr << "Kernel collimatorScatterSysMatCuda launch failed: " << cudaGetErrorString(err) << std::endl;
	}
	float* h_Collimator_SysMat;
	cudaMallocHost(&h_Collimator_SysMat, sizeof(float) * numProjectionSingle * numImagebin);
	cudaMemcpyAsync(h_Collimator_SysMat, deviceMatrix_collimator, sizeof(float) * numProjectionSingle * numImagebin, cudaMemcpyDeviceToHost, stream);
	cudaCheckError(cudaGetLastError());
	cudaStreamSynchronize(stream);
	
	/* // Write CollimatorScatter_SysMat.sysmat for debug
	sprintf(Fnametmp, "CollimatorScatter_SysMat.sysmat");
	fptmp = fopen(Fnametmp, "wb+");
	if (fptmp == 0) { puts("error"); exit(0); }
	fwrite(h_Collimator_SysMat, sizeof(float), numProjectionSingle * numImagebin, fptmp);
	fclose(fptmp);
	cout << "Write Results of CollimatorScatterSysMatCuda function finished " << endl;
	*/
	auto end_CollimatorScatterSysMatCuda = std::chrono::high_resolution_clock::now();
	auto duration_CollimatorScatterSysMatCuda = std::chrono::duration_cast<std::chrono::milliseconds>(end_CollimatorScatterSysMatCuda - start_CollimatorScatterSysMatCuda);
	cout << "Time of  collimatorScatterSysMatCuda function: " << duration_CollimatorScatterSysMatCuda.count() / 1000.0 / 60.0 << " min" << endl;

	cout << "########################" << endl;
	double crystal_scatter_sum = 0.0;
	double collimator_scatter_sum = 0.0;
	for (size_t i = 0; i < static_cast<size_t>(numProjectionSingle) * numImagebin; i++)
	{
		crystal_scatter_sum += h_Crystal_SysMat[i];
		collimator_scatter_sum += h_Collimator_SysMat[i];
		dst[i] = h_Collimator_SysMat[i] + h_Crystal_SysMat[i];
	}
	double scatter_sum = crystal_scatter_sum + collimator_scatter_sum;
	cout << "Scatter component sums: crystal=" << crystal_scatter_sum
		<< " collimator=" << collimator_scatter_sum
		<< " collimator_fraction="
		<< (scatter_sum > 0.0 ? collimator_scatter_sum / scatter_sum : 0.0)
		<< endl;
	if (write_scatter_components)
	{
		const int rotation_index
			= static_cast<int>(floorf(parameter_Image[20] + 0.5f));
		writeScatterComponentSlice("C_collimator_to_crystal.sysmat",
			h_Collimator_SysMat, matrix_element_count, rotation_index);
		writeScatterComponentSlice("C_total.sysmat", dst,
			matrix_element_count, rotation_index);
	}
	// Release Sources
	cudaFreeHost(h_parameter_Collimator);
	cudaFreeHost(h_parameter_Detector);
	cudaFreeHost(h_parameter_Image);
	cudaFreeHost(h_parameter_Physics);
	cudaFreeHost(h_Collimator_SysMat);
	cudaFreeHost(h_Crystal_SysMat);
	if (h_Component_SysMat != NULL) cudaFreeHost(h_Component_SysMat);

	cudaFree(deviceparameter_Collimator);
	cudaFree(deviceparameter_Detector);
	cudaFree(deviceparameter_Image);
	cudaFree(deviceparameter_Physics);
	cudaFree(deviceDetectorMaterial);
	cudaFree(deviceMatrix_collimator);
	if (deviceComptonPhasePrefix != NULL) cudaFree(deviceComptonPhasePrefix);
	
	cudaFree(deviceGeometryRelationShip_Collimator2Crystal);
	cudaFree(deviceCollimatorSamples);
	cudaStreamDestroy(stream);

	return numImagebin;
}
