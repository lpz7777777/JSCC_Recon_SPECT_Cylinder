#define _CRT_SECURE_NO_WARNINGS

// ScatterGen_CircularHole.cpp
// Description:
//   This program calculates the scatter system matrix, considering only primary Compton events.
//   It should be executed after the PE (photoelectric effect) system matrix generation.
//
// Usage:
//   ./ScatterGen_CircularHole -PE <path_to_PE_SystemMatrix> 
//                -GeoCrystal <path_to_CrystalGeometryRelationship>
//                -GeoCollimator <path_to_CollimatorGeometryRelationship>
//                -cuda <cuda_device_id>
//
// Author: Xingchun Zheng @ tsinghua university
// Last Modified: 2024/12/21
// Version: 1.0



#include <algorithm>
#include <fstream> 
#include <stdio.h>  
#include <stdlib.h>

#include <cstring>
#include <string> 
#include <chrono> 
#include <math.h>
#include <time.h>   
#include <iostream>
#include <vector>

#include<cuda_runtime.h>
#include <device_launch_parameters.h>

#include "scatter.h"
#include "../common/energy_window.h"

using namespace std;

static size_t readFloatFile(
	const char* filename,
	float* destination,
	size_t capacity,
	bool require_full = false)
{
	FILE* file = fopen(filename, "rb");
	if (file == NULL)
	{
		perror(filename);
		exit(EXIT_FAILURE);
	}
	size_t count = fread(destination, sizeof(float), capacity, file);
	if (ferror(file) || count == 0 || (require_full && count != capacity))
	{
		fprintf(stderr, "Cannot read expected float data from %s.\n", filename);
		fclose(file);
		exit(EXIT_FAILURE);
	}
	fclose(file);
	return count;
}

int main(int argc, char* argv[])
{
	float* parameter_Collimator = new float[200000]();
	float* parameter_Detector = new float[200000]();
	float* parameter_Image = new float[100]();
	float* parameter_Physics = new float[100]();

	readFloatFile("Params_Collimator.dat", parameter_Collimator, 200000);
	readFloatFile("Params_Detector.dat", parameter_Detector, 200000);
	readFloatFile("Params_Image.dat", parameter_Image, 100);
	readFloatFile("Params_Physics.dat", parameter_Physics, 100);
	////////////////////////////////////////////////////
	
	int numCollimatorLayers = (int)parameter_Collimator[0];
	float FOV2Collimator0 = parameter_Image[11];
	for (int id_CollimatorLayer = 0; id_CollimatorLayer < numCollimatorLayers; id_CollimatorLayer++)
	{
		cout << "############ Collimator " << id_CollimatorLayer << " ############" << endl;
		cout << "Number of collimator holes = " << parameter_Collimator[(id_CollimatorLayer + 1) * 10 + 0] << endl;	
		cout << "Width of collimator layer(X direction) = " << parameter_Collimator[(id_CollimatorLayer + 1) * 10 + 1] << "mm" << endl;
		cout << "Thickness of collimator layer(Y direction) = " << parameter_Collimator[(id_CollimatorLayer + 1) * 10 + 2] << "mm" << endl;
		cout << "Height of collimator layer(Z direction) = " << parameter_Collimator[(id_CollimatorLayer + 1) * 10 + 3] << "mm" << endl;
		cout << "Collimator Layer to 1st Collimator Layer = " << parameter_Collimator[(id_CollimatorLayer + 1) * 10 + 4] << "mm" << endl;
		cout << "Coeff of collimator layer = " << parameter_Collimator[(id_CollimatorLayer + 1) * 10 + 5] << endl;

	}

	cout << "FOV center to 1st Collimator = " << FOV2Collimator0 << endl;

	////////////////////////////////////////////////////
	int numProjectionsingle =(int) parameter_Detector[0];

	int numImageVoxelX = (int)parameter_Image[0];
	int numImageVoxelY = (int)parameter_Image[1];
	int numImageVoxelZ = (int)parameter_Image[2];
	float widthImageVoxelX = parameter_Image[3];
	float widthImageVoxelY = parameter_Image[4];
	float widthImageVoxelZ = parameter_Image[5];
	int numRotation_ = (int)floor(parameter_Image[6]+0.001);
	float shiftFOVX= parameter_Image[8];
	float shiftFOVY = parameter_Image[9];
	float shiftFOVZ = parameter_Image[10];

	const int numProjectionSingle = numProjectionsingle;
	const int numImagebin = numImageVoxelX * numImageVoxelY * numImageVoxelZ;
	const int numRotation = numRotation_;

	string FnamePE;
	string FnameGeoCrystal = "GeometryRelationShip_Crystal2Crystal"; 
	string FnameGeoCollimator = "GeometryRelationShip_Collimator2Crystal";
	int cuda_id = 0; 

	for (int i = 1; i < argc; ++i)
	{
		if (strcmp(argv[i], "-PE") == 0 && i + 1 < argc)
		{
			FnamePE = argv[i + 1];
			i++; 
		}
		else if (strcmp(argv[i], "-GeoCrystal") == 0 && i + 1 < argc)
		{
			FnameGeoCrystal = argv[i + 1];
			parameter_Physics[8] = 0;
			i++; 
		}
		else if (strcmp(argv[i], "-GeoCollimator") == 0 && i + 1 < argc)
		{
			FnameGeoCollimator= argv[i + 1];
			parameter_Physics[9] = 0;
			i++;
		}
		else if (strcmp(argv[i], "-cuda") == 0 && i + 1 < argc)
		{
			cuda_id = atoi(argv[i + 1]);
			i++;
		}
		else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
		{
			cout << "Usage: " << argv[0] << " [-PE PE_SysMat_path] [-GeoCrystal GeometryRelationShip_Crystal2Crystal_path] [-GeoCollimator GeometryRelationShip_Collimator2Crystal_path] [-cuda GPU_ID]" << endl;
			return 0;
		}
		else
		{
			cerr << "Unknown parameter or missing argument: " << argv[i] << endl;
			cout << "Usage: " << argv[0] << " [-PE PE_SysMat_path] [-GeoCrystal GeometryRelationShip_Crystal2Crystal_path] [-GeoCollimator GeometryRelationShip_Collimator2Crystal_path] " << endl;
			return EXIT_FAILURE;
		}
	}

	//////////////////////////// PE SysMat Loading ///////////////////////////////
	float* PE_SysMat = new float[numProjectionSingle * numImagebin * numRotation]();

	if (FnamePE.empty())
	{
		char bufferPE[2048];
		snprintf(bufferPE, sizeof(bufferPE), "PE_SysMat_shift_%f_%f_%f_v3.sysmat", shiftFOVX, shiftFOVY, shiftFOVZ);
		FnamePE = bufferPE;
	}

	cout << "Photon Electric SysMat: " << FnamePE << endl;

	auto start_ioPE = std::chrono::high_resolution_clock::now();
	readFloatFile(FnamePE.c_str(), PE_SysMat,
		static_cast<size_t>(numProjectionSingle) * numImagebin * numRotation, true);
	auto end_ioPE = std::chrono::high_resolution_clock::now();
	auto duration_ioPE = std::chrono::duration_cast<std::chrono::milliseconds>(end_ioPE - start_ioPE);
	cout << "Time of io PE System Matrix: " << duration_ioPE.count() << " ms" << endl;

	//////////////////////////// Scatter Function Start ///////////////////////////////
	cout << "Geometry RelationShip Crystal2Crystal:  " << FnameGeoCrystal << endl;
	cout << "Geometry RelationShip Collimator2Crystal:  " << FnameGeoCollimator << endl;

	auto start_scatter = std::chrono::high_resolution_clock::now();

	float* out = new float[numProjectionSingle * numImagebin * numRotation]();

	printf("FOV dimension : %d %d %d\n", numImageVoxelX, numImageVoxelY, numImageVoxelZ);
	printf("FOV Voxel Size(mm) : %f %f %f\n", widthImageVoxelX, widthImageVoxelY, widthImageVoxelZ);
	for (int idxRotation = 0; idxRotation < numRotation; idxRotation++)
	{
		cout << "########################" << endl;
		cout << "Rotation (" << idxRotation << ") processing ..." << endl;
		cout << "########################" << endl;


		cout << "Shift FOV in X = " << shiftFOVX << "mm" << endl;
		cout << "Shift FOV in Y = " << shiftFOVY << "mm" << endl;
		cout << "Shift FOV in Z = " << shiftFOVZ << "mm" << endl;

		parameter_Image[20] = float(idxRotation);

		const size_t rotationOffset = static_cast<size_t>(idxRotation)
			* static_cast<size_t>(numProjectionSingle) * static_cast<size_t>(numImagebin);
		int q = scatter(parameter_Collimator, parameter_Detector, parameter_Image,
			parameter_Physics, PE_SysMat + rotationOffset,
			FnameGeoCrystal.c_str(), FnameGeoCollimator.c_str(),
			out + rotationOffset, cuda_id);

		printf("numImagebin = %d\n", q);
	}

	auto end_scatter = std::chrono::high_resolution_clock::now();
	auto duration_scatter = std::chrono::duration_cast<std::chrono::milliseconds>(end_scatter - start_scatter);
	cout << "Time of scatter function: " << duration_scatter.count()/1000.0/60.0 << " min" << endl;

	if (parameter_Physics[2] == 1)
	{
		char Fname[2048];
		sprintf(Fname, "Scatter_SysMat_shift_%f_%f_%f.sysmat", shiftFOVX, shiftFOVY, shiftFOVZ);
		FILE* fp1;
		fp1 = fopen(Fname, "wb+");
		if (fp1 == 0) { puts("error"); exit(0); }
		fwrite(out, sizeof(float), numProjectionSingle * numImagebin * numRotation, fp1);
		fclose(fp1);

		cout << "########################" << endl;
		cout << "Compton Scatter Sysmat written." << endl;
		cout << "########################" << endl;
	}
	if (parameter_Physics[3] == 1) 
	{
		vector<float> photopeak_acceptance(numProjectionSingle, 0.0f);
		for (int row = 0; row < numProjectionSingle; ++row)
		{
			photopeak_acceptance[row] = photopeak_energy_window_acceptance(
				parameter_Physics, parameter_Detector + row * 12 + 1);
		}
		const size_t total_elements = static_cast<size_t>(numProjectionSingle)
			* static_cast<size_t>(numImagebin) * static_cast<size_t>(numRotation);
		char Fname3[2048];
		sprintf(Fname3, "SysMat_withScatter_shift_%f_%f_%f.sysmat", shiftFOVX, shiftFOVY, shiftFOVZ);
		FILE* fp2;
		fp2 = fopen(Fname3, "wb+");
		if (fp2 == 0) { perror(Fname3); exit(EXIT_FAILURE); }
		const size_t combined_chunk_elements = 1 << 20;
		vector<float> combined_chunk(combined_chunk_elements);
		for (size_t offset = 0; offset < total_elements;
			offset += combined_chunk_elements)
		{
			const size_t count = std::min(combined_chunk_elements,
				total_elements - offset);
			for (size_t local_index = 0; local_index < count; ++local_index)
			{
				const size_t index = offset + local_index;
				const size_t row = (index / static_cast<size_t>(numImagebin))
					% numProjectionSingle;
				combined_chunk[local_index] = PE_SysMat[index]
					* photopeak_acceptance[row] + out[index];
			}
			if (fwrite(combined_chunk.data(), sizeof(float), count, fp2) != count)
			{
				perror(Fname3);
				fclose(fp2);
				exit(EXIT_FAILURE);
			}
		}
		fclose(fp2);

		cout << "########################" << endl;
		cout << "Full Sysmat written." << endl;
		cout << "########################" << endl;
	}
	return 0;
}

