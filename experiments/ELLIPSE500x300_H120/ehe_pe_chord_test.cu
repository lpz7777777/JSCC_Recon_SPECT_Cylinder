#define main frozen_pe_main
#include "engine/PEGen_RayTracing_CircularHole/PEGen_V4_Production.cu"
#undef main

__global__ void chord_test_kernel(const float* pairs,const CollimatorLayerGpu* col,float* output,int n)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)output[i]=eheMaterialChord(pairs+i*6,pairs+i*6+3,col[0]);
}

__global__ void config_test_kernel(const CollimatorLayerGpu* col,float* output)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<1250)for(int k=0;k<5;++k)output[5*i+k]=col[0].holes[i][k];
    if(i==0){
        const auto& c=col[0];
        output[6250]=c.hole_count;output[6251]=c.count_x;output[6252]=c.count_z;
        output[6253]=c.max_radius;output[6254]=c.offsets[c.count_x*c.count_z];
        const float s[3]={c.holes[0][0],0,c.holes[0][3]};
        const float e[3]={c.holes[0][0],354,c.holes[0][3]};
        output[6255]=eheCylinderFraction(s,e,c.holes[0],0,1);
        int gx=static_cast<int>(floorf((s[0]-c.origin_x)/c.cell_size));
        int gz=static_cast<int>(floorf((s[2]-c.origin_z)/c.cell_size));
        int cell=gx*c.count_z+gz;
        output[6256]=c.offsets[cell+1]-c.offsets[cell];
        output[6257]=c.hole_ids[c.offsets[cell]];
    }
}

int main(int argc,char** argv)
{
    try{
        if(argc!=3)throw std::runtime_error("Expected input and output paths");
        cudaCheck(cudaSetDevice(0),"set device");
        auto image=readFloatFile("Params_Image.dat");
        auto params=readFloatFile("Params_Collimator.dat");
        auto col=buildCollimator(params,image);
        auto pairs=readFloatFile(argv[1]);
        if(col.size()!=1 || pairs.size()%6)throw std::runtime_error("EHE test shape mismatch");
        CollimatorLayerGpu* dc=nullptr;float* dp=nullptr;float* dout=nullptr;
        allocateAndCopy(&dc,col,"test collimator");allocateAndCopy(&dp,pairs,"test pairs");
        std::vector<float> output(pairs.size()/6);
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&dout),output.size()*sizeof(float)),"test output");
        chord_test_kernel<<<(output.size()+127)/128,128>>>(dp,dc,dout,output.size());
        cudaCheck(cudaGetLastError(),"test kernel");cudaCheck(cudaDeviceSynchronize(),"test sync");
        cudaCheck(cudaMemcpy(output.data(),dout,output.size()*sizeof(float),cudaMemcpyDeviceToHost),"test copy");
        std::ofstream f(argv[2],std::ios::binary);f.write(reinterpret_cast<const char*>(output.data()),output.size()*sizeof(float));
        float* config=nullptr;std::vector<float> config_host(6258);
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&config),config_host.size()*sizeof(float)),"config output");
        config_test_kernel<<<10,128>>>(dc,config);
        cudaCheck(cudaDeviceSynchronize(),"config sync");
        cudaCheck(cudaMemcpy(config_host.data(),config,config_host.size()*sizeof(float),cudaMemcpyDeviceToHost),"config copy");
        std::ofstream cf(std::string(argv[2])+".config",std::ios::binary);
        cf.write(reinterpret_cast<const char*>(config_host.data()),config_host.size()*sizeof(float));
        cudaFree(config);
        cudaFree(dc);cudaFree(dp);cudaFree(dout);
        return f?0:1;
    }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
