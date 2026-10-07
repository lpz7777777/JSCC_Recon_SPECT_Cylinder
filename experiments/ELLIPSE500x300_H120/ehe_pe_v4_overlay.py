"""Extend frozen PE-v4 surface quadrature with exact finite circular-hole chords.

The JSCC engine stays untouched. Spatial buckets enumerate every hole whose
circle intersects the ray's conservative collimator-plane bounding rectangle.
No fixed nearest-hole shortlist, random sampling or aperture approximation.
"""
from pathlib import Path
from ehe_common import digest,write

STRUCT='''struct CollimatorLayerGpu
{
    float half_x, center_y, half_y, half_z, mu_total;
    int hole_count, count_x, count_z;
    float origin_x, origin_z, cell_size, max_radius;
    float holes[1250][5];
    int offsets[4097], hole_ids[1250];
};'''

BUILD='''        output.hole_count=hole_count;
        if(hole_count<0 || hole_count>1250)throw std::runtime_error("EHE hole capacity exceeded");
        output.cell_size=6.0f;output.origin_x=-output.half_x;output.origin_z=-output.half_z;
        output.count_x=static_cast<int>(std::ceil(2*output.half_x/output.cell_size))+1;
        output.count_z=static_cast<int>(std::ceil(2*output.half_z/output.cell_size))+1;
        const int cells=output.count_x*output.count_z;
        if(cells>4096)throw std::runtime_error("EHE spatial grid capacity exceeded");
        std::vector<std::vector<int>> buckets(cells);
        int previous_holes=0;
        for(int previous=0;previous<layer;++previous)previous_holes+=static_cast<int>(values[(previous+1)*10]);
        if(values.size()<100+static_cast<std::size_t>(previous_holes+hole_count)*9)
            throw std::runtime_error("Incomplete circular-hole Params");
        for(int h=0;h<hole_count;++h){
            const int hb=100+(previous_holes+h)*9;
            output.holes[h][0]=values[hb];output.holes[h][1]=values[hb+1]+image[11];
            output.holes[h][2]=values[hb+2]+image[11];output.holes[h][3]=values[hb+3];output.holes[h][4]=values[hb+4];
            if(!(output.holes[h][4]>0) || !(output.holes[h][2]>output.holes[h][1]))
                throw std::runtime_error("Invalid finite circular hole");
            output.max_radius=std::max(output.max_radius,output.holes[h][4]);
            const int gx=static_cast<int>(std::floor((values[hb]-output.origin_x)/output.cell_size));
            const int gz=static_cast<int>(std::floor((values[hb+3]-output.origin_z)/output.cell_size));
            if(gx<0 || gx>=output.count_x || gz<0 || gz>=output.count_z)throw std::runtime_error("Hole outside spatial grid");
            buckets[gx*output.count_z+gz].push_back(h);
        }
        int offset=0;
        for(int c=0;c<cells;++c){output.offsets[c]=offset;for(int h:buckets[c])output.hole_ids[offset++]=h;}
        output.offsets[cells]=offset;
        if(offset!=hole_count)throw std::runtime_error("Circular-hole grid omitted entries");
'''

CHORD='''__device__ float eheCylinderFraction(const float* start,const float* end,const float* h,double lo,double hi)
{
    const double dy=static_cast<double>(end[1])-start[1];
    if(fabs(dy)<1e-12){if(start[1]<h[1] || start[1]>h[2])return 0;}
    else{double a=(static_cast<double>(h[1])-start[1])/dy,b=(static_cast<double>(h[2])-start[1])/dy;if(a>b){double t=a;a=b;b=t;}lo=lo>a?lo:a;hi=hi<b?hi:b;}
    const double dx=static_cast<double>(end[0])-start[0],dz=static_cast<double>(end[2])-start[2];
    const double sx=static_cast<double>(start[0])-h[0],sz=static_cast<double>(start[2])-h[3];
    const double aa=dx*dx+dz*dz,bb=2*(sx*dx+sz*dz),cc=sx*sx+sz*sz-static_cast<double>(h[4])*h[4];
    if(aa<1e-20){if(cc>0)return 0;}
    else{const double disc=bb*bb-4*aa*cc;if(disc<=0)return 0;
         const double root=sqrt(disc),a=(-bb-root)/(2*aa),b=(-bb+root)/(2*aa);lo=lo>a?lo:a;hi=hi<b?hi:b;}
    return hi>lo?static_cast<float>(hi-lo):0.0f;
}

__device__ float eheMaterialChord(const float* start,const float* end,const CollimatorLayerGpu& col)
{
    const float local_start[3]={start[0],start[1]-col.center_y,start[2]};
    const float local_end[3]={end[0],end[1]-col.center_y,end[2]};
    const float ext[3]={col.half_x,col.half_y,col.half_z};float lo=0,hi=0;
    if(!segmentBoxInterval(local_start,local_end,ext,&lo,&hi))return 0;
    const float dx=end[0]-start[0],dy=end[1]-start[1],dz=end[2]-start[2];
    const float length=sqrtf(dx*dx+dy*dy+dz*dz);
    if(col.hole_count==0)return (hi-lo)*length;
    const float x0=start[0]+lo*dx,x1=start[0]+hi*dx,z0=start[2]+lo*dz,z1=start[2]+hi*dz;
    int gx0=static_cast<int>(floorf((fminf(x0,x1)-col.max_radius-col.origin_x)/col.cell_size));
    int gx1=static_cast<int>(floorf((fmaxf(x0,x1)+col.max_radius-col.origin_x)/col.cell_size));
    int gz0=static_cast<int>(floorf((fminf(z0,z1)-col.max_radius-col.origin_z)/col.cell_size));
    int gz1=static_cast<int>(floorf((fmaxf(z0,z1)+col.max_radius-col.origin_z)/col.cell_size));
    gx0=max(0,gx0);gz0=max(0,gz0);gx1=min(col.count_x-1,gx1);gz1=min(col.count_z-1,gz1);
    float removed=0;
    for(int gx=gx0;gx<=gx1;++gx)for(int gz=gz0;gz<=gz1;++gz){
        int cell=gx*col.count_z+gz;
        for(int i=col.offsets[cell];i<col.offsets[cell+1];++i)
            removed+=eheCylinderFraction(start,end,col.holes[col.hole_ids[i]],lo,hi);
    }
    return fmaxf(0,(hi-lo)-removed)*length;
}

'''

def overlay(source,target):
    s=Path(source).read_text();original=s
    old='''struct CollimatorLayerGpu
{
    float half_x;
    float center_y;
    float half_y;
    float half_z;
    float mu_total;
};'''
    replacements=[(old,STRUCT),('''        if (hole_count != 0)
            throw std::runtime_error(
                "PE v4 production currently supports zero-hole collimators only");''',''),
        ('        layers.push_back(output);',BUILD+'        layers.push_back(output);'),
        ('__device__ float targetExitDistance(',CHORD+'__device__ float targetExitDistance('),
        ('const CollimatorLayerGpu collimator = collimators[layer];','const CollimatorLayerGpu& collimator = collimators[layer];'),
        ('''attenuation += collimator.mu_total * axisAlignedSegmentChord(
                    source, entry_world, collimator.center_y,
                    collimator.half_x, collimator.half_y, collimator.half_z);''',
         'attenuation += collimator.mu_total * eheMaterialChord(source, entry_world, collimator);')]
    for old,new in replacements:
        if s.count(old)!=1:raise ValueError('Frozen PE-v4 overlay anchor changed')
        s=s.replace(old,new)
    Path(target).write_text(s,encoding='utf-8',newline='\n')
    write(Path(target).with_suffix('.overlay.json'),dict(base_sha256=digest(source),derived_sha256=digest(target),
        changes='finite circular-hole chords only; unchanged PE-v4 quadrature, detector attenuation, windowing',
        buckets='complete conservative rectangle, no hole truncation',capacity_holes=1250))
