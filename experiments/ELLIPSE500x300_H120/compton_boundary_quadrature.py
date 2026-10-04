"""Object-frame polar-cell integration with exact ellipse angular breakpoints.

Weights include dV. The supplied B field must be divided by full cell volume
before interpolation. No response row normalization or event selection occurs
here; a production builder must explicitly supply its full-circle reference.
"""
import math
import numpy as np
from scipy.spatial import Delaunay

def angular_breakpoints(cell,semi_axes=(250.,150.)):
    lo,hi,start,end=cell;a,b=semi_axes;points=[start,end]
    for radius in (lo,hi):
        if radius <= 0:continue
        value=(1/radius**2-1/b**2)/(1/a**2-1/b**2)
        if not 0 < value < 1:continue
        t=math.acos(math.sqrt(value))
        for base in (t,-t,math.pi-t,math.pi+t):
            for shift in range(-2,3):
                angle=base+shift*2*math.pi
                if start<angle<end:points.append(angle)
    return np.unique(points)

def cell_quadrature(cell,z,order=8,radial_order=4,axial_order=4,ellipse=True,semi_axes=(250.,150.)):
    lo,hi,start,end=cell;a,b=semi_axes
    breaks=angular_breakpoints(cell,semi_axes) if ellipse else np.array([start,end])
    tn,tw=np.polynomial.legendre.leggauss(order)
    rn,rw=np.polynomial.legendre.leggauss(radial_order)
    zn,zw=np.polynomial.legendre.leggauss(axial_order)
    points=[];weights=[]
    for start,end in zip(breaks[:-1],breaks[1:]):
        theta=(start+end)/2+(end-start)/2*tn
        top=np.minimum(hi**2,1/((np.cos(theta)/a)**2+(np.sin(theta)/b)**2)) if ellipse else np.full_like(theta,hi**2)
        span=np.maximum(top-lo**2,0)
        if not np.any(span>0):continue
        r=np.sqrt(lo**2+span[:,None]*(rn[None,:]+1)/2)
        shape=(order,radial_order,axial_order)
        p=np.stack((np.broadcast_to((r*np.cos(theta[:,None]))[:,:,None],shape),
                    np.broadcast_to((r*np.sin(theta[:,None]))[:,:,None],shape),
                    np.broadcast_to(z+1.5*zn[None,None,:],shape)),axis=-1).reshape(-1,3)
        w=(tw[:,None,None]*(end-start)/2*span[:,None,None]/4*
           rw[None,:,None]*zw[None,None,:]*1.5).reshape(-1)
        valid=w>0;points.append(p[valid]);weights.append(w[valid])
    return (np.concatenate(points),np.concatenate(weights)) if points else (np.empty((0,3)),np.empty(0))

def rotate_to_detector(points,view):
    if not 0<=view<20:raise ValueError('View out of range')
    angle=view*math.pi/10;c,s=math.cos(angle),math.sin(angle)
    result=points.copy();result[:,0]=points[:,0]*c+points[:,1]*s
    result[:,1]=points[:,1]*c-points[:,0]*s
    return result

class PolarResponseField:
    """Cached barycentric interpolation; transverse extrapolation is forbidden."""
    def __init__(self,xy,layer_count=40):
        self.triangulation=Delaunay(xy);self.layer_count=layer_count

    def cache(self,points,axial_rule='linear_endpoint'):
        tri=self.triangulation;simplex=tri.find_simplex(points[:,:2])
        if np.any(simplex<0):raise ValueError('Quadrature outside transverse response field')
        delta=points[:,:2]-tri.transform[simplex,2,:]
        first=np.einsum('nij,nj->ni',tri.transform[simplex,:2,:],delta)
        bary=np.column_stack((first,1-first.sum(1)))
        z=points[:,2]
        if np.any(np.abs(z)>60+1e-8):raise ValueError('Outside axial support')
        if axial_rule=='clamped_endpoint':z=np.clip(z,-58.5,58.5)
        elif axial_rule!='linear_endpoint':raise ValueError('Unknown axial interpolation')
        layer=(z+58.5)/3;lower=np.clip(np.floor(layer).astype(int),0,self.layer_count-2)
        return tri.simplices[simplex],bary,lower,layer-lower

    @staticmethod
    def evaluate(field,cache):
        vertices,bary,lower,t=cache
        left=np.sum(field[lower[:,None],vertices]*bary,axis=1)
        right=np.sum(field[(lower+1)[:,None],vertices]*bary,axis=1)
        result=left*(1-t)+right*t
        if not np.isfinite(result).all():raise ValueError('Nonfinite interpolated A')
        # Linear endpoint extension may cross zero; explicitly track this rule.
        return np.maximum(result,0)


class GuardedPolarResponseField:
    """Sampled radial/axial halo with unchanged interior interpolation stencils.

    All original polar vertices precede the additional r=258 vertices. The
    original convex-hull edges must remain edges of the extended mesh, so the
    inside/outside interpolation joins continuously. Axial extrapolation and
    clamping are both prohibited in this explicitly sampled field.
    """
    def __init__(self, xy, original_count, z_centres):
        self.original_count = original_count
        self.inner = Delaunay(xy[:original_count]); self.outer = Delaunay(xy)
        edges = {tuple(sorted(e)) for t in self.outer.simplices
                 for e in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0]))}
        if any(tuple(sorted(e)) not in edges for e in self.inner.convex_hull):
            raise ValueError('Halo triangulation breaks original hull; cannot join A continuously')
        self.z = np.asarray(z_centres,dtype=float)
        if not np.all(np.diff(self.z)>0) or self.z[0]>-60 or self.z[-1]<60:
            raise ValueError('Sampled axial field does not cover full cells')

    def cache(self, points):
        if not np.isfinite(points).all():raise ValueError('Nonfinite integration coordinates')
        points=np.asarray(points);vertices=np.empty((len(points),3),dtype=int)
        bary=np.empty((len(points),3),dtype=float)
        inside=self.inner.find_simplex(points[:,:2])>=0
        for mask,tri in ((inside,self.inner),(~inside,self.outer)):
            if not mask.any():continue
            xy=points[mask,:2];simplex=tri.find_simplex(xy)
            if np.any(simplex<0):raise ValueError('Outside sampled radial halo')
            delta=xy-tri.transform[simplex,2,:]
            first=np.einsum('nij,nj->ni',tri.transform[simplex,:2,:],delta)
            vertices[mask]=tri.simplices[simplex]
            bary[mask]=np.column_stack((first,1-first.sum(1)))
        z=points[:,2]
        if np.any(z<self.z[0]-1e-10) or np.any(z>self.z[-1]+1e-10):
            raise ValueError('Outside sampled axial halo')
        lower=np.clip(np.searchsorted(self.z,z,side='right')-1,0,len(self.z)-2)
        t=(z-self.z[lower])/(self.z[lower+1]-self.z[lower])
        if np.any(t < -1e-10) or np.any(t > 1+1e-10):raise ValueError('Axial extrapolation forbidden')
        return vertices,bary,lower,np.clip(t,0,1)

    @staticmethod
    def evaluate(field,cache):
        vertices,bary,lower,t=cache
        left=np.sum(field[lower[:,None],vertices]*bary,axis=1)
        right=np.sum(field[(lower+1)[:,None],vertices]*bary,axis=1)
        result=left*(1-t)+right*t
        if not np.isfinite(result).all() or np.any(result < -1e-20):
            raise ValueError('Invalid sampled A interpolation')
        return np.maximum(result,0)
