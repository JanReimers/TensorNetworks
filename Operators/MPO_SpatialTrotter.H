#ifndef MPO_SPATIALTROTTER_H
#define MPO_SPATIALTROTTER_H

#include "TensorNetworks/Namespace.H"
#include "Operators/MPOImp.H"

namespace TensorNetworks
{

//
//  MPO for spatial trotter decomposition for odd and even sites.
//  This MPO with odd/even decomposotion can only work for NN interactions that natrually
//  decompose into two non-interacting sublattices.
//  For Hamiltonians with further neighour interactions or NN interactions with non-trivial
//  geometries (Kagome, CST, FCC, Tri) we will need a more general
//  decomposition in N non-interacting sublattices.
//
class MPO_SpatialTrotter
    : public virtual MPO
    , private MPOImp
{
    public:
        MPO_SpatialTrotter(double dt, Trotter type,int L, double S, const Matrix4RT& Hlocal);
        MPO_SpatialTrotter(double dt, Trotter type,int L, int d   , const Matrix4RT& Hlocal) = delete;
        virtual ~MPO_SpatialTrotter();
};

}
#endif // MPO_SPATIALTROTTER_H
