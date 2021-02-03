#ifndef iMPO_SPATIALTROTTER_H
#define iMPO_SPATIALTROTTER_H

#include "TensorNetworks/Namespace.H"
#include "Operators/iMPOImp.H"

namespace TensorNetworks
{

//
//  iMPO for spatial trotter decomposition for odd and even sites.
//  This iMPO with odd/even decomposition can only work for NN interactions that naturally
//  decompose into two non-interacting sublattices.
//  For Hamiltonians with further neighour interactions or NN interactions with non-trivial
//  geometries (Kagome, CST, FCC, Tri) we will need a more general
//  decomposition in N non-interacting sublattices.
//
class iMPO_SpatialTrotter
    : public virtual iMPO
    , private iMPOImp
{
    public:
        iMPO_SpatialTrotter(double dt, Trotter type, const iHamiltonian*);
        virtual ~iMPO_SpatialTrotter();
};

}
#endif // MPO_SPATIALTROTTER_H
