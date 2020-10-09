#ifndef MPO_SPATIALTROTTER_H
#define MPO_SPATIALTROTTER_H

#include "TensorNetworks/Enums.H"
#include "TensorNetworks/Typedefs.H"
#include "TensorNetworks/Operator.H"
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
class MPO_SpatialTrotter : public virtual Operator
{
    public:
        MPO_SpatialTrotter(double dt, Trotter type,int L, int d, const Matrix4RT& Hlocal);
        virtual ~MPO_SpatialTrotter();

        virtual int   GetL() const {return itsL;}
        virtual const SiteOperator* GetSiteOperator(int isite) const;

    protected:

    private:

        Trotter itsOddEven;
        int itsL;   //Number of lattice sites
        int itsd;   //Number ot states for the spin on each site = 2S+1
        SiteOperator* itsLeft_Site; //Left  site of pair
        SiteOperator* itsRightSite; //Right site of pair
        SiteOperator* itsUnit_Site; //Site with unit operator for chain ends.

};

}
#endif // MPO_SPATIALTROTTER_H
