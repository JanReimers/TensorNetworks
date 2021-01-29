#ifndef ITEBDGATES_H
#define ITEBDGATES_H

#include "TensorNetworksImp/iTEBD/iTEBDStateImp.H"

namespace TensorNetworks
{

class iTEBDGates
    : public virtual iTEBDState
    , private iTEBDStateImp
{
    using MultigateType=iTEBDStateImp::MultigateType;
    public:
        iTEBDGates(int L,double S, int D,double normEps,double epsSV);
        virtual ~iTEBDGates();

        virtual void   InitGates (const iHamiltonian*,double dt,TrotterOrder,CompressType,double eps);
        virtual void   Apply     (SVCompressorC*, int center);

    private:
        MultigateType itsGates;
};

} //namespace
#endif // ITEBDGATES_H
