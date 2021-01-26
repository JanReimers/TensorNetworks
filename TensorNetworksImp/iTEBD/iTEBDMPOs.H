#ifndef ITEBDMPOS_H
#define ITEBDMPOS_H

#include "TensorNetworksImp/iTEBD/iTEBDStateImp.H"

namespace TensorNetworks
{

class iTEBDMPOs
    : public virtual iTEBDState
    , private iTEBDStateImp
{
    using Multi_iMPOType=iTEBDStateImp::Multi_iMPOType;
    public:
        iTEBDMPOs(int L,double S, int D,double normEps,double epsSV);
        virtual ~iTEBDMPOs();

        virtual void   InitGates (const Hamiltonian*,double dt,TrotterOrder, double eps);
        virtual void   Apply     (SVCompressorC*, int center);

    private:
        Multi_iMPOType itsGates;
};

} //namespace
#endif // ITEBDMPOS_H
