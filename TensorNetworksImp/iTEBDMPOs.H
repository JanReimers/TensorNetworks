#ifndef ITEBDMPOS_H
#define ITEBDMPOS_H

#include "TensorNetworksImp/iTEBDStateImp.H"

namespace TensorNetworks
{

class iTEBDMPOs
    : public virtual iTEBDState
    , private iTEBDStateImp
{
    using MultiMPOType=iTEBDStateImp::MultiMPOType;
    public:
        iTEBDMPOs(int L,double S, int D,double normEps,double epsSV);
        virtual ~iTEBDMPOs();

        virtual void   InitGates (const Hamiltonian*,double dt,TrotterOrder);
        virtual void   Apply     (SVCompressorC*, int center);

    private:
        MultiMPOType itsGates;
};

} //namespace
#endif // ITEBDMPOS_H
