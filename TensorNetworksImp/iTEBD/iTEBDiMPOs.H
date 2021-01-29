#ifndef ITEBDiMPOS_H
#define ITEBDiMPOS_H

#include "TensorNetworksImp/iTEBD/iTEBDStateImp.H"

namespace TensorNetworks
{

class iTEBDiMPOs
    : public virtual iTEBDState
    , private iTEBDStateImp
{
    using Multi_iMPOType=iTEBDStateImp::Multi_iMPOType;
    public:
        iTEBDiMPOs(int L,double S, int D,double normEps,double epsSV);
        virtual ~iTEBDiMPOs();

        virtual void   InitGates (const iHamiltonian*,double dt,TrotterOrder,CompressType,double eps);
        virtual void   Apply     (SVCompressorC*, int center);

    private:
        Multi_iMPOType itsGates;
};

} //namespace
#endif // ITEBDiMPOS_H
