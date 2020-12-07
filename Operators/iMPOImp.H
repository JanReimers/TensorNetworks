#ifndef iMPOImp_H
#define iMPOImp_H

#include "TensorNetworks/iMPO.H"
#include "Operators/MPOImp.H"

namespace TensorNetworks
{

class iMPOImp : public virtual iMPO
    , private MPOImp
{
    public:
        iMPOImp(int L, double S,MPOImp::LoadWith);
        virtual ~iMPOImp();

        virtual int   GetL() const {return MPOImp::GetL();}
        virtual SiteOperator* GetSiteOperator(int isite)
        {
            return MPOImp::GetSiteOperator(isite);
        }
//        virtual double Compress(const SVCompressorR*);
};

}
#endif // iMPO_H
