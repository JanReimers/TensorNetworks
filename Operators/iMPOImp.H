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
        iMPOImp(int L, double S,const TensorT& W);
        iMPOImp(int L, double S,const OperatorClient* W);

        virtual ~iMPOImp();

        virtual iMPO* MakeUnitcelliMPO(int unitcell) const; //Contract horizontally to make iMPO for the whole unit cell.

        virtual int   GetL() const {return MPOImp::GetL();}
        virtual const SiteOperator* GetSiteOperator(int isite) const
        {
            return MPO::GetSiteOperator(isite);
        }
        virtual SiteOperator* GetSiteOperator(int isite)
        {
            return MPOImp::GetSiteOperator(isite);
        }

        virtual double Compress(const SVCompressorR*);
    private:
         virtual void LinkSites();

};

}
#endif // iMPO_H
