#ifndef iMPOImp_H
#define iMPOImp_H

#include "TensorNetworks/iMPO.H"
#include "Containers/ptr_vector.h"

namespace TensorNetworks
{

class iMPOImp : public virtual iMPO
{
        typedef Matrix<MatrixRT> TensorT;
   public:
        enum LoadWith {Identity,LoadLater};
        iMPOImp(int L, double S,iMPOImp::LoadWith);
        iMPOImp(int L, double S,const TensorT& W);
        iMPOImp(int L, double S,const OperatorClient* W);

        virtual ~iMPOImp();

        virtual int   GetL() const {return itsL;}
//        virtual const SiteOperator* GetSiteOperator(int isite) const
//        {
//            return iMPO::GetSiteOperator(isite);
//        }
        virtual SiteOperator* GetSiteOperator(int isite)
        {
            assert(isite>0);
            assert(isite<=itsL);
            assert(areSitesLinked);
            return itsSites[isite];
        }

        iMPO* MakeUnitcelliMPO(int unitcell) const;
//        virtual double Compress(const SVCompressorR*);
protected:
        void Insert(SiteOperator*);
        virtual void LinkSites();

    private:
        typedef optr_vector<SiteOperator*> SitesType;

        int itsL;
        double itsS;
        bool areSitesLinked;
        SitesType itsSites;

};

}
#endif // iMPO_H
