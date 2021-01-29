#ifndef iMPOImp_H
#define iMPOImp_H

#include "TensorNetworks/iMPO.H"
//#include "Operators/MPO_Common.H"
#include "Containers/ptr_vector.h"

namespace TensorNetworks
{

class iMPOImp
    : public virtual iMPO
//    , private MPO_Common
{
        typedef Matrix<MatrixRT> TensorT;
   public:
        enum LoadWith {Identity,LoadLater};
        iMPOImp(int L, double S,iMPOImp::LoadWith);
        iMPOImp(int L, double S,const TensorT& W);
        iMPOImp(int L, double S,const OperatorClient* W);

        virtual ~iMPOImp();

        virtual int    GetL() const {return itsL;}
        virtual int    Getd() const {return 2*itsS+1;}
        virtual double GetS() const {return itsS;}
        virtual const  SiteOperator* GetSiteOperator(int isite) const;
        virtual        SiteOperator* GetSiteOperator(int isite);

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
