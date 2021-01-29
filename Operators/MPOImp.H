#ifndef MPOImp_H
#define MPOImp_H

#include "TensorNetworks/MPO.H"
//#include "Operators/MPO_Common.H"
#include "Containers/ptr_vector.h"

namespace TensorNetworks
{

//
//  All this class does is
//    1) store a string MPOs, each with left and right neighbour links.
//    2) serve up the site operators by site number
//  All operator are initialized as unit operators with Dw=1.
//  Non trivial MPOs are then created using the Combine function in MPO.H
//
class MPOImp
    : public virtual MPO
//    , private MPO_Common
{
protected:
    typedef Matrix<MatrixRT> TensorT;

    public:
        enum LoadWith {Identity,LoadLater};
        MPOImp(int L, double S,LoadWith);
        MPOImp(int L, double S,const TensorT& W);
        virtual ~MPOImp();

        virtual int    GetL() const {return itsL;}
        virtual double GetS() const {return itsS;}
        virtual const SiteOperator* GetSiteOperator(int isite) const;
        virtual       SiteOperator* GetSiteOperator(int isite);

    protected:
        virtual void LinkSites();
        void Insert(SiteOperator*);
        typedef optr_vector<SiteOperator*> SitesType;

        int itsL;
        double itsS;
        bool areSitesLinked;
        SitesType itsSites;

};

}
#endif // MPO_H
