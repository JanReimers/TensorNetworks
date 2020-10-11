#ifndef MPOImp_H
#define MPOImp_H

#include "TensorNetworks/MPO.H"
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
class MPOImp : public virtual MPO
{
    public:
        MPOImp(int L, double S);
        virtual ~MPOImp();

        virtual int   GetL() const {return itsL;}
        virtual SiteOperator* GetSiteOperator(int isite)
        {
            assert(isite>0);
            assert(isite<=itsL);
            return itsSites[isite];
        }

    protected:

    private:
        typedef optr_vector<SiteOperator*> SitesType;

        int itsL;
        SitesType itsSites;

};

}
#endif // MPO_H
