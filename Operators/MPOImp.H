#ifndef MPOImp_H
#define MPOImp_H

#include "TensorNetworks/MPO.H"
#include "Misc/ptr_vector.h"

namespace TensorNetworks
{

//
//  General MPO with possibly independent operators on each site
//
class MPOImp : public virtual MPO
{
    public:
        MPOImp(int L, double S);
        virtual ~MPOImp();

        virtual int   GetL() const {return itsL;}
        virtual const SiteOperator* GetSiteOperator(int isite) const
        {
            assert(isite>0);
            assert(isite<=itsL);
            return itsSites[isite];
        }

        virtual int    GetMaxDw() const;
        virtual void   Combine(const Operator* O2);
        virtual double Compress(int Dmax, double minSV);
        virtual void   Report(std::ostream&) const;

    protected:

    private:
        typedef optr_vector<SiteOperator*> SitesType;

        int itsL;
        int itsd;

        SitesType itsSites;

};

}
#endif // MPO_H
