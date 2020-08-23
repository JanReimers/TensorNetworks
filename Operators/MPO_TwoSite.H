#ifndef MPO_TwoSite_H
#define MPO_TwoSite_H

#include "TensorNetworks/Operator.H"
#include "TensorNetworks/Enums.H"
#include "Misc/ptr_vector.h"

class SiteOperator;
class OperatorWRepresentation;

//
//  Represent an MPO for one site only. The rest of the sites are just identity operators.
//
class MPO_TwoSite : public virtual Operator
{
    public:
        MPO_TwoSite(int L, double S ,int isite1,int isite2, TensorNetworks::SpinOperator so1,TensorNetworks::SpinOperator so2);
        virtual ~MPO_TwoSite();

        virtual int   GetL() const {return itsL;}
        const SiteOperator* GetSiteOperator(int isite) const
        {
            assert(isite>=1);
            assert(isite<=itsL);
            return isite==itsSite1Index ? itsSite1Operator :
                (isite==itsSite2Index ? itsSite2Operator : itsIndentityOperator);
        }

    private:

        int itsL;   //Number of lattice sites
        int itsp;   //Number ot states for the spin on each site = 2S+1
        int itsSite1Index;
        int itsSite2Index;

        OperatorWRepresentation* itsSpin1WOp;
        OperatorWRepresentation* itsSpin2WOp;
        OperatorWRepresentation* itsIdentityWOp;
        SiteOperator* itsSite1Operator;
        SiteOperator* itsSite2Operator;
        SiteOperator* itsIndentityOperator;
};

#endif // MPO_TwoSite_H
