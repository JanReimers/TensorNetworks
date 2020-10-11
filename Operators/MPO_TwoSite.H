#ifndef MPO_TwoSite_H
#define MPO_TwoSite_H

#include "TensorNetworks/Namespace.H"
#include "TensorNetworks/Operator.H"
#include "Containers/ptr_vector.h"


namespace TensorNetworks
{
//
//  Represent an MPO for one site only. The rest of the sites are just identity operators.
//
class MPO_TwoSite : public virtual Operator
{
    public:
        MPO_TwoSite(int L, double S ,int isite1,int isite2, SpinOperator so1,SpinOperator so2);
        virtual ~MPO_TwoSite();

        virtual int   GetL() const {return itsL;}
        SiteOperator* GetSiteOperator(int isite)
        {
            assert(isite>=1);
            assert(isite<=itsL);
            return isite==itsSite1Index ? itsSite1Operator :
                (isite==itsSite2Index ? itsSite2Operator : itsIndentityOperator);
        }

    private:

        int itsL;   //Number of lattice sites
        int itsd;   //Number ot states for the spin on each site = 2S+1
        int itsSite1Index;
        int itsSite2Index;

        OperatorWRepresentation* itsSpin1WOp;
        OperatorWRepresentation* itsSpin2WOp;
        OperatorWRepresentation* itsIdentityWOp;
        SiteOperator* itsSite1Operator;
        SiteOperator* itsSite2Operator;
        SiteOperator* itsIndentityOperator;
};

}

#endif // MPO_TwoSite_H
