#ifndef MPO_OneSIte_H
#define MPO_OneSIte_H

#include "TensorNetworks/Operator.H"
#include "TensorNetworks/Enums.H"
#include "Misc/ptr_vector.h"

class SiteOperator;
class OperatorWRepresentation;

//
//  Represent an MPO for one site only. The rest of the sites are just identity operators.
//
class MPO_OneSite : public virtual Operator
{
    public:
        MPO_OneSite(int L, double S ,int isite, TensorNetworks::SpinOperator);
        virtual ~MPO_OneSite();

        const SiteOperator* GetSiteOperator(int isite) const
        {
            assert(isite>=0);
            assert(isite<itsL);
            return isite==itsSiteIndex ? itsSiteOperator : itsIndentityOperator;
        }

    private:

        int itsL;   //Number of lattice sites
        int itsp;   //Number ot states for the spin on each site = 2S+1
        int itsSiteIndex;

        OperatorWRepresentation* itsSpinWOp;
        OperatorWRepresentation* itsIdentityWOp;
        SiteOperator* itsSiteOperator;
        SiteOperator* itsIndentityOperator;
};

#endif // MPO_OneSIte_H