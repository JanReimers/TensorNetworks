#ifndef MPO_OneSIte_H
#define MPO_OneSIte_H

#include "TensorNetworks/Operator.H"
#include "TensorNetworks/Enums.H"
#include "Misc/ptr_vector.h"


namespace TensorNetworks
{

class OperatorWRepresentation;
class SiteOperator;

//
//  Represent an MPO for one site only. The rest of the sites are just identity operators.
//
class MPO_OneSite : public virtual Operator
{
    public:
        MPO_OneSite(int L, double S ,int isite, SpinOperator);
        virtual ~MPO_OneSite();

        virtual int   GetL() const {return itsL;}
        const SiteOperator* GetSiteOperator(int isite) const
        {
            assert(isite>=1);
            assert(isite<=itsL);
            return isite==itsSiteIndex ? itsSiteOperator : itsIndentityOperator;
        }

    private:

        int itsL;   //Number of lattice sites
        int itsd;   //Number ot states for the spin on each site = 2S+1
        int itsSiteIndex;

        OperatorWRepresentation* itsSpinWOp;
        OperatorWRepresentation* itsIdentityWOp;
        SiteOperator* itsSiteOperator;
        SiteOperator* itsIndentityOperator;
};

}
#endif // MPO_OneSIte_H
