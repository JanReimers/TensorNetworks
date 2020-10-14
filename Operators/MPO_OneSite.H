#ifndef MPO_OneSIte_H
#define MPO_OneSIte_H

#include "Operators/MPOImp.H"


namespace TensorNetworks
{
//
//  Represent an MPO for one site only. The rest of the sites are just identity operators.
//
class MPO_OneSite
    : public virtual MPO
    , private MPOImp
{
    public:
        MPO_OneSite(int L, double S ,int isite, SpinOperator);
        virtual ~MPO_OneSite();
};

}
#endif // MPO_OneSIte_H
