#ifndef MPO_TwoSite_H
#define MPO_TwoSite_H

#include "Operators/MPOImp.H"


namespace TensorNetworks
{
//
//  Represent an MPO for one site only. The rest of the sites are just identity operators.
//
class MPO_TwoSite
    : public virtual MPO
    , private MPOImp
{
    public:
        MPO_TwoSite(int L, double S ,int isite1,int isite2, SpinOperator so1,SpinOperator so2);
        virtual ~MPO_TwoSite();
};

}

#endif // MPO_TwoSite_H
