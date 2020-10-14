#include "Operators/MPO_TwoSite.H"
#include "Operators/SiteOperatorImp.H"
#include "TensorNetworks/CheckSpin.H"

namespace TensorNetworks
{


MPO_TwoSite::MPO_TwoSite(int L, double S ,int isite1,int isite2, SpinOperator so1, SpinOperator so2)
    : MPOImp(L,S,MPOImp::LoadLater)
{
    assert(isValidSpin(S));
    int d=2*S+1;
    assert(L>1);
    assert(d>1);

    for (int ia=1;ia<=L;ia++)
    {
        if (ia==isite1)
            Insert(new SiteOperatorImp(d,S,so1));
        else if (ia==isite2)
            Insert(new SiteOperatorImp(d,S,so2));
        else
            Insert(new SiteOperatorImp(d)); //Identity op
    }

}


MPO_TwoSite::~MPO_TwoSite()
{
}

} //namespace

