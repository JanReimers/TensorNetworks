#include "Operators/MPO_OneSite.H"
#include "Operators/SiteOperatorImp.H"
#include "TensorNetworks/CheckSpin.H"

namespace TensorNetworks
{

MPO_OneSite::MPO_OneSite(int L, double S ,int isite, SpinOperator o)
    : MPOImp(L,S,MPOImp::LoadLater)
{
    assert(isValidSpin(S));
    int d=2*S+1;
    assert(L>1);
    assert(d>1);

    for (int ia=1;ia<=L;ia++)
    {
        Position lbr = ia==1 ? PLeft : (ia==L ? PRight : PBulk);
        if (ia==isite)
            Insert(new SiteOperatorImp(d,lbr,S,o));
        else
            Insert(new SiteOperatorImp(d,lbr)); //Identity op
    }
    LinkSites();
}


MPO_OneSite::~MPO_OneSite()
{
}


} //namespace

