#include "Operators/MPO_LRB.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/CheckSpin.H"

namespace TensorNetworks
{


MPO_LRB::MPO_LRB(int L, double S)
    : itsL(L)
    , itsd(2*S+1)
{
    assert(isValidSpin(S));
}

MPO_LRB::~MPO_LRB()
{
    //dtor
}


Position MPO_LRB::GetPosition(int isite) const
{
    return isite==1 ? PLeft :
           (isite==itsL ? PRight : PBulk);
}

}
