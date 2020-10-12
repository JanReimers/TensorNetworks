#include "Operators/MPO_LRB.H"
#include "Operators/SiteOperatorImp.H"
#include "TensorNetworks/CheckSpin.H"

namespace TensorNetworks
{


MPO_LRB::MPO_LRB(int L, double S)
    : itsL(L)
    , itsd(2*S+1)
{
    assert(isValidSpin(S));
}

MPO_LRB::MPO_LRB(const OperatorWRepresentation* O,int L, double S)
    : itsL(L)
    , itsd(2*S+1)
{
    assert(isValidSpin(S));
    assert(S>=0.5);
    Init(O);
}

MPO_LRB::~MPO_LRB()
{
    //dtor
}

void MPO_LRB::Init(const OperatorWRepresentation* O)
{
    assert(O);
    //
    //  Load W matrices for the left edge,bulk and right edge
    //
    SiteOperator* left =new SiteOperatorImp(PLeft,O,itsd);
    SiteOperator* bulk =new SiteOperatorImp(PBulk,O,itsd);
    SiteOperator* right=new SiteOperatorImp(PRight,O,itsd);
    itsSites.push_back(left );     //Left edge
    itsSites.push_back(bulk );     //bulk
    itsSites.push_back(right);     //Right edge
}

Position MPO_LRB::GetPosition(int isite) const
{
    return isite==1 ? PLeft :
           (isite==itsL ? PRight : PBulk);
}

}
