#include "Operators/MPO_TwoSite.H"
#include "Operators/SiteOperatorImp.H"
#include "Operators/OneSiteSpinOperator.H"
#include "Operators/IdentityOperator.H"

namespace TensorNetworks
{


MPO_TwoSite::MPO_TwoSite(int L, double S ,int isite1,int isite2, SpinOperator so1, SpinOperator so2)
    : itsL(L)
    , itsd(2*S+1)
    , itsSite1Index(isite1)
    , itsSite2Index(isite2)
    , itsSpin1WOp(new OneSiteSpinOperator(S,so1))
    , itsSpin2WOp(new OneSiteSpinOperator(S,so2))
    , itsIdentityWOp(new IdentityOperator())
    , itsSite1Operator(new SiteOperatorImp(PBulk,itsSpin1WOp,itsd))
    , itsSite2Operator(new SiteOperatorImp(PBulk,itsSpin2WOp,itsd))
    , itsIndentityOperator(new SiteOperatorImp(PBulk,itsIdentityWOp,itsd))
{

}


MPO_TwoSite::~MPO_TwoSite()
{
   delete itsIndentityOperator;
   delete itsSite2Operator;
   delete itsSite1Operator;
   delete itsIdentityWOp;
   delete itsSpin2WOp;
   delete itsSpin1WOp;
}

} //namespace

