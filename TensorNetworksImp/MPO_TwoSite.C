#include "TensorNetworksImp/MPO_TwoSite.H"
#include "TensorNetworksImp/SiteOperatorImp.H"
#include "TensorNetworksImp/OneSiteSpinOperator.H"
#include "TensorNetworksImp/IdentityOperator.H"

MPO_TwoSite::MPO_TwoSite(int L, double S ,int isite1,int isite2, TensorNetworks::SpinOperator so1, TensorNetworks::SpinOperator so2)
    : itsL(L)
    , itsp(2*S+1)
    , itsSite1Index(isite1)
    , itsSite2Index(isite2)
    , itsSpin1WOp(new OneSiteSpinOperator(S,so1))
    , itsSpin2WOp(new OneSiteSpinOperator(S,so2))
    , itsIdentityWOp(new IdentityOperator())
    , itsSite1Operator(new SiteOperatorImp(TensorNetworks::Bulk,itsSpin1WOp,itsp))
    , itsSite2Operator(new SiteOperatorImp(TensorNetworks::Bulk,itsSpin2WOp,itsp))
    , itsIndentityOperator(new SiteOperatorImp(TensorNetworks::Bulk,itsIdentityWOp,itsp))
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


