#include "Operators/MPO_OneSite.H"
#include "Operators/SiteOperatorImp.H"
#include "Operators/OneSiteSpinOperator.H"
#include "Operators/IdentityOperator.H"

MPO_OneSite::MPO_OneSite(int L, double S ,int isite, TensorNetworks::SpinOperator o)
    : itsL(L)
    , itsp(2*S+1)
    , itsSiteIndex(isite)
    , itsSpinWOp(new OneSiteSpinOperator(S,o))
    , itsIdentityWOp(new IdentityOperator())
    , itsSiteOperator(new SiteOperatorImp(TensorNetworks::Bulk,itsSpinWOp,itsp))
    , itsIndentityOperator(new SiteOperatorImp(TensorNetworks::Bulk,itsIdentityWOp,itsp))
{

}


MPO_OneSite::~MPO_OneSite()
{
   delete itsIndentityOperator;
   delete itsSiteOperator;
   delete itsIdentityWOp;
   delete itsSpinWOp;
}


