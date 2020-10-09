#include "Operators/MPO_OneSite.H"
#include "Operators/SiteOperatorImp.H"
#include "Operators/OneSiteSpinOperator.H"
#include "Operators/IdentityOperator.H"

namespace TensorNetworks
{

MPO_OneSite::MPO_OneSite(int L, double S ,int isite, SpinOperator o)
    : itsL(L)
    , itsd(2*S+1)
    , itsSiteIndex(isite)
    , itsSpinWOp(new OneSiteSpinOperator(S,o))
    , itsIdentityWOp(new IdentityOperator())
    , itsSiteOperator(new SiteOperatorImp(PBulk,itsSpinWOp,itsd))
    , itsIndentityOperator(new SiteOperatorImp(PBulk,itsIdentityWOp,itsd))
{

}


MPO_OneSite::~MPO_OneSite()
{
   delete itsIndentityOperator;
   delete itsSiteOperator;
   delete itsIdentityWOp;
   delete itsSpinWOp;
}

} //namespace

