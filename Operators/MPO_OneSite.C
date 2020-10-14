#include "Operators/MPO_OneSite.H"
#include "Operators/SiteOperatorImp.H"

namespace TensorNetworks
{

MPO_OneSite::MPO_OneSite(int L, double S ,int isite, SpinOperator o)
    : itsL(L)
    , itsd(2*S+1)
    , itsSiteIndex(isite)
    , itsSiteOperator(new SiteOperatorImp(itsd,S,o))
    , itsIndentityOperator(new SiteOperatorImp(itsd))
{

}


MPO_OneSite::~MPO_OneSite()
{
   delete itsIndentityOperator;
   delete itsSiteOperator;
}

} //namespace

