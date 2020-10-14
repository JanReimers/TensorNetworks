#include "Operators/MPO_TwoSite.H"
#include "Operators/SiteOperatorImp.H"

namespace TensorNetworks
{


MPO_TwoSite::MPO_TwoSite(int L, double S ,int isite1,int isite2, SpinOperator so1, SpinOperator so2)
    : itsL(L)
    , itsd(2*S+1)
    , itsSite1Index(isite1)
    , itsSite2Index(isite2)
    , itsSite1Operator(new SiteOperatorImp(itsd,S,so1))
    , itsSite2Operator(new SiteOperatorImp(itsd,S,so2))
    , itsIndentityOperator(new SiteOperatorImp(itsd))
{

}


MPO_TwoSite::~MPO_TwoSite()
{
   delete itsIndentityOperator;
   delete itsSite2Operator;
   delete itsSite1Operator;
}

} //namespace

