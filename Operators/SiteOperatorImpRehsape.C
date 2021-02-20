#include "Operators/SiteOperatorImp.H"

namespace TensorNetworks
{

//
//  For right canonization we need preserve the last row of W.  These are the d, b blocks.
//
void SiteOperatorImp::NewBondDimensions(int D1, int D2, bool saveData)
{
    assert(D1>0);
    assert(D2>0);
    if (itsDw.Dw1==D1 && itsDw.Dw2==D2) return;
//    std::cout << "Reshape from " << itsDw.Dw1 << "," << itsDw.Dw2 << "   to " << D1 << "," << D2 << std::endl;

    itsWs.SetChi12(D1-2,D2-2,saveData);
    itsDw.Dw1=D1;
    itsDw.Dw2=D2;
    SetLimits();
}




} //namespace
