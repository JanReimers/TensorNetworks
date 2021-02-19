#include "Operators/SiteOperatorImp.H"

namespace TensorNetworks
{


//
//
//  For right canonization we need preserve the last row of W.  These are the d, b blocks.
//
void SiteOperatorImp::NewBondDimensions(int D1, int D2, bool saveData)
{
    assert(D1>0);
    assert(D2>0);
    if (itsDw.Dw1==D1 && itsDw.Dw2==D2) return;
//    std::cout << "Reshape from " << itsDw.Dw1 << "," << itsDw.Dw2 << "   to " << D1 << "," << D2 << std::endl;


    for (int m=0; m<itsd; m++)
        for (int n=0; n<itsd; n++)
        {
            VectorRT lastRow=itsWs(m+1,n+1).GetRow(itsDw.Dw1);
            itsWs(m+1,n+1).SetLimits(D1,D2,saveData);
            itsWs(m+1,n+1).GetRow(D1)=lastRow.SubVector(D2);

        }
    itsDw.Dw1=D1;
    itsDw.Dw2=D2;
    SyncWtoO();
}




} //namespace
