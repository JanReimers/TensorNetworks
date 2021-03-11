#include "Operators/SiteOperatorImp.H"
#include "Operators/OperatorBond.H"
#include "TensorNetworks/SVCompressor.H"
#include "TensorNetworks/TNSLogger.H"
#include "NumericalMethods/LapackSVDSolver.H"
#include "NumericalMethods/LapackQRSolver.H"
#include "NumericalMethods/LapackLinearSolver.H"
#include "oml/diagonalmatrix.h"

namespace TensorNetworks
{

using std::cout;
using std::endl;

void SiteOperatorImp::InitQRIter()
{
    assert(itsOpRange.Dw1==itsOpRange.Dw2); //Make sure we are square
    int X=itsOpRange.Dw1-2; //Chi
    itsG.SetLimits(0,X+1,0,X+1);
    Unit(itsG);
}

double SiteOperatorImp::QRStep(Direction lr,double eps)
{
    double eta=99.0;
    MatrixRT L=itsWs.QXRR(lr,eps); //Solves V=Q*L, Q is stored in W
    if (L.GetNumRows()==L.GetNumCols())
    {
        MatrixRT Id(L.GetLimits());
        Unit(Id);
        eta=Max(fabs(L-Id));
//            cout << " L=" << L.GetLimits() << "eta=" << eta << endl;
    }
    else
    {
//            cout << " L=" << L.GetLimits() << endl;

    }

    GetBond(lr)->GaugeTransfer(lr,L); //  Do W->L*W
    switch(lr)
    {
    case DLeft:
        itsG=L*itsG; //Update gauge transform
        break;
    case DRight:
        itsG=itsG*L; //Update gauge transform
        break;
    }
    return eta;
}


} //namespace
