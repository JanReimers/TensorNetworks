#include "Operators/SiteOperatorImp.H"
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

//
//  Hunt for the zero pivots in L and remove those rows from L and columns from Q
//
bool Shrink(MatrixRT& L, MatrixOR& Q,double eps)
{
    VectorRT diag=L.GetDiagonal();
    std::vector<index_t> remove;
    for (index_t i:diag.indices())
    {
        if (fabs(diag(i))<eps) //fast feasibility study
        {
            double s=Sum(fabs(L.GetRow(i))); //Now check the whole row
            if (s<eps) remove.push_back(i);
        }
    }
    for (auto i=remove.rbegin();i!=remove.rend();i++)
    {
        assert(Sum(fabs(L.GetRow(*i)))<eps); //Make sure we got the correct row
        L.RemoveRow   (*i);
        Q.RemoveColumn(*i);
    }
    return remove.size()>0;
}

MatrixRT SiteOperatorImp::iCanonicalFormQRIter(Direction lr)
{
    assert(itsOpRange.Dw1==itsOpRange.Dw2); //Make sure we are square
    int X=itsOpRange.Dw1-2; //Chi
    MatrixRT G(0,X+1,0,X+1),Id(0,X+1,0,X+1);
    Unit(G);
    Unit(Id);

    double eta=1.0;
    int niter=1;
    do
    {
        MatrixRT L=itsWs.QXRR(lr,1e-13); //Solves V=Q*L, Q is stored in W
        if (L.GetNumRows()==L.GetNumCols())
        {
            Id.SetLimits(L.GetLimits(),true);
            eta=Max(fabs(L-Id));
            cout << " L=" << L.GetLimits() << "eta=" << eta << endl;
        }
        else
        {
            cout << " L=" << L.GetLimits() << endl;

        }
        if (niter++>100) break; // Get out here so we leave the Ws left normalized.
        //
        //  Do W->L*W
        //
        switch(lr)
        {
        case DLeft:
            itsWs=L*itsWs;
            G=L*G; //Update gauge transform
            break;
        case DRight:
            itsWs=itsWs*L;
            G=G*L; //Update gauge transform
            break;
        }

    } while (eta>1e-13);
//    cout << std::fixed << std::setprecision(2) << "Lp=" << Lp << endl;
//    cout << std::fixed << std::setprecision(2) << "LpT*Lp=" << Transpose(Lp)*Lp << endl;

    SetLimits();
    return G;
}



} //namespace
