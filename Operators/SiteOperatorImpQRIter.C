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

void SiteOperatorImp::iCanonicalFormQRIter(Direction lr)
{
    assert(itsDw.Dw1==itsDw.Dw2); //Make sure we are square
//    CheckSync();
    int X=itsDw.Dw1-2; //Chi
    MatrixRT Lp(0,X+1,0,X+1),Id(0,X+1,0,X+1);
    Unit(Lp);
    Unit(Id);

    double eta=8.111111;
    int niter=1;
    do
    {
        auto [Q,L]=itsWOvM.BlockQX(lr); //Solves V=Q*L
        X=itsDw.Dw1-2; //Chi
        assert(L.GetNumRows()==X+2);
        assert(L.GetNumCols()==X+2);
        if (Shrink(L,Q,1e-13))
        {
//            cout << "L*Q-V=" << Max(fabs(Q*L-V)) << endl;
//            assert(Max(fabs(Q*L-V))<1e-13);
        }
        eta=8.111;
        if (L.GetNumRows()==L.GetNumCols())
        {
            Id.SetLimits(L.GetLimits(),true);
            eta=Max(fabs(L-Id));
        }
        cout << "eta=" << eta << endl;
        itsWOvM.SetV(lr,Q);
        // Get out here so we leave the Ws left normalized.
        if (niter++>100) break;
        //
        //  Do W->L*W
        //
        itsWOvM=MatrixOR(L*itsWOvM); //ul gets lost in mul op.
        itsWOvM.SetUpperLower(Lower);
        assert(itsWOvM.GetUpperLower()==Lower);
        itsDw.Dw1=L.GetNumRows();
        Lp=MatrixRT(L*Lp);

    } while (eta>1e-13);
//    cout << std::fixed << std::setprecision(2) << "Lp=" << Lp << endl;
//    cout << std::fixed << std::setprecision(2) << "LpT*Lp=" << Transpose(Lp)*Lp << endl;

    SyncOtoW();
}



} //namespace
