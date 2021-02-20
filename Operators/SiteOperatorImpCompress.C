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

double SiteOperatorImp::Compress(CompressType ct,Direction lr,const SVCompressorR* comp)
{
    double terror=0.0;
    switch (ct)
    {
    case Std:
        terror=CompressStd(lr,comp);
        break;
    case Parker:
        terror=CompressParker(lr,comp);
        break;
    case CNone:
        break;
    default:
        assert(false);
    }
    SetLimits();
    return terror;
}


double SiteOperatorImp::CompressStd(Direction lr,const SVCompressorR* comp)
{
    assert(comp);
    MatrixRT  A=itsWs.Flatten(lr);
    MatLimits lim=A.ReBase(1,1);
    LapackSVDSolver<double> solver;
    auto [U,sm,VT]=solver.SolveAll(A,1e-14); //Solves A=U * s * VT
    //
    //  Rescaling
    //
    double s_avg=Sum(sm.GetDiagonal())/sm.size();
    sm*=1.0/s_avg;

    AccumulateTruncationError(comp->Compress(U,sm,VT));
//    cout << "s_avg, sm=" << s_avg << " " << sm.GetDiagonal() << endl;ing site.
    switch (lr)
    {
        case DRight:
        {
            VT.ReBase(lim);
            MatrixRT Us=U*sm;
            Us.ReBase(lim);
            itsWs.UnFlatten(VT*s_avg);
            if (itsLeft_Neighbour) itsLeft_Neighbour->QLTransfer(lr,Us);
            break;
        }
        case DLeft:
        {
            U.ReBase(lim);
            MatrixRT sV=sm*VT;
            sV.ReBase(lim);
            itsWs.UnFlatten(U*s_avg);
            if (itsRightNeighbour) itsRightNeighbour->QLTransfer(lr,sV);
            break;
        }
    }
    SetLimits();
    return itsTruncationError;
}

double SiteOperatorImp::CompressParker(Direction lr,const SVCompressorR* comp)
{
    auto [Q,RL]=itsWs.BlockSVD(lr,comp); // Do QX=QR/RQ/QL/LQ decomposition of the V-block
    GetNeighbour(lr)->QLTransfer(lr,RL);
    SetLimits();
    return itsWs.GetTruncationError();
}


void SiteOperatorImp::CanonicalForm(Direction lr)
{
    auto [Q,RL]=itsWs.BlockQX(lr); // Do QX=QR/RQ/QL/LQ decomposition of the V-block
    itsWs.SetV(lr,Q); // Can'y move this inside BlockQX because SVD needs to modify Q before call SetV
    GetNeighbour(lr)->QLTransfer(lr,RL);
    SetLimits();
}



//
//  Do W = W*L/L*W
//
void SiteOperatorImp::QLTransfer(Direction lr,const MatrixRT& L)
{
    switch (lr)
    {
    case DRight:
    {
        int N1=L.GetNumCols(); //N1=0 on the first site.
        if (N1>0 && N1!=itsOpRange.Dw2)
        {
            if (itsWs.GetNumCols()!=L.GetNumRows())
                NewBondDimensions(itsOpRange.Dw1,N1,true);
            else
                itsOpRange.Dw2=N1; //The contraction below will automatically reshape the Ws.
        }
        assert(itsWs.GetColLimits()==L.GetRowLimits());
        itsWs=itsWs*L;
        break;
    }
    case DLeft:
    {
        int N1=L.GetNumRows(); //N1=0 on the first site.
        if (N1>0 && N1!=itsOpRange.Dw1)
        {
            if (itsWs.GetNumRows()!=L.GetNumCols())
                NewBondDimensions(N1,itsOpRange.Dw2,true);
            else
                itsOpRange.Dw1=N1; //The contraction below will automatically reshape the As.
        }
        assert(L.GetColLimits()==itsWs.GetRowLimits());
        itsWs=L*itsWs;
        break;
    }

    }
    SetLimits();
}


} //namespace
