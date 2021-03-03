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
    auto [U,s,VT]=solver.SolveAll(A,1e-14); //Solves A=U * s * VT
    //
    //  Rescaling
    //
    double s_avg=::Sum(s.GetDiagonal())/s.size();
    s*=1.0/s_avg;

    double truncationError=comp->Compress(U,s,VT);
//    cout << "s_avg, sm=" << s_avg << " " << sm.GetDiagonal() << endl;ing site.
    MatrixRT Rtrans; //Gauge transform that is transferred to the next site.
    MatrixRT Q; //Orthogonal matrix for this site.
    U.ReBase(lim);
    s.ReBase(lim);
    VT.ReBase(lim);
    switch (lr)
    {
        case DRight:
        {
            Rtrans=U*s;
            Q=VT*s_avg;
            break;
        }
        case DLeft:
        {
            Rtrans=s*VT;
            Q=U*s_avg;
            break;
        }
    }
    itsWs.UnFlatten(Q);
    s.ReBase(1);
    GetBond(lr)->GaugeTransfer(lr,truncationError,s,Rtrans);
    SetLimits();
    return truncationError;
}

double SiteOperatorImp::CompressParker(Direction lr,const SVCompressorR* comp)
{
    auto [truncError,s,Rtrans]=itsWs.SVD(lr,comp); // Do QX=QR/RQ/QL/LQ decomposition of the V-block
    GetBond(lr)->GaugeTransfer(lr,truncError,s,Rtrans);
    SetLimits();
    return truncError;
}


// Do QX=QR/RQ/QL/LQ decomposition of the V-block and pass it on the the next site.
void SiteOperatorImp::CanonicalForm(Direction lr)
{
    GetBond(lr)->GaugeTransfer(lr,itsWs.QX(lr));
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
