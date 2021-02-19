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
        terror=CompressParkerOvM(lr,comp);
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
    CheckSync();
    MatrixRT  A=itsWOvM.Flatten(lr);
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
            itsWOvM.UnFlatten(VT*s_avg);
            if (itsLeft_Neighbour) itsLeft_Neighbour->QLTransfer(lr,Us);
            break;
        }
        case DLeft:
        {
            U.ReBase(lim);
            MatrixRT sV=sm*VT;
            sV.ReBase(lim);
            itsWOvM.UnFlatten(U*s_avg);
            if (itsRightNeighbour) itsRightNeighbour->QLTransfer(lr,sV);
            break;
        }
    }
    SyncOtoW(); //Get Q into the Ws.
    return itsTruncationError;
}

double SiteOperatorImp::CompressParkerOvM(Direction lr,const SVCompressorR* comp)
{
    CheckSync();
    auto [Q,RL]=itsWOvM.BlockSVD(lr,comp); // Do QX=QR/RQ/QL/LQ decomposition of the V-block
    itsWOvM.SetV(lr,Q); //Could be move inside BlockQX
    SyncOtoW(); //Get Q into the Ws.
    GetNeighbour(lr)->QLTransfer(lr,RL);
    return itsWOvM.GetTruncationError();
}


void SiteOperatorImp::CanonicalFormOvM(Direction lr)
{
    CheckSync();
    auto [Q,RL]=itsWOvM.BlockQX(lr); // Do QX=QR/RQ/QL/LQ decomposition of the V-block
    itsWOvM.SetV(lr,Q); //Could be move inside BlockQX
    SyncOtoW(); //Get Q into the Ws.
    GetNeighbour(lr)->QLTransfer(lr,RL);
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
        if (N1>0 && N1!=itsDw.Dw2)
        {
            if (itsWOvM.GetNumCols()!=L.GetNumRows())
                NewBondDimensions(itsDw.Dw1,N1,true);
            else
                itsDw.Dw2=N1; //The contraction below will automatically reshape the Ws.
        }
        assert(itsWOvM.GetColLimits()==L.GetRowLimits());
        TriType ul=itsWOvM.GetUpperLower();
        itsWOvM=MatrixOR(itsWOvM*L);
        itsWOvM.SetUpperLower(ul);
        break;
    }
    case DLeft:
    {
        int N1=L.GetNumRows(); //N1=0 on the first site.
        if (N1>0 && N1!=itsDw.Dw1)
        {
            if (itsWOvM.GetNumRows()!=L.GetNumCols())
                NewBondDimensions(N1,itsDw.Dw2,true);
            else
                itsDw.Dw1=N1; //The contraction below will automatically reshape the As.
        }
        assert(L.GetColLimits()==itsWOvM.GetRowLimits());
        TriType ul=itsWOvM.GetUpperLower();
        itsWOvM=MatrixOR(L*itsWOvM);
        itsWOvM.SetUpperLower(ul);
        break;
    }

    }
    SyncOtoW();
    SetLimits();
    CheckSync();
}


} //namespace
