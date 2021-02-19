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

MatrixRT MakeBlockMatrix(const MatrixRT& M,int Dr,int Dc,int offset)
{
    MatrixRT ret(Dr,Dc);
    Fill(ret,0.0);
    if (offset==0)
        ret(Dr,Dc)=1.0;
    else
        ret(1,1)=1.0;

#ifdef DEBUG
    int nr=M.GetNumRows();
    int nc=M.GetNumCols();
#endif
    assert(Dr>nr);
    assert(Dc>nc);
    assert(offset>=0);
    assert(nr+offset<=Dr);
    assert(nc+offset<=Dc);
    for (int i:M.rows())  //Sub matrix multiply
        for (int j:M.cols())
            ret(i+offset,j+offset)=M(i,j);
    return ret;
}


MatrixRT MakeBlockMatrix(const MatrixRT& M,int D,int offset)
{
    MatrixRT ret(D,D);
    Unit(ret);
#ifdef DEBUG
    int n=M.GetNumRows();
#endif
    assert(n==M.GetNumCols());
    assert(D>n);
    assert(offset>=0);
    assert(n+offset<=D);
    for (int i:M.rows())  //Sub matrix multiply
        for (int j:M.cols())
            ret(i+offset,j+offset)=M(i,j);
    return ret;
}

MatrixRT ExtractLprime(Direction lr,const MatrixRT& Lp)
{
//    cout << "Lp=" << Lp << endl;
    int X1=Lp.GetNumRows()-1;
    int X2=Lp.GetNumCols()-1;
    MatrixRT Lprime;

    switch(lr)
    {
    case DLeft:
        Lprime.SetLimits(X2+2,X2+2);
        Unit(Lprime);
        for (int j=2;j<=X2;j++)
            Lprime(X2+2,j)=Lp(X1+1,j); //Copy the tvector part of Lp
        break;
    case DRight:
        Lprime.SetLimits(X1+2,X1+2);
        Unit(Lprime);
        for (int i=2;i<=X1;i++)
            Lprime(i,X2+2)=Lp(i,X2+1); //Copy the tvector part of Lp
        break;
    }

    return Lprime;
}

MatrixRT ExtractM1(Direction lr,const MatrixRT& Lp)
{
//    cout << "Lp=" << Lp << endl;
    int X1=Lp.GetNumRows()-1;
    int X2=Lp.GetNumCols()-1;
    MatrixRT M(X1,X2);
    switch(lr)
    {
    case DLeft:
        M=Lp.SubMatrix(MatLimits(1,X1,1,X2));
        break;
    case DRight:
        for (index_t i:M.rows())
            for (index_t j:M.cols())
                M(i,j)=Lp(i+1,j+1);

        break;
    }

    return M;
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

MatrixRT SiteOperatorImp::ContractDel(Direction lr) const
{
    int X1=itsDw.Dw1-2;
    int X2=itsDw.Dw2-2;
    MatrixRT del;
    switch (lr)
    {
    case DLeft:
        del.SetLimits(X2+1,X2+1);
        for (int b=1; b<=X2+1; b++)
        for (int c=1; c<=X2+1; c++)
        {
            double t=0.0;
            for (int a=1; a<=X1+1; a++)
                t+=Contract(a,b,a,c); // <Wab,Wac> dot product between columns b and c
            del(b,c)=t;
        }
        break;
    case DRight:
        del.SetLimits(X1+1,X1+1);
        for (int b=0; b<=X1; b++)
        for (int c=0; c<=X1; c++)
        {
            double t=0.0;
            for (int a=0; a<=X2; a++)
                t+=Contract(b,a,c,a); // <Wba,Wba> dot product between rows b and c
            del(b+1,c+1)=t;
        }
        break;
    }

    return del;
}
MatrixRT MakeBlockMatrix(Direction lr,const MatrixRT& LR)
{
    int NR=LR.GetNumRows()+1;
    int NC=LR.GetNumRows()+1;
    MatrixRT LRplus;
    switch (lr)
    {
    case DLeft:
        LRplus=MakeBlockMatrix(LR,NR,NC,1);
        break;
    case DRight:
        LRplus=MakeBlockMatrix(LR,NR,NC,0);
        break;
    }
    return LRplus;
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
    MatrixOR Wcopy=itsWOvM;
    auto [Q,RL]=itsWOvM.BlockQX(lr); // Do QX=QR/RQ/QL/LQ decomposition of the V-block
    itsWOvM.SetV(lr,Q); //Could be move inside BlockQX
    SyncOtoW(); //Get Q into the Ws.
    GetNeighbour(lr)->QLTransfer(lr,RL);
}

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

//
//  <Wdagger(w11,w12),W(w21,w22)>.  The ws are zero based.
//
double SiteOperatorImp::ContractT(int w11, int w12, int w21, int w22) const
{
    return Contract(w12,w11,w22,w21);
}

double SiteOperatorImp::Contract(int w11, int w12, int w21, int w22) const
{
    double r1=0.0;
//    cout << " m  n   " << " Wt(" << w11+1 << "," << w12+1 << ") " << " W(" << w21+1 << "," << w22+1 << ")" << endl;
    for (int m=0; m<itsd; m++)
    for (int n=0; n<itsd; n++)
    {
        const MatrixRT& W=GetW(m,n); //Work on lower triangular version for now.
        r1+=W(w11+1,w12+1)*W(w21+1,w22+1);
//        cout << m << " " << n << " " << W(w11+1,w12+1) << " " << W(w21+1,w22+1) << " " << r1 << endl ;
    }
    return r1/itsd; //Divide by Tr[I]
}

MatrixRT SiteOperatorImp::BuildK(int M) const
{
    MatrixRT K(M,M),I(M,M);
    Unit(I);
    for (int b=0;b<=M-1;b++)
    for (int a=0;a<=M-1;a++)
    {
        K(b+1,a+1)=I(b+1,a+1)-Contract(b,a,M,M);
    }
    return K;
}
VectorRT SiteOperatorImp::Buildc(int M) const
{
    VectorRT c(M);
    Fill(c,0.0);
    for (int b=0;b<=M-1;b++)
    for (int a=0;a<=M-1;a++)
    {
        c(b+1)+=Contract(M,b,M,a);
    }
    return c;
}

void SiteOperatorImp::GaugeTransform(const MatrixRT& R, const MatrixRT& Rinv)
{
    assert(R.GetLimits()==Rinv.GetLimits());
    assert(IsUnit(R*Rinv,1e-13));
    TriType ul=itsWOvM.GetUpperLower();
    itsWOvM=MatrixOR(R*itsWOvM*Rinv);
    itsWOvM.SetUpperLower(ul);
//    for (int m=0; m<itsd; m++)
//    for (int n=0; n<itsd; n++)
//    {
//        const MatrixRT& W=Transpose(GetW(m,n));
//        MatrixRT RWR=R*W*Rinv;
//        SetW(m,n,Transpose(RWR));
//    }
}

double SiteOperatorImp::Contract_sM(int M) const
{
    double dM=ContractT(M,M,M,M);
    assert(dM<1.0);
    assert(dM>=0.0);
    double sM=0.0;
    for (int a=0;a<=M;a++)
        sM+=ContractT(a,M,a,M);
    cout << "sM,dM = " << sM << " " << dM << endl;
    assert(sM>=0.0);
    sM/=(1-dM);
    assert(sM>=0.0);
    return sqrt(sM);

}
double SiteOperatorImp::Contract_sM1(int M) const
{
    int X=itsDw.Dw1-2;
    double sM=0.0;
    for (int a=1;a<=X+1;a++)
        sM+=ContractT(M,a,M,a);
    return sqrt(sM);

}
void SiteOperatorImp::iCanonicalFormTriangular(Direction lr)
{
    assert(itsDw.Dw1==itsDw.Dw2); //Make sure we are square
    int X=itsDw.Dw1-2; //Chi
    MatrixRT RT(X+2,X+2); //Accumulated gauge transform
    LinearSolver<double>* solver=new LapackLinearSolver<double>();;
    for (int M=1;M<=X;M++)
    {
//        cout << "Init del=" << ContractDel(DLeft) << endl;
        MatrixRT K=BuildK(M);
        VectorRT c=Buildc(M);
//        cout << "M=" << M << endl;
//        cout << "K=" << K << endl;
//        cout << "c=" << c << endl;
        VectorRT r=solver->SolveLowerTri(K,c);
//        cout << "r=" << r << endl;
        MatrixRT R(X+2,X+2),Rinv(X+2,X+2);
        Unit(R);
        for (int b=0;b<=M-1;b++)
            R(b+1,M+1)=r(b+1);
//        cout << "R=" << R << endl;
        Unit(Rinv);
        for (int b=0;b<=M-1;b++)
            Rinv(b+1,M+1)=-r(b+1);
//        cout << "Rinv=" << Rinv << endl;
//        cout << "R*Rinv=" << R*Rinv << endl;

        GaugeTransform(R,Rinv);
//        cout << "After gauge del=" << ContractDel(DLeft) << endl;
        RT=R*RT;
//        {
//            MatrixRT QL=ReshapeV(DLeft);
//            cout << "Before norm QT*Q=" << Transpose(QL)*QL << endl;
//        }
        double sM=Contract_sM1(M);
        assert(fabs(sM)>1e-14);
 //       cout << "sM^2=" << sM*sM << endl;
        if (fabs(sM)>1e-14)
        {
            Unit(R);
            Unit(Rinv);
            R   (M+1,M+1)=1.0/sM;
            Rinv(M+1,M+1)=sM;
            GaugeTransform(R,Rinv);
            RT=R*RT;
        }
        else
        {
            assert(false);
            //Need code to remove rows and columns
        }
//        cout << "After norm del=" << ContractDel(DLeft) << endl;
//        {
//            MatrixRT QL=ReshapeV(DLeft);
//            cout << "After norm QT*Q=" << Transpose(QL)*QL << endl;
//        }

    }
    delete solver;
}


void SiteOperatorImp::iCanonicalFormQRIter(Direction lr)
{
    assert(itsDw.Dw1==itsDw.Dw2); //Make sure we are square
    CheckSync();
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


//
//  Do W = W*L, but L is (Dw-1)x(Dw-1) is smaller than W
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
            if (GetW(0,0).GetNumCols()!=L.GetNumRows())
                NewBondDimensions(itsDw.Dw1,N1,true);
            else
                itsDw.Dw2=N1; //The contraction below will automatically reshape the Ws.
        }
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
            if (GetW(0,0).GetNumRows()!=L.GetNumCols())
                NewBondDimensions(N1,itsDw.Dw2,true);
            else
                itsDw.Dw1=N1; //The contraction below will automatically reshape the As.
        }
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

//void SiteOperatorImp::SVDTransfer(Direction lr,const DiagonalMatrixRT& s,const MatrixRT& UV)
//{
////    cout << "SVD transfer s=" << s << " UV=" << UV << endl;
//    switch (lr)
//    {
//    case DRight:
//    {
//        int N1=s.GetNumRows(); //N1=0 on the first site.
//        if (N1>0 && N1!=itsDw.Dw2)
//        {
//            if (GetW(0,0).GetNumCols()!=UV.GetNumRows())
//                NewBondDimensions(itsDw.Dw1,N1,true);
//            else
//                itsDw.Dw2=N1; //The contraction below will automatically reshape the As.
//        }
//        for (int m=0; m<itsd; m++)
//            for (int n=0; n<itsd; n++)
//            {
//                MatrixRT W=GetW(m,n);
//                assert(W.GetNumCols()==UV.GetNumRows());
//                MatrixRT temp=W*UV*s;
//                W.SetLimits(0,0);
//                W=temp; //Shallow copy
////                cout << "SVD transfer W(" << m << "," << n << ")=" << W << endl;
//                assert(W.GetNumCols()==itsDw.Dw2); //Verify shape is correct;
//                SetW(m,n,W);
//            }
//        break;
//    }
//    case DLeft:
//    {
//        int N1=s.GetNumRows(); //N1=0 on the first site.
//        if (N1>0 && N1!=itsDw.Dw1)
//        {
//            if (GetW(0,0).GetNumRows()!=UV.GetNumCols())
//                NewBondDimensions(N1,itsDw.Dw2,true);
//            else
//                itsDw.Dw1=N1; //The contraction below will automatically reshape the As.
//        }
//
//        for (int m=0; m<itsd; m++)
//            for (int n=0; n<itsd; n++)
//            {
//                MatrixRT W=GetW(m,n);
//                assert(UV.GetNumCols()==W.GetNumRows());
//                MatrixRT temp=s*UV*W;
//                W.SetLimits(0,0);
//                W=temp; //Shallow copy
////                cout << "SVD transfer W(" << m << "," << n << ")=" << W << endl;
//                assert(W.GetNumRows()==itsDw.Dw1); //Verify shape is correct;
//                SetW(m,n,W);
//            }
//        break;
//    }
//
//    }
//    SetLimits();
//    CheckSync();
//
//}
//


} //namespace
