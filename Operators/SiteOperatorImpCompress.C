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


double SiteOperatorImp::CompressParker(Direction lr,const SVCompressorR* comp)
{
    assert(comp);
    LapackQRSolver <double>  QRsolver;
    LapackSVDSolver<double> SVDsolver;
    int X1=itsDw.Dw1-2,X2=itsDw.Dw2-2; //Chi and Chi_prime
    if (X1<0) X1=0;
    if (X2<0) X2=0;
    int Xs=X2;

    MatrixRT  V=ReshapeV(lr);
    assert(V.size()!=0);
//    cout << "X1,X2,V=" << X1 << " " << X2 << " " << V.GetLimits() << endl;

    switch (lr)
    {
        case DLeft:
        {
            assert(V.GetNumRows()==itsd*itsd*(X1+1)); // Treate these like enforced comments on the
            assert(V.GetNumCols()==X2+1);             // dimensions of each matrix.
            auto [Qp,Lp]=QRsolver.SolveThinQL(V); //Solves V=Q*L
            Lp*=1.0/sqrt(itsd);
            Qp*=sqrt(itsd);
            double QLerr=Max(fabs(Qp*Lp-V));
            if (QLerr>1e-13)
               Logger->LogWarnV(1,"SiteOperatorImp::CompressParker QL error=%.1e ",QLerr);

            int Xq = (V.GetNumRows()>=V.GetNumCols()) ? X2 : itsd*itsd*(X1+1)-1;
            assert(Qp.GetNumRows()==itsd*itsd*(X1+1));
            assert(Qp.GetNumCols()==Xq+1);
            assert(Lp.GetNumRows()==Xq+1);
            assert(Lp.GetNumCols()==X2+1);
            assert(isOrthonormal(lr,Qp));

            MatrixRT Lpp;
            MatrixRT M=ExtractM1(lr,Lp);
            if (M.size()==0) return 0.0; //THis will happen if we have already compressed down to a unit matrix.
           {
                auto [U,s,VT]=SVDsolver.SolveAll(M,1e-14); //Solves M=U * s * VT
                AccumulateTruncationError(comp->Compress(U,s,VT));
//                cout << std::fixed << std::setprecision(2) << "DLeft s=" << s.GetDiagonal() << endl;
                Xs=s.GetDiagonal().size();
                MatrixRT sV=s*VT;
                assert(sV.GetNumRows()==Xs);
                assert(sV.GetNumCols()==X2);
                assert( U.GetNumRows()==Xq);
                assert( U.GetNumCols()==Xs);
                assert(IsUnit(Transpose(U)*U,1e-13));
                // Add lower right row and column
                MatrixRT  Up=MakeBlockMatrix( U,Xq+1,Xs+1,0);
                MatrixRT sVp=MakeBlockMatrix(sV,Xs+1,X2+1,0);
                assert(IsUnit(Transpose(Up)*Up,1e-13));
                // Add upper left row and column
                MatrixRT sVpp=MakeBlockMatrix(sVp,Xs+2,X2+2,1);
                MatrixRT Lprime=ExtractLprime(lr,Lp);

                Lpp=sVpp*Lprime; //This get passed on to the next site over.
                Qp*=Up;
                assert(isOrthonormal(lr,Qp));
                ReshapeV(lr,Qp);  //W is now Qp
#ifdef DEBUG
                char ns=GetNormStatus(1e-13);
                assert(ns=='L' || ns=='I');
#endif
            }

            if (itsRightNeighbour) itsRightNeighbour->QLTransfer(lr,Lpp);
            break;
        }
        case DRight:
        {
            assert(V.GetNumCols()==itsd*itsd*(X2+1)); // Treate these like enforced comments on the
            assert(V.GetNumRows()==X1+1);             // dimensions of each matrix.
            auto [Lp,Qp]=QRsolver.SolveThinLQ(V); //Solves V=Q*L
            Lp*=1.0/sqrt(itsd);
            Qp*=sqrt(itsd);
            double QLerr=Max(fabs(Lp*Qp-V));
            if (QLerr>1e-13)
               Logger->LogWarnV(1,"SiteOperatorImp::CompressParker QL error=%.1e ",QLerr);

            int Xq = (V.GetNumCols()>=V.GetNumRows()) ? X1 : itsd*itsd*(X2+1)-1;
            assert(Qp.GetNumCols()==itsd*itsd*(X2+1));
            assert(Qp.GetNumRows()==Xq+1);
            assert(Lp.GetNumCols()==Xq+1);
            assert(Lp.GetNumRows()==X1+1);
            assert(isOrthonormal(lr,Qp));

            MatrixRT Lpp;
            MatrixRT M=ExtractM1(lr,Lp);
            if (M.size()==0) return 0.0; //THis will happen if we have already compressed down to a unit matrix.
            {
                auto [U,s,VT]=SVDsolver.SolveAll(M,1e-14); //Solves M=U * s * VT
                AccumulateTruncationError(comp->Compress(U,s,VT));
//                cout << std::fixed << std::setprecision(2) << "DRight s=" << s.GetDiagonal() << endl;
                Xs=s.GetDiagonal().size();
                MatrixRT Us=U*s;
                assert(Us.GetNumRows()==X1);
                assert(Us.GetNumCols()==Xs);
                assert(VT.GetNumRows()==Xs);
                assert(VT.GetNumCols()==Xq);
                assert(IsUnit(VT*Transpose(VT),1e-13));

                // Add upper left row and column
                MatrixRT VTp=MakeBlockMatrix(VT,Xs+1,Xq+1,1);
                MatrixRT Usp=MakeBlockMatrix(Us,X1+1,Xs+1,1);
                // Add lower right row and column
                MatrixRT Uspp=MakeBlockMatrix(Usp,X1+2,Xs+2,0);
                MatrixRT Lprime=ExtractLprime(lr,Lp);

                Lpp=Lprime*Uspp; //This get passed on to the next site over.
                Qp=MatrixRT(VTp*Qp);
                assert(isOrthonormal(lr,Qp));
                ReshapeV(lr,Qp);  //W is now Qp
#ifdef DEBUG
                char ns=GetNormStatus(1e-13);
                assert(ns=='R' || ns=='I');
#endif
            }

            if (itsLeft_Neighbour) itsLeft_Neighbour->QLTransfer(lr,Lpp);
            break;
        }

    }
    return itsTruncationError;
}


double SiteOperatorImp::CompressStd(Direction lr,const SVCompressorR* comp)
{
    assert(comp);
    MatrixRT  A=ReshapeW(lr);
//    cout << "A=" << A << endl;
    LapackSVDSolver<double> solver;
    auto [U,sm,VT]=solver.SolveAll(A,1e-14); //Solves A=U * s * VT
//    cout << "U=" << U << endl;
    //
    //  Rescaling
    //
    double s_avg=Sum(sm.GetDiagonal())/sm.size();
//    cout << "VT=" << VT << endl;
    sm*=1.0/s_avg;
     switch (lr)
    {
        case DLeft:
            U*=s_avg;
            break;
        case DRight:
            VT*=s_avg;
            break;
    }

    AccumulateTruncationError(comp->Compress(U,sm,VT));
//    cout << "s_avg, sm=" << s_avg << " " << sm.GetDiagonal() << endl;
    MatrixRT UV;// This get transferred through the bond to a neighbouring site.
    switch (lr)
    {
        case DRight:
        {
            UV=U;
            ReshapeW(lr,VT);  //A is now Vdagger
            if (itsLeft_Neighbour) itsLeft_Neighbour->SVDTransfer(lr,sm,UV);
            break;
        }
        case DLeft:
        {
            UV=VT; //Set Vdagger
            ReshapeW(lr,U);  //A is now U
            if (itsRightNeighbour) itsRightNeighbour->SVDTransfer(lr,sm,UV);
            break;
        }
    }
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

void SiteOperatorImp::CanonicalForm(Direction lr)
{
#ifdef DEBUG
    MatrixRT Worig=ReshapeW(lr);
//    cout << "Worig=" << Worig << endl;
#endif
    MatrixRT  V=ReshapeV(lr);
    if(V.size()==0) return; //We only this for unit operators with Dw1==Dw2==1;
//    cout << "V=" << V << endl;
    LapackQRSolver<double> solver;
    switch (lr)
    {
        case DLeft:
        {
            auto [Q,L]=solver.SolveThinQL(V); //Solves V=Q*L
            L*=1.0/sqrt(itsd);
            Q*=sqrt(itsd);
//            cout << "L=" << L << endl;
//            cout << "Q=" << Q << endl;
//            cout << "Qt*Q=" << Transpose(Q)*Q << endl;
            ReshapeV(lr,Q);  //A is now U
            MatrixRT Lplus=MakeBlockMatrix(L,L.GetNumRows()+1,L.GetNumCols()+1,1);
            if (itsRightNeighbour) itsRightNeighbour->QLTransfer(lr,Lplus);
#ifdef DEBUG
            V=ReshapeV(lr);
            assert(V.GetLimits()==Q.GetLimits());
            assert(Max(fabs(Q-V))<1e-15);
            MatrixRT Qplus=ReshapeW(lr);
            MatrixRT QL=Qplus*Lplus;
//            cout << "Lplus=" << Lplus << endl;
//            cout << "Qplus=" << Qplus << endl;
//            cout << "QL=" << QL << endl;
//            cout << "Worig-QL=" << Worig-QL << endl;
            assert(Max(fabs(Worig-QL ))<1e-8);
            if (itsDw.Dw1>1 && itsDw.Dw2>1)
            {
                MatrixRT del=ContractDel(lr);
                if(!IsUnit(del,1e-13))
                {
                    cout << std::fixed << std::setprecision(0) << "del=" << del << endl;
                    assert(false);
                }
            }

#endif
            break;
        }
        case DRight:
        {
            auto [L,Q]=solver.SolveThinLQ(V); //Solves V=L*Q
            L*=1.0/sqrt(itsd);
            Q*=sqrt(itsd);
            ReshapeV(lr,Q);  //A is now U
            MatrixRT Lplus=MakeBlockMatrix(L,L.GetNumRows()+1,L.GetNumCols()+1,0);
            if (itsLeft_Neighbour) itsLeft_Neighbour->QLTransfer(lr,Lplus);
#ifdef DEBUG
            V=ReshapeV(lr);
            assert(V.GetLimits()==Q.GetLimits());
            assert(Max(fabs(Q-V))<1e-15);
            MatrixRT Qplus=ReshapeW(lr);
            MatrixRT LQ=Lplus*Qplus;
//            cout << "Worig-LQ=" << Worig-LQ << endl;
            assert(Max(fabs(Worig-LQ ))<1e-13);

            if (itsDw.Dw1>1 && itsDw.Dw2>1)
            {
                MatrixRT del=ContractDel(lr);
                if(!IsUnit(del,1e-13))
                {
                    cout << std::fixed << std::setprecision(0) << "del=" << del << endl;
                    assert(false);
                }
            }
#endif
            break;
        }

    }
    SetLimits();

}

//
//  Hunt for the zero pivots in L and remove those rows from L and columns from Q
//
bool Shrink(MatrixRT& L, MatrixRT& Q,double eps)
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
    for (int m=0; m<itsd; m++)
    for (int n=0; n<itsd; n++)
    {
        const MatrixRT& W=Transpose(GetW(m,n));
        MatrixRT RWR=R*W*Rinv;
        SetW(m,n,Transpose(RWR));
    }
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
    LapackQRSolver <double>  QRsolver;

    int X=itsDw.Dw1-2; //Chi
//    if (X1<0) X1=0;
//    if (X2<0) X2=0;
    MatrixRT Lp(X+1,X+1),LpOld(X+1,X+1);
    Unit(Lp);
    MatrixRT Id(X+1,X+1);
    Unit(Id);

    double eta=8.111111;
    int niter=1;
    do
    {
        MatrixRT  V=ReshapeV(lr);
        auto [Q,L]=QRsolver.SolveThinQL(V); //Solves V=Q*L
        X=itsDw.Dw1-2; //Chi
        assert(L.GetNumRows()==X+1);
        assert(L.GetNumCols()==X+1);
        double sgn=L(X+1,X+1);
        L*=1.0/sgn;
        Q*=sgn;

//        cout << std::scientific << std::setprecision(8) << "Min(Diag(L)=" << Min(L.GetDiagonal()) << endl;
        if (Shrink(L,Q,1e-13))
        {
            cout << "L*Q-V=" << Max(fabs(Q*L-V)) << endl;
            assert(Max(fabs(Q*L-V))<1e-13);
        }
         eta=8.111;
        if (L.GetNumRows()==L.GetNumCols())
        {
            Id.SetLimits(L.GetLimits(),true);
            eta=Max(fabs(L-Id));
        }
        cout << "eta=" << eta << "  sgn=" << sgn << endl;
        MatrixRT Lplus=MakeBlockMatrix(L,L.GetNumRows()+1,L.GetNumCols()+1,1);

        ReshapeV(lr,Q);
        // Get out here so we leave the Ws left normalized.
        if (niter++>100) break;
        //
        //  Do W->L*W
        //
        for (int m=0; m<itsd; m++)
        for (int n=0; n<itsd; n++)
        {
            MatrixRT W=GetW(m,n);
            itsWs(m+1,n+1)=Lplus*W;
        }
        itsDw.Dw1=Lplus.GetNumRows();
        MatrixRT LLp=L*Lp;
        Lp=LLp;

//        cout << "L-Id=" << L-Id << endl;

    } while (eta>1e-13);
    cout << std::fixed << std::setprecision(2) << "Lp=" << Lp << endl;
//    cout << std::fixed << std::setprecision(2) << "LpT*Lp=" << Transpose(Lp)*Lp << endl;


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
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++)
            {
                MatrixRT W=GetW(m,n);
                assert(W.GetNumCols()==L.GetNumRows());
                SetW(m,n,W*L);
                assert(GetW(m,n).GetNumCols()==itsDw.Dw2); //Verify shape is correct;
            }
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
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++)
            {
                MatrixRT W=GetW(m,n);
                assert(L.GetNumCols()==W.GetNumRows());
                SetW(m,n,L*W);
                assert(GetW(m,n).GetNumRows()==itsDw.Dw1); //Verify shape is correct;
            }
        break;
    }

    }
    SetLimits();
}

void SiteOperatorImp::SVDTransfer(Direction lr,const DiagonalMatrixRT& s,const MatrixRT& UV)
{
//    cout << "SVD transfer s=" << s << " UV=" << UV << endl;
    switch (lr)
    {
    case DRight:
    {
        int N1=s.GetNumRows(); //N1=0 on the first site.
        if (N1>0 && N1!=itsDw.Dw2)
        {
            if (GetW(0,0).GetNumCols()!=UV.GetNumRows())
                NewBondDimensions(itsDw.Dw1,N1,true);
            else
                itsDw.Dw2=N1; //The contraction below will automatically reshape the As.
        }
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++)
            {
                MatrixRT W=GetW(m,n);
                assert(W.GetNumCols()==UV.GetNumRows());
                MatrixRT temp=W*UV*s;
                W.SetLimits(0,0);
                W=temp; //Shallow copy
//                cout << "SVD transfer W(" << m << "," << n << ")=" << W << endl;
                assert(W.GetNumCols()==itsDw.Dw2); //Verify shape is correct;
                SetW(m,n,W);
            }
        break;
    }
    case DLeft:
    {
        int N1=s.GetNumRows(); //N1=0 on the first site.
        if (N1>0 && N1!=itsDw.Dw1)
        {
            if (GetW(0,0).GetNumRows()!=UV.GetNumCols())
                NewBondDimensions(N1,itsDw.Dw2,true);
            else
                itsDw.Dw1=N1; //The contraction below will automatically reshape the As.
        }

        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++)
            {
                MatrixRT W=GetW(m,n);
                assert(UV.GetNumCols()==W.GetNumRows());
                MatrixRT temp=s*UV*W;
                W.SetLimits(0,0);
                W=temp; //Shallow copy
//                cout << "SVD transfer W(" << m << "," << n << ")=" << W << endl;
                assert(W.GetNumRows()==itsDw.Dw1); //Verify shape is correct;
                SetW(m,n,W);
            }
        break;
    }

    }
    SetLimits();
}



} //namespace
