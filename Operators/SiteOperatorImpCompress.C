#include "Operators/SiteOperatorImp.H"
#include "TensorNetworks/SVCompressor.H"
#include "TensorNetworks/TNSLogger.H"
#include "NumericalMethods/LapackSVDSolver.H"
#include "NumericalMethods/LapackQRSolver.H"
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
    default:
        assert(false);
    }
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

//std::tuple<MatrixRT,MatrixRT> ExtractM(Direction lr,const MatrixRT& Lp)
//{
////    cout << "Lp=" << Lp << endl;
//    int X1=Lp.GetNumRows()-1;
//    int X2=Lp.GetNumCols()-1;
//    MatrixRT M(X1,X2),Lprime;
//    switch(lr)
//    {
//    case DLeft:
//        M=Lp.SubMatrix(MatLimits(1,X1,1,X2));
//        Lprime=  MakeBlockMatrix(Lp,X1+2,X2+2,1);
////        cout << "Lprime=" << Lprime << endl;
//        for (int i=2;i<=X1+1;i++)
//        { //Clear out the M part of Lp
//            Lprime(i,i)=1.0;
//            for (int j=1;j<=i-1;j++)
//                Lprime(i,j)=0.0;
//            for (int j=i+1;j<=X2+1;j++)
//                Lprime(i,j)=0.0;
//        }
//        break;
//    case DRight:
//        for (index_t i:M.rows())
//            for (index_t j:M.cols())
//                M(i,j)=Lp(i+1,j+1);
//
//        Lprime=  MakeBlockMatrix(Lp,X1+2,X2+2,0);
//        for (int i=2;i<=X1+1;i++)
//        { //Clear out the M part of Lp
//            Lprime(i,i)=1.0;
//            for (int j=1;j<=i-1;j++)
//                Lprime(i,j)=0.0;
//            for (int j=i+1;j<=X2+1;j++)
//                Lprime(i,j)=0.0;
//        }
//        break;
//    }
//
//    return std::make_tuple(M,Lprime);
//}

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
    int Dw1=itsDw.Dw1,Dw2=itsDw.Dw2;
    int X1=Dw1-2,X2=Dw2-2,Xs=X2; //Chi and Chi_prime

    MatrixRT  V=Reshape(lr,1);
//    cout << "X1,X2,V=" << X1 << " " << X2 << " " << V.GetLimits() << endl;

    switch (lr)
    {
        case DLeft:
        {
            if (V.size()==0) return 0.0; //This will happen at the edges of an MPO
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
//                cout << std::scientific << std::setprecision(8) << "s=" << s.GetDiagonal() << endl;
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
                Reshape(lr,1,Qp);  //W is now Qp
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
            if (V.size()==0) return 0.0; //This will happen at the edges of an MPO
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
//                cout << std::fixed << std::setprecision(2) << "s=" << s.GetDiagonal() << endl;
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
                Reshape(lr,1,Qp);  //W is now Qp
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
    MatrixRT  A=Reshape(lr,0);
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
            Reshape(lr,0,VT);  //A is now Vdagger
            if (itsLeft_Neighbour) itsLeft_Neighbour->SVDTransfer(lr,sm,UV);
            break;
        }
        case DLeft:
        {
            UV=VT; //Set Vdagger
            Reshape(lr,0,U);  //A is now U
            if (itsRightNeighbour) itsRightNeighbour->SVDTransfer(lr,sm,UV);
            break;
        }
    }
    return itsTruncationError;
}



void SiteOperatorImp::CanonicalForm(Direction lr)
{

    MatrixRT  V=Reshape(lr,1);
    LapackQRSolver<double> solver;
    switch (lr)
    {
        case DLeft:
        {
//           cout << "V=" << V.GetLimits() << Max(fabs(V)) << endl;
            if (V.size()==0) return; //This will happen at the edges of an MPO
            auto [Q,L]=solver.SolveThinQL(V); //Solves V=Q*L
            L*=1.0/sqrt(itsd);
            Q*=sqrt(itsd);
            Reshape(lr,1,Q);  //A is now U
            MatrixRT Lplus=MakeBlockMatrix(L,L.GetNumRows()+1,L.GetNumCols()+1,1);
            if (itsRightNeighbour) itsRightNeighbour->QLTransfer(lr,Lplus);
            break;
        }
        case DRight:
        {
            if (V.size()==0) return; //This will happen at the edges of an MPO
            auto [L,Q]=solver.SolveThinLQ(V); //Solves V=L*Q
            L*=1.0/sqrt(itsd);
            Q*=sqrt(itsd);
            Reshape(lr,1,Q);  //A is now U
            MatrixRT Lplus=MakeBlockMatrix(L,L.GetNumRows()+1,L.GetNumCols()+1,0);
            if (itsLeft_Neighbour) itsLeft_Neighbour->QLTransfer(lr,Lplus);
            break;
        }

    }
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
            if (GetiW(0,0).GetNumCols()!=L.GetNumRows())
                NewBondDimensions(itsDw.Dw1,N1,true);
            else
                itsDw.Dw2=N1; //The contraction below will automatically reshape the Ws.
        }
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++)
            {
                MatrixRT W=GetiW(m,n);
                assert(W.GetNumCols()==L.GetNumRows());
                MatrixRT temp=W*L;
                W=temp; //Shallow copy
                SetiW(m,n,W);
                assert(W.GetNumCols()==itsDw.Dw2); //Verify shape is correct;
            }
        break;
    }
    case DLeft:
    {
        int N1=L.GetNumRows(); //N1=0 on the first site.
        if (N1>0 && N1!=itsDw.Dw1)
        {
            if (GetiW(0,0).GetNumRows()!=L.GetNumCols())
                NewBondDimensions(N1,itsDw.Dw2,true);
            else
                itsDw.Dw1=N1; //The contraction below will automatically reshape the As.
        }
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++)
            {
                MatrixRT W=GetiW(m,n);
                assert(L.GetNumCols()==W.GetNumRows());
                MatrixRT temp=L*W;
                W=temp; //Shallow copy
                assert(W.GetNumRows()==itsDw.Dw1); //Verify shape is correct;
                SetiW(m,n,W);
            }
        break;
    }

    }
    Update();
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
            if (GetiW(0,0).GetNumCols()!=UV.GetNumRows())
                NewBondDimensions(itsDw.Dw1,N1,true);
            else
                itsDw.Dw2=N1; //The contraction below will automatically reshape the As.
        }
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++)
            {
                MatrixRT W=GetiW(m,n);
                assert(W.GetNumCols()==UV.GetNumRows());
                MatrixRT temp=W*UV*s;
                W.SetLimits(0,0);
                W=temp; //Shallow copy
//                cout << "SVD transfer W(" << m << "," << n << ")=" << W << endl;
                assert(W.GetNumCols()==itsDw.Dw2); //Verify shape is correct;
                SetiW(m,n,W);
            }
        break;
    }
    case DLeft:
    {
        int N1=s.GetNumRows(); //N1=0 on the first site.
        if (N1>0 && N1!=itsDw.Dw1)
        {
            if (GetiW(0,0).GetNumRows()!=UV.GetNumCols())
                NewBondDimensions(N1,itsDw.Dw2,true);
            else
                itsDw.Dw1=N1; //The contraction below will automatically reshape the As.
        }

        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++)
            {
                MatrixRT W=GetiW(m,n);
                assert(UV.GetNumCols()==W.GetNumRows());
                MatrixRT temp=s*UV*W;
                W.SetLimits(0,0);
                W=temp; //Shallow copy
//                cout << "SVD transfer W(" << m << "," << n << ")=" << W << endl;
                assert(W.GetNumRows()==itsDw.Dw1); //Verify shape is correct;
                SetiW(m,n,W);
            }
        break;
    }

    }
    Update();
}



} //namespace
