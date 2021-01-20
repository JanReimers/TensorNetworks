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

std::tuple<MatrixRT,MatrixRT> ExtractM(Direction lr,const MatrixRT& Lp)
{
    int X=Lp.GetNumRows()-1;
    assert(X==Lp.GetNumCols()-1);
    MatrixRT M(X,X),Lprime;
    switch(lr)
    {
    case DLeft:
        M=Lp.SubMatrix(MatLimits(1,X,1,X));
        Lprime=  MakeBlockMatrix(Lp,X+2,X+2,1);
        for (int i=2;i<=X+1;i++)
        { //Clear out the M part of Lp
            Lprime(i,i)=1.0;
            for (int j=i+1;j<=X+1;j++)
                Lprime(i,j)=Lprime(j,i)=0.0;
        }
        break;
    case DRight:
        for (index_t i:M.rows())
            for (index_t j:M.cols())
                M(i,j)=Lp(i+1,j+1);

        Lprime=  MakeBlockMatrix(Lp,X+2,X+2,0);
        for (int i=2;i<=X+1;i++)
        { //Clear out the M part of Lp
            Lprime(i,i)=1.0;
            for (int j=i+1;j<=X+1;j++)
                Lprime(i,j)=Lprime(j,i)=0.0;
        }
        break;
    }

    return std::make_tuple(M,Lprime);
}


void SiteOperatorImp::CompressParker(Direction lr,const SVCompressorR* comp)
{
    assert(comp);
    LapackQRSolver <double>  QRsolver;
    LapackSVDSolver<double> SVDsolver;
    int Dw1=itsDw.Dw1,Dw2=itsDw.Dw2;
    int X1=Dw1-2,X2=Dw2-2,Xs=X2; //Chi and Chi_prime

    MatrixRT  V=ReshapeV(lr);

    switch (lr)
    {
        case DLeft:
        {
            assert(V.GetNumRows()==itsd*itsd*(X1+1)); // Treate these like enforced comments on the
            assert(V.GetNumCols()==X2+1);             // dimensions of each matrix.
            auto [Qp,Lp]=QRsolver.SolveThinQL(V); //Solves V=Q*L
//            cout << "Lp(X2+1,X2+1)=" << Lp(X2+1,X2+1) << endl;
            assert(fabs(Lp(X2+1,X2+1)-1.0)<1e-15);
            double QLerr=Max(fabs(Qp*Lp-V));
            if (QLerr>1e-13)
               Logger->LogWarnV(1,"SiteOperatorImp::CompressParker QL error=%.1e ",QLerr);

            assert(Qp.GetNumRows()==itsd*itsd*(X1+1));
            assert(Qp.GetNumCols()==X2+1);
            assert(Lp.GetNumRows()==X2+1);
            assert(Lp.GetNumCols()==X2+1);
            assert(IsUnit(Transpose(Qp)*Qp,1e-13));

            MatrixRT Lpp;
            auto [M,Lprime]=ExtractM(lr,Lp);
            if (IsDiagonal(M,1e-14))
            {
                cout << std::fixed << std::setprecision(2) << "M=" << M.GetDiagonal() << endl;
                assert(Min(fabs(M.GetDiagonal()))>1e-13); //If this trigger we need to compress
                ReshapeV(lr,Qp);  //W is now Q
                Lpp=MakeBlockMatrix(Lp,X2+2,1);
            }
            else
            {
                auto [U,s,VT]=SVDsolver.SolveAll(M,1e-14); //Solves M=U * s * VT
                itsTruncationError=comp->Compress(U,s,VT);
                cout << std::fixed << std::setprecision(2) << "s=" << s.GetDiagonal() << endl;
                Xs=s.GetDiagonal().size();
                MatrixRT sV=s*VT;
                assert(sV.GetNumRows()==Xs);
                assert(sV.GetNumCols()==X2);
                assert( U.GetNumRows()==X2);
                assert( U.GetNumCols()==Xs);
                assert(IsUnit(Transpose(U)*U,1e-13));

                // Add lower right row and column
                MatrixRT  Up=MakeBlockMatrix( U,X2+1,Xs+1,0);
                MatrixRT sVp=MakeBlockMatrix(sV,Xs+1,X2+1,0);
                assert(IsUnit(Transpose(Up)*Up,1e-13));
                // Add upper left row and column
                MatrixRT sVpp=MakeBlockMatrix(sVp,Xs+2,X2+2,1);

                Lpp=sVpp*Lprime; //This get passed on to the next site over.
                Qp*=Up;
                assert(IsUnit(Transpose(Qp)*Qp,1e-13));
                ReshapeV(lr,Qp);  //W is now Qp
                assert(GetNormStatus(1e-13)=='L');
            }

            if (itsRightNeighbour) itsRightNeighbour->QLTransfer(lr,Lpp);
            break;
        }
        case DRight:
        {
            assert(V.GetNumCols()==itsd*itsd*(X2+1)); // Treate these like enforced comments on the
            assert(V.GetNumRows()==X1+1);             // dimensions of each matrix.
            auto [Lp,Qp]=QRsolver.SolveThinLQ(V); //Solves V=Q*L
            double QLerr=Max(fabs(Lp*Qp-V));
            if (QLerr>1e-13)
               Logger->LogWarnV(1,"SiteOperatorImp::CompressParker QL error=%.1e ",QLerr);
            assert(Qp.GetNumCols()==itsd*itsd*(X2+1));
            assert(Qp.GetNumRows()==X1+1);
            assert(Lp.GetNumRows()==X1+1);
            assert(Lp.GetNumCols()==X1+1);
            assert(IsUnit(Qp*Transpose(Qp),1e-13));

            MatrixRT Lpp;
            auto [M,Lprime]=ExtractM(lr,Lp);
//            cout << "Lp=" << Lp << endl;
//            cout << "M=" << M << endl;
//            cout << "Lprime=" << Lprime << endl;
            if (IsDiagonal(M,1e-14))
            {
                cout << std::fixed << std::setprecision(2) << "M=" << M.GetDiagonal() << endl;
                assert(Min(fabs(M.GetDiagonal()))>1e-13); //If this trigger we need to compress
                ReshapeV(lr,Qp);  //W is now Q
                Lpp=MakeBlockMatrix(Lp,X1+2,0);
            }
            else
            {
                auto [U,s,VT]=SVDsolver.SolveAll(M,1e-14); //Solves M=U * s * VT
                itsTruncationError=comp->Compress(U,s,VT);
                cout << std::fixed << std::setprecision(2) << "s=" << s.GetDiagonal() << endl;
                Xs=s.GetDiagonal().size();
                MatrixRT Us=U*s;
                assert(Us.GetNumRows()==X1);
                assert(Us.GetNumCols()==Xs);
                assert(VT.GetNumRows()==Xs);
                assert(VT.GetNumCols()==X1);
                assert(IsUnit(VT*Transpose(VT),1e-13));

                // Add upper left row and column
                MatrixRT VTp=MakeBlockMatrix(VT,Xs+1,X1+1,1);
                MatrixRT Usp=MakeBlockMatrix(Us,X1+1,Xs+1,1);
                // Add lower right row and column
                MatrixRT Uspp=MakeBlockMatrix(Usp,X1+2,Xs+2,0);

                Lpp=Lprime*Uspp; //This get passed on to the next site over.
                Qp=MatrixRT(VTp*Qp);
                assert(IsUnit(Qp*Transpose(Qp),1e-13));
//                cout << "Qp=" << Qp << endl;
                ReshapeV(lr,Qp);  //W is now Qp
                assert(GetNormStatus(1e-13)=='R');
            }

            if (itsLeft_Neighbour) itsLeft_Neighbour->QLTransfer(lr,Lpp);
            break;
        }

    }
}


void SiteOperatorImp::CompressStd(Direction lr,const SVCompressorR* comp)
{
    assert(comp);
    MatrixRT  A=Reshape(lr);
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

    itsTruncationError=comp->Compress(U,sm,VT);
//    cout << "s_avg, sm=" << s_avg << " " << sm.GetDiagonal() << endl;
    MatrixRT UV;// This get transferred through the bond to a neighbouring site.
    switch (lr)
    {
        case DRight:
        {
            UV=U;
            Reshape(lr,VT);  //A is now Vdagger
            if (itsLeft_Neighbour) itsLeft_Neighbour->SVDTransfer(lr,sm,UV);
            break;
        }
        case DLeft:
        {
            UV=VT; //Set Vdagger
            Reshape(lr,U);  //A is now U
            if (itsRightNeighbour) itsRightNeighbour->SVDTransfer(lr,sm,UV);
            break;
        }
    }
}



void SiteOperatorImp::CanonicalForm(Direction lr)
{

    MatrixRT  V=ReshapeV(lr);
    LapackQRSolver<double> solver;
    switch (lr)
    {
        case DLeft:
        {
            auto [Q,L]=solver.SolveThinQL(V); //Solves V=Q*L
            ReshapeV(lr,Q);  //A is now U
            MatrixRT Lplus=MakeBlockMatrix(L,L.GetNumRows()+1,1);
            if (itsRightNeighbour) itsRightNeighbour->QLTransfer(lr,Lplus);
            break;
        }
        case DRight:
        {
            auto [L,Q]=solver.SolveThinLQ(V); //Solves V=L*Q
            ReshapeV(lr,Q);  //A is now U
            MatrixRT Lplus=MakeBlockMatrix(L,L.GetNumRows()+1,0);
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
                Reshape(itsDw.Dw1,N1,true);
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
                Reshape(N1,itsDw.Dw2,true);
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
    Init_lr();
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
                Reshape(itsDw.Dw1,N1,true);
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
                Reshape(N1,itsDw.Dw2,true);
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
    Init_lr();
}



} //namespace
