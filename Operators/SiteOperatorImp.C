#include "Operators/SiteOperatorImp.H"
#include "TensorNetworks/SVCompressor.H"
#include "TensorNetworksImp/SpinCalculator.H"
#include "NumericalMethods/LapackSVDSolver.H"
#include "NumericalMethods/LapackQRSolver.H"
#include "Containers/Vector3.H"
#include "oml/diagonalmatrix.h"
#include <complex>

namespace TensorNetworks
{

//
//  Build with Dw=1 identity operators
//
SiteOperatorImp::SiteOperatorImp(int d)
    : itsd(d)
    , itsDw12(1,1,Vector<int>(1),Vector<int>(1))
    , itsTruncationError(0.0)
    , itsWs(d,d)
{
    itsDw12.w1_first(1)=1;
    itsDw12.w2_last (1)=1;

    MatrixRT I0(1,1),I1(1,1);
    I0(1,1)=0.0;
    I1(1,1)=1.0;

    for (int m=0; m<itsd; m++)
        for (int n=0; n<itsd; n++)
        {
            itsWs(m+1,n+1)= (m==n) ? I1 : I0;
            assert(itsWs(m+1,n+1).GetNumRows()==itsDw12.Dw1);
            assert(itsWs(m+1,n+1).GetNumCols()==itsDw12.Dw2);
        }
}

SiteOperatorImp::SiteOperatorImp(int d, double S, SpinOperator so) //Construct spin operator
    : itsd(d)
    , itsDw12(1,1,Vector<int>(1),Vector<int>(1))
    , itsTruncationError(0.0)
    , itsWs(d,d)
{
    itsDw12.w1_first(1)=1;
    itsDw12.w2_last (1)=1;
    SpinCalculator sc(S);
    for (int m=0; m<itsd; m++)
        for (int n=0; n<itsd; n++)
        {
            itsWs(m+1,n+1)=sc.Get(m,n,so);
            assert(itsWs(m+1,n+1).GetNumRows()==itsDw12.Dw1);
            assert(itsWs(m+1,n+1).GetNumCols()==itsDw12.Dw2);
        }
}
//
//  Build from a W rep object
//
SiteOperatorImp::SiteOperatorImp(int d, Position lbr, const OperatorClient* H)
    : itsd(d)
    , itsDw12(H->GetDw12(lbr))
    , itsTruncationError(0.0)
    , itsWs(d,d)
{
    for (int m=0; m<itsd; m++)
        for (int n=0; n<itsd; n++)
        {
            itsWs(m+1,n+1)=H->GetW(lbr,m,n);
            assert(itsWs(m+1,n+1).GetNumRows()==itsDw12.Dw1);
            assert(itsWs(m+1,n+1).GetNumCols()==itsDw12.Dw2);
        }
}
//
// Build from a trotter decomp.
//
SiteOperatorImp::SiteOperatorImp(int d, Direction lr,const MatrixRT& U, const DiagonalMatrixRT& s)
    : itsd(d)
    , itsDw12()
    , itsTruncationError(0.0)
    , itsWs(d,d)
{
    int Dw=s.GetNumRows();
    if (lr==DLeft)
    {
        // Build up w limits
        Vector<int> first(Dw);
        Vector<int> last (1);
        Fill(first,1);
        Fill(last,Dw);
        itsDw12=Dw12(1,Dw,first,last);
        int i1=1; //Linear index for (m,n) = 1+m+p*n
        //  Fill W^(m,n)_w matrices
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++,i1++)
            {
                itsWs(m+1,n+1)=MatrixRT(1,Dw);
                for (int w=1; w<=Dw; w++)
                    itsWs(m+1,n+1)(1,w)=U(i1,w)*sqrt(s(w,w));
                //cout << "Left itsWs(" << m << "," << n << ") = " << itsWs(m+1,n+1) << endl;
                assert(itsWs(m+1,n+1).GetNumRows()==itsDw12.Dw1);
                assert(itsWs(m+1,n+1).GetNumCols()==itsDw12.Dw2);
            }
    }
    else if (lr==DRight)
    {
        // Build up w limits
        Vector<int> first(1);
        Vector<int> last (Dw);
        Fill(first,1);
        Fill(last,Dw);
        itsDw12=Dw12(Dw,1,first,last);
        int i2=1; //Linear index for (m,n) = 1+m+p*n
        //  Fill W^(m,n)_w matrices
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++,i2++)
            {
                itsWs(m+1,n+1)=MatrixRT(Dw,1);
                for (int w=1; w<=Dw; w++)
                    itsWs(m+1,n+1)(w,1)=sqrt(s(w,w))*U(w,i2); //U is actually VT
                //cout << "Right itsWs(" << m << "," << n << ") = " << itsWs(m+1,n+1) << endl;
                assert(itsWs(m+1,n+1).GetNumRows()==itsDw12.Dw1);
                assert(itsWs(m+1,n+1).GetNumCols()==itsDw12.Dw2);
            }
    }
    else
    {
        // Must be been called with one of the spin decomposition types.
        assert(false);
    }
}
//
// Construct with W operator
//
SiteOperatorImp::SiteOperatorImp(int d, const TensorT& W)
    : itsd(d)
    , itsDw12()
    , itsTruncationError(0.0)
    , itsWs(W)
{
    int Dw=itsWs(1,1).GetNumRows();
    assert(itsWs(1,1).GetNumCols()==Dw);
// Build up w limits
    Vector<int> first(Dw);
    Vector<int> last (Dw);
    Fill(first,1);
    Fill(last,Dw);
    itsDw12=Dw12(Dw,Dw,first,last);
}


SiteOperatorImp::~SiteOperatorImp()
{
    //dtor
}

void SiteOperatorImp::SetNeighbours(SiteOperator* left, SiteOperator* right)
{
    assert(left || right); //At least one needs to be non zero
    itsLeft_Neighbour=dynamic_cast<SiteOperatorImp*>(left);
    itsRightNeighbour=dynamic_cast<SiteOperatorImp*>(right);
    assert(!left  || itsLeft_Neighbour); //if left is nonzero then did the cast work?
    assert(!right || itsRightNeighbour);
}

void SiteOperatorImp::SetLimits()
{
    itsDw12.w1_first.SetLimits(itsDw12.Dw2);
    itsDw12.w2_last .SetLimits(itsDw12.Dw1);
    Fill(itsDw12.w1_first,1);
    Fill(itsDw12.w2_last ,itsDw12.Dw2);
//    Fill(itsDw12.w1_first,itsDw12.Dw1);
//    Fill(itsDw12.w2_last ,1);
//    for (int m=0; m<itsd; m++)
//        for (int n=0; n<itsd; n++)
//        {
//            const MatrixRT& W=GetW(m,n);
//            for (int w1=1; w1<=itsDw12.Dw1; w1++)
//                for (int w2=1; w2<=itsDw12.Dw2; w2++)
//                    if (W(w1,w2)!=0.0)
//                    {
//                        if (itsDw12.w1_first(w2)>w1) itsDw12.w1_first(w2)=w1;
//                        if (itsDw12.w2_last (w1)<w2) itsDw12.w2_last (w1)=w2;
//
//                    }
//            cout << "W(" << m << "," << n << ")=" << W << endl;
//            cout << "w1_first=" << itsDw12.w1_first << endl;
//            cout << "w2_last =" << itsDw12.w2_last  << endl;
//        }

}



void SiteOperatorImp::Combine(const SiteOperator* O2,double factor)
{
    //const SiteOperatorImp* O2=dynamic_cast<const SiteOperatorImp*>(_O2);
    //assert(O2);

    Dw12 O2Dw=O2->GetDw12();
    Dw12 Dw(itsDw12.Dw1*O2Dw.Dw1,itsDw12.Dw2*O2Dw.Dw2);

//    cout << "MPO D1,D2=" << itsDw12.Dw1 << " " << itsDw12.Dw2 << " ";
//    cout << "O2  D1,D2=" << O2Dw.Dw1 << " " << O2Dw.Dw2 << " ";
//    cout << "New D1,D2=" << Dw.Dw1 << " " << Dw.Dw2 << endl;


    TensorT newWs(itsd,itsd);
    for (int m=0; m<itsd; m++)
        for (int o=0; o<itsd; o++)
        {
            MatrixRT Wmo(Dw.Dw1,Dw.Dw2);
            Fill(Wmo,0.0);
            for (int n=0; n<itsd; n++)
            {
                const MatrixRT& W1=GetW(m,n);
                const MatrixRT& W2=O2->GetW(n,o);
                int w1=1;
                for (int w11=1; w11<=itsDw12.Dw1; w11++)
                    for (int w12=1; w12<=   O2Dw.Dw1; w12++,w1++)
                    {
                        int w2=1;
                        for (int w21=1; w21<=itsDw12.Dw2; w21++)
                            for (int w22=1; w22<=   O2Dw.Dw2; w22++,w2++)
                                Wmo(w1,w2)+=W1(w11,w21)*W2(w12,w22);
                    }
            }
            newWs(m+1,o+1)=factor*Wmo;
        }
    itsWs=newWs;
    itsDw12=Dw;
}


MatrixRT MakeBlockMatrix(const MatrixRT& M,int Dr,int Dc,int offset)
{
    MatrixRT ret(Dr,Dc);
    Fill(ret,0.0);
    if (offset==0)
        ret(Dr,Dc)=1.0;
    else
        ret(1,1)=1.0;

    int nr=M.GetNumRows();
    int nc=M.GetNumCols();
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
    int n=M.GetNumRows();
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
    int Dw1=itsDw12.Dw1,Dw2=itsDw12.Dw2;
    int X1=Dw1-2,X2=Dw2-2,Xs=X2; //Chi and Chi_prime

    MatrixRT  V=ReshapeV(lr);

    switch (lr)
    {
        case DLeft:
        {
            assert(V.GetNumRows()==itsd*itsd*(X1+1)); // Treate these like enforced comments on the
            assert(V.GetNumCols()==X2+1);             // dimensions of each matrix.
            auto [Qp,Lp]=QRsolver.SolveThinQL(V); //Solves V=Q*L
            cout << "Lp(X2+1,X2+1)=" << Lp(X2+1,X2+1) << endl;
            assert(fabs(Lp(X2+1,X2+1)-1.0)<1e-15);
            assert(Max(fabs(Qp*Lp-V))<1e-13);
            assert(Qp.GetNumRows()==itsd*itsd*(X1+1));
            assert(Qp.GetNumCols()==X2+1);
            assert(Lp.GetNumRows()==X2+1);
            assert(Lp.GetNumCols()==X2+1);
            assert(IsUnit(Transpose(Qp)*Qp,1e-13));

            MatrixRT Lpp;
            auto [M,Lprime]=ExtractM(lr,Lp);
            if (IsDiagonal(M,1e-14))
            {
                ReshapeV(lr,Qp);  //W is now Q
                Lpp=MakeBlockMatrix(Lp,X2+2,1);
            }
            else
            {
                auto [U,s,VT]=SVDsolver.SolveAll(M,1e-14); //Solves M=U * s * VT
                itsTruncationError=comp->Compress(U,s,VT);
                cout << std::fixed << "s=" << s.GetDiagonal() << endl;
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
            assert(fabs(Lp(1,1)-1.0)<1e-15);
//            cout << std::setprecision(1) << std::fixed << "Lp=" << Lp << endl;
            assert(Max(fabs(Lp*Qp-V))<1e-13);
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
                ReshapeV(lr,Qp);  //W is now Q
                Lpp=MakeBlockMatrix(Lp,X1+2,0);
            }
            else
            {
                auto [U,s,VT]=SVDsolver.SolveAll(M,1e-14); //Solves M=U * s * VT
                itsTruncationError=comp->Compress(U,s,VT);
                cout << std::fixed << "s=" << s.GetDiagonal() << endl;
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
    SetLimits();
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
//    cout << "s_avg, sm=" << s_avg << " " << sm << endl;
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
    SetLimits();
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
    SetLimits();
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
        if (N1>0 && N1!=itsDw12.Dw2)
        {
            if (GetW(0,0).GetNumCols()!=L.GetNumRows())
                Reshape(itsDw12.Dw1,N1,true);
            else
                itsDw12.Dw2=N1; //The contraction below will automatically reshape the Ws.
        }
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++)
            {
                MatrixRT& W=GetW(m,n);
                assert(W.GetNumCols()==L.GetNumRows());
                MatrixRT temp=W*L;
                W=temp; //Shallow copy
                assert(W.GetNumCols()==itsDw12.Dw2); //Verify shape is correct;
            }
        break;
    }
    case DLeft:
    {
        int N1=L.GetNumRows(); //N1=0 on the first site.
        if (N1>0 && N1!=itsDw12.Dw1)
        {
            if (GetW(0,0).GetNumRows()!=L.GetNumCols())
                Reshape(N1,itsDw12.Dw2,true);
            else
                itsDw12.Dw1=N1; //The contraction below will automatically reshape the As.
        }
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++)
            {
                MatrixRT& W=GetW(m,n);
                assert(L.GetNumCols()==W.GetNumRows());
                MatrixRT temp=L*W;
                W=temp; //Shallow copy
                assert(W.GetNumRows()==itsDw12.Dw1); //Verify shape is correct;
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
        if (N1>0 && N1!=itsDw12.Dw2)
        {
            if (GetW(0,0).GetNumCols()!=UV.GetNumRows())
                Reshape(itsDw12.Dw1,N1,true);
            else
                itsDw12.Dw2=N1; //The contraction below will automatically reshape the As.
        }
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++)
            {
                MatrixRT& W=GetW(m,n);
                assert(W.GetNumCols()==UV.GetNumRows());
                MatrixRT temp=W*UV*s;
                W.SetLimits(0,0);
                W=temp; //Shallow copy
//                cout << "SVD transfer W(" << m << "," << n << ")=" << W << endl;
                assert(W.GetNumCols()==itsDw12.Dw2); //Verify shape is correct;
            }
        break;
    }
    case DLeft:
    {
        int N1=s.GetNumRows(); //N1=0 on the first site.
        if (N1>0 && N1!=itsDw12.Dw1)
        {
            if (GetW(0,0).GetNumRows()!=UV.GetNumCols())
                Reshape(N1,itsDw12.Dw2,true);
            else
                itsDw12.Dw1=N1; //The contraction below will automatically reshape the As.
        }

        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++)
            {
                MatrixRT& W=GetW(m,n);
                assert(UV.GetNumCols()==W.GetNumRows());
                MatrixRT temp=s*UV*W;
                W.SetLimits(0,0);
                W=temp; //Shallow copy
//                cout << "SVD transfer W(" << m << "," << n << ")=" << W << endl;
                assert(W.GetNumRows()==itsDw12.Dw1); //Verify shape is correct;
            }
        break;
    }

    }
    SetLimits();
}

//
//  For LR = {Left/Right} we need to reshape with only {bottom right/top left} portion of matrix which
//  has the intrinsic portion of W.
//  In simple terms we just leave out the {first/last} row and column
//
MatrixRT SiteOperatorImp::ReshapeV(Direction lr) const
{
    MatrixRT A;
    switch (lr)
    {
    case DLeft:
    { //Leave out **first** row and column of W
        A.SetLimits(itsd*itsd*(itsDw12.Dw1-1),itsDw12.Dw2-1);
        int w=1;
        for (int w1=2; w1<=itsDw12.Dw1; w1++)
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++,w++)
            {
                const MatrixRT& W=GetW(m,n);
                for (int w2=2; w2<=itsDw12.Dw2; w2++)
                    A(w,w2-1)=W(w1,w2);
            }
        break;
    }
    case DRight:
    { //Leave out **last** row and column of W
        A.SetLimits(itsDw12.Dw1-1,itsd*itsd*(itsDw12.Dw2-1));
        int w=1;
        for (int w2=1; w2<=itsDw12.Dw2-1; w2++)
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++,w++)
            {
                const MatrixRT& W=GetW(m,n);
                for (int w1=1; w1<=itsDw12.Dw1-1; w1++)
                    A(w1,w)=W(w1,w2);
            }
        break;
    }
    }
    return A;
}
MatrixRT SiteOperatorImp::ReshapeV1(Direction lr) const
{
    MatrixRT A;
    switch (lr)
    {
    case DLeft:
    { //Leave out **first** row and column
        A.SetLimits(itsd*itsd*(itsDw12.Dw1-1),itsDw12.Dw2-1);
        int w=1;
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++)
            {
                const MatrixRT& W=GetW(m,n);
                for (int w1=2; w1<=itsDw12.Dw1; w1++,w++)
                    for (int w2=2; w2<=itsDw12.Dw2; w2++)
                        A(w,w2-1)=W(w1,w2);
            }
        break;
    }
    case DRight:
    { //Leave out **last** row and column
        A.SetLimits(itsDw12.Dw1-1,itsd*itsd*(itsDw12.Dw2-1));
        int w=1;
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++)
            {
                const MatrixRT& W=GetW(m,n);
                for (int w2=1; w2<=itsDw12.Dw2-1; w2++,w++)
                   for (int w1=1; w1<=itsDw12.Dw1-1; w1++)
                        A(w1,w)=W(w1,w2);
            }
        break;
    }
    }
    return A;
}

void  SiteOperatorImp::ReshapeV(Direction lr,const MatrixRT& Q)
{
    switch (lr)
    {
    case DLeft:
    {
        //  If L has less columns than the Ws then we need to reshape the whole site.
        //  Typically this will happen at the edges of the lattice.
        //
        if (Q.GetNumCols()+1<itsDw12.Dw2)
            Reshape(itsDw12.Dw1,Q.GetNumCols()+1,true);//we must save the old since Q only holds part of W
        //Leave out **first** row and column of W
        int w=1;
        for (int w1=2; w1<=itsDw12.Dw1; w1++)
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++,w++)
            {
                MatrixRT& W=GetW(m,n);
                for (int w2=2; w2<=itsDw12.Dw2; w2++)
                    W(w1,w2)=Q(w,w2-1);
            }
//        for (int m=0; m<itsd; m++)
//            for (int n=0; n<itsd; n++,w++)
//               cout << "Wnew(" << m << n << ")=" << GetW(m,n) << endl;
        break;
    }
    case DRight:
    {
        //  If Vdagger has less rows than the Ws then we need to reshape the whole site.
        //  Typically this will happen at the edges of the lattice.
        //
        if (Q.GetNumRows()+1<itsDw12.Dw1)
            Reshape(Q.GetNumRows()+1,itsDw12.Dw2,true);//we must save the old since Q only holds part of W
        //Leave out **last** row and column of W
        int w=1;
        for (int w2=1; w2<=itsDw12.Dw2-1; w2++)
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++,w++)
            {
                MatrixRT& W=GetW(m,n);
                for (int w1=1; w1<=itsDw12.Dw1-1; w1++)
                    W(w1,w2)=Q(w1,w);
            }
        break;
    }
    }
}
void  SiteOperatorImp::ReshapeV1(Direction lr,const MatrixRT& Q)
{
    switch (lr)
    {
    case DLeft:
    {
        //  If L has less columns than the Ws then we need to reshape the whole site.
        //  Typically this will happen at the edges of the lattice.
        //
        if (Q.GetNumCols()+1<itsDw12.Dw2) Reshape(itsDw12.Dw1,Q.GetNumCols()+1);//This throws away the old data
        int w=1;
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++)
            {
                MatrixRT& W=GetW(m,n);
                for (int w1=2; w1<=itsDw12.Dw1; w1++,w++)
                    for (int w2=2; w2<=itsDw12.Dw2; w2++)
                        W(w1,w2)=Q(w,w2-1);
            }
//        for (int m=0; m<itsd; m++)
//            for (int n=0; n<itsd; n++,w++)
//               cout << "Wnew(" << m << n << ")=" << GetW(m,n) << endl;
        break;
    }
    case DRight:
    {
        //  If Vdagger has less rows than the Ws then we need to reshape the whole site.
        //  Typically this will happen at the edges of the lattice.
        //
        if (Q.GetNumRows()+1<itsDw12.Dw1) Reshape(Q.GetNumRows(),itsDw12.Dw2+1,false);//This throws away the old data
        int w=1;
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++)
            {
                MatrixRT& W=GetW(m,n);
                for (int w2=1; w2<=itsDw12.Dw2-1; w2++,w++)
                    for (int w1=1; w1<=itsDw12.Dw1-1; w1++)
                        W(w1,w2)=Q(w1,w);
            }
        break;
    }
    }
}


MatrixRT SiteOperatorImp::Reshape(Direction lr) const
{
    MatrixRT A;
    switch (lr)
    {
    case DLeft:
    {
        A.SetLimits(itsd*itsd*itsDw12.Dw1,itsDw12.Dw2);
        int w=1;
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++)
            {
                const MatrixRT& W=GetW(m,n);
                for (int w1=1; w1<=itsDw12.Dw1; w1++,w++)
                    for (int w2=1; w2<=itsDw12.Dw2; w2++)
                        A(w,w2)=W(w1,w2);
            }
        break;
    }
    case DRight:
    {
        A.SetLimits(itsDw12.Dw1,itsd*itsd*itsDw12.Dw2);
        int w=1;
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++)
            {
                const MatrixRT& W=GetW(m,n);
                for (int w2=1; w2<=itsDw12.Dw2; w2++,w++)
                   for (int w1=1; w1<=itsDw12.Dw1; w1++)
                        A(w1,w)=W(w1,w2);
            }
        break;
    }
    }
    return A;
}

void SiteOperatorImp::Reshape(int D1, int D2, bool saveData)
{
    assert(D1>0);
    assert(D2>0);
    if (itsDw12.Dw1==D1 && itsDw12.Dw2==D2) return;
//    cout << "Reshape from " << itsDw12.Dw1 << "," << itsDw12.Dw2 << "   to " << D1 << "," << D2 << endl;
    itsDw12.Dw1=D1;
    itsDw12.Dw2=D2;
    for (int m=0; m<itsd; m++)
        for (int n=0; n<itsd; n++)
            GetW(m,n).SetLimits(itsDw12.Dw1,itsDw12.Dw2,saveData);

}

void  SiteOperatorImp::Reshape(Direction lr,const MatrixRT& UV)
{
    switch (lr)
    {
    case DLeft:
    {
        //  If U has less columns than the Ws then we need to reshape the whole site.
        //  Typically this will happen at the edges of the lattice.
        //
        if (UV.GetNumCols()<itsDw12.Dw2) Reshape(itsDw12.Dw1,UV.GetNumCols());//This throws away the old data
        int w=1;
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++)
            {
                MatrixRT& W=GetW(m,n);
                for (int w1=1; w1<=itsDw12.Dw1; w1++,w++)
                    for (int w2=1; w2<=itsDw12.Dw2; w2++)
                        W(w1,w2)=UV(w,w2);
            }
        break;
    }
    case DRight:
    {
        //  If Vdagger has less rows than the Ws then we need to reshape the whole site.
        //  Typically this will happen at the edges of the lattice.
        //
        if (UV.GetNumRows()<itsDw12.Dw1) Reshape(UV.GetNumRows(),itsDw12.Dw2,false);//This throws away the old data
        int w=1;
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++)
            {
                MatrixRT& W=GetW(m,n);
                for (int w2=1; w2<=itsDw12.Dw2; w2++,w++)
                    for (int w1=1; w1<=itsDw12.Dw1; w1++)
                        W(w1,w2)=UV(w1,w);
            }
        break;
    }
    }
}




void SiteOperatorImp::Report(std::ostream& os) const
{
    os << itsDw12.Dw1 << " " << itsDw12.Dw2 << " " << itsTruncationError
//    << " " << itsDw12.w1_first << " " << itsDw12.w2_last
    ;
}

char SiteOperatorImp::GetNormStatus(double eps) const
{
    char ret='W'; //Not normalized
    {
        MatrixRT QL=ReshapeV(DLeft);
        if (QL.GetNumRows()==0)
            ret='l';
        else if (IsUnit(Transpose(QL)*QL,eps))
            ret='L';
    }
    if (ret!='l')
    {
        MatrixRT QR=ReshapeV(DRight);
        if (QR.GetNumCols()==0)
            ret='r';
        else if (IsUnit(QR*Transpose(QR),eps))
        {
            if (ret=='L')
                ret='I';
            else
                ret='R';
        }
    }
    return ret;
}


} //namespace
//---------------------------------------------------------------------------------
//
//  Make template instance
//
#define TYPE Matrix<double>
#include "oml/src/matrix.cpp"
#undef TYPE
#define TYPE int
#include "oml/src/vector.cpp"
