#include "Operators/SiteOperatorImp.H"
#include "TensorNetworks/SVCompressor.H"
#include "TensorNetworksImp/SpinCalculator.H"
#include "NumericalMethods/LapackSVD.H"
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
        Fill(last,5);
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
        Fill(last,5);
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

void SiteOperatorImp::Compress(Direction lr,const SVCompressorR* comp)
{
    assert(comp);
    MatrixRT  A=Reshape(lr);
    LapackSVDSolver<double> solver;
    auto [U,sm,VT]=solver.SolveAll(A,1e-14); //Solves A=U * s * VT
    //
    //  Rescaling
    //
    double s_avg=Sum(sm.GetDiagonal())/sm.size();
//    cout << "s_avg, sm=" << s_avg << " " << sm << endl;
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

}

MatrixRT SiteOperatorImp::Reshape(Direction lr)
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

}

void SiteOperatorImp::Report(std::ostream& os) const
{
    os << itsDw12.Dw1 << " " << itsDw12.Dw2 << " " << itsTruncationError;
}

} //namespace
//---------------------------------------------------------------------------------
//
//  Make template instance
//
#define TYPE Matrix<double>
#include "oml/src/matrix.cpp"
