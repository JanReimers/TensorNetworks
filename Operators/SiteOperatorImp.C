#include "Operators/SiteOperatorImp.H"
#include "TensorNetworks/OperatorWRepresentation.H"
#include "TensorNetworks/IterationSchedule.H"
#include "oml/minmax.h"
#include "NumericalMethods/LapackSVD.H"
#include <complex>

//
//  Build from a W rep opbject
//
SiteOperatorImp::SiteOperatorImp(TensorNetworks::Position lbr, const OperatorWRepresentation* H,int d)
    : itsd(d)
    , itsDw12(H->GetDw12(lbr))
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
SiteOperatorImp::SiteOperatorImp(TensorNetworks::Direction lr,const MatrixRT& U, const VectorRT& s, int d)
    : itsd(d)
    , itsDw12()
    , itsWs(d,d)
{
    int Dw=s.size();
    if (lr==TensorNetworks::DLeft)
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
                    itsWs(m+1,n+1)(1,w)=U(i1,w)*sqrt(s(w));
                //cout << "Left itsWs(" << m << "," << n << ") = " << itsWs(m+1,n+1) << endl;
                assert(itsWs(m+1,n+1).GetNumRows()==itsDw12.Dw1);
                assert(itsWs(m+1,n+1).GetNumCols()==itsDw12.Dw2);
            }
    }
    else if (lr==TensorNetworks::DRight)
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
                    itsWs(m+1,n+1)(w,1)=U(i2,w)*sqrt(s(w));
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


void SiteOperatorImp::Combine(const SiteOperator* O2)
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
            newWs(m+1,o+1)=Wmo;
        }
    itsWs=newWs;
    itsDw12=Dw;
}

using TensorNetworks::MatrixCT;
using TensorNetworks::VectorRT;

void SiteOperatorImp::Compress(TensorNetworks::Direction lr,SVCompressorR* comp)
{
    assert(comp);
    MatrixRT  A=Reshape(lr);

    auto [U,sm,VT]=LaSVDecomp(A); //Solves A=U * s * Vdagger  returns V not Vdagger
//    VectorRT s=sm.GetDiagonal();
//    cout << "error1=" << std::scientific << Max(abs(MatrixRT(U*sm*VT-A))) << endl;
//    assert(Max(abs(MatrixRT(U*sm*VT-A)))<1e-10);
    //
    //  Rescaling
    //
    double s_avg=Sum(sm.GetDiagonal())/sm.size();
    sm*=1.0/s_avg;
     switch (lr)
    {
        case TensorNetworks::DLeft:
            U*=s_avg;
            break;
        case TensorNetworks::DRight:
            VT*=s_avg;
            break;
    }

    comp->Compress(U,sm,VT);
//    cout << "error2=" << Max(abs(MatrixRT(U*sm*VT-A))) << endl;
//    s=sm.GetDiagonal();
    MatrixRT UV;// This get transferred through the bond to a neighbouring site.
    switch (lr)
    {
        case TensorNetworks::DRight:
        {
            UV=U;

            //cout << "After compress V=" << " "<< V << endl;
            Reshape(lr,VT);  //A is now Vdagger
            if (itsLeft_Neighbour) itsLeft_Neighbour->SVDTransfer(lr,sm,UV);
            break;
        }
        case TensorNetworks::DLeft:
        {
            UV=VT; //Set Vdagger
//            cout << "After compress A=" << " "<< A << endl;
            Reshape(lr,U);  //A is now U
            if (itsRightNeighbour) itsRightNeighbour->SVDTransfer(lr,sm,UV);
            break;
        }
    }

}

TensorNetworks::MatrixRT SiteOperatorImp::Reshape(TensorNetworks::Direction lr)
{
    MatrixRT A;
    switch (lr)
    {
    case TensorNetworks::DLeft:
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
    case TensorNetworks::DRight:
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

void  SiteOperatorImp::Reshape(TensorNetworks::Direction lr,const MatrixRT& UV)
{
    switch (lr)
    {
    case TensorNetworks::DLeft:
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
    case TensorNetworks::DRight:
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

void SiteOperatorImp::SVDTransfer(TensorNetworks::Direction lr,const DiagonalMatrixRT& s,const MatrixRT& UV)
{
//    cout << "SVD transfer s=" << s << " UV=" << UV << endl;
    switch (lr)
    {
    case TensorNetworks::DRight:
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
    case TensorNetworks::DLeft:
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

/*//
//  Anew(j,i) =  s(j)*VA(j,k)
//
SiteOperatorImp::MatrixRT SiteOperatorImp::Contract1(const VectorRT& s, const MatrixRT& VA)
{
    int N1=VA.GetNumRows();
    int N2=VA.GetNumCols();
    assert(s.GetHigh()==N1);

    MatrixRT Anew(N1,N2);
    for(int i2=1; i2<=N2; i2++)
        for(int i1=1; i1<=N1; i1++)
            Anew(i1,i2)=s(i1)*VA(i1,i2);

    return Anew;
}

//
//  Anew(j,i) =  s(j)*VA(j,k)
//
SiteOperatorImp::MatrixRT SiteOperatorImp::Contract1(const MatrixRT& AU,const VectorRT& s)
{
    int N1=AU.GetNumRows();
    int N2=AU.GetNumCols();
    assert(s.GetHigh()==N2);

    MatrixRT Anew(N1,N2);
    for(int i2=1; i2<=N2; i2++)
        for(int i1=1; i1<=N1; i1++)
            Anew(i1,i2)=AU(i1,i2)*s(i2);

    return Anew;
}
*/
void SiteOperatorImp::Report(std::ostream& os) const
{
    os << itsDw12.Dw1 << " " << itsDw12.Dw2;
}
//---------------------------------------------------------------------------------
//
//  Make template instance
//
#define TYPE DMatrix<double>
#include "oml/src/dmatrix.cc"
