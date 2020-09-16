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
SiteOperatorImp::SiteOperatorImp(TensorNetworks::Direction lr,const MatrixT& U, const VectorT& s, int d)
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
                itsWs(m+1,n+1)=MatrixT(1,Dw);
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
                itsWs(m+1,n+1)=MatrixT(Dw,1);
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
            MatrixT Wmo(Dw.Dw1,Dw.Dw2);
            Fill(Wmo,0.0);
            for (int n=0; n<itsd; n++)
            {
                const MatrixT& W1=GetW(m,n);
                const MatrixT& W2=O2->GetW(n,o);
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
using TensorNetworks::VectorT;

void SiteOperatorImp::Compress(TensorNetworks::Direction lr,int DwMax, double sMin)
{
    //assert(sMin>=0.0);
    //  If DwMax==0 then only use sMin.  If sMin==0.0 then only use DwMax
    //
    assert(DwMax>=0);
    assert(sMin>=0.0);
    assert(!(DwMax==0 && sMin==0.0));

    MatrixT  A=Reshape(lr);
    MatrixT  Acopy=A;
    int      M=A.GetNumRows(),N=A.GetNumCols();
    int      mn=Min(M,N);
    VectorT  s(mn);
    MatrixT  VT(N,N);
//    cout << "Before Compress Dw1 Dw2 A=" << itsDw12.Dw1 << " " << itsDw12.Dw2 << " "<< A << endl;
    LaSVDecomp(A,s,VT); //Solves A=U * s * Vdagger  returns V not Vdagger
    //if (V.GetNumCols()!=s.size())
    //{
    //    MatrixT VT=Transpose(V);
    //    V.SetLimits(0,0);
    //    V=VT;
   // }


 //   cout << "SVD U s V =" << A << s << VT << endl;
//    cout << "U*s*Trans(V)=" << MatrixT(Contract1(A,s)*VT) << endl;
//    cout << "U*s*Trans(V)=" << MatrixT(A*Contract1(s,VT))<< endl;
//    cout << "U*s*Trans(V)=" << Max(abs(MatrixT(Contract1(A,s)*VT-Acopy))) << endl;
//    cout << "U*s*Trans(V)=" << Max(abs(MatrixT(A*Contract1(s,VT)-Acopy))) << endl;
    assert(Max(abs(MatrixT(Contract1(A,s)*VT-Acopy)))<1e-12);
    //
    //  Rescaling
    //
    double s_avg=Sum(s)/s.size();
    s*=1.0/s_avg;
     switch (lr)
    {
        case TensorNetworks::DLeft:
            A*=s_avg;
            break;
        case TensorNetworks::DRight:
            VT*=s_avg;
            break;
    }

    // At this point we have N singular values but we only Dmax of them or only the ones >=epsMin;
    int D=DwMax>0 ? Min(mn,DwMax) : mn; //Ignore Dmax if it is 0
    // Shrink so that all s(is<=D)>=epsMin;
    for (int is=D; is>=1; is--)
        if (s(is)>=sMin)
        {
            D=is;
            break;
        }
    if (D<s.size())
    {
//        cout << "Smin=" << s(D) << "  Sum of rejected singular values=" << Sum(s.SubVector(D+1,s.size())) << endl;
//        cout << "S=" << s << endl;
    }
    double Sums=Sum(s);
    assert(Sums>0.0);
    s.SetLimits(D,true);  // Resize s
    A.SetLimits(A.GetNumRows(),D,true);
    VT.SetLimits(D,VT.GetNumCols(),true);
    assert(Sum(s)>0.0);
    double rescaleS=Sums/Sum(s);
    s*=rescaleS;

//    cout << "After Compress U s V =" << A << s << VT << endl;
//    cout << "U*s*Trans(V)=" << Max(abs(MatrixT(Contract1(A,s)*VT-Acopy))) << endl;
//    cout << "U*s*Trans(V)=" << Max(abs(MatrixT(A*Contract1(s,VT)-Acopy))) << endl;
    assert(Max(abs(MatrixT(Contract1(A,s)*VT-Acopy)))<10*sMin);

    MatrixT UV;// This get transferred through the bond to a neighbouring site.
    switch (lr)
    {
        case TensorNetworks::DRight:
        {
            UV=A;

            //cout << "After compress V=" << " "<< V << endl;
            Reshape(lr,VT);  //A is now Vdagger
            if (itsLeft_Neighbour) itsLeft_Neighbour->SVDTransfer(lr,s,UV);
            break;
        }
        case TensorNetworks::DLeft:
        {
            UV=VT; //Set Vdagger
//            cout << "After compress A=" << " "<< A << endl;
            Reshape(lr,A);  //A is now U
            if (itsRightNeighbour) itsRightNeighbour->SVDTransfer(lr,s,UV);
            break;
        }
    }

}

TensorNetworks::MatrixT SiteOperatorImp::Reshape(TensorNetworks::Direction lr)
{
    MatrixT A;
    switch (lr)
    {
    case TensorNetworks::DLeft:
    {
        A.SetLimits(itsd*itsd*itsDw12.Dw1,itsDw12.Dw2);
        int w=1;
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++)
            {
                const MatrixT& W=GetW(m,n);
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
                const MatrixT& W=GetW(m,n);
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

void  SiteOperatorImp::Reshape(TensorNetworks::Direction lr,const MatrixT& UV)
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
                MatrixT& W=GetW(m,n);
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
                MatrixT& W=GetW(m,n);
                for (int w2=1; w2<=itsDw12.Dw2; w2++,w++)
                    for (int w1=1; w1<=itsDw12.Dw1; w1++)
                        W(w1,w2)=UV(w1,w);
            }
        break;
    }
    }
}

void SiteOperatorImp::SVDTransfer(TensorNetworks::Direction lr,const VectorT& s,const MatrixT& UV)
{
//    cout << "SVD transfer s=" << s << " UV=" << UV << endl;
    switch (lr)
    {
    case TensorNetworks::DRight:
    {
        int N1=s.GetHigh(); //N1=0 on the first site.
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
                MatrixT& W=GetW(m,n);
                assert(W.GetNumCols()==UV.GetNumRows());
                MatrixT temp=Contract1(W*UV,s);
                W.SetLimits(0,0);
                W=temp; //Shallow copy
//                cout << "SVD transfer W(" << m << "," << n << ")=" << W << endl;
                assert(W.GetNumCols()==itsDw12.Dw2); //Verify shape is correct;
            }
        break;
    }
    case TensorNetworks::DLeft:
    {
        int N1=s.GetHigh(); //N1=0 on the first site.
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
                MatrixT& W=GetW(m,n);
                assert(UV.GetNumCols()==W.GetNumRows());
                MatrixT temp=Contract1(s,UV*W);
                W.SetLimits(0,0);
                W=temp; //Shallow copy
//                cout << "SVD transfer W(" << m << "," << n << ")=" << W << endl;
                assert(W.GetNumRows()==itsDw12.Dw1); //Verify shape is correct;
            }
        break;
    }

    }

}

//
//  Anew(j,i) =  s(j)*VA(j,k)
//
SiteOperatorImp::MatrixT SiteOperatorImp::Contract1(const VectorT& s, const MatrixT& VA)
{
    int N1=VA.GetNumRows();
    int N2=VA.GetNumCols();
    assert(s.GetHigh()==N1);

    MatrixT Anew(N1,N2);
    for(int i2=1; i2<=N2; i2++)
        for(int i1=1; i1<=N1; i1++)
            Anew(i1,i2)=s(i1)*VA(i1,i2);

    return Anew;
}

//
//  Anew(j,i) =  s(j)*VA(j,k)
//
SiteOperatorImp::MatrixT SiteOperatorImp::Contract1(const MatrixT& AU,const VectorT& s)
{
    int N1=AU.GetNumRows();
    int N2=AU.GetNumCols();
    assert(s.GetHigh()==N2);

    MatrixT Anew(N1,N2);
    for(int i2=1; i2<=N2; i2++)
        for(int i1=1; i1<=N1; i1++)
            Anew(i1,i2)=AU(i1,i2)*s(i2);

    return Anew;
}

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
