#include "Operators/SiteOperatorImp.H"
#include "TensorNetworks/OperatorWRepresentation.H"
#include "oml/cnumeric.h"
#include <complex>

//
//  Build from a W rep opbject
//
SiteOperatorImp::SiteOperatorImp(TensorNetworks::Position lbr, const OperatorWRepresentation* H,int p)
    : itsp(p)
    , itsDw12(H->GetDw12(lbr))
    , itsWs(p,p)
{
    for (int m=0; m<itsp; m++)
        for (int n=0; n<itsp; n++)
        {
            itsWs(m+1,n+1)=H->GetW(lbr,m,n);
            assert(itsWs(m+1,n+1).GetNumRows()==itsDw12.Dw1);
            assert(itsWs(m+1,n+1).GetNumCols()==itsDw12.Dw2);
        }
}
//
// Build from a trotter decomp.
//
SiteOperatorImp::SiteOperatorImp(TensorNetworks::Direction lr,const MatrixT& U, const VectorT& s, int p)
    : itsp(p)
    , itsDw12()
    , itsWs(p,p)
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
        for (int m=0; m<itsp; m++)
            for (int n=0; n<itsp; n++,i1++)
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
        for (int m=0; m<itsp; m++)
            for (int n=0; n<itsp; n++,i2++)
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

void SiteOperatorImp::Combine(const SiteOperator* O2)
{
    //const SiteOperatorImp* O2=dynamic_cast<const SiteOperatorImp*>(_O2);
    //assert(O2);

    Dw12 O2Dw=O2->GetDw12();
    Dw12 Dw(itsDw12.Dw1*O2Dw.Dw1,itsDw12.Dw2*O2Dw.Dw2);

//    cout << "MPO D1,D2=" << itsDw12.Dw1 << " " << itsDw12.Dw2 << " ";
//    cout << "O2  D1,D2=" << O2Dw.Dw1 << " " << O2Dw.Dw2 << " ";
//    cout << "New D1,D2=" << Dw.Dw1 << " " << Dw.Dw2 << endl;


    TensorT newWs(itsp,itsp);
    for (int m=0; m<itsp; m++)
        for (int o=0; o<itsp; o++)
        {
            MatrixT Wmo(Dw.Dw1,Dw.Dw2);
            Fill(Wmo,0.0);
            for (int n=0; n<itsp; n++)
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
            /*{
                TensorNetworks::MatrixCT A(Wmo.GetLimits());
                for (int w1=1;w1<=Dw.Dw1;w1++)
                    for (int w2=1;w2<=Dw.Dw2;w2++)
                        A(w1,w2)=Wmo(w1,w2);
                int N=Min(Wmo.GetNumRows(),Wmo.GetNumCols());
                //int N=Wmo.GetNumCols();
                VectorT s(N);
                TensorNetworks::MatrixCT V(N,Wmo.GetNumCols());
                CSVDecomp(A,s,V); //So
                cout << "m,o s=" << m << " " << o << " " << s << endl;
            }*/
        }
//    cout.precision(3);
//    cout << std::fixed << "newWs=" << newWs << endl;
    itsWs=newWs;
    itsDw12=Dw;
}

//---------------------------------------------------------------------------------
//
//  Make template instance
//
#define TYPE DMatrix<std::complex<double> >
#include "oml/src/dmatrix.cc"
