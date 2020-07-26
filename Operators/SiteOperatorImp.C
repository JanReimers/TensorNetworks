#include "Operators/SiteOperatorImp.H"
#include "TensorNetworks/OperatorWRepresentation.H"

//
//  Build from a W rep opbject
//
SiteOperatorImp::SiteOperatorImp(TensorNetworks::Position lbr, const OperatorWRepresentation* H,int p)
    : itsp(p)
    , itsDw12(H->GetDw12(lbr))
    , itsWs(p,p)
{
    for (int m=0;m<itsp;m++)
        for (int n=0;n<itsp;n++)
        {
            itsWs(m+1,n+1)=H->GetW(lbr,m,n);
            assert(itsWs(m+1,n+1).GetNumRows()==itsDw12.Dw1);
            assert(itsWs(m+1,n+1).GetNumCols()==itsDw12.Dw2);
        }
}
//
// Build from a trotter decomp.
//
SiteOperatorImp::SiteOperatorImp(TensorNetworks::Direction lr,const MatrixT& U, const VectorT& expEvs, int p)
    : itsp(p)
    , itsDw12()
    , itsWs(p,p)
{
    int Dw=expEvs.size();
    if (lr==TensorNetworks::DLeft)
    {
        // Build up w limits
        Vector<int> first(Dw);
        Vector<int> last (1);
        Fill(first,1);
        Fill(last ,5);
        itsDw12=Dw12(1,Dw,first,last);
        int index=1; //Linear index for (m,n) = 1+m+p*n
        //  Fill W^(m,n)_w matrices
        for (int m=0;m<itsp;m++)
            for (int n=0;n<itsp;n++)
            {
                itsWs(m+1,n+1)=MatrixT(1,Dw);
                for (int w=1;w<=Dw;w++)
                    itsWs(m+1,n+1)(1,w)=U(index,w)*expEvs(w);
                //cout << "Left itsWs(" << m << "," << n << ") = " << itsWs(m+1,n+1) << endl;
                assert(itsWs(m+1,n+1).GetNumRows()==itsDw12.Dw1);
                assert(itsWs(m+1,n+1).GetNumCols()==itsDw12.Dw2);
                index++;
            }
    }
    else if (lr==TensorNetworks::DRight)
    {
        // Build up w limits
        Vector<int> first(1);
        Vector<int> last (Dw);
        Fill(first,5);
        Fill(last ,1);
        itsDw12=Dw12(Dw,1,first,last);
        int index=1; //Linear index for (m,n) = 1+m+p*n
        //  Fill W^(m,n)_w matrices
        for (int m=0;m<itsp;m++)
            for (int n=0;n<itsp;n++)
            {
                itsWs(m+1,n+1)=MatrixT(Dw,1);
                for (int w=1;w<=Dw;w++)
                    itsWs(m+1,n+1)(w,1)=U(index,w)*expEvs(w);
                //cout << "Right itsWs(" << m << "," << n << ") = " << itsWs(m+1,n+1) << endl;
                assert(itsWs(m+1,n+1).GetNumRows()==itsDw12.Dw1);
                assert(itsWs(m+1,n+1).GetNumCols()==itsDw12.Dw2);
                index++;
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

//---------------------------------------------------------------------------------
//
//  Make template instance
//
#define TYPE DMatrix<std::complex<double> >
#include "oml/src/dmatrix.cc"
