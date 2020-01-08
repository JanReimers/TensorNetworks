#include "MPOSite.H"
#include "Hamiltonian.H"
#include <complex>


MPOSite::MPOSite(const Operator* H,int p, int D1, int D2)
    : itsp(p)
    , itsD1(D1)
    , itsD2(D2)
    , itsDw1(H->GetDw())
    , itsDw2(H->GetDw())
    , itsWs(p,p)
{
    Fill(itsWs,MatrixT(itsD1,itsD2));
    Operator::Position lbr =
        itsD1==1 ? Operator::Left :
            itsD2==1 ? Operator::Right : Operator::Bulk;
    if (lbr==Operator::Left ) itsDw1=1;
    if (lbr==Operator::Right) itsDw2=1;

    for (int m=0;m<itsp;m++)
        for (int n=0;n<itsp;n++)
        {
            itsWs(m+1,n+1)=H->GetW(lbr,m,n);
            assert(itsWs(m+1,n+1).GetNumRows()==itsDw1);
            assert(itsWs(m+1,n+1).GetNumCols()==itsDw2);
        }
}

MPOSite::~MPOSite()
{
    //dtor
}

//---------------------------------------------------------------------------------
//
//  Make template instance
//
#define TYPE DMatrix<std::complex<double> >
#include "oml/src/dmatrix.cc"
