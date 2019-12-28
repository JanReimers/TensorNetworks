#include "MPOSite.H"
#include "Hamiltonian.H"
#include <complex>


MPOSite::MPOSite(const Hamiltonian* H,int p, int D1, int D2)
    : itsp(p)
    , itsD1(D1)
    , itsD2(D2)
    , itsWs(p,p)
{
    Fill(itsWs,MatrixT(itsD1,itsD2));
    Hamiltonian::Position lbr =
        itsD1==1 ? Hamiltonian::Left :
            itsD2==1 ? Hamiltonian::Right : Hamiltonian::Bulk;
    for (int m=0;m<itsp;m++)
        for (int n=0;n<itsp;n++)
            itsWs(m+1,n+1)=H->GetW(lbr,m,n);
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
