#include "MPOSite.H"

#include <complex>


MPOSite::MPOSite(int p, int D1, int D2)
    : itsp(p)
    , itsD1(D1)
    , itsD2(D2)
    , itsWs(p,p)
{
    Fill(itsWs,MatrixT(itsD1,itsD2));
    //ctor
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
