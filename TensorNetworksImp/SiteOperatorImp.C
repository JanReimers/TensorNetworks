#include "TensorNetworksImp/SiteOperatorImp.H"
#include "TensorNetworks/OperatorWRepresentation.H"


SiteOperatorImp::SiteOperatorImp(TensorNetworks::Position lbr, const OperatorWRepresentation* H,int p)
    : itsp(p)
    , itsDws(H->GetDw(lbr))
    , itsWs(p,p)
{
    for (int m=0;m<itsp;m++)
        for (int n=0;n<itsp;n++)
        {
            itsWs(m+1,n+1)=H->GetW(lbr,m,n);
            assert(itsWs(m+1,n+1).GetNumRows()==itsDws.first);
            assert(itsWs(m+1,n+1).GetNumCols()==itsDws.second);
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
