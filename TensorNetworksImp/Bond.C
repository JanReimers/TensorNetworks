#include "Bond.H"
#include "TensorNetworksImp/MPSSite.H"
#include "oml/vector_io.h"
#include "oml/diagonalmatrix.h"
#include <iostream>

namespace TensorNetworks
{

Bond::Bond()
    : itsBondEntropy(0.0)
    , itsMinSV(0.0)
    , itsIntegratedS2(-99)
    , itsRank(0)
    , itsLeft_Site(0)
    , itsRightSite(0)
{
}

Bond::~Bond()
{
    //dtor
}

void Bond::CloneState(const Bond* b2)
{
    itsSingularValues=b2->itsSingularValues;
    itsBondEntropy   =b2->itsBondEntropy;
    itsMinSV         =b2->itsMinSV;
    itsRank          =b2->itsRank;
}

void Bond::SetSites(MPSSite* left, MPSSite* right)
{
    itsLeft_Site=left;
    itsRightSite=right;
    assert(itsLeft_Site);
    assert(itsRightSite);
}

void Bond::NewBondDimension(int D)
{
    assert(D>=1);
    assert(itsRank==itsSingularValues.size());
    itsSingularValues.SetLimits(D,true);
    for (int i=itsRank+1;i<=D;i++)
        itsSingularValues(i)=0.0;
    itsRank=D;
}

void Bond::SetSingularValues(const DiagonalMatrixRT& s,double integratedS2)
{
    //std::cout << "SingularValues=" << s << std::endl;
    int N=s.GetNumRows();
    itsSingularValues.SetLimits(N);
    itsSingularValues=s.GetDiagonal();

    itsRank=N;
    itsMinSV=itsSingularValues(N);
    itsIntegratedS2=integratedS2;

    itsBondEntropy=0.0;
    for (int i=1;i<=N;i++)
    {
        double s2=itsSingularValues(i)*itsSingularValues(i);
        if (s2>0.0) itsBondEntropy-=s2*log(s2);
        if (fabs(itsSingularValues(i))<1e-12)
        {
            //cout << "Auto rank reduction s=" << s << endl;
            itsRank--;
        }
    }

}

//
//  Direction is the normaliztions direction, which i opposite to the direction that UV gets tranferred.
//
void Bond::SVDTransfer(Direction lr,double integratedS2,const DiagonalMatrixRT& s,const MatrixCT& UV)
{
    SetSingularValues(s,integratedS2);
    assert(GetSite(lr));
    GetSite(lr)->SVDTransfer(lr,s,UV);
}

void Bond::CanonicalTransfer(Direction lr,double integratedS2,const DiagonalMatrixRT& s,const MatrixCT& UV)
{
    SetSingularValues(s,integratedS2);
    assert(GetSite(lr));
    GetSite(lr)->SVDTransfer(lr,UV);
}

void Bond::Report(std::ostream& os) const
{
    os
                                                  << std::setw(4) << itsRank
       << std::fixed      << std::setprecision(6) << std::setw(12) << itsBondEntropy
       << std::scientific << std::setprecision(1) << std::setw(10) << itsMinSV
       << std::scientific << std::setprecision(1) << std::setw(10) << itsIntegratedS2
       ;

}


} //namespace
