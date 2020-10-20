#include "Bond.H"
#include "TensorNetworksImp/MPSSite.H"
#include "oml/diagonalmatrix.h"
#include <iostream>
#include <iomanip>

namespace TensorNetworks
{

Bond::Bond(double epsSV)
    : itsEpsSV(epsSV)
    , itsBondEntropy(0.0)
    , itsMinSV(0.0)
    , itsIntegratedS2(-99)
    , itsD(0)
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
    itsEpsSV         =b2->itsEpsSV;
    itsBondEntropy   =b2->itsBondEntropy;
    itsIntegratedS2  =b2->itsIntegratedS2;
    itsMinSV         =b2->itsMinSV;
    itsD             =b2->itsD;
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
    assert(itsD==itsSingularValues.size());
    itsSingularValues.SetLimits(D,true);
    for (int i=itsD+1;i<=D;i++)
        itsSingularValues(i)=0.0;
    itsD=D;
}

void Bond::SetSingularValues(const DiagonalMatrixRT& s,double integratedS2)
{
    //std::cout << "SingularValues=" << s << std::endl;
    itsD=s.GetNumRows();
    itsSingularValues.SetLimits(itsD);
    itsSingularValues=s.GetDiagonal();

    itsRank=itsD;
    itsMinSV=itsSingularValues(itsD);
    itsIntegratedS2=integratedS2;

    itsBondEntropy=0.0;
    for (int i=1;i<=itsD;i++)
    {
        double s2=itsSingularValues(i)*itsSingularValues(i);
        if (s2>0.0) itsBondEntropy-=s2*log(s2);
        if (fabs(itsSingularValues(i))<itsEpsSV)
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
                                                  << std::setw(4)  << itsD
                                                  << std::setw(4)  << itsRank
       << std::fixed      << std::setprecision(6) << std::setw(12) << itsBondEntropy
       << std::scientific << std::setprecision(1) << std::setw(10) << itsMinSV
       << std::scientific << std::setprecision(1) << std::setw(10) << itsIntegratedS2
       ;

}


} //namespace
