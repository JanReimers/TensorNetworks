#include "Bond.H"
#include "TensorNetworksImp/MPSSite.H"
#include "oml/vector_io.h"
#include "oml/diagonalmatrix.h"
#include <iostream>

Bond::Bond()
    : itsBondEntropy(0.0)
    , itsMinSV(0.0)
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

void Bond::SetSingularValues(const DiagonalMatrixT& s)
{
    //std::cout << "SingularValues=" << s << std::endl;
    int N=s.GetNumRows();
    itsSingularValues.SetLimits(N);
    itsSingularValues=s.GetDiagonal();

    itsRank=N;
    itsMinSV=itsSingularValues(N);
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
void Bond::SVDTransfer(TensorNetworks::Direction lr,const DiagonalMatrixT& s,const MatrixCT& UV)
{
    SetSingularValues(s);
    assert(GetSite(lr));
    GetSite(lr)->SVDTransfer(lr,s,UV);
}

void Bond::CanonicalTransfer(TensorNetworks::Direction lr,const DiagonalMatrixT& s,const MatrixCT& UV)
{
    SetSingularValues(s);
    assert(GetSite(lr));
    GetSite(lr)->SVDTransfer(lr,UV);
}
