#include "Bond.H"
#include "TensorNetworksImp/MPSSite.H"
#include "oml/vector_io.h"
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
    itsSingularValues.SetSize(D,true);
    for (int i=itsRank;i<D;i++)
        itsSingularValues[i]=0.0;
    itsRank=D;
}

void Bond::SetSingularValues(const VectorT& s)
{
    //std::cout << "SingularValues=" << s << std::endl;
    int N=s.size();
    itsSingularValues.SetSize(N);

    itsRank=N;
    itsMinSV=s(N);
    itsBondEntropy=0.0;
    for (int i=1;i<=N;i++)
    {
        itsSingularValues[i-1]=log10(s(i)); //Transload from vector to array. OML shouldreally do this automatically with a shallow copy.
        double s2=s(i)*s(i);
        if (s2>0.0) itsBondEntropy-=s2*log(s2);
        if (fabs(s(i))<1e-12)
        {
            //cout << "Auto rank reduction s=" << s << endl;
            itsRank--;
        }
    }

}

//
//  Direction is the normaliztions direction, which i opposite to the direction that UV gets tranferred.
//
void Bond::SVDTransfer(TensorNetworks::Direction lr,const VectorT& s,const MatrixCT& UV)
{
    SetSingularValues(s);
    assert(GetSite(lr));
    GetSite(lr)->Contract(lr,s,UV);
}

void Bond::CanonicalTransfer(TensorNetworks::Direction lr,const VectorT& s,const MatrixCT& UV)
{
    SetSingularValues(s);
    assert(GetSite(lr));
    GetSite(lr)->Contract(lr,UV);
}
