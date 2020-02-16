#include "Bond.H"
#include "TensorNetworksImp/MatrixProductSite.H"
#include "oml/vector_io.h"
#include <iostream>

Bond::Bond(double eps)
    : itsEps(eps)
    , itsBondEntropy(0.0)
    , itsMinSV(0.0)
    , itsRank(0)
    , itsLeft_Site(0)
    , itsRightSite(0)
{
    assert(itsEps>=0.0);
}

Bond::~Bond()
{
    //dtor
}

void Bond::SetSites(MatrixProductSite* left, MatrixProductSite* right)
{
    itsLeft_Site=left;
    itsRightSite=right;
    assert(itsLeft_Site);
    assert(itsRightSite);
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
        if (fabs(s(i))<1e-12) itsRank--;
    }

}

void Bond::SVDTransfer(TensorNetworks::Direction lr,const VectorT& s,const MatrixCT& UV)
{
    SetSingularValues(s);
    switch(lr)
    {
        case TensorNetworks::DLeft:
        {
            assert(itsLeft_Site);
            itsLeft_Site->Contract(TensorNetworks::DRight,s,UV);
            break;
        }
        case TensorNetworks::DRight:
        {
            itsRightSite->Contract(TensorNetworks::DLeft,s,UV);
            break;
        }

    }
}
