#include "Bond.H"
#include "oml/vector_io.h"
#include <iostream>

Bond::Bond(double eps)
    : itsEps(eps)
    , itsBondEntropy(0.0)
    , itsMinSV(0.0)
    , itsRank(0)
{
    assert(itsEps>=0.0);
}

Bond::~Bond()
{
    //dtor
}

void Bond::SetSingularValues(const Vector<double> s)
{
    //std::cout << "SingularValues=" << s << std::endl;
    itsSingularValues=s;
    int N=s.size();
    itsRank=N;
    itsMinSV=s(N);
    itsBondEntropy=0.0;
    for (int i=1;i<=N;i++)
    {
        double s2=s(i)*s(i);
        if (s2>0.0) itsBondEntropy-=s2*log(s2);
        if (fabs(s(i))<1e-12) itsRank--;
    }

}
