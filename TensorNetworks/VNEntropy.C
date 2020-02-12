
#include "TensorNetworks/VNEntropy.H"
#include "oml/vector.h"
#include <iostream>

double VNEntropyFromEVs(const Vector<double>& s)
{
    static double eps=1e-14;
    int N=s.size();
    double ret=0.0;
    for (int i=1;i<=N;i++)
    {
        assert(s(i)>=-eps);
        if (s(i)>1.0) std::cerr << "Warning VNEntropyFromEVs S(i)-1=" << s(i)-1 << std::endl;
        assert(s(i)<=1.0+1000*eps);  //This check is really tough to satisfy
        if (s(i)>0.0) ret+=s(i)*std::log(s(i));
    }
    return -ret;
}

double VNEntropyFromSVs(const Vector<double>& s)
{
    static double eps=1e-14;
    int N=s.size();
    double ret=0.0;
    for (int i=1;i<=N;i++)
    {
        assert(s(i)>=-eps);
        if (s(i)>1.0) std::cerr << "Warning VNEntropyFromSVs S(i)-1=" << s(i)-1 << std::endl;
        assert(s(i)<=1.0+1000*eps);  //This check is really tough to satisfy
        if (s(i)>0.0) ret+=s(i)*s(i)*std::log(s(i)*s(i));
    }
    return -ret;
}

