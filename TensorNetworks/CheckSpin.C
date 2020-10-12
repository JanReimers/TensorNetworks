#include "TensorNetworks/CheckSpin.H"
#include <cmath>

bool isValidSpin(double S)
{
    double ipart;
    double frac=std::modf(2.0*S,&ipart);
    return frac==0.0 && (S>=0.5);

}
