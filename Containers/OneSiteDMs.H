#ifndef ONESITEDMS_H
#define ONESITEDMS_H

#include "oml/matrix.h"
#include "oml/vector.h"
#include <complex>
#include <vector>
//---------------------------------------------------------------------------------------
//
//  Store and manipulate 2-site density matricies
//
class OneSiteDMs
{
public:
    typedef std::complex<double> dcmplx;
    typedef Matrix<dcmplx>       DMType;
    typedef Matrix<double>      OperatorT;
    typedef Matrix<dcmplx>       OperatorCT;
    typedef Vector<double>       ExpectationT;

    OneSiteDMs(int L,int d);
    ~OneSiteDMs();
    void Insert(int ia, const DMType& dm);
    int    GetL() const {return itsL;}
    double GetS() const {return (itsd-1.0)/2.0;}

    template <class O> ExpectationT Contract(const O& op) const;
    ExpectationT GetTraces() const;
    ExpectationT GetVNEntropies() const;

private:
    typedef std::vector<DMType> DMsType;
    typedef Matrix<dcmplx> MatrixCT;

    int     itsL;
    int     itsd;
    DMsType itsDMs;
};

#endif // ONESITEDMS_H
