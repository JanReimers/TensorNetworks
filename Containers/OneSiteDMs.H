#ifndef ONESITEDMS_H
#define ONESITEDMS_H

#include "oml/dmatrix.h"
#include "oml/vector.h"
#include <complex>
//---------------------------------------------------------------------------------------
//
//  Store and manipulate 2-site density matricies
//
class OneSiteDMs
{
public:
    typedef std::complex<double> dcmplx;
    typedef DMatrix<dcmplx>       DMType;
    typedef DMatrix<double>      OperatorT;
    typedef DMatrix<dcmplx>       OperatorCT;
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
    typedef Vector<DMType> DMsType;
    typedef DMatrix<dcmplx> MatrixCT;

    int     itsL;
    int     itsd;
    DMsType itsDMs;
};

#endif // ONESITEDMS_H
