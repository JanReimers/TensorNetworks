#ifndef ONESITEDMS_H
#define ONESITEDMS_H

#include "oml/dmatrix.h"
#include "oml/smatrix.h"
#include "oml/array.h"
#include <complex>
//---------------------------------------------------------------------------------------
//
//  Store and manipulate 2-site density matricies
//
class OneSiteDMs
{
public:
    typedef std::complex<double> eType;
    typedef DMatrix<eType> DMType;
    typedef DMatrix<double> OperatorT;
    typedef DMatrix<eType>  OperatorCT;
    typedef Array<double> ExpectationT;

    OneSiteDMs(int L,int p);
    ~OneSiteDMs();
    void Insert(int ia, const DMType& dm);
    int    GetL() const {return itsL;}
    double GetS() const {return (itsp-1.0)/2.0;}

    template <class O> ExpectationT Contract(const O& op) const;
    ExpectationT GetTraces() const;
    ExpectationT GetVNEntropies() const;

private:
    typedef Array<DMType> DMsType;
    typedef DMatrix<eType> MatrixCT;

    int    itsL;
    int    itsp;
    DMsType itsDMs;
};

#endif // ONESITEDMS_H
