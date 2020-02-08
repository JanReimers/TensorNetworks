#ifndef TWOSITEDMS_H
#define TWOSITEDMS_H

#include "Containers/Matrix4.H"
#include "oml/smatrix.h"
#include <complex>
//---------------------------------------------------------------------------------------
//
//  Store and manipulate 2-site density matricies
//
class TwoSiteDMs
{
public:
    typedef std::complex<double> eType;
    typedef Matrix4<eType> DMType;
    typedef Matrix4<double> OperatorT;
    typedef Matrix4<eType>  OperatorCT;
    typedef SMatrix<double> ExpectationT;

    TwoSiteDMs(int L,int p);
    ~TwoSiteDMs();
    void Insert(int ia, int ib,const DMType& dm);

    template <class O> ExpectationT Contract(const O& op) const;
    ExpectationT GetTraces() const;
    ExpectationT GetVNEntropies() const;

private:
    typedef SMatrix<DMType> DMsType;
    typedef DMatrix<eType> MatrixCT;

    int itsL;
    int itsp;
    DMsType itsDMs;
};

#endif // TWOSITEDMS_H
