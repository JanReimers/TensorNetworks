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
    typedef std::complex<double> dcmplx;
    typedef Matrix4<dcmplx> DMType;
    typedef Matrix4<double> OperatorT;
    typedef Matrix4<dcmplx>  OperatorCT;
    typedef SMatrix<double> ExpectationT;

    TwoSiteDMs(int L,int d);
    ~TwoSiteDMs();
    void Insert(int ia, int ib,const DMType& dm);
    int    GetL() const {return itsL;}
    double GetS() const {return (itsd-1.0)/2.0;}

    template <class O> ExpectationT Contract(const O& op) const;
    ExpectationT GetTraces() const;
    ExpectationT GetVNEntropies() const;

private:
    typedef SMatrix<DMType> DMsType;
    typedef Matrix<dcmplx> MatrixCT;

    int itsL;
    int itsd;
    DMsType itsDMs;
};

#endif // TWOSITEDMS_H
