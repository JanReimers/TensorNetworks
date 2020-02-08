#ifndef SPINCALCULATOR_H
#define SPINCALCULATOR_H

#include "oml/dmatrix.h"
#include <complex>
#include <cassert>

//
//  The best place to look these up for general S is
//    http://easyspin.org/easyspin/documentation/spinoperators.html
//
class SpinCalculator
{
        typedef std::complex<double> complx;
        typedef DMatrix<complx> MatrixCT;
        typedef DMatrix<double> MatrixT;

        typedef double(SpinCalculator::*dfp)(int,int) const;
        typedef complx(SpinCalculator::*cfp)(int,int) const;

    public:
        SpinCalculator(double S); //Shoule be 1/2,1,3/2,...
        ~SpinCalculator();

        double GetSm (int m, int n) const;
        double GetSp (int m, int n) const;
        double GetSx (int m, int n) const;
        complx GetSy (int m, int n) const;
        double GetSz (int m, int n) const;
        //
        //  Matrix versions
        //
        MatrixT  GetSm (int lowerIndex=1) const;
        MatrixT  GetSp (int lowerIndex=1) const;
        MatrixT  GetSx (int lowerIndex=1) const;
        MatrixCT GetSy (int lowerIndex=1) const;
        MatrixT  GetSz (int lowerIndex=1) const;

    private:
        double ConvertToSpin(int n) const;
        template <class M, class F> M BuildMatrix(int lowerIndex,F) const;
        double itsS; // Total spin
        int    itsd;
};

inline double SpinCalculator::ConvertToSpin(int n) const
{
    double s=n-itsS;
    assert(s>=-itsS);
    assert(s<=+itsS);
    return s;
}


#endif // SPINCALCULATOR_H
