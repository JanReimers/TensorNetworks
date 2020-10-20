#ifndef SPINCALCULATOR_H
#define SPINCALCULATOR_H

#include "TensorNetworks/Enums.H"
#include "Containers/Matrix4.H"
#include "oml/matrix.h"
#include <complex>
#include <cassert>

namespace TensorNetworks
{

//
//  The best place to look these up for general S is
//    http://easyspin.org/easyspin/documentation/spinoperators.html
//
class SpinCalculator
{
        typedef std::complex<double> complx;
        typedef DMatrix<complx> MatrixCT;
        typedef DMatrix<double> MatrixRT;
        typedef Matrix4<complx> Matrix4CT;
        typedef Matrix4<double> Matrix4T;
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
        MatrixRT  GetSm (int lowerIndex=1) const;
        MatrixRT  GetSp (int lowerIndex=1) const;
        MatrixRT  GetSx (int lowerIndex=1) const;
        MatrixCT  GetSy (int lowerIndex=1) const;
        MatrixRT  GetSz (int lowerIndex=1) const;

        MatrixRT  Get   (int m, int n,SpinOperator so) const;
        //
        //  Two site Matrix versions
        //
        Matrix4T  GetSxSx (int lowerIndex=1) const;
        Matrix4CT GetSxSy (int lowerIndex=1) const;
        Matrix4T  GetSxSz (int lowerIndex=1) const;
        Matrix4CT GetSySx (int lowerIndex=1) const;
        Matrix4CT GetSySy (int lowerIndex=1) const;
        Matrix4CT GetSySz (int lowerIndex=1) const;
        Matrix4T  GetSzSx (int lowerIndex=1) const;
        Matrix4CT GetSzSy (int lowerIndex=1) const;
        Matrix4T  GetSzSz (int lowerIndex=1) const;


    private:
        double ConvertToSpin(int n) const;
        template <class M, class F> M BuildMatrix(int lowerIndex,F) const;
        template <class M, class F1,class F2> M BuildMatrix(int lowerIndex,F1,F2) const;
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

}
#endif // SPINCALCULATOR_H
