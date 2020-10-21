#ifndef _TYPEDEFS_H_
#define _TYPEDEFS_H_

#include <complex>

template <class T> class Matrix;
template <class T> class DiagonalMatrix;
template <class T> class Vector;
template <class T> class Matrix4;

//--------------------------------------------------------------------------------------
//
//  Typedefs for various containers
//
namespace TensorNetworks
{
    typedef std::complex<double> dcmplx;
    typedef Matrix  <dcmplx>  MatrixCT;
    typedef Matrix  <double>  MatrixRT;
    typedef Vector   <double>  VectorRT;
    typedef Vector   <dcmplx>  VectorCT;
    typedef Matrix4  <double>  Matrix4RT;

    typedef DiagonalMatrix<double> DiagonalMatrixRT;
};

#endif //_TYPEDEFS_H_
