#ifndef _TYPEDEFS_H_
#define _TYPEDEFS_H_

#include <complex>

template <class T> class DMatrix;
template <class T> class DiagonalMatrix;
template <class T> class Vector;
template <class T> class Array;
template <class T> class Matrix4;

//--------------------------------------------------------------------------------------
//
//  Typedefs for various containers
//
namespace TensorNetworks
{
    typedef std::complex<double> eType;
    typedef DMatrix  <eType>   MatrixCT;
    typedef DMatrix  <double>  MatrixRT;
    typedef Vector   <double>  VectorRT;
    typedef Vector   <eType>   VectorCT;
    typedef Matrix4  <double>  Matrix4RT;
    typedef Array    <double>  ArrayRT;

    typedef DiagonalMatrix<double> DiagonalMatrixRT;
};

#endif //_TYPEDEFS_H_
