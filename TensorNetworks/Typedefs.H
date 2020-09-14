#ifndef _TYPEDEFS_H_
#define _TYPEDEFS_H_

#include "oml/dmatrix.h"
#include "Containers/Matrix4.H"
#include <utility>


//--------------------------------------------------------------------------------------
//
//  Typedefs for various containers
//
namespace TensorNetworks
{
    typedef std::complex<double> eType;
    typedef DMatrix  <eType>   MatrixCT;
    typedef DMatrix  <double>  MatrixT;
    typedef Vector   <double>  VectorT;
    typedef Matrix4  <double>  Matrix4T;
    typedef Array    <double>  ArrayT;
};

#endif //_TYPEDEFS_H_
