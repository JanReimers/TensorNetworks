#ifndef _TYPEDEFS_H_
#define _TYPEDEFS_H_

#include "oml/dmatrix.h"
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
    typedef Array    <double>  ArrayT;
    typedef std::pair<int,int> ipairT;
};

#endif //_TYPEDEFS_H_
