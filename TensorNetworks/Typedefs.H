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
    typedef DMatrix  <double>  MatrixT;
    typedef std::pair<int,int> ipairT;
};

#endif //_TYPEDEFS_H_
