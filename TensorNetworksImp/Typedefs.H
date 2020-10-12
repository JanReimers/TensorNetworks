#ifndef _IMPTYPEDEFS_H_
#define _IMPTYPEDEFS_H_

#include "TensorNetworks/Typedefs.H"
#include "Containers/Matrix6.H"
#include "Containers/Matrix4.H"
#include "Containers/Vector3.H"
#include "Containers/Vector4.H"
#include "oml/vector.h"
#include "oml/dmatrix.h"
#include "oml/diagonalmatrix.h"
#include "oml/array.h"

//--------------------------------------------------------------------------------------
//
//  Typedefs for various containers
//
namespace TensorNetworks
{
        typedef Vector <double>  VectorRT;
        typedef Matrix6<double> Matrix6RT;

        typedef Array  <dcmplx>    ArrayCT;
        typedef Matrix6<dcmplx>  Matrix6CT;
        typedef Matrix4<dcmplx>  Matrix4CT;
        typedef Vector3<dcmplx>  Vector3CT;
        typedef Vector4<dcmplx>  Vector4CT;
};

#endif //_IMPTYPEDEFS_H_
