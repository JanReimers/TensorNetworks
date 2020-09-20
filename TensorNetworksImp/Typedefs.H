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
        typedef Vector <double> VectorRT;
        typedef Vector <eType > VectorCT;
        typedef Array  <eType >  ArrayCT;

        typedef DiagonalMatrix<double> DiagonalMatrixRT;

        typedef Matrix6<eType>  Matrix6CT;
        typedef Matrix4<eType>  Matrix4CT;
        typedef Vector3<eType>  Vector3CT;
        typedef Vector4<eType>  Vector4CT;
};

#endif //_IMPTYPEDEFS_H_
