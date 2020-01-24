#ifndef _IMPTYPEDEFS_H_
#define _IMPTYPEDEFS_H_

#include "TensorNetworks/Typedefs.H"
#include "Containers/Matrix6.H"
#include "Containers/Matrix4.H"
#include "Containers/Vector3.H"
#include "Containers/Vector4.H"
#include "oml/vector.h"

//--------------------------------------------------------------------------------------
//
//  Typedefs for various containers
//
namespace TensorNetworks
{
        typedef Vector <double> VectorT;

        typedef std::complex<double> eType;
        typedef DMatrix<eType>  MatrixCT;
        typedef Vector <eType > VectorCT;
        typedef Matrix6<eType>  Matrix6T;
        typedef Matrix4<eType>  Matrix4T;
        typedef Vector3<eType>  Vector3T;
        typedef Vector4<eType>  Vector4T;
};

#endif //_IMPTYPEDEFS_H_
