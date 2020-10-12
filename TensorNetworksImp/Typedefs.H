#ifndef _IMPTYPEDEFS_H_
#define _IMPTYPEDEFS_H_

#include "TensorNetworks/Typedefs.H"

template <class T> class Matrix6;
template <class T> class Vector3;
template <class T> class Vector4;

//--------------------------------------------------------------------------------------
//
//  Typedefs for various containers
//
namespace TensorNetworks
{
        typedef Vector <double>  VectorRT;
        typedef Matrix6<double>  Matrix6RT;

        typedef Matrix6<dcmplx>  Matrix6CT;
        typedef Matrix4<dcmplx>  Matrix4CT;
        typedef Vector3<dcmplx>  Vector3CT;
        typedef Vector4<dcmplx>  Vector4CT;
};

#endif //_IMPTYPEDEFS_H_
