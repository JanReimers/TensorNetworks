#ifndef SVCOMPRESSOR_H
#define SVCOMPRESSOR_H

#include "TensorNetworks/Namespace.H"

namespace TensorNetworks
{

template <class T> class SVCompressor
{
    public:
        typedef DMatrix<T> MatrixT;

        SVCompressor() {};
        virtual ~SVCompressor() {};

        virtual int  GetDmax() const=0;
        virtual bool Donly  () const=0;
        // Returns integrated weight of discarded vector space.
        virtual double Compress(MatrixT& U, DiagonalMatrixRT& s, MatrixT& Vdagger)=0;
};


}
#endif // SVCOMPRESSOR_H
