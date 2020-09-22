#ifndef SVCOMPRESSOR_H
#define SVCOMPRESSOR_H

#include "TensorNetworks/Typedefs.H"

template <class T> class SVCompressor
{
    protected:
        typedef TensorNetworks::DiagonalMatrixRT DiagonalMatrixRT;

    public:
        typedef DMatrix<T> MatrixT;

        SVCompressor() {};
        virtual ~SVCompressor() {};

        virtual int  GetDmax() const=0;
        virtual bool Donly  () const=0;
        // Returns integrated weight of discarded vector space.
        virtual double Compress(MatrixT& U, DiagonalMatrixRT& s, MatrixT& Vdagger)=0;
};

typedef SVCompressor<TensorNetworks::eType> SVCompressorC;
typedef SVCompressor<double               > SVCompressorR;

#endif // SVCOMPRESSOR_H
