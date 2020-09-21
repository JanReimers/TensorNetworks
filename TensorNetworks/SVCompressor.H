#ifndef SVCOMPRESSOR_H
#define SVCOMPRESSOR_H

#include "TensorNetworks/Typedefs.H"

class SVCompressor
{
    protected:
        typedef TensorNetworks::MatrixRT MatrixRT;
        typedef TensorNetworks::MatrixCT MatrixCT;
        typedef TensorNetworks::DiagonalMatrixRT DiagonalMatrixRT;

    public:
        SVCompressor() {};
        virtual ~SVCompressor() {};

        virtual int GetDmax() const=0;
        // Returns integrated weight of discarded vector space.
        virtual double Compress(MatrixCT& U, DiagonalMatrixRT& s, MatrixCT& Vdagger)=0;
};

#endif // SVCOMPRESSOR_H
