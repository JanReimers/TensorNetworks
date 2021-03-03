#ifndef SVCOMPRESSOR_H
#define SVCOMPRESSOR_H

#include "TensorNetworks/Namespace.H"
#include <vector>

namespace TensorNetworks
{

template <class T> class SVCompressor
{
    public:
        typedef Matrix<T> MatrixT;
        typedef std::vector<MatrixT> dVectorT;

        SVCompressor() {};
        virtual ~SVCompressor() {};

        virtual int  GetDmax() const=0;
        virtual bool Donly  () const=0;
        // Returns compression error of discarded vector space. =Sum(s^2) for discarded Svs
        virtual double Compress(MatrixT& U, DiagonalMatrixRT& s, MatrixT& Vdagger) const=0;
        virtual double Compress(dVectorT& gamma, DiagonalMatrixRT& lambda) const=0;
};


}
#endif // SVCOMPRESSOR_H
