#ifndef SVMPOCOMPRESSOR_H
#define SVMPOCOMPRESSOR_H

#include "TensorNetworksImp/SVCompressorImp.H"

namespace TensorNetworks
{

class SVMPOCompressor
    : public virtual SVCompressorR
    , private SVCompressorImpR
{
        typedef typename SVCompressorR::MatrixT MatrixT;
    public:
        SVMPOCompressor(int Dmax, double eps);
        virtual ~SVMPOCompressor();

        virtual int GetDmax() const {return SVCompressorImp::GetDmax();}
        virtual double Compress(MatrixT& U, DiagonalMatrixRT& s, MatrixT& Vdagger);

    protected:

    private:
};

}
#endif // SVMPOCOMPRESSOR_H
