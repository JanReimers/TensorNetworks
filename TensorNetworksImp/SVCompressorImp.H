#ifndef SVCOMPRESSORIMP_H
#define SVCOMPRESSORIMP_H

#include "TensorNetworks/SVCompressor.H"

namespace TensorNetworks
{

template <class T> class SVCompressorImp : public virtual SVCompressor<T>
{
        typedef typename SVCompressor<T>:: MatrixT  MatrixT;
        typedef typename SVCompressor<T>::dVectorT dVectorT;
    public:
        SVCompressorImp(int Dmax, double eps);
        virtual ~SVCompressorImp();

        virtual int GetDmax() const {return itsDmax;}
        virtual bool Donly () const {return itsDmax>0 && itsSVeps==0.0;}

        virtual double Compress( MatrixT& U    , DiagonalMatrixRT& s     , MatrixT& Vdagger) const;
        virtual double Compress(dVectorT& gamma, DiagonalMatrixRT& lambda) const;

    protected:
        int    itsDmax;
        double itsSVeps;
};

typedef SVCompressorImp<dcmplx > SVCompressorImpC;
typedef SVCompressorImp<double> SVCompressorImpR;
}

#endif // SVCOMPRESSORIMP_H
