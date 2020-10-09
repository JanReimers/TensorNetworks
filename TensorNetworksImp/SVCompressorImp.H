#ifndef SVCOMPRESSORIMP_H
#define SVCOMPRESSORIMP_H

#include "TensorNetworks/SVCompressor.H"

namespace TensorNetworks
{

template <class T> class SVCompressorImp : public virtual SVCompressor<T>
{
        typedef typename SVCompressor<T>::MatrixT MatrixT;
    public:
        SVCompressorImp(int Dmax, double eps);
        virtual ~SVCompressorImp();

        virtual int GetDmax() const {return itsDmax;}
        virtual bool Donly () const {return itsDmax>0 && itsSVeps==0.0;}

        virtual double Compress(MatrixT& U, DiagonalMatrixRT& s, MatrixT& Vdagger);

    protected:
        int    itsDmax;
        double itsSVeps;
};

typedef SVCompressorImp<eType> SVCompressorImpC;
typedef SVCompressorImp<double               > SVCompressorImpR;
}

#endif // SVCOMPRESSORIMP_H
