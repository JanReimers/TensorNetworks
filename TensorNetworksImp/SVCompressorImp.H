#ifndef SVCOMPRESSORIMP_H
#define SVCOMPRESSORIMP_H

#include "TensorNetworks/SVCompressor.H"

template <class T> class SVCompressorImp : public virtual SVCompressor<T>
{
    public:
        typedef typename SVCompressor<T>::MatrixT MatrixT;
        typedef typename SVCompressor<T>::DiagonalMatrixRT DiagonalMatrixRT;

        SVCompressorImp(int Dmax, double eps);
        virtual ~SVCompressorImp();

        virtual int GetDmax() const {return itsDmax;}
        virtual bool Donly () const {return itsDmax>0 && itsSVeps==0.0;}

        virtual double Compress(MatrixT& U, DiagonalMatrixRT& s, MatrixT& Vdagger);

    protected:
        int    itsDmax;
        double itsSVeps;
};

typedef SVCompressorImp<TensorNetworks::eType> SVCompressorImpC;
typedef SVCompressorImp<double               > SVCompressorImpR;


#endif // SVCOMPRESSORIMP_H
