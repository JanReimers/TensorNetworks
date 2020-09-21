#ifndef SVCOMPRESSORIMP_H
#define SVCOMPRESSORIMP_H

#include "TensorNetworks/SVCompressor.H"

class SVCompressorImp : public virtual SVCompressor
{
    public:
        SVCompressorImp(int Dmax, double eps);
        virtual ~SVCompressorImp();

        virtual int GetDmax() const {return itsDmax;}
        virtual double Compress(MatrixCT& U, DiagonalMatrixRT& s, MatrixCT& Vdagger);

    private:
        int    itsDmax;
        double itsSVeps;
};

#endif // SVCOMPRESSORIMP_H
