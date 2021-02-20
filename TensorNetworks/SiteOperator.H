#ifndef SITEOPERATOR_H
#define SITEOPERATOR_H

#include "TensorNetworks/Namespace.H"

namespace TensorNetworks
{

//
//  The implementation of this class cache all the  <m|W|n> matrix elements
//  so it can used in hot loops.
//
class SiteOperator
{
public:
    SiteOperator() {};
    virtual ~SiteOperator() {};

    virtual int         Getd   () const=0;
    virtual const Dw12& GetDw12() const=0;
    virtual double      GetFrobeniusNorm() const=0;
    //
    //  Generate MPO matrices.  m,n are quantum numbers for <m|Oa|n>
    //
    virtual const MatrixRT& GetW(int m, int n) const=0;
    virtual const MatrixOR& GetW(            ) const=0;
    //
    //  Contract MPOs together
    //
    virtual void Product(const SiteOperator* O2)=0;
    //
    //  SVD compress and normalize to reduce Dw Returns accumulated SVD truncation error.
    //
    virtual double Compress(CompressType,Direction,const SVCompressorR*)=0;
    //
    //  as per: Parker, D. E.; Cao, X. & Zaletel, M. P.
    //          Local Matrix Product Operators: Canonical Form, Compression, and Control Theory
    //          Phys. Rev. B 102, 035147 (2020), 2019
    //
    virtual void  CanonicalFormOvM(Direction LR)=0;
    virtual void iCanonicalFormQRIter(Direction LR)=0; //Infinite lattice iMPO version
    virtual void iCanonicalFormTriangular(Direction LR)=0; //Infinite lattice iMPO version

    virtual void Report(std::ostream&) const=0;
    virtual char GetNormStatus(double eps) const=0;
    virtual char GetUpperLower(double eps) const=0;
};

}
#endif // SITEOPERATOR_H
