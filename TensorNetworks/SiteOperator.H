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
    //
    //  Generate MPO matrices.  m,n are quantum numbers for <m|Oa|n>
    //
    virtual const MatrixRT& GetW(int m, int n) const=0;
    //
    //  Contract MPOs together
    //
    virtual void Combine(const SiteOperator* O2,double factor)=0;
    //
    //  SVD compress and normalize to reduce Dw
    //
    virtual void Compress(Direction,const SVCompressorR*)=0;

    virtual void Report(std::ostream&) const=0;

};

}
#endif // SITEOPERATOR_H
