#ifndef SITEOPERATOR_H
#define SITEOPERATOR_H

#include "TensorNetworks/Typedefs.H"

class Dw12;
//
//  The implementation of this class cache all the  <m|W|n> matrix elements
//  so it can used in hot loops.
//
class SiteOperator
{
public:
    typedef TensorNetworks::MatrixT MatrixT;

    SiteOperator() {};
    virtual ~SiteOperator() {};
    virtual const Dw12& GetDw12() const=0;
    //
    //  Generate MPO matrices.  m,n are quantum numbers for <m|Oa|n>
    //
    virtual const MatrixT& GetW(int m, int n) const=0;
    //
    //  Contract MPOs together
    //
    virtual void Combine(const SiteOperator* O2)=0;
};

#endif // SITEOPERATOR_H
