#ifndef SITEOPERATOR_H
#define SITEOPERATOR_H

#include "TensorNetworks/Namespace.H"
#include "oml/imp/index_t.h"
#include <vector>

class MatLimits;

namespace TensorNetworks
{

struct Range
{
    index_t Low,High;
};

typedef std::vector<Range> vRange;

struct OpRange
{
    vRange rows; //Tight range for each row.
    vRange cols; //Tight range for each col.
    Range  row;  //Full range
    Range  col;  //Full range
    int    Dw1,Dw2; //Number of rows and cols.

    void resize(const MatLimits& l);
    void NonZeroAt(index_t i,index_t j);
};

std::ostream& operator<<(std::ostream&,const OpRange&);
//
//  The implementation of this class cache all the  <m|W|n> matrix elements
//  so it can used in hot loops.
//
class SiteOperator
{
public:
    typedef std::tuple<index_t  ,index_t   > IIType;

    SiteOperator() {};
    virtual ~SiteOperator() {};

    virtual int            Getd   () const=0;
    virtual IIType         GetDws () const=0;
    virtual const OpRange& GetRanges() const=0; //Summation ranges for contractions. Use this to avoid contracting over zeros elements
    virtual double         GetFrobeniusNorm() const=0;
    //
    //  Get Operator Valued Matrix
    //
    virtual const MatrixOR& GetW() const=0;
    //
    //  Combine MPOs together with tensor sum and product
    //
    virtual void Product(const SiteOperator* O2)=0;
    virtual void Sum    (const SiteOperator* O2, double factor)=0;
    //
    //  SVD compress and normalize to reduce Dw Returns accumulated SVD truncation error.
    //
    virtual double Compress(CompressType,Direction,const SVCompressorR*)=0;
    //
    //  as per: Parker, D. E.; Cao, X. & Zaletel, M. P.
    //          Local Matrix Product Operators: Canonical Form, Compression, and Control Theory
    //          Phys. Rev. B 102, 035147 (2020), 2019
    //
    virtual void     CanonicalForm(Direction LR)=0;
    virtual MatrixRT iCanonicalFormQRIter(Direction LR)=0; //Infinite lattice iMPO version
    virtual void     iCanonicalFormTriangular(Direction LR)=0; //Infinite lattice iMPO version

    virtual void Report(std::ostream&) const=0;
    virtual char GetNormStatus(double eps) const=0;
    virtual char GetUpperLower(double eps) const=0;
};

}
#endif // SITEOPERATOR_H
