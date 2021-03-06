#ifndef SITEOPERATORIMP_H
#define SITEOPERATORIMP_H

#include "TensorNetworks/Namespace.H"
#include "TensorNetworks/SiteOperator.H"
#include "Operators/OperatorValuedMatrix.H"
#include "TensorNetworksImp/Typedefs.H"
#include "oml/matrix.h"

class MPOTests;



namespace TensorNetworks
{

class OperatorBond;

class SiteOperatorImp : public virtual SiteOperator
{
public:
    typedef Matrix<MatrixRT> TensorT;
    SiteOperatorImp(int d,MPOForm); //Construct with identity operator and predetermined MPO form
    SiteOperatorImp(int d, SpinOperator so); //Construct with identity operator
    SiteOperatorImp(int d, Position lbr, const OperatorClient* O,MPOForm);
    SiteOperatorImp(int d, Direction lr , Position lbr, const MatrixRT& U, const DiagonalMatrixRT& expEvs);
    SiteOperatorImp(const MatrixOR& W); //Construct with W operator
    virtual ~SiteOperatorImp();
    void SetNeighbours(OperatorBond* leftBond, OperatorBond* rightBond);

    virtual int             Getd     () const {return itsd;}
    virtual const OpRange&  GetRanges() const {return itsOpRange;}
    virtual IIType          GetDws   () const {return std::make_tuple(itsOpRange.Dw1,itsOpRange.Dw2);}
    virtual const MatrixOR& GetW     () const {return itsWs;}
    //
    //  Combine MPOs together with tensor sum and product
    //
    virtual void   Product(const SiteOperator* O2);
    virtual void   Sum    (const SiteOperator* O2, double factor);
    //
    //  SVD compress and normalize to reduce Dw. Returns accumulated SVD truncation error
    //
    virtual double  Compress(CompressType,Direction,const SVCompressorR*);
    virtual double iCompress(CompressType,Direction,const SVCompressorR*);
    //
    //  As per Parker et. al.
    //
    virtual void      CanonicalForm(Direction LR);
    virtual void     iCanonicalFormTriangular(Direction LR); //Infinite lattice iMPO version
    //
    //  Low level QRiter and SVD routines (for iMPOs only)
    //
    virtual void     InitQRIter(); //Set itsG to unit matrix.
    virtual double   QRStep(Direction lr,double eps); //return eta = Max(fabs(L-Id))
    virtual MatrixRT GetGaugeTransform() const {return itsG;}
    virtual void     ZeroRowCol(Direction lr);

    virtual void   Report(std::ostream&) const;
    virtual char   GetNormStatus(double eps) const;
    virtual double GetFrobeniusNorm() const;
    virtual char   GetMeasuredShape(double eps) const;
    virtual char   GetLRB() const {return PBulk;}
    virtual void   SetLimits(); //Fix up itsDw12 by inspecting zeros in Wmn
    virtual char   GetUpperLower(double eps) const {return GetMeasuredShape(eps);}

private:
    friend class OperatorBond;
    OperatorBond* GetBond(Direction) const;
    void Init_lr(Position lbr, int lindex,int rindex);
    char GetForm() const;


    double  CompressStd   (Direction,const SVCompressorR*);
    double  CompressParker(Direction,const SVCompressorR*);
    double iCompressParker(Direction,const SVCompressorR*);
    void    QLTransfer    (Direction lr,const MatrixRT& L);

    void     NewBondDimensions(int D1, int D2, bool saveData=false);

    void     AccumulateTruncationError(double err);

    MatrixRT BuildK(int M,MPOForm ul) const;
    VectorRT Buildc(int M,MPOForm ul) const;
    double   ContractUL(int w11, int w12, int w21, int w22,MPOForm) const;
    void     GaugeTransform(const MatrixRT& R, const MatrixRT& Rinv);
    double   Contract_sM(int M,MPOForm ul) const;
    double   Contract_sM1(int M,MPOForm ul) const;
protected:

    int      itsd; //2*S+1
    OpRange  itsOpRange;
    Position itsLBR;
    MatrixOR itsWs; //Operator values matrix version of W.
    MatrixRT itsG; //Accumulated gauge transform from QR iterations.
    OperatorBond* itsLeft_Bond;
    OperatorBond* itsRightBond;
};

}
#endif // SITEOPERATORIMP_H
