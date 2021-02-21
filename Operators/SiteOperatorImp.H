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

class SiteOperatorImp : public virtual SiteOperator
{
public:
    typedef Matrix<MatrixRT> TensorT;
    SiteOperatorImp(int d); //Construct with identity operator
    SiteOperatorImp(int d, SpinOperator so); //Construct with identity operator
    SiteOperatorImp(int d, Position lbr, const OperatorClient* O);
    SiteOperatorImp(int d, Direction lr , Position lbr, const MatrixRT& U, const DiagonalMatrixRT& expEvs);
    SiteOperatorImp(const MatrixOR& W); //Construct with W operator
    virtual ~SiteOperatorImp();
    void SetNeighbours(SiteOperator* left, SiteOperator* right);

    virtual int             Getd     () const {return itsd;}
    virtual const OpRange&  GetRanges() const {return itsOpRange;}
    virtual IIType          GetDws   () const {return std::make_tuple(itsOpRange.Dw1,itsOpRange.Dw2);}
    virtual const MatrixOR& GetW     () const {return itsWs;}
    //
    //  Contract MPOs together
    //
    virtual void   Product(const SiteOperator* O2);
    //
    //  SVD compress and normalize to reduce Dw. Returns accumulated SVD truncation error
    //
    virtual double Compress(CompressType,Direction,const SVCompressorR*);
    //
    //  As per Parker et. al.
    //
    virtual void    CanonicalForm(Direction LR);
    virtual void   iCanonicalFormQRIter(Direction LR); //Infinite lattice iMPO version
    virtual void   iCanonicalFormTriangular(Direction LR); //Infinite lattice iMPO version

    virtual void   Report(std::ostream&) const;
    virtual char   GetNormStatus(double eps) const;
    virtual double GetFrobeniusNorm() const;
    virtual char   GetUpperLower(double eps) const;
    virtual char   GetLRB() const {return PBulk;}
    virtual void   SetLimits(); //Fix up itsDw12 by inspecting zeros in Wmn

private:
    SiteOperatorImp* GetNeighbour(Direction) const;
    void Init_lr(Position lbr, int lindex,int rindex);

    double CompressStd   (Direction,const SVCompressorR*);
    double CompressParker(Direction,const SVCompressorR*);
    void   QLTransfer    (Direction lr,const MatrixRT& L);

    void     NewBondDimensions(int D1, int D2, bool saveData=false);

    void     AccumulateTruncationError(double err);

    MatrixRT BuildK(int M) const;
    VectorRT Buildc(int M) const;
    double   Contract (int w11, int w12, int w21, int w22) const;
    double   ContractT(int w11, int w12, int w21, int w22) const;
    void     GaugeTransform(const MatrixRT& R, const MatrixRT& Rinv);
    double   Contract_sM(int M) const;
    double   Contract_sM1(int M) const;
protected:

    int      itsd; //2*S+1
    OpRange  itsOpRange;
    double   itsTruncationError;
    MatrixOR itsWs; //Operator values matrix version of W.
    SiteOperatorImp* itsLeft_Neighbour;
    SiteOperatorImp* itsRightNeighbour;
};

}
#endif // SITEOPERATORIMP_H
