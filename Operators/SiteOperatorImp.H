#ifndef SITEOPERATORIMP_H
#define SITEOPERATORIMP_H

#include "TensorNetworks/Namespace.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/Dw12.H"
#include "TensorNetworksImp/Typedefs.H"
#include "oml/matrix.h"

class MPOTests;

namespace TensorNetworks
{
//
//  Derive from this in order to load up your site MPO/W matrices
//
class OperatorClient
{
protected:
    OperatorClient() {}
    virtual ~OperatorClient() {}
public:
    virtual MatrixRT GetW   (int m, int n) const=0;
    virtual Dw12     GetDw12() const=0;
    friend class ::MPOTests;
};

class SiteOperatorImp : public virtual SiteOperator
{
    protected:
        typedef Matrix<MatrixRT> TensorT;
        SiteOperatorImp(int d); //Construct with identity operator
        SiteOperatorImp(int d, double S, SpinOperator so); //Construct with identity operator
        SiteOperatorImp(int d, const OperatorClient* O);
        SiteOperatorImp(int d, Direction lr , const MatrixRT& U, const DiagonalMatrixRT& expEvs);
        SiteOperatorImp(int d, const TensorT& W); //Construct with W operator
    public:
        virtual ~SiteOperatorImp();
        void SetNeighbours(SiteOperator* left, SiteOperator* right);

        virtual int             Getd   () const {return itsd;}
        virtual const Dw12&     GetDw12() const=0;
        virtual const MatrixRT& GetW(int m, int n) const =0;
        virtual const MatrixRT& GetiW(int m, int n) const
        {
            return itsWs(m+1,n+1);
        }
        //
        //  Contract MPOs together
        //
        virtual void Combine(const SiteOperator* O2,double factor);
        //
        //  SVD compress and normalize to reduce Dw
        //
        virtual void CompressStd   (Direction,const SVCompressorR*);
        virtual void CompressParker(Direction,const SVCompressorR*);
        //
        //  As per Parker et. al.
        //
        virtual void CanonicalForm(Direction LR);

        virtual void Report(std::ostream&) const;
        virtual char GetNormStatus(double eps) const;

        virtual void Init_lr()=0;
        virtual void CheckDws() const;
        virtual void SetLimits(); //Fix up itsDw12 by inspecting zeros in Wmn
    private:
        friend class MPOTesting;
        friend class iMPOImp;
        void SetiW(int m, int n, const MatrixRT& W);

        void SVDTransfer(Direction lr,const DiagonalMatrixRT& s,const MatrixRT& UV);
        void  QLTransfer(Direction lr,const MatrixRT& L);

        MatrixRT Reshape  (Direction lr) const;
        MatrixRT ReshapeV (Direction lr) const;
        MatrixRT ReshapeV1(Direction lr) const;
        void     Reshape  (int D1, int D2, bool saveData=false);
        void     Reshape  (Direction lr,const MatrixRT& UV);
        void     ReshapeV (Direction lr,const MatrixRT& Q);
        void     ReshapeV1(Direction lr,const MatrixRT& Q);

        bool     isOrthonormal(Direction lr,const MatrixRT& Q) const;
        void     AccumulateTruncationError(double err);
    protected:

        int      itsd; //2*S+1
        Dw12     itsDw;
        double   itsTruncationError;
        bool     isShapeDirty; //Matrices got reshaped
        bool     isData_Dirty; //Matrices got new data

        TensorT  itsWs;
        SiteOperatorImp* itsLeft_Neighbour;
        SiteOperatorImp* itsRightNeighbour;
};

}
#endif // SITEOPERATORIMP_H
