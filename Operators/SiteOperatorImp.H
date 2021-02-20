#ifndef SITEOPERATORIMP_H
#define SITEOPERATORIMP_H

#include "TensorNetworks/Namespace.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/Dw12.H"
#include "Operators/OperatorValuedMatrix.H"
#include "TensorNetworksImp/Typedefs.H"
#include "oml/matrix.h"

class MPOTests;



namespace TensorNetworks
{

class SiteOperatorImp : public virtual SiteOperator
{
    protected:
        typedef Matrix<MatrixRT> TensorT;
        SiteOperatorImp(int d); //Construct with identity operator
        SiteOperatorImp(int d, double S, SpinOperator so); //Construct with identity operator
        SiteOperatorImp(int d, const OperatorClient* O);
        SiteOperatorImp(int d, Direction lr , const MatrixRT& U, const DiagonalMatrixRT& expEvs);
        SiteOperatorImp(const MatrixOR& W); //Construct with W operator
    public:
        virtual ~SiteOperatorImp();
        void SetNeighbours(SiteOperator* left, SiteOperator* right);

        virtual int             Getd   () const {return itsd;}
        virtual const Dw12&     GetDw12() const {return itsDw;}
        virtual const MatrixOR& GetW   () const {return itsWOvM;}
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
        virtual void    CanonicalFormOvM(Direction LR);
        virtual void   iCanonicalFormQRIter(Direction LR); //Infinite lattice iMPO version
        virtual void   iCanonicalFormTriangular(Direction LR); //Infinite lattice iMPO version

        virtual void   Report(std::ostream&) const;
        virtual char   GetNormStatus(double eps) const;
        virtual double GetFrobeniusNorm() const;
        virtual char   GetUpperLower(double eps) const;
        virtual char   GetLRB() const=0;
        virtual void   SetLimits(); //Fix up itsDw12 by inspecting zeros in Wmn
    protected:
//        void SyncWtoO();
        void SyncOtoW();
//        void CheckSync();

    private:
        friend class MPOTesting;
        friend class iMPOImp;

        SiteOperatorImp* GetNeighbour(Direction) const;

        double CompressStd   (Direction,const SVCompressorR*);
        double CompressParkerOvM(Direction,const SVCompressorR*);
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
        Dw12     itsDw;
        double   itsTruncationError;
        MatrixOR itsWOvM; //Operator values matrix version of W.
        SiteOperatorImp* itsLeft_Neighbour;
        SiteOperatorImp* itsRightNeighbour;
};

}
#endif // SITEOPERATORIMP_H
