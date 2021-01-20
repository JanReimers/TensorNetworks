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
    virtual MatrixRT GetW   (Position,int m, int n) const=0;
    virtual Dw12     GetDw12(Position) const=0;
//    friend class SiteOperatorImp;
    friend class ::MPOTests;
};

class SiteOperatorImp : public virtual SiteOperator
{
    protected:
        typedef Matrix<MatrixRT> TensorT;
    public:
        SiteOperatorImp(int d, Position  lbr); //Construct with identity operator
        SiteOperatorImp(int d, Position  lbr, double S, SpinOperator so); //Construct with identity operator
        SiteOperatorImp(int d, Position  lbr, const OperatorClient* O);
        SiteOperatorImp(int d, Position  lbr, Direction lr , const MatrixRT& U, const DiagonalMatrixRT& expEvs);
        SiteOperatorImp(int d, Position  lbr, const TensorT& W); //Construct with W operator
        virtual ~SiteOperatorImp();
        void SetNeighbours(SiteOperator* left, SiteOperator* right);

        virtual int             Getd   () const {return itsd;}
        virtual const Dw12&     GetDw12() const
        {
            static Dw12 Dw;
            Dw=itsDw;
            switch (itsLBR)
            {
            case PBulk:
                break;
            case PLeft:
                Dw.Dw1=1;
                Dw.w1_first.SetLimits(Dw.Dw2);
                Dw.w2_last .SetLimits(Dw.Dw1);
                Fill(Dw.w1_first,1);
                Fill(Dw.w2_last ,Dw.Dw2);
                break;
            case PRight:
                Dw.Dw2=1;
                Dw.w1_first.SetLimits(Dw.Dw2);
                Dw.w2_last .SetLimits(Dw.Dw1);
                Fill(Dw.w1_first,1);
                Fill(Dw.w2_last ,Dw.Dw2);
                break;
            default:
                assert(false); //This can happen if itsLBR was never initialized.
            }
            return Dw;
        }
        virtual const MatrixRT& GetW(int m, int n) const
        {
//            std::cout << "LBR,Dw1,Dw2,iW,W,lr=" << itsLBR << " " << itsDw.Dw1 << " " << itsDw.Dw2 <<
//            std::endl << itsWs(m+1,n+1).GetLimits() << itslr.GetLimits() << std::endl << "---------------" << std::endl;
            switch (itsLBR)
            {
            case PBulk:
                itsW=itsWs(m+1,n+1);
                break;
            case PLeft:
                itsW=itslr*itsWs(m+1,n+1);
                break;
            case PRight:
                itsW=itsWs(m+1,n+1)*itslr;
                break;
            default:
                assert(false); //This can happen if itsLBR was never initialized.
            }
            return itsW;
        }
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
        void     SetLimits(); //Fix up itsDw12 by inspecting zeros in Wmn
        void     Init_lr();
        void     CheckDws() const;

        int      itsd; //2*S+1
        Dw12     itsDw;
        Position itsLBR;
        double   itsTruncationError;
        bool     isShapeDirty; //Matrices got reshaped
        bool     isData_Dirty; //Matrices got new data

        TensorT  itsWs;
        MatrixRT itslr; //l or r in l*W*W*...*W*W*r
        mutable MatrixRT itsW; //Used for return
        SiteOperatorImp* itsLeft_Neighbour;
        SiteOperatorImp* itsRightNeighbour;
};

}
#endif // SITEOPERATORIMP_H
