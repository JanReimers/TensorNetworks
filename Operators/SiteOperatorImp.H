#ifndef MPOSITE_H
#define MPOSITE_H

#include "TensorNetworks/Namespace.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/Dw12.H"
#include "TensorNetworksImp/Typedefs.H"


namespace TensorNetworks
{
class SiteOperatorImp : public virtual SiteOperator
{
    public:
        SiteOperatorImp(Position  lbr, const OperatorWRepresentation* O,int d);
        SiteOperatorImp(Direction lr , const MatrixRT& U, const DiagonalMatrixRT& expEvs, int d);
        virtual ~SiteOperatorImp();
        void SetNeighbours(SiteOperator* left, SiteOperator* right);

        virtual const Dw12&    GetDw12() const {return itsDw12;}
        virtual const MatrixRT& GetW(int m, int n) const
        {
            return itsWs(m+1,n+1);
        }
        //
        //  Contract MPOs together
        //
        virtual void Combine(const SiteOperator* O2);
        //
        //  SVD compress and normalize to reduce Dw
        //
        virtual void Compress(Direction,const SVCompressorR*);

        virtual void Report(std::ostream&) const;

    private:
        friend class MPOTesting;
        MatrixRT& GetW(int m, int n) {return itsWs(m+1,n+1);}
        void SVDTransfer(Direction lr,const DiagonalMatrixRT& s,const MatrixRT& UV);

        MatrixRT Reshape(Direction lr);
        void     Reshape(int D1, int D2, bool saveData=false);
        void     Reshape(Direction lr,const MatrixRT& UV);

        typedef DMatrix<MatrixRT> TensorT;

        int itsd; //2*S+1
        Dw12 itsDw12;

        TensorT itsWs;
        SiteOperatorImp* itsLeft_Neighbour;
        SiteOperatorImp* itsRightNeighbour;
};

}
#endif // MPOSITE_H
