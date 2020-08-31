#ifndef MPOSITE_H
#define MPOSITE_H

#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/Enums.H"
#include "TensorNetworks/Dw12.H"
#include "TensorNetworksImp/Typedefs.H"


class OperatorWRepresentation;

class SiteOperatorImp : public virtual SiteOperator
{
        typedef TensorNetworks::VectorT VectorT;
    public:
        SiteOperatorImp(TensorNetworks::Position  lbr, const OperatorWRepresentation* O,int d);
        SiteOperatorImp(TensorNetworks::Direction lr , const MatrixT& U, const VectorT& expEvs, int d);
        virtual ~SiteOperatorImp();
        void SetNeighbours(SiteOperator* left, SiteOperator* right);

        virtual const Dw12&    GetDw12() const {return itsDw12;}
        virtual const MatrixT& GetW(int m, int n) const
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
        virtual void Compress(TensorNetworks::Direction,int Dmax, double minSV);

       virtual void Report(std::ostream&) const;
    protected:

    private:
        friend class MPOTesting;
//        typedef TensorNetworks::MatrixCT MatrixCT;
        MatrixT& GetW(int m, int n) {return itsWs(m+1,n+1);}
        void SVDTransfer(TensorNetworks::Direction lr,const VectorT& s,const MatrixT& UV);
        static MatrixT  Contract1(const VectorT & s, const MatrixT& VA);
        static MatrixT  Contract1(const MatrixT& AU, const VectorT & s);


        MatrixT  Reshape(TensorNetworks::Direction lr);
        void     Reshape(int D1, int D2, bool saveData=false);
        void     Reshape(TensorNetworks::Direction lr,const MatrixT& UV);

        typedef DMatrix<TensorNetworks::MatrixT> TensorT;

        int itsd; //2*S+1
        Dw12 itsDw12;

        TensorT itsWs;
        SiteOperatorImp* itsLeft_Neighbour;
        SiteOperatorImp* itsRightNeighbour;
};

#endif // MPOSITE_H
