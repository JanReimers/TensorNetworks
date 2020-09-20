#ifndef MPOSITE_H
#define MPOSITE_H

#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/Enums.H"
#include "TensorNetworks/Dw12.H"
#include "TensorNetworksImp/Typedefs.H"


class OperatorWRepresentation;

class SiteOperatorImp : public virtual SiteOperator
{
        typedef TensorNetworks::VectorRT VectorRT;
    public:
        SiteOperatorImp(TensorNetworks::Position  lbr, const OperatorWRepresentation* O,int d);
        SiteOperatorImp(TensorNetworks::Direction lr , const MatrixRT& U, const VectorRT& expEvs, int d);
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
        virtual void Compress(TensorNetworks::Direction,int Dmax, double minSV);

       virtual void Report(std::ostream&) const;
    protected:

    private:
        friend class MPOTesting;
//        typedef TensorNetworks::MatrixCT MatrixCT;
        MatrixRT& GetW(int m, int n) {return itsWs(m+1,n+1);}
        void SVDTransfer(TensorNetworks::Direction lr,const VectorRT& s,const MatrixRT& UV);
        static MatrixRT  Contract1(const VectorRT & s, const MatrixRT& VA);
        static MatrixRT  Contract1(const MatrixRT& AU, const VectorRT & s);


        MatrixRT  Reshape(TensorNetworks::Direction lr);
        void     Reshape(int D1, int D2, bool saveData=false);
        void     Reshape(TensorNetworks::Direction lr,const MatrixRT& UV);

        typedef DMatrix<TensorNetworks::MatrixRT> TensorT;

        int itsd; //2*S+1
        Dw12 itsDw12;

        TensorT itsWs;
        SiteOperatorImp* itsLeft_Neighbour;
        SiteOperatorImp* itsRightNeighbour;
};

#endif // MPOSITE_H
