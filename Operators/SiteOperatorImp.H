#ifndef MPOSITE_H
#define MPOSITE_H

#include "TensorNetworks/Namespace.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/Dw12.H"
#include "TensorNetworksImp/Typedefs.H"
#include "oml/dmatrix.h"

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

    virtual MatrixRT GetW   (Position,int m, int n) const=0;
    virtual Dw12     GetDw12(Position) const=0;
    friend class SiteOperatorImp;
    friend class ::MPOTests;
};

class SiteOperatorImp : public virtual SiteOperator
{
    public:
        SiteOperatorImp(int d); //Construct with identity operator
        SiteOperatorImp(int d, double S, SpinOperator so); //Construct with identity operator
        SiteOperatorImp(int d, Position  lbr, const OperatorClient* O);
        SiteOperatorImp(int d, Direction lr , const MatrixRT& U, const DiagonalMatrixRT& expEvs);
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
