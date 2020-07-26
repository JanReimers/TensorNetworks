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
        SiteOperatorImp(TensorNetworks::Position  lbr, const OperatorWRepresentation* O,int p);
        SiteOperatorImp(TensorNetworks::Direction lr , const MatrixT& U, const VectorT& expEvs, int p);
        virtual ~SiteOperatorImp();


        virtual const Dw12&    GetDw12() const {return itsDw12;}
        virtual const MatrixT& GetW(int m, int n) const
        {
            return itsWs(m+1,n+1);
        }

    protected:

    private:
        friend class MPOTesting;
        typedef DMatrix<TensorNetworks::MatrixT> TensorT;

        int itsp; //2*S+1
        Dw12 itsDw12;

        TensorT itsWs;
};

#endif // MPOSITE_H
