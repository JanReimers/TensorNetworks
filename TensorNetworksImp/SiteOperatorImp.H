#ifndef MPOSITE_H
#define MPOSITE_H

#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/Enums.H"
class OperatorWRepresentation;

class SiteOperatorImp : public virtual SiteOperator
{
//        typedef MatrixProductSite::MatrixT MatrixT;
//        typedef MatrixProductSite::MatrixCT MatrixCT;
    public:
        SiteOperatorImp(TensorNetworks::Position lbr, const OperatorWRepresentation* O,int p);
        virtual ~SiteOperatorImp();

        virtual TensorNetworks::ipairT GetDw  () const {return itsDws;}
        virtual const Dw12*            GetDw12() const {return itsDW12;}
        virtual const TensorNetworks::MatrixT GetW(int m, int n) const
        {
            return itsWs(m+1,n+1);
        }

    protected:

    private:
        friend class MPOTesting;
        typedef DMatrix<TensorNetworks::MatrixT> TensorT;

        int itsp; //2*S+1
        TensorNetworks::ipairT itsDws;
        const Dw12* itsDW12;

        TensorT itsWs;
};

#endif // MPOSITE_H
