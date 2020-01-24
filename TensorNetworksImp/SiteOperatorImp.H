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

        TensorNetworks::ipairT GetDw() const {return itsDws;}
        const TensorNetworks::MatrixT GetW(int m, int n) const
        {
            return itsWs(m+1,n+1);
        }

    protected:

    private:
        friend class MPOTesting;
        typedef DMatrix<TensorNetworks::MatrixT> TensorT;

        int itsp; //2*S+1
        TensorNetworks::ipairT itsDws;

        TensorT itsWs;
};

#endif // MPOSITE_H
