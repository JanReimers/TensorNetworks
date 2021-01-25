#ifndef SITEOPERATORBULK_H
#define SITEOPERATORBULK_H

#include "Operators/SiteOperatorImp.H"


namespace TensorNetworks
{

class SiteOperatorBulk
    : public virtual SiteOperator
    , private SiteOperatorImp
{
    public:
        SiteOperatorBulk(int d); //Construct with identity operator
        SiteOperatorBulk(int d,double S, SpinOperator so); //Construct with identity operator
        SiteOperatorBulk(int d, const OperatorClient* O);
        SiteOperatorBulk(int d, Direction lr , const MatrixRT& U, const DiagonalMatrixRT& expEvs);
        SiteOperatorBulk(int d, const TensorT& W); //Construct with W operator
        virtual ~SiteOperatorBulk();

        virtual const Dw12&     GetDw12() const
        {
            return itsDw;
        }
        virtual const MatrixRT& GetW(int m, int n) const
        {
            return itsWs(m+1,n+1);
        }

        virtual double GetFrobeniusNorm() const {return SiteOperatorImp::GetFrobeniusNorm();};
        virtual void Init_lr();
        virtual void CheckDws() const;
};

}
#endif // SITEOPERATORBULK_H
