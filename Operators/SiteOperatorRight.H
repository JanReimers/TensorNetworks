#ifndef SITEOPERATORRIGHT_H
#define SITEOPERATORRIGHT_H

#include "Operators/SiteOperatorImp.H"


namespace TensorNetworks
{

class SiteOperatorRight
    : public virtual SiteOperator
    , private SiteOperatorImp
{
    public:
        SiteOperatorRight(int d); //Construct with identity operator
        SiteOperatorRight(int d,double S, SpinOperator so); //Construct with identity operator
        SiteOperatorRight(int d, const OperatorClient* O);
        SiteOperatorRight(int d, Direction lr , const MatrixRT& U, const DiagonalMatrixRT& expEvs);
        SiteOperatorRight(int d, const TensorT& W); //Construct with W operator
        virtual ~SiteOperatorRight();

        virtual const Dw12&     GetDw12() const
        {
            return itsDw;
        }
        virtual const MatrixRT& GetW(int m, int n) const
        {
            return itsWrs(m+1,n+1);
        }

        virtual void Combine(const SiteOperator* O2,double factor);
        virtual double GetFrobeniusNorm() const {return SiteOperatorImp::GetFrobeniusNorm();};
        virtual char   GetLRB() const {return 'R';}
        virtual void Init_lr(int oneIndex);
        virtual void Update();
        virtual void CheckDws() const;
        virtual void SetLimits(); //Fix up itsDw12 by inspecting zeros in Wmn

    private:

        Dw12     itsDw;
        MatrixRT itsr; //r in l*W*W*...*W*W*r
        TensorT  itsWrs;
};

}
#endif // SITEOPERATORRIGHT_H
