#ifndef SITEOPERATORLEFT_H
#define SITEOPERATORLEFT_H

#include "Operators/SiteOperatorImp.H"


namespace TensorNetworks
{

class SiteOperatorLeft
    : public virtual SiteOperator
    , private SiteOperatorImp
{
    public:
        SiteOperatorLeft(int d); //Construct with identity operator
        SiteOperatorLeft(int d,double S, SpinOperator so); //Construct with identity operator
        SiteOperatorLeft(int d, const OperatorClient* O);
        SiteOperatorLeft(int d, Direction lr , const MatrixRT& U, const DiagonalMatrixRT& expEvs);
        SiteOperatorLeft(int d, const TensorT& W); //Construct with W operator
        virtual ~SiteOperatorLeft();

        virtual const Dw12&     GetDw12() const
        {
            return itsDw;
        }
        virtual const MatrixRT& GetW(int m, int n) const
        {
            return itslWs(m+1,n+1);
        }

        virtual void Combine(const SiteOperator* O2,double factor);
        virtual double GetFrobeniusNorm() const {return SiteOperatorImp::GetFrobeniusNorm();};

        virtual void Init_lr(int oneIndex);
        virtual void Update();
        virtual void SetLimits(); //Fix up itsDw12 by inspecting zeros in Wmn
        virtual void CheckDws() const;

    private:

        Dw12     itsDw;
        MatrixRT itsl; //l in l*W*W*...*W*W*r
        TensorT  itslWs;
};

}
#endif // SITEOPERATORLEFT_H
