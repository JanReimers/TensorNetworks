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
        virtual ~SiteOperatorLeft();

//        virtual MatrixRT GetV(Direction,int m, int n) const;
//        virtual void     SetV(Direction,int m, int n, const MatrixRT& V);

        virtual char GetLRB() const {return 'L';}
private:
        virtual void Init_lr(int oneIndex);
};

}
#endif // SITEOPERATORLEFT_H
