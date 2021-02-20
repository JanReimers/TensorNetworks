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
        virtual ~SiteOperatorRight();

        virtual char GetLRB() const {return 'R';}
private:
        virtual void Init_lr(int oneIndex);
};

}
#endif // SITEOPERATORRIGHT_H
