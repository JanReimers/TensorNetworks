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
        SiteOperatorBulk(const MatrixOR& W); //Construct with W operator
        virtual ~SiteOperatorBulk();

        virtual char GetLRB  () const {return 'B';}
};

}
#endif // SITEOPERATORBULK_H
