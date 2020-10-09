#ifndef OneSiteSpinOPERATOR_H
#define OneSiteSpinOPERATOR_H

#include "TensorNetworks/OperatorWRepresentation.H"
#include "TensorNetworks/Dw12.H"

namespace TensorNetworks
{

class OneSiteSpinOperator  : public virtual OperatorWRepresentation
{
    public:
        OneSiteSpinOperator(double S, SpinOperator);
        virtual ~OneSiteSpinOperator();

        virtual MatrixRT GetW  (Position,int m, int n) const;
        virtual Dw12    GetDw12(Position) const {return itsDw12;}
    private:
        double itsS;
        int    itsd;
        SpinOperator itsOperator;
        Dw12   itsDw12;
};

}
#endif // OneSiteSpinOPERATOR_H
