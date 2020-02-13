#ifndef OneSiteSpinOPERATOR_H
#define OneSiteSpinOPERATOR_H

#include "TensorNetworks/OperatorWRepresentation.H"
#include "TensorNetworks/Dw12.H"


class OneSiteSpinOperator  : public virtual OperatorWRepresentation
{
    public:
        OneSiteSpinOperator(double S, TensorNetworks::SpinOperator);
        virtual ~OneSiteSpinOperator();

        virtual MatrixT GetW   (TensorNetworks::Position,int m, int n) const;
        virtual Dw12    GetDw12(TensorNetworks::Position) const {return itsDw12;}
    private:
        double itsS;
        int    itsp;
        TensorNetworks::SpinOperator itsOperator;
        Dw12   itsDw12;
};

#endif // OneSiteSpinOPERATOR_H
