#ifndef OneSiteSpinOPERATOR_H
#define OneSiteSpinOPERATOR_H

#include "TensorNetworks/OperatorWRepresentation.H"

class OneSiteSpinOperator  : public virtual OperatorWRepresentation
{
    public:
        OneSiteSpinOperator(double S, TensorNetworks::SpinOperator);
        virtual ~OneSiteSpinOperator();

        virtual TensorNetworks::MatrixT GetW (TensorNetworks::Position,int m, int n) const;
        virtual TensorNetworks::ipairT  GetDw(TensorNetworks::Position) const;
        virtual const Dw12*             GetDw12(TensorNetworks::Position) const {return itsDw12;}
    private:
        Dw12* itsDw12;
        double itsS;
        int    itsp;
        TensorNetworks::SpinOperator itsOperator;
};

#endif // OneSiteSpinOPERATOR_H
