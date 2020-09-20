#ifndef IDENTITYOPERATOR_H
#define IDENTITYOPERATOR_H

#include "TensorNetworks/OperatorWRepresentation.H"
#include "TensorNetworks/Dw12.H"

class IdentityOperator  : public virtual OperatorWRepresentation
{
    public:
        IdentityOperator();
        virtual ~IdentityOperator();

        virtual MatrixRT GetW   (TensorNetworks::Position,int m, int n) const;
        virtual Dw12    GetDw12(TensorNetworks::Position) const {return itsDw12;}
    private:
        Dw12 itsDw12;
};

#endif // IDENTITYOPERATOR_H
