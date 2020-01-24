#ifndef IDENTITYOPERATOR_H
#define IDENTITYOPERATOR_H

#include "TensorNetworks/OperatorWRepresentation.H"

class IdentityOperator  : public virtual OperatorWRepresentation
{
    public:
        IdentityOperator();
        virtual ~IdentityOperator();

        virtual TensorNetworks::MatrixT GetW (TensorNetworks::Position,int m, int n) const;
        virtual TensorNetworks::ipairT  GetDw(TensorNetworks::Position) const;
};

#endif // IDENTITYOPERATOR_H
