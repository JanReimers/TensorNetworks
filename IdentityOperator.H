#ifndef IDENTITYOPERATOR_H
#define IDENTITYOPERATOR_H

#include "Operator.H"

class IdentityOperator  : public virtual Operator
{
    public:
        IdentityOperator();
        virtual ~IdentityOperator();
        virtual int GetDw() const;
        //
        //  Generate MPO matrices.  m,n are quantum numbers for <m|Ha|n>
        //
        virtual MatrixT GetW (Position lbr,int m, int n) const;

    protected:

    private:
};

#endif // IDENTITYOPERATOR_H
