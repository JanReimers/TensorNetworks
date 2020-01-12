#include "IdentityOperator.H"

IdentityOperator::IdentityOperator()
{
    //ctor
}

IdentityOperator::~IdentityOperator()
{
    //dtor
}

int IdentityOperator::GetDw() const
{
    return 1;
}

Operator::MatrixT IdentityOperator::GetW (Position lbr,int m, int n) const
{
    MatrixT W(1,1);
    W(1,1)=m==n ? 1.0 : 0.0;
    return W;
}

/*
Operator::MatrixT IdentityOperator::GetW (Position lbr,int m, int n) const
{
    MatrixT W;
    switch (lbr)
    {
        case Left:
        {
            W.SetLimits(1,GetDw() );
            W(1,1)=0.0;
            W(1,2)=1.0;
            break;
        }
        case Bulk:
        {
            W.SetLimits(GetDw(),GetDw() );
            W(1,1)=1.0;
            W(1,2)=0.0;
            W(2,1)=0.0;
            W(2,2)=1.0;
            break;
        }
        case Right:
        {
            W.SetLimits(GetDw(),1 );
            W(1,1)=1.0;
            W(2,1)=0.0;
            break;
        }
    }
    return W;
}
*/

