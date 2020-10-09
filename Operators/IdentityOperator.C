#include "Operators/IdentityOperator.H"
#include "oml/dmatrix.h"

namespace TensorNetworks
{

IdentityOperator::IdentityOperator()
 : itsDw12(1,1,Vector<int>(1),Vector<int>(1))
{
    itsDw12.w1_first(1)=1;
    itsDw12.w2_last (1)=1;
}

IdentityOperator::~IdentityOperator()
{
}

MatrixRT IdentityOperator::GetW (Position lbr,int m, int n) const
{
    MatrixRT W(1,1);
    W(1,1)=m==n ? 1.0 : 0.0;
    return W;
}

}
