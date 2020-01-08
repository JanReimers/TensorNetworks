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
    return 2;
}

//Operator::MatrixT IdentityOperator::GetW (Position lbr,int m, int n) const
//{
//    MatrixT W(1,1);
//    W(1,1)=m==n ? 1.0 : 0.0;
//    return W;
//}

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


//Operator::MatrixT IdentityOperator::GetW (Position lbr,int m, int n) const
//{
//    MatrixT W;
//    switch (lbr)
//    {
////
////  Implement W=[ 0, J/2*S-, J/2*S+, JSz, 1 ]
////
//    case Left:
//    {
//        W.SetLimits(1,GetDw() );
//        W(1,1)=0.0;
//        W(1,2)=0.0;
//        W(1,3)=0.0;
//        W(1,4)=0.0;
//        W(1,5)=1.0;
//    }
//    break;
////      [ 1    0      0    0   0 ]
////      [ S+   0      0    0   0 ]
////  W = [ S-   0      0    0   0 ]
////      [ Sz   0      0    0   0 ]
////      [ 0  J/2*S- J/2*S+ JSz 1 ]
////
//    case Bulk :
//    {
//        W.SetLimits(GetDw() ,GetDw() );
//        Fill(W,ElementT(0.0));
//        W(1,1)=1.0;
//        W(2,1)=0.0;
//        W(3,1)=0.0;
//        W(4,1)=0.0;
//        //W(5,1)=0.0;
//        W(5,2)=0.0;
//        W(5,3)=0.0;
//        W(5,4)=0.0; //The get return 2*Sz to avoid half integers
//        W(5,5)=1.0;
//    }
//    break;
////
////      [ 1  ]
////      [ S+ ]
////  W = [ S- ]
////      [ Sz ]
////      [ 0  ]
////
//    case  Right :
//    {
//
//        W.SetLimits(GetDw() ,1);
//        W(1,1)=1.0;
//        W(2,1)=0.0;
//        W(3,1)=0.0;
//        W(4,1)=0.0; //The get return 2*Sz to avoid half integers
//        W(5,1)=0.0;
//    }
//    break;
//    }
//    return W;
//}
//
////
