#include "Operators/OperatorClient.H"
#include "Containers/Matrix4.H"

namespace TensorNetworks
{

Matrix4RT OperatorClient::GetH12 () const
{
    int d=2*GetS()+1;
    Matrix4RT H12(d,d,d,d,0);
    for (int n1=0;n1<d;n1++)
        for (int n2=0;n2<d;n2++)
            for (int m1=0;m1<d;m1++)
                for (int m2=0;m2<d;m2++)
                    H12(m1,m2,n1,n2)=GetH(m1,n1,m2,n2);
    return H12;
}

Matrix4RT OperatorClient1::GetH12 () const
{
    int d=2*GetS()+1;
    Matrix4RT H12(d,d,d,d,0);
    for (int n1=0;n1<d;n1++)
        for (int n2=0;n2<d;n2++)
            for (int m1=0;m1<d;m1++)
                for (int m2=0;m2<d;m2++)
                    H12(m1,m2,n1,n2)=GetH(m1,n1,m2,n2);
    return H12;
}


} //namespace

