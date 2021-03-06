#include "TensorNetworks/Hamiltonian.H"
#include "Containers/Matrix4.H"
#include "oml/numeric.h"

namespace TensorNetworks
{

MPO* Hamiltonian::CreateH2Operator  () const
{
    MPO* H2=CreateUnitOperator();
    H2->Product(this);
    H2->Product(this);
    H2->CanonicalForm();
    H2->Compress(Parker,0,1e-13);
    return H2;
}
Matrix4RT Hamiltonian::GetExponentH(double dt) const
{
    return GetExponentH(dt,GetLocalMatrix());
}

Matrix4RT Hamiltonian::GetExponentH(double dt,const Matrix4RT& H12)
{
    MatrixRT U12=H12.Flatten();
    int N=U12.GetNumRows();
    assert(N==U12.GetNumCols());
    int d=sqrt(N);
    assert(d*d==N);
    VectorRT evs=Diagonalize(U12);
    VectorRT expEvs=exp(-dt*evs);
    Matrix4RT expH(d,d,d,d,0);
    Fill(expH.Flatten(),0.0);
    int i1=1;
    for (int m1=0; m1<d; m1++)
        for (int m2=0; m2<d; m2++,i1++)
        {
            int i2=1;
            for (int n1=0; n1<d; n1++)
                for (int n2=0; n2<d; n2++,i2++)
                    for (int k=1; k<=N; k++)
                        expH(m1,n1,m2,n2)+=U12(i1,k)*expEvs(k)*U12(i2,k);
        }
    return expH;
}

}
