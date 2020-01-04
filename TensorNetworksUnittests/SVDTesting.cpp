#include "Tests.H"

#include "TensorNetworks/MatrixProductState.H"
#include "oml/stream.h"
#include "oml/numeric.h"
#include "oml/cnumeric.h"
#include "oml/random.h"
#include "oml/vector_io.h"
#include "oml/array_io.h"
#include <complex>
#include <iostream>

using std::cout;
using std::endl;

class SVDTesting : public ::testing::Test
{
public:
    typedef MatrixProductSite::MatrixT MatrixT;
    SVDTesting()
    : mps(10,1,2)
  //  , itsSites(mps.itsSites)
    , eps(1.0e-13)
    {
        StreamableObject::SetToPretty();
        mps.InitializeWithProductState();
    }
 //   const MatrixT& GetA(int i,int ip) const {return itsSites[i]->itsAs[ip]; }


    MatrixProductState mps;
//    MatrixProductState::SitesType& itsSites;
    double eps;
};


template <class T,class T2>
DMatrix<T2> contract(const DMatrix<T2> U, const Vector<T>& s, const DMatrix<T2>& V)
{
    assert(U.GetNumCols()==V.GetNumCols());
    assert(U.GetNumCols()==s.GetHigh());
    int ni=U.GetNumRows();
    int nk=U.GetNumCols();
    int nj=V.GetNumRows();
    DMatrix<T2> M(ni,nj);
    Fill(M,T2(0.0));
    for(int i=1;i<=ni;i++)
        for(int j=1;j<=nj;j++)
            for(int k=1;k<=nk;k++)
            {
                M(i,j)+=U(i,k)*s(k)*V(j,k);
            }
    return M;
}

template <class T,class T2>
DMatrix<T2> ConstractsVT(const Vector<T>& s, const DMatrix<T2>& VT)
{
    assert(VT.GetNumRows()==s.GetHigh());
    int nk=s.GetHigh();
    int nj=VT.GetNumRows();
    DMatrix<T2> Vs(nk,nj);
    for(int j=1;j<=nj;j++)
        for(int k=1;k<=nk;k++)
            Vs(k,j)=s(k)*VT(j,k);
    return Vs;
}
template <class T,class T2>
DMatrix<T2> ContractVstar(const Vector<T>& s, const DMatrix<T2>& Vstar)
{
    assert(Vstar.GetNumCols()==s.GetHigh());
    int nk=s.GetHigh();
    int nj=Vstar.GetNumRows();
    DMatrix<T2> Vs(nk,nj);
    for(int j=1;j<=nj;j++)
        for(int k=1;k<=nk;k++)
            Vs(k,j)=s(k)*Vstar(j,k);
    return Vs;
}


TEST_F(SVDTesting,OML_SVDRandomSquareRealMatrix)
{
    int N=10;
    typedef DMatrix<double> Mtype;
    Mtype M(N,N),VT(N,N),UnitMatrix(N,N);
    Vector<double>  s(N);
    FillRandom(M);
    Mtype Mcopy(M);
    Unit(UnitMatrix);
    SVDecomp(M,s,VT); //Solve M=U*s*VT
    Mtype V=Transpose(VT);
    EXPECT_NEAR(Max(fabs(Transpose(M)*M-UnitMatrix)),0.0,eps);
    EXPECT_NEAR(Max(fabs(M*Transpose(M)-UnitMatrix)),0.0,eps);
    EXPECT_NEAR(Max(fabs(V*VT-UnitMatrix)),0.0,eps);
    EXPECT_NEAR(Max(fabs(VT*V-UnitMatrix)),0.0,eps);
    EXPECT_NEAR(Max(fabs(M*ConstractsVT(s,VT)-Mcopy)),0.0,eps);
//    EXPECT_NEAR(Max(fabs(M*V-Mcopy)),0.0,eps);
}

TEST_F(SVDTesting,OML_SVDRandomRectRealMatrix_10x5)
{
    int N1=10,N2=5;
    typedef DMatrix<double> Mtype;
    Mtype M(N1,N2),VT(N2,N2),UnitMatrix(N1,N1);
    Vector<double>  s(N2);
    FillRandom(M);
    Mtype Mcopy(M);
    Unit(UnitMatrix);
    SVDecomp(M,s,VT); //Solve M=U*s*VT
    Mtype V=Transpose(VT);
    EXPECT_NEAR(Max(fabs(Transpose(M)*M-UnitMatrix)),0.0,eps);
    EXPECT_NEAR(Max(fabs(VT*V-UnitMatrix)),0.0,eps);
    EXPECT_NEAR(Max(fabs(V*VT-UnitMatrix)),0.0,eps);
    EXPECT_NEAR(Max(fabs(M*ConstractsVT(s,VT)-Mcopy)),0.0,eps);
}


TEST_F(SVDTesting,OML_SVDRandomRectRealMatrix_5x10)
{
    int N1=5,N2=10;
    typedef DMatrix<double> Mtype;
    Mtype M(N1,N2),VT(N2,N2),UnitMatrix(N1,N1);
    Vector<double>  s(N2);
    FillRandom(M);
    Mtype Mcopy(M);
    Unit(UnitMatrix);
    SVDecomp(M,s,VT); //Solve M=U*s*VT
    Mtype V=Transpose(VT);
    EXPECT_NEAR(Max(fabs(Transpose(M)*M-UnitMatrix)),0.0,eps);
    EXPECT_NEAR(Max(fabs(VT*V-UnitMatrix)),0.0,eps);
    EXPECT_NEAR(Max(fabs(M*ConstractsVT(s,VT)-Mcopy)),0.0,eps);
}



//The eigen header file 'lib' is another option for later.  It takes a long time to complile.
/*#include <Eigen/Dense>
#include "oml/random.h"

TEST_F(SVDTesting,OML_SVDRandomSquareComplexMatrix)
{
    using namespace Eigen;
    int N=5;
    MatrixXd m(N,N);
    for (int i=0;i<N;i++)
        for (int j=0;i<N;i++)
            m(i,j)=OMLRandPos<double>();
    BDCSVD< MatrixXd > svd(m);
    std::cout << m << std::endl;
}
*/


TEST_F(SVDTesting,SVDComplexSquare_N10)
{
    int N=10;
    typedef DMatrix<eType> Mtype;
    Mtype A(N,N),V(N,N),UnitMatrix(N,N);
    Vector<double>  s(N);
    FillRandom(A);
    Mtype Mcopy(A);
    Unit(UnitMatrix);
    CSVDecomp(A,s,V); //Solve A=U*s*conj(V)
    Mtype Vdagger=Transpose(conj(V));
    EXPECT_NEAR(Max(abs(Transpose(conj(A))*A-UnitMatrix)),0.0,eps);
    EXPECT_NEAR(Max(abs(V*Vdagger-UnitMatrix)),0.0,eps);
    Mtype Vstar=conj(V);
    EXPECT_NEAR(Max(abs(A*ContractVstar(s,Vstar)-Mcopy)),0.0,eps);
    EXPECT_NEAR(Max(abs(contract(A,s,Vstar)-Mcopy)),0.0,eps);

}

TEST_F(SVDTesting,OML_SVDRandomRectComplexMatrix_10x5)
{
    int M=10,N=5;
    typedef DMatrix<eType> Mtype;
    Mtype A(M,N),V(N,N),UnitMatrix(N,N);
    Vector<double>  s(N);
    FillRandom(A);
    Mtype Mcopy(A);
    Unit(UnitMatrix);
    CSVDecomp(A,s,V);
    EXPECT_NEAR(Max(abs(Transpose(conj(A))*A-UnitMatrix)),0.0,eps);
    EXPECT_NEAR(Max(abs(V*Transpose(conj(V))-UnitMatrix)),0.0,eps);
    Mtype Vstar=conj(V);
    EXPECT_NEAR(Max(abs(contract(A,s,Vstar)-Mcopy)),0.0,eps);
    EXPECT_NEAR(Max(abs(A*ContractVstar(s,Vstar)-Mcopy)),0.0,eps);
}


TEST_F(SVDTesting,OML_SVDRandomRectComplexMatrix_5x10)
{
    int M=5,N=10;
    typedef DMatrix<eType> Mtype;
    Mtype A(M,N),V(M,N),UnitMatrix(M,M);
    Vector<double>  s(M);
    FillRandom(A);
    Mtype Mcopy(A);
    Unit(UnitMatrix);
    CSVDecomp(A,s,V);
    Mtype Vstar=conj(V);
    Mtype Vdagger=Transpose(Vstar);
    EXPECT_NEAR(Max(abs(Transpose(conj(A))*A-UnitMatrix)),0.0,eps);
    EXPECT_NEAR(Max(abs(Vdagger*V-UnitMatrix)),0.0,eps);
    EXPECT_NEAR(Max(abs(contract(A,s,Vstar)-Mcopy)),0.0,eps);
    EXPECT_NEAR(Max(abs(A*ContractVstar(s,Vstar)-Mcopy)),0.0,eps);
}

TEST_F(SVDTesting,OML_EigenSolverComplexHermitian)
{
    int N=50;
    typedef DMatrix<eType> Mtype;
    Mtype A(N,N);//,V(M,N),UnitMatrix(M,M);
    Vector<double>  w(N);
    FillRandom(A);
    Mtype Ah=A+Transpose(conj(A)); //Make it hermitian
    Mtype Mcopy(Ah);
    int ierr=0;
    ch(Ah, w ,true,ierr);
    EXPECT_EQ(ierr,0);
    Mtype diag=Transpose(conj(Ah))*Mcopy*Ah;
    for (int i=1;i<=N;i++) diag(i,i)-=w(i);
    EXPECT_NEAR(Max(abs(diag)),0.0,eps);
}
