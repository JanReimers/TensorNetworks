#include "Tests.H"

#include "TensorNetworksImp/MPSImp.H"
#include "TensorNetworks/Epsilons.H"
#include "TensorNetworks/LRPSupervisor.H"

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
    typedef TensorNetworks::MatrixCT MatrixT;
    SVDTesting()
    : itsEps()
    , mps(10,1,2,itsEps,new LRPSupervisor())
    , eps(1.0e-13)
    {
        StreamableObject::SetToPretty();
        mps.InitializeWith(TensorNetworks::Product);
    }


    Epsilons               itsEps;
    MPSImp mps;
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
    int nj=VT.GetNumCols();
    DMatrix<T2> Vs(nk,nj);
    for(int j=1;j<=nj;j++)
        for(int k=1;k<=nk;k++)
            Vs(k,j)=s(k)*VT(k,j);
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



TEST_F(SVDTesting,SVDComplexMatrix_10x10)
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

TEST_F(SVDTesting,OML_SVDRandomComplexMatrix_10x5)
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


TEST_F(SVDTesting,OML_SVDRandomComplexMatrix_5x10)
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


#include "Containers/SparseMatrix.H"

TEST_F(SVDTesting,SparseMatrixClass)
{
    int N=20;
    typedef DMatrix<eType> Mtype;
    Mtype A(N,N);
    FillRandom(A);
    for (int i=0;i<N*N;i++)
    {
        int ir=static_cast<int>(OMLRand<float>()*N)+1;
        int ic=static_cast<int>(OMLRand<float>()*N)+1;
        A(ir,ic)=0.0;
    }
    SparseMatrix<eType> sm(A,1e-12);
    cout << "Density=" << sm.GetDensity() << "%" << endl;
}

#include "TensorNetworksImp/PrimeEigenSolver.H"

TEST_F(SVDTesting,Prime_EigenSolverSparseComplexHermitian200x200)
{
    int N=200,Ne=10;
    typedef DMatrix<eType> Mtype;
    Mtype A(N,N);
    FillRandom(A);
    for (int i=0;i<1.0*N*N;i++)
    {
        int ir=static_cast<int>(OMLRand<float>()*N)+1;
        int ic=static_cast<int>(OMLRand<float>()*N)+1;
        A(ir,ic)=0.0;
    }
    Mtype Ah=A+Transpose(conj(A)); //Make it hermitian

    PrimeEigenSolver<eType> solver;
    itsEps.itsEigenSolverEpsilon=1e-4;
    solver.Solve(Ah,Ne,itsEps);
    itsEps.itsEigenSolverEpsilon=1e-6;
    solver.Solve(Ah,Ne,itsEps);
    itsEps.itsEigenSolverEpsilon=1e-8;
    solver.Solve(Ah,Ne,itsEps);
    itsEps.itsEigenSolverEpsilon=1e-11;
    solver.Solve(Ah,Ne,itsEps);
    itsEps.itsEigenSolverEpsilon=1e-14;
    solver.Solve(Ah,Ne,itsEps);

    Mtype diag=Transpose(conj(solver.GetEigenVectors()))*Ah*solver.GetEigenVectors();
    Vector<double> evals=solver.GetEigenValues();
    for (int i=1;i<=Ne;i++) diag(i,i)-=evals(i);
    EXPECT_NEAR(Max(abs(diag)),0.0,100*eps);
}

TEST_F(SVDTesting,Prime_EigenSolverDenseComplexHermitian200x200)
{
    int N=200,Ne=10;
    typedef DMatrix<eType> Mtype;
    Mtype A(N,N);
    FillRandom(A);
    for (int i=0;i<0.5*N*N;i++)
    {
        int ir=static_cast<int>(OMLRand<float>()*N)+1;
        int ic=static_cast<int>(OMLRand<float>()*N)+1;
        A(ir,ic)=0.0;
    }
    Mtype Ah=A+Transpose(conj(A)); //Make it hermitian

    PrimeEigenSolver<eType> solver;
    itsEps.itsEigenSolverEpsilon=1e-4;
    solver.Solve(Ah,Ne,itsEps);
    itsEps.itsEigenSolverEpsilon=1e-6;
    solver.Solve(Ah,Ne,itsEps);
    itsEps.itsEigenSolverEpsilon=1e-8;
    solver.Solve(Ah,Ne,itsEps);
    itsEps.itsEigenSolverEpsilon=1e-11;
    solver.Solve(Ah,Ne,itsEps);
    itsEps.itsEigenSolverEpsilon=1e-14;
    solver.Solve(Ah,Ne,itsEps);

    Mtype diag=Transpose(conj(solver.GetEigenVectors()))*Ah*solver.GetEigenVectors();
    Vector<double> evals=solver.GetEigenValues();
    for (int i=1;i<=Ne;i++) diag(i,i)-=evals(i);
    EXPECT_NEAR(Max(abs(diag)),0.0,100*eps);
}


TEST_F(SVDTesting,Prime_SVDComplex4004Matrix_1x4)
{
    int N1=1,N2=4;
    typedef DMatrix<eType> Mtype;
    Mtype M(N1,N2),V(N2,N1),UnitMatrix(N1,N1);
    Vector<double>  s(N1);
    M(1,1)=4.0;
    M(1,2)=0.0;
    M(1,3)=0.0;
    M(1,4)=4.0;
    Mtype Mcopy(M);
    Unit(UnitMatrix);
    CSVDecomp(M,s,V); //Solve M=U*s*VT
    Mtype VT=Transpose(V);

//    cout << "U S VT=" << M << " " << s << " " << VT << endl;
//    cout << "Mcopy=" << Mcopy << endl;
//    cout << "ConstractsVT(s,VT)=" << ConstractsVT(s,VT) << endl;
//    cout << "U*s*VT" <<  Mtype(M*ConstractsVT(s,VT))-Mcopy << endl;
    EXPECT_NEAR(Max(abs(Transpose(M)*M-UnitMatrix)),0.0,eps);
    EXPECT_NEAR(Max(abs(VT*V-UnitMatrix)),0.0,eps);
    EXPECT_NEAR(Max(abs(M*ConstractsVT(s,VT)-Mcopy)),0.0,eps);
}



#include "TensorNetworksImp/LapackSVD.H"

TEST_F(SVDTesting,LAPACK_SVDReal4004Matrix_1x4a)
{
    int M=1,N=4,mn=Min(M,N);
    typedef DMatrix<double> Mtype;
    Mtype A(M,N),VT(N,N),UnitMatrix(M,M);
    Vector<double>  s(mn);
    //FillRandom(A);
    A(1,1)=4.0;
    A(1,2)=0.0;
    A(1,3)=0.0;
    A(1,4)=4.0;
    Mtype Acopy(A);
    Unit(UnitMatrix);

    LaSVDecomp(A,s,VT);
    Mtype V=Transpose(VT);
    EXPECT_NEAR(Max(fabs(Transpose(A)*A-UnitMatrix)),0.0,eps);
    EXPECT_NEAR(Max(fabs(VT*V-UnitMatrix)),0.0,eps);
    EXPECT_NEAR(Max(fabs(A*ConstractsVT(s,VT)-Acopy)),0.0,eps);
}

TEST_F(SVDTesting,LAPACK_SVDRealRandomMatrix_1x4a)
{
    int M=1,N=4,mn=Min(M,N);
    typedef DMatrix<double> Mtype;
    Mtype A(M,N),VT(N,N),UnitMatrix(M,M);
    Vector<double>  s(mn);
    FillRandom(A);
    Mtype Acopy(A);
    Unit(UnitMatrix);

    LaSVDecomp(A,s,VT);
    Mtype V=Transpose(VT);
    EXPECT_NEAR(Max(fabs(Transpose(A)*A-UnitMatrix)),0.0,eps);
    EXPECT_NEAR(Max(fabs(VT*V-UnitMatrix)),0.0,eps);
    EXPECT_NEAR(Max(fabs(A*ConstractsVT(s,VT)-Acopy)),0.0,eps);
}

TEST_F(SVDTesting,LAPACK_SVDReal4004Matrix_4x1a)
{
    int M=4,N=1,mn=Min(M,N);
    typedef DMatrix<double> Mtype;
    Mtype A(M,N),VT(N,N),UnitMatrix(M,M);
    Vector<double>  s(mn);
    A(1,1)=4;
    A(2,1)=0;
    A(3,1)=0;
    A(4,1)=4;
    Mtype Acopy(A);
    Unit(UnitMatrix);

    LaSVDecomp(A,s,VT);
    Mtype V=Transpose(VT);
    EXPECT_NEAR(Max(fabs(Transpose(A)*A-UnitMatrix)),0.0,eps);
    EXPECT_NEAR(Max(fabs(VT*V-UnitMatrix)),0.0,eps);
    EXPECT_NEAR(Max(fabs(A*ConstractsVT(s,VT)-Acopy)),0.0,eps);
}




TEST_F(SVDTesting,LAPACK_SVDRealRandomMatrix_4x1a)
{
    int M=4,N=1,mn=Min(M,N);
    typedef DMatrix<double> Mtype;
    Mtype A(M,N),VT(N,N),UnitMatrix(M,M);
    Vector<double>  s(mn);
    FillRandom(A);
    Mtype Acopy(A);
    Unit(UnitMatrix);

    LaSVDecomp(A,s,VT);
    Mtype V=Transpose(VT);
    EXPECT_NEAR(Max(fabs(Transpose(A)*A-UnitMatrix)),0.0,eps);
    EXPECT_NEAR(Max(fabs(VT*V-UnitMatrix)),0.0,eps);
    EXPECT_NEAR(Max(fabs(A*ConstractsVT(s,VT)-Acopy)),0.0,eps);
}



TEST_F(SVDTesting,LAPACK_SVDRealRandomMatrix_200x3a)
{
    int M=200,N=3,mn=Min(M,N);
    typedef DMatrix<double> Mtype;
    Mtype A(M,N),VT(N,N),UnitMatrix(M,M);
    Vector<double>  s(mn);
    FillRandom(A);
    Mtype Acopy(A);
    Unit(UnitMatrix);

    LaSVDecomp(A,s,VT);
    Mtype V=Transpose(VT);
    EXPECT_NEAR(Max(fabs(Transpose(A)*A-UnitMatrix)),0.0,eps);
    EXPECT_NEAR(Max(fabs(VT*V-UnitMatrix)),0.0,eps);
    EXPECT_NEAR(Max(fabs(A*ConstractsVT(s,VT)-Acopy)),0.0,eps);
}

TEST_F(SVDTesting,LAPACK_SVDRealRandomMatrix_3x200a)
{
    int M=3,N=200,mn=Min(M,N);
    typedef DMatrix<double> Mtype;
    Mtype A(M,N),VT(N,N),UnitMatrix(M,M);
    Vector<double>  s(mn);
    FillRandom(A);
    Mtype Acopy(A);
    Unit(UnitMatrix);

    LaSVDecomp(A,s,VT);
    Mtype V=Transpose(VT);
    EXPECT_NEAR(Max(fabs(Transpose(A)*A-UnitMatrix)),0.0,eps);
    EXPECT_NEAR(Max(fabs(VT*V-UnitMatrix)),0.0,eps);
    EXPECT_NEAR(Max(fabs(A*ConstractsVT(s,VT)-Acopy)),0.0,eps);
}

TEST_F(SVDTesting,LAPACK_SVDRealRandomMatrix_100x100)
{
    int M=100,N=100,mn=Min(M,N);
    typedef DMatrix<double> Mtype;
    Mtype A(M,N),VT(N,N),UnitMatrix(M,M);
    Vector<double>  s(mn);
    FillRandom(A);
    Mtype Acopy(A);
    Unit(UnitMatrix);

    LaSVDecomp(A,s,VT);
    Mtype V=Transpose(VT);
    EXPECT_NEAR(Max(fabs(Transpose(A)*A-UnitMatrix)),0.0,eps);
    EXPECT_NEAR(Max(fabs(A*Transpose(A)-UnitMatrix)),0.0,eps);
    EXPECT_NEAR(Max(fabs(VT*V-UnitMatrix)),0.0,eps);
    EXPECT_NEAR(Max(fabs(V*VT-UnitMatrix)),0.0,eps);
    EXPECT_NEAR(Max(fabs(A*ConstractsVT(s,VT)-Acopy)),0.0,eps);
}



