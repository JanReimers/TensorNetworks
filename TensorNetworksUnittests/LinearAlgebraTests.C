#include "Tests.H"

#include "TensorNetworksUnittests/LinearAlgebraTests.H"
#include "NumericalMethods/PrimeSVDSolver.H"
#include "NumericalMethods/PrimeEigenSolver.H"
#include "NumericalMethods/LapackSVD.H"
#include "NumericalMethods/LapackEigenSolver.H"
#include "NumericalMethods/ArpackEigenSolver.H"
#include "Containers/SparseMatrix.H"
//#include "oml/stream.h"
#include "oml/numeric.h"
#include "oml/cnumeric.h"
#include "oml/diagonalmatrix.h"

class LinearAlgebraTests : public ::testing::Test
{
public:
    typedef DMatrix<dcmplx> MatrixCT;
    typedef DMatrix<double> MatrixRT;
    typedef Vector <double> VectorRT;
    typedef DiagonalMatrix<double> DiagonalMatrixRT;
    typedef SparseMatrix<dcmplx> SparseMatrixCT;
    typedef SparseMatrix<double> SparseMatrixRT;

    LinearAlgebraTests()
    : itsEps()
    , eps(1.0e-13)
#ifdef DEBUG
    , Nsvd  (30)
    , Neigen(30)
#else
    , Nsvd  (100)
    , Neigen(200)
#endif
    , svdDensity(0.2)
    , eigenDensity(0.1)
    {
        StreamableObject::SetToPretty();
    }
    void SetupC(int M,int N)
    {
        int mn=Min(M,N);
        itsAC.SetLimits(M,N);
        itsIC.SetLimits(mn,mn);
        FillRandom(itsAC);
        Unit(itsIC);
    }
    void SetupH(int N)
    {
        MatrixCT A(N,N);
        itsWR.SetLimits(N);
        itsIC.SetLimits(N,N);
        FillRandom(A);
        itsAC=A+Transpose(conj(A)); //Make it hermitian
        Unit(itsIC);
    }

    SparseMatrixRT itsARs;
    SparseMatrixCT itsACs;
    MatrixCT  itsAC,itsIC;
    MatrixRT  itsAR,itsIR;
    VectorRT  itsWR; //Eigen values

    TensorNetworks::Epsilons  itsEps;
    double eps;
    int Nsvd,Neigen;
    double svdDensity;
    double eigenDensity;

};

TEST_F(LinearAlgebraTests,SparseMatrix)
{
    DMatrix<double> d(5,6);
    FillRandom(d);
    SparseMatrix<double> m=d;
//    cout << m << endl;
}


TEST_F(LinearAlgebraTests,Primme_SVDSolverDenseReal)
{
    SVDTester<double,DMatrix,PrimeSVDSolver>(Nsvd  ,Nsvd  ).RunTests();
    SVDTester<double,DMatrix,PrimeSVDSolver>(Nsvd/2,Nsvd  ).RunTests();
    SVDTester<double,DMatrix,PrimeSVDSolver>(Nsvd  ,Nsvd/2).RunTests();
}

TEST_F(LinearAlgebraTests,Primme_SVDSolverDenseComplex)
{
    SVDTester<dcmplx,DMatrix,PrimeSVDSolver>(Nsvd  ,Nsvd  ).RunTests();
    SVDTester<dcmplx,DMatrix,PrimeSVDSolver>(Nsvd/2,Nsvd  ).RunTests();
    SVDTester<dcmplx,DMatrix,PrimeSVDSolver>(Nsvd  ,Nsvd/2).RunTests();
}

TEST_F(LinearAlgebraTests,Primme_SVDSolverSparseReal)
{
    SVDTester<double,SparseMatrix,PrimeSVDSolver>(Nsvd  ,Nsvd  ,svdDensity).RunTests();
    SVDTester<double,SparseMatrix,PrimeSVDSolver>(Nsvd/2,Nsvd  ,svdDensity).RunTests();
    SVDTester<double,SparseMatrix,PrimeSVDSolver>(Nsvd  ,Nsvd/2,svdDensity).RunTests();
}
TEST_F(LinearAlgebraTests,Primme_SVDSolverSparseComplex)
{
    SVDTester<dcmplx,SparseMatrix,PrimeSVDSolver>(Nsvd  ,Nsvd  ,svdDensity).RunTests();
    SVDTester<dcmplx,SparseMatrix,PrimeSVDSolver>(Nsvd/2,Nsvd  ,svdDensity).RunTests();
    SVDTester<dcmplx,SparseMatrix,PrimeSVDSolver>(Nsvd  ,Nsvd/2,svdDensity).RunTests();
}

TEST_F(LinearAlgebraTests,Primme_EigenSolverDenseReal)
{
    SymEigenTester<double,DMatrix,PrimeEigenSolver>(Neigen).RunTests();
}
TEST_F(LinearAlgebraTests,Primme_EigenSolverDenseComplex)
{
    SymEigenTester<dcmplx,DMatrix,PrimeEigenSolver>(Neigen).RunTests();
}

TEST_F(LinearAlgebraTests,Primme_EigenSolverSparseReal)
{
    SymEigenTester<double,SparseMatrix,PrimeEigenSolver>(Neigen,eigenDensity).RunTests();
}

TEST_F(LinearAlgebraTests,Primme_EigenSolverSparseComplex)
{
    SymEigenTester<dcmplx,SparseMatrix,PrimeEigenSolver>(Neigen,eigenDensity).RunTests();
}

TEST_F(LinearAlgebraTests,Lapack_SVDSolverDenseReal)
{
    SVDTester<double,DMatrix,LapackSVDSolver>(Nsvd  ,Nsvd  ).RunTests();
    SVDTester<double,DMatrix,LapackSVDSolver>(Nsvd/2,Nsvd  ).RunTests();
    SVDTester<double,DMatrix,LapackSVDSolver>(Nsvd  ,Nsvd/2).RunTests();
}
TEST_F(LinearAlgebraTests,Lapack_SVDSolverDenseComplex)
{
    SVDTester<dcmplx,DMatrix,LapackSVDSolver>(Nsvd  ,Nsvd  ).RunTests();
    SVDTester<dcmplx,DMatrix,LapackSVDSolver>(Nsvd/2,Nsvd  ).RunTests();
    SVDTester<dcmplx,DMatrix,LapackSVDSolver>(Nsvd  ,Nsvd/2).RunTests();
}



TEST_F(LinearAlgebraTests,Lapack_EigenSolverDenseReal)
{
    SymEigenTester<double,DMatrix,LapackEigenSolver>(Neigen).RunTests();
}
TEST_F(LinearAlgebraTests,Lapack_EigenSolverDenseComplex)
{
    SymEigenTester<dcmplx,DMatrix,LapackEigenSolver>(Neigen).RunTests();
}

TEST_F(LinearAlgebraTests,Lapack_EigenSolverDenseNonSymReal)
{
    NonSymEigenTester<double,DMatrix,LapackEigenSolver>(Neigen).RunTests();
}
TEST_F(LinearAlgebraTests,Lapack_EigenSolverDenseNonSymComplex)
{
    NonSymEigenTester<dcmplx,DMatrix,LapackEigenSolver>(Neigen).RunTests();
}


TEST_F(LinearAlgebraTests,Arpack_EigenSolverDenseReal)
{
    NonSymEigenTester<double,DMatrix,ArpackEigenSolver>(Neigen).RunTests();
}

TEST_F(LinearAlgebraTests,Arpack_EigenSolverDenseComplex)
{
    NonSymEigenTester<dcmplx,DMatrix,ArpackEigenSolver>(Neigen).RunTests();
}

TEST_F(LinearAlgebraTests,Arpack_EigenSolverSparseReal)
{
    NonSymEigenTester<double,SparseMatrix,ArpackEigenSolver>(Neigen,eigenDensity).RunTests();
}

TEST_F(LinearAlgebraTests,Arpack_EigenSolverSparseComplex)
{
    NonSymEigenTester<dcmplx,SparseMatrix,ArpackEigenSolver>(Neigen,eigenDensity).RunTests();
}




TEST_F(LinearAlgebraTests,oml_SVDRandomComplexMatrix_10x10)
{
    SetupC(10,10);
    auto [U,s,Vdagger]=oml_CSVDecomp(itsAC); //Solve A=U*s*conj(V)

    MatrixCT V=Transpose(conj(Vdagger));
    EXPECT_NEAR(Max(fabs(Transpose(conj(U))*U-itsIC)),0.0,eps);
    EXPECT_NEAR(Max(fabs(V*Vdagger-itsIC)),0.0,eps);
    EXPECT_NEAR(Max(fabs(U*s*Vdagger-itsAC)),0.0,eps);
}


TEST_F(LinearAlgebraTests,oml_SVDRandomComplexMatrix_10x5)
{
    SetupC(10,5);
    auto [U,s,Vdagger]=oml_CSVDecomp(itsAC);

    MatrixCT V=Transpose(conj(Vdagger));
//    EXPECT_NEAR(Max(fabs(Transpose(conj(U))*U-itsIC)),0.0,eps); //Not true for 10x5
    EXPECT_NEAR(Max(fabs(V*Vdagger-itsIC)),0.0,eps);
    EXPECT_NEAR(Max(fabs(U*s*Vdagger-itsAC)),0.0,eps);
}


TEST_F(LinearAlgebraTests,oml_SVDRandomComplexMatrix_5x10)
{
    SetupC(5,10);
    auto [U,s,Vdagger]=oml_CSVDecomp(itsAC);

    MatrixCT V=Transpose(conj(Vdagger));
    EXPECT_NEAR(Max(fabs(Transpose(conj(U))*U-itsIC)),0.0,eps);
    //EXPECT_NEAR(Max(fabs(V*Vdagger-itsIC)),0.0,eps); Not true for 5x10
    EXPECT_NEAR(Max(fabs(U*s*Vdagger-itsAC)),0.0,eps);
}

TEST_F(LinearAlgebraTests,OML_EigenSolverComplexHermitian_oldUI)
{
    SetupH(50);
    MatrixCT Acopy(itsAC);
    int ierr=0;
    ch(itsAC, itsWR ,true,ierr);
    EXPECT_EQ(ierr,0);
    MatrixCT diag=Transpose(conj(itsAC))*Acopy*itsAC;
    for (int i=1;i<=itsWR.size();i++) diag(i,i)-=itsWR(i);
    EXPECT_NEAR(Max(fabs(diag)),0.0,Neigen*eps);
}

TEST_F(LinearAlgebraTests,OML_EigenSolverComplexHermitian)
{
    SetupH(50);
    auto [U,w]=oml_Diagonalize(itsAC);
    MatrixCT diag=Transpose(conj(U))*itsAC*U;
    for (int i=1;i<=w.size();i++) diag(i,i)-=w(i);
    EXPECT_NEAR(Max(fabs(diag)),0.0,Neigen*eps);
}

TEST_F(LinearAlgebraTests,omlDiagonalMatrix_double)
{
    int N=10;
    Vector<double> v(N);
    Fill(v,-1.);
    DiagonalMatrixRT d(v);
    MatrixRT M(N,N);
    FillRandom(M);
    MatrixRT Md=M*d;
    EXPECT_NEAR(Max(fabs(M+Md)),0.0,eps);
    MatrixRT dM=d*M;
    EXPECT_NEAR(Max(fabs(M+dM)),0.0,eps);
    MatrixRT dMdMdM=d*M*d*M*d*M;
    EXPECT_NEAR(Max(fabs(M*M*M+dMdMdM)),0.0,eps);

}

TEST_F(LinearAlgebraTests,omlDiagonalMatrix_complex)
{
    int N=10;
    Vector<dcmplx> v(N);
    Fill(v,dcmplx(-1.0));
    DiagonalMatrix<dcmplx> d(v);
    MatrixCT M(N,N);
    FillRandom(M);
    MatrixCT Md=M*d;
    EXPECT_NEAR(Max(fabs(M+Md)),0.0,eps);
    MatrixCT dM=d*M;
    EXPECT_NEAR(Max(fabs(M+dM)),0.0,eps);
    MatrixCT dMdMdM=d*M*d*M*d*M;
    EXPECT_NEAR(Max(fabs(M*M*M+dMdMdM)),0.0,eps);

}
TEST_F(LinearAlgebraTests,omlDiagonalMatrix_complex_double)
{
    int N=10;
    Vector<double> v(N);
    Fill(v,-1.0);
    DiagonalMatrix<double> d(v);
    MatrixCT M(N,N);
    FillRandom(M);
    MatrixCT Md=M*d;
    EXPECT_NEAR(Max(fabs(M+Md)),0.0,eps);
    MatrixCT dM=d*M;
    EXPECT_NEAR(Max(fabs(M+dM)),0.0,eps);
    MatrixCT dMdMdM=d*M*d*M*d*M;
    EXPECT_NEAR(Max(fabs(M*M*M+dMdMdM)),0.0,eps);

}


