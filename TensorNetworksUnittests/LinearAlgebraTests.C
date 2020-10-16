#include "Tests.H"

#include "NumericalMethods/PrimeSVDSolver.H"
#include "NumericalMethods/PrimeEigenSolver.H"
#include "NumericalMethods/ArpackEigenSolver.H"
#include "Containers/SparseMatrix.H"
#include "oml/stream.h"
#include "oml/numeric.h"
#include "oml/cnumeric.h"
#include "oml/random.h"
#include "oml/diagonalmatrix.h"
#include <iostream>

using std::cout;
using std::endl;

typedef std::complex<double> dcmplx;

template <typename T> DMatrix<double> MyAbs(const DMatrix<T> &);
template <> DMatrix<double> MyAbs(const DMatrix<double> &m)
{
    return fabs(m);
};

template <> DMatrix<double> MyAbs(const DMatrix<dcmplx> &m)
{
    return abs(m);
};

const DMatrix<double>& conj(const DMatrix<double>& m) {return m;}

template <typename T, template <typename> class Mat> class Tester
{
public:
    Tester(int M, int N,double Density)
    : A(M,N), eps(1e-13)
    {
        Fill(Density);
    }

    void Fill(double TargetDensity)
    {
        DMatrix<T> temp(A.GetLimits());
        FillRandom(temp);
        int Ntotal=temp.size(),N=temp.size();
        int Nr=A.GetNumRows(),Nc=A.GetNumCols();
        double density=1.0;
        while (density>TargetDensity)
        {
            int ir=static_cast<int>(OMLRand<float>()*Nr)+1;
            int ic=static_cast<int>(OMLRand<float>()*Nc)+1;
            if (temp(ir,ic)!=0.0)
            {
                temp(ir,ic)=0.0;
                N--;
                density=static_cast<double>(N)/Ntotal;
            }

        }
        A=temp;
    }

    Mat<T> A;
    double eps;
};


template <typename T, template <typename> class Mat> class SVDTester : public Tester<T,Mat>
{
public:
    using Tester<T,Mat>::A;
    using Tester<T,Mat>::eps;
    SVDTester(int M, int N,double Density=1.0)
    : Tester<T,Mat>(M,N,Density)
    {
        auto [U,s,VT]=solver.Solve(A,Min(M,N),eps);
        DMatrix<T> A1=U*s*VT;
        DMatrix<T> dA=A-A1;
        EXPECT_NEAR(Max(MyAbs(dA)),0.0,10*eps);
    }

    PrimeSVDSolver<T> solver;
};

template <typename T, template <typename> class Mat> class EigenTester : public Tester<T,Mat>
{
public:
    using Tester<T,Mat>::A;
    using Tester<T,Mat>::eps;

    EigenTester(int N,double Density=1.0)
    : Tester<T,Mat>(N,N,Density),I(N,N)
    {
        A=Mat<T>(A+Transpose(conj(A)));
        Unit(I);
        int Ne=N/2;
        auto [U,d]=solver.Solve(A,Ne,eps);
        EXPECT_NEAR(Max(abs(Transpose(conj(U))*U-I)),0.0,100*eps);
        DMatrix<T> diag=Transpose(conj(U))*A*U;
        for (int i=1;i<=Ne;i++) diag(i,i)-=d(i);
        EXPECT_NEAR(Max(abs(diag)),0.0,100*eps);
    }

    PrimeEigenSolver<T> solver;
    DMatrix<T> I;
};

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
    , Nsvd(30)
    , Neigen(10)
    , svdDensity(0.1)
    {
        StreamableObject::SetToPretty();
    }

    void SetupR(int N)
    {
        SetupR(N,N);
    }
    void SetupR(int M,int N)
    {
        int mn=Min(M,N);
        itsAR.SetLimits(M,N);
        itsIR.SetLimits(mn,mn);
        FillRandom(itsAR);
        Unit(itsIR);
    }
    void SetupSym(int N)
    {
        MatrixRT A(N,N);
        itsWR.SetLimits(N);
        itsAR.SetLimits(N,N);
        FillRandom(A);
        itsAR=A+Transpose(A); //Make it hermitian
        Unit(itsIR);
    }
    void SetupSparseSym(int N)
    {
        MatrixRT A(N,N);
        FillRandom(A);
        for (int i=0;i<1.0*N*N;i++)
        {
            int ir=static_cast<int>(OMLRand<float>()*N)+1;
            int ic=static_cast<int>(OMLRand<float>()*N)+1;
            A(ir,ic)=0.0;
        }
        itsAR=A+Transpose(A); //Make it hermitian
    }

    void SetupSparseR(int M,int N)
    {
        MatrixRT A(M,N);
        FillRandom(A);
        for (int i=0;i<0.5*log(N*M)*N*M;i++)
        {
            int ir=static_cast<int>(OMLRand<float>()*M)+1;
            int ic=static_cast<int>(OMLRand<float>()*N)+1;
            A(ir,ic)=0.0;
        }
        itsARs=A; //Make it hermitian
        cout << "Density=" << itsARs.GetDensity() << "%" << endl;
    }
    void SetupSparseC(int M,int N)
    {
        MatrixCT A(M,N);
        FillRandom(A);
        for (int i=0;i<0.5*log(N*M)*N*M;i++)
        {
            int ir=static_cast<int>(OMLRand<float>()*M)+1;
            int ic=static_cast<int>(OMLRand<float>()*N)+1;
            A(ir,ic)=0.0;
        }
        itsACs=A; //Make it hermitian
        cout << "Density=" << itsACs.GetDensity() << "%" << endl;
    }

    void SetupC(int N)
    {
        SetupC(N,N);
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
    void SetupSparseH(int N)
    {
        MatrixCT A(N,N);
        FillRandom(A);
        for (int i=0;i<1.0*N*N;i++)
        {
            int ir=static_cast<int>(OMLRand<float>()*N)+1;
            int ic=static_cast<int>(OMLRand<float>()*N)+1;
            A(ir,ic)=0.0;
        }
        itsAC=A+Transpose(conj(A)); //Make it hermitian

    }
    //
    // This trivial matrix cuased horrendous problems for the numerical recipes solver
    // So we test the crap out of it here.
    //
    template <class T> void Load4004(DMatrix<T>& M)
    {
        if (M.GetNumRows()==1)
        {
            assert(M.GetNumCols()==4);
            M(1,1)=4.0;
            M(1,2)=0.0;
            M(1,3)=0.0;
            M(1,4)=4.0;
        }
        else if (M.GetNumCols()==1)
        {
            assert(M.GetNumRows()==4);
            M(1,1)=4.0;
            M(2,1)=0.0;
            M(3,1)=0.0;
            M(4,1)=4.0;
        }
        else
        {
            assert(false);
        }
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

};

TEST_F(LinearAlgebraTests,Primme_SVDSolverDenseReal)
{
    SVDTester<double,DMatrix>(Nsvd  ,Nsvd  );
    SVDTester<double,DMatrix>(Nsvd/2,Nsvd  );
    SVDTester<double,DMatrix>(Nsvd  ,Nsvd/2);
}
TEST_F(LinearAlgebraTests,Primme_SVDSolverDenseComplex)
{
    SVDTester<dcmplx,DMatrix>(Nsvd  ,Nsvd  );
    SVDTester<dcmplx,DMatrix>(Nsvd/2,Nsvd  );
    SVDTester<dcmplx,DMatrix>(Nsvd  ,Nsvd/2);
}

TEST_F(LinearAlgebraTests,Primme_SVDSolverSparseReal)
{
    SVDTester<double,SparseMatrix>(Nsvd  ,Nsvd  ,svdDensity);
    SVDTester<double,SparseMatrix>(Nsvd/2,Nsvd  ,svdDensity);
    SVDTester<double,SparseMatrix>(Nsvd  ,Nsvd/2,svdDensity);
}
TEST_F(LinearAlgebraTests,Primme_SVDSolverSparseComplex)
{
    SVDTester<dcmplx,SparseMatrix>(Nsvd  ,Nsvd  ,svdDensity);
    SVDTester<dcmplx,SparseMatrix>(Nsvd/2,Nsvd  ,svdDensity);
    SVDTester<dcmplx,SparseMatrix>(Nsvd  ,Nsvd/2,svdDensity);
}

TEST_F(LinearAlgebraTests,Primme_EigenSolverDenseReal)
{
    EigenTester<double,DMatrix>(Neigen,1.0);
}
TEST_F(LinearAlgebraTests,Primme_EigenSolverDenseComplex)
{
    EigenTester<dcmplx,DMatrix>(Neigen,1.0);
}
/*


TEST_F(LinearAlgebraTests,oml_SVDRandomComplexMatrix_10x10)
{
    SetupC(10);
    auto [U,s,Vdagger]=oml_CSVDecomp(itsAC); //Solve A=U*s*conj(V)

    MatrixCT V=Transpose(conj(Vdagger));
    EXPECT_NEAR(Max(abs(Transpose(conj(U))*U-itsIC)),0.0,eps);
    EXPECT_NEAR(Max(abs(V*Vdagger-itsIC)),0.0,eps);
    EXPECT_NEAR(Max(abs(U*s*Vdagger-itsAC)),0.0,eps);
}


TEST_F(LinearAlgebraTests,oml_SVDRandomComplexMatrix_10x5)
{
    SetupC(10,5);
    auto [U,s,Vdagger]=oml_CSVDecomp(itsAC);

    MatrixCT V=Transpose(conj(Vdagger));
//    EXPECT_NEAR(Max(abs(Transpose(conj(U))*U-itsIC)),0.0,eps); //Not true for 10x5
    EXPECT_NEAR(Max(abs(V*Vdagger-itsIC)),0.0,eps);
    EXPECT_NEAR(Max(abs(U*s*Vdagger-itsAC)),0.0,eps);
}


TEST_F(LinearAlgebraTests,oml_SVDRandomComplexMatrix_5x10)
{
    SetupC(5,10);
    auto [U,s,Vdagger]=oml_CSVDecomp(itsAC);

    MatrixCT V=Transpose(conj(Vdagger));
    EXPECT_NEAR(Max(abs(Transpose(conj(U))*U-itsIC)),0.0,eps);
    //EXPECT_NEAR(Max(abs(V*Vdagger-itsIC)),0.0,eps); Not true for 5x10
    EXPECT_NEAR(Max(abs(U*s*Vdagger-itsAC)),0.0,eps);
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
    EXPECT_NEAR(Max(abs(diag)),0.0,2*eps);
}

TEST_F(LinearAlgebraTests,OML_EigenSolverComplexHermitian)
{
    SetupH(50);
    auto [U,w]=oml_Diagonalize(itsAC);
    MatrixCT diag=Transpose(conj(U))*itsAC*U;
    for (int i=1;i<=w.size();i++) diag(i,i)-=w(i);
    EXPECT_NEAR(Max(abs(diag)),0.0,2*eps);
}


TEST_F(LinearAlgebraTests,SparseMatrixClass)
{
    SetupSparseH(20);
    SparseMatrix<dcmplx> sm(itsAC,1e-12);
//    cout << "Density=" << sm.GetDensity() << "%" << endl;
}


TEST_F(LinearAlgebraTests,Primme_EigenSolverSparseRealSymmetric200x200)
{
    int Ne=10; //Number of eigenvector/values to calculate.
    SetupSparseSym(200);

    PrimeEigenSolver<double> solver;
    itsEps.itsEigenSolverEpsilon=1e-4;
    solver.Solve(itsAR,Ne,itsEps);
    itsEps.itsEigenSolverEpsilon=1e-6; //Previous solution at eps=1e-4 should be cached and used as a starting point.
    solver.Solve(itsAR,Ne,itsEps);
    itsEps.itsEigenSolverEpsilon=1e-8;
    solver.Solve(itsAR,Ne,itsEps);
    itsEps.itsEigenSolverEpsilon=1e-11;
    solver.Solve(itsAR,Ne,itsEps);
    itsEps.itsEigenSolverEpsilon=1e-14;
    solver.Solve(itsAR,Ne,itsEps);

    MatrixRT diag=Transpose(solver.GetEigenVectors())*itsAR*solver.GetEigenVectors();
    itsWR=solver.GetEigenValues();
    for (int i=1;i<=Ne;i++) diag(i,i)-=itsWR(i);
    EXPECT_NEAR(Max(abs(diag)),0.0,100*eps);
}

TEST_F(LinearAlgebraTests,Primme_EigenSolverDenseRealSymmetric200x200)
{
    int Ne=10;
    SetupSym(200);
    PrimeEigenSolver<double> solver;
    itsEps.itsEigenSolverEpsilon=1e-4;
    solver.Solve(itsAR,Ne,itsEps);
    itsEps.itsEigenSolverEpsilon=1e-6;
    solver.Solve(itsAR,Ne,itsEps);
    itsEps.itsEigenSolverEpsilon=1e-8;
    solver.Solve(itsAR,Ne,itsEps);
    itsEps.itsEigenSolverEpsilon=1e-11;
    solver.Solve(itsAR,Ne,itsEps);
    itsEps.itsEigenSolverEpsilon=1e-14;
    solver.Solve(itsAR,Ne,itsEps);

    MatrixRT diag=Transpose(solver.GetEigenVectors())*itsAR*solver.GetEigenVectors();
    MatrixRT UU=Transpose(solver.GetEigenVectors())*solver.GetEigenVectors();
    itsWR=solver.GetEigenValues();
    for (int i=1;i<=Ne;i++) diag(i,i)-=itsWR(i);
    EXPECT_NEAR(Max(abs(diag)),0.0,100*eps);
}

TEST_F(LinearAlgebraTests,Primme_EigenSolverSparseComplexHermitian200x200)
{
    int Ne=10; //Number of eigenvector/values to calculate.
    SetupSparseH(200);

    PrimeEigenSolver<dcmplx> solver;
    itsEps.itsEigenSolverEpsilon=1e-4;
    solver.Solve(itsAC,Ne,itsEps);
    itsEps.itsEigenSolverEpsilon=1e-6; //Previous solution at eps=1e-4 should be cached and used as a starting point.
    solver.Solve(itsAC,Ne,itsEps);
    itsEps.itsEigenSolverEpsilon=1e-8;
    solver.Solve(itsAC,Ne,itsEps);
    itsEps.itsEigenSolverEpsilon=1e-11;
    solver.Solve(itsAC,Ne,itsEps);
    itsEps.itsEigenSolverEpsilon=1e-14;
    solver.Solve(itsAC,Ne,itsEps);

    MatrixCT diag=Transpose(conj(solver.GetEigenVectors()))*itsAC*solver.GetEigenVectors();
    itsWR=solver.GetEigenValues();
    for (int i=1;i<=Ne;i++) diag(i,i)-=itsWR(i);
    EXPECT_NEAR(Max(abs(diag)),0.0,100*eps);
}
*/

/*

#ifndef DEBUG
TEST_F(LinearAlgebraTests,Primme_EigenSolverDenseComplexHermitian200x200)
{
    int Ne=10;
    SetupH(200);
    PrimeEigenSolver<dcmplx> solver;
    itsEps.itsEigenSolverEpsilon=1e-4;
    solver.Solve(itsAC,Ne,itsEps);
    itsEps.itsEigenSolverEpsilon=1e-6;
    solver.Solve(itsAC,Ne,itsEps);
    itsEps.itsEigenSolverEpsilon=1e-8;
    solver.Solve(itsAC,Ne,itsEps);
    itsEps.itsEigenSolverEpsilon=1e-11;
    solver.Solve(itsAC,Ne,itsEps);
    itsEps.itsEigenSolverEpsilon=1e-14;
    solver.Solve(itsAC,Ne,itsEps);

    MatrixCT diag=Transpose(conj(solver.GetEigenVectors()))*itsAC*solver.GetEigenVectors();
    itsWR=solver.GetEigenValues();
    for (int i=1;i<=Ne;i++) diag(i,i)-=itsWR(i);
    EXPECT_NEAR(Max(abs(diag)),0.0,100*eps);
}

#endif // DEBUG

TEST_F(LinearAlgebraTests,oml_SVDComplex4004Matrix_1x4)
{
    SetupC(1,4);
    Load4004(itsAC);
    auto [U,s,VT]=oml_CSVDecomp(itsAC); //Solve M=U*s*VT
    MatrixCT V=Transpose(VT);
    EXPECT_NEAR(Max(abs(Transpose(U)*U-itsIC)),0.0,eps);
    EXPECT_NEAR(Max(abs(VT*V-itsIC)),0.0,eps);
    EXPECT_NEAR(Max(abs(U*s*VT-itsAC)),0.0,eps);
}



#include "NumericalMethods/LapackSVD.H"

TEST_F(LinearAlgebraTests,LAPACK_SVDReal4004Matrix_1x4a)
{
    SetupR(1,4);
    Load4004(itsAR);
    auto [U,s,VT]=LaSVDecomp(itsAR); //Solve M=U*s*VT
    EXPECT_NEAR(Max(abs(Transpose(U)*U-itsIR)),0.0,eps);
    EXPECT_NEAR(Max(abs(VT*Transpose(VT)-itsIR)),0.0,eps);
    EXPECT_NEAR(Max(abs(U*s*VT-itsAR)),0.0,eps);
}

TEST_F(LinearAlgebraTests,LAPACK_SVDRealRandomMatrix_1x4a)
{
    SetupR(1,4);
    auto [U,s,VT]=LaSVDecomp(itsAR); //Solve M=U*s*VT
    EXPECT_NEAR(Max(abs(Transpose(U)*U-itsIR)),0.0,eps);
    EXPECT_NEAR(Max(abs(VT*Transpose(VT)-itsIR)),0.0,eps);
    EXPECT_NEAR(Max(abs(U*s*VT-itsAR)),0.0,eps);
}

TEST_F(LinearAlgebraTests,LAPACK_SVDReal4004Matrix_4x1a)
{
    SetupR(4,1);
    Load4004(itsAR);
    auto [U,s,VT]=LaSVDecomp(itsAR); //Solve M=U*s*VT
    EXPECT_NEAR(Max(abs(Transpose(U)*U-itsIR)),0.0,eps);
    EXPECT_NEAR(Max(abs(VT*Transpose(VT)-itsIR)),0.0,eps);
    EXPECT_NEAR(Max(abs(U*s*VT-itsAR)),0.0,eps);
}


TEST_F(LinearAlgebraTests,LAPACK_SVDRealRandomMatrix_4x1a)
{
    SetupR(4,1);
    auto [U,s,VT]=LaSVDecomp(itsAR); //Solve M=U*s*VT
    EXPECT_NEAR(Max(abs(Transpose(U)*U-itsIR)),0.0,eps);
    EXPECT_NEAR(Max(abs(VT*Transpose(VT)-itsIR)),0.0,eps);
    EXPECT_NEAR(Max(abs(U*s*VT-itsAR)),0.0,eps);
}



TEST_F(LinearAlgebraTests,LAPACK_SVDRealRandomMatrix_200x3a)
{
    SetupR(300,3);
    auto [U,s,VT]=LaSVDecomp(itsAR); //Solve M=U*s*VT
    EXPECT_NEAR(Max(abs(Transpose(U)*U-itsIR)),0.0,eps);
    EXPECT_NEAR(Max(abs(VT*Transpose(VT)-itsIR)),0.0,eps);
    EXPECT_NEAR(Max(abs(U*s*VT-itsAR)),0.0,eps);
}

TEST_F(LinearAlgebraTests,LAPACK_SVDRealRandomMatrix_3x200a)
{
    SetupR(3,200);
    auto [U,s,VT]=LaSVDecomp(itsAR); //Solve M=U*s*VT
    EXPECT_NEAR(Max(abs(Transpose(U)*U-itsIR)),0.0,eps);
    EXPECT_NEAR(Max(abs(VT*Transpose(VT)-itsIR)),0.0,eps);
    EXPECT_NEAR(Max(abs(U*s*VT-itsAR)),0.0,eps);
}

TEST_F(LinearAlgebraTests,LAPACK_SVDRealRandomMatrix_100x100)
{
    SetupR(100,100);
    auto [U,s,VT]=LaSVDecomp(itsAR); //Solve M=U*s*VT
    EXPECT_NEAR(Max(abs(Transpose(U)*U-itsIR)),0.0,eps);
    EXPECT_NEAR(Max(abs(VT*Transpose(VT)-itsIR)),0.0,eps);
    EXPECT_NEAR(Max(abs(U*s*VT-itsAR)),0.0,eps);
}

TEST_F(LinearAlgebraTests,ArpackEigenSolver_ComplexNonHermition)
{
    int Nev=4;
    SetupC(100);

    ArpackEigenSolver solver;
    auto [D,U]=solver.Solve(itsAC,Nev,eps);

    for (int i=1;i<=Nev;i++)
    {
        Vector<dcmplx> residuals=itsAC*U.GetColumn(i)-D(i)*U.GetColumn(i);
        double res=Max(abs(residuals));
        EXPECT_NEAR(res,0.0,15*itsEps.itsEigenSolverEpsilon);
    }
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
    EXPECT_NEAR(Max(abs(M+Md)),0.0,eps);
    MatrixRT dM=d*M;
    EXPECT_NEAR(Max(abs(M+dM)),0.0,eps);
    MatrixRT dMdMdM=d*M*d*M*d*M;
    EXPECT_NEAR(Max(abs(M*M*M+dMdMdM)),0.0,eps);

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
    EXPECT_NEAR(Max(abs(M+Md)),0.0,eps);
    MatrixCT dM=d*M;
    EXPECT_NEAR(Max(abs(M+dM)),0.0,eps);
    MatrixCT dMdMdM=d*M*d*M*d*M;
    EXPECT_NEAR(Max(abs(M*M*M+dMdMdM)),0.0,eps);

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
    EXPECT_NEAR(Max(abs(M+Md)),0.0,eps);
    MatrixCT dM=d*M;
    EXPECT_NEAR(Max(abs(M+dM)),0.0,eps);
    MatrixCT dMdMdM=d*M*d*M*d*M;
    EXPECT_NEAR(Max(abs(M*M*M+dMdMdM)),0.0,eps);

}


*/
