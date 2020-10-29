#ifndef LINEARALGEBRATESTS_H_INCLUDED
#define LINEARALGEBRATESTS_H_INCLUDED

#include "NumericalMethods/SVDSolver.H"
#include "NumericalMethods/SparseSVDSolver.H"
#include "NumericalMethods/EigenSolver.H"
#include "NumericalMethods/SparseEigenSolver.H"
#include "oml/random.h"
#include "oml/fakedouble.h"
#include <iostream>
#include <complex>

using std::cout;
using std::endl;

typedef std::complex<double> dcmplx;

template <typename T> Matrix<double> MyAbs(const Matrix<T> &);
template <> Matrix<double> MyAbs(const Matrix<double> &m)
{
    return fabs(m);
};

template <> Matrix<double> MyAbs(const Matrix<dcmplx> &m)
{
    return fabs(m);
};

//
//  Randomly put zeros into a dense matrix until a request density is achieved
//
template <class T> void MakeSparse(Matrix<T>& m,double TargetDensity)
{
    int Ntotal=m.size(),N=m.size();
    int Nr=m.GetNumRows(),Nc=m.GetNumCols();
    double density=1.0;
    while (density>TargetDensity)
    {
        int ir=static_cast<int>(OMLRand<float>()*Nr)+1;
        int ic=static_cast<int>(OMLRand<float>()*Nc)+1;
        if (m(ir,ic)!=0.0)
        {
            m(ir,ic)=0.0;
            N--;
            density=static_cast<double>(N)/Ntotal;
        }

    }

}

template <typename T> class EigenBase
{
public:
    EigenBase(int n,double density=1.0)
        : N(n), Nev(n),Density(density),eps(1e-13)
        , I(N,N)
        {
            Unit(I);
        }
    EigenBase(int n,int nev,double density=1.0)
        : N(n), Nev(nev),Density(density),eps(1e-13)
        , I(N,N)
        {
            Unit(I);
        }
    virtual Matrix<T> MakeMatrix()
    {
        Matrix<T> ret(N,N);
        FillRandom(ret);
        MakeSparse(ret,Density);
        return ret;
    }

    virtual void ValidateSymmetric(const Matrix<T>& U)
    {
        Matrix<T> UT=Transpose(conj(U));
        //cout << "U*UT=" << UT*U << endl;
        EXPECT_NEAR(Max(fabs(UT*U-I)),0.0,2*N*eps);
    }


protected:
    int N;
    int Nev;
    double Density;
    double eps;
    Matrix<T> I;
};

template <typename T>
class EigenTester : public EigenBase<T>
{
    using Base=EigenBase<T>;
    using Base::N;
    using Base::Nev;
    using Base::eps;
    using Base::I;
public:
    EigenTester(EigenSolver<T>* s, int n,double density=1.0)
        : Base(n,density)
        , itsSolver(s)
        , A(n,n)
     {}
    EigenTester(EigenSolver<T>* s, int n,int Nev,double density=1.0)
        : Base(n,Nev,density)
        , itsSolver(s)
        , A(n,n)
     {}
    ~EigenTester() {delete itsSolver;}

    virtual void RunTests()
    {
        A=MakeMatrix();
        {
            auto [U,d]=itsSolver->Solve(A,eps,Nev);
            ValidateSymmetric(U,d);
            itsSolver->Reset();
        }
        if (Nev/2>=1)
        {
            auto [U,d]=itsSolver->Solve(A,eps,Nev/2); //Get only half the eigen values
            ValidateSymmetric(U,d);
            itsSolver->Reset();
        }
        if (Nev==N)
        {
            auto [U,d]=itsSolver->SolveAll(A,eps);
            ValidateSymmetric(U,d);
            itsSolver->Reset();
        }
        A=Base::MakeMatrix(); //Make a non sym matrix
        {
            auto [U,d]=itsSolver->SolveRightNonSym(A,eps,Nev);
            ValidateRight(U,d);
            itsSolver->Reset();
        }
        {
            auto [U,d]=itsSolver->SolveLeft_NonSym(A,eps,Nev);
            ValidateLeft(U,d);
            itsSolver->Reset();
        }
        if (Nev/2>=1)
        {
            auto [U,d]=itsSolver->SolveRightNonSym(A,eps,Nev/2);
            ValidateRight(U,d);
            itsSolver->Reset();
        }
        if (Nev/2>=1)
        {
            auto [U,d]=itsSolver->SolveLeft_NonSym(A,eps,Nev/2);
            ValidateLeft(U,d);
            itsSolver->Reset();
        }
        if (Nev==N)
        {
            auto [U,d]=itsSolver->SolveAllRightNonSym(A,eps);
            ValidateRight(U,d);
            itsSolver->Reset();
        }
    }
    virtual Matrix<T> MakeMatrix()
    {
        Matrix<T> ret=Base::MakeMatrix();
        ret=Matrix<T>(ret+Transpose(conj(ret)));
        return ret;
    }
    virtual void ValidateSymmetric(const Matrix<T>& U, const Vector<double>& d)
    {
        Matrix<T> UT=Transpose(conj(U));
//        cout << "U*UT=" << UT*U << endl;
        EXPECT_NEAR(Max(fabs(UT*U-I)),0.0,2*N*eps);
        Matrix<T> diag=UT*A*U;
//        cout << "diag=" << diag << endl;
        int Ne=d.size();
        for (int i=1;i<=Ne;i++) diag(i,i)-=d(i);
        EXPECT_NEAR(Max(fabs(diag)),0.0,N*eps);
    }
    virtual void ValidateRight(const Matrix<dcmplx>& U, const Vector<dcmplx>& d)
    {
        int Ne=d.size();
        if (Ne>0)
        {
            for (int i=1;i<=Ne;i++)
            {
                Vector<dcmplx> Ui=U.GetColumn(i);
                Vector<dcmplx> residuals=A*Ui-d(i)*Ui;
                double res=Max(fabs(residuals));
                EXPECT_NEAR(res,0.0,15*eps);
            }
        }
    }
    virtual void ValidateLeft(const Matrix<dcmplx>& U, const Vector<dcmplx>& d)
    {
        int Ne=d.size();
        if (Ne>0)
        {
            for (int i=1;i<=Ne;i++)
            {
                Vector<dcmplx> Ui=U.GetColumn(i);
                Vector<dcmplx> residuals=Ui*A-d(i)*Ui;
                double res=Max(fabs(residuals));
                EXPECT_NEAR(res,0.0,15*eps);
            }
        }
    }

    EigenSolver<T>* itsSolver;
    Matrix     <T>  A;
 };

template <typename T>
class SparseEigenTester
    : public EigenBase<T>
    , public virtual SparseEigenSolverClient<T>
{
    using Base=EigenBase<T>;
    using Base::N;
    using Base::Nev;
    using Base::eps;
    using Base::I;
public:
    SparseEigenTester(SparseEigenSolver<T>* s, int n,double density=1.0)
        : Base(n,density)
        , itsSolver(s)
        , A(n,n)
     {}
    SparseEigenTester(SparseEigenSolver<T>* s, int n,int Nev,double density=1.0)
        : Base(n,Nev,density)
        , itsSolver(s)
        , A(n,n)
     {}
    ~SparseEigenTester() {delete itsSolver;}

    virtual void RunTests()
    {
        A=MakeMatrix();
        {
            auto [U,d]=itsSolver->Solve(A,eps,Nev);
            ValidateSymmetric(U,d);
            itsSolver->Reset();
        }
        {
            auto [U,d]=itsSolver->Solve(this,eps,Nev);
            ValidateSymmetric(U,d);
            itsSolver->Reset();
        }
        if (Nev/2>=1)
        {
            auto [U,d]=itsSolver->Solve(A,eps,Nev/2); //Get only half the eigen values
            ValidateSymmetric(U,d);
            itsSolver->Reset();
        }
        if (Nev/2>=1)
        {
            auto [U,d]=itsSolver->Solve(this,eps,Nev/2); //Get only half the eigen values
            ValidateSymmetric(U,d);
            itsSolver->Reset();
        }
        A=Base::MakeMatrix(); //Make a non sym matrix
        {
            auto [U,d]=itsSolver->SolveRightNonSym(A,eps,Nev);
            ValidateRight(U,d);
            itsSolver->Reset();
        }
        {
            auto [U,d]=itsSolver->SolveLeft_NonSym(A,eps,Nev);
            ValidateLeft(U,d);
            itsSolver->Reset();
        }
        {
            auto [U,d]=itsSolver->SolveRightNonSym(this,eps,Nev);
            ValidateRight(U,d);
            itsSolver->Reset();
        }
        if (Nev/2>=1)
        {
            auto [U,d]=itsSolver->SolveRightNonSym(A,eps,Nev/2);
            ValidateRight(U,d);
            itsSolver->Reset();
        }
        if (Nev/2>=1)
        {
            auto [U,d]=itsSolver->SolveLeft_NonSym(A,eps,Nev/2);
            ValidateLeft(U,d);
            itsSolver->Reset();
        }
        if (Nev/2>=1)
        {
            auto [U,d]=itsSolver->SolveRightNonSym(this,eps,Nev/2);
            ValidateRight(U,d);
            itsSolver->Reset();
        }
    }
    virtual Matrix<T> MakeMatrix()
    {
        Matrix<T> ret=Base::MakeMatrix();
        ret=Matrix<T>(ret+Transpose(conj(ret)));
        return ret;
    }
    virtual void ValidateSymmetric(const Matrix<T>& U, const Vector<double>& d)
    {
        Matrix<T> UT=Transpose(conj(U));
        //cout << "U*UT=" << UT*U << endl;
        EXPECT_NEAR(Max(fabs(UT*U-I)),0.0,2*N*eps);
        Matrix<T> diag=UT*A*U;
//        cout << "diag=" << diag << endl;
        int Ne=d.size();
        for (int i=1;i<=Ne;i++) diag(i,i)-=d(i);
        EXPECT_NEAR(Max(fabs(diag)),0.0,2*N*eps);
    }
    virtual void ValidateRight(const Matrix<dcmplx>& U, const Vector<dcmplx>& d)
    {
        int Ne=d.size();
        if (Ne>0)
        {
            for (int i=1;i<=Ne;i++)
            {
                Vector<dcmplx> Ui=U.GetColumn(i);
                Vector<dcmplx> residuals=A*Ui-d(i)*Ui;
                double res=Max(fabs(residuals));
                EXPECT_NEAR(res,0.0,15*eps);
            }
        }
    }
    virtual void ValidateLeft(const Matrix<dcmplx>& U, const Vector<dcmplx>& d)
    {
        int Ne=d.size();
        if (Ne>0)
        {
            for (int i=1;i<=Ne;i++)
            {
                Vector<dcmplx> Ui=U.GetColumn(i);
                Vector<dcmplx> residuals=Ui*A-d(i)*Ui;
                double res=Max(fabs(residuals));
                EXPECT_NEAR(res,0.0,15*eps);
            }
        }
    }

    virtual long int GetSize  () const {return N;}; //2^L Hilbert space size
    virtual void     DoMatVecContraction (int N, const T* xvec, T* yvec) const
    {
        A.DoMVMultiplication(N,xvec,yvec);
    };


    SparseEigenSolver<T>* itsSolver;
    SparseMatrix     <T>  A;
 };

//
//   SVD Tester classes
//
template <typename T> class SVDBase
{
public:
    using DMatRT=DiagonalMatrix<double>;

    SVDBase(int m, int n,double density=1.0)
        : M(m), N(n), Density(density),eps(1e-13)
        , I(Min(M,N),Min(M,N))
        {
            Unit(I);
        }
    Matrix<T> MakeMatrix()
    {
        Matrix<T> ret(M,N);
        FillRandom(ret);
        MakeSparse(ret,Density);
        return ret;
    }

    virtual void Validate(const Matrix<T>& U, const DMatRT& s,const Matrix<T>& VT)
    {
        if (N<=M)
        {
            Matrix<T> V=Transpose(conj(VT));
            EXPECT_NEAR(Max(fabs(V*VT-I)),0.0,2*sqrt(N*M)*eps);
        }
        if (N>=M)
        {
            Matrix<T> UT=Transpose(conj(U));
            //cout << "U*UT=" << UT*U << endl;
            EXPECT_NEAR(Max(fabs(UT*U-I)),0.0,2*sqrt(N*M)*eps);
        }
    }


protected:
    int M,N;
    double Density;
    double eps;
    Matrix<T> I;
};

template <typename T>
class SVDTester : public SVDBase<T>
{
    using Base=SVDBase<T>;
    using DMatRT=typename Base::DMatRT;
    using Base::M;
    using Base::N;
    using Base::eps;
    using Base::I;
public:
    SVDTester(SVDSolver<T>* s, int m, int n,double density=1.0)
        : Base(m,n,density)
        , itsSolver(s)
        , A(m,n)
     {}
    ~SVDTester() {delete itsSolver;}

    virtual void RunTests()
    {
        A=Base::MakeMatrix();
        {
            auto [U,s,VT]=itsSolver->Solve(A,eps,Min(M,N));
            Validate(U,s,VT);
        }
        {
            auto [U,s,VT]=itsSolver->SolveAll(A,eps);
            Validate(U,s,VT);
        }
    }
    virtual void Validate(const Matrix<T>& U, const DMatRT& s,const Matrix<T>& VT)
    {
        Base::Validate(U,s,VT); //Check U,V orthonormality
        Matrix<T> A1=U*s*VT;
        Matrix<T> dA=A-A1;
        EXPECT_NEAR(Max(MyAbs(dA)),0.0,2*sqrt(N*M)*eps);
    }

    SVDSolver<T>* itsSolver;
    Matrix   <T>  A;
 };

template <typename T>
class SparseSVDTester
    : public SVDBase<T>
    , public virtual SparseSVDSolverClient<T>
{
    using Base=SVDBase<T>;
    using DMatRT=typename Base::DMatRT;
    using Base::M;
    using Base::N;
    using Base::eps;
    using Base::I;

public:
    SparseSVDTester(SparseSVDSolver<T>* s, int m, int n,double density=1.0)
        : Base(m,n,density)
        , itsSolver(s)
        , A(m,n)
     {}
    ~SparseSVDTester() {delete itsSolver;}

    virtual void RunTests()
    {
        A=Base::MakeMatrix();
        {
            auto [U,s,VT]=itsSolver->Solve(A,eps,Min(M,N));
            Validate(U,s,VT);
        }
        {  //Client version
            auto [U,s,VT]=itsSolver->Solve(this,eps,Min(M,N));
            Validate(U,s,VT);
        }
    }
    virtual void Validate(const Matrix<T>& U, const DMatRT& s,const Matrix<T>& VT)
    {
        Base::Validate(U,s,VT); //Check U,V orthonormality
        Matrix<T> A1=U*s*VT;
        Matrix<T> dA=A-A1;
        EXPECT_NEAR(Max(MyAbs(dA)),0.0,2*sqrt(N*M)*eps);
    }
//
// SparseSVDSolverClient interface
//
    virtual int  GetNumRows() const {return M;}
    virtual int  GetNumCols() const {return N;}
    virtual void DoMatVecContraction (int N, const T* xvec, T* yvec,int transpose) const
    {
        A.DoMVMultiplication(M,N,xvec,yvec,transpose);
    }

private:
    SparseSVDSolver<T>* itsSolver;
    SparseMatrix   <T>  A;
 };

#endif // LINEARALGEBRATESTS_H_INCLUDED
