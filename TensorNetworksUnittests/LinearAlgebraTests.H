#ifndef LINEARALGEBRATESTS_H_INCLUDED
#define LINEARALGEBRATESTS_H_INCLUDED

#include "oml/random.h"
#include <iostream>
#include <complex>

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

//
//  Randomly put zeros into a dense matrix until a request density is achieved
//
template <class T> void MakeSparse(DMatrix<T>& m,double TargetDensity)
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

template <typename T, template <typename> class Mat,template <typename> class Solver> class Tester
{
public:
    Tester(int m,int n,double density)
    : M(m), N(n), Density(density),
    A(M,N), I(Min(M,N),Min(M,N)),eps(1e-13)
    {
        Unit(I);
    }

    virtual void RunTests()=0;
    virtual DMatrix<T> MakeMatrix()
    {
        DMatrix<T> ret(M,N);
        FillRandom(ret);
        MakeSparse(ret,Density);
        return ret;
    }

    int M,N;
    double Density;
    Solver<T> solver;
    Mat<T> A;
    DMatrix<T> I;
    double eps;
};


template <typename T, template <typename> class Mat,template <typename> class Solver>
class SVDTester : public Tester<T,Mat,Solver>
{
public:
    typedef Tester<T,Mat,Solver> Base;
    using Base::solver;
    using Base::A;
    using Base::I;
    using Base::M;
    using Base::N;
    using Base::eps;
    SVDTester(int M, int N,double density=1.0) : Base(M,N,density) {}

    virtual void RunTests()
    {
        A=Base::MakeMatrix();
        auto [U,s,VT]=solver.Solve(A,Min(M,N),eps);
        if (N<=M)
        {
            DMatrix<T> V=Transpose(conj(VT));
            EXPECT_NEAR(Max(abs(V*VT-I)),0.0,100*eps);
        }
        if (N>=M)
        {
            DMatrix<T> UT=Transpose(conj(U));
            //cout << "U*UT=" << UT*U << endl;
            EXPECT_NEAR(Max(abs(UT*U-I)),0.0,sqrt(N*M)*eps);
        }

        DMatrix<T> A1=U*s*VT;
        DMatrix<T> dA=A-A1;
        EXPECT_NEAR(Max(MyAbs(dA)),0.0,sqrt(N*M)*eps);

    }

};

template <typename T, template <typename> class Mat,template <typename> class Solver>
class SymEigenTester : public Tester<T,Mat,Solver>
{
public:
    typedef Tester<T,Mat,Solver> Base;
    using Base::solver;
    using Base::A;
    using Base::I;
    using Base::M;
    using Base::N;
    using Base::eps;

    SymEigenTester(int N,double density=1.0) : Base(N,N,density) {}

    virtual DMatrix<T> MakeMatrix()
    {
        DMatrix<T> ret=Base::MakeMatrix();
        ret=DMatrix<T>(ret+Transpose(conj(ret)));
        return ret;
    }
    virtual void RunTests()
    {
        A=MakeMatrix();
        int Ne=N/2;
//        cout << "A=" << A << endl;
        auto [U,d]=solver.Solve(A,Ne,eps);
//        cout << "U,d=" << U << d << endl;
        DMatrix<T> UT=Transpose(conj(U));
        EXPECT_NEAR(Max(abs(UT*U-I)),0.0,10*eps);
//        cout << "UT*U=" << UT*U << endl;
        DMatrix<T> diag=UT*A*U;
//        cout << "diag=" << diag << endl;
        for (int i=1;i<=Ne;i++) diag(i,i)-=d(i);
        EXPECT_NEAR(Max(abs(diag)),0.0,100*eps);
    }

};

template <typename T, template <typename> class Mat,template <typename> class Solver>
class NonSymEigenTester : public Tester<T,Mat,Solver>
{
public:
    typedef Tester<T,Mat,Solver> Base;
    using Base::solver;
    using Base::A;
    using Base::I;
    using Base::M;
    using Base::N;
    using Base::eps;

    NonSymEigenTester(int N,double density=1.0) : Base(N,N,density) {}

    virtual void RunTests()
    {
        A=Base::MakeMatrix();
        //Fill(A,1.0);
        int Ne=N/2;
        auto [U,d]=solver.SolveNonSym(A,Ne,eps);
//        cout << "U,d=" << U << d << endl;
        //Eigen vectors are not orthogonal so no ortho check here
        for (int i=1;i<=Ne;i++)
        {
            Vector<dcmplx> Ui=U.GetColumn(i);
            Vector<dcmplx> residuals=A*Ui-d(i)*Ui;
            double res=Max(abs(residuals));
            EXPECT_NEAR(res,0.0,15*eps);
        }

    }

};


#endif // LINEARALGEBRATESTS_H_INCLUDED