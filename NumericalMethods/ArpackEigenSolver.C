
#include "ArpackEigenSolver.H"
#include "TensorNetworks/Epsilons.H"

#include "arpack/arpackdef.h"
#include "oml/dmatrix.h"
#include <tuple>
#include <iostream>

using std::cout;
using std::endl;

//
//  Hand modified version of arpack.h with the horrible C-style __complex__ crap removed.
//  This code uses the fork arpack-ng https://github.com/opencollab/arpack-ng
//  These routines are very hard ot use becuase they force the user to implement the interation loop
//  They dont seem to undestand how to pass function names for matvec operations like lapack and primme do.
//  See here for docs: https://www.caam.rice.edu/software/ARPACK/UG/node138.html
//                     https://www.caam.rice.edu/software/ARPACK/UG/node44.html
//
extern "C"
{
    void cnaupd_c(a_int* ido, char const* bmat, a_int n, char const* which, a_int nev, float  tol, std::complex<float>* resid, a_int ncv, std::complex<float>* v, a_int ldv, a_int* iparam, a_int* ipntr, std::complex<float>* workd, std::complex<float>* workl, a_int lworkl, float* rwork, a_int* info);
    void dnaupd_c(a_int* ido, char const* bmat, a_int n, char const* which, a_int nev, double tol, double* resid, a_int ncv, double* v, a_int ldv, a_int* iparam, a_int* ipntr, double* workd, double* workl, a_int lworkl, a_int* info);
    void dsaupd_c(a_int* ido, char const* bmat, a_int n, char const* which, a_int nev, double tol, double* resid, a_int ncv, double* v, a_int ldv, a_int* iparam, a_int* ipntr, double* workd, double* workl, a_int lworkl, a_int* info);
    void snaupd_c(a_int* ido, char const* bmat, a_int n, char const* which, a_int nev, float  tol, float* resid, a_int ncv, float* v, a_int ldv, a_int* iparam, a_int* ipntr, float* workd, float* workl, a_int lworkl, a_int* info);
    void ssaupd_c(a_int* ido, char const* bmat, a_int n, char const* which, a_int nev, float  tol, float* resid, a_int ncv, float* v, a_int ldv, a_int* iparam, a_int* ipntr, float* workd, float* workl, a_int lworkl, a_int* info);
    void znaupd_c(a_int* ido, char const* bmat, a_int n, char const* which, a_int nev, double tol, std::complex<double>* resid, a_int ncv, std::complex<double>* v, a_int ldv, a_int* iparam, a_int* ipntr, std::complex<double>* workd, std::complex<double>* workl, a_int lworkl, double* rwork, a_int* info);

    void cneupd_c(bool rvec, char const* howmny, a_int const* select, std::complex<float>*  d, std::complex<float>* z, a_int ldz, std::complex<float> sigma, std::complex<float>* workev, char const* bmat, a_int n, char const* which, a_int nev, float tol, std::complex<float>* resid, a_int ncv, std::complex<float>* v, a_int ldv, a_int* iparam, a_int* ipntr, std::complex<float>* workd, std::complex<float>* workl, a_int lworkl, float* rwork, a_int* info);
    void dneupd_c(bool rvec, char const* howmny, a_int const* select, double*              dr, double* di, double* z, a_int ldz, double sigmar, double sigmai, double * workev, char const* bmat, a_int n, char const* which, a_int nev, double tol, double* resid, a_int ncv, double* v, a_int ldv, a_int* iparam, a_int* ipntr, double* workd, double* workl, a_int lworkl, a_int* info);
    void dseupd_c(bool rvec, char const* howmny, a_int const* select, double*               d, double* z, a_int ldz, double sigma, char const* bmat, a_int n, char const* which, a_int nev, double tol, double* resid, a_int ncv, double* v, a_int ldv, a_int* iparam, a_int* ipntr, double* workd, double* workl, a_int lworkl, a_int* info);
    void sneupd_c(bool rvec, char const* howmny, a_int const* select, float*               dr, float* di, float* z, a_int ldz, float sigmar, float sigmai, float * workev, char const* bmat, a_int n, char const* which, a_int nev, float tol, float* resid, a_int ncv, float* v, a_int ldv, a_int* iparam, a_int* ipntr, float* workd, float* workl, a_int lworkl, a_int* info);
    void sseupd_c(bool rvec, char const* howmny, a_int const* select, float*                d, float* z, a_int ldz, float sigma, char const* bmat, a_int n, char const* which, a_int nev, float tol, float* resid, a_int ncv, float* v, a_int ldv, a_int* iparam, a_int* ipntr, float* workd, float* workl, a_int lworkl, a_int* info);
    void zneupd_c(bool rvec, char const* howmny, a_int const* select, std::complex<double>* d, std::complex<double>* z, a_int ldz, std::complex<double> sigma, std::complex<double>* workev, char const* bmat, a_int n, char const* which, a_int nev, double tol, std::complex<double>* resid, a_int ncv, std::complex<double>* v, a_int ldv, a_int* iparam, a_int* ipntr, std::complex<double>* workd, std::complex<double>* workl, a_int lworkl, double* rwork, a_int* info);
}

typedef TensorNetworks::VectorRT  VectorRT;


ArpackEigenSolver::ArpackEigenSolver()
{
    //ctor
}

ArpackEigenSolver::~ArpackEigenSolver()
{
    //dtor
}

void matvec(int N, const TensorNetworks::MatrixCT& A, const std::complex<double> * x, std::complex<double> * y)
{
    assert(A.GetNumRows()==N);
    assert(A.GetNumCols()==N);
    for (int i=1;i<=N;i++)
    {
        y[i-1]=std::complex<double> (0.0);
        for (int j=1;j<=N;j++)
            y[i-1]+=A(i,j)*x[j-1];
    }
}

std::tuple<ArpackEigenSolver::VectorCT,ArpackEigenSolver::MatrixCT> ArpackEigenSolver::Solve(const MatrixCT& A, int Nev,const TensorNetworks::Epsilons& eps)
{
    int N=A.GetNumRows();
    assert(N==A.GetNumCols());
    assert(Nev>0);
    assert(Nev<N);

    int IDO=0,INFO=0;
    int Ncv=5*Nev; //*** THis has a huge effect on convergence, bigger is better.
    if (Ncv>N) Ncv=N;
    int Lworkl=3*Ncv*Ncv + 5*Ncv;

    int MaxIter=1000;

    VectorCT  residuals(N),Workd(3*N),Workl(Lworkl);
    MatrixCT  V(N,Ncv);
    VectorRT   rwork(Ncv);
    Vector<int> iParam(11);
    iParam(1)=1; //ISHIFT
    iParam(3)=MaxIter;
    iParam(4)=1; //NB only 1 works
    iParam(7)=1; //Mode
    int iPntr[14];
    char arI='I';

    // Arnaldi iteration loop
    do
    {
        znaupd_c(&IDO,&arI,N,"LM",Nev,eps.itsEigenSolverEpsilon,&residuals(1),Ncv,&V(1,1)
        ,N,&iParam(1),iPntr,&Workd(1),&Workl(1),Lworkl,&rwork(1),&INFO);
//        cout << "IDO=" << IDO << endl;
        if (IDO==-1 || IDO==1)
            matvec(N,A,&Workd(iPntr[0]),&Workd(iPntr[1]));
        else
            break;

    } while(true);
    cout << "Info=" << INFO << endl;
    cout << "nIter=" << iParam(3) << endl;
    assert(INFO==0);
    //
    //  Post processing to the eigen vectors.
    //
    Vector<int> select(Ncv);
    VectorCT D(Nev+1),Workev(2*Ncv);
    MatrixCT U(N,Nev);
    std::complex<double> sigma(0.0);
    zneupd_c(true, "All", &select(1), &D(1), &U(1,1),N,
        sigma, &Workev(1), &arI,N,"LM", Nev, eps.itsEigenSolverEpsilon,
          &residuals(1), Ncv, &V(1,1),N, &iParam(1), iPntr, &Workd(1), &Workl(1),
          Lworkl, &rwork(1), &INFO );

    return {D,U};
}
