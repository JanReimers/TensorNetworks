
#include "ArpackEigenSolver.H"
#include "Containers/SparseMatrix.H"

#include "arpack/arpackdef.h"
#include "oml/matrix.h"
#include <tuple>
#include <iostream>

using std::cout;
using std::endl;

typedef std::complex<double> dcmplx;

//
//  Hand modified version of arpack.h with the horrible C-style __complex__ crap removed.
//  This code uses the fork arpack-ng https://github.com/opencollab/arpack-ng
//  These routines are very hard ot use becuase they force the user to implement the interation loop
//  They dont seem to undestand how to pass function names for matvec operations like lapack and primme do.
//  See here for docs: https://www.caam.rice.edu/software/ARPACK/UG/node138.html
//                     https://www.caam.rice.edu/software/ARPACK/UG/node44.html
//                      https://scm.cs.kuleuven.be/scm/svn/numerics_software/ARPACK/SRC/dneupd.f//
extern "C"
{
    // Non symmetric complex single
    void cnaupd_c(a_int* ido, char const* bmat, a_int n, char const* which, a_int nev, float  tol, std::complex<float>* resid, a_int ncv, std::complex<float>* v, a_int ldv, a_int* iparam, a_int* ipntr, std::complex<float>* workd, std::complex<float>* workl, a_int lworkl, float* rwork, a_int* info);
    // Non symmetric double
    void dnaupd_c(a_int* ido, char const* bmat, a_int n, char const* which, a_int nev, double tol, double* resid, a_int ncv, double* v, a_int ldv, a_int* iparam, a_int* ipntr, double* workd, double* workl, a_int lworkl, a_int* info);
    // Symmetric double
    void dsaupd_c(a_int* ido, char const* bmat, a_int n, char const* which, a_int nev, double tol, double* resid, a_int ncv, double* v, a_int ldv, a_int* iparam, a_int* ipntr, double* workd, double* workl, a_int lworkl, a_int* info);
    // Non symmetric single
    void snaupd_c(a_int* ido, char const* bmat, a_int n, char const* which, a_int nev, float  tol, float* resid, a_int ncv, float* v, a_int ldv, a_int* iparam, a_int* ipntr, float* workd, float* workl, a_int lworkl, a_int* info);
    // Symmetric single
    void ssaupd_c(a_int* ido, char const* bmat, a_int n, char const* which, a_int nev, float  tol, float* resid, a_int ncv, float* v, a_int ldv, a_int* iparam, a_int* ipntr, float* workd, float* workl, a_int lworkl, a_int* info);
    // Non symmetric complex double
    void znaupd_c(a_int* ido, char const* bmat, a_int n, char const* which, a_int nev, double tol, dcmplx* resid, a_int ncv, dcmplx* v, a_int ldv, a_int* iparam, a_int* ipntr, dcmplx* workd, dcmplx* workl, a_int lworkl, double* rwork, a_int* info);

    // Call these to extract eigen vectors after the iterations are complete
    void cneupd_c(bool rvec, char const* howmny, a_int const* select, std::complex<float>*  d, std::complex<float>* z, a_int ldz, std::complex<float> sigma, std::complex<float>* workev, char const* bmat, a_int n, char const* which, a_int nev, float tol, std::complex<float>* resid, a_int ncv, std::complex<float>* v, a_int ldv, a_int* iparam, a_int* ipntr, std::complex<float>* workd, std::complex<float>* workl, a_int lworkl, float* rwork, a_int* info);
    void dneupd_c(bool rvec, char const* howmny, a_int const* select, double* dr, double* di, double* z, a_int ldz, double sigmar, double sigmai, double * workev, char const* bmat, a_int n, char const* which, a_int nev, double tol, double* resid, a_int ncv, double* v, a_int ldv, a_int* iparam, a_int* ipntr, double* workd, double* workl, a_int lworkl, a_int* info);
    void dseupd_c(bool rvec, char const* howmny, a_int const* select, double* d , double*  z, a_int ldz, double sigma, char const* bmat, a_int n, char const* which, a_int nev, double tol, double* resid, a_int ncv, double* v, a_int ldv, a_int* iparam, a_int* ipntr, double* workd, double* workl, a_int lworkl, a_int* info);
    void sneupd_c(bool rvec, char const* howmny, a_int const* select, float*  dr, float*  di, float* z, a_int ldz, float sigmar, float sigmai, float * workev, char const* bmat, a_int n, char const* which, a_int nev, float tol, float* resid, a_int ncv, float* v, a_int ldv, a_int* iparam, a_int* ipntr, float* workd, float* workl, a_int lworkl, a_int* info);
    void sseupd_c(bool rvec, char const* howmny, a_int const* select, float*  d , float*   z, a_int ldz, float sigma, char const* bmat, a_int n, char const* which, a_int nev, float tol, float* resid, a_int ncv, float* v, a_int ldv, a_int* iparam, a_int* ipntr, float* workd, float* workl, a_int lworkl, a_int* info);
    void zneupd_c(bool rvec, char const* howmny, a_int const* select, dcmplx* d , dcmplx*  z, a_int ldz, dcmplx sigma, dcmplx* workev, char const* bmat, a_int n, char const* which, a_int nev, double tol, dcmplx* resid, a_int ncv, dcmplx* v, a_int ldv, a_int* iparam, a_int* ipntr, dcmplx* workd, dcmplx* workl, a_int lworkl, double* rwork, a_int* info);
}

template <class T> void matvec(int N, const DMatrix<T>& A, const T * x, T * y)
{
    assert(A.GetNumRows()==N);
    assert(A.GetNumCols()==N);
    for (int i=1;i<=N;i++)
    {
        y[i-1]=T (0.0);
        for (int j=1;j<=N;j++)
            y[i-1]+=A(i,j)*x[j-1];
    }
}

template <class T> void matvec(int N, const SparseMatrix<T>& A, const T * x, T * y)
{
    assert(A.GetNumRows()==N);
    assert(A.GetNumCols()==N);
    A.DoMVMultiplication(N,x,y);
//    for (int i=1;i<=N;i++)
//    {
//        y[i-1]=T (0.0);
//        for (int j=1;j<=N;j++)
//            y[i-1]+=A(i,j)*x[j-1];
//    }
}

template <class T> typename ArpackEigenSolver<T>::UdType
ArpackEigenSolver<T>::SolveNonSym(const DMatrix<T>& A, int Nev,double eps)
{
    return SolveG(A,Nev,eps);
}

template <class T> typename ArpackEigenSolver<T>::UdType
ArpackEigenSolver<T>::SolveNonSym(const SparseMatrix<T>& A, int Nev,double eps)
{
    return SolveG(A,Nev,eps);
}

template <> template <template <typename> class Mat> typename ArpackEigenSolver<double>::UdType
ArpackEigenSolver<double>::SolveG(const Mat<double>& A, int Nev,double eps)
{
    int N=A.GetNumRows();
    assert(N==A.GetNumCols());
    assert(Nev>0);
    assert(Nev<N);

    int IDO=0,INFO=0;
    int Ncv=2*Nev; //*** THis has a huge effect on convergence, bigger is better.
    if (Ncv>N) Ncv=N;
    int Lworkl=3*Ncv*Ncv + 6*Ncv;

    int MaxIter=1000;

    Vector<double>  residuals(N);
    Vector<double>  Workd(3*N);
    Vector<double>  Workl(Lworkl);
    DMatrix<double>  V(N,Ncv);
    Vector<double> rwork(Ncv);
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
        dnaupd_c(&IDO,&arI,N,"LM",Nev,eps,&residuals(1),Ncv,&V(1,1)
        ,N,&iParam(1),iPntr,&Workd(1),&Workl(1),Lworkl,&INFO);
//        cout << "IDO=" << IDO << endl;
        if (IDO==-1 || IDO==1)
            matvec<double>(N,A,&Workd(iPntr[0]),&Workd(iPntr[1]));
        else
            break;

    } while(true);
//    cout << "Info=" << INFO << endl;
//    cout << "nIter=" << iParam(3) << endl;
    assert(INFO==0);
    //
    //  Post processing to the eigen vectors.
    //
    Vector<int> select(Ncv);
    Vector<double> DR(Nev+1),DI(Nev+1),Workev(3*Ncv);
//    DMatrix<double> U(N,Nev+1);
    double sigmar(0),sigmai(0);
    char how_many('A');
    dneupd_c(true, &how_many, &select(1), &DR(1),&DI(1), &V(1,1),N,
        sigmar,sigmai, &Workev(1), &arI,N,"LM", Nev, eps,
          &residuals(1), Ncv, &V(1,1),N, &iParam(1), iPntr, &Workd(1), &Workl(1),
          Lworkl, &INFO );

    Vector<dcmplx> D(Nev);
    DMatrix<dcmplx> UC(N,Nev);
    int ne=1;
    while (ne<Nev)
    {
        if (fabs(DI(ne))<eps)
        {
            D(ne)=dcmplx(DR(ne),0.0);
            for (int i=1;i<=N;i++)
                UC(i,ne  )=dcmplx(V(i,ne),0.0);
            ne+=1;
        }
        else
        {
            D(ne)=dcmplx(DR(ne),DI(ne));
            D(ne+1)=dcmplx(DR(ne+1),DI(ne+1));
            for (int i=1;i<=N;i++)
            {
                UC(i,ne  )=dcmplx(V(i,ne), V(i,ne+1));
                UC(i,ne+1)=dcmplx(V(i,ne),-V(i,ne+1));
            }
            ne+=2;
        }
    }
    return make_tuple(UC,D);
}

template <> template <template <typename> class Mat> typename ArpackEigenSolver<dcmplx>::UdType
ArpackEigenSolver<dcmplx>::SolveG(const Mat<dcmplx>& A, int Nev,double eps)
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

    Vector<dcmplx>  residuals(N),Workd(3*N),Workl(Lworkl);
    DMatrix<dcmplx>  V(N,Ncv);
    Vector<double> rwork(Ncv);
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
        znaupd_c(&IDO,&arI,N,"LM",Nev,eps,&residuals(1),Ncv,&V(1,1)
        ,N,&iParam(1),iPntr,&Workd(1),&Workl(1),Lworkl,&rwork(1),&INFO);
//        cout << "IDO=" << IDO << endl;
        if (IDO==-1 || IDO==1)
            matvec<dcmplx>(N,A,&Workd(iPntr[0]),&Workd(iPntr[1]));
        else
            break;

    } while(true);
    //cout << "Info=" << INFO << endl;
    //cout << "nIter=" << iParam(3) << endl;
    assert(INFO==0);
    //
    //  Post processing to the eigen vectors.
    //
    Vector<int> select(Ncv);
    Vector<dcmplx> D(Nev+1),Workev(2*Ncv);
    DMatrix<dcmplx> U(N,Nev);
    dcmplx sigma(0);
    zneupd_c(true, "All", &select(1), &D(1), &U(1,1),N,
        sigma, &Workev(1), &arI,N,"LM", Nev, eps,
          &residuals(1), Ncv, &V(1,1),N, &iParam(1), iPntr, &Workd(1), &Workl(1),
          Lworkl, &rwork(1), &INFO );

    return make_tuple(U,D);
}

template class ArpackEigenSolver<double>;
template class ArpackEigenSolver<dcmplx>;

