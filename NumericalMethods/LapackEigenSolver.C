#include "LapackEigenSolver.H"
#include "oml/dmatrix.h"
#include "oml/vector.h"
//#include "oml/matrix_io.h"
#include <complex>
#include <iostream>
//
// See http://www.netlib.org/lapack/explore-html/d1/d7e/group__double_g_esing_ga84fdf22a62b12ff364621e4713ce02f2.html
// for detailed docs
// handy search tool: https://software.intel.com/content/www/us/en/develop/articles/intel-mkl-function-finding-advisor.html
// you also need to add -llapack to the link command
typedef std::complex<double> dcmplx;

extern"C" {

void dsyevx_(char* JOBZ,char* RANGE,char* UPLO,int* N,double* A,int* LDA,double* VL,double* VU,int* IL,int* IU,double* ABSTOL,int* M,double* W,double* Z,int* LDZ,double* WORK,int* LWORK,              int* IWORK,int* IFAIL,int* INFO);
void zheevx_(char* JOBZ,char* RANGE,char* UPLO,int* N,dcmplx* A,int* LDA,double* VL,double* VU,int* IL,int* IU,double* ABSTOL,int* M,double* W,dcmplx* Z,int* LDZ,dcmplx* WORK,int* LWORK,double* RWORK,int* IWORK,int* IFAIL,int* INFO);
void dgeev_ (char* JOBVL,char* JOBVR,int* N,double* A,int* LDA,double* WR,double* WI,double* VL,int* LDVL,double* VR,int* LDVR,double* WORK,int* LWORK,              int* INFO);
void zgeev_ (char* JOBVL,char* JOBVR,int* N,dcmplx* A,int* LDA,dcmplx* W            ,dcmplx* VL,int* LDVL,dcmplx* VR,int* LDVR,dcmplx* WORK,int* LWORK,double* RWORK,int* INFO);
}
// Symmetric
template <class T> void evx  (char* JOBZ,char* RANGE,char* UPLO,int* N,T     * A,int* LDA,double* VL,double* VU,int* IL,int* IU,double* ABSTOL,int* M,double* W,T     * Z,int* LDZ,T     * WORK,int* LWORK,int* IWORK,int* IFAIL,int* INFO);
template <> void evx<double> (char* JOBZ,char* RANGE,char* UPLO,int* N,double* A,int* LDA,double* VL,double* VU,int* IL,int* IU,double* ABSTOL,int* M,double* W,double* Z,int* LDZ,double* WORK,int* LWORK,int* IWORK,int* IFAIL,int* INFO)
{
    dsyevx_(JOBZ,RANGE,UPLO,N,A,LDA,VL,VU,IL,IU,ABSTOL,M,W,Z,LDZ,WORK,LWORK,IWORK,IFAIL,INFO); //double
}
template <> void evx<dcmplx> (char* JOBZ,char* RANGE,char* UPLO,int* N,dcmplx* A,int* LDA,double* VL,double* VU,int* IL,int* IU,double* ABSTOL,int* M,double* W,dcmplx* Z,int* LDZ,dcmplx* WORK,int* LWORK,int* IWORK,int* IFAIL,int* INFO)
{
    Vector<double> rwork(7*(*N));
    zheevx_(JOBZ,RANGE,UPLO,N,A,LDA,VL,VU,IL,IU,ABSTOL,M,W,Z,LDZ,WORK,LWORK,&rwork(1),IWORK,IFAIL,INFO); //complex<double>
}
// Non symmetric
template <class T> void ev (char* JOBVL,char* JOBVR,int* N,T     * A,int* LDA,T     * WR,T     * WI,T     * VL,int* LDVL,T     * VR,int* LDVR,T     * WORK,int* LWORK,int* INFO);
template <> void ev<double>(char* JOBVL,char* JOBVR,int* N,double* A,int* LDA,double* WR,double* WI,double* VL,int* LDVL,double* VR,int* LDVR,double* WORK,int* LWORK,int* INFO)
{
    dgeev_ (JOBVL,JOBVR,N,A,LDA,WR,WI,VL,LDVL,VR,LDVR,WORK,LWORK,      INFO);
}
template <> void ev<dcmplx>(char* JOBVL,char* JOBVR,int* N,dcmplx* A,int* LDA,dcmplx* W,dcmplx*    ,dcmplx* VL,int* LDVL,dcmplx* VR,int* LDVR,dcmplx* WORK,int* LWORK,int* INFO)
{
    Vector<double> rwork(7*(*N));
    zgeev_ (JOBVL,JOBVR,N,A,LDA,W    ,VL,LDVL,VR,LDVR,WORK,LWORK,&rwork(1),INFO);
}


inline const double& real(const double& d) {return d;}
using std::real;

template <class T> typename LapackEigenSolver<T>::UdType
LapackEigenSolver<T>::Solve(const MatrixT& A, int NumEigenValues,double eps)
{
    int N=A.GetNumRows();
    assert(N==A.GetNumCols());
    assert(NumEigenValues<=N);
    assert(IsHermitian(A,eps));
    //
    //  Dicey deciding how much work space lapack needs. more is faster
    //
    int info=0,lwork=-1,IL=1,IU=NumEigenValues;
    double VL,VU;
    Vector<double> W(N);
    Vector<T> work(1);
    Vector<int> iwork(5*N),ifail(N);
    DMatrix<T> U(N,NumEigenValues),Alower(A);
    char jobz='V',range='I',uplo='L';
    //
    //  Initial call to see how much work space is needed
    //
    evx<T>(&jobz,&range,&uplo,&N,&Alower(1,1),&N,&VL,&VU,&IL,&IU,&eps,&NumEigenValues,&W(1),&U(1,1),&N, &work(1),&lwork,&iwork(1),&ifail(1),&info);
    lwork=real(work(1));
    work.SetLimits(lwork);
    //
    //  Now do the actual SVD work
    //
    evx<T>(&jobz,&range,&uplo,&N,&Alower(1,1),&N,&VL,&VU,&IL,&IU,&eps,&NumEigenValues,&W(1),&U(1,1),&N, &work(1),&lwork,&iwork(1),&ifail(1),&info);
    assert(info==0);
    //
    //  Now fix up the matrix limits
    W.SetLimits(NumEigenValues,true);
    return std::make_tuple(U,W);
}

template <> typename LapackEigenSolver<double>::UdTypeN
LapackEigenSolver<double>::SolveNonSym(const MatrixT& A, int NumEigenValues,double eps)
{
    int N=A.GetNumRows();
    assert(N==A.GetNumCols());
    if (NumEigenValues<=N)
        std::cerr << "Warning: Lapack does not support subset of eigen values for non-symmtric matrcies" << std::endl;
    //
    //  Dicey deciding how much work space lapack needs. more is faster
    //
    int info=0,lwork=-1;
    Vector<double> WR(N),WI(N);
    Vector<double> work(1);
    Vector<int> iwork(5*N),ifail(N);
    DMatrix<double> VL(N,N),VR(N,N),Acopy(A);
    char jobvl='N',jobvr='V';
    //
    //  Initial call to see how much work space is needed
    //
    dgeev_(&jobvl,&jobvr,&N,&Acopy(1,1),&N,&WR(1),&WI(1),&VL(1,1),&N,&VR(1,1),&N, &work(1),&lwork,&info);
    lwork=real(work(1));
    work.SetLimits(lwork);
    //
    //  Now do the actual SVD work
    //
    dgeev_(&jobvl,&jobvr,&N,&Acopy(1,1),&N,&WR(1),&WI(1),&VL(1,1),&N,&VR(1,1),&N, &work(1),&lwork,&info);
    assert(info==0);
    //
    //  Unpack the eigen pairs
    //
    Vector<dcmplx> W(N);
    DMatrix<dcmplx> V(N,N);
    for (int j=1;j<=N;j++)
    {
        if (fabs(WI(j))<eps)
        {
            W(j)=dcmplx(WR(j),WI(j));
            for (int i=1;i<=N;i++)
                V(i,j)=dcmplx(VR(i,j),0.0);
        }
        else
        {
            W(j  )=dcmplx(WR(j  ),WI(j  ));
            W(j+1)=dcmplx(WR(j+1),WI(j+1));
            for (int i=1;i<=N;i++)
            {
                V(i,j  )=dcmplx(VR(i,j), VR(i,j+1));
                V(i,j+1)=dcmplx(VR(i,j),-VR(i,j+1));
            }
            j++;
        }
    }
    return std::make_tuple(V,W);
}

template <> typename LapackEigenSolver<dcmplx>::UdTypeN
LapackEigenSolver<dcmplx>::SolveNonSym(const MatrixT& A, int NumEigenValues,double eps)
{
    int N=A.GetNumRows();
    assert(N==A.GetNumCols());
    if (NumEigenValues<=N)
        std::cerr << "Warning: Lapack does not support subset of eigen values for non-symmtric matrcies" << std::endl;
    //
    //  Dicey deciding how much work space lapack needs. more is faster
    //
    int info=0,lwork=-1;
    Vector<dcmplx> W(N),work(1);
    Vector<double> rwork(2*N);
    DMatrix<dcmplx> VL(N,N),VR(N,N),Acopy(A);
    char jobvl='N',jobvr='V';
    //
    //  Initial call to see how much work space is needed
    //
    zgeev_(&jobvl,&jobvr,&N,&Acopy(1,1),&N,&W(1),&VL(1,1),&N,&VR(1,1),&N, &work(1),&lwork,&rwork(1),&info);
    lwork=real(work(1));
    work.SetLimits(lwork);
    //
    //  Now do the actual SVD work
    //
    zgeev_(&jobvl,&jobvr,&N,&Acopy(1,1),&N,&W(1),&VL(1,1),&N,&VR(1,1),&N, &work(1),&lwork,&rwork(1),&info);
    assert(info==0);
    return std::make_tuple(VR,W);
}

//
//  Make template instances
//
template class LapackEigenSolver<std::complex<double> >;
template class LapackEigenSolver<double>;
