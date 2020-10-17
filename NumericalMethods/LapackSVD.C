#include "LapackSVD.H"
#include "oml/dmatrix.h"
#include "oml/diagonalmatrix.h"
#include "oml/vector.h"
#include "oml/minmax.h"
#include <complex>
//
// See http://www.netlib.org/lapack/explore-html/d1/d7e/group__double_g_esing_ga84fdf22a62b12ff364621e4713ce02f2.html
// for detailed docs
// you also need to add -llapack to the link command
typedef std::complex<double> dcmplx;

extern"C" {
void dgesvd_(char* JOBU,char* JOBVT,int* M,int* N,double* A,int* LDA, double* S,double* U,
int* LDU,double* VT,int* LDVT,double* WORK,int* LWORK,int* INFO);
void zgesvd_(char* JOBU,char* JOBVT,int* M,int* N,dcmplx* A,int* LDA, double* S,dcmplx* U,
int* LDU,dcmplx* VT,int* LDVT,dcmplx* WORK,int* LWORK,double* RWORK, int* INFO);
}


template <class TM> void LaSVDecomp(TM& A, Vector<double>& s, TM& VT)
{
    int M=A.GetNumRows(),N=A.GetNumCols(),mn=Min(M,N);
    assert(s.size()==mn);
    assert(VT.GetNumRows()==N);
    assert(VT.GetNumCols()==N);

    //
    //  Decide how much work space lapack needs
    //
    int info=0,lwork=-1;
    Vector<double> work(1);
    char jobu='O',jobv='A';
    dgesvd_(&jobu,&jobv,&M,&N,&A(1,1),&M,&s(1),0,&M,&VT(1,1),&N,&work(1),&lwork,&info);
    lwork=work(1);
    work.SetLimits(lwork);
    //
    //  Now do the actual SVD work
    //
    dgesvd_(&jobu,&jobv,&M,&N,&A(1,1),&M,&s(1),0,&M,&VT(1,1),&N,&work(1),&lwork,&info);
    assert(info==0);
    //
    //  Now fix up the matrix limits
    //
    A.SetLimits(M,mn,true); //Throw away last N-mn columns
    VT.SetLimits(mn,N,true); // Throw away last N-mn rows

    return;
}

std::tuple<DMatrix<double>,DiagonalMatrix<double>,DMatrix<double>> LaSVDecomp(const DMatrix<double>& A)
{
    int M=A.GetNumRows(),N=A.GetNumCols(),mn=Min(M,N);

    //
    //  Diced how much work space lapack needs
    //
    int info=0,lwork=-1;
    Vector<double> s(mn),work(1);
    DMatrix<double> U(A),VT(N,N);
    char jobu='O',jobv='A';
    //
    //  Initial call to see how much work space is needed
    //
    dgesvd_(&jobu,&jobv,&M,&N,&U(1,1),&M,&s(1),0,&M,&VT(1,1),&N,&work(1),&lwork,&info);
    lwork=work(1);
    work.SetLimits(lwork);
    //
    //  Now do the actual SVD work
    //
    dgesvd_(&jobu,&jobv,&M,&N,&U(1,1),&M,&s(1),0,&M,&VT(1,1),&N,&work(1),&lwork,&info);
    assert(info==0);
    //
    //  Now fix up the matrix limits
    //
    U .SetLimits(M,mn,true); //Throw away last N-mn columns
    VT.SetLimits(mn,N,true); // Throw away last N-mn rows
    DiagonalMatrix<double> ds(s);
    return std::make_tuple(U,ds,VT);
}

template <class T> void gesvd(char* JOBU,char* JOBV,int* M,int* N,T* A,int* LDA, double* S,T* U,
int* LDU,T* VT,int* LDVT,T* WORK,int* LWORK,int* INFO);

template <> void gesvd<double> (char* JOBU,char* JOBV,int* M,int* N,double* A,int* LDA, double* S,double* U,
int* LDU,double* VT,int* LDVT,double* WORK,int* LWORK,int* INFO)
{
    dgesvd_(JOBU,JOBV,M,N,A,LDA,S,U,LDU,VT,LDVT,WORK,LWORK,INFO); //double
}
template <> void gesvd<dcmplx> (char* JOBU,char* JOBV,int* M,int* N,dcmplx* A,int* LDA, double* S,dcmplx* U,
int* LDU,dcmplx* VT,int* LDVT,dcmplx* WORK,int* LWORK,int* INFO)
{
    Vector<double> rwork(5*Min(*M,*N));
    zgesvd_(JOBU,JOBV,M,N,A,LDA,S,U,LDU,VT,LDVT,WORK,LWORK,&rwork(1),INFO); //complex<double>
}

inline const double& real(const double& d) {return d;}
using std::real;

template <class T> typename LapackSVDSolver<T>::UsVType
LapackSVDSolver<T>::Solve(const MatrixT& A, int NumSingularValues,double eps)
{
    assert(NumSingularValues<=Min(A.GetNumRows(),A.GetNumCols()));
    int M=A.GetNumRows(),N=A.GetNumCols(),mn=Min(M,N);

    //
    //  Dicey deciding how much work space lapack needs. more is faster
    //
    int info=0,lwork=-1;
    Vector<double> s(mn);
    Vector<T> work(1);
    DMatrix<T> U(A),VT(N,N);
    char jobu='O',jobv='A';
    //
    //  Initial call to see how much work space is needed
    //
//    dgesvd_(&jobu,&jobv,&M,&N,&U(1,1),&M,&s(1),0,&M,&VT(1,1),&N,&work(1),&lwork,&info);
    gesvd<T>(&jobu,&jobv,&M,&N,&U(1,1),&M,&s(1),0,&M,&VT(1,1),&N,&work(1),&lwork,&info);
    lwork=real(work(1));
    work.SetLimits(lwork);
    //
    //  Now do the actual SVD work
    //
    gesvd<T>(&jobu,&jobv,&M,&N,&U(1,1),&M,&s(1),0,&M,&VT(1,1),&N,&work(1),&lwork,&info);
    assert(info==0);
    //
    //  Now fix up the matrix limits
    //
    U .SetLimits(M,mn,true); //Throw away last N-mn columns
    VT.SetLimits(mn,N,true); // Throw away last N-mn rows
    DiagonalMatrix<double> ds(s);
    return std::make_tuple(U,ds,VT);
}

template class LapackSVDSolver<double>;
template class LapackSVDSolver<dcmplx>;
template void LaSVDecomp<DMatrix<double> >(DMatrix<double>& A, Vector<double>& s, DMatrix<double>& VT);
