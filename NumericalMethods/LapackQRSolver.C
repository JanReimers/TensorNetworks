#include "LapackQRSolver.H"
#include "oml/matrix.h"
#include "oml/vector.h"
#include <complex>
//
// See http://www.netlib.org/lapack/explore-html/d1/d7e/group__double_g_esing_ga84fdf22a62b12ff364621e4713ce02f2.html
// for detailed docs
// you also need to add -llapack to the link command
typedef std::complex<double> dcmplx;

extern"C" {
void dgeqrf_(int* M,int* N,double* A,int* LDA,double* TAU,double* WORK,int* LWORK,int* INFO);
void zgeqrf_(int* M,int* N,dcmplx* A,int* LDA,dcmplx* TAU,dcmplx* WORK,int* LWORK,int* INFO);

void dorgqr_(int* M,int* N,int*	K,double* A,int* LDA,double* TAU,double* WORK,int* LWORK,int* INFO);
void zungqr_(int* M,int* N,int*	K,dcmplx* A,int* LDA,dcmplx* TAU,dcmplx* WORK,int* LWORK,int* INFO);

}

template <class T> void xgeqrf  (int* M,int* N,T* A,int* LDA,T* TAU,T* WORK,int* LWORK,int* INFO);
template <> void xgeqrf<double> (int* M,int* N,double* A,int* LDA,double* TAU,double* WORK,int* LWORK,int* INFO)
{
    dgeqrf_(M,N,A,LDA,TAU,WORK,LWORK,INFO); //double
}
template <> void xgeqrf<dcmplx> (int* M,int* N,dcmplx* A,int* LDA,dcmplx* TAU,dcmplx* WORK,int* LWORK,int* INFO)
{
    zgeqrf_(M,N,A,LDA,TAU,WORK,LWORK,INFO); //complex<double>
}

template <class T> void xungqr  (int* M,int* N,int*	K,T* A,int* LDA,T* TAU,T* WORK,int* LWORK,int* INFO);
template <> void xungqr<double> (int* M,int* N,int*	K,double* A,int* LDA,double* TAU,double* WORK,int* LWORK,int* INFO)
{
    dorgqr_(M,N,K,A,LDA,TAU,WORK,LWORK,INFO); //double
}
template <> void xungqr<dcmplx> (int* M,int* N,int*	K,dcmplx* A,int* LDA,dcmplx* TAU,dcmplx* WORK,int* LWORK,int* INFO)
{
    zungqr_(M,N,K,A,LDA,TAU,WORK,LWORK,INFO); //complex<double>
}


template <class T> typename LapackQRSolver<T>::QRType LapackQRSolver<T>::SolveThin(const MatrixT& Ain)
{
    int M=Ain.GetNumRows(),N=Ain.GetNumCols(),mn=Min(M,N);

    //
    //  Diced how much work space lapack needs
    //
    int info=0,lwork=-1;
    Vector<T> tau(mn),work(1);
    Matrix<T> A(Ain);
    //
    //  Initial call to see how much work space is needed
    //
    xgeqrf<T>(&M,&N,&A(1,1),&M,&tau(1),&work(1),&lwork,&info);
    lwork=static_cast<int>(real(work(1)));
    work.SetLimits(lwork);
    //
    //  Now do the actual QR work
    //
    xgeqrf<T>(&M,&N,&A(1,1),&M,&tau(1),&work(1),&lwork,&info);
    assert(info==0);
    //
    //  Grab R before xungqr clobbers it.
    //
    Matrix<T> R(mn,N);
    for (int i=1;i<=mn;i++)
    {
        for (int j=1;j<i;j++)
            R(i,j)=0.0;
        for (int j=i;j<=N;j++)
            R(i,j)=A(i,j);

    }
    //
    //  unpack Q
    //
    lwork=-1;
    info=0;
    xungqr<T>(&M,&mn,&mn,&A(1,1),&M,&tau(1),&work(1),&lwork,&info);
    lwork=static_cast<int>(real(work(1)));
    work.SetLimits(lwork);
    xungqr<T>(&M,&mn,&mn,&A(1,1),&M,&tau(1),&work(1),&lwork,&info);
    if (mn<N) A.SetLimits(M,mn,true);
    return std::make_tuple(std::move(A),std::move(R));
}

template class LapackQRSolver<double>;
template class LapackQRSolver<dcmplx>;
