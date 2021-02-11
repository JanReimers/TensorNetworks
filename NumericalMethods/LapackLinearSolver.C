#include "LapackLinearSolver.H"
#include "oml/matrix.h"
#include "oml/vector.h"
#include <complex>
//
// See http://www.netlib.org/lapack/explore-html/da/dba/group__double_o_t_h_e_rcomputational_ga7068947990361e55177155d044435a5c.html
// for detailed docs
// you also need to add -llapack to the link command
typedef std::complex<double> dcmplx;

extern"C" {
void dtrtrs_(char* UPLO,char* TRANS, char* DIAG, int* N, int* NRHS, const double* A,int* LDA,double* B,int* LDB,int* INFO);
void ztrtrs_(char* UPLO,char* TRANS, char* DIAG, int* N, int* NRHS, const dcmplx* A,int* LDA,dcmplx* B,int* LDB,int* INFO);
}

template <class T> void xtrtrs  (char* UPLO,char* TRANS, char* DIAG, int* N, int* NRHS, const T     * A,int* LDA,T     * B,int* LDB,int* INFO);
template <> void xtrtrs<double> (char* UPLO,char* TRANS, char* DIAG, int* N, int* NRHS, const double* A,int* LDA,double* B,int* LDB,int* INFO)
{
    dtrtrs_(UPLO,TRANS,DIAG,N,NRHS,A,LDA,B,LDB,INFO); //double
}
template <> void xtrtrs<dcmplx> (char* UPLO,char* TRANS, char* DIAG, int* N, int* NRHS, const dcmplx* A,int* LDA,dcmplx* B,int* LDB,int* INFO)
{
    ztrtrs_(UPLO,TRANS,DIAG,N,NRHS,A,LDA,B,LDB,INFO); //complex<double>
}

template <class T> typename LapackLinearSolver<T>::VectorT LapackLinearSolver<T>::SolveUpperTri(const MatrixT& A,const VectorT& b)
{
    assert(IsUpperTriangular(A));
    return SolveTri(A,b,'U');
}

template <class T> typename LapackLinearSolver<T>::VectorT LapackLinearSolver<T>::SolveLowerTri(const MatrixT& A,const VectorT& b)
{
    assert(IsLowerTriangular(A));
    return SolveTri(A,b,'L');
}

template <class T> typename LapackLinearSolver<T>::VectorT LapackLinearSolver<T>::SolveTri(const MatrixT& A,const VectorT& b,char UL)
{
    int N=A.GetNumRows();
    assert(A.GetNumCols()==N);
    assert(b.size()==N);
    int info=0,nrhs=1;
    char cN('N');
    Vector<T> x(b);
    //
    //  Initial call to see how much work space is needed
    //
    xtrtrs<T>(&UL,&cN,&cN,&N,&nrhs,&A(1,1),&N,&x(1),&N,&info);
    assert(info==0);
    return x;
}

template class LapackLinearSolver<double>;
template class LapackLinearSolver<dcmplx>;