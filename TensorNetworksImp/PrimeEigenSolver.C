#include "PrimeEigenSolver.H"
#include <primme.h>
#include "oml/vector_io.h"

using std::cout;
using std::endl;


void DenseMatvec(void *x, PRIMME_INT *ldx, void *y, PRIMME_INT *ldy, int *blockSize, primme_params *primme, int *ierr);
void SparseMatvec(void *x, PRIMME_INT *ldx, void *y, PRIMME_INT *ldy, int *blockSize, primme_params *primme, int *ierr);

template<class T> const SparseMatrix<T>* PrimeEigenSolver<T>::theSparseMatrix = 0;
template<class T> const      DMatrix<T>* PrimeEigenSolver<T>::theDenseMatrix = 0;

template <class T> PrimeEigenSolver<T>::PrimeEigenSolver(double eps)
: itsEps(eps)
, itsNumGuesses(0)
{

}

template <class T> PrimeEigenSolver<T>::~PrimeEigenSolver()
{

    //dtor
}

#include <iostream>
template <class T> void PrimeEigenSolver<T>::Solve(const DMatrix<T>& m, int NumEigenValues)
{
    assert(&m);
    assert(m.GetNumRows()==m.GetNumCols());
    SparseMatrix<T> sparsem(m,itsEps);
       //cout << "Sparsisty=" << itsSparsem.GetSparsisty() << "%" << endl;
    if (sparsem.GetSparsisty()<60)
    {
        theSparseMatrix=&sparsem;
        SolveSparse(NumEigenValues);
    }
    else
    {
        theDenseMatrix=&m;
        SolveDense(NumEigenValues);
    }
}

template <class T> void PrimeEigenSolver<T>::SolveSparse(int NumEigenValues)
{
    assert(theSparseMatrix);
    assert(theSparseMatrix->GetNumRows()==theSparseMatrix->GetNumCols());
    int N=theSparseMatrix->GetNumRows();
    primme_params primme;
    primme_initialize(&primme);
    primme.matrixMatvec = SparseMatvec;
    primme.n = N; /* set problem dimension */
    primme.numEvals = NumEigenValues;   /* Number of wanted eigenpairs */
    primme.eps = itsEps;      /* ||r|| <= eps * ||matrix|| */
    primme.target = primme_smallest; /* Wanted the smallest eigenvalues */

    primme.initSize=itsNumGuesses;
    primme_set_method(PRIMME_DYNAMIC, &primme);

    itsEigenValues.SetLimits(NumEigenValues);
    itsEigenVectors.SetLimits(N,NumEigenValues);
    Vector<double> rnorms(NumEigenValues);
    int ret = zprimme(&itsEigenValues(1), &itsEigenVectors(1,1), &rnorms(1), &primme);
    assert(ret==0);
    (void)ret; //avoid compiler warning in release mode
//    std::cout << "Max(abs(rnorms))=" <<  Max(abs(rnorms)) << " " << itsEps << std::endl;
    if (Max(abs(rnorms))>1000*itsEps)
        cout << "Warning high rnorms in PrimeEigenSolver::SolveSparse rnorma=" << std::scientific << rnorms << endl;

    primme_free(&primme);
    itsNumGuesses=NumEigenValues; //Set up using guesses for next time around

}
template <class T> void PrimeEigenSolver<T>::SolveDense(int NumEigenValues)
{
    assert(theDenseMatrix);
    assert(theDenseMatrix->GetNumRows()==theDenseMatrix->GetNumCols());
    int N=theDenseMatrix->GetNumRows();
    primme_params primme;
    primme_initialize(&primme);
    primme.matrixMatvec = DenseMatvec;
    primme.n = N; /* set problem dimension */
    primme.numEvals = NumEigenValues;   /* Number of wanted eigenpairs */
    primme.eps = itsEps;      /* ||r|| <= eps * ||matrix|| */
    primme.target = primme_smallest; /* Wanted the smallest eigenvalues */
    primme_set_method(PRIMME_DYNAMIC, &primme);

    itsEigenValues.SetLimits(NumEigenValues);
    itsEigenVectors.SetLimits(N,NumEigenValues);
    Vector<double> rnorms(NumEigenValues);
    int ret = zprimme(&itsEigenValues(1), &itsEigenVectors(1,1), &rnorms(1), &primme);
    assert(ret==0);
    (void)ret; //avoid compiler warning in release mode
//    std::cout << "Max(abs(rnorms))=" <<  Max(abs(rnorms)) << " " << itsEps << std::endl;
    if (Max(abs(rnorms))>100*itsEps)
        cout << "Warning high rnorms in PrimeEigenSolver::SolveDense rnorma=" << std::scientific << rnorms << endl;

    primme_free(&primme);

}

// Get lowest eigen value and vector with initial guess
//template <class T> void PrimeEigenSolver<T>::SolveLowest(const Vector<T>& EigenVectorGuess);

template <class T> Vector<T>  PrimeEigenSolver<T>::GetEigenVector(int index) const
{
    return itsEigenVectors.GetColumn(index);
}

//  matrix-vector product, Y = A * X
//   - X, input dense matrix of size primme.n x blockSize;
//   - Y, output dense matrix of size primme.n x blockSize;
//   - A, tridiagonal square matrix of dimension primme.n with this form:

void SparseMatvec(void *x, PRIMME_INT *_ldx, void *y, PRIMME_INT *_ldy, int *_blockSize, primme_params *primme, int *ierr)
{
    typedef PrimeEigenSolver<std::complex<double> > primmeT;
    assert(primmeT::theSparseMatrix);
    long int& ldx(*_ldx);
    long int& ldy(*_ldy);
    int& blockSize(*_blockSize);

    for (int ib=0; ib<blockSize; ib++)
    {
        std::complex<double>* xvec = static_cast<std::complex<double> *>(x) + ldx*ib;
        std::complex<double>* yvec = static_cast<std::complex<double> *>(y) + ldy*ib;
        primmeT::theSparseMatrix->DoMVMultiplication(primme->n,xvec,yvec);
    }
    *ierr = 0;

}
void DenseMatvec(void *x, PRIMME_INT *_ldx, void *y, PRIMME_INT *_ldy, int *_blockSize, primme_params *primme, int *ierr)
{
    typedef PrimeEigenSolver<std::complex<double> > primmeT;
    assert(primmeT::theDenseMatrix);
    long int& ldx(*_ldx);
    long int& ldy(*_ldy);
    int& blockSize(*_blockSize);
    int N=primme->n;

    for (int ib=0; ib<blockSize; ib++)
    {
        std::complex<double>* xvec = static_cast<std::complex<double> *>(x) + ldx*ib;
        std::complex<double>* yvec = static_cast<std::complex<double> *>(y) + ldy*ib;
        for (int ir=1;ir<=N;ir++)
        {
            yvec[ir-1]=0.0;
            for (int ic=1;ic<=N;ic++)
                yvec[ir-1]+=(*primmeT::theDenseMatrix)(ir,ic)*xvec[ic-1];
        }
    }
    *ierr = 0;

}


template class PrimeEigenSolver<std::complex<double> >;

