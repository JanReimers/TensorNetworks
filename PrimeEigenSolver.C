#include "PrimeEigenSolver.H"
#include <primme.h>

void MyMatvec(void *x, PRIMME_INT *ldx, void *y, PRIMME_INT *ldy, int *blockSize, primme_params *primme, int *ierr);

template<class T> const SparseMatrix<T>* PrimeEigenSolver<T>::theMatrix = 0;

template <class T> PrimeEigenSolver<T>::PrimeEigenSolver(const SparseMatrix<T>& m, double eps)
: itsEps(eps)
{
    assert(&m);
    assert(m.GetNumRows()==m.GetNumCols());
    theMatrix=&m;
    //ctor
}

template <class T> PrimeEigenSolver<T>::~PrimeEigenSolver()
{

    //dtor
}

#include <iostream>

template <class T> void PrimeEigenSolver<T>::Solve(int NumEigenValues)
{
    int N=theMatrix->GetNumRows();
    assert(theMatrix);
    assert(theMatrix->GetNumRows()==theMatrix->GetNumCols());
    primme_params primme;
    primme_initialize(&primme);
    primme.matrixMatvec = MyMatvec;
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
    assert(Max(abs(rnorms))<100*itsEps);

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

void MyMatvec(void *x, PRIMME_INT *_ldx, void *y, PRIMME_INT *_ldy, int *_blockSize, primme_params *primme, int *ierr)
{
    typedef PrimeEigenSolver<std::complex<double> > primmeT;
    assert(primmeT::theMatrix);
    long int& ldx(*_ldx);
    long int& ldy(*_ldy);
    int& blockSize(*_blockSize);

    for (int ib=0; ib<blockSize; ib++)
    {
        std::complex<double>* xvec = static_cast<std::complex<double> *>(x) + ldx*ib;
        std::complex<double>* yvec = static_cast<std::complex<double> *>(y) + ldy*ib;
        primmeT::theMatrix->DoMVMultiplication(primme->n,xvec,yvec);
    }
    *ierr = 0;

}

template class PrimeEigenSolver<std::complex<double> >;

