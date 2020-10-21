#include "SparseMatrix.H"
#include <algorithm>

using std::cout;
using std::endl;

template <class T>  SparseMatrix<T>::SparseMatrix(const Matrix<T>& denseMatrix,double eps)
 : itsNr(denseMatrix.GetNumRows())
 , itsNc(denseMatrix.GetNumCols())
 , itsTotalNumElements(0)
{
      AssignFrom(denseMatrix,eps);
}

template <class T> SparseMatrix<T>::SparseMatrix(const MatLimits& lim)
 : itsNr(lim.GetNumRows())
 , itsNc(lim.GetNumCols())
 , itsTotalNumElements(0)
{}

template <class T> SparseMatrix<T>::SparseMatrix(int Nr, int Nc)
 : itsNr(Nr)
 , itsNc(Nc)
 , itsTotalNumElements(0)
{}

template <class T> SparseMatrix<T>::SparseMatrix()
 : itsNr(0)
 , itsNc(0)
 , itsTotalNumElements(0)
{}

template <class T> SparseMatrix<T>& SparseMatrix<T>::operator=(const Matrix<T>& denseMatrix)
{
      AssignFrom(denseMatrix,0);
      return *this;
}

template <class T> void SparseMatrix<T>::AssignFrom(const Matrix<T>& denseMatrix,double eps)
{
    itsTotalNumElements=0;
    itsNr=denseMatrix.GetNumRows();
    itsNc=denseMatrix.GetNumCols();
    for (int i=1;i<=itsNr;i++)
        for (int j=1;j<=itsNc;j++)
            if (T val=denseMatrix(i,j);fabs(val)>eps)
            {
                Insert(val,i,j);
                itsTotalNumElements++;
            }

}


template <class T> T SparseMatrix<T>::operator()(int i, int j) const
{
    assert(i>=1);
    assert(j>=1);
    assert(i<=itsNr);
    assert(j<=itsNc);
    T ret=0.0;
    if (auto ir=RowIndexes.find(i);ir!=RowIndexes.end())
        if (auto ic=ir->second.find(j);ic!=ir->second.end())
            ret=itsValues[ic->second];

    return ret;
}

template <class T> void SparseMatrix<T>::Insert(const T& val,int i, int j)
{
    if (val==0.0) return;
    itsValues.push_back(val);
    int index=itsValues.size()-1;
    RowIndexes[i][j]=index;
    ColIndexes[j][i]=index;
}


template <class T> void SparseMatrix<T>::Dump(std::ostream& os) const
{
//    for (const_iterator i(*this);!i.end();i++)
//        os << "(" << i.RowIndex() << "," << i.ColIndex() << ")=" << i.Value() << endl;
//    os << "---------------------------------------" << endl;
    os << endl;
   for (int ir=1;ir<=itsNr;ir++)
   {
        for (int ic=1;ic<=itsNc;ic++)
             os << (*this)(ir,ic) << " ";
         os << endl;
    }
}

template <class T> void SparseMatrix<T>::DoMVMultiplication(int N, const T* xvec,T* yvec) const
{
    assert(N==itsNr);
    assert(N==itsNc);
    for (int i=1;i<=itsNr;i++) yvec[i-1]=0.0;
    for (const_iterator i(*this);!i.end();i++)
    {
//        cout << "i,j,y[i],A(i,j),x[j]=" << i.RowIndex() << " " << i.ColIndex() << " " << yvec[i.RowIndex()-1] << " " << i.Value() << " " << xvec[i.ColIndex()-1] << endl;
         yvec[i.RowIndex()-1]+=xvec[i.ColIndex()-1] * i.Value();
    }
}


template <class T> void SparseMatrix<T>::DoMVMultiplication(int M, int N, const T* xvec,T* yvec, int transpose) const
{
    assert(M==itsNr);
    assert(N==itsNc);
    if (transpose==0)
    {
        for (int i=1;i<=itsNr;i++) yvec[i-1]=0.0;
        for (const_iterator i(*this);!i.end();i++)
             yvec[i.RowIndex()-1]+=xvec[i.ColIndex()-1] * i.Value();
    }
    else
    {
        for (int i=1;i<=itsNc;i++) yvec[i-1]=0.0;
        for (const_iterator i(*this);!i.end();i++)
            yvec[i.ColIndex()-1]+=xvec[i.RowIndex()-1] * conj(i.Value());
    } //if else
}

#include <complex>
template class SparseMatrix<double>;
template class SparseMatrix<std::complex<double> >;


