#include "SparseMatrix.H"

using std::cout;
using std::endl;

template <class T>  SparseMatrix<T>::SparseMatrix(const DMatrix<T>& denseMatrix,double eps)\
 : itsNr(denseMatrix.GetNumRows())
 , itsNc(denseMatrix.GetNumCols())
 ,itsTotalNumElements(0)
{
    for (int i=1;i<=itsNr;i++)
        for (int j=1;j<=itsNc;j++)
            if (abs(denseMatrix(i,j))>eps)
            {
                if (nonZeroRows.size()==0 || nonZeroRows.back()!=i)
                {
                    nonZeroRows.push_back(i);
                    Rows.push_back(Row());
                }
                Row& r=Rows.back();
                r.nonZeroColumns.push_back(j);
                r.values.push_back(denseMatrix(i,j));
                itsTotalNumElements++;
            }
}

template <class T> void SparseMatrix<T>::Dump(std::ostream& os) const
{
    for (long unsigned int ir=0;ir<nonZeroRows.size();ir++)
    {
        const Row& r=Rows[ir];
        for (long unsigned int ic=0;ic<r.nonZeroColumns.size();ic++)
            os << "[" << nonZeroRows[ir] << "," << r.nonZeroColumns[ic] << "]=" << r.values[ic] << std::endl;
    }
}

template <class T> void SparseMatrix<T>::DoMVMultiplication(int N, T* xvec,T* yvec) const
{
    assert(N==itsNr);
    assert(N==itsNc);
    for (long unsigned int ir=0; ir<nonZeroRows.size(); ir++)
    {
        const Row& r=Rows[ir];
        yvec[nonZeroRows[ir]-1]=T(0.0);
        for (long unsigned int ic=0; ic<r.nonZeroColumns.size(); ic++)
        {
//            cout << "" << nonZeroRows[ir]-1 << " " << r.nonZeroColumns[ic]-1 << " " << r.values[ic] << endl;
            yvec[nonZeroRows[ir]-1]+=xvec[r.nonZeroColumns[ic]-1] * r.values[ic];
        }
    }
}

#include <complex>
template class SparseMatrix<std::complex<double> >;


