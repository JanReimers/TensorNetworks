#include "SparseMatrix.H"
#include <algorithm>

using std::cout;
using std::endl;

template <class T>  SparseMatrix<T>::SparseMatrix(const DMatrix<T>& denseMatrix,double eps)
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

template <class T> SparseMatrix<T>::SparseMatrix(int Nr, int Nc)
 : itsNr(Nr)
 , itsNc(Nc)
 , itsTotalNumElements(0)
{

}

template <class T> T SparseMatrix<T>::operator()(int i, int j) const
{
    T ret=0.0;
    std::vector<int>::const_iterator ir=std::find(nonZeroRows.begin(),nonZeroRows.end(),i);
    if (ir!=nonZeroRows.end())
    {
        int rowIndex= std::distance(nonZeroRows.begin(), ir);
        const Row& r=Rows[rowIndex];
        std::vector<int>::const_iterator ic=std::find(r.nonZeroColumns.begin(),r.nonZeroColumns.end(),j);
        if (ic!=r.nonZeroColumns.end())
        {
            int colIndex= std::distance(r.nonZeroColumns.begin(), ic);
            ret=r.values[colIndex];
        }

    }
    return ret;
}

template <class T> void SparseMatrix<T>::Insert(const T& val,int i, int j)
{
    if (val==0.0) return;
    int rowIndex=-1,colIndex=-1;
    std::vector<int>::iterator ir=std::find(nonZeroRows.begin(),nonZeroRows.end(),i);
    if (ir==nonZeroRows.end())
    {
        nonZeroRows.push_back(i);
        rowIndex=nonZeroRows.size()-1;
        Rows.push_back(Row());
    }
    else
    {
        rowIndex= std::distance(nonZeroRows.begin(), ir);
    }
    Row& r=Rows[rowIndex];

    std::vector<int>::iterator ic=std::find(r.nonZeroColumns.begin(),r.nonZeroColumns.end(),j);
    if (ic==r.nonZeroColumns.end())
    {
        r.nonZeroColumns.push_back(j);
        colIndex=r.nonZeroColumns.size()-1;
        r.values.push_back(0.0);
        itsTotalNumElements++;
    }
    else
    {
        colIndex= std::distance(r.nonZeroColumns.begin(), ic);
    }
    r.values[colIndex]=val;
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


