#ifndef SPARSEMATRIX_H
#define SPARSEMATRIX_H

#include "oml/dmatrix.h"
#include <vector>
#include <iostream>

template <class T> class SparseMatrix
{
    public:
        SparseMatrix(const DMatrix<T>& denseMatrix,double eps=0.0);
        SparseMatrix(int Nr, int Nc);
        SparseMatrix();
        ~SparseMatrix() {};

        SparseMatrix& operator=(const DMatrix<T>&);

        void Insert(const T&,int i, int j);
        T operator()(int i, int j) const;

        friend std::ostream& operator<<(std::ostream& os,const SparseMatrix<T>& m)
        {
            m.Dump(os);
            return os;
        }

        void DoMVMultiplication(int N, T* xvec,T* yvec) const;
        void DoMVMultiplication(int M, int N, T* xvec,T* yvec,int transpose=0) const;
        MatLimits GetLimits() const {return MatLimits(itsNr,itsNc);}
        int GetNumRows() const {return itsNr;}
        int GetNumCols() const {return itsNc;}
        double GetDensity() const {return (100.0*itsTotalNumElements)/(itsNr*itsNc);}
    protected:
        void Dump(std::ostream& os) const;
        void AssignFrom(const DMatrix<T>& denseMatrix,double eps);

    private:
        int itsNr,itsNc;
        struct Row
        {
            std::vector<int> nonZeroColumns;
            std::vector<T>   values;
        };
        std::vector<int> nonZeroRows;
        std::vector<Row> Rows;
        int itsTotalNumElements;
};

template <class T> DMatrix<T> operator-(const SparseMatrix<T>& a, const DMatrix<T>& b)
{
    assert(a.GetLimits()==b.GetLimits());
    DMatrix<T> ret(a.GetLimits());
    for (int i=1;i<=a.GetNumRows();i++)
        for (int j=1;j<=b.GetNumCols();j++)
            ret(i,j)=a(i,j)-b(i,j);
    return ret;
}

#endif // SPARSEMATRIX_H
