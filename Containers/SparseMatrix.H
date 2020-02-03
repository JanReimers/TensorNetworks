#ifndef SPARSEMATRIX_H
#define SPARSEMATRIX_H

#include "oml/dmatrix.h"
#include <vector>
#include <iostream>

template <class T> class SparseMatrix
{
    public:
        SparseMatrix(const DMatrix<T>& denseMatrix,double eps);
        ~SparseMatrix() {};

        friend std::ostream& operator<<(std::ostream& os,const SparseMatrix<T>& m)
        {
            m.Dump(os);
            return os;
        }

        void DoMVMultiplication(int N, T* xvec,T* yvec) const;
        int GetNumRows() const {return itsNr;}
        int GetNumCols() const {return itsNc;}
        double GetDensity() const {return (100.0*itsTotalNumElements)/(itsNr*itsNc);}
    protected:
        void Dump(std::ostream& os) const;


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

#endif // SPARSEMATRIX_H
