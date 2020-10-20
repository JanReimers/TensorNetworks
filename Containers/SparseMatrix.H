#ifndef SPARSEMATRIX_H
#define SPARSEMATRIX_H

#include "oml/matrix.h"
#include <vector>
#include <map>
#include <iostream>

template <class T> class SparseMatrix
{
    public:
        SparseMatrix(const DMatrix<T>& denseMatrix,double eps=0.0);
        SparseMatrix(const MatLimits&);
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

        void DoMVMultiplication(int N, const T* xvec,T* yvec) const;
        void DoMVMultiplication(int M, int N, const T* xvec,T* yvec,int transpose=0) const;
        MatLimits GetLimits() const {return MatLimits(itsNr,itsNc);}
        int GetNumRows() const {return itsNr;}
        int GetNumCols() const {return itsNc;}
        double GetDensity() const {return (100.0*itsTotalNumElements)/(itsNr*itsNc);}

    protected:
        void Dump(std::ostream& os) const;
        void AssignFrom(const DMatrix<T>& denseMatrix,double eps);

    private:
        int itsNr,itsNc;
        int itsTotalNumElements;

        typedef typename std::vector<T>::iterator v_iterator;
        typedef typename std::map<int,int > mapV;
        typedef typename std::map<int,mapV> mapT;
        typedef typename std::vector<T> vecT;
        mapT RowIndexes;
        mapT ColIndexes;
        vecT itsValues;

private:
        class iterator_base
        {
            public:
            iterator_base(mapT& i,vecT& v)
                : RCIndexes(i), Values(v)
                , r(i.begin()), c(r->second.begin()) {}

                bool end() const {return r==RCIndexes.end();}
                void operator++(int)
                {
                    if (++c;c==r->second.end  ())
                    {
                        ++r;
                        c =r->second.begin();
                    }
                }

                int RowIndex() const {return r->first;}
                int ColIndex() const {return c->first;}
        protected:
            mapT& RCIndexes;
            vecT& Values;
            mapT::iterator r;
            mapV::iterator c;
        };
public:
    class iterator : public iterator_base
    {
        public:
            iterator(SparseMatrix& m) : iterator_base(m.RowIndexes,m.itsValues) {}
            const T & Value() const {return iterator_base::Values[iterator_base::c->second];}
    };

    class const_iterator : public iterator_base
    {
        public:
            const_iterator(const SparseMatrix& m)
            : iterator_base(const_cast<SparseMatrix&>(m).RowIndexes,const_cast<SparseMatrix&>(m).itsValues)
             {}
            const T & Value() const {return iterator_base::Values[iterator_base::c->second];}
    };

};


template <class T> DMatrix<T> operator-(const SparseMatrix<T>& a, const DMatrix<T>& b)
{
    assert(a.GetLimits()==b.GetLimits());
    DMatrix<T> ret(a.GetLimits());
    Fill(ret,T(0));
    for (int i=1;i<=a.GetNumRows();i++)
        for (int j=1;j<=b.GetNumCols();j++)
            ret(i,j)=a(i,j)-b(i,j);
    return ret;
}

template <class TA, class TB> auto operator*(const SparseMatrix<TA>& a, const DMatrix<TB>& b)
{
    assert(a.GetNumCols()==b.GetNumRows());
    typedef typename BinaryRetType<TA,TB,OpMul<TA,TB> >::RetType TR;

    DMatrix<TR> ret(a.GetNumRows(),b.GetNumCols());
    Fill(ret,TR(0));
    for (typename SparseMatrix<TA>::const_iterator ia(a);!ia.end();ia++)
        for (int j=1;j<=b.GetNumCols();j++)
                ret(ia.RowIndex(),j)+=ia.Value()*b(ia.ColIndex(),j);
    return ret;
}

template <class TA, class TB> auto operator*(const DMatrix<TA>& a, const SparseMatrix<TB>& b)
{
    assert(a.GetNumCols()==b.GetNumRows());
    typedef typename BinaryRetType<TA,TB,OpMul<TA,TB> >::RetType TR;
    DMatrix<TR> ret(a.GetNumRows(),b.GetNumCols());
    Fill(ret,TR(0));
    for (typename SparseMatrix<TB>::const_iterator ib(b);!ib.end();ib++)
        for (int i=1;i<=a.GetNumRows();i++)
            ret(i,ib.ColIndex())+=a(i,ib.RowIndex())*ib.Value();
//        for (int j=1;j<=b.GetNumCols();j++)
//            for (int k=1;k<=a.GetNumCols();k++)
//                ret(i,j)+=a(i,k)*b(k,j);
    return ret;
}

template <class TA, class TB> auto operator*(const SparseMatrix<TA>& a, const Vector<TB>& b)
{
    assert(a.GetNumCols()==b.size());
    typedef typename BinaryRetType<TA,TB,OpMul<TA,TB> >::RetType TR;

    Vector<TR> ret(a.GetNumRows());
    Fill(ret,TR(0));
    for (typename SparseMatrix<TA>::const_iterator i(a);!i.end();i++)
         ret(i.RowIndex())+=i.Value()*b(i.ColIndex());

//    for (int i=1;i<=a.GetNumRows();i++)
//        for (int j=1;j<=b.size();j++)
//            ret(i)+=a(i,j)*b(j);
    return ret;
}

inline const double& conj(const double& d) { return d;}

template <class T> bool IsHermitian(const SparseMatrix<T>& m, double eps=0.0)
{
    bool ret=true;
    for (int i=1;i<=m.GetNumRows();i++)
        for (int j=i+1;j<=m.GetNumCols();j++)
            ret=ret && (fabs(m(i,j)-conj(m(j,i)))<eps);
    return ret;
}
#endif // SPARSEMATRIX_H
