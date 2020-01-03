#ifndef MATRIX6_H
#define MATRIX6_H

#include "oml/dmatrix.h"

using std::cout;
using std::endl;
//
//  Store 6 index tensor as a flattened 2 index matrix.
//
template <class T> class Matrix6
{
    typedef DMatrix<T> MatrixT;
    public:
        Matrix6(int Dmn1,int D1,int Dmn2, int D2, int mnLow=0)
         : itsDmn1(Dmn1), itsD1(D1), itsDmn2(Dmn2), itsD2(D2), itsmnLow(1-mnLow)
         , itsData(itsDmn1*itsD1*itsD1,itsDmn2*itsD2*itsD2)
        {
            assert(itsDmn1>=itsmnLow);
            assert(itsDmn2>=itsmnLow);
            assert(itsD1>0);
            assert(itsD2>0);
            assert(itsmnLow>=0);
        }

        Matrix6(int Dw1,int Dw2)
         : itsDmn1(Dw1), itsD1(1), itsDmn2(Dw2), itsD2(1), itsmnLow(0)
         , itsData(itsDmn1*itsD1*itsD1,itsDmn2*itsD2*itsD2)
        {
            assert(itsDmn1>=itsmnLow);
            assert(itsDmn2>=itsmnLow);
            assert(itsD1>0);
            assert(itsD2>0);
            assert(itsmnLow>=0);
        }

        void Fill(const T& fill) {::Fill(itsData,fill);}

        T& operator()(int m, int i1, int i2, int n, int j1, int j2)
        {
            //cout << "m,i1,12,index=" << m << " " << i1 << " " << i2 << " " << index1(m,i1,i2) <<endl;
            //cout << "n,j1,j2,index=" << n << " " << j1 << " " << j2 << " " << index2(n,j1,j2) <<endl;
            return itsData(index1(m,i1,i2),index2(n,j1,j2));
        }
        const T& operator()(int m, int i1, int i2, int n, int j1, int j2) const
        {
            return itsData(index1(m,i1,i2),index2(n,j1,j2));
        }

        Matrix6& operator*=(const Matrix6& m6)
        {
            itsData*=m6.itsData;
            return *this;
        }

        void ClearLimits() {itsData.SetLimits(0,0);}

//        Matrix6& RightMultiply(const Matrix6& m6)
//        {
//            itsData*=m6.itsData;
//            return *this;
//        }


        friend std::ostream& operator<<(std::ostream& os,const Matrix6& m) {return os << m.itsData;}

    protected:
        int index1(int m, int i1, int i2) const {return m+itsmnLow+itsDmn1*(i1-1+itsD1*(i2-1));}
        int index2(int n, int j1, int j2) const {return n+itsmnLow+itsDmn2*(j1-1+itsD1*(j2-1));}

    private:
        int itsDmn1,itsD1,itsDmn2,itsD2, itsmnLow;
        MatrixT itsData;
};

#endif // MATRIX6_H
