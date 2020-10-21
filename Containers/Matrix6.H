#ifndef MATRIX6_H
#define MATRIX6_H

#include "oml/matrix.h"

using std::cout;
using std::endl;
//
//  Store 6 index tensor as a flattened 2 index matrix.
//
template <class T> class Matrix6
{
    typedef Matrix<T> MatrixRT;
    public:
        Matrix6(int _M1,int _M2,int _M3, int _N1, int _N2, int _N3, int mnLow=0)
         : M1(_M1),M2(_M2),M3(_M3),N1(_N1),N2(_N2),N3(_N3), itsmnLow(1-mnLow)
         , itsData(M1*M2*M3,N1*N2*N3)
        {
            assert(M1>=itsmnLow);
            assert(N1>=itsmnLow);
            assert(M2>0);
            assert(N2>0);
            assert(M3>0);
            assert(N3>0);
            assert(itsmnLow>=0);
        }

        Matrix6(int _M1,int _N1)
         : M1(_M1),M2(1),M3(1),N1(_N1),N2(1),N3(1), itsmnLow(0)
         , itsData(M1*M2*M3,N1*N2*N3)
        {
            assert(M1>=itsmnLow);
            assert(N1>=itsmnLow);
            assert(M2>0);
            assert(N2>0);
            assert(M3>0);
            assert(N3>0);
            assert(itsmnLow>=0);
        }

        void Fill(const T& fill) {::Fill(itsData,fill);}

        T& operator()(int m, int i1, int i2, int n, int j1, int j2)
        {
            //cout << "m,i1,12,index=" << m << " " << i1 << " " << i2 << " " << index1(m,i1,i2) <<endl;
            //cout << "N1,n,j1,j2,index=" << N1 << " " << n << " " << j1 << " " << j2 << " " << index2(n,j1,j2) <<endl;

            assert(m+itsmnLow<=M1);
            assert(n+itsmnLow<=N1);
            assert(i1<=M2);
            assert(i2<=M3);
            assert(j1<=N2);
            assert(j2<=N3);
            return itsData(index1(m,i1,i2),index2(n,j1,j2));
        }
        const T operator()(int m, int i1, int i2, int n, int j1, int j2) const
        {
            //cout << "m,i1,12,index=" << m << " " << i1 << " " << i2 << " " << index1(m,i1,i2) <<endl;
            //cout << "N1,n,j1,j2,index=" << N1 << " " << n << " " << j1 << " " << j2 << " " << index2(n,j1,j2) <<endl;
            assert(m+itsmnLow<=M1);
            assert(n+itsmnLow<=N1);
            assert(i1<=M2);
            assert(i2<=M3);
            assert(j1<=N2);
            assert(j2<=N3);
            return itsData(index1(m,i1,i2),index2(n,j1,j2));
        }

        Matrix6& operator*=(const Matrix6& m6)
        {
            itsData*=m6.itsData;
            N1=m6.N1;
            N2=m6.N2;
            N3=m6.N3;
            return *this;
        }

        void ClearLimits() {itsData.SetLimits(0,0);}

        MatrixRT& Flatten() {return itsData;}

        friend std::ostream& operator<<(std::ostream& os,const Matrix6& m) {return os << m.itsData;}

        void Dump(std::ostream& os) const
        {
        for (int w1=1-itsmnLow; w1<=M1-itsmnLow; w1++)
            for (int w2=1-itsmnLow; w2<=N1-itsmnLow; w2++)
                for (int i1=1; i1<=M2; i1++)
                    for (int j1=1; j1<=M3; j1++)
                        for (int i2=1; i2<=N2; i2++)
                            for (int j2=1; j2<=N3; j2++)
                                os << "   [" << w1 << ","  << i1 << ","  << j1 << ","  << w2 << ","  << i2 << "," << j2 << "]=" << (*this)(w1,i1,j1,w2,i2,j2) << endl;
        }

    protected:
        int index1(int m, int i1, int i2) const {return m+itsmnLow+M1*(i1-1+M2*(i2-1));}
        int index2(int n, int j1, int j2) const {return n+itsmnLow+N1*(j1-1+N2*(j2-1));}

    private:
        int M1,M2,M3,N1,N2,N3;
        int itsmnLow;
        MatrixRT itsData;
};

#endif // MATRIX6_H
