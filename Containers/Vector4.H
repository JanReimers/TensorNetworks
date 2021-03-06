#ifndef VECTOR4_H
#define VECTOR4_H

#include "oml/vector.h"

using std::cout;
using std::endl;
//
//  Store 6 index tensor as a flattened 2 index matrix.
//
template <class T> class Vector4
{
    typedef Vector<T> VectorRT;
    public:
        Vector4()
         : M1(0),M2(0),M3(0),M4(0), itsmnLow(1)
         , itsData(0)
        {}

        Vector4(int _M1,int _M2,int _M3, int _M4, const VectorRT& v, int mnLow=0)
         : M1(_M1),M2(_M2),M3(_M3),M4(_M4), itsmnLow(1-mnLow)
         , itsData(v)
        {
            assert(M1>=itsmnLow);
            assert(M2>0);
            assert(M3>0);
            assert(M4>0);
            assert(itsmnLow>=0);
            assert(itsData.size()==M1*M2*M3*M4);
        }

        Vector4(int _M1,int _M2,int _M3, int _M4, int mnLow=0)
         : M1(_M1),M2(_M2),M3(_M3),M4(_M4), itsmnLow(1-mnLow)
         , itsData(M1*M2*M3*M4)
        {
            assert(M1>=itsmnLow);
            assert(M2>0);
            assert(M3>0);
            assert(M4>0);
            assert(itsmnLow>=0);
            assert(itsData.size()==M1*M2*M3*M4);
        }

        void Fill(const T& fill) {::Fill(itsData,fill);}

        T& operator()(int m, int n, int i1, int i2)
        {
            //cout << "m,i1,12,index=" << m << " " << i1 << " " << i2 << " " << index1(m,i1,i2) <<endl;
            //cout << "N1,n,j1,j2,index=" << N1 << " " << n << " " << j1 << " " << j2 << " " << index2(n,j1,j2) <<endl;

            assert(m+itsmnLow<=M1);
            assert(n+itsmnLow<=M2);
            assert(i1<=M3);
            assert(i2<=M4);
            return itsData(index1(m,n,i1,i2));
        }
        const T operator()(int m, int n, int i1, int i2) const
        {
            //cout << "m,i1,12,index=" << m << " " << i1 << " " << i2 << " " << index1(m,i1,i2) <<endl;
            //cout << "N1,n,j1,j2,index=" << N1 << " " << n << " " << j1 << " " << j2 << " " << index2(n,j1,j2) <<endl;
            assert(m+itsmnLow<=M1);
            assert(n+itsmnLow<=M2);
            assert(i1<=M3);
            assert(i2<=M4);
            return itsData(index1(m,n,i1,i2));
        }


        void ClearLimits() {itsData.SetLimits(0,0);}

        VectorRT& Flatten() {return itsData;}

        friend std::ostream& operator<<(std::ostream& os,const Vector4& m) {return os << m.itsData;}

    protected:
        int index1(int m, int n, int i1, int i2) const {return m+itsmnLow+M1*(n+itsmnLow-1+M2*(i1-1+M3*(i2-1)));}

    private:
        int M1,M2,M3,M4;
        int itsmnLow;
        VectorRT itsData;
};

#endif // VECTOR4_H
