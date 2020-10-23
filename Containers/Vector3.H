#ifndef VECTOR3_H
#define VECTOR3_H

#include "oml/vector.h"
//
//  Store 6 index tensor as a flattened 2 index matrix.
//
template <class T> class Vector3
{
    typedef Vector<T> VectorT;
    public:
        Vector3()
         : M1(0),M2(0),M3(0), itsmnLow(1)
         , itsData(0)
        {}

        Vector3(int _M1,int _M2,int _M3, const VectorT& v, int mnLow=0)
         : M1(_M1),M2(_M2),M3(_M3), itsmnLow(1-mnLow)
         , itsData(v)
        {
            assert(M1>=itsmnLow);
            assert(M2>0);
            assert(M3>0);
            assert(itsmnLow>=0);
            assert(itsData.size()==M1*M2*M3);
        }

        Vector3(int _M1,int _M2,int _M3, int mnLow=0)
         : M1(_M1),M2(_M2),M3(_M3), itsmnLow(1-mnLow)
         , itsData(M1*M2*M3)
        {
            assert(M1>=itsmnLow);
            assert(M2>0);
            assert(M3>0);
            assert(itsmnLow>=0);
            assert(itsData.size()==M1*M2*M3);
        }

        void Fill(const T& fill) {::Fill(itsData,fill);}
       class Subscriptor
        {
        public:
            Subscriptor(Vector3& m) : itsS(m.itsData), itsM(m) {}
            T& operator()(int m, int i1, int i2)
            {
                return itsS(itsM.index1(m,i1,i2));
            }
            private:
            typename VectorT::Subscriptor itsS;
            const Vector3& itsM;
            };

        T& operator()(int m, int i1, int i2)
        {
            //cout << "m,i1,12,index=" << m << " " << i1 << " " << i2 << " " << index1(m,i1,i2) <<endl;
            //cout << "N1,n,j1,j2,index=" << N1 << " " << n << " " << j1 << " " << j2 << " " << index2(n,j1,j2) <<endl;

            assert(m+itsmnLow<=M1);
            assert(i1<=M2);
            assert(i2<=M3);
            return itsData(index1(m,i1,i2));
        }
        const T operator()(int m, int i1, int i2) const
        {
            //cout << "m,i1,12,index=" << m << " " << i1 << " " << i2 << " " << index1(m,i1,i2) <<endl;
            //cout << "N1,n,j1,j2,index=" << N1 << " " << n << " " << j1 << " " << j2 << " " << index2(n,j1,j2) <<endl;
            assert(m+itsmnLow<=M1);
            assert(i1<=M2);
            assert(i2<=M3);
            return itsData(index1(m,i1,i2));
        }


        void ClearLimits() {itsData.SetLimits(0,0);}

        VectorT& Flatten() {return itsData;}

        friend std::ostream& operator<<(std::ostream& os,const Vector3& m) {return os << m.itsData;}

    protected:
        int index1(int m, int i1, int i2) const {return m+itsmnLow+M1*(i1-1+M2*(i2-1));}

    private:
        int M1,M2,M3;
        int itsmnLow;
        VectorT itsData;
};

#endif // VECTOR3_H
