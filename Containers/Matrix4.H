#ifndef MATRIX4_H
#define MATRIX4_H

#include "oml/dmatrix.h"

using std::cout;
using std::endl;
//
//  Store 4 index tensor as a flattened 2 index matrix.
//
template <class T> class Matrix4
{
    typedef DMatrix<T> MatrixRT;
    public:
        Matrix4(int Dw1,int D1, int Dw2, int D2, int wLow=1)
         : itsDw1(Dw1), itsD1(D1), itsDw2(Dw2), itsD2(D2), itswLow(1-wLow)
         , itsData(itsDw1*itsD1,itsDw2*itsD2)
        {
            assert(itsDw1>=itswLow);
            assert(itsDw2>=itswLow);
            assert(itsD1>0);
            assert(itsD2>0);
            assert(wLow>=0);
        }
        Matrix4(int wLow=1)
         : itsDw1(0), itsD1(0), itsDw2(0), itsD2(0), itswLow(1-wLow)
         , itsData()
        {
        }

        T& operator()(int w1, int i1, int w2, int i2)
        {
            //cout << "w1,i1,index=" << w1 << " " << i1 << " " << index1(w1,i1) <<endl;
            //cout << "w2,i2,index=" << w2 << " " << i2 << " " << index2(w2,i2) <<endl;
            return itsData(index1(w1,i1),index2(w2,i2));
        }
        const T& operator()(int w1, int i1, int w2, int i2) const
        {
            return itsData(index1(w1,i1),index2(w2,i2));;
        }

              MatrixRT& Flatten()       {return itsData;}
        const MatrixRT& Flatten() const {return itsData;}
        void Fill(const T& fill) {::Fill(itsData,fill);}
        bool IsHermitian(double eps) const
        {
            //return itsData.IsHermitian(eps);
            return Max(fabs(itsData-conj(Transpose(itsData))))<eps;
        }

        friend std::ostream& operator<<(std::ostream& os,const Matrix4& m) {return os << m.itsData;}
        friend std::istream& operator>>(std::istream& is,      Matrix4& m) {return is >> m.itsData;}

    protected:
        int index1(int w1, int i1) const {return w1+itswLow+itsDw1*(i1-1+itswLow);}
        int index2(int w2, int i2) const {return w2+itswLow+itsDw2*(i2-1+itswLow);}

    private:
        int itsDw1,itsD1,itsDw2,itsD2, itswLow;
        MatrixRT itsData;
};

#endif // MATRIX4_H
