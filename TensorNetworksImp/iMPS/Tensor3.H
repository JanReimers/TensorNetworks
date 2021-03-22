#ifndef TENSOR3_H
#define TENSOR3_H

#include "TensorNetworks/Enums.H"
#include "TensorNetworksImp/Typedefs.H"
#include "oml/matrix.h"
#include <vector>
#include <tuple>

namespace TensorNetworks
{

class Tensor3
{
public:
    typedef std::tuple<int,int,int> DimT;
    typedef std::tuple<MatrixCT,MatrixCT> MMType;
    typedef std::tuple<double,Tensor3> LRWType;


    Tensor3(int d, int D1, int D2, int mnlow=0);
    ~Tensor3() {}
    void InitializeWith(State,int sgn);
    void Unit(int n); //Make this(n,i,j)=delta_ij = unit matrix
    MatrixCT   Flatten(Direction) const;
    void     UnFlatten(Direction,const MatrixCT& F);
    void     UnFlatten(const VectorCT&);

    int       Getd     () const {return itsMs.size();}
    int       GetD1    () const {return itsMs[0].GetNumRows();}
    int       GetD2    () const {return itsMs[0].GetNumCols();}
    DimT GetDimensions() const {return std::make_tuple(Getd(),GetD1(),GetD2());}
    MatLimits GetLimits() const;
    bool      IsSquare () const;

    const MatrixCT& operator()(int n) const;
          MatrixCT& operator()(int n)      ;

    MatrixCT  QLRR             (Direction,double eps); // Rank Revealing QL/LQ
    MatrixCT  GetNorm          (Direction) const;
    MatrixCT  GetNorm          (Direction, const DiagonalMatrixRT& lambda) const;
    MatrixCT  GetTransferMatrix(Direction) const;
    Matrix6CT GetTransferMatrix(Direction,const MatrixOR& W) const;
    VectorCT  GetTMEigenVector (Direction) const;
    LRWType   GetLW            (const MatrixOR& W) const;
    LRWType   GetRW            (const MatrixOR& W) const;
    double    GetExpectation   (const MatrixOR& W) const;

    friend std::ostream& operator<<(std::ostream& os, const Tensor3& m)
    {
        for (auto& M:m.itsMs)
            os << M;
        return os;
    }

private:

    MatrixCT InitNorm (Direction) const;

    int itsmnLow;
    std::vector<MatrixCT> itsMs; //
};

template <class T> Vector<T> Flatten(const Matrix<T>& m)
{
    Vector<T> v(m.GetNumRows()*m.GetNumCols());
    int ij=1;
    for (index_t j:m.cols())
        for (index_t i:m.rows())
            v(ij++)=m(i,j);
    return v;
}
template <class T> Matrix<T> UnFlatten(const Vector<T>& v)
{
    int N=sqrt(v.size());
    Matrix<T> m(N,N);
    int ij=1;
    for (index_t j:m.cols())
        for (index_t i:m.rows())
            m(i,j)=v(ij++);
    return m;
}

} //namespace

#endif // TENSOR3_H
