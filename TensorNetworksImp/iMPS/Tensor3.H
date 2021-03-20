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


    Tensor3(int d, int D1, int D2, int mnlow=0);
    ~Tensor3() {}
    void InitializeWith(State,int sgn);
    void Unit(int n); //Make this(n,i,j)=delta_ij = unit matrix

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

    friend std::ostream& operator<<(std::ostream& os, const Tensor3& m)
    {
        for (auto& M:m.itsMs)
            os << M;
        return os;
    }

private:

    MatrixCT InitNorm (Direction) const;
    MatrixCT   Flatten(Direction) const;
    void     UnFlatten(Direction,const MatrixCT& F);

    int itsmnLow;
    std::vector<MatrixCT> itsMs; //
};

} //namespace

#endif // TENSOR3_H
