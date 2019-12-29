#include "gtest/gtest.h"
#include "TensorNetworks/MatrixProductState.H"
#include "oml/stream.h"
#include "oml/numeric.h"
#include "oml/random.h"
#include "oml/vector_io.h"
#include "oml/array_io.h"
#include <complex>
#include <iostream>

using std::cout;
using std::endl;

class SVDTesting : public ::testing::Test
{
public:
    typedef MatrixProductSite::MatrixT MatrixT;
    SVDTesting()
    : mps(10,1,2)
  //  , itsSites(mps.itsSites)
    , eps(1.0e-13)
    {
        StreamableObject::SetToPretty();
        mps.InitializeWithProductState();
    }
 //   const MatrixT& GetA(int i,int ip) const {return itsSites[i]->itsAs[ip]; }


    MatrixProductState mps;
//    MatrixProductState::SitesType& itsSites;
    double eps;
};

template <class Ob> std::string ToString(const Ob& result)
{
    std::stringstream res_stream;
    res_stream << result;
    return res_stream.str();
}

DMatrix<double> contract(const DMatrix<double>& U, const Vector<double>& s, const DMatrix<double>& V)
{
    int ni=U.GetNumRows();
    int nk=U.GetNumCols();
    int nj=V.GetNumRows();
    DMatrix<double> M(ni,nj);
    Fill(M,0.0);
    for(int i=1;i<=ni;i++)
        for(int j=1;j<=nj;j++)
            for(int k=1;k<=nk;k++)
            {
                M(i,j)+=U(i,k)*s(k)*V(j,k);
            }
    return M;
}
DMatrix<double> ConstractsV(const Vector<double>& s, const DMatrix<double>& V)
{
    int ni=V.GetNumRows();
    int nk=s.GetHigh();
    int nj=V.GetNumRows();
    DMatrix<double> Vs(ni,nj);
    for(int i=1;i<=ni;i++)
        for(int j=1;j<=nj;j++)
            for(int k=1;k<=nk;k++)
                Vs(i,j)=s(i)*V(j,i);
    return Vs;
}

TEST_F(SVDTesting,OML_SVDRandomSquareRealMatrix)
{
    int N=10;
    typedef DMatrix<double> Mtype;
    Mtype M(N,N),VT(N,N),UnitMatrix(N,N);
    Vector<double>  s(N);
    FillRandom(M);
    Mtype Mcopy(M);
    Unit(UnitMatrix);
    SVDecomp(M,s,VT);
//    cout << "V=" << V;
    EXPECT_NEAR(Max(fabs(Transpose(M)*M-UnitMatrix)),0.0,eps);
    EXPECT_NEAR(Max(fabs(M*Transpose(M)-UnitMatrix)),0.0,eps);
    EXPECT_NEAR(Max(fabs(Transpose(VT)*VT-UnitMatrix)),0.0,eps);
    EXPECT_NEAR(Max(fabs(VT*Transpose(VT)-UnitMatrix)),0.0,eps);
    EXPECT_NEAR(Max(fabs(M*ConstractsV(s,VT)-Mcopy)),0.0,eps);
//    EXPECT_NEAR(Max(fabs(M*V-Mcopy)),0.0,eps);
}

TEST_F(SVDTesting,OML_SVDRandomRectRealMatrix_10x5)
{
    int N1=10,N2=5;
    typedef DMatrix<double> Mtype;
    Mtype M(N1,N2),VT(N2,N2),UnitMatrix(N1,N1);
    Vector<double>  s(N2);
    FillRandom(M);
    Mtype Mcopy(M);
    Unit(UnitMatrix);
    SVDecomp(M,s,VT);
    EXPECT_NEAR(Max(fabs(Transpose(M)*M-UnitMatrix)),0.0,eps);
    EXPECT_NEAR(Max(fabs(VT*Transpose(VT)-UnitMatrix)),0.0,eps);
    EXPECT_NEAR(Max(fabs(Transpose(VT)*VT-UnitMatrix)),0.0,eps);
    EXPECT_NEAR(Max(fabs(M*ConstractsV(s,VT)-Mcopy)),0.0,eps);
}


TEST_F(SVDTesting,OML_SVDRandomRectRealMatrix_5x10)
{
    int N1=3,N2=5;
    typedef DMatrix<double> Mtype;
    Mtype M(N1,N2),VT(N2,N2),UnitMatrix(N1,N1);
    Vector<double>  s(N2);
    FillRandom(M);
    Mtype Mcopy(M);
    Unit(UnitMatrix);
    SVDecomp(M,s,VT);
    EXPECT_NEAR(Max(fabs(Transpose(M)*M-UnitMatrix)),0.0,eps);
    EXPECT_NEAR(Max(fabs(VT*Transpose(VT)-UnitMatrix)),0.0,eps);
    EXPECT_NEAR(Max(fabs(M*ConstractsV(s,VT)-Mcopy)),0.0,eps);
}


