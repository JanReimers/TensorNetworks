#include "gtest/gtest.h"
#include "TensorNetworks/MatrixProductState.H"
#include "oml/stream.h"
#include "oml/numeric.h"
#include "oml/cnumeric.h"
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

template <class T,class T2>
DMatrix<T2> contract(const DMatrix<T2> U, const Vector<T>& s, const DMatrix<T2>& V)
{
    int ni=U.GetNumRows();
    int nk=U.GetNumCols();
    int nj=V.GetNumRows();
    DMatrix<T2> M(ni,nj);
    Fill(M,T2(0.0));
    for(int i=1;i<=ni;i++)
        for(int j=1;j<=nj;j++)
            for(int k=1;k<=nk;k++)
            {
                M(i,j)+=U(i,k)*s(k)*V(j,k);
            }
    return M;
}

template <class T,class T2>
DMatrix<T2> ConstractsV(const Vector<T>& s, const DMatrix<T2>& V)
{
    int ni=V.GetNumRows();
    int nk=s.GetHigh();
    int nj=V.GetNumRows();
    DMatrix<T2> Vs(ni,nj);
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



//The eigen header file 'lib' is another option for later.  It takes a long time to complile.
/*#include <Eigen/Dense>
#include "oml/random.h"

TEST_F(SVDTesting,OML_SVDRandomSquareComplexMatrix)
{
    using namespace Eigen;
    int N=5;
    MatrixXd m(N,N);
    for (int i=0;i<N;i++)
        for (int j=0;i<N;i++)
            m(i,j)=OMLRandPos<double>();
    BDCSVD< MatrixXd > svd(m);
    std::cout << m << std::endl;
}
*/


TEST_F(SVDTesting,SVDComplexSquare_N10)
{
    int N=10;
    typedef DMatrix<eType> Mtype;
    Mtype A(N,N),V(N,N),UnitMatrix(N,N);
    Vector<double>  s(N);
    FillRandom(A);
    Mtype Mcopy(A);
    Unit(UnitMatrix);
    CSVDecomp(A,s,V);
    EXPECT_NEAR(Max(abs(Transpose(conj(A))*A-UnitMatrix)),0.0,eps);
    EXPECT_NEAR(Max(abs(V*Transpose(conj(V))-UnitMatrix)),0.0,eps);
    Mtype Vstar=conj(V);
    EXPECT_NEAR(Max(abs(A*ConstractsV(s,Vstar)-Mcopy)),0.0,eps);

}

/*
TEST_F(SVDTesting,OML_SVDRandomSquareComplexMatrix)
{
    int N=4;
    typedef DMatrix<std::complex<double> > Mtype;
    Mtype M(N,N),VT(N,N),UnitMatrix(N,N);
    Vector<std::complex<double> >  s(N);
    FillRandom(M);
    Mtype Mcopy(M+Transpose(conj(M)));
    M=Mcopy;
    Unit(UnitMatrix);
//    cout << "UnitMatrix=" << UnitMatrix;
//    cout << "M=" << M;
    SVDecomp(M,s,VT);

    cout << "s=" << s;
    cout << "U=" << M;
     cout << "Ut*U=" << Mtype(Transpose(conj(M))*M);
     cout << "U*Ut=" << Mtype(M*Transpose(conj(M)));


    //EXPECT_NEAR(Max(abs(Transpose(conj(M))*M-UnitMatrix)),0.0,eps);
    //EXPECT_NEAR(Max(abs(M*Transpose(conj(M))-UnitMatrix)),0.0,eps);
    //EXPECT_NEAR(Max(abs(Transpose(conj(VT))*VT-UnitMatrix)),0.0,eps);
    //EXPECT_NEAR(Max(abs(VT*Transpose(conj(VT))-UnitMatrix)),0.0,eps);
//    EXPECT_NEAR(Max(abs(M*ConstractsV(s,VT)-Mcopy)),0.0,eps);
//    EXPECT_NEAR(Max(fabs(M*V-Mcopy)),0.0,eps);
}
*/



