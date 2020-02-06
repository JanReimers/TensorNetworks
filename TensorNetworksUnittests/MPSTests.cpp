#include "gtest/gtest.h"
#include "Tests.H"
#include "TensorNetworks/Epsilons.H"
#include "oml/stream.h"
#include "oml/random.h"
#include <complex>

class MatrixProductTesting : public ::testing::Test
{
public:
    MatrixProductTesting()
    : itsMPS(0)
    , itsEps()
    , eps(1.0e-10)
    {
        Setup(10,1,2);
        StreamableObject::SetToPretty();
    }
    ~MatrixProductTesting() {delete itsMPS;}



    void Setup(int L, int S2, int D)
    {
        delete itsMPS;
        itsMPS=new MatrixProductStateImp(L,S2,D,itsEps);
    }


    typedef MatrixProductSite::MatrixCT MatrixCT;

    MatrixCT GetA(int i,int ip) const {return GetSite(i)->itsAs[ip]; }
    const MatrixProductSite* GetSite(int isite) const {return itsMPS->itsSites[isite];}
    MatrixCT GetLeftNorm(int isite) const {return GetSite(isite)->GetLeftNorm();}
    MatrixCT GetRightNorm(int isite) const {return GetSite(isite)->GetRightNorm();}

    MatrixProductStateImp* itsMPS;
    Epsilons               itsEps;


    double eps;
};





TEST_F(MatrixProductTesting,Constructor)
{
    EXPECT_EQ(itsMPS->GetL(),10);
    EXPECT_EQ(itsMPS->GetD(),2);
    EXPECT_EQ(itsMPS->Getp(),2);
}

TEST_F(MatrixProductTesting,MatrixOpMul)
{
    DMatrix<double> A(10,10),B(10,9);
    FillRandom(A);
    FillRandom(B);
    A*=B;
}


TEST_F(MatrixProductTesting,LeftNormalMatricies)
{
    itsMPS->InitializeWith(TensorNetworks::Product);
    VerifyUnit(GetLeftNorm(0),eps);
    VerifyUnit(GetLeftNorm(1),eps);
    VerifyUnit(GetLeftNorm(8),eps);
}


TEST_F(MatrixProductTesting,RightNormalMatricies)
{
    itsMPS->InitializeWith(TensorNetworks::Product);
    // The first site will not be right normalized
    EXPECT_EQ(ToString(GetRightNorm(0)),"(1:1),(1:1) \n[ (2,0) ]\n");
    VerifyUnit(GetRightNorm(1),eps);
    VerifyUnit(GetRightNorm(8),eps);
    VerifyUnit(GetRightNorm(9),eps);
}


TEST_F(MatrixProductTesting,LeftNormalMatricies_S2)
{
    Setup(10,3,2);
    itsMPS->InitializeWith(TensorNetworks::Product);
    VerifyUnit(GetLeftNorm(0),eps);
    VerifyUnit(GetLeftNorm(1),eps);
    VerifyUnit(GetLeftNorm(8),eps);
}

TEST_F(MatrixProductTesting,RightNormalMatricies_S2)
{
    Setup(10,3,3);
    itsMPS->InitializeWith(TensorNetworks::Product);
    VerifyUnit(GetRightNorm(1),eps);
    VerifyUnit(GetRightNorm(8),eps);
    VerifyUnit(GetRightNorm(9),eps);
    EXPECT_EQ(ToString(GetRightNorm(0)),"(1:1),(1:1) \n[ (3,0) ]\n");
}




