#include "gtest/gtest.h"
#include "Tests.H"
#include "oml/stream.h"
#include "oml/random.h"
#include <complex>

class MatrixProductTesting : public ::testing::Test
{
public:
    MatrixProductTesting()
    : itsMPS(0)
    , eps(1.0e-10)
    {
        Setup(10,1,2);
        StreamableObject::SetToPretty();
    }
    ~MatrixProductTesting() {delete itsMPS;}

    void Setup(int L, int S2, int D)
    {
        delete itsMPS;
        itsMPS=new MatrixProductStateImp(L,S2,D);
    }


    typedef MatrixProductSite::MatrixCT MatrixCT;

    MatrixCT GetA(int i,int ip) const {return GetSite(i)->itsAs[ip]; }
    MatrixCT GetMLeft(int isite) const {return itsMPS->GetMLeft(isite);}
    MatrixCT GetMRight(int isite) const {return itsMPS->GetMRight(isite);}
    MatrixCT GetNeff   (int isite) const {return itsMPS->GetNeff(isite);}
    const MatrixProductSite* GetSite(int isite) const {return itsMPS->itsSites[isite];}

    MatrixProductStateImp* itsMPS;
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
    VerifyUnit(GetSite(0)->GetLeftNorm(),eps);
    VerifyUnit(GetSite(1)->GetLeftNorm(),eps);
    VerifyUnit(GetSite(8)->GetLeftNorm(),eps);
}


TEST_F(MatrixProductTesting,RightNormalMatricies)
{
    itsMPS->InitializeWith(TensorNetworks::Product);
    // The first site will not be right normalized
    EXPECT_EQ(ToString(GetSite(0)->GetRightNorm()),"(1:1),(1:1) \n[ (2,0) ]\n");
    VerifyUnit(GetSite(1)->GetRightNorm(),eps);
    VerifyUnit(GetSite(8)->GetRightNorm(),eps);
    VerifyUnit(GetSite(9)->GetRightNorm(),eps);
}


TEST_F(MatrixProductTesting,LeftNormalMatricies_S2)
{
    Setup(10,3,2);
    itsMPS->InitializeWith(TensorNetworks::Product);
    VerifyUnit(GetSite(0)->GetLeftNorm(),eps);
    VerifyUnit(GetSite(1)->GetLeftNorm(),eps);
    VerifyUnit(GetSite(8)->GetLeftNorm(),eps);
}

TEST_F(MatrixProductTesting,RightNormalMatricies_S2)
{
    Setup(10,3,3);
    itsMPS->InitializeWith(TensorNetworks::Product);
    VerifyUnit(GetSite(1)->GetRightNorm(),eps);
    VerifyUnit(GetSite(8)->GetRightNorm(),eps);
    VerifyUnit(GetSite(9)->GetRightNorm(),eps);
    EXPECT_EQ(ToString(GetSite(0)->GetRightNorm()),"(1:1),(1:1) \n[ (3,0) ]\n");
}


//
//  Evaluate <Psi|Psi> for L=3,S=1/2,D=2
//
TEST_F(MatrixProductTesting,GetOverlap)
{
    itsMPS->InitializeWith(TensorNetworks::Product);
    ASSERT_NEAR(itsMPS->GetOverlap(),2.0,eps);
}

//
//  Evaluate <Psi|Psi> for L=10,S=1,D=3
//
TEST_F(MatrixProductTesting,GetOverlapS2D3)
{
    MatrixProductStateImp mps_local(10,2,3);
    mps_local.InitializeWith(TensorNetworks::Product);
    ASSERT_NEAR(mps_local.GetOverlap(),3.0,eps);
}

//
//  Evaluate <Psi|Psi> for L=10,S=3/2,D=4
//
TEST_F(MatrixProductTesting,GetOverlapS3D4)
{
    MatrixProductStateImp mps_local(10,3,4);
    mps_local.InitializeWith(TensorNetworks::Product);
    ASSERT_NEAR(mps_local.GetOverlap(),4.0,eps);
}

TEST_F(MatrixProductTesting,GetOverlapS1D1)
{
    MatrixProductStateImp mps_local(10,1,1);
    mps_local.InitializeWith(TensorNetworks::Product);
    ASSERT_NEAR(mps_local.GetOverlap(),1.0,eps);
}



TEST_F(MatrixProductTesting,GetMLeft_Site_1)
{
    itsMPS->InitializeWith(TensorNetworks::Product);
    VerifyUnit(GetMLeft(1),eps);
    EXPECT_EQ(ToString(GetMLeft(1)),"(1:2),(1:2) \n[ (1,0) (0,0) ]\n[ (0,0) (1,0) ]\n");
}
TEST_F(MatrixProductTesting,GetMLeft_Site_2)
{
    itsMPS->InitializeWith(TensorNetworks::Product);
    VerifyUnit(GetMLeft(2),eps);
    EXPECT_EQ(ToString(GetMLeft(2)),"(1:2),(1:2) \n[ (1,0) (0,0) ]\n[ (0,0) (1,0) ]\n");
}
TEST_F(MatrixProductTesting,GetMLeft_Site_8)
{
    itsMPS->InitializeWith(TensorNetworks::Product);
    VerifyUnit(GetMLeft(8),eps);
    EXPECT_EQ(ToString(GetMLeft(8)),"(1:2),(1:2) \n[ (1,0) (0,0) ]\n[ (0,0) (1,0) ]\n");
}

TEST_F(MatrixProductTesting,GetMLeft_Site_9)
{
    itsMPS->InitializeWith(TensorNetworks::Product);
    VerifyUnit(GetMLeft(9),eps);
    EXPECT_EQ(ToString(GetMLeft(9)),"(1:2),(1:2) \n[ (1,0) (0,0) ]\n[ (0,0) (1,0) ]\n");
}

TEST_F(MatrixProductTesting,GetMLeft_Site_0)
{
    itsMPS->InitializeWith(TensorNetworks::Product);
    EXPECT_EQ(ToString(GetMLeft(0)),"(1:1),(1:1) \n[ (1,0) ]\n");
}

TEST_F(MatrixProductTesting,GetMRight_Site_8)
{
    itsMPS->InitializeWith(TensorNetworks::Product);
    VerifyUnit(GetMRight(8),eps);
    EXPECT_EQ(ToString(GetMRight(8)),"(1:2),(1:2) \n[ (1,0) (0,0) ]\n[ (0,0) (1,0) ]\n");
}

TEST_F(MatrixProductTesting,GetMRight_Site_9)
{
    itsMPS->InitializeWith(TensorNetworks::Product);
    EXPECT_EQ(ToString(GetMRight(9)),"(1:1),(1:1) \n[ (1,0) ]\n");
}

TEST_F(MatrixProductTesting,GetMRight_Site_0)
{
    itsMPS->InitializeWith(TensorNetworks::Product);
    VerifyUnit(GetMRight(0),eps);
    EXPECT_EQ(ToString(GetMRight(0)),"(1:2),(1:2) \n[ (1,0) (0,0) ]\n[ (0,0) (1,0) ]\n");
}
TEST_F(MatrixProductTesting,GetMRight_Site_1)
{
    itsMPS->InitializeWith(TensorNetworks::Product);
    VerifyUnit(GetMRight(1),eps);
    EXPECT_EQ(ToString(GetMRight(1)),"(1:2),(1:2) \n[ (1,0) (0,0) ]\n[ (0,0) (1,0) ]\n");
}


TEST_F(MatrixProductTesting,GetMLeft_Site_S2)
{
    Setup(10,2,3);
    itsMPS->InitializeWith(TensorNetworks::Product);
    VerifyUnit(GetMLeft(1),eps);
    EXPECT_EQ(ToString(GetMLeft(1)),"(1:3),(1:3) \n[ (1,0) (0,0) (0,0) ]\n[ (0,0) (1,0) (0,0) ]\n[ (0,0) (0,0) (1,0) ]\n");
}


//
//  Evaluate overlap for site 0 for L=10,S=1/2,D=2
//

TEST_F(MatrixProductTesting,GetOverlapForSite0)
{
    itsMPS->InitializeWith(TensorNetworks::Product);
    VerifyUnit(GetNeff(0),eps);
}

//
//  Evaluate overlap for site 1 for L=10,S=1/2,D=2
//
TEST_F(MatrixProductTesting,GetOverlapForSite1)
{
    itsMPS->InitializeWith(TensorNetworks::Product);
    VerifyUnit(GetNeff(1),eps);
}

//
//  Evaluate overlap for site 8 for L=10,S=1/2,D=2
//
TEST_F(MatrixProductTesting,GetOverlapForSite8)
{
    itsMPS->InitializeWith(TensorNetworks::Product);
    VerifyUnit(GetNeff(8),eps);
}

//
//  Evaluate overlap for site 9 for L=10,S=1/2,D=2
//
TEST_F(MatrixProductTesting,GetOverlapForSite9)
{
    itsMPS->InitializeWith(TensorNetworks::Product);
    VerifyUnit(GetNeff(9),eps);
}

