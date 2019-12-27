#include "gtest/gtest.h"

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#include "TensorNetworks/MatrixProductState.H"
#include "oml/stream.h"
#include "oml/random.h"
#include <complex>

class MatrixProductTesting : public ::testing::Test
{
public:
    MatrixProductTesting()
    : mps(10,1,2)
    , mps3(10,2,3)
    , itsSites(mps.itsSites)
    , itsSites3(mps3.itsSites)
    , eps(1.0e-10)
    {
        StreamableObject::SetToPretty();
        mps.itsSites.size();
    }
    typedef MatrixProductSite::MatrixT MatrixT;

    const MatrixT& GetA(int i,int ip) const {return itsSites[i]->itsAs[ip]; }
    const MatrixT& GetA3(int i,int ip) const {return mps3.itsSites[i]->itsAs[ip]; }

    MatrixProductState mps;
    MatrixProductState mps3;
    MatrixProductState::SitesType& itsSites;
    MatrixProductState::SitesType& itsSites3;
    double eps;
};

template <class Ob> std::string ToString(const Ob& result)
{
    std::stringstream res_stream;
    res_stream << result;
    return res_stream.str();
}



TEST_F(MatrixProductTesting,Constructor)
{
    EXPECT_EQ(itsSites.size(),10);
    EXPECT_EQ(mps.GetL(),10);
    EXPECT_EQ(mps.GetD(),2);
    EXPECT_EQ(mps.Getp(),2);
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
    mps.InitializeWithProductState();
    EXPECT_EQ(ToString(itsSites[0]->GetLeftNorm()),"(1:2),(1:2) \n[ (1,0) (0,0) ]\n[ (0,0) (1,0) ]\n");
    EXPECT_EQ(ToString(itsSites[1]->GetLeftNorm()),"(1:2),(1:2) \n[ (1,0) (0,0) ]\n[ (0,0) (1,0) ]\n");
    EXPECT_EQ(ToString(itsSites[8]->GetLeftNorm()),"(1:2),(1:2) \n[ (1,0) (0,0) ]\n[ (0,0) (1,0) ]\n");
//    EXPECT_EQ(ToString(itsSites[9]->GetLeftNorm()),"(1:1),(1:1) \n[ (1,0) ]\n");
}


TEST_F(MatrixProductTesting,RightNormalMatricies)
{
    mps.InitializeWithProductState();
    // The first site will not be right normalized
    EXPECT_EQ(ToString(itsSites[0]->GetRightNorm()),"(1:1),(1:1) \n[ (2,0) ]\n");
    EXPECT_EQ(ToString(itsSites[1]->GetRightNorm()),"(1:2),(1:2) \n[ (1,0) (0,0) ]\n[ (0,0) (1,0) ]\n");
    EXPECT_EQ(ToString(itsSites[8]->GetRightNorm()),"(1:2),(1:2) \n[ (1,0) (0,0) ]\n[ (0,0) (1,0) ]\n");
    EXPECT_EQ(ToString(itsSites[9]->GetRightNorm()),"(1:2),(1:2) \n[ (1,0) (0,0) ]\n[ (0,0) (1,0) ]\n");
}


TEST_F(MatrixProductTesting,LeftNormalMatricies_S2)
{
    mps3.InitializeWithProductState();
    EXPECT_EQ(ToString(itsSites3[0]->GetLeftNorm()),"(1:3),(1:3) \n[ (1,0) (0,0) (0,0) ]\n[ (0,0) (1,0) (0,0) ]\n[ (0,0) (0,0) (1,0) ]\n");
    EXPECT_EQ(ToString(itsSites3[1]->GetLeftNorm()),"(1:3),(1:3) \n[ (1,0) (0,0) (0,0) ]\n[ (0,0) (1,0) (0,0) ]\n[ (0,0) (0,0) (1,0) ]\n");
    EXPECT_EQ(ToString(itsSites3[8]->GetLeftNorm()),"(1:3),(1:3) \n[ (1,0) (0,0) (0,0) ]\n[ (0,0) (1,0) (0,0) ]\n[ (0,0) (0,0) (1,0) ]\n");
//    EXPECT_EQ(ToString(itsSites3[9]->GetLeftNorm()),"(1:1),(1:1) \n[ (1,0) ]\n");
}

TEST_F(MatrixProductTesting,RightNormalMatricies_S2)
{
    mps3.InitializeWithProductState();
    EXPECT_EQ(ToString(itsSites3[0]->GetRightNorm()),"(1:1),(1:1) \n[ (3,0) ]\n");
    EXPECT_EQ(ToString(itsSites3[1]->GetRightNorm()),"(1:3),(1:3) \n[ (1,0) (0,0) (0,0) ]\n[ (0,0) (1,0) (0,0) ]\n[ (0,0) (0,0) (1,0) ]\n");
    EXPECT_EQ(ToString(itsSites3[8]->GetRightNorm()),"(1:3),(1:3) \n[ (1,0) (0,0) (0,0) ]\n[ (0,0) (1,0) (0,0) ]\n[ (0,0) (0,0) (1,0) ]\n");
    EXPECT_EQ(ToString(itsSites3[9]->GetRightNorm()),"(1:3),(1:3) \n[ (1,0) (0,0) (0,0) ]\n[ (0,0) (1,0) (0,0) ]\n[ (0,0) (0,0) (1,0) ]\n");
}


//
//  Evaluate <Psi|Psi> for L=3,S=1/2,D=2
//
TEST_F(MatrixProductTesting,GetOverlap)
{
    mps.InitializeWithProductState();
    ASSERT_NEAR(mps.GetOverlap(),2.0,eps);
}

//
//  Evaluate <Psi|Psi> for L=10,S=1,D=3
//
TEST_F(MatrixProductTesting,GetOverlapS2D3)
{
    MatrixProductState mps_local(10,2,3);
    mps_local.InitializeWithProductState();
    ASSERT_NEAR(mps_local.GetOverlap(),3.0,eps);
}

//
//  Evaluate <Psi|Psi> for L=10,S=3/2,D=4
//
TEST_F(MatrixProductTesting,GetOverlapS3D4)
{
    MatrixProductState mps_local(10,3,4);
    mps_local.InitializeWithProductState();
    ASSERT_NEAR(mps_local.GetOverlap(),4.0,eps);
}



TEST_F(MatrixProductTesting,GetMLeft_Site_1)
{
    mps.InitializeWithProductState();
    EXPECT_EQ(ToString(mps.GetMLeft(1)),"(1:2),(1:2) \n[ (1,0) (0,0) ]\n[ (0,0) (1,0) ]\n");
}
TEST_F(MatrixProductTesting,GetMLeft_Site_2)
{
    mps.InitializeWithProductState();
    EXPECT_EQ(ToString(mps.GetMLeft(2)),"(1:2),(1:2) \n[ (1,0) (0,0) ]\n[ (0,0) (1,0) ]\n");
}
TEST_F(MatrixProductTesting,GetMLeft_Site_8)
{
    mps.InitializeWithProductState();
    EXPECT_EQ(ToString(mps.GetMLeft(8)),"(1:2),(1:2) \n[ (1,0) (0,0) ]\n[ (0,0) (1,0) ]\n");
}

TEST_F(MatrixProductTesting,GetMLeft_Site_9)
{
    mps.InitializeWithProductState();
    EXPECT_EQ(ToString(mps.GetMLeft(9)),"(1:2),(1:2) \n[ (1,0) (0,0) ]\n[ (0,0) (1,0) ]\n");
}

TEST_F(MatrixProductTesting,GetMLeft_Site_0)
{
    mps.InitializeWithProductState();
    EXPECT_EQ(ToString(mps.GetMLeft(0)),"(1:1),(1:1) \n[ (1,0) ]\n");
}

TEST_F(MatrixProductTesting,GetMRight_Site_8)
{
    mps.InitializeWithProductState();
    EXPECT_EQ(ToString(mps.GetMRight(8)),"(1:2),(1:2) \n[ (1,0) (0,0) ]\n[ (0,0) (1,0) ]\n");
}

TEST_F(MatrixProductTesting,GetMRight_Site_9)
{
    mps.InitializeWithProductState();
    EXPECT_EQ(ToString(mps.GetMRight(9)),"(1:1),(1:1) \n[ (1,0) ]\n");
}

TEST_F(MatrixProductTesting,GetMRight_Site_0)
{
    mps.InitializeWithProductState();
    EXPECT_EQ(ToString(mps.GetMRight(0)),"(1:2),(1:2) \n[ (1,0) (0,0) ]\n[ (0,0) (1,0) ]\n");
}
TEST_F(MatrixProductTesting,GetMRight_Site_1)
{
    mps.InitializeWithProductState();
    EXPECT_EQ(ToString(mps.GetMRight(1)),"(1:2),(1:2) \n[ (1,0) (0,0) ]\n[ (0,0) (1,0) ]\n");
}


TEST_F(MatrixProductTesting,GetMLeft_Site_S2)
{
    mps3.InitializeWithProductState();
    EXPECT_EQ(ToString(mps3.GetMLeft(1)),"(1:3),(1:3) \n[ (1,0) (0,0) (0,0) ]\n[ (0,0) (1,0) (0,0) ]\n[ (0,0) (0,0) (1,0) ]\n");
}


//
//  Evaluate overlap for site 0 for L=10,S=1/2,D=2
//

TEST_F(MatrixProductTesting,GetOverlapForSite0)
{
    mps.InitializeWithProductState();
    EXPECT_EQ(ToString(mps.GetOverlap(0)),"(1:1),(1:2) \n[ (1,0) (1,0) ]\n");
}

//
//  Evaluate overlap for site 1 for L=10,S=1/2,D=2
//
TEST_F(MatrixProductTesting,GetOverlapForSite1)
{
    mps.InitializeWithProductState();
    EXPECT_EQ(ToString(mps.GetOverlap(1)),"(1:2),(1:2) \n[ (-1.41421,0) (0,0) ]\n[ (0,0) (-1.41421,0) ]\n");
}

//
//  Evaluate overlap for site 8 for L=10,S=1/2,D=2
//
TEST_F(MatrixProductTesting,GetOverlapForSite8)
{
    mps.InitializeWithProductState();
    EXPECT_EQ(ToString(mps.GetOverlap(8)),"(1:2),(1:2) \n[ (1.41421,0) (0,0) ]\n[ (0,0) (1.41421,0) ]\n");
}

//
//  Evaluate overlap for site 9 for L=10,S=1/2,D=2
//
TEST_F(MatrixProductTesting,GetOverlapForSite9)
{
    mps.InitializeWithProductState();
    EXPECT_EQ(ToString(mps.GetOverlap(9)),"(1:2),(1:1) \n[ (-1,0) ]\n[ (-1,0) ]\n");
}

