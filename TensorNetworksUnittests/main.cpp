#include "gtest/gtest.h"

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#include "TensorNetworks/MatrixProductState.H"
#include "oml/stream.h"

class MatrixProductTesting : public ::testing::Test
{
public:
    MatrixProductTesting()
    : mps(10,1,2)
    , itsSites(mps.itsSites)
    {
        StreamableObject::SetToPretty();
        mps.itsSites.size();
    }
    typedef MatrixProductSite::MatrixT MatrixT;

    const MatrixT& GetA(int i,int ip) const {return itsSites[i]->itsAs[ip]; }

    MatrixProductState mps;
    MatrixProductState::SitesType& itsSites;
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

TEST_F(MatrixProductTesting,InitializeWithProductState)
{
    mps.InitializeWithProductState();
    EXPECT_EQ(ToString(GetA(0,0)),"(1:1),(1:2) \n[ (1,0) (0,0) ]\n");
    EXPECT_EQ(ToString(GetA(1,0)),"(1:2),(1:2) \n[ (1,0) (0,0) ]\n[ (0,0) (0,0) ]\n");
    EXPECT_EQ(ToString(GetA(8,0)),"(1:2),(1:2) \n[ (1,0) (0,0) ]\n[ (0,0) (0,0) ]\n");
    EXPECT_EQ(ToString(GetA(9,0)),"(1:2),(1:1) \n[ (1,0) ]\n[ (0,0) ]\n");
    EXPECT_EQ(ToString(GetA(0,1)),"(1:1),(1:2) \n[ (1,0) (0,0) ]\n");
    EXPECT_EQ(ToString(GetA(1,1)),"(1:2),(1:2) \n[ (1,0) (0,0) ]\n[ (0,0) (0,0) ]\n");
    EXPECT_EQ(ToString(GetA(8,1)),"(1:2),(1:2) \n[ (1,0) (0,0) ]\n[ (0,0) (0,0) ]\n");
    EXPECT_EQ(ToString(GetA(9,1)),"(1:2),(1:1) \n[ (1,0) ]\n[ (0,0) ]\n");
}

//
//  Evaluate <Psi|Psi> for L=10,S=1/2,D=2
//
TEST_F(MatrixProductTesting,GetOverlap)
{
    mps.InitializeWithProductState();
    EXPECT_EQ(mps.GetOverlap(),1024.0);
}

//
//  Evaluate <Psi|Psi> for L=10,S=1,D=3
//
TEST_F(MatrixProductTesting,GetOverlapS2D3)
{
    MatrixProductState mps_local(10,2,3);
    mps_local.InitializeWithProductState();
    EXPECT_EQ(mps_local.GetOverlap(),59049.0);
}

//
//  Evaluate <Psi|Psi> for L=10,S=3/2,D=4
//
TEST_F(MatrixProductTesting,GetOverlapS3D4)
{
    MatrixProductState mps_local(10,3,4);
    mps_local.InitializeWithProductState();
    EXPECT_EQ(mps_local.GetOverlap(),1048576.0);
}
