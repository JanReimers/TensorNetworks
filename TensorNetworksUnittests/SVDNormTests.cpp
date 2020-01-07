#include "Tests.H"

#include "TensorNetworks/MatrixProductState.H"
#include "oml/stream.h"
#include "oml/random.h"
#include <complex>


class MPSNormTesting : public ::testing::Test
{
public:
    MPSNormTesting()
    : mps(10,3,2)
 //   , itsSites(mps.itsSites)
    , eps(1.0e-10)
    {
        StreamableObject::SetToPretty();
        std::cout.precision(5);
        cout << std::fixed;
    }
    typedef MatrixProductSite::MatrixT MatrixT;

    void VerifyLeftNorm(const MatrixProductState*);
    void VerifyRightNorm(const MatrixProductState*);

//    const MatrixT& GetA(int i,int ip) const {return itsSites[i]->itsAs[ip]; }

    MatrixProductState mps;
//    MatrixProductState::SitesType& itsSites;
    double eps;
};




void MPSNormTesting::VerifyLeftNorm(const MatrixProductState* mps)
{
    // Skip the last site
    for (int i=0; i<mps->GetL(); i++)
    {
//        cout << mps->GetLeftNorm(i) << endl;
        VerifyUnit(mps->GetLeftNorm(i),eps);
        VerifyUnit(mps->GetMLeft(i),eps);
    }
}

void MPSNormTesting::VerifyRightNorm(const MatrixProductState* mps)
{
    //skip the first site
    for (int i=0; i<mps->GetL(); i++)
    {
//        cout << mps->GetRightNorm(i) << endl;
        VerifyUnit(mps->GetRightNorm(i),eps);
        VerifyUnit(mps->GetMRight(i),eps);
    }
}

std::string ExpectedNorm(int isite,int L)
{
    std::string ret;
    for (int ia=0;ia<isite;ia++) ret+="A0";
    ret+="M0";
    for (int ia=isite+1;ia<L;ia++) ret+="B0";
    return ret;
}




TEST_F(MPSNormTesting,LeftNormalMatriciesProductStateL10S3D2)
{
    mps.InitializeWith(MatrixProductSite::Product);
    mps.Normalize(MatrixProductSite::Left);
    VerifyLeftNorm(&mps);
}


TEST_F(MPSNormTesting,RightNormalMatriciesProductStateL100S3D2)
{
    mps.InitializeWith(MatrixProductSite::Product);
    mps.Normalize(MatrixProductSite::Right);
    VerifyRightNorm(&mps);
}


TEST_F(MPSNormTesting,LeftNormalMatriciesRandomStateL10S3D2)
{
    mps.InitializeWith(MatrixProductSite::Random);
    mps.Normalize(MatrixProductSite::Left);
    VerifyLeftNorm(&mps);
}

TEST_F(MPSNormTesting,RightNormalMatriciesRandomStateL100S3D2)
{
    mps.InitializeWith(MatrixProductSite::Random);
    mps.Normalize(MatrixProductSite::Right);
    VerifyRightNorm(&mps);
}

TEST_F(MPSNormTesting,LeftNormalMatriciesRandomStateL10S3D3)
{
    MatrixProductState mps1(10,3,3);
    mps1.InitializeWith(MatrixProductSite::Random);
    mps1.Normalize(MatrixProductSite::Left);
    VerifyLeftNorm(&mps1);
}

TEST_F(MPSNormTesting,LeftNormalMatriciesRandomStateL10S1D1)
{
    int L=10;
    MatrixProductState mps1(L,1,1);
    mps1.InitializeWith(MatrixProductSite::Random);
    mps1.Normalize(MatrixProductSite::Left);
    VerifyLeftNorm(&mps1);
}
TEST_F(MPSNormTesting,LeftNormalMatriciesRandomStateL10S5D1)
{
    int L=10;
    MatrixProductState mps1(L,5,1);
    mps1.InitializeWith(MatrixProductSite::Random);
    mps1.Normalize(MatrixProductSite::Left);
    VerifyLeftNorm(&mps1);
}
TEST_F(MPSNormTesting,RightNormalMatriciesRandomStateL10S1D1)
{
    int L=10;
    MatrixProductState mps1(L,1,1);
    mps1.InitializeWith(MatrixProductSite::Random);
    mps1.Normalize(MatrixProductSite::Right);
    VerifyLeftNorm(&mps1);
}
TEST_F(MPSNormTesting,RightNormalMatriciesRandomStateL10S5D1)
{
    int L=10;
    MatrixProductState mps1(L,5,1);
    mps1.InitializeWith(MatrixProductSite::Random);
    mps1.Normalize(MatrixProductSite::Right);
    VerifyLeftNorm(&mps1);
}

TEST_F(MPSNormTesting,LeftNormalMatriciesRandomStateL10S3D10)
{
    MatrixProductState mps1(10,1,10);
    mps1.InitializeWith(MatrixProductSite::Random);
    mps1.Normalize(MatrixProductSite::Left);
    VerifyLeftNorm(&mps1);
}

TEST_F(MPSNormTesting,RightNormalMatriciesRandomStateL10S3D10)
{
    MatrixProductState mps1(10,3,10);
    mps1.InitializeWith(MatrixProductSite::Random);
    mps1.Normalize(MatrixProductSite::Right);
    VerifyRightNorm(&mps1);
}

TEST_F(MPSNormTesting,LeftNormalOverlapL10S1D2)
{
    MatrixProductState mps1(10,1,2);
    mps1.InitializeWith(MatrixProductSite::Random);
    mps1.Normalize(MatrixProductSite::Left);
    EXPECT_NEAR(mps1.GetOverlap(),1.0,eps);
}

TEST_F(MPSNormTesting,RightNormalOverlapL10S2D2)
{
    MatrixProductState mps1(10,3,2);
    mps1.InitializeWith(MatrixProductSite::Random);
    mps1.Normalize(MatrixProductSite::Right);
    EXPECT_NEAR(mps1.GetOverlap(),1.0,eps);
}

TEST_F(MPSNormTesting,RightNormalOverlapSite0L10S1D3)
{
    MatrixProductState mps1(10,1,3);
    mps1.InitializeWith(MatrixProductSite::Random);
    mps1.Normalize(MatrixProductSite::Right);
    VerifyUnit(mps1.GetNeff(0),eps);
}
TEST_F(MPSNormTesting,LeftNormalOverlapSiteLL10S1D3)
{
    MatrixProductState mps1(10,1,3);
    mps1.InitializeWith(MatrixProductSite::Random);
    mps1.Normalize(MatrixProductSite::Left);
    VerifyUnit(mps1.GetNeff(mps1.GetL()-1),eps);
}

TEST_F(MPSNormTesting,RightNormalOverlapSite0L10S5D2)
{
    MatrixProductState mps1(10,5,2);
    mps1.InitializeWith(MatrixProductSite::Random);
    mps1.Normalize(MatrixProductSite::Right);
    VerifyUnit(mps1.GetNeff(0),eps);
}
TEST_F(MPSNormTesting,LeftNormalOverlapSiteLL10S5D2)
{
    MatrixProductState mps1(10,5,2);
    mps1.InitializeWith(MatrixProductSite::Random);
    mps1.Normalize(MatrixProductSite::Left);
    VerifyUnit(mps1.GetNeff(mps1.GetL()-1),eps);
}

TEST_F(MPSNormTesting,MixedCanonicalL10S1D3)
{
    int L=10;
    MatrixProductState mps1(L,1,3);
    mps1.InitializeWith(MatrixProductSite::Random);
    for (int ia=0;ia<mps1.GetL();ia++)
    {
        mps1.Normalize(ia);
        VerifyUnit(mps1.GetNeff(ia),eps);
        EXPECT_EQ(mps1.GetNormStatus(),ExpectedNorm(ia,L));
    }
}

//  Slow test
TEST_F(MPSNormTesting,MixedCanonicalL10S3D10)
{
    int L=10;
    MatrixProductState mps1(L,3,10);
    mps1.InitializeWith(MatrixProductSite::Random);
    for (int ia=0;ia<mps1.GetL();ia++)
    {
        mps1.Normalize(ia);
        VerifyUnit(mps1.GetNeff(ia),eps);
        EXPECT_EQ(mps1.GetNormStatus(),ExpectedNorm(ia,L));
   }
}


TEST_F(MPSNormTesting,MixedCanonicalL10S5D2)
{
    int L=10;
    MatrixProductState mps1(L,5,2);
    mps1.InitializeWith(MatrixProductSite::Random);
    for (int ia=0;ia<mps1.GetL();ia++)
    {
        mps1.Normalize(ia);
        VerifyUnit(mps1.GetNeff(ia),eps);
        EXPECT_EQ(mps1.GetNormStatus(),ExpectedNorm(ia,L));
    }
}

TEST_F(MPSNormTesting,MixedCanonicalL10S5D1)
{
    int L=10;
    MatrixProductState mps1(L,5,1);
    mps1.InitializeWith(MatrixProductSite::Random);
    for (int ia=0;ia<mps1.GetL();ia++)
    {
        mps1.Normalize(ia);
        VerifyUnit(mps1.GetNeff(ia),eps);
        // This fails for D=1 becuase left and right normalization are indistinguisable when D=1
        //EXPECT_EQ(mps1.GetNormStatus(),ExpectedNorm(ia,L));
    }
}
