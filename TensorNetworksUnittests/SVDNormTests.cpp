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
    for (int i=0; i<mps->GetL()-1; i++)
    {
 //       cout << mps->GetLeftNorm(i) << endl;
        VerifyUnit(mps->GetLeftNorm(i),eps);
        VerifyUnit(mps->GetMLeft(i),eps);
    }
}

void MPSNormTesting::VerifyRightNorm(const MatrixProductState* mps)
{
    //skip the first site
    for (int i=1; i<mps->GetL(); i++)
    {
//        cout << mps->GetRightNorm(i) << endl;
        VerifyUnit(mps->GetRightNorm(i),eps);
        VerifyUnit(mps->GetMRight(i),eps);
    }
}





TEST_F(MPSNormTesting,LeftNormalMatriciesProductStateL10S3D2)
{
    mps.InitializeWithProductState();
    mps.Normalize(MatrixProductSite::Left);
    VerifyLeftNorm(&mps);
}

TEST_F(MPSNormTesting,RightNormalMatriciesProductStateL100S3D2)
{
    mps.InitializeWithProductState();
    mps.Normalize(MatrixProductSite::Right);
    VerifyRightNorm(&mps);
}


TEST_F(MPSNormTesting,LeftNormalMatriciesRandomStateL10S3D2)
{
    mps.InitializeWithRandomState();
    mps.Normalize(MatrixProductSite::Left);
    VerifyLeftNorm(&mps);
}

TEST_F(MPSNormTesting,RightNormalMatriciesRandomStateL100S3D2)
{
    mps.InitializeWithRandomState();
    mps.Normalize(MatrixProductSite::Right);
    VerifyRightNorm(&mps);
}

TEST_F(MPSNormTesting,LeftNormalMatriciesRandomStateL10S3D3)
{
    MatrixProductState mps1(10,3,3);
    mps1.InitializeWithRandomState();
    mps1.Normalize(MatrixProductSite::Left);
    VerifyLeftNorm(&mps1);
}


TEST_F(MPSNormTesting,LeftNormalMatriciesRandomStateL10S3D10)
{
    MatrixProductState mps1(10,1,10);
    mps1.InitializeWithRandomState();
    mps1.Normalize(MatrixProductSite::Left);
    VerifyLeftNorm(&mps1);
}

TEST_F(MPSNormTesting,RightNormalMatriciesRandomStateL10S3D10)
{
    MatrixProductState mps1(10,3,10);
    mps1.InitializeWithRandomState();
    mps1.Normalize(MatrixProductSite::Right);
    VerifyRightNorm(&mps1);
}

TEST_F(MPSNormTesting,LeftNormalOverlapL10S1D2)
{
    MatrixProductState mps1(10,1,2);
    mps1.InitializeWithRandomState();
    mps1.Normalize(MatrixProductSite::Left);
    EXPECT_NEAR(mps1.GetOverlap(),std::real(mps1.GetLeftNorm(9)(1,1)),0.01);
}

TEST_F(MPSNormTesting,RightNormalOverlapL10S2D2)
{
    MatrixProductState mps1(10,3,2);
    mps1.InitializeWithRandomState();
    mps1.Normalize(MatrixProductSite::Right);
    EXPECT_NEAR(mps1.GetOverlap(),std::real(mps1.GetRightNorm(0)(1,1)),0.01);
}

TEST_F(MPSNormTesting,RightNormalOverlapSite0L10S1D3)
{
    MatrixProductState mps1(10,1,3);
    mps1.InitializeWithRandomState();
    mps1.Normalize(MatrixProductSite::Right);
    VerifyUnit(mps1.GetNeff(0),eps);
}
TEST_F(MPSNormTesting,LeftNormalOverlapSiteLL10S1D3)
{
    MatrixProductState mps1(10,1,3);
    mps1.InitializeWithRandomState();
    mps1.Normalize(MatrixProductSite::Left);
    VerifyUnit(mps1.GetNeff(mps1.GetL()-1),eps);
}

TEST_F(MPSNormTesting,RightNormalOverlapSite0L10S5D2)
{
    MatrixProductState mps1(10,5,2);
    mps1.InitializeWithRandomState();
    mps1.Normalize(MatrixProductSite::Right);
    VerifyUnit(mps1.GetNeff(0),eps);
}
TEST_F(MPSNormTesting,LeftNormalOverlapSiteLL10S5D2)
{
    MatrixProductState mps1(10,5,2);
    mps1.InitializeWithRandomState();
    mps1.Normalize(MatrixProductSite::Left);
    VerifyUnit(mps1.GetNeff(mps1.GetL()-1),eps);
}

TEST_F(MPSNormTesting,MixedCanonicalL10S1D3)
{
    MatrixProductState mps1(10,1,3);
    mps1.InitializeWithRandomState();
    for (int ia=1;ia<mps1.GetL()-1;ia++)
    {
        mps1.Normalize(ia);
        VerifyUnit(mps1.GetNeff(ia),eps);
    }
}

//  Slow test
TEST_F(MPSNormTesting,MixedCanonicalL10S3D10)
{
    MatrixProductState mps1(10,3,10);
    mps1.InitializeWithRandomState();
    for (int ia=1;ia<mps1.GetL()-1;ia++)
    {
        mps1.Normalize(ia);
        VerifyUnit(mps1.GetNeff(ia),eps);
    }
}

TEST_F(MPSNormTesting,MixedCanonicalL10S5D2)
{
    MatrixProductState mps1(10,5,2);
    mps1.InitializeWithRandomState();
    for (int ia=1;ia<mps1.GetL()-1;ia++)
    {
        mps1.Normalize(ia);
        VerifyUnit(mps1.GetNeff(ia),eps);
    }
}


