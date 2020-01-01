#include "gtest/gtest.h"
#include "TensorNetworks/MatrixProductState.H"
#include "oml/stream.h"
#include "oml/random.h"
#include <complex>
#include <iostream>
#include <string>

using std::cout;
using std::endl;


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



template <class Ob> std::string ToString(const Ob& result)
{
    std::stringstream res_stream;
    res_stream << result;
    return res_stream.str();
}

void MPSNormTesting::VerifyLeftNorm(const MatrixProductState* mps)
{
    for (int i=0; i<mps->GetL(); i++)
    {
        MatrixT Norm=mps->GetLeftNorm(i);
        int D=Norm.GetNumRows();
 //       cout << "site " << i << " has D=" << D << endl;
        MatrixT I(D,D);
        Unit(I);
        EXPECT_NEAR(Max(abs(real(Norm-I))),0.0,eps);
        EXPECT_NEAR(Max(abs(imag(Norm  ))),0.0,eps);
        std::string lim="(1:";
        lim= lim + std::to_string(D) + "),(1:" + std::to_string(D) + ") ";
        EXPECT_EQ(ToString(Norm.GetLimits()),lim.c_str());
    }


}

void MPSNormTesting::VerifyRightNorm(const MatrixProductState* mps)
{

    for (int i=0; i<mps->GetL(); i++)
    {
        MatrixT Norm=mps->GetRightNorm(i);
        int D=Norm.GetNumRows();
//        cout << "site " << i << " has D=" << D << endl;
        MatrixT I(D,D);
        Unit(I);
        EXPECT_NEAR(Max(abs(real(Norm-I))),0.0,eps);
        EXPECT_NEAR(Max(abs(imag(Norm  ))),0.0,eps);
        std::string lim="(1:";
        lim= lim + std::to_string(D) + "),(1:" + std::to_string(D) + ") ";
        EXPECT_EQ(ToString(Norm.GetLimits()),lim.c_str());
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


TEST_F(MPSNormTesting,LeftNormalMatriciesRandomStateL10S3D30)
{
    MatrixProductState mps1(10,1,30);
    mps1.InitializeWithRandomState();
    mps1.Normalize(MatrixProductSite::Left);
    VerifyLeftNorm(&mps1);
}

TEST_F(MPSNormTesting,RightNormalMatriciesRandomStateL10S3D30)
{
    MatrixProductState mps1(10,3,30);
    mps1.InitializeWithRandomState();
    mps1.Normalize(MatrixProductSite::Right);
    VerifyRightNorm(&mps1);
}

TEST_F(MPSNormTesting,LeftNormalOverlapL10S3D30)
{
    MatrixProductState mps1(10,1,30);
    mps1.InitializeWithRandomState();
    mps1.Normalize(MatrixProductSite::Left);
    EXPECT_NEAR(mps1.GetOverlap(),1.0,eps);
}
TEST_F(MPSNormTesting,RightNormalOverlapL10S3D30)
{
    MatrixProductState mps1(10,1,30);
    mps1.InitializeWithRandomState();
    mps1.Normalize(MatrixProductSite::Right);
    EXPECT_NEAR(mps1.GetOverlap(),1.0,eps);
}
