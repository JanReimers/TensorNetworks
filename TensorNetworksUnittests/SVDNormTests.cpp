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

    {
        MatrixT Norm=mps->GetMLeft(0);
        EXPECT_EQ(ToString(Norm.GetLimits()),"(1:1),(1:1) ");
        EXPECT_NEAR(real(Norm(1,1)),1.0,eps);
        EXPECT_NEAR(imag(Norm(1,1)),0.0,eps);
    }


    for (int i=1; i<mps->GetL(); i++)
    {
        MatrixT Norm=mps->GetMLeft(i);
        int D=Norm.GetNumRows();
        cout << "site " << i << " has D=" << D << endl;
        MatrixT I(D,D);
        Unit(I);
        EXPECT_NEAR(Max(abs(real(Norm-I))),0.0,eps);
        EXPECT_NEAR(Max(abs(imag(Norm  ))),0.0,eps);
        std::string lim="(1:";
        lim= lim + std::to_string(D) + "),(1:" + std::to_string(D) + ") ";
        EXPECT_EQ(ToString(Norm.GetLimits()),lim.c_str());
    }


}
/*

TEST_F(MPSNormTesting,LeftNormalMatriciesProductStateL10S3D2)
{
    mps.InitializeWithProductState();
    mps.Normalize(MatrixProductSite::Left);
    VerifyLeftNorm(&mps);
}

TEST_F(MPSNormTesting,LeftNormalMatriciesRandomStateL10S3D2)
{
    mps.InitializeWithRandomState();
    mps.Normalize(MatrixProductSite::Left);
    VerifyLeftNorm(&mps);
}

TEST_F(MPSNormTesting,LeftNormalMatriciesRandomStateL10S3D3)
{
    MatrixProductState mps1(10,3,3);
    mps1.InitializeWithRandomState();
    mps1.Normalize(MatrixProductSite::Left);
    VerifyLeftNorm(&mps1);
}
*/

TEST_F(MPSNormTesting,LeftNormalMatriciesRandomStateL10S3D30)
{
    MatrixProductState mps1(10,1,30);
    mps1.InitializeWithRandomState();
    mps1.Normalize(MatrixProductSite::Left);
//    MatrixT Norm=mps1.GetLeftNorm(0);
//    cout << "Norm[0]=" << Norm << endl;
//    Norm=mps1.GetLeftNorm(1);
//    cout << "Norm[1]=" << Norm << endl;
    VerifyLeftNorm(&mps1);
}

