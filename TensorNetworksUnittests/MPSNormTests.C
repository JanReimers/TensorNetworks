#include "Tests.H"

#include "TensorNetworksImp/MPSImp.H"
#include "TensorNetworks/Epsilons.H"

#include "oml/stream.h"
#include "oml/random.h"
#include <complex>


class MPSNormTests : public ::testing::Test
{
public:
    MPSNormTests()
    : itsMPS(0)
    , eps(1.0e-10)
    {
        StreamableObject::SetToPretty();
        Setup(10,1,2);
        std::cout.precision(5);
        cout << std::fixed;
    }
    ~MPSNormTests() {delete itsMPS;}

    void Setup(int L, double S, int D)
    {
        delete itsMPS;
        itsMPS=new TensorNetworks::MPSImp(L,S,D,1e-12,1e-12,0);
    }

    typedef TensorNetworks::MatrixCT MatrixCT;

    TensorNetworks::MPSImp* itsMPS;
    double                  eps;
};




std::string ExpectedNorm(int isite,int L)
{
    std::string ret;
    for (int ia=1;ia<isite;ia++) ret+="A";
    ret+="M";
    for (int ia=isite+1;ia<=L;ia++) ret+="B";
    return ret;
}

std::string BuildNormString(const TensorNetworks::MPSImp* mps,int L)
{
    std::string ret;
    for (int ia=1;ia<=L;ia++) ret+=mps->GetNormStatus(ia);
    return ret;
}





TEST_F(MPSNormTests,LeftNormalMatriciesProductStateL10S3D2)
{
    int L=itsMPS->GetL();
    itsMPS->InitializeWith(TensorNetworks::Product);
    itsMPS->Normalize(TensorNetworks::DLeft);
    EXPECT_EQ(BuildNormString(itsMPS,L),"AIIIIIIIIA");
}


TEST_F(MPSNormTests,RightNormalMatriciesProductStateL100S3D2)
{
    int L=itsMPS->GetL();
    itsMPS->InitializeWith(TensorNetworks::Product);
    itsMPS->Normalize(TensorNetworks::DRight);
    EXPECT_EQ(BuildNormString(itsMPS,L),"BIIIIIIIIB");
}


TEST_F(MPSNormTests,LeftNormalMatriciesRandomStateL10S3D2)
{
    int L=itsMPS->GetL();
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::DLeft);
    EXPECT_EQ(BuildNormString(itsMPS,L),"AAAAAAAAAA");
}

TEST_F(MPSNormTests,RightNormalMatriciesRandomStateL100S3D2)
{
    int L=itsMPS->GetL();
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::DRight);
    EXPECT_EQ(BuildNormString(itsMPS,L),"BBBBBBBBBB");
}

TEST_F(MPSNormTests,LeftNormalMatriciesRandomStateL10S3D3)
{
    int L=10;
    Setup(L,1.5,3);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::DLeft);
    EXPECT_EQ(BuildNormString(itsMPS,L),"AAAAAAAAAA");
}

TEST_F(MPSNormTests,LeftNormalMatriciesRandomStateL10S1D1)
{
    int L=10;
    Setup(L,0.5,1);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::DLeft);
    EXPECT_EQ(BuildNormString(itsMPS,L),"IIIIIIIIII");
}
TEST_F(MPSNormTests,LeftNormalMatriciesRandomStateL10S5D1)
{
    int L=10;
    Setup(L,2.5,1);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::DLeft);
    EXPECT_EQ(BuildNormString(itsMPS,L),"IIIIIIIIII");
}
TEST_F(MPSNormTests,RightNormalMatriciesRandomStateL10S1D1)
{
    int L=10;
    Setup(L,0.5,1);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::DRight);
    EXPECT_EQ(BuildNormString(itsMPS,L),"IIIIIIIIII");
}
TEST_F(MPSNormTests,RightNormalMatriciesRandomStateL10S5D1)
{
    int L=10;
    Setup(L,2.5,1);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::DRight);
    EXPECT_EQ(BuildNormString(itsMPS,L),"IIIIIIIIII");
}

TEST_F(MPSNormTests,LeftNormalMatriciesRandomStateL10S3D10)
{
    int L=10;
    Setup(L,0.5,10);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::DLeft);
    EXPECT_EQ(BuildNormString(itsMPS,L),"AAAAAAAAAA");
}

TEST_F(MPSNormTests,RightNormalMatriciesRandomStateL10S3D10)
{
    int L=10;
    Setup(L,1.5,10);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::DRight);
    EXPECT_EQ(BuildNormString(itsMPS,L),"BBBBBBBBBB");
}

TEST_F(MPSNormTests,LeftNormalOverlapL10S1D2)
{
    int L=10;
    Setup(L,0.5,2);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::DLeft);
    EXPECT_EQ(BuildNormString(itsMPS,L),"AAAAAAAAAA");
}

TEST_F(MPSNormTests,RightNormalOverlapL10S2D2)
{
    int L=10;
    Setup(L,1.5,2);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::DRight);
    EXPECT_EQ(BuildNormString(itsMPS,L),"BBBBBBBBBB");
}



TEST_F(MPSNormTests,MixedCanonicalL10S1D3)
{
    int L=10;
    Setup(L,0.5,3);
    itsMPS->InitializeWith(TensorNetworks::Random);
    for (int ia=1;ia<=itsMPS->GetL();ia++)
    {
        itsMPS->MixedCanonical(ia);
        EXPECT_EQ(BuildNormString(itsMPS,L),ExpectedNorm(ia,L));
    }
}

TEST_F(MPSNormTests,SeflOverlapRandomStateL10S3D10)
{
    int L=10;
    Setup(L,1.5,10);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::DRight);
    EXPECT_NEAR(itsMPS->GetOverlap(itsMPS),1.0,eps);
}


