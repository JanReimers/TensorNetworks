#include "Tests.H"

#include "TensorNetworksImp/MatrixProductStateImp.H"
#include "TensorNetworks/LRPSupervisor.H"
#include "TensorNetworks/Epsilons.H"

#include "oml/stream.h"
#include "oml/random.h"
#include <complex>


class MPSNormTesting : public ::testing::Test
{
public:
    MPSNormTesting()
    : itsMPS(0)
    , eps(1.0e-10)
    , itsSupervisor( new LRPSupervisor())
    , itsEps()
    {
        StreamableObject::SetToPretty();
        Setup(10,1,2);
        std::cout.precision(5);
        cout << std::fixed;
    }
    ~MPSNormTesting() {delete itsMPS;}

    void Setup(int L, int S2, int D)
    {
        delete itsMPS;
        itsMPS=new MatrixProductStateImp(L,S2,D,itsEps);
    }

    typedef TensorNetworks::MatrixCT MatrixCT;

    MatrixProductStateImp* itsMPS;
    double eps;
    LRPSupervisor* itsSupervisor;
    Epsilons               itsEps;
};




std::string ExpectedNorm(int isite,int L)
{
    std::string ret;
    for (int ia=0;ia<isite;ia++) ret+="A0";
    ret+="M0";
    for (int ia=isite+1;ia<L;ia++) ret+="B0";
    return ret;
}

std::string BuildNormString(const MatrixProductStateImp* mps,int L)
{
    std::string ret;
    for (int ia=0;ia<L;ia++) ret+=mps->GetNormStatus(ia);
    return ret;
}





TEST_F(MPSNormTesting,LeftNormalMatriciesProductStateL10S3D2)
{
    int L=itsMPS->GetL();
    itsMPS->InitializeWith(TensorNetworks::Product);
    itsMPS->Normalize(TensorNetworks::Left,new LRPSupervisor());
    EXPECT_EQ(BuildNormString(itsMPS,L),"A0I0I0I0I0I0I0I0I0A0");
}


TEST_F(MPSNormTesting,RightNormalMatriciesProductStateL100S3D2)
{
    int L=itsMPS->GetL();
    itsMPS->InitializeWith(TensorNetworks::Product);
    itsMPS->Normalize(TensorNetworks::Right,new LRPSupervisor());
    EXPECT_EQ(BuildNormString(itsMPS,L),"B0I0I0I0I0I0I0I0I0B0");
}


TEST_F(MPSNormTesting,LeftNormalMatriciesRandomStateL10S3D2)
{
    int L=itsMPS->GetL();
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::Left,new LRPSupervisor());
    EXPECT_EQ(BuildNormString(itsMPS,L),"A0A0A0A0A0A0A0A0A0A0");
}

TEST_F(MPSNormTesting,RightNormalMatriciesRandomStateL100S3D2)
{
    int L=itsMPS->GetL();
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::Right,new LRPSupervisor());
    EXPECT_EQ(BuildNormString(itsMPS,L),"B0B0B0B0B0B0B0B0B0B0");
}

TEST_F(MPSNormTesting,LeftNormalMatriciesRandomStateL10S3D3)
{
    int L=10;
    Setup(L,3,3);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::Left,new LRPSupervisor());
    EXPECT_EQ(BuildNormString(itsMPS,L),"A0A0A0A0A0A0A0A0A0A0");
}

TEST_F(MPSNormTesting,LeftNormalMatriciesRandomStateL10S1D1)
{
    int L=10;
    Setup(L,1,1);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::Left,itsSupervisor);
    EXPECT_EQ(BuildNormString(itsMPS,L),"I0I0I0I0I0I0I0I0I0I0");
}
TEST_F(MPSNormTesting,LeftNormalMatriciesRandomStateL10S5D1)
{
    int L=10;
    Setup(L,5,1);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::Left,itsSupervisor);
    EXPECT_EQ(BuildNormString(itsMPS,L),"I0I0I0I0I0I0I0I0I0I0");
}
TEST_F(MPSNormTesting,RightNormalMatriciesRandomStateL10S1D1)
{
    int L=10;
    Setup(L,1,1);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::Right,itsSupervisor);
    EXPECT_EQ(BuildNormString(itsMPS,L),"I0I0I0I0I0I0I0I0I0I0");
}
TEST_F(MPSNormTesting,RightNormalMatriciesRandomStateL10S5D1)
{
    int L=10;
    Setup(L,5,1);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::Right,itsSupervisor);
    EXPECT_EQ(BuildNormString(itsMPS,L),"I0I0I0I0I0I0I0I0I0I0");
}

TEST_F(MPSNormTesting,LeftNormalMatriciesRandomStateL10S3D10)
{
    int L=10;
    Setup(L,1,10);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::Left,itsSupervisor);
    EXPECT_EQ(BuildNormString(itsMPS,L),"A0A0A0A0A0A0A0A0A0A0");
}

TEST_F(MPSNormTesting,RightNormalMatriciesRandomStateL10S3D10)
{
    int L=10;
    Setup(L,3,10);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::Right,itsSupervisor);
    EXPECT_EQ(BuildNormString(itsMPS,L),"B0B0B0B0B0B0B0B0B0B0");
}

TEST_F(MPSNormTesting,LeftNormalOverlapL10S1D2)
{
    int L=10;
    Setup(L,1,2);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::Left,itsSupervisor);
    EXPECT_EQ(BuildNormString(itsMPS,L),"A0A0A0A0A0A0A0A0A0A0");
}

TEST_F(MPSNormTesting,RightNormalOverlapL10S2D2)
{
    int L=10;
    Setup(L,3,2);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::Right,itsSupervisor);
    EXPECT_EQ(BuildNormString(itsMPS,L),"B0B0B0B0B0B0B0B0B0B0");
}



TEST_F(MPSNormTesting,MixedCanonicalL10S1D3)
{
    int L=10;
    Setup(L,1,3);
    itsMPS->InitializeWith(TensorNetworks::Random);
    for (int ia=0;ia<itsMPS->GetL();ia++)
    {
        itsMPS->Normalize(ia);
        EXPECT_EQ(BuildNormString(itsMPS,L),ExpectedNorm(ia,L));
    }
}


