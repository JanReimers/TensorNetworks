#include "Tests.H"

#include "TensorNetworks/MatrixProductState.H"
#include "oml/stream.h"
#include "oml/random.h"
#include <complex>


class MPSNormTesting : public ::testing::Test
{
public:
    MPSNormTesting()
    : itsMPS(0)
 //   , itsSites(itsMPS->itsSites)
    , eps(1.0e-10)
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
        itsMPS=new MatrixProductState(L,S2,D);
    }

    typedef MatrixProductSite::MatrixCT MatrixCT;
    MatrixCT GetNeff   (int isite) const {return itsMPS->GetNeff(isite);}

    void VerifyLeftNorm();
    void VerifyRightNorm();

//    const MatrixT& GetA(int i,int ip) const {return itsSites[i]->itsAs[ip]; }

    MatrixProductState* itsMPS;
//    MatrixProductState::SitesType& itsSites;
    double eps;
};




void MPSNormTesting::VerifyLeftNorm()
{
    // Skip the last site
    for (int i=0; i<itsMPS->GetL(); i++)
    {
//        cout << mps->GetLeftNorm(i) << endl;
        VerifyUnit(itsMPS->itsSites[i]->GetLeftNorm(),eps);
        VerifyUnit(itsMPS->GetMLeft(i),eps);
    }
}

void MPSNormTesting::VerifyRightNorm()
{
    //skip the first site
    for (int i=0; i<itsMPS->GetL(); i++)
    {
//        cout << mps->GetRightNorm(i) << endl;
        VerifyUnit(itsMPS->itsSites[i]->GetRightNorm(),eps);
        VerifyUnit(itsMPS->GetMRight(i),eps);
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
    itsMPS->InitializeWith(MatrixProductSite::Product);
    itsMPS->Normalize(MatrixProductSite::Left);
    VerifyLeftNorm();
}


TEST_F(MPSNormTesting,RightNormalMatriciesProductStateL100S3D2)
{
    itsMPS->InitializeWith(MatrixProductSite::Product);
    itsMPS->Normalize(MatrixProductSite::Right);
    VerifyRightNorm();
}


TEST_F(MPSNormTesting,LeftNormalMatriciesRandomStateL10S3D2)
{
    itsMPS->InitializeWith(MatrixProductSite::Random);
    itsMPS->Normalize(MatrixProductSite::Left);
    VerifyLeftNorm();
}

TEST_F(MPSNormTesting,RightNormalMatriciesRandomStateL100S3D2)
{
    itsMPS->InitializeWith(MatrixProductSite::Random);
    itsMPS->Normalize(MatrixProductSite::Right);
    VerifyRightNorm();
}

TEST_F(MPSNormTesting,LeftNormalMatriciesRandomStateL10S3D3)
{
    Setup(10,3,3);
    itsMPS->InitializeWith(MatrixProductSite::Random);
    itsMPS->Normalize(MatrixProductSite::Left);
    VerifyLeftNorm();
}

TEST_F(MPSNormTesting,LeftNormalMatriciesRandomStateL10S1D1)
{
    int L=10;
    Setup(L,1,1);
    itsMPS->InitializeWith(MatrixProductSite::Random);
    itsMPS->Normalize(MatrixProductSite::Left);
    VerifyLeftNorm();
}
TEST_F(MPSNormTesting,LeftNormalMatriciesRandomStateL10S5D1)
{
    int L=10;
    Setup(L,5,1);
    itsMPS->InitializeWith(MatrixProductSite::Random);
    itsMPS->Normalize(MatrixProductSite::Left);
    VerifyLeftNorm();
}
TEST_F(MPSNormTesting,RightNormalMatriciesRandomStateL10S1D1)
{
    int L=10;
    Setup(L,1,1);
    itsMPS->InitializeWith(MatrixProductSite::Random);
    itsMPS->Normalize(MatrixProductSite::Right);
    VerifyLeftNorm();
}
TEST_F(MPSNormTesting,RightNormalMatriciesRandomStateL10S5D1)
{
    int L=10;
    Setup(L,5,1);
    itsMPS->InitializeWith(MatrixProductSite::Random);
    itsMPS->Normalize(MatrixProductSite::Right);
    VerifyLeftNorm();
}

TEST_F(MPSNormTesting,LeftNormalMatriciesRandomStateL10S3D10)
{
    Setup(10,1,10);
    itsMPS->InitializeWith(MatrixProductSite::Random);
    itsMPS->Normalize(MatrixProductSite::Left);
    VerifyLeftNorm();
}

TEST_F(MPSNormTesting,RightNormalMatriciesRandomStateL10S3D10)
{
    Setup(10,3,10);
    itsMPS->InitializeWith(MatrixProductSite::Random);
    itsMPS->Normalize(MatrixProductSite::Right);
    VerifyRightNorm();
}

TEST_F(MPSNormTesting,LeftNormalOverlapL10S1D2)
{
    Setup(10,1,2);
    itsMPS->InitializeWith(MatrixProductSite::Random);
    itsMPS->Normalize(MatrixProductSite::Left);
    EXPECT_NEAR(itsMPS->GetOverlap(),1.0,eps);
}

TEST_F(MPSNormTesting,RightNormalOverlapL10S2D2)
{
    Setup(10,3,2);
    itsMPS->InitializeWith(MatrixProductSite::Random);
    itsMPS->Normalize(MatrixProductSite::Right);
    EXPECT_NEAR(itsMPS->GetOverlap(),1.0,eps);
}

TEST_F(MPSNormTesting,RightNormalOverlapSite0L10S1D3)
{
    Setup(10,1,3);
    itsMPS->InitializeWith(MatrixProductSite::Random);
    itsMPS->Normalize(MatrixProductSite::Right);
    VerifyUnit(GetNeff(0),eps);
}
TEST_F(MPSNormTesting,LeftNormalOverlapSiteLL10S1D3)
{
    Setup(10,1,3);
    itsMPS->InitializeWith(MatrixProductSite::Random);
    itsMPS->Normalize(MatrixProductSite::Left);
    VerifyUnit(GetNeff(itsMPS->GetL()-1),eps);
}

TEST_F(MPSNormTesting,RightNormalOverlapSite0L10S5D2)
{
    Setup(10,5,2);
    itsMPS->InitializeWith(MatrixProductSite::Random);
    itsMPS->Normalize(MatrixProductSite::Right);
    VerifyUnit(GetNeff(0),eps);
}
TEST_F(MPSNormTesting,LeftNormalOverlapSiteLL10S5D2)
{
    Setup(10,5,2);
    itsMPS->InitializeWith(MatrixProductSite::Random);
    itsMPS->Normalize(MatrixProductSite::Left);
    VerifyUnit(GetNeff(itsMPS->GetL()-1),eps);
}

TEST_F(MPSNormTesting,MixedCanonicalL10S1D3)
{
    int L=10;
    Setup(L,1,3);
    itsMPS->InitializeWith(MatrixProductSite::Random);
    for (int ia=0;ia<itsMPS->GetL();ia++)
    {
        itsMPS->Normalize(ia);
        VerifyUnit(GetNeff(ia),eps);
        EXPECT_EQ(itsMPS->GetNormStatus(),ExpectedNorm(ia,L));
    }
}

//  Slow test
TEST_F(MPSNormTesting,MixedCanonicalL10S3D10)
{
    int L=10;
    Setup(L,3,10);
    itsMPS->InitializeWith(MatrixProductSite::Random);
    for (int ia=0;ia<itsMPS->GetL();ia++)
    {
        itsMPS->Normalize(ia);
        VerifyUnit(GetNeff(ia),eps);
        EXPECT_EQ(itsMPS->GetNormStatus(),ExpectedNorm(ia,L));
   }
}


TEST_F(MPSNormTesting,MixedCanonicalL10S5D2)
{
    int L=10;
    Setup(L,5,2);
    itsMPS->InitializeWith(MatrixProductSite::Random);
    for (int ia=0;ia<itsMPS->GetL();ia++)
    {
        itsMPS->Normalize(ia);
        VerifyUnit(GetNeff(ia),eps);
        EXPECT_EQ(itsMPS->GetNormStatus(),ExpectedNorm(ia,L));
    }
}

TEST_F(MPSNormTesting,MixedCanonicalL10S5D1)
{
    int L=10;
    Setup(L,5,1);
    itsMPS->InitializeWith(MatrixProductSite::Random);
    for (int ia=0;ia<itsMPS->GetL();ia++)
    {
        itsMPS->Normalize(ia);
        VerifyUnit(GetNeff(ia),eps);
        // This fails for D=1 becuase left and right normalization are indistinguisable when D=1
        //EXPECT_EQ(itsMPS->GetNormStatus(),ExpectedNorm(ia,L));
    }
}
