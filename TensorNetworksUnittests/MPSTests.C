#include "gtest/gtest.h"
#include "Tests.H"
#include "TensorNetworksImp/MPS/MPSImp.H"
#include "TensorNetworks/Typedefs.H"
#include "TensorNetworks/Epsilons.H"
//#include "oml/stream.h"
#include "oml/random.h"
#include <complex>

class MPSTests : public ::testing::Test
{
public:
    MPSTests()
    : itsMPS(0)
    , eps(1.0e-10)
    {
        Setup(10,0.5,2);
        StreamableObject::SetToPretty();
    }
    ~MPSTests()
    {
        if (itsMPS) delete itsMPS;
    }



    void Setup(int L, double  S, int D)
    {
        if (itsMPS) delete itsMPS;
        itsMPS=new TensorNetworks::MPSImp(L,S,D,1e-12,1e-12);
    }


    typedef TensorNetworks::MatrixCT MatrixCT;

    MatrixCT GetA(int i,int ip) const {return GetSite(i)->itsMs[ip]; }
    const TensorNetworks::MPSSite* GetSite(int isite) const {return itsMPS->itsSites[isite];}
    MatrixCT GetNorm(TensorNetworks::Direction lr,int isite) const
    {
        return GetSite(isite)->GetNorm(lr);
    }

    TensorNetworks::MPSImp*     itsMPS;
    double eps;
};





TEST_F(MPSTests,Constructor)
{
    EXPECT_EQ(itsMPS->GetL(),10);
//    EXPECT_EQ(itsMPS->GetD(),2);
    EXPECT_EQ(itsMPS->Getp(),2);
}

TEST_F(MPSTests,MatrixOpMul)
{
    Matrix<double> A(10,10),B(10,9);
    FillRandom(A);
    FillRandom(B);
    A*=B;
}


TEST_F(MPSTests,LeftNormalMatricies)
{
    itsMPS->InitializeWith(TensorNetworks::Product);
    EXPECT_TRUE(IsUnit(GetNorm(TensorNetworks::DLeft,1),eps));
    EXPECT_TRUE(IsUnit(GetNorm(TensorNetworks::DLeft,2),eps));
    EXPECT_TRUE(IsUnit(GetNorm(TensorNetworks::DLeft,9),eps));
}


TEST_F(MPSTests,RightNormalMatricies)
{
    itsMPS->InitializeWith(TensorNetworks::Product);
    // The first site will not be right normalized
    EXPECT_EQ(ToString(GetNorm(TensorNetworks::DRight,1)),"(1:1),(1:1) \n[ (2,0) ]\n");
    EXPECT_TRUE(IsUnit(GetNorm(TensorNetworks::DRight,2),eps));
    EXPECT_TRUE(IsUnit(GetNorm(TensorNetworks::DRight,9),eps));
    EXPECT_TRUE(IsUnit(GetNorm(TensorNetworks::DRight,10),eps));
}


TEST_F(MPSTests,LeftNormalMatricies_S2)
{
    Setup(10,1.5,2);
    itsMPS->InitializeWith(TensorNetworks::Product);
    EXPECT_TRUE(IsUnit(GetNorm(TensorNetworks::DLeft,1),eps));
    EXPECT_TRUE(IsUnit(GetNorm(TensorNetworks::DLeft,2),eps));
    EXPECT_TRUE(IsUnit(GetNorm(TensorNetworks::DLeft,9),eps));
}

TEST_F(MPSTests,RightNormalMatricies_S2)
{
    Setup(10,1.5,3);
    itsMPS->InitializeWith(TensorNetworks::Product);
    EXPECT_TRUE(IsUnit(GetNorm(TensorNetworks::DRight,2),eps));
    EXPECT_TRUE(IsUnit(GetNorm(TensorNetworks::DRight,9),eps));
    EXPECT_TRUE(IsUnit(GetNorm(TensorNetworks::DRight,10),eps));
    EXPECT_EQ(ToString(GetNorm(TensorNetworks::DRight,1)),"(1:1),(1:1) \n[ (3,0) ]\n");
}




