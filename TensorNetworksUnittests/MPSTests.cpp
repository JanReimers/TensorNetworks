#include "gtest/gtest.h"
#include "Tests.H"
#include "TensorNetworks/Epsilons.H"
#include "oml/stream.h"
#include "oml/random.h"
#include <complex>

class MatrixProductTesting : public ::testing::Test
{
public:
    MatrixProductTesting()
    : itsMPS(0)
    , itsEps()
    , eps(1.0e-10)
    {
        Setup(10,1,2);
        StreamableObject::SetToPretty();
    }
    ~MatrixProductTesting() {delete itsMPS;}

    void Setup(int L, int S2, int D)
    {
        delete itsMPS;
        itsMPS=new MatrixProductStateImp(L,S2,D,itsEps);
    }


    typedef MatrixProductSite::MatrixCT MatrixCT;

    MatrixCT GetA(int i,int ip) const {return GetSite(i)->itsAs[ip]; }
    //MatrixCT GetMLeft(int isite) const {return itsMPS->GetMLeft(isite);}
    //MatrixCT GetMRight(int isite) const {return itsMPS->GetMRight(isite);}
    //MatrixCT GetNeff   (int isite) const {return itsMPS->GetNeff(isite);}
    const MatrixProductSite* GetSite(int isite) const {return itsMPS->itsSites[isite];}

    MatrixProductStateImp* itsMPS;
    Epsilons               itsEps;


    double eps;
};





TEST_F(MatrixProductTesting,Constructor)
{
    EXPECT_EQ(itsMPS->GetL(),10);
    EXPECT_EQ(itsMPS->GetD(),2);
    EXPECT_EQ(itsMPS->Getp(),2);
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
    itsMPS->InitializeWith(TensorNetworks::Product);
    VerifyUnit(GetSite(0)->GetLeftNorm(),eps);
    VerifyUnit(GetSite(1)->GetLeftNorm(),eps);
    VerifyUnit(GetSite(8)->GetLeftNorm(),eps);
}


TEST_F(MatrixProductTesting,RightNormalMatricies)
{
    itsMPS->InitializeWith(TensorNetworks::Product);
    // The first site will not be right normalized
    EXPECT_EQ(ToString(GetSite(0)->GetRightNorm()),"(1:1),(1:1) \n[ (2,0) ]\n");
    VerifyUnit(GetSite(1)->GetRightNorm(),eps);
    VerifyUnit(GetSite(8)->GetRightNorm(),eps);
    VerifyUnit(GetSite(9)->GetRightNorm(),eps);
}


TEST_F(MatrixProductTesting,LeftNormalMatricies_S2)
{
    Setup(10,3,2);
    itsMPS->InitializeWith(TensorNetworks::Product);
    VerifyUnit(GetSite(0)->GetLeftNorm(),eps);
    VerifyUnit(GetSite(1)->GetLeftNorm(),eps);
    VerifyUnit(GetSite(8)->GetLeftNorm(),eps);
}

TEST_F(MatrixProductTesting,RightNormalMatricies_S2)
{
    Setup(10,3,3);
    itsMPS->InitializeWith(TensorNetworks::Product);
    VerifyUnit(GetSite(1)->GetRightNorm(),eps);
    VerifyUnit(GetSite(8)->GetRightNorm(),eps);
    VerifyUnit(GetSite(9)->GetRightNorm(),eps);
    EXPECT_EQ(ToString(GetSite(0)->GetRightNorm()),"(1:1),(1:1) \n[ (3,0) ]\n");
}




