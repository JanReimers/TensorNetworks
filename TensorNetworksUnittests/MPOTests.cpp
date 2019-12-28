#include "gtest/gtest.h"
#include "TensorNetworks/MatrixProductOperator.H"
#include "TensorNetworks/Hamiltonian_1D_NN_Heisenberg.H"
#include "oml/stream.h"
#include <complex>

class MPOTesting : public ::testing::Test
{
public:
    typedef MatrixProductSite::MatrixT MatrixT;
    MPOTesting()
    : eps(1.0e-13)
    {
        StreamableObject::SetToPretty();

    }

    void SetSites()
    {
            itsSites=&(itsMPO->itsSites);
    }
    MatrixT GetW(int isite, int m, int n) {return (*itsSites)[isite]->itsWs(m+1,n+1);}

    Hamiltonian* itsH;
    MatrixProductOperator* itsMPO;
    MatrixProductState*    itsMPS;
    MatrixProductOperator::SitesType* itsSites;
    double eps;
};

template <class Ob> std::string ToString(const Ob& result)
{
    std::stringstream res_stream;
    res_stream << result;
    return res_stream.str();
}



TEST_F(MPOTesting,MakeHamiltonian)
{
    itsH=new Hamiltonian_1D_NN_Heisenberg(10,1,1.0);
}
TEST_F(MPOTesting,MakeMatrixProductOperator)
{
    itsMPO=itsH->CreateMPO();
}
TEST_F(MPOTesting,MakeMatrixState)
{
    itsMPS=itsH->CreateMPS(5);
    SetSites();
}

TEST_F(MPOTesting,MPOMemebers)
{
    EXPECT_EQ(itsMPO->GetL(),10);
    EXPECT_EQ(itsMPO->GetD(),5);
    EXPECT_EQ(itsMPO->Getp(),2);
}

TEST_F(MPOTesting,HamiltonianGetLeftW00)
{
    EXPECT_EQ(ToString(itsH->GetW(Hamiltonian::Left,0,0)),"(1:5),(1:1) \n[ (0,0) ]\n[ (0,0) ]\n[ (0,0) ]\n[ (-0.5,0) ]\n[ (0,0) ]\n");
}
TEST_F(MPOTesting,HamiltonianGetLeftW10)
{
    EXPECT_EQ(ToString(itsH->GetW(Hamiltonian::Left,1,0)),"(1:5),(1:1) \n[ (0,0) ]\n[ (0,0) ]\n[ (0.5,0) ]\n[ (0,0) ]\n[ (0,0) ]\n");
}
TEST_F(MPOTesting,HamiltonianGetLeftW01)
{
    EXPECT_EQ(ToString(itsH->GetW(Hamiltonian::Left,0,1)),"(1:5),(1:1) \n[ (0,0) ]\n[ (0.5,0) ]\n[ (0,0) ]\n[ (0,0) ]\n[ (0,0) ]\n");
}
TEST_F(MPOTesting,HamiltonianGetLeftW11)
{
    EXPECT_EQ(ToString(itsH->GetW(Hamiltonian::Left,1,1)),"(1:5),(1:1) \n[ (0,0) ]\n[ (0,0) ]\n[ (0,0) ]\n[ (0.5,0) ]\n[ (0,0) ]\n");
}

TEST_F(MPOTesting,CheckThatWsGotLoaded)
{
    EXPECT_EQ(ToString(GetW(1,0,0)),"(1:5),(1:5) \n[ (1,0) (0,0) (0,0) (0,0) (0,0) ]\n[ (0,0) (0,0) (0,0) (0,0) (0,0) ]\n[ (0,0) (0,0) (0,0) (0,0) (0,0) ]\n[ (-0.5,0) (0,0) (0,0) (0,0) (0,0) ]\n[ (0,0) (0,0) (0,0) (-0.5,0) (1,0) ]\n");
    EXPECT_EQ(ToString(GetW(1,0,1)),"(1:5),(1:5) \n[ (1,0) (0,0) (0,0) (0,0) (0,0) ]\n[ (0,0) (0,0) (0,0) (0,0) (0,0) ]\n[ (1,0) (0,0) (0,0) (0,0) (0,0) ]\n[ (0,0) (0,0) (0,0) (0,0) (0,0) ]\n[ (0,0) (0.5,0) (0,0) (0,0) (1,0) ]\n");
}

TEST_F(MPOTesting,DoHamiltionExpectation)
{
    itsMPS->InitializeWithProductState();
    itsMPO->GetHamiltonianExpectation(itsMPS);
    EXPECT_NEAR(itsMPO->GetHamiltonianExpectation(itsMPS),896.0,1e-11);
}
