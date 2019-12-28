#include "gtest/gtest.h"
#include "TensorNetworks/MatrixProductOperator.H"
#include "TensorNetworks/Hamiltonian_1D_NN_Heisenberg.H"
#include "oml/stream.h"
#include <complex>

class MPOTesting : public ::testing::Test
{
public:
    MPOTesting()
    : eps(1.0e-13)
    {
        StreamableObject::SetToPretty();

    }
    typedef MatrixProductSite::MatrixT MatrixT;

    Hamiltonian* itsH;
    MatrixProductOperator* itsMPO;
    MatrixProductState*    itsMPS;
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
}

TEST_F(MPOTesting,MPOMemebers)
{
    EXPECT_EQ(itsMPO->GetL(),10);
    EXPECT_EQ(itsMPO->GetD(),5);
    EXPECT_EQ(itsMPO->Getp(),2);
}

TEST_F(MPOTesting,HamiltonianGetLeftW00)
{
    EXPECT_EQ(ToString(itsH->GetLeftW(0,0)),"(1:5),(1:1) \n[ (0,0) ]\n[ (0,0) ]\n[ (0,0) ]\n[ (-0.5,0) ]\n[ (0,0) ]\n");
}
TEST_F(MPOTesting,HamiltonianGetLeftW10)
{
    EXPECT_EQ(ToString(itsH->GetLeftW(1,0)),"(1:5),(1:1) \n[ (0,0) ]\n[ (0,0) ]\n[ (0.5,0) ]\n[ (0,0) ]\n[ (0,0) ]\n");
}
TEST_F(MPOTesting,HamiltonianGetLeftW01)
{
    EXPECT_EQ(ToString(itsH->GetLeftW(0,1)),"(1:5),(1:1) \n[ (0,0) ]\n[ (0.5,0) ]\n[ (0,0) ]\n[ (0,0) ]\n[ (0,0) ]\n");
}
TEST_F(MPOTesting,HamiltonianGetLeftW11)
{
    EXPECT_EQ(ToString(itsH->GetLeftW(1,1)),"(1:5),(1:1) \n[ (0,0) ]\n[ (0,0) ]\n[ (0,0) ]\n[ (0.5,0) ]\n[ (0,0) ]\n");
}
