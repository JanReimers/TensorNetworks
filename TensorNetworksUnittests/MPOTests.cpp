#include "Tests.H"
#include "TensorNetworks/MatrixProductOperator.H"
#include "TensorNetworks/Hamiltonian_1D_NN_Heisenberg.H"
//#include "TensorNetworks/Matrix6.H"
#include "oml/stream.h"
#include <complex>

typedef MatrixProductOperator::Matrix6T Matrix6T;
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
    itsMPS=itsH->CreateMPS(2);
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
    EXPECT_EQ(ToString(itsH->GetW(Hamiltonian::Left,0,0)),"(1:1),(1:5) \n[ (0,0) (0,0) (0,0) (-0.5,0) (1,0) ]\n");
}

TEST_F(MPOTesting,HamiltonianGetRightW00)
{
    EXPECT_EQ(ToString(itsH->GetW(Hamiltonian::Right,0,0)),"(1:5),(1:1) \n[ (1,0) ]\n[ (0,0) ]\n[ (0,0) ]\n[ (-0.5,0) ]\n[ (0,0) ]\n");
}
TEST_F(MPOTesting,HamiltonianGetLeftW10)
{
    EXPECT_EQ(ToString(itsH->GetW(Hamiltonian::Left,1,0)),"(1:1),(1:5) \n[ (0,0) (0,0) (0.5,0) (0,0) (1,0) ]\n");
}
TEST_F(MPOTesting,HamiltonianGetRightW10)
{
    EXPECT_EQ(ToString(itsH->GetW(Hamiltonian::Right,1,0)),"(1:5),(1:1) \n[ (1,0) ]\n[ (1,0) ]\n[ (0,0) ]\n[ (0,0) ]\n[ (0,0) ]\n");
}
TEST_F(MPOTesting,HamiltonianGetLeftW01)
{
    EXPECT_EQ(ToString(itsH->GetW(Hamiltonian::Left,0,1)),"(1:1),(1:5) \n[ (0,0) (0.5,0) (0,0) (0,0) (1,0) ]\n");
}
TEST_F(MPOTesting,HamiltonianGetRightW01)
{
    EXPECT_EQ(ToString(itsH->GetW(Hamiltonian::Right,0,1)),"(1:5),(1:1) \n[ (1,0) ]\n[ (0,0) ]\n[ (1,0) ]\n[ (0,0) ]\n[ (0,0) ]\n");
}
TEST_F(MPOTesting,HamiltonianGetLeftW11)
{
    EXPECT_EQ(ToString(itsH->GetW(Hamiltonian::Left,1,1)),"(1:1),(1:5) \n[ (0,0) (0,0) (0,0) (0.5,0) (1,0) ]\n");
}
TEST_F(MPOTesting,HamiltonianGetRightW11)
{
    EXPECT_EQ(ToString(itsH->GetW(Hamiltonian::Right,1,1)),"(1:5),(1:1) \n[ (1,0) ]\n[ (0,0) ]\n[ (0,0) ]\n[ (0.5,0) ]\n[ (0,0) ]\n");
}

TEST_F(MPOTesting,CheckThatWsGotLoaded)
{
    EXPECT_EQ(ToString(GetW(1,0,0)),"(1:5),(1:5) \n[ (1,0) (0,0) (0,0) (0,0) (0,0) ]\n[ (0,0) (0,0) (0,0) (0,0) (0,0) ]\n[ (0,0) (0,0) (0,0) (0,0) (0,0) ]\n[ (-0.5,0) (0,0) (0,0) (0,0) (0,0) ]\n[ (0,0) (0,0) (0,0) (-0.5,0) (1,0) ]\n");
    EXPECT_EQ(ToString(GetW(1,0,1)),"(1:5),(1:5) \n[ (1,0) (0,0) (0,0) (0,0) (0,0) ]\n[ (0,0) (0,0) (0,0) (0,0) (0,0) ]\n[ (1,0) (0,0) (0,0) (0,0) (0,0) ]\n[ (0,0) (0,0) (0,0) (0,0) (0,0) ]\n[ (0,0) (0.5,0) (0,0) (0,0) (1,0) ]\n");
}

TEST_F(MPOTesting,DoHamiltionExpectation)
{
    itsMPS->InitializeWith(MatrixProductSite::Product);
    itsMPO->GetExpectation(itsMPS);
    EXPECT_NEAR(itsMPO->GetExpectation(itsMPS),576.0,1e-11);
}

TEST_F(MPOTesting,LeftNormalizeThenDoHamiltionExpectation)
{
    itsMPS->InitializeWith(MatrixProductSite::Product);
    itsMPS->Normalize(MatrixProductSite::Left);
    itsMPO->GetExpectation(itsMPS);
    EXPECT_NEAR(itsMPO->GetExpectation(itsMPS),288.0,1e-11);
}
TEST_F(MPOTesting,RightNormalizeThenDoHamiltionExpectation)
{
    itsMPS->InitializeWith(MatrixProductSite::Product);
    itsMPS->Normalize(MatrixProductSite::Left);
    itsMPO->GetExpectation(itsMPS);
    EXPECT_NEAR(itsMPO->GetExpectation(itsMPS),288.0,1e-11);
}

TEST_F(MPOTesting,TestHeffWithProductState)
{
//    itsMPS->InitializeWith(MatrixProductSite::Random);
    itsMPS->InitializeWith(MatrixProductSite::Product);
    for (int ia=0; ia<itsMPS->GetL(); ia++)
    {
        itsMPS->Normalize(ia);
        Matrix6T Heff=itsMPO->GetHeff(itsMPS,ia);
 //       cout << "E(" << ia << ")=" << itsMPS->ConstractHeff(ia,Heff) << endl;
        EXPECT_NEAR(itsMPS->ContractHeff(ia,Heff),576.0,1e-11);
        MatrixT HeffF=Heff.Flatten();
        //cout << "Heff=" << Heff << endl;
        MatrixT d=HeffF-Transpose(conj(HeffF));
        EXPECT_NEAR(Max(abs(d)),0.0,1e-11);

//    cout << "d=" << d << endl;
    }
}
TEST_F(MPOTesting,TestHeffWithRandomState)
{
    itsMPS->InitializeWith(MatrixProductSite::Random);
    itsMPS->Normalize(MatrixProductSite::Right);
    double E1=itsMPO->GetExpectation(itsMPS);
    for (int ia=0; ia<itsMPS->GetL(); ia++)
    {
        itsMPS->Normalize(ia);
        Matrix6T Heff=itsMPO->GetHeff(itsMPS,ia);
        double E2=itsMPS->ContractHeff(ia,Heff);
        EXPECT_NEAR(E1,E2,100*eps);
        double E3=itsMPS->ContractHeff(ia,Heff.Flatten());
        EXPECT_NEAR(E1,E3,100*eps);

        MatrixT HeffF=Heff.Flatten();
        MatrixT d=HeffF-Transpose(conj(HeffF));
        EXPECT_NEAR(Max(abs(d)),0.0,100*eps);
    }
}
