#include "Tests.H"
#include "TensorNetworks/MatrixProductOperator.H"
#include "TensorNetworks/Hamiltonian_1D_NN_Heisenberg.H"
//#include "TensorNetworks/Matrix6.H"
#include "oml/stream.h"
#include "oml/vector_io.h"
#include <complex>

typedef MatrixProductOperator::Matrix6T Matrix6T;
class MPOTesting : public ::testing::Test
{
public:
    typedef MatrixProductSite::MatrixT MatrixT;
    typedef MatrixProductSite::Vector3T Vector3T;
    typedef MatrixProductSite::eType eType;
    MPOTesting()
    : eps(1.0e-13)
    {
        StreamableObject::SetToPretty();

    }
    void Setup(int L, int S2, int D)
    {
        itsH=new Hamiltonian_1D_NN_Heisenberg(L,S2,1.0);
        itsMPO=itsH->CreateMPO();
        itsMPS=itsH->CreateMPS(D);
    }


    MatrixT GetW(int isite, int m, int n) {return itsMPO->GetSite(isite)->GetW(m,n);}

    Hamiltonian* itsH;
    MatrixProductOperator* itsMPO;
    MatrixProductState*    itsMPS;
    double eps;
};





TEST_F(MPOTesting,MakeHamiltonian)
{
    Setup(10,1,2);
}


TEST_F(MPOTesting,MPOMemebers)
{
    EXPECT_EQ(itsMPO->GetL(),10);
    EXPECT_EQ(itsMPO->GetD(),5);
    EXPECT_EQ(itsMPO->Getp(),2);
}

TEST_F(MPOTesting,HamiltonianGetLeftW00)
{
    Setup(10,1,2);
    EXPECT_EQ(ToString(itsH->GetW(Hamiltonian::Left,0,0)),"(1:1),(1:5) \n[ (0,0) (0,0) (0,0) (-0.5,0) (1,0) ]\n");
}

TEST_F(MPOTesting,HamiltonianGetRightW00)
{
    Setup(10,1,2);
    EXPECT_EQ(ToString(itsH->GetW(Hamiltonian::Right,0,0)),"(1:5),(1:1) \n[ (1,0) ]\n[ (0,0) ]\n[ (0,0) ]\n[ (-0.5,0) ]\n[ (0,0) ]\n");
}
TEST_F(MPOTesting,HamiltonianGetLeftW10)
{
    Setup(10,1,2);
    EXPECT_EQ(ToString(itsH->GetW(Hamiltonian::Left,1,0)),"(1:1),(1:5) \n[ (0,0) (0,0) (0.5,0) (0,0) (1,0) ]\n");
}
TEST_F(MPOTesting,HamiltonianGetRightW10)
{
    Setup(10,1,2);
    EXPECT_EQ(ToString(itsH->GetW(Hamiltonian::Right,1,0)),"(1:5),(1:1) \n[ (1,0) ]\n[ (1,0) ]\n[ (0,0) ]\n[ (0,0) ]\n[ (0,0) ]\n");
}
TEST_F(MPOTesting,HamiltonianGetLeftW01)
{
    Setup(10,1,2);
    EXPECT_EQ(ToString(itsH->GetW(Hamiltonian::Left,0,1)),"(1:1),(1:5) \n[ (0,0) (0.5,0) (0,0) (0,0) (1,0) ]\n");
}
TEST_F(MPOTesting,HamiltonianGetRightW01)
{
    Setup(10,1,2);
    EXPECT_EQ(ToString(itsH->GetW(Hamiltonian::Right,0,1)),"(1:5),(1:1) \n[ (1,0) ]\n[ (0,0) ]\n[ (1,0) ]\n[ (0,0) ]\n[ (0,0) ]\n");
}
TEST_F(MPOTesting,HamiltonianGetLeftW11)
{
    Setup(10,1,2);
    EXPECT_EQ(ToString(itsH->GetW(Hamiltonian::Left,1,1)),"(1:1),(1:5) \n[ (0,0) (0,0) (0,0) (0.5,0) (1,0) ]\n");
}
TEST_F(MPOTesting,HamiltonianGetRightW11)
{
    Setup(10,1,2);
    EXPECT_EQ(ToString(itsH->GetW(Hamiltonian::Right,1,1)),"(1:5),(1:1) \n[ (1,0) ]\n[ (0,0) ]\n[ (0,0) ]\n[ (0.5,0) ]\n[ (0,0) ]\n");
}

TEST_F(MPOTesting,CheckThatWsGotLoaded)
{
    Setup(10,1,2);
    EXPECT_EQ(ToString(GetW(1,0,0)),"(1:5),(1:5) \n[ (1,0) (0,0) (0,0) (0,0) (0,0) ]\n[ (0,0) (0,0) (0,0) (0,0) (0,0) ]\n[ (0,0) (0,0) (0,0) (0,0) (0,0) ]\n[ (-0.5,0) (0,0) (0,0) (0,0) (0,0) ]\n[ (0,0) (0,0) (0,0) (-0.5,0) (1,0) ]\n");
    EXPECT_EQ(ToString(GetW(1,0,1)),"(1:5),(1:5) \n[ (1,0) (0,0) (0,0) (0,0) (0,0) ]\n[ (0,0) (0,0) (0,0) (0,0) (0,0) ]\n[ (1,0) (0,0) (0,0) (0,0) (0,0) ]\n[ (0,0) (0,0) (0,0) (0,0) (0,0) ]\n[ (0,0) (0.5,0) (0,0) (0,0) (1,0) ]\n");
}

TEST_F(MPOTesting,DoHamiltionExpectationL10S1D2)
{
    Setup(10,1,2);
    itsMPS->InitializeWith(MatrixProductSite::Product);
    itsMPO->GetExpectation(itsMPS);
    EXPECT_NEAR(itsMPO->GetExpectation(itsMPS),576.0,1e-11);
}
TEST_F(MPOTesting,DoHamiltionExpectationProductL10S1D1)
{
    Setup(10,1,1);
    itsMPS->InitializeWith(MatrixProductSite::Product);
    itsMPO->GetExpectation(itsMPS);
    EXPECT_NEAR(itsMPO->GetExpectation(itsMPS),112.0,1e-11);
}
TEST_F(MPOTesting,DoHamiltionExpectationNeelL10S1D1)
{
    Setup(10,1,1);
    itsMPS->InitializeWith(MatrixProductSite::Neel);
    itsMPO->GetExpectation(itsMPS);
    EXPECT_NEAR(itsMPO->GetExpectation(itsMPS),-2.25,1e-11);
}

TEST_F(MPOTesting,LeftNormalizeThenDoHamiltionExpectation)
{
    Setup(10,1,2);
    itsMPS->InitializeWith(MatrixProductSite::Product);
    itsMPS->Normalize(MatrixProductSite::Left);
    itsMPO->GetExpectation(itsMPS);
    EXPECT_NEAR(itsMPO->GetExpectation(itsMPS),288.0,1e-11);
}
TEST_F(MPOTesting,RightNormalizeThenDoHamiltionExpectation)
{
    Setup(10,1,2);
    itsMPS->InitializeWith(MatrixProductSite::Product);
    itsMPS->Normalize(MatrixProductSite::Left);
    itsMPO->GetExpectation(itsMPS);
    EXPECT_NEAR(itsMPO->GetExpectation(itsMPS),288.0,1e-11);
}

TEST_F(MPOTesting,TestHeffWithProductState)
{
    Setup(10,1,2);
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

TEST_F(MPOTesting,TestHeffWithRandomStateL10S1D2)
{
    Setup(10,1,2);
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

TEST_F(MPOTesting,TestHeffWithRandomStateL10S1D1)
{
    Setup(10,1,1);
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

TEST_F(MPOTesting,TestHeffWithRandomStateL10S5D1)
{
    Setup(10,5,1);
    itsMPS->InitializeWith(MatrixProductSite::Random);
    itsMPS->Normalize(MatrixProductSite::Right);
    double E1=itsMPO->GetExpectation(itsMPS);
    for (int ia=0; ia<itsMPS->GetL(); ia++)
    {
        itsMPS->Normalize(ia);
        Matrix6T Heff=itsMPO->GetHeff(itsMPS,ia);
        double E2=itsMPS->ContractHeff(ia,Heff);
        EXPECT_NEAR(E1,E2,100000*eps);
        double E3=itsMPS->ContractHeff(ia,Heff.Flatten());
        EXPECT_NEAR(E1,E3,100000*eps);

        MatrixT HeffF=Heff.Flatten();
        MatrixT d=HeffF-Transpose(conj(HeffF));
        EXPECT_NEAR(Max(abs(d)),0.0,100000*eps);
    }
}


TEST_F(MPOTesting,TestGetLRIterateL10S1D2)
{
    int L=10;
    Setup(L,1,2);
    itsMPS->InitializeWith(MatrixProductSite::Random);
    itsMPS->Normalize(MatrixProductSite::Right);
    Vector3T L3=itsMPS->GetEOLeft_Iterate(itsMPO,L);
    eType EL=L3(1,1,1);
    Vector3T R3=itsMPS->GetEORightIterate(itsMPO,-1);
    eType ER=R3(1,1,1);
    EXPECT_NEAR(std::imag(EL),0.0,eps);
    EXPECT_NEAR(std::imag(ER),0.0,eps);
    EXPECT_NEAR(std::real(ER),std::real(EL),10*eps);
}

TEST_F(MPOTesting,TestGetLRIterateL10S1D1)
{
    int L=10;
    Setup(L,1,1);
    itsMPS->InitializeWith(MatrixProductSite::Random);
    itsMPS->Normalize(MatrixProductSite::Right);
    Vector3T L3=itsMPS->GetEOLeft_Iterate(itsMPO,L);
    eType EL=L3(1,1,1);
    Vector3T R3=itsMPS->GetEORightIterate(itsMPO,-1);
    eType ER=R3(1,1,1);
    EXPECT_NEAR(std::imag(EL),0.0,eps);
    EXPECT_NEAR(std::imag(ER),0.0,eps);
    EXPECT_NEAR(std::real(ER),std::real(EL),10*eps);
}

TEST_F(MPOTesting,TestGetLRIterateL10S1D6)
{
    int L=10;
    Setup(L,1,6);
    itsMPS->InitializeWith(MatrixProductSite::Random);
    itsMPS->Normalize(MatrixProductSite::Right);
    Vector3T L3=itsMPS->GetEOLeft_Iterate(itsMPO,L);
    eType EL=L3(1,1,1);
    Vector3T R3=itsMPS->GetEORightIterate(itsMPO,-1);
    eType ER=R3(1,1,1);
    EXPECT_NEAR(std::imag(EL),0.0,10*eps);
    EXPECT_NEAR(std::imag(ER),0.0,10*eps);
    EXPECT_NEAR(std::real(ER),std::real(EL),10*eps);
}
TEST_F(MPOTesting,TestGetLRIterateL10S5D2)
{
    int L=10;
    Setup(L,5,2);
    itsMPS->InitializeWith(MatrixProductSite::Random);
    itsMPS->Normalize(MatrixProductSite::Right);
    Vector3T L3=itsMPS->GetEOLeft_Iterate(itsMPO,L);
    eType EL=L3(1,1,1);
    Vector3T R3=itsMPS->GetEORightIterate(itsMPO,-1);
    eType ER=R3(1,1,1);
    EXPECT_NEAR(std::imag(EL),0.0,100000*eps);
    EXPECT_NEAR(std::imag(ER),0.0,100000*eps);
    EXPECT_NEAR(std::real(ER),std::real(EL),100000*eps);
}

TEST_F(MPOTesting,TestEold)
{
    int L=10;
    Setup(L,1,2);
    itsMPS->InitializeWith(MatrixProductSite::Random);
    itsMPS->Normalize(MatrixProductSite::Right);
    Vector3T L3=itsMPS->GetEOLeft_Iterate(itsMPO,L);
    eType EL=L3(1,1,1);
    Vector3T R3=itsMPS->GetEORightIterate(itsMPO,-1);
    eType ER=R3(1,1,1);
    double Eold=itsMPO->GetExpectation(itsMPS);
    EXPECT_NEAR(std::real(ER),Eold,100*eps);
    EXPECT_NEAR(std::real(EL),Eold,100*eps);
//    Matrix6T Heff=itsMPS->GetHeffIterate(itsMPO,1);
//    cout << "Heff=" << Heff <<endl;
}

TEST_F(MPOTesting,TestHeff)
{
    int L=10;
    Setup(L,1,2);
    itsMPS->InitializeWith(MatrixProductSite::Random);
    itsMPS->Normalize(MatrixProductSite::Right);
    for (int ia=0;ia<L;ia++)
    {
        Matrix6T HeffI=itsMPS->GetHeffIterate(itsMPO,ia);
    //    cout << "HeffI=" << HeffI <<endl;
        Matrix6T HeffO=itsMPO->GetHeff(itsMPS,ia);
    //    cout << "HeffO=" << HeffO <<endl;
        double error=Max(abs(HeffI.Flatten()-HeffO.Flatten()));
        EXPECT_NEAR(error,0,10*eps);
    }
}

