#include "Tests.H"
#include "TensorNetworks/Hamiltonian.H"
#include "TensorNetworks/OperatorWRepresentation.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/MPO.H"
#include "TensorNetworks/Factory.H"

#include "oml/stream.h"
#include "oml/vector_io.h"
#include "oml/stopw.h"

class MPOTesting : public ::testing::Test
{
public:
    typedef TensorNetworks::Matrix6CT Matrix6CT;
    typedef TensorNetworks::MatrixRT  MatrixRT;
    typedef TensorNetworks::MatrixCT  MatrixCT;
    typedef TensorNetworks::Vector3CT Vector3CT;
    typedef TensorNetworks::eType     eType;
    MPOTesting()
        : eps(1.0e-13)
        , itsFactory(TensorNetworks::Factory::GetFactory())

    {
        assert(itsFactory);
        StreamableObject::SetToPretty();

    }
    void Setup(int L, double S, int D)
    {
        itsH=itsFactory->Make1D_NN_HeisenbergHamiltonian(L,S,1.0,1.0,0.0);
        itsWRep=dynamic_cast<TensorNetworks::OperatorWRepresentation*>(itsH);
        itsMPS=itsH->CreateMPS(D);
    }
    double ENeel(double S) const;
    Matrix6CT GetHeffIterate(int isite) const {return GetMPSImp()->GetHeffIterate(itsH,isite);}
    Vector3CT CalcHeffLeft(int isite,bool cache=false) const {return GetMPSImp()->CalcHeffLeft (itsH,isite,cache);}
    Vector3CT CalcHeffRight(int isite,bool cache=false) const {return GetMPSImp()->CalcHeffRight(itsH,isite,cache);}
    void LoadHeffCaches() {GetMPSImp()->LoadHeffCaches(itsH);}

    MatrixRT GetW(int isite, int m, int n) {return itsH->GetSiteOperator(isite)->GetW(m,n);}

          TensorNetworks::MPSImp* GetMPSImp()       {return dynamic_cast<      TensorNetworks::MPSImp*>(itsMPS);}
    const TensorNetworks::MPSImp* GetMPSImp() const {return dynamic_cast<const TensorNetworks::MPSImp*>(itsMPS);}

    double eps;
    TensorNetworks::Factory*                 itsFactory=TensorNetworks::Factory::GetFactory();
    TensorNetworks::Hamiltonian*             itsH;
    TensorNetworks::OperatorWRepresentation* itsWRep;
    TensorNetworks::MPS*                     itsMPS;
};





TEST_F(MPOTesting,MakeHamiltonian)
{
    Setup(10,0.5,2);
}


TEST_F(MPOTesting,HamiltonianGetLeftW00)
{
    Setup(10,0.5,2);
    EXPECT_EQ(ToString(itsWRep->GetW(TensorNetworks::PLeft,0,0)),"(1:1),(1:5) \n[ -0 0 0 -0.5 1 ]\n");
}

TEST_F(MPOTesting,HamiltonianGetRightW00)
{
    Setup(10,0.5,2);
    EXPECT_EQ(ToString(itsWRep->GetW(TensorNetworks::PRight,0,0)),"(1:5),(1:1) \n[ 1 ]\n[ 0 ]\n[ 0 ]\n[ -0.5 ]\n[ -0 ]\n");
}
TEST_F(MPOTesting,HamiltonianGetLeftW10)
{
    Setup(10,0.5,2);
    EXPECT_EQ(ToString(itsWRep->GetW(TensorNetworks::PLeft,1,0)),"(1:1),(1:5) \n[ 0 0 0.5 0 0 ]\n");
}
TEST_F(MPOTesting,HamiltonianGetRightW10)
{
    Setup(10,0.5,2);
    EXPECT_EQ(ToString(itsWRep->GetW(TensorNetworks::PRight,1,0)),"(1:5),(1:1) \n[ 0 ]\n[ 1 ]\n[ 0 ]\n[ 0 ]\n[ 0 ]\n");
}
TEST_F(MPOTesting,HamiltonianGetLeftW01)
{
    Setup(10,0.5,2);
    EXPECT_EQ(ToString(itsWRep->GetW(TensorNetworks::PLeft,0,1)),"(1:1),(1:5) \n[ 0 0.5 0 0 0 ]\n");
}
TEST_F(MPOTesting,HamiltonianGetRightW01)
{
    Setup(10,0.5,2);
    EXPECT_EQ(ToString(itsWRep->GetW(TensorNetworks::PRight,0,1)),"(1:5),(1:1) \n[ 0 ]\n[ 0 ]\n[ 1 ]\n[ 0 ]\n[ 0 ]\n");
}
TEST_F(MPOTesting,HamiltonianGetLeftW11)
{
    Setup(10,0.5,2);
    EXPECT_EQ(ToString(itsWRep->GetW(TensorNetworks::PLeft,1,1)),"(1:1),(1:5) \n[ 0 0 0 0.5 1 ]\n");
}
TEST_F(MPOTesting,HamiltonianGetRightW11)
{
    Setup(10,0.5,2);
    EXPECT_EQ(ToString(itsWRep->GetW(TensorNetworks::PRight,1,1)),"(1:5),(1:1) \n[ 1 ]\n[ 0 ]\n[ 0 ]\n[ 0.5 ]\n[ 0 ]\n");
}

TEST_F(MPOTesting,CheckThatWsGotLoaded)
{
    Setup(10,0.5,2);
    EXPECT_EQ(ToString(GetW(2,0,0)),"(1:5),(1:5) \n[ 1 0 0 0 0 ]\n[ 0 0 0 0 0 ]\n[ 0 0 0 0 0 ]\n[ -0.5 0 0 0 0 ]\n[ -0 0 0 -0.5 1 ]\n");
    EXPECT_EQ(ToString(GetW(2,0,1)),"(1:5),(1:5) \n[ 0 0 0 0 0 ]\n[ 0 0 0 0 0 ]\n[ 1 0 0 0 0 ]\n[ 0 0 0 0 0 ]\n[ 0 0.5 0 0 0 ]\n");
}

double MPOTesting::ENeel(double s) const
{
    return -s*s*(itsH->GetL()-1);
}

TEST_F(MPOTesting,DoHamiltionExpectationL10S1_5D2)
{
    for (double S=0.5;S<=2.5;S+=0.5)
    {
        Setup(10,S,2);
        itsMPS->InitializeWith(TensorNetworks::Neel);
        EXPECT_NEAR(itsMPS->GetExpectation(itsH),ENeel(S),1e-11);
    }
}

TEST_F(MPOTesting,DoHamiltionExpectationProductL10S1D1)
{
    Setup(10,0.5,1);
    itsMPS->InitializeWith(TensorNetworks::Neel);
    EXPECT_NEAR(itsMPS->GetExpectation(itsH),-2.25,1e-11);
}
TEST_F(MPOTesting,DoHamiltionExpectationNeelL10S1D1)
{
    Setup(10,0.5,1);
    itsMPS->InitializeWith(TensorNetworks::Neel);
    EXPECT_NEAR(itsMPS->GetExpectation(itsH),-2.25,1e-11);
}

TEST_F(MPOTesting,LeftNormalizeThenDoHamiltionExpectation)
{
    Setup(10,0.5,2);
    itsMPS->InitializeWith(TensorNetworks::Neel);
    itsMPS->Normalize(TensorNetworks::DLeft);
    EXPECT_NEAR(itsMPS->GetExpectation(itsH),-2.25,1e-11);
}
TEST_F(MPOTesting,RightNormalizeThenDoHamiltionExpectation)
{
    Setup(10,0.5,2);
    itsMPS->InitializeWith(TensorNetworks::Neel);
    itsMPS->Normalize(TensorNetworks::DLeft);
    EXPECT_NEAR(itsMPS->GetExpectation(itsH),-2.25,1e-11);
}



TEST_F(MPOTesting,TestGetLRIterateL10S1D2)
{
    int L=10;
    Setup(L,0.5,2);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::DRight);
    Vector3CT L3=CalcHeffLeft(L+1);
    eType EL=L3(1,1,1);
    Vector3CT R3=CalcHeffRight(0);
    eType ER=R3(1,1,1);
    EXPECT_NEAR(std::imag(EL),0.0,eps);
    EXPECT_NEAR(std::imag(ER),0.0,eps);
    EXPECT_NEAR(std::real(ER),std::real(EL),10*eps);
}

TEST_F(MPOTesting,TestGetLRIterateL10S1D1)
{
    int L=10;
    Setup(L,0.5,1);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::DRight);
    Vector3CT L3=CalcHeffLeft(L+1);
    eType EL=L3(1,1,1);
    Vector3CT R3=CalcHeffRight(0);
    eType ER=R3(1,1,1);
    EXPECT_NEAR(std::imag(EL),0.0,eps);
    EXPECT_NEAR(std::imag(ER),0.0,eps);
    EXPECT_NEAR(std::real(ER),std::real(EL),10*eps);
}

TEST_F(MPOTesting,TestGetLRIterateL10S1D6)
{
    int L=10;
    Setup(L,0.5,6);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::DRight);
    Vector3CT L3=CalcHeffLeft(L+1);
    eType EL=L3(1,1,1);
    Vector3CT R3=CalcHeffRight(0);
    eType ER=R3(1,1,1);
    EXPECT_NEAR(std::imag(EL),0.0,10*eps);
    EXPECT_NEAR(std::imag(ER),0.0,10*eps);
    EXPECT_NEAR(std::real(ER),std::real(EL),10*eps);
}
TEST_F(MPOTesting,TestGetLRIterateL10S5D2)
{
    int L=10;
    Setup(L,2.5,2);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::DRight);
    Vector3CT L3=CalcHeffLeft(L+1);
    eType EL=L3(1,1,1);
    Vector3CT R3=CalcHeffRight(0);
    eType ER=R3(1,1,1);
    EXPECT_NEAR(std::imag(EL),0.0,100000*eps);
    EXPECT_NEAR(std::imag(ER),0.0,100000*eps);
    EXPECT_NEAR(std::real(ER),std::real(EL),100000*eps);
}

TEST_F(MPOTesting,TestEoldEnew)
{
    int L=10;
    Setup(L,0.5,2);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::DRight);
    Vector3CT L3=CalcHeffLeft(L+1);
    eType EL=L3(1,1,1);
    Vector3CT R3=CalcHeffRight(0);
    eType ER=R3(1,1,1);
    double Enew=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(std::real(ER),Enew,100*eps);
    EXPECT_NEAR(std::real(EL),Enew,100*eps);
}

TEST_F(MPOTesting,TestMPOCombineForH2)
{
    int L=10,D=8;
    Setup(L,0.5,D);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::DRight);
    int Dw=itsH->GetMaxDw();
    TensorNetworks::MPO* H1=itsH->CreateUnitOperator();
    H1->Combine(itsH);
    TensorNetworks::MPO* H2=itsH->CreateUnitOperator();
    H2->Combine(itsH);
    H2->Combine(itsH);
    EXPECT_EQ(H1->GetMaxDw(),Dw);
    EXPECT_EQ(H2->GetMaxDw(),Dw*Dw);
    delete H1;
    delete H2;
}

TEST_F(MPOTesting,TestMPOCompressForH2)
{
    int L=10,D=8;
    Setup(L,0.5,D);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::DRight);
    TensorNetworks::MPO* H1=itsH->CreateUnitOperator();
    H1->Combine(itsH);
    TensorNetworks::MPO* H2=itsH->CreateUnitOperator();
    H2->Combine(itsH);
    H2->Combine(itsH);
    H2->Compress(0,1e-13);
    EXPECT_EQ(H2->GetMaxDw(),9);
    delete H1;
    delete H2;
}

TEST_F(MPOTesting,TestHamiltonianCreateH2)
{
    int L=10,D=8;
    Setup(L,0.5,D);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::DRight);
    TensorNetworks::MPO* H2=itsH->CreateH2Operator();
    EXPECT_EQ(H2->GetMaxDw(),9);
    delete H2;
}

TEST_F(MPOTesting,TestMPOCompressForE2)
{
    int L=10,D=8;
    Setup(L,0.5,D);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::DRight);
    TensorNetworks::MPO* H1=itsH->CreateUnitOperator();
    H1->Combine(itsH);
    TensorNetworks::MPO* H2=itsH->CreateUnitOperator();
    H2->Combine(itsH);
    H2->Combine(itsH);
    double E2a=itsMPS->GetExpectation(H2);
    H2->Compress(0,1e-13);
    EXPECT_EQ(H2->GetMaxDw(),9);
    double E2b=itsMPS->GetExpectation(H2);
    EXPECT_NEAR(E2a,E2b,1e-13);
    delete H1;
    delete H2;
}

#ifndef DEBUG

TEST_F(MPOTesting,TestTimingE2_S5D4)
{
    int L=10,D=4;
    double S=2.5;
    Setup(L,S,D);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::DRight);
    StopWatch sw;
    sw.Start();
    TensorNetworks::MPO* H2=itsH->CreateH2Operator();
    double EE=itsMPS->GetExpectation(H2);
    delete H2;
    sw.Stop();
    cout << "<E^2> contraction for L=" << L << ", S=" << S << ", D=" << D << " took " << sw.GetTime() << " seconds." << endl;
    (void)EE; //Avoid warning
}
TEST_F(MPOTesting,TestTimingE2_S1D16)
{
    int L=10,D=16;
    double S=2.5;
    Setup(L,S,D);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::DRight);
    StopWatch sw;
    sw.Start();
    TensorNetworks::MPO* H2=itsH->CreateH2Operator();
    double EE=itsMPS->GetExpectation(H2);
    delete H2;
    sw.Stop();
    cout << "<E^2> contraction for L=" << L << ", S=" << S << ", D=" << D << " took " << sw.GetTime() << " seconds." << endl;
    (void)EE; //Avoid warning
}
#endif



