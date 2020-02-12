#include "Tests.H"
#include "TensorNetworks/Hamiltonian.H"
#include "TensorNetworks/OperatorWRepresentation.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/Factory.H"
#include "TensorNetworks/LRPSupervisor.H"
#include "TensorNetworks/Epsilons.H"

#include "oml/stream.h"
#include "oml/vector_io.h"
#include "oml/stopw.h"

class MPOTesting : public ::testing::Test
{
public:
    typedef TensorNetworks::Matrix6T Matrix6T;
    typedef TensorNetworks::MatrixT  MatrixT;
    typedef TensorNetworks::MatrixCT MatrixCT;
    typedef TensorNetworks::Vector3T Vector3T;
    typedef TensorNetworks::eType eType;
    MPOTesting()
        : eps(1.0e-13)
        , itsFactory(TensorNetworks::Factory::GetFactory())
        , itsSupervisor( new LRPSupervisor())
        , itsEps()

    {
        assert(itsFactory);
        StreamableObject::SetToPretty();

    }
    void Setup(int L, int S2, int D)
    {
        itsH=itsFactory->Make1D_NN_HeisenbergHamiltonian(L,S2/2.0,1.0,1.0,0.0);
        itsWRep=dynamic_cast<OperatorWRepresentation*>(itsH);
        itsMPS=itsH->CreateMPS(D,itsEps);
    }
    double ENeel(int S2) const;
    //Matrix6T GetHeff(int isite) const {return GetMPSImp()->GetHeff(itsH,isite);}
    Matrix6T GetHeffIterate(int isite) const {return GetMPSImp()->GetHeffIterate(itsH,isite);}
//    double ContractHeff(int isite,const Matrix6T& Heff) const{return GetMPSImp()->itsSites[isite]->ContractHeff(Heff);}
//    double ContractHeff(int isite,const MatrixCT& Heff) const{return GetMPSImp()->itsSites[isite]->ContractHeff(Heff);}
    Vector3T GetEOLeft_Iterate(int isite,bool cache=false) const {return GetMPSImp()->GetEOLeft_Iterate(itsH,itsSupervisor,isite,cache);}
    Vector3T GetEORightIterate(int isite,bool cache=false) const {return GetMPSImp()->GetEORightIterate(itsH,itsSupervisor,isite,cache);}
    void LoadHeffCaches() {GetMPSImp()->LoadHeffCaches(itsH,itsSupervisor);}

    MatrixT GetW(int isite, int m, int n) {return itsH->GetSiteOperator(isite)->GetW(m,n);}

          MatrixProductStateImp* GetMPSImp()       {return dynamic_cast<      MatrixProductStateImp*>(itsMPS);}
    const MatrixProductStateImp* GetMPSImp() const {return dynamic_cast<const MatrixProductStateImp*>(itsMPS);}

    double eps;
    const TensorNetworks::Factory* itsFactory=TensorNetworks::Factory::GetFactory();
    Hamiltonian* itsH;
    OperatorWRepresentation* itsWRep;
    MatrixProductState*    itsMPS;
    LRPSupervisor* itsSupervisor;
    Epsilons             itsEps;
};





TEST_F(MPOTesting,MakeHamiltonian)
{
    Setup(10,1,2);
}


TEST_F(MPOTesting,HamiltonianGetLeftW00)
{
    Setup(10,1,2);
    EXPECT_EQ(ToString(itsWRep->GetW(TensorNetworks::Left,0,0)),"(1:1),(1:5) \n[ -0 0 0 -0.5 1 ]\n");
}

TEST_F(MPOTesting,HamiltonianGetRightW00)
{
    Setup(10,1,2);
    EXPECT_EQ(ToString(itsWRep->GetW(TensorNetworks::Right,0,0)),"(1:5),(1:1) \n[ 1 ]\n[ 0 ]\n[ 0 ]\n[ -0.5 ]\n[ -0 ]\n");
}
TEST_F(MPOTesting,HamiltonianGetLeftW10)
{
    Setup(10,1,2);
    EXPECT_EQ(ToString(itsWRep->GetW(TensorNetworks::Left,1,0)),"(1:1),(1:5) \n[ 0 0 0.5 0 0 ]\n");
}
TEST_F(MPOTesting,HamiltonianGetRightW10)
{
    Setup(10,1,2);
    EXPECT_EQ(ToString(itsWRep->GetW(TensorNetworks::Right,1,0)),"(1:5),(1:1) \n[ 0 ]\n[ 1 ]\n[ 0 ]\n[ 0 ]\n[ 0 ]\n");
}
TEST_F(MPOTesting,HamiltonianGetLeftW01)
{
    Setup(10,1,2);
    EXPECT_EQ(ToString(itsWRep->GetW(TensorNetworks::Left,0,1)),"(1:1),(1:5) \n[ 0 0.5 0 0 0 ]\n");
}
TEST_F(MPOTesting,HamiltonianGetRightW01)
{
    Setup(10,1,2);
    EXPECT_EQ(ToString(itsWRep->GetW(TensorNetworks::Right,0,1)),"(1:5),(1:1) \n[ 0 ]\n[ 0 ]\n[ 1 ]\n[ 0 ]\n[ 0 ]\n");
}
TEST_F(MPOTesting,HamiltonianGetLeftW11)
{
    Setup(10,1,2);
    EXPECT_EQ(ToString(itsWRep->GetW(TensorNetworks::Left,1,1)),"(1:1),(1:5) \n[ 0 0 0 0.5 1 ]\n");
}
TEST_F(MPOTesting,HamiltonianGetRightW11)
{
    Setup(10,1,2);
    EXPECT_EQ(ToString(itsWRep->GetW(TensorNetworks::Right,1,1)),"(1:5),(1:1) \n[ 1 ]\n[ 0 ]\n[ 0 ]\n[ 0.5 ]\n[ 0 ]\n");
}

TEST_F(MPOTesting,CheckThatWsGotLoaded)
{
    Setup(10,1,2);
    EXPECT_EQ(ToString(GetW(1,0,0)),"(1:5),(1:5) \n[ 1 0 0 0 0 ]\n[ 0 0 0 0 0 ]\n[ 0 0 0 0 0 ]\n[ -0.5 0 0 0 0 ]\n[ -0 0 0 -0.5 1 ]\n");
    EXPECT_EQ(ToString(GetW(1,0,1)),"(1:5),(1:5) \n[ 0 0 0 0 0 ]\n[ 0 0 0 0 0 ]\n[ 1 0 0 0 0 ]\n[ 0 0 0 0 0 ]\n[ 0 0.5 0 0 0 ]\n");
}

double MPOTesting::ENeel(int S2) const
{
    double s=0.5*S2;
    return -s*s*(itsH->GetL()-1);
}

TEST_F(MPOTesting,DoHamiltionExpectationL10S1_5D2)
{
    for (int S2=1;S2<=5;S2++)
    {
        Setup(10,S2,2);
        itsMPS->InitializeWith(TensorNetworks::Neel);
        EXPECT_NEAR(itsMPS->GetExpectation(itsH),ENeel(S2),1e-11);
    }
}

TEST_F(MPOTesting,DoHamiltionExpectationProductL10S1D1)
{
    Setup(10,1,1);
    itsMPS->InitializeWith(TensorNetworks::Neel);
    EXPECT_NEAR(itsMPS->GetExpectation(itsH),-2.25,1e-11);
}
TEST_F(MPOTesting,DoHamiltionExpectationNeelL10S1D1)
{
    Setup(10,1,1);
    itsMPS->InitializeWith(TensorNetworks::Neel);
    EXPECT_NEAR(itsMPS->GetExpectation(itsH),-2.25,1e-11);
}

TEST_F(MPOTesting,LeftNormalizeThenDoHamiltionExpectation)
{
    Setup(10,1,2);
    itsMPS->InitializeWith(TensorNetworks::Neel);
    itsMPS->Normalize(TensorNetworks::Left,itsSupervisor);
    EXPECT_NEAR(itsMPS->GetExpectation(itsH),-2.25,1e-11);
}
TEST_F(MPOTesting,RightNormalizeThenDoHamiltionExpectation)
{
    Setup(10,1,2);
    itsMPS->InitializeWith(TensorNetworks::Neel);
    itsMPS->Normalize(TensorNetworks::Left,itsSupervisor);
    EXPECT_NEAR(itsMPS->GetExpectation(itsH),-2.25,1e-11);
}



TEST_F(MPOTesting,TestGetLRIterateL10S1D2)
{
    int L=10;
    Setup(L,1,2);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::Right,itsSupervisor);
    Vector3T L3=GetEOLeft_Iterate(L);
    eType EL=L3(1,1,1);
    Vector3T R3=GetEORightIterate(-1);
    eType ER=R3(1,1,1);
    EXPECT_NEAR(std::imag(EL),0.0,eps);
    EXPECT_NEAR(std::imag(ER),0.0,eps);
    EXPECT_NEAR(std::real(ER),std::real(EL),10*eps);
}

TEST_F(MPOTesting,TestGetLRIterateL10S1D1)
{
    int L=10;
    Setup(L,1,1);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::Right,itsSupervisor);
    Vector3T L3=GetEOLeft_Iterate(L);
    eType EL=L3(1,1,1);
    Vector3T R3=GetEORightIterate(-1);
    eType ER=R3(1,1,1);
    EXPECT_NEAR(std::imag(EL),0.0,eps);
    EXPECT_NEAR(std::imag(ER),0.0,eps);
    EXPECT_NEAR(std::real(ER),std::real(EL),10*eps);
}

TEST_F(MPOTesting,TestGetLRIterateL10S1D6)
{
    int L=10;
    Setup(L,1,6);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::Right,itsSupervisor);
    Vector3T L3=GetEOLeft_Iterate(L);
    eType EL=L3(1,1,1);
    Vector3T R3=GetEORightIterate(-1);
    eType ER=R3(1,1,1);
    EXPECT_NEAR(std::imag(EL),0.0,10*eps);
    EXPECT_NEAR(std::imag(ER),0.0,10*eps);
    EXPECT_NEAR(std::real(ER),std::real(EL),10*eps);
}
TEST_F(MPOTesting,TestGetLRIterateL10S5D2)
{
    int L=10;
    Setup(L,5,2);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::Right,itsSupervisor);
    Vector3T L3=GetEOLeft_Iterate(L);
    eType EL=L3(1,1,1);
    Vector3T R3=GetEORightIterate(-1);
    eType ER=R3(1,1,1);
    EXPECT_NEAR(std::imag(EL),0.0,100000*eps);
    EXPECT_NEAR(std::imag(ER),0.0,100000*eps);
    EXPECT_NEAR(std::real(ER),std::real(EL),100000*eps);
}

TEST_F(MPOTesting,TestEoldEnew)
{
    int L=10;
    Setup(L,1,2);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::Right,itsSupervisor);
    Vector3T L3=GetEOLeft_Iterate(L);
    eType EL=L3(1,1,1);
    Vector3T R3=GetEORightIterate(-1);
    eType ER=R3(1,1,1);
    double Enew=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(std::real(ER),Enew,100*eps);
    EXPECT_NEAR(std::real(EL),Enew,100*eps);
}


TEST_F(MPOTesting,TestGetExpectation2_I_I)
{
    int L=10,S2=1,D=2;
    Setup(L,S2,D);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::Right,itsSupervisor);
    OperatorWRepresentation* IWO=itsFactory->MakeIdentityOperator();
    Operator* IO=itsH->CreateOperator(IWO);

    double E1=itsMPS->GetExpectation(itsH);
    double I1=itsMPS->GetExpectation(IO);
    double II=itsMPS->GetExpectation(IO,IO);
    double IE=itsMPS->GetExpectation(IO,itsH);
    double EI=itsMPS->GetExpectation(itsH,IO);
    double EE=itsMPS->GetExpectation(itsH,itsH);
    EXPECT_NEAR(I1,1.0,eps);
    EXPECT_NEAR(II,1.0,eps);
    EXPECT_NEAR(IE,E1,eps);
    EXPECT_NEAR(EI,E1,eps);
    (void)EE; //Nothing to test this agains right now
}

#ifndef DEBUG

TEST_F(MPOTesting,TestTimingE2_S5D4)
{
    int L=10,S2=5,D=4;
    Setup(L,S2,D);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::Right,itsSupervisor);
    StopWatch sw;
    sw.Start();
    double EE=itsMPS->GetExpectation(itsH,itsH);
    sw.Stop();
    cout << "<E^2> contraction for L=" << L << ", S=" << S2/2.0 << ", D=" << D << " took " << sw.GetTime() << " seconds." << endl;
    (void)EE; //Avoid warning
}
TEST_F(MPOTesting,TestTimingE2_S1D16)
{
    int L=10,S2=1,D=16;
    Setup(L,S2,D);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::Right,itsSupervisor);
    StopWatch sw;
    sw.Start();
    double EE=itsMPS->GetExpectation(itsH,itsH);
    sw.Stop();
    cout << "<E^2> contraction for L=" << L << ", S=" << S2/2.0 << ", D=" << D << " took " << sw.GetTime() << " seconds." << endl;
    (void)EE; //Avoid warning
}
#endif



