#include "Tests.H"
#include "TensorNetworksImp/MatrixProductOperator.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworksImp/Hamiltonian_1D_NN_Heisenberg.H"
#include "TensorNetworksImp/IdentityOperator.H"
#include "oml/stream.h"
#include "oml/vector_io.h"
#include <complex>

typedef TensorNetworks::Matrix6T Matrix6T;
class MPOTesting : public ::testing::Test
{
public:
    typedef TensorNetworks::MatrixT  MatrixT;
    typedef TensorNetworks::MatrixCT MatrixCT;
    typedef TensorNetworks::Vector3T Vector3T;
    typedef TensorNetworks::eType eType;
    MPOTesting()
    : eps(1.0e-13)
    {
        StreamableObject::SetToPretty();

    }
    void Setup(int L, int S2, int D)
    {
        Hamiltonian_1D_NN_Heisenberg* HH=new Hamiltonian_1D_NN_Heisenberg(L,S2,1.0);
        itsH=HH;
        itsWRep=HH;
        itsMPS=itsH->CreateMPS(D);
    }
    double ENeel(int S2) const;
    Matrix6T GetHeff(int isite) const {return GetMPSImp()->GetHeff(itsH,isite);}
    Matrix6T GetHeffIterate(int isite) const {return GetMPSImp()->GetHeffIterate(itsH,isite);}
    double ContractHeff(int isite,const Matrix6T& Heff) const{return GetMPSImp()->itsSites[isite]->ContractHeff(Heff);}
    double ContractHeff(int isite,const MatrixCT& Heff) const{return GetMPSImp()->itsSites[isite]->ContractHeff(Heff);}
    Vector3T GetEOLeft_Iterate(int isite,bool cache=false) const {return GetMPSImp()->GetEOLeft_Iterate(itsH,isite,cache);}
    Vector3T GetEORightIterate(int isite,bool cache=false) const {return GetMPSImp()->GetEORightIterate(itsH,isite,cache);}
    void LoadHeffCaches() {GetMPSImp()->LoadHeffCaches(itsH);}

    MatrixT GetW(int isite, int m, int n) {return itsH->GetSiteOperator(isite)->GetW(m,n);}

          MatrixProductStateImp* GetMPSImp()       {return dynamic_cast<      MatrixProductStateImp*>(itsMPS);}
    const MatrixProductStateImp* GetMPSImp() const {return dynamic_cast<const MatrixProductStateImp*>(itsMPS);}

    Hamiltonian* itsH;
    OperatorWRepresentation* itsWRep;
    MatrixProductState*    itsMPS;
    double eps;
};





TEST_F(MPOTesting,MakeHamiltonian)
{
    Setup(10,1,2);
}


TEST_F(MPOTesting,HamiltonianGetLeftW00)
{
    Setup(10,1,2);
    EXPECT_EQ(ToString(itsWRep->GetW(TensorNetworks::Left,0,0)),"(1:1),(1:5) \n[ 0 0 0 -0.5 1 ]\n");
}

TEST_F(MPOTesting,HamiltonianGetRightW00)
{
    Setup(10,1,2);
    EXPECT_EQ(ToString(itsWRep->GetW(TensorNetworks::Right,0,0)),"(1:5),(1:1) \n[ 1 ]\n[ 0 ]\n[ 0 ]\n[ -0.5 ]\n[ 0 ]\n");
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
    EXPECT_EQ(ToString(GetW(1,0,0)),"(1:5),(1:5) \n[ 1 0 0 0 0 ]\n[ 0 0 0 0 0 ]\n[ 0 0 0 0 0 ]\n[ -0.5 0 0 0 0 ]\n[ 0 0 0 -0.5 1 ]\n");
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
        itsMPS->GetExpectation(itsH);
        EXPECT_NEAR(itsMPS->GetExpectation(itsH),ENeel(S2),1e-11);
    }
}

TEST_F(MPOTesting,DoHamiltionExpectationProductL10S1D1)
{
    Setup(10,1,1);
    itsMPS->InitializeWith(TensorNetworks::Neel);
    itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(itsMPS->GetExpectation(itsH),-2.25,1e-11);
}
TEST_F(MPOTesting,DoHamiltionExpectationNeelL10S1D1)
{
    Setup(10,1,1);
    itsMPS->InitializeWith(TensorNetworks::Neel);
    itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(itsMPS->GetExpectation(itsH),-2.25,1e-11);
}

TEST_F(MPOTesting,LeftNormalizeThenDoHamiltionExpectation)
{
    Setup(10,1,2);
    itsMPS->InitializeWith(TensorNetworks::Neel);
    itsMPS->Normalize(TensorNetworks::Left);
    itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(itsMPS->GetExpectation(itsH),-2.25,1e-11);
}
TEST_F(MPOTesting,RightNormalizeThenDoHamiltionExpectation)
{
    Setup(10,1,2);
    itsMPS->InitializeWith(TensorNetworks::Neel);
    itsMPS->Normalize(TensorNetworks::Left);
    itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(itsMPS->GetExpectation(itsH),-2.25,1e-11);
}

TEST_F(MPOTesting,TestHeffWithProductState)
{
    Setup(10,1,2);
//    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->InitializeWith(TensorNetworks::Neel);
    for (int ia=0; ia<itsH->GetL(); ia++)
    {
        itsMPS->Normalize(ia);
        Matrix6T Heff=GetHeff(ia);
 //       cout << "E(" << ia << ")=" << itsMPS->ConstractHeff(ia,Heff) << endl;
        EXPECT_NEAR(ContractHeff(ia,Heff),-2.25,1e-11);
        MatrixCT HeffF=Heff.Flatten();
        //cout << "Heff=" << Heff << endl;
        MatrixCT d=HeffF-Transpose(conj(HeffF));
        EXPECT_NEAR(Max(abs(d)),0.0,1e-11);

//    cout << "d=" << d << endl;
    }
}

TEST_F(MPOTesting,TestHeffWithRandomStateL10S1D2)
{
    Setup(10,1,2);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::Right);
    double E1=itsMPS->GetExpectation(itsH);
    for (int ia=0; ia<itsH->GetL(); ia++)
    {
        itsMPS->Normalize(ia);
        Matrix6T Heff=GetHeff(ia);
        double E2=ContractHeff(ia,Heff);
        EXPECT_NEAR(E1,E2,100*eps);
        double E3=ContractHeff(ia,Heff.Flatten());
        EXPECT_NEAR(E1,E3,100*eps);

        MatrixCT HeffF=Heff.Flatten();
        MatrixCT d=HeffF-Transpose(conj(HeffF));
        EXPECT_NEAR(Max(abs(d)),0.0,100*eps);
    }
}

TEST_F(MPOTesting,TestHeffWithRandomStateL10S1D1)
{
    Setup(10,1,1);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::Right);
    double E1=itsMPS->GetExpectation(itsH);
    for (int ia=0; ia<itsH->GetL(); ia++)
    {
        itsMPS->Normalize(ia);
        Matrix6T Heff=GetHeff(ia);
        double E2=ContractHeff(ia,Heff);
        EXPECT_NEAR(E1,E2,100*eps);
        double E3=ContractHeff(ia,Heff.Flatten());
        EXPECT_NEAR(E1,E3,100*eps);

        MatrixCT HeffF=Heff.Flatten();
        MatrixCT d=HeffF-Transpose(conj(HeffF));
        EXPECT_NEAR(Max(abs(d)),0.0,100*eps);
    }
}

TEST_F(MPOTesting,TestHeffWithRandomStateL10S5D1)
{
    Setup(10,5,1);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::Right);
    double E1=itsMPS->GetExpectation(itsH);
    for (int ia=0; ia<itsH->GetL(); ia++)
    {
        itsMPS->Normalize(ia);
        Matrix6T Heff=GetHeff(ia);
        double E2=ContractHeff(ia,Heff);
        EXPECT_NEAR(E1,E2,100000*eps);
        double E3=ContractHeff(ia,Heff.Flatten());
        EXPECT_NEAR(E1,E3,100000*eps);

        MatrixCT HeffF=Heff.Flatten();
        MatrixCT d=HeffF-Transpose(conj(HeffF));
        EXPECT_NEAR(Max(abs(d)),0.0,100000*eps);
    }
}


TEST_F(MPOTesting,TestGetLRIterateL10S1D2)
{
    int L=10;
    Setup(L,1,2);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::Right);
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
    itsMPS->Normalize(TensorNetworks::Right);
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
    itsMPS->Normalize(TensorNetworks::Right);
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
    itsMPS->Normalize(TensorNetworks::Right);
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
    itsMPS->Normalize(TensorNetworks::Right);
    Vector3T L3=GetEOLeft_Iterate(L);
    eType EL=L3(1,1,1);
    Vector3T R3=GetEORightIterate(-1);
    eType ER=R3(1,1,1);
    double Enew=itsMPS->GetExpectation(itsH);
    double Eold=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(std::real(ER),Eold,100*eps);
    EXPECT_NEAR(std::real(EL),Eold,100*eps);
    EXPECT_NEAR(Enew,Eold,100*eps);
}

TEST_F(MPOTesting,TestHeff)
{
    int L=10;
    Setup(L,1,2);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::Right);
    LoadHeffCaches();
    // This only work for site 0 since the Left cache only gets updates by the SweepRight routine.
    int ia=0;
        Matrix6T HeffI=GetHeffIterate(ia);
//        cout << "HeffI=" << HeffI <<endl;
        Matrix6T HeffO=GetHeff(ia);
//        cout << "HeffO=" << HeffO <<endl;
        double error=Max(abs(HeffI.Flatten()-HeffO.Flatten()));
        EXPECT_NEAR(error,0,10*eps);
}

TEST_F(MPOTesting,TestGetExpectation2_I_I)
{
    int L=10,S2=1,D=2;
    Setup(L,S2,D);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::Right);
    OperatorWRepresentation* IO=new IdentityOperator();
    MatrixProductOperator* mpoi=new MatrixProductOperator(IO,L,S2);

    double E1=itsMPS->GetExpectation(itsH);
    double I1=itsMPS->GetExpectation(mpoi);
    double II=itsMPS->GetExpectation(mpoi,mpoi);
    double IE=itsMPS->GetExpectation(mpoi,itsH);
    double EI=itsMPS->GetExpectation(itsH,mpoi);
    double EE=itsMPS->GetExpectation(itsH,itsH);
    EXPECT_NEAR(I1,1.0,eps);
    EXPECT_NEAR(II,1.0,eps);
    EXPECT_NEAR(IE,E1,eps);
    EXPECT_NEAR(EI,E1,eps);
    (void)EE; //Nothing to test this agains right now
}


