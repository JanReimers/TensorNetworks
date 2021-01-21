#include "Tests.H"
#include "TensorNetworksImp/MPS/MPSImp.H"
#include "TensorNetworks/Hamiltonian.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/MPO.H"
#include "TensorNetworks/iMPO.H"
#include "TensorNetworks/Factory.H"
#include "Operators/SiteOperatorImp.H"
#include "Containers/Matrix6.H"

//#include "oml/stream.h"
//#include "oml/stopw.h"

class MPOTests : public ::testing::Test
{
public:
    typedef TensorNetworks::Matrix6CT Matrix6CT;
    typedef TensorNetworks::MatrixRT  MatrixRT;
    typedef TensorNetworks::MatrixCT  MatrixCT;
    typedef TensorNetworks::Vector3CT Vector3CT;
    typedef TensorNetworks::dcmplx     dcmplx;
    MPOTests()
        : eps(1.0e-13)
        , itsFactory(TensorNetworks::Factory::GetFactory())
        , itsH(0)
        , itsOperatorClient(0)
        , itsMPS(0)
    {
        assert(itsFactory);
        StreamableObject::SetToPretty();
    }
    ~MPOTests()
    {
        delete itsFactory;
        if (itsH) delete itsH;
//        delete itsOperatorClient; NO! part of itsH.
        if (itsMPS) delete itsMPS;
    }

    void Setup(int L, double S, int D)
    {
        if (itsH) delete itsH;
        if (itsMPS) delete itsMPS;
        itsH=itsFactory->Make1D_NN_HeisenbergHamiltonian(L,S,1.0,1.0,0.0);
        itsOperatorClient=itsH->GetClient();
        assert(itsOperatorClient);
        itsMPS=itsH->CreateMPS(D);
    }
    double ENeel(double S) const;
    Matrix6CT GetHeffIterate(int isite) const {return GetMPSImp()->GetHeffIterate(itsH,isite);}
    Vector3CT CalcHeffLeft(int isite,bool cache=false) const {return GetMPSImp()->CalcHeffLeft (itsH,isite,cache);}
    Vector3CT CalcHeffRight(int isite,bool cache=false) const {return GetMPSImp()->CalcHeffRight(itsH,isite,cache);}
    void LoadHeffCaches() {GetMPSImp()->LoadHeffCaches(itsH);}

    MatrixRT GetW1(int m, int n) {return itsOperatorClient->GetW(m,n);}
    MatrixRT GetW(int isite, int m, int n) {return itsH->GetSiteOperator(isite)->GetW(m,n);}

          TensorNetworks::MPSImp* GetMPSImp()       {return dynamic_cast<      TensorNetworks::MPSImp*>(itsMPS);}
    const TensorNetworks::MPSImp* GetMPSImp() const {return dynamic_cast<const TensorNetworks::MPSImp*>(itsMPS);}

    double eps;
           TensorNetworks::Factory*        itsFactory;
           TensorNetworks::Hamiltonian*    itsH;
    const  TensorNetworks::OperatorClient* itsOperatorClient;
           TensorNetworks::MPS*            itsMPS;
};





TEST_F(MPOTests,MakeHamiltonian)
{
    Setup(10,0.5,2);
}


TEST_F(MPOTests,CheckThatWsGotLoaded)
{
    Setup(10,0.5,2);
    EXPECT_EQ(ToString(GetW(2,0,0)),"(1:5),(1:5) \n[ 1 0 0 0 0 ]\n[ 0 0 0 0 0 ]\n[ 0 0 0 0 0 ]\n[ -0.5 0 0 0 0 ]\n[ -0 0 0 -0.5 1 ]\n");
    EXPECT_EQ(ToString(GetW(2,0,1)),"(1:5),(1:5) \n[ 0 0 0 0 0 ]\n[ 0 0 0 0 0 ]\n[ 1 0 0 0 0 ]\n[ 0 0 0 0 0 ]\n[ 0 0.5 0 0 0 ]\n");
}

double MPOTests::ENeel(double s) const
{
    return -s*s*(itsH->GetL()-1);
}

TEST_F(MPOTests,DoHamiltionExpectationL10S1_5D2)
{
    for (double S=0.5;S<=2.5;S+=0.5)
    {
        Setup(10,S,2);
        itsMPS->InitializeWith(TensorNetworks::Neel);
        EXPECT_NEAR(itsMPS->GetExpectation(itsH),ENeel(S),1e-11);
    }
}

TEST_F(MPOTests,DoHamiltionExpectationProductL10S1D1)
{
    Setup(10,0.5,1);
    itsMPS->InitializeWith(TensorNetworks::Neel);
    EXPECT_NEAR(itsMPS->GetExpectation(itsH),-2.25,1e-11);
}
TEST_F(MPOTests,DoHamiltionExpectationNeelL10S1D1)
{
    Setup(10,0.5,1);
    itsMPS->InitializeWith(TensorNetworks::Neel);
    EXPECT_NEAR(itsMPS->GetExpectation(itsH),-2.25,1e-11);
}

TEST_F(MPOTests,LeftNormalizeThenDoHamiltionExpectation)
{
    Setup(10,0.5,2);
    itsMPS->InitializeWith(TensorNetworks::Neel);
    itsMPS->Normalize(TensorNetworks::DLeft);
    EXPECT_NEAR(itsMPS->GetExpectation(itsH),-2.25,1e-11);
}
TEST_F(MPOTests,RightNormalizeThenDoHamiltionExpectation)
{
    Setup(10,0.5,2);
    itsMPS->InitializeWith(TensorNetworks::Neel);
    itsMPS->Normalize(TensorNetworks::DLeft);
    EXPECT_NEAR(itsMPS->GetExpectation(itsH),-2.25,1e-11);
}



TEST_F(MPOTests,TestGetLRIterateL10S1D2)
{
    int L=10;
    Setup(L,0.5,2);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::DRight);
    Vector3CT L3=CalcHeffLeft(L+1);
    dcmplx EL=L3(1,1,1);
    Vector3CT R3=CalcHeffRight(0);
    dcmplx ER=R3(1,1,1);
    EXPECT_NEAR(std::imag(EL),0.0,eps);
    EXPECT_NEAR(std::imag(ER),0.0,eps);
    EXPECT_NEAR(std::real(ER),std::real(EL),10*eps);
}

TEST_F(MPOTests,TestGetLRIterateL10S1D1)
{
    int L=10;
    Setup(L,0.5,1);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::DRight);
    Vector3CT L3=CalcHeffLeft(L+1);
    dcmplx EL=L3(1,1,1);
    Vector3CT R3=CalcHeffRight(0);
    dcmplx ER=R3(1,1,1);
    EXPECT_NEAR(std::imag(EL),0.0,eps);
    EXPECT_NEAR(std::imag(ER),0.0,eps);
    EXPECT_NEAR(std::real(ER),std::real(EL),10*eps);
}

TEST_F(MPOTests,TestGetLRIterateL10S1D6)
{
    int L=10;
    Setup(L,0.5,6);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::DRight);
    Vector3CT L3=CalcHeffLeft(L+1);
    dcmplx EL=L3(1,1,1);
    Vector3CT R3=CalcHeffRight(0);
    dcmplx ER=R3(1,1,1);
    EXPECT_NEAR(std::imag(EL),0.0,10*eps);
    EXPECT_NEAR(std::imag(ER),0.0,10*eps);
    EXPECT_NEAR(std::real(ER),std::real(EL),10*eps);
}
TEST_F(MPOTests,TestGetLRIterateL10S5D2)
{
    int L=10;
    Setup(L,2.5,2);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::DRight);
    Vector3CT L3=CalcHeffLeft(L+1);
    dcmplx EL=L3(1,1,1);
    Vector3CT R3=CalcHeffRight(0);
    dcmplx ER=R3(1,1,1);
    EXPECT_NEAR(std::imag(EL),0.0,100000*eps);
    EXPECT_NEAR(std::imag(ER),0.0,100000*eps);
    EXPECT_NEAR(std::real(ER),std::real(EL),100000*eps);
}

TEST_F(MPOTests,TestEoldEnew)
{
    int L=10;
    Setup(L,0.5,2);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::DRight);
    Vector3CT L3=CalcHeffLeft(L+1);
    dcmplx EL=L3(1,1,1);
    Vector3CT R3=CalcHeffRight(0);
    dcmplx ER=R3(1,1,1);
    double Enew=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(std::real(ER),Enew,100*eps);
    EXPECT_NEAR(std::real(EL),Enew,100*eps);
}

TEST_F(MPOTests,TestMPOCombineForH2)
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

TEST_F(MPOTests,TestMPOCompressForH2)
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
    H2->CanonicalForm(TensorNetworks::DLeft);
    H2->CanonicalForm(TensorNetworks::DRight);
    H2->CompressStd(0,1e-13);
//    H2->Report(cout);
    EXPECT_EQ(H2->GetMaxDw(),10);
    delete H1;
    delete H2;
}

TEST_F(MPOTests,TestHamiltonianCreateH2)
{
    int L=10;
    Setup(L,0.5,1);
    TensorNetworks::MPO* H2=itsH->CreateH2Operator();
    EXPECT_EQ(H2->GetMaxDw(),10);
    delete H2;
}

TEST_F(MPOTests,TestHamiltonianCompressL2)
{
    int L=2;
    Setup(L,0.5,1);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::DRight);
    double E1=itsMPS->GetExpectation(itsH);
    itsH->CompressStd(0,1e-13);
    double E2=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(E1,E2,1e-13);
    EXPECT_EQ(itsH->GetMaxDw(),5);
}

TEST_F(MPOTests,TestH2CompressL2)
{
    int L=2,D=8;
    Setup(L,0.5,D);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::DRight);
    TensorNetworks::MPO* H2=itsH->CreateUnitOperator();
    H2->Combine(itsH);
    H2->Combine(itsH);
//    H2->Report(cout);
    double E21=itsMPS->GetExpectation(H2);
    EXPECT_EQ(H2->GetMaxDw(),25);
    H2->CompressStd(0,1e-13);
    H2->Report(cout);
    H2->Dump(cout);
    double E22=itsMPS->GetExpectation(H2);
    EXPECT_NEAR(E21,E22,1e-13);
    EXPECT_EQ(H2->GetMaxDw(),10);
}


TEST_F(MPOTests,TestHamiltonianCreateH2L2)
{
    int L=2;
    Setup(L,0.5,1);
    TensorNetworks::MPO* H2=itsH->CreateH2Operator();
    EXPECT_EQ(H2->GetMaxDw(),10);
//    EXPECT_EQ(H2->GetMaxDw(),9); was expecting 9 but SVD for L=2 only gives Dw=4 = d*d
    delete H2;
}

TEST_F(MPOTests,TestMPOCompressForE2)
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
    H2->CanonicalForm(TensorNetworks::DLeft);
    H2->CanonicalForm(TensorNetworks::DRight);
    H2->CompressStd(0,1e-13);
    EXPECT_EQ(H2->GetMaxDw(),10);
    double E2b=itsMPS->GetExpectation(H2);
    EXPECT_NEAR(E2a,E2b,1e-13);
    delete H1;
    delete H2;
}
TEST_F(MPOTests,TestL2MPOTrotter2)
{
    int L=4,D=2;
    double S=0.5,dt=0.1;
    Setup(L,S,D);
    TensorNetworks::MPO* expH=itsH->CreateOperator(dt,TensorNetworks::SecondOrder);
//    expH->CanonicalForm(TensorNetworks::DLeft);
//    expH->CanonicalForm(TensorNetworks::DRight);
    expH->CompressStd(0,1e-4);
//    expH->Report(cout);
    for (int is=2;is<=L-1;is++)
    {
        TensorNetworks::Dw12 Dw=expH->GetSiteOperator(is)->GetDw12();
        EXPECT_EQ(Dw.Dw1,4);
        EXPECT_EQ(Dw.Dw2,4);
    }
}

TEST_F(MPOTests,TestL2iMPOTrotter1)
{
    int L=2,D=2;
    double S=0.5,dt=0.00001,epsMPO=1e-14;
    Setup(L,S,D);
    TensorNetworks::iMPO* expH=itsH->CreateiMPO(dt,TensorNetworks::FirstOrder,epsMPO);
//    expH->Report(cout);
    for (int is=1;is<=L;is++)
    {
        TensorNetworks::Dw12 Dw=expH->GetSiteOperator(is)->GetDw12();
        EXPECT_EQ(Dw.Dw1,4);
        EXPECT_EQ(Dw.Dw2,4);

//        int d=2*S+1;
//        TensorNetworks::SiteOperator* so=expH->GetSiteOperator(is);
//        cout << std::fixed << endl;
//        for (int m=0;m<d;m++)
//            for (int n=0;n<d;n++)
//                cout << "W(" << m << "," << n << ")=" << so->GetW(m,n) << endl;
    }
}
TEST_F(MPOTests,TestL2iMPOTrotter2)
{
    int L=2,D=2;
    double S=0.5,dt=0.1,epsMPO=6e-3;
    Setup(L,S,D);
    TensorNetworks::iMPO* expH=itsH->CreateiMPO(dt,TensorNetworks::SecondOrder,epsMPO);
//    expH->Report(cout);
    for (int is=1;is<=L;is++)
    {
        TensorNetworks::Dw12 Dw=expH->GetSiteOperator(is)->GetDw12();
        EXPECT_EQ(Dw.Dw1,4);
        EXPECT_EQ(Dw.Dw2,4);
    }
}
TEST_F(MPOTests,TestL2iMPOTrotter4)
{
    int L=2,D=2;
    double S=0.5,dt=0.1,epsMPO=1e-4;
    Setup(L,S,D);
    TensorNetworks::iMPO* expH=itsH->CreateiMPO(dt,TensorNetworks::FourthOrder,epsMPO);
    for (int is=1;is<=L;is++)
    {
        TensorNetworks::Dw12 Dw=expH->GetSiteOperator(is)->GetDw12();
        EXPECT_EQ(Dw.Dw1,4);
        EXPECT_EQ(Dw.Dw2,4);
    }
}

TEST_F(MPOTests,TestParkerCanonicalL9)
{
    int L=9,D=2;
    double S=0.5;
    Setup(L,S,D);
    itsMPS->InitializeWith(TensorNetworks::Random);

    EXPECT_EQ(itsH->GetNormStatus(),"WWWWWWWWW");
    double E=itsMPS->GetExpectation(itsH);
    itsH->CanonicalForm(TensorNetworks::DLeft);
    EXPECT_EQ(itsH->GetNormStatus(),"LLLLLLLLW");
    double Eleft=itsMPS->GetExpectation(itsH);
    itsH->CanonicalForm(TensorNetworks::DRight);
    EXPECT_EQ(itsH->GetNormStatus(),"WRRRRRRRR"); //The last site ends up being both right and left normalized
    double Eright=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(E,Eleft ,1e-13);
    EXPECT_NEAR(E,Eright,1e-13);
    EXPECT_EQ(itsH->GetMaxDw(),5);
}

TEST_F(MPOTests,TestParkerSVDCompressHL9)
{
    int L=9,D=2;
    double S=0.5;
    Setup(L,S,D);
    itsMPS->InitializeWith(TensorNetworks::Random);


    EXPECT_EQ(itsH->GetNormStatus(),"WWWWWWWWW");
    double E=itsMPS->GetExpectation(itsH);
    itsH->CanonicalForm(TensorNetworks::DLeft); //Do we need to sweep both ways? Yes!!!!
    EXPECT_EQ(itsH->GetNormStatus(),"LLLLLLLLW");
    itsH->CanonicalForm(TensorNetworks::DRight);
    EXPECT_EQ(itsH->GetNormStatus(),"WRRRRRRRR");
    double Eright=itsMPS->GetExpectation(itsH);
    itsH->CompressParker(0,1e-13);
    EXPECT_EQ(itsH->GetNormStatus(),"WRRRRRRRR");
    itsH->Report(cout);
    double Ecomp=itsMPS->GetExpectation(itsH);
    cout << itsH->GetNormStatus() << endl;;
    EXPECT_NEAR(E,Eright,1e-13);
    EXPECT_NEAR(E,Ecomp ,1e-13);
    EXPECT_EQ(itsH->GetMaxDw(),5);
}

TEST_F(MPOTests,TestParkerSVDCompressH2L9)
{
    int L=9,D=2;
    double S=0.5;
    Setup(L,S,D);
    itsMPS->InitializeWith(TensorNetworks::Random);
//    TensorNetworks::MPO* H2=itsH->CreateH2Operator();
    TensorNetworks::MPO* H2=itsH->CreateUnitOperator();
    H2->Combine(itsH);
    H2->Combine(itsH);

    EXPECT_EQ(H2->GetNormStatus(),"WWWWWWWWW");
    double E2=itsMPS->GetExpectation(H2);
    H2->CanonicalForm(TensorNetworks::DLeft); //Do we need to sweep both ways?
    EXPECT_EQ(H2->GetNormStatus(),"LLLLLLLLW");
    H2->CanonicalForm(TensorNetworks::DRight);
    EXPECT_EQ(H2->GetNormStatus(),"WRRRRRRRR");
    double E2right=itsMPS->GetExpectation(H2);
    H2->CompressParker(0,1e-13);
    EXPECT_EQ(H2->GetNormStatus(),"WRRRRRRRR");
    H2->Report(cout);
    double E2comp=itsMPS->GetExpectation(H2);
    EXPECT_NEAR(E2,E2right,1e-13);
    EXPECT_NEAR(E2,E2comp ,1e-13);
    EXPECT_EQ(H2->GetMaxDw(),10);
}

TEST_F(MPOTests,TestParkerSVDCompressH2L40)
{
    int L=40,D=2;
    double S=0.5;
    Setup(L,S,D);
    itsMPS->InitializeWith(TensorNetworks::Random);
    TensorNetworks::MPO* H2=itsH->CreateUnitOperator();
    H2->Combine(itsH);
    H2->Combine(itsH);

    double E2=itsMPS->GetExpectation(H2);
    H2->CanonicalForm(TensorNetworks::DLeft); //Do we need to sweep both ways?
    H2->CanonicalForm(TensorNetworks::DRight);
    double E2right=itsMPS->GetExpectation(H2);
    H2->CompressParker(0,1e-13);
    H2->Report(cout);
    double E2comp=itsMPS->GetExpectation(H2);
    EXPECT_NEAR(E2,E2right,1e-13);
    EXPECT_NEAR(E2,E2comp ,1e-13);
    EXPECT_EQ(H2->GetMaxDw(),10);
}

/*
TEST_F(MPOTests,TestParkerSVDCompressExpHL9)
{
    int L=9,D=2;
    double S=0.5,dt=0.1;
    Setup(L,S,D);
    itsMPS->InitializeWith(TensorNetworks::Random);
    TensorNetworks::iMPO* expH=itsH->CreateiMPO(dt,TensorNetworks::SecondOrder,1e-13);

    expH->Report(cout);
    expH->CanonicalForm(TensorNetworks::DLeft); //Do we need to sweep both ways?
    expH->Report(cout);
    expH->CanonicalForm(TensorNetworks::DRight);
    expH->Report(cout);
    expH->CompressParker(0,1e-13);
    expH->Report(cout);
    EXPECT_EQ(expH->GetMaxDw(),10);
//    double E2comp=itsMPS->GetExpectation(H2);
//    EXPECT_NEAR(E2,E2right,1e-13);
//    EXPECT_NEAR(E2,E2comp ,1e-13);
}

*/
#ifndef DEBUG

TEST_F(MPOTests,TestTimingE2_S5D4)
{
    int L=10,D=4;
    double S=2.5;
    Setup(L,S,D);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::DRight);
    TensorNetworks::MPO* H2=itsH->CreateH2Operator();
    double EE=itsMPS->GetExpectation(H2);
    delete H2;
  //  cout << "<E^2> contraction for L=" << L << ", S=" << S << ", D=" << D << " took " << sw.GetTime() << " seconds." << endl;
    (void)EE; //Avoid warning
}
#endif

#ifdef RunLongTests

TEST_F(MPOTests,TestTimingE2_S1D16)
{
    int L=10,D=16;
    double S=2.5;
    Setup(L,S,D);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::DRight);
    TensorNetworks::MPO* H2=itsH->CreateH2Operator();
    double EE=itsMPS->GetExpectation(H2);
    delete H2;
    cout << "<E^2> contraction for L=" << L << ", S=" << S << ", D=" << D  << endl;
    (void)EE; //Avoid warning
}
#endif



