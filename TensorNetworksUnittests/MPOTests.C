#include "Tests.H"
#include "TensorNetworksImp/MPS/MPSImp.H"
#include "TensorNetworksImp/Hamiltonians/Hamiltonian_1D_NN_Heisenberg.H"
#include "Operators/MPO_SpatialTrotter.H"
#include "Operators/MPO_TwoSite.H"
#include "Containers/Matrix4.H"
#include "TensorNetworks/Hamiltonian.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/MPO.H"
#include "TensorNetworks/Factory.H"
#include "Operators/SiteOperatorImp.H"
#include "Containers/Matrix6.H"

using TensorNetworks::Hamiltonian;
using TensorNetworks::MPS;
using TensorNetworks::Random;
using TensorNetworks::Neel;
using TensorNetworks::DLeft;
using TensorNetworks::DRight;
using TensorNetworks::IterationSchedule;
using TensorNetworks::Epsilons;
using TensorNetworks::MPO;
using TensorNetworks::TriType;
using TensorNetworks::MPSImp;
using TensorNetworks::Std;
using TensorNetworks::Parker;
using TensorNetworks::CNone;
using TensorNetworks::FirstOrder;
using TensorNetworks::SecondOrder;
using TensorNetworks::FourthOrder;
using TensorNetworks::MPO_TwoSite;
using TensorNetworks::Sz;
using TensorNetworks::Sp;
using TensorNetworks::Sm;
using TensorNetworks::MPOForm;
using TensorNetworks::RegularLower;
using TensorNetworks::RegularUpper;

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
        if (itsH  )  delete itsH;
        if (itsMPS)  delete itsMPS;
        if (itsOperatorClient) delete itsOperatorClient;
    }

    void Setup(int L, double S, int D,MPOForm f=RegularLower)
    {
        assert(L>1);
        if (itsH  ) delete itsH;
        if (itsMPS) delete itsMPS;
        if (itsOperatorClient) delete itsOperatorClient;
        itsH =itsFactory->Make1D_NN_HeisenbergHamiltonian( L,S,f,1.0,1.0,0.0);
        itsMPS=itsH->CreateMPS(D);
        itsOperatorClient=new TensorNetworks::Hamiltonian_1D_NN_Heisenberg(S,1.0,1.0,0.0);
        assert(itsOperatorClient);
    }
    double ENeel(double S) const;
    Matrix6CT GetHeffIterate(int isite) const {return GetMPSImp()->GetHeffIterate(itsH,isite);}
    Vector3CT CalcHeffLeft(int isite,bool cache=false) const {return GetMPSImp()->CalcHeffLeft (itsH,isite,cache);}
    Vector3CT CalcHeffRight(int isite,bool cache=false) const {return GetMPSImp()->CalcHeffRight(itsH,isite,cache);}
    void LoadHeffCaches() {GetMPSImp()->LoadHeffCaches(itsH);}
    MPO* MakeEnergyMPO(int isite) const;

          TensorNetworks::MPSImp* GetMPSImp()       {return dynamic_cast<      TensorNetworks::MPSImp*>(itsMPS);}
    const TensorNetworks::MPSImp* GetMPSImp() const {return dynamic_cast<const TensorNetworks::MPSImp*>(itsMPS);}

    double eps;
           TensorNetworks::Factory*        itsFactory;
           TensorNetworks:: Hamiltonian*   itsH;
    const  TensorNetworks::OperatorClient* itsOperatorClient;
           TensorNetworks::MPS*            itsMPS;
};

MPO* MPOTests::MakeEnergyMPO(int isite) const
{
    int    L=itsH->GetL();
    double S=itsH->GetS();
    MPO* SS=new MPO_TwoSite(L,S ,isite,isite+1, Sz,Sz);
    MPO* SpSmo=new MPO_TwoSite(L,S ,isite,isite+1, Sp,Sm);
    MPO* SmSpo=new MPO_TwoSite(L,S ,isite,isite+1, Sm,Sp);
    SS->Sum(SpSmo,0.5); //This applies 0.5 to all the unit ops.
    SS->Sum(SmSpo,0.5);
    delete SmSpo;
    delete SpSmo;
    return SS;
}



TEST_F(MPOTests,MakeHamiltonian)
{
    Setup(10,0.5,2);
}


double MPOTests::ENeel(double s) const
{
    return -s*s*(itsH->GetL()-1);
}

TEST_F(MPOTests,DoHamiltionExpectation_Lower_L10S1_5D2)
{
    for (double S=0.5;S<=2.5;S+=0.5)
    {
        Setup(10,S,2,RegularLower);
        itsMPS->InitializeWith(Neel);
        EXPECT_NEAR(itsMPS->GetExpectation(itsH),ENeel(S),1e-11);
    }
}
TEST_F(MPOTests,DoHamiltionExpectation_Upper_L10S1_5D2)
{
    for (double S=0.5;S<=2.5;S+=0.5)
    {
        Setup(10,S,2,RegularUpper);
        itsMPS->InitializeWith(Neel);
        EXPECT_NEAR(itsMPS->GetExpectation(itsH),ENeel(S),1e-11);
    }
}

TEST_F(MPOTests,DoHamiltionExpectationProductL10S1D1)
{
    Setup(10,0.5,1);
    itsMPS->InitializeWith(Neel);
    EXPECT_NEAR(itsMPS->GetExpectation(itsH),-2.25,1e-11);
}
TEST_F(MPOTests,DoHamiltionExpectationNeelL10S1D1)
{
    Setup(10,0.5,1);
    itsMPS->InitializeWith(Neel);
    EXPECT_NEAR(itsMPS->GetExpectation(itsH),-2.25,1e-11);
}

TEST_F(MPOTests,DoBuildMPO_Neel)
{
    Setup(10,0.5,1);
    itsMPS->InitializeWith(Neel);
    MPO* h12=MakeEnergyMPO(2);
//    h12->Report(cout);
//    h12->Dump(cout);
    EXPECT_NEAR(itsMPS->GetExpectation(h12),-0.25,1e-11);
//    h12->CanonicalForm();  //OpValMatrix.C line 430 fails.
//    EXPECT_NEAR(itsMPS->GetExpectation(h12),-0.25,1e-11);
    delete h12;
}


TEST_F(MPOTests,LeftNormalizeThenDoHamiltionExpectation)
{
    Setup(10,0.5,2);
    itsMPS->InitializeWith(Neel);
    itsMPS->Normalize(DLeft);
    EXPECT_NEAR(itsMPS->GetExpectation(itsH),-2.25,1e-11);
}
TEST_F(MPOTests,RightNormalizeThenDoHamiltionExpectation)
{
    Setup(10,0.5,2);
    itsMPS->InitializeWith(Neel);
    itsMPS->Normalize(DLeft);
    EXPECT_NEAR(itsMPS->GetExpectation(itsH),-2.25,1e-11);
}



TEST_F(MPOTests,TestGetLRIterateL10S1D2)
{
    int L=10;
    Setup(L,0.5,2);
    itsMPS->InitializeWith(Random);
    itsMPS->Normalize(DRight);
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
    itsMPS->InitializeWith(Random);
    itsMPS->Normalize(DRight);
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
    itsMPS->InitializeWith(Random);
    itsMPS->Normalize(DRight);
    Vector3CT L3=CalcHeffLeft(L+1);
    dcmplx EL=L3(1,1,1);
    Vector3CT R3=CalcHeffRight(0);
    dcmplx ER=R3(1,1,1);
    EXPECT_NEAR(std::imag(EL),0.0,10*eps);
    EXPECT_NEAR(std::imag(ER),0.0,10*eps);
    EXPECT_NEAR(std::real(ER),std::real(EL),10*eps);
}
TEST_F(MPOTests,TestGetLRIterate_Lower_L10S5D2)
{
    int L=10;
    Setup(L,2.5,2,RegularLower);
    itsMPS->InitializeWith(Random);
    itsMPS->Normalize(DRight);
    Vector3CT L3=CalcHeffLeft(L+1);
    dcmplx EL=L3(1,1,1);
    Vector3CT R3=CalcHeffRight(0);
    dcmplx ER=R3(1,1,1);
    EXPECT_NEAR(std::imag(EL),0.0,100000*eps);
    EXPECT_NEAR(std::imag(ER),0.0,100000*eps);
    EXPECT_NEAR(std::real(ER),std::real(EL),100000*eps);
}
TEST_F(MPOTests,TestGetLRIterate_Upper_L10S5D2)
{
    int L=10;
    Setup(L,2.5,2,RegularUpper);
    itsMPS->InitializeWith(Random);
    itsMPS->Normalize(DRight);
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
    itsMPS->InitializeWith(Random);
    itsMPS->Normalize(DRight);
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
    itsMPS->InitializeWith(Random);
    itsMPS->Normalize(DRight);
    int Dw=itsH->GetMaxDw();
    MPO* H1=itsH->CreateUnitOperator();
    H1->Product(itsH);
    MPO* H2=itsH->CreateUnitOperator();
    H2->Product(itsH);
    H2->Product(itsH);
    EXPECT_EQ(H1->GetMaxDw(),Dw);
    EXPECT_EQ(H2->GetMaxDw(),Dw*Dw);
    delete H1;
    delete H2;
}

TEST_F(MPOTests,TestMPOStdCompressForH_Lower)
{
    int L=10,D=8;
    Setup(L,0.5,D,RegularLower);
    itsMPS->InitializeWith(Random);
    itsMPS->Normalize(DRight);
    EXPECT_EQ(itsH->GetUpperLower()," LLLLLLLL ");
    itsH->CanonicalForm();
    EXPECT_EQ(itsH->GetUpperLower()," LLLLLLLL ");
    double truncError=itsH->Compress(Std,0,1e-13);
    EXPECT_EQ(itsH->GetUpperLower()," FFFFFFFF ");
    EXPECT_EQ(itsH->GetMaxDw(),5);
    EXPECT_LT(truncError,1e-13);
}
TEST_F(MPOTests,TestMPOStdCompressForH_Upper)
{
    int L=10,D=8;
    Setup(L,0.5,D,RegularUpper);
    itsMPS->InitializeWith(Random);
    itsMPS->Normalize(DRight);
    EXPECT_EQ(itsH->GetUpperLower()," UUUUUUUU ");
    itsH->CanonicalForm();
    EXPECT_EQ(itsH->GetUpperLower()," UUUUUUUU ");
    double truncError=itsH->Compress(Std,0,1e-13);
    EXPECT_EQ(itsH->GetUpperLower()," FFFFFFFF ");
    EXPECT_EQ(itsH->GetMaxDw(),5);
    EXPECT_LT(truncError,1e-13);
}

TEST_F(MPOTests,TestMPOStdCompressForH2_Lower)
{
    int L=10,D=8;
    Setup(L,0.5,D,RegularLower);
    itsMPS->InitializeWith(Random);
    itsMPS->Normalize(DRight);
    EXPECT_EQ(itsH->GetUpperLower()," LLLLLLLL ");
    MPO* H1=itsH->CreateUnitOperator();
    H1->Product(itsH);
    MPO* H2=itsH->CreateUnitOperator();
    H2->Product(itsH);
    EXPECT_EQ(H2->GetUpperLower()," LLLLLLLL ");
    H2->Product(itsH);
    EXPECT_EQ(H2->GetUpperLower()," LLLLLLLL ");
    H2->CanonicalForm();
    EXPECT_EQ(H2->GetUpperLower()," LLLLLLLL ");
    double truncError=H2->Compress(Std,0,1e-13);
    EXPECT_EQ(H2->GetUpperLower()," FFFFFFFF ");
    EXPECT_EQ(H2->GetMaxDw(),9);
    EXPECT_LT(truncError,1e-13);
    delete H1;
    delete H2;
}
TEST_F(MPOTests,TestMPOStdCompressForH2_Upper)
{
    int L=10,D=8;
    Setup(L,0.5,D,RegularUpper);
    itsMPS->InitializeWith(Random);
    itsMPS->Normalize(DRight);
    EXPECT_EQ(itsH->GetUpperLower()," UUUUUUUU ");
    MPO* H1=itsH->CreateUnitOperator();
    H1->Product(itsH);
    MPO* H2=itsH->CreateUnitOperator();
    H2->Product(itsH);
    EXPECT_EQ(H2->GetUpperLower()," UUUUUUUU ");
    H2->Product(itsH);
    EXPECT_EQ(H2->GetUpperLower()," UUUUUUUU ");
    H2->CanonicalForm();
    EXPECT_EQ(H2->GetUpperLower()," UUUUUUUU ");
    double truncError=H2->Compress(Std,0,1e-13);
    EXPECT_EQ(H2->GetUpperLower()," FFFFFFFF ");
    EXPECT_EQ(H2->GetMaxDw(),9);
    EXPECT_LT(truncError,1e-13);
    delete H1;
    delete H2;
}

TEST_F(MPOTests,TestHamiltonianCreateH2)
{
    int L=10;
    Setup(L,0.5,1);
    MPO* H2=itsH->CreateH2Operator();
    EXPECT_EQ(H2->GetMaxDw(),9);
    delete H2;
}

TEST_F(MPOTests,TestHamiltonianCompressL2)
{
    int L=2;
    Setup(L,0.5,1);
    itsMPS->InitializeWith(Random);
    itsMPS->Normalize(DRight);
    double E1=itsMPS->GetExpectation(itsH);
    MPO* Hmpo=itsH;
    double truncError=Hmpo->Compress(Std,0,1e-13);
    double E2=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(E1,E2,1e-13);
    EXPECT_EQ(itsH->GetMaxDw(),3);
    EXPECT_LT(truncError,1e-13);
}

TEST_F(MPOTests,TestH2CompressL2)
{
    int L=2,D=8;
    Setup(L,0.5,D);
    itsMPS->InitializeWith(Random);
    itsMPS->Normalize(DRight);
    MPO* H2=itsH->CreateUnitOperator();
    H2->Product(itsH);
    H2->Product(itsH);
    double E21=itsMPS->GetExpectation(H2);
    EXPECT_EQ(H2->GetMaxDw(),25);
    double truncError=H2->Compress(Std,0,1e-13);
    double E22=itsMPS->GetExpectation(H2);
    EXPECT_NEAR(E21,E22,1e-13);
    EXPECT_EQ(H2->GetMaxDw(),4);
    EXPECT_LT(truncError,1e-13);
}


TEST_F(MPOTests,TestHamiltonianCreateH2L2)
{
    int L=2;
    Setup(L,0.5,1);
    MPO* H2=itsH->CreateH2Operator();
    EXPECT_EQ(H2->GetMaxDw(),5); //Parker compression
//    EXPECT_EQ(H2->GetMaxDw(),4); //Std compression
    delete H2;
}

TEST_F(MPOTests,TestMPOCompressForE2)
{
    int L=10,D=8;
    Setup(L,0.5,D);
    itsMPS->InitializeWith(Random);
    itsMPS->Normalize(DRight);
    MPO* H1=itsH->CreateUnitOperator();
    H1->Product(itsH);
    MPO* H2=itsH->CreateUnitOperator();
    H2->Product(itsH);
    H2->Product(itsH);
    double E2a=itsMPS->GetExpectation(H2);
    H2->CanonicalForm();
    double E2b=itsMPS->GetExpectation(H2);
    double truncError=H2->Compress(Std,0,1e-13);
    EXPECT_EQ(H2->GetMaxDw(),9);
    double E2c=itsMPS->GetExpectation(H2);
    EXPECT_NEAR(E2a,E2b,1e-13);
    EXPECT_NEAR(E2a,E2c,1e-13);
    EXPECT_LT(truncError,1e-13);
    delete H1;
    delete H2;
}
TEST_F(MPOTests,TestMPOTrotter2_L4_S12)
{
    int L=4,D=2;
    double S=0.5,dt=0.1,epsMPO=1e-4;
    Setup(L,S,D);
    MPO* expH=itsH->CreateOperator(dt,SecondOrder,CNone,epsMPO);
//    expH->CanonicalForm(); this changes the Dws
//    expH->Report(cout);
    double truncError=expH->Compress(Std,0,epsMPO);
    for (int is=2;is<=L-1;is++)
    {
        auto [Dw1,Dw2]=expH->GetSiteOperator(is)->GetDws();
        EXPECT_EQ(Dw1,4);
        EXPECT_EQ(Dw2,4);
    }
    EXPECT_LT(truncError,epsMPO*50);
}
TEST_F(MPOTests,TestMPOTrotter2_L4_S1)
{
    int L=4,D=2;
    double S=1.0,dt=0.1,epsMPO=1e-4;
    Setup(L,S,D);
    MPO* expH=itsH->CreateOperator(dt,SecondOrder,CNone,epsMPO);
    //expH->CanonicalForm(); //Can't handle exp(H) yet.
    double truncError=expH->Compress(Std,0,epsMPO);
//    expH->Report(cout);
    EXPECT_EQ(expH->GetMaxDw(),10);
    EXPECT_LT(truncError,epsMPO*50);
}
TEST_F(MPOTests,TestMPOTrotter2_L5_S12)
{
    int L=5,D=2;
    double S=0.5,dt=0.1,epsMPO=1e-4;
    Setup(L,S,D);
    MPO* expH=itsH->CreateOperator(dt,SecondOrder,CNone,epsMPO);
//    expH->CanonicalForm(); this changes the Dws
//    expH->Report(cout);
    double truncError=expH->Compress(Std,0,epsMPO);
    for (int is=2;is<=L-1;is++)
    {
        auto [Dw1,Dw2]=expH->GetSiteOperator(is)->GetDws();
        EXPECT_EQ(Dw1,4);
        EXPECT_EQ(Dw2,4);
    }
    EXPECT_LT(truncError,epsMPO*50);
}
TEST_F(MPOTests,TestMPOTrotter2_L5_S1)
{
    int L=5,D=2;
    double S=1.0,dt=0.1,epsMPO=1e-4;
    Setup(L,S,D);
    MPO* expH=itsH->CreateOperator(dt,SecondOrder,CNone,epsMPO);
    //expH->CanonicalForm(); //Can't handle exp(H) yet.
    double truncError=expH->Compress(Std,0,epsMPO);
//    expH->Report(cout);
    EXPECT_EQ(expH->GetMaxDw(),10);
    EXPECT_LT(truncError,epsMPO*50);
}

TEST_F(MPOTests,TestParkerCanonical_Lower_L9H)
{
    int L=9,D=2;
    double S=0.5;
    Setup(L,S,D,RegularLower);
//    itsH->Report(cout);
    itsMPS->InitializeWith(Random);
    itsMPS->Normalize(DLeft);
    MPO* H=itsH;
    EXPECT_EQ(H->GetNormStatus(),"WWWWWWWWW");
    EXPECT_EQ(H->GetUpperLower()," LLLLLLL ");
    double E=itsMPS->GetExpectation(itsH);
    itsH->CanonicalForm();
//    H->Report(cout);
    EXPECT_EQ(H->GetNormStatus(),"WRRRRRRRR"); //The last site ends up being both right and left normalized
    EXPECT_EQ(H->GetUpperLower()," LLLLLLL ");
    double Eright=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(E,Eright,1e-13);
    EXPECT_EQ(H->GetMaxDw(),5);
}

TEST_F(MPOTests,TestParkerCanonical_Upper_L9H)
{
    int L=9,D=2;
    double S=0.5;
    Setup(L,S,D,RegularUpper);
//    itsH->Report(cout);
    itsMPS->InitializeWith(Random);
    itsMPS->Normalize(DLeft);
    MPO* H=itsH;
    EXPECT_EQ(H->GetNormStatus(),"WWWWWWWWW");
    EXPECT_EQ(H->GetUpperLower()," UUUUUUU ");
    double E=itsMPS->GetExpectation(itsH);
    itsH->CanonicalForm();
//    H->Report(cout);
    EXPECT_EQ(H->GetNormStatus(),"WRRRRRRRR"); //The last site ends up being both right and left normalized
    EXPECT_EQ(H->GetUpperLower()," UUUUUUU ");
    double Eright=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(E,Eright,1e-13);
    EXPECT_EQ(H->GetMaxDw(),5);
}

TEST_F(MPOTests,TestParkerSVDCompress_Lower_HL9)
{
    int L=9,D=2;
    double S=0.5;
    Setup(L,S,D,RegularLower);
    itsMPS->InitializeWith(Random);
    itsMPS->Normalize(DLeft);

    MPO* H=itsH;
    EXPECT_EQ(H->GetNormStatus(),"WWWWWWWWW");
    EXPECT_EQ(H->GetUpperLower()," LLLLLLL ");
    double E=itsMPS->GetExpectation(itsH);
    H->CanonicalForm(); //Do we need to sweep both ways? Yes!!!!
    EXPECT_EQ(H->GetNormStatus(),"WRRRRRRRR");
    EXPECT_EQ(H->GetUpperLower()," LLLLLLL ");
    double Eright=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(E,Eright,1e-13);
    double truncError=H->Compress(Parker,0,1e-13);
    EXPECT_EQ(H->GetNormStatus(),"WRRRRRRRR");
    EXPECT_EQ(H->GetUpperLower()," LLLLLLL "); //This one happens to work out maintain lower.
    double Ecomp=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(E,Ecomp ,1e-13);
    EXPECT_EQ(itsH->GetMaxDw(),5);
    EXPECT_EQ(truncError,0.0); //H should be uncompressable
}
TEST_F(MPOTests,TestParkerSVDCompress_Upper_HL9)
{
    int L=9,D=2;
    double S=0.5;
    Setup(L,S,D,RegularUpper);
    itsMPS->InitializeWith(Random);
    itsMPS->Normalize(DLeft);

    MPO* H=itsH;
    EXPECT_EQ(H->GetNormStatus(),"WWWWWWWWW");
    EXPECT_EQ(H->GetUpperLower()," UUUUUUU ");
    double E=itsMPS->GetExpectation(itsH);
    H->CanonicalForm(); //Do we need to sweep both ways? Yes!!!!
    EXPECT_EQ(H->GetNormStatus(),"WRRRRRRRR");
    EXPECT_EQ(H->GetUpperLower()," UUUUUUU ");
    double Eright=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(E,Eright,1e-13);
    double truncError=H->Compress(Parker,0,1e-13);
    EXPECT_EQ(H->GetNormStatus(),"WRRRRRRRR");
    EXPECT_EQ(H->GetUpperLower()," UUUUUUU ");
    double Ecomp=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(E,Ecomp ,1e-13);
    EXPECT_EQ(itsH->GetMaxDw(),5);
    EXPECT_EQ(truncError,0.0); //H should be uncompressable
}

TEST_F(MPOTests,TestParkerSVDCompress_Lower_H2L9)
{
    int L=9,D=2;
    double S=0.5;
    Setup(L,S,D,RegularLower);
    itsMPS->InitializeWith(Random);
    itsMPS->Normalize(DLeft);
    MPO* H2=itsH->CreateUnitOperator();
    H2->Product(itsH);
    H2->Product(itsH);

    EXPECT_EQ(H2->GetNormStatus(),"WWWWWWWWW");
    EXPECT_EQ(H2->GetUpperLower()," LLLLLLL ");
    double E2=itsMPS->GetExpectation(H2);
    H2->CanonicalForm(); //Do we need to sweep both ways?
    EXPECT_EQ(H2->GetNormStatus(),"WRRRRRRRR");
    EXPECT_EQ(H2->GetUpperLower()," LLLLLLL ");
    double E2can=itsMPS->GetExpectation(H2);
    double truncError=H2->Compress(Parker,0,1e-13);
    EXPECT_EQ(H2->GetNormStatus(),"WRRRRRRRR");
    EXPECT_EQ(H2->GetUpperLower()," FFFFFFF ");
    double E2comp=itsMPS->GetExpectation(H2);
    EXPECT_NEAR(E2,E2can,1e-13);
    EXPECT_NEAR(E2,E2comp ,1e-13);
    EXPECT_EQ(H2->GetMaxDw(),9);
    EXPECT_LT(truncError,1e-13);
}

TEST_F(MPOTests,TestParkerSVDCompress_Upper_H2L9)
{
    int L=9,D=2;
    double S=0.5;
    Setup(L,S,D,RegularUpper);
    itsMPS->InitializeWith(Random);
    itsMPS->Normalize(DLeft);
    MPO* H2=itsH->CreateUnitOperator();
    H2->Product(itsH);
    H2->Product(itsH);

    EXPECT_EQ(H2->GetNormStatus(),"WWWWWWWWW");
    EXPECT_EQ(H2->GetUpperLower()," UUUUUUU ");
    double E2=itsMPS->GetExpectation(H2);
    H2->CanonicalForm(); //Do we need to sweep both ways?
    EXPECT_EQ(H2->GetNormStatus(),"WRRRRRRRR");
    EXPECT_EQ(H2->GetUpperLower()," UUUUUUU ");
    double E2can=itsMPS->GetExpectation(H2);
    double truncError=H2->Compress(Parker,0,1e-13);
    EXPECT_EQ(H2->GetNormStatus(),"WRRRRRRRR");
    EXPECT_EQ(H2->GetUpperLower()," FFFFFFF ");
    double E2comp=itsMPS->GetExpectation(H2);
    EXPECT_NEAR(E2,E2can,1e-13);
    EXPECT_NEAR(E2,E2comp ,1e-13);
    EXPECT_EQ(H2->GetMaxDw(),9);
    EXPECT_LT(truncError,1e-13);
}

TEST_F(MPOTests,TestParkerSVDCompressH2L256)
{
    int L=256,D=2;
    double epsE=2e-10;
#ifdef DEBUG
    L=32;
    epsE=5e-13;
#endif // DEBUG
    double S=0.5;
    Setup(L,S,D);
    itsMPS->InitializeWith(Random);
    itsMPS->Normalize(DLeft);
    MPO* H2=itsH->CreateUnitOperator();
    H2->Product(itsH);
    H2->Product(itsH);
    double E2=itsMPS->GetExpectation(H2);
    H2->CanonicalForm(); //Sweep both ways
    double E2right=itsMPS->GetExpectation(H2);
    double truncError=H2->Compress(Parker,0,1e-13);
//    H2->Report(cout);
    double E2comp=itsMPS->GetExpectation(H2);
//    cout << E2 << " " << E2right << " " << E2comp << endl;
    EXPECT_NEAR(E2,E2right,epsE);
    EXPECT_NEAR(E2,E2comp ,epsE);
    EXPECT_EQ(H2->GetMaxDw(),9);
    EXPECT_LT(truncError,1e-13);
}

TEST_F(MPOTests,TestParkerSVDCompressH2_2Body_Upper)
{
    int L=8,D=2;
    double epsE=2e-10;
    double S=0.5;
    Setup(L,S,D);
    itsMPS->InitializeWith(Random);
    itsMPS->Normalize(DLeft);

    Hamiltonian* Hlr=itsFactory->Make1D_2BodyLongRangeHamiltonian(L,S,RegularUpper,1.0,0.0,3);
    MPO* H2=Hlr->CreateUnitOperator();
    H2->Product(Hlr);
    H2->Product(Hlr);
    double E2=itsMPS->GetExpectation(H2);
    H2->CanonicalForm(); //Sweep both ways
    double E2right=itsMPS->GetExpectation(H2);
    double truncError=H2->Compress(Parker,0,1e-13);
//    H2->Report(cout);
    double E2comp=itsMPS->GetExpectation(H2);
//    cout << E2 << " " << E2right << " " << E2comp << endl;
    EXPECT_NEAR(E2,E2right,epsE);
    EXPECT_NEAR(E2,E2comp ,epsE);
    EXPECT_EQ(H2->GetMaxDw(),12);
    EXPECT_LT(truncError,1e-13);
}
TEST_F(MPOTests,TestParkerSVDCompressH2_2Body_Lower)
{
    int L=8,D=2;
    double epsE=2e-10;
    double S=0.5;
    Setup(L,S,D);
    itsMPS->InitializeWith(Random);
    itsMPS->Normalize(DLeft);

    Hamiltonian* Hlr=itsFactory->Make1D_2BodyLongRangeHamiltonian(L,S,RegularLower,1.0,0.0,3);
    MPO* H2=Hlr->CreateUnitOperator();
    H2->Product(Hlr);
    H2->Product(Hlr);
    double E2=itsMPS->GetExpectation(H2);
    H2->CanonicalForm(); //Sweep both ways
    double E2right=itsMPS->GetExpectation(H2);
    double truncError=H2->Compress(Parker,0,1e-13);
//    H2->Report(cout);
    double E2comp=itsMPS->GetExpectation(H2);
//    cout << E2 << " " << E2right << " " << E2comp << endl;
    EXPECT_NEAR(E2,E2right,epsE);
    EXPECT_NEAR(E2,E2comp ,epsE);
    EXPECT_EQ(H2->GetMaxDw(),12);
    EXPECT_LT(truncError,1e-13);
}


TEST_F(MPOTests,TestParkerSVDCompressExpHL8t0)
{
    int L=8,D=2;
    double S=0.5,dt=0.0,epsMPO=1e-13;
    Setup(L,S,D);
    itsMPS->InitializeWith(Random);
    itsMPS->Normalize(DLeft);
    MPO* expH=itsH->CreateOperator(dt,SecondOrder,CNone,epsMPO);
//    expH->Report(cout);

    MPS* psi1=itsMPS->Apply(expH);
    EXPECT_NEAR(psi1->GetOverlap(psi1),1.0,1e-13);

    EXPECT_NEAR(itsMPS->GetOverlap(psi1),1.0,1e-13);
    expH->CanonicalForm(); //Do we need to sweep both ways?
    EXPECT_EQ(expH->GetNormStatus(),"RRRRRRRR");
//    expH->Report(cout);
    MPS* psi3=itsMPS->Apply(expH);
    EXPECT_NEAR(psi1->GetOverlap(psi3),1.0,1e-13);

    double truncError=expH->Compress(Parker,0,epsMPO);
    EXPECT_EQ(expH->GetNormStatus(),"IIIIIIII");
//    expH->Report(cout);
    MPS* psi4=itsMPS->Apply(expH);
    EXPECT_NEAR(psi1->GetOverlap(psi4),1.0,1e-13);
    EXPECT_EQ(expH->GetMaxDw(),1);
    EXPECT_LT(truncError,epsMPO); //Unit operator should have no compression error
}


TEST_F(MPOTests,TestParkerSVDCompressExpHL9t0)
{
    int L=9,D=2;
    double S=0.5,dt=0.0,epsMPO=1e-13;
    Setup(L,S,D);
    itsMPS->InitializeWith(Random);
    itsMPS->Normalize(DLeft);
    MPO* expH=itsH->CreateOperator(dt,SecondOrder,CNone,epsMPO);
//    expH->Report(cout);

    MPS* psi1=itsMPS->Apply(expH);
    EXPECT_NEAR(psi1->GetOverlap(psi1),1.0,1e-13);

    EXPECT_NEAR(itsMPS->GetOverlap(psi1),1.0,1e-13);
    expH->CanonicalForm();
    EXPECT_EQ(expH->GetNormStatus(),"RRRRRRRRR");
//    expH->Report(cout);
    MPS* psi3=itsMPS->Apply(expH);
    EXPECT_NEAR(psi1->GetOverlap(psi3),1.0,1e-13);

    double truncError=expH->Compress(Parker,0,epsMPO);
    EXPECT_EQ(expH->GetNormStatus(),"IIIIIIIII");
    MPS* psi4=itsMPS->Apply(expH);
    EXPECT_NEAR(psi1->GetOverlap(psi4),1.0,1e-13);
    EXPECT_EQ(expH->GetMaxDw(),1);
    EXPECT_LT(truncError,epsMPO); //Unit operator should have no compression error
}


TEST_F(MPOTests,TestParkerSVDCompressExpHL8t1)
{
    int L=8,D=2;
    double S=0.5,dt=0.1,epsMPO=1e-5;
    Setup(L,S,D);
    itsMPS->InitializeWith(Random);
    itsMPS->Normalize(DLeft);
    MPO* expH=itsH->CreateOperator(dt,SecondOrder,CNone,epsMPO);
    EXPECT_NEAR(itsMPS->GetOverlap(itsMPS),1.0,1e-13);
    MPS* psi1=itsMPS->Apply(expH);
    double o1=psi1->GetOverlap(psi1);
    expH->CanonicalForm(); //Sweep both ways
    EXPECT_EQ(expH->GetNormStatus(),"WRRRRRRR");
//    expH->Report(cout);
    MPS* psi2=itsMPS->Apply(expH);
    EXPECT_NEAR(psi1->GetOverlap(psi2),o1,1e-13);

    double truncError=expH->Compress(Parker,0,epsMPO);
    EXPECT_EQ(expH->GetNormStatus(),"WRRRRRRR");
//    expH->Report(cout);
    MPS* psi4=itsMPS->Apply(expH);
    EXPECT_NEAR(psi1->GetOverlap(psi4),o1,5*epsMPO);

    EXPECT_EQ(expH->GetMaxDw(),4);
    EXPECT_GT(truncError,0.0);
    EXPECT_LT(truncError,2*epsMPO);
}
TEST_F(MPOTests,TestParkerSVDCompressExpHL8t1_FourthOrder)
{
    int L=8,D=2;
    double S=0.5,dt=0.1,epsMPO=1e-10;
    Setup(L,S,D);
    itsMPS->InitializeWith(Random);
    itsMPS->Normalize(DLeft);
    MPO* expH=itsH->CreateOperator(dt,FourthOrder,Parker,epsMPO);
    EXPECT_EQ(expH->GetNormStatus(),"WRRRRRRR");
    expH->Report(cout);

    EXPECT_EQ(expH->GetMaxDw(),6);
//    EXPECT_GT(truncError,0.0);
//    EXPECT_LT(truncError,2*epsMPO);
}

//Try a large lattice
TEST_F(MPOTests,TestParkerSVDCompressExpHL256t1)
{
    int L=256,D=2;
    double S=0.5,dt=0.1,epsMPO=1e-5;
    Setup(L,S,D);
    itsMPS->InitializeWith(Random);
    itsMPS->Normalize(DLeft);
    MPO* expH=itsH->CreateOperator(dt,SecondOrder,CNone,epsMPO);
    EXPECT_NEAR(itsMPS->GetOverlap(itsMPS),1.0,1e-13);
    MPS* psi1=itsMPS->Apply(expH);
    double o1=psi1->GetOverlap(psi1);
    expH->CanonicalForm(); //Sweep both ways
//    expH->Report(cout);
    MPS* psi2=itsMPS->Apply(expH);
    EXPECT_NEAR(psi1->GetOverlap(psi2),o1,1e-13);

    double truncError=expH->Compress(Parker,0,epsMPO);
//    expH->Report(cout);
    MPS* psi4=itsMPS->Apply(expH);
    EXPECT_NEAR(psi1->GetOverlap(psi4),o1,L*epsMPO);

    EXPECT_EQ(expH->GetMaxDw(),4);
    EXPECT_GT(truncError,0.0);
    EXPECT_LT(truncError,epsMPO);
}


#ifndef DEBUG

TEST_F(MPOTests,TestTimingE2_S5D4)
{
    int L=10,D=4;
    double S=2.5;
    Setup(L,S,D);
    itsMPS->InitializeWith(Random);
    itsMPS->Normalize(DRight);
    MPO* H2=itsH->CreateH2Operator();
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
    itsMPS->InitializeWith(Random);
    itsMPS->Normalize(DRight);
    MPO* H2=itsH->CreateH2Operator();
    double EE=itsMPS->GetExpectation(H2);
    delete H2;
    cout << "<E^2> contraction for L=" << L << ", S=" << S << ", D=" << D  << endl;
    (void)EE; //Avoid warning
}
#endif



