#include "Tests.H"
//#include "TensorNetworks/Hamiltonian.H"
#include "TensorNetworks/iHamiltonian.H"
#include "TensorNetworks/iMPS.H"
#include "TensorNetworks/iMPO.H"
#include "TensorNetworks/Factory.H"
#include "TensorNetworks/SVCompressor.H"
#include "TensorNetworks/IterationSchedule.H"

using std::setw;
//using TensorNetworks::TriType;
using TensorNetworks::Random;
using TensorNetworks::Neel;
using TensorNetworks::DLeft;
using TensorNetworks::DRight;
using TensorNetworks::Gates;
//using TensorNetworks::Std;
using TensorNetworks::IterationSchedule;
using TensorNetworks::Epsilons;
using TensorNetworks::MPOForm;
using TensorNetworks::RegularLower;
using TensorNetworks::RegularUpper;
using TensorNetworks::iMPO;
//    typedef TensorNetworks::MatrixRT MatrixRT;
//    typedef TensorNetworks::Matrix4RT Matrix4RT;



class iVUMPSTests : public ::testing::Test
{
public:

    iVUMPSTests()
    : epsNorm(4e-11)
    , epsOrth(1e-11)
    , itsFactory(TensorNetworks::Factory::GetFactory())
    , itsiH(0)
    , itsiMPS(0)
    , itsCompressor(0)
    {
        StreamableObject::SetToPretty();
    }

    ~iVUMPSTests()
    {
        delete itsFactory;
        if (itsiH)         delete itsiH;
        if (itsiMPS)       delete itsiMPS;
        if (itsCompressor) delete itsCompressor;
    }

    void Setup(int L, double S, int D, double J, double hx, double epsSVD,MPOForm f)
    {
        if (itsiH)         delete itsiH;
        if (itsiMPS)       delete itsiMPS;
        if (itsCompressor) delete itsCompressor;
        itsiH=itsFactory->Make1D_NN_TransverseIsingiHamiltonian(1,S,f,J,hx);
        itsiMPS=itsiH->CreateiMPS(L,D,D*D*epsNorm,epsSVD);
        itsCompressor=itsFactory->MakeMPSCompressor(D,epsSVD);
    }


    double epsNorm,epsOrth;
    TensorNetworks::Factory*       itsFactory=TensorNetworks::Factory::GetFactory();
    TensorNetworks::iHamiltonian*  itsiH;
    TensorNetworks::iMPS*          itsiMPS;
    TensorNetworks::SVCompressorC* itsCompressor;
};


TEST_F(iVUMPSTests,TestSetup)
{
    int UnitCell=1,D=2;
    double S=0.5,J=-1.0,hx=0.0,epsSVD=0.0;
    MPOForm f=RegularLower;
    Setup(UnitCell,S,D,J,hx,epsSVD,f);
    itsiMPS->InitializeWith(Random);
    EXPECT_EQ(itsiMPS->GetNormStatus(),"A"); //We left norm in init
}

TEST_F(iVUMPSTests,TestNormQR_D2_L1)
{
    int UnitCell=1,D=2;
    double J=-1.0,hx=0.0,epsSVD=0.0;
    MPOForm f=RegularLower;
    for (double S=0.5;S<=2.5;S+=0.5)
    {
        Setup(UnitCell,S,D,J,hx,epsSVD,f);
        itsiMPS->InitializeWith(Random);
        itsiMPS->NormalizeQR(DLeft);
        EXPECT_EQ(itsiMPS->GetNormStatus(),"A");
        itsiMPS->NormalizeQR(DRight);
        EXPECT_EQ(itsiMPS->GetNormStatus(),"B");
    }
}

TEST_F(iVUMPSTests,TestNormQR_D6_L1)
{
    int UnitCell=1,D=6;
    double J=-1.0,hx=0.0,epsSVD=0.0;
    MPOForm f=RegularLower;
    for (double S=0.5;S<=2.5;S+=0.5)
    {
        Setup(UnitCell,S,D,J,hx,epsSVD,f);
        itsiMPS->InitializeWith(Random);
        itsiMPS->NormalizeQR(DLeft);
        EXPECT_EQ(itsiMPS->GetNormStatus(),"A");
        itsiMPS->NormalizeQR(DRight);
        EXPECT_EQ(itsiMPS->GetNormStatus(),"B");
    }
}

TEST_F(iVUMPSTests,TestNormQR_D6_L10)
{
    int UnitCell=10,D=6;
    double J=-1.0,hx=0.0,epsSVD=0.0;
    MPOForm f=RegularLower;
    for (double S=0.5;S<=2.5;S+=0.5)
    {
        Setup(UnitCell,S,D,J,hx,epsSVD,f);
        itsiMPS->InitializeWith(Random);
        itsiMPS->NormalizeQR(DLeft);
        EXPECT_EQ(itsiMPS->GetNormStatus(),"AAAAAAAAAA");
        itsiMPS->NormalizeQR(DRight);
        EXPECT_EQ(itsiMPS->GetNormStatus(),"BBBBBBBBBB");
    }
}
//
// L=1 ground states
//
TEST_F(iVUMPSTests,TestFindFerroGS_S12_D2_L1_h01)
{
    int UnitCell=1,D=2,maxIter=30;
    double S=0.5,J=-1.0,hx=0.1,epsSVD=0.0,epsE=1e-14;
    MPOForm f=RegularLower;
    Setup(UnitCell,S,D,J,hx,epsSVD,f);
    itsiMPS->InitializeWith(Random);
    Epsilons eps(epsE);
    eps.itsDeltaLambdaEpsilon=1e-8;

    IterationSchedule is;
    is.Insert({maxIter,D,eps});
    double Eeigen=itsiMPS->FindVariationalGroundState(itsiH,is);
    double Eex   =itsiMPS->GetExpectation(itsiH);
    EXPECT_NEAR(Eeigen,Eex,2e-6); //Using R=G*G_dagger for trandfer matrix eigen vectors reduces eigen value precision
    EXPECT_NEAR(Eex   ,-0.25250795696355421,epsE);
}
TEST_F(iVUMPSTests,TestFindFerroGS_S12_D4_L1_h01)
{
    int UnitCell=1,D=4,maxIter=30;
    double S=0.5,J=-1.0,hx=0.1,epsSVD=0.0,epsE=1e-14;
    MPOForm f=RegularLower;
    Setup(UnitCell,S,D,J,hx,epsSVD,f);
    itsiMPS->InitializeWith(Random);
    Epsilons eps(epsE);
    eps.itsDeltaLambdaEpsilon=1e-8;

    IterationSchedule is;
    is.Insert({maxIter,D,eps});
    double Eeigen=itsiMPS->FindVariationalGroundState(itsiH,is);
    double Eex   =itsiMPS->GetExpectation(itsiH);
    EXPECT_NEAR(Eeigen,Eex,2e-6);
    EXPECT_NEAR(Eex,-0.25250795716599872,epsE);
}

TEST_F(iVUMPSTests,TestFindFerroGS_S12_D8_L1_h04)
{
    int UnitCell=1,D=8,maxIter=100;
    double S=0.5,J=-1.0,hx=0.4,epsSVD=0.0,epsE=1e-12;
    MPOForm f=RegularLower;
    Setup(UnitCell,S,D,J,hx,epsSVD,f);
    itsiMPS->InitializeWith(Random);
    Epsilons eps(epsE);
    eps.itsDeltaLambdaEpsilon=1e-10;

    IterationSchedule is;
    is.Insert({maxIter,D,eps});
    double Eeigen=itsiMPS->FindVariationalGroundState(itsiH,is);
    double Eex   =itsiMPS->GetExpectation(itsiH);
    EXPECT_NEAR(Eeigen,Eex,2e-3);
    EXPECT_NEAR(Eex,-0.2931323911399,epsE);
}

TEST_F(iVUMPSTests,TestFindFerroGS_S10_D8_L1_h04)
{
    int UnitCell=1,D=8,maxIter=100;
    double S=1.0,J=-1.0,hx=0.4,epsSVD=0.0,epsE=1e-12;
    MPOForm f=RegularLower;
    Setup(UnitCell,S,D,J,hx,epsSVD,f);
    itsiMPS->InitializeWith(Random);
    Epsilons eps(epsE);
    eps.itsDeltaLambdaEpsilon=1e-9;

    IterationSchedule is;
    is.Insert({maxIter,D,eps});
    double Eeigen=itsiMPS->FindVariationalGroundState(itsiH,is);
    double Eex   =itsiMPS->GetExpectation(itsiH);
    EXPECT_NEAR(Eeigen,Eex,5e-5);
    EXPECT_NEAR(Eex,-1.0401827662967,epsE);
}

TEST_F(iVUMPSTests,TestFindFerroGS_S32_D8_L1_h04)
{
    int UnitCell=1,D=8,maxIter=100;
    double S=1.5,J=-1.0,hx=0.4,epsSVD=0.0,epsE=1e-12;
    MPOForm f=RegularLower;
    Setup(UnitCell,S,D,J,hx,epsSVD,f);
    itsiMPS->InitializeWith(Random);
    Epsilons eps(epsE);
    eps.itsDeltaLambdaEpsilon=5e-9;

    IterationSchedule is;
    is.Insert({maxIter,D,eps});
    double Eeigen=itsiMPS->FindVariationalGroundState(itsiH,is);
    double Eex   =itsiMPS->GetExpectation(itsiH);
    EXPECT_NEAR(Eeigen,Eex,2e-5);
    EXPECT_NEAR(Eex,-2.2900520735365,epsE);
}
//
// L=2 ground states
//
TEST_F(iVUMPSTests,TestFindFerroGS_S12_D2_L2_h01)
{
    int UnitCell=2,D=2,maxIter=30;
    double S=0.5,J=-1.0,hx=0.1,epsSVD=0.0,epsE=1e-14;
    MPOForm f=RegularLower;
    Setup(UnitCell,S,D,J,hx,epsSVD,f);
    itsiMPS->InitializeWith(Random);
    Epsilons eps(epsE);
    eps.itsDeltaLambdaEpsilon=1e-8;

    IterationSchedule is;
    is.Insert({maxIter,D,eps});
    double Eeigen=itsiMPS->FindVariationalGroundState(itsiH,is);
    double Eex   =itsiMPS->GetExpectation(itsiH);
    EXPECT_NEAR(Eeigen,Eex,2e-6);
    EXPECT_NEAR(Eex,-0.25250795696355421,epsE);
}

TEST_F(iVUMPSTests,TestFindAFGS_S12_D2_L2_h01)
{
    int UnitCell=2,D=2,maxIter=30;
    double S=0.5,J=-1.0,hx=0.1,epsSVD=0.0,epsE=1e-14;
    MPOForm f=RegularLower;
    Setup(UnitCell,S,D,J,hx,epsSVD,f);
    itsiMPS->InitializeWith(Random);
    Epsilons eps(epsE);
    eps.itsDeltaLambdaEpsilon=1e-8;

    IterationSchedule is;
    is.Insert({maxIter,D,eps});
    double Eeigen=itsiMPS->FindVariationalGroundState(itsiH,is);
    double Eex   =itsiMPS->GetExpectation(itsiH);
    EXPECT_NEAR(Eeigen,Eex,2e-6);
    EXPECT_NEAR(Eex,-0.25250795696355421,epsE);
}

