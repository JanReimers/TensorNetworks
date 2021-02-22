#include "Tests.H"
#include "TensorNetworks/MPS.H"
#include "TensorNetworks/Hamiltonian.H"
#include "TensorNetworks/IterationSchedule.H"
#include "TensorNetworks/MPO.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/Factory.H"

using std::setw;
using TensorNetworks::Random;
using TensorNetworks::Neel;
using TensorNetworks::DLeft;
using TensorNetworks::DRight;
using TensorNetworks::IterationSchedule;
using TensorNetworks::Epsilons;
using TensorNetworks::MPO;
using TensorNetworks::TriType;

class VariationalGroundStateTests : public ::testing::Test
{
public:
    VariationalGroundStateTests()
    : eps(1.0e-13)
    , itsFactory(TensorNetworks::Factory::GetFactory())
    , itsH(0)
    , itsMPS(0)
    {
        StreamableObject::SetToPretty();
    }

    ~VariationalGroundStateTests()
    {
        delete itsFactory;
        if (itsH) delete itsH;
        if (itsMPS) delete itsMPS;
    }
// Heisenberg Hamiltonian
    void Setup(int L, double S, int D, TriType ul)
    {
        itsH=itsFactory->Make1D_NN_HeisenbergHamiltonian(L,S,1.0,1.0,0.0,ul);
        itsMPS=itsH->CreateMPS(D,1e-12,1e-12);
    }
// Transverse Ising Hamiltonian
    void SetupTI(int L, double S, int D, double hx, TriType ul)
    {
        itsH=itsFactory->Make1D_NN_TransverseIsingHamiltonian(L,S,1.0,hx,ul);
        itsMPS=itsH->CreateMPS(D,1e-12,1e-12);
    }


    double eps;
    TensorNetworks::Factory*          itsFactory;
    TensorNetworks::Hamiltonian*      itsH;
    TensorNetworks::MPS*              itsMPS;
};


TEST_F(VariationalGroundStateTests,TestIdentityOperator)
{
    Setup(10,0.5,2,Lower);
    itsMPS->InitializeWith(Random);
    itsMPS->Normalize(DLeft);
    MPO* IO=itsH->CreateUnitOperator();
    double S=itsMPS->GetExpectation(IO);
    EXPECT_NEAR(S,1.0,eps);
    delete IO;
}

TEST_F(VariationalGroundStateTests,TestSweep_Lower_L2S1D2)
{
    int L=2,D=2,maxIter=100;
    double S=0.5;
    Setup(L,S,D,Lower);
    itsMPS->InitializeWith(Random);

    Epsilons eps(1e-12);
    eps.itsDelatEnergy1Epsilon=1e-9;
    IterationSchedule is;
    is.Insert({maxIter,D,eps});

    int nSweep=itsMPS->FindVariationalGroundState(itsH,is);

    double E1=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(E1/(L-1),-0.75,1e-7);
    EXPECT_LT(nSweep,maxIter);

    MPO* H2=itsH->CreateH2Operator();
    double E2=itsMPS->GetExpectation(H2);
    EXPECT_EQ(H2->GetMaxDw(),4);
    EXPECT_NEAR(E2,E1*E1,1e-14);
}

TEST_F(VariationalGroundStateTests,TestSweep_Upper_L2S1D2)
{
    int L=2,D=2,maxIter=100;
    double S=0.5;
    Setup(L,S,D,Lower);
    itsMPS->InitializeWith(Random);

    Epsilons eps(1e-12);
    eps.itsDelatEnergy1Epsilon=1e-9;
    IterationSchedule is;
    is.Insert({maxIter,D,eps});

    int nSweep=itsMPS->FindVariationalGroundState(itsH,is);

    double E1=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(E1/(L-1),-0.75,1e-7);
    EXPECT_LT(nSweep,maxIter);

    MPO* H2=itsH->CreateH2Operator();
    double E2=itsMPS->GetExpectation(H2);
    EXPECT_EQ(H2->GetMaxDw(),4);
    EXPECT_NEAR(E2,E1*E1,1e-14);
}


TEST_F(VariationalGroundStateTests,TestSweep_Lower_L9S1D2)
{
    int L=9,D=2,maxIter=100;
    double S=0.5;
    Setup(L,S,D,Lower);
    itsMPS->InitializeWith(Random);

    Epsilons eps(1e-12);
    eps.itsDelatEnergy1Epsilon=1e-9;
    IterationSchedule is;
    is.Insert({maxIter,D,eps});

    int nSweep=itsMPS->FindVariationalGroundState(itsH,is);

    double E=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(E/(L-1),-0.45317425,1e-7);
    EXPECT_LT(nSweep,maxIter);
}
TEST_F(VariationalGroundStateTests,TestSweep_Upper_L9S1D2)
{
    int L=9,D=2,maxIter=100;
    double S=0.5;
    Setup(L,S,D,Upper);
    itsMPS->InitializeWith(Random);
    double Eupper=itsMPS->GetExpectation(itsH);
    TensorNetworks::Hamiltonian* Hlower=itsFactory->Make1D_NN_HeisenbergHamiltonian(L,S,1.0,1.0,0.0,Lower);
    double Elower=itsMPS->GetExpectation(Hlower);
    EXPECT_NEAR(Eupper,Elower,1e-15);

    Epsilons eps(1e-12);
    eps.itsDelatEnergy1Epsilon=1e-9;
    IterationSchedule is;
    is.Insert({maxIter,D,eps});

    int nSweep=itsMPS->FindVariationalGroundState(itsH,is);

    double E=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(E/(L-1),-0.45317425,1e-7);
    EXPECT_LT(nSweep,maxIter);
}

#ifndef DEBUG

TEST_F(VariationalGroundStateTests,TestSweep_Lower_L9S1D8_growD)
{
    int L=9,Dstart=2,D=8,maxIter=100;
    double S=0.5;
    Setup(L,S,Dstart,Lower);
    itsMPS->InitializeWith(Random);

    Epsilons eps(1e-12);
    eps.itsDelatEnergy1Epsilon=1.0;
    eps.itsDeltaLambdaEpsilon =1e-9; //Converge on wave function only.
    IterationSchedule is;
    is.Insert({maxIter,D,eps});

    int nSweep=itsMPS->FindVariationalGroundState(itsH,is);

    double E=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(E/(L-1),-0.46703753899,1e-10);
    EXPECT_LT(nSweep,maxIter);
}

TEST_F(VariationalGroundStateTests,TestSweep_Lower_L9S1D8)
{
    int L=9,D=8,maxIter=100;
    double S=0.5;
    Setup(L,S,D,Lower);
    itsMPS->InitializeWith(Random);

    Epsilons eps(1e-12);
    eps.itsDelatEnergy1Epsilon=1e-9;
    IterationSchedule is;
    is.Insert({maxIter,D,eps});

    int nSweep=itsMPS->FindVariationalGroundState(itsH,is);

    double E=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(E/(L-1),-0.46703753,1e-7);
    EXPECT_LT(nSweep,maxIter);
}
#endif

TEST_F(VariationalGroundStateTests,TestSweep_Lower_L9S1D4)
{
    int L=9,D=4,maxIter=100;
    double S=0.5;
    Setup(L,S,D,Lower);
    itsMPS->InitializeWith(Random);

    Epsilons eps(1e-12);
    eps.itsDelatEnergy1Epsilon=1e-10;
    IterationSchedule is;
    is.Insert({maxIter,D,eps});

    int nSweep=itsMPS->FindVariationalGroundState(itsH,is);

    double E=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(E/(L-1),-0.46664265599414939,1e-7);
    EXPECT_LT(nSweep,maxIter);
}
TEST_F(VariationalGroundStateTests,TestSweep_Upper_L9S1D4)
{
    int L=9,D=4,maxIter=100;
    double S=0.5;
    Setup(L,S,D,Upper);
    itsMPS->InitializeWith(Random);

    Epsilons eps(1e-12);
    eps.itsDelatEnergy1Epsilon=1e-10;
    IterationSchedule is;
    is.Insert({maxIter,D,eps});

    int nSweep=itsMPS->FindVariationalGroundState(itsH,is);

    double E=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(E/(L-1),-0.46664265599414939,1e-7);
    EXPECT_LT(nSweep,maxIter);
}

TEST_F(VariationalGroundStateTests,TestFreeze_Lower_L9S1D2)
{
    int L=9,D=2,maxIter=100;
    double S=0.5;
    Setup(L,S,D,Lower);
    itsMPS->InitializeWith(Neel);
    itsMPS->Freeze(1,0.5); //Site 0 spin up

     Epsilons eps(1e-12);
    eps.itsDelatEnergy1Epsilon=1e-9;
    IterationSchedule is;
    is.Insert({maxIter,D,eps});
    int nSweep=itsMPS->FindVariationalGroundState(itsH,is);

    double E=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(E/(L-1),-0.45317425 ,1e-7);
    EXPECT_LT(nSweep,maxIter);
}

#ifndef DEBUG

TEST_F(VariationalGroundStateTests,TestSweep_Lower_L9S5D2)
{
    int L=9,D=2,maxIter=100;
    double S=2.5;
    Setup(L,S,D,Lower);
    itsMPS->InitializeWith(Random);

    Epsilons eps(1e-12);
    eps.itsDelatEnergy1Epsilon=1e-9;
    IterationSchedule is;
    is.Insert({maxIter,D,eps});
    int nSweep=itsMPS->FindVariationalGroundState(itsH,is);

    double E=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(E/(L-1),-7.025661 ,1e-7);
    EXPECT_LT(nSweep,maxIter);
}
#endif

#ifdef RunLongTests

TEST_F(VariationalGroundStateTests,TestSweep_Lower_L6S1GrowD27)
{
    int L=6,Dstart=2,Dend=27;
    double S=1.0;
    Setup(L,S,Dstart,Lower);
    itsMPS->InitializeWith(Random);

    Epsilons eps(1e-12);
    eps.itsDelatEnergy1Epsilon=1e-6;
    IterationSchedule is;
    is.Insert({30,9,1,eps});
    is.Insert({30,Dend,6,eps});
    eps.itsDelatEnergy1Epsilon=1e-7;
    is.Insert({30,Dend,eps});
    eps.itsDelatEnergy1Epsilon=1e-8;
    is.Insert({30,Dend,eps});
    eps.itsDelatEnergy1Epsilon=1e-9;
    is.Insert({30,Dend,eps});
//    eps.itsDelatEnergy1Epsilon=1e-10;
//    is.Insert({30,Dend,eps});
    itsMPS->FindVariationalGroundState(itsH,is);

    double E=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(E/(L-1), -1.4740549939 ,1e-8);
}



TEST_F(VariationalGroundStateTests,TestSweep_Upper_L7S1GrowD27)
{
    int L=7,Dstart=2,Dend=27;
    double S=1.0;
    Setup(L,S,Dstart,Upper);
    itsMPS->InitializeWith(Random);

    Epsilons eps(1e-12);
    eps.itsDelatEnergy1Epsilon=1e-6;
    IterationSchedule is;
    is.Insert({30,9,1,eps});
    is.Insert({30,Dend,6,eps});
    eps.itsDelatEnergy1Epsilon=1e-7;
    is.Insert({30,Dend,eps});
    itsMPS->FindVariationalGroundState(itsH,is);

    double E=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(E/(L-1), -1.4390886637843641 ,1e-8);
}

TEST_F(VariationalGroundStateTests,TestSweep_Lower_L8S1GrowD27)
{
    int L=8,Dstart=2,Dend=27;
    double S=1.0;
    Setup(L,S,Dstart,Lower);
    itsMPS->InitializeWith(Random);

    Epsilons eps(1e-12);
    eps.itsDelatEnergy1Epsilon=1e-6;
    IterationSchedule is;
    is.Insert({30,9,1,eps});
    is.Insert({30,Dend,6,eps});
    eps.itsDelatEnergy1Epsilon=1e-7;
    is.Insert({30,Dend,eps});
    eps.itsDelatEnergy1Epsilon=1e-8;
    is.Insert({30,Dend,eps});
    eps.itsDelatEnergy1Epsilon=1e-9;
    is.Insert({30,Dend,eps});
//    eps.itsDelatEnergy1Epsilon=1e-10;
//    is.Insert({30,Dend,eps});
    itsMPS->FindVariationalGroundState(itsH,is);

    double E=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(E/(L-1), -1.4463763812511536,1e-8);
}
#endif //RunLongTests

#ifdef RunLongTests

TEST_F(VariationalGroundStateTests,TestSweep_Upper_L19S1D8)
{
    int L=19,D=8,maxIter=100;
    double S=0.5;
    Setup(L,S,D,Upper);
    itsMPS->InitializeWith(Random);

    Epsilons eps(1e-12);
    eps.itsDelatEnergy1Epsilon=1e-9;
    IterationSchedule is;
    is.Insert({maxIter,D,eps});

    int nSweep=itsMPS->FindVariationalGroundState(itsH,is);
    cout << "FindGroundState for L=" << L << ", S=" << S << ", D=" << D  << endl;

    double E=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(E/(L-1),-0.45535447609272839,1e-7);
    EXPECT_LT(nSweep,maxIter);
}


#endif


/*
TEST_F(GroundStateTesting,TestNeelStateSurvey)
{
    cout << " NSweep  L  S  D     E/L    E/(LS^2) " << endl;

    for (int S2=1; S2<=5; S2++)
        for (int L=3; L<=19; L++)
            for (int D=1; D<=8; D*=2)
            {
                Setup(L,S2,D);
                itsMPS->InitializeWith(MatrixProductSite::Random);
                int nSweep=itsMPS->FindVariationalGroundState(itsHamiltonianMPO,100,1e-8);
                double E=itsMPS->GetExpectationIterate(itsHamiltonianMPO);

                cout
                << setw(4) << nSweep << " "
                << setw(2) << L << " "
                << setw(2) << S2 << "/2 "
                << setw(2) << D << " " << std::fixed
                << setw(12) << std::setprecision(6) << E/(L-1)
                << setw(12) << std::setprecision(6) << E/((L-1)*0.25*S2*S2)
                << endl;
            }
}
*/

TEST_F(VariationalGroundStateTests,TestTransverIsing_Lower_L2S1D2Hx0)
{
    int L=2,D=2,maxIter=100;
    double S=0.5,hx=0.0;
    SetupTI(L,S,D,hx,Lower);
    itsMPS->InitializeWith(Random);

    Epsilons eps(1e-12);
    eps.itsDelatEnergy1Epsilon=1e-9;
    IterationSchedule is;
    is.Insert({maxIter,D,eps});

    int nSweep=itsMPS->FindVariationalGroundState(itsH,is);

    double E1=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(E1/(L-1),-0.25,1e-7);
    EXPECT_LT(nSweep,maxIter);

    MPO* H2=itsH->CreateH2Operator();
    double E2=itsMPS->GetExpectation(H2);
    EXPECT_EQ(H2->GetMaxDw(),1); //if hx=0 it compresses a lot
    EXPECT_NEAR(E2,E1*E1,1e-14);
}

TEST_F(VariationalGroundStateTests,TestTransverIsing_Upper_L9S1D2Hx0)
{
    int L=9,D=2,maxIter=100;
    double S=0.5,hx=0.0;
    SetupTI(L,S,D,hx,Upper);
    itsMPS->InitializeWith(Random);

    Epsilons eps(1e-12);
    eps.itsDelatEnergy1Epsilon=1e-9;
    IterationSchedule is;
    is.Insert({maxIter,D,eps});

    int nSweep=itsMPS->FindVariationalGroundState(itsH,is);

    double E1=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(E1/(L-1),-0.25,1e-7);
    EXPECT_LT(nSweep,maxIter);

    MPO* H2=itsH->CreateH2Operator();
    double E2=itsMPS->GetExpectation(H2);
    EXPECT_EQ(H2->GetMaxDw(),5); //if hx=0 it compresses a lot
    EXPECT_NEAR(E2,E1*E1,2e-14);
}


TEST_F(VariationalGroundStateTests,TestTransverIsing_Lower_L9S1D2Hx1)
{
    int L=9,D=2,maxIter=100;
    double S=0.5,hx=1.0;
    SetupTI(L,S,D,hx,Lower);
    itsMPS->InitializeWith(Random);

    Epsilons eps(1e-12);
    eps.itsDelatEnergy1Epsilon=1e-9;
    IterationSchedule is;
    is.Insert({maxIter,D,eps});

    int nSweep=itsMPS->FindVariationalGroundState(itsH,is);

    double E1=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(E1/(L-1),-0.594125875,1e-7);
    EXPECT_LT(nSweep,maxIter);

    MPO* H2=itsH->CreateH2Operator();
    double E2=itsMPS->GetExpectation(H2);
    EXPECT_EQ(H2->GetMaxDw(),5); //if hx=0 it compresses a lot
    EXPECT_NEAR(E2,E1*E1,1e-4);
}

TEST_F(VariationalGroundStateTests,TestTransverIsing_Lower_L9S1D8Hx1)
{
    int L=9,DStart=2,D=8,maxIter=100;
    double S=0.5,hx=1.0;
    SetupTI(L,S,DStart,hx,Lower);
    itsMPS->InitializeWith(Random);

    Epsilons eps(1e-12);
    eps.itsDelatEnergy1Epsilon=1e-9;
    IterationSchedule is;
    is.Insert({maxIter,DStart,eps});
    is.Insert({maxIter,D,eps});

    int nSweep=itsMPS->FindVariationalGroundState(itsH,is);

    double E1=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(E1/(L-1),-0.5941321968,1e-7);
    EXPECT_LT(nSweep,maxIter);

    MPO* H2=itsH->CreateH2Operator();
    double E2=itsMPS->GetExpectation(H2);
    EXPECT_EQ(H2->GetMaxDw(),5); //if hx=0 it compresses a lot
    EXPECT_NEAR(E2,E1*E1,1e-13);
}

TEST_F(VariationalGroundStateTests,TestTransverIsing_Upper_L9S1D8Hx1)
{
    int L=9,DStart=2,D=8,maxIter=100;
    double S=0.5,hx=1.0;
    SetupTI(L,S,DStart,hx,Upper);
    itsMPS->InitializeWith(Random);

    Epsilons eps(1e-12);
    eps.itsDelatEnergy1Epsilon=1e-9;
    IterationSchedule is;
    is.Insert({maxIter,DStart,eps});
    is.Insert({maxIter,D,eps});

    int nSweep=itsMPS->FindVariationalGroundState(itsH,is);

    double E1=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(E1/(L-1),-0.5941321968,1e-7);
    EXPECT_LT(nSweep,maxIter);

    MPO* H2=itsH->CreateH2Operator();
    double E2=itsMPS->GetExpectation(H2);
    EXPECT_EQ(H2->GetMaxDw(),5); //if hx=0 it compresses a lot
    EXPECT_NEAR(E2,E1*E1,1e-13);
}







