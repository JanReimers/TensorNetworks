#include "Tests.H"
#include "TensorNetworks/Hamiltonian.H"
#include "TensorNetworks/IterationSchedule.H"
#include "TensorNetworks/MPO.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/Factory.H"

//#include "oml/stream.h"
//#include "oml/stopw.h"

using std::setw;

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

    void Setup(int L, double S, int D)
    {
        itsH=itsFactory->Make1D_NN_HeisenbergHamiltonian(L,S,1.0,1.0,0.0);
        itsMPS=itsH->CreateMPS(D,1e-12,1e-12);
    }


    double eps;
    TensorNetworks::Factory*          itsFactory;
    TensorNetworks::Hamiltonian*      itsH;
    TensorNetworks::MPS*              itsMPS;
};


TEST_F(VariationalGroundStateTests,TestIdentityOperator)
{
    Setup(10,0.5,2);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::DLeft);
    TensorNetworks::MPO* IO=itsH->CreateUnitOperator();
    double S=itsMPS->GetExpectation(IO);
    EXPECT_NEAR(S,1.0,eps);
    delete IO;
}


TEST_F(VariationalGroundStateTests,TestSweepL9S1D2)
{
    int L=9,D=2,maxIter=100;
    double S=0.5;
    Setup(L,S,D);
    itsMPS->InitializeWith(TensorNetworks::Random);

    TensorNetworks::Epsilons eps(1e-12);
    eps.itsDelatEnergy1Epsilon=1e-9;
    TensorNetworks::IterationSchedule is;
    is.Insert({maxIter,D,eps});

    int nSweep=itsMPS->FindVariationalGroundState(itsH,is);

    double E=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(E/(L-1),-0.45317425,1e-7);
    EXPECT_LT(nSweep,maxIter);
}

#ifndef DEBUG
TEST_F(VariationalGroundStateTests,TestSweepL9S1D8)
{
    int L=9,D=8,maxIter=100;
    double S=0.5;
    Setup(L,S,D);
    itsMPS->InitializeWith(TensorNetworks::Random);

    TensorNetworks::Epsilons eps(1e-12);
    eps.itsDelatEnergy1Epsilon=1e-9;
    TensorNetworks::IterationSchedule is;
    is.Insert({maxIter,D,eps});

    int nSweep=itsMPS->FindVariationalGroundState(itsH,is);

    double E=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(E/(L-1),-0.46703753,1e-7);
    EXPECT_LT(nSweep,maxIter);
}
#endif

TEST_F(VariationalGroundStateTests,TestSweepL9S1D4)
{
    int L=9,D=4,maxIter=100;
    double S=0.5;
    Setup(L,S,D);
    itsMPS->InitializeWith(TensorNetworks::Random);

    TensorNetworks::Epsilons eps(1e-12);
    eps.itsDelatEnergy1Epsilon=1e-10;
    TensorNetworks::IterationSchedule is;
    is.Insert({maxIter,D,eps});

    int nSweep=itsMPS->FindVariationalGroundState(itsH,is);

    double E=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(E/(L-1),-0.46664265599414939,1e-7);
    EXPECT_LT(nSweep,maxIter);
}

TEST_F(VariationalGroundStateTests,TestFreezeL9S1D2)
{
    int L=9,D=2,maxIter=100;
    double S=0.5;
    Setup(L,S,D);
    itsMPS->InitializeWith(TensorNetworks::Neel);
    itsMPS->Freeze(1,0.5); //Site 0 spin up

    TensorNetworks:: Epsilons eps(1e-12);
    eps.itsDelatEnergy1Epsilon=1e-9;
    TensorNetworks::IterationSchedule is;
    is.Insert({maxIter,D,eps});
    int nSweep=itsMPS->FindVariationalGroundState(itsH,is);

    double E=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(E/(L-1),-0.45317425 ,1e-7);
    EXPECT_LT(nSweep,maxIter);
}

#ifndef DEBUG

TEST_F(VariationalGroundStateTests,TestSweepL9S5D2)
{
    int L=9,D=2,maxIter=100;
    double S=2.5;
    Setup(L,S,D);
    itsMPS->InitializeWith(TensorNetworks::Random);

    TensorNetworks::Epsilons eps(1e-12);
    eps.itsDelatEnergy1Epsilon=1e-9;
    TensorNetworks::IterationSchedule is;
    is.Insert({maxIter,D,eps});
    int nSweep=itsMPS->FindVariationalGroundState(itsH,is);

    double E=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(E/(L-1),-7.025661 ,1e-7);
    EXPECT_LT(nSweep,maxIter);
}
#endif

#ifdef RunLongTests

TEST_F(VariationalGroundStateTests,TestSweepL6S1GrowD27)
{
    int L=6,Dstart=2,Dend=27;
    double S=1.0;
    Setup(L,S,Dstart);
    itsMPS->InitializeWith(TensorNetworks::Random);

    TensorNetworks::Epsilons eps(1e-12);
    eps.itsDelatEnergy1Epsilon=1e-6;
    TensorNetworks::IterationSchedule is;
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



TEST_F(VariationalGroundStateTests,TestSweepL7S1GrowD27)
{
    int L=7,Dstart=2,Dend=27;
    double S=1.0;
    Setup(L,S,Dstart);
    itsMPS->InitializeWith(TensorNetworks::Random);

    TensorNetworks::Epsilons eps(1e-12);
    eps.itsDelatEnergy1Epsilon=1e-6;
    TensorNetworks::IterationSchedule is;
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
    EXPECT_NEAR(E/(L-1), -1.4390886637843641 ,1e-8);
}

TEST_F(VariationalGroundStateTests,TestSweepL8S1GrowD27)
{
    int L=8,Dstart=2,Dend=27;
    double S=1.0;
    Setup(L,S,Dstart);
    itsMPS->InitializeWith(TensorNetworks::Random);

    TensorNetworks::Epsilons eps(1e-12);
    eps.itsDelatEnergy1Epsilon=1e-6;
    TensorNetworks::IterationSchedule is;
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


/*
TEST_F(GroundStateTesting,TestSweepL19S5D5)
{
    int L=19,Dstart=1,Dend=5,maxIter=100;
    double S=2.5;
    Setup(L,S,Dstart);
    itsMPS->InitializeWith(TensorNetworks::Random);

    Epsilons eps(1e-12);
    eps.itsDelatEnergy1Epsilon=1e-6;
    IterationSchedule is;
    is.Insert({maxIter,Dend,eps});
    eps.itsDelatEnergy1Epsilon=1e-7;
    is.Insert({maxIter,Dend,eps});
    eps.itsDelatEnergy1Epsilon=1e-9;
    is.Insert({maxIter,Dend,eps});

    StopWatch sw;
    sw.Start();

    itsMPS->FindVariationalGroundState(itsH,is);
    sw.Stop();
    cout << "FindGroundState for L=" << L << ", S=" << S << ", D=" << Dend << " took " << sw.GetTime() << " seconds." << endl;

    double E=itsMPS->GetExpectation(itsH);
// We used to be able to hit this value, but not any longer
//    EXPECT_NEAR(E/(L-1),-7.1969843668336484 ,1e-8);
    EXPECT_NEAR(E/(L-1),-7.1966181668666467,1e-8);
//  OK now get the lower values again!!
//    EXPECT_NEAR(E/(L-1),-7.1969843670044185,1e-8);
    // OK this test seems to randomly land in a false minimim, so lets comment it out for now.

}

*/

#ifdef RunLongTests

TEST_F(VariationalGroundStateTests,TestSweepL19S1D8)
{
    int L=19,D=8,maxIter=100;
    double S=0.5;
    Setup(L,S,D);
    itsMPS->InitializeWith(TensorNetworks::Random);

    TensorNetworks::Epsilons eps(1e-12);
    eps.itsDelatEnergy1Epsilon=1e-9;
    TensorNetworks::IterationSchedule is;
    is.Insert({maxIter,D,eps});

    StopWatch sw;
    sw.Start();
    int nSweep=itsMPS->FindVariationalGroundState(itsH,is);
    sw.Stop();
    cout << "FindGroundState for L=" << L << ", S=" << S << ", D=" << D << " took " << sw.GetTime() << " seconds." << endl;

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




