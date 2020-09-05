#include "Tests.H"
#include "TensorNetworks/Hamiltonian.H"
#include "TensorNetworks/IterationSchedule.H"
#include "TensorNetworks/OperatorWRepresentation.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/Factory.H"
#include "TensorNetworksImp/SPDLogger.H"

#include "oml/stream.h"
#include "oml/stopw.h"

using std::setw;

class GroundStateTesting : public ::testing::Test
{
public:
    GroundStateTesting()
    : eps(1.0e-13)
    , itsFactory(TensorNetworks::Factory::GetFactory())
    , itsLogger(new SPDLogger(0))
    {
        StreamableObject::SetToPretty();

    }

    void Setup(int L, double S, int D)
    {
        itsH=itsFactory->Make1D_NN_HeisenbergHamiltonian(L,S,1.0,1.0,0.0);
        itsMPS=itsH->CreateMPS(D,1e-12,itsLogger);
    }


    double eps;
    const TensorNetworks::Factory* itsFactory=TensorNetworks::Factory::GetFactory();
    Hamiltonian*         itsH;
    MPS*                 itsMPS;
    TNSLogger*           itsLogger;
};


TEST_F(GroundStateTesting,TestIdentityOperator)
{
    Setup(10,0.5,2);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::DLeft);
    OperatorWRepresentation* IWO=itsFactory->MakeIdentityOperator();
    Operator* IO=itsH->CreateOperator(IWO);
    double S=itsMPS->GetExpectation(IO);
    EXPECT_NEAR(S,1.0,eps);

}


TEST_F(GroundStateTesting,TestSweepL9S1D2)
{
    int L=9,D=2,maxIter=100;
    double S=0.5;
    Setup(L,S,D);
    itsMPS->InitializeWith(TensorNetworks::Random);

    Epsilons eps(1e-12);
    eps.itsDelatEnergy1Epsilon=1e-9;
    IterationSchedule is;
    is.Insert({maxIter,D,eps});

    int nSweep=itsMPS->FindVariationalGroundState(itsH,is);

    double E=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(E/(L-1),-0.45317425,1e-7);
    EXPECT_LT(nSweep,maxIter);
}




TEST_F(GroundStateTesting,TestSweepL9S1D8)
{
    int L=9,D=8,maxIter=100;
    double S=0.5;
    Setup(L,S,D);
    itsMPS->InitializeWith(TensorNetworks::Random);

    Epsilons eps(1e-12);
    eps.itsDelatEnergy1Epsilon=1e-9;
    IterationSchedule is;
    is.Insert({maxIter,D,eps});

    int nSweep=itsMPS->FindVariationalGroundState(itsH,is);

    double E=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(E/(L-1),-0.46703753,1e-7);
    EXPECT_LT(nSweep,maxIter);
}

TEST_F(GroundStateTesting,TestSweepL9S5D2)
{
    int L=9,D=2,maxIter=100;
    double S=2.5;
    Setup(L,S,D);
    itsMPS->InitializeWith(TensorNetworks::Random);

    Epsilons eps(1e-12);
    eps.itsDelatEnergy1Epsilon=1e-9;
    IterationSchedule is;
    is.Insert({maxIter,D,eps});
    int nSweep=itsMPS->FindVariationalGroundState(itsH,is);

    double E=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(E/(L-1),-7.025661 ,1e-7);
    EXPECT_LT(nSweep,maxIter);
}

TEST_F(GroundStateTesting,TestFreezeL9S1D2)
{
    int L=9,D=2,maxIter=100;
    double S=0.5;
    Setup(L,S,D);
    itsMPS->InitializeWith(TensorNetworks::Random);
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


TEST_F(GroundStateTesting,TestSweepL6S1GrowD27)
{
    int L=6,Dstart=2,Dend=27;
    double S=1.0;
    Setup(L,S,Dstart);
    itsMPS->InitializeWith(TensorNetworks::Random);

    Epsilons eps(1e-12);
    eps.itsDelatEnergy1Epsilon=1e-6;
    IterationSchedule is;
    is.Insert({30,Dend,eps});
    eps.itsDelatEnergy1Epsilon=1e-7;
    is.Insert({30,Dend,eps});
    eps.itsDelatEnergy1Epsilon=1e-8;
    is.Insert({30,Dend,eps});
    eps.itsDelatEnergy1Epsilon=1e-9;
    is.Insert({30,Dend,eps});
//    eps.itsDelatEnergy1Epsilon=1e-10;
//    is.Insert({30,Dend,eps});
    itsMPS->FindVariationalGroundState(itsH,is);

    itsMPS->Report(cout);

    double E=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(E/(L-1), -1.4740549939 ,1e-8);
}

#ifndef DEBUG

TEST_F(GroundStateTesting,TestSweepL7S1GrowD27)
{
    int L=7,Dstart=2,Dend=27;
    double S=1.0;
    Setup(L,S,Dstart);
    itsMPS->InitializeWith(TensorNetworks::Random);

    Epsilons eps(1e-12);
    eps.itsDelatEnergy1Epsilon=1e-6;
    IterationSchedule is;
    is.Insert({30,Dend,eps});
    eps.itsDelatEnergy1Epsilon=1e-7;
    is.Insert({30,Dend,eps});
    eps.itsDelatEnergy1Epsilon=1e-8;
    is.Insert({30,Dend,eps});
    eps.itsDelatEnergy1Epsilon=1e-9;
    is.Insert({30,Dend,eps});
//    eps.itsDelatEnergy1Epsilon=1e-10;
//    is.Insert({30,Dend,eps});
    itsMPS->FindVariationalGroundState(itsH,is);

    itsMPS->Report(cout);

    double E=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(E/(L-1), -1.4390886637843641 ,1e-8);
}

TEST_F(GroundStateTesting,TestSweepL8S1GrowD27)
{
    int L=8,Dstart=2,Dend=27;
    double S=1.0;
    Setup(L,S,Dstart);
    itsMPS->InitializeWith(TensorNetworks::Random);

    Epsilons eps(1e-12);
    eps.itsDelatEnergy1Epsilon=1e-6;
    IterationSchedule is;
    is.Insert({30,Dend,eps});
    eps.itsDelatEnergy1Epsilon=1e-7;
    is.Insert({30,Dend,eps});
    eps.itsDelatEnergy1Epsilon=1e-8;
    is.Insert({30,Dend,eps});
    eps.itsDelatEnergy1Epsilon=1e-9;
    is.Insert({30,Dend,eps});
//    eps.itsDelatEnergy1Epsilon=1e-10;
//    is.Insert({30,Dend,eps});
    itsMPS->FindVariationalGroundState(itsH,is);

    itsMPS->Report(cout);

    double E=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(E/(L-1), -1.4463763812511536,1e-8);
}


TEST_F(GroundStateTesting,TestSweepL19S5D4)
{
    int L=19,Dstart=2,Dend=5,maxIter=30;
    double S=2.5;
    Setup(L,S,Dstart);
    itsMPS->InitializeWith(TensorNetworks::Random);

    Epsilons eps(1e-12);
    eps.itsDelatEnergy1Epsilon=1e-6;
    IterationSchedule is;
    is.Insert({maxIter,Dend,eps});
    eps.itsDelatEnergy1Epsilon=1e-7;
    is.Insert({maxIter,Dend,eps});
    eps.itsDelatEnergy1Epsilon=1e-8;
    is.Insert({maxIter,Dend,eps});

    StopWatch sw;
    sw.Start();

    itsMPS->FindVariationalGroundState(itsH,is);
    sw.Stop();
    cout << "FindGroundState for L=" << L << ", S=" << S << ", D=" << Dend << " took " << sw.GetTime() << " seconds." << endl;

    double E=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(E/(L-1),-7.1969843668336484 ,1e-8);
}


TEST_F(GroundStateTesting,TestSweepL19S1D8)
{
    int L=19,D=8,maxIter=100;
    double S=0.5;
    Setup(L,S,D);
    itsMPS->InitializeWith(TensorNetworks::Random);

    Epsilons eps(1e-12);
    eps.itsDelatEnergy1Epsilon=1e-9;
    IterationSchedule is;
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




