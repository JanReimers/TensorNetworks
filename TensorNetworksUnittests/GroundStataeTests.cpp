#include "Tests.H"
#include "TensorNetworks/Hamiltonian.H"
#include "TensorNetworks/OperatorWRepresentation.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/Factory.H"
#include "TensorNetworks/LRPSupervisor.H"
#include "TensorNetworks/Epsilons.H"

#include "oml/stream.h"
#include "oml/stopw.h"

using std::setw;

class GroundStateTesting : public ::testing::Test
{
public:
    GroundStateTesting()
    : eps(1.0e-13)
    , itsFactory(TensorNetworks::Factory::GetFactory())
    , itsEps()
    {
        StreamableObject::SetToPretty();

    }

    void Setup(int L, int S2, int D)
    {
        itsH=itsFactory->Make1D_NN_HeisenbergHamiltonian(L,S2/2.0,1.0,1.0,0.0);
        itsMPS=itsH->CreateMPS(D,itsEps);
    }


    double eps;
    const TensorNetworks::Factory* itsFactory=TensorNetworks::Factory::GetFactory();
    Hamiltonian*         itsH;
    MatrixProductState*  itsMPS;
    Epsilons             itsEps;
};


TEST_F(GroundStateTesting,TestIdentityOperator)
{
    Setup(10,1,2);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Normalize(TensorNetworks::Left,new LRPSupervisor());
    OperatorWRepresentation* IWO=itsFactory->MakeIdentityOperator();
    Operator* IO=itsH->CreateOperator(IWO);
    double S=itsMPS->GetExpectation(IO);
    EXPECT_NEAR(S,1.0,eps);

}


TEST_F(GroundStateTesting,TestSweepL9S1D2)
{
    int L=9,S2=1,D=2,maxIter=100;
    Setup(L,S2,D);
    itsMPS->InitializeWith(TensorNetworks::Random);
    int nSweep=itsMPS->FindGroundState(itsH,maxIter,1e-9,new LRPSupervisor());

    double E=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(E/(L-1),-0.45317425,1e-7);
    EXPECT_LT(nSweep,maxIter);
}




TEST_F(GroundStateTesting,TestSweepL9S1D8)
{
    int L=9,S2=1,D=8,maxIter=100;
    Setup(L,S2,D);
    itsMPS->InitializeWith(TensorNetworks::Random);
    int nSweep=itsMPS->FindGroundState(itsH,maxIter,1e-9,new LRPSupervisor());

    double E=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(E/(L-1),-0.46703753,1e-7);
    EXPECT_LT(nSweep,maxIter);
}

TEST_F(GroundStateTesting,TestSweepL9S5D2)
{
    int L=9,S2=5,D=2,maxIter=100;
    Setup(L,S2,D);
    itsMPS->InitializeWith(TensorNetworks::Random);
    int nSweep=itsMPS->FindGroundState(itsH,maxIter,1e-9,new LRPSupervisor());

    double E=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(E/(L-1),-7.025661 ,1e-7);
    EXPECT_LT(nSweep,maxIter);
}

#ifndef DEBUG


TEST_F(GroundStateTesting,TestSweepL19S5D4)
{
    int L=19,S2=5,D=4,maxIter=100;
    Setup(L,S2,D);
    itsMPS->InitializeWith(TensorNetworks::Random);
    StopWatch sw;
    sw.Start();
    int nSweep=itsMPS->FindGroundState(itsH,maxIter,1e-11,new LRPSupervisor());
    sw.Stop();
    cout << "FindGroundState for L=" << L << ", S=" << S2/2.0 << ", D=" << D << " took " << sw.GetTime() << " seconds." << endl;

    double E=itsMPS->GetExpectationIterate(itsH);
    EXPECT_NEAR(E/(L-1),-7.1766766 ,1e-5);
    EXPECT_LT(nSweep,maxIter);
}

TEST_F(GroundStateTesting,TestSweepL19S1D8)
{
    int L=19,S2=1,D=8,maxIter=100;
    Setup(L,S2,D);
    itsMPS->InitializeWith(TensorNetworks::Random);
    StopWatch sw;
    sw.Start();
    int nSweep=itsMPS->FindGroundState(itsH,maxIter,1e-9,new LRPSupervisor());
    sw.Stop();
    cout << "FindGroundState for L=" << L << ", S=" << S2/2.0 << ", D=" << D << " took " << sw.GetTime() << " seconds." << endl;

    double E=itsMPS->GetExpectationIterate(itsH);
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
                int nSweep=itsMPS->FindGroundState(itsHamiltonianMPO,100,1e-8);
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




