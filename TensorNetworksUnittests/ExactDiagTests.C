#include "Tests.H"
#include "TensorNetworks/Hamiltonian.H"
#include "TensorNetworks/Factory.H"
#include "TensorNetworks/FullState.H"
#include "TensorNetworksImp/StateIterator.H"
#include "oml/numeric.h"

using TensorNetworks::MatrixT;

class ExactDiagTesting : public ::testing::Test
{
public:
    typedef std::complex<double> complx;
    typedef TensorNetworks:: ArrayT  ArrayT;

    ExactDiagTesting()
        : itsFactory(TensorNetworks::Factory::GetFactory())
        , itsH(0)
        , itsEps(1e-10)
    {
        itsEps.itsEnergyConvergenceEpsilon=1e-15;
        itsEps.itsEigenConvergenceEpsilon=1e-15;
        itsEps.itsMaxIter=1000;
        StreamableObject::SetToPretty();

    }
    ~ExactDiagTesting()
    {
        delete itsFactory;
        delete itsH;
    }

    void Setup(int L, double S)
    {
        delete itsH;
        itsH=itsFactory->Make1D_NN_HeisenbergHamiltonian(L,S,1.0,1.0,0.0);
    }

    TensorNetworks::MatrixT GetH(double S)
    {
        Setup(2,S);
        return itsH->BuildLocalMatrix().Flatten();
    }


    const TensorNetworks::Factory* itsFactory=TensorNetworks::Factory::GetFactory();
    Hamiltonian*         itsH;
    Epsilons             itsEps;
};


TEST_F(ExactDiagTesting,TestStateIterator)
{

    for (StateIterator is(3,3);!is.end();is++)
        EXPECT_EQ(is.GetLinearIndex(),is.GetIndex(is.GetQuantumNumbers()));
}



TEST_F(ExactDiagTesting,TestHabS12)
{
    Setup(10,0.5);
    MatrixT Hab=itsH->BuildLocalMatrix().Flatten();
    //cout << "Hab=" << Hab << endl;
    EXPECT_EQ(ToString(Hab),"(1:4),(1:4) \n[ 0.25 0 0 0 ]\n[ 0 -0.25 0.5 0 ]\n[ 0 0.5 -0.25 0 ]\n[ 0 0 0 0.25 ]\n");
}


TEST_F(ExactDiagTesting,TestHabS1)
{
    Setup(10,1.0);
    MatrixT Hab=itsH->BuildLocalMatrix().Flatten();
    EXPECT_EQ(ToString(Hab),"(1:9),(1:9) \n[ 1 0 0 0 0 0 0 0 0 ]\n[ 0 0 0 1 0 0 0 0 0 ]\n[ 0 0 -1 0 1 0 0 0 0 ]\n[ 0 1 0 0 0 0 0 0 0 ]\n[ 0 0 1 0 0 0 1 0 0 ]\n[ 0 0 0 0 0 0 0 1 0 ]\n[ 0 0 0 0 1 0 -1 0 0 ]\n[ 0 0 0 0 0 1 0 0 0 ]\n[ 0 0 0 0 0 0 0 0 1 ]\n");
}

TEST_F(ExactDiagTesting,TestHabS32)
{
    Setup(10,1.5);
    TensorNetworks::MatrixT Hab=itsH->BuildLocalMatrix().Flatten();
    EXPECT_EQ(ToString(Hab),"(1:16),(1:16) \n[ 2.25 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ]\n[ 0 0.75 0 0 1.5 0 0 0 0 0 0 0 0 0 0 0 ]\n[ 0 0 -0.75 0 0 1.73205 0 0 0 0 0 0 0 0 0 0 ]\n[ 0 0 0 -2.25 0 0 1.5 0 0 0 0 0 0 0 0 0 ]\n[ 0 1.5 0 0 0.75 0 0 0 0 0 0 0 0 0 0 0 ]\n[ 0 0 1.73205 0 0 0.25 0 0 1.73205 0 0 0 0 0 0 0 ]\n[ 0 0 0 1.5 0 0 -0.25 0 0 2 0 0 0 0 0 0 ]\n[ 0 0 0 0 0 0 0 -0.75 0 0 1.73205 0 0 0 0 0 ]\n[ 0 0 0 0 0 1.73205 0 0 -0.75 0 0 0 0 0 0 0 ]\n[ 0 0 0 0 0 0 2 0 0 -0.25 0 0 1.5 0 0 0 ]\n[ 0 0 0 0 0 0 0 1.73205 0 0 0.25 0 0 1.73205 0 0 ]\n[ 0 0 0 0 0 0 0 0 0 0 0 0.75 0 0 1.5 0 ]\n[ 0 0 0 0 0 0 0 0 0 1.5 0 0 -2.25 0 0 0 ]\n[ 0 0 0 0 0 0 0 0 0 0 1.73205 0 0 -0.75 0 0 ]\n[ 0 0 0 0 0 0 0 0 0 0 0 1.5 0 0 0.75 0 ]\n[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2.25 ]\n");
}

TEST_F(ExactDiagTesting,TestEvsS12)
{
    TensorNetworks::MatrixT H=GetH(1.0/2.0);
    Vector<double> evs=Diagonalize(H);
    EXPECT_EQ(ToString(evs),"(1:4){ -0.75 0.25 0.25 0.25 }");
}

TEST_F(ExactDiagTesting,TestEvsS22)
{
    TensorNetworks::MatrixT H=GetH(2.0/2.0);
    Vector<double> evs=Diagonalize(H);
    EXPECT_EQ(ToString(evs),"(1:9){ -2 -1 -1 -1 1 1 1 1 1 }");
}

TEST_F(ExactDiagTesting,TestEvsS32)
{
    TensorNetworks::MatrixT H=GetH(3.0/2.0);
    Vector<double> evs=Diagonalize(H);
    EXPECT_EQ(ToString(evs),"(1:16){ -3.75 -2.75 -2.75 -2.75 -0.75 -0.75 -0.75 -0.75 -0.75 2.25 2.25 2.25 2.25 2.25 2.25 2.25 }");
}

TEST_F(ExactDiagTesting,TestEvsS42)
{
    TensorNetworks::MatrixT H=GetH(4.0/2.0);
    Vector<double> evs=Diagonalize(H);
    int N=evs.size();
    for (int i=1;i<=N;i++)
        if (fabs(evs(i))<1e-15) evs(i)=0;
    EXPECT_EQ(ToString(evs),"(1:25){ -6 -5 -5 -5 -3 -3 -3 -3 -3 0 0 0 0 0 0 0 4 4 4 4 4 4 4 4 4 }");
}

TEST_F(ExactDiagTesting,TestEvsS52)
{
    TensorNetworks::MatrixT H=GetH(5.0/2.0);
    Vector<double> evs=Diagonalize(H);
    EXPECT_EQ(ToString(evs),"(1:36){ -8.75 -7.75 -7.75 -7.75 -5.75 -5.75 -5.75 -5.75 -5.75 -2.75 -2.75 -2.75 -2.75 -2.75 -2.75 -2.75 1.25 1.25 1.25 1.25 1.25 1.25 1.25 1.25 1.25 6.25 6.25 6.25 6.25 6.25 6.25 6.25 6.25 6.25 6.25 6.25 }");
}

TEST_F(ExactDiagTesting,CreateFullStateL10S12)
{
    Setup(10,0.5);
    FullState* psi=itsH->CreateFullState();
    EXPECT_EQ(psi->GetSize(),1024);
}

TEST_F(ExactDiagTesting,CreateFullStateL10S22)
{
    Setup(10,1.0);
    FullState* psi=itsH->CreateFullState();
    EXPECT_EQ(psi->GetSize(),59049);
}

TEST_F(ExactDiagTesting,CreateFullStateL10S32)
{
    Setup(10,1.5);
    FullState* psi=itsH->CreateFullState();
    EXPECT_EQ(psi->GetSize(),1048576);
}

TEST_F(ExactDiagTesting,CreateFullStateL10S42)
{
    Setup(10,2.0);
    FullState* psi=itsH->CreateFullState();
    EXPECT_EQ(psi->GetSize(),9765625);
}

TEST_F(ExactDiagTesting,CreateFullStateL10S52)
{
    Setup(5,2.5);
    FullState* psi=itsH->CreateFullState();
    EXPECT_EQ(psi->GetSize(),7776);
}



TEST_F(ExactDiagTesting,PowerMethodGroundStateL2S12)
{
    Setup(2,0.5);
    FullState* psi=itsH->CreateFullState();
    psi->PowerIterate(itsEps,*itsH);
    cout << *psi << endl;
    EXPECT_NEAR(psi->GetE(),-0.75,itsEps.itsEnergyConvergenceEpsilon);
}

TEST_F(ExactDiagTesting,PowerMethodGroundStateL3S12)
{
    Setup(3,0.5);
    FullState* psi=itsH->CreateFullState();
    psi->PowerIterate(itsEps,*itsH);
    cout << *psi << endl;
    EXPECT_NEAR(psi->GetE(),-1.0,itsEps.itsEnergyConvergenceEpsilon);
}

TEST_F(ExactDiagTesting,PowerMethodGroundStateL4S12)
{
    Setup(4,0.5);
    FullState* psi=itsH->CreateFullState();
    psi->PowerIterate(itsEps,*itsH);
    EXPECT_NEAR(psi->GetE(), -1.6160254037844393,itsEps.itsEnergyConvergenceEpsilon);
}
TEST_F(ExactDiagTesting,PowerMethodGroundStateL6S12)
{
    Setup(6,0.5);
    FullState* psi=itsH->CreateFullState();
    psi->PowerIterate(itsEps,*itsH);
    EXPECT_NEAR(psi->GetE(),  -2.4935771338879262,itsEps.itsEnergyConvergenceEpsilon);
}
TEST_F(ExactDiagTesting,PowerMethodGroundStateL10S12)
{
    Setup(10,0.5);
    FullState* psi=itsH->CreateFullState();
    psi->PowerIterate(itsEps,*itsH);
    EXPECT_NEAR(psi->GetE(),-4.2580352072828864 ,itsEps.itsEnergyConvergenceEpsilon*10);
}

TEST_F(ExactDiagTesting,PowerMethodGroundStateL2S1)
{
    Setup(2,1.0);
    FullState* psi=itsH->CreateFullState();
    psi->PowerIterate(itsEps,*itsH);
    cout << *psi << endl;
    EXPECT_NEAR(psi->GetE(),-2,itsEps.itsEnergyConvergenceEpsilon);
}

TEST_F(ExactDiagTesting,PowerMethodGroundStateL2S32)
{
    Setup(2,1.5);
    FullState* psi=itsH->CreateFullState();
    psi->PowerIterate(itsEps,*itsH);
    cout << *psi << endl;
    EXPECT_NEAR(psi->GetE(),-3.75,itsEps.itsEnergyConvergenceEpsilon);
}

TEST_F(ExactDiagTesting,PowerMethodGroundStateL2S2)
{
    Setup(2,2.0);
    FullState* psi=itsH->CreateFullState();
    psi->PowerIterate(itsEps,*itsH);
    EXPECT_NEAR(psi->GetE(),-6,itsEps.itsEnergyConvergenceEpsilon);
}

TEST_F(ExactDiagTesting,PowerMethodGroundStateL2S52)
{
    Setup(2,2.5);
    FullState* psi=itsH->CreateFullState();
    psi->PowerIterate(itsEps,*itsH);
    EXPECT_NEAR(psi->GetE(),-8.75,itsEps.itsEnergyConvergenceEpsilon*10);
}

#ifndef DEBUG
TEST_F(ExactDiagTesting,PowerMethodGroundStateL4S52)
{
    Setup(4,2.5);
    FullState* psi=itsH->CreateFullState();
    psi->PowerIterate(itsEps,*itsH);
    EXPECT_NEAR(psi->GetE(),-22.762419480032261,itsEps.itsEigenConvergenceEpsilon*50);
}
#endif // DEBUG

TEST_F(ExactDiagTesting,LanczosGroundStateL2S12)
{
    Setup(2,0.5);
    FullState* psi=itsH->CreateFullState();
    psi->FindGroundState(itsEps,*itsH);
    cout << *psi << endl;
    EXPECT_NEAR(psi->GetE(),-0.75,itsEps.itsEigenConvergenceEpsilon);
}

TEST_F(ExactDiagTesting,LanczosGroundStateL2S52)
{
    Setup(2,2.5);
    FullState* psi=itsH->CreateFullState();
    psi->FindGroundState(itsEps,*itsH);
    cout << *psi << endl;
    EXPECT_NEAR(psi->GetE(),-8.75,itsEps.itsEigenConvergenceEpsilon*10);
}

TEST_F(ExactDiagTesting,LanczosGroundStateL10S12)
{
    Setup(10,0.5);
    FullState* psi=itsH->CreateFullState();
    psi->FindGroundState(itsEps,*itsH);
//    cout << *psi << endl;
    EXPECT_NEAR(psi->GetE(),-4.258035207282882,itsEps.itsEigenConvergenceEpsilon);
}

#ifndef DEBUG
TEST_F(ExactDiagTesting,LanczosGroundStateL12S12)
{
    itsEps.itsEigenConvergenceEpsilon=1e-12;
    Setup(12,0.5);
    FullState* psi=itsH->CreateFullState();
    psi->FindGroundState(itsEps,*itsH);
    EXPECT_NEAR(psi->GetE(),-5.1420906328405298,itsEps.itsEigenConvergenceEpsilon);
}


TEST_F(ExactDiagTesting,LanczosGroundStateL14S12)
{
    itsEps.itsEigenConvergenceEpsilon=1e-12;
    Setup(14,0.5);
    FullState* psi=itsH->CreateFullState();
    psi->FindGroundState(itsEps,*itsH);
    EXPECT_NEAR(psi->GetE(),-6.0267246618621693,itsEps.itsEigenConvergenceEpsilon);
}

TEST_F(ExactDiagTesting,LanczosGroundStateL16S12)
{
    itsEps.itsEigenConvergenceEpsilon=1e-10;
    Setup(16,0.5);
    FullState* psi=itsH->CreateFullState();
    psi->FindGroundState(itsEps,*itsH);
    EXPECT_NEAR(psi->GetE(),-6.9117371455751222,itsEps.itsEigenConvergenceEpsilon);
}

#endif