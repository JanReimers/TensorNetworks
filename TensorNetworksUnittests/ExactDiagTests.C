#include "Tests.H"
#include "TensorNetworks/Typedefs.H"
#include "TensorNetworks/Hamiltonian.H"
#include "TensorNetworks/Factory.H"
#include "TensorNetworks/FullState.H"
#include "TensorNetworksImp/StateIterator.H"
#include "Containers/Matrix4.H"
#include "oml/matrix.h"
#include "oml/numeric.h"

using TensorNetworks::MatrixRT;
using TensorNetworks::Random;
using TensorNetworks::Neel;
using TensorNetworks::DLeft;
using TensorNetworks::DRight;
using TensorNetworks::IterationSchedule;
using TensorNetworks::Epsilons;
using TensorNetworks::MPO;
using TensorNetworks::TriType;

class ExactDiagTests : public ::testing::Test
{
public:
    ExactDiagTests()
        : itsFactory(TensorNetworks::Factory::GetFactory())
        , itsH(0)
        , itsPsi(0)
        , itsMaxIter(1000)
        , itsEpsE(1e-14)
        , itsEpsPsi(1e-10)
    {
        StreamableObject::SetToPretty();
    }
    ~ExactDiagTests()
    {
        delete itsFactory;
        if (itsH) delete itsH;
        if (itsPsi) delete itsPsi;
    }

    void SetupH(int L, double S,TriType ul)
    {
        delete itsH;
        itsH=itsFactory->Make1D_NN_HeisenbergHamiltonian(L,S,1.0,1.0,0.0,ul);
    }
    void Setup(int L, double S,TriType ul)
    {
        SetupH(L,S,ul);
        itsPsi=itsH->CreateFullState();
    }
    TensorNetworks::MatrixRT GetH(double S,TriType ul)
    {
        SetupH(2,S,ul);
        return itsH->GetLocalMatrix().Flatten();
    }


    TensorNetworks::Factory*     itsFactory=TensorNetworks::Factory::GetFactory();
    TensorNetworks::Hamiltonian* itsH;
    TensorNetworks::FullState*   itsPsi;
    int    itsMaxIter;
    double itsEpsE;
    double itsEpsPsi;
};


TEST_F(ExactDiagTests,TestStateIterator)
{

    for (TensorNetworks::StateIterator is(3,3);!is.end();is++)
        EXPECT_EQ(is.GetLinearIndex(),is.GetIndex(is.GetQuantumNumbers()));
}



TEST_F(ExactDiagTests,TestHab_Lower_S12)
{
    SetupH(10,0.5,Lower);
    MatrixRT Hab=itsH->GetLocalMatrix().Flatten();
    //cout << "Hab=" << Hab << endl;
    EXPECT_EQ(ToString(Hab),"(1:4),(1:4) \n[ 0.25 0 0 0 ]\n[ 0 -0.25 0.5 0 ]\n[ 0 0.5 -0.25 0 ]\n[ 0 0 0 0.25 ]\n");
}


TEST_F(ExactDiagTests,TestHab_Upper_S1)
{
    SetupH(10,1.0,Upper);
    MatrixRT Hab=itsH->GetLocalMatrix().Flatten();
    EXPECT_EQ(ToString(Hab),"(1:9),(1:9) \n[ 1 0 0 0 0 0 0 0 0 ]\n[ 0 0 0 1 0 0 0 0 0 ]\n[ 0 0 -1 0 1 0 0 0 0 ]\n[ 0 1 0 0 0 0 0 0 0 ]\n[ 0 0 1 0 0 0 1 0 0 ]\n[ 0 0 0 0 0 0 0 1 0 ]\n[ 0 0 0 0 1 0 -1 0 0 ]\n[ 0 0 0 0 0 1 0 0 0 ]\n[ 0 0 0 0 0 0 0 0 1 ]\n");
}

TEST_F(ExactDiagTests,TestHab_Lower_S32)
{
    SetupH(10,1.5,Lower);
    TensorNetworks::MatrixRT Hab=itsH->GetLocalMatrix().Flatten();
    EXPECT_EQ(ToString(Hab),"(1:16),(1:16) \n[ 2.25 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ]\n[ 0 0.75 0 0 1.5 0 0 0 0 0 0 0 0 0 0 0 ]\n[ 0 0 -0.75 0 0 1.73205 0 0 0 0 0 0 0 0 0 0 ]\n[ 0 0 0 -2.25 0 0 1.5 0 0 0 0 0 0 0 0 0 ]\n[ 0 1.5 0 0 0.75 0 0 0 0 0 0 0 0 0 0 0 ]\n[ 0 0 1.73205 0 0 0.25 0 0 1.73205 0 0 0 0 0 0 0 ]\n[ 0 0 0 1.5 0 0 -0.25 0 0 2 0 0 0 0 0 0 ]\n[ 0 0 0 0 0 0 0 -0.75 0 0 1.73205 0 0 0 0 0 ]\n[ 0 0 0 0 0 1.73205 0 0 -0.75 0 0 0 0 0 0 0 ]\n[ 0 0 0 0 0 0 2 0 0 -0.25 0 0 1.5 0 0 0 ]\n[ 0 0 0 0 0 0 0 1.73205 0 0 0.25 0 0 1.73205 0 0 ]\n[ 0 0 0 0 0 0 0 0 0 0 0 0.75 0 0 1.5 0 ]\n[ 0 0 0 0 0 0 0 0 0 1.5 0 0 -2.25 0 0 0 ]\n[ 0 0 0 0 0 0 0 0 0 0 1.73205 0 0 -0.75 0 0 ]\n[ 0 0 0 0 0 0 0 0 0 0 0 1.5 0 0 0.75 0 ]\n[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2.25 ]\n");
}

TEST_F(ExactDiagTests,TestEvs_Lower_S12)
{
    TensorNetworks::MatrixRT H=GetH(1.0/2.0,Lower);
    Vector<double> evs=Diagonalize(H);
    EXPECT_EQ(ToString(evs),"(1:4){ -0.75 0.25 0.25 0.25 }");
}

TEST_F(ExactDiagTests,TestEvs_Lower_S22)
{
    TensorNetworks::MatrixRT H=GetH(2.0/2.0,Lower);
    Vector<double> evs=Diagonalize(H);
    EXPECT_EQ(ToString(evs),"(1:9){ -2 -1 -1 -1 1 1 1 1 1 }");
}

TEST_F(ExactDiagTests,TestEvs_Lower_S32)
{
    TensorNetworks::MatrixRT H=GetH(3.0/2.0,Lower);
    Vector<double> evs=Diagonalize(H);
    EXPECT_EQ(ToString(evs),"(1:16){ -3.75 -2.75 -2.75 -2.75 -0.75 -0.75 -0.75 -0.75 -0.75 2.25 2.25 2.25 2.25 2.25 2.25 2.25 }");
}

TEST_F(ExactDiagTests,TestEvs_Lower_S42)
{
    TensorNetworks::MatrixRT H=GetH(4.0/2.0,Lower);
    Vector<double> evs=Diagonalize(H);
    int N=evs.size();
    for (int i=1;i<=N;i++)
        if (fabs(evs(i))<1e-15) evs(i)=0;
    EXPECT_EQ(ToString(evs),"(1:25){ -6 -5 -5 -5 -3 -3 -3 -3 -3 0 0 0 0 0 0 0 4 4 4 4 4 4 4 4 4 }");
}

TEST_F(ExactDiagTests,TestEvs_Lower_S52)
{
    TensorNetworks::MatrixRT H=GetH(5.0/2.0,Lower);
    Vector<double> evs=Diagonalize(H);
    EXPECT_EQ(ToString(evs),"(1:36){ -8.75 -7.75 -7.75 -7.75 -5.75 -5.75 -5.75 -5.75 -5.75 -2.75 -2.75 -2.75 -2.75 -2.75 -2.75 -2.75 1.25 1.25 1.25 1.25 1.25 1.25 1.25 1.25 1.25 6.25 6.25 6.25 6.25 6.25 6.25 6.25 6.25 6.25 6.25 6.25 }");
}

TEST_F(ExactDiagTests,CreateFullState_Lower_L10S12)
{
    Setup(10,0.5,Lower);
    EXPECT_EQ(itsPsi->GetSize(),1024);
}

TEST_F(ExactDiagTests,CreateFullStateL10S22)
{
    Setup(10,1.0,Lower);
    EXPECT_EQ(itsPsi->GetSize(),59049);
}

TEST_F(ExactDiagTests,CreateFullStateL10S32)
{
    Setup(10,1.5,Lower);
    EXPECT_EQ(itsPsi->GetSize(),1048576);
}

TEST_F(ExactDiagTests,CreateFullStateL10S42)
{
    Setup(10,2.0,Lower);
    EXPECT_EQ(itsPsi->GetSize(),9765625);
}

TEST_F(ExactDiagTests,CreateFullStateL10S52)
{
    Setup(5,2.5,Lower);
    EXPECT_EQ(itsPsi->GetSize(),7776);
}



TEST_F(ExactDiagTests,PowerMethodGroundState_Lower_L2S12)
{
    Setup(2,0.5,Lower);
    itsPsi->PowerIterate(*itsH,itsEpsE,itsEpsPsi,itsMaxIter);
    EXPECT_NEAR(itsPsi->GetE(),-0.75,itsEpsE);
}

TEST_F(ExactDiagTests,PowerMethodGroundState_Lower_L3S12)
{
    Setup(3,0.5,Lower);
    itsPsi->PowerIterate(*itsH,itsEpsE,itsEpsPsi,itsMaxIter);
    EXPECT_NEAR(itsPsi->GetE(),-1.0,itsEpsE);
}

TEST_F(ExactDiagTests,PowerMethodGroundState_Lower_L4S12)
{
    Setup(4,0.5,Lower);
    itsPsi->PowerIterate(*itsH,itsEpsE,itsEpsPsi,itsMaxIter);
    EXPECT_NEAR(itsPsi->GetE(), -1.6160254037844393,itsEpsE);
}
TEST_F(ExactDiagTests,PowerMethodGroundState_Lower_L6S12)
{
    Setup(6,0.5,Lower);
    itsPsi->PowerIterate(*itsH,itsEpsE,itsEpsPsi,itsMaxIter);
    EXPECT_NEAR(itsPsi->GetE(),  -2.4935771338879262,itsEpsE);
}

#ifndef DEBUG
TEST_F(ExactDiagTests,PowerMethodGroundState_Lower_L10S12)
{
    Setup(10,0.5,Lower);
    itsPsi->PowerIterate(*itsH,itsEpsE,itsEpsPsi,itsMaxIter);
    EXPECT_NEAR(itsPsi->GetE(),-4.2580352072828864 ,itsEpsE*10);
}
#endif // DEBUG

TEST_F(ExactDiagTests,PowerMethodGroundState_Lower_L2S1)
{
    Setup(2,1.0,Lower);
    itsPsi->PowerIterate(*itsH,itsEpsE,itsEpsPsi,itsMaxIter);
    EXPECT_NEAR(itsPsi->GetE(),-2,itsEpsE);
}

TEST_F(ExactDiagTests,PowerMethodGroundState_Lower_L2S32)
{
    Setup(2,1.5,Lower);
    itsPsi->PowerIterate(*itsH,itsEpsE,itsEpsPsi,itsMaxIter);
    EXPECT_NEAR(itsPsi->GetE(),-3.75,itsEpsE);
}
TEST_F(ExactDiagTests,PowerMethodGroundState_Upper_L2S32)
{
    Setup(2,1.5,Upper);
    itsPsi->PowerIterate(*itsH,itsEpsE,itsEpsPsi,itsMaxIter);
    EXPECT_NEAR(itsPsi->GetE(),-3.75,itsEpsE);
}

TEST_F(ExactDiagTests,PowerMethodGroundState_Lower_L2S2)
{
    Setup(2,2.0,Lower);
    itsPsi->PowerIterate(*itsH,itsEpsE,itsEpsPsi,itsMaxIter);
    EXPECT_NEAR(itsPsi->GetE(),-6,itsEpsE);
}

TEST_F(ExactDiagTests,PowerMethodGroundState_Lower_L2S52)
{
    Setup(2,2.5,Lower);
    itsPsi->PowerIterate(*itsH,itsEpsE,itsEpsPsi,itsMaxIter);
    EXPECT_NEAR(itsPsi->GetE(),-8.75,itsEpsE*10);
}

#ifdef RunLongTests
TEST_F(ExactDiagTests,PowerMethodGroundState_Lower_L4S52)
{
    Setup(4,2.5,Lower);
    itsPsi->PowerIterate(*itsH,itsEpsE,itsEpsPsi,itsMaxIter);
    EXPECT_NEAR(itsPsi->GetE(),-22.762419480032261,itsEpsE*50);
}
#endif // RunLongTests

TEST_F(ExactDiagTests,LanczosGroundState_Lower_L2S12)
{
    Setup(2,0.5,Lower);
    itsPsi->FindGroundState(*itsH,itsEpsE);
    EXPECT_NEAR(itsPsi->GetE(),-0.75,itsEpsE);
}

TEST_F(ExactDiagTests,LanczosGroundState_Lower_L2S52)
{
    Setup(2,2.5,Lower);
    itsPsi->FindGroundState(*itsH,itsEpsE);
    EXPECT_NEAR(itsPsi->GetE(),-8.75,itsEpsE*10);
}

TEST_F(ExactDiagTests,LanczosGroundState_Lower_L10S12)
{
    Setup(10,0.5,Lower);
    itsPsi->FindGroundState(*itsH,itsEpsE);
    EXPECT_NEAR(itsPsi->GetE(),-4.258035207282882,itsEpsE);
}
TEST_F(ExactDiagTests,LanczosGroundState_Upper_L10S12)
{
    Setup(10,0.5,Upper);
    itsPsi->FindGroundState(*itsH,itsEpsE);
    EXPECT_NEAR(itsPsi->GetE(),-4.258035207282882,itsEpsE);
}

#ifndef DEBUG
TEST_F(ExactDiagTests,LanczosGroundState_Lower_L12S12)
{
    Setup(12,0.5,Lower);
    itsPsi->FindGroundState(*itsH,itsEpsE);
    EXPECT_NEAR(itsPsi->GetE(),-5.1420906328405342,1.5*itsEpsE);
}
#endif

#ifdef RunLongTests

TEST_F(ExactDiagTests,LanczosGroundState_Lower_L14S12)
{
    Setup(14,0.5,Lower);
    itsPsi->FindGroundState(*itsH,itsEpsE);
    EXPECT_NEAR(itsPsi->GetE(),-6.0267246618621693,itsEpsE);
}

TEST_F(ExactDiagTests,LanczosGroundState_Lower_L16S12)
{
    Setup(16,0.5,Lower);
    itsPsi->FindGroundState(*itsH,itsEpsE);
    EXPECT_NEAR(itsPsi->GetE(),-6.9117371455751222,10*itsEpsE);
}

#endif
