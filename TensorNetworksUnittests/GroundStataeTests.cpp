#include "Tests.H"
#include "TensorNetworks/MatrixProductOperator.H"
#include "TensorNetworks/IdentityOperator.H"
#include "TensorNetworks/Hamiltonian_1D_NN_Heisenberg.H"
#include "oml/stream.h"
#include <complex>

using std::setw;

typedef MatrixProductOperator::Matrix6T Matrix6T;
class GroundStateTesting : public ::testing::Test
{
public:
    typedef MatrixProductSite::MatrixCT MatrixCT;
    GroundStateTesting()
    : eps(1.0e-13)
    {
        StreamableObject::SetToPretty();

    }

    void Setup(int L, int S2, int D)
    {
        itsH=new Hamiltonian_1D_NN_Heisenberg(L,S2,1.0);
        itsHamiltonianMPO=itsH->CreateMPO();
        itsMPS=itsH->CreateMPS(D);
    }


    Hamiltonian* itsH;
    MatrixProductOperator* itsHamiltonianMPO;
    MatrixProductState*    itsMPS;
    double eps;
};


TEST_F(GroundStateTesting,TestSweepL9S1D2)
{
    int L=9,S2=1,D=2,maxIter=100;
    Setup(L,S2,D);
    itsMPS->InitializeWith(MatrixProductSite::Random);
    int nSweep=itsMPS->FindGroundState(itsHamiltonianMPO,maxIter,1e-9);

    double E=itsMPS->GetExpectationIterate(itsHamiltonianMPO);
    double o=itsMPS->GetOverlap();
    EXPECT_NEAR(o,1.0,eps);
    EXPECT_NEAR(E/(L-1),-0.45317425,1e-7);
    EXPECT_LT(nSweep,maxIter);
}

#ifndef DEBUG
TEST_F(GroundStateTesting,TestSweepL9S1D8)
{
    int L=9,S2=1,D=8,maxIter=100;
    Setup(L,S2,D);
    itsMPS->InitializeWith(MatrixProductSite::Random);
    int nSweep=itsMPS->FindGroundState(itsHamiltonianMPO,maxIter,1e-9);

    double E=itsMPS->GetExpectationIterate(itsHamiltonianMPO);
    double o=itsMPS->GetOverlap();
    EXPECT_NEAR(o,1.0,eps);
    EXPECT_NEAR(E/(L-1),-0.46703753,1e-7);
    EXPECT_LT(nSweep,maxIter);
}
#endif

TEST_F(GroundStateTesting,TestSweepL9S5D2)
{
    int L=9,S2=5,D=2,maxIter=100;
    Setup(L,S2,D);
    itsMPS->InitializeWith(MatrixProductSite::Random);
    int nSweep=itsMPS->FindGroundState(itsHamiltonianMPO,maxIter,1e-9);

    double E=itsMPS->GetExpectationIterate(itsHamiltonianMPO);
    double o=itsMPS->GetOverlap();
    EXPECT_NEAR(o,1.0,eps);
    EXPECT_NEAR(E/(L-1),-7.025661 ,1e-7);
    EXPECT_LT(nSweep,maxIter);
}

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

TEST_F(GroundStateTesting,TestIdentityOperator)
{
    Setup(10,1,2);
    itsMPS->InitializeWith(MatrixProductSite::Random);
    itsMPS->Normalize(MatrixProductSite::Left);
    Operator* IO=new IdentityOperator();
    MatrixProductOperator* mpoi=new MatrixProductOperator(IO,10,1,2);

    double S=itsMPS->GetExpectation(mpoi);
    EXPECT_NEAR(S,1.0,eps);

}


