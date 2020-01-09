#include "Tests.H"
#include "TensorNetworks/MatrixProductOperator.H"
#include "TensorNetworks/IdentityOperator.H"
#include "TensorNetworks/Hamiltonian_1D_NN_Heisenberg.H"
#include "oml/stream.h"
#include <complex>

typedef MatrixProductOperator::Matrix6T Matrix6T;
class GroundStateTesting : public ::testing::Test
{
public:
    typedef MatrixProductSite::MatrixT MatrixT;
    GroundStateTesting()
    : eps(1.0e-13)
    {
        StreamableObject::SetToPretty();

    }

    void Setup(int L, int S2, int D)
    {
        itsH=new Hamiltonian_1D_NN_Heisenberg(L,S2,1.0);
        itsMPO=itsH->CreateMPO();
        itsMPS=itsH->CreateMPS(D);
    }


    Hamiltonian* itsH;
    MatrixProductOperator* itsMPO;
    MatrixProductState*    itsMPS;
    double eps;
};

/*
TEST_F(GroundStateTesting,TestSweepL10S1D2)
{
    Setup(10,1,2);
    itsMPS->InitializeWith(MatrixProductSite::Random);
    itsMPS->Normalize(MatrixProductSite::Right);
    itsMPS->SweepRight(itsMPO);
    itsMPS->SweepLeft (itsMPO);
    double E=itsMPS->GetExpectation(itsMPO);
    double o=itsMPS->GetOverlap();
    cout << "After 2 sweeps E=" << E << "  Overlap=" << o << endl;
}


TEST_F(GroundStateTesting,TestSweepL10S1D1)
{
    Setup(10,1,1);
    itsMPS->InitializeWith(MatrixProductSite::Random);
    itsMPS->Normalize(MatrixProductSite::Right);
    itsMPS->SweepRight(itsMPO);
    itsMPS->SweepLeft (itsMPO);
    double E=itsMPS->GetExpectation(itsMPO);
    double o=itsMPS->GetOverlap();
    cout << "After 2 sweeps E=" << E << "  Overlap=" << o << endl;
}
*/

TEST_F(GroundStateTesting,TestNeelStateL10S1D2)
{
    int L=9;
    Setup(L,1,2);
    itsMPS->InitializeWith(MatrixProductSite::Random);
    itsMPS->Normalize(MatrixProductSite::Right);
    itsMPS->Report(cout);
    itsMPS->SweepRight(itsMPO,false);
    itsMPS->Report(cout);
    itsMPS->SweepLeft (itsMPO,false);
    itsMPS->Report(cout);
    itsMPS->SweepRight(itsMPO,false);
    itsMPS->Report(cout);
    itsMPS->SweepLeft (itsMPO,false);
    itsMPS->Report(cout);
    double E=itsMPS->GetExpectationIterate(itsMPO);
    double o=itsMPS->GetOverlap();
    cout << "After 2 sweeps E=" << E/(L-1) << "  Overlap=" << o << endl;
}

/*
TEST_F(GroundStateTesting,TestLandD_DependenceS1)
{
    for (int D=1;D<=5;D++)
    for (int L=4;L<=12;L++)
    {
        Setup(L,1,D);
        itsMPS->InitializeWith(MatrixProductSite::Random);
        itsMPS->Normalize(MatrixProductSite::Left);
        //itsMPS->SweepRight(itsMPO);
        //itsMPS->SweepLeft (itsMPO);
        double E=itsMPS->GetExpectation(itsMPO);
//        EXPECT_NEAR(itsMPS->GetOverlap(),1.0,eps);
        cout << "D,L=" << D << "," << L << "  After 2 sweeps E=" << E << "  E/L=" << E/L << "  E/2^L=" << E/pow(2,L) << endl;
    }
}

TEST_F(GroundStateTesting,TestLandD_DependenceS2)
{
    for (int D=1;D<=3;D++)
    for (int L=4;L<=12;L++)
    {
        Setup(L,2,D);
        itsMPS->InitializeWith(MatrixProductSite::Random);
        itsMPS->Normalize(MatrixProductSite::Left);
//        itsMPS->SweepRight(itsMPO);
//        itsMPS->SweepLeft (itsMPO);
        double E=itsMPS->GetExpectation(itsMPO);
//        EXPECT_NEAR(itsMPS->GetOverlap(),1.0,eps);
        cout << "D,L=" << D << "," << L << "  After 2 sweeps E=" << E << "  E/L=" << E/L << "  E/3^L=" << E/pow(3,L) << endl;
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
    cout << "Identity=" << S << endl;

}


