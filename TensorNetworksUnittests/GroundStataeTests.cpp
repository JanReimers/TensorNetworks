#include "Tests.H"
#include "TensorNetworks/MatrixProductOperator.H"
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

TEST_F(GroundStateTesting,TestSweepL10S1D2)
{
    Setup(10,1,2);
    itsMPS->InitializeWith(MatrixProductSite::Random);
    //    itsMPS->InitializeWith(MatrixProductSite::Product);
    double E=itsMPO->GetExpectation(itsMPS);
    double o=itsMPS->GetOverlap();
    cout << "Before Refine1 E=" << E << "  Overlap=" << o << endl;
    itsMPS->Normalize(MatrixProductSite::Right);
    itsMPS->SweepRight(itsMPO);
    itsMPS->SweepLeft (itsMPO);
}
TEST_F(GroundStateTesting,TestSweepL10S1D1)
{
    Setup(10,1,1);
    itsMPS->InitializeWith(MatrixProductSite::Random);
    //    itsMPS->InitializeWith(MatrixProductSite::Product);
    double E=itsMPO->GetExpectation(itsMPS);
    double o=itsMPS->GetOverlap();
    cout << "Before Refine1 E=" << E << "  Overlap=" << o << endl;
    itsMPS->Normalize(MatrixProductSite::Right);
    itsMPS->SweepRight(itsMPO);
    itsMPS->SweepLeft (itsMPO);
}


TEST_F(GroundStateTesting,TestNeelState)
{
    Setup(10,1,3);
    itsMPS->InitializeWith(MatrixProductSite::Random);
    //    itsMPS->InitializeWith(MatrixProductSite::Product);
    double E=itsMPO->GetExpectation(itsMPS);
    double o=itsMPS->GetOverlap();
    cout << "Before Normalize Neel E=" << E << "  Overlap=" << o << endl;
    cout << "Norm Status=" << itsMPS->GetNormStatus() << endl;
    itsMPS->Normalize(MatrixProductSite::Right);
     E=itsMPO->GetExpectation(itsMPS);
     o=itsMPS->GetOverlap();
    cout << "After NormalizeNeel E=" << E << "  Overlap=" << o << endl;
    cout << "Norm Status=" << itsMPS->GetNormStatus() << endl;
    itsMPS->SweepRight(itsMPO);
    itsMPS->SweepLeft (itsMPO);
//    itsMPS->SweepRight(itsMPO);
//    itsMPS->SweepLeft (itsMPO);
//    itsMPS->SweepRight(itsMPO);
//    itsMPS->SweepLeft (itsMPO);
}
