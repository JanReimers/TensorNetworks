#include "Tests.H"
#include "TensorNetworks/Hamiltonian.H"
#include "TensorNetworks/MPO.H"
#include "TensorNetworks/IterationSchedule.H"
//#include "TensorNetworks/OperatorWRepresentation.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/Factory.H"
#include "TensorNetworks/LRPSupervisor.H"
#include "TensorNetworks/Epsilons.H"
#include "Operators/MPO_SpatialTrotter.H"

#include "oml/matrix.h"
#include "oml/stream.h"
#include "oml/stopw.h"

using std::setw;

class ImaginaryTimeTesting : public ::testing::Test
{
public:
    typedef TensorNetworks::MatrixT MatrixT;

    ImaginaryTimeTesting()
    : eps(1.0e-13)
    , itsFactory(TensorNetworks::Factory::GetFactory())
    , itsEps()
    , itsSupervisor(new LRPSupervisor())
    {
        StreamableObject::SetToPretty();

    }

    void Setup(int L, double S, int D)
    {
        itsH=itsFactory->Make1D_NN_HeisenbergHamiltonian(L,S,1.0,1.0,0.0);
        itsMPS=itsH->CreateMPS(D,itsEps);
    }


    double eps;
    const TensorNetworks::Factory* itsFactory=TensorNetworks::Factory::GetFactory();
    Hamiltonian*         itsH;
    MatrixProductState*  itsMPS;
    Epsilons             itsEps;
    LRPSupervisor*       itsSupervisor;
};

/*
TEST_F(ImaginaryTimeTesting,TestApplyInPlaceOddEven)
{
    double dt=0.1;
    Setup(10,0.5,2);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Report(cout);
    Operator* W_Odd =itsH->CreateOperator(dt,TensorNetworks::Odd);
    Operator* W_Even=itsH->CreateOperator(dt,TensorNetworks::Even);
    itsMPS->ApplyInPlace(W_Odd);
    itsMPS->Report(cout);
    itsMPS->ApplyInPlace(W_Even);
    itsMPS->Report(cout);
    itsMPS->ApplyInPlace(W_Odd);
    itsMPS->Report(cout);
    itsMPS->ApplyInPlace(W_Even);
    itsMPS->Report(cout);


//    EXPECT_NEAR(S,1.0,eps);
}

TEST_F(ImaginaryTimeTesting,TestApplyOddEven)
{
    double dt=0.1;
    Setup(10,0.5,8);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Report(cout);
    Operator* W_Odd =itsH->CreateOperator(dt,TensorNetworks::Odd);
    Operator* W_Even=itsH->CreateOperator(dt,TensorNetworks::Even);
    MatrixProductState* Psi1=itsMPS->Apply(W_Odd);
    MatrixProductState* Psi2=Psi1->Apply(W_Even);
    Psi2->NormalizeAndCompress(TensorNetworks::DLeft,8,itsSupervisor);
    Psi2->Report(cout);
    MatrixProductState* Psi3=Psi2->Apply(W_Odd);
    MatrixProductState* Psi4=Psi3->Apply(W_Even);
    Psi4->NormalizeAndCompress(TensorNetworks::DLeft,8,itsSupervisor);
    Psi4->Report(cout);

    delete Psi4;
    delete Psi3;
    delete Psi2;
    delete Psi1;


//    EXPECT_NEAR(S,1.0,eps);
}

TEST_F(ImaginaryTimeTesting,TestApplyIdentity)
{
    int D=1;
    Setup(3,0.5,D);
    MatrixProductState* Psi1=itsH->CreateMPS(D,itsEps);
    Psi1->InitializeWith(TensorNetworks::Neel);
    double E1=Psi1->GetExpectation(itsH);
    Psi1->Normalize(TensorNetworks::DRight,itsSupervisor);
    Psi1->Normalize(TensorNetworks::DLeft ,itsSupervisor);
    EXPECT_NEAR(Psi1->GetExpectation(itsH) ,E1,eps);

    OperatorWRepresentation* IWO=itsFactory->MakeIdentityOperator();
    Operator* IO=itsH->CreateOperator(IWO);
    MatrixProductState* Psi2=Psi1->Apply(IO);
    EXPECT_NEAR(Psi2->GetExpectation(itsH) ,E1,eps);
    delete Psi1;
    delete Psi2;
}


TEST_F(ImaginaryTimeTesting,TestTryGroundStateDmax8)
{
    int D=8,L=9;
    double dt=0.05000;
    Setup(L,0.5,D);
    MatrixProductState* Psi1=itsH->CreateMPS(D,itsEps);
    MatrixProductState* Psi2=0;
    Psi1->InitializeWith(TensorNetworks::Neel);
    double E1=Psi1->GetExpectation(itsH);
    cout << "E1=" << std::fixed << E1 << endl;

    Operator* W_Odd =itsH->CreateOperator(dt/2.0,TensorNetworks::Odd);
    Operator* W_Even=itsH->CreateOperator(dt,TensorNetworks::Even);

    for (int niter=1;niter<=50;niter++)
    {
        Psi2=Psi1->Apply(W_Odd);
        Psi2->NormalizeAndCompress(TensorNetworks::DLeft,D,itsSupervisor);
        Psi2->ApplyInPlace(W_Even);
        Psi2->NormalizeAndCompress(TensorNetworks::DLeft,D,itsSupervisor);
        Psi2->ApplyInPlace(W_Odd);
        Psi2->NormalizeAndCompress(TensorNetworks::DLeft,D,itsSupervisor);
//        Psi2->Report(cout);
        cout << "E=" << std::fixed << Psi2->GetExpectation(itsH) << endl;
        delete Psi1;
        Psi1=Psi2;
    }

    double E2=Psi2->GetExpectation(itsH);
    EXPECT_NEAR(E2/(L-1),-0.46703753,1e-2);

    delete Psi2;


}
*/

TEST_F(ImaginaryTimeTesting,TestMPPOCombine)
{
    int D=2,L=9;
    double S=0.5,dt=0.05000;
    Setup(L,S,D);

    TensorNetworks::Matrix4T H12=itsH->BuildLocalMatrix(); //Full H matrix for two sites 1&2

    // Create some Trotter 2nd order operators
    MPO_SpatialTrotter W_Odd (dt/2.0,TensorNetworks::Odd ,L,2*S+1,H12);
    MPO_SpatialTrotter W_Even(dt    ,TensorNetworks::Even,L,2*S+1,H12);
    //
    //  Now combine three trotters into one
    //
    MPO* W=itsH->CreateUnitOperator();
    W->Combine(&W_Odd);
    W->Combine(&W_Even);
    W->Combine(&W_Odd);
    //
    //  Make a random normalized wave function
    //
    MatrixProductState* Psi1=itsH->CreateMPS(D,itsEps);
    Psi1->InitializeWith(TensorNetworks::Random);
    Psi1->Normalize(TensorNetworks::DRight,itsSupervisor);
    //
    //  Psi2 = W*Psi1
    //
    MatrixProductState* Psi2=Psi1->Apply(W);
    //
    //  Psi1 = W_Odd * W_Even * W_Odd * Psi1
    //
    Psi1->ApplyInPlace(&W_Odd);
    Psi1->ApplyInPlace(&W_Even);
    Psi1->ApplyInPlace(&W_Odd);
    //
    //  At this point if the Combine function is working Psi1==Psi2
    //
    double O11=Psi1->GetOverlap(Psi1);
    double O12=Psi1->GetOverlap(Psi2);
    double O21=Psi2->GetOverlap(Psi1);
    double O22=Psi2->GetOverlap(Psi2);
    EXPECT_NEAR(O12/O11,1.0,1e-14);
    EXPECT_NEAR(O21/O11,1.0,1e-14);
    EXPECT_NEAR(O22/O11,1.0,1e-14);

    delete Psi1;
    delete Psi2;
    delete W;
}
TEST_F(ImaginaryTimeTesting,TestIterationSchedule)
{
    Epsilons eps(1e-12);
    IterationScheduleLine l1={10,5,8,0.2,eps};

    IterationSchedule is;
    is.Insert(l1);
    is.Insert({10,5,8,0.1,eps});
    cout << is;

}

TEST_F(ImaginaryTimeTesting,TestOptimize)
{
    int D=8,L=9;
    Setup(L,0.5,D);
    MatrixProductState* Psi1=itsH->CreateMPS(D,itsEps);
//    Psi1->Report(cout);
    Psi1->InitializeWith(TensorNetworks::Random);
    Psi1->Normalize(TensorNetworks::DRight,itsSupervisor);
    double E1=Psi1->GetExpectation(itsH);
    cout << "E1=" << std::fixed << E1 << endl;
    cout << "Psi1 overlap=" << Psi1->GetOverlap(Psi1) << endl;

    Epsilons eps(1e-12);

    IterationSchedule is;
    eps.itsDelatNormEpsilon=1e-5;
    eps.itsDelatEnergy1Epsilon=1e-3;
    is.Insert({50,0,8,0.5,eps});
    eps.itsDelatEnergy1Epsilon=1e-5;
    is.Insert({500,0,8,0.2,eps});
    eps.itsDelatEnergy1Epsilon=1e-5;
    is.Insert({500,0,8,0.1,eps});
    eps.itsDelatEnergy1Epsilon=1e-6;
    is.Insert({500,0,8,0.05,eps});
    eps.itsDelatEnergy1Epsilon=3e-7;
    is.Insert({500,1,8,0.02,eps});
    eps.itsDelatEnergy1Epsilon=1e-7;
    is.Insert({500,1,8,0.01,eps});
    eps.itsDelatEnergy1Epsilon=3e-8;
    is.Insert({500,1,8,0.005,eps});
    eps.itsDelatEnergy1Epsilon=1e-9;
    is.Insert({500,1,8,0.002,eps});
    eps.itsDelatEnergy1Epsilon=3e-10;
    is.Insert({500,1,8,0.001,eps});
    cout << is;

    Psi1->FindGroundState(itsH,is,itsSupervisor);


//        double O22=Psi2->GetOverlap(Psi2);
//        double O21=Psi2->GetOverlap(Psi1);
//        double O12=Psi1->GetOverlap(Psi2);
//        double O11=Psi1->GetOverlap(Psi1);
//        cout << "O11 O12 O21 O22 delta=" << O11 << " " << O12 << " " << O21 << " " << O22 << " " << O11-O12-O21+O22 << endl;
//

    double E2=Psi1->GetExpectation(itsH);
    EXPECT_NEAR(E2/(L-1),-0.46703753,1e-7);

    delete Psi1;


}

