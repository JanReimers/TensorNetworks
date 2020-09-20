#include "Tests.H"
#include "TensorNetworks/Hamiltonian.H"
#include "TensorNetworks/MPO.H"
#include "TensorNetworks/IterationSchedule.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/Factory.H"
#include "Operators/MPO_SpatialTrotter.H"

#include "oml/matrix.h"
#include "oml/stream.h"
#include "oml/stopw.h"

using std::setw;
using TensorNetworks::TrotterOrder;
using TensorNetworks::FirstOrder;
using TensorNetworks::SecondOrder;
using TensorNetworks::FourthOrder;

class ImaginaryTimeTesting : public ::testing::Test
{
public:
    typedef TensorNetworks::MatrixT MatrixT;

    ImaginaryTimeTesting()
    : eps(1.0e-13)
    , itsFactory(TensorNetworks::Factory::GetFactory())
    {
        StreamableObject::SetToPretty();

    }

    void Setup(int L, double S, int D)
    {
        itsH=itsFactory->Make1D_NN_HeisenbergHamiltonian(L,S,1.0,1.0,0.0);
        itsMPS=itsH->CreateMPS(D);
    }


    double eps;
    const TensorNetworks::Factory* itsFactory=TensorNetworks::Factory::GetFactory();
    Hamiltonian*         itsH;
    MPS*                 itsMPS;
};



TEST_F(ImaginaryTimeTesting,TestApplyIdentity)
{
    int D=1;
    Setup(3,0.5,D);
    MPS* Psi1=itsH->CreateMPS(D);
    Psi1->InitializeWith(TensorNetworks::Neel);
    double E1=Psi1->GetExpectation(itsH);
    Psi1->Normalize(TensorNetworks::DRight);
    Psi1->Normalize(TensorNetworks::DLeft );
    EXPECT_NEAR(Psi1->GetExpectation(itsH) ,E1,eps);

    OperatorWRepresentation* IWO=itsFactory->MakeIdentityOperator();
    Operator* IO=itsH->CreateOperator(IWO);
    MPS* Psi2=*IO**Psi1;
    EXPECT_NEAR(Psi2->GetExpectation(itsH) ,E1,eps);
    delete Psi1;
    delete Psi2;
}


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
    MPS* Psi1=itsH->CreateMPS(D);
    Psi1->InitializeWith(TensorNetworks::Random);
    Psi1->Normalize(TensorNetworks::DRight);
    //
    //  Psi2 = W*Psi1
    //
    MPS* Psi2=Psi1->Apply(W);
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
    IterationScheduleLine l1={10,5,0.2,TensorNetworks::FirstOrder,eps};

    IterationSchedule is;
    is.Insert(l1);
    is.Insert({10,5,0.1,TensorNetworks::FirstOrder,eps});
    cout << is;

}


TEST_F(ImaginaryTimeTesting,MPOCompressSeconderOrderTrotter_dt0)
{
    int D=8,L=9;
    double dt=0.0,epsSVD=1e-5                                                                                                                                                                                                        ;
    Setup(L,0.5,D);
    MPS* Psi1=itsH->CreateMPS(D);
    Psi1->InitializeWith(TensorNetworks::Random);
    Psi1->Normalize(TensorNetworks::DLeft );
    Psi1->Normalize(TensorNetworks::DRight);

    //Since dt=0 W should be unit operator.
    MPO* W=itsH->CreateOperator(dt,TensorNetworks::SecondOrder);

    MPS* Psi2=Psi1->Apply(W);
    EXPECT_NEAR(Psi1->GetOverlap(Psi2),1.0,eps);

    W->Compress(0,epsSVD);
    MPS* Psi3=Psi1->Apply(W);
    EXPECT_NEAR(Psi1->GetOverlap(Psi3),1.0,1e-7);
    EXPECT_NEAR(Psi2->GetOverlap(Psi3),1.0,1e-7);
    EXPECT_NEAR(Psi3->GetOverlap(Psi3),1.0,1e-6);

    delete Psi3;
    delete Psi2;
    delete Psi1;
    delete W;
}

TEST_F(ImaginaryTimeTesting,MPOCompressSeconderOrderTrotter_dt05)
{
    int D=8,L=9;
    double dt=0.05,epsSVD=1e-5                                                                                                                                                                                                        ;
    Setup(L,0.5,D);
    MPS* Psi1=itsH->CreateMPS(D);
    Psi1->InitializeWith(TensorNetworks::Random);
    Psi1->Normalize(TensorNetworks::DLeft );
    Psi1->Normalize(TensorNetworks::DRight);

    //Since dt=0 W should be unit operator.
    MPO* W=itsH->CreateOperator(dt,TensorNetworks::SecondOrder);

    MPS* Psi2=Psi1->Apply(W);
    W->Compress(0,epsSVD);
    MPS* Psi3=Psi1->Apply(W);
//    EXPECT_NEAR(Psi3->GetOverlap(Psi3),1.0,1e-6);
//
    W->Compress(0,epsSVD);
//    W->Report(cout);
    MPS* Psi4=Psi1->Apply(W);


    double O23=Psi2->GetOverlap(Psi3);
    cout << std::fixed << std::setprecision(9) << "O23=" << O23 << endl;
    double O24=Psi2->GetOverlap(Psi4);
    cout << "O24=" << O24 << endl;
    double O34=Psi3->GetOverlap(Psi4);
    cout << "O34=" << O34 << endl;
    double O33=Psi3->GetOverlap(Psi3);
    cout << "O33=" << O33 << endl;
    double O44=Psi4->GetOverlap(Psi4);
    cout << "O44=" << O44 << endl;
    EXPECT_NEAR(O23,O24,eps);
    EXPECT_NEAR(O34,O44,eps); //The confirm that a second compression on W is a no-op.
    EXPECT_NEAR(O33,O44,eps);

    delete Psi4;
    delete Psi3;
    delete Psi2;
    delete Psi1;
    delete W;
}


TEST_F(ImaginaryTimeTesting,MPOCompressFourthOrderTrotter)
{
    int D=8,L=9;
    double dt=0.1;
    Setup(L,0.5,D);

    MPO* W=itsH->CreateOperator(dt,TensorNetworks::FourthOrder);
    EXPECT_EQ(W->GetMaxDw(),16);

}

#ifndef DEBUG
TEST_F(ImaginaryTimeTesting,TestITimeFirstOrderTrotter)
{
    int D=8,L=9;
    Setup(L,0.5,D);
    MPS* Psi1=itsH->CreateMPS(D);
//    Psi1->Report(cout);
    Psi1->InitializeWith(TensorNetworks::Random);
    Psi1->Normalize(TensorNetworks::DRight);
//    double E1=Psi1->GetExpectation(itsH);
//    cout << "E1=" << std::fixed << E1 << endl;
//    cout << "Psi1 overlap=" << Psi1->GetOverlap(Psi1) << endl;

    Epsilons eps(1e-12);
    eps.itsMPOCompressEpsilon=1e-5;

    IterationSchedule is;
    eps.itsDelatNormEpsilon=1e-5;
    eps.itsDelatEnergy1Epsilon=1e-3;
    is.Insert({50,0,0.5,FirstOrder,eps});
    eps.itsDelatEnergy1Epsilon=1e-5;
    is.Insert({500,0,0.2,FirstOrder,eps});
    eps.itsDelatEnergy1Epsilon=1e-5;
    is.Insert({500,1,0.1,FirstOrder,eps});
    eps.itsDelatEnergy1Epsilon=1e-6;
    is.Insert({500,2,0.05,FirstOrder,eps});
    eps.itsDelatEnergy1Epsilon=1e-7;
    is.Insert({500,3,0.02,FirstOrder,eps});
    eps.itsDelatEnergy1Epsilon=1e-8;
    is.Insert({500,4,0.01,FirstOrder,eps});
    eps.itsDelatEnergy1Epsilon=3e-9;
    is.Insert({500,5,0.005,FirstOrder,eps});
    eps.itsDelatEnergy1Epsilon=1e-9;
    is.Insert({500,6,0.002,FirstOrder,eps});
    cout << is;

    Psi1->FindiTimeGroundState(itsH,is);

    double E2=Psi1->GetExpectation(itsH);
    EXPECT_NEAR(E2/(L-1),-0.46703753,1e-5);

    delete Psi1;


}

/*
TEST_F(ImaginaryTimeTesting,TestITimeSecondOrderTrotter)
{
    int D=8,L=9;
    Setup(L,0.5,D);
    MPS* Psi1=itsH->CreateMPS(D);
//    Psi1->Report(cout);
    Psi1->InitializeWith(TensorNetworks::Random);
    Psi1->Normalize(TensorNetworks::DRight);
    double E1=Psi1->GetExpectation(itsH);
    cout << "E1=" << std::fixed << E1 << endl;
    cout << "Psi1 overlap=" << Psi1->GetOverlap(Psi1) << endl;

    Epsilons eps(1e-12);
    eps.itsMPOCompressEpsilon=1e-5;

    IterationSchedule is;
    eps.itsDelatNormEpsilon=1e-5;
    eps.itsDelatEnergy1Epsilon=1e-5;
    is.Insert({50,0,0.5,SecondOrder,eps});
    eps.itsDelatEnergy1Epsilon=1e-6;
    is.Insert({500,1,0.2,SecondOrder,eps});
    eps.itsDelatEnergy1Epsilon=1e-8;
    is.Insert({500,3,0.1,SecondOrder,eps});
    eps.itsDelatEnergy1Epsilon=1e-9;
    is.Insert({500,5,0.05,SecondOrder,eps});

    cout << is;

    Psi1->FindiTimeGroundState(itsH,is);

    double E2=Psi1->GetExpectation(itsH);
    EXPECT_NEAR(E2/(L-1),-0.467040178,1e-7);

    delete Psi1;
}

TEST_F(ImaginaryTimeTesting,TestITimeFourthOrderTrotter)
{
    int D=8,L=9;
    Setup(L,0.5,D);
    MPS* Psi1=itsH->CreateMPS(D);
//    Psi1->Report(cout);
    Psi1->InitializeWith(TensorNetworks::Random);
    Psi1->Normalize(TensorNetworks::DRight);
    double E1=Psi1->GetExpectation(itsH);
    cout << "E1=" << std::fixed << E1 << endl;
    cout << "Psi1 overlap=" << Psi1->GetOverlap(Psi1) << endl;

    Epsilons eps(1e-12);
    eps.itsMPOCompressEpsilon=1e-8;

    IterationSchedule is;
    eps.itsDelatNormEpsilon=1e-5;
    eps.itsDelatEnergy1Epsilon=1e-5;
    is.Insert({50,0,0.5,FourthOrder,eps});
    eps.itsDelatEnergy1Epsilon=1e-6;
    is.Insert({500,1,0.2,FourthOrder,eps});
    eps.itsDelatEnergy1Epsilon=1e-8;
    is.Insert({500,3,0.1,FourthOrder,eps});
    eps.itsDelatEnergy1Epsilon=1e-9;
    is.Insert({500,5,0.05,FourthOrder,eps});

    cout << is;

    Psi1->FindiTimeGroundState(itsH,is);

    double E2=Psi1->GetExpectation(itsH);
    EXPECT_NEAR(E2/(L-1),-0.46704019232164307,1e-7);

    delete Psi1;
}
*/
#endif
