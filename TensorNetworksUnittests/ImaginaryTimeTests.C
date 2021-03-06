#include "Tests.H"
#include "TensorNetworks/MPS.H"
#include "TensorNetworks/Hamiltonian.H"
#include "TensorNetworks/MPO.H"
#include "TensorNetworks/IterationSchedule.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/Factory.H"
#include "Operators/MPO_SpatialTrotter.H"

#include "oml/matrix.h"
//#include "oml/stream.h"
//#include "oml/stopw.h"

using std::setw;
using TensorNetworks::TrotterOrder;
using TensorNetworks::FirstOrder;
using TensorNetworks::SecondOrder;
using TensorNetworks::FourthOrder;
using TensorNetworks::MPOForm;
using TensorNetworks::RegularLower;
using TensorNetworks::RegularUpper;
using TensorNetworks::MPO;
using TensorNetworks::MPS;
using TensorNetworks::Random;
using TensorNetworks::DLeft;
using TensorNetworks::DRight;
using TensorNetworks::CNone;
using TensorNetworks::Parker;
using TensorNetworks::Epsilons;
using TensorNetworks::IterationSchedule;

class ImaginaryTimeTests : public ::testing::Test
{
public:
    typedef TensorNetworks::MatrixRT MatrixRT;

    ImaginaryTimeTests()
    : eps(1.0e-13)
    , itsFactory(TensorNetworks::Factory::GetFactory())
    , itsH(0)
    , itsMPS(0)
    {
        StreamableObject::SetToPretty();
    }
    ~ImaginaryTimeTests()
    {
        delete itsFactory;
        if (itsH)   delete itsH;
        if (itsMPS) delete itsMPS;
    }

    void Setup(int L, double S, int D)
    {
        if (itsH)   delete itsH;
        if (itsMPS) delete itsMPS;
        itsH=itsFactory->Make1D_NN_HeisenbergHamiltonian(L,S,RegularLower,1.0,1.0,0.0);
        itsMPS=itsH->CreateMPS(D);
        itsMPS->InitializeWith(TensorNetworks::Random);
    }


    double eps;
    TensorNetworks::Factory*          itsFactory;
    TensorNetworks::Hamiltonian*      itsH;
    TensorNetworks::MPS*              itsMPS;
};



TEST_F(ImaginaryTimeTests,TestApplyIdentity)
{
    int D=1;
    Setup(3,0.5,D);
    itsMPS->InitializeWith(TensorNetworks::Neel);
    double E1=itsMPS->GetExpectation(itsH);
    itsMPS->Normalize(TensorNetworks::DRight);
    itsMPS->Normalize(TensorNetworks::DLeft );
    EXPECT_NEAR(itsMPS->GetExpectation(itsH) ,E1,eps);

    TensorNetworks::MPO* IO=itsH->CreateUnitOperator();
    TensorNetworks::MPS* Psi2=*IO**itsMPS;
    EXPECT_NEAR(Psi2->GetExpectation(itsH) ,E1,eps);
    delete Psi2;
    delete IO;
}


TEST_F(ImaginaryTimeTests,TestMPPOCombineL8)
{
    int D=2,L=8;
    double S=0.5,dt=0.05000;
    Setup(L,S,D);

    // Create some Trotter 2nd order operators
    TensorNetworks::MPO_SpatialTrotter W_Odd (dt/2.0,TensorNetworks::Odd ,itsH);
    TensorNetworks::MPO_SpatialTrotter W_Even(dt    ,TensorNetworks::Even,itsH);
    //
    //  Now combine three trotters into one
    //
    TensorNetworks::MPO* W=itsH->CreateUnitOperator();
    W->Product(&W_Odd);
    W->Product(&W_Even);
    W->Product(&W_Odd);
    //
    //  Make a random normalized wave function
    //
    itsMPS->Normalize(TensorNetworks::DRight);
    //
    //  Psi2 = W*Psi1
    //
    TensorNetworks::MPS* Psi2=itsMPS->Apply(W);
    //
    //  Psi1 = W_Odd * W_Even * W_Odd * Psi1
    //
    itsMPS->ApplyInPlace(&W_Odd);
    itsMPS->ApplyInPlace(&W_Even);
    itsMPS->ApplyInPlace(&W_Odd);
    //
    //  At this point if the Combine function is working Psi1==Psi2
    //
    double O11=itsMPS->GetOverlap(itsMPS);
    double O12=itsMPS->GetOverlap(Psi2);
    double O21=Psi2->GetOverlap(itsMPS);
    double O22=Psi2->GetOverlap(Psi2);
    EXPECT_NEAR(O12/O11,1.0,1e-14);
    EXPECT_NEAR(O21/O11,1.0,1e-14);
    EXPECT_NEAR(O22/O11,1.0,1e-14);

    delete Psi2;
    delete W;
}
TEST_F(ImaginaryTimeTests,TestMPPOCombineL9)
{
    int D=2,L=9;
    double S=0.5,dt=0.05000;
    Setup(L,S,D);

    // Create some Trotter 2nd order operators
    TensorNetworks::MPO_SpatialTrotter W_Odd (dt/2.0,TensorNetworks::Odd ,itsH);
    TensorNetworks::MPO_SpatialTrotter W_Even(dt    ,TensorNetworks::Even,itsH);
    //
    //  Now combine three trotters into one
    //
    TensorNetworks::MPO* W=itsH->CreateUnitOperator();
    W->Product(&W_Odd);
    W->Product(&W_Even);
    W->Product(&W_Odd);
    //
    //  Make a random normalized wave function
    //
    itsMPS->Normalize(TensorNetworks::DRight);
    //
    //  Psi2 = W*Psi1
    //
    TensorNetworks::MPS* Psi2=itsMPS->Apply(W);
    //
    //  Psi1 = W_Odd * W_Even * W_Odd * Psi1
    //
    itsMPS->ApplyInPlace(&W_Odd);
    itsMPS->ApplyInPlace(&W_Even);
    itsMPS->ApplyInPlace(&W_Odd);
    //
    //  At this point if the Combine function is working Psi1==Psi2
    //
    double O11=itsMPS->GetOverlap(itsMPS);
    double O12=itsMPS->GetOverlap(Psi2);
    double O21=Psi2->GetOverlap(itsMPS);
    double O22=Psi2->GetOverlap(Psi2);
    EXPECT_NEAR(O12/O11,1.0,1e-14);
    EXPECT_NEAR(O21/O11,1.0,1e-14);
    EXPECT_NEAR(O22/O11,1.0,1e-14);

    delete Psi2;
    delete W;
}

TEST_F(ImaginaryTimeTests,TestIterationSchedule)
{
    TensorNetworks::Epsilons eps(1e-12);
    TensorNetworks::IterationScheduleLine l1={10,5,0.2,TensorNetworks::FirstOrder,eps};

    TensorNetworks::IterationSchedule is;
    is.Insert(l1);
    is.Insert({10,5,0.1,TensorNetworks::FirstOrder,eps});

}


TEST_F(ImaginaryTimeTests,MPOCompressSeconderOrderTrotter_dt0)
{
    int D=8,L=8;
    double dt=0.0,epsSVD=1e-12,epsMPO=1e-4;                                                                                                                                                                                                        ;
    Setup(L,0.5,D);
    itsMPS->InitializeWith(Random);
    itsMPS->Normalize(DLeft );
    itsMPS->Normalize(DRight);

    //Since dt=0 W should be unit operator.
    MPO* W=itsH->CreateOperator(dt,SecondOrder,CNone,epsMPO);

    MPS* Psi2=itsMPS->Apply(W);
    EXPECT_NEAR(itsMPS->GetOverlap(Psi2),1.0,eps);
    W->CanonicalForm();
    W->Compress(Parker,0,epsSVD);
    MPS* Psi3=itsMPS->Apply(W);
    EXPECT_NEAR(itsMPS->GetOverlap(Psi3),1.0,1e-7);
    EXPECT_NEAR(Psi2->GetOverlap(Psi3),1.0,1e-7);
    EXPECT_NEAR(Psi3->GetOverlap(Psi3),1.0,1e-6);

    delete Psi3;
    delete Psi2;
    delete W;
}


TEST_F(ImaginaryTimeTests,MPOCompressSeconderOrderTrotter_dt05_FixedEps)
{
    int D=2,L=9;
    double dt=0.2,epsMPO=1e-6;                                                                                                                                                                                                     ;
    Setup(L,0.5,D);
    MPS* Psi1=itsH->CreateMPS(D);
    Psi1->InitializeWith(Random);
    Psi1->Normalize(DLeft );
    Psi1->Normalize(DRight);

    //Since dt=0 W should be unit operator.
    MPO* Wref=itsH->CreateOperator(dt,SecondOrder,CNone,epsMPO);

    MPS* Psi2=Psi1->Apply(Wref);
    MPO* W=itsH->CreateOperator(dt,SecondOrder,Parker,epsMPO);
//    W->Report(cout);
    double truncError=W->GetTruncationError();
    MPS* Psi3=Psi1->Apply(W);
    double O21=Psi2->GetOverlap(Psi1);
    double O31=Psi3->GetOverlap(Psi1);
    double O22=Psi2->GetOverlap(Psi2);
    double O23=Psi2->GetOverlap(Psi3);
    double O33=Psi3->GetOverlap(Psi3);
    EXPECT_EQ(W->GetMaxDw(),4);
    EXPECT_NEAR(O21,O31,truncError);
    EXPECT_NEAR(O22,O33,truncError); //The confirm that a second compression on W is accurate.
    EXPECT_NEAR(O22,O23,truncError); //The confirm that a second compression on W is accurate.

    delete Psi3;
    delete Psi2;
    delete Psi1;
    delete Wref;
    delete W;
}

TEST_F(ImaginaryTimeTests,MPOCompressSeconderOrderTrotter_dt05_FixedDw)
{
    int D=2,L=9,DwMax=4;
    double dt=0.2,epsSVD=0, epsMPO=1e-6;                                                                                                                                                                                                     ;
    Setup(L,0.5,D);
    MPS* Psi1=itsH->CreateMPS(D);
    Psi1->InitializeWith(Random);
    Psi1->Normalize(DLeft );
    Psi1->Normalize(DRight);

    //Since dt=0 W should be unit operator.
    MPO* W=itsH->CreateOperator(dt,SecondOrder,CNone,epsMPO);

    MPS* Psi2=Psi1->Apply(W);

    W->CanonicalForm();
    W->Compress(Parker,DwMax,epsSVD);
    double truncError=W->GetTruncationError();
//    W->Report(cout);
    MPS* Psi3=Psi1->Apply(W);
    double O22=Psi2->GetOverlap(Psi2);
    double O23=Psi2->GetOverlap(Psi3);
    double O33=Psi3->GetOverlap(Psi3);
    EXPECT_NEAR(O23,O22,truncError);
    EXPECT_NEAR(O23,O33,truncError); //The confirm that a second compression on W is a no-op.
    EXPECT_NEAR(O33,O22,truncError);

    delete Psi3;
    delete Psi2;
    delete Psi1;
    delete W;
}

TEST_F(ImaginaryTimeTests,MPOCompressFourthOrderTrotter)
{
    int D=2,L=8;
    double dt=0.1,epsMPO=1e-12;
    Setup(L,0.5,D);

    TensorNetworks::MPO* W=itsH->CreateOperator(dt,TensorNetworks::FourthOrder,TensorNetworks::Parker,epsMPO);
    EXPECT_EQ(W->GetMaxDw(),7);

}

TEST_F(ImaginaryTimeTests,TestITimeFirstOrderTrotterL2)
{
    int D=4,L=2;
    Setup(L,0.5,D);

    itsMPS->Normalize(DLeft);
    itsMPS->Normalize(DRight);

    Epsilons eps(1e-12);
    eps.itsMPOCompressEpsilon=1e-6;
    eps.itsDelatNormEpsilon=1e-5;
    eps.itsMPSCompressEpsilon=0;
    eps.itsDeltaLambdaEpsilon=1e-3;

    IterationSchedule is;
    eps.itsDeltaLambdaEpsilon=eps.itsDelatEnergy1Epsilon=1e-3;
    is.Insert({50,D,1,0,0.5,FirstOrder,eps});
    eps.itsDeltaLambdaEpsilon=eps.itsDelatEnergy1Epsilon=1e-5;
    is.Insert({500,D,1,0,0.2,FirstOrder,eps});
    eps.itsDeltaLambdaEpsilon=eps.itsDelatEnergy1Epsilon=1e-6;
    is.Insert({500,D,1,1,0.1,FirstOrder,eps});
    eps.itsDeltaLambdaEpsilon=eps.itsDelatEnergy1Epsilon=1e-7;
    is.Insert({500,D,1,2,0.05,FirstOrder,eps});
    eps.itsDeltaLambdaEpsilon=eps.itsDelatEnergy1Epsilon=1e-8;
    is.Insert({500,D,1,3,0.02,FirstOrder,eps});
    eps.itsDeltaLambdaEpsilon=eps.itsDelatEnergy1Epsilon=1e-9;
    is.Insert({500,D,1,4,0.01,FirstOrder,eps});

    itsMPS->FindiTimeGroundState(itsH,is);
    double E1=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(E1/(L-1),-0.75,1e-6);

    TensorNetworks::MPO* H2=itsH->CreateH2Operator();
//    H2->Report(cout);
    double E2=itsMPS->GetExpectation(H2);
    EXPECT_NEAR(E2,E1*E1,1e-6);
}

TEST_F(ImaginaryTimeTests,TestITimeSecondOrderTrotterL2)
{
    int D=2,L=2;
    Setup(L,0.5,D);
    MPS* Psi1=itsH->CreateMPS(D);
    Psi1->InitializeWith(Random);
    Psi1->Normalize(DRight);

    Epsilons eps(1e-12);
    eps.itsMPOCompressEpsilon=1e-14;
    eps.itsMPSCompressEpsilon=0.0; //Just Dmax for compression
    eps.itsDelatNormEpsilon=1e-5;

    IterationSchedule is;
    eps.itsDeltaLambdaEpsilon=eps.itsDelatEnergy1Epsilon=1e-5;
    eps.itsMPSCompressEpsilon=0.0; //Just Dmax for compression
    is.Insert({50 ,D,1,0,0.5,SecondOrder,eps});
    eps.itsDeltaLambdaEpsilon=eps.itsDelatEnergy1Epsilon=1e-8;
    is.Insert({500,D,1,3,0.1,SecondOrder,eps});
    eps.itsDeltaLambdaEpsilon=eps.itsDelatEnergy1Epsilon=1e-10;
    is.Insert({500,D,1,3,0.05,SecondOrder,eps});

 //   cout << is;

    Psi1->FindiTimeGroundState(itsH,is);

    double E1=Psi1->GetExpectation(itsH);
    EXPECT_NEAR(E1/(L-1),-0.75,1e-9);

    TensorNetworks::MPO* H2=itsH->CreateH2Operator();
    double E2=Psi1->GetExpectation(H2);
    EXPECT_NEAR(E2,E1*E1,1e-9);
    delete Psi1;
}

TEST_F(ImaginaryTimeTests,TestITimeFourthOrderTrotterL2)
{
    int D=2,L=2;
    Setup(L,0.5,D);
    TensorNetworks::MPS* Psi1=itsH->CreateMPS(D);
    Psi1->InitializeWith(TensorNetworks::Random);
    Psi1->Normalize(TensorNetworks::DRight);

    TensorNetworks::Epsilons eps(1e-12);
    eps.itsMPOCompressEpsilon=1e-8;
    eps.itsMPSCompressEpsilon=0.0;
    eps.itsDelatNormEpsilon=1e-5;

    TensorNetworks::IterationSchedule is;
    eps.itsDeltaLambdaEpsilon=eps.itsDelatEnergy1Epsilon=1e-5;
    is.Insert({50,D,1,0,0.5,FourthOrder,eps});
    eps.itsDeltaLambdaEpsilon=eps.itsDelatEnergy1Epsilon=1e-8;
    is.Insert({500,D,1,1,0.2,FourthOrder,eps});
    eps.itsDeltaLambdaEpsilon=eps.itsDelatEnergy1Epsilon=1e-9;

    Psi1->FindiTimeGroundState(itsH,is);

    double E1=Psi1->GetExpectation(itsH);
    EXPECT_NEAR(E1/(L-1),-0.75,1e-10);
    TensorNetworks::MPO* H2=itsH->CreateH2Operator();
    double E2=Psi1->GetExpectation(H2);
    EXPECT_NEAR(E2,E1*E1,1e-10);
    delete Psi1;
}


#ifndef DEBUG


TEST_F(ImaginaryTimeTests,TestITimeFirstOrderTrotter)
{
    int D=4,L=9,maxIter=1000,nopt=5;
    Setup(L,0.5,D);

    itsMPS->Normalize(TensorNetworks::DLeft);
    itsMPS->Normalize(TensorNetworks::DRight);

    TensorNetworks::Epsilons eps(1e-12);
    eps.itsMPOCompressEpsilon=1e-6;
    eps.itsDelatNormEpsilon=1e-5;
    eps.itsMPSCompressEpsilon=0;

    TensorNetworks::IterationSchedule is;
    eps.itsDeltaLambdaEpsilon=eps.itsDelatEnergy1Epsilon=1e-5;
    is.Insert({maxIter,D,1,nopt,0.5,FirstOrder,eps});
    eps.itsDeltaLambdaEpsilon=eps.itsDelatEnergy1Epsilon=1e-6;
    eps.itsDelatNormEpsilon=1e-10;
    is.Insert({maxIter,D,1,nopt,0.2,FirstOrder,eps});
    is.Insert({maxIter,D,1,nopt,0.1,FirstOrder,eps});
    is.Insert({maxIter,D,1,nopt,0.05,FirstOrder,eps});
    is.Insert({maxIter,D,1,nopt,0.02,FirstOrder,eps});
    is.Insert({maxIter,D,1,nopt,0.01,FirstOrder,eps});
    is.Insert({maxIter,D,1,nopt,0.005,FirstOrder,eps});

    itsMPS->FindiTimeGroundState(itsH,is);

    double E2=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(E2/(L-1),-0.46664265599414939,2e-6);
}

TEST_F(ImaginaryTimeTests,TestITimeSecondOrderTrotter_EpsLimitedCompression)
{
    int maxIter=100,nopt=5;
    int D=16,Dcompress=16,L=16; //Set DMax high
//    double epsSV=1e-9;
    Setup(L,0.5,D);
    itsMPS->NormalizeAndCompress(TensorNetworks::DLeft ,D,1e-5);
    itsMPS->NormalizeAndCompress(TensorNetworks::DRight,D,1e-5);

    TensorNetworks::Epsilons eps(1e-12);

    TensorNetworks::IterationSchedule is;

    eps.itsDeltaLambdaEpsilon=eps.itsDelatEnergy1Epsilon=1e-3;
    eps.itsMPOCompressEpsilon=1e-4;
    eps.itsMPSCompressEpsilon=1e-4; //Just Eps for compression
    eps.itsDelatNormEpsilon=1e-5;
    is.Insert({maxIter ,Dcompress,1,nopt,2.0,SecondOrder,eps});
    is.Insert({maxIter ,Dcompress,1,nopt,0.5,SecondOrder,eps});

    eps.itsDeltaLambdaEpsilon=eps.itsDelatEnergy1Epsilon=1e-4;
    eps.itsMPOCompressEpsilon=1e-4;
    eps.itsMPSCompressEpsilon=1e-4; //Just Eps for compression
    eps.itsDelatNormEpsilon=1e-7;
    is.Insert({maxIter,Dcompress,1,nopt,0.2,SecondOrder,eps});

    eps.itsDeltaLambdaEpsilon=eps.itsDelatEnergy1Epsilon=1e-5;
    eps.itsMPOCompressEpsilon=1e-6;
    eps.itsMPSCompressEpsilon=1e-6; //Just Eps for compression
    eps.itsDelatNormEpsilon=1e-8;
    is.Insert({maxIter,Dcompress,1,nopt,0.1,SecondOrder,eps});

    eps.itsDeltaLambdaEpsilon=eps.itsDelatEnergy1Epsilon=1e-6;
    eps.itsMPOCompressEpsilon=1e-9;
    eps.itsMPSCompressEpsilon=1e-8; //Just Eps for compression
    eps.itsDelatNormEpsilon=1e-10;
    is.Insert({maxIter,Dcompress,1,nopt,0.1,SecondOrder,eps});

    itsMPS->FindiTimeGroundState(itsH,is);

    double E2=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(E2/(L-1),-0.46078193,1e-7);
}

TEST_F(ImaginaryTimeTests,TestITimeFourthOrderTrotter)
{
    int maxIter=100,nopt=5,Dcompress=0;
    int D=16,L=16; //Set DMax high
//    double epsSV=1e-9;
    Setup(L,0.5,D);
    itsMPS->NormalizeAndCompress(TensorNetworks::DLeft ,D,1e-5);
    itsMPS->NormalizeAndCompress(TensorNetworks::DRight,D,1e-5);

    TensorNetworks::Epsilons eps(1e-12);

    TensorNetworks::IterationSchedule is;

    eps.itsDeltaLambdaEpsilon=eps.itsDelatEnergy1Epsilon=1e-3;
    eps.itsMPOCompressEpsilon=1e-5;
    eps.itsMPSCompressEpsilon=1e-5; //Just Eps for compression
    eps.itsDelatNormEpsilon=1e-5;
    is.Insert({maxIter ,Dcompress,1,nopt,0.5,FourthOrder,eps});

    eps.itsDeltaLambdaEpsilon=eps.itsDelatEnergy1Epsilon=1e-5;
    eps.itsMPOCompressEpsilon=1e-7;
    eps.itsMPSCompressEpsilon=1e-6; //Just Eps for compression
    eps.itsDelatNormEpsilon=1e-10;
    is.Insert({maxIter,Dcompress,1,nopt,0.3,FourthOrder,eps});

    eps.itsDeltaLambdaEpsilon=eps.itsDelatEnergy1Epsilon=1e-7;
    eps.itsMPOCompressEpsilon=1e-12;
    eps.itsMPSCompressEpsilon=1e-7; //Just Eps for compression
    is.Insert({maxIter,Dcompress,1,nopt,0.1,FourthOrder,eps});

    itsMPS->FindiTimeGroundState(itsH,is);

    double E2=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(E2/(L-1),-0.46078193,2e-6);
}
#endif // DEBUG

#ifdef RunLongTests

TEST_F(ImaginaryTimeTests,TestITimeSecondOrderTrotter)
{
    int maxIter=100,nopt=5;
    int D=4,L=9;
    Setup(L,0.5,D);
    TensorNetworks::MPS* Psi1=itsH->CreateMPS(D);
    Psi1->InitializeWith(TensorNetworks::Random);
    Psi1->Normalize(TensorNetworks::DRight);

    TensorNetworks::Epsilons eps(1e-12);
    eps.itsMPOCompressEpsilon=1e-8;
    eps.itsMPSCompressEpsilon=0.0; //Just Dmax for compression
    eps.itsDelatNormEpsilon=1e-5;

    TensorNetworks::IterationSchedule is;
    eps.itsDeltaLambdaEpsilon=eps.itsDelatEnergy1Epsilon=1e-5;
    eps.itsMPSCompressEpsilon=0.0; //Just Dmax for compression
    is.Insert({maxIter,D,1,nopt,2.0,SecondOrder,eps});
    is.Insert({maxIter,D,1,nopt,0.5,SecondOrder,eps});
    eps.itsDeltaLambdaEpsilon=eps.itsDelatEnergy1Epsilon=1e-6;
    is.Insert({maxIter,D,1,nopt,0.2,SecondOrder,eps});
    is.Insert({maxIter,D,1,nopt,0.1,SecondOrder,eps});
    is.Insert({maxIter,D,1,nopt,0.05,SecondOrder,eps});
    is.Insert({maxIter,D,1,nopt,0.02,SecondOrder,eps});
    is.Insert({maxIter,D,1,nopt,0.01,SecondOrder,eps});

    Psi1->FindiTimeGroundState(itsH,is);

    double E2=Psi1->GetExpectation(itsH);
    EXPECT_NEAR(E2/(L-1),-0.46664265599414939,1e-7);

    delete Psi1;
}

TEST_F(ImaginaryTimeTests,TestITimeFourthOrderTrotter)
{
    int maxIter=100,nopt=5;
    int D=4,L=9;
    Setup(L,0.5,D);
    TensorNetworks::MPS* Psi1=itsH->CreateMPS(D);
    Psi1->InitializeWith(TensorNetworks::Random);
    Psi1->Normalize(TensorNetworks::DRight);

    TensorNetworks::Epsilons eps(1e-12);
    eps.itsMPOCompressEpsilon=1e-8;
    eps.itsMPSCompressEpsilon=0.0;
    eps.itsDelatNormEpsilon=1e-5;

    TensorNetworks::IterationSchedule is;
    eps.itsDeltaLambdaEpsilon=eps.itsDelatEnergy1Epsilon=1e-4;
    is.Insert({maxIter,D,1,nopt,0.5,FourthOrder,eps});
    eps.itsDeltaLambdaEpsilon=eps.itsDelatEnergy1Epsilon=1e-6;
    is.Insert({maxIter,D,1,nopt,0.2,FourthOrder,eps});
    is.Insert({maxIter,D,1,nopt,0.1,FourthOrder,eps});
    is.Insert({maxIter,D,1,nopt,0.05,FourthOrder,eps});
    is.Insert({maxIter,D,1,nopt,0.02,FourthOrder,eps});
    is.Insert({maxIter,D,1,nopt,0.01,FourthOrder,eps});

    Psi1->FindiTimeGroundState(itsH,is);

    double E2=Psi1->GetExpectation(itsH);
    EXPECT_NEAR(E2/(L-1),-0.46664265599414939,1e-7);

    delete Psi1;
}

#endif

// Very slow to build fourth order exp(-tH)
//TEST_F(ImaginaryTimeTests,TestITimeFourthOrderTrotter)
//{
//    int D=6,L=3,maxIter=1000,deltaD=1;
//    double S=2.5;
//    Setup(L,S,D);
//    TensorNetworks::MPS* Psi1=itsH->CreateMPS(D);
//    Psi1->InitializeWith(TensorNetworks::Random);
//    Psi1->Normalize(TensorNetworks::DRight);
//
//    TensorNetworks::Epsilons eps(1e-12);
//    eps.itsMPOCompressEpsilon=1e-14;
//    eps.itsMPSCompressEpsilon=0.0;
//    eps.itsDelatNormEpsilon=1e-5;
//
//    TensorNetworks::IterationSchedule is;
//    eps.itsDelatEnergy1Epsilon=1e-5;
//    is.Insert({maxIter,D,deltaD,0,0.5,FourthOrder,eps});
//    eps.itsDelatEnergy1Epsilon=1e-7;
//    is.Insert({maxIter,D,deltaD,1,0.2,FourthOrder,eps});
//    eps.itsDelatEnergy1Epsilon=1e-9;
//    is.Insert({maxIter,D,deltaD,3,0.1,FourthOrder,eps});
//    eps.itsDelatEnergy1Epsilon=1e-11;
//    is.Insert({maxIter,D,deltaD,5,0.01,FourthOrder,eps});
//    is.Insert({maxIter,D,deltaD,5,0.001,FourthOrder,eps});
//
////    cout << is;
//
//    Psi1->FindiTimeGroundState(itsH,is);
//
//    double E2=Psi1->GetExpectation(itsH);
//    EXPECT_NEAR(E2/(L-1),-7.5,1e-7);
//
//    delete Psi1;
//}
