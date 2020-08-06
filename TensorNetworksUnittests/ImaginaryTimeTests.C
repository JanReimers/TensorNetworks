#include "Tests.H"
#include "TensorNetworks/Hamiltonian.H"
//#include "TensorNetworks/OperatorWRepresentation.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/Factory.H"
#include "TensorNetworks/LRPSupervisor.H"
#include "TensorNetworks/Epsilons.H"

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
};


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
    Psi2->NormalizeAndCompress(TensorNetworks::DLeft,8,new LRPSupervisor());
    Psi2->Report(cout);
    MatrixProductState* Psi3=Psi2->Apply(W_Odd);
    MatrixProductState* Psi4=Psi3->Apply(W_Even);
    Psi4->NormalizeAndCompress(TensorNetworks::DLeft,8,new LRPSupervisor());
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
    Psi1->Normalize(TensorNetworks::DRight,new LRPSupervisor());
    Psi1->Normalize(TensorNetworks::DLeft ,new LRPSupervisor());
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
        Psi2->NormalizeAndCompress(TensorNetworks::DLeft,D,new LRPSupervisor());
        Psi2->ApplyInPlace(W_Even);
        Psi2->NormalizeAndCompress(TensorNetworks::DLeft,D,new LRPSupervisor());
        Psi2->ApplyInPlace(W_Odd);
        Psi2->NormalizeAndCompress(TensorNetworks::DLeft,D,new LRPSupervisor());
//        Psi2->Report(cout);
        cout << "E=" << std::fixed << Psi2->GetExpectation(itsH) << endl;
        delete Psi1;
        Psi1=Psi2;
    }

    double E2=Psi2->GetExpectation(itsH);
    EXPECT_NEAR(E2/(L-1),-0.46703753,1e-2);

    delete Psi2;


}
