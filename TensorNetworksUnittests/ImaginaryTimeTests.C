#include "Tests.H"
#include "TensorNetworks/Hamiltonian.H"
//#include "TensorNetworks/OperatorWRepresentation.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/Factory.H"
//#include "TensorNetworks/LRPSupervisor.H"
#include "TensorNetworks/Epsilons.H"

#include "oml/stream.h"
#include "oml/stopw.h"

using std::setw;

class ImaginaryTimeTesting : public ::testing::Test
{
public:
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


TEST_F(ImaginaryTimeTesting,TestIdentityOperator)
{
    double dt=0.1;
    Setup(10,0.5,2);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Report(cout);
    Operator* W_Odd =itsH->CreateOperator(dt,TensorNetworks::Odd);
    Operator* W_Even=itsH->CreateOperator(dt,TensorNetworks::Even);
    itsMPS->Apply(W_Odd);
    itsMPS->Report(cout);
    itsMPS->Apply(W_Even);
    itsMPS->Report(cout);
    itsMPS->Apply(W_Odd);
    itsMPS->Report(cout);
    itsMPS->Apply(W_Even);
    itsMPS->Report(cout);


//    EXPECT_NEAR(S,1.0,eps);
}
