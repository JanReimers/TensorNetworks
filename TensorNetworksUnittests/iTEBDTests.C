#include "Tests.H"
#include "TensorNetworks/Hamiltonian.H"
#include "TensorNetworks/iTEBDState.H"
#include "TensorNetworks/Factory.H"

using std::setw;

class iTEBDTests : public ::testing::Test
{
public:
    typedef TensorNetworks::MatrixRT MatrixRT;
    typedef TensorNetworks::Matrix4RT Matrix4RT;

    iTEBDTests()
    : eps(1.0e-13)
    , itsFactory(TensorNetworks::Factory::GetFactory())
    , itsH(0)
    , itsState(0)
    {
        StreamableObject::SetToPretty();
    }

    ~iTEBDTests()
    {
        delete itsFactory;
        if (itsH) delete itsH;
        if (itsState) delete itsState;
    }

    void Setup(int L, double S, int D)
    {
        if (itsH) delete itsH;
        if (itsState) delete itsState;
        itsH=itsFactory->Make1D_NN_HeisenbergHamiltonian(L,S,1.0,1.0,0.0);
        itsState=itsH->CreateiTEBDState(D);
    }


    double eps;
    TensorNetworks::Factory*     itsFactory=TensorNetworks::Factory::GetFactory();
    TensorNetworks::Hamiltonian* itsH;
    TensorNetworks::iTEBDState*  itsState;
};



TEST_F(iTEBDTests,TestApplyIdentity)
{
    int UnitCell=2,D=8;
    double dt=0.0;
    Setup(UnitCell,0.5,D);
    itsState->InitializeWith(TensorNetworks::Random);
    itsState->Normalize(TensorNetworks::DLeft);
    itsState->Report(cout);

    Matrix4RT expH=TensorNetworks::Hamiltonian::ExponentH(dt,itsH->BuildLocalMatrix());
    itsState->Apply(1,expH);
    itsState->Apply(2,expH);
    //itsState->Report(cout);

}
