#include "Tests.H"
#include "TensorNetworks/Hamiltonian.H"
#include "TensorNetworks/iTEBDState.H"
#include "TensorNetworks/Factory.H"

using std::setw;

class iTEBDTesting : public ::testing::Test
{
public:
    typedef TensorNetworks::MatrixRT MatrixRT;
    typedef TensorNetworks::Matrix4T Matrix4T;

    iTEBDTesting()
    : eps(1.0e-13)
    , itsFactory(TensorNetworks::Factory::GetFactory())
    {
        StreamableObject::SetToPretty();

    }

    void Setup(int L, double S, int D)
    {
        itsH=itsFactory->Make1D_NN_HeisenbergHamiltonian(L,S,1.0,1.0,0.0);
        itsState=itsH->CreateiTEBDState(D);
    }


    double eps;
    const TensorNetworks::Factory* itsFactory=TensorNetworks::Factory::GetFactory();
    Hamiltonian*         itsH;
    iTEBDState*          itsState;
};



TEST_F(iTEBDTesting,TestApplyIdentity)
{
    int UnitCell=2,D=8;
    double dt=0.0;
    Setup(UnitCell,0.5,D);
    itsState->InitializeWith(TensorNetworks::Random);
    itsState->Normalize(TensorNetworks::DLeft);
    itsState->Report(cout);

    Matrix4T expH=Hamiltonian::ExponentH(dt,itsH->BuildLocalMatrix());
    itsState->Apply(1,expH);
    itsState->Apply(2,expH);
    itsState->Report(cout);

}