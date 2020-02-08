#include "Tests.H"
#include "TensorNetworks/Hamiltonian.H"
//#include "TensorNetworks/OperatorWRepresentation.H"
//#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/Factory.H"
#include "TensorNetworks/LRPSupervisor.H"
#include "TensorNetworks/Epsilons.H"
#include "TensorNetworksImp/SpinCalculator.H"

#include "oml/stream.h"
#include "oml/array_io.h"

using std::setw;

class ExpectationsTesting : public ::testing::Test
{
public:
    typedef std::complex<double> complx;
    typedef TensorNetworks:: ArrayT  ArrayT;

    ExpectationsTesting()
        : itsFactory(TensorNetworks::Factory::GetFactory())
        , itsH(0)
        , itsMPS(0)
        , itsLRPSupervisor(new LRPSupervisor())
        , itsEps()
    {
        StreamableObject::SetToPretty();

    }
    ~ExpectationsTesting()
    {
        delete itsFactory;
        delete itsLRPSupervisor;
        delete itsH;
        delete itsMPS;
    }

    void Setup(int L, double S, int D)
    {
        delete itsH;
        delete itsMPS;
       itsH=itsFactory->Make1D_NN_HeisenbergHamiltonian(L,S,1.0,1.0,0.0);
        itsMPS=itsH->CreateMPS(D,itsEps);
        itsMPS->InitializeWith(TensorNetworks::Random);
        itsMPS->FindGroundState(itsH,20,itsEps,itsLRPSupervisor);
    }


    const TensorNetworks::Factory* itsFactory=TensorNetworks::Factory::GetFactory();
    Hamiltonian*         itsH;
    MatrixProductState*  itsMPS;
    LRPSupervisor*       itsLRPSupervisor;
    Epsilons             itsEps;
};


TEST_F(ExpectationsTesting,TestSpinCalculatorS12)
{
    SpinCalculator sc(0.5);
    cout << "Sx=" << sc.GetSx() << endl;
    cout << "Sy=" << sc.GetSy() << endl;
    cout << "Sz=" << sc.GetSz() << endl;
    cout << "Sm=" << sc.GetSm() << endl;
    cout << "Sp=" << sc.GetSp() << endl;
}

/*

TEST_F(ExpectationsTesting,TestSweepL9S1D2)
{
    int L=9,D=2;
    double S=0.5;
    Setup(L,S,D);

    double E=itsMPS->GetExpectationIterate(itsH);
    EXPECT_NEAR(E/(L-1),-0.45317425,1e-7);
}


TEST_F(ExpectationsTesting,TestOneSiteDMs)
{
    int L=9,D=4;
    double S=0.5;
    Setup(L,S,D);
    itsMPS->CalculateOneSiteDMs(itsLRPSupervisor);
}

TEST_F(ExpectationsTesting,TestOneSiteExpectations)
{
    int L=9,D=4;
    double S=0.5;
    Setup(L,S,D);
    itsMPS->CalculateOneSiteDMs(itsLRPSupervisor);

    SpinCalculator sc(S);
    ArrayT Sx=itsMPS->GetOneSiteExpectation(sc.GetSx());
    cout << "Sx=" << Sx << endl;
    ArrayT Sy=itsMPS->GetOneSiteExpectation(sc.GetSy());
    cout << "Sy=" << Sy << endl;
    ArrayT Sz=itsMPS->GetOneSiteExpectation(sc.GetSz());
    cout << "Sz=" << Sz << endl;

    for (int ia=0;ia<L;ia++)
    {
        double S=sqrt(Sx[ia]*Sx[ia] + Sy[ia]*Sy[ia] + Sz[ia]*Sz[ia]);
        cout << "Site " << ia << " S^2=" << S << endl;
    }
    double E=0.0;
    for (int ia=0;ia<L-1;ia++)
    {
        double S=Sx[ia]*Sx[ia+1] + Sy[ia]*Sy[ia+1] + Sz[ia]*Sz[ia+1];
        E+=S;
        cout << "Site " << ia << " Si*Si+1=" << S << endl;
    }
    cout << "E=" << E << endl;
}

*/

TEST_F(ExpectationsTesting,TestTwoSiteDMs)
{
    int L=9,D=4;
    double S=0.5;
    Setup(L,S,D);
    itsMPS->CalculateTwoSiteDMs(1,itsLRPSupervisor);
}

