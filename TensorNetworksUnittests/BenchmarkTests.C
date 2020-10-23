#include "Tests.H"
#include "TensorNetworks/Hamiltonian.H"
#include "TensorNetworks/IterationSchedule.H"
#include "TensorNetworks/MPO.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/Factory.H"
#include "TensorNetworksImp/SPDLogger.H"
#include <omp.h>

//#include "oml/stream.h"
//#include "oml/stopw.h"

using std::setw;

class BenchmarkTests : public ::testing::Test
{
public:
    BenchmarkTests()
    : eps(1.0e-13)
    , itsFactory(TensorNetworks::Factory::GetFactory())
    , itsH(0)
    , itsMPS(0)
    , itsLogger(new TensorNetworks::SPDLogger(-1))
    {
        StreamableObject::SetToPretty();
    }

    ~BenchmarkTests()
    {
        delete itsFactory;
        if (itsH) delete itsH;
        if (itsMPS) delete itsMPS;
    }

    void Setup(int L, double S, int D)
    {
        itsH=itsFactory->Make1D_NN_HeisenbergHamiltonian(L,S,1.0,1.0,0.0);
        itsMPS=itsH->CreateMPS(D,1e-12,1e-12);
        itsMPS->Inject(itsLogger);
    }


    double eps;
    TensorNetworks::Factory*          itsFactory;
    TensorNetworks::Hamiltonian*      itsH;
    TensorNetworks::MPS*              itsMPS;
    rc_ptr<TensorNetworks::TNSLogger> itsLogger;
};


TEST_F(BenchmarkTests,TestSweepL9S1D8)
{
    int L=9,D=8,maxIter=100,Nreplicates=1;
    double S=0.5;
    Setup(L,S,D);


    TensorNetworks::Epsilons eps(0.0);
    TensorNetworks::IterationSchedule is;
    is.Insert({maxIter,D,eps});
    double t_start=omp_get_wtime();
    for (int i=1;i<=Nreplicates;i++)
    {
        itsMPS->InitializeWith(TensorNetworks::Neel);
        itsMPS->FindVariationalGroundState(itsH,is);
    }
    double t_stop=omp_get_wtime();
    cout << "Run time = " << (t_stop-t_start)/Nreplicates << " sec/replicate.";
    itsMPS->Report(cout);

    double E=itsMPS->GetExpectation(itsH);
//    EXPECT_NEAR(E/(L-1),-0.4670402,1e-7); //For D=16
    EXPECT_NEAR(E/(L-1),-0.4670375,1e-7); //For D=8
}
