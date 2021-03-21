#include "Tests.H"
//#include "TensorNetworks/Hamiltonian.H"
#include "TensorNetworks/iHamiltonian.H"
#include "TensorNetworks/iMPS.H"
#include "TensorNetworks/iMPO.H"
#include "TensorNetworks/Factory.H"
#include "TensorNetworks/SVCompressor.H"
#include "TensorNetworks/IterationSchedule.H"

using std::setw;
//using TensorNetworks::TriType;
using TensorNetworks::Random;
using TensorNetworks::Neel;
using TensorNetworks::DLeft;
using TensorNetworks::DRight;
using TensorNetworks::Gates;
//using TensorNetworks::Std;
using TensorNetworks::IterationSchedule;
using TensorNetworks::Epsilons;
using TensorNetworks::MPOForm;
using TensorNetworks::RegularLower;
using TensorNetworks::RegularUpper;
using TensorNetworks::iMPO;
//    typedef TensorNetworks::MatrixRT MatrixRT;
//    typedef TensorNetworks::Matrix4RT Matrix4RT;



class iVUMPSTests : public ::testing::Test
{
public:

    iVUMPSTests()
    : epsNorm(4e-11)
    , epsOrth(1e-11)
    , itsFactory(TensorNetworks::Factory::GetFactory())
    , itsiH(0)
    , itsiMPS(0)
    , itsCompressor(0)
    {
        StreamableObject::SetToPretty();
    }

    ~iVUMPSTests()
    {
        delete itsFactory;
        if (itsiH)         delete itsiH;
        if (itsiMPS)       delete itsiMPS;
        if (itsCompressor) delete itsCompressor;
    }

    void Setup(int L, double S, int D, double hx, double epsSVD,MPOForm f)
    {
        if (itsiH)         delete itsiH;
        if (itsiMPS)       delete itsiMPS;
        if (itsCompressor) delete itsCompressor;
        itsiH=itsFactory->Make1D_NN_TransverseIsingiHamiltonian(1,S,f,-1.0,hx);
        itsiMPS=itsiH->CreateiMPS(L,D,D*D*epsNorm,epsSVD);
        itsCompressor=itsFactory->MakeMPSCompressor(D,epsSVD);
    }


    double epsNorm,epsOrth;
    TensorNetworks::Factory*       itsFactory=TensorNetworks::Factory::GetFactory();
    TensorNetworks::iHamiltonian*  itsiH;
    TensorNetworks::iMPS*          itsiMPS;
    TensorNetworks::SVCompressorC* itsCompressor;
};


TEST_F(iVUMPSTests,TestSetup)
{
    int UnitCell=1,D=2;
    double S=0.5,hx=0.0,epsSVD=0.0;
    MPOForm f=RegularLower;
    Setup(UnitCell,S,D,hx,epsSVD,f);
    itsiMPS->InitializeWith(Random);
    EXPECT_EQ(itsiMPS->GetNormStatus(),"M");
}

TEST_F(iVUMPSTests,TestNormQR_D2_L1)
{
    int UnitCell=1,D=2;
    double hx=0.0,epsSVD=0.0;
    MPOForm f=RegularLower;
    for (double S=0.5;S<=2.5;S+=0.5)
    {
        Setup(UnitCell,S,D,hx,epsSVD,f);
        itsiMPS->InitializeWith(Random);
        itsiMPS->NormalizeQR(DLeft);
        EXPECT_EQ(itsiMPS->GetNormStatus(),"A");
        itsiMPS->NormalizeQR(DRight);
        EXPECT_EQ(itsiMPS->GetNormStatus(),"B");
    }
}

TEST_F(iVUMPSTests,TestNormQR_D6_L1)
{
    int UnitCell=1,D=6;
    double hx=0.0,epsSVD=0.0;
    MPOForm f=RegularLower;
    for (double S=0.5;S<=2.5;S+=0.5)
    {
        Setup(UnitCell,S,D,hx,epsSVD,f);
        itsiMPS->InitializeWith(Random);
        itsiMPS->NormalizeQR(DLeft);
        EXPECT_EQ(itsiMPS->GetNormStatus(),"A");
        itsiMPS->NormalizeQR(DRight);
        EXPECT_EQ(itsiMPS->GetNormStatus(),"B");
    }
}

TEST_F(iVUMPSTests,TestNormQR_D6_L10)
{
    int UnitCell=10,D=6;
    double hx=0.0,epsSVD=0.0;
    MPOForm f=RegularLower;
    for (double S=0.5;S<=2.5;S+=0.5)
    {
        Setup(UnitCell,S,D,hx,epsSVD,f);
        itsiMPS->InitializeWith(Random);
        itsiMPS->NormalizeQR(DLeft);
        EXPECT_EQ(itsiMPS->GetNormStatus(),"AAAAAAAAAA");
        itsiMPS->NormalizeQR(DRight);
        EXPECT_EQ(itsiMPS->GetNormStatus(),"BBBBBBBBBB");
    }
}

TEST_F(iVUMPSTests,TestFindGS_D2_L1)
{
    int UnitCell=1,D=2,maxIter=30;
    double S=0.5,hx=1.0,epsSVD=0.0,eps=1e-14;
    MPOForm f=RegularLower;
    Setup(UnitCell,S,D,hx,epsSVD,f);
    itsiMPS->InitializeWith(Random);

    IterationSchedule is;
    is.Insert({maxIter,D,eps});
    itsiMPS->FindVariationalGroundState(itsiH,is);

}
