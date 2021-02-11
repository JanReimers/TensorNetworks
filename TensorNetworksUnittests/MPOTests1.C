#include "Tests.H"
#include "TensorNetworks/Factory.H"
#include "TensorNetworksImp/Hamiltonians/Hamiltonian_1D_NN_Heisenberg.H"
#include "Operators/OperatorElement.H"


//#include "TensorNetworksImp/MPS/MPSImp.H"
//#include "TensorNetworks/iTEBDState.H"
//#include "Operators/MPO_SpatialTrotter.H"
//#include "Containers/Matrix4.H"
//#include "TensorNetworks/Hamiltonian.H"
//#include "TensorNetworks/iHamiltonian.H"
//#include "TensorNetworks/SiteOperator.H"
//#include "TensorNetworks/MPO.H"
//#include "TensorNetworks/iMPO.H"
//#include "Operators/SiteOperatorImp.H"
//#include "Containers/Matrix6.H"

using dcmplx=TensorNetworks::dcmplx;

class MPOTests1 : public ::testing::Test
{
public:
//    typedef TensorNetworks::Matrix6CT Matrix6CT;
//    typedef TensorNetworks::MatrixRT  MatrixRT;
//    typedef TensorNetworks::MatrixCT  MatrixCT;
//    typedef TensorNetworks::Vector3CT Vector3CT;
//    typedef TensorNetworks::dcmplx     dcmplx;
    MPOTests1()
        : eps(1.0e-13)
        , itsFactory(TensorNetworks::Factory::GetFactory())
        , itsOperatorClient(0)
    {
        assert(itsFactory);
        StreamableObject::SetToPretty();
    }
    ~MPOTests1()
    {
        delete itsFactory;
        if (itsOperatorClient) delete itsOperatorClient;
    }

    void Setup(double S)
    {
        if (itsOperatorClient) delete itsOperatorClient;
        itsOperatorClient=new TensorNetworks::Hamiltonian_1D_NN_Heisenberg(S,1.0,1.0,0.0);
        assert(itsOperatorClient);
    }

    double eps;
           TensorNetworks::Factory*        itsFactory;
    const  TensorNetworks::OperatorClient* itsOperatorClient;
};

TEST_F(MPOTests1,MakeHamiltonian)
{
    Setup(0.5);
}

TEST_F(MPOTests1,OperatorElement1)
{
    double S=0.5;
    {
        TensorNetworks::OperatorSz Sz12(S);
        EXPECT_EQ(Sz12(0,0),-0.5);
        EXPECT_EQ(Sz12(1,0), 0.0);
        EXPECT_EQ(Sz12(0,1), 0.0);
        EXPECT_EQ(Sz12(1,1), 0.5);
    }

    {
        TensorNetworks::OperatorSp Sp12(S);
        EXPECT_EQ(Sp12(0,0), 0.0);
        EXPECT_EQ(Sp12(1,0), 1.0);
        EXPECT_EQ(Sp12(0,1), 0.0);
        EXPECT_EQ(Sp12(1,1), 0.0);
    }

    {
        TensorNetworks::OperatorSm Sm12(S);
        EXPECT_EQ(Sm12(0,0), 0.0);
        EXPECT_EQ(Sm12(1,0), 0.0);
        EXPECT_EQ(Sm12(0,1), 1.0);
        EXPECT_EQ(Sm12(1,1), 0.0);
    }

    {
        TensorNetworks::OperatorSy Sy12(S);
        EXPECT_EQ(Sy12(0,0), 0.0);
        EXPECT_EQ(Sy12(1,0), dcmplx(0.0,-0.5));
        EXPECT_EQ(Sy12(0,1), dcmplx(0.0, 0.5));
        EXPECT_EQ(Sy12(1,1), 0.0);
    }
    {
        TensorNetworks::OperatorSx Sx12(S);
        EXPECT_EQ(Sx12(0,0), 0.0);
        EXPECT_EQ(Sx12(1,0), 0.5);
        EXPECT_EQ(Sx12(0,1), 0.5);
        EXPECT_EQ(Sx12(1,1), 0.0);
    }
}


