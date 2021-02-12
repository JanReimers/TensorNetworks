#include "Tests.H"
#include "TensorNetworks/Factory.H"
#include "TensorNetworksImp/Hamiltonians/Hamiltonian_1D_NN_Heisenberg.H"
#include "Operators/OperatorElement.H"
#include "Operators/OperatorValuedMatrix.H"


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
using TensorNetworks::MatrixOR;
using TensorNetworks::Lower;
using TensorNetworks::Upper;
using TensorNetworks::DLeft;
using TensorNetworks::DRight;

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
           TensorNetworks::Factory*         itsFactory;
    const  TensorNetworks::OperatorClient1* itsOperatorClient;
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

TEST_F(MPOTests1,OperatorValuedMatrix1)
{
    double S=0.5;
    Setup(S);
    MatrixOR OvM(itsOperatorClient->GetMatrixO(Lower));
    EXPECT_EQ(OvM(0,0),TensorNetworks::OperatorI (S));
    EXPECT_EQ(OvM(1,0),TensorNetworks::OperatorSp(S));
    EXPECT_EQ(OvM(2,0),TensorNetworks::OperatorSm(S));
    EXPECT_EQ(OvM(3,0),TensorNetworks::OperatorSz(S));
    EXPECT_EQ(OvM(0,1),TensorNetworks::OperatorZ (S));
    EXPECT_EQ(OvM(0,2),TensorNetworks::OperatorZ (S));
    EXPECT_EQ(OvM(0,3),TensorNetworks::OperatorZ (S));
    EXPECT_EQ(OvM.GetUpperLower(),Lower);
//    cout << "OvM=" << OvM << endl;
}

TEST_F(MPOTests1,OperatorValuedMatrix2)
{
    double S=1.0;
    Setup(S);
    MatrixOR OvM(itsOperatorClient->GetMatrixO(Lower));
    EXPECT_EQ(OvM(0,0),TensorNetworks::OperatorI (S));
    EXPECT_EQ(OvM(1,0),TensorNetworks::OperatorSp(S));
    EXPECT_EQ(OvM(2,0),TensorNetworks::OperatorSm(S));
    EXPECT_EQ(OvM(3,0),TensorNetworks::OperatorSz(S));
    EXPECT_EQ(OvM(0,1),TensorNetworks::OperatorZ (S));
    EXPECT_EQ(OvM(0,2),TensorNetworks::OperatorZ (S));
    EXPECT_EQ(OvM(0,3),TensorNetworks::OperatorZ (S));
    EXPECT_EQ(OvM.GetUpperLower(),Lower);
}

TEST_F(MPOTests1,OperatorValuedMatrix3)
{
    double S=0.5;
    Setup(S);
    MatrixOR OvM(itsOperatorClient->GetMatrixO(Upper));
    EXPECT_EQ(OvM(0,0),TensorNetworks::OperatorI (S));
    EXPECT_EQ(OvM(0,1),TensorNetworks::OperatorSp(S));
    EXPECT_EQ(OvM(0,2),TensorNetworks::OperatorSm(S));
    EXPECT_EQ(OvM(0,3),TensorNetworks::OperatorSz(S));
    EXPECT_EQ(OvM(1,0),TensorNetworks::OperatorZ (S));
    EXPECT_EQ(OvM(2,0),TensorNetworks::OperatorZ (S));
    EXPECT_EQ(OvM(3,0),TensorNetworks::OperatorZ (S));
    EXPECT_EQ(OvM.GetUpperLower(),Upper);

//    cout << "OvM=" << OvM << endl;
}

TEST_F(MPOTests1,OperatorValuedMatrix4)
{
    double S=1.0;
    Setup(S);
    MatrixOR OvM(itsOperatorClient->GetMatrixO(Upper));
    EXPECT_EQ(OvM(0,0),TensorNetworks::OperatorI (S));
    EXPECT_EQ(OvM(0,1),TensorNetworks::OperatorSp(S));
    EXPECT_EQ(OvM(0,2),TensorNetworks::OperatorSm(S));
    EXPECT_EQ(OvM(0,3),TensorNetworks::OperatorSz(S));
    EXPECT_EQ(OvM(1,0),TensorNetworks::OperatorZ (S));
    EXPECT_EQ(OvM(2,0),TensorNetworks::OperatorZ (S));
    EXPECT_EQ(OvM(3,0),TensorNetworks::OperatorZ (S));
    EXPECT_EQ(OvM.GetUpperLower(),Upper);
}


TEST_F(MPOTests1,OperatorValuedMatrixGetVUpper)
{
    double S=0.5;
    Setup(S);
    MatrixOR OvM(itsOperatorClient->GetMatrixO(Upper));
    MatrixOR Vl=OvM.GetV(DLeft);
    MatrixOR Vr=OvM.GetV(DRight);
    EXPECT_EQ(OvM.GetNumRows(),Vl.GetNumRows()+1);
    EXPECT_EQ(OvM.GetNumCols(),Vl.GetNumCols()+1);
    EXPECT_EQ(OvM.GetNumRows(),Vr.GetNumRows()+1);
    EXPECT_EQ(OvM.GetNumCols(),Vr.GetNumCols()+1);
    for (index_t i:Vl.rows())
    for (index_t j:Vl.cols())
        EXPECT_EQ(OvM(i,j),Vl(i,j));
    for (index_t i:Vr.rows())
    for (index_t j:Vr.cols())
        EXPECT_EQ(OvM(i,j),Vr(i,j));
}

TEST_F(MPOTests1,OperatorValuedMatrixGetVLower)
{
    double S=0.5;
    Setup(S);
    MatrixOR OvM(itsOperatorClient->GetMatrixO(Lower));
    MatrixOR Vl=OvM.GetV(DLeft);
    MatrixOR Vr=OvM.GetV(DRight);
    EXPECT_EQ(OvM.GetNumRows(),Vl.GetNumRows()+1);
    EXPECT_EQ(OvM.GetNumCols(),Vl.GetNumCols()+1);
    EXPECT_EQ(OvM.GetNumRows(),Vr.GetNumRows()+1);
    EXPECT_EQ(OvM.GetNumCols(),Vr.GetNumCols()+1);
    for (index_t i:Vl.rows())
    for (index_t j:Vl.cols())
        EXPECT_EQ(OvM(i,j),Vl(i,j));
    for (index_t i:Vr.rows())
    for (index_t j:Vr.cols())
        EXPECT_EQ(OvM(i,j),Vr(i,j));
}

TEST_F(MPOTests1,OperatorValuedMatrixSetVUpper)
{
    double S=0.5;
    Setup(S);
    MatrixOR OvM(itsOperatorClient->GetMatrixO(Upper));
    MatrixOR Copy(OvM);
    MatrixOR Vl=OvM.GetV(DLeft);
    Copy.SetV(Vl);
    EXPECT_EQ(OvM,Copy);
    MatrixOR Vr=OvM.GetV(DRight);
    Copy.SetV(Vr);
    EXPECT_EQ(OvM,Copy);
}

TEST_F(MPOTests1,OperatorValuedMatrixSetVLower)
{
    double S=0.5;
    Setup(S);
    MatrixOR OvM(itsOperatorClient->GetMatrixO(Lower));
    MatrixOR Copy(OvM);
    MatrixOR Vl=OvM.GetV(DLeft);
    Copy.SetV(Vl);
    EXPECT_EQ(OvM,Copy);
    MatrixOR Vr=OvM.GetV(DRight);
    Copy.SetV(Vr);
    EXPECT_EQ(OvM,Copy);
}
