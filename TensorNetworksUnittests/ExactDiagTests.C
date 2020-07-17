#include "Tests.H"
#include "TensorNetworks/Hamiltonian.H"
#include "TensorNetworks/Factory.H"
#include "TensorNetworks/LRPSupervisor.H"
#include "TensorNetworks/Epsilons.H"
//#include "TensorNetworksImp/SpinCalculator.H"
//#include "Operators/MPO_OneSite.H"
//#include "Operators/MPO_TwoSite.H"

//#include "oml/stream.h"
//#include "oml/array_io.h"
//#include "oml/smatrix.h"
#include "oml/numeric.h"

//using std::setw;

class ExactDiagTesting : public ::testing::Test
{
public:
    typedef std::complex<double> complx;
    typedef TensorNetworks:: ArrayT  ArrayT;

    ExactDiagTesting()
        : itsFactory(TensorNetworks::Factory::GetFactory())
        , itsH(0)
        , itsLRPSupervisor(new LRPSupervisor())
        , itsEps()
    {
        StreamableObject::SetToPretty();

    }
    ~ExactDiagTesting()
    {
        delete itsFactory;
        delete itsLRPSupervisor;
        delete itsH;
    }

    void Setup(int L, double S)
    {
        delete itsH;
        itsH=itsFactory->Make1D_NN_HeisenbergHamiltonian(L,S,1.0,1.0,0.0);
    }

    TensorNetworks::MatrixT GetH(double S)
    {
        Setup(2,S);
        SparseMatrixCT Hab=itsH->BuildLocalMatrix();
        int N=Hab.GetNumRows();
        TensorNetworks::MatrixT H(N,N);
        for (int i=1;i<=N;i++)
            for (int j=1;j<=N;j++)
                H(i,j)=real(Hab(i,j));
        return H;
    }


    const TensorNetworks::Factory* itsFactory=TensorNetworks::Factory::GetFactory();
    Hamiltonian*         itsH;
    LRPSupervisor*       itsLRPSupervisor;
    Epsilons             itsEps;
};


TEST_F(ExactDiagTesting,TestHabS12)
{
    Setup(10,0.5);
    SparseMatrixCT Hab=itsH->BuildLocalMatrix();
    //cout << "Hab=" << Hab << endl;
    EXPECT_EQ(ToString(Hab),"[1,1]=(0.25,0)\n[2,2]=(-0.25,0)\n[2,3]=(0.5,0)\n[3,2]=(0.5,0)\n[3,3]=(-0.25,0)\n[4,4]=(0.25,0)\n");
}

TEST_F(ExactDiagTesting,TestHabS1)
{
    Setup(10,1.0);
    SparseMatrixCT Hab=itsH->BuildLocalMatrix();
    EXPECT_EQ(ToString(Hab),"[1,1]=(1,0)\n[2,4]=(1,0)\n[3,3]=(-1,0)\n[3,5]=(1,0)\n[4,2]=(1,0)\n[5,3]=(1,0)\n[5,7]=(1,0)\n[6,8]=(1,0)\n[7,5]=(1,0)\n[7,7]=(-1,0)\n[8,6]=(1,0)\n[9,9]=(1,0)\n");
}

TEST_F(ExactDiagTesting,TestHabS32)
{
    Setup(10,1.5);
    SparseMatrixCT Hab=itsH->BuildLocalMatrix();
    EXPECT_EQ(ToString(Hab),"[1,1]=(2.25,0)\n[2,2]=(0.75,0)\n[2,5]=(1.5,0)\n[3,3]=(-0.75,0)\n[3,6]=(1.73205,0)\n[4,4]=(-2.25,0)\n[4,7]=(1.5,0)\n[5,2]=(1.5,0)\n[5,5]=(0.75,0)\n[6,3]=(1.73205,0)\n[6,6]=(0.25,0)\n[6,9]=(1.73205,0)\n[7,4]=(1.5,0)\n[7,7]=(-0.25,0)\n[7,10]=(2,0)\n[8,8]=(-0.75,0)\n[8,11]=(1.73205,0)\n[9,6]=(1.73205,0)\n[9,9]=(-0.75,0)\n[10,7]=(2,0)\n[10,10]=(-0.25,0)\n[10,13]=(1.5,0)\n[11,8]=(1.73205,0)\n[11,11]=(0.25,0)\n[11,14]=(1.73205,0)\n[12,12]=(0.75,0)\n[12,15]=(1.5,0)\n[13,10]=(1.5,0)\n[13,13]=(-2.25,0)\n[14,11]=(1.73205,0)\n[14,14]=(-0.75,0)\n[15,12]=(1.5,0)\n[15,15]=(0.75,0)\n[16,16]=(2.25,0)\n");
}

TEST_F(ExactDiagTesting,TestEvsS12)
{
    TensorNetworks::MatrixT H=GetH(1.0/2.0);
//    cout << "Hab=" << H << endl;
    Vector<double> evs=Diagonalize(H);
    EXPECT_EQ(ToString(evs),"(1:4){ -0.75 0.25 0.25 0.25 }");
}

TEST_F(ExactDiagTesting,TestEvsS22)
{
    TensorNetworks::MatrixT H=GetH(2.0/2.0);
    Vector<double> evs=Diagonalize(H);
    EXPECT_EQ(ToString(evs),"(1:9){ -2 -1 -1 -1 1 1 1 1 1 }");
}

TEST_F(ExactDiagTesting,TestEvsS32)
{
    TensorNetworks::MatrixT H=GetH(3.0/2.0);
    Vector<double> evs=Diagonalize(H);
    EXPECT_EQ(ToString(evs),"(1:16){ -3.75 -2.75 -2.75 -2.75 -0.75 -0.75 -0.75 -0.75 -0.75 2.25 2.25 2.25 2.25 2.25 2.25 2.25 }");
}

TEST_F(ExactDiagTesting,TestEvsS42)
{
    TensorNetworks::MatrixT H=GetH(4.0/2.0);
    Vector<double> evs=Diagonalize(H);
    int N=evs.size();
    for (int i=1;i<=N;i++)
        if (fabs(evs(i))<1e-15) evs(i)=0;
    EXPECT_EQ(ToString(evs),"(1:25){ -6 -5 -5 -5 -3 -3 -3 -3 -3 0 0 0 0 0 0 0 4 4 4 4 4 4 4 4 4 }");
}

TEST_F(ExactDiagTesting,TestEvsS52)
{
    TensorNetworks::MatrixT H=GetH(5.0/2.0);
    Vector<double> evs=Diagonalize(H);
    EXPECT_EQ(ToString(evs),"(1:36){ -8.75 -7.75 -7.75 -7.75 -5.75 -5.75 -5.75 -5.75 -5.75 -2.75 -2.75 -2.75 -2.75 -2.75 -2.75 -2.75 1.25 1.25 1.25 1.25 1.25 1.25 1.25 1.25 1.25 6.25 6.25 6.25 6.25 6.25 6.25 6.25 6.25 6.25 6.25 6.25 }");
}
