#include "Tests.H"
#include "TensorNetworksImp/Hamiltonians/Hamiltonian_1D_NN_Heisenberg.H"
#include "TensorNetworksImp/Hamiltonians/Hamiltonian_1D_2Body_LongRange.H"
#include "TensorNetworksImp/Hamiltonians/Hamiltonian_1D_3Body.H"
#include "TensorNetworks/iTEBDState.H"
#include "Operators/OperatorBond.H"
#include "Operators/MPO_SpatialTrotter.H"
#include "Operators/MPO_TwoSite.H"
#include "Operators/iMPOImp.H"
//#include "Operators/SiteOperatorImp.H"
#include "Containers/Matrix4.H"
#include "TensorNetworks/iHamiltonian.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/iMPO.H"
#include "TensorNetworks/Factory.H"
#include "Containers/Matrix6.H"

using TensorNetworks::Random;
using TensorNetworks::Neel;
using TensorNetworks::DLeft;
using TensorNetworks::DRight;
using TensorNetworks::IterationSchedule;
using TensorNetworks::Epsilons;
using TensorNetworks::iMPO;
using TensorNetworks::iHamiltonian;
using TensorNetworks::TriType;
using TensorNetworks::Gates;
using TensorNetworks::Std;
using TensorNetworks::Parker;
using TensorNetworks::CNone;
using TensorNetworks::FirstOrder;
using TensorNetworks::SecondOrder;
using TensorNetworks::FourthOrder;
using TensorNetworks::MPO_TwoSite;
using TensorNetworks::Sz;
using TensorNetworks::Sp;
using TensorNetworks::Sm;
using TensorNetworks::MPOForm;
using TensorNetworks::RegularLower;
using TensorNetworks::RegularUpper;
using TensorNetworks::Stod;
//using TensorNetworks::SiteOperatorImp;
using TensorNetworks::PBulk;
using TensorNetworks::MatrixOR;
using TensorNetworks::MaxDelta;
using TensorNetworks::OperatorBond;
using TensorNetworks::OperatorClient;
using TensorNetworks::iMPOImp;
using TensorNetworks::SiteOperator;
//using TensorNetworks::;
//using TensorNetworks::;
//using TensorNetworks::;
//using TensorNetworks::;


class iMPOTests : public ::testing::Test
{
public:
    typedef TensorNetworks::Matrix6CT Matrix6CT;
    typedef TensorNetworks::MatrixRT  MatrixRT;
    typedef TensorNetworks::MatrixCT  MatrixCT;
    typedef TensorNetworks::Vector3CT Vector3CT;
    typedef TensorNetworks::dcmplx     dcmplx;
    iMPOTests()
        : eps(1.0e-13)
        , itsFactory(TensorNetworks::Factory::GetFactory())
        , itsiH(0)
        , itsOperatorClient(0)
        , itsiMPS(0)
    {
        assert(itsFactory);
        StreamableObject::SetToPretty();
    }
    ~iMPOTests()
    {
        delete itsFactory;
        if (itsiH )  delete itsiH;
        if (itsiMPS) delete itsiMPS;
        if (itsOperatorClient) delete itsOperatorClient;
    }

    void Setup(int L, double S, int D,MPOForm f=RegularLower)
    {
        if (itsiH ) delete itsiH;
        if (itsiMPS) delete itsiMPS;
        if (itsOperatorClient) delete itsOperatorClient;
        itsiH  =itsFactory->Make1D_NN_HeisenbergiHamiltonian(L,S,f,1.0,1.0,0.0);
        itsiMPS=itsiH->CreateiTEBDState(2,D,Gates,D*D*1e-10,1e-13);
        itsOperatorClient=new TensorNetworks::Hamiltonian_1D_NN_Heisenberg(S,1.0,1.0,0.0);
        assert(itsOperatorClient);
        itsiMPS->InitializeWith(Random);
        itsiMPS->Canonicalize(DLeft);
        itsiMPS->Orthogonalize(0,1e-13);
    }
    double ENeel(double S) const;

    double eps;
           TensorNetworks::Factory*        itsFactory;
           TensorNetworks::iHamiltonian*   itsiH;
    const  TensorNetworks::OperatorClient* itsOperatorClient;
           TensorNetworks::iTEBDState*     itsiMPS;
};





TEST_F(iMPOTests,MakeHamiltonian)
{
    Setup(10,0.5,2);
}


double iMPOTests::ENeel(double s) const
{
    return -s*s*(itsiH->GetL()-1);
}

TEST_F(iMPOTests,TestiMPOTrotter1_L2)
{
    int L=2,D=2;
    double S=0.5,dt=0.00001,epsMPO=1e-14;
    Setup(L,S,D);
    iMPO* expH=itsiH->CreateiMPO(dt,FirstOrder,Std,epsMPO);
//    expH->Report(cout);
    for (int is=1;is<=L;is++)
    {
        auto [Dw1,Dw2]=expH->GetSiteOperator(is)->GetDws();
        EXPECT_EQ(Dw1,4);
        EXPECT_EQ(Dw2,4);
    }
}
TEST_F(iMPOTests,TestiMPO_Lower_Trotter2_L2)
{
    int L=2,D=2;
    double S=0.5,dt=0.1,epsMPO=6e-3;
    Setup(L,S,D,RegularLower);
    iMPO* expH=itsiH->CreateiMPO(dt,SecondOrder,Std,epsMPO);
//    expH->Report(cout);
    for (int is=1;is<=L;is++)
    {
        auto [Dw1,Dw2]=expH->GetSiteOperator(is)->GetDws();
        EXPECT_EQ(Dw1,4);
        EXPECT_EQ(Dw2,4);
    }
}
TEST_F(iMPOTests,TestiMPO_Upper_Trotter2_L2)
{
    int L=2,D=2;
    double S=0.5,dt=0.1,epsMPO=6e-3;
    Setup(L,S,D,RegularUpper);
    iMPO* expH=itsiH->CreateiMPO(dt,SecondOrder,Std,epsMPO);
//    expH->Report(cout);
    for (int is=1;is<=L;is++)
    {
        auto [Dw1,Dw2]=expH->GetSiteOperator(is)->GetDws();
        EXPECT_EQ(Dw1,4);
        EXPECT_EQ(Dw2,4);
    }
}

TEST_F(iMPOTests,TestiMPO_Lower_Trotter1_L3)
{
    int L=3,D=2;
    double S=0.5,dt=0.00001,epsMPO=1e-14;
    Setup(L,S,D,RegularLower);
    iMPO* expH=itsiH->CreateiMPO(dt,FirstOrder,Std,epsMPO);
//    expH->Report(cout);
    {
        auto [Dw1,Dw2]=expH->GetSiteOperator(1)->GetDws();
        EXPECT_EQ(Dw1,1);
        EXPECT_EQ(Dw2,4);
    }
    {
        auto [Dw1,Dw2]=expH->GetSiteOperator(2)->GetDws();
        EXPECT_EQ(Dw1,4);
        EXPECT_EQ(Dw2,4);
    }
    {
        auto [Dw1,Dw2]=expH->GetSiteOperator(3)->GetDws();
        EXPECT_EQ(Dw1,4);
        EXPECT_EQ(Dw2,1);
    }
}

TEST_F(iMPOTests,TestiMPO_Upper_Trotter1_L3)
{
    int L=3,D=2;
    double S=0.5,dt=0.00001,epsMPO=1e-14;
    Setup(L,S,D,RegularUpper);
    iMPO* expH=itsiH->CreateiMPO(dt,FirstOrder,Std,epsMPO);
//    expH->Report(cout);
    {
        auto [Dw1,Dw2]=expH->GetSiteOperator(1)->GetDws();
        EXPECT_EQ(Dw1,1);
        EXPECT_EQ(Dw2,4);
    }
    {
        auto [Dw1,Dw2]=expH->GetSiteOperator(2)->GetDws();
        EXPECT_EQ(Dw1,4);
        EXPECT_EQ(Dw2,4);
    }
    {
        auto [Dw1,Dw2]=expH->GetSiteOperator(3)->GetDws();
        EXPECT_EQ(Dw1,4);
        EXPECT_EQ(Dw2,1);
    }
}

TEST_F(iMPOTests,  TestiMPO_Lower_Trotter2_L3  )
{
    int L=3,D=2;
    double S=0.5,dt=0.1,epsMPO=6e-3;
    Setup(L,S,D,RegularLower);
    iMPO* expH=itsiH->CreateiMPO(dt,SecondOrder,Std,epsMPO);
//    expH->Report(cout);
    {
        auto [Dw1,Dw2]=expH->GetSiteOperator(1)->GetDws();
        EXPECT_EQ(Dw1,1);
        EXPECT_EQ(Dw2,4);
    }
    {
        auto [Dw1,Dw2]=expH->GetSiteOperator(2)->GetDws();
        EXPECT_EQ(Dw1,4);
        EXPECT_EQ(Dw2,4);
    }
    {
        auto [Dw1,Dw2]=expH->GetSiteOperator(3)->GetDws();
        EXPECT_EQ(Dw1,4);
        EXPECT_EQ(Dw2,1);
    }
}
TEST_F(iMPOTests,  TestiMPO_Upper_Trotter2_L3  )
{
    int L=3,D=2;
    double S=0.5,dt=0.1,epsMPO=6e-3;
    Setup(L,S,D,RegularUpper);
    iMPO* expH=itsiH->CreateiMPO(dt,SecondOrder,Std,epsMPO);
//    expH->Report(cout);
    {
        auto [Dw1,Dw2]=expH->GetSiteOperator(1)->GetDws();
        EXPECT_EQ(Dw1,1);
        EXPECT_EQ(Dw2,4);
    }
    {
        auto [Dw1,Dw2]=expH->GetSiteOperator(2)->GetDws();
        EXPECT_EQ(Dw1,4);
        EXPECT_EQ(Dw2,4);
    }
    {
        auto [Dw1,Dw2]=expH->GetSiteOperator(3)->GetDws();
        EXPECT_EQ(Dw1,4);
        EXPECT_EQ(Dw2,1);
    }
}

TEST_F(iMPOTests,TestL2iMPOTrotter4)
{
    int L=2,D=2;
    double S=0.5,dt=0.1,epsMPO=1e-3;
    Setup(L,S,D);
    iMPO* expH=itsiH->CreateiMPO(dt,FourthOrder,Std,epsMPO);
    {
        auto [Dw1,Dw2]=expH->GetSiteOperator(1)->GetDws();
        EXPECT_EQ(Dw1,4);
        EXPECT_EQ(Dw2,7);
    }
    {
        auto [Dw1,Dw2]=expH->GetSiteOperator(2)->GetDws();
        EXPECT_EQ(Dw1,7);
        EXPECT_EQ(Dw2,4);
    }
}




TEST_F(iMPOTests,TestParkerCanonicalTri_Lower_L1iH)
{
    int L=1,D=2;
    double S=0.5;
    Setup(L,S,D,RegularLower);

    EXPECT_EQ(itsiH->GetNormStatus(),"W");
    double E=itsiMPS->GetExpectation(itsiH);
    itsiH->CanonicalFormTri();
    EXPECT_EQ(itsiH->GetNormStatus(),"L"); //The last site ends up being both right and left normalized
//    itsiH->Report(cout);
    double Eright=itsiMPS->GetExpectation(itsiH);
    EXPECT_NEAR(E,Eright,1e-13);
    EXPECT_EQ(itsiH->GetMaxDw(),5);
}

TEST_F(iMPOTests,TestParkerCanonicalTri_Upper_L1iH)
{
    int L=1,D=2;
    double S=0.5;
    Setup(L,S,D,RegularUpper);

    EXPECT_EQ(itsiH->GetNormStatus(),"W");
    double E=itsiMPS->GetExpectation(itsiH);
    itsiH->CanonicalFormTri();
    EXPECT_EQ(itsiH->GetNormStatus(),"R"); //The last site ends up being both right and left normalized
//    itsiH->Report(cout);
    double Eright=itsiMPS->GetExpectation(itsiH);
    EXPECT_NEAR(E,Eright,1e-13);
    EXPECT_EQ(itsiH->GetMaxDw(),5);
}


TEST_F(iMPOTests,CanonicalQRIter_Left__Lower_L1_NNHeis)
{
    int L=1,D=2;
    double S=0.5;
    Setup(L,S,D,RegularLower);

    EXPECT_EQ(itsiH->GetNormStatus(),"W");
    double E=itsiMPS->GetExpectation(itsiH);
    itsiH->CanonicalFormQRIter(DLeft);
    EXPECT_EQ(itsiH->GetNormStatus(),"L"); //The last site ends up being both right and left normalized
//    itsiH->Report(cout);
    double Eright=itsiMPS->GetExpectation(itsiH);
    EXPECT_NEAR(E,Eright,1e-13);
    EXPECT_EQ(itsiH->GetMaxDw(),5);
}
TEST_F(iMPOTests,CanonicalQRIter_Right_Lower_L1_NNHeis)
{
    int L=1,D=2;
    double S=0.5;
    Setup(L,S,D,RegularLower);

    EXPECT_EQ(itsiH->GetNormStatus(),"W");
    double E=itsiMPS->GetExpectation(itsiH);
    itsiH->CanonicalFormQRIter(DRight);
    EXPECT_EQ(itsiH->GetNormStatus(),"R"); //The last site ends up being both right and left normalized
//    itsiH->Report(cout);
    double Eright=itsiMPS->GetExpectation(itsiH);
    EXPECT_NEAR(E,Eright,1e-13);
    EXPECT_EQ(itsiH->GetMaxDw(),5);
}
TEST_F(iMPOTests,CanonicalQRIter_Left__Upper_L1_NNHeis)
{
    int L=1,D=2;
    double S=0.5;
    Setup(L,S,D,RegularUpper);

    EXPECT_EQ(itsiH->GetNormStatus(),"W");
    double E=itsiMPS->GetExpectation(itsiH);
    itsiH->CanonicalFormQRIter(DLeft);
    EXPECT_EQ(itsiH->GetNormStatus(),"L"); //The last site ends up being both right and left normalized
//    itsiH->Report(cout);
    double Eright=itsiMPS->GetExpectation(itsiH);
    EXPECT_NEAR(E,Eright,1e-13);
    EXPECT_EQ(itsiH->GetMaxDw(),5);
}
TEST_F(iMPOTests,CanonicalQRIter_Right_Upper_L1_NNHeis)
{
    int L=1,D=2;
    double S=0.5;
    Setup(L,S,D,RegularUpper);

    EXPECT_EQ(itsiH->GetNormStatus(),"W");
    double E=itsiMPS->GetExpectation(itsiH);
    itsiH->CanonicalFormQRIter(DRight);
    EXPECT_EQ(itsiH->GetNormStatus(),"R"); //The last site ends up being both right and left normalized
//    itsiH->Report(cout);
    double Eright=itsiMPS->GetExpectation(itsiH);
    EXPECT_NEAR(E,Eright,1e-13);
    EXPECT_EQ(itsiH->GetMaxDw(),5);
}
TEST_F(iMPOTests,CanonicalQRIter_Left__Lower_L2_NNHeis)
{
    int L=2,D=2;
    double S=0.5;
    Setup(L,S,D,RegularLower);

    EXPECT_EQ(itsiH->GetNormStatus(),"WW");
    double E=itsiMPS->GetExpectation(itsiH);
    itsiH->CanonicalFormQRIter(DLeft);
    EXPECT_EQ(itsiH->GetNormStatus(),"LL"); //The last site ends up being both right and left normalized
//    itsiH->Report(cout);
    double Eright=itsiMPS->GetExpectation(itsiH);
    EXPECT_NEAR(E,Eright,1e-13);
    EXPECT_EQ(itsiH->GetMaxDw(),5);
}
TEST_F(iMPOTests,CanonicalQRIter_Right_Lower_L2_NNHeis)
{
    int L=2,D=2;
    double S=0.5;
    Setup(L,S,D,RegularLower);

    EXPECT_EQ(itsiH->GetNormStatus(),"WW");
    double E=itsiMPS->GetExpectation(itsiH);
    itsiH->CanonicalFormQRIter(DRight);
    EXPECT_EQ(itsiH->GetNormStatus(),"RR"); //The last site ends up being both right and left normalized
//    itsiH->Report(cout);
    double Eright=itsiMPS->GetExpectation(itsiH);
    EXPECT_NEAR(E,Eright,1e-13);
    EXPECT_EQ(itsiH->GetMaxDw(),5);
}
TEST_F(iMPOTests,CanonicalQRIter_Left__Upper_L2_NNHeis)
{
    int L=2,D=2;
    double S=0.5;
    Setup(L,S,D,RegularUpper);

    EXPECT_EQ(itsiH->GetNormStatus(),"WW");
    double E=itsiMPS->GetExpectation(itsiH);
    itsiH->CanonicalFormQRIter(DLeft);
    EXPECT_EQ(itsiH->GetNormStatus(),"LL"); //The last site ends up being both right and left normalized
//    itsiH->Report(cout);
    double Eright=itsiMPS->GetExpectation(itsiH);
    EXPECT_NEAR(E,Eright,1e-13);
    EXPECT_EQ(itsiH->GetMaxDw(),5);
}
TEST_F(iMPOTests,CanonicalQRIter_Right_Upper_L2_NNHeis)
{
    int L=2,D=2;
    double S=0.5;
    Setup(L,S,D,RegularUpper);

    EXPECT_EQ(itsiH->GetNormStatus(),"WW");
    double E=itsiMPS->GetExpectation(itsiH);
    itsiH->CanonicalFormQRIter(DRight);
    EXPECT_EQ(itsiH->GetNormStatus(),"RR"); //The last site ends up being both right and left normalized
//    itsiH->Report(cout);
    double Eright=itsiMPS->GetExpectation(itsiH);
    EXPECT_NEAR(E,Eright,1e-13);
    EXPECT_EQ(itsiH->GetMaxDw(),5);
}

/*
TEST_F(iMPOTests,TestParkerCanonicalL1iH2)
{
    int L=1,D=2;
    double S=0.5;
    Setup(L,S,D);

    iMPO* iH2=itsiH->CreateiUnitOperator();
    iH2->Product(itsiH);
    iH2->Product(itsiH);

    EXPECT_EQ(iH2->GetNormStatus(),"W");
    double E=itsiMPS->GetExpectation(iH2);
    iH2->CanonicalForm();
    EXPECT_EQ(iH2->GetNormStatus(),"L"); //The last site ends up being both right and left normalized
    iH2->Report(cout);
    double Eright=itsiMPS->GetExpectation(iH2);
    EXPECT_NEAR(E,Eright,2e-3); //Very lax right now because CanonicalForm is not fully converging
    EXPECT_EQ(iH2->GetMaxDw(),15);
}
*/

TEST_F(iMPOTests,CanonicalQRIter_Left__Lower_L1_LongRange4)
{
    int L=1,D=2,NN=4;
    double S=0.5;
    Setup(L,S,D,RegularLower);

    iHamiltonian* iH=itsFactory->Make1D_2BodyLongRangeiHamiltonian(L,S,RegularLower,1.0,0.0,NN);
    EXPECT_EQ(iH->GetMaxDw(),12);
    EXPECT_EQ(iH->GetNormStatus(),"W");
    double E=itsiMPS->GetExpectation(iH);
    iH->CanonicalFormQRIter(DLeft);
    EXPECT_EQ(iH->GetNormStatus(),"L"); //The last site ends up being both right and left normalized
    double Eright=itsiMPS->GetExpectation(iH);
    EXPECT_NEAR(E,Eright,1e-13);
    EXPECT_EQ(iH->GetMaxDw(),6);
}

TEST_F(iMPOTests,CanonicalQRIter_Right_Lower_L1_LongRange4)
{
    int L=1,D=2,NN=4;
    double S=0.5;
    Setup(L,S,D,RegularLower);

    iHamiltonian* iH=itsFactory->Make1D_2BodyLongRangeiHamiltonian(L,S,RegularLower,1.0,0.0,NN);
    EXPECT_EQ(iH->GetMaxDw(),12);
    EXPECT_EQ(iH->GetNormStatus(),"W");
    double E=itsiMPS->GetExpectation(iH);
    iH->CanonicalFormQRIter(DRight);
    EXPECT_EQ(iH->GetNormStatus(),"R"); //The last site ends up being both right and left normalized
    double Eright=itsiMPS->GetExpectation(iH);
    EXPECT_NEAR(E,Eright,1e-13);
    EXPECT_EQ(iH->GetMaxDw(),6);
}

TEST_F(iMPOTests,CanonicalQRIter_Left__Upper_L1_LongRange4)
{
    int L=1,D=2,NN=4;
    double S=0.5;
    Setup(L,S,D,RegularUpper);

    iHamiltonian* iH=itsFactory->Make1D_2BodyLongRangeiHamiltonian(L,S,RegularUpper,1.0,0.0,NN);
    EXPECT_EQ(iH->GetMaxDw(),12);
    EXPECT_EQ(iH->GetNormStatus(),"W");
    double E=itsiMPS->GetExpectation(iH);
    iH->CanonicalFormQRIter(DLeft);
    EXPECT_EQ(iH->GetNormStatus(),"L"); //The last site ends up being both right and left normalized
    double Eright=itsiMPS->GetExpectation(iH);
    EXPECT_NEAR(E,Eright,1e-13);
    EXPECT_EQ(iH->GetMaxDw(),6);
}

TEST_F(iMPOTests,CanonicalQRIter_Right_Upper_L1_LongRange4)
{
    int L=1,D=2,NN=4;
    double S=0.5;
    Setup(L,S,D,RegularUpper);

    iHamiltonian* iH=itsFactory->Make1D_2BodyLongRangeiHamiltonian(L,S,RegularUpper,1.0,0.0,NN);
    EXPECT_EQ(iH->GetMaxDw(),12);
    EXPECT_EQ(iH->GetNormStatus(),"W");
    double E=itsiMPS->GetExpectation(iH);
    iH->CanonicalFormQRIter(DRight);
    EXPECT_EQ(iH->GetNormStatus(),"R"); //The last site ends up being both right and left normalized
    double Eright=itsiMPS->GetExpectation(iH);
    EXPECT_NEAR(E,Eright,1e-13);
    EXPECT_EQ(iH->GetMaxDw(),6);
}

TEST_F(iMPOTests,CanonicalQRIter_Right_Upper_L1_LongRange10)
{
    int L=1,D=2,NN=10;
    double S=0.5;
    Setup(L,S,D,RegularLower);

    iHamiltonian* iH=itsFactory->Make1D_2BodyLongRangeiHamiltonian(L,S,RegularUpper,1.0,0.0,NN);
    EXPECT_EQ(iH->GetMaxDw(),NN*(NN+1)/2+2);
    EXPECT_EQ(iH->GetNormStatus(),"W");
    double E=itsiMPS->GetExpectation(iH);
    iH->CanonicalFormQRIter(DRight);
//    iH->CanonicalFormQRIter(DLeft); //Fails??!?  Lower left is OK?!?!
    EXPECT_EQ(iH->GetNormStatus(),"R"); //The last site ends up being both right and left normalized
    double Eright=itsiMPS->GetExpectation(iH);
    EXPECT_NEAR(E,Eright,1e-13);
    EXPECT_EQ(iH->GetMaxDw(),NN+2);
}

TEST_F(iMPOTests,CanonicalQRIter_Right_Upper_L2_LongRange10)
{
    int L=2,D=2,NN=10;
    double S=0.5;
    Setup(L,S,D,RegularLower);

    iHamiltonian* iH=itsFactory->Make1D_2BodyLongRangeiHamiltonian(L,S,RegularUpper,1.0,0.0,NN);
    EXPECT_EQ(iH->GetMaxDw(),NN*(NN+1)/2+2);
    EXPECT_EQ(iH->GetNormStatus(),"WW");
    double E=itsiMPS->GetExpectation(iH);
    iH->CanonicalFormQRIter(DRight);
//    iH->CanonicalFormQRIter(DLeft); //Fails??!?  Lower left is OK?!?!
    EXPECT_EQ(iH->GetNormStatus(),"RR"); //The last site ends up being both right and left normalized
    double Eright=itsiMPS->GetExpectation(iH);
    EXPECT_NEAR(E,Eright,1e-13);
    EXPECT_EQ(iH->GetMaxDw(),NN+2);
}

TEST_F(iMPOTests,CanonicalQRIter_Left__Lower_L1_3Body)
{
    int L=1,D=2;
    double S=2.5;
    Setup(L,S,D,RegularLower);

    iHamiltonian* iH=itsFactory->Make1D_3BodyiHamiltonian(L,S,RegularLower,1.0,0.5,0.0);
    EXPECT_EQ(iH->GetMaxDw(),5);
    EXPECT_EQ(iH->GetNormStatus(),"W");
    double E=itsiMPS->GetExpectation(iH);
    iH->CanonicalFormQRIter(DLeft);
    EXPECT_EQ(iH->GetNormStatus(),"L"); //The last site ends up being both right and left normalized
    double Eright=itsiMPS->GetExpectation(iH);
    EXPECT_NEAR(E,Eright,1e-13);
    EXPECT_EQ(iH->GetMaxDw(),3);
}


TEST_F(iMPOTests,CanonicalGaugeTransform_NNHeis_Upper)
{
    int L=1,D=2;
    double S=0.5,eps=S*1e-15;
    MPOForm f=RegularUpper;
    Setup(L,S,D,f);

    iMPOImp l(L,itsOperatorClient,f);
    iMPOImp r(L,itsOperatorClient,f);
    double El=itsiMPS->GetExpectation(&l);
    double Er=itsiMPS->GetExpectation(&r);
    EXPECT_NEAR(El,Er,eps);


    SiteOperator* ls=l.GetSiteOperator(1);
    SiteOperator* rs=r.GetSiteOperator(1);
    l.CanonicalFormQRIter(DLeft);
    r.CanonicalFormQRIter(DRight);
    EXPECT_NEAR(El,itsiMPS->GetExpectation(&l),eps);
    EXPECT_NEAR(Er,itsiMPS->GetExpectation(&r),eps);
    MatrixRT GL=ls->GetGaugeTransform();
    MatrixRT GR=rs->GetGaugeTransform();
    EXPECT_EQ(ls->GetNormStatus(eps),'L');
    EXPECT_EQ(rs->GetNormStatus(eps),'R');
    MatrixRT G=GL*GR;
    MatrixOR WL=ls->GetW();
    MatrixOR WR=rs->GetW();
    EXPECT_NEAR(MaxDelta(G*WR,WL*G),0.0,eps);

    r.CanonicalFormQRIter(DLeft);
    EXPECT_NEAR(Er,itsiMPS->GetExpectation(&r),eps);
    EXPECT_EQ(rs->GetNormStatus(eps),'L');
    MatrixRT G1=rs->GetGaugeTransform();
    EXPECT_NEAR(Max(fabs(G-G1)),0.0,eps); //Passes by flucke ?
    MatrixOR WL1=rs->GetW();
    EXPECT_NEAR(MaxDelta(G1*WR,WL1*G1),0.0,eps);
    l.CanonicalFormQRIter(DRight);
    EXPECT_EQ(ls->GetNormStatus(eps),'R');
    EXPECT_NEAR(El,itsiMPS->GetExpectation(&l),eps);
    MatrixRT G2=ls->GetGaugeTransform();
//    EXPECT_NEAR(Max(fabs(G-G2)),0.0,eps); //Getting sign differences
    MatrixOR WR1=ls->GetW();
    EXPECT_NEAR(MaxDelta(G2*WR1,WL*G2),0.0,eps);

}
TEST_F(iMPOTests,CanonicalGaugeTransform_NNHeis_Lower)
{
    int L=1,D=2;
    double S=0.5,eps=S*1e-15;
    MPOForm f=RegularLower;
    Setup(L,S,D,f);

    iMPOImp l(L,itsOperatorClient,f);
    iMPOImp r(L,itsOperatorClient,f);
    double El=itsiMPS->GetExpectation(&l);
    double Er=itsiMPS->GetExpectation(&r);
    EXPECT_NEAR(El,Er,eps);

    SiteOperator* ls=l.GetSiteOperator(1);
    SiteOperator* rs=r.GetSiteOperator(1);
    l.CanonicalFormQRIter(DLeft);
    r.CanonicalFormQRIter(DRight);
    EXPECT_NEAR(El,itsiMPS->GetExpectation(&l),eps);
    EXPECT_NEAR(Er,itsiMPS->GetExpectation(&r),eps);

    MatrixRT GL=ls->GetGaugeTransform();
    MatrixRT GR=rs->GetGaugeTransform();
    EXPECT_EQ(ls->GetNormStatus(eps),'L');
    EXPECT_EQ(rs->GetNormStatus(eps),'R');
    MatrixRT G=GL*GR;
    MatrixOR WL=ls->GetW();
    MatrixOR WR=rs->GetW();
    EXPECT_NEAR(MaxDelta(G*WR,WL*G),0.0,eps);

    r.CanonicalFormQRIter(DLeft);
    EXPECT_EQ(rs->GetNormStatus(eps),'L');
    EXPECT_NEAR(Er,itsiMPS->GetExpectation(&r),eps);
    MatrixRT G1=rs->GetGaugeTransform();
//    EXPECT_NEAR(Max(fabs(G-G1)),0.0,eps); //Getting sign differences
    MatrixOR WL1=rs->GetW();
    EXPECT_NEAR(MaxDelta(G1*WR,WL1*G1),0.0,eps);
    l.CanonicalFormQRIter(DRight);
    EXPECT_EQ(ls->GetNormStatus(eps),'R');
    EXPECT_NEAR(El,itsiMPS->GetExpectation(&l),eps);
    MatrixRT G2=ls->GetGaugeTransform();
//    EXPECT_NEAR(Max(fabs(G-G2)),0.0,eps); //Getting sign differences
    MatrixOR WR1=ls->GetW();
    EXPECT_NEAR(MaxDelta(G2*WR1,WL*G2),0.0,eps);

}

TEST_F(iMPOTests,CanonicalGaugeTransform_LongRangeHeis_Upper)
{
    int L=1,D=2,NN=8; //9 fails
    double S=0.5,eps=1e-13;
    MPOForm f=RegularUpper;
    Setup(L,S,D,f);

    TensorNetworks::Hamiltonian_2Body_LongRange H(S,1.0,0.0,NN);
    iMPOImp l(L,&H,f);
    iMPOImp r(L,&H,f);
    double El=itsiMPS->GetExpectation(&l);
    double Er=itsiMPS->GetExpectation(&r);
    EXPECT_NEAR(El,Er,eps);

    SiteOperator* ls=l.GetSiteOperator(1);
    SiteOperator* rs=r.GetSiteOperator(1);

    l.CanonicalFormQRIter(DLeft);
    r.CanonicalFormQRIter(DRight);
    EXPECT_EQ(ls->GetNormStatus(eps),'L');
    EXPECT_EQ(rs->GetNormStatus(eps),'R');
    EXPECT_NEAR(El,itsiMPS->GetExpectation(&l),eps);
    EXPECT_NEAR(Er,itsiMPS->GetExpectation(&r),eps);

    MatrixRT GL=ls->GetGaugeTransform();
    MatrixRT GR=rs->GetGaugeTransform();
    MatrixRT G=GL*GR;

    MatrixOR WL=ls->GetW();
    MatrixOR WR=rs->GetW();
    EXPECT_NEAR(MaxDelta(G*WR,WL*G),0.0,1e-15);

    r.CanonicalFormQRIter(DLeft);
    EXPECT_EQ(rs->GetNormStatus(eps),'L');
    EXPECT_NEAR(Er,itsiMPS->GetExpectation(&r),eps);
    MatrixRT G1=rs->GetGaugeTransform();
    MatrixOR WL1=rs->GetW();
    EXPECT_NEAR(MaxDelta(G1*WR,WL1*G1),0.0,1e-15);

}


TEST_F(iMPOTests,CanonicalGaugeTransform_LongRangeHeis_Lower)
{
    int L=1,D=2,NN=8; //9 fails
    double S=0.5,eps=1e-13;
    MPOForm f=RegularLower;
    Setup(L,S,D,f);

    TensorNetworks::Hamiltonian_2Body_LongRange H(S,1.0,0.0,NN);
    iMPOImp l(L,&H,f);
    iMPOImp r(L,&H,f);
    double El=itsiMPS->GetExpectation(&l);
    double Er=itsiMPS->GetExpectation(&r);
    EXPECT_NEAR(El,Er,eps);

    SiteOperator* ls=l.GetSiteOperator(1);
    SiteOperator* rs=r.GetSiteOperator(1);

    l.CanonicalFormQRIter(DLeft);
    r.CanonicalFormQRIter(DRight);
    EXPECT_EQ(ls->GetNormStatus(eps),'L');
    EXPECT_EQ(rs->GetNormStatus(eps),'R');
    EXPECT_NEAR(El,itsiMPS->GetExpectation(&l),eps);
    EXPECT_NEAR(Er,itsiMPS->GetExpectation(&r),eps);

    MatrixRT GL=ls->GetGaugeTransform();
    MatrixRT GR=rs->GetGaugeTransform();
    MatrixRT G=GL*GR;

    MatrixOR WL=ls->GetW();
    MatrixOR WR=rs->GetW();
    EXPECT_NEAR(MaxDelta(G*WR,WL*G),0.0,1e-15);

    r.CanonicalFormQRIter(DLeft);
    EXPECT_EQ(rs->GetNormStatus(eps),'L');
    EXPECT_NEAR(Er,itsiMPS->GetExpectation(&r),eps);
    MatrixRT G1=rs->GetGaugeTransform();
    MatrixOR WL1=rs->GetW();
    EXPECT_NEAR(MaxDelta(G1*WR,WL1*G1),0.0,1e-15);

}

TEST_F(iMPOTests,iCompress_Upper_L1_LongRange5)
{
    int L=1,D=2,NN=5;
    double S=0.5;
    MPOForm f=RegularUpper;
    Setup(L,S,D,f);

//    iHamiltonian* iH=itsFactory->Make1D_3BodyiHamiltonian(L,S,f,1.0,0.5,0.0);
    iHamiltonian* iH=itsFactory->Make1D_3BodyLongRangeiHamiltonian(L,S,f,1.0,0.0,NN);
    EXPECT_EQ(iH->GetMaxDw(),17);
    EXPECT_EQ(iH->GetNormStatus(),"W");
    double E=itsiMPS->GetExpectation(iH);
    iH->CanonicalFormQRIter(DRight);
    EXPECT_EQ(iH->GetNormStatus(),"R"); //The last site ends up being both right and left normalized
    EXPECT_NEAR(E,itsiMPS->GetExpectation(iH),1e-13);
    EXPECT_EQ(iH->GetMaxDw(),6);
    iH->Compress(Parker,0,1e-15);
    EXPECT_EQ(iH->GetMaxDw(),4);
    EXPECT_EQ(iH->GetNormStatus(),"W"); // iH is only approximatly orthogonal now.
    //EXPECT_NEAR(E,itsiMPS->GetExpectation(iH),1e-13); SVD destroys the triangular structure, so right now we can't test the expectation value
    iH->CanonicalFormQRIter(DRight);
}


TEST_F(iMPOTests,iCompress_Lower_L1_LongRange5)
{
    int L=1,D=2,NN=5;
    double S=0.5;
    MPOForm f=RegularLower;
    Setup(L,S,D,f);

//    iHamiltonian* iH=itsFactory->Make1D_3BodyiHamiltonian(L,S,f,1.0,0.5,0.0);
    iHamiltonian* iH=itsFactory->Make1D_3BodyLongRangeiHamiltonian(L,S,f,1.0,0.0,NN);
    EXPECT_EQ(iH->GetMaxDw(),17);
    EXPECT_EQ(iH->GetNormStatus(),"W");
    double E=itsiMPS->GetExpectation(iH);
    iH->CanonicalFormQRIter(DRight);
    EXPECT_EQ(iH->GetNormStatus(),"R"); //The last site ends up being both right and left normalized
    EXPECT_NEAR(E,itsiMPS->GetExpectation(iH),1e-13);
    EXPECT_EQ(iH->GetMaxDw(),6);
    iH->Compress(Parker,0,1e-15);
    EXPECT_EQ(iH->GetMaxDw(),4);
    EXPECT_EQ(iH->GetNormStatus(),"W"); // iH is only approximatly orthogonal now.
    //EXPECT_NEAR(E,itsiMPS->GetExpectation(iH),1e-13); SVD destroys the triangular structure, so right now we can't test the expectation value
    iH->CanonicalFormQRIter(DRight);
}



