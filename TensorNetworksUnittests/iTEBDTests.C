#include "Tests.H"
#include "TensorNetworks/Hamiltonian.H"
#include "TensorNetworks/iHamiltonian.H"
#include "TensorNetworks/iTEBDState.H"
#include "TensorNetworks/iMPO.H"
#include "TensorNetworks/Factory.H"
#include "TensorNetworks/SVCompressor.H"
#include "TensorNetworks/IterationSchedule.H"
#include "Operators/MPO_TwoSite.H"
#include "Containers/Matrix4.H"

using std::setw;
using TensorNetworks::TriType;
using TensorNetworks::Random;
using TensorNetworks::Neel;
using TensorNetworks::DLeft;
using TensorNetworks::DRight;
using TensorNetworks::Gates;
using TensorNetworks::FirstOrder;
using TensorNetworks::SecondOrder;
using TensorNetworks::FourthOrder;
using TensorNetworks::TrotterOrder;
using TensorNetworks::Std;
using TensorNetworks::IterationSchedule;
using TensorNetworks::Epsilons;
using TensorNetworks::MPOForm;
using TensorNetworks::RegularLower;
using TensorNetworks::RegularUpper;



class iTEBDTests : public ::testing::Test
{
public:
    typedef TensorNetworks::MatrixRT MatrixRT;
    typedef TensorNetworks::Matrix4RT Matrix4RT;

    iTEBDTests()
    : epsNorm(4e-11)
    , epsOrth(1e-11)
    , itsFactory(TensorNetworks::Factory::GetFactory())
    , itsH(0)
    , itsiH(0)
    , itsState(0)
    , itsCompressor(0)
    {
        StreamableObject::SetToPretty();
    }

    ~iTEBDTests()
    {
        delete itsFactory;
        if (itsH)          delete itsH;
        if (itsiH)         delete itsiH;
        if (itsState)      delete itsState;
        if (itsCompressor) delete itsCompressor;
    }

    void Setup(int L, double S, int D, double epsSVD,TensorNetworks::iTEBDType itype,MPOForm f=RegularLower)
    {
        if (itsH)          delete itsH;
        if (itsiH)         delete itsiH;
        if (itsState)      delete itsState;
        if (itsCompressor) delete itsCompressor;
        itsH=itsFactory->Make1D_NN_HeisenbergHamiltonian(L,S,f,1.0,1.0,0.0);
        itsiH=itsFactory->Make1D_NN_HeisenbergiHamiltonian(1,S,f,1.0,1.0,0.0);
        itsState=itsiH->CreateiTEBDState(L,D,itype,D*D*epsNorm,epsSVD);
        itsCompressor=itsFactory->MakeMPSCompressor(D,epsSVD);
    }

    using iMPO=TensorNetworks::iMPO;
    using MPO=TensorNetworks::MPO;
//
//    double CalculateE(int L, double S)
//    {
//        MPO* SpSmo=new TensorNetworks::MPO_TwoSite(L,S ,1,2, TensorNetworks::Sp,TensorNetworks::Sm);
//        MPO* SmSpo=new TensorNetworks::MPO_TwoSite(L,S ,1,2, TensorNetworks::Sm,TensorNetworks::Sp);
//        MPO* SzSzo=new TensorNetworks::MPO_TwoSite(L,S ,1,2, TensorNetworks::Sz,TensorNetworks::Sz);
//        double E=0.5*(itsState->GetExpectation(SpSmo)+itsState->GetExpectation(SmSpo))+itsState->GetExpectation(SzSzo);
//        delete SpSmo;
//        delete SmSpo;
//        delete SzSzo;
//        return E;
//    }
//    MPO* MakeEnergyMPO(int L, double S)
//    {
//        MPO* SpSmo=new TensorNetworks::MPO_TwoSite(L,S ,1,2, TensorNetworks::Sp,TensorNetworks::Sm);
//        MPO* SmSpo=new TensorNetworks::MPO_TwoSite(L,S ,1,2, TensorNetworks::Sm,TensorNetworks::Sp);
//        MPO* SzSzo=new TensorNetworks::MPO_TwoSite(L,S ,1,2, TensorNetworks::Sz,TensorNetworks::Sz);
//        MPO* SS=itsiH->CreateUnitOperator();
//        SS->Combine(SpSmo,0.5);
//        SS->Combine(SmSpo,0.5);
//        SS->Combine(SzSzo);
//        SS->Report(cout);
//        delete SzSzo;
//        delete SmSpo;
//        delete SpSmo;
//        return SS;
//    }


    double epsNorm,epsOrth;
    TensorNetworks::Factory*       itsFactory=TensorNetworks::Factory::GetFactory();
    TensorNetworks::Hamiltonian*   itsH;
    TensorNetworks::iHamiltonian*  itsiH;
    TensorNetworks::iTEBDState*    itsState;
    TensorNetworks::SVCompressorC* itsCompressor;
};


TEST_F(iTEBDTests,TestLeftNormalize)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0;
    Setup(UnitCell,S,D,epsSVD,Gates);
    itsState->InitializeWith(Random);
    itsState->Canonicalize(DLeft);
    EXPECT_EQ(itsState->GetNormStatus(),"ll");
}

TEST_F(iTEBDTests,TestRecenterLeftNormalize)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0;
    Setup(UnitCell,S,D,epsSVD,Gates);
    itsState->InitializeWith(Random);
    itsState->ReCenter(2);
    itsState->Canonicalize(DLeft);
    EXPECT_EQ(itsState->GetNormStatus(),"ll");
}

TEST_F(iTEBDTests,TestRightNormalize)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0;
    Setup(UnitCell,S,D,epsSVD,Gates);
    itsState->InitializeWith(Random);
    itsState->Canonicalize(DRight);
    EXPECT_EQ(itsState->GetNormStatus(),"rr");
}

TEST_F(iTEBDTests,TestReCenterRightNormalize)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0;
    Setup(UnitCell,S,D,epsSVD,Gates);
    itsState->InitializeWith(Random);
    itsState->ReCenter(2);
    itsState->Canonicalize(DRight);
    EXPECT_EQ(itsState->GetNormStatus(),"rr");
}

TEST_F(iTEBDTests,TestNormalizeRangeSD)
{
    int UnitCell=2,Dmax=32;
#ifdef DEBUG
    Dmax=8;
#endif // DEBUG
    double epsSVD=0.0;
    for (int D=1;D<Dmax;D++)
        for (double S=0.5;S<=2.5;S+=0.5)
        {
            Setup(UnitCell,S,D,epsSVD,Gates);
//            cout << "S,D=" << S << " " << D << endl;
            itsState->InitializeWith(Random);
            itsState->Canonicalize(DLeft);
            EXPECT_EQ(itsState->GetNormStatus(),D==1 ? "II" : "ll");
            itsState->Canonicalize(DRight);
            EXPECT_EQ(itsState->GetNormStatus(),D==1 ? "II" : "rr");
            itsState->ReCenter(2);
            itsState->Canonicalize(DLeft);
            EXPECT_EQ(itsState->GetNormStatus(),D==1 ? "II" : "ll");
            itsState->Canonicalize(DRight);
            EXPECT_EQ(itsState->GetNormStatus(),D==1 ? "II" : "rr");
        }
}

TEST_F(iTEBDTests,TestOrthogonalLeft)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0;
    Setup(UnitCell,S,D,epsSVD,Gates);
    itsState->InitializeWith(Random);
    itsState->Canonicalize(DLeft);
    EXPECT_LT(itsState->Orthogonalize(itsCompressor),epsOrth);
    EXPECT_EQ(itsState->GetNormStatus(),"GG");
}
TEST_F(iTEBDTests,TestOrthogonalRight)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0;
    Setup(UnitCell,S,D,epsSVD,Gates);
    itsState->InitializeWith(Random);
    itsState->Canonicalize(DRight);
    EXPECT_LT(itsState->Orthogonalize(itsCompressor),epsOrth);
    EXPECT_EQ(itsState->GetNormStatus(),"GG");
}

TEST_F(iTEBDTests,TestOrthogonalLeftReCenter1)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0;
    Setup(UnitCell,S,D,epsSVD,Gates);
    itsState->InitializeWith(Random);
    itsState->Canonicalize(DLeft);
    itsState->ReCenter(2);
    EXPECT_LT(itsState->Orthogonalize(itsCompressor),epsOrth);
    EXPECT_EQ(itsState->GetNormStatus(),"GG");
}
TEST_F(iTEBDTests,TestOrthogonalRightReCenter1)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0;
    Setup(UnitCell,S,D,epsSVD,Gates);
    itsState->InitializeWith(Random);
    itsState->Canonicalize(DRight);
    itsState->ReCenter(2);
    EXPECT_LT(itsState->Orthogonalize(itsCompressor),epsOrth);
    EXPECT_EQ(itsState->GetNormStatus(),"GG");
}
TEST_F(iTEBDTests,TestOrthogonalLeftReCenter2)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0;
    Setup(UnitCell,S,D,epsSVD,Gates);
    itsState->InitializeWith(Random);
    itsState->Canonicalize(DLeft);
    EXPECT_LT(itsState->Orthogonalize(itsCompressor),epsOrth);
    itsState->ReCenter(2);
    EXPECT_EQ(itsState->GetNormStatus(),"GG");
}
TEST_F(iTEBDTests,TestOrthogonalRightReCenter2)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0;
    Setup(UnitCell,S,D,epsSVD,Gates);
    itsState->InitializeWith(Random);
    itsState->Canonicalize(DRight);
    EXPECT_LT(itsState->Orthogonalize(itsCompressor),epsOrth);
    itsState->ReCenter(2);
    EXPECT_EQ(itsState->GetNormStatus(),"GG");
}

TEST_F(iTEBDTests,TestOrthogonalRangeSD)
{
    int UnitCell=2,Dmax=16;
#ifdef DEBUG
    Dmax=4;
#endif // DEBUG
    double epsSVD=0.0;
    for (int D=1;D<=Dmax;D*=2)
        for (double S=0.5;S<=2.5;S+=0.5)
        {
            Setup(UnitCell,S,D,epsSVD,Gates);
//            cout << "S,D=" << S << " " << D << endl;
            itsState->InitializeWith(Random);
            itsState->Canonicalize(DLeft);

            EXPECT_LT(itsState->Orthogonalize(itsCompressor),epsOrth);
            EXPECT_EQ(itsState->GetNormStatus(),D==1 ? "II" : "GG");

            itsState->ReCenter(2);

            EXPECT_LT(itsState->Orthogonalize(itsCompressor),epsOrth);
            EXPECT_EQ(itsState->GetNormStatus(),D==1 ? "II" : "GG");

             itsState->ReCenter(1);
            EXPECT_EQ(itsState->GetNormStatus(),D==1 ? "II" : "GG");
        }
}

TEST_F(iTEBDTests,TestExpectationIdentity)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0;
    Setup(UnitCell,S,D,epsSVD,Gates);
    itsState->InitializeWith(Random);
    itsState->Canonicalize(DLeft);
    iMPO* IdentityOp=itsiH->CreateiUnitOperator();
    double expectation=itsState->GetExpectationDw1(IdentityOp);
    EXPECT_NEAR(expectation,1.0,D*1E-13);
    delete IdentityOp;
}

TEST_F(iTEBDTests,TestExpectationIdentityExpH)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0,dt=0.0;
    Setup(UnitCell,S,D,epsSVD,Gates);
    itsState->InitializeWith(Random);
    itsState->Canonicalize(DLeft);

    Matrix4RT IdentityOp=itsiH->GetExponentH(dt); //dt=0 gives unit oprator

    double expectation=itsState->GetExpectationmnmn(IdentityOp);
    EXPECT_NEAR(expectation,1.0,D*1E-13);
}


TEST_F(iTEBDTests,TestReCenterExpectationIdentity1)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0;
    Setup(UnitCell,S,D,epsSVD,Gates);
    itsState->ReCenter(2);
    itsState->InitializeWith(Random);
    itsState->Canonicalize(DLeft);

    iMPO* IdentityOp=itsiH->CreateiUnitOperator();
    double expectation=itsState->GetExpectationDw1(IdentityOp);
    EXPECT_NEAR(expectation,1.0,D*1E-13);
    delete IdentityOp;
}

TEST_F(iTEBDTests,TestReCenterExpectationIdentity2)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0;
    Setup(UnitCell,S,D,epsSVD,Gates);
    itsState->InitializeWith(Random);
    itsState->Canonicalize(DLeft);
    itsState->Orthogonalize(itsCompressor);
    itsState->ReCenter(2);
    iMPO* IdentityOp=itsiH->CreateiUnitOperator();
    double expectation=itsState->GetExpectationDw1(IdentityOp);
    EXPECT_NEAR(expectation,1.0,D*1E-13);
    delete IdentityOp;
}

TEST_F(iTEBDTests,TestExpectationIdentityRangeSD)
{
        int UnitCell=2,Dmax=16;
#ifdef DEBUG
    Dmax=4;
#endif // DEBUG
    double epsSVD=0.0,dt=0.0;
    for (int D=1;D<=Dmax;D*=2)
        for (double S=0.5;S<=2.5;S+=0.5)
        {
            double eps=2e-14;
            Setup(UnitCell,S,D,epsSVD,Gates);
//            cout << "S,D=" << S << " " << D << endl;
            itsState->InitializeWith(Random);
            itsState->Canonicalize(DLeft);
            Matrix4RT IdentityOp=itsiH->GetExponentH(dt); //dt=0 gives unit oprator
            iMPO* IdentityMPO=itsiH->CreateiUnitOperator();
            EXPECT_NEAR(itsState->GetExpectationmnmn(IdentityOp ),1.0,eps);
            EXPECT_NEAR(itsState->GetExpectationDw1 (IdentityMPO),1.0,eps);
            itsState->Orthogonalize(itsCompressor);
            EXPECT_NEAR(itsState->GetExpectationmnmn(IdentityOp ),1.0,eps);
            EXPECT_NEAR(itsState->GetExpectationDw1 (IdentityMPO),1.0,eps);
            itsState->ReCenter(2);
            EXPECT_NEAR(itsState->GetExpectationmnmn(IdentityOp ),1.0,eps);
            EXPECT_NEAR(itsState->GetExpectationDw1 (IdentityMPO),1.0,eps);
        }

}

TEST_F(iTEBDTests,TestApplyIdentity)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0,dt=0.0;
    Setup(UnitCell,S,D,epsSVD,Gates);
    itsState->InitializeWith(Random);
    itsState->Canonicalize(DLeft);
    Matrix4RT IdentityOp=itsiH->GetExponentH(dt); //dt=0 gives unit oprator
    itsState->ApplyOrtho(IdentityOp,itsCompressor);
    EXPECT_EQ(itsState->GetNormStatus(),"GG");
}
TEST_F(iTEBDTests,TestApplyMPOIdentity)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0,epsMPO=1e-13,dt=0.0;
    Setup(UnitCell,S,D,epsSVD,Gates);
    itsState->InitializeWith(Random);
    itsState->Canonicalize(DLeft);
    iMPO* IdentityOp=itsiH->CreateiMPO(dt,FirstOrder,Std,epsMPO);
    itsState->ApplyOrtho(IdentityOp,itsCompressor);
    EXPECT_EQ(itsState->GetNormStatus(),"GG");
    delete IdentityOp;
}


// Always get a singular Vl
/*
TEST_F(iTEBDTests,TestApplyiMPOIdentity)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0,epsMPO=1e-14,dt=0.0;
    Setup(UnitCell,S,D,epsSVD,Gates);
    itsState->InitializeWith(Random);
    itsState->Canonicalize(DLeft);
    MPO* IdentityOp=itsiH->CreateiMPO(dt,FirstOrder,epsMPO);
    IdentityOp->Report(cout);
    itsState->ApplyOrtho(IdentityOp,itsCompressor);
    EXPECT_EQ(itsState->GetNormStatus(),"GG");
    delete IdentityOp;
}
*/

TEST_F(iTEBDTests,TestApplyIdentityRangeSD)
{
   int UnitCell=2,Dmax=4;
   double Smax=2.0;
#ifdef DEBUG
    Dmax=8;
    Smax=1.5;
#endif // DEBUG
    double epsSVD=0.0,epsMPO=1e-13,dt=0.0;
    for (int D=1;D<=Dmax;D*=2)
        for (double S=0.5;S<=Smax;S+=0.5)
        {
            Setup(UnitCell,S,D,epsSVD,Gates);
 //           cout << "S,D=" << S << " " << D << endl;
            itsState->InitializeWith(Random);
            itsState->Canonicalize(DLeft);
            itsState->Orthogonalize(itsCompressor);
            Matrix4RT IdentityOp=itsiH->GetExponentH(dt); //dt=0 gives unit oprator
            itsState->ApplyOrtho(IdentityOp,itsCompressor);
            itsState->ReCenter(2);
            itsState->ApplyOrtho(IdentityOp,itsCompressor);
            EXPECT_EQ(itsState->GetNormStatus(),D==1 ? "II" : "GG");

            itsState->ReCenter(1);
            iMPO* IMPO=itsiH->CreateiMPO(dt,FirstOrder,Std,epsMPO);
            itsState->ApplyOrtho(IMPO,itsCompressor);
            itsState->ReCenter(2);
            itsState->ApplyOrtho(IMPO,itsCompressor);
            EXPECT_EQ(itsState->GetNormStatus(),D==1 ? "II" : "GG");
            delete IMPO;

            IMPO=itsiH->CreateiMPO(dt,SecondOrder,Std,epsMPO);
            itsState->ApplyOrtho(IMPO,itsCompressor);
            itsState->ReCenter(2);
            itsState->ApplyOrtho(IMPO,itsCompressor);
            EXPECT_EQ(itsState->GetNormStatus(),D==1 ? "II" : "GG");
            delete IMPO;

            IMPO=itsiH->CreateiMPO(dt,FourthOrder,Std,epsMPO);
            itsState->ApplyOrtho(IMPO,itsCompressor);
            itsState->ReCenter(2);
            itsState->ApplyOrtho(IMPO,itsCompressor);
            EXPECT_EQ(itsState->GetNormStatus(),D==1 ? "II" : "GG");
            delete IMPO;
        }
}

TEST_F(iTEBDTests,TestApplyExpH)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0,epsMPO=1e-4,dt=0.2;
    Setup(UnitCell,S,D,epsSVD,Gates);
    itsState->InitializeWith(Random);
    itsState->Canonicalize(DLeft);
    itsState->Orthogonalize(itsCompressor);
    Matrix4RT IdentityOp=itsiH->GetExponentH(dt); //dt=0 gives unit oprator

    itsState->ApplyOrtho(IdentityOp,itsCompressor);
    EXPECT_EQ(itsState->GetNormStatus(),"lr");
    itsState->ApplyOrtho(IdentityOp,itsCompressor,1e-13,100);
    EXPECT_EQ(itsState->GetNormStatus(),"lr");

    iMPO* IMPO=itsiH->CreateiMPO(dt,FirstOrder,Std,epsMPO);
    itsState->ApplyOrtho(IMPO,itsCompressor);
    EXPECT_EQ(itsState->GetNormStatus(),"lr");
    delete IMPO;

    IMPO=itsiH->CreateiMPO(dt,SecondOrder,Std,epsMPO);
    itsState->ApplyOrtho(IMPO,itsCompressor);
    EXPECT_EQ(itsState->GetNormStatus(),"lr");
    delete IMPO;

    IMPO=itsiH->CreateiMPO(dt,FourthOrder,Std,epsMPO);
    itsState->ApplyOrtho(IMPO,itsCompressor);
    EXPECT_EQ(itsState->GetNormStatus(),"lr");
    delete IMPO;
}

TEST_F(iTEBDTests,TestApplyOrthoiMPOExpH)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0,epsMPO=1e-4,dt=0.2;
    Setup(UnitCell,S,D,epsSVD,Gates);
    itsState->InitializeWith(Random);
    itsState->Canonicalize(DLeft);
    itsState->Orthogonalize(itsCompressor);

    iMPO* expH=itsiH->CreateiMPO(dt,FirstOrder,Std,epsMPO);
//    expH->Report(cout);
    itsState->ApplyOrtho(expH,itsCompressor);
    EXPECT_EQ(itsState->GetNormStatus(),"lr");
    delete expH;

    expH=itsiH->CreateiMPO(dt,SecondOrder,Std,epsMPO);
    itsState->ApplyOrtho(expH,itsCompressor);
    EXPECT_EQ(itsState->GetNormStatus(),"lr");
    delete expH;

    expH=itsiH->CreateiMPO(dt,FourthOrder,Std,epsMPO);
    itsState->ApplyOrtho(expH,itsCompressor);
    EXPECT_EQ(itsState->GetNormStatus(),"lr");
    delete expH;
}

TEST_F(iTEBDTests,TestApplyiMPOExpH)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0,epsMPO=1e-4,dt=0.2;
    Setup(UnitCell,S,D,epsSVD,Gates);
    itsState->InitializeWith(Random);
    itsState->Canonicalize(DLeft);
    itsState->Orthogonalize(itsCompressor);

    iMPO* expH=itsiH->CreateiMPO(dt,FirstOrder,Std,epsMPO);
    itsState->Apply(expH,itsCompressor);
    EXPECT_EQ(itsState->GetNormStatus(),"lr");
    delete expH;

    expH=itsiH->CreateiMPO(dt,SecondOrder,Std,epsMPO);
    itsState->Apply(expH,itsCompressor);
    EXPECT_EQ(itsState->GetNormStatus(),"lr");
    delete expH;

    expH=itsiH->CreateiMPO(dt,FourthOrder,Std,epsMPO);
    itsState->Apply(expH,itsCompressor);
    EXPECT_EQ(itsState->GetNormStatus(),"lr");
    delete expH;
}

TEST_F(iTEBDTests,TestApplyExpH2)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0,dt=0.2;
    Setup(UnitCell,S,D,epsSVD,Gates);
    itsState->InitializeWith(Random);
    itsState->Canonicalize(DLeft);
    itsState->Orthogonalize(itsCompressor);
    Matrix4RT expH=itsiH->GetExponentH(dt);
    itsState->ApplyOrtho(expH,itsCompressor);
    itsState->ReCenter(2);
    itsState->ApplyOrtho(expH,itsCompressor);
    EXPECT_EQ(itsState->GetNormStatus(),"rl");
}
TEST_F(iTEBDTests,TestApplyExpH3)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0,dt=0.5;
    Setup(UnitCell,S,D,epsSVD,Gates);
    itsState->InitializeWith(Random);
    itsState->Canonicalize(DLeft);
    itsState->Orthogonalize(itsCompressor);
    Matrix4RT expH=itsiH->GetExponentH(dt);
    itsState->ApplyOrtho(expH,itsCompressor);
    itsState->ReCenter(2);
    itsState->ApplyOrtho(expH,itsCompressor);
    itsState->Orthogonalize(itsCompressor);
    EXPECT_EQ(itsState->GetNormStatus(),"GG");
}
TEST_F(iTEBDTests,TestApplyMPOExpH3)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0,epsMPO=1e-2,dt=0.5;
    Setup(UnitCell,S,D,epsSVD,Gates);
    itsState->InitializeWith(Random);
    itsState->Canonicalize(DLeft);
    itsState->Orthogonalize(itsCompressor);

    iMPO* expH=itsiH->CreateiMPO(dt,FirstOrder,Std,epsMPO);
    itsState->ApplyOrtho(expH,itsCompressor);
    itsState->ReCenter(2);
    itsState->ApplyOrtho(expH,itsCompressor);
    itsState->Orthogonalize(itsCompressor);
    EXPECT_EQ(itsState->GetNormStatus(),"GG");
    delete expH;

    expH=itsiH->CreateiMPO(dt,SecondOrder,Std,epsMPO);
    itsState->ApplyOrtho(expH,itsCompressor);
    itsState->ReCenter(2);
    itsState->ApplyOrtho(expH,itsCompressor);
    itsState->Orthogonalize(itsCompressor);
    EXPECT_EQ(itsState->GetNormStatus(),"GG");
    delete expH;

    expH=itsiH->CreateiMPO(dt,FourthOrder,Std,epsMPO);
    itsState->ApplyOrtho(expH,itsCompressor);
    itsState->ReCenter(2);
    itsState->ApplyOrtho(expH,itsCompressor);
    itsState->Orthogonalize(itsCompressor);
    EXPECT_EQ(itsState->GetNormStatus(),"GG");
    delete expH;
}

TEST_F(iTEBDTests,TestApplyExpHRangeSD)
{
   int UnitCell=2,Dmax=16;
   double Smax=1.5;
#ifdef DEBUG
    Dmax=8;
    Smax=1.0;
#endif // DEBUG
    double epsSVD=0.0,epsMPO=1e-4,dt=0.0;
    for (int D=1;D<=Dmax;D*=2)
        for (double S=0.5;S<=Smax;S+=0.5)
        {
            Setup(UnitCell,S,D,epsSVD,Gates);
//            cout << "S,D=" << S << " " << D << endl;
            itsState->InitializeWith(Random);
            itsState->Canonicalize(DLeft);
            itsState->Orthogonalize(itsCompressor);
            {
                Matrix4RT expH=itsiH->GetExponentH(dt);
                itsState->ApplyOrtho(expH,itsCompressor);
                itsState->ReCenter(2);
                itsState->ApplyOrtho(expH,itsCompressor);
                itsState->Orthogonalize(itsCompressor);
                EXPECT_EQ(itsState->GetNormStatus(),D==1 ? "II" : "GG");
            }
            {
                iMPO* expH=itsiH->CreateiMPO(dt,FourthOrder,Std,epsMPO);
                itsState->ApplyOrtho(expH,itsCompressor);
                itsState->ReCenter(2);
                itsState->ApplyOrtho(expH,itsCompressor);
                itsState->Orthogonalize(itsCompressor);
                EXPECT_EQ(itsState->GetNormStatus(),D==1 ? "II" : "GG");
                delete expH;
            }
        }
}
TEST_F(iTEBDTests,TestNeelEnergy_Lower_S12)
{
    int UnitCell=2,D=1;
    double S=0.5,epsSVD=0.0;
    Setup(UnitCell,S,D,epsSVD,Gates,RegularLower);
    itsState->InitializeWith(Neel);
    EXPECT_EQ(itsState->GetNormStatus(),"II");
    Matrix4RT Hlocal=itsiH->GetLocalMatrix();
    EXPECT_NEAR(itsState->GetExpectationmmnn(Hlocal),-0.25,1e-14);
    EXPECT_NEAR(itsState->GetExpectationDw1(itsH  ),-0.25,1e-14);
//    EXPECT_NEAR(CalculateE(UnitCell,S),-0.25,1e-14);
}
TEST_F(iTEBDTests,TestNeelEnergy_Upper_S12)
{
    int UnitCell=2,D=1;
    double S=0.5,epsSVD=0.0;
    Setup(UnitCell,S,D,epsSVD,Gates,RegularUpper);
    itsState->InitializeWith(Neel);
    EXPECT_EQ(itsState->GetNormStatus(),"II");
    Matrix4RT Hlocal=itsiH->GetLocalMatrix();
    EXPECT_NEAR(itsState->GetExpectationmmnn(Hlocal),-0.25,1e-14);
    EXPECT_NEAR(itsState->GetExpectationDw1(itsH  ),-0.25,1e-14);
//    EXPECT_NEAR(CalculateE(UnitCell,S),-0.25,1e-14);
}

TEST_F(iTEBDTests,TestNeelEnergy_Lower_S1)
{
    int UnitCell=2,D=1;
    double S=1.0,epsSVD=0.0;
    Setup(UnitCell,S,D,epsSVD,Gates,RegularLower);
    itsState->InitializeWith(Neel);
    EXPECT_EQ(itsState->GetNormStatus(),"II");
    Matrix4RT Hlocal=itsiH->GetLocalMatrix();
    EXPECT_NEAR(itsState->GetExpectationmmnn(Hlocal),-1.,1e-14);
    EXPECT_NEAR(itsState->GetExpectationDw1(itsH  ),-1.,1e-14);
//    EXPECT_NEAR(CalculateE(UnitCell,S),-1.,1e-14);
}
TEST_F(iTEBDTests,TestNeelEnergy_Upper_S1)
{
    int UnitCell=2,D=1;
    double S=1.0,epsSVD=0.0;
    Setup(UnitCell,S,D,epsSVD,Gates,RegularUpper);
    itsState->InitializeWith(Neel);
    EXPECT_EQ(itsState->GetNormStatus(),"II");
    Matrix4RT Hlocal=itsiH->GetLocalMatrix();
    EXPECT_NEAR(itsState->GetExpectationmmnn(Hlocal),-1.,1e-14);
    EXPECT_NEAR(itsState->GetExpectationDw1(itsH  ),-1.,1e-14);
//    EXPECT_NEAR(CalculateE(UnitCell,S),-1.,1e-14);
}


TEST_F(iTEBDTests,TestRandomEnergyRangeSD_Lower)
{
    int UnitCell=2,Dmax=16;
#ifdef DEBUG
    Dmax=4;
#endif // DEBUG
    double epsSVD=0.0;
    for (int D=1;D<=Dmax;D*=2)
        for (double S=0.5;S<=2.5;S+=0.5)
        {
            Setup(UnitCell,S,D,epsSVD,Gates,RegularLower);
//            cout << "S,D=" << S << " " << D << endl;
            itsState->InitializeWith(Random);
            itsState->Canonicalize(DLeft);
            itsState->Orthogonalize(itsCompressor);
            EXPECT_EQ(itsState->GetNormStatus(),D==1 ? "II" : "GG");
            Matrix4RT Hlocal=itsiH->GetLocalMatrix();
            EXPECT_NEAR(itsState->GetExpectationmmnn(Hlocal),itsState->GetExpectation(itsiH  ),4e-14);
//            EXPECT_NEAR(CalculateE(UnitCell,S),itsState->GetExpectation(itsiH  ),1e-14);
        }
}

TEST_F(iTEBDTests,TestRandomEnergyRangeSD_Upper)
{
    int UnitCell=2,Dmax=16;
#ifdef DEBUG
    Dmax=4;
#endif // DEBUG
    double epsSVD=0.0;
    for (int D=1;D<=Dmax;D*=2)
        for (double S=0.5;S<=2.5;S+=0.5)
        {
            Setup(UnitCell,S,D,epsSVD,Gates,RegularUpper);
//            cout << "S,D=" << S << " " << D << endl;
            itsState->InitializeWith(Random);
            itsState->Canonicalize(DLeft);
            itsState->Orthogonalize(itsCompressor);
            EXPECT_EQ(itsState->GetNormStatus(),D==1 ? "II" : "GG");
            Matrix4RT Hlocal=itsiH->GetLocalMatrix();
            EXPECT_NEAR(itsState->GetExpectationmmnn(Hlocal),itsState->GetExpectation(itsiH  ),3e-14);
//            EXPECT_NEAR(CalculateE(UnitCell,S),itsState->GetExpectation(itsiH  ),1e-14);
        }
}

IterationSchedule MakeSchedule(int maxIter,int D,TrotterOrder to,int deltaD=1)
{
    IterationSchedule is;
    Epsilons eps(1e-12);
    eps.itsMPSCompressEpsilon=0;
    double dts[] = {0.5,0.2,0.1,0.01,0.001,0.0};
    for (double dt:dts)
    {
         eps.itsDeltaLambdaEpsilon=5e-5*dt;
         is.Insert({maxIter,D,deltaD,dt,to,eps});
    }
    return is;
}

TEST_F(iTEBDTests,FindiTimeGS_Lower_D4S12_Gates_FirstOrder)
{
    int UnitCell=2,D=8,maxIter=10000;
    double S=0.5,epsSVD=0.0;
#ifdef DEBUG
    D=4;
    maxIter=1000;
#endif // DEBUG
    Setup(UnitCell,S,2,epsSVD,Gates,RegularLower);
    itsState->InitializeWith(Random);
    IterationSchedule is=MakeSchedule(maxIter,D,FirstOrder);
    itsState->FindiTimeGroundState(itsH,itsiH,is);
//    itsState->Report(cout);
    double E=itsState->GetExpectation(itsiH);
#ifdef DEBUG
    EXPECT_LT(E,-0.44105);   //D=4 we only seem to get this far right now.
//    EXPECT_LT(E,-0.442607); //D=4 From mps-tools.  This looks more like a D=8 result?!?
#else
    EXPECT_LT(E,-0.442846); //D=8 we only seem to get this far right now.
//    EXPECT_LT(E,-0.4430818); //From mps-tools
#endif
    EXPECT_GT(E,-0.4431471805599453094172);
}

TEST_F(iTEBDTests,FindiTimeGSD4S12_Gates_SecondOrder)
{
    int UnitCell=2,D=8,maxIter=1000;
    double S=0.5,epsSVD=0.0;
#ifdef DEBUG
    D=4;
    maxIter=300;
#endif // DEBUG
    Setup(UnitCell,S,2,epsSVD,Gates);
    itsState->InitializeWith(Random);
    IterationSchedule is=MakeSchedule(maxIter,D,SecondOrder);
    itsState->FindiTimeGroundState(itsH,itsiH,is);
//    itsState->Report(cout);
    double E=itsState->GetExpectation(itsiH);
#ifdef DEBUG
    EXPECT_LT(E,-0.44105);   //D=4 we only seem to get this far right now.
//    EXPECT_LT(E,-0.442607); //D=4 From mps-tools.  This looks more like a D=8 result?!?
#else
    EXPECT_LT(E,-0.442846); //D=8 we only seem to get this far right now.
//    EXPECT_LT(E,-0.4430818); //From mps-tools
#endif
    EXPECT_GT(E,-0.4431471805599453094172);
}
TEST_F(iTEBDTests,FindiTimeGSD4S12_Gates_FourthOrder)
{
    int UnitCell=2,D=8,maxIter=400;
    double S=0.5,epsSVD=0.0;
#ifdef DEBUG
    D=2;
    maxIter=200;
#endif // DEBUG
    Setup(UnitCell,S,2,epsSVD,Gates);
    itsState->InitializeWith(Random);
    IterationSchedule is=MakeSchedule(maxIter,D,FourthOrder);
    itsState->FindiTimeGroundState(itsH,itsiH,is);
//    itsState->Report(cout);
    double E=itsState->GetExpectation(itsiH);
#ifdef DEBUG
    EXPECT_LT(E,-0.427905);   //D=2 we only seem to get this far right now.
//    EXPECT_LT(E,-0.44105);   //D=4 we only seem to get this far right now.
//    EXPECT_LT(E,-0.442607); //D=4 From mps-tools.  This looks more like a D=8 result?!?
#else
    EXPECT_LT(E,-0.442845); //Fourth order does not as well in the sixth digit.
//    EXPECT_LT(E,-0.4430818); //From mps-tools
#endif
    EXPECT_GT(E,-0.4431471805599453094172);
}

/*TEST_F(iTEBDTests,FindiTimeGSD4S12_MPOs_FirstOrder)
{
    int UnitCell=2,D=8,maxIter=10000;
    double S=0.5,epsSVD=0.0;
#ifdef DEBUG
    D=4;
    maxIter=1000;
#endif // DEBUG
    Setup(UnitCell,S,2,epsSVD,MPOs);
    itsState->InitializeWith(Random);
    IterationSchedule is=MakeSchedule(maxIter,D,FirstOrder);
    itsState->FindiTimeGroundState(itsiH,is);
    itsState->Report(cout);
    double E=itsState->GetExpectation(itsiH);
#ifdef DEBUG
    EXPECT_LT(E,-0.44105);   //D=4 we only seem to get this far right now.
//    EXPECT_LT(E,-0.442607); //D=4 From mps-tools.  This looks more like a D=8 result?!?
#else
    EXPECT_LT(E,-0.442846); //D=8 we only seem to get this far right now.
//    EXPECT_LT(E,-0.4430818); //From mps-tools
#endif
    EXPECT_GT(E,-0.4431471805599453094172);
}

TEST_F(iTEBDTests,FindiTimeGSD4S12_MPOs_SecondOrder)
{
    int UnitCell=2,D=8,maxIter=10000;
    double S=0.5,epsSVD=0.0;
#ifdef DEBUG
    D=4;
    maxIter=1000;
#endif // DEBUG
    Setup(UnitCell,S,2,epsSVD,MPOs);
    itsState->InitializeWith(Random);
    IterationSchedule is=MakeSchedule(maxIter,D,SecondOrder);
    itsState->FindiTimeGroundState(itsiH,is);
    itsState->Report(cout);
    double E=itsState->GetExpectation(itsiH);
#ifdef DEBUG
    EXPECT_LT(E,-0.44105);   //D=4 we only seem to get this far right now.
//    EXPECT_LT(E,-0.442607); //D=4 From mps-tools.  This looks more like a D=8 result?!?
#else
    EXPECT_LT(E,-0.442846); //D=8 we only seem to get this far right now.
//    EXPECT_LT(E,-0.4430818); //From mps-tools
#endif
    EXPECT_GT(E,-0.4431471805599453094172);
}
TEST_F(iTEBDTests,FindiTimeGSD4S12_MPOs_FourthOrder)
{
    int UnitCell=2,D=8,maxIter=10000;
    double S=0.5,epsSVD=0.0;
#ifdef DEBUG
    D=4;
    maxIter=1000;
#endif // DEBUG
    Setup(UnitCell,S,2,epsSVD,MPOs);
    itsState->InitializeWith(Random);
    IterationSchedule is=MakeSchedule(maxIter,D,FourthOrder);
    itsState->FindiTimeGroundState(itsiH,is);
    itsState->Report(cout);
    double E=itsState->GetExpectation(itsiH);
#ifdef DEBUG
    EXPECT_LT(E,-0.44105);   //D=4 we only seem to get this far right now.
//    EXPECT_LT(E,-0.442607); //D=4 From mps-tools.  This looks more like a D=8 result?!?
#else
    EXPECT_LT(E,-0.442846); //D=8 we only seem to get this far right now.
//    EXPECT_LT(E,-0.4430818); //From mps-tools
#endif
    EXPECT_GT(E,-0.4431471805599453094172);
}

TEST_F(iTEBDTests,FindiTimeGSD4S12_iMPOs_FirstOrder)
{
    int UnitCell=2,D=8,maxIter=200;
    double S=0.5,epsSVD=0.0;
#ifdef DEBUG
    D=4;
    maxIter=100;
#endif // DEBUG
    Setup(UnitCell,S,2,epsSVD,iMPOs);
    itsState->InitializeWith(Random);
    IterationSchedule is=MakeSchedule(maxIter,D,FirstOrder);
    itsState->FindiTimeGroundState(itsiH,is);
    itsState->Report(cout);
    double E=itsState->GetExpectation(itsiH);
#ifdef DEBUG
    EXPECT_LT(E,-0.4268);   //The algo actually goes up in energy for small dt
//    EXPECT_LT(E,-0.442607); //D=4 From mps-tools.  This looks more like a D=8 result?!?
#else
    EXPECT_LT(E,-0.4296); //The algo actually goes up in energy for small dt
//    EXPECT_LT(E,-0.4430818); //From mps-tools
#endif
    EXPECT_GT(E,-0.4431471805599453094172);
}

TEST_F(iTEBDTests,FindiTimeGSD4S12_iMPOs_SecondOrder)
{
    int UnitCell=2,D=8,maxIter=100;
    double S=0.5,epsSVD=0.0;
#ifdef DEBUG
    D=4;
    maxIter=100;
#endif // DEBUG
    Setup(UnitCell,S,2,epsSVD,iMPOs);
    itsState->InitializeWith(Random);
    IterationSchedule is=MakeSchedule(maxIter,D,SecondOrder);
    itsState->FindiTimeGroundState(itsiH,is);
    itsState->Report(cout);
    double E=itsState->GetExpectation(itsiH);
#ifdef DEBUG
    EXPECT_LT(E,-0.4269);   //The algo actually goes up in energy for small dt
//    EXPECT_LT(E,-0.442607); //D=4 From mps-tools.  This looks more like a D=8 result?!?
#else
    EXPECT_LT(E,-0.4296); //The algo actually goes up in energy for small dt
//    EXPECT_LT(E,-0.4430818); //From mps-tools
#endif
    EXPECT_GT(E,-0.4431471805599453094172);
}
*/

#ifndef DEBUG
TEST_F(iTEBDTests,FindiTimeGSD32S12)
{
    int UnitCell=2,Dstart=8,Dmax=32,deltaD=8,maxIter=1000;
    double S=0.5,epsSVD=0.0;
#ifdef DEBUG
    Dmax=8;
    Dstart=2;
    deltaD=2;
    maxIter=1000;
#endif // DEBUG
    Setup(UnitCell,S,Dstart,epsSVD,Gates);
    itsState->InitializeWith(Random);
    IterationSchedule is=MakeSchedule(maxIter,Dmax,SecondOrder,deltaD);
    itsState->FindiTimeGroundState(itsH,itsiH,is);
    itsState->Report(cout);
    double E=itsState->GetExpectation(itsiH);
#ifdef DEBUG
//    EXPECT_LT(E,-0.44308); //From mp-toolkit
    EXPECT_LT(E,-0.442846);
#else
//    EXPECT_LT(E,-0.4431447); //From mp-toolkit
    EXPECT_LT(E,-0.443137); //We only seem to get this far
#endif
    EXPECT_GT(E,-0.4431471805599453094172);
}

#endif

#include "Operators/iMPOImp.H"
TEST_F(iTEBDTests,TestiMPOExpectation)
{
    int UnitCell=2,Dstart=2,D=2,deltaD=1,maxIter=100;
    double S=0.5,epsSVD=0.0;
    Setup(UnitCell,S,Dstart,epsSVD,Gates);


    itsState->InitializeWith(Random);
    iMPO* iH=itsiH;

    IterationSchedule is=MakeSchedule(maxIter,D,SecondOrder,deltaD);
    itsState->FindiTimeGroundState(itsH,itsiH,is);

    double E=itsState->GetExpectation(itsiH);
    double Er=itsState->GetExpectation(iH);
    EXPECT_NEAR(Er,E,1e-10);
//  Recursive contraction of H^2 does not work because we get unsolvable, singular equations
//  when diagonal Wmn(i,i) operators are present.
//    iMPO* iH2=itsiH->CreateiH2Operator();
//    iH2->CanonicalForm();
//    iH2->Report(cout);
//    double Er2=itsState->GetExpectation(iH2); //Fail because shape of W is no longer lower triangular
//    cout << std::setprecision(8) << "E, Er, Er2, <E^2>-<E>^2=" << E << " " << Er << " " << Er2 << " " << Er2-Er*Er <<  endl;
}

