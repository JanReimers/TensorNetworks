#include "Tests.H"
#include "TensorNetworks/Hamiltonian.H"
#include "TensorNetworks/iTEBDState.H"
#include "TensorNetworks/Factory.H"
#include "Operators/MPO_TwoSite.H"

using std::setw;

class iTEBDTests : public ::testing::Test
{
public:
    typedef TensorNetworks::MatrixRT MatrixRT;
    typedef TensorNetworks::Matrix4RT Matrix4RT;

    iTEBDTests()
    : epsNorm(1e-11)
    , epsOrth(1e-11)
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

    void Setup(int L, double S, int D, double epsSVD)
    {
        if (itsH) delete itsH;
        if (itsState) delete itsState;
        itsH=itsFactory->Make1D_NN_HeisenbergHamiltonian(L,S,1.0,1.0,0.0);
        itsState=itsH->CreateiTEBDState(D,epsNorm,epsSVD);
    }

    using MPO=TensorNetworks::MPO;
    MPO* MakeEnergyMPO(int L, double S)
    {
        MPO* SpSmo=new TensorNetworks::MPO_TwoSite(L,S ,1,2, TensorNetworks::Sp,TensorNetworks::Sm);
        MPO* SmSpo=new TensorNetworks::MPO_TwoSite(L,S ,1,2, TensorNetworks::Sm,TensorNetworks::Sp);
        MPO* SzSzo=new TensorNetworks::MPO_TwoSite(L,S ,1,2, TensorNetworks::Sz,TensorNetworks::Sz);
        MPO* SS=itsH->CreateUnitOperator();
        SS->Combine(SpSmo,0.5);
        SS->Combine(SmSpo,0.5);
        SS->Combine(SzSzo);
        delete SzSzo;
        delete SmSpo;
        delete SpSmo;
        return SS;
    }


    double epsNorm,epsOrth;
    TensorNetworks::Factory*     itsFactory=TensorNetworks::Factory::GetFactory();
    TensorNetworks::Hamiltonian* itsH;
    TensorNetworks::iTEBDState*  itsState;
};

TEST_F(iTEBDTests,TestLeftNormalize)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0;
    Setup(UnitCell,S,D,epsSVD);
    itsState->InitializeWith(TensorNetworks::Random);
    itsState->Canonicalize(TensorNetworks::DLeft);
    EXPECT_EQ(itsState->GetNormStatus(),"ll");
}

TEST_F(iTEBDTests,TestRecenterLeftNormalize)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0;
    Setup(UnitCell,S,D,epsSVD);
    itsState->InitializeWith(TensorNetworks::Random);
    itsState->ReCenter(2);
    itsState->Canonicalize(TensorNetworks::DLeft);
    EXPECT_EQ(itsState->GetNormStatus(),"ll");
}

TEST_F(iTEBDTests,TestRightNormalize)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0;
    Setup(UnitCell,S,D,epsSVD);
    itsState->InitializeWith(TensorNetworks::Random);
    itsState->Canonicalize(TensorNetworks::DRight);
    EXPECT_EQ(itsState->GetNormStatus(),"rr");
}

TEST_F(iTEBDTests,TestReCenterRightNormalize)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0;
    Setup(UnitCell,S,D,epsSVD);
    itsState->InitializeWith(TensorNetworks::Random);
    itsState->ReCenter(2);
    itsState->Canonicalize(TensorNetworks::DRight);
    EXPECT_EQ(itsState->GetNormStatus(),"rr");
}

TEST_F(iTEBDTests,TestOrthogonalLeft)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0;
    Setup(UnitCell,S,D,epsSVD);
    itsState->InitializeWith(TensorNetworks::Random);
    itsState->Normalize(TensorNetworks::DLeft);
    itsState->Orthogonalize();
    EXPECT_TRUE(itsState->TestOrthogonal(epsOrth));
    EXPECT_EQ(itsState->GetNormStatus(),"GG");
}
TEST_F(iTEBDTests,TestOrthogoanlRight)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0;
    Setup(UnitCell,S,D,epsSVD);
    itsState->InitializeWith(TensorNetworks::Random);
    itsState->Normalize(TensorNetworks::DRight);
    itsState->Orthogonalize();
    EXPECT_TRUE(itsState->TestOrthogonal(epsOrth));
    EXPECT_EQ(itsState->GetNormStatus(),"GG");
}

TEST_F(iTEBDTests,TestOrthogoanlLeftReCenter1)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0;
    Setup(UnitCell,S,D,epsSVD);
    itsState->InitializeWith(TensorNetworks::Random);
    itsState->Normalize(TensorNetworks::DLeft);
    itsState->ReCenter(2);
    itsState->Orthogonalize();
    EXPECT_TRUE(itsState->TestOrthogonal(epsOrth));
    EXPECT_EQ(itsState->GetNormStatus(),"GG");
}
TEST_F(iTEBDTests,TestOrthogoanlRightReCenter1)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0;
    Setup(UnitCell,S,D,epsSVD);
    itsState->InitializeWith(TensorNetworks::Random);
    itsState->Normalize(TensorNetworks::DLeft);
    itsState->ReCenter(2);
    itsState->Orthogonalize();
    EXPECT_TRUE(itsState->TestOrthogonal(epsOrth));
    EXPECT_EQ(itsState->GetNormStatus(),"GG");
}
TEST_F(iTEBDTests,TestOrthogoanlLeftReCenter2)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0;
    Setup(UnitCell,S,D,epsSVD);
    itsState->InitializeWith(TensorNetworks::Random);
    itsState->Normalize(TensorNetworks::DLeft);
    itsState->Orthogonalize();
    itsState->ReCenter(2);
    EXPECT_TRUE(itsState->TestOrthogonal(epsOrth));
    EXPECT_EQ(itsState->GetNormStatus(),"GG");
}
TEST_F(iTEBDTests,TestOrthogoanlRightReCenter2)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0;
    Setup(UnitCell,S,D,epsSVD);
    itsState->InitializeWith(TensorNetworks::Random);
    itsState->Normalize(TensorNetworks::DLeft);
    itsState->Orthogonalize();
    itsState->ReCenter(2);
    EXPECT_TRUE(itsState->TestOrthogonal(epsOrth));
    EXPECT_EQ(itsState->GetNormStatus(),"GG");
}


TEST_F(iTEBDTests,TestExpectationIdentity)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0;
    Setup(UnitCell,S,D,epsSVD);
    itsState->InitializeWith(TensorNetworks::Random);
    itsState->Normalize(TensorNetworks::DLeft);
    itsState->Orthogonalize();

    MPO* IdentityOp=itsH->CreateUnitOperator();
    double expectation=itsState->GetExpectation(IdentityOp);
    EXPECT_NEAR(expectation,1.0,1E-14);
    delete IdentityOp;
}


TEST_F(iTEBDTests,TestReCenterExpectationIdentity1)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0;
    Setup(UnitCell,S,D,epsSVD);
    itsState->ReCenter(2);
    itsState->InitializeWith(TensorNetworks::Random);
    itsState->Normalize(TensorNetworks::DLeft);
    itsState->Orthogonalize();
//    itsState->ReCenter(1);
    EXPECT_TRUE(itsState->TestOrthogonal(epsOrth));
    EXPECT_EQ(itsState->GetNormStatus(),"GG");
    MPO* IdentityOp=itsH->CreateUnitOperator();
    double expectation=itsState->GetExpectation(IdentityOp);
    EXPECT_NEAR(expectation,1.0,1E-14);
    delete IdentityOp;
}

TEST_F(iTEBDTests,TestReCenterExpectationIdentity2)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0;
    Setup(UnitCell,S,D,epsSVD);
    itsState->InitializeWith(TensorNetworks::Random);
    itsState->Normalize(TensorNetworks::DLeft);
    itsState->Orthogonalize();
    itsState->ReCenter(2);
    MPO* IdentityOp=itsH->CreateUnitOperator();
    double expectation=itsState->GetExpectation(IdentityOp);
    EXPECT_NEAR(expectation,1.0,1E-14);
    delete IdentityOp;
}

TEST_F(iTEBDTests,TestApplyIdentity)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0,dt=0.0;
    Setup(UnitCell,S,D,epsSVD);
    itsState->InitializeWith(TensorNetworks::Random);
    itsState->Normalize(TensorNetworks::DLeft);
    itsState->Orthogonalize();
    TensorNetworks::SVCompressorC* mps_compressor =itsFactory->MakeMPSCompressor(D,epsSVD);
    Matrix4RT Hlocal=itsH->BuildLocalMatrix();
    Matrix4RT IdentityOp=TensorNetworks::Hamiltonian::ExponentH(dt,Hlocal); //dt=0 gives unit oprator
    itsState->Apply(IdentityOp,mps_compressor);
    EXPECT_EQ(itsState->GetNormStatus(),"lr");
}
TEST_F(iTEBDTests,TestApplyIdentity2)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0,dt=0.0;
    Setup(UnitCell,S,D,epsSVD);
    itsState->InitializeWith(TensorNetworks::Random);
    itsState->Normalize(TensorNetworks::DLeft);
    itsState->Orthogonalize();
    TensorNetworks::SVCompressorC* mps_compressor =itsFactory->MakeMPSCompressor(D,epsSVD);
    Matrix4RT Hlocal=itsH->BuildLocalMatrix();
    Matrix4RT IdentityOp=TensorNetworks::Hamiltonian::ExponentH(dt,Hlocal); //dt=0 gives unit oprator
    itsState->Apply(IdentityOp,mps_compressor);
    itsState->ReCenter(2);
    itsState->Apply(IdentityOp,mps_compressor);
    EXPECT_EQ(itsState->GetNormStatus(),"GG");
}

TEST_F(iTEBDTests,TestApplyExpH)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0,dt=0.5;
    Setup(UnitCell,S,D,epsSVD);
    itsState->InitializeWith(TensorNetworks::Random);
    itsState->Normalize(TensorNetworks::DLeft);
    itsState->Orthogonalize();
    TensorNetworks::SVCompressorC* mps_compressor =itsFactory->MakeMPSCompressor(D,epsSVD);
    Matrix4RT Hlocal=itsH->BuildLocalMatrix();
    Matrix4RT IdentityOp=TensorNetworks::Hamiltonian::ExponentH(dt,Hlocal); //dt=0 gives unit oprator
    itsState->Apply(IdentityOp,mps_compressor);
    EXPECT_EQ(itsState->GetNormStatus(),"lr");
}

TEST_F(iTEBDTests,TestApplyExpH2)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0,dt=0.5;
    Setup(UnitCell,S,D,epsSVD);
    itsState->InitializeWith(TensorNetworks::Random);
    itsState->Normalize(TensorNetworks::DLeft);
    itsState->Orthogonalize();
    TensorNetworks::SVCompressorC* mps_compressor =itsFactory->MakeMPSCompressor(D,epsSVD);
    Matrix4RT Hlocal=itsH->BuildLocalMatrix();
    Matrix4RT IdentityOp=TensorNetworks::Hamiltonian::ExponentH(dt,Hlocal); //dt=0 gives unit oprator
    itsState->Apply(IdentityOp,mps_compressor);
    itsState->ReCenter(2);
    itsState->Apply(IdentityOp,mps_compressor);
    EXPECT_EQ(itsState->GetNormStatus(),"rl");
}
TEST_F(iTEBDTests,TestApplyExpH3)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0,dt=0.5;
    Setup(UnitCell,S,D,epsSVD);
    itsState->InitializeWith(TensorNetworks::Random);
    itsState->Normalize(TensorNetworks::DLeft);
    itsState->Orthogonalize();
    TensorNetworks::SVCompressorC* mps_compressor =itsFactory->MakeMPSCompressor(D,epsSVD);
    Matrix4RT Hlocal=itsH->BuildLocalMatrix();
    Matrix4RT IdentityOp=TensorNetworks::Hamiltonian::ExponentH(dt,Hlocal); //dt=0 gives unit oprator
    itsState->Apply(IdentityOp,mps_compressor);
    itsState->ReCenter(2);
    itsState->Apply(IdentityOp,mps_compressor);
    itsState->Normalize(TensorNetworks::DLeft);
    itsState->Orthogonalize();
    EXPECT_EQ(itsState->GetNormStatus(),"GG");
}


TEST_F(iTEBDTests,TestiTimeIterate)
{
    int UnitCell=2,D=2;
    double S=0.5,dt=0.05,epsSVD=0.0;
    Setup(UnitCell,S,D,epsSVD);
    itsState->InitializeWith(TensorNetworks::Random);
    itsState->Normalize(TensorNetworks::DLeft);
    itsState->Orthogonalize();
    itsState->Report(cout);
    TensorNetworks::SVCompressorC* mps_compressor =itsFactory->MakeMPSCompressor(D,epsSVD);
//    MPO* SS=MakeEnergyMPO(UnitCell,S);
//    SS->Report(cout);
//    double E=itsState->GetExpectation(SS);
//    cout << "E=" << E << endl;

    Matrix4RT Hlocal=itsH->BuildLocalMatrix();
    cout << std::fixed << std::setprecision(5) << "E=" << itsState->GetExpectation(Hlocal) << " " << itsState->GetExpectation(itsH) << endl;
    for (int it=1;it<=8;it++)
    {
        Matrix4RT expH=TensorNetworks::Hamiltonian::ExponentH(dt,Hlocal);
        //cout << std::setprecision(6) << "expH=" << expH << endl;
        for (int i=1;i<10;i++)
        {
            itsState->ReCenter(1);
            itsState->Apply(expH,mps_compressor);
            itsState->Normalize(TensorNetworks::DLeft);
//            itsState->Orthogonalize();
            itsState->ReCenter(2);
            itsState->Apply(expH,mps_compressor);
            itsState->Normalize(TensorNetworks::DLeft);
            itsState->Orthogonalize();
            itsState->Report(cout);
            cout << std::fixed << std::setprecision(5) << "E=" << itsState->GetExpectation(Hlocal) << " " << itsState->GetExpectation(itsH) << endl;
        }
        dt/=2.0;
    }

}
