#include "Tests.H"
#include "TensorNetworks/Hamiltonian.H"
#include "TensorNetworks/iTEBDState.H"
#include "TensorNetworks/Factory.H"
#include "TensorNetworks/SVCompressor.H"
#include "Operators/MPO_TwoSite.H"

using std::setw;

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
    , itsState(0)
    , itsCompressor(0)
    {
        StreamableObject::SetToPretty();
    }

    ~iTEBDTests()
    {
        delete itsFactory;
        if (itsH) delete itsH;
        if (itsState) delete itsState;
        if (itsCompressor) delete itsCompressor;
    }

    void Setup(int L, double S, int D, double epsSVD)
    {
        if (itsH) delete itsH;
        if (itsState) delete itsState;
        if (itsCompressor) delete itsCompressor;
        itsH=itsFactory->Make1D_NN_HeisenbergHamiltonian(L,S,1.0,1.0,0.0);
        itsState=itsH->CreateiTEBDState(D,D*D*epsNorm,epsSVD);
        itsCompressor=itsFactory->MakeMPSCompressor(D,epsSVD);
    }

    void Check(const TensorNetworks::ONErrors& err) const
    {
//        int D=itsState->GetD();
        EXPECT_LT(err.RightNormError,epsOrth);
        EXPECT_LT(err.RightOrthError,epsOrth);
        EXPECT_LT(err.Left_NormError,epsOrth);
        EXPECT_LT(err.Left_OrthError,epsOrth);
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
    TensorNetworks::SVCompressorC* itsCompressor;
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
            Setup(UnitCell,S,D,epsSVD);
//            cout << "S,D=" << S << " " << D << endl;
            itsState->InitializeWith(TensorNetworks::Random);
            itsState->Canonicalize(TensorNetworks::DLeft);
            EXPECT_EQ(itsState->GetNormStatus(),D==1 ? "II" : "ll");
            itsState->Canonicalize(TensorNetworks::DRight);
            EXPECT_EQ(itsState->GetNormStatus(),D==1 ? "II" : "rr");
            itsState->ReCenter(2);
            itsState->Canonicalize(TensorNetworks::DLeft);
            EXPECT_EQ(itsState->GetNormStatus(),D==1 ? "II" : "ll");
            itsState->Canonicalize(TensorNetworks::DRight);
            EXPECT_EQ(itsState->GetNormStatus(),D==1 ? "II" : "rr");
        }
}

TEST_F(iTEBDTests,TestOrthogonalLeft)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0;
    Setup(UnitCell,S,D,epsSVD);
    itsState->InitializeWith(TensorNetworks::Random);
    itsState->Normalize(TensorNetworks::DLeft);
    Check(itsState->Orthogonalize(itsCompressor));
    EXPECT_EQ(itsState->GetNormStatus(),"GG");
}
TEST_F(iTEBDTests,TestOrthogonalRight)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0;
    Setup(UnitCell,S,D,epsSVD);
    itsState->InitializeWith(TensorNetworks::Random);
    itsState->Normalize(TensorNetworks::DRight);
    Check(itsState->Orthogonalize(itsCompressor));
    EXPECT_EQ(itsState->GetNormStatus(),"GG");
}

TEST_F(iTEBDTests,TestOrthogonalLeftReCenter1)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0;
    Setup(UnitCell,S,D,epsSVD);
    itsState->InitializeWith(TensorNetworks::Random);
    itsState->Normalize(TensorNetworks::DLeft);
    itsState->ReCenter(2);
    Check(itsState->Orthogonalize(itsCompressor));
    EXPECT_EQ(itsState->GetNormStatus(),"GG");
}
TEST_F(iTEBDTests,TestOrthogonalRightReCenter1)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0;
    Setup(UnitCell,S,D,epsSVD);
    itsState->InitializeWith(TensorNetworks::Random);
    itsState->Normalize(TensorNetworks::DRight);
    itsState->ReCenter(2);
    Check(itsState->Orthogonalize(itsCompressor));
    EXPECT_EQ(itsState->GetNormStatus(),"GG");
}
TEST_F(iTEBDTests,TestOrthogonalLeftReCenter2)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0;
    Setup(UnitCell,S,D,epsSVD);
    itsState->InitializeWith(TensorNetworks::Random);
    itsState->Normalize(TensorNetworks::DLeft);
    Check(itsState->Orthogonalize(itsCompressor));
    itsState->ReCenter(2);
    EXPECT_EQ(itsState->GetNormStatus(),"GG");
}
TEST_F(iTEBDTests,TestOrthogonalRightReCenter2)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0;
    Setup(UnitCell,S,D,epsSVD);
    itsState->InitializeWith(TensorNetworks::Random);
    itsState->Normalize(TensorNetworks::DRight);
    Check(itsState->Orthogonalize(itsCompressor));
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
            Setup(UnitCell,S,D,epsSVD);
//            cout << "S,D=" << S << " " << D << endl;
            itsState->InitializeWith(TensorNetworks::Random);
            itsState->Canonicalize(TensorNetworks::DLeft);
            itsState->Normalize(TensorNetworks::DLeft);

            Check(itsState->Orthogonalize(itsCompressor));
            EXPECT_EQ(itsState->GetNormStatus(),D==1 ? "II" : "GG");

            itsState->ReCenter(2);

            Check(itsState->Orthogonalize(itsCompressor));
            EXPECT_EQ(itsState->GetNormStatus(),D==1 ? "II" : "GG");

             itsState->ReCenter(1);
            EXPECT_EQ(itsState->GetNormStatus(),D==1 ? "II" : "GG");
        }
}

TEST_F(iTEBDTests,TestExpectationIdentity)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0;
    Setup(UnitCell,S,D,epsSVD);
    itsState->InitializeWith(TensorNetworks::Random);
    itsState->Canonicalize(TensorNetworks::DLeft);
    itsState->Normalize(TensorNetworks::DLeft);
//    itsState->Orthogonalize();
//    itsState->Normalize(TensorNetworks::DLeft);
//    EXPECT_TRUE(itsState->TestOrthogonal(epsOrth));
//    EXPECT_EQ(itsState->GetNormStatus(),"GG");

    MPO* IdentityOp=itsH->CreateUnitOperator();
    double expectation=itsState->GetExpectation(IdentityOp);
    EXPECT_NEAR(expectation,1.0,D*1E-13);
    delete IdentityOp;
}

TEST_F(iTEBDTests,TestExpectationIdentityExpH)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0,dt=0.0;
    Setup(UnitCell,S,D,epsSVD);
    itsState->InitializeWith(TensorNetworks::Random);
    itsState->Canonicalize(TensorNetworks::DLeft);
    itsState->Normalize(TensorNetworks::DLeft);
//    itsState->Orthogonalize();
//    itsState->Normalize(TensorNetworks::DLeft);
//    EXPECT_TRUE(itsState->TestOrthogonal(epsOrth));
//    EXPECT_EQ(itsState->GetNormStatus(),"GG");
    Matrix4RT Hlocal=itsH->BuildLocalMatrix();
    Matrix4RT IdentityOp=TensorNetworks::Hamiltonian::ExponentH(dt,Hlocal); //dt=0 gives unit oprator

    double expectation=itsState->GetExpectationmnmn(IdentityOp);
    EXPECT_NEAR(expectation,1.0,D*1E-13);
}


TEST_F(iTEBDTests,TestReCenterExpectationIdentity1)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0;
    Setup(UnitCell,S,D,epsSVD);
    itsState->ReCenter(2);
    itsState->InitializeWith(TensorNetworks::Random);
    itsState->Canonicalize(TensorNetworks::DLeft);
    itsState->Normalize(TensorNetworks::DLeft);
//    itsState->Orthogonalize();
//    itsState->ReCenter(1);
//    EXPECT_TRUE(itsState->TestOrthogonal(epsOrth));
//    EXPECT_EQ(itsState->GetNormStatus(),"GG");

    MPO* IdentityOp=itsH->CreateUnitOperator();
    double expectation=itsState->GetExpectation(IdentityOp);
    EXPECT_NEAR(expectation,1.0,D*1E-13);
    delete IdentityOp;
}

TEST_F(iTEBDTests,TestReCenterExpectationIdentity2)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0;
    Setup(UnitCell,S,D,epsSVD);
    itsState->InitializeWith(TensorNetworks::Random);
    itsState->Canonicalize(TensorNetworks::DLeft);
    itsState->Normalize(TensorNetworks::DLeft);
    itsState->Orthogonalize(itsCompressor);
    itsState->ReCenter(2);
    MPO* IdentityOp=itsH->CreateUnitOperator();
    double expectation=itsState->GetExpectation(IdentityOp);
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
            Setup(UnitCell,S,D,epsSVD);
//            cout << "S,D=" << S << " " << D << endl;
            itsState->InitializeWith(TensorNetworks::Random);
            itsState->Canonicalize(TensorNetworks::DLeft);
            itsState->Normalize(TensorNetworks::DLeft);
            Matrix4RT Hlocal=itsH->BuildLocalMatrix();
            Matrix4RT IdentityOp=TensorNetworks::Hamiltonian::ExponentH(dt,Hlocal); //dt=0 gives unit oprator
            MPO* IdentityMPO=itsH->CreateUnitOperator();
            EXPECT_NEAR(itsState->GetExpectationmnmn(IdentityOp ),1.0,eps);
            EXPECT_NEAR(itsState->GetExpectation    (IdentityMPO),1.0,eps);
            itsState->Orthogonalize(itsCompressor);
            EXPECT_NEAR(itsState->GetExpectationmnmn(IdentityOp ),1.0,eps);
            EXPECT_NEAR(itsState->GetExpectation    (IdentityMPO),1.0,eps);
            itsState->ReCenter(2);
            EXPECT_NEAR(itsState->GetExpectationmnmn(IdentityOp ),1.0,eps);
            EXPECT_NEAR(itsState->GetExpectation    (IdentityMPO),1.0,eps);
        }

}

TEST_F(iTEBDTests,TestApplyIdentity)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0,dt=0.0;
    Setup(UnitCell,S,D,epsSVD);
    itsState->InitializeWith(TensorNetworks::Random);
    itsState->Canonicalize(TensorNetworks::DLeft);
    Matrix4RT Hlocal=itsH->BuildLocalMatrix();
    Matrix4RT IdentityOp=TensorNetworks::Hamiltonian::ExponentH(dt,Hlocal); //dt=0 gives unit oprator
    itsState->Apply(IdentityOp,itsCompressor);
    EXPECT_EQ(itsState->GetNormStatus(),"GG");
}


TEST_F(iTEBDTests,TestApplyIdentityRangeSD)
{
   int UnitCell=2,Dmax=16;
#ifdef DEBUG
    Dmax=4;
#endif // DEBUG
    double epsSVD=0.0,dt=0.0;
    for (int D=1;D<=Dmax;D*=2)
        for (double S=0.5;S<=2.5;S+=0.5)
        {
            Setup(UnitCell,S,D,epsSVD);
 //           cout << "S,D=" << S << " " << D << endl;
            itsState->InitializeWith(TensorNetworks::Random);
            itsState->Canonicalize(TensorNetworks::DLeft);
            itsState->Normalize(TensorNetworks::DLeft);
            itsState->Orthogonalize(itsCompressor);
            Matrix4RT Hlocal=itsH->BuildLocalMatrix();
            Matrix4RT IdentityOp=TensorNetworks::Hamiltonian::ExponentH(dt,Hlocal); //dt=0 gives unit oprator
            itsState->Apply(IdentityOp,itsCompressor);
            itsState->ReCenter(2);
            itsState->Apply(IdentityOp,itsCompressor);
            EXPECT_EQ(itsState->GetNormStatus(),D==1 ? "II" : "GG");
        }
}

TEST_F(iTEBDTests,TestApplyExpH)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0,dt=0.2;
    Setup(UnitCell,S,D,epsSVD);
    itsState->InitializeWith(TensorNetworks::Random);
    itsState->Normalize(TensorNetworks::DLeft);
    itsState->Orthogonalize(itsCompressor);
    Matrix4RT Hlocal=itsH->BuildLocalMatrix();
    Matrix4RT IdentityOp=TensorNetworks::Hamiltonian::ExponentH(dt,Hlocal); //dt=0 gives unit oprator
    itsState->Apply(IdentityOp,itsCompressor);
    EXPECT_EQ(itsState->GetNormStatus(),"lr");
}

TEST_F(iTEBDTests,TestApplyExpH2)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0,dt=0.2;
    Setup(UnitCell,S,D,epsSVD);
    itsState->InitializeWith(TensorNetworks::Random);
    itsState->Normalize(TensorNetworks::DLeft);
    itsState->Orthogonalize(itsCompressor);
    Matrix4RT Hlocal=itsH->BuildLocalMatrix();
    Matrix4RT expH=TensorNetworks::Hamiltonian::ExponentH(dt,Hlocal); //dt=0 gives unit oprator
    itsState->Apply(expH,itsCompressor);
    itsState->ReCenter(2);
    itsState->Apply(expH,itsCompressor);
    EXPECT_EQ(itsState->GetNormStatus(),"rl");
}
TEST_F(iTEBDTests,TestApplyExpH3)
{
    int UnitCell=2,D=2;
    double S=0.5,epsSVD=0.0,dt=0.5;
    Setup(UnitCell,S,D,epsSVD);
    itsState->InitializeWith(TensorNetworks::Random);
    itsState->Normalize(TensorNetworks::DLeft);
    itsState->Orthogonalize(itsCompressor);
    Matrix4RT Hlocal=itsH->BuildLocalMatrix();
    Matrix4RT expH=TensorNetworks::Hamiltonian::ExponentH(dt,Hlocal); //dt=0 gives unit oprator
    itsState->Apply(expH,itsCompressor);
    itsState->ReCenter(2);
    itsState->Apply(expH,itsCompressor);
    itsState->Normalize(TensorNetworks::DLeft);
    itsState->Orthogonalize(itsCompressor);
    EXPECT_EQ(itsState->GetNormStatus(),"GG");
}

TEST_F(iTEBDTests,TestApplyExpHRangeSD)
{
       int UnitCell=2,Dmax=16;
#ifdef DEBUG
    Dmax=4;
#endif // DEBUG
    double epsSVD=0.0,dt=0.0;
    for (int D=1;D<=Dmax;D*=2)
        for (double S=0.5;S<=2.5;S+=0.5)
        {
            Setup(UnitCell,S,D,epsSVD);
//            cout << "S,D=" << S << " " << D << endl;
            itsState->InitializeWith(TensorNetworks::Random);
            itsState->Normalize(TensorNetworks::DLeft);
            itsState->Orthogonalize(itsCompressor);
            Matrix4RT Hlocal=itsH->BuildLocalMatrix();
            Matrix4RT expH=TensorNetworks::Hamiltonian::ExponentH(dt,Hlocal); //dt=0 gives unit oprator
            itsState->Apply(expH,itsCompressor);
            itsState->ReCenter(2);
            itsState->Apply(expH,itsCompressor);
            itsState->Normalize(TensorNetworks::DLeft);
            itsState->Orthogonalize(itsCompressor);
            EXPECT_EQ(itsState->GetNormStatus(),D==1 ? "II" : "GG");
        }
}
TEST_F(iTEBDTests,TestNeelEnergyS12)
{
    int UnitCell=2,D=1;
    double S=0.5,epsSVD=0.0;
    Setup(UnitCell,S,D,epsSVD);
    itsState->InitializeWith(TensorNetworks::Neel);
//    itsState->Normalize(TensorNetworks::DLeft);
//    itsState->Orthogonalize();
    EXPECT_EQ(itsState->GetNormStatus(),"II");
    Matrix4RT Hlocal=itsH->BuildLocalMatrix();
    EXPECT_NEAR(itsState->GetExpectationmmnn(Hlocal),-0.25,1e-14);
    EXPECT_NEAR(itsState->GetExpectation(itsH  ),-0.25,1e-14);
}

TEST_F(iTEBDTests,TestNeelEnergyS1)
{
    int UnitCell=2,D=1;
    double S=1.0,epsSVD=0.0;
    Setup(UnitCell,S,D,epsSVD);
    itsState->InitializeWith(TensorNetworks::Neel);
//    itsState->Normalize(TensorNetworks::DLeft);
//    itsState->Orthogonalize();
    EXPECT_EQ(itsState->GetNormStatus(),"II");
    Matrix4RT Hlocal=itsH->BuildLocalMatrix();
    EXPECT_NEAR(itsState->GetExpectationmmnn(Hlocal),-1.,1e-14);
    EXPECT_NEAR(itsState->GetExpectation(itsH  ),-1.,1e-14);
}


TEST_F(iTEBDTests,TestRandomEnergyS12)
{
    int UnitCell=2,D=8;
    double S=0.5,epsSVD=0.0;
    Setup(UnitCell,S,D,epsSVD);
    itsState->InitializeWith(TensorNetworks::Random);
    itsState->Canonicalize(TensorNetworks::DLeft);
    itsState->Normalize(TensorNetworks::DLeft);
    itsState->Orthogonalize(itsCompressor);
    EXPECT_EQ(itsState->GetNormStatus(),"GG");
    Matrix4RT Hlocal=itsH->BuildLocalMatrix();
    EXPECT_NEAR(itsState->GetExpectationmmnn(Hlocal),itsState->GetExpectation(itsH  ),1e-14);
}


TEST_F(iTEBDTests,TestiTimeIterate)
{
    int UnitCell=2,D=8;
    double S=0.5,dt=0.2,epsSVD=0.0;
#ifdef DEBUG
    D=4;
#endif // DEBUG
    Setup(UnitCell,S,D,epsSVD);
    itsState->InitializeWith(TensorNetworks::Random);
    itsState->Canonicalize(TensorNetworks::DLeft);
    itsState->Normalize(TensorNetworks::DLeft);
    itsState->Report(cout);

    Matrix4RT Hlocal=itsH->BuildLocalMatrix();
    for (int it=1;it<=8;it++)
    {
        Matrix4RT expH=TensorNetworks::Hamiltonian::ExponentH(dt,Hlocal);
        //cout << std::setprecision(6) << "expH=" << expH << endl;
        for (int i=1;i<40;i++)
        {
            itsState->ReCenter(1);
            itsState->Apply(expH,itsCompressor);
//            itsState->Normalize(TensorNetworks::DLeft);
//            itsState->Orthogonalize();
            itsState->ReCenter(2);
            itsState->Apply(expH,itsCompressor);
            itsState->ReCenter(1);
        }
        itsState->Normalize(TensorNetworks::DLeft);
        itsState->Orthogonalize(itsCompressor);
//            itsState->Report(cout);
        cout << std::fixed << std::setprecision(5) << "E=" << itsState->GetExpectationmmnn(Hlocal) << " " << itsState->GetExpectation(itsH) << endl;
        dt/=2.0;
    }
    itsState->Report(cout);
    EXPECT_GT(itsState->GetExpectationmmnn(Hlocal),-0.4431471805599453094172);
}

