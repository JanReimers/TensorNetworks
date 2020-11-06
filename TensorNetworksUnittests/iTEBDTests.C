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
    : epsNorm(1e-13)
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


    double epsNorm;
    TensorNetworks::Factory*     itsFactory=TensorNetworks::Factory::GetFactory();
    TensorNetworks::Hamiltonian* itsH;
    TensorNetworks::iTEBDState*  itsState;
};



TEST_F(iTEBDTests,TestApplyIdentity)
{
    int UnitCell=2,D=8;
    double S=0.5,dt=0.5,epsSVD=0.0;
    Setup(UnitCell,S,D,epsSVD);
    itsState->InitializeWith(TensorNetworks::Random);
    itsState->Canonicalize(TensorNetworks::DLeft);
    itsState->Normalize(TensorNetworks::DLeft);
    itsState->Normalize(TensorNetworks::DLeft);
    itsState->Report(cout);
    TensorNetworks::SVCompressorC* mps_compressor =itsFactory->MakeMPSCompressor(D,epsSVD);
//    MPO* SS=MakeEnergyMPO(UnitCell,S);
//    SS->Report(cout);
//    double E=itsState->GetExpectation(SS);
//    cout << "E=" << E << endl;

    Matrix4RT Hlocal=itsH->BuildLocalMatrix();
    double E1=itsState->GetExpectation(1,Hlocal);
    double E2=itsState->GetExpectation(2,Hlocal);
    cout << "E1,E2=" << E1 << " " << E2 << endl;
    for (int it=1;it<=8;it++)
    {
        Matrix4RT expH=TensorNetworks::Hamiltonian::ExponentH(dt,Hlocal);
        //cout << std::setprecision(6) << "expH=" << expH << endl;
        for (int i=1;i<10;i++)
        {
            itsState->Apply(2,expH,mps_compressor);
//            itsState->Normalize(TensorNetworks::DLeft);
            itsState->Apply(1,expH,mps_compressor);
            itsState->Normalize(TensorNetworks::DLeft);
//            itsState->Orthogonalize(1);
//            itsState->Orthogonalize(2);
            itsState->Report(cout);
            cout << std::fixed << std::setprecision(5) << "E1=" << itsState->GetExpectation(1,Hlocal) << endl;
            cout << std::fixed << std::setprecision(5) << "E2=" << itsState->GetExpectation(2,Hlocal) << endl;
//            cout << std::fixed << std::setprecision(5) << "E3=" << itsState->GetExpectation(1,itsH) << endl;
//            cout << std::fixed << std::setprecision(5) << "E4=" << itsState->GetExpectation(2,itsH) << endl;
        }
        dt/=2.0;
    }

}
