#include "Tests.H"
#include "TensorNetworks/Hamiltonian.H"
#include "TensorNetworks/Factory.H"
#include "TensorNetworks/IterationSchedule.H"
#include "TensorNetworksImp/SpinCalculator.H"
#include "Operators/MPO_OneSite.H"
#include "Operators/MPO_TwoSite.H"

#include "oml/stream.h"
#include "oml/array_io.h"
#include "oml/smatrix.h"
#include "oml/numeric.h"

using std::setw;

class ExpectationsTests : public ::testing::Test
{
public:
    typedef std::complex<double> complx;

    ExpectationsTests()
        : itsFactory(TensorNetworks::Factory::GetFactory())
        , itsH(0)
        , itsMPS(0)
        , itsEps()
    {
        StreamableObject::SetToPretty();

    }
    ~ExpectationsTests()
    {
        delete itsFactory;
        delete itsH;
        delete itsMPS;
    }

    void Setup(int L, double S, int D)
    {
        delete itsH;
        delete itsMPS;
        itsH=itsFactory->Make1D_NN_HeisenbergHamiltonian(L,S,1.0,1.0,0.0);
        itsMPS=itsH->CreateMPS(D);
        itsMPS->InitializeWith(TensorNetworks::Random);

        TensorNetworks::IterationSchedule is;
        is.Insert({20,D,itsEps});
        itsMPS->FindVariationalGroundState(itsH,is);
    }


    TensorNetworks::Factory*     itsFactory=TensorNetworks::Factory::GetFactory();
    TensorNetworks::Hamiltonian* itsH;
    TensorNetworks::MPS*         itsMPS;
    TensorNetworks::Epsilons     itsEps;
};


TEST_F(ExpectationsTests,TestSpinCalculatorS12)
{
    TensorNetworks::SpinCalculator sc(0.5);

    EXPECT_EQ(ToString(sc.GetSx()),"(1:2),(1:2) \n[ 0 0.5 ]\n[ 0.5 0 ]\n");
    EXPECT_EQ(ToString(sc.GetSy()),"(1:2),(1:2) \n[ (0,0) (0,0.5) ]\n[ (0,-0.5) (0,0) ]\n");
    EXPECT_EQ(ToString(sc.GetSz()),"(1:2),(1:2) \n[ -0.5 0 ]\n[ 0 0.5 ]\n");
    EXPECT_EQ(ToString(sc.GetSm()),"(1:2),(1:2) \n[ 0 1 ]\n[ 0 0 ]\n");
    EXPECT_EQ(ToString(sc.GetSp()),"(1:2),(1:2) \n[ 0 0 ]\n[ 1 0 ]\n");

}



TEST_F(ExpectationsTests,TestSweepL9S1D2)
{
    int L=9,D=2;
    double S=0.5;
    Setup(L,S,D);

    double E=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(E/(L-1),-0.45317425,1e-7);
}


TEST_F(ExpectationsTests,TestOneSiteDMs)
{
    int L=9,D=4;
    double S=0.5;
    Setup(L,S,D);
    OneSiteDMs ro1=itsMPS->CalculateOneSiteDMs();
    OneSiteDMs::ExpectationT traces=ro1.GetTraces();
    double eps=1e-14;
    for (int ia=1;ia<=L;ia++)
    {
         EXPECT_NEAR(traces(ia),1.0,eps);
    }
}

TEST_F(ExpectationsTests,TestOneSiteExpectations)
{
    int L=9,D=4;
    double S=0.5;
    Setup(L,S,D);

    OneSiteDMs::ExpectationT Sx_mpo(L),Sz_mpo(L);
    Vector<std::complex<double> > Sp_mpo(L),Sm_mpo(L);
    for (int ia=1;ia<=L;ia++)
    {
        TensorNetworks::Operator* Sxo=new TensorNetworks::MPO_OneSite(L,S ,ia, TensorNetworks::Sx);
        TensorNetworks::Operator* Szo=new TensorNetworks::MPO_OneSite(L,S ,ia, TensorNetworks::Sz);
        TensorNetworks::Operator* Spo=new TensorNetworks::MPO_OneSite(L,S ,ia, TensorNetworks::Sp);
        TensorNetworks::Operator* Smo=new TensorNetworks::MPO_OneSite(L,S ,ia, TensorNetworks::Sm);
        Sx_mpo(ia)=itsMPS->GetExpectation(Sxo);
        Sz_mpo(ia)=itsMPS->GetExpectation(Szo);
        Sp_mpo(ia)=itsMPS->GetExpectationC(Smo);
        Sm_mpo(ia)=itsMPS->GetExpectationC(Spo);
        delete Sxo;
        delete Szo;
        delete Spo;
        delete Smo;
    }
    OneSiteDMs::ExpectationT Sy_mpo=-0.5*imag(Sp_mpo-Sm_mpo);

    TensorNetworks::SpinCalculator sc(S);

    OneSiteDMs ro1=itsMPS->CalculateOneSiteDMs();
    OneSiteDMs::ExpectationT Sx=ro1.Contract(sc.GetSx());
    OneSiteDMs::ExpectationT Sy=ro1.Contract(sc.GetSy());
    OneSiteDMs::ExpectationT Sz=ro1.Contract(sc.GetSz());

    double eps=1e-14;
    for (int ia=1;ia<=L;ia++)
    {
        EXPECT_NEAR(Sx(ia),Sx_mpo(ia),eps);
        EXPECT_NEAR(Sy(ia),Sy_mpo(ia),eps);
        EXPECT_NEAR(Sz(ia),Sz_mpo(ia),eps);
    }

/*    for (int ia=0;ia<L;ia++)
    {
        double S=sqrt(Sx(ia)*Sx(ia) + Sy(ia)*Sy(ia) + Sz(ia)*Sz(ia));
        cout << "Site " << ia << " S^2=" << S << endl;
    }
    double E=0.0;
    for (int ia=0;ia<L-1;ia++)
    {
        double S=Sx(ia)*Sx[ia+1] + Sy(ia)*Sy[ia+1] + Sz(ia)*Sz[ia+1];
        E+=S;
        cout << "Site " << ia << " Si*Si+1=" << S << endl;
    }
    cout << "E=" << E << endl;
*/
}
TEST_F(ExpectationsTests,TestFreezeL9S1D2)
{
    int L=9,D=2,maxIter=100;
    double S=0.5;
    Setup(L,S,D);
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->Freeze(L,S); //Site 0 spin up


    TensorNetworks::IterationSchedule is;
    itsEps.itsDelatEnergy1Epsilon=1e-9;
    is.Insert({maxIter,D,itsEps});

    int nSweep=itsMPS->FindVariationalGroundState(itsH,is);

    double E=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(E/(L-1),-0.45317425 ,1e-7);
    EXPECT_LT(nSweep,maxIter);

    TensorNetworks::SpinCalculator sc(S);
    OneSiteDMs ro1=itsMPS->CalculateOneSiteDMs();
    OneSiteDMs::ExpectationT Sx=ro1.Contract(sc.GetSx());
    OneSiteDMs::ExpectationT Sy=ro1.Contract(sc.GetSy());
    OneSiteDMs::ExpectationT Sz=ro1.Contract(sc.GetSz());

//    cout << "Sx=" << Sx << endl;
//    cout << "Sy=" << Sy << endl;
//    cout << "Sz=" << Sz << endl;
}



SMatrix<DMatrix<double> > SuseptibilityTensor(const OneSiteDMs& dm1,const TwoSiteDMs& dm2)
{
    int L=dm1.GetL();
    assert(dm2.GetL()==L);
    assert(dm2.GetS()==dm1.GetS());
    SMatrix<DMatrix<double> > ret(L,L);
    TensorNetworks::SpinCalculator sc(dm1.GetS());

    OneSiteDMs::ExpectationT Sx=dm1.Contract(sc.GetSx());
    OneSiteDMs::ExpectationT Sy=dm1.Contract(sc.GetSy());
    OneSiteDMs::ExpectationT Sz=dm1.Contract(sc.GetSz());
    TwoSiteDMs::ExpectationT SxSx=dm2.Contract(sc.GetSxSx());
    TwoSiteDMs::ExpectationT SxSy=dm2.Contract(sc.GetSxSy());
    TwoSiteDMs::ExpectationT SxSz=dm2.Contract(sc.GetSxSz());
    TwoSiteDMs::ExpectationT SySx=dm2.Contract(sc.GetSySx());
    TwoSiteDMs::ExpectationT SySy=dm2.Contract(sc.GetSySy());
    TwoSiteDMs::ExpectationT SySz=dm2.Contract(sc.GetSySz());
    TwoSiteDMs::ExpectationT SzSx=dm2.Contract(sc.GetSzSx());
    TwoSiteDMs::ExpectationT SzSy=dm2.Contract(sc.GetSzSy());
    TwoSiteDMs::ExpectationT SzSz=dm2.Contract(sc.GetSzSz());
    for (int ia=1;ia<=L-1;ia++)
        for (int ib=ia+1;ib<=L;ib++)
        {
            DMatrix<double> Sus(3,3);
            Sus(1,1)=SxSx(ia,ib)-Sx(ia)*Sx(ib);
            Sus(1,2)=SxSy(ia,ib)-Sx(ia)*Sy(ib);
            Sus(1,3)=SxSz(ia,ib)-Sx(ia)*Sz(ib);
            Sus(2,1)=SySx(ia,ib)-Sy(ia)*Sx(ib);
            Sus(2,2)=SySy(ia,ib)-Sy(ia)*Sy(ib);
            Sus(2,3)=SySz(ia,ib)-Sy(ia)*Sz(ib);
            Sus(3,1)=SzSx(ia,ib)-Sz(ia)*Sx(ib);
            Sus(3,2)=SzSy(ia,ib)-Sz(ia)*Sy(ib);
            Sus(3,3)=SzSz(ia,ib)-Sz(ia)*Sz(ib);
            ret(ia,ib)=Sus;
        }
    return ret;
}

SMatrix<DMatrix<double> > SuseptibilityTensor(const TensorNetworks::MPS* mps,const TwoSiteDMs& dm2)
{
    int L=dm2.GetL();
    double S=dm2.GetS();
//    assert(dm2.GetL()==L);
//    assert(dm2.GetS()==mps.GetS());

    OneSiteDMs::ExpectationT Sx_mpo(L),Sz_mpo(L);
    Vector<std::complex<double> > Sp_mpo(L),Sm_mpo(L);
    for (int ia=1; ia<=L; ia++)
    {
        TensorNetworks::Operator* Sxo=new TensorNetworks::MPO_OneSite(L,S,ia, TensorNetworks::Sx);
        TensorNetworks::Operator* Szo=new TensorNetworks::MPO_OneSite(L,S,ia, TensorNetworks::Sz);
        TensorNetworks::Operator* Spo=new TensorNetworks::MPO_OneSite(L,S,ia, TensorNetworks::Sp);
        TensorNetworks::Operator* Smo=new TensorNetworks::MPO_OneSite(L,S,ia, TensorNetworks::Sm);
        Sx_mpo(ia)=mps->GetExpectation(Sxo);
        Sz_mpo(ia)=mps->GetExpectation(Szo);
        Sp_mpo(ia)=mps->GetExpectationC(Smo);
        Sm_mpo(ia)=mps->GetExpectationC(Spo);
        delete Sxo;
        delete Szo;
        delete Spo;
        delete Smo;
    }
    OneSiteDMs::ExpectationT Sy_mpo=-0.5*imag(Sp_mpo-Sm_mpo);

    SMatrix<DMatrix<double> > ret(L,L);
    TensorNetworks::SpinCalculator sc(dm2.GetS());


    TwoSiteDMs::ExpectationT SxSx=dm2.Contract(sc.GetSxSx());
    TwoSiteDMs::ExpectationT SxSy=dm2.Contract(sc.GetSxSy());
    TwoSiteDMs::ExpectationT SxSz=dm2.Contract(sc.GetSxSz());
    TwoSiteDMs::ExpectationT SySx=dm2.Contract(sc.GetSySx());
    TwoSiteDMs::ExpectationT SySy=dm2.Contract(sc.GetSySy());
    TwoSiteDMs::ExpectationT SySz=dm2.Contract(sc.GetSySz());
    TwoSiteDMs::ExpectationT SzSx=dm2.Contract(sc.GetSzSx());
    TwoSiteDMs::ExpectationT SzSy=dm2.Contract(sc.GetSzSy());
    TwoSiteDMs::ExpectationT SzSz=dm2.Contract(sc.GetSzSz());
    for (int ia=1; ia<L; ia++)
        for (int ib=ia+1; ib<=L; ib++)
        {
            DMatrix<double> Sus(3,3);
            Sus(1,1)=SxSx(ia,ib)-Sx_mpo(ia)*Sx_mpo(ib);
            Sus(1,2)=SxSy(ia,ib)-Sx_mpo(ia)*Sy_mpo(ib);
            Sus(1,3)=SxSz(ia,ib)-Sx_mpo(ia)*Sz_mpo(ib);
            Sus(2,1)=SySx(ia,ib)-Sy_mpo(ia)*Sx_mpo(ib);
            Sus(2,2)=SySy(ia,ib)-Sy_mpo(ia)*Sy_mpo(ib);
            Sus(2,3)=SySz(ia,ib)-Sy_mpo(ia)*Sz_mpo(ib);
            Sus(3,1)=SzSx(ia,ib)-Sz_mpo(ia)*Sx_mpo(ib);
            Sus(3,2)=SzSy(ia,ib)-Sz_mpo(ia)*Sy_mpo(ib);
            Sus(3,3)=SzSz(ia,ib)-Sz_mpo(ia)*Sz_mpo(ib);
            ret(ia,ib)=Sus;
        }
    return ret;
}

TEST_F(ExpectationsTests,TestTwoSiteDMs)
{
    int L=9,D=4;
    double S=0.5;
    Setup(L,S,D);

    TwoSiteDMs::ExpectationT SxSx_mpo(L,L),SxSz_mpo(L,L),SzSx_mpo(L,L),SzSz_mpo(L,L);
    SMatrix<std::complex<double> > SmSm_mpo(L,L),SmSp_mpo(L,L),SpSm_mpo(L,L),SpSp_mpo(L,L);
    Fill(SxSx_mpo,0.0);
    Fill(SxSz_mpo,0.0);
    Fill(SzSx_mpo,0.0);
    Fill(SzSz_mpo,0.0);
    Fill(SmSm_mpo,std::complex<double>(0.0));
    Fill(SmSp_mpo,std::complex<double>(0.0));
    Fill(SpSm_mpo,std::complex<double>(0.0));
    Fill(SpSp_mpo,std::complex<double>(0.0));
    for (int ia=1;ia<L;ia++)
        for (int ib=ia+1;ib<=L;ib++)
    {
        TensorNetworks::Operator* SxSxo=new TensorNetworks::MPO_TwoSite(L,S ,ia,ib, TensorNetworks::Sx,TensorNetworks::Sx);
        TensorNetworks::Operator* SxSzo=new TensorNetworks::MPO_TwoSite(L,S ,ia,ib, TensorNetworks::Sx,TensorNetworks::Sz);
        TensorNetworks::Operator* SzSxo=new TensorNetworks::MPO_TwoSite(L,S ,ia,ib, TensorNetworks::Sz,TensorNetworks::Sx);
        TensorNetworks::Operator* SzSzo=new TensorNetworks::MPO_TwoSite(L,S ,ia,ib, TensorNetworks::Sz,TensorNetworks::Sz);
        TensorNetworks::Operator* SmSmo=new TensorNetworks::MPO_TwoSite(L,S ,ia,ib, TensorNetworks::Sm,TensorNetworks::Sm);
        TensorNetworks::Operator* SmSpo=new TensorNetworks::MPO_TwoSite(L,S ,ia,ib, TensorNetworks::Sm,TensorNetworks::Sp);
        TensorNetworks::Operator* SpSmo=new TensorNetworks::MPO_TwoSite(L,S ,ia,ib, TensorNetworks::Sp,TensorNetworks::Sm);
        TensorNetworks::Operator* SpSpo=new TensorNetworks::MPO_TwoSite(L,S ,ia,ib, TensorNetworks::Sp,TensorNetworks::Sp);
        SxSx_mpo(ia,ib)=itsMPS->GetExpectation (SxSxo);
        SxSz_mpo(ia,ib)=itsMPS->GetExpectation (SxSzo);
        SzSx_mpo(ia,ib)=itsMPS->GetExpectation (SzSxo);
        SzSz_mpo(ia,ib)=itsMPS->GetExpectation (SzSzo);
        SmSm_mpo(ia,ib)=itsMPS->GetExpectationC(SmSmo);
        SmSp_mpo(ia,ib)=itsMPS->GetExpectationC(SmSpo);
        SpSm_mpo(ia,ib)=itsMPS->GetExpectationC(SpSmo);
        SpSp_mpo(ia,ib)=itsMPS->GetExpectationC(SpSpo);
        delete SxSxo;
        delete SxSzo;
        delete SzSxo;
        delete SzSzo;
        delete SmSmo;
        delete SmSpo;
        delete SpSmo;
        delete SpSpo;
    }




    TwoSiteDMs ros=itsMPS->CalculateTwoSiteDMs();
    TwoSiteDMs::ExpectationT traces=ros.GetTraces();
    TwoSiteDMs::ExpectationT VNs=ros.GetVNEntropies();
     for (int ia=1; ia<L; ia++)
        for (int ib=ia+1; ib<=L; ib++)
        {
            EXPECT_NEAR(traces(ia,ib),1.0,1e-13);
            EXPECT_GT(VNs(ia,ib),0.0);
            EXPECT_LE(VNs(ia,ib),log(D));
        }
    TensorNetworks::SpinCalculator sc(S);
    TwoSiteDMs::ExpectationT SxSx=ros.Contract(sc.GetSxSx());
    TwoSiteDMs::ExpectationT SxSy=ros.Contract(sc.GetSxSy());
    TwoSiteDMs::ExpectationT SxSz=ros.Contract(sc.GetSxSz());
    TwoSiteDMs::ExpectationT SySx=ros.Contract(sc.GetSySx());
    TwoSiteDMs::ExpectationT SySy=ros.Contract(sc.GetSySy());
    TwoSiteDMs::ExpectationT SySz=ros.Contract(sc.GetSySz());
    TwoSiteDMs::ExpectationT SzSx=ros.Contract(sc.GetSzSx());
    TwoSiteDMs::ExpectationT SzSy=ros.Contract(sc.GetSzSy());
    TwoSiteDMs::ExpectationT SzSz=ros.Contract(sc.GetSzSz());
    double eps=1e-14;
    for (int ia=1; ia<L; ia++)
        for (int ib=ia+1; ib<=L; ib++)
        {
            EXPECT_NEAR(SxSx(ia,ib),SxSx_mpo(ia,ib),eps);
            EXPECT_NEAR(SxSz(ia,ib),SxSz_mpo(ia,ib),eps);
            EXPECT_NEAR(SzSx(ia,ib),SzSx_mpo(ia,ib),eps);
            EXPECT_NEAR(SzSz(ia,ib),SzSz_mpo(ia,ib),eps);
        }

    OneSiteDMs ro1=itsMPS->CalculateOneSiteDMs();
//    OneSiteDMs::ExpectationT Sx=ro1.Contract(sc.GetSx());
//    OneSiteDMs::ExpectationT Sy=ro1.Contract(sc.GetSy());
//    OneSiteDMs::ExpectationT Sz=ro1.Contract(sc.GetSz());
    SMatrix<DMatrix<double> > Sus1=SuseptibilityTensor(ro1,ros);
    SMatrix<DMatrix<double> > Sus2=SuseptibilityTensor(itsMPS,ros);

    for (int ia=1;ia<L;ia++)
        for (int ib=ia+1;ib<=L;ib++)
        {
 //           cout << "Sites (" << ia << "," << ib << "): <S_a*S_b>-<S_a>*<S_b>=" << Sus(ia,ib) <<endl;
            double err=Max(abs(Sus1(ia,ib)-Sus2(ia,ib)));
            double err1=Max(abs(Sus1(ia,ib)-Transpose(Sus1(ia,ib))));
            double err2=Max(abs(Sus2(ia,ib)-Transpose(Sus2(ia,ib))));
            EXPECT_NEAR(err,0.0,1e-14);
            EXPECT_NEAR(err1,0.0,2e-8);
            EXPECT_NEAR(err2,0.0,2e-8);
//            cout << "Err 12,nonsym1,nonsym2=" << err << " " << err1 << " " << err2 << endl;
//            DMatrix<double> SusSym=0.5*(Sus2(ia,ib)+ Transpose(Sus2(ia,ib)));
//            cout << "Sites (" << ia << "," << ib << "): Eigen Values=" << Diagonalize(SusSym) <<endl;
        }
    //
    // Check ground state energy
    //
    double E1=0.0;
    for (int ia=1;ia<L;ia++)
        E1+=SxSx(ia,ia+1)+SySy(ia,ia+1)+SzSz(ia,ia+1);
    double E2=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(E1,E2,1e-13);
//    cout << "E1-E2=" << E1-E2 << endl;
}


#define TYPE DMatrix<double>
#include "oml/src/smatrix.cc"

