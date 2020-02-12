#include "Tests.H"
#include "TensorNetworks/Hamiltonian.H"
//#include "TensorNetworks/OperatorWRepresentation.H"
//#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/Factory.H"
#include "TensorNetworks/LRPSupervisor.H"
#include "TensorNetworks/Epsilons.H"
#include "TensorNetworksImp/SpinCalculator.H"
#include "TensorNetworksImp/MPO_OneSite.H"
#include "TensorNetworksImp/MPO_TwoSite.H"

#include "oml/stream.h"
#include "oml/array_io.h"
#include "oml/smatrix.h"
#include "oml/numeric.h"

using std::setw;

class ExpectationsTesting : public ::testing::Test
{
public:
    typedef std::complex<double> complx;
    typedef TensorNetworks:: ArrayT  ArrayT;

    ExpectationsTesting()
        : itsFactory(TensorNetworks::Factory::GetFactory())
        , itsH(0)
        , itsMPS(0)
        , itsLRPSupervisor(new LRPSupervisor())
        , itsEps()
    {
        StreamableObject::SetToPretty();

    }
    ~ExpectationsTesting()
    {
        delete itsFactory;
        delete itsLRPSupervisor;
        delete itsH;
        delete itsMPS;
    }

    void Setup(int L, double S, int D)
    {
        delete itsH;
        delete itsMPS;
       itsH=itsFactory->Make1D_NN_HeisenbergHamiltonian(L,S,1.0,1.0,0.0);
        itsMPS=itsH->CreateMPS(D,itsEps);
        itsMPS->InitializeWith(TensorNetworks::Random);
        itsMPS->FindGroundState(itsH,20,itsEps,itsLRPSupervisor);
    }


    const TensorNetworks::Factory* itsFactory=TensorNetworks::Factory::GetFactory();
    Hamiltonian*         itsH;
    MatrixProductState*  itsMPS;
    LRPSupervisor*       itsLRPSupervisor;
    Epsilons             itsEps;
};


TEST_F(ExpectationsTesting,TestSpinCalculatorS12)
{
    SpinCalculator sc(0.5);

    EXPECT_EQ(ToString(sc.GetSx()),"(1:2),(1:2) \n[ 0 0.5 ]\n[ 0.5 0 ]\n");
    EXPECT_EQ(ToString(sc.GetSy()),"(1:2),(1:2) \n[ (0,0) (0,0.5) ]\n[ (0,-0.5) (0,0) ]\n");
    EXPECT_EQ(ToString(sc.GetSz()),"(1:2),(1:2) \n[ -0.5 0 ]\n[ 0 0.5 ]\n");
    EXPECT_EQ(ToString(sc.GetSm()),"(1:2),(1:2) \n[ 0 1 ]\n[ 0 0 ]\n");
    EXPECT_EQ(ToString(sc.GetSp()),"(1:2),(1:2) \n[ 0 0 ]\n[ 1 0 ]\n");

}



TEST_F(ExpectationsTesting,TestSweepL9S1D2)
{
    int L=9,D=2;
    double S=0.5;
    Setup(L,S,D);

    double E=itsMPS->GetExpectation(itsH);
    EXPECT_NEAR(E/(L-1),-0.45317425,1e-7);
}


TEST_F(ExpectationsTesting,TestOneSiteDMs)
{
    int L=9,D=4;
    double S=0.5;
    Setup(L,S,D);
    OneSiteDMs ro1=itsMPS->CalculateOneSiteDMs(itsLRPSupervisor);
    OneSiteDMs::ExpectationT traces=ro1.GetTraces();
    double eps=1e-14;
    for (int ia=0;ia<L;ia++)
    {
         EXPECT_NEAR(traces[ia],1.0,eps);
    }
}

TEST_F(ExpectationsTesting,TestOneSiteExpectations)
{
    int L=9,D=4;
    double S=0.5;
    Setup(L,S,D);

    OneSiteDMs::ExpectationT Sx_mpo(L),Sz_mpo(L);
    Array<std::complex<double> > Sp_mpo(L),Sm_mpo(L);
    for (int ia=0;ia<L;ia++)
    {
        Operator* Sxo=new MPO_OneSite(L,S ,ia, TensorNetworks::Sx);
        Operator* Szo=new MPO_OneSite(L,S ,ia, TensorNetworks::Sz);
        Operator* Spo=new MPO_OneSite(L,S ,ia, TensorNetworks::Sp);
        Operator* Smo=new MPO_OneSite(L,S ,ia, TensorNetworks::Sm);
        Sx_mpo[ia]=itsMPS->GetExpectation(Sxo);
        Sz_mpo[ia]=itsMPS->GetExpectation(Szo);
        Sp_mpo[ia]=itsMPS->GetExpectationC(Smo);
        Sm_mpo[ia]=itsMPS->GetExpectationC(Spo);
        delete Sxo;
        delete Szo;
        delete Spo;
        delete Smo;
    }
    OneSiteDMs::ExpectationT Sy_mpo=-0.5*imag(Sp_mpo-Sm_mpo);

    SpinCalculator sc(S);

    OneSiteDMs ro1=itsMPS->CalculateOneSiteDMs(itsLRPSupervisor);
    OneSiteDMs::ExpectationT Sx=ro1.Contract(sc.GetSx());
    OneSiteDMs::ExpectationT Sy=ro1.Contract(sc.GetSy());
    OneSiteDMs::ExpectationT Sz=ro1.Contract(sc.GetSz());

    double eps=1e-14;
    for (int ia=0;ia<L;ia++)
    {
        EXPECT_NEAR(Sx[ia],Sx_mpo[ia],eps);
        EXPECT_NEAR(Sy[ia],Sy_mpo[ia],eps);
        EXPECT_NEAR(Sz[ia],Sz_mpo[ia],eps);
    }

/*    for (int ia=0;ia<L;ia++)
    {
        double S=sqrt(Sx[ia]*Sx[ia] + Sy[ia]*Sy[ia] + Sz[ia]*Sz[ia]);
        cout << "Site " << ia << " S^2=" << S << endl;
    }
    double E=0.0;
    for (int ia=0;ia<L-1;ia++)
    {
        double S=Sx[ia]*Sx[ia+1] + Sy[ia]*Sy[ia+1] + Sz[ia]*Sz[ia+1];
        E+=S;
        cout << "Site " << ia << " Si*Si+1=" << S << endl;
    }
    cout << "E=" << E << endl;
*/
}



SMatrix<DMatrix<double> > SuseptibilityTensor(const OneSiteDMs& dm1,const TwoSiteDMs& dm2)
{
    int L=dm1.GetL();
    assert(dm2.GetL()==L);
    assert(dm2.GetS()==dm1.GetS());
    SMatrix<DMatrix<double> > ret(0,L-1,0,L-1);
    SpinCalculator sc(dm1.GetS());

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
    for (int ia=0;ia<L-1;ia++)
        for (int ib=ia+1;ib<L;ib++)
        {
            DMatrix<double> Sus(3,3);
            Sus(1,1)=SxSx(ia,ib)-Sx[ia]*Sx[ib];
            Sus(1,2)=SxSy(ia,ib)-Sx[ia]*Sy[ib];
            Sus(1,3)=SxSz(ia,ib)-Sx[ia]*Sz[ib];
            Sus(2,1)=SySx(ia,ib)-Sy[ia]*Sx[ib];
            Sus(2,2)=SySy(ia,ib)-Sy[ia]*Sy[ib];
            Sus(2,3)=SySz(ia,ib)-Sy[ia]*Sz[ib];
            Sus(3,1)=SzSx(ia,ib)-Sz[ia]*Sx[ib];
            Sus(3,2)=SzSy(ia,ib)-Sz[ia]*Sy[ib];
            Sus(3,3)=SzSz(ia,ib)-Sz[ia]*Sz[ib];
            ret(ia,ib)=Sus;
        }
    return ret;
}

TEST_F(ExpectationsTesting,TestTwoSiteDMs)
{
    int L=9,D=4;
    double S=0.5;
    Setup(L,S,D);

    TwoSiteDMs::ExpectationT SxSx_mpo(0,L-1,0,L-1),SxSz_mpo(0,L-1,0,L-1),SzSx_mpo(0,L-1,0,L-1),SzSz_mpo(0,L-1,0,L-1);
    SMatrix<std::complex<double> > SmSm_mpo(0,L-1,0,L-1),SmSp_mpo(0,L-1,0,L-1),SpSm_mpo(0,L-1,0,L-1),SpSp_mpo(0,L-1,0,L-1);
    Fill(SxSx_mpo,0.0);
    Fill(SxSz_mpo,0.0);
    Fill(SzSx_mpo,0.0);
    Fill(SzSz_mpo,0.0);
    Fill(SmSm_mpo,std::complex<double>(0.0));
    Fill(SmSp_mpo,std::complex<double>(0.0));
    Fill(SpSm_mpo,std::complex<double>(0.0));
    Fill(SpSp_mpo,std::complex<double>(0.0));
    for (int ia=0;ia<L-1;ia++)
        for (int ib=ia+1;ib<L;ib++)
    {
        Operator* SxSxo=new MPO_TwoSite(L,S ,ia,ib, TensorNetworks::Sx,TensorNetworks::Sx);
        Operator* SxSzo=new MPO_TwoSite(L,S ,ia,ib, TensorNetworks::Sx,TensorNetworks::Sz);
        Operator* SzSxo=new MPO_TwoSite(L,S ,ia,ib, TensorNetworks::Sz,TensorNetworks::Sx);
        Operator* SzSzo=new MPO_TwoSite(L,S ,ia,ib, TensorNetworks::Sz,TensorNetworks::Sz);
        Operator* SmSmo=new MPO_TwoSite(L,S ,ia,ib, TensorNetworks::Sm,TensorNetworks::Sm);
        Operator* SmSpo=new MPO_TwoSite(L,S ,ia,ib, TensorNetworks::Sm,TensorNetworks::Sp);
        Operator* SpSmo=new MPO_TwoSite(L,S ,ia,ib, TensorNetworks::Sp,TensorNetworks::Sm);
        Operator* SpSpo=new MPO_TwoSite(L,S ,ia,ib, TensorNetworks::Sp,TensorNetworks::Sp);
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




    TwoSiteDMs ros=itsMPS->CalculateTwoSiteDMs(itsLRPSupervisor);
    cout << "Traces=" << ros.GetTraces() << endl;
    cout << "Von Neumann entropies=" << ros.GetVNEntropies() << endl;
    SpinCalculator sc(S);
    TwoSiteDMs::ExpectationT SxSx=ros.Contract(sc.GetSxSx());
    TwoSiteDMs::ExpectationT SxSy=ros.Contract(sc.GetSxSy());
    TwoSiteDMs::ExpectationT SxSz=ros.Contract(sc.GetSxSz());
    TwoSiteDMs::ExpectationT SySx=ros.Contract(sc.GetSySx());
    TwoSiteDMs::ExpectationT SySy=ros.Contract(sc.GetSySy());
    TwoSiteDMs::ExpectationT SySz=ros.Contract(sc.GetSySz());
    TwoSiteDMs::ExpectationT SzSx=ros.Contract(sc.GetSzSx());
    TwoSiteDMs::ExpectationT SzSy=ros.Contract(sc.GetSzSy());
    TwoSiteDMs::ExpectationT SzSz=ros.Contract(sc.GetSzSz());
    cout << "SxSx    =" << SxSx << endl;
    cout << "SxSx_mpo=" << SxSx_mpo << endl;
    cout << "SxSy=" << SxSy << endl;
    cout << "SxSz=" << SxSz << endl;
    cout << "SxSz_mpo=" << SxSz_mpo << endl;
    cout << "SySx=" << SySx << endl;
    cout << "SySy=" << SySy << endl;
    cout << "SySz=" << SySz << endl;
    cout << "SzSx=" << SzSx << endl;
    cout << "SzSx_mpo=" << SzSx_mpo << endl;
    cout << "SzSy=" << SzSy << endl;
    cout << "SzSz=" << SzSz << endl;
    cout << "SzSz_mpo=" << SzSz_mpo << endl;

 /*   OneSiteDMs ro1=itsMPS->CalculateOneSiteDMs(itsLRPSupervisor);
    OneSiteDMs::ExpectationT Sx=ro1.Contract(sc.GetSx());
    OneSiteDMs::ExpectationT Sy=ro1.Contract(sc.GetSy());
    OneSiteDMs::ExpectationT Sz=ro1.Contract(sc.GetSz());
    SMatrix<DMatrix<double> > Sus=SuseptibilityTensor(ro1,ros);

    for (int ia=0;ia<L-1;ia++)
        for (int ib=ia+1;ib<L;ib++)
        {
            cout << "Sites (" << ia << "," << ib << "): <S_a*S_b>-<S_a>*<S_b>=" << Sus(ia,ib) <<endl;
//            cout << "Sites (" << ia << "," << ib << "): Eigen Values=" << Diagonalize(Sus(ia,ib)) <<endl;
        }
        */
}

#define TYPE DMatrix<double>
#include "oml/src/smatrix.cc"

