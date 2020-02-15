#include "TensorNetworksImp/MatrixProductStateImp.H"
#include "TensorNetworksImp/Bond.H"
#include "TensorNetworks/Hamiltonian.H"
#include "TensorNetworks/LRPSupervisor.H"
#include "Containers/Matrix4.H"
#include "Functions/Mesh/PlotableMesh.H"
#include "Plotting/CurveUnits.H"
#include "Plotting/Factory.H"
#include "Plotting/MultiGraph.H"
#include "Misc/Dimension.H"
#include "Misc/NamedUnits.H"
#include "oml/cnumeric.h"
#include "oml/vector_io.h"
#include <iostream>
#include <iomanip>

using std::cout;
using std::endl;
using Dimensions::PureNumber;

//-------------------------------------------------------------------------------
//
//  Init/construction zone
//
MatrixProductStateImp::MatrixProductStateImp(int L, double S, int D,const Epsilons& eps)
    : itsL(L)
    , itsDmax(D)
    , itsS(S)
    , itsp(2*S+1)
    , itsNSweep(0)
    , itsSelectedSite(1)
    , itsEpsilons(eps)
    , itsSitesMesh(0)
    , itsBondsMesh(0)
{
    assert(itsL>0);
#ifdef DEBUG
    double ipart;
    double frac=std::modf(2.0*itsS,&ipart);
    assert(frac==0.0);
#endif
    assert(itsS>=0.5);
    assert(itsDmax>0);
    //
    //  Create bond objects
    //
    for (int i=0;i<itsL-1;i++)
        itsBonds.push_back(new Bond(eps.itsSingularValueZeroEpsilon));
    //
    //  Create Sites
    //
    itsSites.push_back(new MatrixProductSite(TensorNetworks::Left,NULL,itsBonds[0],itsp,1,D));
    for (int i=1;i<itsL-1;i++)
        itsSites.push_back(new MatrixProductSite(TensorNetworks::Bulk,itsBonds[i-1],itsBonds[i],itsp,D,D));
    itsSites.push_back(new MatrixProductSite(TensorNetworks::Right,itsBonds[L-2],NULL,itsp,D,1));

    Range rsites(1.0,itsL);
    itsSitesMesh=new UniformMesh(rsites,1.0);
    itsBondsMesh=new Mesh(itsSitesMesh->CenterPoints());
    itsSVMesh=new UniformMesh(Range(1,D+1),1.0); //Range 1->D causes problems when D=1

    itsSitesPMesh = new PlotableMesh(*itsSitesMesh,"none");
    itsBondsPMesh = new PlotableMesh(*itsBondsMesh,"none");
    itsSVMeshPMesh= new PlotableMesh(*itsSVMesh   ,"none");

    itsSiteEnergies .SetSize(itsL);
    itsSiteEGaps    .SetSize(itsL);
    itsBondEntropies.SetSize(itsL-1);
    itsBondMinSVs   .SetSize(itsL-1);
    itsBondRanks    .SetSize(itsL-1);
    itssSelectedEntropySpectrum.SetSize(D);

    Fill(itsSiteEnergies ,0.0);
    Fill(itsSiteEGaps    ,0.0);
    Fill(itsBondEntropies,0.0);
    Fill(itsBondMinSVs   ,0.0);
    Fill(itsBondRanks    ,0.0);
    Fill(itssSelectedEntropySpectrum,1.0);

    NamedUnit EJ("none","Site E/J");
    itsSitesPMesh->Insert
    (
        new TPlotableMeshClient<PureNumber,Array<double> >
        (*itsSitesPMesh,"Site E/J","E/J",EJ,PureNumber(1.0),itsSiteEnergies, Plotting::Green)
        ,Plotting::Circle,false
    );
    itsSitesPMesh->Insert
    (
        new TPlotableMeshClient<PureNumber,Array<double> >
        (*itsSitesPMesh,"Site Egap/J","Egap/J",NamedUnit("none","Egap/J"),PureNumber(1.0),itsSiteEGaps, Plotting::Green)
        ,Plotting::Circle,false
    );
    itsBondsPMesh->Insert
    (
        new TPlotableMeshClient<PureNumber,Array<double> >
        (*itsBondsPMesh,"Bond Entropy","Entropy",NamedUnit("none","Entropy"),PureNumber(1.0),itsBondEntropies, Plotting::Red)
        ,Plotting::Circle,false
    );
    itsBondsPMesh->Insert
    (
        new TPlotableMeshClient<PureNumber,Array<double> >
        (*itsBondsPMesh,"Bond log(Min(s))","log(Min(s))",NamedUnit("none","log(Min(s))"),PureNumber(1.0),itsBondMinSVs, Plotting::Red)
        ,Plotting::Circle,false
    );
    itsBondsPMesh->Insert
    (
        new TPlotableMeshClient<PureNumber,Array<double> >
        (*itsBondsPMesh,"Bond Rank","Rank",NamedUnit("none","Rank"),PureNumber(1.0),itsBondRanks, Plotting::Red)
        ,Plotting::Circle,false
    );
    itsSVMeshPMesh->Insert
    (
        new TPlotableMeshClient<PureNumber,Array<double> >
        (*itsSVMeshPMesh,"Singular Values","SVs",NamedUnit("none","SVs"),PureNumber(1.0),itssSelectedEntropySpectrum, Plotting::Red)
        ,Plotting::Circle,false
    );

}

MatrixProductStateImp::~MatrixProductStateImp()
{
//    cout << "MatrixProductStateImp destructor." << endl;
    delete itsBondsPMesh;
    delete itsSitesPMesh;
    delete itsBondsMesh;
    delete itsSitesMesh;
}

void MatrixProductStateImp::InitializeWith(TensorNetworks::State state)
{
    int sgn=1;
    for (SIter i=itsSites.begin();i!=itsSites.end();i++)
    {
        i->InitializeWith(state,sgn);
        sgn*=-1;
    }

}

c_str SiteMessage(c_str message,int site)
{
    static std::string ret;
    ret=message+std::to_string(site+1); //user sees 1 based site number as opposed to zero based in code
    return ret.c_str();
}
//-------------------------------------------------------------------------------
//
//   Normalization routines
//
void MatrixProductStateImp::Normalize(TensorNetworks::Position LR,LRPSupervisor* Supervisor)
{
    VectorT s; // This get passed from one site to the next.
    if (LR==TensorNetworks::Left)
    {
        MatrixCT Vdagger;// This get passed from one site to the next.
        for (int ia=0;ia<itsL-1;ia++)
        {
            itsSites[ia]->SVDLeft_Normalize(s,Vdagger);
            UpdateBondData(ia);
            Supervisor->DoneOneStep(2,SiteMessage("SVD Left Normalize site ",ia),ia);
            itsSites[ia+1]->Contract(s,Vdagger);
            Supervisor->DoneOneStep(2,SiteMessage("Transfer M=s*V_dagger*M on site",ia+1),ia+1);
        }
        itsSites[itsL-1]->ReshapeAndNormFromLeft(s.GetHigh());
        UpdateBondData(itsL-2);
        Supervisor->DoneOneStep(2,SiteMessage("Rescale site ",itsL-1),itsL-1);
    }
    else if (LR==TensorNetworks::Right)
    {
        MatrixCT U;// This get passed from one site to the next.
        for (int ia=itsL-1;ia>0;ia--)
        {
            itsSites[ia]->SVDRightNormalize(U,s);
            UpdateBondData(ia-1);
            Supervisor->DoneOneStep(2,SiteMessage("SVD Right Normalize site ",ia),ia);
            itsSites[ia-1]->Contract(U,s);
            Supervisor->DoneOneStep(2,SiteMessage("Transfer M=M*U*s on site ",ia-1),ia-1);
        }
        itsSites[0]->ReshapeAndNormFromRight(s.GetHigh());
        UpdateBondData(0);
        Supervisor->DoneOneStep(2,SiteMessage("Rescale site ",0),0);
    }

}

//
//  Mixed canonical  A*A*A*A...A*M(isite)*B*B...B*B
//
void MatrixProductStateImp::Normalize(int isite)
{
    if (isite>0)
    {
        VectorT s; // This get passed from one site to the next.
        MatrixCT Vdagger;// This get passed from one site to the next.
        for (int ia=0; ia<isite; ia++)
        {
            itsSites[ia]->SVDLeft_Normalize(s,Vdagger);
            itsSites[ia+1]->Contract(s,Vdagger);
        }
        itsSites[isite]->ReshapeFromLeft(s.GetHigh());
    }

    if (isite<itsL-1)
    {
        VectorT s; // This get passed from one site to the next.
        MatrixCT U;// This get passed from one site to the next.
        for (int ia=itsL-1; ia>isite; ia--)
        {
            itsSites[ia]->SVDRightNormalize(U,s);
            itsSites[ia-1]->Contract(U,s);
        }
        itsSites[isite]->ReshapeFromRight(s.GetHigh());
    }
}


GraphDefinition MatrixProductStateImp::theGraphs[]=
{
    {"Site E/J"         ,"none"     ,"none"  ,"Sites"     ,"Lattice Site #"},
    {"Site Egap/J"      ,"none"     ,"none"  ,"Sites"     ,"Lattice Site #"},
    {"Singular Values"  ,"none"     ,"none"  ,"Bonds"     ,"SV index"},
    {"Bond Entropy"     ,"none"     ,"none"  ,"Bonds"     ,"Lattice Site #"},
    {"Bond log(Min(s))" ,"none"     ,"none"  ,"Bonds"     ,"Lattice Site #"},
    {"Bond Rank"        ,"none"     ,"none"  ,"Bonds"     ,"Lattice Site #"},
    {"Iter E/J"         ,"none"     ,"none"  ,"Iterations","Iteration #"},
    {"Iter log(dE/J)"   ,"none"     ,"none"  ,"Iterations","Iteration #"},
};

const int MatrixProductStateImp::n_graphs=sizeof(MatrixProductStateImp::theGraphs)/sizeof(GraphDefinition);


void MatrixProductStateImp::MakeAllGraphs()
{
    NamedUnit Xunits("none","Iteration #");
    Plotting::Line* l=0;
    l=InsertLine("Iter E/J"        ,"E/J"       ,Plotting::CurveUnits(Xunits,"none"));
    l->SetLineType(Plotting::NoLine);
    l->SetSymbolType(Plotting::Circle);
    l->SetSymbolColour(Plotting::Blue);
    l=InsertLine("Iter log(dE/J)"  ,"log(dE/J)" ,Plotting::CurveUnits(Xunits,"none"));
    l->SetLineType(Plotting::NoLine);
    l->SetSymbolType(Plotting::Circle);
    l->SetSymbolColour(Plotting::Blue);
    MultiPlotableImp::Insert(itsSitesPMesh);
    MultiPlotableImp::Insert(itsBondsPMesh);
    MultiPlotableImp::Insert(itsSVMeshPMesh);

    Plotting::Graph* g=0;
    for (int i=0; i<n_graphs; i++)
    {
        const GraphDefinition& gd=theGraphs[i];
        NamedUnit x(gd.Xunits,gd.Xtitle);
        NamedUnit y(gd.Yunits,gd.Title);
        g=Plotting::Factory::GetFactory()->MakeGraph(gd.Title,Plotting::CurveUnits(x,y));
//        cout << "New graph " << gd.Title << endl;
        g->SetVerbose();
        if (std::string(gd.Title)=="Bond Entropy")
            g->SetLimits(0.0,1.0,y,Plotting::yAxis);
        if (std::string(gd.Xtitle)=="Lattice Site #")
            g->SetLimits(1.0,itsL,x,Plotting::xAxis);
        if (std::string(gd.Title)=="Singular Values")
        {
            g->SetLogAxis(y,Plotting::yAxis);
            g->SetLimits(1.0,itsDmax,x,Plotting::xAxis);
        }

        MultiPlotableImp::Insert(g,gd.Layer);
    }
}

void MatrixProductStateImp::Select(int index)
{
    assert(index>=0);
    assert(index<itsL-1);
    if (itsSelectedSite!=index)
        itssSelectedEntropySpectrum=itsBonds[index]->GetSVs();
    itsSelectedSite=index;
}


void MatrixProductStateImp::UpdateBondData(int isite)
{
    itsBondEntropies[isite]=itsBonds[isite]->GetBondEntropy();
    itsBondMinSVs   [isite]=log10(itsBonds[isite]->GetMinSV());
    itsBondRanks    [isite]=itsBonds[isite]->GetRank();
    if (isite==itsSelectedSite)
        itssSelectedEntropySpectrum=itsBonds[isite]->GetSVs();
}

//--------------------------------------------------------------------------------------
//
// Find ground state
//
double MatrixProductStateImp::FindGroundState(const Hamiltonian* hamiltonian, int maxIter, const Epsilons& eps,LRPSupervisor* Supervisor)
{
    Supervisor->ReadyToStart("Right normalize");
    Normalize(TensorNetworks::Right,Supervisor);
    Supervisor->DoneOneStep(0,"Load L&R caches");
    LoadHeffCaches(hamiltonian,Supervisor);

    double E1=0;
    for (int in=0; in<maxIter; in++)
    {
        Supervisor->DoneOneStep(0,"Sweep Right");
        E1=SweepRight(hamiltonian,Supervisor,eps,true);
        Supervisor->DoneOneStep(0,"Sweep Left");
        E1=SweepLeft (hamiltonian,Supervisor,eps,true);
        if (GetMaxDeltaE()<eps.itsEnergyConvergenceEpsilon) break;
    }
    Supervisor->DoneOneStep(0,"Contracting <E^2>"); //Supervisor will update the graphs
    double E2=GetExpectation(hamiltonian,hamiltonian);
    return E2-E1*E1;
}


double MatrixProductStateImp::SweepRight(const Hamiltonian* h,LRPSupervisor* Supervisor,const Epsilons& eps,bool quiet)
{
    if (!quiet)
    {
        cout.precision(10);
        cout << "SweepRight  E=" << GetExpectation(h)/(itsL-1) << endl;
    }
    for (int ia=0; ia<itsL-1; ia++)
    {
        Refine(h,Supervisor,eps,ia);

        VectorT s;
        MatrixCT Vdagger;
        Supervisor->DoneOneStep(2,SiteMessage("SVD Left Normalize site ",ia),ia);
        itsSites[ia]->SVDLeft_Normalize(s,Vdagger);
        UpdateBondData(ia);
        Supervisor->DoneOneStep(2,SiteMessage("Transfer M=s*V_dagger*M on site ",ia+1),ia);
        itsSites[ia+1]->Contract(s,Vdagger);
        Supervisor->DoneOneStep(1,SiteMessage("Update L&R caches for site ",ia));
        itsSites[ia]->UpdateCache(h->GetSiteOperator(ia),GetHLeft_Cache(ia-1),GetHRightCache(ia+1));
        if (weHaveGraphs())
        {
            double de=fabs(itsSites[ia]->GetIterDE());
            //cout << "ia,diter,de=" << ia << " " << diter << " " << de << endl;
            if (de<1e-16) de=1e-16;
            double diter=itsNSweep+static_cast<double>(ia)/(itsL-1); //Fractional iter count for log(dE) plot
            AddPoint("Iter log(dE/J)",Plotting::Point(diter,log10(de)));
        }
        if (!quiet) cout << "SweepRight post constract  E=" << GetExpectation(h)/(itsL-1) << endl;
    }
    itsNSweep++;
    Supervisor->DoneOneStep(0,"Calculating Expectation <E>"); //Supervisor will update the graphs
    double E1=GetExpectation(h);
    if (weHaveGraphs())
    {
        AddPoint("Iter E/J",Plotting::Point(itsNSweep,E1));
    }
    return E1;
}

double MatrixProductStateImp::SweepLeft(const Hamiltonian* h,LRPSupervisor* Supervisor,const Epsilons& eps,bool quiet)
{
    if (!quiet)
    {
        cout.precision(10);
        cout << "SweepLeft  entry  E=" << GetExpectation(h)/(itsL-1) << endl;
    }

    for (int ia=itsL-1; ia>0; ia--)
    {
        Refine(h,Supervisor,eps,ia);

        VectorT s;
        MatrixCT U;
        Supervisor->DoneOneStep(2,SiteMessage("SVD Rigft Normalize site ",ia),ia);
        itsSites[ia]->SVDRightNormalize(U,s);
        UpdateBondData(ia-1);
        Supervisor->DoneOneStep(2,SiteMessage("Transfer M=M*U*s on site ",ia-1),ia);
        itsSites[ia-1]->Contract(U,s);
        Supervisor->DoneOneStep(1,SiteMessage("Update L&R caches for site ",ia));
        itsSites[ia]->UpdateCache(h->GetSiteOperator(ia),GetHLeft_Cache(ia-1),GetHRightCache(ia+1));
        if (weHaveGraphs()&&itsNSweep>=1)
        {
            double de=fabs(itsSites[ia]->GetIterDE());
            //cout << "ia,diter,de=" << ia << " " << diter << " " << de << endl;
            if (de<1e-16) de=1e-16;
            double diter=itsNSweep+static_cast<double>(itsL-1-ia)/(itsL-1); //Fractional iter count for log(dE) plot
            AddPoint("Iter log(dE/J)",Plotting::Point(diter,log10(de)));
        }

        if (!quiet)
            cout << "SweepLeft  post contract  E=" << GetExpectation(h)/(itsL-1) << endl;

    }
    itsNSweep++;
    Supervisor->DoneOneStep(0,"Calculating Expectation <E>"); //Supervisor will update the graphs
    double E1=GetExpectation(h);
    if (weHaveGraphs())
    {
        AddPoint("Iter E/J",Plotting::Point(itsNSweep,E1));
    }
    return E1;
}

void MatrixProductStateImp::Refine(const Hamiltonian *h,LRPSupervisor* Supervisor,const Epsilons& eps,int isite)
{
//    assert(CheckNormalized(isite,eps.itsNormalizationEpsilon));
    Supervisor->DoneOneStep(2,"Calculating Heff",isite); //Supervisor will update the graphs
    Matrix6T Heff6=GetHeffIterate(h,isite); //New iterative version
    Supervisor->DoneOneStep(2,"Running eigen solver",isite); //Supervisor will update the graphs
    itsSites[isite]->Refine(Heff6.Flatten(),eps);
    itsSiteEnergies[isite]=itsSites[isite]->GetSiteEnergy();
    itsSiteEGaps   [isite]=itsSites[isite]->GetEGap      ();

}


MatrixProductStateImp::Vector3T MatrixProductStateImp::GetHLeft_Cache (int isite) const
{
    Vector3T HLeft(1,1,1,1);
    HLeft(1,1,1)=eType(1.0);
    if (isite>=0)  HLeft=itsSites[isite]->GetHLeft_Cache();
    return HLeft;
}
MatrixProductStateImp::Vector3T MatrixProductStateImp::GetHRightCache(int isite) const
{
    Vector3T HRight(1,1,1,1);
    HRight(1,1,1)=eType(1.0);
    if (isite<itsL)  HRight=itsSites[isite]->GetHRightCache();
    return HRight;
}


MatrixProductStateImp::Matrix6T MatrixProductStateImp::GetHeffIterate   (const Hamiltonian* h,int isite) const
{
    Vector3T Lcache=GetHLeft_Cache(isite-1);
    Vector3T Rcache=GetHRightCache(isite+1);
    return itsSites[isite]->GetHeff(h->GetSiteOperator(isite),Lcache,Rcache);
}
void MatrixProductStateImp::LoadHeffCaches(const Hamiltonian* h,LRPSupervisor* supervisor)
{
    assert(supervisor);
    GetEOLeft_Iterate(h,supervisor,0,true);
    GetEORightIterate(h,supervisor,0,true);
}

double  MatrixProductStateImp::GetMaxDeltaE() const
{
    double MaxDeltaE=0.0;
    for (int ia=0; ia<itsL; ia++)
    {
        double de=fabs(itsSites[ia]->GetIterDE());
        if (de>MaxDeltaE) MaxDeltaE=de;
    }
    return MaxDeltaE;
}


void MatrixProductStateImp::Insert(Plotting::MultiGraph* graphs)
{
    graphs->InsertLayer("Iterations");
    graphs->InsertLayer("Sites");
    graphs->InsertLayer("Bonds");
    MultiPlotableImp::Insert(graphs);
}




double   MatrixProductStateImp::GetExpectation   (const Operator* o) const
{
    Vector3T F(1,1,1,1);
    F(1,1,1)=eType(1.0);
    for (int ia=0; ia<itsL; ia++)
        F=itsSites[ia]->IterateLeft_F(o->GetSiteOperator(ia),F);

    double iE=std::imag(F(1,1,1));
    if (fabs(iE)>1e-10)
        cout << "Warning: MatrixProductState::GetExpectation Imag(E)=" << std::imag(F(1,1,1)) << endl;

    return std::real(F(1,1,1));
}

eType   MatrixProductStateImp::GetExpectationC(const Operator* o) const
{
    Vector3T F(1,1,1,1);
    F(1,1,1)=eType(1.0);
    for (int ia=0; ia<itsL; ia++)
        F=itsSites[ia]->IterateLeft_F(o->GetSiteOperator(ia),F);

    return F(1,1,1);
}

double   MatrixProductStateImp::GetExpectation(const Operator* o1,const Operator* o2) const
{
    Vector4T F(1,1,1,1,1);
    F(1,1,1,1)=eType(1.0);
    for (int ia=0; ia<itsL; ia++)
        F=itsSites[ia]->IterateLeft_F(o1->GetSiteOperator(ia),o2->GetSiteOperator(ia),F);

    double iE=std::imag(F(1,1,1,1));
    if (fabs(iE)>1e-10)
        cout << "Warning: MatrixProductState::GetExpectation Imag(E)=" << std::imag(F(1,1,1,1)) << endl;

    return std::real(F(1,1,1,1));
}

//--------------------------------------------------------------------------------------
//
//    Reporting
//
void MatrixProductStateImp::Report(std::ostream& os) const
{
    os.precision(3);
    os << "Matrix product state for " << itsL << " lattice sites." << endl;
    os << "  Site  D1  D2  Bond Entropy   #updates  Rank  Sparsisty     Emin      Egap    dA" << endl;
    for (int ia=0; ia<itsL; ia++)
    {
        os << std::setw(3) << ia << "  ";
        itsSites[ia]->Report(os);
        os << endl;
    }
}

std::string MatrixProductStateImp::GetNormStatus(int isite) const
{
    return itsSites[isite]->GetNormStatus(itsEpsilons.itsNormalizationEpsilon);
}

//
//  Used ot L&R caches for Heff calcultions.
//
MatrixProductStateImp::Vector3T MatrixProductStateImp::GetEOLeft_Iterate(const Operator* o,LRPSupervisor* supervisor,int isite, bool cache) const
{
    assert(supervisor);
    Vector3T F(1,1,1,1);
    F(1,1,1)=eType(1.0);
    for (int ia=0; ia<isite; ia++)
    {
        supervisor->DoneOneStep(1,SiteMessage("Calculating L cache for site ",ia));
        F=itsSites[ia]->IterateLeft_F(o->GetSiteOperator(ia),F,cache);
    }
    return F;
}
//
//  Used ot L&R caches for Heff calcultions.
//
MatrixProductStateImp::Vector3T MatrixProductStateImp::GetEORightIterate(const Operator* o,LRPSupervisor* supervisor,int isite, bool cache) const
{
    assert(supervisor);
    Vector3T F(1,1,1,1);
    F(1,1,1)=eType(1.0);
    for (int ia=itsL-1; ia>isite; ia--)
    {
        supervisor->DoneOneStep(1,SiteMessage("Calculating R cache for site ",ia));
        F=itsSites[ia]->IterateRightF(o->GetSiteOperator(ia),F,cache);
    }
    return F;
}


OneSiteDMs MatrixProductStateImp::CalculateOneSiteDMs(LRPSupervisor* supervisor)
{
    OneSiteDMs ret(itsL,itsp);
    Normalize(TensorNetworks::Right,supervisor);
    VectorT s; // This get passed from one site to the next.
    MatrixCT Vdagger;// This get passed from one site to the next.
    for (int ia=0; ia<itsL-1; ia++)
    {
        supervisor->DoneOneStep(2,SiteMessage("Calculate ro(mn) site: ",ia),ia);
        ret.Insert(ia,itsSites[ia]->CalculateOneSiteDM());
        supervisor->DoneOneStep(2,SiteMessage("SVD Left Normalize site: ",ia),ia);
        itsSites[ia]->SVDLeft_Normalize(s,Vdagger);
        UpdateBondData(ia);
        supervisor->DoneOneStep(2,SiteMessage("Transfer M=s*V_dagger*M on site: ",ia+1),ia+1);
        itsSites[ia+1]->Contract(s,Vdagger);
    }
    UpdateBondData(itsL-2);
    supervisor->DoneOneStep(2,SiteMessage("Calculate ro(mn) site: ",itsL-1),itsL-1);
    ret.Insert(itsL-1,itsSites[itsL-1]->CalculateOneSiteDM());
    return ret;
}

MatrixProductStateImp::Matrix4T MatrixProductStateImp::CalculateTwoSiteDM(int ia,int ib) const
{
    assert(ia>=0);
    assert(ib<itsL);
#ifdef DEBUG
    for (int is=0; is<ia; is++)
        assert(GetNormStatus(is)[0]=='A');
    for (int is=ib+1; is<itsL; is++)
        assert(GetNormStatus(is)[0]=='B');
#endif
    Matrix4T ret(itsp,itsp,itsp,itsp,1);
    ret.Fill(eType(0.0));
    // Start the zipper
    for (int m=0; m<itsp; m++)
        for (int n=0; n<itsp; n++)
        {
            MatrixCT C=itsSites[ia]->InitializeTwoSiteDM(m,n);
            for (int ix=ia+1; ix<ib; ix++)
                C=itsSites[ix]->IterateTwoSiteDM(C);
            C=itsSites[ib]->FinializeTwoSiteDM(C);

            for (int m2=0; m2<itsp; m2++)
                for (int n2=0; n2<itsp; n2++)
                    ret(m+1,m2+1,n+1,n2+1)=C(m2+1,n2+1);
        }
    assert(IsHermitian(ret.Flatten(),1e-14));
    return ret;
}


TwoSiteDMs MatrixProductStateImp::CalculateTwoSiteDMs(LRPSupervisor* supervisor)
{
    Normalize(TensorNetworks::Right,supervisor);
    VectorT s; // This get passed from one site to the next.
    MatrixCT Vdagger;// This get passed from one site to the next.
    TwoSiteDMs ret(itsL,itsp);
    for (int ia=0; ia<itsL-1; ia++)
        for (int ib=ia+1;ib<itsL;ib++)
        {
            Matrix4T ro=CalculateTwoSiteDM(ia,ib);
            ret.Insert(ia,ib,ro);
            itsSites[ia]->SVDLeft_Normalize(s,Vdagger);
            UpdateBondData(ia);
            itsSites[ia+1]->Contract(s,Vdagger);
        }
    return ret;
}
