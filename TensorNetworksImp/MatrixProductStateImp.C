#include "TensorNetworksImp/MatrixProductStateImp.H"
#include "TensorNetworksImp/Bond.H"
#include "TensorNetworks/Hamiltonian.H"
#include "TensorNetworks/LRPSupervisor.H"
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
MatrixProductStateImp::MatrixProductStateImp(int L, int S2, int D)
    : itsL(L)
    , itsS2(S2)
    , itsD(D)
    , itsp(itsS2+1)
    , itsNSweep(0)
    , itsSitesMesh(0)
    , itsBondsMesh(0)
{
    //
    //  Create bond objects
    //
    for (int i=0;i<itsL-1;i++)
        itsBonds.push_back(new Bond(1e-12));
    //
    //  Create Sites
    //
    itsSites.push_back(new MatrixProductSite(TensorNetworks::Left,NULL,itsBonds[0],itsp,1,itsD));
    for (int i=1;i<itsL-1;i++)
        itsSites.push_back(new MatrixProductSite(TensorNetworks::Bulk,itsBonds[i-1],itsBonds[i],itsp,itsD,itsD));
    itsSites.push_back(new MatrixProductSite(TensorNetworks::Right,itsBonds[L-2],NULL,itsp,itsD,1));

    Range rsites(1.0,itsL);
    itsSitesMesh=new UniformMesh(rsites,1.0);
    itsBondsMesh=new Mesh(itsSitesMesh->CenterPoints());

    itsSitesPMesh = new PlotableMesh(*itsSitesMesh,"none");
    itsBondsPMesh = new PlotableMesh(*itsBondsMesh,"none");

    itsSiteEnergies .SetSize(itsL);
    itsSiteEGaps    .SetSize(itsL);
    itsBondEntropies.SetSize(itsL-1);
    itsBondMinSVs   .SetSize(itsL-1);
    itsBondRanks    .SetSize(itsL-1);
    Fill(itsSiteEnergies ,0.0);
    Fill(itsSiteEGaps    ,0.0);
    Fill(itsBondEntropies,0.0);
    Fill(itsBondMinSVs   ,0.0);
    Fill(itsBondRanks    ,0.0);

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

}

MatrixProductStateImp::~MatrixProductStateImp()
{
    //cout << "MatrixProductStateImp destructor." << endl;
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
        itsSites[itsL-1]->ReshapeFromLeft(s.GetHigh());
        double norm=std::real(itsSites[itsL-1]->GetLeftNorm()(1,1));
        itsSites[itsL-1]->Rescale(sqrt(norm));
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
        itsSites[0]->ReshapeFromRight(s.GetHigh());
        double norm=std::real(itsSites[0]->GetRightNorm()(1,1));
        itsSites[0]->Rescale(sqrt(norm));
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

bool MatrixProductStateImp::CheckNormalized(int isite,double eps) const
{
    MatrixCT Neff=GetNeff(isite);
    int N=Neff.GetNumRows();
    MatrixCT I(N,N);
    Unit(I);
    double error=Max(abs(Neff-I));
    if (error>1e-12)
        cout << "Warning: Normalization site=" << isite << "  Neff-I error " << error << endl;
    return error<eps;
}

GraphDefinition MatrixProductStateImp::theGraphs[]=
{
    {"Site E/J"         ,"none"     ,"none"  ,"Sites"     ,"Lattice Site #"},
    {"Site Egap/J"      ,"none"     ,"none"  ,"Sites"     ,"Lattice Site #"},
    {"Bond Entropy"     ,"none"     ,"none"  ,"Sites"     ,"Lattice Site #"},
    {"Bond log(Min(s))" ,"none"     ,"none"  ,"Sites"     ,"Lattice Site #"},
    {"Bond Rank"        ,"none"     ,"none"  ,"Sites"     ,"Lattice Site #"},
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

    Plotting::Graph* g=0;
    for (int i=0; i<n_graphs; i++)
    {
        const GraphDefinition& gd=theGraphs[i];
        NamedUnit x(gd.Xunits,gd.Xtitle);
        NamedUnit y(gd.Yunits,gd.Title);
        g=Plotting::Factory::GetFactory()->MakeGraph(gd.Title,Plotting::CurveUnits(x,y));
        g->SetVerbose();
        if (std::string(gd.Title)=="Bond Entropy")
            g->SetLimits(0.0,1.0,y,Plotting::yAxis);
        if (std::string(gd.Xtitle)=="Lattice Site #")
            g->SetLimits(1.0,itsL,y,Plotting::xAxis);
        MultiPlotableImp::Insert(g,gd.Layer);
    }
}

void MatrixProductStateImp::UpdateBondData(int isite)
{
    itsBondEntropies[isite]=itsBonds[isite]->GetBondEntropy();
    itsBondMinSVs   [isite]=log10(itsBonds[isite]->GetMinSV());
    itsBondRanks    [isite]=itsBonds[isite]->GetRank();
}

//--------------------------------------------------------------------------------------
//
// Find ground state
//
double MatrixProductStateImp::FindGroundState(const Hamiltonian* hamiltonian, int maxIter, double eps,LRPSupervisor* Supervisor)
{
    Supervisor->ReadyToStart("Right normalize");
    Normalize(TensorNetworks::Right,Supervisor);
    Supervisor->DoneOneStep(0,"Load L&R caches");
    LoadHeffCaches(hamiltonian,Supervisor);

    double E1=0;
    for (int in=0; in<maxIter; in++)
    {
        Supervisor->DoneOneStep(0,"Sweep Right");
        E1=SweepRight(hamiltonian,Supervisor,true);
        Supervisor->DoneOneStep(0,"Sweep Left");
        E1=SweepLeft (hamiltonian,Supervisor,true);
        if (GetMaxDeltaE()<eps) break;
    }
    Supervisor->DoneOneStep(0,"Contracting <E^2>"); //Supervisor will update the graphs
    double E2=GetExpectation(hamiltonian,hamiltonian);
    return E2-E1*E1;
}


double MatrixProductStateImp::SweepRight(const Hamiltonian* h,LRPSupervisor* Supervisor,bool quiet)
{
    if (!quiet)
    {
        cout.precision(10);
        cout << "SweepRight  E=" << GetExpectationIterate(h)/(itsL-1) << endl;
    }
    double diter=itsNSweep; //Fractional iter count for log(dE) plot
    for (int ia=0; ia<itsL-1; ia++)
    {
        Refine(h,Supervisor,ia);

        VectorT s;
        MatrixCT Vdagger;
        Supervisor->DoneOneStep(2,SiteMessage("SVD Left Normalize site ",ia),ia);
        itsSites[ia]->SVDLeft_Normalize(s,Vdagger);
        UpdateBondData(ia);
        Supervisor->DoneOneStep(2,SiteMessage("Transfer M=s*V_dagger*M on site ",ia+1),ia);
        itsSites[ia+1]->Contract(s,Vdagger);
        Supervisor->DoneOneStep(1,SiteMessage("Update L&R caches for site ",ia));
        itsSites[ia]->UpdateCache(h->GetSiteOperator(ia),GetHLeft_Cache(ia-1),GetHRightCache(ia+1));
        if (weHaveGraphs()&&itsNSweep>=1)
        {
            double de=fabs(itsSites[ia]->GetIterDE());
            //cout << "ia,diter,de=" << ia << " " << diter << " " << de << endl;
            if (de>0.0)
                AddPoint("Iter log(dE/J)",Plotting::Point(diter,log10(de)));
        }
        diter+=1.0/itsL;
        if (!quiet) cout << "SweepRight post constract  E=" << GetExpectationIterate(h)/(itsL-1) << endl;
    }
    itsNSweep++;
    Supervisor->DoneOneStep(0,"Calculating Expectation <E>"); //Supervisor will update the graphs
    double E1=GetExpectationIterate(h);
    if (weHaveGraphs())
    {
        AddPoint("Iter E/J",Plotting::Point(itsNSweep,E1));
    }
    return E1;
}

double MatrixProductStateImp::SweepLeft(const Hamiltonian* h,LRPSupervisor* Supervisor,bool quiet)
{
    if (!quiet)
    {
        cout.precision(10);
        cout << "SweepLeft  entry  E=" << GetExpectationIterate(h)/(itsL-1) << endl;
    }
    double diter=itsNSweep; //Fractional iter count for log(dE) plot
    for (int ia=itsL-1; ia>0; ia--)
    {
        Refine(h,Supervisor,ia);

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
            if (de>0.0)
                AddPoint("Iter log(dE/J)",Plotting::Point(diter,log10(de)));
        }
        diter+=1.0/itsL;
        if (!quiet)
            cout << "SweepLeft  post contract  E=" << GetExpectationIterate(h)/(itsL-1) << endl;

    }
    itsNSweep++;
    Supervisor->DoneOneStep(0,"Calculating Expectation <E>"); //Supervisor will update the graphs
    double E1=GetExpectationIterate(h);
    if (weHaveGraphs())
    {
        AddPoint("Iter E/J",Plotting::Point(itsNSweep,E1));
    }
    return E1;
}

void MatrixProductStateImp::Refine(const Hamiltonian *h,LRPSupervisor* Supervisor,int isite)
{
    assert(CheckNormalized(isite,1e-11));
    Supervisor->DoneOneStep(2,"Calculating Heff",isite); //Supervisor will update the graphs
    Matrix6T Heff6=GetHeffIterate(h,isite); //New iterative version
    Supervisor->DoneOneStep(2,"Running eigen solver",isite); //Supervisor will update the graphs
    itsSites[isite]->Refine(Heff6.Flatten());
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
    graphs->InsertLayer("Sites");
    graphs->InsertLayer("Iterations");
    MultiPlotableImp::Insert(graphs);
}



//--------------------------------------------------------------------------------------
//
//    Overlap and expectation contractions
//
double MatrixProductStateImp::GetOverlap() const
{
    cSIter i=itsSites.begin();
    MatrixCT E=i->GetE();
    i++;
    for (;i!=itsSites.end();i++)
        E=i->GetELeft(E);
    assert(E.GetNumRows()==1);
    assert(E.GetNumCols()==1);
    double iE=fabs(std::imag(E(1,1)));
    if (iE>1e-12)
        cout << "Warning MatrixProductState::GetOverlap imag(E)=" << iE << endl;
    return std::real(E(1,1));
}

double   MatrixProductStateImp::GetExpectationIterate   (const Operator* o) const
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

double MatrixProductStateImp::GetExpectation(const Operator* o) const
{
    assert(o);
    Matrix6T E(1,1);
    E.Fill(std::complex<double>(1.0));

    // E is now a unit row vector
    for (int isite=0;isite<itsL;isite++)
    { //loop over sites
//        Matrix6T temp=mps->GetEO(isite,itsSites[lbr]);
        E*=itsSites[isite]->GetEO(o->GetSiteOperator(isite));
//        cout << "E[" << isite << "]=" << endl;
//        E.Dump(cout);
//        cout << "o Elimits=" << E.GetLimits() << " lbr=" << lbr << endl;
    }
    // at this point E is 1xDw so we need to dot it with a unit vector
 //   Matrix6T Unit(itsp,1);
   // Unit.Fill(std::complex<double>(1.0));
   // E*=Unit;

//    cout << "E =" << E << endl;
//    assert(E.GetNumRows()==1);
//    assert(E.GetNumCols()==1);
    double iE=std::imag(E(1,1,1,1,1,1));
    if (fabs(iE)>1e-10)
        cout << "Warning: MatrixProduct::GetExpectation Imag(E)=" << std::imag(E(1,1,1,1,1,1)) << endl;

    return std::real(E(1,1,1,1,1,1));
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
    return itsSites[isite]->GetNormStatus();
}

//--------------------------------------------------------------------------------------
//
//  Allows unit test classes inside.
//
MatrixProductStateImp::MatrixCT MatrixProductStateImp::GetMLeft(int isite) const
{
   assert(isite<itsL);
    MatrixCT Eleft;
    if (isite==0)
    {
        Eleft.SetLimits(1,1);
        Fill(Eleft,std::complex<double>(1.0));
    }
    else
    {
        Eleft=itsSites[0]->GetE();
    }
    //
    //  Zip from left to right up to isite
    //
//    cout << "ELeft(0)=" << Eleft << endl;
    for (int ia=1;ia<isite;ia++)
    {
            Eleft=itsSites[ia]->GetELeft(Eleft);
 //       cout << "ELeft(" << ia << ")=" << Eleft << endl;
            }
    return Eleft;
}


MatrixProductStateImp::MatrixCT MatrixProductStateImp::GetMRight(int isite) const
{
    MatrixCT Eright;
    if (isite==itsL-1)
    {
        Eright.SetLimits(1,1);
        Fill(Eright,std::complex<double>(1.0));
    }
    else
    {
        Eright=itsSites[itsL-1]->GetE();
    }
    // Zip right to left
    for (int ia=itsL-2;ia>=isite+1;ia--)
        Eright=itsSites[ia]->GetERight(Eright);

    return Eright;
}

 MatrixProductStateImp::MatrixCT MatrixProductStateImp::GetNeff(int isite) const
 {
    return itsSites[isite]->GetNeff(GetMLeft(isite),GetMRight(isite));
 }


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






MatrixProductStateImp::Matrix6T MatrixProductStateImp::GetHeff(const Hamiltonian *h,int isite) const
{
    Matrix6T NLeft =GetEOLeft (h,isite);
    Matrix6T NRight=GetEORight(h,isite);
//    cout << "NLeft " << NLeft  << endl;
//    cout << "NRight" << NRight << endl;
//    assert(NLeft .GetNumRows()==1);
 //   assert(NRight.GetNumCols()==1);
    return itsSites[isite]->GetHeff(h->GetSiteOperator(isite),NLeft,NRight);
}

MatrixProductStateImp::Matrix6T MatrixProductStateImp::GetEOLeft(const Operator *o,int isite) const
{
    Matrix6T NLeft(1,1);
    NLeft.Fill(std::complex<double>(1.0));
    for (int ia=0;ia<isite;ia++)
    { //loop over sites
        NLeft*=itsSites[ia]->GetEO(o->GetSiteOperator(ia));
//        cout << "o Elimits=" << E.GetLimits() << " lbr=" << lbr << endl;
    }
    return NLeft;
}

MatrixProductStateImp::Matrix6T MatrixProductStateImp::GetEORight(const Operator *o,int isite) const
{
    Matrix6T NRight(1,1);
    NRight.Fill(std::complex<double>(1.0));
    for (int ia=itsL-1;ia>isite;ia--)
    { //loop over sites
        Matrix6T temp=NRight;
        Matrix6T E=itsSites[ia]->GetEO(o->GetSiteOperator(ia));

//        cout << "NRight=" <<  NRight << endl;
//        cout << "E=" <<  E << endl;
//        Matrix6T Et=E*temp;
//        cout << "Et=" <<  Et << endl;
        NRight.ClearLimits();
        NRight=E*=temp;
//        cout << "o Elimits=" << E.GetLimits() << " lbr=" << lbr << endl;
    }
    return NRight;
}


