#include "TensorNetworksImp/MPSPlotableImp.H"
#include "TensorNetworksImp/Bond.H"
#include "Functions/Mesh/PlotableMesh.H"
#include "Plotting/Factory.H"
#include "Plotting/MultiGraph.H"
#include "Misc/Dimension.H"

#include <iostream>
#include <iomanip>

using Dimensions::PureNumber;

using std::cout;
using std::endl;

namespace TensorNetworks
{

//-------------------------------------------------------------------------------
//
//  Init/construction zone
//
MPSPlotableImp::MPSPlotableImp(int L, double S, int D,double normEps,TNSLogger* logger)
    : MPSImp(L,S,D,normEps,logger)
    , itsSelectedSite(1)
    , itsSitesMesh(0)
    , itsBondsMesh(0)
{
    InitPlotting();
}

MPSPlotableImp::MPSPlotableImp(const MPSPlotableImp& mps)
    : MPSImp(mps)
    , itsSelectedSite(mps.itsSelectedSite)
    , itsSitesMesh   (0)
    , itsBondsMesh   (0)
{
    InitPlotting();
}

//Used for iTEBD states
/*MPSPlotableImp::MPSPlotableImp(int L, double S, int D,Direction lr,double normEps,TNSLogger* logger)
    : MPSImp(L,S,D,lr,normEps,logger)
    , itsSelectedSite(1)
    , itsSitesMesh(0)
    , itsBondsMesh(0)
{
    InitPlotting();
}
*/

MPSPlotableImp::~MPSPlotableImp()
{
//    cout << "MatrixProductStateImp destructor." << endl;
    delete itsBondsPMesh;
    delete itsSitesPMesh;
    delete itsBondsMesh;
    delete itsSitesMesh;
}

MPS* MPSPlotableImp::Clone() const
{
    assert(this->itsDmax>0);
    return new MPSPlotableImp(*this);
}



void MPSPlotableImp::InitPlotting()
{
    Range rsites(1.0,itsL);
    itsSitesMesh=new UniformMesh(rsites,1.0);
    itsBondsMesh=new Mesh(itsSitesMesh->CenterPoints());
    itsSVMesh=new UniformMesh(Range(1,itsDmax+1),1.0); //Range 1->D causes problems when D=1

    itsSitesPMesh = new PlotableMesh(*itsSitesMesh,"none");
    itsBondsPMesh = new PlotableMesh(*itsBondsMesh,"none");
    itsSVMeshPMesh= new PlotableMesh(*itsSVMesh   ,"none");

    itsSiteEnergies .SetSize(itsL+1);
    itsSiteEGaps    .SetSize(itsL+1);
    itsBondEntropies.SetSize(itsL);
    itsBondMinSVs   .SetSize(itsL);
    itsBondRanks    .SetSize(itsL);
    itssSelectedEntropySpectrum.SetLimits(itsDmax);

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
    // Todo: Get this to compile.
/*    itsSVMeshPMesh->Insert
    (
        new TPlotableMeshClient<PureNumber,Vector<double> >
        (*itsSVMeshPMesh,"Singular Values","SVs",NamedUnit("none","SVs"),PureNumber(1.0),itssSelectedEntropySpectrum, Plotting::Red)
        ,Plotting::Circle,false
    );
*/
}

void MPSPlotableImp::UpdateBondData(int isite)
{
    CheckBondNumber(isite);
    itsBondEntropies[isite]=itsBonds[isite]->GetBondEntropy();
    itsBondMinSVs   [isite]=log10(itsBonds[isite]->GetMinSV());
    itsBondRanks    [isite]=itsBonds[isite]->GetRank();
    if (isite==itsSelectedSite)
        itssSelectedEntropySpectrum=itsBonds[isite]->GetSVs();

}
void MPSPlotableImp::UpdateEnergyData(int isite)
{
    itsSiteEnergies[isite]=itsSites[isite]->GetSiteEnergy();
    itsSiteEGaps   [isite]=itsSites[isite]->GetEGap      ();
}

void MPSPlotableImp::IterationDeltaE(int iter,double dE)
{
    if (weHaveGraphs())
    {
        if (dE<1e-16) dE=1e-16;
        double diter=itsNSweep+static_cast<double>(iter)/(itsL-1); //Fractional iter count for log(dE) plot
        //cout << "ia,diter,de=" << ia << " " << diter << " " << de << endl;
        AddPoint("Iter log(dE/J)",Plotting::Point(diter,log10(dE)));
    }

}

void MPSPlotableImp::IterationEnergy(double E)
{
    if (weHaveGraphs())
        AddPoint("Iter E/J",Plotting::Point(itsNSweep,E));
}



GraphDefinition MPSPlotableImp::theGraphs[]=
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

const int MPSPlotableImp::n_graphs=sizeof(MPSPlotableImp::theGraphs)/sizeof(GraphDefinition);


void MPSPlotableImp::MakeAllGraphs()
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

#define CheckSiteNumber(ia)\
    assert(ia>=1);\
    assert(ia<=itsL);\

void MPSPlotableImp::Select(int index)
{
    CheckSiteNumber(index);
    if (itsSelectedSite!=index)
        itssSelectedEntropySpectrum=itsBonds[index]->GetSVs();
    itsSelectedSite=index;
}

void MPSPlotableImp::Insert(Plotting::MultiGraph* graphs)
{
    graphs->InsertLayer("Iterations");
    graphs->InsertLayer("Sites");
    graphs->InsertLayer("Bonds");
    MultiPlotableImp::Insert(graphs);
}

} // namespace
