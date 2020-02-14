#include "Tests.H"
#include "TensorNetworks/Factory.H"
#include "TensorNetworks/Hamiltonian.H"
#include "TensorNetworks/MatrixProductState.H"
#include "TensorNetworks/Epsilons.H"
#include "TensorNetworks/LRPSupervisor.H"


#include "wxDriver/wxFactory.H"
#include "Plotting/MultiGraph.H"
#include "Plotting/CurveUnits.H"

#include "wx/app.h"
#include "wx/frame.h"


class wxMPSApp : public wxApp
{
public:
    virtual ~wxMPSApp() {};
    virtual bool OnInit();
};

void BuildSinCurve(Plotting::BufferedLine* bl,int N, double A, double f, double phi)
{
    static double Pi=3.14159262;
    for (int i=0;i<=N;i++)
    {
        double x=(1.0/N)*i*2.0*Pi;
        double y=A*sin(f*(x+phi));
        bl->Add(x,y);
    }
}

IMPLEMENT_APP(wxMPSApp)

/*
bool wxMPSApp::OnInit()
{
    Plotting::wxFactoryMain fact;
    TensorNetworks::FactoryMain fMain;
    const Plotting::wxFactory* wxfactory=Plotting::wxFactory::GetFactory();
    assert(wxfactory);


    wxFrame *frame = new wxFrame(0,0,"D(x,y,t)",wxPoint(50, 50), wxSize(650, 540));

    Plotting::MultiGraph* graphs=wxfactory->MakewxMultiGraph(frame);
    assert(graphs);
    graphs->InsertLayer("Layer1");
    graphs->InsertLayer("Layer2");

    // Create a Graph object.
    Plotting::CurveUnits cu("none","none");

    Plotting::Graph* g1=wxfactory->MakeGraph("D(x,y,t)",cu);
    assert(g1);
    Plotting::BufferedLine* bl=wxfactory->MakeLine("line 1",cu);
    BuildSinCurve(bl,200,1.0,1.0,0.0);
    bl->DrawOn(g1);
    graphs->InsertGraph(g1,"Layer1");
    g1->Show();

    Plotting::Graph* g3=wxfactory->MakeGraph("F(x,y,t)",cu);
    assert(g3);
    Plotting::BufferedLine* bl3=wxfactory->MakeLine("line 3",cu);
    BuildSinCurve(bl3,200,1.1,2.0,0.5);
    bl3->DrawOn(g3);
    graphs->InsertGraph(g3,"Layer1");
    g3->Show();


    Plotting::Graph* g2=wxfactory->MakeGraph("E(x,y,t)",cu);
    assert(g2);
    Plotting::BufferedLine* bl2=wxfactory->MakeLine("line 2",cu);
    BuildSinCurve(bl2,200,1.1,2.0,0.5);
    bl2->DrawOn(g2);
    graphs->InsertGraph(g2,"Layer2");
    g2->Show();




    frame->Show(TRUE);

    return true;
}
*/

bool wxMPSApp::OnInit()
{
    const Plotting::wxFactory* wxfactory=Plotting::wxFactory::GetFactory();
    assert(wxfactory);
    const TensorNetworks::Factory* tnfactory =TensorNetworks::Factory::GetFactory();


    wxFrame *frame = new wxFrame(0,0,"MPS Studio",wxPoint(50, 50), wxSize(650, 540));

    Plotting::MultiGraph* graphs=wxfactory->MakewxMultiGraph(frame);
    frame->Show(TRUE);

    int L=9,D=2,maxIter=100;
    double S=0.5;
    Epsilons eps;
    LRPSupervisor* supervisor=new LRPSupervisor();
    Hamiltonian* itsH=tnfactory->Make1D_NN_HeisenbergHamiltonian(L,S,1.0,1.0,0.0);
    MatrixProductState* itsMPS=itsH->CreateMPS(D,eps);
    itsMPS->Insert(graphs);

    // This point we need to return true and let the wx event loop take over.
    itsMPS->InitializeWith(TensorNetworks::Random);
    itsMPS->FindGroundState(itsH,maxIter,eps,supervisor);


    return true;
}


