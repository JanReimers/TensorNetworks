// File: MultiPlotableImp.C  Helper class for any class the has acumulated lines of data to be plotted.

// Copyright (2002-2003), Jan N. Reimers

#include "TensorNetworksImp/MultiPlotableImp.H"
#include "Plotting/MultiGraph.H"
#include "Plotting/Factory.H"
#include "Plotting/CurveUnits.H"
#include <iostream>
#include <cstdlib>

typedef optr_map   <std::string,Plotting::Graph*>::iterator  GITER;
typedef  ptr_vector<            Plotting::Graph*>::iterator OGITER;
typedef optr_map   <std::string,Plotting::BufferedLine*>::      iterator  LITER;
typedef optr_map   <std::string,Plotting::BufferedLine*>::const_iterator CLITER;
typedef  ptr_vector<const Plotting::Plotable*>::const_iterator CPITER;

MultiPlotableImp::MultiPlotableImp()
  : itsMultiGraph(0)
{}

MultiPlotableImp::~MultiPlotableImp()
{
     std::cout << "MultiPlotableImp destructor." << std::endl;
}

void MultiPlotableImp::Insert(Plotting::Graph* g, c_str layer)
{
  assert(g);
  c_str key=g->GetName();
  if (itsGraphs.find(key)!=itsGraphs.end())
  {
    std::cerr << "Graph " << key << " already in inserted into  MultiPlotableImp" << std::endl;
  }

  assert(itsGraphs.find(key)==itsGraphs.end());
  itsGraphs[key]=g;
  itsOrderedGraphs.push_back(g);
  DrawOn(g);
  if (layer) itsLayers[key]=layer;
}


void MultiPlotableImp::InsertLine(c_str key, c_str title,Plotting::CurveUnits u,Plotting::Colour c)
{
  assert(key);
  assert(title);
  const Plotting::Factory* fac=Plotting::Factory::GetFactory();
  Plotting::BufferedLine* l=fac->MakeLine(title,u);
  assert(l);
  l->SetLineColour(c);
  itsLines[key]=l;
}

void MultiPlotableImp::Insert(const Plotting::Plotable* p)
{
  assert(p);
  itsPlotables.push_back(p);
}

void MultiPlotableImp::DrawLineOn(c_str key,Plotting::Plotter* p)
{
  Plotting::BufferedLine* l=FindLine(key);
  l->DrawOn(p);
}


void MultiPlotableImp::AddPoint(c_str key,Plotting::Point p)
{
  FindLine(key)->Add(p);
}

void MultiPlotableImp::Clear(c_str key)
{
  FindLine(key)->Clear();
}

const Plotting::BufferedLine* MultiPlotableImp::FindLine(c_str key) const
{
  assert(key);
  CLITER l=itsLines.find(key);
  if (l==itsLines.end())
  {
    std::cerr << "MultiPlotableImp::FindLine unknown line named '" << key << "'" << std::endl;
    std::cerr << "  Know line names are: " << std::endl;
    for (CLITER i=itsLines.begin();i!=itsLines.end();i++)
      std::cerr << i.GetKey() << ", ";
    std::cerr << std::endl;
    exit(-1);
  }
  return &l;
}

Plotting::BufferedLine* MultiPlotableImp::FindLine(c_str key)
{
  const MultiPlotableImp* pmi(this);
  return const_cast<Plotting::BufferedLine*>(pmi->FindLine(key));
}

//
//  Hunt through all graphs and find one with the same title.
//
void MultiPlotableImp::Show(Plotting::Driver* d,c_str title)
{
  GITER i=itsGraphs.find(title);
  if (i!=itsGraphs.end())
      i->Show(d);
}

void MultiPlotableImp::DrawOn(Plotting::Plotter* p) const
{
    static Plotting::CurveUnits none("none","none");

    for (CLITER i=itsLines.begin(); i!=itsLines.end(); i++)
    {
        if (i->GetUnits()==none && p->CheckUnits(&i))
        {
            //Resort to name checks if there are no units
//            std::cout << "MultiPlotableImp::DrawOn comparing line='" << i.GetKey() << "', plotter name='" << p->GetName() << "'" << std::endl;
            if (i.GetKey()==std::string(p->GetName()))
                i->DrawOn(p);
        }
        else
            i->DrawOn(p);
    }
    for (CPITER i=itsPlotables.begin(); i!=itsPlotables.end(); i++)
        i->DrawOn(p);
}

//
//  Assumes drivers are already made.
//
void MultiPlotableImp::Attach(Plotting::MultiGraph* graphs)
{
  assert(graphs);
  itsMultiGraph=graphs;
  for (GITER g=itsGraphs.begin();g!=itsGraphs.end();g++)
  {
    assert(itsLayers.find(g.GetKey())!=itsLayers.end());
    graphs->Show(&g,itsLayers[g.GetKey()].c_str());
  }
}

void MultiPlotableImp::Insert(Plotting::MultiGraph* graphs)
{
  assert(graphs);
  itsMultiGraph=graphs;
  MakeAllGraphs();

  int Ng=itsOrderedGraphs.size();
  for (int i=0;i<Ng;i++)
  {
    Plotting::Graph* g=itsOrderedGraphs[i];
    assert(g);
    std::string layer=itsLayers[g->GetName()];
    graphs->InsertGraph(g,layer.c_str());
  }
  //  Attach(graphs);
}

void MultiPlotableImp::Restart()
{
  for (LITER i=itsLines.begin();i!=itsLines.end();i++)
    i->Clear();
}

void MultiPlotableImp::ReplotActiveGraph(bool InGUIThread)
{
    if (itsMultiGraph)
        itsMultiGraph->ReplotActiveGraph(InGUIThread);
}
