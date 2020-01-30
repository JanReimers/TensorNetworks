// File: MultiPlotableImp.H  Helper class for any class the has acumulated lines of data to be plotted.
#ifndef _MultiPlotableImp_H_
#define _MultiPlotableImp_H_

// Copyright (2002-2004), Jan N. Reimers

#include "TensorNetworks/MultiPlotable.H"
#include "Plotting/BufferedLine.H"
#include "Plotting/Graph.H"
#include "Misc/ptr_map.h"
#include "Misc/ptr_vector.h"
#include <string>


struct GraphDefinition
{
  c_str Title;
  c_str Xunits;
  c_str Yunits;
  c_str Layer;
};

class MultiPlotableImp
  : public virtual MultiPlotable
{
public:
  MultiPlotableImp();
  virtual ~MultiPlotableImp();
  //
  //  Implement public interface.
  //
  virtual void Show  (Plotting::Driver*,c_str title);
  virtual void DrawOn(Plotting::Plotter*) const; // Required by Plotable.
  virtual void Attach(Plotting::MultiGraph*); //Temporarily attach drivers to graphs.

  virtual void Insert(Plotting::MultiGraph*); //Assumes all layers are present.
  virtual void Insert(const Plotting::Plotable*);
protected:
  //
  //  New graphs all call virtual DrawOn, which cannot be called from a base class constructor.
  //  So we make all graphs in separate function called from dervied class constructor, or from Insert(dd).
  //
  virtual void MakeAllGraphs()=0;
  //
  //  Helper functiosn for derived classes.
  //
  void Insert    (Plotting::Graph*,c_str layer=0);
  //
  //  TODO: document what the rules are for key, are they key=LayerName-GraphName?
  //  TODO: The corollory question: On what layer and what graph is the line plotted?
  //  As far as I can tell this is done strickly based on XY units of the line matching seom graph.
  //     This enforces only one graph for set of XY units.  Or all line of the same units
  //     will end up on all graphs with the same units.  This is not going to for in computational physics
  //     where possibly all graphs and lines will have no units.
  //  We may need to make a new rule if CoordUnits=(none,none) then the key must GraphName-LineName.
  //
  void InsertLine(c_str key, c_str title,Plotting::CurveUnits,Plotting::Colour=Plotting::Black);
  void DrawLineOn(c_str key,Plotting::Plotter*); //Do we need this?
  void AddPoint  (c_str key,Plotting::Point);
  void Clear     (c_str key);

  virtual void Restart();
  virtual void ReplotActiveGraph(bool InGUIThread);
  bool weHaveGraphs() const {return itsMultiGraph;}

protected:
  const Plotting::BufferedLine* FindLine(c_str key) const;
        Plotting::BufferedLine* FindLine(c_str key);
  //
  //  Accumulated data lines, for quatities like V(q) or V(t) that build as the simulation runs.
  //
  optr_map<std::string,Plotting::BufferedLine*> itsLines;
  //
  //  Other plotable objects.
  //
  ptr_vector<const Plotting::Plotable*> itsPlotables;

private:
  MultiPlotableImp(const MultiPlotableImp&) {}; //Don't copy any data.
  //
  //  Graphs.
  //
  optr_map<std::string,Plotting::Graph*> itsGraphs;
  ptr_vector<Plotting::Graph*>           itsOrderedGraphs;
  std::map<std::string,std::string     > itsLayers;
  Plotting::MultiGraph*                  itsMultiGraph;
};

#endif //_MultiPlotableImp_H_
