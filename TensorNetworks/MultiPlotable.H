// File: MultiPlotable.H  Interface for any class the has data to be plotted.
#ifndef _MultiPlotable_H_
#define _MultiPlotable_H_

// Copyright (2002-2003), Jan N. Reimers

#include "Plotting/Plotable.H"
#include "Plotting/Namespace.H"
#include "Misc/c_str.H"

class MultiPlotable
  : public virtual Plotting::Plotable
{
public:
  virtual ~MultiPlotable() {};
  //
  //  Plotting interface.
  //
  //! show on a specific device.
  virtual void Show  (Plotting::Driver*,c_str title)=0;
  //! If any lines match plotter units and/or name , plot them.
  virtual void DrawOn(Plotting::Plotter*) const=0; // Required by Plotable.
  //! Permenantly attach drivers to graphs.  Unhide form base class.  Only call this once.
  virtual void Insert     (Plotting::MultiGraph*)=0;
  //! Temporarily attach drivers to graphs.
  virtual void Attach     (Plotting::MultiGraph*)=0;
  //! Insert misc. plotabel objects.
  virtual void Insert(const Plotting::Plotable*)=0;
};

#endif //_MultiPlotable_H_
