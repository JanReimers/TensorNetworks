#ifndef MPSPLOTABLEIMP_H
#define MPSPLOTABLEIMP_H

#include "TensorNetworks/MPSPlotable.H"
#include "TensorNetworksImp/MPSImp.H"
#include "TensorNetworksImp/MultiPlotableImp.H"
#include "Functions/Mesh/TMesh.H"
#include "Misc/ptr_vector.h"

class PlotableMesh;

class MPSPlotableImp
: public virtual MPSPlotable
, public virtual MultiPlotable
, private MPSImp
, private MultiPlotableImp
{
//--------------------------------------------------------------------------------------
//
//  Construction and initialization zone
//
public:
        MPSPlotableImp(int L, double S, int D,double normEps=1e-12,TNSLogger* s=0);
        MPSPlotableImp(const MPSPlotableImp& mps);
        MPS* Clone() const;

        virtual ~MPSPlotableImp();

        virtual void   UpdateBondData(int isite);
        virtual void   UpdateEnergyData(int isite);
        virtual void   IterationDeltaE(int iter,double dE);
        virtual void   IterationEnergy(double E);

//--------------------------------------------------------------------------------------
//
//      Multi Plotting interface
//
        virtual void MakeAllGraphs();
        virtual void Insert(Plotting::MultiGraph* graphs);
        virtual void Select(int index);

private:
        void InitPlotting();


//--------------------------------------------------------------------------------------
//
//  Data members
//
        int    itsSelectedSite; //Selected site for plots which show lots of data for one site.

        UniformMesh*  itsSitesMesh;
        Mesh*         itsBondsMesh;
        UniformMesh*  itsSVMesh;

        PlotableMesh* itsSitesPMesh;
        PlotableMesh* itsBondsPMesh;
        PlotableMesh* itsSVMeshPMesh;

        ArrayRT itsSiteEnergies;  //Minimum eigen value of Heff on each site.
        ArrayRT itsSiteEGaps;     //Gap to second eigen value on each site.
        ArrayRT itsBondEntropies; // Sum(s^2*ln(s^2)) for each bond
        ArrayRT itsBondMinSVs;    // smallest singluar value for each bond
        ArrayRT itsBondRanks;     // Effective rank based on non zero singular values
        VectorRT itssSelectedEntropySpectrum; //full spectrum of SVs for a selected site.

        static GraphDefinition theGraphs[];
        static const int n_graphs;

};

#endif // MPSPLOTABLEIMP_H
