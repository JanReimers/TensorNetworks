#ifndef ITEBDSTATEIMP_H
#define ITEBDSTATEIMP_H

#include "TensorNetworks/iTEBDState.H"
#include "TensorNetworksImp/MPSImp.H"

namespace TensorNetworks
{

//--------------------------------------------------------------------------------
//
//  Implement Vidals  ....Lambda-Gamma-Lambda-Gamma-Lambda... canonical form.
//  Try and re-use some of the MPSImp code through private inheritance
//  L in this class is interpreted as the number of Gammas in the (infinityl) repeating unit cell.
//
class iTEBDStateImp : public virtual iTEBDState,
    private MPSImp
{
    public:
        iTEBDStateImp(int L,double S, int D,double normEps,double epsSV);
        virtual ~iTEBDStateImp();
        virtual void InitializeWith(State);
 //--------------------------------------------------------------------------------------
//
//  Access to information
//
        int     GetL        () const {return MPSImp::GetL();}
        int     Getp        () const {return MPSImp::Getp();}
//--------------------------------------------------------------------------------------
//
//  Normalization members with and without SVD compression
//
        virtual void NormalizeAndCompress(Direction LR,SVCompressorC*);

        virtual void Apply(int isite,const Matrix4RT& expH);
        virtual void Report(std::ostream&) const;

    protected:
        int GetModSite(int isite);
        VectorRT&  GetLambda(int isite);
        MatrixCT& GetGamma (int isite,int n);

    private:
        void InitSitesAndBonds(int D, double epsSV);
};

}
#endif // ITEBDSTATEIMP_H
