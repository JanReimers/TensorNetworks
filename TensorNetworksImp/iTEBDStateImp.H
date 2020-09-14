#ifndef ITEBDSTATEIMP_H
#define ITEBDSTATEIMP_H

#include "TensorNetworks/iTEBDState.H"
#include "TensorNetworksImp/MPSImp.H"

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
        iTEBDStateImp(int L,double S, int D,double normEps=1e-12,TNSLogger* s=0);
        virtual ~iTEBDStateImp();
        virtual void InitializeWith(TensorNetworks::State);
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
        virtual void Normalize(TensorNetworks::Direction LR);
//        virtual void NormalizeAndCompress(TensorNetworks::Direction LR,int Dmax,double epsMin);
        virtual void Apply(int isite,const Matrix4T& expH);
        virtual void        Report    (std::ostream&) const;

    protected:
        int GetModSite(int isite);
    private:
        void InitSitesAndBonds();

        // MPSImp::itsBonds; is used to store sing values, or lambda's, Schmidt weights.
        // MPSImp::itsSites; is used to store Gamma matrices

};

#endif // ITEBDSTATEIMP_H
