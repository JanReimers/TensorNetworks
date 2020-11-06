#ifndef ITEBDSTATEIMP_H
#define ITEBDSTATEIMP_H

#include "TensorNetworks/iTEBDState.H"
#include "TensorNetworks/Typedefs.H"
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
        virtual void Canonicalize(Direction);  //Vidals L-G-L-G-L canonical form
        virtual void Normalize(Direction LR);
        virtual void Orthogonalize(int isite);

        virtual double GetExpectation (int isite, const MPO* o) const;
        virtual double GetExpectation (int isite, const Matrix4RT& Hlocal) const;
        virtual void Apply(int isite,const Matrix4RT& expH,SVCompressorC*);
        virtual void Report(std::ostream&) const;

    protected:
        using dVectorT=MPSSite::dVectorT;
        typedef std::tuple<dVectorT,DiagonalMatrixRT> GLType;

        int GetModSite(int isite) const;
        const DiagonalMatrixRT& GetLambda(int isite) const;
        const         MatrixCT& GetGamma (int isite,int n) const ;
        Matrix4CT GetTransferMatrix() const;
        GLType Orthogonalize(const dVectorT& gamma, const DiagonalMatrixRT& lambda);
        bool TestOrthogonal(int site) ;
        Matrix4CT GetTransferMatrix(const dVectorT& M);

        struct Sites
        {
            Sites(int leftSite,iTEBDStateImp*);
            MPSSite* siteA;
            MPSSite* siteB;
            Bond*    bondA;
            Bond*    bondB;
            dVectorT& MA;
            dVectorT& MB;
            const DiagonalMatrixRT& lambdaA;
            const DiagonalMatrixRT& lambdaB;
        };



    private:
        void InitSitesAndBonds(int D, double epsSV);
};

}
#endif // ITEBDSTATEIMP_H
