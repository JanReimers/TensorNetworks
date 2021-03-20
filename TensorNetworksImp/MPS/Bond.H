#ifndef BOND_H
#define BOND_H

#include "TensorNetworksImp/Namespace.H"
#include "TensorNetworks/Typedefs.H"
#include "oml/diagonalmatrix.h"


namespace TensorNetworks
{

class Site
{
public:
    virtual ~Site() {};
    virtual void SVDTransfer(Direction lr,const DiagonalMatrixRT& s, const MatrixCT& UV)=0;
    virtual void SVDTransfer(Direction lr,const MatrixCT& UV)=0;
    virtual void TransferQR (Direction lr,const MatrixCT& R)=0;

};

class Bond
{

   public:
        Bond(int D, double epsSV);
        virtual ~Bond();
        void CloneState(const Bond* b2);
        Bond* Clone() const;
        void SetSites(Site* left, Site* right);
        void NewBondDimension(int D);

        void SVDTransfer      (Direction lr,double compessionError,const DiagonalMatrixRT& s,const MatrixCT& UV);
        void TransferQR       (Direction lr,const MatrixCT& R);
        void CanonicalTransfer(Direction lr,double compessionError,const DiagonalMatrixRT& s,const MatrixCT& UV);

        double GetBondEntropy() const {return itsBondEntropy;}
        double GetMinSV      () const {return itsMinSV;}
        double GetMaxDelta   () const {return itsMaxDelta;}
        double GetMaxDelta   (const Bond& cache) const;
        int    GetD          () const {return itsD;}
        int    GetRank       () const {return itsRank;}
        const DiagonalMatrixRT& GetSVs () const {return itsSingularValues;}

        void Report    (std::ostream&) const;

    protected:

    private:
        friend class iTEBDStateImp;
        void SetSingularValues(const DiagonalMatrixRT& s, double compessionError);
        void ClearSingularValues(Direction lr,const MatrixCT& R);
        void UpdateBondEntropy();
        static double GetNorm(const DiagonalMatrixRT&);
        //
        //  lr is the normalization direction.  For normalization we need to access the right bond.
        //
        Site* GetSite(Direction lr)
        {
            return lr==DLeft ? itsRightSite : itsLeft_Site;
        }

        DiagonalMatrixRT itsSingularValues;
        double itsEpsSV; //SVs < itsEpsSV are deemed to be zero when determining rank and entropies.
        double itsBondEntropy;
        double itsMinSV;
        double itsMaxDelta; //Maximum delta lambda
        double itsCompessionError; // sum of reject S_i^2
        int    itsD;
        int    itsRank; //Rank <=D depending on zero SVs
        Site* itsLeft_Site;
        Site* itsRightSite;

};
}
#endif // BOND_H
