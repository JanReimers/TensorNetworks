#ifndef BOND_H
#define BOND_H

#include "TensorNetworksImp/Namespace.H"
#include "TensorNetworks/Typedefs.H"
#include "oml/diagonalmatrix.h"


namespace TensorNetworks
{

class Bond
{

   public:
        Bond(double epsSV);
        virtual ~Bond();
        void CloneState(const Bond* b2);
        void SetSites(MPSSite* left, MPSSite* right);
        void NewBondDimension(int D);

        void SVDTransfer      (Direction lr,double integratedS2,const DiagonalMatrixRT& s,const MatrixCT& UV);
        void CanonicalTransfer(Direction lr,double integratedS2,const DiagonalMatrixRT& s,const MatrixCT& UV);

        double GetBondEntropy() const {return itsBondEntropy;}
        double GetMinSV      () const {return itsMinSV;}
        int    GetD          () const {return itsD;}
        int    GetRank       () const {return itsRank;}
        const DiagonalMatrixRT& GetSVs () const {return itsSingularValues;}

        void Report    (std::ostream&) const;

    protected:

    private:
        friend class iTEBDStateImp;
        void SetSingularValues(const DiagonalMatrixRT& s, double integratedS2);
        void UpdateBondEntropy();
        //
        //  lr is the normalization direction.  For normalization we need to access the right bond.
        //
        MPSSite* GetSite(Direction lr)
        {
            return lr==DLeft ? itsRightSite : itsLeft_Site;
        }

        DiagonalMatrixRT itsSingularValues;
        double itsEpsSV; //SVs < itsEpsSV are deemed to be zero when determining rank and entropies.
        double itsBondEntropy;
        double itsMinSV;
        double itsIntegratedS2; // sum of reject S_i^2
        int    itsD;
        int    itsRank; //Rank <=D depending on zero SVs
        MPSSite* itsLeft_Site;
        MPSSite* itsRightSite;

};
}
#endif // BOND_H
