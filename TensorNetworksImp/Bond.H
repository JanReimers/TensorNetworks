#ifndef BOND_H
#define BOND_H

#include "TensorNetworks/Enums.H"
#include "TensorNetworksImp/Typedefs.H"
#include "oml/vector.h"


namespace TensorNetworks
{
    class iTEBDStateImp;
    class MPSSite;


class Bond
{
   public:
        Bond();
        virtual ~Bond();
        void CloneState(const Bond* b2);
        void SetSites(MPSSite* left, MPSSite* right);
        void NewBondDimension(int D);

        void SVDTransfer      (Direction lr,const DiagonalMatrixRT& s,const MatrixCT& UV);
        void CanonicalTransfer(Direction lr,const DiagonalMatrixRT& s,const MatrixCT& UV);

        double GetBondEntropy() const {return itsBondEntropy;}
        double GetMinSV      () const {return itsMinSV;}
        int    GetRank       () const {return itsRank;}
        Vector<double> GetSVs () const {return itsSingularValues;}
    protected:

    private:
        friend class iTEBDStateImp;
        void SetSingularValues(const DiagonalMatrixRT& s);
        //
        //  lr is the normalization direction.  For normalization we need to access the right bond.
        //
        MPSSite* GetSite(Direction lr)
        {
            return lr==DLeft ? itsRightSite : itsLeft_Site;
        }

        Vector<double> itsSingularValues;
        double itsBondEntropy;
        double itsMinSV;
        int    itsRank;
        MPSSite* itsLeft_Site;
        MPSSite* itsRightSite;

};
}
#endif // BOND_H
