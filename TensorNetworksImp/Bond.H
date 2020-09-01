#ifndef BOND_H
#define BOND_H

#include "TensorNetworks/Enums.H"
#include "TensorNetworksImp/Typedefs.H"
#include "oml/vector.h"

class MPSSite;

class Bond
{
     typedef TensorNetworks::VectorT  VectorT;
     typedef TensorNetworks::MatrixCT MatrixCT;
   public:
        Bond();
        virtual ~Bond();
        void CloneState(const Bond* b2);
        void SetSites(MPSSite* left, MPSSite* right);

        void SVDTransfer(TensorNetworks::Direction lr,const VectorT& s,const MatrixCT& UV);

        double GetBondEntropy() const {return itsBondEntropy;}
        double GetMinSV      () const {return itsMinSV;}
        int    GetRank       () const {return itsRank;}
        Array<double> GetSVs () const {return itsSingularValues;}
    protected:

    private:
        void SetSingularValues(const VectorT& s);
        //
        //  lr is the normalization direction.  For normalization we need to access the right bond.
        //
        MPSSite* GetSite(TensorNetworks::Direction lr)
        {
            return lr==TensorNetworks::DLeft ? itsRightSite : itsLeft_Site;
        }

        Array<double> itsSingularValues;
        double itsBondEntropy;
        double itsMinSV;
        int    itsRank;
        MPSSite* itsLeft_Site;
        MPSSite* itsRightSite;

};

#endif // BOND_H
