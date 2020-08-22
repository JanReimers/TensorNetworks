#ifndef BOND_H
#define BOND_H

#include "TensorNetworks/Enums.H"
#include "TensorNetworksImp/Typedefs.H"
#include "oml/vector.h"

class MatrixProductSite;

class Bond
{
     typedef TensorNetworks::VectorT  VectorT;
     typedef TensorNetworks::MatrixCT MatrixCT;
   public:
        Bond(double eps);
        virtual ~Bond();
        void CloneState(const Bond* b2);
        void SetSites(MatrixProductSite* left, MatrixProductSite* right);

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
        MatrixProductSite* GetSite(TensorNetworks::Direction lr)
        {
            return lr==TensorNetworks::DLeft ? itsRightSite : itsLeft_Site;
        }

        Array<double> itsSingularValues;
        double itsEps; //SVs < eps are considered zero
        double itsBondEntropy;
        double itsMinSV;
        int    itsRank;
        MatrixProductSite* itsLeft_Site;
        MatrixProductSite* itsRightSite;

};

#endif // BOND_H
