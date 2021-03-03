#ifndef OPERATORBOND_H
#define OPERATORBOND_H

#include "TensorNetworksImp/Namespace.H"
#include "TensorNetworks/Typedefs.H"
#include "oml/diagonalmatrix.h"


namespace TensorNetworks
{

class OperatorBond
{

   public:
        OperatorBond(int D);
        virtual ~OperatorBond();
        void SetSites(SiteOperator* left, SiteOperator* right);
        void NewBondDimension(int Dw);

        void GaugeTransfer    (Direction lr,double compessionError,const DiagonalMatrixRT& s,const MatrixRT& UV);
        void GaugeTransfer    (Direction lr,const MatrixRT& R);
        void CanonicalTransfer(Direction lr,double compessionError,const DiagonalMatrixRT& s,const MatrixRT& UV);

        double GetBondEntropy() const {return itsBondEntropy;}
        double GetMinSV      () const {return itsMinSV;}
        double GetMaxDelta   () const {return itsMaxDelta;}
//        double GetMaxDelta   (const Bond& cache) const;
        int    GetDw         () const {return itsD;}
        int    GetRank       () const {return itsRank;}
        const DiagonalMatrixRT& GetSVs () const {return itsSingularValues;}

        void Report    (std::ostream&) const;

    protected:

    private:
        void SetSingularValues(const DiagonalMatrixRT& s, double compessionError);
        void ClearSingularValues(Direction lr,const MatrixRT& R);
        void UpdateBondEntropy();
        //static double GetNorm(const DiagonalMatrixRT&);
        //
        //  lr is the normalization direction.  For normalization we need to access the right bond.
        //
        SiteOperatorImp* GetSite(Direction lr)
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
        SiteOperatorImp* itsLeft_Site;
        SiteOperatorImp* itsRightSite;

};
}
#endif // OPERATORBOND_H
