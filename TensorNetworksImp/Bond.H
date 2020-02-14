#ifndef BOND_H
#define BOND_H

#include "oml/vector.h"

class Bond
{
    public:
        Bond(double eps);
        virtual ~Bond();

        void SetSingularValues(const Vector<double> s);

        double GetBondEntropy() const {return itsBondEntropy;}
        double GetMinSV      () const {return itsMinSV;}
        int    GetRank       () const {return itsRank;}
        Array<double> GetSVs () const {return itsSingularValues;}
    protected:

    private:
        Array<double> itsSingularValues;
        double itsEps; //SVs < eps are considered zero
        double itsBondEntropy;
        double itsMinSV;
        int    itsRank;
};

#endif // BOND_H
