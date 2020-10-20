#ifndef STATEITERATOR_H
#define STATEITERATOR_H

#include "oml/vector.h"
#include <iostream>

namespace TensorNetworks
{

class StateIterator
{
    public:
        StateIterator(int L, int d);
        virtual ~StateIterator();

        void               Restart          ();
        int                operator++       (int); //Return linear index
        bool               end              () const;
        const Vector<int>& GetQuantumNumbers() const {return itsQuantumNumbers;}

        int                GetIndex         (const Vector<int>& QNs) const;
        int                GetLinearIndex   () const {return itsLinearIndex;}
        friend std::ostream& operator<<(std::ostream& os,const StateIterator& i)
        {
            return os << i.itsNextSiteToIncrement << " " << i.itsLinearIndex << " " << i.itsQuantumNumbers << std::endl;
        }

    protected:

    private:
        int itsL; //Number of basis sites/spins
        int itsdmax; //max QN on each site/spin
        mutable int         itsNextSiteToIncrement;
        mutable int         itsLinearIndex;
        mutable Vector<int> itsQuantumNumbers; //Vector of quantum numbers |n_1,n_2...n_L>
};

}
#endif // STATEITERATOR_H
