#include "StateIterator.H"

StateIterator::StateIterator(int L, int p)
    : itsL(L)
    , itspmax(p-1)
    , itsNextSiteToIncrement(L)
    , itsLinearIndex(0)
    , itsQuantumNumbers(L)
{
    assert(itsL>0);
    assert(itspmax>0);
    Fill(itsQuantumNumbers,0);
}

StateIterator::~StateIterator()
{
    //dtor
}

void StateIterator::Restart()
{
    itsNextSiteToIncrement=itsL;
    itsLinearIndex=0;
    Fill(itsQuantumNumbers,0);
}

//
//  This is the meat of the iterator.
//  The key to understaning this increment is exactly what is the the interpretation of
//  itsNextSiteToIncrement which should be clear from the long name
//
 int StateIterator::operator++(int)
 {
    assert(itsQuantumNumbers(itsNextSiteToIncrement)<=itspmax);
    itsQuantumNumbers(itsNextSiteToIncrement)++;
    while (itsQuantumNumbers(itsNextSiteToIncrement)>itspmax)
    {
         itsNextSiteToIncrement--;
         if (itsNextSiteToIncrement==0)
            return ++itsLinearIndex; //We are done iterating
         itsQuantumNumbers(itsNextSiteToIncrement)++;
    }
    for (int ia=itsNextSiteToIncrement+1;ia<=itsL;ia++)
        itsQuantumNumbers(ia)=0;
    itsNextSiteToIncrement=itsL;

    itsLinearIndex++;
    return itsLinearIndex;
 }

 //
 //  Even this one is non-trivial, we need to know when all
 //  QNs==pmax.  But that requires looping the whole
 //  vector.
 //
 bool StateIterator::end() const
 {
    return itsNextSiteToIncrement==0;
 }

 // Caluclate the linear index from a rank L tensor index stored in state;
int StateIterator::GetIndex(const Vector<int>& state) const
{
    int ret=0,p=itspmax+1;
    for (int ia=1; ia<=itsL; ia++)
        ret=p*ret+state(ia);
    return ret; //0 based of Array type used for amplitudes
}

