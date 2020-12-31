#include "StateIterator.H"

namespace TensorNetworks
{

StateIterator::StateIterator(int L, int d)
    : itsL(L)
    , itsdmax(d-1)
    , itsNextSiteToIncrement(L)
    , itsLinearIndex(1)
    , itsQuantumNumbers(L)
{
    assert(itsL>0);
    assert(itsdmax>0);
    Fill(itsQuantumNumbers,0);
}

StateIterator::~StateIterator()
{
    //dtor
}

void StateIterator::Restart()
{
    itsNextSiteToIncrement=itsL;
    itsLinearIndex=1;
    Fill(itsQuantumNumbers,0);
}

//
//  This is the meat of the iterator.
//  The key to understaning this increment is exactly what is the the interpretation of
//  itsNextSiteToIncrement which should be clear from the long name
//
 int StateIterator::operator++(int)
 {
    assert(itsQuantumNumbers(itsNextSiteToIncrement)<=itsdmax);
    itsQuantumNumbers(itsNextSiteToIncrement)++;
    while (itsQuantumNumbers(itsNextSiteToIncrement)>itsdmax)
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
    int ret=0,p=itsdmax+1;
    for (int ia=1; ia<=itsL; ia++)
        ret=p*ret+state(ia);
    return ret+1; //1 based of Vector type used for amplitudes
}

} // namespace

