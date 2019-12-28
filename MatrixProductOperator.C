#include "MatrixProductOperator.H"
#include "Hamiltonian.H"

MatrixProductOperator::MatrixProductOperator(const Hamiltonian* H, int L, int S2, int D)
    : itsL(L)
    , itsD(D)
    , itsp(S2+1)
    , itsHamiltonian(H)
{
    assert(itsHamiltonian);
    //
    //  Load W matrices for the left edge,bulk and right edge
    //
    MPOSite* left =new MPOSite(itsHamiltonian,itsp,1,   itsD);
    MPOSite* bulk =new MPOSite(itsHamiltonian,itsp,itsD,itsD);
    MPOSite* right=new MPOSite(itsHamiltonian,itsp,itsD,1   );
    itsSites.push_back(left );     //Left edge
    itsSites.push_back(bulk );     //bulk
    itsSites.push_back(right);     //Right edge


}

MatrixProductOperator::~MatrixProductOperator()
{
    //dtor
}
