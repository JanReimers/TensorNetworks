#include "MatrixProductOperator.H"
#include "Hamiltonian.H"
#include "Matrix6.H"
#include "oml/vector_io.h"
#include <iostream>

using std::cout;
using std::endl;

/*MatrixProductOperator::MatrixProductOperator(const Hamiltonian* H, int L, int S2, int D)
    : itsL(L)
    , itsD(D)
    , itsp(S2+1)
    , itsOperator(H)
{
    assert(itsOperator);
    //
    //  Load W matrices for the left edge,bulk and right edge
    //
    MPOSite* left =new MPOSite(itsOperator,itsp,1,   itsD);
    MPOSite* bulk =new MPOSite(itsOperator,itsp,itsD,itsD);
    MPOSite* right=new MPOSite(itsOperator,itsp,itsD,1   );
    itsSites.push_back(left );     //Left edge
    itsSites.push_back(bulk );     //bulk
    itsSites.push_back(right);     //Right edge


}*/

MatrixProductOperator::MatrixProductOperator(const Operator* O, int L, int S2, int D)
    : itsL(L)
    , itsD(D)
    , itsp(S2+1)
    , itsOperator(O)
{
    assert(itsOperator);
    //
    //  Load W matrices for the left edge,bulk and right edge
    //
    MPOSite* left =new MPOSite(itsOperator,itsp,1,   itsD);
    MPOSite* bulk =new MPOSite(itsOperator,itsp,itsD,itsD);
    MPOSite* right=new MPOSite(itsOperator,itsp,itsD,1   );
    itsSites.push_back(left );     //Left edge
    itsSites.push_back(bulk );     //bulk
    itsSites.push_back(right);     //Right edge


}

MatrixProductOperator::~MatrixProductOperator()
{
    //dtor
}

 Operator::Position MatrixProductOperator::GetPosition(int isite) const
        {
            return isite==0 ? Operator::Left :
                (isite==itsL-1 ? Operator::Right : Operator::Bulk);
            }

