#include "MatrixProductOperator.H"

MatrixProductOperator::MatrixProductOperator(int L, int S2, int D)
    : itsL(L)
    , itsD(D)
    , itsp(S2+1)
{
    itsSites.push_back(new MPOSite(itsp,1,itsD));
    for (int i=1;i<itsL-1;i++)
        itsSites.push_back(new MPOSite(itsp,itsD,itsD));
    itsSites.push_back(new MPOSite(itsp,itsD,1));

}

MatrixProductOperator::~MatrixProductOperator()
{
    //dtor
}
