#include "MatrixProductOperator.H"
#include "Hamiltonian.H"
#include <iostream>

using std::cout;
using std::endl;

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

double MatrixProductOperator::GetHamiltonianExpectation(const MatrixProductState *mps) const
{
    assert(mps);
    int Dw=itsHamiltonian->GetDw();
    MatrixT E(1,Dw);
    Fill(E,std::complex<double>(1.0));
    // E is now a unit row vector
    for (int isite=0;isite<itsL;isite++)
    { //loop over sites
        Hamiltonian::Position lbr =
        isite==0 ? Hamiltonian::Left :
            isite==itsL-1 ? Hamiltonian::Right : Hamiltonian::Bulk;
        E*=mps->GetE(isite,itsSites[lbr]);
//        cout << "MPO Elimits=" << E.GetLimits() << " lbr=" << lbr << endl;
    }
    // at this point E is 1xDw so we need to dot it with a unit vector
    MatrixT UnitCol(Dw,1);
    Fill(UnitCol,std::complex<double>(1.0));
    E*=UnitCol;
//    cout << "E =" << E << endl;
    assert(E.GetNumRows()==1);
    assert(E.GetNumCols()==1);
//    cout << std::imag(E(1,1)) << endl;
    assert(std::imag(E(1,1))==0.0);
    return std::real(E(1,1));
}
