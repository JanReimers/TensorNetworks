#include "MatrixProductOperator.H"
#include "Hamiltonian.H"
#include "Matrix6.H"
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

 Hamiltonian::Position MatrixProductOperator::GetPosition(int isite) const
        {
            return isite==0 ? Hamiltonian::Left :
                (isite==itsL-1 ? Hamiltonian::Right : Hamiltonian::Bulk);
            }

MatrixProductOperator::Matrix6T MatrixProductOperator::GetHeff(const MatrixProductState *mps,int isite) const
{
    Matrix6T NLeft =GetNLeft (mps,isite);
    Matrix6T NRight=GetNRight(mps,isite);
    cout << "NLeft " << NLeft  << endl;
    cout << "NRight" << NRight << endl;
//    assert(NLeft .GetNumRows()==1);
 //   assert(NRight.GetNumCols()==1);
//    Subscriptor s=mps->GetSuperMatrixSubscriptor(isite);

    const MPOSite* mops=itsSites[isite];
    ipairT Ds=mps->GetDs(isite);
    int D1=Ds.first;
    int D2=Ds.first;
    int p=mps->Getp();

    Matrix6<eType> Heff(p,D1,p,D2);

    for (int m=0; m<p; m++)
        for (int i1=1; i1<=D1; i1++)
            for (int j1=1; j1<=D2; j1++)
            {
                for (int n=0; n<p; n++)
                {
                    MatrixT W=mops->GetW(m,n);
                    for (int i2=1; i2<=D1; i2++)
                        for (int j2=1; j2<=D2; j2++)
                        {
                            eType temp(0.0);
                            for (int w1=1; w1<=W.GetNumRows(); w1++)
                                for (int w2=1; w2<=W.GetNumCols(); w2++)
                                {
                                    temp+=NLeft(1,1,1,w1,i1,j1)*W(w1,w2)*NRight(w2,i2,j2,1,1,1);
                                }

                            Heff(m,i1,i2,n,j1,j2)=temp;
                        }
                }
            }
    return Heff;
}

MatrixProductOperator::Matrix6T MatrixProductOperator::GetNLeft(const MatrixProductState *mps,int isite) const
{
    Matrix6T NLeft(1,1);
    NLeft.Fill(std::complex<double>(1.0));
    for (int ia=0;ia<isite;ia++)
    { //loop over sites
        Hamiltonian::Position lbr = GetPosition(ia);
        NLeft*=mps->GetE(ia,itsSites[lbr]);
//        cout << "MPO Elimits=" << E.GetLimits() << " lbr=" << lbr << endl;
    }
    return NLeft;
}

MatrixProductOperator::Matrix6T MatrixProductOperator::GetNRight(const MatrixProductState *mps,int isite) const
{
    Matrix6T NRight(1,1);
    NRight.Fill(std::complex<double>(1.0));
    for (int ia=itsL-1;ia>=isite;ia--)
    { //loop over sites
        Hamiltonian::Position lbr = GetPosition(ia);
        Matrix6T temp=NRight;
        Matrix6T E=mps->GetE(ia,itsSites[lbr]);

        cout << "NRight=" <<  NRight << endl;
        cout << "E=" <<  E << endl;
//        Matrix6T Et=E*temp;
//        cout << "Et=" <<  Et << endl;
        NRight.ClearLimits();
        NRight=E*=temp;
//        cout << "MPO Elimits=" << E.GetLimits() << " lbr=" << lbr << endl;
    }
    return NRight;
}

double MatrixProductOperator::GetHamiltonianExpectation(const MatrixProductState *mps) const
{
    assert(mps);
    int Dw=itsHamiltonian->GetDw();
    Matrix6T E(1,1);
    E.Fill(std::complex<double>(1.0));

    // E is now a unit row vector
    for (int isite=0;isite<itsL;isite++)
    { //loop over sites
        Hamiltonian::Position lbr = GetPosition(isite);
        cout << "E=" << E << endl;
        cout << "GetE="<< mps->GetE(isite,itsSites[lbr]) << endl;
        E*=mps->GetE(isite,itsSites[lbr]);
//        cout << "MPO Elimits=" << E.GetLimits() << " lbr=" << lbr << endl;
    }
    // at this point E is 1xDw so we need to dot it with a unit vector
    cout << "E =" << E << endl;
//    assert(E.GetNumRows()==1);
//    assert(E.GetNumCols()==1);
//    cout << std::imag(E(1,1)) << endl;
    assert(std::imag(E(1,1,1,1,1,1))==0.0);
    return std::real(E(1,1,1,1,1,1));
}
