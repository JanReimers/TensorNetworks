#include "TensorNetworksImp/MatrixProductStateImp.H"
#include "TensorNetworksImp/Bond.H"
#include "TensorNetworks/Hamiltonian.H"
#include "TensorNetworks/LRPSupervisor.H"
#include "Functions/Mesh/PlotableMesh.H"
#include "Plotting/CurveUnits.H"
#include "Plotting/Factory.H"
#include "Plotting/MultiGraph.H"
#include "Misc/Dimension.H"
#include "Misc/NamedUnits.H"
#include "oml/cnumeric.h"
#include "oml/vector_io.h"
#include <iostream>
#include <iomanip>

using std::cout;
using std::endl;

/--------------------------------------------------------------------------------------
//
//    Overlap and expectation contractions
//
double MatrixProductStateImp::GetOverlap() const
{
    cSIter i=itsSites.begin();
    MatrixCT E=i->GetE();
    i++;
    for (;i!=itsSites.end();i++)
        E=i->GetELeft(E);
    assert(E.GetNumRows()==1);
    assert(E.GetNumCols()==1);
    double iE=fabs(std::imag(E(1,1)));
    if (iE>1e-12)
        cout << "Warning MatrixProductState::GetOverlap imag(E)=" << iE << endl;
    return std::real(E(1,1));
}


MatrixProductStateImp::Matrix6T MatrixProductStateImp::GetEOLeft(const Operator *o,int isite) const
{
    Matrix6T NLeft(1,1);
    NLeft.Fill(std::complex<double>(1.0));
    for (int ia=0;ia<isite;ia++)
    { //loop over sites
        NLeft*=itsSites[ia]->GetEO(o->GetSiteOperator(ia));
//        cout << "o Elimits=" << E.GetLimits() << " lbr=" << lbr << endl;
    }
    return NLeft;
}

MatrixProductStateImp::MatrixCT MatrixProductStateImp::GetNeff(int isite) const
 {
    return itsSites[isite]->GetNeff(GetMLeft(isite),GetMRight(isite));
 }
//--------------------------------------------------------------------------------------
//
//  Allows unit test classes inside.
//
MatrixProductStateImp::MatrixCT MatrixProductStateImp::GetMLeft(int isite) const
{
   assert(isite<itsL);
    MatrixCT Eleft;
    if (isite==0)
    {
        Eleft.SetLimits(1,1);
        Fill(Eleft,std::complex<double>(1.0));
    }
    else
    {
        Eleft=itsSites[0]->GetE();
    }
    //
    //  Zip from left to right up to isite
    //
//    cout << "ELeft(0)=" << Eleft << endl;
    for (int ia=1;ia<isite;ia++)
    {
            Eleft=itsSites[ia]->GetELeft(Eleft);
 //       cout << "ELeft(" << ia << ")=" << Eleft << endl;
            }
    return Eleft;
}


MatrixProductStateImp::MatrixCT MatrixProductStateImp::GetMRight(int isite) const
{
    MatrixCT Eright;
    if (isite==itsL-1)
    {
        Eright.SetLimits(1,1);
        Fill(Eright,std::complex<double>(1.0));
    }
    else
    {
        Eright=itsSites[itsL-1]->GetE();
    }
    // Zip right to left
    for (int ia=itsL-2;ia>=isite+1;ia--)
        Eright=itsSites[ia]->GetERight(Eright);

    return Eright;
}

MatrixProductStateImp::Matrix6T MatrixProductStateImp::GetHeff(const Hamiltonian *h,int isite) const
{
    Matrix6T NLeft =GetEOLeft (h,isite);
    Matrix6T NRight=GetEORight(h,isite);
//    cout << "NLeft " << NLeft  << endl;
//    cout << "NRight" << NRight << endl;
//    assert(NLeft .GetNumRows()==1);
 //   assert(NRight.GetNumCols()==1);
    return itsSites[isite]->GetHeff(h->GetSiteOperator(isite),NLeft,NRight);
}

bool MatrixProductStateImp::CheckNormalized(int isite,double eps) const
{
    MatrixCT Neff=GetNeff(isite);
    int N=Neff.GetNumRows();
    MatrixCT I(N,N);
    Unit(I);
    double error=Max(abs(Neff-I));
    if (error>1e-12)
        cout << "Warning: Normalization site=" << isite << "  Neff-I error " << error << endl;
    return error<eps;
}
double MatrixProductStateImp::GetExpectation(const Operator* o) const
{
    assert(o);
    Matrix6T E(1,1);
    E.Fill(std::complex<double>(1.0));

    // E is now a unit row vector
    for (int isite=0;isite<itsL;isite++)
    { //loop over sites
//        Matrix6T temp=mps->GetEO(isite,itsSites[lbr]);
        E*=itsSites[isite]->GetEO(o->GetSiteOperator(isite));
//        cout << "E[" << isite << "]=" << endl;
//        E.Dump(cout);
//        cout << "o Elimits=" << E.GetLimits() << " lbr=" << lbr << endl;
    }
    // at this point E is 1xDw so we need to dot it with a unit vector
 //   Matrix6T Unit(itsp,1);
   // Unit.Fill(std::complex<double>(1.0));
   // E*=Unit;

//    cout << "E =" << E << endl;
//    assert(E.GetNumRows()==1);
//    assert(E.GetNumCols()==1);
    double iE=std::imag(E(1,1,1,1,1,1));
    if (fabs(iE)>1e-10)
        cout << "Warning: MatrixProduct::GetExpectation Imag(E)=" << std::imag(E(1,1,1,1,1,1)) << endl;

    return std::real(E(1,1,1,1,1,1));
}



MatrixProductStateImp::Matrix6T MatrixProductStateImp::GetEORight(const Operator *o,int isite) const
{
    Matrix6T NRight(1,1);
    NRight.Fill(std::complex<double>(1.0));
    for (int ia=itsL-1;ia>isite;ia--)
    { //loop over sites
        Matrix6T temp=NRight;
        Matrix6T E=itsSites[ia]->GetEO(o->GetSiteOperator(ia));

//        cout << "NRight=" <<  NRight << endl;
//        cout << "E=" <<  E << endl;
//        Matrix6T Et=E*temp;
//        cout << "Et=" <<  Et << endl;
        NRight.ClearLimits();
        NRight=E*=temp;
//        cout << "o Elimits=" << E.GetLimits() << " lbr=" << lbr << endl;
    }
    return NRight;
}


