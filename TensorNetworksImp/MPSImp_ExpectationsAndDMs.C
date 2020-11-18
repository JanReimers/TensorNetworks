#include "TensorNetworksImp/MPSImp.H"
#include "TensorNetworks/MPO.H"
#include "TensorNetworks/TNSLogger.H"
//#include <iostream>
//#include <iomanip>

//using std::cout;
//using std::endl;

namespace TensorNetworks
{

double   MPSImp::GetOverlap(const MPS* Psi2) const
{
    const MPSImp* Psi2Imp=dynamic_cast<const MPSImp*>(Psi2);
    assert(Psi2Imp);

    MatrixCT F(1,1);
    F(1,1)=dcmplx(1.0);
    SiteLoop(ia)
    F=itsSites[ia]->IterateLeft_F(Psi2Imp->itsSites[ia],F);

    assert(F.GetLimits()==MatLimits(1,1));
    double iO=std::imag(F(1,1))/itsL/itsL;
    if (fabs(iO)>1e-10)
        cout << "Warning: MatrixProductState::GetOverlap Imag(O)=" << iO << endl;

    return std::real(F(1,1));
}



double   MPSImp::GetExpectation   (const MPO* o) const
{
    Vector3CT F(1,1,1,1);
    F(1,1,1)=dcmplx(1.0);
    SiteLoop(ia)
        F=itsSites[ia]->IterateLeft_F(o->GetSiteOperator(ia),F);

    double iE=std::imag(F(1,1,1))/itsL/itsL;
    if (fabs(iE)>1e-10)
        cout << "Warning: MatrixProductState::GetExpectation Imag(E)=" << iE << endl;

    return std::real(F(1,1,1));
}

dcmplx   MPSImp::GetExpectationC(const MPO* o) const
{
    Vector3CT F(1,1,1,1);
    F(1,1,1)=dcmplx(1.0);
    SiteLoop(ia)
        F=itsSites[ia]->IterateLeft_F(o->GetSiteOperator(ia),F);

    return F(1,1,1);
}



OneSiteDMs MPSImp::CalculateOneSiteDMs()
{
    assert(Logger); //Make sure we have global logger.
    OneSiteDMs ret(itsL,itsd);
    Normalize(DRight);
    SiteLoop(ia)
    {
        Logger->LogInfo(2,ia,"Calculate ro(mn)");
        ret.Insert(ia,itsSites[ia]->CalculateOneSiteDM());
        NormalizeSite(DLeft,ia);
    }
    return ret;
}

Matrix4CT MPSImp::CalculateTwoSiteDM(int ia,int ib) const
{
    CheckSiteNumber(ia);
    CheckSiteNumber(ib);
#ifdef DEBUG
    for (int is=1; is<ia; is++)
        assert(GetNormStatus(is)=='A');
    for (int is=ib+1; is<=itsL; is++)
        assert(GetNormStatus(is)=='B');
#endif
    Matrix4CT ret(itsd,itsd,itsd,itsd,1);
    ret.Fill(dcmplx(0.0));
    // Start the zipper
    for (int m=0; m<itsd; m++)
        for (int n=0; n<itsd; n++)
        {
            MatrixCT C=itsSites[ia]->InitializeTwoSiteDM(m,n);
            for (int ix=ia+1; ix<ib; ix++)
                C=itsSites[ix]->IterateTwoSiteDM(C);
            C=itsSites[ib]->FinializeTwoSiteDM(C);

            for (int m2=0; m2<itsd; m2++)
                for (int n2=0; n2<itsd; n2++)
                    ret(m+1,m2+1,n+1,n2+1)=C(m2+1,n2+1);
        }
    assert(IsHermitian(ret.Flatten(),1e-14));
    return ret;
}


TwoSiteDMs MPSImp::CalculateTwoSiteDMs()
{
    Normalize(DRight);
    TwoSiteDMs ret(itsL,itsd);
    SiteLoop(ia)
    for (int ib=ia+1; ib<=itsL; ib++)
    {
        Matrix4CT ro=CalculateTwoSiteDM(ia,ib);
        ret.Insert(ia,ib,ro);
        NormalizeSite(DLeft,ia);
    }
    // Normalize the last to keep things tidy
    NormalizeSite(DLeft,itsL);
    return ret;
}

}; // namespace
