#include "TensorNetworksImp/iTEBDStateImp.H"
#include "TensorNetworksImp/Bond.H"
#include "oml/cnumeric.h"
#include <iomanip>

namespace TensorNetworks
{

iTEBDStateImp::iTEBDStateImp(int L,double S, int D,double normEps,TNSLogger* s)
    : MPSImp(L,S,D,DLeft,normEps,s)
{
    InitSitesAndBonds();
}

iTEBDStateImp::~iTEBDStateImp()
{
    itsBonds[0]=0; //Avoid double deletion in optr_vector destructor
    //dtor
}


void iTEBDStateImp::InitSitesAndBonds()
{
    //
    //  Create bond objects
    //
    itsBonds.push_back(0);  //Dummy space holder. We want this array to be 1 based.
    for (int i=1; i<=itsL; i++)
        itsBonds.push_back(new Bond());
    itsBonds[0]=itsBonds[itsL];  //Periodic boundary conditions
    //
    //  Create Sites
    //
    itsSites.push_back(0);  //Dummy space holder. We want this array to be 1 based.
    for (int i=1; i<=itsL-1; i++)
        itsSites.push_back(new MPSSite(PBulk,itsBonds[i-1],itsBonds[i],itsd,itsDmax,itsDmax));
    itsSites.push_back(new MPSSite(PRight,itsBonds[itsL-1],itsBonds[0],itsd,itsDmax,itsDmax));
    //
    //  Tell each bond about its left and right sites.
    //
    for (int i=1; i<=itsL; i++)
        itsBonds[i]->SetSites(itsSites[i],itsSites[GetModSite(i+1)]);
}

void iTEBDStateImp::InitializeWith(State s)
{
    MPSImp::InitializeWith(s);
}



void iTEBDStateImp::NormalizeAndCompress(Direction LR,SVCompressorC* comp)
{
    ForLoop(LR)
        MPSImp::CanonicalizeSite(LR,ia,comp);
}

//void iTEBDStateImp::NormalizeAndCompress(Direction LR,int Dmax,double epsMin);
int iTEBDStateImp::GetModSite(int isite)
{
    int modSite=((isite-1)%itsL)+1;
    assert(modSite>=1);
    assert(modSite<=itsL);
    return modSite;
}

VectorRT&  iTEBDStateImp::GetLambda(int isite)
{
    MPSSite* site=itsSites[GetModSite(isite  )];
    assert(site);
    Bond*    bond=site->itsRightBond;
    assert(bond);
    return bond->itsSingularValues;
}

MatrixCT& iTEBDStateImp::GetGamma (int isite,int n)
{
    assert(n>=0);
    assert(n<itsd);
    MPSSite* site=itsSites[GetModSite(isite  )];
    assert(site);
    return site->itsMs[n];
}

void iTEBDStateImp::Apply(int isite,const Matrix4RT& expH)
{
    MPSSite* siteA=itsSites[GetModSite(isite  )];
    Bond*    bondA=siteA->itsRightBond;
    MPSSite* siteB=itsSites[GetModSite(isite+1)];
    Bond*    bondB=siteB->itsRightBond;
    assert(siteA);
    assert(siteB);
    assert(bondA);
    assert(bondB);
    assert(siteA!=siteB);
    assert(bondA!=bondB);
    MPSSite::dVectorT& MA(siteA->itsMs);
    MPSSite::dVectorT& MB(siteB->itsMs);
    Vector<double> lambdaA=bondA->GetSVs();
    Vector<double> lambdaB=bondB->GetSVs();
    //
    //  New we need to contract   Theta(nA,i1,nB,i3) =
    //                         sB(i1)*MA(mA,i1,i2)*sA(i2)*MB(mB,i2,i3)*sB(i3)
    //                                   |                   |
    //                              expH(mA,nA,              mB,nB)
    //
    Matrix4CT Theta(itsd,itsDmax,itsd,itsDmax,0);
    Fill(Theta.Flatten(),eType(0.0));
    for (int na=0;na<itsd;na++)
    for (int nb=0;nb<itsd;nb++)
        for (int i1=1;i1<=itsDmax;i1++)
            for (int i3=1;i3<=itsDmax;i3++)
            {
                eType t(0.0);
                for (int ma=0;ma<itsd;ma++)
                for (int mb=0;mb<itsd;mb++)
                for (int i2=1;i2<=itsDmax;i2++)
                    t+=lambdaB(i1)*MA[ma](i1,i2)*lambdaA(i2)*MB[mb](i2,i3)*lambdaB(i3)*expH(ma,na,mb,nb);
                Theta(na,i1-1,nb,i3-1)=t;
            }

    //
    //  Now SVD Theta
    //
    MatrixCT ThetaF=Theta.Flatten();
    assert(ThetaF.GetNumRows()==ThetaF.GetNumCols());
//    cout << "Before Compress Dw1 Dw2 A=" << itsDw12.Dw1 << " " << itsDw12.Dw2 << " "<< A << endl;
    auto [U,s,Vdagger]=oml_CSVDecomp(ThetaF);
    cout << std::scientific << std::setprecision(1) << "s=" << s.GetDiagonal() << endl;
    int nai1=1;
    for (int na=0;na<itsd;na++)
        for (int i1=1;i1<=itsDmax;i1++,nai1++)
            for (int i2=1;i2<=itsDmax;i2++)
                MA[na](i1,i2)=U(nai1,i2)/lambdaB(i1);
    int nbi2=1;
    for (int nb=0;nb<itsd;nb++)
        for (int i2=1;i2<=itsDmax;i2++,nbi2++)
            for (int i3=1;i3<=itsDmax;i3++)
                MB[nb](i2,i3)=Vdagger(nbi2,i3)/lambdaB(i3);

   bondA->SetSingularValues(s);

}

void iTEBDStateImp::Report(std::ostream& os) const
{
    MPSImp::Report(os);
}

}
