#include "TensorNetworksImp/MPS/MPSSite.H"
#include "TensorNetworksImp/MPS/Bond.H"
#include "Operators/OperatorValuedMatrix.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/Dw12.H"
#include "Containers/Matrix6.H"
#include "Containers/Matrix4.H"
#include "oml/cnumeric.h"
#include "oml/random.h"
#include "oml/diagonalmatrix.h"
#include <iostream>

using std::cout;
using std::endl;

namespace TensorNetworks
{

MatrixCT  MPSSite::ReshapeBeforeSVD(Direction lr)
{
    MatrixCT A;
    switch (lr)
    {
    case DLeft:
    {
        A.SetLimits(itsd*itsD1,itsD2);
        int i2_1=1;
        for (int in=0; in<itsd; in++)
            for (int i1=1; i1<=itsD1; i1++,i2_1++)
                for (int i2=1; i2<=itsD2; i2++)
                    A(i2_1,i2)=itsMs[in](i1,i2);
        break;
    }
    case DRight:
    {
        A.SetLimits(itsD1,itsd*itsD2);
        int i2_2=1;
        for (int in=0; in<itsd; in++)
            for (int i2=1; i2<=itsD2; i2++,i2_2++)
                for (int i1=1; i1<=itsD1; i1++)
                    A(i1,i2_2)=itsMs[in](i1,i2);
        break;
    }
    }
    return A;
}



void  MPSSite::ReshapeAfter_SVD(Direction lr,const MatrixCT& UV)
{
    switch (lr)
    {
    case DLeft:
    {
        //  If U has less columns than the As then we need to reshape the whole site.
        //  Typically this will happen at the edges of the lattice.
        //
        if (UV.GetNumCols()<itsD2) NewBondDimensions(itsD1,UV.GetNumCols());//This throws away the old data
        int i2_1=1;
        for (int in=0; in<itsd; in++)
            for (int i1=1; i1<=itsD1; i1++,i2_1++)
                for (int i2=1; i2<=itsD2; i2++)
                    itsMs[in](i1,i2)=UV(i2_1,i2);
        break;
    }
    case DRight:
    {
        //  If Vdagger has less row than the As then we need to reshape the whole site.
        //  Typically this will happen at the edges of the lattice.
        //
        if (UV.GetNumRows()<itsD1) NewBondDimensions(UV.GetNumRows(),itsD2,false);//This throws away the old data
        int i2_2=1;
        for (int in=0; in<itsd; in++)
            for (int i2=1; i2<=itsD2; i2++,i2_2++)
                for (int i1=1; i1<=itsD1; i1++)
                    itsMs[in](i1,i2)=UV(i1,i2_2);
        break;
    }
    }
}

MatrixCT MPSSite::GetNorm(Direction lr) const
{
    MatrixCT ret;
    switch(lr)
    {
    case DLeft:
    {
        ret.SetLimits(itsD2,itsD2);
        Fill(ret,std::complex<double>(0.0));
        //
        //  Sum_ip A^t(id) * A(id)
        //
        for (cdIterT id=itsMs.begin(); id!=itsMs.end(); id++)
            ret+=conj(Transpose((*id)))*(*id);
        break;
    }
    case DRight:
    {
        ret.SetLimits(itsD1,itsD1);
        Fill(ret,std::complex<double>(0.0));
        //
        //  Sum_ip B(id)*B^t(id)
        //
        for (cdIterT id=itsMs.begin(); id!=itsMs.end(); id++)
            ret+=(*id)*conj(Transpose((*id)));
        break;
    }
    }
    return ret;
}

MatrixCT MPSSite::GetCanonicalNorm(Direction lr) const
{
    MatrixCT ret;
    switch(lr)
    {
    case DLeft:
    {
        ret.SetLimits(itsD2,itsD2);
        Fill(ret,std::complex<double>(0.0));
        if (itsLeft_Bond)
        {
            const DiagonalMatrixRT& lambda=itsLeft_Bond->GetSVs();
            //
            //  Sum_ip A^t(id) * gamma^2 * A(id)
            //
            for (cdIterT id=itsMs.begin(); id!=itsMs.end(); id++)
                ret+=conj(Transpose((*id)))*lambda*lambda*(*id);
        }

        break;
    }
    case DRight:
    {
        ret.SetLimits(itsD1,itsD1);
        Fill(ret,std::complex<double>(0.0));
        if (itsRightBond)
        {
            const DiagonalMatrixRT& lambda=itsRightBond->GetSVs();
            //
            //  Sum_ip B(id) * gamma^2 * B^t(id)
            //
            for (cdIterT id=itsMs.begin(); id!=itsMs.end(); id++)
            {
                ret+=(*id)*lambda*lambda*conj(Transpose((*id)));
            }
        }
        break;
    }
    }
    return ret;
}

Matrix6CT MPSSite::
GetHeff(const SiteOperator* mops,const Vector3CT& L,const Vector3CT& R) const
{
    assert(mops);
    Matrix6<dcmplx> Heff(itsd,itsD1,itsD2,itsd,itsD1,itsD2);
    Matrix6<dcmplx>::Subscriptor SHeff(Heff);
    const Dw12& Dws=mops->GetDw12();
    const MatrixOR& WOvM=mops->GetW();
    for (int m=0; m<itsd; m++)
        for (int n=0; n<itsd; n++)
        {
            assert(WOvM.GetNumRows()==Dws.Dw1);
            Vector3CT WR(Dws.Dw1,itsD2,itsD2,1);
            #pragma omp parallel for collapse(2)
            for (int i2=1; i2<=itsD2; i2++)
                for (int j2=1; j2<=itsD2; j2++)
                {
                    VectorCT WR(Dws.Dw1,FillType::Zero);
                    VectorCT::Subscriptor sWR(WR);
                    for (int w1=1; w1<=Dws.Dw1; w1++)
                    {
                        for (int w2=1; w2<=Dws.w2_last(w1); w2++)
                            if (WOvM(w1-1,w2-1)(m,n)!=0.0)
                                WR(w1)+=WOvM(w1-1,w2-1)(m,n)*R(w2,i2,j2);
                    }
                    for (int i1=1; i1<=itsD1; i1++)
                        for (int j1=1; j1<=itsD1; j1++)
                        {
                            dcmplx LWR(0.0);
                            for (int w1=1; w1<=Dws.Dw1; w1++)
                                LWR+=L(w1,i1,j1)*sWR(w1);
                            SHeff(m,i1,i2,n,j1,j2)=LWR;
                        }
                }

        }

    return Heff;

}

dcmplx MPSSite::
ContractWR(int w1, int i2, int j2,const MatrixRT& W, int Dw2,const Vector3CT& R) const
{
    dcmplx WR(0.0);
    for (int w2=1; w2<=Dw2; w2++)
        if (W(w1,w2)!=0.0)
            WR+=W(w1,w2)*R(w2,i2,j2);
    return WR;
}

MatrixCT MPSSite::IterateLeft_F(const MPSSite* Psi2, const MatrixCT& Fam1,bool cache) const
{
//    cout << "IterateLeft_F D1,D2,DwD1,DwD2,Fam=" << itsAs[0].GetLimits() << " " << Psi2->itsAs[0].GetLimits() << " " << Fam1.GetLimits() << endl;
    assert(Fam1.GetNumRows()==      itsD1);
    assert(Fam1.GetNumCols()==Psi2->itsD1);
    MatrixCT F(itsD2,Psi2->itsD2);
    Fill(F,dcmplx(0.0));
    for (int m=0; m<itsd; m++)
        for (int i2=1; i2<=itsD2; i2++)
            for (int j1=1; j1<=Psi2->itsD1; j1++)
            {
                dcmplx FM(0);
                for (int i1=1; i1<=itsD1; i1++)
                    FM+=Fam1(i1,j1)*conj(itsMs[m](i1,i2));
                for (int j2=1; j2<=Psi2->itsD2; j2++)
                    F(i2,j2)+=FM*Psi2->itsMs[m](j1,j2);
            }
    if (cache) itsLeft_Cache=F;
//    cout << "Lcache=" << itsLeft_Cache.GetLimits() << endl;
    return F;
}

MatrixCT MPSSite::IterateRightF(const MPSSite* Psi2, const MatrixCT& Fap1,bool cache) const
{
//    cout << "IterateRightF D1,D2,DwD1,DwD2,Fap=" << itsAs[0].GetLimits() << " " << Psi2->itsAs[0].GetLimits() << " " << Fap1.GetLimits() << endl;
    MatrixCT F(itsD1,Psi2->itsD1);
    Fill(F,dcmplx(0.0));
//    cout << "F=" << F.GetLimits() << " Fap1=" << Fap1.GetLimits() << endl;
    assert(Fap1.GetNumRows()>=      itsD2);
    assert(Fap1.GetNumCols()==Psi2->itsD2);

    for (int m=0; m<itsd; m++)
        for (int i1=1; i1<=itsD1; i1++)
            for (int j2=1; j2<=Psi2->itsD2; j2++)
            {
                dcmplx FM(0);
                for (int i2=1; i2<=itsD2; i2++)
                    FM+=Fap1(i2,j2)*conj(itsMs[m](i1,i2));
                for (int j1=1; j1<=Psi2->itsD1; j1++)
                    F(i1,j1)+=FM*Psi2->itsMs[m](j1,j2); //Not Optimized
            }
    if (cache) itsRightCache=F;
//    cout << "Rcache=" << itsRightCache.GetLimits() << endl;

    return F;
}

MatrixCT MPSSite::IterateF(Direction lr,const MatrixCT& Mold) const
{
//    cout << "IterateF D1,D2 Mold=" << itsD1 << "," << itsD2 << " " << Mold.GetLimits() << endl;
    MatrixCT M;
    if (lr==DLeft)
    {
        assert(Mold.GetNumRows()>=itsD1);
        assert(Mold.GetNumCols()>=itsD1);
        if (Mold.GetNumRows()!=itsD1 || Mold.GetNumCols()!=itsD1)
        {
            cout << "D1,D2,Mold=" << itsD1 << " " << itsD2 << " " << Mold.GetLimits() << endl;
        }
        M.SetLimits(itsD2,itsD2);
        Fill(M,dcmplx(0.0));
//    cout << "F=" << F.GetLimits() << " Fap1=" << Fap1.GetLimits() << endl;
        for (int m=0; m<itsd; m++)
            for (int i2=1; i2<=itsD2; i2++)
                for (int j2=1; j2<=itsD2; j2++)
                    for (int i1=1; i1<=itsD1; i1++)
                        for (int j1=1; j1<=itsD1; j1++)
                            M(i2,j2)+=Mold(i1,j1)*conj(itsMs[m](i1,i2))*itsMs[m](j1,j2); //Not Optimized

    }
    else
    {
        assert(lr==DRight);
        assert(Mold.GetNumRows()==itsD2);
        assert(Mold.GetNumCols()==itsD2);
        M.SetLimits(itsD1,itsD1);
        Fill(M,dcmplx(0.0));
        for (int m=0; m<itsd; m++)
            for (int i1=1; i1<=itsD1; i1++)
                for (int j1=1; j1<=itsD1; j1++)
                    for (int i2=1; i2<=itsD2; i2++)
                        for (int j2=1; j2<=itsD2; j2++)
                            M(i1,j1)+=Mold(i2,j2)*conj(itsMs[m](i1,i2))*itsMs[m](j1,j2); //Not Optimized
    }
    return M;
}

Vector3CT MPSSite::IterateLeft_F(const SiteOperator* so, const Vector3CT& Fam1,bool cache)
{
    int Dw2=so->GetDw12().Dw2;
    Vector3CT F(Dw2,itsD2,itsD2,1);
    Vector3CT::Subscriptor sF(F);
    for (int w2=1; w2<=Dw2; w2++)
        //#pragma omp parallel for collapse(2)
        for (int i2=1; i2<=itsD2; i2++)
            for (int j2=1; j2<=itsD2; j2++)
                sF(w2,i2,j2)=ContractAWFA(w2,i2,j2,so,Fam1);
    if (cache) itsHLeft_Cache=F;
    return F;
}

dcmplx MPSSite::ContractAWFA(int w2, int i2, int j2, const SiteOperator* so, const Vector3CT& Fam1) const
{
    dcmplx awfa(0.0);
    for (int m=0; m<itsd; m++)
        for (int i1=1; i1<=itsD1; i1++)
            awfa+=conj(itsMs[m](i1,i2))*ContractWFA(m,w2,i1,j2,so,Fam1);

    return awfa;
}

dcmplx MPSSite::ContractWFA(int m, int w2, int i1, int j2, const SiteOperator* so, const Vector3CT& Fam1) const
{
    const Dw12& Dws1=so->GetDw12();
    dcmplx wfa(0.0);
    const MatrixOR& WOvM=so->GetW();
    for (int n=0; n<itsd; n++)
    {
        assert(WOvM.GetNumRows()==Dws1.Dw1);
        for (int w1=1; w1<Dws1.w1_first(w2); w1++)
        {
            if (WOvM(w1-1,w2-1)(m,n)!=0.0)
            {
                cout << "W"<< m << n <<"(" << w1 << "," << w2 << ")=" << WOvM(w1-1,w2-1) << endl;
                cout << "Dws1.w1_first=" << Dws1.w1_first << endl;
            }
            assert(WOvM(w1-1,w2-1)(m,n)==0.0);
        }
        for (int w1=Dws1.w1_first(w2); w1<=Dws1.Dw1; w1++)
            if (WOvM(w1-1,w2-1)(m,n)!=0.0)
            {
//                assert(fabs(Wmn(w1,w2))>0.0);
                wfa+=WOvM(w1-1,w2-1)(m,n)*ContractFA(n,w1,i1,j2,Fam1);
            }
    }
    return wfa;
}

dcmplx MPSSite::ContractFA(int n, int w1, int i1, int j2, const Vector3CT& Fam1) const
{
    dcmplx fa(0.0);
    for (int j1=1; j1<=itsD1; j1++)
        fa+=Fam1(w1,i1,j1)*itsMs[n](j1,j2);
    return fa;
}



Vector3CT MPSSite::IterateRightF(const SiteOperator* so, const Vector3CT& Fap1, bool cache)
{
    int Dw1=so->GetDw12().Dw1;;
    Vector3CT F(Dw1,itsD1,itsD1,1);
    Vector3CT::Subscriptor sF(F);
    for (int w1=1; w1<=Dw1; w1++)
        //#pragma omp parallel for collapse(2)
        for (int i1=1; i1<=itsD1; i1++)
            for (int j1=1; j1<=itsD1; j1++)
                sF(w1,i1,j1)=ContractBWFB(w1,i1,j1,so,Fap1);
    if (cache) itsHRightCache=F;
    return F;
}

dcmplx MPSSite::ContractBWFB(int w1, int i1, int j1, const SiteOperator* so, const Vector3CT& Fap1) const
{
    dcmplx bwfb(0.0);
    for (int m=0; m<itsd; m++)
        for (int i2=1; i2<=itsD2; i2++)
            bwfb+=conj(itsMs[m](i1,i2))*ContractWFB(m,w1,i2,j1,so,Fap1);

    return bwfb;
}

dcmplx MPSSite::ContractWFB(int m, int w1, int i2, int j1, const SiteOperator* so, const Vector3CT& Fap1) const
{
    const Dw12& Dws=so->GetDw12();
    dcmplx wfb(0.0);
    const MatrixOR& WOvM=so->GetW();
    for (int n=0; n<itsd; n++)
    {
        assert(WOvM.GetNumCols()==Dws.Dw2);
        for (int w2=1; w2<=Dws.w2_last(w1); w2++)
            if (WOvM(w1-1,w2-1)(m,n)!=0.0)
            {
                assert(fabs(WOvM(w1-1,w2-1)(m,n))>0.0);
                wfb+=WOvM(w1-1,w2-1)(m,n)*ContractFB(n,w2,i2,j1,Fap1);
            }
    }
    return wfb;
}

dcmplx MPSSite::ContractFB(int n, int w2, int i2, int j1, const Vector3CT& Fap1) const
{
    dcmplx fb(0.0);
    for (int j2=1; j2<=itsD2; j2++)
        fb+=Fap1(w2,i2,j2)*itsMs[n](j1,j2);
    return fb;
}

void MPSSite::SVDTransfer(Direction lr,const DiagonalMatrixRT& s, const MatrixCT& UV)
{
    switch (lr)
    {
    case DRight:
    {
        int N1=s.GetNumRows(); //N1=0 on the first site.
        if (N1>0 && N1<itsD2)
        {
            if (itsMs[0].GetNumCols()!=UV.GetNumRows())
                NewBondDimensions(itsD1,N1,true);
            else
                itsD2=N1; //The contraction below will automatically reshape the As.
        }
        for (int in=0; in<itsd; in++)
        {
            // Getting Heisenbug trap.  Leave in place
            if (itsMs[in].GetNumCols()!=UV.GetNumRows())
            {
                // D2=2 oldD2=8 in=0 N1=2 M=(1:8),(1:2)  UV=(1:4),(1:2)
                // D2=2 oldD2=8 in=1 N1=2 M=(1:8),(1:2)  UV=(1:4),(1:2)
                cout << "D2=" << itsD2 << " in=" << in << " N1=" << N1 << " M=" << itsMs[in].GetLimits() << " UV=" << UV.GetLimits() << endl;
                abort();
            }
            assert(itsMs[in].GetNumCols()==UV.GetNumRows());
            MatrixCT MU=itsMs[in]*UV;
            itsMs[in].SetLimits(0,0);
            itsMs[in]=MU*s;
            assert(itsMs[in].GetNumCols()==itsD2); //Verify shape is correct;
        }
        break;
    }
    case DLeft:
    {
        int N1=s.GetNumRows(); //N1=0 on the first site.
        if (N1>0 && N1<itsD1)
        {
            if (itsMs[0].GetNumRows()!=UV.GetNumCols())
                NewBondDimensions(N1,itsD2,true);
            else
                itsD1=N1; //The contraction below will automatically reshape the As.
        }

        for (int in=0; in<itsd; in++)
        {
            assert(UV.GetNumCols()==itsMs[in].GetNumRows());
            MatrixCT VM=UV*itsMs[in];
            itsMs[in].SetLimits(0,0);
            itsMs[in]=s*VM;
            //        cout << "A[" << in << "]=" << itsAs[in] << endl;
            assert(itsMs[in].GetNumRows()==itsD1); //Verify shape is correct;
        }
        break;
    }

    }
}

void MPSSite::TransferQR(Direction lr,const MatrixCT& R)
{
    SVDTransfer(lr,R);
}

void MPSSite::SVDTransfer(Direction lr,const MatrixCT& UV)
{
    switch (lr)
    {
    case DRight:
    {
        int N1=UV.GetNumCols(); //N1=0 on the first site.
        if (N1>0 && N1<itsD2)
        {
            if (itsMs[0].GetNumCols()!=UV.GetNumRows())
                NewBondDimensions(itsD1,N1,true);
            else
                itsD2=N1; //The contraction below will automatically reshape the As.
        }
        for (int in=0; in<itsd; in++)
        {
            assert(itsMs[in].GetNumCols()==UV.GetNumRows());
            MatrixCT temp=itsMs[in]*UV;
            itsMs[in].SetLimits(0,0);
            itsMs[in]=temp; //Shallow copy
            assert(itsMs[in].GetNumCols()==itsD2); //Verify shape is correct;
        }
        break;
    }
    case DLeft:
    {
        int N1=UV.GetNumRows(); //N1=0 on the first site.
        if (N1>0 && N1<itsD1)
        {
            if (itsMs[0].GetNumRows()!=UV.GetNumCols())
                NewBondDimensions(N1,itsD2,true);
            else
                itsD1=N1; //The contraction below will automatically reshape the As.
        }

        for (int in=0; in<itsd; in++)
        {
            assert(UV.GetNumCols()==itsMs[in].GetNumRows());
            MatrixCT temp=UV*itsMs[in];
            itsMs[in].SetLimits(0,0);
            itsMs[in]=temp; //Shallow copy
            //        cout << "A[" << in << "]=" << itsAs[in] << endl;
            assert(itsMs[in].GetNumRows()==itsD1); //Verify shape is correct;
        }
        break;
    }

    }
}


MatrixCT MPSSite::CalculateOneSiteDM()
{
    MatrixCT ro(itsd,itsd); //These can't be zero based if we want run them through eigen routines, which are hard ocded for 1 based matricies
    Fill(ro,dcmplx(0.0));
    for (int m=0; m<itsd; m++)
        for (int n=0; n<itsd; n++)
            for (int j1=1; j1<=itsD1; j1++)
                for (int j2=1; j2<=itsD2; j2++)
                    ro(m+1,n+1)+=std::conj(itsMs[m](j1,j2))*itsMs[n](j1,j2);
    return ro;
}

MatrixCT MPSSite::InitializeTwoSiteDM(int m, int n)
{
    MatrixCT C(itsD2,itsD2);
    Fill(C,dcmplx(0.0));
    for (int i2=1; i2<=itsD2; i2++)
        for (int j2=1; j2<=itsD2; j2++)
            for (int i1=1; i1<=itsD1; i1++)
                C(i2,j2)+=std::conj(itsMs[m](i1,i2))*itsMs[n](i1,j2);
    return C;
}

MatrixCT MPSSite::IterateTwoSiteDM(MatrixCT& Cmn)
{
    MatrixCT ret(itsD2,itsD2);
    Fill(ret,dcmplx(0.0));
    for (int n2=0; n2<itsd; n2++)
    {
        MatrixCT CAmn=ContractCA(n2,Cmn);
        for (int i2=1; i2<=itsD2; i2++)
            for (int j2=1; j2<=itsD2; j2++)
                for (int i1=1; i1<=itsD1; i1++)
                    ret(i2,j2)+=std::conj(itsMs[n2](i1,i2))*CAmn(i1,j2);
    }
    return ret;
}

MatrixCT MPSSite::ContractCA(int n2, const MatrixCT& Cmn) const
{
    MatrixCT ret(itsD1,itsD2);
    Fill(ret,dcmplx(0.0));
    for (int j2=1; j2<=itsD2; j2++)
        for (int i1=1; i1<=itsD1; i1++)
            for (int j1=1; j1<=itsD1; j1++)
                ret(i1,j2)+=Cmn(i1,j1)*itsMs[n2](j1,j2);
    return ret;
}

MatrixCT MPSSite::FinializeTwoSiteDM(const MatrixCT & Cmn)
{
    MatrixCT ret(itsd,itsd);
    Fill(ret,dcmplx(0.0));
    for (int n2=0; n2<itsd; n2++)
    {
        MatrixCT CAmn=ContractCA(n2,Cmn);
        for (int m2=0; m2<itsd; m2++)
            for (int j2=1; j2<=itsD2; j2++)
                for (int i1=1; i1<=itsD1; i1++)
                    ret(m2+1,n2+1)+=std::conj(itsMs[m2](i1,j2))*CAmn(i1,j2);
    }


    return ret;
}

void MPSSite::Contract(dVectorT& newAs,const SiteOperator* so)
{
    newAs.clear();
    const Dw12& Dws=so->GetDw12();
    int newD1=itsD1*Dws.Dw1;
    int newD2=itsD2*Dws.Dw2;
    const MatrixOR& WOvM=so->GetW();

    for (int n=0; n<itsd; n++)
    {
        newAs.push_back(MatrixCT(newD1,newD2));
        Fill(newAs[n],dcmplx(0.0));
        for (int m=0; m<itsd; m++)
        {
            assert(WOvM.GetNumRows()==Dws.Dw1);
            assert(WOvM.GetNumCols()==Dws.Dw2);
            int i1=1; //i1=(w1,j1)
            for (int w1=1; w1<=Dws.Dw1; w1++)
                for (int j1=1; j1<=itsD1; j1++,i1++)
                {
                    int i2=1; //i2=(w2,j2)
                    for (int w2=1; w2<=Dws.Dw2; w2++)
                        for (int j2=1; j2<=itsD2; j2++,i2++)
                            newAs[n](i1,i2)+=WOvM(w1-1,w2-1)(n,m)*itsMs[m](j1,j2);
                }
        }
        //  cout << "newAs[" << n << "]=" << newAs[n] << endl;
    }

}

MatrixCT MPSSite::ContractLRM(const MatrixCT& M, const MatrixCT& L, const MatrixCT& R) const
{
//    cout << "D1,D2=" << itsD1 << " " << itsD2 << endl;
//    cout << "ContractLRM L=" << L.GetLimits() << endl;
//    cout << "ContractLRM R=" << R.GetLimits() << endl;
    assert(R.GetNumRows()==itsD2);
    assert(L.GetNumRows()==itsD1);

    MatrixCT M_tilde(itsD1,itsD2);
    Fill(M_tilde,dcmplx(0.0));

//    cout << "ContractLRM RM=" << RM.GetLimits() << endl;
    for (int i2=1; i2<=itsD2; i2++)
        for (int j1=1; j1<=L.GetNumCols(); j1++)
        {
            dcmplx RM(0);
            for (int j2=1; j2<=R.GetNumCols(); j2++)
                RM+=R(i2,j2)*M(j1,j2);
            for (int i1=1; i1<=itsD1; i1++)
                M_tilde(i1,i2)+=L(i1,j1)*RM;
        }
    return M_tilde;
}
/*
MatrixCT MPSSite::Contract_RM(const MatrixCT& R, const MatrixCT& M) const
{
    assert(R.GetNumCols()==M.GetNumCols());
    assert(R.GetNumRows()==itsD2);
    MatrixCT RM(R.GetNumRows(),M.GetNumRows());
    Fill(RM,dcmplx(0.0));
    for (int i2=1; i2<=R.GetNumRows(); i2++)
        for (int j1=1; j1<=M.GetNumRows(); j1++)
            for (int j2=1; j2<=R.GetNumCols(); j2++)
                RM(i2,j1)+=R(i2,j2)*M(j1,j2);
    return RM;
}
*/

Matrix4CT  MPSSite::GetTransferMatrix(Direction lr) const
{
    assert(itsD1==itsD2);
    const DiagonalMatrixRT& lambda=GetBond(Invert(lr))->GetSVs();
    int D=itsD1;
    Matrix4CT E(D,D,D,D);
    for (int m=0; m<itsd; m++)
    {
//        cout << "lambda,M=" << lambda.size() << " " << itsMs[m].GetLimits() << endl;
        MatrixCT theta;
        if (lr==DLeft)
        {
            assert(lambda.size()==itsMs[m].GetNumRows());
            theta=lambda*itsMs[m];
        }
        else
        {
            assert(lambda.size()==itsMs[m].GetNumCols());
            theta=itsMs[m]*lambda;
        }
        assert(theta.GetNumRows()==D);
        assert(theta.GetNumCols()==D);
//        cout << "theta,D1,D2=" << theta.GetLimits() << " " << itsD1 << " " << itsD2 << endl;
        for (index_t i1:theta.rows())
            for (index_t i2:theta.cols())
                for (index_t j1:theta.rows())
                    for (index_t j2:theta.cols())
                        E(i1,j1,i2,j2)+=conj(theta(i1,i2))*theta(j1,j2);
    }
    return E;
}


void  MPSSite::ApplyInPlace(const SiteOperator* so)
{
    assert(so->GetFrobeniusNorm()>0.0);
    dVectorT newAs;
    Contract(newAs,so);

    const Dw12& Dws=so->GetDw12();
    itsD1=itsD1*Dws.Dw1;
    itsD2=itsD2*Dws.Dw2;
    itsMs=newAs;
}

void  MPSSite::Apply(const SiteOperator* so, MPSSite* psiPrime)
{
    assert(so->GetFrobeniusNorm()>0.0);
    Contract(psiPrime->itsMs,so);

    const Dw12& Dws=so->GetDw12();
    psiPrime->itsD1=itsD1*Dws.Dw1;
    psiPrime->itsD2=itsD2*Dws.Dw2;
}

} //namespace


