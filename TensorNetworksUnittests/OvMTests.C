#include "Tests.H"
#include "TensorNetworks/Factory.H"
#include "TensorNetworks/SVCompressor.H"
#include "TensorNetworksImp/Hamiltonians/Hamiltonian_1D_NN_Heisenberg.H"
#include "Operators/OperatorElement.H"
#include "Operators/OperatorValuedMatrix.H"

using dcmplx=TensorNetworks::dcmplx;
using TensorNetworks::MatrixOR;
using TensorNetworks::TriType;
using TensorNetworks::Lower;
using TensorNetworks::Upper;
using TensorNetworks::Direction;
using TensorNetworks::DLeft;
using TensorNetworks::DRight;
using TensorNetworks::SVCompressorR;
using TensorNetworks::MatrixRT;
using TensorNetworks::Position;
using TensorNetworks::PBulk;
using TensorNetworks::PLeft;
using TensorNetworks::PRight;
using TensorNetworks::OperatorSz;
using TensorNetworks::OperatorSx;
using TensorNetworks::OperatorSy;
using TensorNetworks::OperatorSp;
using TensorNetworks::OperatorSm;
using TensorNetworks::OperatorI;
using TensorNetworks::OperatorZ;
using TensorNetworks::OperatorElement;

class OvMTests : public ::testing::Test
{
public:
    OvMTests()
        : eps(3.0e-15)
        , itsFactory(TensorNetworks::Factory::GetFactory())
        , itsOperatorClient(0)
    {
        assert(itsFactory);
        StreamableObject::SetToPretty();
    }
    ~OvMTests()
    {
        delete itsFactory;
        if (itsOperatorClient) delete itsOperatorClient;
    }

    void Setup(double S)
    {
        if (itsOperatorClient) delete itsOperatorClient;
        itsOperatorClient=new TensorNetworks::Hamiltonian_1D_NN_Heisenberg(S,1.0,1.0,0.0);
        assert(itsOperatorClient);
    }

    void TestQR (MatrixOR OvM,Direction,TriType,Position);
    void TestSVD(MatrixOR OvM,Direction,TriType,Position);
    void TestShuffle(MatrixOR OvM,Direction lr,double eps,double S);
    Direction Invert(Direction lr) const
    {
        if (lr==DLeft)
            lr=DRight;
        else if (lr==DRight)
            lr=DLeft;
        return lr;
    }

    double eps;
           TensorNetworks::Factory*         itsFactory;
    const  TensorNetworks::OperatorClient* itsOperatorClient;
};

void MakeLRBOperator(MatrixOR& OvM,TriType ul,Position lbr)
{
    auto [X1,X2]=OvM.GetChi12();
    if (lbr==PLeft)
    {
        MatrixRT l(0,0,0,X2+1);
        Fill(l,0.0);
        switch (ul)
        {
        case Upper:
            l(0,0)=1.0;
            break;
        case Lower:
            l(0,X2+1)=1.0;
            break;
        default:
            assert(false);
        }
        OvM=MatrixOR(l*OvM);
        OvM.SetUpperLower(ul);
    }
    if (lbr==PRight)
    {
        MatrixRT r(0,X1+1,0,0);
        Fill(r,0.0);
        switch (ul)
        {
        case Upper:
            r(X1+1,0)=1.0;
            break;
        case Lower:
            r(0,0)=1.0;
            break;
        default:
            assert(false);
        }
        OvM=MatrixOR(OvM*r);
        OvM.SetUpperLower(ul);
    }
}

void OvMTests::TestQR(MatrixOR OvM,Direction lr,TriType ul,Position lbr)
{
    int d=2*itsOperatorClient->GetS()+1;
    MakeLRBOperator(OvM,ul,lbr);
    MatrixOR V=OvM.GetV(lr);
    auto [Q,R]=OvM.BlockQX(lr);
    MatrixRT R1=R;
    if (lr==DLeft)
        R1.SetLimits(MatLimits(Q.GetColLimits(),V.GetColLimits()),true); //Shrink R back to Q size so we can multiply.
    else if (lr==DRight)
        R1.SetLimits(MatLimits(V.GetRowLimits(),Q.GetRowLimits()),true); //Shrink R back to Q size so we can multiply.
    else
        assert(false);

    MatrixOR V1;
    if (lr==DLeft)
        V1=Q*R1;
    else
        V1=R1*Q;
    EXPECT_NEAR(MaxDelta(V,V1),0.0,d*eps);

    if (ul==Upper)
    {
        EXPECT_TRUE(IsUpperTriangular(R,1e-13));
        if (Q.GetNumCols()>1)
        {
            EXPECT_TRUE(IsUpperTriangular(Q));
        }
    }
    if (ul==Lower)
    {
        EXPECT_TRUE(IsLowerTriangular(R));
        if (Q.GetNumRows()>1)
        {
            EXPECT_TRUE(IsLowerTriangular(Q));
        }
    }
    EXPECT_TRUE(IsUnit(Q.GetOrthoMatrix(lr),d*eps));
    EXPECT_FALSE(IsUnit(Q.GetOrthoMatrix(Invert(lr)),d*eps));
}

void OvMTests::TestSVD(MatrixOR OvM,Direction lr,TriType ul,Position lbr)
{
    int d=2*itsOperatorClient->GetS()+1;
    SVCompressorR* comp=itsFactory->MakeMPOCompressor(0,1e-14);
    MakeLRBOperator(OvM,ul,lbr);
    MatrixOR V=OvM.GetV(lr);
    auto [Q,R]=OvM.BlockSVD(lr,comp);
    MatrixOR V1;
    if (lr==DLeft)
    {
        R.SetLimits(MatLimits(Q.GetColLimits(),V.GetColLimits()),true); //Shrink R back to Q size so we can multiply.
        V1=Q*R;
    }
    else
    {
        R.SetLimits(MatLimits(V.GetRowLimits(),Q.GetRowLimits()),true); //Shrink R back to Q size so we can multiply.
        V1=R*Q;
    }

    EXPECT_NEAR(MaxDelta(V,V1),0.0,d*d*eps);

// Currently the SVD does not support this. We need to swap rows/columns of U/VT in degenerate blocks to fix this.
//    if (ul==Upper)
//    {
////        EXPECT_TRUE(IsUpperTriangular(R)); Not guaranteed for SVD
//        EXPECT_TRUE(IsUpperTriangular(Q,eps));
//    }
//    if (ul==Lower)
//    {
////        EXPECT_TRUE(IsLowerTriangular(R)); Not guaranteed for SVD
//        cout << "Q=" << Q << endl;
//        EXPECT_TRUE(IsLowerTriangular(Q,eps));
//    }
    EXPECT_TRUE(IsUnit(Q.GetOrthoMatrix(lr),eps));
    EXPECT_FALSE(IsUnit(Q.GetOrthoMatrix(Invert(lr)),eps));
}

TEST_F(OvMTests,Setup)
{
    Setup(0.5);
}


TEST_F(OvMTests,OperatorElement1)
{
    double S=0.5;
    {
        OperatorSz Sz12(S);
        EXPECT_EQ(Sz12(0,0),-0.5);
        EXPECT_EQ(Sz12(1,0), 0.0);
        EXPECT_EQ(Sz12(0,1), 0.0);
        EXPECT_EQ(Sz12(1,1), 0.5);
    }

    {
        OperatorSp Sp12(S);
        EXPECT_EQ(Sp12(0,0), 0.0);
        EXPECT_EQ(Sp12(1,0), 1.0);
        EXPECT_EQ(Sp12(0,1), 0.0);
        EXPECT_EQ(Sp12(1,1), 0.0);
    }

    {
        OperatorSm Sm12(S);
        EXPECT_EQ(Sm12(0,0), 0.0);
        EXPECT_EQ(Sm12(1,0), 0.0);
        EXPECT_EQ(Sm12(0,1), 1.0);
        EXPECT_EQ(Sm12(1,1), 0.0);
    }

    {
        OperatorSy Sy12(S);
        EXPECT_EQ(Sy12(0,0), 0.0);
        EXPECT_EQ(Sy12(1,0), dcmplx(0.0,-0.5));
        EXPECT_EQ(Sy12(0,1), dcmplx(0.0, 0.5));
        EXPECT_EQ(Sy12(1,1), 0.0);
    }
    {
        OperatorSx Sx12(S);
        EXPECT_EQ(Sx12(0,0), 0.0);
        EXPECT_EQ(Sx12(1,0), 0.5);
        EXPECT_EQ(Sx12(0,1), 0.5);
        EXPECT_EQ(Sx12(1,1), 0.0);
    }
}

TEST_F(OvMTests,OperatorElement2)
{
    OperatorElement<double> Oe(0.5);
    Oe=1.0;
    EXPECT_EQ(Oe(0,0),1.0);
    EXPECT_EQ(Oe(1,0),0.0);
    EXPECT_EQ(Oe(0,1),0.0);
    EXPECT_EQ(Oe(1,1),1.0);
    Oe=0.0;
    EXPECT_EQ(Oe(0,0),0.0);
    EXPECT_EQ(Oe(1,0),0.0);
    EXPECT_EQ(Oe(0,1),0.0);
    EXPECT_EQ(Oe(1,1),0.0);
    Oe=1.1;
    EXPECT_EQ(Oe(0,0),1.1);
    EXPECT_EQ(Oe(1,0),0.0);
    EXPECT_EQ(Oe(0,1),0.0);
    EXPECT_EQ(Oe(1,1),1.1);
}
TEST_F(OvMTests,OperatorElement3)
{
    double f=1.1;
    OperatorElement<double> Oe(2,f);
    EXPECT_EQ(Oe(0,0),f);
    EXPECT_EQ(Oe(1,0),0.0);
    EXPECT_EQ(Oe(0,1),0.0);
    EXPECT_EQ(Oe(1,1),f);
}


TEST_F(OvMTests,OperatorValuedMatrix1)
{
    double S=0.5;
    Setup(S);
    MatrixOR OvM(itsOperatorClient->GetMatrixO(Lower));
    EXPECT_EQ(OvM(0,0),OperatorI (S));
    EXPECT_EQ(OvM(1,0),OperatorSp(S));
    EXPECT_EQ(OvM(2,0),OperatorSm(S));
    EXPECT_EQ(OvM(3,0),OperatorSz(S));
    EXPECT_EQ(OvM(0,1),OperatorZ (S));
    EXPECT_EQ(OvM(0,2),OperatorZ (S));
    EXPECT_EQ(OvM(0,3),OperatorZ (S));
    EXPECT_EQ(OvM.GetUpperLower(),Lower);
}

TEST_F(OvMTests,OperatorValuedMatrix2)
{
    double S=1.0;
    Setup(S);
    MatrixOR OvM(itsOperatorClient->GetMatrixO(Lower));
    EXPECT_EQ(OvM(0,0),OperatorI (S));
    EXPECT_EQ(OvM(1,0),OperatorSp(S));
    EXPECT_EQ(OvM(2,0),OperatorSm(S));
    EXPECT_EQ(OvM(3,0),OperatorSz(S));
    EXPECT_EQ(OvM(0,1),OperatorZ (S));
    EXPECT_EQ(OvM(0,2),OperatorZ (S));
    EXPECT_EQ(OvM(0,3),OperatorZ (S));
    EXPECT_EQ(OvM.GetUpperLower(),Lower);
}

TEST_F(OvMTests,OperatorValuedMatrix3)
{
    double S=0.5;
    Setup(S);
    MatrixOR OvM(itsOperatorClient->GetMatrixO(Upper));
    EXPECT_EQ(OvM(0,0),OperatorI (S));
    EXPECT_EQ(OvM(0,1),OperatorSp(S));
    EXPECT_EQ(OvM(0,2),OperatorSm(S));
    EXPECT_EQ(OvM(0,3),OperatorSz(S));
    EXPECT_EQ(OvM(1,0),OperatorZ (S));
    EXPECT_EQ(OvM(2,0),OperatorZ (S));
    EXPECT_EQ(OvM(3,0),OperatorZ (S));
    EXPECT_EQ(OvM.GetUpperLower(),Upper);
}

TEST_F(OvMTests,OperatorValuedMatrix4)
{
    double S=1.0;
    Setup(S);
    MatrixOR OvM(itsOperatorClient->GetMatrixO(Upper));
    EXPECT_EQ(OvM(0,0),OperatorI (S));
    EXPECT_EQ(OvM(0,1),OperatorSp(S));
    EXPECT_EQ(OvM(0,2),OperatorSm(S));
    EXPECT_EQ(OvM(0,3),OperatorSz(S));
    EXPECT_EQ(OvM(1,0),OperatorZ (S));
    EXPECT_EQ(OvM(2,0),OperatorZ (S));
    EXPECT_EQ(OvM(3,0),OperatorZ (S));
    EXPECT_EQ(OvM.GetUpperLower(),Upper);
}

TEST_F(OvMTests,OperatorValuedMatrix5)
{
    double S=0.5;
    {
        MatrixOR OvM(1,1,S,Lower);
        Unit(OvM);
        EXPECT_EQ(OvM(0,0),OperatorI (S));
    }
    {
        MatrixOR OvM(3,3,S,Lower);
        Unit(OvM);
        EXPECT_EQ(OvM(0,0),OperatorI(S));
        EXPECT_EQ(OvM(1,1),OperatorI(S));
        EXPECT_EQ(OvM(2,2),OperatorI(S));
        EXPECT_EQ(OvM(0,1),OperatorZ(S));
        EXPECT_EQ(OvM(0,2),OperatorZ(S));
        EXPECT_EQ(OvM(1,0),OperatorZ(S));
        EXPECT_EQ(OvM(1,2),OperatorZ(S));
        EXPECT_EQ(OvM(2,0),OperatorZ(S));
        EXPECT_EQ(OvM(2,1),OperatorZ(S));
    }

}



TEST_F(OvMTests,OperatorValuedMatrixGetVUpper)
{
    double S=0.5;
    Setup(S);
    MatrixOR OvM(itsOperatorClient->GetMatrixO(Upper));
    MatrixOR Vl=OvM.GetV(DLeft);
    MatrixOR Vr=OvM.GetV(DRight);
    EXPECT_EQ(OvM.GetNumRows(),Vl.GetNumRows()+1);
    EXPECT_EQ(OvM.GetNumCols(),Vl.GetNumCols()+1);
    EXPECT_EQ(OvM.GetNumRows(),Vr.GetNumRows()+1);
    EXPECT_EQ(OvM.GetNumCols(),Vr.GetNumCols()+1);
    for (index_t i:Vl.rows())
    for (index_t j:Vl.cols())
        EXPECT_EQ(OvM(i,j),Vl(i,j));
    for (index_t i:Vr.rows())
    for (index_t j:Vr.cols())
        EXPECT_EQ(OvM(i,j),Vr(i,j));
}

TEST_F(OvMTests,OperatorValuedMatrixGetVLower)
{
    double S=0.5;
    Setup(S);
    MatrixOR OvM(itsOperatorClient->GetMatrixO(Lower));
    MatrixOR Vl=OvM.GetV(DLeft);
    MatrixOR Vr=OvM.GetV(DRight);
    EXPECT_EQ(OvM.GetNumRows(),Vl.GetNumRows()+1);
    EXPECT_EQ(OvM.GetNumCols(),Vl.GetNumCols()+1);
    EXPECT_EQ(OvM.GetNumRows(),Vr.GetNumRows()+1);
    EXPECT_EQ(OvM.GetNumCols(),Vr.GetNumCols()+1);
    for (index_t i:Vl.rows())
    for (index_t j:Vl.cols())
        EXPECT_EQ(OvM(i,j),Vl(i,j));
    for (index_t i:Vr.rows())
    for (index_t j:Vr.cols())
        EXPECT_EQ(OvM(i,j),Vr(i,j));
}

TEST_F(OvMTests,OperatorValuedMatrixSetVUpper)
{
    double S=0.5;
    Setup(S);
    MatrixOR OvM(itsOperatorClient->GetMatrixO(Upper));
    MatrixOR Copy(OvM);
    MatrixOR Vl=OvM.GetV(DLeft);
    Copy.SetV(DLeft,Vl);
    EXPECT_EQ(OvM,Copy);
    MatrixOR Vr=OvM.GetV(DRight);
    Copy.SetV(DRight,Vr);
    EXPECT_EQ(OvM,Copy);
}

TEST_F(OvMTests,OperatorValuedMatrixSetVLower)
{
    double S=0.5;
    Setup(S);
    MatrixOR OvM(itsOperatorClient->GetMatrixO(Lower));
    MatrixOR Copy(OvM);
    MatrixOR Vl=OvM.GetV(DLeft);
    Copy.SetV(DLeft,Vl);
    EXPECT_EQ(OvM,Copy);
    MatrixOR Vr=OvM.GetV(DRight);
    Copy.SetV(DRight,Vr);
    EXPECT_EQ(OvM,Copy);
}

TEST_F(OvMTests,OperatorValuedMatrixFlattenUpper)
{
    double S=0.5;
    Setup(S);
    MatrixOR OvM(itsOperatorClient->GetMatrixO(Upper));
    MatrixOR Copy(OvM);
    Matrix<double> F=Copy.Flatten(DLeft);
    Copy.UnFlatten(F);
    EXPECT_EQ(OvM,Copy);
    F=Copy.Flatten(DRight);
    Copy.UnFlatten(F);
    EXPECT_EQ(OvM,Copy);
}

TEST_F(OvMTests,OperatorValuedMatrixFlattenLower)
{
    double S=0.5;
    Setup(S);
    MatrixOR OvM(itsOperatorClient->GetMatrixO(Lower));
    MatrixOR Copy(OvM);
    Matrix<double> F=Copy.Flatten(DLeft);
    Copy.UnFlatten(F);
    EXPECT_EQ(OvM,Copy);
    F=Copy.Flatten(DRight);
    Copy.UnFlatten(F);
    EXPECT_EQ(OvM,Copy);
}

TEST_F(OvMTests,OperatorValuedMatrixFlattenVUpper)
{
    double S=0.5;
    Setup(S);
    MatrixOR OvM(itsOperatorClient->GetMatrixO(Upper));
    {
        MatrixOR V=OvM.GetV(DLeft);
        MatrixOR Copy(V);
        Matrix<double> F=V.Flatten(DLeft);
        Copy.UnFlatten(F);
        EXPECT_EQ(V,Copy);
        F=Copy.Flatten(DRight);
        Copy.UnFlatten(F);
        EXPECT_EQ(V,Copy);
    }
    {
        MatrixOR V=OvM.GetV(DRight);
        MatrixOR Copy(V);
        Matrix<double> F=V.Flatten(DRight);
        Copy.UnFlatten(F);
        EXPECT_EQ(V,Copy);
        F=Copy.Flatten(DRight);
        Copy.UnFlatten(F);
        EXPECT_EQ(V,Copy);
    }
}

TEST_F(OvMTests,OperatorValuedMatrixFlattenVLower)
{
    double S=0.5;
    Setup(S);
    MatrixOR OvM(itsOperatorClient->GetMatrixO(Lower));
    {
        MatrixOR V=OvM.GetV(DLeft);
        MatrixOR Copy(V);
        Matrix<double> F=V.Flatten(DLeft);
        Copy.UnFlatten(F);
        EXPECT_EQ(V,Copy);
        F=Copy.Flatten(DRight);
        Copy.UnFlatten(F);
        EXPECT_EQ(V,Copy);
    }
    {
        MatrixOR V=OvM.GetV(DRight);
        MatrixOR Copy(V);
        Matrix<double> F=V.Flatten(DRight);
        Copy.UnFlatten(F);
        EXPECT_EQ(V,Copy);
        F=Copy.Flatten(DRight);
        Copy.UnFlatten(F);
        EXPECT_EQ(V,Copy);
    }
}

TEST_F(OvMTests,Grow1)
{
    MatrixRT A(1,4,1,24);
    Unit(A);
    MatLimits l(0,4,0,24);
    TensorNetworks::Grow(A,l);
    EXPECT_TRUE(IsUnit(A));
}
TEST_F(OvMTests,Grow2)
{
    MatrixRT A(1,24,1,4);
    Unit(A);
    MatLimits l(0,24,0,4);
    TensorNetworks::Grow(A,l);
    EXPECT_TRUE(IsUnit(A));
}
TEST_F(OvMTests,Grow3)
{
    MatrixRT A(0,4,0,24);
    Unit(A);
    MatLimits l(0,5,0,25);
    TensorNetworks::Grow(A,l);
    for (index_t i:A.rows())
    {
        if (i==5)
            EXPECT_EQ(A(i,25),1.0);
        else
            EXPECT_EQ(A(i,25),0.0);
    }
    for (index_t i:A.cols())
    {
        if (i==25)
            EXPECT_EQ(A(5,i),1.0);
        else
            EXPECT_EQ(A(5,i),0.0);
    }
    EXPECT_FALSE(IsUnit(A));
}
TEST_F(OvMTests,Grow4)
{
    MatrixRT A(0,24,0,4);
    Unit(A);
    MatLimits l(0,25,0,5);
    TensorNetworks::Grow(A,l);
    for (index_t i:A.rows())
    {
        if (i==25)
            EXPECT_EQ(A(i,5),1.0);
        else
            EXPECT_EQ(A(i,5),0.0);
    }
    for (index_t i:A.cols())
    {
        if (i==5)
            EXPECT_EQ(A(25,i),1.0);
        else
            EXPECT_EQ(A(25,i),0.0);
    }
    EXPECT_FALSE(IsUnit(A));
}

TEST_F(OvMTests,Grow5)
{
    MatrixRT A(0,4,0,4);
    Unit(A);
    MatLimits l(0,5,0,5);
    TensorNetworks::Grow(A,l);
    EXPECT_TRUE(IsUnit(A));
}

TEST_F(OvMTests,Lower1)
{
    int M=4,N=24;
    MatrixRT A(M,N);
    Fill(A,1.0);
    EXPECT_FALSE(IsLowerTriangular(A));
    int delta=Max(1,N-M+1);
    for (index_t i:A.rows())
    for (index_t j:A.cols(i+delta))
        A(i,j)=0;

    EXPECT_TRUE(IsLowerTriangular(A));
    EXPECT_TRUE(IsLowerTriangular(A,0.0));
    EXPECT_FALSE(IsUpperTriangular(A));
    EXPECT_FALSE(IsUpperTriangular(A,0.0));

}
TEST_F(OvMTests,Lower2)
{
    int M=24,N=4;
    MatrixRT A(M,N);
    Fill(A,1.0);
    EXPECT_FALSE(IsLowerTriangular(A));
    int delta=Max(1,N-M+1);
    for (index_t i:A.rows())
    for (index_t j:A.cols(i+delta))
        A(i,j)=0;

    EXPECT_TRUE(IsLowerTriangular(A));
    EXPECT_TRUE(IsLowerTriangular(A,0.0));
    EXPECT_FALSE(IsUpperTriangular(A));
    EXPECT_FALSE(IsUpperTriangular(A,0.0));

}

TEST_F(OvMTests,Upper1)
{
    int M=4,N=24;
    MatrixRT A(M,N);
    Fill(A,1.0);
    EXPECT_FALSE(IsLowerTriangular(A));
    int delta=Max(1,M-N+1);
    for (index_t j:A.cols())
    for (index_t i:A.rows(j+delta))
        A(i,j)=0;

    EXPECT_TRUE(IsUpperTriangular(A));
    EXPECT_TRUE(IsUpperTriangular(A,0.0));
    EXPECT_FALSE(IsLowerTriangular(A));
    EXPECT_FALSE(IsLowerTriangular(A,0.0));

}

TEST_F(OvMTests,Upper2)
{
    int M=24,N=4;
    MatrixRT A(M,N);
    Fill(A,1.0);
    EXPECT_FALSE(IsLowerTriangular(A));
    int delta=Max(1,M-N+1);
    for (index_t j:A.cols())
    for (index_t i:A.rows(j+delta))
        A(i,j)=0;

    EXPECT_TRUE(IsUpperTriangular(A));
    EXPECT_TRUE(IsUpperTriangular(A,0.0));
    EXPECT_FALSE(IsLowerTriangular(A));
    EXPECT_FALSE(IsLowerTriangular(A,0.0));

}
TEST_F(OvMTests,OpMulMO)
{
    int M=8,N=4,d=2;
    TriType ul=Lower;
    VecLimits vl1(0,M-1);
    VecLimits vl2(0,N-1);
    MatLimits la(vl1,vl2);
    MatLimits lb(vl2,vl1);
    MatrixRT A(la);
    MatrixOR B(d,ul,lb);
    Fill(A,1.0);
    Fill(B,1.0);

    MatrixOR C=A*B;
    EXPECT_EQ(C.Getd(),d);
    EXPECT_EQ(C.GetUpperLower(),ul);

    MatrixOR D=B*A;
    EXPECT_EQ(D.Getd(),d);
    EXPECT_EQ(D.GetUpperLower(),ul);
}

TEST_F(OvMTests,TensorProduct)
{
    int M=8,N=4,d=2;
    TriType ul=Lower;
    VecLimits vl1(0,M-1);
    VecLimits vl2(0,N-1);
    MatLimits la(vl1,vl2);
    MatLimits lb(vl2,vl1);
    MatrixOR A(d,ul,la);
    MatrixOR B(d,ul,lb);
    Fill(A,1.0);
    Fill(B,1.0);

    MatrixOR C=TensorProduct(A,B);
    EXPECT_EQ(C.Getd(),d);
    EXPECT_EQ(C.GetUpperLower(),ul);

    MatrixOR D=TensorProduct(B,A);
    EXPECT_EQ(D.Getd(),d);
    EXPECT_EQ(D.GetUpperLower(),ul);
}


TEST_F(OvMTests,OperatorValuedMatrixQRBulk)
{
    for (double S=0.5;S<=2.5;S+=0.5)
    {
        Setup(S);
        {
            MatrixOR OvM(itsOperatorClient->GetMatrixO(Upper));
            TestQR(OvM,DLeft ,Upper,PBulk);
            TestQR(OvM,DRight,Upper,PBulk);
        }
        {
            MatrixOR OvM(itsOperatorClient->GetMatrixO(Lower));
            TestQR(OvM,DLeft ,Lower,PBulk);
            TestQR(OvM,DRight,Lower,PBulk);
        }
    }
}

TEST_F(OvMTests,OperatorValuedMatrixQRBulkH2)
{
    for (double S=0.5;S<=2.5;S+=0.5)
    {
        Setup(S);
        {
            MatrixOR H(itsOperatorClient->GetMatrixO(Upper));
            MatrixOR H2=TensorProduct(H,H);
            TestQR(H2,DLeft ,Upper,PBulk);
            TestQR(H2,DRight,Upper,PBulk);
        }
        {
            MatrixOR H(itsOperatorClient->GetMatrixO(Lower));
            MatrixOR H2=TensorProduct(H,H);
            TestQR(H2,DLeft ,Lower,PBulk);
            TestQR(H2,DRight,Lower,PBulk);
        }
    }
}

TEST_F(OvMTests,OperatorValuedMatrixQRLeft)
{
    for (double S=0.5;S<=2.5;S+=0.5)
    {
        Setup(S);
        {
            MatrixOR OvM(itsOperatorClient->GetMatrixO(Upper));
            TestQR(OvM,DLeft ,Upper,PLeft);
        }
        {
            MatrixOR OvM(itsOperatorClient->GetMatrixO(Lower));
            TestQR(OvM,DLeft ,Lower,PLeft);
        }
    }
}

TEST_F(OvMTests,OperatorValuedMatrixQRLeftH2)
{
    for (double S=0.5;S<=2.5;S+=0.5)
    {
        Setup(S);
        {
            MatrixOR H(itsOperatorClient->GetMatrixO(Upper));
            MatrixOR H2=TensorProduct(H,H);
            TestQR(H2,DLeft ,Upper,PLeft);
        }
        {
            MatrixOR H(itsOperatorClient->GetMatrixO(Lower));
            MatrixOR H2=TensorProduct(H,H);
            TestQR(H2,DLeft ,Lower,PLeft);
        }
    }
}

TEST_F(OvMTests,OperatorValuedMatrixQRRight)
{
    for (double S=0.5;S<=2.5;S+=0.5)
    {
        Setup(S);
        {
            MatrixOR OvM(itsOperatorClient->GetMatrixO(Upper));
            TestQR(OvM,DRight ,Upper,PRight);
        }
        {
            MatrixOR OvM(itsOperatorClient->GetMatrixO(Lower));
            TestQR(OvM,DRight ,Lower,PRight);
        }
    }

}

TEST_F(OvMTests,OperatorValuedMatrixQRRightH2)
{
    for (double S=0.5;S<=2.5;S+=0.5)
    {
        Setup(S);
        {
            MatrixOR H(itsOperatorClient->GetMatrixO(Upper));
            MatrixOR H2=TensorProduct(H,H);
            TestQR(H2,DRight,Upper,PRight);
        }
        {
            MatrixOR H(itsOperatorClient->GetMatrixO(Lower));
            MatrixOR H2=TensorProduct(H,H);
            TestQR(H2,DRight,Lower,PRight);
        }
    }
}


TEST_F(OvMTests,OperatorValuedMatrixSVD)
{
    for (double S=0.5;S<=2.5;S+=0.5)
    {
        Setup(S);
        {
            MatrixOR H(itsOperatorClient->GetMatrixO(Upper));
            TestSVD(H,DLeft ,Upper,PLeft);
            TestSVD(H,DRight,Upper,PRight);
            TestSVD(H,DLeft ,Upper,PBulk);
            TestSVD(H,DRight,Upper,PBulk);
        }
        {
            MatrixOR H(itsOperatorClient->GetMatrixO(Lower));
            TestSVD(H,DLeft ,Lower,PLeft);
            TestSVD(H,DRight,Lower,PRight);
            TestSVD(H,DLeft ,Lower,PBulk);
            TestSVD(H,DRight,Lower,PBulk);
        }
    }
}

TEST_F(OvMTests,OperatorValuedMatrixSVDH2BulkOnly)
{
    for (double S=0.5;S<=2.5;S+=0.5)
    {
        Setup(S);
        {
            MatrixOR H(itsOperatorClient->GetMatrixO(Upper));
            MatrixOR H2=TensorProduct(H,H);
//            TestSVD(H2,DLeft ,Upper,PLeft); //These won't work, we need canonical form 1,d,d^2 etc for Dws
//            TestSVD(H2,DRight,Upper,PRight);
            TestSVD(H2,DLeft ,Upper,PBulk);
            TestSVD(H2,DRight,Upper,PBulk);
        }
        {
            MatrixOR H(itsOperatorClient->GetMatrixO(Lower));
            MatrixOR H2=TensorProduct(H,H);
//            TestSVD(H2,DLeft ,Lower,PLeft); //These won't work, we need canonical form 1,d,d^2 etc for Dws
//            TestSVD(H2,DRight,Lower,PRight);
            TestSVD(H2,DLeft ,Lower,PBulk);
            TestSVD(H2,DRight,Lower,PBulk);
        }
    }
}
#include "NumericalMethods/LapackSVDSolver.H"
#include "oml/diagonalmatrix.h"




void OvMTests::TestShuffle(MatrixOR O,Direction lr,double eps,double S)
{
    SVCompressorR* comp=itsFactory->MakeMPOCompressor(0,eps);
    LapackSVDSolver<double> solver;
    MatrixOR Ocopy=O; //Save the testing later.
    //
    //  Make sure we are starting with a triangular operator valued matrix.
    //
    TriType ul=O.GetUpperLower();
    EXPECT_TRUE(IsTriangular(ul,O,eps));
    MatrixRT Of=O.Flatten(lr);
    EXPECT_TRUE(IsTriangular(ul,Of,eps));
    //
    //  SVD->Compress->Shuffle
    //
    MatLimits lim=Of.ReBase(1,1); //Numerical routines only work with fortran indexing.
    auto [U,s,VT]=solver.SolveAll(Of,eps);
    comp->Compress(U,s,VT);
    SVDShuffle(ul,lr,U,s,VT,eps);
    EXPECT_TRUE(IsTriangular(ul,VT,eps));
    EXPECT_TRUE(IsTriangular(ul,U ,eps));
    //
    //  Make sure we can rebuild O from the heavily processed U*s*VT
    //  And the O is now triangular
    //
    MatrixRT Of1=U*s*VT;
    EXPECT_TRUE(IsTriangular(ul,Of1,eps));
    Of1.ReBase(lim);
    O.UnFlatten(Of1);
    EXPECT_TRUE(IsTriangular(ul,O,eps));
    EXPECT_NEAR(MaxDelta(O,Ocopy),0.0,eps);
    //
    //  Make sure OpVal versions of U/VT are also triangular.
    //
    switch (lr)
    {
    case DLeft:
        {
            MatrixOR Uo(O);
            U.ReBase(0,0);
            Uo.UnFlatten(U);
            EXPECT_TRUE(IsTriangular(ul,Uo,eps));
        }
        break;
    case DRight:
        {
            MatrixOR Vo(O);
            VT.ReBase(0,0);
            Vo.UnFlatten(VT);
            EXPECT_TRUE(IsTriangular(ul,Vo,eps));
        }
        break;
    }
}

TEST_F(OvMTests,SVDShuffleH)
{
    double eps=1e-14;
    for (double S=0.5;S<=2.5;S+=0.5)
    {
        Setup(S);
        {
            MatrixOR O(itsOperatorClient->GetMatrixO(Lower));
            TestShuffle(O,DLeft ,eps,S);
            TestShuffle(O,DRight,eps,S);
        }
        {
            MatrixOR O(itsOperatorClient->GetMatrixO(Upper));
            TestShuffle(O,DLeft ,eps,S);
            TestShuffle(O,DRight,eps,S);
        }
    }
}

TEST_F(OvMTests,SVDShuffleH2)
{
    double eps=1e-14;
    for (double S=0.5;S<=0.5;S+=0.5)
    {
        Setup(S);
        {
            MatrixOR H(itsOperatorClient->GetMatrixO(Lower));
            MatrixOR H2=TensorProduct(H,H);
            TestShuffle(H2,DLeft ,eps,S);
            TestShuffle(H2,DRight,eps,S);
        }
        {
            MatrixOR H(itsOperatorClient->GetMatrixO(Upper));
            MatrixOR H2=TensorProduct(H,H);
            TestShuffle(H2,DLeft ,eps,S);
            TestShuffle(H2,DRight,eps,S);
        }
    }
}

