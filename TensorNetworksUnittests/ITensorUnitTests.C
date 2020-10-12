#include "Tests.H"
#include "itensor/itensor.h"

class ITensorTests : public ::testing::Test
{
public:

    ITensorTests()
    : eps(1.0e-13)
    {
//        StreamableObject::SetToPretty();
    }

    double eps;
};

using namespace itensor;

TEST_F(ITensorTests,TestITensor)
{
    double S=0.5;
    int d=2*S+1;
    int D=2;//,NSites=4,NBond=6;

    Index bond12=Index(D,"b12");
    Index bond13=Index(D,"b13");
    Index bond14=Index(D,"b14");
    Index bond23=Index(D,"b23");
    Index bond24=Index(D,"b24");
    Index bond34=Index(D,"b34");

    std::vector<ITensor> sites;
    sites.push_back(ITensor(Index(d,"n1,Site"),bond12,bond13,bond14));
    sites.push_back(ITensor(Index(d,"n2,Site"),bond12,bond23,bond24));
    sites.push_back(ITensor(Index(d,"n3,Site"),bond13,bond23,bond34));
    sites.push_back(ITensor(Index(d,"n4,Site"),bond14,bond24,bond34));

    //PrintData(sites[0]);
}


ITensor
makeSp(Index const& s)
    {
    auto Sp = ITensor(s,prime(s));
    Sp.set(s=2,prime(s)=1, 1);
    return Sp;
    }

ITensor
makeSm(Index const& s)
    {
    auto Sm = ITensor(s,prime(s));
    Sm.set(s=1,prime(s)=2,1);
    return Sm;
    }

ITensor
makeSz(Index const& s)
    {
    auto Sz = ITensor(s,prime(s));
    Sz.set(s=1,prime(s)=1,+0.5);
    Sz.set(s=2,prime(s)=2,-0.5);
    return Sz;
    }

ITensor
makeIdentity(Index const& s)
    {
    auto I = ITensor(s,prime(s));
    I.set(s=1,prime(s)=1,+1.0);
    I.set(s=2,prime(s)=2,+1.0);
    return I;
    }

ITensor MakeH(Index const& s1,Index const& s2)
{
    auto Sz1 = makeSz(s1);
    auto Sz2 = makeSz(s2);
    auto Sp1 = makeSp(s1);
    auto Sp2 = makeSp(s2);
    auto Sm1 = makeSm(s1);
    auto Sm2 = makeSm(s2);
    return (Sz1*Sz2 + 0.5*(Sp1*Sm2 + Sm1*Sp2));
}
#include "itensor/decomp.h"

TEST_F(ITensorTests,Test3SiteHamiltonian)
{
    double S=0.5;
    int d=2*S+1;
//    int D=2;//,NSites=4,NBond=6;

    auto s1 = Index(d,"s1");
    auto s2 = Index(d,"s2");
    auto s3 = Index(d,"s3");


    auto I1  = makeIdentity(s1);
    auto I2  = makeIdentity(s2);
    auto I3  = makeIdentity(s3);


    auto H12 = MakeH(s1,s2);
    auto H23 = MakeH(s2,s3);
    auto H31 = MakeH(s3,s1);
    auto H=H12*I3+H23*I1+H31*I2;
    //PrintData(H);


    auto [U,Evs] = diagHermitian(H);
//    PrintData(U);
//    PrintData(Evs);

    auto expH=expHermitian(H);
//    PrintData(expH);

}
TEST_F(ITensorTests,Test4SiteHamiltonian)
{
    double S=0.5;
    int d=2*S+1;
//    int D=2;//,NSites=4,NBond=6;

    auto s1 = Index(d,"s1");
    auto s2 = Index(d,"s2");
    auto s3 = Index(d,"s3");
    auto s4 = Index(d,"s4");


    auto I1  = makeIdentity(s1);
    auto I2  = makeIdentity(s2);
    auto I3  = makeIdentity(s3);
    auto I4  = makeIdentity(s4);


    auto H12 = MakeH(s1,s2);
    auto H13 = MakeH(s1,s3);
    auto H14 = MakeH(s1,s4);
    auto H23 = MakeH(s2,s3);
    auto H24 = MakeH(s2,s4);
    auto H34 = MakeH(s3,s4);
    auto H=H12*I3*I4+H13*I2*I4+H14*I2*I3+H23*I1*I4+H24*I1*I3+H34*I1*I2;
    //PrintData(H);


    auto [U,Evs] = diagHermitian(H);
//    PrintData(U);
//    PrintData(Evs);

    auto expH=expHermitian(H);
//    PrintData(expH);

}

