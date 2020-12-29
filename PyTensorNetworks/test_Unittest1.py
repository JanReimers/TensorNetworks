import PyTensorNetworks as TN
import pytest

@pytest.mark.parametrize("L, S, Dfinal, Eexpected, E2expected",[
#L S  D   E                   E2
(2 ,0.5,2,-0.750000000000000 ,1e-13),
(2 ,1.0,3,-2.000000000000000 ,1e-13),
(2 ,1.5,4,-3.750000000000000 ,1e-13),
(2 ,2.0,5,-6.000000000000000 ,1e-13),
(2 ,2.5,6,-8.750000000000000 ,1e-13),
(3 ,0.5,2,-0.500000000000000 ,1e-13),
(3 ,1.0,3,-1.500000000000000 ,1e-13),
(3 ,1.5,4,-3.000000000000000 ,1e-13),
(3 ,2.0,5,-5.000000000000000 ,2e-13),
(3 ,2.5,6,-7.500000000000000 ,1e-12),
(4 ,0.5,4,-0.538675134594813 ,1e-13),
(4 ,1.0,9,-1.548583770354863 ,1e-13),
(5 ,0.5,4,-0.481971563329499 ,1e-13),
(6 ,0.5,8,-0.498715426777585 ,1e-13),
(7 ,0.5,8,-0.472706613447776 ,1e-13),
(8 ,0.5,8,-0.482132838467071 ,2e-05),
(10,0.5,8,-0.473113399629092 ,5e-05),
(16,0.5,8,-0.460770570536169 ,5e-04),
#(24,0.5,8,-0.454476296840250 ,2e-03)
])
def test_MPS_VariationalGS(L,S,Dfinal,Eexpected,E2expected):

    Jxy=1.0;Jz=1.0;Hz=0.0;
    epsNorm=1e-13;epsSV=1e-13
    epsE=1e-11
    Dstart=2;maxIter=100
    f=TN.Factory.GetFactory()
    logger=f.MakeSPDLogger(1)
    H=f.Make1D_NN_HeisenbergHamiltonian(L,S,Jxy,Jz,Hz)
    psi=H.CreateMPS(Dstart,epsNorm,epsSV)
    psi.InitializeWith(TN.Random)

    eps=TN.Epsilons()
    isched=TN.IterationSchedule()
    isched.Insert(TN.IterationScheduleLine(maxIter,Dfinal,eps))
    E2=psi.FindVariationalGroundState(H,isched)
    E1=psi.GetExpectation(H)/(L-1)
    assert E1-Eexpected <= epsE
    assert abs(E1-Eexpected) <= epsE
    assert E2<E2expected

@pytest.mark.parametrize("L, S, Eexpected",[
#L S         E
(2 ,0.5,-0.750000000000000),
(2 ,1.0,-2.000000000000000),
(2 ,1.5,-3.750000000000000),
(2 ,2.0,-6.000000000000000),
(2 ,2.5,-8.750000000000000),
(3 ,0.5,-0.500000000000000),
(3 ,1.0,-1.500000000000000),
(3 ,1.5,-3.000000000000000),
(3 ,2.0,-5.000000000000000),
(3 ,2.5,-7.500000000000000),
(4 ,0.5,-0.538675134594813),
(4 ,1.0,-1.548583770354863),
(5 ,0.5,-0.481971563329499),
(6 ,0.5,-0.498715426777585),
(7 ,0.5,-0.472706613447776),
(8 ,0.5,-0.482133228383984),
(10,0.5,-0.473115023031432),
(12,0.5,-0.467462784803685),
])
def test_FullWavefunctionGS(L,S,Eexpected):

    Jxy=1.0;Jz=1.0;Hz=0.0;
    epsE=1e-11
    f=TN.Factory.GetFactory()
    logger=f.MakeSPDLogger(1)
    H=f.Make1D_NN_HeisenbergHamiltonian(L,S,Jxy,Jz,Hz)
    psi=H.CreateFullState()

    E1a=psi.FindGroundState(H,epsE)
    assert E1a==psi.GetE()
    E1=psi.GetE()/(L-1)
    assert E1-Eexpected <= epsE
    assert abs(E1-Eexpected) <= epsE

@pytest.mark.parametrize("L, S, Dfinal, Eexpected, epsE, E2expected, TO",[
#L S  D   E                   E2
(2 ,0.5,2,-0.750000000000000 ,1e-09,1e-9,TN.FirstOrder),
(2 ,0.5,2,-0.750000000000000 ,1e-09,1e-9,TN.SecondOrder),
(2 ,0.5,2,-0.750000000000000 ,1e-09,1e-9,TN.FourthOrder),
(2 ,1.0,3,-2.000000000000000 ,1e-09,1e-9,TN.FirstOrder),
(2 ,1.0,3,-2.000000000000000 ,1e-09,1e-9,TN.SecondOrder),
(2 ,1.0,3,-2.000000000000000 ,1e-09,1e-9,TN.FourthOrder),
(2 ,1.5,4,-3.750000000000000 ,1e-09,1e-9,TN.FirstOrder),
(2 ,1.5,4,-3.750000000000000 ,1e-09,1e-9,TN.SecondOrder),
(2 ,1.5,4,-3.750000000000000 ,1e-09,1e-9,TN.FourthOrder),
(2 ,2.0,5,-6.000000000000000 ,1e-09,1e-9,TN.FirstOrder),
(2 ,2.0,5,-6.000000000000000 ,1e-09,1e-9,TN.SecondOrder),
(2 ,2.0,5,-6.000000000000000 ,1e-09,1e-9,TN.FourthOrder),
(2 ,2.5,6,-8.750000000000000 ,1e-09,1e-9,TN.FirstOrder),
(2 ,2.5,6,-8.750000000000000 ,1e-09,1e-9,TN.SecondOrder),
(2 ,2.5,6,-8.750000000000000 ,1e-09,1e-9,TN.FourthOrder),
(3 ,0.5,2,-0.500000000000000 ,1e-07,2e-7,TN.FirstOrder),
(3 ,0.5,2,-0.500000000000000 ,1e-10,1e-9,TN.SecondOrder),
(3 ,0.5,2,-0.500000000000000 ,1e-10,1e-9,TN.FourthOrder),
(3 ,1.0,3,-1.500000000000000 ,1e-06,1e-6,TN.FirstOrder),
(3 ,1.0,3,-1.500000000000000 ,1e-09,2e-9,TN.SecondOrder),
(3 ,1.0,3,-1.500000000000000 ,1e-10,1e-9,TN.FourthOrder),
(3 ,1.5,4,-3.000000000000000 ,5e-07,2e-6,TN.FirstOrder),
(3 ,1.5,4,-3.000000000000000 ,1e-09,5e-9,TN.SecondOrder),
#(3 ,1.5,4,-3.000000000000000 ,1e-9,1e-9,TN.FourthOrder), hangs
(3 ,2.0,5,-5.000000000000000 ,2e-09,5e-9,TN.SecondOrder),
#(3 ,2.5,6,-7.500000000000000 ,1e-8,1e-8,TN.SecondOrder), hangs
(4 ,0.5,4,-0.538675134594813 ,1e-09,2e-9,TN.SecondOrder),
(4 ,0.5,4,-0.538675134594813 ,1e-09,2e-9,TN.FourthOrder),
(4 ,1.0,9,-1.548583770354863 ,1e-09,5e-9,TN.SecondOrder),
#(4 ,1.0,9,-1.548583770354863 ,1e-9,5e-9,TN.FourthOrder), very slow
(5 ,0.5,4,-0.481971563329499 ,1e-09,3e-9,TN.SecondOrder),
(5 ,0.5,4,-0.481971563329499 ,1e-09,3e-9,TN.FourthOrder),
(6 ,0.5,8,-0.498715426777585 ,1e-09,5e-9,TN.SecondOrder),
(6 ,0.5,8,-0.498715426777585 ,1e-09,3e-9,TN.FourthOrder),
(7 ,0.5,8,-0.472706613447776 ,1e-09,5e-9,TN.SecondOrder),
(7 ,0.5,8,-0.472706613447776 ,1e-09,3e-9,TN.FourthOrder),
(8 ,0.5,8,-0.482132838467071 ,1e-09,2e-5,TN.SecondOrder),
(10,0.5,8,-0.473113399629092 ,1e-09,5e-5,TN.SecondOrder),
(16,0.5,8,-0.460770570536169 ,1e-09,5e-4,TN.SecondOrder),
])
def test_TEBD_GS(L,S,Dfinal,Eexpected,epsE,E2expected,TO):

    Jxy=1.0;Jz=1.0;Hz=0.0;
    epsNorm=1e-13;epsSV=1e-13
    Dstart=2;deltaD=1;maxIter=1000
    f=TN.Factory.GetFactory()
    logger=f.MakeSPDLogger(1)
    H=f.Make1D_NN_HeisenbergHamiltonian(L,S,Jxy,Jz,Hz)
    psi=H.CreateMPS(Dstart,epsNorm,epsSV)
    psi.InitializeWith(TN.Random)
    psi.Normalize(TN.DRight)

    eps=TN.Epsilons()
    eps.itsMPOCompressEpsilon=1e-14;
    eps.itsDelatNormEpsilon=1e-5;
    eps.itsMPSCompressEpsilon=0;

    isched=TN.IterationSchedule()
    eps.itsDelatEnergy1Epsilon=1e-5;
    isched.Insert(TN.IterationScheduleLine(maxIter,Dfinal,deltaD,0,0.5,TO,eps));
    eps.itsDelatEnergy1Epsilon=1e-7;
    isched.Insert(TN.IterationScheduleLine(maxIter,Dfinal,deltaD,0,0.2,TO,eps));
    eps.itsDelatEnergy1Epsilon=1e-9;
    isched.Insert(TN.IterationScheduleLine(maxIter,Dfinal,deltaD,1,0.1,TO,eps));
    eps.itsDelatEnergy1Epsilon=1e-11;
    isched.Insert(TN.IterationScheduleLine(maxIter,Dfinal,deltaD,2,0.05,TO,eps));
    isched.Insert(TN.IterationScheduleLine(maxIter,Dfinal,deltaD,3,0.02,TO,eps));
    isched.Insert(TN.IterationScheduleLine(maxIter,Dfinal,deltaD,4,0.01,TO,eps));
    isched.Insert(TN.IterationScheduleLine(maxIter,Dfinal,deltaD,4,0.005,TO,eps));
    isched.Insert(TN.IterationScheduleLine(maxIter,Dfinal,deltaD,4,0.002,TO,eps));
    isched.Insert(TN.IterationScheduleLine(maxIter,Dfinal,deltaD,4,0.001,TO,eps));

    isched.Insert(TN.IterationScheduleLine(maxIter,Dfinal,eps))
    psi.FindiTimeGroundState(H,isched)
    E1=psi.GetExpectation(H)

    H2=H.CreateH2Operator();
    E2=psi.GetExpectation(H2)-E1*E1;
    E1=E1/(L-1)

    assert E1-Eexpected <= epsE
    assert abs(E1-Eexpected) <= epsE
    assert E2<E2expected
