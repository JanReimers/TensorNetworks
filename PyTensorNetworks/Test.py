import PyTensorNetworks as TN
logger=TN.SPDLogger(1);
f=TN.Factory.GetFactory();
H=f.Make1D_NN_HeisenbergHamiltonian(9,0.5,1.,1.,0.);
psi=H.CreateMPS(4,1e-12,1e-12);
psi.InitializeWith(TN.Random);

eps=TN.Epsilons();
isched=TN.IterationSchedule();
isched.Insert(TN.IterationScheduleLine(100,4,eps))
psi.FindVariationalGroundState(H,isched)

