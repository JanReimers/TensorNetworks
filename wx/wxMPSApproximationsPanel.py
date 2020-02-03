import wx
import PyTensorNetworks

EVT_NEW_MPS = wx.PyEventBinder(wx.NewEventType(), 1)
EVT_NEW_EPS = wx.PyEventBinder(wx.NewEventType(), 1)

class wxMPSApproximationsPanel(wx.Panel):
    def __init__(self,parent):
        super().__init__(parent)
        dx=120
        self.DControl=wx.SpinCtrl(self,min=1,max=10000,initial=4,size=(dx,-1))
        self.MaxIterControl=wx.SpinCtrl(self,min=1,max=1000,initial=20,size=(dx,-1))
        self.EnergyConvergenceControl=wx.SpinCtrl(self,min=-16,max=0,initial=-12,size=(dx,-1))
        self. EigenConvergenceControl=wx.SpinCtrl(self,min=-16,max=0,initial=-12,size=(dx,-1))
        self.   EnergyVarienceControl=wx.SpinCtrl(self,min=-16,max=0,initial=-12,size=(dx,-1))
        self. NormVerificationControl=wx.SpinCtrl(self,min=-16,max=0,initial=-12,size=(dx,-1))
        self.          SVDRankControl=wx.SpinCtrl(self,min=-16,max=0,initial=-12,size=(dx,-1))
        self.     SparseMatrixControl=wx.SpinCtrl(self,min=-16,max=0,initial=-12,size=(dx,-1))

        fgsizer=wx.FlexGridSizer(2,5,5)
        fgsizer.Add(wx.StaticText(self,label='Bond Dimension D:'),flag=wx.ALIGN_CENTER_VERTICAL|wx.ALIGN_RIGHT)
        fgsizer.Add(self.DControl)
        fgsizer.Add(wx.StaticText(self,label='Max iterations:'),flag=wx.ALIGN_CENTER_VERTICAL|wx.ALIGN_RIGHT)
        fgsizer.Add(self.MaxIterControl)
        fgsizer.Add(wx.StaticText(self,label='Energy convergence eps: 1e'),flag=wx.ALIGN_CENTER_VERTICAL|wx.ALIGN_RIGHT)
        fgsizer.Add(self.EnergyConvergenceControl)
        fgsizer.Add(wx.StaticText(self,label='<E^2>-<E>^2) eps: 1e'),flag=wx.ALIGN_CENTER_VERTICAL|wx.ALIGN_RIGHT)
        fgsizer.Add(self.EnergyVarienceControl)
        fgsizer.Add(wx.StaticText(self,label='Eigen solver eps: 1e'),flag=wx.ALIGN_CENTER_VERTICAL|wx.ALIGN_RIGHT)
        fgsizer.Add(self.EigenConvergenceControl)
        fgsizer.Add(wx.StaticText(self,label='Normalization eps: 1e'),flag=wx.ALIGN_CENTER_VERTICAL|wx.ALIGN_RIGHT)
        fgsizer.Add(self.NormVerificationControl)
        fgsizer.Add(wx.StaticText(self,label='SVD rank eps: 1e'),flag=wx.ALIGN_CENTER_VERTICAL|wx.ALIGN_RIGHT)
        fgsizer.Add(self.SVDRankControl)
        fgsizer.Add(wx.StaticText(self,label='Sparse matrix eps: 1e'),flag=wx.ALIGN_CENTER_VERTICAL|wx.ALIGN_RIGHT)
        fgsizer.Add(self.SparseMatrixControl)

        bsizer=wx.BoxSizer(wx.VERTICAL) #only purpose is to get a border
        bsizer.Add(fgsizer,border=5,flag=wx.ALL)
        self.SetSizer(bsizer)

        self.DControl.Bind(wx.EVT_SPINCTRL,self.OnChange) #this should capture changes from only the D spin control

    def OnChange(self,e):
        event = wx.PyCommandEvent(EVT_NEW_MPS.typeId, self.GetId())
        self.GetEventHandler().ProcessEvent(event)

    def CreateMPS(self,Hamiltonian):
        D=self.DControl.GetValue()
        return Hamiltonian.CreateMPS(D,self.GetEpsilons())

    def GetEpsilons(self):
        eps=PyTensorNetworks.Epsilons()
        eps.itsEnergyConvergenceEpsilon=pow(10,self.EnergyConvergenceControl.GetValue())
        eps.itsEnergyVarienceEpsilon   =pow(10,self.EnergyVarienceControl.GetValue())
        eps.itsEigenConvergenceEpsilon =pow(10,self.EigenConvergenceControl.GetValue())
        eps.itsNormalizationEpsilon    =pow(10,self.NormVerificationControl.GetValue())
        eps.itsSingularValueZeroEpsilon=pow(10,self.SVDRankControl.GetValue())
        eps.itsSparseMatrixEpsilon     =pow(10,self.SparseMatrixControl.GetValue())
        return eps

    def GetMaxIter(self):
        return self.MaxIterControl.GetValue()

