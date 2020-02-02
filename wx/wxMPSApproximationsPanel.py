import wx
import PyTensorNetworks

EVT_NEW_MPS = wx.PyEventBinder(wx.NewEventType(), 1)
EVT_NEW_EPS = wx.PyEventBinder(wx.NewEventType(), 1)

class wxMPSApproximationsPanel(wx.Panel):
    def __init__(self,parent):
        super().__init__(parent)
        dx=120
        self.DControl=wx.SpinCtrl(self,min=1,max=10000,initial=4,size=(dx,-1))
        self.EnergyConvergenceControl=wx.SpinCtrl(self,min=-16,max=0,initial=-12,size=(dx,-1))
        self. EigenConvergenceControl=wx.SpinCtrl(self,min=-16,max=0,initial=-12,size=(dx,-1))
        self. NormVerificationControl=wx.SpinCtrl(self,min=-16,max=0,initial=-12,size=(dx,-1))
        self.          SVDRankControl=wx.SpinCtrl(self,min=-16,max=0,initial=-12,size=(dx,-1))

        fgsizer=wx.FlexGridSizer(2,5,5)
        fgsizer.Add(wx.StaticText(self,label='Bond Dimension D:'))
        fgsizer.Add(self.DControl)
        fgsizer.Add(wx.StaticText(self,label='Energy convergence eps: 1e-'))
        fgsizer.Add(self.EnergyConvergenceControl)
        fgsizer.Add(wx.StaticText(self,label='Eigen solver eps: 1e-'))
        fgsizer.Add(self.EigenConvergenceControl)
        fgsizer.Add(wx.StaticText(self,label='Normalization eps: 1e-'))
        fgsizer.Add(self.NormVerificationControl)
        fgsizer.Add(wx.StaticText(self,label='SVD rank eps: 1e-'))
        fgsizer.Add(self.SVDRankControl)
        fgsizer.SetSizeHints(self)
        self.SetSizer(fgsizer)

        self.DControl.Bind(wx.EVT_SPINCTRL,self.OnChange) #this should capture changes from only the D spin control

    def OnChange(self,e):
        event = wx.PyCommandEvent(EVT_NEW_MPS.typeId, self.GetId())
        self.GetEventHandler().ProcessEvent(event)

    def CreateMPS(self,Hamiltonian):
        D=self.DControl.GetValue()
        return Hamiltonian.CreateMPS(D)

