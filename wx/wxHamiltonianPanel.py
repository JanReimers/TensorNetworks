import wx
import wx.html
import PyTensorNetworks

EVT_NEW_HAMILTONIAN = wx.PyEventBinder(wx.NewEventType(), 1) # the 1 is number of expected IDs for the __call__ interface

class wxHamiltonianPanel(wx.Panel):
    def __init__(self,parent):
        super().__init__(parent)

        sizer=wx.BoxSizer(wx.VERTICAL)

        Hexpression0 = wx.Image(name='HeisenbergHamiltonian0x.png',type=wx.BITMAP_TYPE_PNG)
        HexpressionSBM0 = wx.StaticBitmap(self, wx.ID_ANY, wx.Bitmap(Hexpression0))
        sizer.Add(HexpressionSBM0,border=10,flag=wx.ALL)
        Hexpression1 = wx.Image(name='HeisenbergHamiltonian1x.png',type=wx.BITMAP_TYPE_PNG)
        HexpressionSBM1 = wx.StaticBitmap(self, wx.ID_ANY, wx.Bitmap(Hexpression1))
        sizer.Add(HexpressionSBM1,border=10,flag=wx.ALL)

        fgsizer=wx.FlexGridSizer(2,5,5)
        self.lattice=wx.SpinCtrl(self,min=2,max=10000,initial=9)
        latticet=wx.StaticText(self,label='#Lattice sites L: ')
        fgsizer.Add(latticet,border=5,flag=wx.ALL|wx.ALIGN_CENTER_VERTICAL)
        fgsizer.Add(self.lattice)

        self.spin=wx.SpinCtrlDouble(parent=self,min=0.5,max=9.5,initial=0.5,inc=0.5)
        spint=wx.StaticText(self,label='Spin S: ')
        fgsizer.Add(spint,border=5,flag=wx.ALL|wx.ALIGN_CENTER_VERTICAL)
        fgsizer.Add(self.spin)

        self.Jxy=self.JxAddrow('J<sub>xy</sub>',1.0,fgsizer)
        self.Jz =self.JxAddrow('J<sub>z</sub>' ,1.0,fgsizer)
        self.hz =self.JxAddrow('h<sub>z</sub>' ,0.0,fgsizer)

        sizer.Add(fgsizer)
        sizer.SetSizeHints(self)
        self.SetSizer(sizer)

        self.Bind(wx.EVT_SPINCTRL,self.OnChange) #this should capture changes from any of spin controls
        self.Bind(wx.EVT_SPINCTRLDOUBLE,self.OnChange) #this should capture changes from any of spin controls

    def OnChange(self,e):
        event = wx.PyCommandEvent(EVT_NEW_HAMILTONIAN.typeId, self.GetId())
        self.GetEventHandler().ProcessEvent(event)

    def JxAddrow(self,text,initialValue,sizer):
        Jexchange=wx.SpinCtrlDouble(parent=self,min=-10,max=10,initial=initialValue,inc=0.1)
        W,H=Jexchange.GetSize()
        html = wx.html.HtmlWindow(self,size=wx.Size(30,H) ,style=wx.html.HW_SCROLLBAR_NEVER | wx.html.HW_NO_SELECTION)
        html.SetPage("<html><body><i>"+text+":</i></body></html>")
        sizer.Add(html,flag=wx.ALIGN_TOP|wx.ALIGN_RIGHT )
        sizer.Add(Jexchange)
        return Jexchange

    def MakeHamiltonian(self,TNfactory):
        L  =self.lattice.GetValue()
        S =self.spin.GetValue()
        Jxy=self.Jxy.GetValue()
        Jz =self.Jz .GetValue()
        hz =self.hz .GetValue()
        return TNfactory.Make1D_NN_HeisenbergHamiltonian(L,S,Jxy,Jz,hz)

