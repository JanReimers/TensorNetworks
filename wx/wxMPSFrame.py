import wx
import wx.html
import PyPlotting
import PyTensorNetworks
import _thread
import time

class GUIHandler:
    def __init__(self):
        pass

    def Update(level):
        pass

class MPSSupervisor(PyTensorNetworks.LRPSupervisor):
    def __init__(self,_GUIhandler):
        super().__init__()

        self.GUIhandler=_GUIhandler
        self.Pause=True
        self.Started=False
        self.Step=0
    def ReadyToStart(self):
        self.Started=True

    def DoneOneStep(self,level):
        wx.CallAfter(self.GUIhandler.Update,(level))
        print("Enter DoOneStep")
        if self.Started and self.Step>0 :
            self.Step=self.Step-1
            if self.Step==0:
                self.Pause=True
        while (self.Pause):
            time.sleep(0.1)
        print("Exit  DoOneStep")

    def OnPlay(self,e):
        self.Started=True
        self.Pause=False

    def OnPause(self,e):
        self.Pause=True

    def OnStep(self,e):
        self.Step=1
        self.Pause=False

    def OnStop(self,e):
        self.Pause=True
        self.Started=False



class wxHamiltonianPanel(wx.Panel):
    def __init__(self,parent):
        super().__init__(parent)

        sizer=wx.BoxSizer(wx.VERTICAL)

        Hexpression = wx.Image(name='HeisenbergHamiltonian0x.png',type=wx.BITMAP_TYPE_PNG)
        HexpressionBM = wx.Bitmap(Hexpression)
        HexpressionSBM = wx.StaticBitmap(self, wx.ID_ANY, HexpressionBM)
        sizer.Add(HexpressionSBM,border=10,flag=wx.ALL)

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


class wxMPSApproximationsPanel(wx.Panel):
    def __init__(self,parent):
        super().__init__(parent)
        self.DControl=wx.SpinCtrl(self,min=1,max=10000,initial=2)
        self.EnergyConvergenceControl=wx.SpinCtrl(self,min=-16,max=0,initial=-12)
        self. EigenConvergenceControl=wx.SpinCtrl(self,min=-16,max=0,initial=-12)
        self. NormVerificationControl=wx.SpinCtrl(self,min=-16,max=0,initial=-12)
        self.          SVDRankControl=wx.SpinCtrl(self,min=-16,max=0,initial=-12)

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

    def CreateMPS(self,Hamiltonian):
        D=self.DControl.GetValue()
        return Hamiltonian.CreateMPS(D)

class wxMPSStatusPanel(wx.Panel):
    def __init__(self,parent):
        super().__init__(parent)

        self.colours={"A":wx.BLUE,"B":wx.RED, "M":wx.BLACK, "I":wx.LIGHT_GREY }
        self.text=wx.TextCtrl(self,style=wx.TE_MULTILINE,size=(180,-1))

    def Update(self,MPS):
        status=MPS.GetNormStatus()
        print(status)
        L=len(status)
        #self.text=wx.TextCtrl(self,style=wx.TE_MULTILINE,size=(20*L,-1))
        #self.text.SetSize((20*L,-1))
        self.text.Clear()
        for i in range(0,L-1):
            letters=status[i]
            letter=letters[0]
            colour=self.colours[letter]
            self.text.SetDefaultStyle(wx.TextAttr(colour))
            self.text.AppendText(letters)

ID_PLAY   =wx.NewIdRef()
ID_PAUSE   =wx.NewIdRef()
ID_STEP    =wx.NewIdRef()
ID_STOP    =wx.NewIdRef()
ID_RESTART =wx.NewIdRef()

class wxMPSControlsPanel(wx.Panel):
    def __init__(self,parent,supervisor):
        super().__init__(parent)
        b1=wx.Button(self,ID_PLAY   ,"Play")
        b2=wx.Button(self,ID_PAUSE  ,"Pause")
        b3=wx.Button(self,ID_STEP   ,"Step")
        b4=wx.Button(self,ID_STOP   ,"Stop")
        b5=wx.Button(self,ID_RESTART,"Restart")
        sizer=wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(b1)
        sizer.Add(b2)
        sizer.Add(b3)
        sizer.Add(b4)
        sizer.Add(b5)
        self.SetSizer(sizer)
        self.Layout()

        b1.Bind(wx.EVT_BUTTON,supervisor.OnPlay)
        b2.Bind(wx.EVT_BUTTON,supervisor.OnPause)
        b3.Bind(wx.EVT_BUTTON,supervisor.OnStep)
        b4.Bind(wx.EVT_BUTTON,supervisor.OnStop)
        b5.Bind(wx.EVT_BUTTON,parent.OnRestart)

class MPSFrame(wx.Frame,GUIHandler):
    def __init__(self,*args,**kwargs):
        super(MPSFrame,self).__init__(*args,**kwargs)

        self.PlottingFactory=PyPlotting.wxFactory.GetFactory()
        self.TNFactory=PyTensorNetworks.Factory.GetFactory()

        self.BuildMenus()

        self.supervisor=MPSSupervisor(self)

        self.BuildLayout()

        #self.Hamiltonian=self.TNFactory.Make1D_NN_HeisenbergHamiltonian(9,1,1.0,1.0,1.0,0)
        self.Hamiltonian=self.HamiltonianPanel.MakeHamiltonian(self.TNFactory)
        self.MPS=self.ApproximationsPanel.CreateMPS(self.Hamiltonian)
        self.MPS.Insert(self.graphs) #Tell the MPS where to plot data
        self.statusPanel.Update(self.MPS)
        #self.Bind(wx.EVT_BUTTON,self.ReplotActiveGraph,id=PyPlotting.ID_ReplotActiveGraph)
        _thread.start_new_thread(self.FindGroundState,())


    def BuildMenus(self):
        menuBar=wx.MenuBar()
        fileMenu=wx.Menu()
        qmi=wx.MenuItem(fileMenu,wx.ID_EXIT,'&Quit\tCtrl+Q')
        qmi.SetBitmap(wx.ArtProvider.GetBitmap(wx.ART_CLOSE))
        fileMenu.Append(qmi)
        menuBar.Append(fileMenu, '&File')
        self.SetMenuBar(menuBar)
        self.Bind(wx.EVT_MENU,self.OnQuit,id=wx.ID_EXIT)

    def BuildLayout(self):

        #
        #  Build up the left notebook
        #
        self.LeftNB=wx.Notebook(self)
        self.HamiltonianPanel=wxHamiltonianPanel(self.LeftNB)
        self.ApproximationsPanel=wxMPSApproximationsPanel(self.LeftNB)
        self.LeftNB.AddPage(self.HamiltonianPanel,'Hamiltonian')
        self.LeftNB.AddPage(self.ApproximationsPanel,'MPS Approximations')
        #
        #  Build up the right side
        #
        self.graphs=self.PlottingFactory.MakewxMultiGraph(self) #tabbed notebook of graphs
        graphsPanel=self.PlottingFactory.GetPanel(self.graphs) #dynamic cross cast to panel for sizing
        self.statusPanel=wxMPSStatusPanel(self)
        controlsPanel=wxMPSControlsPanel(self,self.supervisor)
        rightsizer=wx.BoxSizer(wx.VERTICAL)
        rightsizer.Add(graphsPanel,proportion=1,flag=wx.EXPAND) #Give the graphs as much space as possible
        rightsizer.Add(self.statusPanel,proportion=0)
        rightsizer.Add(controlsPanel,proportion=0)
        #
        #  Combine left nad right and size everything
        #
        allsizer=wx.BoxSizer(wx.HORIZONTAL)
        allsizer.Add(self.LeftNB)
        allsizer.Add(rightsizer,proportion=1,flag=wx.EXPAND) #Again give the graphs as much space as possible
        self.SetSizer(allsizer)
        self.Layout()
        self.Centre()


    def OnQuit(self, e):
        self.Close()


    def FindGroundState(self):
        n=self.MPS.FindGroundState(self.Hamiltonian,20,1e-8,self.supervisor)
        print("FindGroundState nsweeps=",n)
        #wx.CallAfter(self.graphs.ReplotActiveGraph)



    def OnRestart(self,e):
        print("Restart")

    def Update(self,level):
        print("Update level=",level)
        self.statusPanel.Update(self.MPS)
        self.graphs.ReplotActiveGraph()


