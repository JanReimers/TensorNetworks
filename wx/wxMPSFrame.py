import wx
import wx.html
import PyPlotting
import PyTensorNetworks
import threading
import time

class GUIHandler:
    def __init__(self):
        pass

    def Update(level,isite):
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

    def DoneOneStep(self,level,isite=-1):
        #print("Enter DoOneStep isite=",isite)
        wx.CallAfter(self.GUIhandler.Update,level,isite)
        wx.YieldIfNeeded() #this is supposed to give the GUI a time slice but it's NOT working!!  Same for wxYield
        time.sleep(0.1) #this is kludge to give the GUI a time slice.
        #el=wx.EventLoopBase.GetActive()
        #if not el==None:
        #    print ("Pending=",el.Pending())
        #    while el.Pending():
        #        time.sleep(0.01)


        if self.Started and self.Step>0 :
            self.Step=self.Step-1
            if self.Step==0:
                self.Pause=True
        while (self.Pause):
            time.sleep(0.1)

        #print("Exit  DoOneStep")

    def OnPlay(self,e):
        print("OnPlay")
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

    def CreateMPS(self,Hamiltonian):
        D=self.DControl.GetValue()
        return Hamiltonian.CreateMPS(D)

class wxMPSStatusPanel(wx.Panel):
    def __init__(self,parent):
        super().__init__(parent)

        self.colours={"A":wx.BLUE,"B":wx.RED, "M":wx.BLACK, "I":wx.LIGHT_GREY }
        self.text=wx.TextCtrl(self,style=wx.TE_MULTILINE,size=(400,-1))
        self.L=0

    def NewLattice(self,L):
        self.L=L
        fgsizer=wx.FlexGridSizer(L+1,2,0) #2 rows, L+1 columns, gap=5
        fgsizer.Add(wx.StaticText(self,label='Site #:'),flag=wx.ALIGN_RIGHT)
        for i in range(1,L+1):
            fgsizer.Add(wx.StaticText(self,label=str(i)),flag=wx.ALIGN_CENTER)
        fgsizer.Add(wx.StaticText(self,label='Norm Status:'))
        self.NormTextControls=[]
        for i in range(1,L+1):
            #We don't actually want multiline, but that is only way to get colours t work!!??
            tc=wx.TextCtrl(self,style=wx.TE_MULTILINE|wx.TE_CENTRE|wx.BORDER_NONE,size=(30,20))
            tc.AppendText("   ")
            self.NormTextControls.append(tc)
            fgsizer.Add(tc)
        fgsizer.SetSizeHints(self)
        self.SetSizer(fgsizer)

    def UpdateSite(self,MPS,isite):
        status=MPS.GetNormStatus(isite)
        tc=self.NormTextControls[isite]
        tc.Clear()
        letter=status[0]
        colour=self.colours[letter]
        tc.SetDefaultStyle(wx.TextAttr(colour))
        tc.AppendText(status)

    def Update(self,MPS):
        for isite in range(0,self.L):
            self.UpdateSite(MPS,isite)

ID_START   =wx.NewIdRef()
ID_PLAY    =wx.NewIdRef()
ID_PAUSE   =wx.NewIdRef()
ID_STEP    =wx.NewIdRef()
ID_STOP    =wx.NewIdRef()
ID_RESTART =wx.NewIdRef()

class wxMPSControlsPanel(wx.Panel):
    def __init__(self,parent,supervisor):
        super().__init__(parent)
        b0=wx.Button(self,ID_PLAY   ,"Start")
        b1=wx.Button(self,ID_PLAY   ,"Play")
        b2=wx.Button(self,ID_PAUSE  ,"Pause")
        b3=wx.Button(self,ID_STEP   ,"Step")
        b4=wx.Button(self,ID_STOP   ,"Stop")
        b5=wx.Button(self,ID_RESTART,"Restart")
        sizer=wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(b0)
        sizer.Add(b1)
        sizer.Add(b2)
        sizer.Add(b3)
        sizer.Add(b4)
        sizer.Add(b5)
        self.SetSizer(sizer)
        self.Layout()

        b0.Bind(wx.EVT_BUTTON,parent.OnStart)
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

        self.supervisor=MPSSupervisor(self) #conrol panel needs this

        self.BuildLayout()

        self.Hamiltonian=self.HamiltonianPanel.MakeHamiltonian(self.TNFactory)
        self.MPS=self.ApproximationsPanel.CreateMPS(self.Hamiltonian)
        self.MPS.InitializeWith(PyTensorNetworks.Random)
        self.MPS.Insert(self.graphs) #Tell the MPS where to plot data
        self.statusPanel.NewLattice(self.Hamiltonian.GetL())
        self.statusPanel.Update(self.MPS)

        self.inUpdate=False



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

    def OnStart(self,e):
        x = threading.Thread(target=self.FindGroundState, args=(), daemon=True)
        x.start()
        self.supervisor.OnPlay(e)

    def FindGroundState(self):
        n=self.MPS.FindGroundState(self.Hamiltonian,20,1e-8,self.supervisor)
        print("FindGroundState nsweeps=",n)
        #wx.CallAfter(self.graphs.ReplotActiveGraph)



    def OnRestart(self,e):
        print("Restart")

    def Update(self,level,isite):
        if (not self.inUpdate):
            self.inUpdate=True
            #print("Update level=",level)
            if level==0: #end of each sweep
                self.graphs.ReplotActiveGraph()
            #if level=1: #finished refine on one site

            if level==2: #individual steps on one site
                self.statusPanel.UpdateSite(self.MPS,isite)

            self.inUpdate=False
        else:
            print("Update clash")



