import wx
import PyPlotting
import PyTensorNetworks
import threading
import time
import ctypes

from wxHamiltonianPanel import *
from wxMPSApproximationsPanel import *
from wxMPSStatusPanel import *

class GUIHandler:
    def __init__(self):
        pass

    def Update(level,currentOperation,isite):
        pass

    def OnStop():
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

    def DoneOneStep(self,level,currentOperation,isite=-1):
        #print("Enter DoOneStep isite=",isite)
        wx.CallAfter(self.GUIhandler.Update,level,currentOperation,isite)
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

    def get_id(self):
        # returns id of the respective thread
        if hasattr(self, '_thread_id'):
            return self._thread_id
        for id, thread in threading._active.items():
            if thread is self:
                return id

    def raise_exception(self):
        #killing threads is tough, one way is to raise and excpetion
        thread_id = self.get_id()
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id,
              ctypes.py_object(SystemExit))
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
            print('Exception raise failure')

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
        self.raise_exception()
        self.GUIhandler.OnStop(e)

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

        self.Bind(EVT_NEW_HAMILTONIAN, self.OnNewHamiltonian)
        self.Bind(EVT_NEW_MPS        , self.OnNewMPS)

        self.inUpdate=False
        self.OnNewHamiltonian(None)



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

    def OnNewHamiltonian(self,e):
        self.Hamiltonian=self.HamiltonianPanel.MakeHamiltonian(self.TNFactory)
        self.statusPanel.NewLattice(self.Hamiltonian.GetL())
        self.OnNewMPS(e)

    def OnNewMPS(self,e):
        self.MPS=self.ApproximationsPanel.CreateMPS(self.Hamiltonian)
        self.MPS.InitializeWith(PyTensorNetworks.Random)
        self.graphs.Clear()
        self.MPS.Insert(self.graphs) #Tell the MPS where to plot data
        self.graphs.ReplotActiveGraph()
        self.statusPanel.Update(self.MPS)

    def OnStart(self,e):

        self.crunchThread = threading.Thread(target=self.FindGroundState, args=(), daemon=True)
        self.crunchThread.start()
        self.supervisor.OnPlay(e)

    def OnStop(self,e):
        #self.crunchThread.join()
        print("Crunch thread stopped")
        del self.MPS
        self.OnNewMPS(e)

    def FindGroundState(self):
        dE=self.MPS.FindGroundState(self.Hamiltonian,20,1e-8,self.supervisor)
        print("FindGroundState finished, <E^2>-<E>^2=",dE)




    def OnRestart(self,e):
        print("Restart")

    def Update(self,level,currentOperation,isite):
        if (not self.inUpdate):
            self.inUpdate=True
            #print("Update level=",level," current op=",currentOperation)
            self.statusPanel.UpdateCurrentOperation(currentOperation)
            self.graphs.ReplotActiveGraph()
            #if level==0: #end of each sweep
            #    self.graphs.ReplotActiveGraph()
            #if level==1: #finished refine on one site
            #    self.graphs.ReplotActiveGraph()

            if level==2: #individual steps on one site
                self.statusPanel.UpdateSite(self.MPS,isite)

            self.inUpdate=False
        else:
            print("Update clash")



