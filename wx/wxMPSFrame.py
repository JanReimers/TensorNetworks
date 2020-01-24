import wx
import wx.html
import PyPlotting

class wxHamiltonianPanel(wx.Panel):
    def __init__(self,parent):
        super().__init__(parent)
        
        sizer=wx.BoxSizer(wx.VERTICAL)

        Hexpression = wx.Image(name='HeisenbergHamiltonian0x.png',type=wx.BITMAP_TYPE_PNG)
        HexpressionBM = wx.Bitmap(Hexpression)
        HexpressionSBM = wx.StaticBitmap(self, wx.ID_ANY, HexpressionBM)
        sizer.Add(HexpressionSBM,border=10,flag=wx.ALL)
        
        fgsizer=wx.FlexGridSizer(2,5,5)
        lattice=wx.SpinCtrl(self,min=2,max=10000,initial=10)
        latticet=wx.StaticText(self,label='#Lattice sites L: ')
        fgsizer.Add(latticet,border=5,flag=wx.ALL|wx.ALIGN_CENTER_VERTICAL)
        fgsizer.Add(lattice)
        
        self.JxAddrow('J<sub>x</sub>',fgsizer)
        self.JxAddrow('J<sub>y</sub>',fgsizer)
        self.JxAddrow('J<sub>z</sub>',fgsizer)
        self.JxAddrow('h<sub>z</sub>',fgsizer)
       
        sizer.Add(fgsizer)
        sizer.SetSizeHints(self)
        self.SetSizer(sizer)
        
    def JxAddrow(self,text,sizer):
        Jexchange=wx.SpinCtrlDouble(parent=self,min=-10,max=10,initial=1.0,inc=0.1)
        W,H=Jexchange.GetSize()
        html = wx.html.HtmlWindow(self,size=wx.Size(30,H) ,style=wx.html.HW_SCROLLBAR_NEVER | wx.html.HW_NO_SELECTION)
        html.SetPage("<html><body><i>"+text+":</i></body></html>")
        sizer.Add(html,flag=wx.ALIGN_TOP|wx.ALIGN_RIGHT )
        sizer.Add(Jexchange)
        
class wxMPSApproximationsPanel(wx.Panel):
    def __init__(self,parent):
        super().__init__(parent)
        fgsizer=wx.FlexGridSizer(2,5,5)
        fgsizer.Add(wx.StaticText(self,label='Bond Dimension D:'))
        fgsizer.Add(wx.SpinCtrl(self,min=1,max=10000,initial=8))
        fgsizer.Add(wx.StaticText(self,label='Energy convergence eps: 1e-'))
        fgsizer.Add(wx.SpinCtrl(self,min=-16,max=0,initial=-12))
        fgsizer.Add(wx.StaticText(self,label='Eigen solver eps:'))
        fgsizer.Add(wx.SpinCtrlDouble(parent=self,min=1e-16,max=1.0,initial=1e-12,inc=1e-12))
        fgsizer.Add(wx.StaticText(self,label='Normalization eps:'))
        fgsizer.Add(wx.SpinCtrlDouble(parent=self,min=1e-16,max=1.0,initial=1e-12,inc=1e-12))
        fgsizer.Add(wx.StaticText(self,label='SVD rank eps:'))
        fgsizer.Add(wx.SpinCtrlDouble(parent=self,min=1e-16,max=1.0,initial=1e-12,inc=1e-12))
        fgsizer.SetSizeHints(self)
        self.SetSizer(fgsizer)
        
class wxMPSStatusPanel(wx.Panel):
    def __init__(self,parent):
        super().__init__(parent)
        #
        #  let's mock things for now
        #
        L=10
        status='A1A1A1M0B0B0B0B0B0B0'
        #
        #
        #
        colours={"A":wx.BLUE,"B":wx.RED, "M":wx.GREEN, "I":wx.LIGHT_GREY }
        text=wx.TextCtrl(self,style=wx.TE_MULTILINE,size=(L*20,-1))
        for i in range(0,L-1):
            index=2*i
            letters=status[index:index+2]
            letter=letters[0]
            colour=colours[letter]
            text.SetDefaultStyle(wx.TextAttr(colour))
            text.AppendText(letters)
            
class MPSFrame(wx.Frame):
    def __init__(self,*args,**kwargs):
        super(MPSFrame,self).__init__(*args,**kwargs)
               
        self.BuildMenus()
        self.BuildLayout()
        
        self.Centre()
 
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
        factory=PyPlotting.wxFactory.GetFactory()

        self.Splitter=wx.SplitterWindow(self)
        leftPanel=wx.Panel(self.Splitter)
        self.LeftNB=wx.Notebook(leftPanel)
        h=wxHamiltonianPanel(self.LeftNB)
        self.LeftNB.AddPage(h,'Hamiltonian')
        approx=wxMPSApproximationsPanel(self.LeftNB)
        self.LeftNB.AddPage(approx,'MPS Approximations')
        
        nbsizer=wx.BoxSizer(wx.VERTICAL)
        nbsizer.Add(self.LeftNB,border=5,flag=wx.ALL|wx.EXPAND)
        leftPanel.SetSizer(nbsizer)
        nbsizer.SetSizeHints(leftPanel)
        leftPanel.SetAutoLayout(True)

        rightPanel=wx.Panel(self.Splitter)
        self.graphs=factory.MakewxMultiGraph(rightPanel)
        graphsPanel=factory.GetPanel(self.graphs) #dynamic cross cast
        statusPanel=wxMPSStatusPanel(rightPanel)
        rbsizer=wx.BoxSizer(wx.VERTICAL)
        rbsizer.Add(graphsPanel)
        rbsizer.Add(statusPanel)
        rightPanel.SetSizer(rbsizer)
        rightPanel.Layout()
        
        #rbsizer.SetSizeHints(rightPanel)
        #rightPanel.SetAutoLayout(True)
        
        W,H = leftPanel.GetSize()
        self.Splitter.SplitVertically(leftPanel,rightPanel,W)
 
        
        
    def OnQuit(self, e):
        self.Close()


