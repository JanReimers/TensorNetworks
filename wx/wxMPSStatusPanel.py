import wx
import PyTensorNetworks

class wxMPSStatusPanel(wx.Panel):
    def __init__(self,parent):
        super().__init__(parent)

        self.colours={"A":wx.BLUE,"B":wx.RED, "M":wx.BLACK, "I":wx.LIGHT_GREY }
        self.L=0

    def NewLattice(self,L):
        self.L=L
        fgsizer=wx.FlexGridSizer(L+1,3,0) #2 rows, L+2 columns, gap=5
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

        #fgsizer.SetSizeHints(self)

        hsizer=wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self,label='Current Operation:'))
        self.CurrentOperationTextControl=wx.TextCtrl(self,style=wx.TE_MULTILINE|wx.TE_LEFT,size=(400,20))
        hsizer.Add(self.CurrentOperationTextControl,flag=wx.EXPAND)

        vsizer=wx.BoxSizer(wx.VERTICAL)
        vsizer.Add(fgsizer)
        vsizer.Add(hsizer,flag=wx.EXPAND)

        self.SetSizer(vsizer)
        self.GetParent().Layout()

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

    def UpdateCurrentOperation(self,currentOP):
        self.CurrentOperationTextControl.SetValue(currentOP)

