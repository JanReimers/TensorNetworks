import wx
import PyPlotting
import PyTensorNetworks

from wxMPSFrame import MPSFrame


class MPSapp(wx.App):
    def OnInit(self):
        frame = MPSFrame(None,title='MPS Studio',size=(1000,500))
        frame.Show()
        return True


app = MPSapp()
app.MainLoop()

