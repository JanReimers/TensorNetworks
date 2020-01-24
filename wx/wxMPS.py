import wx
import PyPlotting

from wxMPSFrame import MPSFrame


class MPSapp(wx.App):
    def OnInit(self):
        self.PlottingFactory = PyPlotting.wxFactoryMain()
        frame = MPSFrame(None,title='MPS Studio',size=(1000,500))
        frame.Show()
        return True


app = MPSapp()
app.MainLoop()

