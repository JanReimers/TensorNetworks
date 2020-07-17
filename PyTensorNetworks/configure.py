import os
import sipconfig

# The name of the SIP build file generated by SIP and used by the build
# system.
build_file = "PyTensorNetworks.sbf"

# Get the SIP configuration information.
config = sipconfig.Configuration()

# Run SIP to generate the code.
# Add includes to pick .sip from the WxPython build tree at /home/jan/wxPython-4.1
# Note: we are not picking C headers (*.h) YET, that comes below
# The secret sauce seems to be the the -n wx.siplib without that the module won't load into python3
#     I think this tells sip to use Robin's build of siblib (c files in wxPython-4.1/src) instead of building a local copy
os.system(" ".join([config.sip_bin, "-c", ".", "-I../../../wxPython-4/sip/gen/","-I../../../wxPython-4/src","-I../../Plotting/PyPlotting", "-n wx.siplib","-b", build_file, "PyTensorNetworks.sip"]))

# Create the Makefile.
makefile = sipconfig.SIPModuleMakefile(config, build_file,warnings=None,debug=1)

#
#  Here is where we tell the build where to find sip.h and any other headers in wxPython-4.1 source tree
#
makefile.extra_include_dirs=["../../../wxPython-4/sip/siplib","../" ,"../../","../../Plotting","/usr/lib/x86_64-linux-gnu/wx/include/gtk3-unicode-3.0","/usr/include/wx-3.0"]
#
#  Some compile flags required by wxWidgets
#
makefile.extra_defines=["_FILE_OFFSET_BITS=64","WXUSINGDLL","__WXGTK__"]
makefile.extra_cxxflags=["-pthread"]
#
#  replace ../bin/Debug with a path to your extension lib
#  /home/jan/.local/lib/python3.7/site-packages/wx is the location of the shared wx libs used by wxPython
#  You will see that these are sent to the linker with -rpath, which should set the runtime path as well.
#  This may be important because its easy to wxWidget .so's sprinkled in various places in a linux dist.
#
makefile.extra_lib_dirs=["../../Debug","/usr/local/lib/python3.7/dist-packages/wx"]
makefile.extra_lflags=["-pthread"]
# Add the library we are wrapping.  The name doesn't include any platform
# specific prefixes or extensions (e.g. the "lib" prefix on UNIX, or the
# ".dll" extension on Windows).
makefile.extra_libs = ["TensorNetworks","Functions","Plotting","Misc","oml","primme","lapack","blas"]

# Generate the Makefile itself.
makefile.generate()
