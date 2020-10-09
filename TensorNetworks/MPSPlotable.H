#ifndef MPSPLOTTABLE_H
#define MPSPLOTTABLE_H

#include "TensorNetworks/MPS.H"
#include "TensorNetworks/MultiPlotable.H"

namespace TensorNetworks
{

class MPSPlotable
    : public virtual MPS
    , public virtual MultiPlotable
{
    public:
        MPSPlotable() {}
        virtual ~MPSPlotable() {}
};

}
#endif // MPSPLOTTABLE_H
