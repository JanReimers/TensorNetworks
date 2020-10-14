#include "TensorNetworks/MPS.H"
#include "TensorNetworks/Factory.H"
#include "TensorNetworks/SVCompressor.H"

namespace TensorNetworks
{

void MPS::Normalize(Direction lr)
{
    NormalizeAndCompress(lr,0);
}

void MPS::NormalizeAndCompress(Direction lr,int Dmax,double epsSV)
{
    SVCompressorC* comp=Factory::GetFactory()->MakeMPSCompressor(Dmax,epsSV);
    NormalizeAndCompress(lr,comp);
    delete comp;
}

}
