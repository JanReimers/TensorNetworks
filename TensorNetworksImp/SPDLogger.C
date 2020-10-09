#include "SPDLogger.H"

namespace TensorNetworks
{

SPDLogger::SPDLogger(int level)
    : itsLevel(level)
{
    //Defulat logger is stdout
}

SPDLogger::~SPDLogger()
{
    //dtor
}

void SPDLogger::ReadyToStart(c_str message)
{
    spdlog::info("Start {}", message);
}

void SPDLogger::LogInfo(int level,c_str message)
{
    if (itsLevel>=level)
    {
        spdlog::info(message);
    }
}

void SPDLogger::LogInfo(int level,int site, c_str message)
{
    if (itsLevel>=level)
    {
        spdlog::info("Site {}: {}",site,message);
    }
}

} // namespace
