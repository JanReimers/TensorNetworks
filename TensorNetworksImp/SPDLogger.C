#include "SPDLogger.H"


namespace TensorNetworks
{
TNSLogger* Logger=nullptr;

SPDLogger::SPDLogger(int level)
    : itsLevel(level)
{
    //Defulat logger is stdout
    if (!Logger) Logger=this;
}

SPDLogger::~SPDLogger()
{
    if (Logger==this) Logger=nullptr;
}

void SPDLogger::ReadyToStart(c_str message)
{
    if (itsLevel>=0)
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
