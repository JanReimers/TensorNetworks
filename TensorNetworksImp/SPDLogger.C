#include "TensorNetworksImp/SPDLogger.H"
#include "TensorNetworksImp/NullLogger.H"
#include "spdlog/spdlog.h"
#include <iostream>

namespace TensorNetworks
{

SPDLogger::SPDLogger(int level)
    : itsLevel(level)
{
    //Logger is null default the override.
    if (!Logger  || dynamic_cast<NullLogger*>(Logger))
    {
        Logger=this;
        std::cout << "Setting logger to SPD" << std::endl;
    }
}

SPDLogger::~SPDLogger()
{
    if (Logger==this) Logger=nullptr;
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

void SPDLogger::LogWarn(int level,c_str message)
{
    if (itsLevel>=level)
    {
        spdlog::warn(message);
    }
}


} // namespace
