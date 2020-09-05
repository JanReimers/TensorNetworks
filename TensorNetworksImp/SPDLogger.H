#ifndef SPDLOGGER_H
#define SPDLOGGER_H

#include "TensorNetworks/TNSLogger.H"
#include "spdlog/spdlog.h"

class SPDLogger : public virtual TNSLogger
{
    public:
        SPDLogger(int level);
        virtual ~SPDLogger();
        virtual void ReadyToStart(c_str /*currentOperation*/);
        virtual void LogInfo(int level,c_str message);
        virtual void LogInfo(int level,int site,c_str message);

    protected:

    private:
        int itsLevel;
};

#endif // SPDLOGGER_H
