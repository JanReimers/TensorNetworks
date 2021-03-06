#ifndef SPDLOGGER_H
#define SPDLOGGER_H

#include "TensorNetworks/TNSLogger.H"

namespace TensorNetworks
{

class SPDLogger : public virtual TNSLogger
{
    public:
        SPDLogger(int level);
        virtual ~SPDLogger();
        virtual void LogInfo(int level,c_str message);
        virtual void LogInfo(int level,int site,c_str message);
        virtual void LogWarn(int level,c_str message);

    protected:

    private:
        int itsLevel;
};

}
#endif // SPDLOGGER_H
