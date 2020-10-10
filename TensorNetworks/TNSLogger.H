#ifndef TNSLOGGER_H
#define TNSLOGGER_H

#include "Containers/c_str.H"
#include <memory>

namespace TensorNetworks
{
//
//  LOgger for a long running process (LRP).  The public interface is for use by
//  the LRP, it simply calls ReadyToStart at the begining and then
//  repeatedly calls DoneOneStep(level) where level defines a hierarcy of steps.
//  These calls may be blocking ... that's the point.  The GUI user can step through
//  a calculation pla/pause it.  The logger can also update the GUI at each step if required.
//
class TNSLogger
{
    public:
        TNSLogger() {};
        virtual ~TNSLogger() {};

        virtual void ReadyToStart(c_str /*currentOperation*/) {};
        virtual void LogInfo(int level,c_str message) {(void)level;} //comment name to avoid warning
        virtual void LogInfo(int level,int site,c_str message) {(void)level;} //comment name to avoid warning

        void LogInfo(int level,int site,const std::string& message) {LogInfo(level,site,message.c_str());} //comment name to avoid warning
        void LogInfo(int level,const std::string& message) {LogInfo(level,message.c_str());} //comment name to avoid warning

        template<typename... Args>
        void LogInfoV(int level,c_str fmt, const Args &... args)
        {
            size_t size = snprintf( nullptr, 0, fmt,args...) + 1; // Extra space for '\0'
            std::unique_ptr<char[]> buf( new char[ size ] );
            snprintf( buf.get(), size, fmt, args ... );
            LogInfo(level,buf.get());
        }
};

}
#endif // TNSLOGGER_H
