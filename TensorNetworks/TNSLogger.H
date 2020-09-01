#ifndef TNSLOGGER_H
#define TNSLOGGER_H

#include "Misc/c_str.H"
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
        virtual void DoneOneStep(int /*level*/,c_str /*currentOperation*/, int isite=-1) {(void)isite;} //comment name to avoid warning
        virtual TNSLogger* Clone() const {return new TNSLogger();}
};

#endif // TNSLOGGER_H
