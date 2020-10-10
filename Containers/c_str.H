// File: c_str.H  Make a short typedef.
#ifndef _c_str_H_
#define _c_str_H_

// Copyright (1994-2003), Jan N. Reimers

#include <cassert>

typedef const char* c_str;

inline c_str Check_c_str(c_str s)
{
  assert(s);
  return s;
}


#endif //_c_str_H_
