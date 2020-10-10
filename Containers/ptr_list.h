// File: ptr_list.h  Experimental pointer list, derived from and STL list<void*>.
#ifndef _ptr_list_h_
#define _ptr_list_h_

// Copyright (1994-2003), Jan N. Reimers

#include <list>
#include <Misc/void_types.h>

//
// Primary template for un-owned pointers.
//
template <class T> class ptr_list;

//
//  Iterator.  Has different symmantics than a T* iterator, op* returns a T&
//  and op& returns a T*.
//
template<class T, class Ref, class Ptr,class Base>
struct ptr_list_iterator
: public Base
{
  typedef ptr_list_iterator<T,Ref,Ptr,Base> Self;

  ptr_list_iterator(                 ) : Base( ) {};
  template <class B> ptr_list_iterator(const B & x) : Base(x) {} //Not type safe.

  Ptr& operator& () const { return *reinterpret_cast<Ptr*>(&Base::operator*()); }
  Ref  operator* () const { return *(operator&()); }
  Ptr  operator->() const { return   operator&() ; }

  Self& operator++(   ) {Base::operator++();return *this;}
  Self  operator++(int) {Self tmp = *this;++*this;return tmp;}
  Self& operator--(   ) {Base::operator--();return *this;}
  Self  operator--(int) {Self tmp = *this;--*this;return tmp;}
 private:
  friend class ptr_list<T*>;
};

//
//  Specialize for any pointer type.
//
template <class T> class ptr_list<T*>
: private std::list<typename VoidType<T*>::void_type>
{
public:
  typedef T     element_type;
private:
  typedef T*    value_type;
  typedef       value_type& reference;
  typedef const value_type& const_reference;
  typedef typename VoidType<T*>::void_type void_type;
  typedef typename std::list<void_type> Base;
  typedef typename Base::_Node* node_ptr;
  typedef typename Base::      iterator  BI;
  typedef typename Base::const_iterator CBI;
 public:

  typedef ptr_list_iterator<T,      T&,      T*,BI > iterator;
  typedef ptr_list_iterator<T,const T&,T* const,CBI> const_iterator;

  explicit ptr_list() : Base() {};
  ~ptr_list() {};  // Somebody else owns the pointers.

  iterator       begin()       { return Base::begin(); }
  const_iterator begin() const { return Base::begin(); }
  iterator       end  ()       { return Base::end(); }
  const_iterator end  () const { return Base::end(); }

  using Base::empty;
  using Base::size;
  using Base::erase;
  void clear() {Base::clear();}  // Somebody else owns the pointers.

  reference       front()       { return *begin(); }
  const_reference front() const { return *begin(); }
  reference       back ()       { return *(--end()); }
  const_reference back () const { return *(--end()); }

  void push_front(const value_type x) {Base::push_front(static_cast<void_type>(x));}
  void push_back (const value_type x) {Base::push_back (static_cast<void_type>(x));}
  void merge(ptr_list<T*>& l) {Base::merge(l);}

};

//
// Primary template for un-owned pointers.
//
template <class T> class optr_list;
//
//  Specialize for any pointer type.
//
template <class T> class optr_list<T*>
: private ptr_list<T*>
{
  typedef ptr_list<T*> Base;
 public:
  using Base::element_type;

  explicit optr_list() : Base() {};
  ~optr_list() {clear();}
  void clear();

  using Base::iterator;
  using Base::const_iterator;
  using Base::begin;
  using Base::end;
  using Base::empty;
  using Base::size;
  using Base::front;
  using Base::back;
  using Base::push_front;
  using Base::push_back;
 private:
  optr_list(const optr_list&);
  optr_list& operator=(const optr_list&);
};

template <class T> void optr_list<T*>::clear()
{
  for (typename optr_list::iterator i=begin();i!=end();i++) delete &i;
  Base::clear();
}

#endif // _ptr_list_h_
