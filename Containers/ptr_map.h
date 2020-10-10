// File: ptr_map.h  Experimental pointer map, derived from and STL map<void*>.
#ifndef _ptr_map_h_
#define _ptr_map_h_

// Copyright (1994-2003), Jan N. Reimers

#include <map>
#include <Misc/void_types.h>
//
// Primary template for un-owned pointers.
//
template <class Key,class T> class  ptr_map;
template <class Key,class T> class optr_map;

//
//  Iterator.  Has different symmantics than a T* iterator, op* returns a T&
//  and op& returns a T*.
//
template<class Key, class T, class Ref, class Ptr,class Base>
struct ptr_map_iterator
        : public Base
{
public:
    typedef ptr_map_iterator<Key,T,Ref,Ptr,Base> Self;
    ptr_map_iterator(                 ) : Base( ) {};
    template <class B> ptr_map_iterator(const B      & x) : Base(x) {} //Not type safe.

    Ptr& operator& () const
    {
        void* const& vpr=Base::operator*().second;
        Ptr const& pr=reinterpret_cast<Ptr const&>(vpr);
        return const_cast<Ptr&>(pr);
    }
    Ref  operator* () const
    {
        return *(operator&());
    }
    Ptr  operator->() const
    {
        return   operator&() ;
    }
    const Key& GetKey() const
    {
        return Base::operator*().first;
    }

    Self& operator++(   )
    {
        Base::operator++();
        return *this;
    }
    Self& operator--(   )
    {
        Base::operator--();
        return *this;
    }
    Self  operator++(int)
    {
        Self tmp = *this;
        ++*this;
        return tmp;
    }
    Self  operator--(int)
    {
        Self tmp = *this;
        --*this;
        return tmp;
    }
private:
    friend class  ptr_map<Key,T*>;
    friend class optr_map<Key,T*>;
    friend class std::pair<Self,bool>;
    //  friend class pair<iterator,bool>;
};

/*! \class ptr_map<Key,T*> ptr_map.h Misc/ptr_map.h
  \brief STL like \c map container specialized for un-owned pointers.

  The class is derived from \c map<Key,void*> in order to reduce executable size. Only the
  most important members of \c map are overriden. The iterator class supports:
  - \c op&, \c op*, \c op-> which return object refs. and pointers rather than \c pair<Key,T*> pairs.
  - pre and post \c op++, pre and post \c op--
  - \c GetKey() to access key values.

  There seems to be a number \c multi_map members in here (\c count, \c equal_range), not sure why!
*/
template <class Key,class T> class ptr_map<Key,T*>
    : private std::map<Key,typename VoidType<T*>::void_type>
{
    typedef typename VoidType<T*>::void_type void_type;
    typedef std::map<Key,void_type> Base;
    typedef typename Base::      iterator  BI;
    typedef typename Base::const_iterator CBI;
    typedef typename Base::      reverse_iterator  BRI;
    typedef typename Base::const_reverse_iterator CBRI;
public:
    typedef std::pair<Key,T*> value_type;
    typedef       value_type& reference;
    typedef const value_type& const_reference;
    typedef typename Base::key_type key_type;

    typedef ptr_map_iterator<Key,T,      T&,      T*, BI>       iterator;
    typedef ptr_map_iterator<Key,T,const T&,T* const,CBI> const_iterator;
    typedef ptr_map_iterator<Key,T,      T&,      T*, BRI>       reverse_iterator;
    typedef ptr_map_iterator<Key,T,const T&,T* const,CBRI> const_reverse_iterator;

    explicit ptr_map() : Base() {}
    ~ptr_map() {};  // Somebody else owns the pointers.

//! Get an STL like read/write iterator.
    iterator       begin()
    {
        return Base::begin();
    }
//! Get an STL like read only iterator.
    const_iterator begin() const
    {
        return Base::begin();
    }
//! Get an STL like read/write iterator.
    iterator       end  ()
    {
        return Base::end  ();
    }
//! Get an STL like read only iterator.
    const_iterator end  () const
    {
        return Base::end  ();
    }
//! Get an STL like read/write iterator.
    reverse_iterator       rbegin()
    {
        return Base::rbegin();
    }
//! Get an STL like read only iterator.
    const_reverse_iterator rbegin() const
    {
        return Base::rbegin();
    }
//! Get an STL like read/write iterator.
    reverse_iterator       rend  ()
    {
        return Base::rend  ();
    }
//! Get an STL like read only iterator.
    const_reverse_iterator rend  () const
    {
        return Base::rend  ();
    }

//! Associative array subscript operator.
    T*&  operator[](const key_type& k)
    {
        return *reinterpret_cast<T**>(&Base::operator[](k));
    }

//! Binary search for object associated with a key.
    iterator       find(const key_type& k)
    {
        return Base::find(k);
    }
//! Binary search for const object associated with a key.
    const_iterator find(const key_type& k) const
    {
        return Base::find(k);
    }


//! First object matching key.
    iterator       lower_bound(const key_type& k)
    {
        return Base::lower_bound(k);
    }
//! First const object matching key.
    const_iterator lower_bound(const key_type& k) const
    {
        return Base::lower_bound(k);
    }
//! Last object matching key.
    iterator       upper_bound(const key_type& k)
    {
        return Base::upper_bound(k);
    }
//! Last object matching key.
    const_iterator upper_bound(const key_type& k) const
    {
        return Base::upper_bound(k);
    }

//! First and last object matching key.
    std::pair<      iterator,      iterator> equal_range(const key_type& k)
    {
        return Base::equal_range(k);
    }
//! First and last const object matching key.
    std::pair<const_iterator,const_iterator> equal_range(const key_type& k) const
    {
        return Base::equal_range(k);
    }

//! Add a pair<Key,T*> pair.
    std::pair<iterator,bool> insert(const value_type& x)
    {
        return Base::insert(x);
    }
//! Add a \c pair<Key,T*> pair \b replacing any prior matching elements (not a std. STL member).
    iterator force_insert(const value_type&);

//! Is the map empty()?
    using Base::empty;
//! Get number of elements.
    using Base::size;
//! Count number of elements matching a key.
    using Base::count;
//! Clear but do not delete, one element.
    using Base::erase;
//! Clears out the map, \b does \b not delete the objects.
    void  clear()
    {
        Base::clear();    // Somebody else owns the pointers.
    }

};

//
// Primary template for un-owned pointers.
//
template <class Key,class T> class optr_map;
//
//  Specialize for any pointer type.
//
/*! \class optr_map\<T*\> ptr_map.h Misc/ptr_map.h
  \brief STL like \c map container specialized for \b owned pointers.
  Same as \c ptr_map but objects get deleted when \c optr_map does.
  Copying is not allowed, because only one \c optr_map can own the pointers.
*/
template <class Key,class T> class optr_map<Key,T*>
    : private ptr_map<Key,T*>
{
    typedef ptr_map<Key,T*> Base;
public:
    typedef std::pair<Key,T*> value_type;
    typedef typename Base::iterator iterator;
    typedef typename Base::reverse_iterator reverse_iterator;

    explicit optr_map() : Base() {};
    ~optr_map()
    {
        clear();
    }
    void clear();
    void erase(iterator);
    void erase_save(iterator i)
    {
        Base::erase(i);
    }

    iterator force_insert(const value_type& x);

    using Base::insert;
    using Base::const_iterator;
    using Base::begin;
    using Base::end;
    using Base::rbegin;
    using Base::rend;
    using Base::empty;
    using Base::size;
    using Base::count;
    using Base::operator[];
    using Base::find;
    using Base::lower_bound;
    using Base::upper_bound;
    using Base::equal_range;
private:
    // Copying is not allowed.
    optr_map(const optr_map&);
    optr_map& operator=(const optr_map&);
};

template <class Key,class T> void optr_map<Key,T*>::clear()
{
    for (iterator i=begin(); i!=end(); i++) delete &i;
    Base::clear();
}

template <class Key,class T> void optr_map<Key,T*>::erase(iterator i)
{
    delete &i;
    Base::erase(i);
}

template <class Key,class T> typename ptr_map<Key,T*>::iterator ptr_map<Key,T*>::
force_insert(const value_type& x)
{
    std::pair<iterator,bool> p=insert(x);
    if (!p.second) &p.first=x.second;
    return p.first;
}

template <class Key,class T> typename optr_map<Key,T*>::iterator optr_map<Key,T*>::
force_insert(const value_type& x)
{
    std::pair<iterator,bool> p=insert(x);
    if (!p.second)
    {
        delete &p.first;
        &p.first=x.second;
    }
    return p.first;
}

#endif // _ptr_map_h_
