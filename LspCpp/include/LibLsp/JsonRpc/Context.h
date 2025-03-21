//===--- Context.h - Mechanism for passing implicit data --------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Context for storing and retrieving implicit data. Useful for passing implicit
// parameters on a per-request basis.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <memory>
#include <type_traits>

namespace lsp
{

/// Values in a Context are indexed by typed keys.
/// Key<T> serves two purposes:
///   - it provides a lookup key for the context (each Key is unique),
///   - it makes lookup type-safe: a Key<T> can only map to a T (or nothing).
///
/// Example:
///    Key<int> RequestID;
///    Key<int> Version;
///
///    Context Ctx = Context::empty().derive(RequestID, 10).derive(Version, 3);
///    assert(*Ctx.get(RequestID) == 10);
///    assert(*Ctx.get(Version) == 3);
///
/// Keys are typically used across multiple functions, so most of the time you
/// would want to make them static class members or global variables.
template<class Type>
class Key
{
public:
    static_assert(!std::is_reference<Type>::value, "Reference arguments to Key<> are not allowed");

    constexpr Key() = default;

    Key(Key const&) = delete;
    Key& operator=(Key const&) = delete;
    Key(Key&&) = delete;
    Key& operator=(Key&&) = delete;
};

/// A context is an immutable container for per-request data that must be
/// propagated through layers that don't care about it. An example is a request
/// ID that we may want to use when logging.
///
/// Conceptually, a context is a heterogeneous map<Key<T>, T>. Each key has
/// an associated value type, which allows the map to be typesafe.
///
/// There is an "ambient" context for each thread, Context::current().
/// Most functions should read from this, and use WithContextValue or
/// WithContext to extend or replace the context within a block scope.
/// Only code dealing with threads and extension points should need to use
/// other Context objects.
///
/// You can't add data to an existing context, instead you create a new
/// immutable context derived from it with extra data added. When you retrieve
/// data, the context will walk up the parent chain until the key is found.
class Context
{
public:
    /// Returns an empty root context that contains no data.
    static Context empty();
    /// Returns the context for the current thread, creating it if needed.
    static Context const& current();
    // Sets the current() context to Replacement, and returns the old context.
    // Prefer to use WithContext or WithContextValue to do this safely.
    static Context swapCurrent(Context Replacement);

private:
    struct Data;
    Context(std::shared_ptr<Data const> DataPtr);

public:
    /// Same as Context::empty(), please use Context::empty() instead.
    Context() = default;

    /// Copy operations for this class are deleted, use an explicit clone() method
    /// when you need a copy of the context instead.
    Context(Context const&) = delete;
    Context& operator=(Context const&) = delete;

    Context(Context&&) = default;
    Context& operator=(Context&&) = default;

    /// Get data stored for a typed \p Key. If values are not found
    /// \returns Pointer to the data associated with \p Key. If no data is
    /// specified for \p Key, return null.
    template<class Type>
    Type const* get(Key<Type> const& Key) const
    {
        for (Data const* DataPtr = this->dataPtr.get(); DataPtr != nullptr; DataPtr = DataPtr->parent.get())
        {
            if (DataPtr->KeyPtr == &Key)
            {
                return static_cast<Type const*>(DataPtr->value->getValuePtr());
            }
        }
        return nullptr;
    }

    /// A helper to get a reference to a \p Key that must exist in the map.
    /// Must not be called for keys that are not in the map.
    template<class Type>
    Type const& getExisting(Key<Type> const& Key) const
    {
        auto Val = get(Key);
        assert(Val && "Key does not exist");
        return *Val;
    }

    /// Derives a child context
    /// It is safe to move or destroy a parent context after calling derive().
    /// The child will keep its parent alive, and its data remains accessible.
    template<class Type>
    Context derive(Key<Type> const& Key, typename std::decay<Type>::type Value) const&
    {
        return Context(std::make_shared<Data>(Data {
            /*parent=*/dataPtr, &Key,
            std::make_unique<TypedAnyStorage<typename std::decay<Type>::type>>(std::move(Value))
        }));
    }

    template<class Type>
    Context derive(Key<Type> const& Key, typename std::decay<Type>::type Value) && /* takes ownership */
    {
        return Context(std::make_shared<Data>(Data {
            /*parent=*/std::move(dataPtr), &Key,
            std::make_unique<TypedAnyStorage<typename std::decay<Type>::type>>(std::move(Value))
        }));
    }

    /// Derives a child context, using an anonymous key.
    /// Intended for objects stored only for their destructor's side-effect.
    template<class Type>
    Context derive(Type&& Value) const&
    {
        static Key<typename std::decay<Type>::type> Private;
        return derive(Private, std::forward<Type>(Value));
    }

    template<class Type>
    Context derive(Type&& Value) &&
    {
        static Key<typename std::decay<Type>::type> Private;
        return std::move(*this).derive(Private, std::forward<Type>(Value));
    }

    /// Clone this context object.
    Context clone() const;

private:
    class AnyStorage
    {
    public:
        virtual ~AnyStorage() = default;
        virtual void* getValuePtr() = 0;
    };

    template<class T>
    class TypedAnyStorage : public Context::AnyStorage
    {
        static_assert(
            std::is_same<typename std::decay<T>::type, T>::value, "Argument to TypedAnyStorage must be decayed"
        );

    public:
        TypedAnyStorage(T&& Value) : value(std::move(Value))
        {
        }

        void* getValuePtr() override
        {
            return &value;
        }

    private:
        T value;
    };

    struct Data
    {
        // We need to make sure parent outlives the value, so the order of members
        // is important. We do that to allow classes stored in Context's child
        // layers to store references to the data in the parent layers.
        std::shared_ptr<Data const> parent;
        void const* KeyPtr;
        std::unique_ptr<AnyStorage> value;
    };

    std::shared_ptr<Data const> dataPtr;
};

/// WithContext replaces Context::current() with a provided scope.
/// When the WithContext is destroyed, the original scope is restored.
/// For extending the current context with new value, prefer WithContextValue.
class WithContext
{
public:
    WithContext(Context C) : restore(Context::swapCurrent(std::move(C)))
    {
    }
    ~WithContext()
    {
        Context::swapCurrent(std::move(restore));
    }
    WithContext(WithContext const&) = delete;
    WithContext& operator=(WithContext const&) = delete;
    WithContext(WithContext&&) = delete;
    WithContext& operator=(WithContext&&) = delete;

private:
    Context restore;
};

/// WithContextValue extends Context::current() with a single value.
/// When the WithContextValue is destroyed, the original scope is restored.
class WithContextValue
{
public:
    template<typename T>
    WithContextValue(Key<T> const& K, typename std::decay<T>::type V)
        : restore(Context::current().derive(K, std::move(V)))
    {
    }

    // Anonymous values can be used for the destructor side-effect.
    template<typename T>
    WithContextValue(T&& V) : restore(Context::current().derive(std::forward<T>(V)))
    {
    }

private:
    WithContext restore;
};

} // namespace lsp
