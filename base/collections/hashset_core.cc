// C++ core for the compound port of base/collections/hashset.asy.
//
// Owns the data structure (buckets, the doubly-linked oldest/newest list,
// the find / changeCapacity / makeZombie logic and the mutation counter).
// Knows nothing about Set_T, late-bound defaults, unravel super, or
// autounravel — those concerns live in the sibling wrapper hashset.asy.

#include <asybind/asybind.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace {

// Allocate a zero-filled array via asy's GC so that the contents are
// scanned conservatively and any HashEntry* stored inside remains
// reachable.  Using std::vector here is unsafe: its backing buffer
// lives in the system heap and is invisible to the GC, so entries can
// be collected while still "in use", producing crashes in unrelated
// code after a later allocation.
template <class T>
T* gc_array(size_t n) {
    if (n == 0) return nullptr;
    void* mem = ay::detail::current_api()->alloc_obj(n * sizeof(T));
    if (!mem) ay::raise("hashset_core: gc_array allocation failed");
    std::memset(mem, 0, n * sizeof(T));
    return static_cast<T*>(mem);
}

struct HashEntry {
    ay::Any  item;
    long long  hash  = -1;
    HashEntry* newer = nullptr;
    HashEntry* older = nullptr;
};

struct HashSetCore_T;

struct Cursor {
    HashSetCore_T* owner = nullptr;
    HashEntry*     current = nullptr;
    int            expectedChanges = 0;

    void check() const;
    bool   valid()   const;
    ay::Any get()    const;
    void   advance();
};

struct HashSetCore_T {
    // Callbacks installed at construction; held as ay::callable so the
    // GC can reach the asy-side closures from `this`.
    // Asy integers are 64-bit, so we return long long to avoid losing
    // entropy at the asybind boundary.
    ay::callable<long long(ay::Any)>         hashFn;
    ay::callable<bool(ay::Any, ay::Any)>   equivFn;
    ay::callable<bool(ay::Any)>            isNullTFn;   // may be null

    // GC-allocated bucket array (an asy::vector<...> equivalent that the
    // GC can scan).  Holds n_buckets HashEntry* slots.
    HashEntry** buckets   = nullptr;
    int       n_buckets   = 0;
    int       size_      = 0;
    int       zombies    = 0;
    int       numChanges = 0;
    HashEntry* newest    = nullptr;
    HashEntry* oldest    = nullptr;

    // Map a (possibly negative) hash to a bucket index in [0, n).
    int bucketIndex(long long hash, int n) const {
        long long j = hash % n;
        if (j < 0) j += n;
        return static_cast<int>(j);
    }

    HashEntry*& bucketAt(int i) {
        return buckets[i];
    }

    bool isNullTItem(ay::Any item) const {
        return isNullTFn && isNullTFn(item);
    }

    void changeCapacity() {
        ++numChanges;
        const int n = n_buckets;
        const int newCap = (zombies > size_) ? n : 2 * n;
        zombies = 0;
        HashEntry** next = gc_array<HashEntry*>(newCap);
        for (HashEntry* cur = oldest; cur; cur = cur->newer) {
            const int start = bucketIndex(cur->hash, newCap);
            bool placed = false;
            for (int i = 0; i < newCap; ++i) {
                int idx = start + i;
                if (idx >= newCap) idx -= newCap;
                if (!next[idx]) {
                    next[idx] = cur;
                    placed = true;
                    break;
                }
            }
            if (!placed)
                ay::raise("No space in hash table; "
                          "is the linked list circular?");
        }
        buckets = next;
        n_buckets = newCap;
    }

    // Returns the bucket index in [0, n) of either the existing entry
    // matching `item`, or the first empty slot encountered.
    // Returns -1 if all n probes saw zombie/non-matching entries.
    int find(ay::Any item, long long hash) {
        const int n = n_buckets;
        const int start = bucketIndex(hash, n);
        for (int i = 0; i < n; ++i) {
            int idx = start + i;
            if (idx >= n) idx -= n;
            HashEntry* e = buckets[idx];
            if (!e) return idx;
            if (e->hash == hash && equivFn(e->item, item)) return idx;
        }
        return -1;
    }

    void makeZombie(HashEntry* entry) {
        ++numChanges;
        entry->hash = -1;
        ++zombies;
        if (entry->older) entry->older->newer = entry->newer;
        else              oldest = entry->newer;
        if (entry->newer) entry->newer->older = entry->older;
        else              newest = entry->older;
        --size_;
    }

    void reset(ay::callable<long long(ay::Any)> hashFn_,
               ay::callable<bool(ay::Any, ay::Any)> equivFn_,
               ay::callable<bool(ay::Any)> isNullTFn_,
               int initialCapacity) {
        hashFn    = hashFn_;
        equivFn   = equivFn_;
        isNullTFn = isNullTFn_;
        if (initialCapacity < 1) initialCapacity = 1;
        buckets = gc_array<HashEntry*>(initialCapacity);
        n_buckets = initialCapacity;
        size_ = 0;
        zombies = 0;
        numChanges = 0;
        newest = oldest = nullptr;
    }

    int  size()        const { return size_; }
    int  capacity()    const { return n_buckets; }
    int  numChangesV() const { return numChanges; }
    bool hasNullT()    const { return static_cast<bool>(isNullTFn); }

    bool contains(ay::Any item) {
        if (isNullTItem(item)) return false;
        const int n = capacity();
        const long long hash = hashFn(item);
        const int start = bucketIndex(hash, n);
        for (int i = 0; i < n; ++i) {
            int idx = start + i;
            if (idx >= n) idx -= n;
            HashEntry* e = buckets[idx];
            if (!e) return false;
            if (e->hash == hash && equivFn(e->item, item)) return true;
        }
        return false;
    }

    ay::result<ay::Any> lookup(ay::Any item) {
        if (isNullTItem(item)) return {};
        const int n = capacity();
        const long long hash = hashFn(item);
        const int start = bucketIndex(hash, n);
        for (int i = 0; i < n; ++i) {
            int idx = start + i;
            if (idx >= n) idx -= n;
            HashEntry* e = buckets[idx];
            if (!e) return {};
            if (e->hash == hash && equivFn(e->item, item))
                return { true, e->item };
        }
        return {};
    }

    bool add(ay::Any item) {
        if (isNullTItem(item)) return false;
        int cap = capacity();
        if (2 * (size_ + zombies) >= cap) {
            changeCapacity();
            cap = capacity();
        }
        const long long hash = hashFn(item);
        int index = find(item, hash);
        if (index == -1) {
            ++numChanges;
            changeCapacity();
            cap = capacity();
            index = find(item, hash);
            if (index == -1) ay::raise("No space in hash table");
        }
        HashEntry*& slot = bucketAt(index);
        if (slot) return false;

        ++numChanges;
        HashEntry* entry = ay::gc_new<HashEntry>();
        entry->item  = item;
        entry->hash  = hash;
        entry->older = newest;
        if (newest) newest->newer = entry;
        newest = entry;
        if (!oldest) oldest = entry;
        bucketAt(index) = entry;
        ++size_;
        return true;
    }

    ay::result<ay::Any> push(ay::Any item) {
        if (isNullTItem(item)) return {};
        int cap = capacity();
        if (2 * (size_ + zombies) >= cap) {
            changeCapacity();
        }
        const long long hash = hashFn(item);
        int index = find(item, hash);
        if (index == -1) {
            changeCapacity();
            index = find(item, hash);
            if (index == -1) ay::raise("No space in hash table");
        }
        HashEntry*& slot = bucketAt(index);
        if (slot) {
            ay::Any old = slot->item;
            slot->item = item;
            return { true, old };
        }
        ++numChanges;
        HashEntry* entry = ay::gc_new<HashEntry>();
        entry->item  = item;
        entry->hash  = hash;
        entry->older = newest;
        if (newest) newest->newer = entry;
        newest = entry;
        if (!oldest) oldest = entry;
        bucketAt(index) = entry;
        ++size_;
        return {};
    }

    ay::result<ay::Any> extract(ay::Any item) {
        if (isNullTItem(item)) return {};
        int index = find(item, hashFn(item));
        if (index == -1)
            ay::raise("Overcrowded hash table");
        HashEntry* entry = bucketAt(index);
        if (!entry) return {};
        ay::Any old = entry->item;
        makeZombie(entry);
        return { true, old };
    }

    bool deleteItem(ay::Any item) {
        if (isNullTItem(item)) return false;
        int index = find(item, hashFn(item));
        if (index == -1)
            ay::raise("Overcrowded hash table");
        HashEntry* entry = bucketAt(index);
        if (!entry) return false;
        makeZombie(entry);
        return true;
    }

    ay::result<ay::Any> getRandom() {
        if (size_ == 0) return {};
        const int n = capacity();
        if (size_ > 0 && size_ / 2 > n / size_) {
            int idx = static_cast<int>(ay::rand(0, size_ - 1));
            for (HashEntry* cur = oldest; cur; cur = cur->newer) {
                if (idx == 0) return { true, cur->item };
                --idx;
            }
            ay::raise("Unreachable code");
        }
        for (;;) {
            int i = static_cast<int>(ay::rand(0, n - 1));
            HashEntry* e = buckets[i];
            if (e && e->hash != -1) return { true, e->item };
        }
    }

    Cursor* beginCursor() {
        Cursor* c = ay::gc_new<Cursor>();
        c->owner = this;
        c->current = this->oldest;
        c->expectedChanges = this->numChanges;
        return c;
    }
};

inline void Cursor::check() const {
    if (owner->numChanges != expectedChanges)
        ay::raise("Concurrent modification");
}
inline bool Cursor::valid() const {
    check();
    return current != nullptr;
}
inline ay::Any Cursor::get() const {
    check();
    if (!current) ay::raise("Invalid iterator");
    return current->item;
}
inline void Cursor::advance() {
    check();
    if (!current) ay::raise("Invalid iterator");
    current = current->newer;
}

}  // namespace

ASY_TEMPLATED_MODULE(hashset_core, m, "T") {
    // Bind T (resolved per instantiation); no requires_method since the
    // wrapper supplies hash/equiv as ordinary asy callables.
    (void)m.type_param("T");

    ay::class_<HashSetCore_T> core(m, "HashSetCore_T");
    ay::class_<Cursor> cursor(m, "Cursor_T");

    core.def(ay::init<>());
    core.def<&HashSetCore_T::reset>     ("reset");
    core.def<&HashSetCore_T::size>      ("size");
    core.def<&HashSetCore_T::capacity>  ("capacity");
    core.def<&HashSetCore_T::numChangesV>("numChanges");
    core.def<&HashSetCore_T::hasNullT>  ("hasNullT");
    core.def<&HashSetCore_T::contains>  ("contains");
    core.def<&HashSetCore_T::add>("add");
    core.def<&HashSetCore_T::deleteItem>("deleteItem");
    core.def<&HashSetCore_T::lookup>    ("lookup");
    core.def<&HashSetCore_T::push>      ("push");
    core.def<&HashSetCore_T::extract>   ("extract");
    core.def<&HashSetCore_T::getRandom> ("getRandom");
    core.def<&HashSetCore_T::beginCursor>("beginCursor");

    cursor.def<&Cursor::valid>  ("valid");
    cursor.def<&Cursor::get>    ("get");
    cursor.def<&Cursor::advance>("advance");
}
