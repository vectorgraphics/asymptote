// C++ core for the compound port of base/collections/hashset.asy.
//
// Owns the data structure (buckets, the doubly-linked oldest/newest list,
// the find / changeCapacity / makeZombie logic and the mutation counter).
// Knows nothing about Set_T, late-bound defaults, unravel super, or
// autounravel — those concerns live in the sibling wrapper hashset.asy.

#include <asybind/asybind.h>

#include <cstdint>

namespace {

struct HashEntry {
    asy::Any  item;
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
    asy::Any get()    const;
    void   advance();
};

struct HashSetCore_T {
    // Callbacks installed at construction; held as asy::callable so the
    // GC can reach the asy-side closures from `this`.
    // Asy integers are 64-bit, so we return long long to avoid losing
    // entropy at the asybind boundary.
    asy::callable<long long(asy::Any)>         hashFn;
    asy::callable<bool(asy::Any, asy::Any)>   equivFn;
    asy::callable<bool(asy::Any)>            isNullTFn;   // may be null

    // GC-tracked bucket array.  `asy::mem::vector` allocates its
    // backing buffer on asy's scanned GC heap, so any non-null
    // HashEntry* stored inside is kept reachable.
    asy::mem::vector<HashEntry*> buckets;
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

    bool isNullTItem(asy::Any item) const {
        return isNullTFn && isNullTFn(item);
    }

    void changeCapacity() {
        ++numChanges;
        const int n = static_cast<int>(buckets.size());
        const int newCap = (zombies > size_) ? n : 2 * n;
        zombies = 0;
        asy::mem::vector<HashEntry*> next(static_cast<std::size_t>(newCap),
                                         nullptr);
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
                asy::raise("No space in hash table; "
                          "is the linked list circular?");
        }
        buckets = std::move(next);
    }

    // Returns the bucket index in [0, n) of either the existing entry
    // matching `item`, or the first empty slot encountered.
    // Returns -1 if all n probes saw zombie/non-matching entries.
    int find(asy::Any item, long long hash) {
        const int n = static_cast<int>(buckets.size());
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

    void reset(asy::callable<long long(asy::Any)> hashFn_,
               asy::callable<bool(asy::Any, asy::Any)> equivFn_,
               asy::callable<bool(asy::Any)> isNullTFn_,
               int initialCapacity) {
        hashFn    = hashFn_;
        equivFn   = equivFn_;
        isNullTFn = isNullTFn_;
        if (initialCapacity < 1) initialCapacity = 1;
        buckets.assign(static_cast<std::size_t>(initialCapacity), nullptr);
        size_ = 0;
        zombies = 0;
        numChanges = 0;
        newest = oldest = nullptr;
    }

    int  size()        const { return size_; }
    int  capacity()    const { return static_cast<int>(buckets.size()); }
    int  numChangesV() const { return numChanges; }
    bool hasNullT()    const { return static_cast<bool>(isNullTFn); }

    bool contains(asy::Any item) {
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

    asy::result<asy::Any> lookup(asy::Any item) {
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

    bool add(asy::Any item) {
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
            if (index == -1) asy::raise("No space in hash table");
        }
        HashEntry*& slot = bucketAt(index);
        if (slot) return false;

        ++numChanges;
        HashEntry* entry = asy::gc_new<HashEntry>();
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

    asy::result<asy::Any> push(asy::Any item) {
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
            if (index == -1) asy::raise("No space in hash table");
        }
        HashEntry*& slot = bucketAt(index);
        if (slot) {
            asy::Any old = slot->item;
            slot->item = item;
            return { true, old };
        }
        ++numChanges;
        HashEntry* entry = asy::gc_new<HashEntry>();
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

    asy::result<asy::Any> extract(asy::Any item) {
        if (isNullTItem(item)) return {};
        int index = find(item, hashFn(item));
        if (index == -1)
            asy::raise("Overcrowded hash table");
        HashEntry* entry = bucketAt(index);
        if (!entry) return {};
        asy::Any old = entry->item;
        makeZombie(entry);
        return { true, old };
    }

    bool deleteItem(asy::Any item) {
        if (isNullTItem(item)) return false;
        int index = find(item, hashFn(item));
        if (index == -1)
            asy::raise("Overcrowded hash table");
        HashEntry* entry = bucketAt(index);
        if (!entry) return false;
        makeZombie(entry);
        return true;
    }

    asy::result<asy::Any> getRandom() {
        if (size_ == 0) return {};
        const int n = capacity();
        if (size_ > 0 && size_ / 2 > n / size_) {
            int idx = static_cast<int>(asy::rand(0, size_ - 1));
            for (HashEntry* cur = oldest; cur; cur = cur->newer) {
                if (idx == 0) return { true, cur->item };
                --idx;
            }
            asy::raise("Unreachable code");
        }
        for (;;) {
            int i = static_cast<int>(asy::rand(0, n - 1));
            HashEntry* e = buckets[i];
            if (e && e->hash != -1) return { true, e->item };
        }
    }

    Cursor* beginCursor() {
        Cursor* c = asy::gc_new<Cursor>();
        c->owner = this;
        c->current = this->oldest;
        c->expectedChanges = this->numChanges;
        return c;
    }
};

inline void Cursor::check() const {
    if (owner->numChanges != expectedChanges)
        asy::raise("Concurrent modification");
}
inline bool Cursor::valid() const {
    check();
    return current != nullptr;
}
inline asy::Any Cursor::get() const {
    check();
    if (!current) asy::raise("Invalid iterator");
    return current->item;
}
inline void Cursor::advance() {
    check();
    if (!current) asy::raise("Invalid iterator");
    current = current->newer;
}

}  // namespace

ASY_TEMPLATED_MODULE(hashset_core, m, "T") {
    // Bind T (resolved per instantiation); no requires_method since the
    // wrapper supplies hash/equiv as ordinary asy callables.
    (void)m.type_param("T");

    asy::class_<HashSetCore_T> core(m, "HashSetCore_T");
    asy::class_<Cursor> cursor(m, "Cursor_T");

    core.def(asy::init<>());
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
