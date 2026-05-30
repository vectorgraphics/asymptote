// C++ core for the compound port of base/collections/hashset.asy.
//
// Owns the data structure (buckets, the doubly-linked oldest/newest list,
// the find / changeCapacity / makeZombie logic and the mutation counter).
// Knows nothing about Set_T, late-bound defaults, unravel super, or
// autounravel — those concerns live in the sibling wrapper hashset.asy.

#include <asybind/asybind.h>

#include <cstdint>
#include <vector>

namespace {

struct HashEntry {
    ay::Any  item;
    int      hash  = -1;
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
    ay::callable<int(ay::Any)>             hashFn;
    ay::callable<bool(ay::Any, ay::Any)>   equivFn;
    ay::callable<bool(ay::Any)>            isNullTFn;   // may be null

    std::vector<HashEntry*> buckets;
    int       size_      = 0;
    int       zombies    = 0;
    int       numChanges = 0;
    HashEntry* newest    = nullptr;
    HashEntry* oldest    = nullptr;

    HashEntry*& bucketAt(int i) {
        const int n = static_cast<int>(buckets.size());
        int j = i % n;
        if (j < 0) j += n;
        return buckets[j];
    }

    bool isNullTItem(ay::Any item) const {
        return isNullTFn && isNullTFn(item);
    }

    void changeCapacity() {
        ++numChanges;
        const int n = static_cast<int>(buckets.size());
        const int newCap = (zombies > size_) ? n : 2 * n;
        zombies = 0;
        std::vector<HashEntry*> next(newCap, nullptr);
        for (HashEntry* cur = oldest; cur; cur = cur->newer) {
            const int bucket = cur->hash;
            bool placed = false;
            for (int i = 0; i < newCap; ++i) {
                int idx = bucket % newCap;
                idx = (idx + i) % newCap;
                if (idx < 0) idx += newCap;
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
        buckets = std::move(next);
    }

    int find(ay::Any item, int hash) {
        const int n = static_cast<int>(buckets.size());
        for (int i = hash - n; i < hash; ++i) {
            HashEntry* e = bucketAt(i);
            if (!e) return i;
            if (e->hash == hash && equivFn(e->item, item)) return i;
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

    void reset(ay::callable<int(ay::Any)> hashFn_,
               ay::callable<bool(ay::Any, ay::Any)> equivFn_,
               ay::callable<bool(ay::Any)> isNullTFn_,
               int initialCapacity) {
        hashFn    = hashFn_;
        equivFn   = equivFn_;
        isNullTFn = isNullTFn_;
        if (initialCapacity < 1) initialCapacity = 1;
        buckets.assign(initialCapacity, nullptr);
        size_ = 0;
        zombies = 0;
        numChanges = 0;
        newest = oldest = nullptr;
    }

    int  size()        const { return size_; }
    int  capacity()    const { return static_cast<int>(buckets.size()); }
    int  numChangesV() const { return numChanges; }
    bool hasNullT()    const { return static_cast<bool>(isNullTFn); }

    bool contains(ay::Any item) {
        if (isNullTItem(item)) return false;
        const int n = capacity();
        const int bucket = hashFn(item);
        for (int i = bucket - n; i < bucket; ++i) {
            HashEntry* e = bucketAt(i);
            if (!e) return false;
            if (e->hash == bucket && equivFn(e->item, item)) return true;
        }
        return false;
    }

    ay::result<ay::Any> lookup(ay::Any item) {
        if (isNullTItem(item)) return {};
        const int n = capacity();
        const int bucket = hashFn(item);
        for (int i = bucket - n; i < bucket; ++i) {
            HashEntry* e = bucketAt(i);
            if (!e) return {};
            if (e->hash == bucket && equivFn(e->item, item))
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
        const int bucket = hashFn(item);
        int index = find(item, bucket);
        if (index == -1) {
            ++numChanges;
            changeCapacity();
            cap = capacity();
            index = find(item, bucket);
            if (index == -1) ay::raise("No space in hash table");
        }
        HashEntry*& slot = bucketAt(index);
        if (slot) return false;

        ++numChanges;
        HashEntry* entry = ay::gc_new<HashEntry>();
        entry->item  = item;
        entry->hash  = bucket;
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
        const int bucket = hashFn(item);
        int index = find(item, bucket);
        if (index == -1) {
            changeCapacity();
            index = find(item, bucket);
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
        entry->hash  = bucket;
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
