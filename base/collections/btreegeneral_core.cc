// C++ core for the compound port of base/collections/btreegeneral.asy.
//
// Owns the B-tree node arena, the lt/isNullT callbacks, the size and
// mutation counters, and the path-tracking machinery used by insertions
// and deletions.  Knows nothing about Set_T / SortedSet_T / unravel
// super / autounravel — those concerns live in the sibling wrapper
// btreegeneral.asy.
//
// Design notes:
//   * `BTreeNode` lives in GC memory (allocated via `ay::gc_new`).  Its
//     pivots and children arrays use `ay::mem::vector`, an STL-style
//     vector whose backing buffer is allocated on asy's scanned GC
//     heap.  No bespoke "gc array" helpers are required: the BDW GC
//     conservatively scans both the enclosing node and the vector
//     buffer, so any pointer stored inside (whether to an `ay::Any`'s
//     payload or to a sibling `BTreeNode`) is kept alive.
//   * Path stacks used during locate/forceAdd/delete are fixed-size
//     arrays on the C stack.  At maxPivots=8 (worst supported case in
//     the bench), depth ≤ ceil(log_{minPivots+1}(N)) — bounded well
//     under 64 for any plausible N.
//   * No callbacks fire during forceAdd / delete after `locate` has
//     returned, so GC pressure stays bounded.  Newly-allocated split
//     nodes are stored in local variables (which BDW GC scans
//     conservatively on the C stack) before being attached to the
//     tree.

#include <asybind/asybind.h>

#include <cstddef>
#include <cstdint>

namespace {

constexpr int kMaxDepth = 64;

struct BTreeNode {
    asy::mem::vector<asy::Any>     pivots;
    asy::mem::vector<BTreeNode*>  children;  // empty for a leaf
    bool isLeaf() const { return children.empty(); }
};

struct BTreeSetCore_T;

struct Cursor {
    BTreeSetCore_T* owner = nullptr;
    int            expectedChanges = 0;

    // Path from root to the current node; `idx[d]` is the index within
    // path[d]->pivots of the element to emit on `get()`.  Always ends
    // at a leaf (path[depth-1] is a leaf with idx in [0, pivots.size)).
    BTreeNode* path[kMaxDepth];
    int        idx [kMaxDepth];
    int        depth = 0;   // 0 ⇒ exhausted

    void   check() const;
    bool   valid()   const;
    asy::Any get()    const;
    void   advance();

    // Push (node, 0) and descend through children[0] until we hit a leaf.
    void descend_left(BTreeNode* n) {
        while (true) {
            if (depth >= kMaxDepth)
                asy::raise("btree_core: cursor stack overflow");
            path[depth] = n;
            idx [depth] = 0;
            ++depth;
            if (n->isLeaf()) return;
            n = n->children[0];
        }
    }
};

struct BTreeSetCore_T {
    asy::callable<bool(asy::Any, asy::Any)> ltFn;
    asy::callable<bool(asy::Any)>          isNullTFn;  // may be null

    BTreeNode* root = nullptr;
    int size_      = 0;
    int numChanges = 0;
    int maxPivots  = 128;
    int minPivots  = 64;

    bool isNullTItem(asy::Any item) const {
        return isNullTFn && isNullTFn(item);
    }

    // Binary search on pivots: returns the largest i with pivots[i] <= x,
    // or -1 if pivots[0] > x.  Calls ltFn O(log size) times.
    // Equivalent to asy's `search(pivots, x, lt)`.
    int search_pivots(BTreeNode* n, asy::Any x) {
        const int sz = static_cast<int>(n->pivots.size());
        if (sz == 0 || ltFn(x, n->pivots[0])) return -1;
        int u = sz - 1;
        if (!ltFn(x, n->pivots[u])) return u;
        int l = 0;
        // Invariants:  pivots[l] <= x  and  pivots[u] > x.
        while (l < u) {
            int i = (l + u) / 2;
            if (ltFn(x, n->pivots[i])) u = i;
            else if (ltFn(x, n->pivots[i + 1])) return i;
            else l = i + 1;
        }
        return l;
    }

    void reset(asy::callable<bool(asy::Any, asy::Any)> ltFn_,
               asy::callable<bool(asy::Any)>          isNullTFn_,
               long long maxPivots_) {
        ltFn      = ltFn_;
        isNullTFn = isNullTFn_;
        if (maxPivots_ < 2) maxPivots_ = 2;
        maxPivots = static_cast<int>(maxPivots_);
        minPivots = maxPivots / 2;
        root      = asy::gc_new<BTreeNode>();
        size_     = 0;
        numChanges = 0;
    }

    int  size()        const { return size_; }
    int  numChangesV() const { return numChanges; }
    bool hasNullT()    const { return static_cast<bool>(isNullTFn); }

    // --------- read-only path-free queries (closest to Node::*) ---------

    bool contains(asy::Any x) {
        BTreeNode* node = root;
        while (true) {
            int i = search_pivots(node, x);
            if (i >= 0 && !ltFn(node->pivots[i], x)) return true;
            if (node->isLeaf()) return false;
            node = node->children[i + 1];
        }
    }

    asy::result<asy::Any> lookup(asy::Any x) {
        BTreeNode* node = root;
        while (true) {
            int i = search_pivots(node, x);
            if (i >= 0 && !ltFn(node->pivots[i], x))
                return { true, node->pivots[i] };
            if (node->isLeaf()) return {};
            node = node->children[i + 1];
        }
    }

    asy::result<asy::Any> minOpt() {
        if (size_ == 0) return {};
        BTreeNode* n = root;
        while (!n->isLeaf()) n = n->children[0];
        return { true, n->pivots[0] };
    }

    asy::result<asy::Any> maxOpt() {
        if (size_ == 0) return {};
        BTreeNode* n = root;
        while (!n->isLeaf()) n = n->children.back();
        return { true, n->pivots.back() };
    }

    // Helpers for after/before:  these mirror the asy logic exactly,
    // including the "extra check via child.max()/child.min()" fallback
    // for callers without nullT.

    // Returns { true, y } where y = min { z | z >= x }, or {} if none.
    asy::result<asy::Any> atOrAfter(asy::Any x) {
        BTreeNode* node = root;
        asy::result<asy::Any> best;  // best.found = false
        while (true) {
            int i = search_pivots(node, x);
            // pivots[i] <= x, but maybe pivots[i] == x:
            if (i >= 0 && !ltFn(node->pivots[i], x))
                return { true, node->pivots[i] };
            ++i;
            // pivots[i] > x  or  i == pivots.size()
            if (i < static_cast<int>(node->pivots.size()))
                best = { true, node->pivots[i] };
            if (node->isLeaf()) return best;
            node = node->children[i];
        }
    }

    asy::result<asy::Any> after(asy::Any x) {
        BTreeNode* node = root;
        asy::result<asy::Any> best;
        while (true) {
            int i = search_pivots(node, x) + 1;
            // pivots[i] > x  or  i == pivots.size()
            if (i < static_cast<int>(node->pivots.size()))
                best = { true, node->pivots[i] };
            if (node->isLeaf()) return best;
            node = node->children[i];
        }
    }

    asy::result<asy::Any> atOrBefore(asy::Any x) {
        BTreeNode* node = root;
        asy::result<asy::Any> best;
        while (true) {
            int i = search_pivots(node, x);
            // pivots[i] <= x  or  i == -1
            if (i >= 0)
                best = { true, node->pivots[i] };
            if (node->isLeaf()) return best;
            node = node->children[i + 1];
        }
    }

    asy::result<asy::Any> before(asy::Any x) {
        BTreeNode* node = root;
        asy::result<asy::Any> best;
        while (true) {
            int i = search_pivots(node, x);
            if (i >= 0 && !ltFn(node->pivots[i], x)) --i;
            // pivots[i] < x  or  i == -1
            if (i >= 0)
                best = { true, node->pivots[i] };
            if (node->isLeaf()) return best;
            node = node->children[i + 1];
        }
    }

    // ------------------------ locate/forceAdd/delete -----------------

    // Walks from root to a leaf, pushing nodes on `stack` and the
    // chosen child index at each step on `indices`.  Returns true iff
    // an exact match was found (in which case the top of stack/indices
    // identifies (node, pivot_index) holding the match).
    bool locate(asy::Any x,
                BTreeNode** stack, int* indices, int& depth) {
        BTreeNode* node = root;
        while (true) {
            int i = search_pivots(node, x);
            stack[depth] = node;
            if (i >= 0 && !ltFn(node->pivots[i], x)) {
                indices[depth] = i;
                ++depth;
                return true;
            }
            int ci = i + 1;
            indices[depth] = ci;
            ++depth;
            if (node->isLeaf()) return false;
            node = node->children[ci];
        }
    }

    void locateMin(BTreeNode** stack, int* indices, int& depth) {
        BTreeNode* node = root;
        while (true) {
            stack[depth] = node;
            indices[depth] = 0;
            ++depth;
            if (node->isLeaf()) return;
            node = node->children[0];
        }
    }

    void locateMax(BTreeNode** stack, int* indices, int& depth) {
        BTreeNode* node = root;
        while (true) {
            stack[depth] = node;
            int plen = static_cast<int>(node->pivots.size());
            if (node->isLeaf()) {
                indices[depth] = plen - 1;
                ++depth;
                return;
            }
            indices[depth] = plen;
            ++depth;
            node = node->children[plen];
        }
    }

    // Inserts `item` at (stack[depth-1], indices[depth-1]) and
    // performs splits up the tree as needed.  Consumes (partially)
    // `stack` and `indices`.
    void forceAdd(asy::Any item,
                  BTreeNode** stack, int* indices, int depth) {
        --depth;
        int i = indices[depth];
        BTreeNode* node = stack[depth];
        node->pivots.insert(node->pivots.begin() + i, item);

        while (static_cast<int>(node->pivots.size()) > maxPivots) {
            const int plen = static_cast<int>(node->pivots.size());
            const int mid  = plen / 2;
            asy::Any pivot  = node->pivots[mid];

            BTreeNode* left  = asy::gc_new<BTreeNode>();
            BTreeNode* right = asy::gc_new<BTreeNode>();
            left ->pivots.assign(node->pivots.begin(),
                                 node->pivots.begin() + mid);
            right->pivots.assign(node->pivots.begin() + mid + 1,
                                 node->pivots.end());
            if (!node->isLeaf()) {
                left ->children.assign(node->children.begin(),
                                       node->children.begin() + mid + 1);
                right->children.assign(node->children.begin() + mid + 1,
                                       node->children.end());
            }

            if (depth == 0) {
                // Replace root with a new internal node.
                root = asy::gc_new<BTreeNode>();
                root->pivots.push_back(pivot);
                root->children.push_back(left);
                root->children.push_back(right);
                break;
            }
            --depth;
            i    = indices[depth];
            node = stack[depth];
            node->pivots.insert(node->pivots.begin() + i, pivot);
            node->children[i] = left;
            node->children.insert(node->children.begin() + i + 1, right);
        }

        ++size_;
        ++numChanges;
    }

    // Deletes the entry identified by (stack[depth-1], indices[depth-1]).
    // Consumes (partially) `stack` and `indices`.
    void deletePath(BTreeNode** stack, int* indices, int depth) {
        ++numChanges;
        --size_;
        --depth;
        BTreeNode* node = stack[depth];
        int i = indices[depth];

        if (!node->isLeaf()) {
            // Swap with in-order successor (leftmost pivot of children[i+1]).
            BTreeNode* right = node->children[i + 1];
            // Re-extend the path: starting from `right` go leftmost.
            stack  [depth] = node;
            indices[depth] = i + 1;
            ++depth;
            BTreeNode* cur = right;
            while (true) {
                stack[depth] = cur;
                indices[depth] = 0;
                ++depth;
                if (cur->isLeaf()) break;
                cur = cur->children[0];
            }
            BTreeNode* minNode = stack[depth - 1];
            int        minIdx  = indices[depth - 1];
            --depth;  // pop the leaf back so the loop below sees correct top
            node->pivots[i] = minNode->pivots[minIdx];
            node = minNode;
            i = minIdx;
        }
        node->pivots.erase(node->pivots.begin() + i);

        while (static_cast<int>(node->pivots.size()) < minPivots) {
            if (depth == 0) {
                // node == root.
                if (node->pivots.empty() && size_ > 0) {
                    root = node->children[0];
                }
                break;
            }
            --depth;
            BTreeNode* parent       = stack[depth];
            int        parentIndex  = indices[depth];
            BTreeNode* left  = nullptr;
            BTreeNode* right = nullptr;

            // Try rotating from a sibling with > minPivots.
            if (parentIndex < static_cast<int>(parent->pivots.size())) {
                right = parent->children[parentIndex + 1];
                if (static_cast<int>(right->pivots.size()) > minPivots) {
                    asy::Any oldPivot = parent->pivots[parentIndex];
                    asy::Any newPivot = right->pivots.front();
                    parent->pivots[parentIndex] = newPivot;
                    right->pivots.erase(right->pivots.begin());
                    node->pivots.push_back(oldPivot);
                    if (!node->isLeaf()) {
                        node->children.push_back(right->children.front());
                        right->children.erase(right->children.begin());
                    }
                    return;
                }
            }
            if (parentIndex > 0) {
                left = parent->children[parentIndex - 1];
                if (static_cast<int>(left->pivots.size()) > minPivots) {
                    asy::Any oldPivot = parent->pivots[parentIndex - 1];
                    asy::Any newPivot = left->pivots.back();
                    left->pivots.pop_back();
                    parent->pivots[parentIndex - 1] = newPivot;
                    node->pivots.insert(node->pivots.begin(), oldPivot);
                    if (!node->isLeaf()) {
                        BTreeNode* movedChild = left->children.back();
                        left->children.pop_back();
                        node->children.insert(node->children.begin(),
                                              movedChild);
                    }
                    return;
                }
            }

            // Merge with a sibling.
            if (left != nullptr) {
                right = node;
                --parentIndex;
            } else {
                // right was set above when parentIndex < parent->pivots.size().
                left = node;
            }
            left->pivots.push_back(parent->pivots[parentIndex]);
            parent->pivots.erase(parent->pivots.begin() + parentIndex);
            left->pivots.insert(left->pivots.end(),
                                right->pivots.begin(), right->pivots.end());
            if (!left->isLeaf()) {
                left->children.insert(left->children.end(),
                                      right->children.begin(),
                                      right->children.end());
            }
            parent->children.erase(parent->children.begin() + parentIndex + 1);
            node = parent;
        }
    }

    bool add(asy::Any item) {
        if (isNullTItem(item)) return false;
        BTreeNode* stack[kMaxDepth];
        int        indices[kMaxDepth];
        int        depth = 0;
        if (locate(item, stack, indices, depth)) return false;
        forceAdd(item, stack, indices, depth);
        return true;
    }

    asy::result<asy::Any> push(asy::Any item) {
        if (isNullTItem(item)) return {};
        BTreeNode* stack[kMaxDepth];
        int        indices[kMaxDepth];
        int        depth = 0;
        if (locate(item, stack, indices, depth)) {
            int i = indices[depth - 1];
            BTreeNode* node = stack[depth - 1];
            asy::Any result = node->pivots[i];
            node->pivots[i] = item;
            return { true, result };
        }
        forceAdd(item, stack, indices, depth);
        return {};
    }

    asy::result<asy::Any> extract(asy::Any item) {
        BTreeNode* stack[kMaxDepth];
        int        indices[kMaxDepth];
        int        depth = 0;
        if (!locate(item, stack, indices, depth)) return {};
        BTreeNode* node = stack[depth - 1];
        int i = indices[depth - 1];
        asy::Any result = node->pivots[i];
        deletePath(stack, indices, depth);
        return { true, result };
    }

    bool deleteItem(asy::Any item) {
        BTreeNode* stack[kMaxDepth];
        int        indices[kMaxDepth];
        int        depth = 0;
        if (!locate(item, stack, indices, depth)) return false;
        deletePath(stack, indices, depth);
        return true;
    }

    asy::result<asy::Any> popMin() {
        if (size_ == 0) return {};
        BTreeNode* stack[kMaxDepth];
        int        indices[kMaxDepth];
        int        depth = 0;
        locateMin(stack, indices, depth);
        asy::Any result = stack[depth - 1]->pivots[indices[depth - 1]];
        deletePath(stack, indices, depth);
        return { true, result };
    }

    asy::result<asy::Any> popMax() {
        if (size_ == 0) return {};
        BTreeNode* stack[kMaxDepth];
        int        indices[kMaxDepth];
        int        depth = 0;
        locateMax(stack, indices, depth);
        asy::Any result = stack[depth - 1]->pivots[indices[depth - 1]];
        deletePath(stack, indices, depth);
        return { true, result };
    }

    Cursor* beginCursor() {
        Cursor* c = asy::gc_new<Cursor>();
        c->owner = this;
        c->expectedChanges = numChanges;
        c->depth = 0;
        if (size_ > 0) c->descend_left(root);
        return c;
    }
};

inline void Cursor::check() const {
    if (owner->numChanges != expectedChanges)
        asy::raise("Concurrent modification");
}
inline bool Cursor::valid() const {
    check();
    return depth > 0;
}
inline asy::Any Cursor::get() const {
    check();
    if (depth == 0) asy::raise("Invalid iterator");
    return path[depth - 1]->pivots[idx[depth - 1]];
}
inline void Cursor::advance() {
    check();
    if (depth == 0) asy::raise("Invalid iterator");

    BTreeNode* top = path[depth - 1];
    if (top->isLeaf()) {
        ++idx[depth - 1];
        if (idx[depth - 1] < static_cast<int>(top->pivots.size())) return;
        // Exhausted leaf.  Pop until we find an internal node with
        // more pivots to emit (or run out).
        --depth;
        while (depth > 0) {
            int i = idx[depth - 1];
            BTreeNode* n = path[depth - 1];
            if (i < static_cast<int>(n->pivots.size())) {
                // Sitting at internal node ready to emit pivots[i].
                return;
            }
            --depth;
        }
        return;  // exhausted
    }
    // Top is an internal node; we just emitted pivots[idx].  Descend
    // into children[idx+1] (which exists since idx < pivots.size()).
    int next = idx[depth - 1] + 1;
    idx[depth - 1] = next;
    descend_left(top->children[next]);
}

}  // namespace

ASY_TEMPLATED_MODULE(btreegeneral_core, m, "T") {
    (void)m.type_param("T");

    asy::class_<BTreeSetCore_T> core(m, "BTreeSetCore_T");
    asy::class_<Cursor>         cursor(m, "Cursor_T");

    core.def(asy::init<>());
    core.def<&BTreeSetCore_T::reset>       ("reset");
    core.def<&BTreeSetCore_T::size>        ("size");
    core.def<&BTreeSetCore_T::numChangesV> ("numChanges");
    core.def<&BTreeSetCore_T::hasNullT>    ("hasNullT");
    core.def<&BTreeSetCore_T::contains>    ("contains");
    core.def<&BTreeSetCore_T::lookup>      ("lookup");
    core.def<&BTreeSetCore_T::minOpt>      ("minOpt");
    core.def<&BTreeSetCore_T::maxOpt>      ("maxOpt");
    core.def<&BTreeSetCore_T::atOrAfter>   ("atOrAfter");
    core.def<&BTreeSetCore_T::after>       ("after");
    core.def<&BTreeSetCore_T::atOrBefore>  ("atOrBefore");
    core.def<&BTreeSetCore_T::before>      ("before");
    core.def<&BTreeSetCore_T::add>         ("add");
    core.def<&BTreeSetCore_T::push>        ("push");
    core.def<&BTreeSetCore_T::extract>     ("extract");
    core.def<&BTreeSetCore_T::deleteItem>  ("deleteItem");
    core.def<&BTreeSetCore_T::popMin>      ("popMin");
    core.def<&BTreeSetCore_T::popMax>      ("popMax");
    core.def<&BTreeSetCore_T::beginCursor> ("beginCursor");

    cursor.def<&Cursor::valid>   ("valid");
    cursor.def<&Cursor::get>     ("get");
    cursor.def<&Cursor::advance> ("advance");
}
