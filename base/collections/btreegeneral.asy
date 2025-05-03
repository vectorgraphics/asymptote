typedef import(T);

from collections.iter(T=T) access Iter_T, Iterable_T, Iterable;
from collections.sortedset(T=T) access Set_T, SortedSet_T;

struct BTreeSet_T {
  struct _ { autounravel restricted SortedSet_T super; }
  from super unravel nullT, isNullT;
  private bool lt(T, T) = null;
  private int size = 0;
  private int versionNo = 0;
  private int maxPivots = 128;
  private int minPivots = maxPivots # 2;
  
  private bool leq(T a, T b) { return !lt(b, a); };
  // The following functions have been manually inlined to avoid the overhead of
  // function calls.
  // private bool gt(T a, T b) { return lt(b, a); };
  // private bool geq(T a, T b) { return !lt(b, a); };
  private bool equiv(T a, T b) { return !(lt(a, b) || lt(b, a)); };

  struct Node {
    T[] pivots;
    Node[] children = null;
    T get(T x) {
      int i = search(pivots, x, lt);
      if (i >= 0) {
        // Known: pivots[i] <= x
        T candidate = pivots[i];
        if (/*leq(x, candidate)*/!lt(candidate, x)) return candidate;
      }
      if (alias(children, null)) {
        assert(isNullT != null, 'Item is not present.');
        return nullT;
      }
      return children[i + 1].get(x);
    }
    bool contains(T x) {
      int i = search(pivots, x, lt);
      if (i >= 0) {
        // Known: pivots[i] <= x
        T candidate = pivots[i];
        if (/*leq(x, candidate)*/!lt(candidate, x)) return true;
      }
      if (alias(children, null)) return false;
      return children[i + 1].contains(x);
    }
    T min() {
      if (alias(children, null)) return pivots[0];
      return children[0].min();
    }
    T max() {
      int plen = pivots.length;
      if (alias(children, null)) return pivots[plen - 1];
      return children[plen].max();
    }
    T firstGEQ(T x) {
      // Want: min { y | y >= x }
      int i = search(pivots, x, lt);
      // Known: pivots[i + 1] > x or i == pivots.length - 1
      // Handle the case that pivots[i] == x:
      if (i >= 0 && /*leq(x, pivots[i])*/!lt(pivots[i], x)) return pivots[i];
      ++i;
      // Known: pivots[i] > x or i == pivots.length
      // Handle the childless case:
      if (alias(children, null)) {
        if (i < pivots.length) return pivots[i];
        assert(isNullT != null, 'No element after item to return');
        return nullT;
      }
      Node child = children[i];
      // Now we need to search child to see if it has any nodes >= x.
      if (isNullT != null) {
        T candidate = child.firstGEQ(x);
        if (i < pivots.length && isNullT(candidate)) return pivots[i];
        return candidate;
      }
      // If child.firstGEQ() cannot return nullT to indicate an empty result, we
      // need an extra check to see if child has anything to return.
      if (i < pivots.length && lt(child.max(), x)) {
        return pivots[i];
      }
      return child.firstGEQ(x);
    }
    T after(T x) {
      // Want: min { y | y > x }
      int i = search(pivots, x, lt) + 1;
      // Known: pivots[i] > x or i == pivots.length
      // Handle the childless case:
      if (alias(children, null)) {
        if (i < pivots.length) return pivots[i];
        assert(isNullT != null, 'No element after item to return');
        return nullT;
      }
      Node child = children[i];
      // Now we need to search child to see if it has any nodes > x.
      if (isNullT != null) {
        T candidate = child.after(x);
        if (i < pivots.length && isNullT(candidate)) return pivots[i];
        return candidate;
      }
      // If child.after() cannot return nullT to indicate an empty result, we
      // need an extra check to see if child has anything to return.
      if (i < pivots.length && /*leq(child.max(), x)*/
          !lt(x, child.max())) {
        return pivots[i];
      }
      return child.after(x);
    }
    T firstLEQ(T x) {
      // Want: max { y | y <= x }
      int i = search(pivots, x, lt);
      // Known: pivots[i] <= x or i == -1
      // Handle the childless case:
      if (alias(children, null)) {
        if (i >= 0) return pivots[i];
        assert(isNullT != null, 'No element before item to return');
        return nullT;
      }
      Node child = children[i + 1];
      // Now we need to search child to see if it has any nodes <= x.
      if (isNullT != null) {
        T candidate = child.firstLEQ(x);
        if (i >= 0 && isNullT(candidate)) return pivots[i];
        return candidate;
      }
      // If child.firstLEQ() cannot return nullT to indicate an empty result, we
      // need an extra check to see if child has anything to return.
      if (i >= 0 && lt(x, child.min())) {
        return pivots[i];
      }
      return child.firstLEQ(x);
    }
    T before(T x) {
      // Want: max { y | y < x }
      int i = search(pivots, x, lt);
      // Known: pivots[i] <= x or i == -1
      // Handle the case that pivots[i] == x:
      if (i >= 0 && /*leq(x, pivots[i])*/!lt(pivots[i], x)) {
        --i;
      }
      // Known: pivots[i] < x or i == -1
      // Handle the childless case:
      if (alias(children, null)) {
        if (i >= 0) return pivots[i];
        assert(isNullT != null, 'No element before item to return');
        return nullT;
      }
      Node child = children[i + 1];
      // Now we need to search child to see if it has any nodes < x.
      if (isNullT != null) {
        T candidate = child.before(x);
        if (i >= 0 && isNullT(candidate)) return pivots[i];
        return candidate;
      }
      // If child.before() cannot return nullT to indicate an empty result, we
      // need an extra check to see if child has anything to return.
      if (i >= 0 && /*geq(child.min(), x)*/
          !lt(child.min(), x)) {
        return pivots[i];
      }
      return child.before(x);
    }

    // NOTE: Does NOT perform concurrent modification detection.
    // (Should be wrapped in another iterator that does.)
    Iter_T operator iter() {
      if (alias(children, null)) return Iter_T(pivots);
      Iter_T result;
      int i = 0;
      Iter_T childIter = children[0].operator iter();
      result.get = new T() {
        if (childIter != null) return childIter.get();
        return pivots[i];
      };
      result.advance = new void() {
        if (childIter != null) {
          childIter.advance();
          if (!childIter.valid()) {
            childIter = null;
          }
        } else {
          ++i;
          childIter = children[i].operator iter();
        }
      };
      result.valid = new bool() {
        return i < pivots.length || (childIter != null && childIter.valid());
      };
      return result;
    }
    bool locate(T x, Node[] stack, int[] indices) {
      int i = search(pivots, x, lt);
      stack.push(this);
      if (i >= 0 && /*leq(x, pivots[i])*/!lt(pivots[i], x)) {
        indices.push(i);
        return true;
      }
      int ci = i + 1;
      indices.push(ci);
      if (alias(children, null)) {
        return false;
      }
      return children[ci].locate(x, stack, indices);
    }
    void locateMin(Node[] stack, int[] indices) {
      stack.push(this);
      indices.push(0);
      if (alias(children, null)) {
        return;
      }
      children[0].locateMin(stack, indices);
    }
    void locateMax(Node[] stack, int[] indices) {
      stack.push(this);
      int plen = pivots.length;
      if (alias(children, null)) {
        indices.push(plen - 1);
        return;
      }
      indices.push(plen);
      children[plen].locateMax(stack, indices);
    }
  }
  private Node root = new Node;

  // NOTE: The default isNullT uses operator == rather than equiv. Consequently,
  // lessThan does not necessarily need to be defined for nullT.
  void operator init(bool lessThan(T, T), T nullT,
                     bool isNullT(T) = new bool(T t) { return t == nullT; }) {
    this.lt = lessThan;
    super.operator init(nullT, equiv, isNullT);
  }

  // Allows for adjusting the maximum number of pivots in a node. Intended
  // primarily for testing.
  void operator init(bool lessThan(T, T), T nullT,
                     bool isNullT(T) = new bool(T t) { return t == nullT; },
                     int keyword maxPivots) {
    this.maxPivots = maxPivots;
    this.minPivots = maxPivots # 2;
    this.operator init(lessThan, nullT, isNullT);
  }

  void operator init(bool lessThan(T, T)) {
    this.lt = lessThan;
    using Initializer = void();
    ((Initializer)super.operator init)();
  }

  super.size = new int() { return size; };
  super.contains = new bool(T x) { return root.contains(x); };
  super.get = new T(T x) { return root.get(x); };
  super.after = new T(T x) { return root.after(x); };
  super.before = new T(T x) { return root.before(x); };
  super.firstGEQ = new T(T x) { return root.firstGEQ(x); };
  super.firstLEQ = new T(T x) { return root.firstLEQ(x); };
  super.min = new T() {
    if (size == 0) {
      assert(isNullT != null, 'No minimum element to return');
      return nullT;
    }
    return root.min();
  };
  super.max = new T() {
    if (size == 0) {
      assert(isNullT != null, 'No maximum element to return');
      return nullT;
    }
    return root.max();
  };

  super.operator iter = new Iter_T() {
    Iter_T unsafe = root.operator iter();
    int expectedVersion = versionNo;
    Iter_T result;
    result.get = new T() {
      assert(versionNo == expectedVersion, 'Concurrent modification');
      return unsafe.get();
    };
    result.advance = new void() {
      assert(versionNo == expectedVersion, 'Concurrent modification');
      unsafe.advance();
    };
    result.valid = new bool() {
      assert(versionNo == expectedVersion, 'Concurrent modification');
      return unsafe.valid();
    };
    return result;
  };

  // NOTE: stack and indices are (partially) consumed.
  private void forceAdd(T item, Node[] stack, int[] indices) {
    int i = indices.pop();
    Node node = stack.pop();
    node.pivots.insert(i, item);
    while (node.pivots.length > maxPivots) {
      T[] pivots = node.pivots;
      int mid = pivots.length # 2;
      T pivot = pivots[mid];
      Node left = new Node;
      Node right = new Node;
      left.pivots = pivots[0:mid];
      right.pivots = pivots[mid + 1:];
      if (!alias(node.children, null)) {
        left.children = node.children[0:mid + 1];
        right.children = node.children[mid + 1:];
      }
      if (stack.length == 0) {
        assert(alias(root, node));
        root = new Node;
        root.pivots.push(pivot);
        root.children = new Node[] {left, right};
        break;
      }
      i = indices.pop();
      node = stack.pop();
      node.pivots.insert(i, pivot);
      node.children[i] = left;
      node.children.insert(i + 1, right);
    }
    ++size;
    ++versionNo;
  }

  super.add = new bool(T item) {
    if (isNullT != null && isNullT(item)) {
      return false;
    }
    Node[] stack;
    int[] indices;
    if (root.locate(item, stack, indices)) {
      return false;
    }
    forceAdd(item, stack, indices);
    return true;
  };

  super.swap = new T(T item) {
    if (isNullT != null && isNullT(item)) {
      return nullT;
    }
    Node[] stack;
    int[] indices;
    if (root.locate(item, stack, indices)) {
      int i = indices.pop();
      Node node = stack.pop();
      T result = node.pivots[i];
      node.pivots[i] = item;
      return result;
    }
    assert(isNullT != null, 'Adding item via swap() without defining nullT.');
    forceAdd(item, stack, indices);
    return nullT;
  };

  private void delete(Node[] stack, int[] indices) {
    ++versionNo;
    --size;
    Node node = stack.pop();
    int i = indices.pop();
    T result = node.pivots[i];
    if (!alias(node.children, null)) {
      Node right = node.children[i + 1];
      stack.push(node);
      indices.push(i + 1);
      right.locateMin(stack, indices);
      Node minNode = stack.pop();
      int minIndex = indices.pop();
      node.pivots[i] = minNode.pivots[minIndex];
      node = minNode;
      i = minIndex;
    }
    node.pivots.delete(i);
    while (node.pivots.length < minPivots) {
      if (stack.length == 0) {
        assert(alias(root, node));
        if (node.pivots.length == 0 && size > 0) {
          root = node.children[0];
        }
        break;
      }
      Node parent = stack.pop();
      int parentIndex = indices.pop();
      Node left = null, right = null;
      // Try to rotate from a sibling with more than enough pivots.
      if (parentIndex < parent.pivots.length) {
        right = parent.children[parentIndex + 1];
        if (right.pivots.length > minPivots) {
          T oldPivot = parent.pivots[parentIndex];
          T newPivot = right.pivots[0];
          parent.pivots[parentIndex] = newPivot;
          right.pivots.delete(0);
          node.pivots.push(oldPivot);
          if (!alias(node.children, null)) {
            node.children.push(right.children[0]);
            right.children.delete(0);
          }
          break;
        }
      }
      if (parentIndex > 0) {
        left = parent.children[parentIndex - 1];
        if (left.pivots.length > minPivots) {
          T oldPivot = parent.pivots[parentIndex - 1];
          T newPivot = left.pivots.pop();
          parent.pivots[parentIndex - 1] = newPivot;
          node.pivots.insert(0, oldPivot);
          if (!alias(node.children, null)) {
            node.children.insert(0, left.children.pop());
          }
          break;
        }
      }
      // Merge with a sibling.
      if (left != null) {
        right = node;
        --parentIndex;
      } else {
        assert(right != null);
        left = node;
      }
      left.pivots.push(parent.pivots[parentIndex]);
      parent.pivots.delete(parentIndex);
      left.pivots.append(right.pivots);
      if (!alias(left.children, null)) {
        left.children.append(right.children);
      }
      parent.children.delete(parentIndex + 1);
      node = parent;
    }
  }

  super.extract = new T(T item) {
    Node[] stack;
    int[] indices;
    if (!root.locate(item, stack, indices)) {
      assert(isNullT != null, 'Item not found');
      return nullT;
    }
    Node node = stack[stack.length - 1];
    int i = indices[indices.length - 1];
    T result = node.pivots[i];
    delete(stack, indices);
    return result;
  };

  super.delete = new bool(T item) {
    Node[] stack;
    int[] indices;
    if (!root.locate(item, stack, indices)) {
      return false;
    }
    delete(stack, indices);
    return true;
  };

  super.popMin = new T() {
    if (size == 0) {
      assert(isNullT != null, 'No minimum element to pop');
      return nullT;
    }
    Node[] stack;
    int[] indices;
    root.locateMin(stack, indices);
    T result = stack[stack.length - 1].pivots[indices[indices.length - 1]];
    delete(stack, indices);
    return result;
  };

  super.popMax = new T() {
    if (size == 0) {
      assert(isNullT != null, 'No maximum element to pop');
      return nullT;
    }
    Node[] stack;
    int[] indices;
    root.locateMax(stack, indices);
    T result = stack[stack.length - 1].pivots[indices[indices.length - 1]];
    delete(stack, indices);
    return result;
  };

  // cast operators
  autounravel SortedSet_T operator cast(BTreeSet_T set) {
    return set.super;
  }
  autounravel Set_T operator cast(BTreeSet_T set) {
    return (SortedSet_T)set;  // Compose with the above cast.
  }
  autounravel Iterable_T operator cast(BTreeSet_T set) {
    return Iterable_T(set.super.operator iter);
  }

  from super unravel *;
}