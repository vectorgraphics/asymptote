typedef import(T);

from collections.sortedset(T=T) access Set_T, SortedSet_T;
from collections.iter(T=T) access Iter_T, Iterable_T;

private struct TreeNode {
  TreeNode leftchild;
  TreeNode rightchild;
  T value;
  void operator init(T value) {
    this.value = value;
  }

  bool inOrder(bool run(T)) {
    if (leftchild != null) {
      if (!leftchild.inOrder(run)) return false;
    }
    if (!run(value)) return false;
    if (rightchild != null) {
      if (!rightchild.inOrder(run)) return false;
    }
    return true;
  }
}

private struct NodeProgressEnum {
  restricted static int num = 0;
  private static int make() {
    return (++num - 1);
  }
  static int NOT_STARTED = make();
  static int LEFT_DONE = make();
  static int SELF_DONE = make();
  static int RIGHT_DONE = make();
}

private struct NodeInProgress {
  TreeNode node;
  int progress = NodeProgressEnum.NOT_STARTED;
  void operator init(TreeNode node) {
    this.node = node;
  }
}

void inOrderNonRecursive(TreeNode root, bool run(T)) {
  if (root == null) return;
  NodeInProgress[] stack = new NodeInProgress[0];
  stack.cyclic = true;
  stack.push(NodeInProgress(root));
  while (stack.length > 0) {
    NodeInProgress current = stack[-1];
    if (current.progress == NodeProgressEnum.NOT_STARTED) {
      if (current.node.leftchild != null) {
        stack.push(NodeInProgress(current.node.leftchild));
      }
      current.progress = NodeProgressEnum.LEFT_DONE;
    } else if (current.progress == NodeProgressEnum.LEFT_DONE) {
      if (!run(current.node.value)) return;
      current.progress = NodeProgressEnum.SELF_DONE;
    } else if (current.progress == NodeProgressEnum.SELF_DONE) {
      if (current.node.rightchild != null) {
        stack.push(NodeInProgress(current.node.rightchild));
      }
      current.progress = NodeProgressEnum.RIGHT_DONE;
    } else {
      assert(current.progress == NodeProgressEnum.RIGHT_DONE);
      stack.pop();
    }
  }
}


private TreeNode splay(TreeNode[] ancestors, bool lessthan(T a, T b)) {
  bool operator < (T a, T b) = lessthan;
  
  if (ancestors.length == 0) return null;

  TreeNode root = ancestors[0];
  TreeNode current = ancestors.pop();
  
  while (ancestors.length >= 2) {
    TreeNode parent = ancestors.pop();
    TreeNode grandparent = ancestors.pop();

    if (ancestors.length > 0) {
      TreeNode greatparent = ancestors[-1];
      if (greatparent.leftchild == grandparent) {
        greatparent.leftchild = current;
      } else greatparent.rightchild = current;
    }

    bool currentside = (parent.leftchild == current);
    bool grandside = (grandparent.leftchild == parent);

    if (currentside == grandside) { // zig-zig
      if (currentside) { // both left
        TreeNode B = current.rightchild;
        TreeNode C = parent.rightchild;

        current.rightchild = parent;
        parent.leftchild = B;
        parent.rightchild = grandparent;
        grandparent.leftchild = C;
      } else { // both right
        TreeNode B = parent.leftchild;
        TreeNode C = current.leftchild;

        current.leftchild = parent;
        parent.leftchild = grandparent;
        parent.rightchild = C;
        grandparent.rightchild = B;
      }
    } else { // zig-zag
      if (grandside) {  // left-right
        TreeNode B = current.leftchild;
        TreeNode C = current.rightchild;

        current.leftchild = parent;
        current.rightchild = grandparent;
        parent.rightchild = B;
        grandparent.leftchild = C;
      } else { //right-left
        TreeNode B = current.leftchild;
        TreeNode C = current.rightchild;

        current.leftchild = grandparent;
        current.rightchild = parent;
        grandparent.rightchild = B;
        parent.leftchild = C;
      }
    }
  }

  if (ancestors.length > 0) {
    ancestors.pop();
    if (current == root.leftchild) {
      TreeNode B = current.rightchild;
      current.rightchild = root;
      root.leftchild = B;
    } else {
      TreeNode B = current.leftchild;
      current.leftchild = root;
      root.rightchild = B;
    }
  }

  return current;
}

struct SplayTree_T {
  private TreeNode root = null;
  restricted int size = 0;

  private T emptyresponse;

  private bool operator < (T a, T b);
  void operator init(bool lessthan(T,T), T emptyresponse) {
    operator< = lessthan;
    this.emptyresponse = emptyresponse;
  }

  int size() {
    return size;
  }

  bool empty() {
    assert((root == null) == (size == 0));
    return root == null;
  }

  bool contains(T value) {
    TreeNode[] parentStack = new TreeNode[0];
    parentStack.cyclic = true;
    parentStack.push(root);
    while (true) {
      TreeNode current = parentStack[-1];
      if (current == null) {
        parentStack.pop();
        root = splay(parentStack, operator<);
        return false;
      }
      if (value < current.value) {
        parentStack.push(current.leftchild);
      } else if (current.value < value) {
        parentStack.push(current.rightchild);
      } else break;
    }
    root = splay(parentStack, operator<);
    return true;
  }

  T after(T item) {
    TreeNode[] parentStack = new TreeNode[0];
    parentStack.cyclic = true;
    parentStack.push(root);
    T strictUpperBound = emptyresponse;
    bool found = false;
    while (true) {
      TreeNode current = parentStack[-1];
      if (current == null) {
        parentStack.pop();
        root = splay(parentStack, operator<);
        return strictUpperBound;
      }
      if (found || item < current.value) {
        strictUpperBound = current.value;
        parentStack.push(current.leftchild);
      } else {
        parentStack.push(current.rightchild);
        if (!(current.value < item))
          found = true;
      }
    }
    assert(false, "Unreachable code");
    return emptyresponse;
  }

  T before(T item) {
    TreeNode[] parentStack = new TreeNode[0];
    parentStack.cyclic = true;
    parentStack.push(root);
    T strictLowerBound = emptyresponse;
    bool found = false;
    while (true) {
      TreeNode current = parentStack[-1];
      if (current == null) {
        parentStack.pop();
        root = splay(parentStack, operator<);
        return strictLowerBound;
      }
      if (found || current.value < item) {
        strictLowerBound = current.value;
        parentStack.push(current.rightchild);
      } else {
        parentStack.push(current.leftchild);
        if (!(item < current.value))
          found = true;
      }
    }
    assert(false, "Unreachable code");
    return emptyresponse;
  }

  T atOrAfter(T item) {
    TreeNode[] parentStack = new TreeNode[0];
    parentStack.cyclic = true;
    parentStack.push(root);
    T upperBound = emptyresponse;
    while (true) {
      TreeNode current = parentStack[-1];
      if (current == null) {
        parentStack.pop();
        root = splay(parentStack, operator<);
        return upperBound;
      }
      if (current.value < item) {
        parentStack.push(current.rightchild);
      } else if (item < current.value) {
        upperBound = current.value;
        parentStack.push(current.leftchild);
      } else {
        root = splay(parentStack, operator<);
        return current.value;
      }
    }
    assert(false, "Unreachable code");
    return emptyresponse;
  }

  T atOrBefore(T item) {
    TreeNode[] parentStack = new TreeNode[0];
    parentStack.cyclic = true;
    parentStack.push(root);
    T lowerBound = emptyresponse;
    while (true) {
      TreeNode current = parentStack[-1];
      if (current == null) {
        parentStack.pop();
        root = splay(parentStack, operator<);
        return lowerBound;
      }
      if (item < current.value) {
        parentStack.push(current.leftchild);
      } else if (current.value < item) {
        lowerBound = current.value;
        parentStack.push(current.rightchild);
      } else {
        root = splay(parentStack, operator<);
        return current.value;
      }
    }
    assert(false, "Unreachable code");
    return emptyresponse;
  }

  T min() {
    if (root == null) return emptyresponse;
    TreeNode[] ancestors = new TreeNode[0];
    ancestors.cyclic = true;
    TreeNode current = root;
    while (current != null) {
      ancestors.push(current);
      current = current.leftchild;
    }
    root = splay(ancestors, operator<);
    return root.value;
  }

  T popMin() {
    if (root == null) return emptyresponse;
    TreeNode[] ancestors = new TreeNode[0];
    ancestors.cyclic = true;
    TreeNode current = root;
    while (current != null) {
      ancestors.push(current);
      current = current.leftchild;
    }
    root = splay(ancestors, operator<);
    T toReturn = root.value;
    root = root.rightchild;
    --size;
    return toReturn;
  }

  T max() {
    if (root == null) return emptyresponse;
    TreeNode[] ancestors = new TreeNode[0];
    ancestors.cyclic = true;
    TreeNode current = root;
    while (current != null) {
      ancestors.push(current);
      current = current.rightchild;
    }
    root = splay(ancestors, operator<);
    return root.value;
  }

  T popMax() {
    if (root == null) return emptyresponse;
    TreeNode[] ancestors = new TreeNode[0];
    ancestors.cyclic = true;
    TreeNode current = root;
    while (current != null) {
      ancestors.push(current);
      current = current.rightchild;
    }
    root = splay(ancestors, operator<);
    T toReturn = root.value;
    root = root.leftchild;
    --size;
    return toReturn;
  }

  /*
   * returns true iff the tree was modified
   */
  bool add(T value) {
    if (root == null) {
      root = TreeNode(value);
      ++size;
      return true;
    }
    TreeNode[] ancestors = new TreeNode[0];
    ancestors.cyclic = true;
    ancestors.push(root);

    bool toReturn = false;
    
    while (!toReturn) {
      TreeNode current = ancestors[-1];
      if (value < current.value) {
        if (current.leftchild == null) {
          current.leftchild = TreeNode(value);
          toReturn = true;
        }
        ancestors.push(current.leftchild);			
      } else if (current.value < value) {
        if (current.rightchild == null) {
          current.rightchild = TreeNode(value);
          toReturn = true;
        }
        ancestors.push(current.rightchild);
      } else {
        root = splay(ancestors, operator<);
        return false;
      }
    }

    root = splay(ancestors, operator<);
    ++size;
    return true;
  }

  T push(T item) {
    if (root == null) {
      add(item);
      return emptyresponse;
    }
    TreeNode[] ancestors = new TreeNode[0];
    ancestors.cyclic = true;
    ancestors.push(root);
    TreeNode current = root;
    while (true) {
      if (item < current.value) {
        if (current.leftchild == null) {
          current.leftchild = TreeNode(item);
          ancestors.push(current.leftchild);
          break;
        }
        ancestors.push(current.leftchild);
        current = current.leftchild;
      } else if (current.value < item) {
        if (current.rightchild == null) {
          current.rightchild = TreeNode(item);
          ancestors.push(current.rightchild);
          break;
        }
        ancestors.push(current.rightchild);
        current = current.rightchild;
      } else {
        T toReturn = current.value;
        current.value = item;
        root = splay(ancestors, operator<);
        return toReturn;
      }
    }
    root = splay(ancestors, operator<);
    ++size;
    return emptyresponse;
  }

  T get(T item) {
    if (root == null) return emptyresponse;
    TreeNode[] parentStack = new TreeNode[0];
    parentStack.cyclic = true;
    parentStack.push(root);
    while (true) {
      TreeNode current = parentStack[-1];
      if (current == null) {
        parentStack.pop();
        root = splay(parentStack, operator<);
        return emptyresponse;
      }
      if (item < current.value) {
        parentStack.push(current.leftchild);
      } else if (current.value < item) {
        parentStack.push(current.rightchild);
      } else {
        root = splay(parentStack, operator<);
        return current.value;
      }
    }
    assert(false, "Unreachable code");
    return emptyresponse;
  }

  /*
   * returns the removed item, or emptyresponse if the item was not found
   */
  T extract(T value) {
    TreeNode[] ancestors = new TreeNode[0];
    ancestors.cyclic = true;  // Makes ancestors[-1] refer to the last entry.
    ancestors.push(root);

    while (true) {
      TreeNode current = ancestors[-1];
      if (current == null) {
        ancestors.pop();
        root = splay(ancestors, operator<);
        return emptyresponse;
      }
      if (value < current.value)
        ancestors.push(current.leftchild);
      else if (current.value < value)
        ancestors.push(current.rightchild);
      else break;
    }

    TreeNode toDelete = ancestors.pop();
    T retv = toDelete.value;
    TreeNode parent = null;
    if (ancestors.length > 0) parent = ancestors[-1];
    
    if (toDelete.leftchild == null) {
      if (parent != null)  {
        if (parent.rightchild == toDelete) {
          parent.rightchild = toDelete.rightchild;
        } else {
          parent.leftchild = toDelete.rightchild;
        }
      } else root = toDelete.rightchild;
    } else if (toDelete.rightchild == null) {
      if (parent == null) {
        root = toDelete.leftchild;
      } else if (parent.rightchild == toDelete) {
        parent.rightchild = toDelete.leftchild;
      } else parent.leftchild = toDelete.leftchild;
    } else {
      TreeNode[] innerStack = new TreeNode[0];
      innerStack.cyclic = true;
      TreeNode current = toDelete.rightchild;
      while (current != null) {
        innerStack.push(current);
        current = current.leftchild;
      }
      toDelete.rightchild = splay(innerStack, operator<);
      toDelete.value = toDelete.rightchild.value;
      toDelete.rightchild = toDelete.rightchild.rightchild;
    }

    if (parent != null) root = splay(ancestors, operator<);
    --size;
    return retv;    
  }

  void forEach(bool run(T)) {
    inOrderNonRecursive(root, run);
  }

  Iter_T operator iter() {
    Iter_T result = new Iter_T;
    if (root == null) {
      result.valid = new bool() { return false; };
      result.get = new T() { assert(false, 'Invalid iterator'); return emptyresponse; };
      result.advance = new void() { assert(false, 'Invalid iterator'); };
      return result;
    }
    TreeNode[] stack = new TreeNode[0];
    bool[] selfDone = new bool[0];
    stack.cyclic = true;
    selfDone.cyclic = true;
    stack.push(root);
    selfDone.push(false);
    while (stack[-1].leftchild != null) {
      stack.push(stack[-1].leftchild);
      selfDone.push(false);
    }
    result.valid = new bool() {
      return stack.length > 0;
    };
    result.get = new T() {
      return stack[-1].value;
    };
    result.advance = new void() {
      assert(stack.length > 0, 'Invalid iterator');
      selfDone[-1] = true;
      TreeNode current = stack[-1];
      if (current.rightchild != null) {
        current = current.rightchild;
        stack.push(current);
        selfDone.push(false);
        while (current.leftchild != null) {
          current = current.leftchild;
          stack.push(current);
          selfDone.push(false);
        }
      } else {
        while (stack.length > 0 && selfDone[-1]) {
          stack.pop();
          selfDone.pop();
        }
      }
    };
    return result;
  }

  autounravel SortedSet_T operator cast(SplayTree_T splaytree) {
    SortedSet_T result = new SortedSet_T;
    result.size = splaytree.size;
    result.empty = splaytree.empty;
    result.contains = splaytree.contains;
    result.after = splaytree.after;
    result.before = splaytree.before;
    result.atOrAfter = splaytree.atOrAfter;
    result.atOrBefore = splaytree.atOrBefore;
    result.min = splaytree.min;
    result.popMin = splaytree.popMin;
    result.max = splaytree.max;
    result.popMax = splaytree.popMax;
    result.add = splaytree.add;
    result.push = splaytree.push;
    result.get = splaytree.get;
    result.extract = splaytree.extract;
    result.operator iter = splaytree.operator iter;
    return result;
  }

  autounravel Set_T operator cast(SplayTree_T splaytree) {
    return (SortedSet_T)splaytree;
  }
}