typedef import(T);

from sortedset(T=T) access Set_T, SortedSet_T, operator cast,
    makeNaiveSortedSet;  // needed for bug workaround: https://github.com/vectorgraphics/asymptote/issues/429

private struct treenode {
  treenode leftchild;
  treenode rightchild;
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
  treenode node;
  int progress = NodeProgressEnum.NOT_STARTED;
  void operator init(treenode node) {
    this.node = node;
  }
}

void inOrderNonRecursive(treenode root, bool run(T)) {
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


private treenode splay(treenode[] ancestors, bool lessthan(T a, T b)) {
  bool operator < (T a, T b) = lessthan;
  
  if (ancestors.length == 0) return null;

  treenode root = ancestors[0];
  treenode current = ancestors.pop();
  
  while (ancestors.length >= 2) {
    treenode parent = ancestors.pop();
    treenode grandparent = ancestors.pop();

    if (ancestors.length > 0) {
      treenode greatparent = ancestors[-1];
      if (greatparent.leftchild == grandparent) {
        greatparent.leftchild = current;
      } else greatparent.rightchild = current;
    }

    bool currentside = (parent.leftchild == current);
    bool grandside = (grandparent.leftchild == parent);

    if (currentside == grandside) { // zig-zig
      if (currentside) { // both left
        treenode B = current.rightchild;
        treenode C = parent.rightchild;

        current.rightchild = parent;
        parent.leftchild = B;
        parent.rightchild = grandparent;
        grandparent.leftchild = C;
      } else { // both right
        treenode B = parent.leftchild;
        treenode C = current.leftchild;

        current.leftchild = parent;
        parent.leftchild = grandparent;
        parent.rightchild = C;
        grandparent.rightchild = B;
      }
    } else { // zig-zag
      if (grandside) {  // left-right
        treenode B = current.leftchild;
        treenode C = current.rightchild;

        current.leftchild = parent;
        current.rightchild = grandparent;
        parent.rightchild = B;
        grandparent.leftchild = C;
      } else { //right-left
        treenode B = current.leftchild;
        treenode C = current.rightchild;

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
      treenode B = current.rightchild;
      current.rightchild = root;
      root.leftchild = B;
    } else {
      treenode B = current.leftchild;
      current.leftchild = root;
      root.rightchild = B;
    }
  }

  return current;
}

struct SplayTree_T {
  private treenode root = null;
  restricted int size = 0;
  private bool operator < (T a, T b);
  private T emptyresponse;

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
    treenode[] parentStack = new treenode[0];
    parentStack.cyclic = true;
    parentStack.push(root);
    while (true) {
      treenode current = parentStack[-1];
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
    treenode[] parentStack = new treenode[0];
    parentStack.cyclic = true;
    parentStack.push(root);
    T strictUpperBound = emptyresponse;
    bool found = false;
    while (true) {
      treenode current = parentStack[-1];
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
    treenode[] parentStack = new treenode[0];
    parentStack.cyclic = true;
    parentStack.push(root);
    T strictLowerBound = emptyresponse;
    bool found = false;
    while (true) {
      treenode current = parentStack[-1];
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

  T firstGEQ(T item) {
    treenode[] parentStack = new treenode[0];
    parentStack.cyclic = true;
    parentStack.push(root);
    T upperBound = emptyresponse;
    while (true) {
      treenode current = parentStack[-1];
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

  T firstLEQ(T item) {
    treenode[] parentStack = new treenode[0];
    parentStack.cyclic = true;
    parentStack.push(root);
    T lowerBound = emptyresponse;
    while (true) {
      treenode current = parentStack[-1];
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
    treenode[] ancestors = new treenode[0];
    ancestors.cyclic = true;
    treenode current = root;
    while (current != null) {
      ancestors.push(current);
      current = current.leftchild;
    }
    root = splay(ancestors, operator<);
    return root.value;
  }

  T popMin() {
    if (root == null) return emptyresponse;
    treenode[] ancestors = new treenode[0];
    ancestors.cyclic = true;
    treenode current = root;
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
    treenode[] ancestors = new treenode[0];
    ancestors.cyclic = true;
    treenode current = root;
    while (current != null) {
      ancestors.push(current);
      current = current.rightchild;
    }
    root = splay(ancestors, operator<);
    return root.value;
  }

  T popMax() {
    if (root == null) return emptyresponse;
    treenode[] ancestors = new treenode[0];
    ancestors.cyclic = true;
    treenode current = root;
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
  bool insert(T value) {
    if (root == null) {
      root = treenode(value);
      ++size;
      return true;
    }
    treenode[] ancestors = new treenode[0];
    ancestors.cyclic = true;
    ancestors.push(root);

    bool toReturn = false;
    
    while (!toReturn) {
      treenode current = ancestors[-1];
      if (value < current.value) {
        if (current.leftchild == null) {
          current.leftchild = treenode(value);
          toReturn = true;
        }
        ancestors.push(current.leftchild);			
      } else if (current.value < value) {
        if (current.rightchild == null) {
          current.rightchild = treenode(value);
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

  T replace(T item) {
    if (root == null) {
      insert(item);
      return emptyresponse;
    }
    treenode[] ancestors = new treenode[0];
    ancestors.cyclic = true;
    ancestors.push(root);
    treenode current = root;
    while (true) {
      if (item < current.value) {
        if (current.leftchild == null) {
          current.leftchild = treenode(item);
          ancestors.push(current.leftchild);
          break;
        }
        ancestors.push(current.leftchild);
        current = current.leftchild;
      } else if (current.value < item) {
        if (current.rightchild == null) {
          current.rightchild = treenode(item);
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
    treenode[] parentStack = new treenode[0];
    parentStack.cyclic = true;
    parentStack.push(root);
    while (true) {
      treenode current = parentStack[-1];
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
   * returns true iff the tree was modified
   */
  bool delete(T value) {
    treenode[] ancestors = new treenode[0];
    ancestors.cyclic = true;  // Makes ancestors[-1] refer to the last entry.
    ancestors.push(root);

    while (true) {
      treenode current = ancestors[-1];
      if (current == null) {
        ancestors.pop();
        root = splay(ancestors, operator<);
        return false;
      }
      if (value < current.value)
        ancestors.push(current.leftchild);
      else if (current.value < value)
        ancestors.push(current.rightchild);
      else break;
    }

    treenode toDelete = ancestors.pop();
    treenode parent = null;
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
      treenode[] innerStack = new treenode[0];
      innerStack.cyclic = true;
      treenode current = toDelete.rightchild;
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
    return true;    
  }

  void forEach(bool run(T)) {
    inOrderNonRecursive(root, run);
  }
  
}

SortedSet_T operator cast(SplayTree_T splaytree) {
  SortedSet_T result = new SortedSet_T;
  result.size = splaytree.size;
  result.empty = splaytree.empty;
  result.contains = splaytree.contains;
  result.after = splaytree.after;
  result.before = splaytree.before;
  result.firstGEQ = splaytree.firstGEQ;
  result.firstLEQ = splaytree.firstLEQ;
  result.min = splaytree.min;
  result.popMin = splaytree.popMin;
  result.max = splaytree.max;
  result.popMax = splaytree.popMax;
  result.insert = splaytree.insert;
  result.replace = splaytree.replace;
  result.get = splaytree.get;
  result.delete = splaytree.delete;
  result.forEach = splaytree.forEach;
  return result;
}

Set_T operator cast(SplayTree_T splaytree) {
  return (SortedSet_T)splaytree;
}

T[] operator cast(SplayTree_T splaytree) {
  return (SortedSet_T)splaytree;
}