typedef import(T);

private struct treenode {
  treenode leftchild;
  treenode rightchild;
  T value;
  void operator init(T value) {
    this.value = value;
  }

  void inOrder(void run(T)) {
    if (leftchild != null) leftchild.inOrder(run);
    run(value);
    if (rightchild != null) rightchild.inOrder(run);
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

struct splaytree_T {
  treenode root = null;
  restricted int size = 0;
  private bool operator < (T a, T b);

  void operator init(bool lessthan(T,T)) {
    operator< = lessthan;
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

  /*
   * returns true iff the tree was modified
   */
  bool add(T value) {
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

  T popMax(T default) {
    if (root == null) return default;
    treenode[] ancestors = new treenode[0];
    ancestors.cyclic = true;
    treenode current = root;
    while (current != null) {
      ancestors.push(current);
      current = current.rightchild;
    }
    root = splay(ancestors, operator<);
    T toReturn = root.value;
    // TODO(optimization): Refactor deleteRoot out of the delete function,
    // and call deleteRoot instead of delete.
    delete(toReturn);  
    return toReturn;
  }

  bool empty() {
    assert((root == null) == (size == 0));
    return root == null;
  }

  void forEach(void run(T)) {
    if (root == null) return;
    root.inOrder(run); 
  }
  
}
