/*****
 * treedef.asy
 * Andy Hammerlindl 2003/10/25
 *
 * Implements a dynamic binary search tree.
 *****/

struct tree
{
  tree left;
  tree right;
  int key = 0;
  int value = 0;

  // Writes the entries in order of their keys, one per line.
  autounravel void write(file out, tree t, suffix suffix)
  {
    bool first = true;
    void visit(tree t) {
      if (t == null)
        return;
      visit(t.left);
      if (!first)
        write(out,endl);
      first = false;
      write(out,t.key);
      write(out,"->");
      write(out,t.value);
      visit(t.right);
    }
    visit(t);
    suffix(out);
  }

  autounravel void write(file out=stdout, tree t)
  {
    if (t != null)
      write(out,t,endl);
  }
}

tree newtree()
{
  return null;
}

tree add(tree t, int key, int value)
{
  if (t == null) {
    tree tt;
    tt.key = key; tt.value = value;
    return tt;
  }
  else if (key == t.key) {
    return t;
  }
  else if (key < t.key) {
    tree tt;
    tt.left = add(t.left, key, value);
    tt.key = t.key;
    tt.value = t.value;
    tt.right = t.right;
    return tt;
  }
  else {
    tree tt;
    tt.left = t.left;
    tt.key = t.key;
    tt.value = t.value;
    tt.right = add(t.right, key, value);
    return tt;
  }
}

bool contains(tree t, int key)
{
  if (t == null)
    return false;
  else if (key == t.key)
    return true;
  else if (key < t.key)
    return contains(t.left, key);
  else
    return contains(t.right, key);
}

int lookup(tree t, int key)
{
  if (t == null)
    return 0;
  else if (key == t.key)
    return t.value;
  else if (key < t.key)
    return lookup(t.left, key);
  else
    return lookup(t.right, key);
}
