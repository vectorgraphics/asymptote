/*****
 * treetest.asy
 * Andy Hammerlindl 2003/11/09
 *
 * Tests the import feature by importing tree and using some simple
 * examples.
 *****/

import tree;

tree t = newtree();
t = add(t, 5, 3);
t = add(t, 4, 7);
t = add(t, 10, 100);
write(t);
