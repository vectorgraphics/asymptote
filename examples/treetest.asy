import drawtree;

treeLevelStep = 2cm;

TreeNode root = makeNode( "Root" );
TreeNode child1 = makeNode( root, "Child\_1" );
TreeNode child2 = makeNode( root, "Child\_2" );

TreeNode gchild1 = makeNode( child1, "Grandchild\_1" );
TreeNode gchild2 = makeNode( child1, "Grandchild\_2" );

TreeNode gchild3 = makeNode( child1, "Grandchild\_3" );
TreeNode gchild4 = makeNode( child1, "Grandchild\_4" );


TreeNode gchild11 = makeNode( child2, "Grandchild\_1" );
TreeNode gchild22 = makeNode( child2, "Grandchild\_2" );

TreeNode ggchild1 = makeNode( gchild1, "Great Grandchild\_1" );

draw( root, (0,0) );

