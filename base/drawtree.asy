// A simple tree drawing module contributed by adarovsky
// See example treetest.asy

real treeNodeStep = 0.5cm;
real treeLevelStep = 1cm;
real treeMinNodeWidth = 2cm;

struct TreeNode {
  TreeNode parent;
  TreeNode[] children;

  frame content;

  pair pos;
  real adjust;
}

void add( TreeNode child, TreeNode parent )
{
  child.parent = parent;
  parent.children.push( child );
}

TreeNode makeNode( TreeNode parent = null, frame f )
{
  TreeNode child = new TreeNode;
  child.content = f;
  if( parent != null ) {
    add( child, parent );
  }
  return child;
}

TreeNode makeNode( TreeNode parent = null, Label label )
{
  frame f;
  box( f, label);
  return makeNode( parent, f );
}


real layout( int level, TreeNode node )
{
  if( node.children.length > 0 ) {
    real width[] = new real[node.children.length];
    real curWidth = 0;

    for( int i = 0; i < node.children.length; ++i ) {
      width[i] = layout( level+1, node.children[i] );

      node.children[i].pos = (curWidth + width[i]/2,
                              -level*treeLevelStep);
      curWidth += width[i] + treeNodeStep;
    }

    real midPoint = ( sum( width )+treeNodeStep*(width.length-1)) / 2;
    for( int i = 0; i < node.children.length; ++i ) {
      node.children[i].adjust = - midPoint;
    }

    return max( (max(node.content)-min(node.content)).x,
                sum(width)+treeNodeStep*(width.length-1) );
  }
  else {
    return max( treeMinNodeWidth, (max(node.content)-min(node.content)).x );
  }
}

void drawAll( TreeNode node, frame f )
{
  pair pos;
  if( node.parent != null )
    pos = (node.parent.pos.x+node.adjust, 0);
  else
    pos = (node.adjust, 0);
  node.pos += pos;

  node.content = shift(node.pos)*node.content;
  add( f, node.content );


  if( node.parent != null ) {
    path p = point(node.content, N)--point(node.parent.content,S);
    draw( f, p, currentpen );
  }

  for( int i = 0; i < node.children.length; ++i )
    drawAll( node.children[i], f );
}

void draw( TreeNode root, pair pos )
{
  frame f;

  root.pos = (0,0);
  layout( 1, root );

  drawAll( root, f );

  add(f,pos);
}
