// A simple tree drawing module contributed by adarovsky
// See example treetest.asy

public real treeNodeStep = 0.5cm;
public real treeLevelStep = 1cm;

struct TreeNode {
public TreeNode parent;
public TreeNode[] children;

public frame content;

public pair pos;
public real adjust;
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
  if( !alias(parent, null) ) {
    add( child, parent );
  }
  return child;
}

TreeNode makeNode( TreeNode parent = null, string label )
{
  frame f;
  labelbox( f, label, (0,0) );
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

    real midPoint = sum( width ) / 2;
    for( int i = 0; i < node.children.length; ++i ) {
      node.children[i].adjust = - midPoint;
    }

    return max( (max(node.content)-min(node.content)).x,
		sum(width)+treeNodeStep*width.length );
  }
  else {
    return max( 2cm, (max(node.content)-min(node.content)).x );
  }
}

void drawAll( TreeNode node, frame f )
{
  pair pos;
  if( !alias(node.parent, null) )
    pos = (node.parent.pos.x+node.adjust, 0);
  else
    pos = (node.adjust, 0);
  node.pos += pos;

  node.content = shift(node.pos)*node.content;
  add( f, node.content );


  if( !alias(node.parent, null) ) {
    path p = point(node.content, N)--point(node.parent.content,S);
    draw( f, p, currentpen );
  }

  for( int i = 0; i < node.children.length; ++i )
    drawAll( node.children[i], f );
}

void draw( TreeNode root, pair pos )
{
  frame f;

  root.pos = pos;
  layout( 1, root );

  drawAll( root, f );

  add(pos,f);
}
