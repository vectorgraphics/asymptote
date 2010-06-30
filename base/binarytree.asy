/* **********************************************************************
 * binarytree: An Asymptote module to draw binary trees                 *
 *                                                                      *
 * Copyright (C) 2006                                                   *
 * Tobias Langner tobias[at]langner[dot]nightlabs[dot]de                *
 *                                                                      *
 * Modified by John Bowman                                              *
 *                                                                      *
 ************************************************************************
 *                                                                      *
 * This library is free software; you can redistribute it and/or        *
 * modify it under the terms of the GNU Lesser General Public           *
 * License as published by the Free Software Foundation; either         *
 * version 3 of the License, or (at your option) any later version.     *
 *                                                                      *
 * This library is distributed in the hope that it will be useful,      *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of       *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU    *
 * Lesser General Public License for more details.                      *
 *                                                                      *
 * You should have received a copy of the GNU Lesser General Public     *
 * License along with this library; if not, write to the                *
 *     Free Software Foundation, Inc.,                                  *
 *     51 Franklin St, Fifth Floor,                                     *
 *     Boston, MA  02110-1301  USA                                      *
 *                                                                      *
 * Or get it online:                                                    *
 *     http://www.gnu.org/copyleft/lesser.html                          *
 *                                                                      *
 ***********************************************************************/

/**
 * default values
 */
real minDistDefault=0.2cm;
real nodeMarginDefault=0.1cm;

/**
 * structure to represent nodes in a binary tree
 */
struct binarytreeNode {
  int key;
  binarytreeNode left;
  binarytreeNode right;
  binarytreeNode parent;
        
  /**
   * sets the left child of this node
   */
  void setLeft(binarytreeNode left) {
    this.left=left;
    this.left.parent=this;
  }

  /**
   * sets the right child of this node
   */
  void setRight(binarytreeNode right) {
    this.right=right;
    this.right.parent=this;
  }

  /**
   * returns a boolean indicating whether this node is the root
   */
  bool isRoot() {
    return parent == null;
  }

  /**
   * Returns the level of the subtree rooted at this node.
   */
  int getLevel() {
    if(isRoot())
      return 1;
    else
      return parent.getLevel()+1;
  }
        
  /**
   * sets the children of this binarytreeNode
   */
  void setChildren(binarytreeNode left, binarytreeNode right) {
    setLeft(left);
    setRight(right);
  }
        
  /**
   * creates a new binarytreeNode with key <key> 
   */
  static binarytreeNode binarytreeNode(int key) {
    binarytreeNode toReturn=new binarytreeNode;
    toReturn.key=key;
    return toReturn;
  }
        
  /**
   * returns the height of the subtree rooted at this node.
   */
  int getHeight() {
    if(left == null && right == null)
      return 1;
    if(left == null)
      return right.getHeight()+1;
    if(right == null)
      return left.getHeight()+1;
                
    return max(left.getHeight(),right.getHeight())+1;
  }
}

binarytreeNode operator init() {return null;}

/**
 * "constructor" for binarytreeNode
 */
binarytreeNode binarytreeNode(int key)=binarytreeNode.binarytreeNode;


/**
 * draws the tree rooted at the given <node> at the given position <pos>, with
 * <height>: the height of the containing tree,
 * <minDist>: the minimal horizontal distance of two nodes at the lowest level,
 * <levelDist> the vertical distance between two levels,
 * <nodeDiameter>: the diameter of one node.
 */
object draw(picture pic=currentpicture, binarytreeNode node, pair pos,
            int height, real minDist, real levelDist, real nodeDiameter,
            pen p=currentpen) {
  Label label=Label(math((string) node.key),pos);
        
  binarytreeNode left=node.left;        
  binarytreeNode right=node.right;

  /**
   * returns the distance for two nodes at the given <level> when the
   * containing tree has height <height> 
   * and the minimal distance between two nodes is <minDist>.
   */
  real getDistance(int level, int height, real minDist) {
    return(nodeDiameter+minDist)*2^(height-level);
  }

  real dist=getDistance(node.getLevel(),height,minDist)/2;

  /**
   * draws the connection between the two nodes at the given positions
   * by calculating the connection points
   * and then drawing the corresponding arrow.
   */
  void deferredDrawNodeConnection(pair parentPos, pair childPos) {
    pic.add(new void(frame f, transform t) {
        pair start,end; 
        // calculate connection path 
        transform T=shift(nodeDiameter/2*unit(t*childPos-t*parentPos));  
        path arr=(T*t*parentPos)--(inverse(T)*t*childPos);  
        draw(f,PenMargin(arr,p).g,p,Arrow(5));  
      }); 
    pic.addPoint(parentPos);
    pic.addPoint(childPos);
  } 

  if(left != null) {
    pair childPos=pos-(0,levelDist)-(dist/2,0);
    draw(pic,left,childPos,height,minDist,levelDist,nodeDiameter,p);
    deferredDrawNodeConnection(pos,childPos);
  }

  if(right != null) {
    pair childPos=pos-(0,levelDist)+(dist/2,0);
    draw(pic,right,childPos,height,minDist,levelDist,nodeDiameter,p);
    deferredDrawNodeConnection(pos,childPos);
  }
        
  picture obj;
  draw(obj,circle((0,0),nodeDiameter/2),p);
  label(obj,label,(0,0),p);
        
  add(pic,obj,pos);
        
  return label;
}

struct key {
  int n;
  bool active;
}

key key(int n, bool active=true) {key k; k.n=n; k.active=active; return k;}

key operator cast(int n) {return key(n);}
int operator cast(key k) {return k.n;}
int[] operator cast(key[] k) {
  int[] I;
  for(int i=0; i < k.length; ++i)
    I[i]=k[i].n;
  return I;
}

key nil=key(0,false);

/**
 * structure to represent a binary tree.
 */
struct binarytree {
  binarytreeNode root;
  int[] keys;
        
  /**
   * adds the given < key > to the tree by searching for its place and inserting it there.
   */
  void addKey(int key) {
    binarytreeNode newNode=binarytreeNode(key);
                
    if(root == null) {
      root=newNode;
      keys.push(key);
      return; 
    }
                
    binarytreeNode n=root;
    while(n != null) {
      if(key < n.key) {
        if(n.left != null)
          n=n.left;
        else {
          n.setLeft(newNode);
          keys.push(key);
          return;
        }
      } else if(key > n.key) {
        if(n.right != null)
          n=n.right;
        else {
          n.setRight(newNode);
          keys.push(key);
          return;
        }
      }
    }
  }
        
  /**
   * returns the height of the tree
   */
  int getHeight() {
    if(root == null)
      return 0;
    else
      return root.getHeight();
  }
        
  /**
   * adds all given keys to the tree sequentially
   */
  void addSearchKeys(int[] keys) {
    for(int i=0; i < keys.length; ++i) {
      int key=keys[i];
      // Ignore duplicate keys
      if(find(this.keys == key) == -1)
        addKey(key);
    }
  }
        
  binarytreeNode build(key[] keys, int[] ind) {
    if(ind[0] >= keys.length) return null;
    key k=keys[ind[0]];
    ++ind[0];
    if(!k.active) return null;
    binarytreeNode bt=binarytreeNode(k);
    binarytreeNode left=build(keys,ind);
    binarytreeNode right=build(keys,ind);
    bt.left=left; bt.right=right;
    if(left != null) left.parent=bt;
    if(right != null) right.parent=bt;
    return bt;
  }

  void addKeys(key[] keys) {
    int[] ind={0};
    root=build(keys,ind);
    this.keys=keys;
  }


  /**
   * returns all key in the tree
   */
  int[] getKeys() {
    return keys;
  }
}

binarytree searchtree(...int[] keys)
{
  binarytree bt;
  bt.addSearchKeys(keys);
  return bt;
}

binarytree binarytree(...key[] keys)
{
  binarytree bt;
  bt.addKeys(keys);
  return bt;
}

/**
 * draws the given binary tree.
 */
void draw(picture pic=currentpicture, binarytree tree,
          real minDist=minDistDefault, real nodeMargin=nodeMarginDefault,
          pen p=currentpen)
{
  int[] keys=tree.getKeys();
        
  // calculate the node diameter so that all keys fit into it
  frame f; 
  for(int i=0; i < keys.length; ++i)
    label(f,math(string(keys[i])),p);

  real nodeDiameter=abs(max(f)-min(f))+2*nodeMargin;
  real levelDist=nodeDiameter*1.8;

  draw(pic,tree.root,(0,0),tree.getHeight(),minDist,levelDist,nodeDiameter,p);
}
