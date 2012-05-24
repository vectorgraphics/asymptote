import three;

size(300);

// A structure to subdivide two intersecting patches about their intersection.
struct split
{
  surface[] S=sequence(new surface(int i) {return new surface;},1);
  surface[] T=sequence(new surface(int i) {return new surface;},1);

  struct tree {
    tree[] tree=new tree[2];
  }
  // Default subdivision depth.
  int n=20;

  // Subdivide p and q to depth n if they overlap.
  void write(tree pt, tree qt, triple[][] p, triple[][] q, int depth=n) {
    --depth;
    triple[][][] Split(triple[][] P)=depth % 2 == 0 ? hsplit : vsplit;
    triple[][][] P=Split(p);
    triple[][][] Q=Split(q);

    for(int i=0; i < 2; ++i) {
      triple[][] Pi=P[i];
      for(int j=0; j < 2; ++j) {
        triple[][] Qj=Q[j];
        if(overlap(Pi,Qj)) {
          if(!pt.tree.initialized(i))
            pt.tree[i]=new tree;
          if(!qt.tree.initialized(j))
            qt.tree[j]=new tree;
          if(depth > 0)
            write(pt.tree[i],qt.tree[j],Pi,Qj,depth);
        }
      }    
    }
  }
  
  // Output the subpatches of p from subdivision.
  void read(surface[] S, tree t, triple[][] p, int depth=n) {
    --depth;
    triple[][][] Split(triple[][] P)=depth % 2 == 0 ? hsplit : vsplit;
    triple[][][] P=Split(p);

    for(int i=0; i < 2; ++i) {
      if(t.tree.initialized(i)) 
        read(S,t.tree[i],P[i],depth);
      else {
        S[0].push(patch(P[i]));
      }
    }
  }

  void operator init(triple[][] p, triple[][] q, int depth=n) {
    tree ptrunk,qtrunk;
    write(ptrunk,qtrunk,p,q,depth);
    read(T,ptrunk,p,depth);
    read(S,qtrunk,q,depth);
  }
}

currentprojection=orthographic(0,0,1);

triple[][] A={
  {(0,0,0),(1,0,0),(1,0,0),(2,0,0)},
  {(0,4/3,0),(2/3,4/3,2),(4/3,4/3,2),(2,4/3,0)},
  {(0,2/3,0),(2/3,2/3,0),(4/3,2/3,0),(2,2/3,0)},
  {(0,2,0),(2/3,2,0),(4/3,2,0),(2,2,0)}
};

triple[][] B={
  {(0.5,0,-1),(0.5,1,-1),(0.5,2,-1),(0.5,3,-1)},
  {(0.5,0,0),(0.5,1,0),(0.5,2,0),(0.5,3,0)},
  {(0.5,0,1),(0.5,1,1),(0.5,2,1),(0.5,3,1)},
  {(0.5,0,2),(0.5,1,2),(0.5,2,2),(0.5,3,2)}
};

split S=split(B,A);

defaultrender.merge=true;

for(int i=0; i < S.S[0].s.length; ++i)
  draw(surface(S.S[0].s[i]),Pen(i));

for(int i=0; i < S.T[0].s.length; ++i)
  draw(surface(S.T[0].s[i]),Pen(i));
