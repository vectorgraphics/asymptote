private import math;
import three;

real epsilon=10*realEpsilon;

// Routines for hidden surface removal (via binary space partition):
// Structure face is derived from picture.
struct face {
  picture pic;
  transform t;
  frame fit;
  triple normal,point;
  triple min,max;
  void operator init(path3 p) {
    this.normal=normal(p);
    if(this.normal == O) abort("path is linear");
    this.point=point(p,0);
    min=min(p);
    max=max(p);
  }
  face copy() {
    face f=new face;
    f.pic=pic.copy();
    f.t=t;
    f.normal=normal;
    f.point=point;
    f.min=min;
    f.max=max;
    add(f.fit,fit);
    return f;
  }
}

picture operator cast(face f) {return f.pic;}
face operator cast(path3 p) {return face(p);}
  
struct line {
  triple point;
  triple dir;
}

private line intersection(face a, face b) 
{
  line L;
  L.point=intersectionpoint(a.normal,a.point,b.normal,b.point);
  L.dir=unit(cross(a.normal,b.normal));
  return L;
}

struct half {
  pair[] left,right;
  
  // Sort the points in the pair array z according to whether they lie on the
  // left or right side of the line L in the direction dir passing through P.
  // Points exactly on L are considered to be on the right side.
  // Also push any points of intersection of L with the path operator --(... z)
  // onto each of the arrays left and right. 
  void operator init(pair dir, pair P ... pair[] z) {
    pair lastz;
    pair invdir=dir != 0 ? 1/dir : 0;
    bool left,last;
    for(int i=0; i < z.length; ++i) {
      left=(invdir*z[i]).y > (invdir*P).y;
      if(i > 0 && last != left) {
        pair w=extension(P,P+dir,lastz,z[i]);
        this.left.push(w);
        this.right.push(w);
      }
      if(left) this.left.push(z[i]);
      else this.right.push(z[i]);
      last=left;
      lastz=z[i];
    }
  }
}
  
struct splitface {
  face back,front;
}

// Return the pieces obtained by splitting face a by face cut.
splitface split(face a, face cut, projection P)
{
  splitface S;

  void nointersection() {
    if(abs(dot(a.point-P.camera,a.normal)) >= 
       abs(dot(cut.point-P.camera,cut.normal))) {
      S.back=a;
      S.front=null;
    } else {
      S.back=null;
      S.front=a;
    }
  }

  if(P.infinity) {
    P=P.copy();
    static real factor=1/sqrtEpsilon;
    P.camera *= factor*max(abs(a.min),abs(a.max),
                           abs(cut.min),abs(cut.max));
  }

  if((abs(a.normal-cut.normal) < epsilon ||
      abs(a.normal+cut.normal) < epsilon)) {
    nointersection();
    return S;
  }

  line L=intersection(a,cut);

  if(dot(P.camera-L.point,P.camera-P.target) < 0) {
    nointersection();
    return S;
  }
    
  pair point=a.t*project(L.point,P);
  pair dir=a.t*project(L.point+L.dir,P)-point;
  pair invdir=dir != 0 ? 1/dir : 0;
  triple apoint=L.point+cross(L.dir,a.normal);
  bool left=(invdir*(a.t*project(apoint,P))).y >= (invdir*point).y;

  real t=intersect(apoint,P.camera,cut.normal,cut.point);
  bool rightfront=left ^ (t <= 0 || t >= 1);
  
  face back=a, front=a.copy();
  pair max=max(a.fit);
  pair min=min(a.fit);
  half h=half(dir,point,max,(min.x,max.y),min,(max.x,min.y),max);
  if(h.right.length == 0) {
    if(rightfront) front=null;
    else back=null;
  } else if(h.left.length == 0) {
    if(rightfront) back=null;
    else front=null;
  }
  if(front != null)
    clip(front.fit,operator --(... rightfront ? h.right : h.left)--cycle,
         zerowinding);
  if(back != null)
    clip(back.fit,operator --(... rightfront ? h.left : h.right)--cycle,
         zerowinding);
  S.back=back;
  S.front=front;
  return S;
}

// A binary space partition
struct bsp
{
  bsp back;
  bsp front;
  face node;
  
  // Construct the bsp.
  void operator init(face[] faces, projection P) {
    if(faces.length != 0) {
      this.node=faces.pop();
      face[] front,back;
      for(int i=0; i < faces.length; ++i) {
        splitface split=split(faces[i],this.node,P);
        if(split.front != null) front.push(split.front);
        if(split.back != null) back.push(split.back);
      }
      this.front=bsp(front,P);
      this.back=bsp(back,P);
    }
  }
  
  // Draw from back to front.
  void add(frame f) {
    if(back != null) back.add(f);
    add(f,node.fit,group=true);
    if(labels(node.fit)) layer(f); // Draw over any existing TeX layers.
    if(front != null) front.add(f);
  }
}

void add(picture pic=currentpicture, face[] faces,
         projection P=currentprojection)
{
  int n=faces.length;
  face[] Faces=new face[n];
  for(int i=0; i < n; ++i)
    Faces[i]=faces[i].copy();
  
  pic.add(new void (frame f, transform t, transform T,
                                pair m, pair M) {
                        // Fit all of the pictures so we know their exact sizes.
                        face[] faces=new face[n];
                        for(int i=0; i < n; ++i) {
                          faces[i]=Faces[i].copy();
                          face F=faces[i];
                          F.t=t*T*F.pic.T;
                          F.fit=F.pic.fit(t,T*F.pic.T,m,M);
                        }
    
                        bsp bsp=bsp(faces,P);
                        if(bsp != null) bsp.add(f);
          });
    
  for(int i=0; i < n; ++i) {
    picture F=Faces[i].pic;
    pic.userBox3(F.userMin3(), F.userMax3());
    pic.bounds.append(F.T, F.bounds);
    // The above 2 lines should be replaced with a routine in picture which
    // copies only sizing data from another picture.
  }
}
