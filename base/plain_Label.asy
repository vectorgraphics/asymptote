real angle(transform t)
{
  pair z=(2t.xx*t.yy,t.yx*t.yy-t.xx*t.xy);
  if(t.xx < 0 || t.yy < 0) z=-z;
  return degrees(z,warn=false);
}

transform rotation(transform t)
{
  return rotate(angle(t));
}

transform scaleless(transform t)
{
  real a=t.xx, b=t.xy, c=t.yx, d=t.yy;
  real arg=(a-d)^2+4b*c;
  pair delta=arg >= 0 ? sqrt(arg) : I*sqrt(-arg);
  real trace=a+d; 
  pair l1=0.5(trace+delta);
  pair l2=0.5(trace-delta);
  
  if(abs(delta) < sqrtEpsilon*max(abs(l1),abs(l2))) {
    real s=abs(0.5trace);
    return (s != 0) ? scale(1/s)*t : t;
  }

  if(abs(l1-d) < abs(l2-d)) {pair temp=l1; l1=l2; l2=temp;}

  pair dot(pair[] u, pair[] v) {return conj(u[0])*v[0]+conj(u[1])*v[1];}

  pair[] unit(pair[] u) {
    real norm2=abs(u[0])^2+abs(u[1])^2;
    return norm2 != 0 ? u/sqrt(norm2) : u;
  }

  pair[] u={l1-d,b};
  pair[] v={c,l2-a};
  u=unit(u);
  pair d=dot(u,u);
  if(d != 0) v -= dot(u,v)/d*u;
  v=unit(v);

  pair[][] U={{u[0],v[0]},{u[1],v[1]}};
  pair[][] A={{a,b},{c,d}};

  pair[][] operator *(pair[][] a, pair[][] b) {
    pair[][] c=new pair[2][2];
    for(int i=0; i < 2; ++i) {
      for(int j=0; j < 2; ++j) {
        c[i][j]=a[i][0]*b[0][j]+a[i][1]*b[1][j];
      }
    }
    return c;
  }     

  pair[][] conj(pair[][] a) {
    pair[][] c=new pair[2][2];
    for(int i=0; i < 2; ++i) {
      for(int j=0; j < 2; ++j) {
        c[i][j]=conj(a[j][i]);
      }
    }
    return c;
  }     

  A=conj(U)*A*U;

  real D=abs(A[0][0]);
  if(D != 0) {
    A[0][0] /= D;
    A[0][1] /= D;
  }
  
  D=abs(A[1][1]);
  if(D != 0) {
    A[1][0] /= D;
    A[1][1] /= D;
  }

  A=U*A*conj(U);

  return (0,0,A[0][0].x,A[0][1].x,A[1][0].x,A[1][1].x);
}

struct align {
  pair dir;
  triple dir3;
  bool relative=false;
  bool default=true;
  bool is3D=false;
  void init(pair dir=0, bool relative=false, bool default=false) {
    this.dir=dir;
    this.relative=relative;
    this.default=default;
    is3D=false;
  }
  void init(triple dir=(0,0,0), bool relative=false, bool default=false) {
    this.dir3=dir;
    this.relative=relative;
    this.default=default;
    is3D=true;
  }
  align copy() {
    align align=new align;
    align.init(dir,relative,default);
    align.dir3=dir3;
    align.is3D=is3D;
    return align;
  }
  void align(align align) {
    if(!align.default) {
      bool is3D=align.is3D;
      init(align.dir,align.relative);
      dir3=align.dir3;
      this.is3D=is3D;
    }
  }
  void align(align align, align default) {
    align(align);
    if(this.default) {
      init(default.dir,default.relative,default.default);
      dir3=default.dir3;
      is3D=default.is3D;
    }
  }
  void write(file file=stdout, suffix suffix=endl) {
    if(!default) {
      if(relative) {
        write(file,"Relative(");
        if(is3D)
          write(file,dir3);
        else
          write(file,dir);
        write(file,")",suffix);
      } else {
        if(is3D)
          write(file,dir3,suffix);
        else
          write(file,dir,suffix);
      }
    }
  }
  bool Center() {
    return relative && (is3D ? dir3 == (0,0,0) : dir == 0);
  }
}

struct side {
  pair align;
}

side Relative(explicit pair align)
{
  side s;
  s.align=align;
  return s;
}
  
restricted side NoSide;
restricted side LeftSide=Relative(W);
restricted side Center=Relative((0,0));
restricted side RightSide=Relative(E);

side operator * (real x, side s) 
{
  side S;
  S.align=x*s.align;
  return S;
}

align operator cast(pair dir) {align A; A.init(dir,false); return A;}
align operator cast(triple dir) {align A; A.init(dir,false); return A;}
align operator cast(side side) {align A; A.init(side.align,true); return A;}
restricted align NoAlign;

void write(file file=stdout, align align, suffix suffix=endl)
{
  align.write(file,suffix);
}

struct position {
  pair position;
  bool relative;
}

position Relative(real position)
{
  position p;
  p.position=position;
  p.relative=true;
  return p;
}
  
restricted position BeginPoint=Relative(0);
restricted position MidPoint=Relative(0.5);
restricted position EndPoint=Relative(1);

position operator cast(pair x) {position P; P.position=x; return P;}
position operator cast(real x) {return (pair) x;}
position operator cast(int x) {return (pair) x;}

pair operator cast(position P) {return P.position;}

typedef transform embed(transform);
transform Shift(transform t) {return identity();}
transform Rotate(transform t) {return rotation(t);}
transform Slant(transform t) {return scaleless(t);}
transform Scale(transform t) {return t;}

embed Rotate(pair z) {
  return new transform(transform t) {return rotate(degrees(shiftless(t)*z,
                                                           warn=false));};
}

path[] texpath(string s, pen p, bool tex=settings.tex != "none",
               bool bbox=false);

struct Label {
  string s,size;
  position position;
  bool defaultposition=true;
  align align;
  pen p=nullpen;
  transform T;
  transform3 T3=identity(4);
  bool defaulttransform=true;
  bool defaulttransform3=true;
  embed embed=Rotate; // Shift, Rotate, Slant, or Scale with embedded picture
  filltype filltype=NoFill;
  
  void init(string s="", string size="", position position=0, 
            bool defaultposition=true, align align=NoAlign, pen p=nullpen,
            transform T=identity(), transform3 T3=identity4,
            bool defaulttransform=true, bool defaulttransform3=true,
            embed embed=Rotate, filltype filltype=NoFill) {
    this.s=s;
    this.size=size;
    this.position=position;
    this.defaultposition=defaultposition;
    this.align=align.copy();
    this.p=p;
    this.T=T;
    this.T3=copy(T3);
    this.defaulttransform=defaulttransform;
    this.defaulttransform3=defaulttransform3;
    this.embed=embed;
    this.filltype=filltype;
  }
  
  void initalign(string s="", string size="", align align, pen p=nullpen,
                 embed embed=Rotate, filltype filltype=NoFill) {
    init(s,size,align,p,embed,filltype);
  }
  
  void transform(transform T) {
    this.T=T;
    defaulttransform=false;
  }
  
  void transform3(transform3 T) {
    this.T3=copy(T);
    defaulttransform3=false;
  }

  Label copy(transform3 T3=this.T3) {
    Label L=new Label;
    L.init(s,size,position,defaultposition,align,p,T,T3,defaulttransform,
           defaulttransform3,embed,filltype);
    return L;
  }
  
  void position(position pos) {
    this.position=pos;
    defaultposition=false;
  }
  
  void align(align a) {
    align.align(a);
  }
  void align(align a, align default) {
    align.align(a,default);
  }
  
  void p(pen p0) {
    if(this.p == nullpen) this.p=p0;
  }
  
  void filltype(filltype filltype0) {
    if(this.filltype == NoFill) this.filltype=filltype0;
  }
  
  void label(frame f, transform t=identity(), pair position, pair align) {
    pen p0=p == nullpen ? currentpen : p;
    align=length(align)*unit(rotation(t)*align);
    pair S=t*position+align*labelmargin(p0)+shift(T)*0;
    if(settings.tex != "none")
      label(f,s,size,embed(t)*shiftless(T),S,align,p0);
    else
      fill(f,align(texpath(s,p0),S,align,p0),p0);
  }

  void out(frame f, transform t=identity(), pair position=position.position,
           pair align=align.dir) {
    if(filltype == NoFill)
      label(f,t,position,align);
    else {
      frame d;
      label(d,t,position,align);
      add(f,d,filltype);
    }
  }
  
  void label(picture pic=currentpicture, pair position, pair align) {
    if(s == "") return;
    pic.add(new void (frame f, transform t) {
        out(f,t,position,align);
      },true);
    frame f;
    // Create a picture with label at the origin to extract its bbox truesize.
    label(f,(0,0),align);
    pic.addBox(position,position,min(f),max(f));
  }

  void out(picture pic=currentpicture) {
    label(pic,position.position,align.dir);
  }
  
  void out(picture pic=currentpicture, path g) {
    bool relative=position.relative;
    real position=position.position.x;
    pair Align=align.dir;
    bool alignrelative=align.relative;
    if(defaultposition) {relative=true; position=0.5;}
    if(relative) position=reltime(g,position);
    if(align.default) {
      alignrelative=true;
      Align=position <= sqrtEpsilon ? S :
        position >= length(g)-sqrtEpsilon ? N : E;
    }

    pic.add(new void (frame f, transform t) {
        out(f,t,point(g,position),alignrelative ?
            inverse(rotation(t))*-Align*dir(t*g,position)*I : Align);
      },!alignrelative);

    frame f;
    pair align=alignrelative ? -Align*dir(g,position)*I : Align;
    label(f,(0,0),align);
    pair position=point(g,position);
    pic.addBox(position,position,min(f),max(f));
  }
  
  void write(file file=stdout, suffix suffix=endl) {
    write(file,"\""+s+"\"");
    if(!defaultposition) write(file,", position=",position.position);
    if(!align.default) write(file,", align=");
    write(file,align);
    if(p != nullpen) write(file,", pen=",p);
    if(!defaulttransform)
      write(file,", transform=",T);
    if(!defaulttransform3) {
      write(file,", transform3=",endl);
      write(file,T3);
    }
    write(file,"",suffix);
  }
  
  real relative() {
    return defaultposition ? 0.5 : position.position.x;
  };
  
  real relative(path g) {
    return position.relative ? reltime(g,relative()) : relative();
  };
}

Label Label;

void add(frame f, transform t=identity(), Label L)
{
  L.out(f,t);
}
  
void add(picture pic=currentpicture, Label L)
{
  L.out(pic);
}
  
Label operator * (transform t, Label L)
{
  Label tL=L.copy();
  tL.align.dir=L.align.dir;
  tL.transform(t*L.T);
  return tL;
}

Label operator * (transform3 t, Label L)
{
  Label tL=L.copy(t*L.T3);
  tL.align.dir=L.align.dir;
  tL.defaulttransform3=false;
  return tL;
}

Label Label(string s, string size="", explicit position position,
            align align=NoAlign, pen p=nullpen, embed embed=Rotate,
            filltype filltype=NoFill)
{
  Label L;
  L.init(s,size,position,false,align,p,embed,filltype);
  return L;
}

Label Label(string s, string size="", pair position, align align=NoAlign,
            pen p=nullpen, embed embed=Rotate, filltype filltype=NoFill)
{
  return Label(s,size,(position) position,align,p,embed,filltype);
}

Label Label(explicit pair position, align align=NoAlign, pen p=nullpen,
            embed embed=Rotate, filltype filltype=NoFill)
{
  return Label((string) position,position,align,p,embed,filltype);
}

Label Label(string s="", string size="", align align=NoAlign, pen p=nullpen,
            embed embed=Rotate, filltype filltype=NoFill)
{
  Label L;
  L.initalign(s,size,align,p,embed,filltype);
  return L;
}

Label Label(Label L, align align=NoAlign, pen p=nullpen, embed embed=L.embed,
            filltype filltype=NoFill)
{
  Label L=L.copy();
  L.align(align);
  L.p(p);
  L.embed=embed;
  L.filltype(filltype);
  return L;
}

Label Label(Label L, explicit position position, align align=NoAlign,
            pen p=nullpen, embed embed=L.embed, filltype filltype=NoFill)
{
  Label L=Label(L,align,p,embed,filltype);
  L.position(position);
  return L;
}

Label Label(Label L, pair position, align align=NoAlign,
            pen p=nullpen, embed embed=L.embed, filltype filltype=NoFill)
{
  return Label(L,(position) position,align,p,embed,filltype);
}

void write(file file=stdout, Label L, suffix suffix=endl)
{
  L.write(file,suffix);
}

void label(frame f, Label L, pair position, align align=NoAlign,
           pen p=currentpen, filltype filltype=NoFill)
{
  add(f,Label(L,position,align,p,filltype));
}
  
void label(frame f, Label L, align align=NoAlign,
           pen p=currentpen, filltype filltype=NoFill)
{
  add(f,Label(L,L.position,align,p,filltype));
}
  
void label(picture pic=currentpicture, Label L, pair position,
           align align=NoAlign, pen p=currentpen, filltype filltype=NoFill)
{
  Label L=Label(L,position,align,p,filltype);
  add(pic,L);
}
  
void label(picture pic=currentpicture, Label L, align align=NoAlign,
           pen p=currentpen, filltype filltype=NoFill)
{
  label(pic,L,L.position,align,p,filltype);
}

// Label, but with postscript coords instead of asy
void label(pair origin, picture pic=currentpicture, Label L, align align=NoAlign,
           pen p=currentpen, filltype filltype=NoFill)
{
  picture opic;
  label(opic,L,L.position,align,p,filltype);
  add(pic,opic,origin);
}
  
void label(picture pic=currentpicture, Label L, explicit path g,
           align align=NoAlign, pen p=currentpen, filltype filltype=NoFill)
{
  Label L=Label(L,align,p,filltype);
  L.out(pic,g);
}

void label(picture pic=currentpicture, Label L, explicit guide g,
           align align=NoAlign, pen p=currentpen, filltype filltype=NoFill)
{
  label(pic,L,(path) g,align,p,filltype);
}

Label operator cast(string s) {return Label(s);}

// A structure that a string, Label, or frame can be cast to.
struct object {
  frame f;
  Label L=Label;
  path g; // Bounding path

  void operator init(frame f) {
    this.f=f;
    g=box(min(f),max(f));
  }

  void operator init(Label L) {
    this.L=L.copy();
    if(L != Label) L.out(f);
    g=box(min(f),max(f));
  }
}

object operator cast(frame f) {
  return object(f);
}

object operator cast(Label L) 
{
  return object(L);
}

object operator cast(string s) 
{
  return object(s);
}

Label operator cast(object F)
{
  return F.L;
}

frame operator cast(object F)
{
  return F.f;
}

object operator * (transform t, explicit object F)
{
  object f;
  f.f=t*F.f;
  f.L=t*F.L;
  f.g=t*F.g;
  return f;
}

// Returns a copy of object F aligned in the direction align
object align(object F, pair align) 
{
  return shift(F.f,align)*F;
}

void add(picture dest=currentpicture, object F, pair position=0,
         bool group=true, filltype filltype=NoFill, bool above=true)
{
  add(dest,F.f,position,group,filltype,above);
}

// Pack a list of objects into a frame.
frame pack(pair align=2S ... object inset[])
{
  frame F;
  int n=inset.length;
  pair z;
  for (int i=0; i < n; ++i) {
    add(F,inset[i].f,z);
    z += align+realmult(unit(align),size(inset[i].f));
  }
  return F;
}

path[] texpath(Label L, bool tex=settings.tex != "none", bool bbox=false)
{
  struct stringfont
  {
    string s;
    real fontsize;
    string font;

    void operator init(Label L) 
    {
      s=replace(L.s,'\n',' ');
      fontsize=fontsize(L.p);
      font=font(L.p);
    }

    pen pen() {return fontsize(fontsize)+fontcommand(font);}
  }
  
  bool lexorder(stringfont a, stringfont b) {
    return a.s < b.s || (a.s == b.s && (a.fontsize < b.fontsize ||
                                        (a.fontsize == b.fontsize &&
                                         a.font < b.font)));
  }

  static stringfont[] stringcache;
  static path[][] pathcache;

  static stringfont[] stringlist;
  static bool adjust[];
  
  path[] G;

  stringfont s=stringfont(L);
  pen p=s.pen();

  int i=search(stringcache,s,lexorder);
  if(i == -1 || lexorder(stringcache[i],s)) {
    int k=search(stringlist,s,lexorder);
    if(k == -1 || lexorder(stringlist[k],s)) {
      ++k;
      stringlist.insert(k,s);
      // PDF tex engines lose track of the baseline.
      adjust.insert(k,tex && basealign(L.p) == 1 && pdf());
    }
  }

  path[] transform(path[] g, Label L) {
    if(g.length == 0) return g;
    pair m=min(g);
    pair M=max(g);
    pair dir=rectify(inverse(L.T)*-L.align.dir);
    if(tex && basealign(L.p) == 1)
      dir -= (0,(1-dir.y)*m.y/(M.y-m.y));
    pair a=m+realmult(dir,M-m);

    return shift(L.position+L.align.dir*labelmargin(L.p))*L.T*shift(-a)*g;
  }

  if(tex && bbox) {
    frame f;
    label(f,L);
    return transform(box(min(f),max(f)),L);
  }
  
  if(stringlist.length > 0) {
    path[][] g;
    int n=stringlist.length;
    string[] s=new string[n];
    pen[] p=new pen[n];
    for(int i=0; i < n; ++i) {
      stringfont S=stringlist[i];
      s[i]=adjust[i] ? "."+S.s : S.s;
      p[i]=adjust[i] ? S.pen()+basealign : S.pen();
    }
        
    g=tex ? _texpath(s,p) : textpath(s,p);
      
    if(tex)
      for(int i=0; i < n; ++i)
        if(adjust[i]) {
          real y=min(g[i][0]).y;
          g[i].delete(0);
          g[i]=shift(0,-y)*g[i];
        }
    
  
    for(int i=0; i < stringlist.length; ++i) {
      stringfont s=stringlist[i];
      int j=search(stringcache,s,lexorder)+1;
      stringcache.insert(j,s);
      pathcache.insert(j,g[i]);
    }
    stringlist.delete();
    adjust.delete();
  }

  return transform(pathcache[search(stringcache,stringfont(L),lexorder)],L);
}

texpath=new path[](string s, pen p, bool tex=settings.tex != "none", bool bbox=false)
{
  return texpath(Label(s,p));
};
