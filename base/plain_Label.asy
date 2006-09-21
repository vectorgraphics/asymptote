// A rotation in the direction dir limited to [-90,90]
// This is useful for rotating text along a line in the direction dir.
transform rotate(explicit pair dir)
{
  real angle=degrees(dir);
  if(angle > 90 && angle < 270) angle -= 180;
  return rotate(angle);
} 

transform scale(transform t)
{
  transform t0=shiftless(t);
  return rotate(-degrees(t0*(1,0)))*t0;
}

struct align {
  pair dir;
  bool relative=false;
  bool default=true;
  void init(pair dir=0, bool relative=false, bool default=false) {
    this.dir=dir;
    this.relative=relative;
    this.default=default;
  }
  align copy() {
    align align=new align;
    align.init(dir,relative,default);
    return align;
  }
  void align(align align) {
    if(!align.default) init(align.dir,align.relative);
  }
  void align(align align, align default) {
    align(align);
    if(this.default) init(default.dir,default.relative,default.default);
  }
  void write(file file=stdout, suffix suffix=endl) {
    if(!default) {
      if(relative) {
        write(file,"Relative(");
        write(file,dir);
        write(file,")",suffix);
      } else write(file,dir,suffix);
    }
  }
  bool Center() {
    return relative && dir == 0;
  }
}

struct side {
  pair align;
}

side operator init() {return new side;}
  
side Relative(explicit pair align)
{
  side s;
  s.align=align;
  return s;
}
  
side NoSide;
side LeftSide=Relative(W);
side Center=Relative((0,0));
side RightSide=Relative(E);

side operator * (real x, side s) 
{
  side S;
  S.align=x*s.align;
  return S;
}

align operator init() {return new align;}
align operator cast(pair dir) {align A; A.init(dir,false); return A;}
align operator cast(side side) {align A; A.init(side.align,true); return A;}
align NoAlign;

void write(file file=stdout, align align, suffix suffix=endl)
{
  align.write(file,suffix);
}

struct position {
  pair position;
  bool relative;
}

position operator init() {return new position;}
  
position Relative(real position)
{
  position p;
  p.position=position;
  p.relative=true;
  return p;
}
  
position BeginPoint=Relative(0);
position MidPoint=Relative(0.5);
position EndPoint=Relative(1);

position operator cast(pair x) {position P; P.position=x; return P;}
position operator cast(real x) {return (pair) x;}
position operator cast(int x) {return (pair) x;}

pair operator cast(position P) {return P.position;}

struct Label {
  string s,size;
  position position;
  bool defaultposition=true;
  align align;
  pen p=nullpen;
  real angle;
  bool defaultangle=true;
  pair shift;
  pair scale=(1,1);
  filltype filltype=NoFill;
  
  void init(string s="", string size="", position position=0, 
            bool defaultposition=true,
            align align=NoAlign, pen p=nullpen, real angle=0,
            bool defaultangle=true, pair shift=0, pair scale=(1,1),
            filltype filltype=NoFill) {
    this.s=s;
    this.size=size;
    this.position=position;
    this.defaultposition=defaultposition;
    this.align=align.copy();
    this.p=p;
    this.angle=angle;
    this.defaultangle=defaultangle;
    this.shift=shift;
    this.scale=scale;
    this.filltype=filltype;
  }
  
  void initalign(string s="", string size="", align align, pen p=nullpen,
                 filltype filltype=NoFill) {
    init();
    this.s=s;
    this.size=size;
    this.align=align.copy();
    this.p=p;
    this.filltype=filltype;
  }
  
  Label copy() {
    Label L=new Label;
    L.init(s,size,position,defaultposition,align,p,angle,defaultangle,shift,
           scale,filltype);
    return L;
  }
  
  void angle(real a) {
    this.angle=a;
    defaultangle=false;
  }
  
  void shift(pair a) {
    this.shift=a;
  }
  
  void scale(pair a) {
    this.scale=a;
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
    if(t == identity()) {
      label(f,s,size,angle,position+align*labelmargin(p0)+shift,align,scale,
            p0);
    } else {
      transform t0=shiftless(t);
      real angle=degrees(t0*dir(angle));
      pair position=t*position+align*labelmargin(p0)+shift;
      label(f,s,size,angle,position,length(align)*unit(t0*align),scale,p0);
    }
  }

  void out(frame f, transform t=identity()) {
    if(filltype == NoFill) label(f,t,position.position,align.dir);
    else {
      frame d;
      label(d,t,position.position,align.dir);
      add(f,d,filltype);
    }
  }
  
  void label(picture pic=currentpicture, pair position, pair align) {
    pic.add(new void (frame f, transform t) {
        if(filltype == NoFill)
          label(f,t,position,align);
        else {
          frame d;
          label(d,t,position,align);
          add(f,d,filltype);
        }
      });
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
      Align=position <= 0 ? S : position >= length(g) ? N : E;
    }
    label(pic,point(g,position),
          alignrelative ? Align*dir(g,position)/N : Align);
  }
  
  void write(file file=stdout, suffix suffix=endl) {
    write(file,"s=\""+s+"\"");
    if(!defaultposition) write(file,", position=",position.position);
    if(!align.default) write(file,", align=");
    write(file,align);
    if(p != nullpen) write(file,", pen=",p);
    if(!defaultangle) write(file,", angle=",angle);
    if(shift != 0) write(file,", shift=",shift);
    write(file,"",suffix);
  }
  
  real relative() {
    return defaultposition ? 0.5 : position.position.x;
  };
  
  real relative(path g) {
    return position.relative ? reltime(g,relative()) : relative();
  };
}

Label operator init() {return new Label;}

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
  transform t0=shiftless(t);
  tL.align.dir=length(L.align.dir)*unit(t0*L.align.dir);
  tL.angle(degrees(t0*dir(L.angle)));
  tL.shift(shift(t)*L.shift);
  tL.scale(scale(t)*L.scale);
  return tL;
}

Label Label(string s, string size="", explicit position position,
            align align=NoAlign, pen p=nullpen, filltype filltype=NoFill)
{
  Label L;
  L.init(s,size,position,false,align,p,filltype);
  return L;
}

Label Label(string s, string size="", pair position, align align=NoAlign,
            pen p=nullpen, filltype filltype=NoFill)
{
  return Label(s,size,(position) position,align,p,filltype);
}

Label Label(explicit pair position, align align=NoAlign, pen p=nullpen,
            filltype filltype=NoFill)
{
  return Label((string) position,position,align,p,filltype);
}

Label Label(string s="", string size="", align align=NoAlign, pen p=nullpen,
            filltype filltype=NoFill)
{
  Label L;
  L.initalign(s,size,align,p,filltype);
  return L;
}

Label Label(Label L, explicit position position, align align=NoAlign,
	    pen p=nullpen, filltype filltype=NoFill)
{
  Label L=L.copy();
  L.position(position);
  L.align(align);
  L.p(p);
  L.filltype(filltype);
  return L;
}

Label Label(Label L, pair position, align align=NoAlign,
	    pen p=nullpen, filltype filltype=NoFill)
{
  return Label(L,(position) position,align,p,filltype);
}

Label Label(Label L, align align=NoAlign, pen p=nullpen,
            filltype filltype=NoFill)
{
  Label L=L.copy();
  L.align(align);
  L.p(p);
  L.filltype(filltype);
  return L;
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
           align align=NoAlign, pen p=nullpen, filltype filltype=NoFill)
{
  Label L=L.copy();
  L.position(position);
  L.align(align);
  L.p(p);
  L.filltype(filltype);
  add(pic,L);
}
  
void label(picture pic=currentpicture, Label L, align align=NoAlign,
           pen p=nullpen, filltype filltype=NoFill)
{
  label(pic,L,L.position,align,p,filltype);
}
  
void label(picture pic=currentpicture, Label L, explicit path g,
           align align=NoAlign, pen p=nullpen, filltype filltype=NoFill)
{
  Label L=L.copy();
  L.align(align);
  L.p(p);
  L.filltype(filltype);
  L.out(pic,g);
}

void label(picture pic=currentpicture, Label L, explicit guide g,
           align align=NoAlign, pen p=nullpen, filltype filltype=NoFill)
{
  label(pic,L,(path) g,align,p,filltype);
}

Label operator cast(string s) {return Label(s);}

// A structure that a string, Label, or frame can be cast to.
struct object {
  frame f;
  Label L=Label;
  frame fit() {
    if(L != Label) L.out(f);
    return f;
  }
}

object operator init() {return new object;}
object operator cast(frame f) {
  object o;
  o.f=f;
  return o;
}

object operator cast(Label L) 
{
  object o;
  o.L=L.copy();
  return o;
}

object operator cast(string s) 
{
  object o;
  o.L=s;
  return o;
}

// Pack a list of objects into a frame.
frame pack(pair align=2S ... object inset[])
{
  frame F;
  int n=inset.length;
  pair z;
  for (int i=0; i < n; ++i) {
    frame f=inset[i].fit();
    add(F,f,z);
    z += align+realmult(unit(align),max(f)-min(f));
  }
  return F;
}

