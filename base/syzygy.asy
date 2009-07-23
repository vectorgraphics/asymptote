/***** syzygy.asy {{{1
 * Andy Hammerlindl  2006/12/02
 *
 * Automates the drawing of braids, relations, and syzygies, along with the
 * corresponding equations.
 *
 * See
 *   http://katlas.math.toronto.edu/drorbn/index.php?title=06-1350/Syzygies_in_Asymptote
 * For more information.
 *****/
struct Component { // {{{1
  // The number of strings coming in or out of the component.
  int in;
  int out;

  // Which 'out' string each 'in' string is connected to.  For deriving
  // equations.
  int[] connections;

  string symbol;    // For pullback notation.
  string lsym;      // For linear equations.
  string codename;  // For Mathematica code.

  guide[] draw(picture pic, guide[] ins);
}

// Utility functions {{{1
pair[] endpoints(guide[] a) {
  pair[] z;
  for (int i=0; i<a.length; ++i)
    z.push(endpoint(a[i]));
  return z;
}

pair min(pair[] z) {
  pair m=(infinity, infinity);
  for (int i=0; i<z.length; ++i) {
    if (z[i].x < m.x)
      m=(z[i].x,m.y);
    if (z[i].y < m.y)
      m=(m.x,z[i].y);
  }
  return m;
}

pair max(pair[] z) {
  pair M=(-infinity, -infinity);
  for (int i=0; i<z.length; ++i) {
    if (z[i].x > M.x)
      M=(z[i].x,M.y);
    if (z[i].y > M.y)
      M=(M.x,z[i].y);
  }
  return M;
}

// Component Definitions {{{1
real hwratio=1.4;
real gapfactor=6;

Component bp=new Component;
bp.in=2; bp.out=2;
bp.connections=new int[] {1,0};
bp.symbol="B^+"; bp.lsym="b^+"; bp.codename="bp";
bp.draw=new guide[] (picture pic, guide[] ins) {
  pair[] z=endpoints(ins);
  pair m=min(z), M=max(z);
  real w=M.x-m.x, h=hwratio*w;
  pair centre=(0.5(m.x+M.x),M.y+h/2);

  /*
    return new guide[] {ins[1]..centre{NW}..z[0]+h*N,
    ins[0]..centre{NE}..z[1]+h*N};
  */

  real offset=gapfactor*linewidth(currentpen);
  draw(pic, ins[1]..(centre-offset*NW){NW});
  return new guide[] {(centre+offset*NW){NW}..z[0]+h*N,
                                                ins[0]..centre{NE}..z[1]+h*N};
};
    
Component bm=new Component;
bm.in=2; bm.out=2;
bm.connections=new int[] {1,0};
bm.symbol="B^-"; bm.lsym="b^-"; bm.codename="bm";
bm.draw=new guide[] (picture pic, guide[] ins) {
  pair[] z=endpoints(ins);
  pair m=min(z), M=max(z);
  real w=M.x-m.x, h=hwratio*w;
  pair centre=(0.5(m.x+M.x),M.y+h/2);

  /*
    return new guide[] {ins[1]..centre{NW}..z[0]+h*N,
    ins[0]..centre{NE}..z[1]+h*N};
  */

  real offset=gapfactor*linewidth(currentpen);
  draw(pic, ins[0]..(centre-offset*NE){NE});
  return new guide[] {ins[1]..centre{NW}..z[0]+h*N,
                                            (centre+offset*NE){NE}..z[1]+h*N};
};
    
Component phi=new Component;
phi.in=2; phi.out=1;
phi.connections=new int[] {0,0};
phi.symbol="\Phi"; phi.lsym="\phi"; phi.codename="phi";
phi.draw=new guide[] (picture pic, guide[] ins) {
  pair[] z=endpoints(ins);
  pair m=min(z), M=max(z);
  real w=M.x-m.x, h=hwratio*w;
  pair centre=(0.5(m.x+M.x),M.y+h/2);


  //real offset=4*linewidth(currentpen);
  draw(pic, ins[0]..centre{NE});
  draw(pic, ins[1]..centre{NW});
  draw(pic, centre,linewidth(5*linewidth(currentpen)));
  dot(pic, centre);
  return new guide[] {centre..centre+0.5h*N};
};

Component wye=new Component;
wye.in=1; wye.out=2;
wye.connections=null; // TODO: Fix this!
wye.symbol="Y"; wye.lsym="y"; wye.codename="wye";
wye.draw=new guide[] (picture pic, guide[] ins) {
  pair z=endpoint(ins[0]);
  real w=10, h=hwratio*w; // The 10 is a guess here, and may produce badness.
  pair centre=(z.x,z.y+h/2);


  draw(pic, ins[0]..centre);
  draw(pic, centre,linewidth(5*linewidth(currentpen)));
  return new guide[] {centre{NW}..centre+(-0.5w,0.5h),
                                    centre{NE}..centre+(0.5w,0.5h)};
};


struct Braid { // {{{1
  // Members {{{2
  // Number of lines initially.
  int n;

  struct Placement {
    Component c;
    int place;

    Placement copy() {
      Placement p=new Placement;
      p.c=this.c; p.place=this.place;
      return p;
    }
  }
  Placement[] places;

  void add(Component c, int place) {
    Placement p=new Placement;
    p.c=c; p.place=place;
    places.push(p);
  }

  void add(Braid sub, int place) {
    for (int i=0; i<sub.places.length; ++i)
      add(sub.places[i].c,sub.places[i].place+place);
  }

  // Drawing {{{2
  guide[] drawStep(picture pic, Placement p, guide[] ins) {
    int i=0,j=0;

    // Draw the component.
    Component c=p.c;
    //write("drawing "+c.symbol+" at place "+(string)p.place);
    guide[] couts=c.draw(pic, ins[sequence(c.in)+p.place]);

    pair M=max(endpoints(couts));

    // Extend lines not in the component.
    guide[] outs;
    pair[] z=endpoints(ins);
    while (i<p.place) {
      outs.push(ins[i]..(z[i].x,M.y));
      ++i;
    }

    outs.append(couts);
    i+=c.in;

    while (i<ins.length) {
      outs.push(ins[i]..(z[i].x,M.y));
      ++i;
    }

    return outs;
  }

  void drawEnd(picture pic, guide[] ins, real minheight=0) {
    pair[] z=endpoints(ins);
    for (int i=0; i<ins.length; ++i) {
      draw(pic, z[i].y >= minheight ? ins[i] : ins[i]..(z[i].x,minheight));
    }
  }

  void draw(picture pic, guide[] ins, real minheight=0) {
    int steps=places.length;

    guide[] nodes=ins;
    for (int i=0; i<steps; ++i) {
      Placement p=places[i];
      nodes=drawStep(pic, places[i], nodes);
    }

    drawEnd(pic, nodes, minheight);
  }

  void draw(picture pic=currentpicture, real spacing=15,
            real minheight=2hwratio*spacing) {
    pair[] ins;
    for (int i=0; i<n; ++i)
      ins.push((spacing*i,0));

    draw(pic, ins, minheight);
  }

  // Utilities {{{2
  int in() {
    return n;
  }
  int out() {
    int steps=places.length;
    int num=n; // The number of nodes at this step.

    for (int i=0; i<steps; ++i) {
      Placement p=places[i];
      int nextNum=num-p.c.in+p.c.out;
      num=nextNum;
    }
    return num;
  }

  // Deep copy of a braid.
  Braid copy() {
    Braid b=new Braid;
    b.n=this.n;
    for (int i=0; i<this.places.length; ++i)
      b.add(this.places[i].c,this.places[i].place);
    return b;
  }

  // Matching {{{2
  // Tests if a component p can be swapped with a component q which is assumed
  // to be directly above it.
  static bool swapable(Placement p, Placement q) {
    return  p.place + p.c.out <= q.place || // p is left of q or
      q.place + q.c.in <= p.place;    // q is left of p
  }

  // Creates a new braid with a transposition of two components.
  Braid swap(int i, int j) {
    if (i>j)
      return swap(j,i);
    else {
      assert(j==i+1); assert(swapable(places[i],places[j]));

      Placement p=places[i].copy();
      Placement q=places[j].copy();
      /*write("swap:");
        write("p originally at " + (string)p.place);
        write("q originally at " + (string)q.place);
        write("p.c.in: " + (string)p.c.in + " p.c.out: " + (string)p.c.out);
        write("q.c.in: " + (string)q.c.in + " q.c.out: " + (string)q.c.out);*/
      if (q.place + q.c.in <= p.place)
        // q is left of p - adjust for q renumbering strings.
        p.place+=q.c.out-q.c.in;
      else if (p.place + p.c.out <= q.place)
        // q is right of p - adjust for p renumbering strings.
        q.place+=p.c.in-p.c.out;
      else
        // q is directly on top of p
        assert(false, "swapable");

      /*write("q now at " + (string)q.place);
        write("p now at " + (string)p.place);*/

      Braid b=this.copy();
      b.places[i]=q;
      b.places[j]=p;
      return b;
    }
  }

  // Tests if the component at index 'start' can be moved to index 'end'
  // without interfering with other components.
  bool moveable(int start, int end) {
    assert(start<places.length); assert(end<places.length);
    if (start==end)
      return true;
    else if (end<start)
      return moveable(end,start);
    else {
      assert(start<end);
      Placement p=places[start].copy();
      for (int step=start; step<end; ++step) {
        Placement q=places[step+1];
        if (q.place + q.c.in <= p.place)
          // q is left of p - adjust for q renumbering strings.
          p.place+=q.c.out-q.c.in;
        else if (p.place + p.c.out <= q.place)
          // q is right of p - nothing to do.
          continue;
        else
          // q is directly on top of p
          return false;
      }
      return true;
    }
  }

  bool matchComponent(Braid sub, int subindex, int place, int step) {
    int i=subindex;
    return sub.places[i].c == this.places[step].c &&
      sub.places[i].place + place == this.places[step].place;
  }

  // Returns true if a sub-braid occurs within the one at the specified
  // coordinates with no component occuring anywhere inbetween.
  bool exactMatch(Braid sub, int place, int step) {
    for (int i=0; i<sub.places.length; ++i) {
      if (!matchComponent(sub, i, place, i+step)) {
        write("match failed at iteration: ", i);
        return false;
      }
    }
    return true;
  }

  /*
    bool findSubsequence(Braid sub, int place, int size, int[] acc) {
    // If we've matched all the components, we've won.
    if (acc.length >= sub.places.length)
    return true;

    // The next component to match.
    Placement p=sub.places[acc.length];

    // Start looking immediately after the last match.
    for (int step=acc[acc.length-1]+1; step<this.places.length; ++step) {
    Placement q=this.places[step];
  */

  bool tryMatch(Braid sub, int place, int size, int[] acc) {
    // If we've matched all the components, we've won.
    if (acc.length >= sub.places.length)
      return true;

    // The next component to match.
    Placement p=sub.places[acc.length];

    // Start looking immediately after the last match.
    for (int step=acc[acc.length-1]+1; step<this.places.length; ++step) {
      Placement q=this.places[step];
      // Check if the next component is in the set of strings used by the
      // subbraid.
      if (q.place + q.c.in > place && q.place < place + size) {
        // It's in the window, so it must match the next component in the
        // subbraid.
        if (p.c==q.c && p.place+place==q.place) {
          // A match - go on to the next component.
          acc.push(step);
          return tryMatch(sub, place, size, acc); // TODO: Adjust place/size.
        }
        else
          return false;
      }

      // TODO: Adjust place and size.
    }

    // We've run out of components to match.
    return false;
  }


  // This attempts to find a subbraid within the braid.  It allows other
  // components to be interspersed with the components of the subbraid so long
  // as they don't occur on the same string as the ones the subbraid lies on.
  // Returns null on failure.
  int[] match(Braid sub, int place) {
    for (int i=0; i<=this.places.length-sub.places.length; ++i) {
      // Find where the first component of the subbraid matches and try to
      // match the rest of the braid starting from there.
      if (matchComponent(sub, 0, place, i)) {
        int[] result;
        result.push(i);
        if (tryMatch(sub,place,sub.n,result))
          return result;
      }
    }
    return null;
  }

  // Equations {{{2
  // Returns the string that 'place' moves to when going through the section
  // with Placement p.
  static int advancePast(Placement p, int place) {
    // If it's to the left of the component, it is unaffected.
    return place<p.place ? place :
      // If it's to the right of the component, adjust the numbering due
      // to the change of the number of strings in the component.
      p.place+p.c.in <= place ? place - p.c.in + p.c.out :
      // If it's in the component, ask the component to do the work.
      p.place + p.c.connections[place-p.place];
  }

  // Adjust the place (at step 0) to the step given, to find which string it is
  // on in that part of the diagram.
  int advanceToStep(int step, int place) {
    assert(place>=0 && place<n);
    assert(step>=0 && step<places.length);

    for (int i=0; i<step; ++i)
      place=advancePast(places[i], place);

    return place;
  }

  int pullbackWindowPlace(int step, int place,
                          int w_place, int w_size) {
    place=advanceToStep(step,place);
    return place < w_place           ? 1 : // The shielding.
      w_place + w_size <= place ? 0 : // The string doesn't touch it.
      place-w_place+2;
  }

  int pullbackPlace(int step, int place) {
    // Move to the right step.
    //write("advance: ", step, place, advanceToStep(step,place));
    //place=advanceToStep(step,place);
    Placement p=places[step];
    return pullbackWindowPlace(step,place, p.place, p.c.in);
    /*return place < p.place           ? 1 : // The shielding.
      p.place + p.c.in <= place ? 0 : // The string doesn't touch it.
      place-p.place+2;*/
  }
                
  int[] pullbackWindow(int step, int w_place, int w_size) {
    int[] a={1};
    for (int place=0; place<n; ++place)
      a.push(pullbackWindowPlace(step, place, w_place, w_size));
    return a;
  }

  int[] pullback(int step) {
    Placement p=places[step];
    return pullbackWindow(step, p.place, p.c.in);
    /*int[] a={1};
      for (int place=0; place<n; ++place)
      a.push(pullbackPlace(step, place));
      return a;*/
  }

  string stepToFormula(int step) {
    // Determine the pullbacks.
    string s="(1";
    for (int place=0; place<n; ++place)
      //write("pullback: ", step, place, pullbackString(step,place));
      s+=(string)pullbackPlace(step, place);
    s+=")^\star "+places[step].c.symbol;
    return s;
  }

  // Write it as a formula with pullback notation.
  string toFormula() {
    if (places.length==0)
      return "1";
    else {
      string s;
      for (int step=0; step<places.length; ++step) {
        if (step>0)
          s+=" ";
        s+=stepToFormula(step);
      }
      return s;
    }
  }

  string windowToLinear(int step, int w_place, int w_size) {
    int[] a=pullbackWindow(step, w_place, w_size);
    string s="(";
    for (int arg=1; arg<=w_size+1; ++arg) {
      if (arg>1)
        s+=",";
      bool first=true;
      for (int var=0; var<a.length; ++var) {
        if (a[var]==arg) {
          if (first)
            first=false;
          else
            s+="+";
          s+="x_"+(string)(var+1);
        }
      }
    }
    return s+")";
  }

  string windowToCode(int step, int w_place, int w_size) {
    int[] a=pullbackWindow(step, w_place, w_size);
    string s="[";
    for (int arg=1; arg<=w_size+1; ++arg) {
      if (arg>1)
        s+=", ";
      bool first=true;
      for (int var=0; var<a.length; ++var) {
        if (a[var]==arg) {
          if (first)
            first=false;
          else
            s+=" + ";
          s+="x"+(string)(var+1);
        }
      }
    }
    return s+"]";
  }

  string stepToLinear(int step) {
    //int[] a=pullback(step);
    Placement p=places[step];
    return p.c.lsym+windowToLinear(step, p.place, p.c.in);

    /*string s=p.c.lsym+"(";
      for (int arg=1; arg<=p.c.in+1; ++arg) {
      if (arg>1)
      s+=",";
      bool first=true;
      for (int var=0; var<a.length; ++var) {
      if (a[var]==arg) {
      if (first)
      first=false;
      else
      s+="+";
      s+="x_"+(string)(var+1);
      }
      }
      }
      return s+")";*/
  }

  string stepToCode(int step) {
    Placement p=places[step];
    return p.c.codename+windowToCode(step, p.place, p.c.in);
  }

  string toLinear(bool subtract=false) {
    if (places.length==0)
      return subtract ? "0" : "";  // or "1" ?
    else {
      string s = subtract ? " - " : "";
      for (int step=0; step<places.length; ++step) {
        if (step>0)
          s+= subtract ? " - " : " + ";
        s+=stepToLinear(step);
      }
      return s;
    }
  }

  string toCode(bool subtract=false) {
    if (places.length==0)
      return subtract ? "0" : "";  // or "1" ?
    else {
      string s = subtract ? " - " : "";
      for (int step=0; step<places.length; ++step) {
        if (step>0)
          s+= subtract ? " - " : " + ";
        s+=stepToCode(step);
      }
      return s;
    }
  }
}

struct Relation { // {{{1
  Braid lhs, rhs;

  string lsym, codename;
  bool inverted=false;

  string toFormula() {
    return lhs.toFormula() + " = " + rhs.toFormula();
  }

  string linearName() {
    assert(lhs.n==rhs.n);
    assert(lsym!="");

    string s=(inverted ? "-" : "") + lsym+"(";
    for (int i=1; i<=lhs.n+1; ++i) {
      if (i>1)
        s+=",";
      s+="x_"+(string)i;
    }
    return s+")";
  }

  string fullCodeName() {
    assert(lhs.n==rhs.n);
    assert(codename!="");

    string s=(inverted ? "minus" : "") + codename+"[";
    for (int i=1; i<=lhs.n+1; ++i) {
      if (i>1)
        s+=", ";
      s+="x"+(string)i+"_";
    }
    return s+"]";
  }

  string toLinear() {
    return linearName() + " = " + lhs.toLinear() + rhs.toLinear(true);
  }

  string toCode() {
    return fullCodeName() + " :> " + lhs.toCode() + rhs.toCode(true);
  }

  void draw(picture pic=currentpicture) {
    picture left; lhs.draw(left);
    frame l=left.fit();
    picture right; rhs.draw(right);
    frame r=right.fit();

    real xpad=30;

    add(pic, l);
    label(pic, "=", (max(l).x + 0.5xpad, 0.25(max(l).y+max(r).y)));
    add(pic, r, (max(l).x+xpad,0));
  }
}

Relation operator- (Relation r) {
  Relation opposite;
  opposite.lhs=r.rhs;
  opposite.rhs=r.lhs;
  opposite.lsym=r.lsym;
  opposite.codename=r.codename;
  opposite.inverted=!r.inverted;
  return opposite;
}


Braid apply(Relation r, Braid b, int step, int place) {
  bool valid=b.exactMatch(r.lhs,place,step);
  if (valid) {
    Braid result=new Braid;
    result.n=b.n;
    for (int i=0; i<step; ++i)
      result.places.push(b.places[i]);
    result.add(r.rhs,place);
    for (int i=step+r.lhs.places.length; i<b.places.length; ++i)
      result.places.push(b.places[i]);
    return result;
  }
  else {
    write("Invalid match!");
    return null;
  }
}

// Tableau {{{1

// Draw a number of frames in a nice circular arrangement.
picture tableau(frame[] cards, bool number=false) {
  int n=cards.length;

  // Calculate the max height and width of the frames (assuming min(f)=(0,0)).
  pair M=(0,0);
  for (int i=0; i<n; ++i) {
    pair z=max(cards[i]);
    if (z.x > M.x)
      M=(z.x,M.y);
    if (z.y > M.y)
      M=(M.x,z.y);
  }

  picture pic;
  real xpad=2.0, ypad=1.3;
  void place(int index, real row, real column) {
    pair z=((M.x*xpad)*column,(M.y*ypad)*row);
    add(pic, cards[index], z);
    if (number) {
      label(pic,(string)index, z+(0.5M.x,0), S);
    }
  }

  // Handle small collections.
  if (n<=4) {
    for (int i=0; i<n; ++i)
      place(i,0,i);
  }
  else {
    int rows=quotient(n-1,2), columns=3;

    // Add the top middle card.
    place(0,rows-1,1);

    // place cards down the right side.
    for (int i=1; i<rows; ++i)
      place(i, rows-i,2);

    // place cards at the bottom.
    if (n%2==0) {
      place(rows,0,2);
      place(rows+1,0,1);
      place(rows+2,0,0);
    }
    else {
      place(rows,0,1.5);
      place(rows+1,0,0.5);
    }

    // place cards up the left side.
    for (int i=1; i<rows; ++i)
      place(i+n-rows,i,0);
  }

  return pic;
}

struct Syzygy { // {{{1
  // Setup {{{2
  Braid initial=null;
  bool cyclic=true;
  bool showall=false;
  bool number=false;  // Number the diagrams when drawn.

  string lsym, codename; 

  bool watched=false;
  bool uptodate=true;

  struct Move {
    Braid action(Braid);
    Relation rel;
    int place, step;
  }

  Move[] moves;

  void apply(Relation r, int step, int place) {
    Move m=new Move;
    m.rel=r;
    m.place=place; m.step=step;
    m.action=new Braid (Braid b) {
      return apply(r, b, step, place);
    };
    moves.push(m);

    uptodate = false;
  }

  void swap(int i, int j) {
    Move m=new Move;
    m.rel=null;
    m.action=new Braid (Braid b) {
      return b.swap(i, j);
    };
    moves.push(m);

    uptodate = false;
  }

  // Drawing {{{2
  picture[] drawMoves() {
    picture[] pics;

    assert(initial!=null, "must set initial braid");
    Braid b=initial;

    picture pic;
    b.draw(pic);
    pics.push(pic);

    for (int i=0; i<moves.length; ++i) {
      b=moves[i].action(b);
      if (showall || moves[i].rel != null) {
        picture pic;
        b.draw(pic);
        pics.push(pic);
      }
    }

    // Remove the last picture.
    if (this.cyclic)
      pics.pop();

    return pics;
  }

  void draw(picture pic=currentpicture) {
    pic.add(tableau(fit(drawMoves()), this.number));
  }

  void updatefunction() {
    if (!uptodate) {
      picture pic; this.draw(pic);
      shipout(pic);
      uptodate = true;
    }
  }

  void oldupdatefunction() = null;

  void watch() {
    if (!watched) {
      watched = true;
      oldupdatefunction = atupdate();
      atupdate(this.updatefunction);
      uptodate = false;
    }
  }

  void unwatch() {
    assert(watched == true);
    atupdate(oldupdatefunction);
    uptodate = false;
  }

  // Writing {{{2
  string linearName() {
    assert(lsym!="");

    string s=lsym+"(";
    for (int i=1; i<=initial.n+1; ++i) {
      if (i>1)
        s+=",";
      s+="x_"+(string)i;
    }
    return s+")";
  }

  string fullCodeName() {
    assert(codename!="");

    string s=codename+"[";
    for (int i=1; i<=initial.n+1; ++i) {
      if (i>1)
        s+=", ";
      s+="x"+(string)i+"_";
    }
    return s+"]";
  }

  string toLinear() {
    string s=linearName()+" = ";

    Braid b=initial;
    bool first=true;
    for (int i=0; i<moves.length; ++i) {
      Move m=moves[i];
      if (m.rel != null) {
        if (first) {
          first=false;
          if (m.rel.inverted)
            s+=" - ";
        }
        else
          s+=m.rel.inverted ? " - " : " + ";
        s+=m.rel.lsym+b.windowToLinear(m.step, m.place, m.rel.lhs.n);
      }
      b=m.action(b);
    }

    return s;
  }

  string toCode() {
    string s=fullCodeName()+" :> ";

    Braid b=initial;
    bool first=true;
    for (int i=0; i<moves.length; ++i) {
      Move m=moves[i];
      if (m.rel != null) {
        if (first) {
          first=false;
          if (m.rel.inverted)
            s+=" - ";
        }
        else
          s+=m.rel.inverted ? " - " : " + ";
        s+=m.rel.codename+b.windowToCode(m.step, m.place, m.rel.lhs.n);
      }
      b=m.action(b);
    }

    return s;
  }

}

// Relation definitions {{{1
// If you define more relations that you think would be useful, please email
// them to me, and I'll add them to the script.  --Andy.
Relation r3;
r3.lhs.n=3;
r3.lsym="\rho_3"; r3.codename="rho3";
r3.lhs.add(bp,0); r3.lhs.add(bp,1); r3.lhs.add(bp,0);
r3.rhs.n=3;
r3.rhs.add(bp,1); r3.rhs.add(bp,0); r3.rhs.add(bp,1);

Relation r4a;
r4a.lhs.n=3;
r4a.lsym="\rho_{4a}"; r4a.codename="rho4a";
r4a.lhs.add(bp,0); r4a.lhs.add(bp,1); r4a.lhs.add(phi,0);
r4a.rhs.n=3;
r4a.rhs.add(phi,1); r4a.rhs.add(bp,0);

Relation r4b;
r4b.lhs.n=3;
r4b.lsym="\rho_{4b}"; r4b.codename="rho4b";
r4b.lhs.add(bp,1); r4b.lhs.add(bp,0); r4b.lhs.add(phi,1);
r4b.rhs.n=3;
r4b.rhs.add(phi,0); r4b.rhs.add(bp,0);

