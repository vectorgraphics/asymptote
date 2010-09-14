private import plain_scaling;

struct bounds {
  coords2 point,min,max;
  bool exact=true; // An accurate picture bounds is provided by the user.
  void erase() {
    point.erase();
    min.erase();
    max.erase();
  }

  bounds copy() {
    bounds b=new bounds;
    b.point=point.copy();
    b.min=min.copy();
    b.max=max.copy();
    b.exact=exact;
    return b;
  }

  bounds transformed(transform t) {
    bounds b = new bounds;
    b.point.push(t,this.point,this.point);
    // Add in all 4 corner b.points, to properly size rectangular pictures.
    b.point.push(t,this.min,this.min);
    b.point.push(t,this.min,this.max);
    b.point.push(t,this.max,this.min);
    b.point.push(t,this.max,this.max);
    b.exact = this.exact;
    return b;
  }

  void append(bounds b) {
    this.point.append(b.point);
    this.min.append(b.min);
    this.max.append(b.max);
    if (!b.exact)
      this.exact = false;
  }

  void append(transform t, bounds b) {
    if (t == identity())
      append(b);
    else
      append(b.transformed(t));
  }

  void addPoint(pair user, pair truesize) {
    point.push(user, truesize);
  }

  void addBox(pair userMin, pair userMax, pair trueMin, pair trueMax) {
    // TODO: Change to add all four corners (for translations).
    this.min.push(userMin, trueMin); 
    this.max.push(userMax, trueMax);
  }

  void addPath(path g) {
    // TODO: Store actual path (again for accurate translations of sizing).
    if(size(g) > 0)
      addBox(min(g), max(g), (0,0), (0,0));
  }

  void addPath(path g, pen p) {
    // TODO: Store actual path, pen.
    if(size(g) > 0)
      addBox(min(g), max(g), min(p), max(p));
  }

  void xclip(real Min, real Max) {
    point.xclip(Min,Max);
    min.xclip(Min,Max);
    max.xclip(Min,Max);
  }
  void yclip(real Min, real Max) {
    point.yclip(Min,Max);
    min.yclip(Min,Max);
    max.yclip(Min,Max);
  }
  void clip(triple Min, triple Max) {
    xclip(Min.x,Max.x);
    yclip(Min.y,Max.y);
  }

  // Calculate the min for the final frame, given the coordinate transform.
  pair min(transform t) {
    if(this.min.x.length == 0 && this.point.x.length == 0 &&
       this.max.x.length == 0) return 0;
    pair a=t*(1,1)-t*(0,0), b=t*(0,0);
    scaling xs=scaling.build(a.x,b.x);
    scaling ys=scaling.build(a.y,b.y);
    return (min(min(min(infinity,xs,this.point.x),xs,this.min.x),
                xs,this.max.x),
            min(min(min(infinity,ys,this.point.y),ys,this.min.y),
                ys,this.max.y));
  }

  // Calculate the max for the final frame, given the coordinate transform.
  pair max(transform t) {
    if(this.min.x.length == 0 && this.point.x.length == 0 &&
       this.max.x.length == 0) return 0;
    pair a=t*(1,1)-t*(0,0), b=t*(0,0);
    scaling xs=scaling.build(a.x,b.x);
    scaling ys=scaling.build(a.y,b.y);
    return (max(max(max(-infinity,xs,this.point.x),xs,this.min.x),
                xs,this.max.x),
            max(max(max(-infinity,ys,this.point.y),ys,this.min.y),
                ys,this.max.y));
  }

  // Returns the transform for turning user-space pairs into true-space pairs.
  transform scaling(real xsize, real ysize,
                    real xunitsize, real yunitsize,
                    bool keepAspect, bool warn) {
    if(xsize == 0 && xunitsize == 0 && ysize == 0 && yunitsize == 0)
      return identity();

    coords2 Coords;
    
    // This is unnecessary if both xunitsize and yunitsize are non-zero.
    //append(Coords,Coords,Coords,T,bounds);
    Coords.append(this.point);
    Coords.append(this.min);
    Coords.append(this.max);
    
    real sx;
    if(xunitsize == 0) {
      if(xsize != 0) sx=calculateScaling("x",Coords.x,xsize,warn);
    } else sx=xunitsize;

    /* Possible alternative code : 
    real sx = xunitsize != 0 ? xunitsize :
              xsize != 0     ? calculateScaling("x", Coords.x, xsize, warn) :
                               0; */

    real sy;
    if(yunitsize == 0) {
      if(ysize != 0) sy=calculateScaling("y",Coords.y,ysize,warn);
    } else sy=yunitsize;

    if(sx == 0) {
      sx=sy;
      if(sx == 0)
        return identity();
    } else if(sy == 0) sy=sx;


    if(keepAspect && (xunitsize == 0 || yunitsize == 0))
      return scale(min(sx,sy));
    else
      return scale(sx,sy);
  }
}

bounds operator *(transform t, bounds b) {
  return b.transformed(t);
}
