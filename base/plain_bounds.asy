private import plain_scaling;

// After an transformation, produce new coordinate bounds.  For paths that
// have been added, this is only an approximation since it takes the bounds of
// their transformed bounding box.
private
void addTransformedCoords(coords2 dest, transform t,
                          coords2 point, coords2 min, coords2 max)
{
  dest.push(t, point, point);

  // Add in all 4 corner coords, to properly size rectangular pictures.
  dest.push(t,min,min);
  dest.push(t,min,max);
  dest.push(t,max,min);
  dest.push(t,max,max);
}

// This stores a list of sizing bounds for picture data.  If the object is
// frozen, then it cannot be modified further, and therefore can be safely
// passed by reference and stored in the sizing data for multiple pictures.
private struct freezableBounds {
  restricted bool frozen = false;
  void freeze() {
    frozen = true;
  }

  // Optional links to further (frozen) sizing data.
  private freezableBounds[] links;

  // Links to (frozen) sizing data that is transformed when added here.
  private static struct transformedBounds {
    transform t;
    freezableBounds link;
  };
  private transformedBounds[] tlinks;

  // The sizing data.  It cannot be modified once this object is frozen.
  private coords2 point, min, max;

  // A bound represented by a path.  Using the path instead of the bounding
  // box means it will be accurate after a transformation by coordinates.
  private path[] pathBounds;

  // A bound represented by a path and a pen.
  private static struct pathpen {
    path g; pen p;

    void operator init(path g, pen p) {
      this.g = g;
      this.p = p;
    }
  }
  private static pathpen operator *(transform t, pathpen pp) {
    // Should the pen be transformed?
    return pathpen(t*pp.g, pp.p);
  }
  private pathpen[] pathpenBounds;

  // Once frozen, getMutable returns a new object based on this one, which can
  // be modified.
  freezableBounds getMutable() {
    assert(frozen);
    var f = new freezableBounds;
    f.links.push(this);
    return f;
  }

  freezableBounds transformed(transform t) {
    // Freeze these bounds, as we are storing a reference to them.
    freeze();

    var tlink = new transformedBounds;
    tlink.t = t;
    tlink.link = this;

    var b = new freezableBounds;
    b.tlinks.push(tlink);

    return b;
  }

  void append(freezableBounds b) {
    // Check that we can modify the object.
    assert(!frozen);

    //TODO: If b is "small", ie. a single tlink or cliplink, just copy the
    //link.

    // As we only reference b, we must freeze it to ensure it does not change.
    b.freeze();
    links.push(b);
  }

  void addPoint(pair user, pair truesize) {
    assert(!frozen);
    point.push(user, truesize);
  }

  void addBox(pair userMin, pair userMax, pair trueMin, pair trueMax) {
    assert(!frozen);
    this.min.push(userMin, trueMin); 
    this.max.push(userMax, trueMax);
  }

  void addPath(path g) {
    assert(!frozen);
    this.pathBounds.push(g);
  }

  void addPath(path g, pen p) {
    assert(!frozen);
    this.pathpenBounds.push(pathpen(g,p));
  }

  // Transform the sizing info by t then add the result to the coords
  // structure.
  private void accumulateCoords(transform t, coords2 coords) {
    for (var link : links)
      link.accumulateCoords(t, coords);

    for (var tlink : tlinks)
      tlink.link.accumulateCoords(t*tlink.t, coords);

    addTransformedCoords(coords, t, this.point, this.min, this.max);

    for (var g : pathBounds) {
      g = t*g;
      coords.push(min(g), (0,0));
      coords.push(max(g), (0,0));
    }

    for (var pp: pathpenBounds) {
      pp = t*pp;
      coords.push(min(pp.g), min(pp.p));
      coords.push(max(pp.g), max(pp.p));
    }
  }

  // Add all of the sizing info to the given coords structure.
  private void accumulateCoords(coords2 coords) {
    for (var link : links)
      link.accumulateCoords(coords);

    for (var tlink : tlinks)
      tlink.link.accumulateCoords(tlink.t, coords);

    coords.append(this.point);
    coords.append(this.min);
    coords.append(this.max);

    for (var g : pathBounds) {
      coords.push(min(g), (0,0));
      coords.push(max(g), (0,0));
    }

    for (var pp: pathpenBounds) {
      coords.push(min(pp.g), min(pp.p));
      coords.push(max(pp.g), max(pp.p));
    }
  }

  // Returns all of the coords that this sizing data represents.
  private coords2 allCoords() {
    coords2 coords;
    accumulateCoords(coords);
    return coords;
  }

  // A temporary measure.  Stuffs all of the data from the links and paths
  // into the coords.
  private void flatten() {
    assert(!frozen);

    // Calculate all coordinates.
    coords2 coords = allCoords();

    // Erase all the old data.
    point.erase();
    min.erase();
    max.erase();
    pathBounds.delete();
    pathpenBounds.delete();
    links.delete();
    tlinks.delete();

    // Put all of the coordinates into point.
    point = coords;
  }

  void xclip(real Min, real Max) {
    assert(!frozen);
    flatten();
    point.xclip(Min,Max);
    min.xclip(Min,Max);
    max.xclip(Min,Max);
  }

  void yclip(real Min, real Max) {
    assert(!frozen);
    flatten();
    point.yclip(Min,Max);
    min.yclip(Min,Max);
    max.yclip(Min,Max);
  }


  // Calculate the min for the final frame, given the coordinate transform.
  pair min(transform t) {
    coords2 coords = allCoords();
    if (coords.x.length == 0)
      return 0;

    pair a=t*(1,1)-t*(0,0), b=t*(0,0);
    scaling xs=scaling.build(a.x,b.x);
    scaling ys=scaling.build(a.y,b.y);

    return (min(infinity, xs, coords.x), min(infinity, ys, coords.y));
  }

  // Calculate the max for the final frame, given the coordinate transform.
  pair max(transform t) {
    coords2 coords = allCoords();
    if (coords.x.length == 0)
      return 0;

    pair a=t*(1,1)-t*(0,0), b=t*(0,0);
    scaling xs=scaling.build(a.x,b.x);
    scaling ys=scaling.build(a.y,b.y);

    return (max(-infinity, xs, coords.x), max(-infinity, ys, coords.y));
  }

  // Returns the transform for turning user-space pairs into true-space pairs.
  transform scaling(real xsize, real ysize,
                    real xunitsize, real yunitsize,
                    bool keepAspect, bool warn) {
    if(xsize == 0 && xunitsize == 0 && ysize == 0 && yunitsize == 0)
      return identity();

    // This is unnecessary if both xunitsize and yunitsize are non-zero.
    coords2 Coords = allCoords();
    
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

struct smartBounds {
  private var base = new freezableBounds;

  // We should probably put this back into picture.
  bool exact = true;

  // Called just before modifying the sizing data.  It ensures base is
  // non-frozen.
  private void makeMutable() {
    if (base.frozen)
      base = base.getMutable();
    assert(!base.frozen);
  }

  void erase() {
    // Just discard the old bounds.
    base = new freezableBounds;
  }

  smartBounds copy() {
    // Freeze the underlying bounds and make a shallow copy.
    base.freeze();

    var b = new smartBounds;
    b.base = this.base;
    b.exact = this.exact;
    return b;
  }

  smartBounds transformed(transform t) {
    var b = new smartBounds;
    b.base = base.transformed(t);
    b.exact = this.exact;
    return b;
  }

  void append(smartBounds b) {
    makeMutable();
    base.append(b.base);
  }
    
  void append(transform t, smartBounds b) {
    // makeMutable will be called by append.
    if (t == identity())
      append(b);
    else
      append(b.transformed(t));
  }

  void addPoint(pair user, pair truesize) {
    makeMutable();
    base.addPoint(user, truesize);
  }

  void addBox(pair userMin, pair userMax, pair trueMin, pair trueMax) {
    makeMutable();
    base.addBox(userMin, userMax, trueMin, trueMax);
  }

  void addPath(path g) {
    makeMutable();
    base.addPath(g);
  }

  void addPath(path g, pen p) {
    makeMutable();
    base.addPath(g, p);
  }

  void xclip(real Min, real Max) {
    makeMutable();
    base.xclip(Min,Max);
  }

  void yclip(real Min, real Max) {
    makeMutable();
    base.yclip(Min,Max);
  }
  
  void clip(triple Min, triple Max) {
    xclip(Min.x,Max.x);
    yclip(Min.y,Max.y);
  }

  pair min(transform t) {
    return base.min(t);
  }

  pair max(transform t) {
    return base.max(t);
  }

  transform scaling(real xsize, real ysize,
                    real xunitsize, real yunitsize,
                    bool keepAspect, bool warn) {
    return base.scaling(xsize, ysize, xunitsize, yunitsize, keepAspect, warn);
  }
}

struct bounds {
  coords2 point,min,max;
  bool exact=true; // An accurate picture bounds is provided by the user.

  bool empty() {
    return min.x.length == 0 && point.x.length == 0 && max.x.length == 0;
  }

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
    if (this.empty())
      return 0;

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
    if (this.empty())
      return 0;

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

smartBounds operator *(transform t, smartBounds b) {
  return b.transformed(t);
}
bounds operator *(transform t, bounds b) {
  return b.transformed(t);
}
