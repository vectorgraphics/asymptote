include plain_scaling;

// After an transformation, produce new coordinate bounds.  For paths that
// have been added, this is only an approximation since it takes the bounds of
// their transformed bounding box.
private void addTransformedCoords(coords2 dest, transform t,
                          coords2 point, coords2 min, coords2 max)
{
  dest.push(t, point, point);

  // Add in all 4 corner coords, to properly size rectangular pictures.
  dest.push(t,min,min);
  dest.push(t,min,max);
  dest.push(t,max,min);
  dest.push(t,max,max);
}

// Adds another sizing restriction to the coordinates, but only if it is
// maximal, that is, if under some scaling, this coordinate could be the
// largest.
private void addIfMaximal(coord[] coords, real user, real truesize) {
  // TODO: Test promoting coordinates for efficiency.

  for (coord c : coords)
    if (user <= c.user && truesize <= c.truesize)
      // Not maximal.
      return;

  // The coordinate is not dominated by any existing extreme, so it is
  // maximal and will be added, but first remove any coords it now dominates.
  int i = 0;
  while (i < coords.length) {
    coord c = coords[i];
    if (c.user <= user && c.truesize <= truesize)
      coords.delete(i);
    else
      ++i;
  }

  // Add the coordinate to the extremes.
  coords.push(coord.build(user, truesize));
}

private void addIfMaximal(coord[] dest, coord[] src)
{
  // This may be inefficient, as it rebuilds the coord struct when adding it.
  for (coord c : src)
    addIfMaximal(dest, c.user, c.truesize);
}
      
// Same as addIfMaximal, but testing for minimal coords.
private void addIfMinimal(coord[] coords, real user, real truesize) {
  for (coord c : coords)
    if (user >= c.user && truesize >= c.truesize)
      return;

  int i = 0;
  while (i < coords.length) {
    coord c = coords[i];
    if (c.user >= user && c.truesize >= truesize)
      coords.delete(i);
    else
      ++i;
  }

  coords.push(coord.build(user, truesize));
}

private void addIfMinimal(coord[] dest, coord[] src)
{
  for (coord c : src)
    addIfMinimal(dest, c.user, c.truesize);
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
  // As often many paths use the same pen, we store an array of paths.
  private static struct pathpen {
    path[] g; pen p;

    void operator init(path g, pen p) {
      this.g.push(g);
      this.p = p;
    }
  }
  private static pathpen operator *(transform t, pathpen pp) {
    // Should the pen be transformed?
    pathpen newpp;
    for (path g : pp.g)
      newpp.g.push(t*g);
    newpp.p = pp.p;
    return newpp;
  }

  // WARNING: Due to crazy optimizations, if this array is changed between an
  // empty and non-empty state, the assignment of a method to
  // addPath(path,pen) must also change.
  private pathpen[] pathpenBounds;

  // Once frozen, the sizing is immutable, and therefore we can compute and
  // store the extremal coordinates.
  public static struct extremes {
    coord[] left, bottom, right, top;

    void operator init(coord[] left, coord[] bottom,
                       coord[] right, coord[] top) {
      this.left = left;
      this.bottom = bottom; 
      this.right = right;
      this.top = top;
    }

  }
  private static void addMaxToExtremes(extremes e, pair user, pair truesize) {
    addIfMaximal(e.right, user.x, truesize.x);
    addIfMaximal(e.top, user.y, truesize.y);
  }
  private static void addMinToExtremes(extremes e, pair user, pair truesize) {
    addIfMinimal(e.left, user.x, truesize.x);
    addIfMinimal(e.bottom, user.y, truesize.y);
  }
  private static void addMaxToExtremes(extremes e, coords2 coords) {
    addIfMaximal(e.right, coords.x);
    addIfMaximal(e.top, coords.y);
  }
  private static void addMinToExtremes(extremes e, coords2 coords) {
    addIfMinimal(e.left, coords.x);
    addIfMinimal(e.bottom, coords.y);
  }

  private extremes cachedExtremes = null;

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
    // This, and other asserts have been removed to speed things up slightly.
    //assert(!frozen);
    this.pathBounds.push(g);
  }

  void addPath(path[] g) {
    //assert(!frozen);
    this.pathBounds.append(g);
  }

  // To squeeze out a bit more performance, this method is either assigned
  // addPathToNonEmptyArray or addPathToEmptyArray depending on the state of
  // the pathpenBounds array.
  void addPath(path g, pen p);

  private void addPathToNonEmptyArray(path g, pen p) {
    //assert(!frozen);
    //assert(!pathpenBounds.empty());
    var pp = pathpenBounds[0];

    // Test if the pens are equal or have the same bounds.
    if (pp.p == p || (min(pp.p) == min(p) && max(pp.p) == max(p))) {
      // If this path has the same pen as the last one, just add it to the
      // array corresponding to that pen.
      pp.g.push(g);
    }
    else {
      // A different pen.  Start a new bound and put it on the front.  Put
      // the old bound at the end of the array.
      pathpenBounds[0] = pathpen(g,p);
      pathpenBounds.push(pp);
    }
  }
  void addPathToEmptyArray(path g, pen p) {
    //assert(!frozen);
    //assert(pathpenBounds.empty());

    pathpenBounds.push(pathpen(g,p));
    addPath = addPathToNonEmptyArray;
  }

  // Initial setting for addPath.
  addPath = addPathToEmptyArray;

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
      pair pm = min(pp.p), pM = max(pp.p);
      for (var g : pp.g) {
        g = t*g;
        coords.push(min(g), pm);
        coords.push(max(g), pM);
      }
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
      pair pm = min(pp.p), pM = max(pp.p);
      for (var g : pp.g) {
        coords.push(min(g), pm);
        coords.push(max(g), pM);
      }
    }
  }

  // Returns all of the coords that this sizing data represents.
  private coords2 allCoords() {
    coords2 coords;
    accumulateCoords(coords);
    return coords;
  }

  private void addLocalsToExtremes(transform t, extremes e) {
    coords2 coords;
    addTransformedCoords(coords, t, this.point, this.min, this.max);
    addMinToExtremes(e, coords);
    addMaxToExtremes(e, coords);

    if (pathBounds.length > 0) {
      addMinToExtremes(e, minAfterTransform(t, pathBounds), (0,0));
      addMaxToExtremes(e, maxAfterTransform(t, pathBounds), (0,0));
    }

    for (var pp : pathpenBounds) {
      if (pp.g.length > 0) {
        addMinToExtremes(e, minAfterTransform(t, pp.g), min(pp.p));
        addMaxToExtremes(e, maxAfterTransform(t, pp.g), max(pp.p));
      }
    }
  }

  private void addToExtremes(transform t, extremes e) {
    for (var link : links)
      link.addToExtremes(t, e);

    for (var tlink : tlinks)
      tlink.link.addToExtremes(t*tlink.t, e);

    addLocalsToExtremes(t, e);
  }
    
  private void addLocalsToExtremes(extremes e) {
    addMinToExtremes(e, point);
    addMaxToExtremes(e, point);
    addMinToExtremes(e, min);
    addMaxToExtremes(e, max);

    if (pathBounds.length > 0) {
      addMinToExtremes(e, min(pathBounds), (0,0));
      addMaxToExtremes(e, max(pathBounds), (0,0));
    }

    for(var pp : pathpenBounds) {
      pair m=min(pp.p);
      pair M=max(pp.p);
      for(path gg : pp.g) {
        if (size(gg) > 0) {
          addMinToExtremes(e,min(gg),m);
          addMaxToExtremes(e,max(gg),M);
        }
      }
    }
  }

  private void addToExtremes(extremes e) {
    for (var link : links)
      link.addToExtremes(e);

    for (var tlink : tlinks)
      tlink.link.addToExtremes(tlink.t, e);

    addLocalsToExtremes(e);
  }

  private static void write(extremes e) {
    static void write(coord[] coords) {
      for (coord c : coords)
        write("  " + (string)c.user + " u + " + (string)c.truesize);
    }
    write("left:");
    write(e.left);
    write("bottom:");
    write(e.bottom);
    write("right:");
    write(e.right);
    write("top:");
    write(e.top);
  }

  // Returns the extremal coordinates of the sizing data.
  public extremes extremes() {
    if (cachedExtremes == null) {
      freeze();

      extremes e;
      addToExtremes(e);
      cachedExtremes = e;
    }

    return cachedExtremes;
  }

  // Helper functions for computing the usersize bounds.  usermin and usermax
  // would be easily computable from extremes, except that the picture
  // interface actually allows calls that manually change the usermin and
  // usermax values.  Therefore, we have to compute these values separately.
  private static struct userbounds {
    bool areSet=false;
    pair min;
    pair max;
  }
  private static struct boundsAccumulator {
    pair[] mins;
    pair[] maxs;

    void push(pair m, pair M) {
      mins.push(m);
      maxs.push(M);
    }

    void push(userbounds b) {
      if (b.areSet)
        push(b.min, b.max);
    }

    void push(transform t, userbounds b) {
      if (b.areSet) {
        pair[] box = { t*(b.min.x,b.max.y), t*b.max,
                       t*b.min,             t*(b.max.x,b.min.y) };
        for (var z : box)
          push(z,z);
      }
    }

    void pushUserCoords(coords2 min, coords2 max) {
      int n = min.x.length;
      assert(min.y.length == n);
      assert(max.x.length == n);
      assert(max.y.length == n);

      for (int i = 0; i < n; ++i)
        push((min.x[i].user, min.y[i].user),
             (max.x[i].user, max.y[i].user));
    }

    userbounds collapse() {
      userbounds b;
      if (mins.length > 0) {
        b.areSet = true;
        b.min = minbound(mins);
        b.max = maxbound(maxs);
      }
      else {
        b.areSet = false;
      }
      return b;
    }
  }

  // The user bounds already calculated for this data.
  private userbounds storedUserBounds = null;

  private void accumulateUserBounds(boundsAccumulator acc)
  {
    if (storedUserBounds != null) {
      assert(frozen);
      acc.push(storedUserBounds);
    } else {
      acc.pushUserCoords(point, point);
      acc.pushUserCoords(min, max);
      if (pathBounds.length > 0)
        acc.push(min(pathBounds), max(pathBounds));
      for (var pp : pathpenBounds)
        acc.push(min(pp.g), max(pp.g));
      for (var link : links)
        link.accumulateUserBounds(acc);

      // Transforms are handled as they were in the old system.
      for (var tlink : tlinks) {
        boundsAccumulator tacc;
        tlink.link.accumulateUserBounds(tacc);
        acc.push(tlink.t, tacc.collapse());
      }
    }
  }

  private void computeUserBounds() {
    freeze();
    boundsAccumulator acc;
    accumulateUserBounds(acc);
    storedUserBounds = acc.collapse();
  }

  private userbounds userBounds() {
    if (storedUserBounds == null)
      computeUserBounds();

    assert(storedUserBounds != null);
    return storedUserBounds;
  }

  // userMin/userMax returns the minimal/maximal userspace coordinate of the
  // sizing data.  As coordinates for objects such as labels can have
  // significant truesize dimensions, this userMin/userMax values may not
  // correspond closely to the end of the screen, and are of limited use.
  // userSetx and userSety determine if there is sizing data in order to even
  // have userMin/userMax defined.
  public bool userBoundsAreSet() {
    return userBounds().areSet;
  }

  public pair userMin() {
    return userBounds().min;
  }
  public pair userMax() {
    return userBounds().max;
  }

  // To override the true userMin and userMax bounds, first compute the
  // userBounds as they should be at this point, then change the values.
  public void alterUserBound(string which, real val) {
    // We are changing the bounds data, so it cannot be frozen yet.  After the
    // user bounds are set, however, the sizing data cannot change, so it will
    // be frozen.
    assert(!frozen);
    computeUserBounds();
    assert(frozen);

    var b = storedUserBounds;
    if (which == "minx")
      b.min = (val, b.min.y);
    else if (which == "miny")
      b.min = (b.min.x, val);
    else if (which == "maxx")
      b.max = (val, b.max.y);
    else {
      assert(which == "maxy");
      b.max = (b.max.x, val);
    }
  }

  // A temporary measure.  Stuffs all of the data from the links and paths
  // into the coords.
  private void flatten() {
    assert(!frozen);

    // First, compute the user bounds, taking into account any manual
    // alterations.
    computeUserBounds();

    // Calculate all coordinates.
    coords2 coords = allCoords();

    // Erase all the old data.
    point.erase();
    min.erase();
    max.erase();
    pathBounds.delete();
    pathpenBounds.delete();
    addPath = addPathToEmptyArray;
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

    // Cap the userBounds.
    userbounds b = storedUserBounds;
    b.min = (max(Min, b.min.x), b.min.y);
    b.max = (min(Max, b.max.x), b.max.y);
  }

  void yclip(real Min, real Max) {
    assert(!frozen);
    flatten();
    point.yclip(Min,Max);
    min.yclip(Min,Max);
    max.yclip(Min,Max);

    // Cap the userBounds.
    userbounds b = storedUserBounds;
    b.min = (b.min.x, max(Min, b.min.y));
    b.max = (b.max.x, min(Max, b.max.y));
  }

  // Calculate the min for the final frame, given the coordinate transform.
  pair min(transform t) {
    extremes e = extremes();
    if (e.left.length == 0)
      return 0;

    pair a=t*(1,1)-t*(0,0), b=t*(0,0);
    scaling xs=scaling.build(a.x,b.x);
    scaling ys=scaling.build(a.y,b.y);

    return (min(infinity, xs, e.left), min(infinity, ys, e.bottom));
  }

  // Calculate the max for the final frame, given the coordinate transform.
  pair max(transform t) {
    extremes e = extremes();
    if (e.right.length == 0)
      return 0;

    pair a=t*(1,1)-t*(0,0), b=t*(0,0);
    scaling xs=scaling.build(a.x,b.x);
    scaling ys=scaling.build(a.y,b.y);

    return (max(-infinity, xs, e.right), max(-infinity, ys, e.top));
  }

  // Returns the transform for turning user-space pairs into true-space pairs.
  transform scaling(real xsize, real ysize,
                    real xunitsize, real yunitsize,
                    bool keepAspect, bool warn) {
    if(xsize == 0 && xunitsize == 0 && ysize == 0 && yunitsize == 0)
      return identity();

    // Get the extremal coordinates.
    extremes e = extremes();
    
    real sx;
    if(xunitsize == 0) {
      if(xsize != 0) sx=calculateScaling("x",e.left,e.right,xsize,warn);
    } else sx=xunitsize;

    /* Possible alternative code : 
       real sx = xunitsize != 0 ? xunitsize :
       xsize != 0     ? calculateScaling("x", Coords.x, xsize, warn) :
       0; */

    real sy;
    if(yunitsize == 0) {
      if(ysize != 0) sy=calculateScaling("y",e.bottom,e.top,ysize,warn);
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

struct bounds {
  private var base = new freezableBounds;

  // We should probably put this back into picture.
  bool exact = true;

  // Called just before modifying the sizing data.  It ensures base is
  // non-frozen.
  // Note that this is manually inlined for speed reasons in a couple often
  // called methods below.
  private void makeMutable() {
    if (base.frozen)
      base = base.getMutable();
    //assert(!base.frozen); // Disabled for speed reasons.
  }

  void erase() {
    // Just discard the old bounds.
    base = new freezableBounds;

    // We don't reset the 'exact' field, for backward compatibility.
  }

  bounds copy() {
    // Freeze the underlying bounds and make a shallow copy.
    base.freeze();

    var b = new bounds;
    b.base = this.base;
    b.exact = this.exact;
    return b;
  }

  bounds transformed(transform t) {
    var b = new bounds;
    b.base = base.transformed(t);
    b.exact = this.exact;
    return b;
  }

  void append(bounds b) {
    makeMutable();
    base.append(b.base);
  }
    
  void append(transform t, bounds b) {
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
    //makeMutable(); // Manually inlined here for speed reasons.
    if (base.frozen)
      base = base.getMutable();
    base.addPath(g);
  }

  void addPath(path[] g) {
    //makeMutable(); // Manually inlined here for speed reasons.
    if (base.frozen)
      base = base.getMutable();
    base.addPath(g);
  }

  void addPath(path g, pen p) {
    //makeMutable(); // Manually inlined here for speed reasons.
    if (base.frozen)
      base = base.getMutable();
    base.addPath(g, p);
  }

  public bool userBoundsAreSet() {
    return base.userBoundsAreSet();
  }
  public pair userMin() {
    return base.userMin();
  }
  public pair userMax() {
    return base.userMax();
  }
  public void alterUserBound(string which, real val) {
    makeMutable();
    base.alterUserBound(which, val);
  }

  void xclip(real Min, real Max) {
    makeMutable();
    base.xclip(Min,Max);
  }

  void yclip(real Min, real Max) {
    makeMutable();
    base.yclip(Min,Max);
  }
  
  void clip(pair Min, pair Max) {
    // TODO: If the user bounds have been manually altered, they may be
    // incorrect after the clip.
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

bounds operator *(transform t, bounds b) {
  return b.transformed(t);
}
