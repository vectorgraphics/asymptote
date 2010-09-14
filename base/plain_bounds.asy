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
}
