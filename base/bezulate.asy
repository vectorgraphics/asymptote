// Bezier triangulation routines written by Orest Shardt, 2008.

private real fuzz=sqrtEpsilon;
real duplicateFuzz=1e-3; // Work around font errors.

private real[][] intersections(pair a, pair b, path p)
{
  pair delta=fuzz*unit(b-a);
  return intersections(a-delta--b+delta,p,fuzz);
}

int countIntersections(path[] p, pair start, pair end)
{
  int intersects=0;
  for(path q : p)
    intersects += intersections(start,end,q).length;
  return intersects;
}

path[][] containmentTree(path[] paths)
{
  path[][] result;
  for(path g : paths) {
    // check if current curve contains or is contained in a group of curves
    int j;
    for(j=0; j < result.length; ++j) {
      path[] resultj=result[j];
      int test=inside(g,resultj[0],zerowinding);
      if(test == 1) {
        // current curve contains group's toplevel curve;
        // replace toplevel curve with current curve
        resultj.insert(0,g);
        // check to see if any other groups are contained within this curve
        for(int k=j+1; k < result.length;) {
          if(inside(g,result[k][0]) == 1) {
            resultj.append(result[k]);
            result.delete(k);
          } else ++k;
        }
        break;
      } else if(test == -1) {
        // current curve contained within group's toplevel curve
        resultj.push(g);
        break;
      }
    }
    // create a new group if this curve does not belong to another group
    if(j == result.length)
      result.push(new path[] {g});
  }
  return result;
}

bool isDuplicate(pair a, pair b, real relSize)
{
  return abs(a-b) <= duplicateFuzz*relSize;
}

path removeDuplicates(path p)
{
  real relSize = abs(max(p)-min(p));
  bool cyclic=cyclic(p);
  for(int i=0; i < length(p); ++i) {
    if(isDuplicate(point(p,i),point(p,i+1),relSize)) {
      p=subpath(p,0,i)&subpath(p,i+1,length(p));
      --i;
    }
  }
  return cyclic ? p&cycle : p;
}

path section(path p, real t1, real t2, bool loop=false)
{
  if(t2 < t1 || loop && t1 == t2)
    t2 += length(p);
  return subpath(p,t1,t2);
}

path uncycle(path p, real t)
{
  return subpath(p,t,t+length(p));
}

// returns outer paths
void connect(path[] paths, path[] result, path[] patch)
{
  path[][] tree=containmentTree(paths);
  for(path[] group : tree) {
    path outer = group[0];
    group.delete(0);
    path[][] innerTree = containmentTree(group);
    path[] remainingCurves;
    path[] inners;
    for(path[] innerGroup:innerTree)
      {
        inners.push(innerGroup[0]);
        if(innerGroup.length>1)
          remainingCurves.append(innerGroup[1:]);
      }
    connect(remainingCurves,result,patch);
    real d=2*abs(max(outer)-min(outer));
    while(inners.length > 0) {
      int curveIndex = 0;
      pair direction=I*dir(inners[curveIndex],0,1); // Use outgoing direction
      if(direction == 0) // Try a random direction
        direction=expi(2pi*unitrand());
      pair start=point(inners[curveIndex],0);

      // find first intersection of line segment with outer curve
      real[][] ints=intersections(start,start+d*direction,outer);
      assert(ints.length != 0);
      real endtime=ints[0][1]; // endtime is time on outer
      pair end = point(outer,endtime);

      // find first intersection of end--start with any inner curve
      real starttime=0; // starttime is time on inners[curveIndex]
      real earliestTime=1;
      for(int j=0; j < inners.length; ++j) {
        real[][] ints=intersections(end,start,inners[j]);
        if(ints.length > 0 && ints[0][0] < earliestTime) {
          earliestTime=ints[0][0]; // time on end--start
          starttime=ints[0][1]; // time on inner curve
          curveIndex=j;
        }
      }
      start=point(inners[curveIndex],starttime);

      real timeoffset=2;
      bool found=false;
      path portion;
      path[] allCurves = {outer};
      allCurves.append(inners);

      while(!found && timeoffset > fuzz) {
        timeoffset /= 2;
        if(countIntersections(allCurves,start,
                              point(outer,endtime+timeoffset)) == 2)
          {
            portion = subpath(outer,endtime,endtime+timeoffset)--start--cycle;
            found=true;
            // check if an inner curve is inside the portion
            for(int k = 0; found && k < inners.length; ++k)
              {
                if(k!=curveIndex &&
                   inside(portion,point(inners[k],0),zerowinding))
                  found = false;
              }
          }
      }

      if(!found) timeoffset=-2;
      while(!found && timeoffset < -fuzz) {
        timeoffset /= 2;
        if(countIntersections(allCurves,start,
                              point(outer,endtime+timeoffset))==2)
          {
            portion = subpath(outer,endtime+timeoffset,endtime)--start--cycle;
            found = true;
            // check if an inner curve is inside the portion
            for(int k = 0; found && k < inners.length; ++k)
              {
                if(k!=curveIndex &&
                   inside(portion,point(inners[k],0),zerowinding))
                  found = false;
              }
          }
      }
      assert(found);
      endtime=min(endtime,endtime+timeoffset);
      timeoffset=abs(timeoffset);

      // depends on the curves having opposite orientations
      path remainder=section(outer,endtime+timeoffset,endtime)
        --uncycle(inners[curveIndex],
                  starttime)--cycle;
      inners.delete(curveIndex);
      outer = remainder;
      patch.append(portion);
    }
    result.append(outer);
  }
}

bool checkSegment(path g, pair p, pair q)
{
  pair mid=0.5*(p+q);
  return intersections(p,q,g).length == 2 &&
    inside(g,mid,zerowinding) && intersections(g,mid).length == 0;
}

path subdivide(path p)
{
  path q;
  int l=length(p);
  for(int i=0; i < l; ++i)
    q=q&subpath(p,i,i+0.5)&subpath(p,i+0.5,i+1);
  return cyclic(p) ? q&cycle : q;
}

path[] bezulate(path[] p)
{
  if(p.length == 1 && length(p[0]) <= 4) return p;
  path[] patch;
  path[] result;
  connect(p,result,patch);
  for(int i=0; i < result.length; ++i) {
    path p=result[i];
    int refinements=0;
    if(size(p) <= 1) return p;
    if(!cyclic(p))
      abort("path must be cyclic and nonselfintersecting.");
    p=removeDuplicates(p);
    if(length(p) > 4) {
      static real SIZE_STEPS=10;
      static real factor=1.05/SIZE_STEPS;
      for(int k=1; k <= SIZE_STEPS; ++k) {
        real L=factor*k*abs(max(p)-min(p));
        for(int i=0; length(p) > 4 && i < length(p); ++i) {
          bool found=false;
          pair start=point(p,i);
          //look for quadrilaterals and triangles with one line, 4 | 3 curves
          for(int desiredSides=4; !found && desiredSides >= 3;
              --desiredSides) {
            if(desiredSides == 3 && length(p) <= 3)
              break;
            pair end;
            int endi=i+desiredSides-1;
            end=point(p,endi);
            found=checkSegment(p,start,end) && abs(end-start) < L;
            if(found) {
              path p1=subpath(p,endi,i+length(p))--cycle;
              patch.append(subpath(p,i,endi)--cycle);
              p=removeDuplicates(p1);
              i=-1; // increment will make i be 0
            }
          }
          if(!found && k == SIZE_STEPS && length(p) > 4 && i == length(p)-1) {
            // avoid infinite recursion
            ++refinements;
            if(refinements > mantissaBits) {
              warning("subdivisions","too many subdivisions",position=true);
            } else {
              p=subdivide(p);
              i=-1;
            }
          }
        }
      }
    }
    if(length(p) <= 4)
      patch.append(p);
  }
  return patch;
}
