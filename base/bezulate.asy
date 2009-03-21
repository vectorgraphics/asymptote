// Bezier triangulation routines written by Orest Shardt, 2008.

int countIntersections(path[] p, pair start, pair end)
{
  int intersects=0;
  for(path q : p)
    intersects += intersections(q,start--end).length;
  return intersects;
}

path[][] containmentTree(path[] paths)
{
  path[][] result;
  for(int i=0; i < paths.length; ++i) {
    bool classified=false;
    // check if current curve contains or is contained in a group of curves
    for(int j=0; !classified && j < result.length; ++j)
    {
      int test = inside(paths[i],result[j][0],zerowinding);
      if(test == 1) // current curve contains group's toplevel curve
      {
        // replace toplevel curve with current curve
        result[j].insert(0,paths[i]);
        classified = true;
      }
      else if(test == -1) // current curve contained in group's toplevel curve
      {
        result[j].push(paths[i]);
        classified = true;
      }
    }
    // create a new group if this curve does not belong to another group
    if(!classified)
      result.push(new path[] {paths[i]});
  }

  // sort group so that later paths in the array are contained in previous paths
  bool comparepaths(path i, path j) {return inside(i,j,zerowinding)==1;}
  for(int i=0; i < result.length; ++i)
    result[i] = sort(result[i],comparepaths);

  return result;
}

bool isDuplicate(pair a, pair b, real relSize)
{
  return abs(a-b) <= sqrtEpsilon*relSize;
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
path[] connect(path[] paths, path[] result, path[] patch, int depth=0)
{
  bool flag=depth % 2 == 0;
  path[][] tree=containmentTree(paths);
  path[] outers;
  for(path[] group : tree) {
    if(group.length == 1) {
      if(flag)
        result.push(group[0]);
      else
        outers.push(group[0]);
    } else { // optimize case where group.length == 2 to avoid call to connect
      path[][] tree=containmentTree(group[1:]);
      path[] inners;
      for(path[] subgroup : tree) {
        //connect outer curve to result of connecting inner curves
        if(!flag) {
          outers.append(connect(subgroup,result,patch,depth+1));
        } else {
          path[] conn=connect(subgroup,result,patch,depth+1);
          inners.append(conn);
        }
      }
      path outer=group[0];
      if(flag) {
        real d=2*abs(max(outer)-min(outer));
        for(int i=0; i < inners.length; ++i) {
          path[] allCurves={outer};
          allCurves.append(inners[i:]);

          path inner=inners[i];
          pair direction=I*dir(inner,0);
          pair start=point(inner,0);
          real starttime=0;
          // find an outer point on inner curve in the chosen direction
          starttime=intersections(start+d*direction--start,inner)[0][1];
          start=point(inner,starttime);

          // find earliest intersection
          real[][] ints=intersections(start--start+d*direction,outer);
          assert(ints.length != 0);
          real endtime=ints[0][1];
          real earliestTime=intersections(start--start+d*direction,outer)[0][0];
          int curveIndex=0;
          for(int j=i+1; j < inners.length; ++j) {
            real[][] ints=intersections(start--start+d*direction,inners[j]);
            if(ints.length > 0 && ints[0][0] < earliestTime) {
              earliestTime=ints[0][0];
              endtime=ints[0][1];
              curveIndex=j+1;
            }
          }
          pair end=point(allCurves[curveIndex],endtime);

          real timeoffset=2;
          bool found=false;
          while(!found && timeoffset > sqrtEpsilon) {
            timeoffset /= 2;
            if(countIntersections(allCurves,start,
                                  point(allCurves[curveIndex],
                                        endtime+timeoffset)) == 2)
              found=true;
          }
          if(!found)timeoffset=-2;
          while(!found && timeoffset > sqrtEpsilon) {
            timeoffset /= 2;
            if(countIntersections(allCurves,start,
                                  point(allCurves[curveIndex],
                                        endtime+timeoffset)) == 2)
              found=true;
          }
          assert(found);
          endtime=min(endtime,endtime+timeoffset);
          timeoffset=abs(timeoffset);

          path remainder=section(allCurves[curveIndex],endtime+timeoffset,
                                 endtime)--uncycle(inner,starttime)--cycle;
          if(curveIndex == 0)
            outer=remainder;
          else
            inners[curveIndex-1]=remainder;
          patch.append(start--section(allCurves[curveIndex],endtime,
                                      endtime+timeoffset)--cycle);
        }
      }
      outers.push(outer);
    }
  }
  return outers;
}

int countIntersections(path g, pair p, pair q)
{
  int ints=0;
  int l=length(g);
  for(int i=1; i <= l; ++i)
    ints += intersections(subpath(g,i-1,i),p--q).length;
  return ints;
}

bool checkSegment(path g, pair p, pair q)
{
  pair mid=0.5*(p+q);
  return countIntersections(g,p,q) == 4 && inside(g,mid,zerowinding) && 
    intersections(g,mid).length == 0;
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
  result.append(connect(p,result,patch));
  for(int i=0; i < result.length; ++i) {
    path p=result[i];
    int refinements=0;
    static int maxR=ceil(-log(realEpsilon)/log(2))+1;
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
            if(refinements > maxR) {
              write("warning: too many subdivisions");
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
