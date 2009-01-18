// a function to round sharp edges of open and cyclic paths
// written by stefan knorr

path roundedpath(path A, real R, real S = 1)                   
// create rounded path from path A with radius R and scale S = 1
{
  path RoundPath;                   // returned path
  path LocalPath;                   // local straight subpath
  path LocalCirc;                   // local edge circle for intersection
  real LocalTime;                   // local intersectiontime between . and ..
  pair LocalPair;                   // local point to be added to 'RoundPath'
            
  int len=length(A);              // length of given path 'A' 
  bool PathClosed=cyclic(A);      // true, if given path 'A' is cyclic

  // initialisation: define first Point of 'RoundPath' as
  if (PathClosed)                         // ? is 'A' cyclic
    RoundPath=scale(S)*point(point(A,0)--point(A,1), 0.5);  // centerpoint of first straight subpath of 'A'  
  else
    RoundPath=scale(S)*point(A,0);            // first point of 'A'

  // doing everything between start and end 
  // create round paths subpath by subpath for every i-th edge 
  for(int i=1; i < len; ++i)
    { 
      // straight subpath towards i-th edge
      LocalPath=point(A,i-1)---point(A,i);
      // circle with radius 'R' around i-th edge
      LocalCirc=circle(point(A,i),R);
      // calculate intersection time between straight subpath and circle
      real[] t=intersect(LocalPath, LocalCirc);
      if(t.length > 0) {
	LocalTime=t[0];
	// define intersectionpoint between both paths
	LocalPair=point(subpath(LocalPath, 0, LocalTime), 1);
	// add straight subpath towards i-th curvature to 'RoundPath'
	RoundPath=RoundPath--scale(S)*LocalPair;
      }
    
      // straight subpath from i-th edge to (i+1)-th edge
      LocalPath=point(A,i)---point(A,i+1);
      // calculate intersection-time between straight subpath and circle
      real[] t=intersect(LocalPath, LocalCirc);
      if(t.length > 0) {
	LocalTime=t[0];
	// define intersectionpoint between both paths  
	LocalPair=point(subpath(LocalPath, 0, LocalTime), 1);
	// add curvature near i-th edge to 'RoundPath'
	RoundPath=RoundPath..scale(S)*LocalPair;
      }
    } 

  // final steps to have a correct termination 
  if(PathClosed) { // Is 'A' cyclic?
    // straight subpath towards 0-th edge 
    LocalPath=point(A,len-1)---point(A,0);
    // circle with radius 'R' around 0-th edge
    LocalCirc=circle(point(A,0),R);
    // calculate intersection-time between straight subpath and circle
    real[] t=intersect(LocalPath, LocalCirc);
    if(t.length > 0) {
      LocalTime=t[0];
      // define intersectionpoint between both paths
      LocalPair=point(subpath(LocalPath, 0, LocalTime), 1);
      // add straight subpath towards 0-th curvature to 'RoundPath'
      RoundPath=RoundPath--scale(S)*LocalPair;
    }
    
    
    // straight subpath from 0-th edge to 1st edge
    LocalPath=point(A,0)---point(A,1);
    // calculate intersection-time between straight subpath and circle
    real[] t=intersect(LocalPath, LocalCirc);
    if(t.length > 0) {
      LocalTime=t[0];
      // define intersectionpoint between both paths  
      LocalPair=point(subpath(LocalPath, 0, LocalTime), 1);
      // add curvature near 0-th edge to 'RoundPath' and close path
      RoundPath=RoundPath..scale(S)*LocalPair--cycle;
    }
  } else
    RoundPath=RoundPath--scale(S)*point(A,len);
  return RoundPath;
}
