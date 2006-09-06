// a function to round sharp edges of open and cyclic paths
// written by stefan knorr

path roundedpath(path A, real R, real S = 1)                   
// create rounded path from path A with radius R and scale S = 1
{
  path RoundPath;                   // returned path
  path LocalPath;                   // local straight subpath
  path LocalCirc;                   // local edge circle for intersection
  pair LocalTime;                   // local intersectiontime between . and ..
  pair LocalPair;                   // local point to be added to 'RoundPath'
            
  int len = length(A);              // length of given path 'A' 
  bool PathClosed = cyclic(A);      // true, if given path 'A' is cyclic

  // initialisation: define first Point of 'RoundPath' as
  if (PathClosed)                         // ? is 'A' cyclic
    RoundPath = scale(S)*point(point(A,0)--point(A,1), 0.5);  // centerpoint of first straight subpath of 'A'  
  else
    RoundPath = scale(S)*point(A,0);            // first point of 'A'

// doing everything between start and end 
  for(int i = 1; i < len; ++i)                  // create round paths subpath by subpath for every i-th edge 
    { 
      LocalPath = point(A,i-1)---point(A,i);      // staight subpath towards i-th edge 
      LocalCirc = circle(point(A,i),R);           // circle with radius 'R' around i-th edge
      LocalTime = intersect(LocalPath, LocalCirc);// calculate intersection-time between straight subpath and circle
      LocalPair = point(subpath(LocalPath, 0, LocalTime.x), 1); // define intersectionpoint between both paths
      RoundPath = RoundPath--scale(S)*LocalPair;  // add straight subpath towards i-th curvature to 'RoundPath'
    
      LocalPath = point(A,i)---point(A,i+1);      // staight subpath from i-th edge to (i+1)-th edge
      LocalTime = intersect(LocalPath, LocalCirc); // calculate intersection-time between straight subpath and circle
      LocalPair = point(subpath(LocalPath, 0, LocalTime.x), 1); // define intersectionpoint between both paths  
      RoundPath = RoundPath..scale(S)*LocalPair;  // add curvature near i-th edge to 'RoundPath'
    } 

// final steps to have a correct termination 
  if (PathClosed)                               // ? is 'A' cyclic                       
    {
      LocalPath = point(A,len-1)---point(A,0);    // staight subpath towards 0-th edge 
      LocalCirc = circle(point(A,0),R);           // circle with radius 'R' around 0-th edge
      LocalTime = intersect(LocalPath, LocalCirc);// calculate intersection-time between straight subpath and circle
      LocalPair = point(subpath(LocalPath, 0, LocalTime.x), 1); // define intersectionpoint between both paths
      RoundPath = RoundPath--scale(S)*LocalPair;  // add straight subpath towards 0-th curvature to 'RoundPath'
    
      LocalPath = point(A,0)---point(A,1);        // staight subpath from 0-th edge to 1st edge
      LocalTime = intersect(LocalPath, LocalCirc); // calculate intersection-time between straight subpath and circle
      LocalPair = point(subpath(LocalPath, 0, LocalTime.x), 1); // define intersectionpoint between both paths  
      RoundPath = RoundPath..scale(S)*LocalPair--cycle; // add curvature near 0-th edge to 'RoundPath' and close path
    }
  else
    RoundPath = RoundPath--scale(S)*point(A,len);
  return RoundPath;
}
