import NURBS_to_Bezier;
size(10cm);
real[] knot={0,0,0,1,2,3,4,4,6,5,5};
triple[] P=
  {
    (-6,-1,0),
    (-5,2,0),
    (-3,3,0),
    (-1,2,0),
    (0,0,0),
    (3,1,0),
    (3,3,0),
    (1,5,0)
  };

triple[] testPoints={
  ( 4.0,-6.0,6.0),
  (-4.0,1.0,0.0),
  (-1.5,5.0,-6.0),
  ( 0.0,2.0,-2.0),
  ( 1.5,5.0,-6.0),
  ( 4.0,1.0,0.0),
  (-4.0,-6.0,6.0),
};
  triple[] testPoints ={
   (4,6,0),(7,12,0),(11,6,0),(15,2,0),(20,6,0)
   };
   real[] testKnots={0,0,0,0,0.5,1,1,1,1}; 
real[] testKnots={1,1,1,2,3,4,5,6,6,6};
real[] weights=array(testPoints.length,1.0);
weights[3]=0;
NURBScurve n=NURBScurve(testPoints,testKnots,weights);
n.draw();
dot(n.data.controlPoints,red);
draw(box(n.min(),n.max()),red);