size(500);
import graph3;

currentprojection=perspective(-5,-4,2);

path3 g=randompath3(10);

draw(g,red);

triple[][] P={
  {(0,0,0),(1,0,0),(1,0,0),(2,0,0)},
  {(0,4/3,0),(2/3,4/3,2),(4/3,4/3,2),(2,4/3,0)},
  {(0,2/3,0),(2/3,2/3,0),(4/3,2/3,0),(2,2/3,0)},
  {(0,2,0),(2/3,2,0),(4/3,2,0),(2,2,0)}};

surface s=surface(patch(P));
s.append(unitplane);

draw(s,lightgray+opacity(0.9));
dot(intersectionpoints(g,s),blue);
