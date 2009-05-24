import graph;
unitsize(1cm);

real Floor(real x) {return floor(x);}

pair[] Close;
pair[] Open;

bool3 branch(real x) {
  static real lasty;
  static bool first=true;
  real y=floor(x);
  bool samebranch=first || lasty == y; 
  first=false;
  if(samebranch) lasty=x;
  else {
    Close.push((x,lasty));
    Open.push((x,y));
  }
  lasty=y;
  return samebranch ? true : default;
};

draw(graph(Floor,-5.5,5.5,500,branch)); 
axes("$x$",rotate(0)*"$\lfloor x\rfloor$",red);

dot(Close);
dot(Open,UnFill);
