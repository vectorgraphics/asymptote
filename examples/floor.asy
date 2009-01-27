import graph;
unitsize(1cm);

real Floor(real x) {return floor(x);}

pair[] Open;
pair[] Close;

bool3 branch(real x) {
  static real last;
  static bool first=true;
  real current=floor(x);
  bool samebranch=first || last == current; 
  first=false;
  if(samebranch) last=x;
  else {
    Close.push((x,last));
    Open.push((x,current));
  }
  last=current;
  return samebranch ? true : default;
};

draw(graph(Floor,-5.5,5.5,500,branch)); 
axes("$x$","$\lfloor x\rfloor$",red);

write(Close);
write(Open);

dot(Close);
dot(Open,UnFill);
