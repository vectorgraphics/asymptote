import graph;
unitsize(1cm);

real Floor(real x) {return floor(x);}

pair[] open;
pair[] close;

bool3 branch(real x) {
  static int lastint;
  static bool first=true;
  static pair last;
  int currint=floor(x);
  pair z=(x,currint);
  bool samebranch=first || lastint == currint; 
  lastint=currint;
  first=false;
  if(samebranch) last=z;
  else {
    open.push(z);
    close.push(last);
  }
  return samebranch ? true : default;
};

draw(graph(Floor,-5.5,5.5,1000,branch)); 
axes("$x$","$\lfloor x\rfloor$",red);

dot(close);
dot(open,UnFill);
