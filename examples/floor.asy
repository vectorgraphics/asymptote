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
  bool samebranch=first || lastint == currint; 
  first=false;
  if(samebranch) last=(x,currint);
  else {
    open.push((x,currint));
    close.push(last);
  }
  lastint=currint;
  return samebranch ? true : default;
};

draw(graph(Floor,-5.5,5.5,1000,branch)); 
axes("$x$","$y$",red);

dot(close);
dot(open,UnFill);
