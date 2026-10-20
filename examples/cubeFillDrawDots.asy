import three;

unitsize(3cm);

currentlight=nolight;
currentprojection=orthographic(dir(64,36-90),center=false);

string[] l6={"R","Y","G","C","B","M","$O$","W"};
pen[] c6={red,yellow,green,cyan,blue,magenta,black,white};
triple[] t6={
  (1,0,0),(1,1,0),(0,1,0),(0,1,1),(0,0,1),(1,0,1),(0,0,0),(1,1,1)
};
path3[] ph=new path3[6]; // polyhedron == cube
for(int i: sequence(6)) {
  int j=(i+1) % 6,k=(i+2) % 6,o=6+i % 2;
  ph[i]=t6[o]--t6[i]--t6[j]--t6[k]--cycle;
}
transform3 T=scale3(sqrt(3/2));
ph=T*ph;
t6=T*t6;
pair[] a6={S,S,S,N,W,N,S,N};
a6 *= 2.5; // scale for align

draw(Label("$R$",EndPoint),-2.5X--2.5X,Arrow3);
draw(Label("$G$",EndPoint),-2.5Y--2.5Y,Arrow3);
draw(Label("$B$",EndPoint),O--3Z,Arrow3);
pen p=black+1bp;
for(int i: sequence(6))
  draw(surface(ph[i]),white+opacity(0.6),p);

dotfactor=10;

for(int i: sequence(8))
  dot(Label(l6[i]),t6[i],p,FillDraw(c6[i]),align=a6[i]);
