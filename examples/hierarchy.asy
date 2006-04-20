texpreamble("\def\Ham{\mathop {\rm Ham}\nolimits}");
pair align=2N;
frame f;
ellipse(f,Label("$\Ham(r,2)$",(0,0)),lightblue,Fill,Below);
ellipse(f,Label("BCH Codes",point(f,N),align),green,Fill,Below);
ellipse(f,Label("Cyclic Codes",point(f,N),align),lightmagenta,Fill,Below);
ellipse(f,Label("Linear Codes",point(f,N),align),-4mm,orange,Fill,Below);
box(f,Label("General Codes",point(f,N),align),2mm,yellow,Fill,Below);
add(f);
