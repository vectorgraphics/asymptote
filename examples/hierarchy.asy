texpreamble("\def\Ham{\mathop {\rm Ham}\nolimits}");
pair align=2N;
frame f;
labelellipse(f,"$\Ham(r,2)$",(0,0),black,lightblue,Fill);
labelellipse(f,"BCH Codes",point(f,N),align,black,green,Fill);
labelellipse(f,"Cyclic Codes",point(f,N),align,black,pink,Fill);
labelellipse(f,-4mm,"Linear Codes",point(f,N),align,black,orange,Fill);
labelbox(f,2mm,"General Codes",point(f,N),align,black,yellow,Fill);
add(f);
