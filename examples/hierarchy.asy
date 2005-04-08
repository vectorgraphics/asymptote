texpreamble("\def\Ham{\mathop {\rm Ham}\nolimits}");
pair align=2N;
frame f;
labelellipse(f,"$\Ham(r,2)$",(0,0));
labelellipse(f,"BCH Codes",point(f,N),align);
labelellipse(f,"Cyclic Codes",point(f,N),align);
labelellipse(f,-4mm,"Linear Codes",point(f,N),align);
labelbox(f,2mm,"General Codes",point(f,N),align);
add(f);

