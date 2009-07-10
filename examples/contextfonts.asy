settings.tex="context";
// Work around ConTeXT bug for font sizes less than 12pt:
//texpreamble("\setupbodyfont[10pt]"); 

usetypescript("iwona");
usetypescript("antykwa-torunska");

label("$A$",0,N,font("iwona"));
label("$A$",0,S,font("antykwa",10pt)+red);

