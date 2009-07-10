settings.tex="context";
// Work around ConTeXT bug for font sizes less than 12pt:
texpreamble("\setupbodyfont[8pt]"); 

usetypescript("iwona");
usetypescript("antykwa-torunska");

label("$A$",0,N,font("iwona"));
label("$A$",0,S,font("antykwa",8pt)+red);

