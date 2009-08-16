import graph;
usepackage("ocg");
settings.tex="pdflatex";

// Dan Bruton algorithm
pen nm2rgb(real wl, real gamma=0.8, bool intensity=true) {
  triple rgb;
  if(wl >= 380 && wl <= 440) {rgb=((440-wl)/60,0,1);}
  if(wl >  440 && wl <= 490) {rgb=(0,(wl-440)/50,1);}
  if(wl >  490 && wl <= 510) {rgb=(0,1,(510-wl)/20);}
  if(wl >  510 && wl <= 580) {rgb=((wl-510)/70,1,0);}
  if(wl >  580 && wl <= 645) {rgb=(1,(645-wl)/65,0);}
  if(wl >  645 && wl <= 780) {rgb=(1,0,0);}
  
  real Intensity=1;
  if(intensity) {
    if(wl >= 700) {Intensity=0.3+0.7*(780-wl)/80;}
    else if(wl <= 420) {Intensity=0.3+0.7*(wl-380)/40;}
  }

  return rgb((Intensity*rgb.x)**gamma,(Intensity*rgb.y)**gamma,
             (Intensity*rgb.z)**gamma);
}

real width=1;
real height=50;

begin("spectrum");
for(real i=380 ; i <= 780 ; i += width) {
  draw((i,0)--(i,height),width+nm2rgb(wl=i,false)+squarecap);
}
begin("Extinction",false); // nested
for(real i=380 ; i <= 780 ; i += width) {
  draw((i,0)--(i,height),width+nm2rgb(wl=i,true)+squarecap);
}
end();
end();

begin("Wavelength");
xaxis(scale(0.5)*"$\lambda$(nm)",BottomTop,380,780,
      RightTicks(scale(0.5)*rotate(90)*Label(),step=2,Step=10),above=true);
end();

// From Astronomical Data Center(NASA)
// Neutral only
real[] Na={423.899, 424.208, 427.364, 427.679, 428.784, 429.101,
           432.14, 432.462, 434.149, 434.474, 439.003, 439.334, 441.989, 442.325,
           449.418, 449.766, 454.163, 454.519, 568.2633, 568.8204, 588.995,
           589.5924};
begin("Na absorption");
for(int i=0; i < Na.length; ++i) {
  draw((Na[i],0)--(Na[i],height),0.1*width+squarecap);
}
end();

begin("Na emission");
for(int i=0; i < Na.length; ++i) {
  draw((Na[i],0)--(Na[i],-height),0.1*width+nm2rgb(Na[i],false)+squarecap);
}
end();

// Neutral only
real[] Zn={388.334, 396.543, 411.321, 429.288, 429.833, 462.981,
           468.014, 472.215, 481.053 , 506.866, 506.958, 518.198, 530.865,
           531.024, 531.102, 577.21, 577.55, 577.711, 623.79, 623.917, 636.234,
           647.918, 692.832, 693.847, 694.32, 779.936};
begin("Zn absorption",false);
for(int i=0; i < Zn.length; ++i) {
  draw((Zn[i],0)--(Zn[i],height),width+squarecap);
}
end();

begin("Zn emission",false);
for(int i=0; i < Zn.length; ++i) {
  draw((Zn[i],0)--(Zn[i],-height),width+nm2rgb(Zn[i],false)+squarecap);
}
end();

shipout(bbox(2mm,Fill(white)));
