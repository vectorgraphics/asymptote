access settings;
if(!settings.multipleView)
 settings.batchView=false;
settings.tex="pdflatex";
settings.inlinetex=true;
deletepreamble();

// Beginning of Asymptote Figure 1
eval(quote{
defaultfilename='inlinemovie_1';

import animate;
  animation A=animation("movie1");
  real h=2pi/10;

  picture pic;
  unitsize(pic,2cm);
  for (int i=0; i < 10; ++i){
    draw(pic,expi(i*h)--expi((i+1)*h));
    A.add(pic);
  }
  label(A.pdf("controls"));
});
// End of Asymptote Figure 1

// Beginning of Asymptote Figure 2
eval(quote{
defaultfilename='inlinemovie_2';

import animate;
  animation A=animation("movie2");
  real h=2pi/10;

  picture pic;
  unitsize(pic,2cm);
  for (int i=0; i < 10; ++i){
    draw(pic,expi(-i*h)--expi(-(i+1)*h),red);
    A.add(pic);
  }
  label(A.pdf("controls"));
});
// End of Asymptote Figure 2
