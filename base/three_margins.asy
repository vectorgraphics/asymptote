struct marginT3 {
  path3 g;
  real begin,end;
};

typedef marginT3 margin3(path3, pen);

path3 trim(path3 g, real begin, real end) {
  real a=arctime(g,begin);
  real b=arctime(g,arclength(g)-end);
  return a <= b ? subpath(g,a,b) : point(g,a);
}

margin3 operator +(margin3 ma, margin3 mb)
{
  return new marginT3(path3 g, pen p) {
    marginT3 margin;
    real ba=ma(g,p).begin < 0 ? 0 : ma(g,p).begin;
    real bb=mb(g,p).begin < 0 ? 0 : mb(g,p).begin;
    real ea=ma(g,p).end < 0   ? 0 : ma(g,p).end;
    real eb=mb(g,p).end < 0   ? 0 : mb(g,p).end;
    margin.begin=ba+bb;
    margin.end=ea+eb;
    margin.g=trim(g,margin.begin,margin.end);
    return margin;
  };
}

margin3 NoMargin3()
{ 
  return new marginT3(path3 g, pen) {
    marginT3 margin;
    margin.begin=margin.end=0;
    margin.g=g;
    return margin;
  };
}
                                                      
margin3 Margin3(real begin, real end)
{ 
  return new marginT3(path3 g, pen p) {
    marginT3 margin;
    real factor=labelmargin(p);
    real w=0.5*linewidth(p);
    margin.begin=begin*factor-w;
    margin.end=end*factor-w;
    margin.g=trim(g,margin.begin,margin.end);
    return margin;
  };
}
                                                           
margin3 PenMargin3(real begin, real end)
{ 
  return new marginT3(path3 g, pen p) {
    marginT3 margin;
    real factor=linewidth(p);
    margin.begin=begin*factor;
    margin.end=end*factor;
    margin.g=trim(g,margin.begin,margin.end);
    return margin;
  };
}
                                              
margin3 DotMargin3(real begin, real end)
{ 
  return new marginT3(path3 g, pen p) {
    marginT3 margin;
    real margindot(real x) {return x > 0 ? dotfactor*x : x;}
    real factor=linewidth(p);
    margin.begin=margindot(begin)*factor;
    margin.end=margindot(end)*factor;
    margin.g=trim(g,margin.begin,margin.end);
    return margin;
  };
}
                                                      
margin3 TrueMargin3(real begin, real end)
{ 
  return new marginT3(path3 g, pen p) {
    marginT3 margin;
    margin.begin=begin;
    margin.end=end;
    margin.g=trim(g,begin,end);
    return margin;
  };
}
                                                      
margin3 NoMargin3=NoMargin3(),
  BeginMargin3=Margin3(1,0),
  Margin3=Margin3(0,1),
  EndMargin3=Margin3,
  Margins3=Margin3(1,1),
  BeginPenMargin3=PenMargin3(0.5,-0.5),
  BeginPenMargin2=PenMargin3(1.0,-0.5),
  PenMargin3=PenMargin3(-0.5,0.5),
  PenMargin2=PenMargin3(-0.5,1.0),
  EndPenMargin3=PenMargin3,
  EndPenMargin2=PenMargin2,
  PenMargins3=PenMargin3(0.5,0.5),
  PenMargins2=PenMargin3(1.0,1.0),
  BeginDotMargin3=DotMargin3(0.5,-0.5),
  DotMargin3=DotMargin3(-0.5,0.5),
  EndDotMargin3=DotMargin3,
  DotMargins3=DotMargin3(0.5,0.5);
