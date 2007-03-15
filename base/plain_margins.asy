struct marginT {
  path g;
  real begin,end;
};

typedef marginT margin(path, pen);

path trim(path g, real begin, real end) {
  real a=arctime(g,begin);
  real b=arctime(g,arclength(g)-end);
  return a <= b ? subpath(g,a,b) : point(g,a);
}

margin operator +(margin ma, margin mb)
{
  return new marginT(path g, pen p) {
    marginT margin;
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

margin NoMargin()
{ 
  return new marginT(path g, pen) {
    marginT margin;
    margin.begin=margin.end=0;
    margin.g=g;
    return margin;
  };
}
                                                      
margin Margin(real begin, real end)
{ 
  return new marginT(path g, pen p) {
    marginT margin;
    real factor=labelmargin(p);
    margin.begin=begin*factor;
    margin.end=end*factor;
    margin.g=trim(g,margin.begin,margin.end);
    return margin;
  };
}
                                                           
margin PenMargin(real begin, real end)
{ 
  return new marginT(path g, pen p) {
    marginT margin;
    real factor=linewidth(p);
    margin.begin=(begin+0.5)*factor;
    margin.end=(end+0.5)*factor;
    margin.g=trim(g,margin.begin,margin.end);
    return margin;
  };
}
                                              
margin DotMargin(real begin, real end)
{ 
  return new marginT(path g, pen p) {
    marginT margin;
    real margindot(real x) {return x > 0 ? dotfactor*x : x;}
    real factor=linewidth(p);
    margin.begin=(margindot(begin)+0.5)*factor;
    margin.end=(margindot(end)+0.5)*factor;
    margin.g=trim(g,margin.begin,margin.end);
    return margin;
  };
}
                                                      
margin TrueMargin(real begin, real end)
{ 
  return new marginT(path g, pen p) {
    marginT margin;
    margin.begin=begin;
    margin.end=end;
    margin.g=trim(g,begin,end);
    return margin;
  };
}
                                                      
margin NoMargin=NoMargin(),
  BeginMargin=Margin(1,0),
  Margin=Margin(0,1),
  EndMargin=Margin,
  Margins=Margin(1,1),
  BeginPenMargin=PenMargin(0.5,-0.5),
  PenMargin=PenMargin(-0.5,0.5),
  EndPenMargin=PenMargin,
  PenMargins=PenMargin(0.5,0.5),
  BeginDotMargin=DotMargin(0.5,-0.5),
  DotMargin=DotMargin(-0.5,0.5),
  EndDotMargin=DotMargin,
  DotMargins=DotMargin(0.5,0.5);
