/*****
 * pen.h
 * John Bowman 2003/03/23
 *
 *****/

#ifndef PEN_H
#define PEN_H

#include <string>
#include "pool.h"
#include "transform.h"
#include "settings.h"
#include "bbox.h"

namespace camp {
  class pen;
}

namespace settings {
  extern camp::pen defaultpen;
}

namespace camp {

static const std::string DEFPAT="<default>";
static const std::string DEFFONT="\\usefont{OT1}{cmr}{m}{n}";
static const double DEFWIDTH=-1;
static const int DEFCAP=-1;
static const int DEFJOIN=-1;
  
static const struct invisiblepen_t {} invisiblepen={};
static const struct setlinewidth_t {} setlinewidth={};
static const struct setfont_t {} setfont={};
static const struct setfontsize_t {} setfontsize={};
static const struct setpattern_t {} setpattern={};
static const struct setlinecap_t {} setlinecap={};
static const struct setlinejoin_t {} setlinejoin={};
static const struct setoverwrite_t {} setoverwrite={};
static const struct initialpen_t {} initialpen={};
static const struct resolvepen_t {} resolvepen={};
  
static const std::string Cap[]={"square","round","extended"};
static const std::string Join[]={"miter","round","bevel"};
const int nCap=sizeof(Cap)/sizeof(std::string);
const int nJoin=sizeof(Join)/sizeof(std::string);
  
enum overwrite_t {DEFWRITE=-1,ALLOW,SUPPRESS,SUPPRESSQUIET,MOVE,MOVEQUIET};
static const std::string OverwriteTag[]={"Allow","Suppress","SupressQuiet",
					 "Move","MoveQuiet"};
const int nOverwrite=sizeof(OverwriteTag)/sizeof(std::string);
  
enum FillRule {DEFFILL=-1,ZEROWINDING,EVENODD};
static const std::string FillRuleTag[]={"ZeroWinding","EvenOdd"};
const int nFill=sizeof(FillRuleTag)/sizeof(std::string);
  
enum BaseLine {DEFBASE=-1,NOBASEALIGN,BASEALIGN};
static const std::string BaseLineTag[]={"NoAlign","Align"};
const int nBaseLine=sizeof(BaseLineTag)/sizeof(std::string);
  
enum ColorSpace {DEFCOLOR=0,INVISIBLE,GRAYSCALE,RGB,CMYK,PATTERN};
static const int ColorComponents[]={0,0,1,3,4,0};
static const std::string ColorDeviceSuffix[]={"","","Gray","RGB","CMYK",""};
const int nColorSpace=sizeof(ColorDeviceSuffix)/sizeof(std::string);
  
using settings::defaultpen;
  
class LineType
{
public:  
  std::string pattern;	// The string for the PostScript style line pattern.
  bool scale;		// Scale the line type values by the pen width?
  
  LineType(std::string pattern, bool scale) : pattern(pattern), scale(scale) {}
};
  
static const LineType DEFLINE("default",true);
  
inline bool operator == (LineType a, LineType b) {
  return a.pattern == b.pattern && a.scale == b.scale;
}
  
class pen : public mempool::pooled<pen>
{ 
  LineType line;

  // Width of line, in PS units.
  double linewidth;
  std::string font;
  double fontsize;  
  double lineskip;  
  
  ColorSpace color;
  double r,g,b; 	// RGB or CMY value
  double grey; 		// grayscale or K value
  
  std::string pattern;	// The name of the user-defined fill/draw pattern.
  FillRule fillrule; 	// Zero winding-number (default) or even-odd rule
  BaseLine baseline;	// Align to TeX baseline?
  int linecap;
  int linejoin;
  overwrite_t overwrite;
  
  // The transformation applied to the pen nib for calligraphic effects.
  // Null means the identity transformation.
  const transform *t;
  
public:
  static double pos0(double x) {return x >= 0 ? x : 0;}
  
  void greyrange() {if(grey > 1.0) grey=1.0;}
  
  void rgbrange() {
    double sat=rgbsaturation();
    if(sat > 1.0) {
      double scale=1.0/sat;
      r *= scale;
      g *= scale;
      b *= scale;
    }
  }
  
  void cmykrange() {
    double sat=cmyksaturation();
    if(sat > 1.0) {
      double scale=1.0/sat;
      r *= scale;
      g *= scale;
      b *= scale;
      grey *= scale;
    }
  }
  
  pen() :
    line(DEFLINE), linewidth(DEFWIDTH),
    font(""), fontsize(0.0), lineskip(0.0), color(DEFCOLOR),
    r(defaultpen.r), g(defaultpen.g), b(defaultpen.b), grey(defaultpen.grey),
    pattern(DEFPAT), fillrule(DEFFILL), baseline(DEFBASE),
    linecap(DEFCAP), linejoin(DEFJOIN), overwrite(DEFWRITE), t(0) {}

  pen(const LineType& line, double linewidth,
      const std::string& font, double fontsize, double lineskip,
      ColorSpace color, double r, double g, double b,  double grey,
      const std::string& pattern, FillRule fillrule, BaseLine baseline,
      int linecap, int linejoin, overwrite_t overwrite, const transform *t) :
    line(line), linewidth(linewidth),
    font(font), fontsize(fontsize), lineskip(lineskip), color(color),
    r(r), g(g), b(b), grey(grey),
    pattern(pattern), fillrule(fillrule), baseline(baseline),
    linecap(linecap), linejoin(linejoin), overwrite(overwrite), t(t) {}
      
  pen(invisiblepen_t) : 
    line(DEFLINE), linewidth(DEFWIDTH),
    font(""), fontsize(0.0), lineskip(0.0), color(INVISIBLE),
    r(defaultpen.r), g(defaultpen.g), b(defaultpen.b), grey(defaultpen.grey),
    pattern(DEFPAT), fillrule(DEFFILL), baseline(DEFBASE),
    linecap(DEFCAP), linejoin(DEFJOIN), overwrite(DEFWRITE), t(0) {}
  
  pen(setlinewidth_t, double linewidth) : 
    line(DEFLINE), linewidth(linewidth),
    font(""), fontsize(0.0), lineskip(0.0), color(DEFCOLOR),
    r(defaultpen.r), g(defaultpen.g), b(defaultpen.b), grey(defaultpen.grey),
    pattern(DEFPAT), fillrule(DEFFILL), baseline(DEFBASE),
    linecap(DEFCAP), linejoin(DEFJOIN), overwrite(DEFWRITE), t(0) {}
  
  pen(const LineType& line) :
    line(line), linewidth(DEFWIDTH),
    font(""), fontsize(0.0), lineskip(0.0), color(DEFCOLOR),
    r(defaultpen.r), g(defaultpen.g), b(defaultpen.b), grey(defaultpen.grey),
    pattern(DEFPAT), fillrule(DEFFILL), baseline(DEFBASE),
    linecap(DEFCAP), linejoin(DEFJOIN), overwrite(DEFWRITE), t(0) {}
  
  pen(setfont_t, std::string font) :
    line(DEFLINE), linewidth(DEFWIDTH),
    font(font), fontsize(0.0), lineskip(0.0), color(DEFCOLOR),
    r(defaultpen.r), g(defaultpen.g), b(defaultpen.b), grey(defaultpen.grey),
    pattern(DEFPAT), fillrule(DEFFILL), baseline(DEFBASE),
    linecap(DEFCAP), linejoin(DEFJOIN), overwrite(DEFWRITE), t(0) {}
  
  pen(setfontsize_t, double fontsize, double lineskip) :
    line(DEFLINE), linewidth(DEFWIDTH),
    font(""), fontsize(fontsize), lineskip(lineskip), color(DEFCOLOR),
    r(defaultpen.r), g(defaultpen.g), b(defaultpen.b), grey(defaultpen.grey),
    pattern(DEFPAT), fillrule(DEFFILL), baseline(DEFBASE),
    linecap(DEFCAP), linejoin(DEFJOIN), overwrite(DEFWRITE), t(0) {}
  
  pen(setpattern_t, const std::string& pattern) :
    line(DEFLINE), linewidth(DEFWIDTH),
    font(""), fontsize(0.0), lineskip(0.0), color(PATTERN),
    r(defaultpen.r), g(defaultpen.g), b(defaultpen.b), grey(defaultpen.grey),
    pattern(pattern), fillrule(DEFFILL), baseline(DEFBASE),
    linecap(DEFCAP), linejoin(DEFJOIN), overwrite(DEFWRITE), t(0) {}
  
  pen(FillRule fillrule) :
    line(DEFLINE), linewidth(DEFWIDTH),
    font(""), fontsize(0.0), lineskip(0.0), color(DEFCOLOR),
    r(defaultpen.r), g(defaultpen.g), b(defaultpen.b), grey(defaultpen.grey),
    pattern(DEFPAT), fillrule(fillrule), baseline(DEFBASE),
    linecap(DEFCAP), linejoin(DEFJOIN), overwrite(DEFWRITE), t(0) {}
  
  pen(BaseLine baseline) :
    line(DEFLINE), linewidth(DEFWIDTH),
    font(""), fontsize(0.0), lineskip(0.0), color(DEFCOLOR),
    r(defaultpen.r), g(defaultpen.g), b(defaultpen.b), grey(defaultpen.grey),
    pattern(DEFPAT), fillrule(DEFFILL), baseline(baseline),
    linecap(DEFCAP), linejoin(DEFJOIN), overwrite(DEFWRITE), t(0) {}
  
  pen(setlinecap_t, int linecap) :
    line(DEFLINE), linewidth(DEFWIDTH),
    font(""), fontsize(0.0), lineskip(0.0), color(DEFCOLOR),
    r(defaultpen.r), g(defaultpen.g), b(defaultpen.b), grey(defaultpen.grey),
    pattern(DEFPAT), fillrule(DEFFILL), baseline(DEFBASE),
    linecap(linecap), linejoin(DEFJOIN), overwrite(DEFWRITE), t(0) {}
  
  pen(setlinejoin_t, int linejoin) :
    line(DEFLINE), linewidth(DEFWIDTH),
    font(""), fontsize(0.0), lineskip(0.0), color(DEFCOLOR),
    r(defaultpen.r), g(defaultpen.g), b(defaultpen.b), grey(defaultpen.grey),
    pattern(DEFPAT), fillrule(DEFFILL), baseline(DEFBASE),
    linecap(DEFCAP), linejoin(linejoin), overwrite(DEFWRITE), t(0) {}
  
  pen(setoverwrite_t, overwrite_t overwrite) :
    line(DEFLINE), linewidth(DEFWIDTH),
    font(""), fontsize(0.0), lineskip(0.0), color(DEFCOLOR),
    r(defaultpen.r), g(defaultpen.g), b(defaultpen.b), grey(defaultpen.grey),
    pattern(DEFPAT), fillrule(DEFFILL), baseline(DEFBASE),
    linecap(DEFCAP), linejoin(DEFJOIN), overwrite(overwrite), t(0) {}
  
  explicit pen(double grey) :
    line(DEFLINE), linewidth(DEFWIDTH),
    font(""), fontsize(0.0), lineskip(0.0), color(GRAYSCALE),
    r(0.0), g(0.0), b(0.0), grey(pos0(grey)),
    pattern(DEFPAT), fillrule(DEFFILL), baseline(DEFBASE),
    linecap(DEFCAP), linejoin(DEFJOIN), overwrite(DEFWRITE), t(0)
  {greyrange();}
  
  pen(double r, double g, double b) : 
    line(DEFLINE), linewidth(DEFWIDTH),
    font(""), fontsize(0.0), lineskip(0.0), color(RGB),
    r(pos0(r)), g(pos0(g)), b(pos0(b)),  grey(0.0), 
    pattern(DEFPAT), fillrule(DEFFILL), baseline(DEFBASE),
    linecap(DEFCAP), linejoin(DEFJOIN), overwrite(DEFWRITE), t(0)
  {rgbrange();}
  
  pen(double c, double m, double y, double k) :
    line(DEFLINE), linewidth(DEFWIDTH),
    font(""), fontsize(0.0), lineskip(0.0), color(CMYK),
    r(pos0(c)), g(pos0(m)), b(pos0(y)), grey(pos0(k)),
    pattern(DEFPAT), fillrule(DEFFILL), baseline(DEFBASE),
    linecap(DEFCAP), linejoin(DEFJOIN), overwrite(DEFWRITE), t(0)
  {cmykrange();}
  
  // Construct one pen from another, resolving defaults
  pen(resolvepen_t, const pen& p) : 
    line(LineType(p.stroke(),p.scalestroke())), linewidth(p.width()),
    font(p.Font()), fontsize(p.size()), lineskip(p.Lineskip()),
    color(p.colorspace()),
    r(p.red()), g(p.green()), b(p.blue()), grey(p.gray()),
    pattern(""), fillrule(p.Fillrule()), baseline(p.Baseline()),
    linecap(p.cap()), linejoin(p.join()),overwrite(p.Overwrite()), t(p.t) {}
  
  static pen startupdefaultpen() {
    return pen(LineType("",true),0.5,DEFFONT,12.0,12.0*1.2,GRAYSCALE,
	       0.0,0.0,0.0,0.0,"",ZEROWINDING,NOBASEALIGN,1,1,ALLOW,0);
  }
  
  pen(initialpen_t) : 
    line(DEFLINE), linewidth(-2.0),
    font("<invalid>"), fontsize(-1.0), lineskip(-1.0), color(GRAYSCALE),
    r(0.0), g(0.0), b(0.0), grey(0.0),
    pattern(DEFPAT), fillrule(DEFFILL), baseline(NOBASEALIGN),
    linecap(-2), linejoin(-2), overwrite(DEFWRITE), t(0) {}
  
  double width() const {
    return linewidth == DEFWIDTH ? defaultpen.linewidth : linewidth;
  }
  
  std::string Font() const {
    return font == "" ? defaultpen.font : font;
  }
  
  double size() const {
    return fontsize == 0.0 ? defaultpen.fontsize : fontsize;
  }
  
  double Lineskip() const {
    return lineskip == 0.0 ? defaultpen.lineskip : lineskip;
  }
  
  std::string stroke() const {
    return line == DEFLINE ? defaultpen.line.pattern : line.pattern;
  }
  
  bool scalestroke() const {
    return line.scale;
  }
  
  void setstroke(const std::string& s) {line.pattern=s;}
  
  std::string fillpattern() const {
    return pattern == DEFPAT ? "" : pattern;
  }
  
  FillRule Fillrule() const {
    return fillrule == DEFFILL ? defaultpen.fillrule : fillrule;
  }
  
  BaseLine Baseline() const {
    return baseline == DEFBASE ? defaultpen.baseline : baseline;
  }
  
  int cap() const {
    return linecap == DEFCAP ? defaultpen.linecap : linecap;
  }
  
  int join() const {
    return linejoin == DEFJOIN ? defaultpen.linejoin : linejoin;
  }
  
  overwrite_t Overwrite() const {
    return overwrite == DEFWRITE ? defaultpen.overwrite : overwrite;
  }
  
  ColorSpace colorspace() const {
    return color == DEFCOLOR ? defaultpen.color : color;
  }
  
  bool invisible() const {return colorspace() == INVISIBLE;}
  
  bool grayscale() const {return colorspace() == GRAYSCALE;}
  
  bool rgb() const {return colorspace() == RGB;}
  
  bool cmyk() const {return colorspace() == CMYK;}
  
  double gray() const {return color == DEFCOLOR ? defaultpen.grey : grey;}
  
  double red() const {return color == DEFCOLOR ? defaultpen.r : r;}
  
  double green() const {return color == DEFCOLOR ? defaultpen.g : g;}
  
  double blue() const {return color == DEFCOLOR ? defaultpen.b : b;}
  
  double cyan() const {return red();}
  
  double magenta() const {return green();}
  
  double yellow() const {return blue();}
  
  double black() const {return gray();}
  
  double rgbsaturation() const {return max(max(r,g),b);}
  
  double cmyksaturation() const {return max(rgbsaturation(),black());}
  
  void greytorgb() {
    r=g=b=grey; grey=0.0;
    color=RGB;
  }
  
  void rgbtogrey() {
    grey=0.299*r+0.587*g+0.114*b; // Standard YUV luminosity coefficients
    r=g=b=0.0;
    color=GRAYSCALE;
  }
  
  void greytocmyk() {
    grey=1.0-grey;
    color=CMYK;
  }
  
  void rgbtocmyk() {
    double sat=rgbsaturation();
    r=1.0-r;
    g=1.0-g;
    b=1.0-b;
    grey=1.0-sat;
    if(sat) {
      double scale=1.0/sat;
      r=(r-grey)*scale;
      g=(g-grey)*scale;
      b=(b-grey)*scale;
    }
    color=CMYK;
  }

  void cmyktorgb() {
    double sat=1.0-grey;
    r=(1.0-r)*sat;
    g=(1.0-g)*sat;
    b=(1.0-b)*sat;
    grey=0.0;
    color=RGB;
  }

  void cmyktogrey() {
    cmyktorgb();
    rgbtogrey();
  }
  
  void convert() {
    if(settings::grayonly || settings::bwonly) {
      if(rgb()) rgbtogrey();
      else if(cmyk()) cmyktogrey();
      if(settings::bwonly) {grey=(grey == 1.0) ? 1.0 : 0.0;}
    }
    else if(settings::rgbonly && cmyk()) cmyktorgb();
    else if(settings::cmykonly && rgb()) rgbtocmyk();
  }    
  
  friend pen operator * (double x, const pen& q) {
    pen p=q;
    if(x < 0.0) x = 0.0;
    switch(p.color) {
    case PATTERN:
    case INVISIBLE:
    case DEFCOLOR:
      break;
    case GRAYSCALE:
      {
	p.grey *= x;
	p.greyrange();
	break;
      }
      break;
    case RGB:
      {
	p.r *= x;
	p.g *= x;
	p.b *= x;
	p.rgbrange();
	break;
      }
    case CMYK:
      {
	p.r *= x;
	p.g *= x;
	p.b *= x;
	p.grey *= x;
	p.cmykrange();
	break;
      }
    }
    return p;
  }
  
  friend pen operator + (const pen& p, const pen& q) {
    double R=0.0,G=0.0,B=0.0,greyval=0.0;
    pen P=p;
    pen Q=q;
    
    if(P.color == PATTERN && P.pattern == "") P.color=DEFCOLOR;
    ColorSpace colorspace=(ColorSpace) max(P.color,Q.color);
    
  switch(colorspace) {
    case PATTERN:
    case INVISIBLE:
    case DEFCOLOR:
      break;
    case GRAYSCALE:
      {
	if(p.color != GRAYSCALE) greyval=q.grey;
	else {
	  if(q.color != GRAYSCALE) greyval=p.grey;
	  else greyval=0.5*(p.grey+q.grey);
	}
	break;
      }
      
    case RGB:
      {
	double sat;
	if (P.color == GRAYSCALE) {
	  P.greytorgb();
	  sat=0.5*(P.rgbsaturation()+Q.rgbsaturation());
	} else {
	  if (Q.color == GRAYSCALE) {
	    Q.greytorgb();
	    sat=0.5*(P.rgbsaturation()+Q.rgbsaturation());
	  } else sat=max(P.rgbsaturation(),Q.rgbsaturation());
	}
	// Mix colors
	P.r += Q.r;
	P.g += Q.g;
	P.b += Q.b;
	double newsat=P.rgbsaturation();
	double scale=newsat ? sat/newsat: 1.0;
	R=P.r*scale;
	G=P.g*scale;
	B=P.b*scale;
	break;
      }
      
    case CMYK:
      {
	if (P.color == GRAYSCALE) P.greytocmyk();
	else if (Q.color == GRAYSCALE) Q.greytocmyk();
	
	if (P.color == RGB) P.rgbtocmyk();
	else if (Q.color == RGB) Q.rgbtocmyk();
	
	double sat=max(P.cmyksaturation(),Q.cmyksaturation());      
	// Mix colors
	P.r += Q.r;
	P.g += Q.g;
	P.b += Q.b;
	P.grey += Q.grey;
	double newsat=P.cmyksaturation();
	double scale=newsat ? sat/newsat: 1.0;
	R=P.r*scale;
	G=P.g*scale;
	B=P.b*scale;
	greyval=P.grey*scale;
	break;
      }
    }
    
    return pen(q.line == DEFLINE ? p.line : q.line,
	       q.linewidth == DEFWIDTH ? p.linewidth : q.linewidth,
	       q.font == "" ? p.font : q.font,
	       q.fontsize == 0.0 ? p.fontsize : q.fontsize,
	       q.lineskip == 0.0 ? p.lineskip : q.lineskip,
	       colorspace,R,G,B,greyval,
	       q.pattern == DEFPAT ? p.pattern : q.pattern,
	       q.fillrule == DEFFILL ? p.fillrule : q.fillrule,
	       q.baseline == DEFBASE ? p.baseline : q.baseline,
	       q.linecap == DEFCAP ? p.linecap : q.linecap,
	       q.linejoin == DEFJOIN ? p.linejoin : q.linejoin,
	       q.overwrite == DEFWRITE ? p.overwrite : q.overwrite,
	       q.t == NULL ? p.t : q.t);
  }

  friend bool operator == (const pen& p, const pen& q) {
    return p.stroke() == q.stroke() 
      && p.scalestroke() == q.scalestroke() 
      && p.width() == q.width() 
      && p.colorspace() == q.colorspace()
      && p.Font() == q.Font()
      && p.Lineskip() == q.Lineskip()
      && p.size() == q.size()
      && (!(p.grayscale() || p.cmyk()) || p.gray() == q.gray())
      && (!(p.rgb() || p.cmyk()) || (p.red() == q.red() &&
				     p.green() == q.green() &&
				     p.blue() == q.blue()))
      && p.pattern == q.pattern
      && p.Fillrule() == q.Fillrule()
      && p.Baseline() == q.Baseline()
      && p.cap() == q.cap()
      && p.join() == q.join()
      && p.Overwrite() == q.Overwrite()
      && (p.t ? *p.t : identity()) == (q.t ? *q.t : identity());
  }
  
  friend ostream& operator << (ostream& out, const pen& p) {
    out << "([" << p.line.pattern << "]";
    if(!p.line.scale)
      out << " bp";
    if(p.linewidth != DEFWIDTH)
      out << ", linewidth=" << p.linewidth;
    if(p.linecap != DEFCAP)
      out << ", linecap=" << Cap[p.linecap];
    if(p.linejoin != DEFJOIN)
      out << ", linejoin=" << Join[p.linejoin];
    if(p.font != "")
      out << ", font=\"" << p.font << "\"";
    if(p.fontsize)
      out << ", fontsize=" << p.fontsize;
    if(p.lineskip)
      out << ", lineskip=" << p.lineskip;
    if(p.color == INVISIBLE)
      out << ", invisible";
    else if(p.color == GRAYSCALE)
      out << ", gray=" << p.grey;
    else if(p.color == RGB)
      out << ", red=" << p.red() << ", green=" << p.green() 
	  << ", blue=" << p.blue();
    else if(p.color == CMYK)
      out << ", cyan=" << p.cyan() << ", magenta=" << p.magenta() 
	  << ", yellow=" << p.yellow() << ", black=" << p.black();
    if(p.pattern != DEFPAT)
      out << ", " << "\"" << p.pattern << "\"";
    if(p.fillrule != DEFFILL)
      out << ", fillrule=" << FillRuleTag[p.fillrule];
    if(p.baseline != DEFBASE)
      out << ", baseline=" << BaseLineTag[p.baseline];
    if(p.overwrite != DEFWRITE)
      out << ", overwrite=" << OverwriteTag[p.overwrite];
    if(p.t)
      out << ", transform=" << *(p.t);
    out << ")";
    
    return out;
  }

  // The bounds of the circle or ellipse the pen describes.
  bbox bounds() const
  {
    double maxx, maxy;
    pair shift;

    if (t != 0) {
      double xx = t->getxx(), xy = t->getxy(),
             yx = t->getyx(), yy = t->getyy();

      // These are the maximum x and y values that a linear transform can map
      // a point in the unit circle.  This can be proven by the Lagrange
      // Multiplier theorem or by properties of the dot product.
      maxx = length(pair(xx,xy));
      maxy = length(pair(yx,yy));

      shift = *t*pair(0,0);
    }
    else {
      maxx = 1;
      maxy = 1;
      shift = pair(0,0);
    }

    bbox b;
    pair z = width() * pair(maxx,maxy);
    b += z + shift;
    b += -z + shift;

    return b;
  }

  friend pen transformed(const transform* t, const pen& p) {
    pen ret = p;
    ret.t = p.t ? new transform((*t)*(*p.t)) : t;
    return ret;
  }

  const transform *getTransform() const {
    return t;
  }
};
  
}

#endif
