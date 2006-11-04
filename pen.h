/*****
 * pen.h
 * John Bowman 2003/03/23
 *
 *****/

#ifndef PEN_H
#define PEN_H

#include "transform.h"
#include "settings.h"
#include "bbox.h"
#include "memory.h"
#include "path.h"

namespace camp {
  class pen;
}

namespace settings {
  extern camp::pen defaultpen;
}

namespace camp {

static const mem::string DEFPAT="<default>";
static const mem::string DEFLATEXFONT="\\usefont{OT1}{cmr}{m}{n}";
static const mem::string DEFTEXFONT="\\font\\ASYfont=cmr12\\ASYfont";
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
  
static const mem::string Cap[]={"square","round","extended"};
static const mem::string Join[]={"miter","round","bevel"};
const int nCap=sizeof(Cap)/sizeof(mem::string);
const int nJoin=sizeof(Join)/sizeof(mem::string);
  
enum overwrite_t {DEFWRITE=-1,ALLOW,SUPPRESS,SUPPRESSQUIET,MOVE,MOVEQUIET};
static const mem::string OverwriteTag[]={"Allow","Suppress","SupressQuiet",
					 "Move","MoveQuiet"};
const int nOverwrite=sizeof(OverwriteTag)/sizeof(mem::string);
  
enum FillRule {DEFFILL=-1,ZEROWINDING,EVENODD};
static const mem::string FillRuleTag[]={"ZeroWinding","EvenOdd"};

const int nFill=sizeof(FillRuleTag)/sizeof(mem::string);
  
enum BaseLine {DEFBASE=-1,NOBASEALIGN,BASEALIGN};
static const mem::string BaseLineTag[]={"NoAlign","Align"};
const int nBaseLine=sizeof(BaseLineTag)/sizeof(mem::string);
  
enum ColorSpace {DEFCOLOR=0,INVISIBLE,GRAYSCALE,RGB,CMYK,PATTERN};
extern const int ColorComponents[];
static const mem::string ColorDeviceSuffix[]={"","","Gray","RGB","CMYK",""};
const int nColorSpace=sizeof(ColorDeviceSuffix)/sizeof(mem::string);
  
using settings::defaultpen;
  
class LineType
{
public:  
  mem::string pattern;	// The string for the PostScript style line pattern.
  double offset;        // The offset in the pattern at which to start drawing.
  bool scale;		// Scale the line type values by the pen width?
  bool adjust;		// Adjust the line type values to fit the arclength?
  
  LineType(mem::string pattern, double offset, bool scale, bool adjust) : 
    pattern(pattern), offset(offset), scale(scale), adjust(adjust) {}
};
  
static const LineType DEFLINE("default",0,true,true);
  
inline bool operator == (LineType a, LineType b) {
  return a.pattern == b.pattern && a.offset == b.offset && 
    a.scale == b.scale && a.adjust == b.adjust;
}
  
class Transparency
{
public:  
  mem::string blend;
  double opacity;
  Transparency(mem::string blend, double opacity) :
    blend(blend), opacity(opacity) {}
};
  
static const Transparency DEFTRANSP("Compatible",1.0);
  
inline bool operator == (Transparency a, Transparency b) {
  return a.blend == b.blend && a.opacity == b.opacity;
}
  
static const mem::string BlendMode[]={"Compatible","Normal","Multiply","Screen",
				      "Overlay","SoftLight","HardLight",
				      "ColorDodge","ColorBurn","Darken",
				      "Lighten","Difference","Exclusion",
				      "Hue","Saturation","Color","Luminosity"};
const int nBlendMode=sizeof(BlendMode)/sizeof(mem::string);

  
class pen : public gc { 
  LineType line;

  // Width of line, in PS units.
  double linewidth;
  path *P;              // A polygonal path defining a custom pen nib
                        // NULL means the default circular nib.
  mem::string font;
  double fontsize;  
  double lineskip;  
  
  ColorSpace color;
  double r,g,b; 	// RGB or CMY value
  double grey; 		// grayscale or K value
  
  mem::string pattern;	// The name of the user-defined fill/draw pattern
  FillRule fillrule; 	// Zero winding-number (default) or even-odd rule
  BaseLine baseline;	// Align to TeX baseline?
  Transparency transparency;
  int linecap;
  int linejoin;
  overwrite_t overwrite;
  
  // The transformation applied to the pen nib for calligraphic effects.
  // NULL means the identity transformation.
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
  
  void colorless() {
    r=g=b=grey=0.0;
    color=DEFCOLOR;
  }
  
  pen() :
    line(DEFLINE), linewidth(DEFWIDTH), P(0),
    font(""), fontsize(0.0), lineskip(0.0), color(DEFCOLOR),
    r(0), g(0), b(0), grey(0),
    pattern(DEFPAT), fillrule(DEFFILL), baseline(DEFBASE),
    transparency(DEFTRANSP),
    linecap(DEFCAP), linejoin(DEFJOIN), overwrite(DEFWRITE), t(0) {}

  pen(const LineType& line, double linewidth, path *P,
      const mem::string& font, double fontsize, double lineskip,
      ColorSpace color, double r, double g, double b,  double grey,
      const mem::string& pattern, FillRule fillrule, BaseLine baseline,
      const Transparency& transparency,
      int linecap, int linejoin, overwrite_t overwrite, const transform *t) :
    line(line), linewidth(linewidth), P(P),
    font(font), fontsize(fontsize), lineskip(lineskip), color(color),
    r(r), g(g), b(b), grey(grey),
    pattern(pattern), fillrule(fillrule), baseline(baseline),
    transparency(transparency),
    linecap(linecap), linejoin(linejoin), overwrite(overwrite), t(t) {}
      
  pen(invisiblepen_t) : 
    line(DEFLINE), linewidth(DEFWIDTH), P(0),
    font(""), fontsize(0.0), lineskip(0.0), color(INVISIBLE),
    r(0), g(0), b(0), grey(0),
    pattern(DEFPAT), fillrule(DEFFILL), baseline(DEFBASE),
    transparency(DEFTRANSP),
    linecap(DEFCAP), linejoin(DEFJOIN), overwrite(DEFWRITE), t(0) {}
  
  pen(setlinewidth_t, double linewidth) : 
    line(DEFLINE), linewidth(linewidth), P(0),
    font(""), fontsize(0.0), lineskip(0.0), color(DEFCOLOR),
    r(0), g(0), b(0), grey(0),
    pattern(DEFPAT), fillrule(DEFFILL), baseline(DEFBASE),
    transparency(DEFTRANSP),
    linecap(DEFCAP), linejoin(DEFJOIN), overwrite(DEFWRITE), t(0) {}
  
  pen(path *P) : 
    line(DEFLINE), linewidth(DEFWIDTH), P(P),
    font(""), fontsize(0.0), lineskip(0.0), color(DEFCOLOR),
    r(0), g(0), b(0), grey(0),
    pattern(DEFPAT), fillrule(DEFFILL), baseline(DEFBASE),
    transparency(DEFTRANSP),
    linecap(DEFCAP), linejoin(DEFJOIN), overwrite(DEFWRITE), t(0) {}
  
  pen(const LineType& line) :
    line(line), linewidth(DEFWIDTH), P(0),
    font(""), fontsize(0.0), lineskip(0.0), color(DEFCOLOR),
    r(0), g(0), b(0), grey(0),
    pattern(DEFPAT), fillrule(DEFFILL), baseline(DEFBASE),
    transparency(DEFTRANSP),
    linecap(DEFCAP), linejoin(DEFJOIN), overwrite(DEFWRITE), t(0) {}
  
  pen(setfont_t, mem::string font) :
    line(DEFLINE), linewidth(DEFWIDTH), P(0),
    font(font), fontsize(0.0), lineskip(0.0), color(DEFCOLOR),
    r(0), g(0), b(0), grey(0),
    pattern(DEFPAT), fillrule(DEFFILL), baseline(DEFBASE),
    transparency(DEFTRANSP),
    linecap(DEFCAP), linejoin(DEFJOIN), overwrite(DEFWRITE), t(0) {}
  
  pen(setfontsize_t, double fontsize, double lineskip) :
    line(DEFLINE), linewidth(DEFWIDTH), P(0),
    font(""), fontsize(fontsize), lineskip(lineskip), color(DEFCOLOR),
    r(0), g(0), b(0), grey(0),
    pattern(DEFPAT), fillrule(DEFFILL), baseline(DEFBASE),
    transparency(DEFTRANSP),
    linecap(DEFCAP), linejoin(DEFJOIN), overwrite(DEFWRITE), t(0) {}
  
  pen(setpattern_t, const mem::string& pattern) :
    line(DEFLINE), linewidth(DEFWIDTH), P(0),
    font(""), fontsize(0.0), lineskip(0.0), color(PATTERN),
    r(0), g(0), b(0), grey(0),
    pattern(pattern), fillrule(DEFFILL), baseline(DEFBASE),
    transparency(DEFTRANSP),
    linecap(DEFCAP), linejoin(DEFJOIN), overwrite(DEFWRITE), t(0) {}
  
  pen(FillRule fillrule) :
    line(DEFLINE), linewidth(DEFWIDTH), P(0),
    font(""), fontsize(0.0), lineskip(0.0), color(DEFCOLOR),
    r(0), g(0), b(0), grey(0),
    pattern(DEFPAT), fillrule(fillrule), baseline(DEFBASE),
    transparency(DEFTRANSP),
    linecap(DEFCAP), linejoin(DEFJOIN), overwrite(DEFWRITE), t(0) {}
  
  pen(BaseLine baseline) :
    line(DEFLINE), linewidth(DEFWIDTH), P(0),
    font(""), fontsize(0.0), lineskip(0.0), color(DEFCOLOR),
    r(0), g(0), b(0), grey(0),
    pattern(DEFPAT), fillrule(DEFFILL), baseline(baseline),
    transparency(DEFTRANSP),
    linecap(DEFCAP), linejoin(DEFJOIN), overwrite(DEFWRITE), t(0) {}
  
  pen(const Transparency& transparency) :
    line(DEFLINE), linewidth(DEFWIDTH), P(0),
    font(""), fontsize(0.0), lineskip(0.0), color(DEFCOLOR),
    r(0), g(0), b(0), grey(0),
    pattern(DEFPAT), fillrule(DEFFILL), baseline(DEFBASE),
    transparency(transparency),
    linecap(DEFCAP), linejoin(DEFJOIN), overwrite(DEFWRITE), t(0) {}
  
  pen(setlinecap_t, int linecap) :
    line(DEFLINE), linewidth(DEFWIDTH), P(0),
    font(""), fontsize(0.0), lineskip(0.0), color(DEFCOLOR),
    r(0), g(0), b(0), grey(0),
    pattern(DEFPAT), fillrule(DEFFILL), baseline(DEFBASE),
    transparency(DEFTRANSP),
    linecap(linecap), linejoin(DEFJOIN), overwrite(DEFWRITE), t(0) {}
  
  pen(setlinejoin_t, int linejoin) :
    line(DEFLINE), linewidth(DEFWIDTH), P(0),
    font(""), fontsize(0.0), lineskip(0.0), color(DEFCOLOR),
    r(0), g(0), b(0), grey(0),
    pattern(DEFPAT), fillrule(DEFFILL), baseline(DEFBASE),
    transparency(DEFTRANSP),
    linecap(DEFCAP), linejoin(linejoin), overwrite(DEFWRITE), t(0) {}
  
  pen(setoverwrite_t, overwrite_t overwrite) :
    line(DEFLINE), linewidth(DEFWIDTH), P(0),
    font(""), fontsize(0.0), lineskip(0.0), color(DEFCOLOR),
    r(0), g(0), b(0), grey(0),
    pattern(DEFPAT), fillrule(DEFFILL), baseline(DEFBASE),
    transparency(DEFTRANSP),
    linecap(DEFCAP), linejoin(DEFJOIN), overwrite(overwrite), t(0) {}
  
  explicit pen(double grey) :
    line(DEFLINE), linewidth(DEFWIDTH), P(0),
    font(""), fontsize(0.0), lineskip(0.0), color(GRAYSCALE),
    r(0.0), g(0.0), b(0.0), grey(pos0(grey)),
    pattern(DEFPAT), fillrule(DEFFILL), baseline(DEFBASE),
    transparency(DEFTRANSP),
    linecap(DEFCAP), linejoin(DEFJOIN), overwrite(DEFWRITE), t(0)
  {greyrange();}
  
  pen(double r, double g, double b) : 
    line(DEFLINE), linewidth(DEFWIDTH), P(0),
    font(""), fontsize(0.0), lineskip(0.0), color(RGB),
    r(pos0(r)), g(pos0(g)), b(pos0(b)),  grey(0.0), 
    pattern(DEFPAT), fillrule(DEFFILL), baseline(DEFBASE),
    transparency(DEFTRANSP),
    linecap(DEFCAP), linejoin(DEFJOIN), overwrite(DEFWRITE), t(0)
  {rgbrange();}
  
  pen(double c, double m, double y, double k) :
    line(DEFLINE), linewidth(DEFWIDTH), P(0),
    font(""), fontsize(0.0), lineskip(0.0), color(CMYK),
    r(pos0(c)), g(pos0(m)), b(pos0(y)), grey(pos0(k)),
    pattern(DEFPAT), fillrule(DEFFILL), baseline(DEFBASE),
    transparency(DEFTRANSP),
    linecap(DEFCAP), linejoin(DEFJOIN), overwrite(DEFWRITE), t(0)
  {cmykrange();}
  
  // Construct one pen from another, resolving defaults
  pen(resolvepen_t, const pen& p) : 
    line(LineType(p.stroke(),p.line.offset,p.line.scale,p.line.adjust)),
    linewidth(p.width()), P(p.Path()),
    font(p.Font()), fontsize(p.size()), lineskip(p.Lineskip()),
    color(p.colorspace()),
    r(p.red()), g(p.green()), b(p.blue()), grey(p.gray()),
    pattern(""), fillrule(p.Fillrule()), baseline(p.Baseline()),
    transparency(Transparency(p.blend(),p.opacity())),
    linecap(p.cap()), linejoin(p.join()),overwrite(p.Overwrite()), t(p.t) {}
  
  static pen startupdefaultpen() {
    return pen(LineType("",0,true,true),0.5,0,"",12.0,12.0*1.2,
	       GRAYSCALE,
	       0.0,0.0,0.0,0.0,"",ZEROWINDING,NOBASEALIGN,
	       DEFTRANSP,1,1,ALLOW,0);
  }
  
  pen(initialpen_t) : 
    line(DEFLINE), linewidth(-2.0), P(0),
    font("<invalid>"), fontsize(-1.0), lineskip(-1.0), color(INVISIBLE),
    r(0.0), g(0.0), b(0.0), grey(0.0),
    pattern(DEFPAT), fillrule(DEFFILL), baseline(NOBASEALIGN),
    transparency(DEFTRANSP),linecap(-2), linejoin(-2), overwrite(DEFWRITE),
    t(0) {}
  
  double width() const {
    return linewidth == DEFWIDTH ? defaultpen.linewidth : linewidth;
  }
  
  path *Path() const {
    return P == NULL ? defaultpen.P : P;
  }
  
  mem::string Font() const {
    if(font.empty()) {
      if(defaultpen.font.empty())
	return settings::latex(settings::getSetting<mem::string>("tex")) ? 
	  DEFLATEXFONT : DEFTEXFONT;
      else return defaultpen.font;
    }
    return font;
  }
  
  double size() const {
    return fontsize == 0.0 ? defaultpen.fontsize : fontsize;
  }
  
  double Lineskip() const {
    return lineskip == 0.0 ? defaultpen.lineskip : lineskip;
  }
  
  mem::string stroke() const {
    return line == DEFLINE ? defaultpen.line.pattern : line.pattern;
  }
  
  LineType linetype() const {
    return line == DEFLINE ? defaultpen.line : line;
  }
  
  void setstroke(const mem::string& s) {line.pattern=s;}
  void setoffset(const double& offset) {line.offset=offset;}
  
  mem::string fillpattern() const {
    return pattern == DEFPAT ? (mem::string)"" : pattern;
  }
  
  FillRule Fillrule() const {
    return fillrule == DEFFILL ? defaultpen.fillrule : fillrule;
  }
  
  bool evenodd() const {
    return fillrule == EVENODD;
  }
  
  bool inside(int count) const {
    return evenodd() ? count % 2 : count != 0;
  }
  
  BaseLine Baseline() const {
    return baseline == DEFBASE ? defaultpen.baseline : baseline;
  }
  
  Transparency transp() const {
    return transparency == DEFTRANSP ? defaultpen.transparency : transparency;
  }
  
  mem::string blend() const {
    return transparency == DEFTRANSP ? defaultpen.transparency.blend :
      transparency.blend;
  }
  
  double opacity() const {
    return transparency == DEFTRANSP ? defaultpen.transparency.opacity :
      transparency.opacity;
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
    if (settings::getSetting<bool>("gray") ||
        settings::getSetting<bool>("bw")) {
      if(rgb()) rgbtogrey();
      else if(cmyk()) cmyktogrey();
      if(settings::getSetting<bool>("bw")) {grey=(grey == 1.0) ? 1.0 : 0.0;}
    }
    else if(settings::getSetting<bool>("rgb") && cmyk()) cmyktorgb();
    else if(settings::getSetting<bool>("cmyk") && rgb()) rgbtocmyk();
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
    
    if(P.color == PATTERN && P.pattern.empty()) P.color=DEFCOLOR;
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
	       q.P == NULL ? p.P : q.P,
	       q.font.empty() ? p.font : q.font,
	       q.fontsize == 0.0 ? p.fontsize : q.fontsize,
	       q.lineskip == 0.0 ? p.lineskip : q.lineskip,
	       colorspace,R,G,B,greyval,
	       q.pattern == DEFPAT ? p.pattern : q.pattern,
	       q.fillrule == DEFFILL ? p.fillrule : q.fillrule,
	       q.baseline == DEFBASE ? p.baseline : q.baseline,
	       q.transparency == DEFTRANSP ? p.transparency : q.transparency,
	       q.linecap == DEFCAP ? p.linecap : q.linecap,
	       q.linejoin == DEFJOIN ? p.linejoin : q.linejoin,
	       q.overwrite == DEFWRITE ? p.overwrite : q.overwrite,
	       q.t == NULL ? p.t : q.t);
  }

  friend bool operator == (const pen& p, const pen& q) {
    return  p.linetype() == q.linetype() 
      && p.width() == q.width() 
      && p.Path() == q.Path()
      && p.Font() == q.Font()
      && p.Lineskip() == q.Lineskip()
      && p.size() == q.size()
      && p.colorspace() == q.colorspace()
      && (!(p.grayscale() || p.cmyk()) || p.gray() == q.gray())
      && (!(p.rgb() || p.cmyk()) || (p.red() == q.red() &&
				     p.green() == q.green() &&
				     p.blue() == q.blue()))
      && p.pattern == q.pattern
      && p.Fillrule() == q.Fillrule()
      && p.Baseline() == q.Baseline()
      && p.transp() == q.transp()
      && p.cap() == q.cap()
      && p.join() == q.join()
      && p.Overwrite() == q.Overwrite()
      && (p.t ? *p.t : identity()) == (q.t ? *q.t : identity());
  }
  
  friend bool operator != (const pen& p, const pen& q) {
    return !(p == q);
  }
  
  friend ostream& operator << (ostream& out, const pen& p) {
    out << "([" << p.line.pattern << "]";
    if(p.line.offset)
      out << p.line.offset;
    if(!p.line.scale)
      out << " bp";
    if(!p.line.adjust)
      out << " fixed";
    if(p.linewidth != DEFWIDTH)
      out << ", linewidth=" << p.linewidth;
    if(p.P)
      out << ", path=" << *p.P;
    if(p.linecap != DEFCAP)
      out << ", linecap=" << Cap[p.linecap];
    if(p.linejoin != DEFJOIN)
      out << ", linejoin=" << Join[p.linejoin];
    if(!p.font.empty())
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
      out << ", pattern=" << "\"" << p.pattern << "\"";
    if(p.fillrule != DEFFILL)
      out << ", fillrule=" << FillRuleTag[p.fillrule];
    if(p.baseline != DEFBASE)
      out << ", baseline=" << BaseLineTag[p.baseline];
    if(!(p.transparency == DEFTRANSP)) {
      out << ", opacity=" << p.transparency.opacity;
      out << ", blend=" << p.transparency.blend;
    }
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

    if(P) return P->bounds();
    
    if (t != 0) {
      double xx = t->getxx(), xy = t->getxy(),
             yx = t->getyx(), yy = t->getyy();

      // These are the maximum x and y values that a linear transform can map
      // a point in the unit circle.  This can be proven by the Lagrange
      // Multiplier theorem or by properties of the dot product.
      maxx = length(pair(xx,xy));
      maxy = length(pair(yx,yy));

      shift = *t*pair(0,0);
    } else {
      maxx = 1;
      maxy = 1;
      shift = pair(0,0);
    }

    bbox b;
    pair z=0.5*width()*pair(maxx,maxy);
    b += z + shift;
    b += -z + shift;

    return b;
  }

  friend pen transformed(const transform& t, pen p) {
    pen ret = p;
    if(p.P) ret.P=new path(p.P->transformed(t));
    ret.t = new transform(p.t ? t*(*p.t) : t);
    return ret;
  }

  const transform *getTransform() const {
    return t;
  }
};
  
pen transformed(const transform& t, pen p);
}

#endif
