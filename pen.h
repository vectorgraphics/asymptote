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

using std::string;
using settings::defaultlinewidth;
using settings::defaultfontsize;

namespace camp {

static const string DEFLINE="default";
static const double DEFWIDTH=-1;
static const int DEFCAP=-1;
static const int DEFJOIN=-1;
  
static const struct transparentpen_t {} transparentpen={};
static const struct setlinewidth_t {} setlinewidth={};
static const struct setfontsize_t {} setfontsize={};
static const struct setpattern_t {} setpattern={};
static const struct setlinecap_t {} setlinecap={};
static const struct setlinejoin_t {} setlinejoin={};
static const struct initialpen_t {} initialpen={};
  
static const std::string Cap[]={"square","round","extended"};
static const std::string Join[]={"miter","round","bevel"};
  
const int nCap=sizeof(Cap)/sizeof(string);
const int nJoin=sizeof(Join)/sizeof(string);
  
enum ColorSpace {TRANSPARENT,DEFCOLOR,GRAYSCALE,RGB,CMYK,PATTERN};
static const int ColorComponents[]={0,1,1,3,4,0};
static const std::string ColorDeviceSuffix[]={"","Gray","Gray","RGB","CMYK",
					      ""};
class pen : public mempool::pooled<pen>
{ 
  // The string for the PostScript style line pattern.
  string line;

  // Width of line, in PS units.
  double linewidth;
  double fontsize;  
  
  ColorSpace color;
  double r,g,b; // RGB or CMY value
  double grey; // grayscale or K value

  // The name of the user-defined fill/draw pattern.
  string pattern;
  int linecap;
  int linejoin;
  
  // The transformation applied to the pen nib for calligraphic effects.
  // Null means the identity transformation.
  const transform *t;
  
public:
  pen() : line(DEFLINE), linewidth(DEFWIDTH), fontsize(0.0),
	  color(DEFCOLOR), r(0.0), g(0.0), b(0.0), grey(0.0), pattern(""),
	  linecap(DEFCAP), linejoin(DEFJOIN), t(0) {}

  pen(transparentpen_t) : 
    line(DEFLINE), linewidth(DEFWIDTH), fontsize(0.0),
    color(TRANSPARENT), r(0.0), g(0.0), b(0.0), grey(0.0), pattern (""),
    linecap(DEFCAP), linejoin(DEFJOIN), t(0) {}
  
  pen(const string& line, double linewidth, double fontsize,
      ColorSpace color, double r, double g, double b,  double grey,
      const string& pattern, int linecap, int linejoin, const transform *t)
    : line(line), linewidth(linewidth), fontsize(fontsize),
      color(color), r(r), g(g), b(b), grey(grey), pattern(pattern),
      linecap(linecap), linejoin(linejoin), t(t) {}
      
  
  explicit pen(setlinewidth_t, double linewidth) : 
    line(DEFLINE), linewidth(linewidth), fontsize(0.0), color(DEFCOLOR),
    r(0.0), g(0.0), b(0.0), grey(0.0), pattern(""),
    linecap(DEFCAP), linejoin(DEFJOIN), t(0) {}
  
  explicit pen(const string& line) : line(line), linewidth(DEFWIDTH),
				     fontsize(0.0), color(DEFCOLOR),
				     r(0.0), g(0.0), b(0.0), grey(0.0),
				     pattern(""), linecap(DEFCAP),
				     linejoin(DEFJOIN), t(0) {}
  
  pen(setfontsize_t, double fontsize) : line(DEFLINE), linewidth(DEFWIDTH),
					fontsize(fontsize), color(DEFCOLOR), 
					r(0.0), g(0.0), b(0.0), grey(0.0),
					pattern(""), linecap(DEFCAP),
					linejoin(DEFJOIN), t(0) {}
  
  pen(initialpen_t) : line(DEFLINE), linewidth(-2), fontsize(-1),
		      color(DEFCOLOR), r(0.0), g(0.0), b(0.0), grey(0.0),
		      pattern(""), linecap(-2), linejoin(-2), t(0) {}
  
  static double pos0(double x) {return x >= 0 ? x : 0;}
  
  void greyrange() {if(grey > 1.0) grey=1.0;}
  
  explicit pen(double grey) : line(DEFLINE), linewidth(DEFWIDTH),
			      fontsize(0.0),
		              color(GRAYSCALE), r(0.0), g(0.0), b(0.0),
		              grey(pos0(grey)), pattern(""),
			      linecap(DEFCAP), linejoin(DEFJOIN), t(0) {
    greyrange();
  }
  
  explicit pen(setpattern_t, const string& pattern) :
    line(DEFLINE), linewidth(DEFWIDTH), fontsize(0.0), color(PATTERN),
    r(0.0), g(0.0), b(0.0), grey(0.0), pattern(pattern),
    linecap(DEFCAP), linejoin(DEFJOIN), t(0) {}
  
  explicit pen(setlinecap_t, int linecap) :
    line(DEFLINE), linewidth(DEFWIDTH), fontsize(0.0), color(DEFCOLOR),
    r(0.0), g(0.0), b(0.0), grey(0.0), pattern(""),
    linecap(linecap), linejoin(DEFJOIN), t(0) {}
  
  explicit pen(setlinejoin_t, int linejoin) :
    line(DEFLINE), linewidth(DEFWIDTH), fontsize(0.0), color(DEFCOLOR),
    r(0.0), g(0.0), b(0.0), grey(0.0), pattern(""),
    linecap(DEFCAP), linejoin(linejoin), t(0) {}
  
  void rgbrange() {
    double sat=rgbsaturation();
    if(sat > 1.0) {
      double scale=1.0/sat;
      r *= scale;
      g *= scale;
      b *= scale;
    }
  }
  
  pen(double r, double g, double b) : 
    line(DEFLINE), linewidth(DEFWIDTH), fontsize(0.0),
    color(RGB), r(pos0(r)), g(pos0(g)), b(pos0(b)),  grey(0.0), 
    pattern(""), linecap(DEFCAP), linejoin(DEFJOIN), t(0) {
    rgbrange();
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
  
  pen(double c, double m, double y, double k) :
    line(DEFLINE), linewidth(DEFWIDTH), fontsize(0.0),
    color(CMYK), r(pos0(c)), g(pos0(m)), b(pos0(y)), grey(pos0(k)),
    pattern(""), linecap(DEFCAP), linejoin(DEFJOIN), t(0) {
    cmykrange();
  }
  
  double width() const {
    return linewidth == DEFWIDTH ? defaultlinewidth : linewidth;
  }
  
  double size() const {return fontsize == 0.0 ? defaultfontsize : fontsize;}
  
  string stroke() {return line == DEFLINE ? "" : line;}
  
  void setstroke(const string& s) {line=s;}
  
  string fillpattern() {return pattern;}
  
  int cap() {return linecap == DEFCAP ? 1 : linecap;}
  
  int join() {return linejoin == DEFJOIN ? 1 : linejoin;}
  
  ColorSpace colorspace() {return color;}
  
  bool transparent() const {return color == TRANSPARENT;}
  
  bool mono() const {return color == GRAYSCALE || color == DEFCOLOR;}
  
  bool grayscale() const {return color == GRAYSCALE;}
  
  bool rgb() const {return color == RGB;}
  
  bool cmyk() const {return color == CMYK;}
  
  double red() const {return r;}
  
  double green() const {return g;}
  
  double blue() const {return b;}
  
  double cyan() const {return r;}
  
  double magenta() const {return g;}
  
  double yellow() const {return b;}
  
  double black() const {return grey;}
  
  double gray() const {return grey;}
  
  double rgbsaturation() const {return max(max(r,g),b);}
  
  double cmyksaturation() const {return max(rgbsaturation(),black());}
  
  void greytorgb() {
    r=g=b=grey; grey=0.0;
    color=RGB;
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

  void convert() {
    if(settings::rgbonly && cmyk()) cmyktorgb();
    else if(settings::cmykonly && rgb()) rgbtocmyk();
  }    
  
  friend pen operator * (double x, const pen& q) {
    pen p=q;
    if(x < 0.0) x = 0.0;
    switch(p.color) {
    case PATTERN:
    case TRANSPARENT:
      break;
    case DEFCOLOR:
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
    ColorSpace colorspace=(ColorSpace) max(p.color,q.color);
    
    double R=0.0,G=0.0,B=0.0,greyval=0.0;
    pen P=p;
    pen Q=q;
    
    switch(colorspace) {
    case PATTERN:
    case TRANSPARENT:
      break;
    case DEFCOLOR:
    case GRAYSCALE:
      {
	if(!p.grayscale()) greyval=q.grey;
	else {
	  if(!q.grayscale()) greyval=p.grey;
	  else greyval=0.5*(p.grey+q.grey);
	}
	break;
      }
      
    case RGB:
      {
	double sat;
	if (P.grayscale()) {
	  P.greytorgb();
	  sat=0.5*(P.rgbsaturation()+Q.rgbsaturation());
	} else {
	  if (Q.grayscale()) {
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
	if (P.grayscale()) P.greytocmyk();
	else if (Q.grayscale()) Q.greytocmyk();
	
	if (P.rgb()) P.rgbtocmyk();
	else if (Q.rgb()) Q.rgbtocmyk();
	
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
	       q.fontsize == 0.0 ? p.fontsize : q.fontsize,
	       colorspace,R,G,B,greyval,
	       q.pattern == "" ? p.pattern : q.pattern,
	       q.linecap == DEFCAP ? p.linecap : q.linecap,
	       q.linejoin == DEFJOIN ? p.linejoin : q.linejoin,
	       q.t == NULL ? p.t : q.t);
  }

  friend bool operator == (const pen& p, const pen& q) {
    return p.line == q.line && p.linewidth == q.linewidth 
      && p.color == q.color
      && p.fontsize == q.fontsize
      && (!(p.grayscale() || p.cmyk())  || p.grey == q.grey)
      && (!(p.rgb() || p.cmyk()) || (p.r == q.r && p.g == q.g && p.b == q.b))
      && p.pattern == q.pattern
      && p.linecap == q.linecap
      && p.linejoin == q.linejoin
      && (p.t ? *p.t : identity()) == (q.t ? *q.t : identity());
  }
  
  friend ostream& operator << (ostream& out, const pen& p) {
    out << "([" << p.line << "]";
    if(p.linewidth != DEFWIDTH)
      out << ", linewidth=" << p.linewidth;
    if(p.linecap != DEFCAP)
      out << ", linecap=" << Cap[p.linecap];
    if(p.linejoin != DEFJOIN)
      out << ", linejoin=" << Join[p.linejoin];
    if(p.fontsize)
      out << ", fontsize=" << p.fontsize;
    if(p.grayscale())
      out << ", gray=" << p.grey;
    if(p.rgb()) 
      out << ", red=" << p.red() << ", green=" << p.green() 
	  << ", blue=" << p.blue();
    if(p.cmyk()) 
      out << ", cyan=" << p.cyan() << ", magenta=" << p.magenta() 
	  << ", yellow=" << p.yellow() << ", black=" << p.black();
    if(p.pattern != "")
      out << ", " << "\"" << p.pattern << "\"";
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
