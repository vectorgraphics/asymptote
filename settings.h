/*****
 * settings.h
 * Andy Hammerlindl 2004/05/10
 *
 * Declares a list of global variables that act as settings in the system.
 *****/

#ifndef SETTINGS_H
#define SETTINGS_H

#include <fstream>
#include <sys/stat.h>

#include "common.h"
#include "pair.h"
#include "item.h"

namespace types {
class record;
}

namespace camp {
void glrenderWrapper();
}

namespace gl {
extern bool glthread;
extern bool initialize;

#ifdef HAVE_PTHREAD
extern pthread_t mainthread;
extern pthread_cond_t initSignal;
extern pthread_mutex_t initLock;
extern pthread_cond_t readySignal;
extern pthread_mutex_t readyLock;
void wait(pthread_cond_t& signal, pthread_mutex_t& lock);
void endwait(pthread_cond_t& signal, pthread_mutex_t& lock);
#endif
}

namespace settings {
extern char *argv0;

void Warn(const string& s);
void noWarn(const string& s);
bool warn(const string& s);
extern string systemDir;
extern string docdir;
extern const string dirsep;

extern bool safe;

bool globalread();
bool globalwrite();

extern const string suffix;
extern const string guisuffix;
extern const string standardprefix;

extern string historyname;

void SetPageDimensions();

types::record *getSettingsModule();

vm::item& Setting(string name);
vm::item& Setting(int optionId);

// This approach to setting up an enum with access to enum names as strings is
// based on https://stackoverflow.com/a/1801402/2318074 and
// https://stackoverflow.com/a/1809701/2318074.
#define ASYOPTIONLIST \
  OPTION(leftbutton) \
  OPTION(middlebutton) \
  OPTION(rightbutton) \
  OPTION(wheelup) \
  OPTION(wheeldown) \
  OPTION(suppress) \
  OPTION(warn) \
  OPTION(View) \
  OPTION(batchView) \
  OPTION(multipleView) \
  OPTION(interactiveView) \
  OPTION(outformat) \
  OPTION(svgemulation) \
  OPTION(prc) \
  OPTION(v3d) \
  OPTION(toolbar) \
  OPTION(axes3) \
  OPTION(ibl) \
  OPTION(image) \
  OPTION(imageDir) \
  OPTION(imageURL) \
  OPTION(render) \
  OPTION(devicepixelratio) \
  OPTION(antialias) \
  OPTION(multisample) \
  OPTION(twosided) \
  OPTION(GPUindexing) \
  OPTION(GPUinterlock) \
  OPTION(GPUcompress) \
  OPTION(GPUlocalSize) \
  OPTION(GPUblockSize) \
  OPTION(position) \
  OPTION(maxviewport) \
  OPTION(viewportmargin) \
  OPTION(webgl2) \
  OPTION(absolute) \
  OPTION(maxtile) \
  OPTION(iconify) \
  OPTION(thick) \
  OPTION(thin) \
  OPTION(autobillboard) \
  OPTION(threads) \
  OPTION(fitscreen) \
  OPTION(interactiveWrite) \
  OPTION(help) \
  OPTION(environment) \
  OPTION(version) \
  OPTION(offset) \
  OPTION(aligndir) \
  OPTION(align) \
  OPTION(debug) \
  OPTION(verbose) \
  OPTION(vv) \
  OPTION(novv) \
  OPTION(keep) \
  OPTION(keepaux) \
  OPTION(tex) \
  OPTION(twice) \
  OPTION(inlinetex) \
  OPTION(embed) \
  OPTION(auto3D) \
  OPTION(autoplay) \
  OPTION(loop) \
  OPTION(interrupt) \
  OPTION(animating) \
  OPTION(reverse) \
  OPTION(inlineimage) \
  OPTION(compress) \
  OPTION(parseonly) \
  OPTION(translate) \
  OPTION(tabcompletion) \
  OPTION(prerender) \
  OPTION(lossy) \
  OPTION(listvariables) \
  OPTION(where) \
  OPTION(mask) \
  OPTION(batchMask) \
  OPTION(interactiveMask) \
  OPTION(bw) \
  OPTION(gray) \
  OPTION(rgb) \
  OPTION(cmyk) \
  OPTION(safe) \
  OPTION(globalwrite) \
  OPTION(globalread) \
  OPTION(outname) \
  OPTION(cd) \
  OPTION(compact) \
  OPTION(divisor) \
  OPTION(prompt) \
  OPTION(prompt2) \
  OPTION(multiline) \
  OPTION(xasy) \
  OPTION(lsp) \
  OPTION(lspport) \
  OPTION(lsphost) \
  OPTION(wsl) \
  OPTION(wait) \
  OPTION(inpipe) \
  OPTION(outpipe) \
  OPTION(exitonEOF) \
  OPTION(quiet) \
  OPTION(localhistory) \
  OPTION(historylines) \
  OPTION(scroll) \
  OPTION(level) \
  OPTION(autoplain) \
  OPTION(autorotate) \
  OPTION(offline) \
  OPTION(pdfreload) \
  OPTION(pdfreloaddelay) \
  OPTION(autoimport) \
  OPTION(command) \
  OPTION(user) \
  OPTION(zoomfactor) \
  OPTION(zoomPinchFactor) \
  OPTION(zoomPinchCap) \
  OPTION(zoomstep) \
  OPTION(shiftHoldDistance) \
  OPTION(shiftWaitTime) \
  OPTION(vibrateTime) \
  OPTION(spinstep) \
  OPTION(framerate) \
  OPTION(resizestep) \
  OPTION(digits) \
  OPTION(paperwidth) \
  OPTION(paperheight) \
  OPTION(dvipsOptions) \
  OPTION(dvisvgmOptions) \
  OPTION(dvisvgmMultipleFiles) \
  OPTION(convertOptions) \
  OPTION(gsOptions) \
  OPTION(htmlviewerOptions) \
  OPTION(psviewerOptions) \
  OPTION(pdfviewerOptions) \
  OPTION(pdfreloadOptions) \
  OPTION(glOptions) \
  OPTION(hyperrefOptions) \
  OPTION(config) \
  OPTION(htmlviewer) \
  OPTION(pdfviewer) \
  OPTION(psviewer) \
  OPTION(gs) \
  OPTION(libgs) \
  OPTION(epsdriver) \
  OPTION(psdriver) \
  OPTION(pngdriver) \
  OPTION(asygl) \
  OPTION(texpath) \
  OPTION(texcommand) \
  OPTION(dvips) \
  OPTION(dvisvgm) \
  OPTION(convert) \
  OPTION(display) \
  OPTION(animate) \
  OPTION(papertype) \
  OPTION(dir) \
  OPTION(sysdir) \
  OPTION(textcommand) \
  OPTION(textcommandOptions) \
  OPTION(textextension) \
  OPTION(textoutformat) \
  OPTION(textprologue) \
  OPTION(textinitialfont) \
  OPTION(textepilogue) \

namespace optionList {
# define OPTION(name) name,  // Add a comma after each name
  enum Option {
    ASYOPTIONLIST
    numOptions
  };
# undef OPTION
}  // namespace optionList

struct option;

namespace internal {
  extern mem::vector<option *> options;
  extern const char *optionNames[];
}  // namespace internal

template <typename T>
inline T getSetting(int optionId)
{
  return vm::get<T>(Setting(optionId));
}

extern Int verbose;
extern bool compact;
extern bool gray;
extern bool bw;
extern bool rgb;
extern bool cmyk;

bool view();
bool trap();
string outname();

void setOptions(int argc, char *argv[]);

// Access the arguments once options have been parsed.
int numArgs();
char *getArg(int n);

Int getScroll();

#if !defined(_MSC_VER)
extern mode_t mask;
#endif

bool xe(const string& texengine);
bool lua(const string& texengine);
bool pdf(const string& texengine);
bool latex(const string& texengine);
bool context(const string& texengine);

string nativeformat();
string defaultformat();

const char *newpage(const string& texengine);
const char *beginlabel(const string& texengine);
const char *endlabel(const string& texengine);
const char *rawpostscript(const string& texengine);
const char *beginpicture(const string& texengine);
const char *endpicture(const string& texengine);
const char *beginspecial(const string& texengine);
const char *endspecial();

string texcommand();
string texprogram();

const double inches=72.0;
const double cm=inches/2.54;
const double tex2ps=72.0/72.27;
const double ps2tex=1.0/tex2ps;

const string AsyGL="webgl/asygl.js";
const string WebGLheader="webgl/WebGLheader.html";
const string WebGLfooter="webgl/WebGLfooter.html";
}

extern const char *REVISION;
extern const char *AsyGLVersion;

#endif
