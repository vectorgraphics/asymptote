import fontsize;
usepackage("asycolors");

bool reverse=false; // Set to true to enable reverse video
bool stepping=false; // Set to true to enable stepping
bool itemstep=true;  // Set to false to disable stepping on each item

bool allowstepping=false; // Allow stepping for current slide.

real pagemargin=0.5cm;
real pagewidth=-2pagemargin;
real pageheight=-2pagemargin;

bool landscape=orientation == Landscape || orientation == Seascape;

if(landscape) {
  orientation=Portrait;
  real temp=settings.paperwidth;
  settings.paperwidth=settings.paperheight;
  settings.paperheight=temp;
  pagewidth += settings.paperwidth;
  pageheight += settings.paperheight;
 } else {
  pagewidth += settings.paperwidth;
  pageheight += settings.paperheight;
 }

size(pagewidth,pageheight,IgnoreAspect);
picture background;

real minipagemargin=1inch;
real minipagewidth=pagewidth-2minipagemargin;

transform tinv=inverse(fixedscaling((-1,-1),(1,1),currentpen));
  
pen itempen=fontsize(24pt);
pen codepen=fontsize(20pt);
pen titlepagepen=fontsize(36pt);
pen authorpen=fontsize(24pt);
pen institutionpen=authorpen;
pen datepen=fontsize(18pt);
pen urlpen=datepen;

real itemskip=0.5;
real codeskip=0.25;
pair dateskip=(0,0.1);
pair urlskip=(0,0.2);

pair titlealign=2S;
pen titlepen=fontsize(32pt);
real titleskip=0.5;

string oldbulletcolor;
string newbulletcolor="red";
string bullet="{\bulletcolor\textbullet}";
                                              
pair pagenumberposition=S+E;
pair pagenumberalign=4NW;
pen pagenumberpen=fontsize(12);
pen steppagenumberpen=colorless(pagenumberpen);

real figureborder=0.25cm;
pen figuremattpen;

pen backgroundcolor;
pen foregroundcolor;

pair titleposition=(-0.8,0.4);
pair startposition=(-0.8,0.9);
pair currentposition=startposition;

string bulletcolor(string color)
{
  return "\def\bulletcolor{"+'\\'+"color{"+color+"}}%";
}

int[] firstnode=new int[] {currentpicture.nodes.length};
int[] lastnode=new int[];
bool firststep=true;

int page=1;
bool havepagenumber=true;

int preamblenodes=2;

bool empty()
{
  return currentpicture.nodes.length <= preamblenodes;
}

void background() 
{
  if(!background.empty()) {
    add(background);
    layer();
    preamblenodes += 2;
  }
}

void color(string name, string color)
{
  texpreamble("\def"+'\\'+name+"#1{{\color{"+color+"}#1}}%");
}

string texcolor(pen p)
{
  real[] colors=colors(p);
  string s;
  if(colors.length > 0) {
    s="{"+colorspace(p)+"}{";
    for(int i=0; i < colors.length-1; ++i)
      s += string(colors[i])+",";
    s += string(colors[colors.length-1])+"}";
  }
  return s;
}

void setpens(pen red=red, pen blue=blue, pen steppen=red)
{
  itempen=colorless(itempen);
  codepen=colorless(codepen);
  pagenumberpen=colorless(pagenumberpen);
  steppagenumberpen=colorless(steppagenumberpen)+steppen;
  titlepagepen=colorless(titlepagepen)+red;
  authorpen=colorless(authorpen)+blue;
  institutionpen=colorless(institutionpen)+blue;
  datepen=colorless(datepen);
  urlpen=colorless(urlpen);
}

void reversevideo()
{
  backgroundcolor=black;
  foregroundcolor=white;
  fill(background,box((-1,-1),(1,1)),backgroundcolor);
  setpens(mediumred,paleblue,mediumblue);
  // Work around pdflatex bug, in which white is mapped to black!
  figuremattpen=pdf() ? cmyk(0,0,0,1/255) : white;
  color("Red","mediumred");
  color("Green","green");
  color("Blue","paleblue");
  color("Foreground","white");
  color("Background","black");
  oldbulletcolor="white";
  defaultpen(itempen+foregroundcolor);
}

void normalvideo() {
  backgroundcolor=invisible;
  foregroundcolor=black;
  background=new picture;
  size(background,currentpicture);
  setpens();
  figuremattpen=invisible;
  color("Red","red");
  color("Green","heavygreen");
  color("Blue","blue");
  color("Foreground","black");
  color("Background","white");
  oldbulletcolor="black";
  defaultpen(itempen+foregroundcolor);
}

normalvideo();

texpreamble(bulletcolor(newbulletcolor));
texpreamble("\hyphenpenalty=10000\tolerance=1000");

// Evaluate user command line option.
void usersetting()
{
  plain.usersetting();
  if(reverse) { // Black background
    reversevideo();
  } else { // White background
    normalvideo();
  }
}

void numberpage(pen p=pagenumberpen)
{
  if(havepagenumber) {
    label((string) page,pagenumberposition,pagenumberalign,p);
    ++page;
  }
}

void nextpage(pen p=pagenumberpen)
{
  if(!empty()) {
    numberpage(p);
    newpage();
  }
  background();
  firststep=true;
}

void newslide(bool stepping=true) 
{
  allowstepping=stepping;
  nextpage();
  havepagenumber=true;
  currentposition=startposition;
  firstnode=new int[] {currentpicture.nodes.length};
  lastnode=new int[];
}

bool checkposition()
{
  if(abs(currentposition.x) > 1 || abs(currentposition.y) > 1) {
    newslide();
    return false;
  }
  return true;
}

void step()
{
  if(!stepping || !allowstepping) return;
  if(!checkposition()) return;
  lastnode.push(currentpicture.nodes.length-1);
  nextpage(steppagenumberpen);
  for(int i=0; i < firstnode.length; ++i) {
    for(int j=firstnode[i]; j <= lastnode[i]; ++j) {
      tex(bulletcolor(oldbulletcolor));
      currentpicture.add(currentpicture.nodes[j]);
    }
  }
  firstnode.push(currentpicture.nodes.length-1);
  tex(bulletcolor(newbulletcolor));
}

void incrementposition(pair z)
{
  currentposition += z;
}

void title(string s, pair position=N, pair align=titlealign,
           pen p=titlepen, bool newslide=true)
{
  if(newslide) newslide();
  checkposition();
  frame f;
  label(f,minipage("\center "+s,minipagewidth),(0,0),align,p);
  add(f,position,labelmargin(p)*align);
  currentposition=(currentposition.x,position.y+
                   (tinv*(min(f)-titleskip*I*lineskip(p)*pt)).y);
}

void outline(string s="Outline", pair position=N, pair align=titlealign,
             pen p=titlepen)
{
  newslide(stepping=false);
  title(s,position,align,p,newslide=false);
}

void remark(bool center=false, string s, pair align=0, pen p=itempen,
            real indent=0, bool minipage=true, real skip=itemskip,
            filltype filltype=NoFill, bool step=false) 
{
  checkposition();
  if(minipage) s=minipage(s,minipagewidth);
  
  pair offset;
  if(center) {
    if(align == 0) align=S;
    offset=(0,currentposition.y);
  } else {
    if(align == 0) align=SE;
    offset=currentposition;
  }
  
  frame f;
  label(f,s,(indent,0),align,p,filltype);
  pair m=tinv*min(f);
  pair M=tinv*min(f);
  
  if(abs(offset.x+M.x) > 1)
    write("warning: slide too wide on page "+(string) page+':\n'+(string) s);

  if(abs(offset.y+M.y) > 1) {
    void toohigh() {
      write("warning: slide too high on page "+(string) page+':\n'+(string) s);
    }
    if(M.y-m.y < 2) {
      newslide(); offset=(offset.x,currentposition.y);
      if(offset.y+M.y > 1 || offset.y+m.y < -1) toohigh();
    } else toohigh();
  }

  if(step) {
    if(!firststep) step();
    firststep=false;
  }

  add(f,offset);
  incrementposition((0,(tinv*(min(f)-skip*I*lineskip(p)*pt)).y));
}

void center(string s, pen p=itempen)
{
  remark("\center "+s,p);
}

void equation(string s, pen p=itempen)
{
  remark(center=true,"\vbox{$$"+s+"$$}",p,minipage=false,skip=0);
}

void vbox(string s, pen p=itempen)
{
  remark(center=true,"\vbox{"+s+"}",p,minipage=false,skip=0);
}

void equations(string s, pen p=itempen)
{
  vbox("\begin{eqnarray*}"+s+"\end{eqnarray*}",p);
}

void skip(real n=1)
{
  incrementposition((0,(tinv*(-n*itemskip*I*lineskip(itempen)*pt)).y));
}

void display(frame[] f, real margin=0, pair align=S, pen p=itempen,
             pen figuremattpen=figuremattpen)
{
  if(f.length == 0) return;
  real[] width=new real[f.length];
  real sum;
  for(int i=0; i < f.length; ++i) {
    width[i]=size(f[i]).x;
    sum += width[i];
  }
  if(sum > pagewidth)
    write("warning: slide too wide on page "+(string) page);
  else margin=(pagewidth-sum)/(f.length+1);
  real pos;
  frame F;
  for(int i=0; i < f.length; ++i) {
    real w=0.5*(margin+width[i]);
    pos += w;
    add(F,f[i],(pos,0),Fill(figureborder,figuremattpen));
    pos += w;
  }
  add(F,(0,currentposition.y),align);
  real a=0.5(unit(align).y-1);
  incrementposition((0,(tinv*(a*(max(F)-min(F))-itemskip*I*lineskip(p)*pt)).y));
}

void display(frame f, real margin=0, pair align=S, pen p=itempen,
             pen figuremattpen=figuremattpen)
{
  display(new frame[] {f},margin,align,p,figuremattpen);
}

void display(string[] s, real margin=0, string[] captions=new string[],
             string caption="", pair align=S, pen p=itempen,
             pen figuremattpen=figuremattpen)
{
  frame[] f=new frame[s.length];
  frame F;
  for(int i=0; i < s.length; ++i) {
    f[i]=newframe;
    label(f[i],s[i]);
    add(F,f[i],(0,0));
  }
  real y=point(F,S).y;
  int stop=min(s.length,captions.length);
  for(int i=0; i < stop; ++i) {
    if(captions[i] != "")
      label(f[i],captions[i],point(f[i],S).x+I*y,S);
  }
  display(f,margin,align,p,figuremattpen);
  if(caption != "") center(caption,p);
}

void display(string s, string caption="", pair align=S, pen p=itempen,
             pen figuremattpen=figuremattpen)
{
  display(new string[] {s},caption,align,p,figuremattpen);
}

void figure(string[] s, string options="", real margin=0, 
            string[] captions=new string[], string caption="",
            pair align=S, pen p=itempen, pen figuremattpen=figuremattpen)
{
  string[] S;
  for(int i=0; i < s.length; ++i) {
    S[i]=graphic(s[i],options);
  }

  display(S,margin,captions,caption,align,itempen,figuremattpen);
}

void figure(string s, string options="", string caption="", pair align=S,
            pen p=itempen, pen figuremattpen=figuremattpen)
{
  figure(new string[] {s},options,caption,align,p,figuremattpen);
}

string cropcode(string s)
{
  while(substr(s,0,1) == '\n') s=substr(s,1,length(s));
  while(substr(s,length(s)-1,1) == '\n') s=substr(s,0,length(s)-1);
  return s;
}

void code(bool center=false, string s, pen p=codepen,
          real indent=0, real skip=codeskip,
          filltype filltype=NoFill) 
{
  remark(center,"{\tt "+verbatim(cropcode(s))+"}",p,indent,skip,filltype);
}

void filecode(bool center=false, string s, pen p=codepen, real indent=0,
              real skip=codeskip, filltype filltype=NoFill)
{
  code(center,file(s),p,indent,skip,filltype);
}

void asyfigure(string s, string options="", string caption="", pair align=S,
               pen p=codepen, pen figuremattpen=figuremattpen,
               filltype filltype=NoFill, bool newslide=false)
{
  string a=s+".asy";
  asy(nativeformat(),s);
  s += "."+nativeformat();
  if(newslide && !empty()) {
    newslide();
    currentposition=(currentposition.x,0);
    align=0;
  }
  figure(s,options,caption,align,p,figuremattpen);
}

string[] codefile;

string asywrite(string s, string preamble="")
{
  static int count=0;
  string name="_slide"+(string) count;
  ++count;
  file temp=output(name+".asy");
  write(temp,preamble);
  write(temp,s);
  close(temp);
  codefile.push(name);
  return name;
}

void asycode(bool center=false, string s, string options="",
             string caption="", string preamble="",
             pair align=S, pen p=codepen, pen figuremattpen=figuremattpen,
             real indent=0, real skip=codeskip,
             filltype filltype=NoFill, bool newslide=false)
{
  code(center,s,p,indent,skip,filltype);
  asyfigure(asywrite(s,preamble),options,caption,align,p,figuremattpen,filltype,
            newslide);
}

void asyfilecode(bool center=false, string s, string options="",
                 string caption="",
                 pair align=S, pen p=codepen, pen figuremattpen=figuremattpen,
                 real indent=0, real skip=codeskip,
                 filltype filltype=NoFill, bool newslide=false)
{
  filecode(center,s+".asy",p,indent,skip,filltype);
  asyfigure(s,options,caption,align,p,figuremattpen,filltype,newslide);
}

void item(string s, pen p=itempen, bool step=itemstep)
{
  frame b;
  label(b,bullet,(0,0),p);
  real bulletwidth=max(b).x-min(b).x;
  remark(bullet+"\hangindent"+(string) bulletwidth+"pt$\,$"+s,p,
         -bulletwidth*pt,step=step);
}

void subitem(string s, pen p=itempen)
{
  remark("\quad -- "+s,p);
}

void titlepage(string title, string author, string institution="",
               string date="", string url="", bool newslide=false)
{
  newslide();
  currentposition=titleposition;
  center(title,titlepagepen);
  center(author,authorpen);
  if(institution != "") center(institution,institutionpen);
  currentposition -= dateskip;
  if(date != "") center(date,datepen);
  currentposition -= urlskip;
  if(url != "") center("{\tt "+url+"}",urlpen);
}

// Resolve optional bibtex citations:
void bibliographystyle(string name)
{
  settings.twice=true;
  settings.keepaux=true;
  texpreamble("\bibliographystyle{"+name+"}");
}

void bibliography(string name) 
{
  numberpage();
  havepagenumber=false;
  string s=texcolor(backgroundcolor);
  if(s != "") tex("\definecolor{Background}"+s+"\pagecolor{Background}%");
  label("",itempen);
  tex("\eject\def\refname{\fontsize{"+string(fontsize(titlepen))+"}{"+
      string(lineskip(titlepen))+"}\selectfont References}%");
  real hmargin,vmargin;
  if(pdf()) {
    hmargin=1;
    vmargin=0;
  } else {
    hmargin=1.5;
    vmargin=1;
  }
  string s;
  if(landscape) {
    s="{\centering\textheight="+string(pageheight-1inch)+"bp\textwidth="+
      string(pagewidth-1.5inches)+"bp"+
      "\vsize=\textheight\hsize=\textwidth\linewidth=\hsize"+
      "\topmargin="+string(vmargin)+"in\oddsidemargin="+string(hmargin)+"in";
  } else
    s="{\centering\textheight="+string(pageheight-0.5inches)+"bp\textwidth="+
      string(pagewidth-0.5inches)+
      "bp\hsize=\textwidth\linewidth=\textwidth\vsize=\textheight"+
      "\topmargin=0.5in\oddsidemargin=1in";
  s += "\evensidemargin=\oddsidemargin\bibliography{"+name+"}\eject}";
  tex(s);
}

void exitfunction()
{
  numberpage();
  plain.exitfunction();
  if(!settings.keep)
    for(int i=0; i < codefile.length; ++i) {
      string name=codefile[i];
      delete(name+"."+nativeformat());
      delete(name+"_.aux");
      delete(name+".asy");
    }
}

atexit(exitfunction);
