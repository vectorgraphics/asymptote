import fontsize;
usepackage("colordvi");

public bool reverse=false; // Set to true to enable reverse video
public bool stepping=false; // Set to true to enable stepping
public bool itemstep=true;  // Set to false to disable stepping on each item

public bool allowstepping=false; // Allow stepping for current slide.

public real margin=0.5cm;
public real pagewidth=-2margin;
public real pageheight=-2margin;

access settings;
								 
if(orientation == Portrait || orientation == UpsideDown) {
  pagewidth += settings.paperwidth;
  pageheight += settings.paperheight;
} else {
  pagewidth += settings.paperheight;
  pageheight += settings.paperwidth;
}

size(pagewidth,pageheight,IgnoreAspect);

real minipagemargin=1inch;
public real minipagewidth=pagewidth-2minipagemargin;

public transform tinv=inverse(fixedscaling((-1,-1),(1,1)));
  
public pen itempen=fontsize(24pt);
public real itemskip=0.5;

public pair titlealign=2S;
public pen titlepen=fontsize(32pt);
public real titleskip=0.5;

public pen urlpen=fontsize(20pt);
public pair urlskip=(0,0.3);

public pen titlepagepen=fontsize(36pt)+red;
public pen authorpen=fontsize(24pt)+blue;
public pen datepen=urlpen;
public pair dateskip=(0,0.05);

public string oldbulletcolor="Black";
public string newbulletcolor="Red";
public string bullet="\bulletcolor{$\bullet$}";
					      
public pair pagenumberposition=S+E;
public pair pagenumberalign=4NW;
public pen pagenumberpen=fontsize(12);
public pen steppagenumberpen=pagenumberpen+red;

public real figureborder=0.25cm;
public pen figuremattpen=invisible;

public pair titleposition=(-0.8,0.4);
public pair startposition=(-0.8,0.9);
public pair currentposition=startposition;

texpreamble("\hyphenpenalty=10000\tolerance=1000");
texpreamble("\let\bulletcolor"+'\\'+newbulletcolor);

public picture background;
size(background,pagewidth,pageheight,IgnoreAspect);

defaultpen(itempen);

int[] firstnode=new int[] {currentpicture.nodes.length};
int[] lastnode=new int[];
bool firststep=true;

public int page=1;
bool havepagenumber=false;

void background() 
{
  if(!background.empty()) {
    add(background);
    layer();
  }
}

// Evaluate user command line option.
void usersetting()
{
  plain.usersetting();
  if(reverse) { // Black background
    fill(background,box((-1,-1),(1,1)),black);
    itempen=white;
    defaultpen(itempen);
    oldbulletcolor="White";
    pagenumberpen=fontsize(pagenumberpen)+white;
    steppagenumberpen=pagenumberpen+mediumblue;
    authorpen=fontsize(authorpen)+paleblue;
    figuremattpen=white;
    usepackage("asycolors");
    texpreamble("\def\Blue#1{{\paleblue #1}}");
  } else { // White background
    texpreamble("\let\Green\OliveGreen");
  }
}

void numberpage(pen p=pagenumberpen)
{
  label((string) page,pagenumberposition,pagenumberalign,p);
  havepagenumber=true;
}

void nextpage(pen p=pagenumberpen)
{
  numberpage(p);
  newpage();
  background();
  firststep=true;
}

void newslide(bool stepping=true) 
{
  allowstepping=stepping;
  nextpage();
  ++page;
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
      tex("\let\bulletcolor"+'\\'+oldbulletcolor);
      currentpicture.add(currentpicture.nodes[j]);
    }
  }
  firstnode.push(currentpicture.nodes.length-1);
  tex("\let\bulletcolor"+'\\'+newbulletcolor);
}

void incrementposition(pair z)
{
  currentposition += z;
}

void title(string s, pair position=N, pair align=titlealign,
	   pen p=titlepen, bool newslide=true)
{
  if(newslide && !currentpicture.empty()) newslide();
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
	    real indent=0, bool minipage=true, filltype filltype=NoFill) 
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
    abort("slide too wide on page "+(string) page);

  if(abs(offset.y+M.y) > 1) {
    void toohigh() {
      abort("slide too high on page "+(string) page);
    }
    if(M.y-m.y < 2) {
      newslide(); offset=(offset.x,currentposition.y);
      if(offset.y+M.y > 1 || offset.y+m.y < -1) toohigh();
    } else toohigh();
  }

  add(f,offset);
  incrementposition((0,(tinv*(min(f)-itemskip*I*lineskip(p)*pt)).y));
}

void center(string s, pen p=itempen)
{
  remark("\center "+s,p);
}

void equation(string s, pen p=itempen)
{
  remark(center=true,"{$\displaystyle "+s+"$}",p,minipage=false);
}

void figure(string[] s, string options="", real margin=50bp, 
	    pen figuremattpen=figuremattpen,
	    string caption="", pair align=S, pen p=itempen)
{
  string S;
  if(s.length == 0) return;
  S=graphic(s[0],options);
  for(int i=1; i < s.length; ++i)
    S += "\kern "+(string) (margin/pt)+"pt "+graphic(s[i],options);
  remark(center=true,S,align,minipage=false,Fill(figureborder,figuremattpen));
  if(caption != "") center(caption,p);
}

void figure(string s, string options="", real margin=50bp,
	    pen figuremattpen=figuremattpen,
	    string caption="", pair align=S, pen p=itempen)
{
  figure(new string[] {s},options,margin,figuremattpen,caption,align,p);
}

void item(string s, pen p=itempen, bool step=itemstep)
{
  if(step && !firststep) step();
  firststep=false;
  frame b;
  label(b,bullet,(0,0),p);
  real bulletwidth=max(b).x-min(b).x;
  remark(bullet+"\hangindent"+(string) bulletwidth+"pt$\,$"+s,p,
	 -bulletwidth*pt);
}

void subitem(string s, pen p=itempen, bool step=itemstep)
{
  remark("\quad -- "+s,p);
}

void skip(real n=1)
{
  incrementposition((0,(tinv*(-n*itemskip*I*lineskip(itempen)*pt)).y));
}

void titlepage(string title, string author, string date="", string url="",
	       bool newslide=false)
{
  if(newslide && !currentpicture.empty()) newslide();
  background();
  currentposition=titleposition;
  center(title,titlepagepen);
  center(author,authorpen);
  currentposition -= dateskip;
  if(date != "") center(date,datepen);
  currentposition -= urlskip;
  if(url != "") center("{\tt "+url+"}",urlpen);
}

void exitfunction()
{
  if(havepagenumber) numberpage();
  plain.exitfunction();
}

atexit(exitfunction);
