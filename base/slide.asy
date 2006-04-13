import fontsize;
usepackage("colordvi");

access settings;
orientation=Landscape;
public real margin=1cm;
public real pagewidth=settings.pageheight-margin;
public real pageheight=settings.pagewidth-margin;
size(pagewidth,pageheight,IgnoreAspect);

real minipagemargin=1inch;
real minipagewidth=pagewidth-2minipagemargin;

texpreamble("\hyphenpenalty=5000\tolerance=1000");
texpreamble("\let\bulletcolor\Red");

draw((-1,-1),invisible);
draw((1,1),invisible);

transform t=currentpicture.calculateTransform();
transform tinv=inverse(t);
  
public pen itempen=fontsize(24pt);
public real itemskip=0.5;

public pair titlealign=2S;
public pen titlepen=fontsize(32pt);
public real titleskip=0.5;

public pen urlpen=fontsize(20pt);
public pair urlskip=(0,0.3);

public pen titlepagepen=fontsize(36pt)+red;
public pen authorpen=fontsize(36pt)+blue;
public pen datepen=urlpen;
public pair dateskip=(0,0.05);

//public string bullet="{\newcmykcolor{ASYcolor}{0 1 1 0}\ASYcolor$\bullet$}";
public string bullet="\bulletcolor{$\bullet$}";
					      
public pair pagenumberposition=S+E;
public pair pagenumberalign=4NW;
public pen pagenumberpen=fontsize(12);
public pen steppagenumberpen=pagenumberpen+red;

public pair titleposition=(-0.8,0.4);
public pair startposition=(-0.8,0.9);
public pair currentposition=startposition;

defaultpen(itempen);

public bool stepping=false;
public bool itemstep=true;

int[] firstnode=new int[] {currentpicture.nodes.length};
int[] lastnode=new int[];
bool firststep=true;

public int page=1;
bool havepagenumber=false;

void numberpage(pen p=pagenumberpen)
{
  label((string) page,pagenumberposition,pagenumberalign,p);
  havepagenumber=true;
}

void nextpage(pen p=pagenumberpen)
{
  numberpage(p);
  newpage();
  firststep=true;
}

void newslide() 
{
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
  if(!stepping) return;
  if(!checkposition()) return;
  lastnode.push(currentpicture.nodes.length-1);
  nextpage(steppagenumberpen);
  for(int i=0; i < firstnode.length; ++i) {
    for(int j=firstnode[i]; j <= lastnode[i]; ++j) {
      tex("\let\bulletcolor\Black");
      currentpicture.add(currentpicture.nodes[j]);
    }
  }
  firstnode.push(currentpicture.nodes.length-1);
  tex("\let\bulletcolor\Red");
}

void incrementposition(pair z)
{
  currentposition += z;
}

void title(string s, pair position=N, pair align=titlealign,
	   pen p=titlepen) 
{
  checkposition();
  frame f;
  label(f,minipage("\center "+s,minipagewidth),(0,0),align,p);
  add(position,f,labelmargin(p)*align);
  currentposition=(currentposition.x,position.y+
		   (tinv*(min(f)-titleskip*I*lineskip(p)*pt)).y);
}

void remark(bool center=false, string s, pair align=0, pen p=itempen,
	    real indent=0, bool minipage=true) 
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
  label(f,s,(indent,0),align,p);
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

  add(offset,f);
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

void figure(string s, string options="", string caption="", pair align=S)
{
  remark(center=true,graphic(s,options),align,minipage=false);
  if(caption != "") remark(center=true,caption,align);
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

void titlepage(string title, string author, string date="", string url="")
{
  currentposition=titleposition;
  center(title,titlepagepen);
  center(author,authorpen);
  currentposition -= dateskip;
  if(date != "") center(date,datepen);
  currentposition -= urlskip;
  if(url != "") center("{\tt "+url+"}",urlpen);
  newslide();
}

void exitfunction()
{
  if(havepagenumber) numberpage();
  frame f=currentpicture.fit(t);
  if(interact() || (!shipped && !currentpicture.empty())) 
    shipout(orientation(f));
}
atexit(exitfunction);
