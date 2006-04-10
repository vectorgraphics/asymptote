import fontsize;

access settings;
orientation=Landscape;
public real margin=1cm;
public real pagewidth=settings.pageheight-margin;
public real pageheight=settings.pagewidth-margin;
size(pagewidth,pageheight,IgnoreAspect);

texpreamble("\hyphenpenalty=10000");

draw((-1,-1),invisible);
draw((1,1),invisible);
transform t=inverse(currentpicture.calculateTransform());
  
public pen itempen=fontsize(24pt);
public real itemskip=0.5;

public pair titlealign=2S;
public pen titlepen=fontsize(32pt);
public real titleskip=0.5;

public pen urlpen=fontsize(20pt);
public pair urlskip=(0,0.3);

public pen titlepagepen=fontsize(36pt)+red;
public pen authorpen=titlepen;
public pen datepen=urlpen+blue;
public pair dateskip=(0,0.05);

public string bullet="{\newcmykcolor{ASYcolor}{0 1 1 0}\ASYcolor$\bullet$}";
					      
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

void step()
{
  if(!stepping) return;
  lastnode.push(currentpicture.nodes.length-1);
  nextpage(steppagenumberpen);
  for(int i=0; i < firstnode.length; ++i)
    for(int j=firstnode[i]; j <= lastnode[i]; ++j)
      currentpicture.add(currentpicture.nodes[j]);
  firstnode.push(currentpicture.nodes.length-1);
}

void newslide() 
{
  nextpage();
  ++page;
  currentposition=startposition;
  firstnode=new int[] {currentpicture.nodes.length};
  lastnode=new int[];
}

void incrementposition(pair z)
{
  if(abs(currentposition.x) > 1 || abs(currentposition.y) > 1)
    abort("Overfull slide on page "+(string) page);
  currentposition += z;
}

void title(string s, pair position=N, pair align=titlealign,
	   pen p=titlepen) 
{
  frame f;
  label(f,s,(0,0),align,p);
  add(position,f,labelmargin(p)*align);
  currentposition=(currentposition.x,position.y+
		   (t*(min(f)-titleskip*I*lineskip(p)*pt)).y);
}

void remark(string s, pen p=itempen, real indent=0)
{
  frame f;
  label(f,minipage(s,0.75*pagewidth),(indent,0),SE,p);
  add(currentposition,f);
  incrementposition((0,(t*(min(f)-itemskip*I*lineskip(p)*pt)).y));
}

void center(string s, pen p=itempen)
{
  remark("\center "+s,p);
}

void equation(string s, pen p=itempen)
{
  center("{$\displaystyle "+s+"$}",p);
}

void figure(string s, string options="", string caption="")
{
  center(graphic(s,options));
  if(caption != "") center(caption);
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
  if(stepping && havepagenumber) numberpage();
  plain.exitfunction();
}
atexit(exitfunction);
