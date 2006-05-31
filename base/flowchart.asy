// Flowchart routines written by Jacques Pienaar and Steve Melenchuk.

private import math;

// N.B.: Relative coordinates take the lower left corner of the block: (0,0).

struct flowblock {
  // The relative maximum coordinates of the block. 
  public pair bound;

  // The absolute lower left corner of the block.
  public pair llcorner;

  // Returns the relative center of the block.
  public pair f_center();

  // Returns the absolute center of the block.
  pair center() {return shift(this.llcorner)*this.f_center();}
 
  // Returns the relative position along the boundary of the block.
  public pair f_position(real x);

  // Returns the absolute position along the boundary of the block.
  pair position(real x) {return shift(this.llcorner)*this.f_position(x);}

  // These eight functions return the appropriate location on the block
  // in relative coordinates.
  public pair f_top();
  public pair f_left();
  public pair f_right();
  public pair f_bottom();
  public pair f_topleft();
  public pair f_topright();
  public pair f_bottomleft();
  public pair f_bottomright();

  // These eight functions return the appropriate location on the block
  // in absolute coordinates.
  pair top() {return shift(this.llcorner)*this.f_top();} 
  pair bottom() {return shift(this.llcorner)*this.f_bottom();} 
  pair left() {return shift(this.llcorner)*this.f_left();} 
  pair right() {return shift(this.llcorner)*this.f_right();} 
  pair topleft() {return shift(this.llcorner)*this.f_topleft();} 
  pair topright() {return shift(this.llcorner)*this.f_topright();} 
  pair bottomleft() {return shift(this.llcorner)*this.f_bottomleft();} 
  pair bottomright() {return shift(this.llcorner)*this.f_bottom();} 
  
  // Centers the block on the given coordinate.
  void center(pair loc) {this.llcorner=loc-this.f_center();} 
  
  // Returns a picture representing the block.
  public picture draw(pen p=currentpen);
};

flowblock operator init() {return new flowblock;}

// Given two pictures, returns a flowblock with those as a header and
// body of a rectangular block.
flowblock flowrectangle(picture header, picture body, 
                        pen headercolor=mediumgray,
                        pen bodycolor=currentpen, 
                        pair center=(0,0), real dx=3) 
{
  pair bound[]=new pair[2];
  real width;
  path shape;
  bound[0]=max(header.fit())-min(header.fit());
  bound[1]=max(body.fit())-min(body.fit());
  width=max(bound[0].x,bound[1].x);
  shape=(0,0)--(0,bound[1].y+2*dx)--
    (0,bound[1].y+4*dx+bound[0].y)--
    (width+2*dx,bound[1].y+4*dx+bound[0].y)--
    (width+2*dx,bound[1].y+2*dx)--
    (width+2*dx,0)--cycle;

  flowblock block;
  block.draw=new picture(pen p) {
      picture block;
      filldraw(block,shift(0,bound[1].y+2*dx)*
	       ((0,0)--
		(0,bound[0].y+2*dx)--
		(width+2*dx,bound[0].y+2*dx)--
		(width+2*dx,0)--cycle),headercolor);
      add(block,shift(width/2,bound[0].y+bound[1].y+dx)*header);
      draw(block,(0,0)--(0,bound[1].y+2*dx)--
	   (width+2*dx,bound[1].y+2*dx)--
	   (width+2*dx,0)--cycle,bodycolor);
      add(block,shift(width/2,bound[0].y/2+dx)*body);
      return block;
    };
  block.f_position=new pair(real x) {
      return point(shape,x);
    };
  block.f_center=new pair() {
      return interp(point(shape,0),point(shape,3),0.5);
    };
  block.f_top=new pair() {
      return point(shape,2.5); 
    }; 
  block.f_left=new pair() {
      return point(shape,0.5); 
    };
  block.f_right=new pair() {
      return point(shape,4.5); 
    };
  block.f_bottom=new pair() {
      return point(shape,5.5); 
    };
  block.f_topleft=new pair() {
      return point(shape,2.0); 
    };
  block.f_topright=new pair() {
      return point(shape,3.0); 
    };
  block.f_bottomleft=new pair() {
      return point(shape,0.0); 
    };
  block.f_bottomright=new pair() {
      return point(shape,5.0); 
    };
  block.center(center);
  block.bound=point(shape,3);
  return block;
}

// As above, but without the header.
flowblock flowrectangle(picture body, pen bodycolor=currentpen, 
                        pair center=(0,0), real dx=3) 
{
  pair bound;
  path shape;
  bound=max(body.fit())-min(body.fit());
  shape=(0,0)--(0,bound.y+2*dx)--
    (bound.x+2*dx,bound.y+2*dx)--
    (bound.x+2*dx,0)--cycle;

  flowblock block;
  block.draw=new picture(pen p) {
      picture block;
      draw(block,(0,0)--(0,bound.y+2*dx)--
	   (bound.x+2*dx,bound.y+2*dx)--
	   (bound.x+2*dx,0)--cycle,bodycolor);
      add(block,shift(bound.x/2,bound.y/2)*body);
      return block;
    };
  block.f_position=new pair(real x) {
      return point(shape,x);
    };
  block.f_center=new pair() {
      return(dx+bound.x/2,dx+bound.y/2);
    };
  block.center(center);
  block.bound=(2*dx+bound.x,2*dx+bound.y);
  block.f_top=new pair() {
      return point(shape,1.5); 
    };
  block.f_left=new pair() {
      return point(shape,0.5); 
    };
  block.f_right=new pair() {
      return point(shape,2.5); 
    };
  block.f_bottom=new pair() {
      return point(shape,3.5); 
    };
  block.f_topleft=new pair() {
      return point(shape,1.0); 
    };
  block.f_topright=new pair() {
      return point(shape,2.0); 
    };
  block.f_bottomleft=new pair() {
      return point(shape,0.0); 
    };
  block.f_bottomright=new pair() {
      return point(shape,3.0); 
    };
  return block;
}

flowblock flowdiamond(picture pic, pair center=(0,0), 
                      real ds=5,
                      real dw=1,
                      real height=20,
                      real dh=0)
{
  pair bound;
  path shape;
  real a,b,c,d,e,m;
  
  bound=max(pic.fit())-min(pic.fit());
  
  e=ds;
  a=(bound.x-dw*2)/2;
  b=bound.y/2;
  c=b+height;
  m=(a*c-a*b-e*sqrt(-e*e+c*c-2*b*c+b*b+a*a))/(e*e-a*a);
  d=abs(c/m);

  shape=(0,c)--(d,2*c)--(2*d,c)--(d,0)--cycle;

  flowblock block;
  block.draw=new picture(pen p) {
      picture block;
      draw(block,shape);
      add(block,shift(d,c)*pic);
      return block;
    };
  block.f_position=new pair(real x) {
      return point(shape,x);
    };
  block.f_center=new pair() {
      return(point(shape,1).x,point(shape,0).y);
    };
  block.center(center);
  block.bound=(point(shape,2).x,point(shape,1).y);
  block.f_top=new pair() {
      return point(shape,1.0); 
    }; 
  block.f_left=new pair() {
      return point(shape,0.0); 
    };
  block.f_right=new pair() {
      return point(shape,2.0); 
    };
  block.f_bottom=new pair() {
      return point(shape,3.0); 
    };
  block.f_topleft=new pair() {
      return point(shape,0.5); 
    };
  block.f_topright=new pair() {
      return point(shape,1.5); 
    };
  block.f_bottomleft=new pair() {
      return point(shape,3.5); 
    };
  block.f_bottomright=new pair() {
      return point(shape,2.5); 
    };
  return block;
}

flowblock flowcircle(picture pic, pair center=(0,0), real dr=0)
{
  real r;
  path shape;
  pair bound;
  
  bound=max(pic.fit())-min(pic.fit());
  r=length(bound)/2+dr;
  
  shape=(0,r)..(r,2*r)..(2*r,r)..(r,0)..cycle;
  
  flowblock block;
  block.draw=new picture(pen p) {
      picture block;
      draw(block,shape);
      add(block,shift(bound/2)*shift(0,r)*
	  shift(0,(min(pic.fit())-max(pic.fit())).y/2)*pic);
      return block;
    };
  block.f_position=new pair(real x) {
      return point(shape,x);
    };
  block.f_center=new pair() {
      return(r,r);
    };
  block.center(center);
  block.bound=(2r,2r);
  block.f_top=new pair() {
      return point(shape,1.0); 
    };
  block.f_left=new pair() {
      return point(shape,0.0); 
    };
  block.f_right=new pair() {
      return point(shape,2.0); 
    };
  block.f_bottom=new pair() {
      return point(shape,3.0); 
    };
  block.f_topleft=new pair() {
      return point(shape,0.5); 
    };
  block.f_topright=new pair() {
      return point(shape,1.5); 
    };
  block.f_bottomleft=new pair() {
      return point(shape,3.5); 
    };
  block.f_bottomright=new pair() {
      return point(shape,2.5); 
    };
  return block;
}

flowblock flowroundrectangle(picture pic, pair center=(0,0),
                             real ds=5, real dw=0)
{
  real a,b;
  path shape;
  pair bound;
  
  bound=max(pic.fit())-min(pic.fit());
  a=bound.x;
  b=bound.y;
  
  shape=(0,ds+dw)--(0,ds+b-dw){up}..
    {right}(ds+dw,2*ds+b)--(ds+a-dw,2*ds+b){right}..
	     {down}(2*ds+a,ds+b-dw)--(2*ds+a,ds+dw){down}..
		     {left}(ds+a-dw,0)--
			     (ds+dw,0){left}..{up}cycle;
  
  flowblock block;
  block.draw=new picture(pen p) {
      picture block;
      draw(block,shape);
      add(block,shift(ds,ds)*shift(bound/2)*pic);
      return block;
    };
  block.f_position=new pair(real x) {
      return point(shape,x);
    };
  block.f_center=new pair() {
      return(ds+a/2,ds+b/2);
    };
  block.center(center);
  block.bound=(2*ds+a,2*ds+b);
  block.f_top=new pair() {
      return point(shape,2.5); 
    };
  block.f_left=new pair() {
      return point(shape,0.5); 
    };
  block.f_right=new pair() {
      return point(shape,4.5); 
    };
  block.f_bottom=new pair() {
      return point(shape,6.5); 
    };
  block.f_topleft=new pair() {
      return point(shape,1.5); 
    };
  block.f_topright=new pair() {
      return point(shape,3.5); 
    };
  block.f_bottomleft=new pair() {
      return point(shape,7.5); 
  };
  block.f_bottomright=new pair() {
      return point(shape,5.5); 
    };
  return block;
}

flowblock flowbevel(picture pic, pair center=(0,0), real dh=5, real dw=5)
{
  real a,b;
  path shape;
  pair bound;
  
  bound=max(pic.fit())-min(pic.fit());
  a=bound.x;
  b=bound.y/2;

  shape=(0,b+dh)--(dw,2*b+2*dh)--(dw+a,2*b+2*dh)--
    (2*dw+a,b+dh)--(dw+a,0)--(dw,0)--cycle;
  flowblock block;
  block.draw=new picture(pen p) {
      picture block;
      draw(block,shape);
      add(block,shift(bound/2+(dw,dh))*pic);
      return block;
    };
  block.f_position=new pair(real x) {
      return point(shape,x);
    };
  block.f_center=new pair() {
      return(dw+a/2,dh+b);
    };
  block.center(center);
  block.bound=(2*dw+a,2*dh+2*b);
  block.f_top=new pair() {
      return point(shape,1.5); 
    };
  block.f_left=new pair() {
      return point(shape,0.0); 
    };
  block.f_right=new pair() {
      return point(shape,3.0); 
    };
  block.f_bottom=new pair() {
      return point(shape,4.5); 
    };
  block.f_topleft=new pair() {
      return point(shape,0.5); 
    };
  block.f_topright=new pair() {
      return point(shape,2.5); 
    };
  block.f_bottomleft=new pair() {
      return point(shape,5.5); 
    };
  block.f_bottomright=new pair() {
      return point(shape,3.5); 
    };
  return block;
}

path flowpath(pair point[], bool horizontal[] )
{
  path line=point[0];
  pair current, prev=point[0];
  for(int i=1; i < point.length; ++i) {
    if(horizontal[i-1])
      current=(point[i].x,point[i-1].y);
    else 
      current=(point[i-1].x,point[i].y);
    
    if(current != prev) {
      line=line--current;
      prev=current;
    }
    
    current=point[i];
    if(current != prev) {
      line=line--current;
      prev=current;
    }
  }
  return line;
}

void drawflow(picture pic=currentpicture, path flow, 
              picture text=new picture, real pos=0.5, 
              pair align=(0,0), pen p=currentpen)
{
  frame f=text.fit();
  pair center=(min(f)+max(f))/2,
    textloc=((1.2align.x-1)*center.x,(1.2align.y-1)*center.y)
    +point(flow,pos*length(flow));
  draw(pic,flow,p,Arrow);
  add(pic,shift(textloc)*text);
}

picture picture(string text, real dy=0.2cm, 
		real angle=0, pair align=0, pen p=currentpen)
{
  Label tl;
  frame f;
  tl.label(f,(0,0),align);
  picture block;
  tl.init(text,-min(f),align,p,(real)angle);
  label(block,tl);
  return block;
}

void draw(picture pic=currentpicture, flowblock block, pen p=currentpen)
{
  add(pic,shift(block.llcorner)*block.draw(p));
}
