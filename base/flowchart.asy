// Flowchart routines written by Jacques Pienaar and Steve Melenchuk.

private import math;

public real flowmargin;

// N.B.: Relative coordinates take the lower left corner of the block: (0,0)

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
  pair bound0=max(header.fit())-min(header.fit());
  pair bound1=max(body.fit())-min(body.fit());
  real width=max(bound0.x,bound1.x);
  pair z0=(width+2dx,bound0.y+2dx);
  pair z1=(width+2dx,bound1.y+2dx);
  path shape=(0,0)--(0,z1.y)--(0,z0.y+z1.y)--(z0.x,z0.y+z1.y)--z1--(z0.x,0)--
    cycle;

  flowblock block;
  block.draw=new picture(pen p) {
      picture block;
      filldraw(block,shift(0,z1.y)*box((0,0),z0),headercolor);
      add(block,shift((0,z1.y)+0.5z0)*header);
      draw(block,box((0,0),z1));
      add(block,shift(0.5*z1)*body);
      return block;
    };
  block.f_position=new pair(real x) {
      return point(shape,x);
    };
  block.f_center=new pair() {
      return interp(point(shape,0),point(shape,3),0.5);
    };
  block.f_bottomleft=new pair() {
      return point(shape,0); 
    };
  block.f_bottom=new pair() {
      return point(shape,5.5); 
    };
  block.f_bottomright=new pair() {
      return point(shape,5); 
    };
  block.f_right=new pair() {
      return point(shape,4.5); 
    };
  block.f_topright=new pair() {
      return point(shape,3); 
    };
  block.f_top=new pair() {
      return point(shape,2.5); 
    }; 
  block.f_topleft=new pair() {
      return point(shape,2); 
    };
  block.f_left=new pair() {
      return point(shape,0.5); 
    };
  block.center(center);
  block.bound=point(shape,3);
  return block;
}

// As above, but without the header.
flowblock flowrectangle(picture body, pen bodycolor=currentpen, 
                        pair center=(0,0), real dx=3) 
{
  pair bound=max(body.fit())-min(body.fit());
  pair z=(bound.x+2dx,bound.y+2dx);
  path shape=box((0,0),z);

  flowblock block;
  block.draw=new picture(pen p) {
      picture block;
      draw(block,shape,bodycolor);
      add(block,shift(0.5*z)*body);
      return block;
    };
  block.f_position=new pair(real x) {
      return point(shape,x);
    };
  block.f_center=new pair() {
      return 0.5*z;
    };
  block.center(center);
  block.bound=z;
  block.f_bottomleft=new pair() {
      return point(shape,0);
    };
  block.f_bottom=new pair() {
      return point(shape,0.5); 
    };
  block.f_bottomright=new pair() {
      return point(shape,1); 
    };
  block.f_right=new pair() {
      return point(shape,1.5); 
    };
  block.f_topright=new pair() {
      return point(shape,2); 
    };
  block.f_top=new pair() {
      return point(shape,2.5); 
    };
  block.f_topleft=new pair() {
      return point(shape,3); 
    };
  block.f_left=new pair() {
      return point(shape,3.5); 
    };
  return block;
}

flowblock flowdiamond(picture pic, pair center=(0,0), 
                      real ds=5,
                      real dw=1,
                      real height=20,
                      real dh=0)
{
  real a,b,c,d,e,m;
  
  pair bound=max(pic.fit())-min(pic.fit());
  
  e=ds;
  a=0.5(bound.x-dw*2);
  b=0.5bound.y;
  c=b+height;
  m=(a*c-a*b-e*sqrt(-e*e+c*c-2b*c+b*b+a*a))/(e*e-a*a);
  d=abs(c/m);

  path shape=(2d,c)--(d,2c)--(0,c)--(d,0)--cycle;

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
  block.f_bottomleft=new pair() {
      return point(shape,2.5); 
    };
  block.f_bottom=new pair() {
      return point(shape,3); 
    };
  block.f_bottomright=new pair() {
      return point(shape,3.5); 
    };
  block.f_right=new pair() {
      return point(shape,0); 
    };
  block.f_topright=new pair() {
      return point(shape,0.5); 
    };
  block.f_top=new pair() {
      return point(shape,1); 
    }; 
  block.f_topleft=new pair() {
      return point(shape,1.5); 
    };
  block.f_left=new pair() {
      return point(shape,2); 
    };
  return block;
}

flowblock flowcircle(picture pic, pair center=(0,0), real dr=3)
{
  real r;
  
  pair bound=max(pic.fit())-min(pic.fit());
  r=0.5length(bound)+dr;
  
  path shape=(0,r)..(r,2r)..(2r,r)..(r,0)..cycle;
  
  flowblock block;
  block.draw=new picture(pen p) {
      picture block;
      draw(block,shape);
      add(block,shift((r,r))*pic);
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
  block.f_left=new pair() {
      return point(shape,0);
    };
  block.f_topleft=new pair() {
      return point(shape,0.5); 
    };
  block.f_top=new pair() {
      return point(shape,1); 
    };
  block.f_topright=new pair() {
      return point(shape,1.5); 
    };
  block.f_right=new pair() {
      return point(shape,2); 
    };
  block.f_bottomright=new pair() {
      return point(shape,2.5); 
    };
  block.f_bottom=new pair() {
      return point(shape,3); 
    };
  block.f_bottomleft=new pair() {
      return point(shape,3.5); 
    };
  return block;
}

flowblock flowroundrectangle(picture pic, pair center=(0,0),
                             real ds=5, real dw=0)
{
  real a,b;
  
  pair bound=max(pic.fit())-min(pic.fit());
  a=bound.x;
  b=bound.y;
  
  path shape=(0,ds+dw)--(0,ds+b-dw){up}..
    {right}(ds+dw,2ds+b)--(ds+a-dw,2ds+b){right}..
	     {down}(2ds+a,ds+b-dw)--(2ds+a,ds+dw){down}..
		     {left}(ds+a-dw,0)--
			     (ds+dw,0){left}..{up}cycle;
  
  flowblock block;
  block.draw=new picture(pen p) {
      picture block;
      draw(block,shape);
      add(block,shift(ds,ds)*shift(0.5bound)*pic);
      return block;
    };
  block.f_position=new pair(real x) {
      return point(shape,x);
    };
  block.f_center=new pair() {
      return(ds+0.5a,ds+0.5b);
    };
  block.center(center);
  block.bound=(2ds+a,2ds+b);
  block.f_bottomleft=new pair() {
      return point(shape,7.5); 
    };
  block.f_bottom=new pair() {
      return point(shape,6.5); 
    };
  block.f_bottomright=new pair() {
      return point(shape,5.5); 
    };
  block.f_right=new pair() {
      return point(shape,4.5); 
    };
  block.f_topright=new pair() {
      return point(shape,3.5); 
    };
  block.f_top=new pair() {
      return point(shape,2.5); 
    };
  block.f_topleft=new pair() {
      return point(shape,1.5); 
    };
  block.f_left=new pair() {
      return point(shape,0.5); 
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
  b=0.5bound.y;

  shape=(2dw+a,b+dh)--(dw+a,2b+2dh)--(dw,2b+2dh)--(0,b+dh)--(dw,0)--(dw+a,0)--
    cycle;
  flowblock block;
  block.draw=new picture(pen p) {
      picture block;
      draw(block,shape);
      add(block,shift(0.5bound+(dw,dh))*pic);
      return block;
    };
  block.f_position=new pair(real x) {
      return point(shape,x);
    };
  block.f_center=new pair() {
      return(dw+0.5a,dh+b);
    };
  block.center(center);
  block.bound=(2dw+a,2dh+2b);
  block.f_bottomleft=new pair() {
      return point(shape,4); 
    };
  block.f_bottom=new pair() {
      return point(shape,4.5); 
    };
  block.f_bottomright=new pair() {
      return point(shape,5); 
    };
  block.f_right=new pair() {
      return point(shape,0); 
    };
  block.f_topright=new pair() {
      return point(shape,1); 
    };
  block.f_top=new pair() {
      return point(shape,1.5); 
    };
  block.f_topleft=new pair() {
      return point(shape,2); 
    };
  block.f_left=new pair() {
      return point(shape,3); 
    };
  return block;
}

path flowpath(pair point[], bool horizontal[])
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

picture flow(picture text, path flow, real pos=0.5,
	     pair align=(0,0), pen p=currentpen, arrowbar arrow=Arrow,
	     margin margin=PenMargin)
{
  picture pic;
  draw(pic,flow,p,arrow,margin);
  add(pic,text.fit(),point(flow,pos),align);
  return pic;
}

picture picture(Label L, pair align=0, pen p=currentpen)
{
  picture block;
  L.align(align);
  L.p(p);
  L.out(block);
  return block;
}

void draw(picture pic=currentpicture, flowblock block, pen p=currentpen)
{
  add(pic,shift(block.llcorner)*block.draw(p));
}
