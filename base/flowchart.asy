// Flowchart routines written by Jacques Pienaar and Steve Melenchuk.

private import math;

real flowmargin;
bool Horizontal=true;
bool Vertical=false;

// N.B.: Relative coordinates take the lower left corner of the block: (0,0)

struct block {
  // The relative maximum coordinates of the block. 
  pair bound;

  // The absolute lower left corner of the block.
  pair llcorner;

  // Returns the relative center of the block.
  pair f_center();

  // Returns the absolute center of the block.
  pair center() {return shift(this.llcorner)*this.f_center();}
 
  // Returns the relative position along the boundary of the block.
  pair f_position(real x);

  // Returns the absolute position along the boundary of the block.
  pair position(real x) {return shift(this.llcorner)*this.f_position(x);}

  // These eight functions return the appropriate location on the block
  // in relative coordinates.
  pair f_top();
  pair f_left();
  pair f_right();
  pair f_bottom();
  pair f_topleft();
  pair f_topright();
  pair f_bottomleft();
  pair f_bottomright();

  // These eight functions return the appropriate location on the block
  // in absolute coordinates.
  pair top() {return shift(this.llcorner)*this.f_top();} 
  pair bottom() {return shift(this.llcorner)*this.f_bottom();} 
  pair left() {return shift(this.llcorner)*this.f_left();} 
  pair right() {return shift(this.llcorner)*this.f_right();} 
  pair topleft() {return shift(this.llcorner)*this.f_topleft();} 
  pair topright() {return shift(this.llcorner)*this.f_topright();} 
  pair bottomleft() {return shift(this.llcorner)*this.f_bottomleft();} 
  pair bottomright() {return shift(this.llcorner)*this.f_bottomright();} 
  
  // Center the block on the given coordinate.
  void center(pair loc) {this.llcorner=loc-this.f_center();} 
  
  // Return a frame representing the block.
  frame draw(pen p=currentpen);
};

block operator init() {return new block;}

// Construct a rectangular block with header and body objects.
block rectangle(object header, object body,
                pen headercolor=mediumgray,
                pen bodycolor=currentpen, 
                pair center=(0,0), real dx=3) 
{
  frame fbody=body.fit();
  frame fheader=header.fit();
  pair mheader=min(fheader);
  pair Mheader=max(fheader);
  pair mbody=min(fbody);
  pair Mbody=max(fbody);
  pair bound0=Mheader-mheader;
  pair bound1=Mbody-mbody;
  real width=max(bound0.x,bound1.x);
  pair z0=(width+2dx,bound0.y+2dx);
  pair z1=(width+2dx,bound1.y+2dx);
  path shape=(0,0)--(0,z1.y)--(0,z0.y+z1.y)--(z0.x,z0.y+z1.y)--z1--(z0.x,0)--
    cycle;

  block block;
  block.draw=new frame(pen p) {
    frame block;
    filldraw(block,shift(0,z1.y)*box((0,0),z0),headercolor);
    add(block,shift(-0.5*(Mheader+mheader))*fheader,(0,z1.y)+0.5z0);
    draw(block,box((0,0),z1));
    add(block,shift(-0.5*(Mbody+mbody))*fbody,0.5z1);
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
block rectangle(object body, pen bodycolor=currentpen, pair center=(0,0),
                real dx=3) 
{
  frame f=body.fit();
  pair m=min(f);
  pair M=max(f);
  pair bound=M-m;
  pair z=(bound.x+2dx,bound.y+2dx);
  path shape=box((0,0),z);

  block block;
  block.draw=new frame(pen p) {
    frame block;
    draw(block,shape,bodycolor);
    add(block,shift(-0.5*(M+m))*f,0.5z);
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

block diamond(object body, pair center=(0,0), real ds=5, real dw=1,
              real height=20, real dh=0)
{
  frame f=body.fit();
  pair m=min(f);
  pair M=max(f);
  pair bound=M-m;
  
  real e=ds;
  real a=0.5bound.x-dw;
  real b=0.5bound.y;
  real c=b+height;

  real arg=a^2+b^2+c^2-2b*c-e^2;
  real denom=e^2-a^2;
  real slope=arg >= 0 && denom != 0 ? (a*(c-b)-e*sqrt(arg))/denom : 1.0;
  real d=abs(c/slope);

  path shape=(2d,c)--(d,2c)--(0,c)--(d,0)--cycle;

  block block;
  block.draw=new frame(pen p) {
    frame block;
    draw(block,shape);
    add(block,shift(-0.5*(M+m))*f,(d,c));
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

block circle(object body, pair center=(0,0), real dr=3)
{
  frame f=body.fit();
  pair m=min(f);
  pair M=max(f);
  pair bound=M-m;
  real r=0.5length(bound)+dr;
  
  path shape=(0,r)..(r,2r)..(2r,r)..(r,0)..cycle;
  
  block block;
  block.draw=new frame(pen p) {
    frame block;
    draw(block,shape);
    add(block,shift(-0.5*(M+m))*f,(r,r));
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

block roundrectangle(object body, pair center=(0,0), real ds=5, real dw=0)
{
  frame f=body.fit();
  pair m=min(f);
  pair M=max(f);
  pair bound=M-m;

  real a=bound.x;
  real b=bound.y;
  
  path shape=(0,ds+dw)--(0,ds+b-dw){up}..
    {right}(ds+dw,2ds+b)--(ds+a-dw,2ds+b){right}..
             {down}(2ds+a,ds+b-dw)--(2ds+a,ds+dw){down}..
                     {left}(ds+a-dw,0)--(ds+dw,0){left}..{up}cycle;
  
                     block block;
                     block.draw=new frame(pen p) {
                       frame block;
                       draw(block,shape);
                       add(block,shift(-0.5*(M+m))*f,(ds,ds)+0.5bound);
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

block bevel(object body, pair center=(0,0), real dh=5, real dw=5)
{
  frame f=body.fit();
  pair m=min(f);
  pair M=max(f);
  pair bound=M-m;

  real a=bound.x;
  real b=0.5bound.y;

  path shape=(2dw+a,b+dh)--(dw+a,2b+2dh)--(dw,2b+2dh)--(0,b+dh)--(dw,0)--
    (dw+a,0)--cycle;
  block block;
  block.draw=new frame(pen p) {
    frame block;
    draw(block,shape);
    add(block,shift(-0.5*(M+m))*f,(0.5bound+(dw,dh)));
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

path path(pair point[] ... bool horizontal[])
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

void draw(picture pic=currentpicture, block block, pen p=currentpen)
{
  add(pic,shift(block.llcorner)*block.draw(p));
}
