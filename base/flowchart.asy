// Flowchart routines written by Jacques Pienaar, Steve Melenchuk, John Bowman.

private import math;

struct flowdir {}

restricted flowdir Horizontal;
restricted flowdir Vertical;

real minblockwidth=0;
real minblockheight=0;
real mincirclediameter=0;
real defaultexcursion=0.1;

struct block {
  // The absolute center of the block in user coordinates.
  pair center;

  // The size of the block
  pair size;

  // The relative center of the block.
  pair f_center;

  // These eight variables return the appropriate location on the block
  // in relative coordinates, where the lower left corner of the block is (0,0).
  pair f_top;
  pair f_left;
  pair f_right;
  pair f_bottom;
  pair f_topleft;
  pair f_topright;
  pair f_bottomleft;
  pair f_bottomright;

  void operator init(pair z) {
    center=z;
  }

  void operator init(real x, real y) {
    center=(x,y);
  }

  pair shift(transform t=identity()) {
    return t*center-f_center;
  }

  // Returns the relative position along the boundary of the block.
  pair f_position(real x);

  // Returns the absolute position along the boundary of the block.
  pair position(real x, transform t=identity()) {
    return shift(t)+f_position(x);
  }

  // These eight functions return the appropriate location on the block
  // in absolute coordinates.
  pair top(transform t=identity()) {
    return shift(t)+f_top;
  } 
  pair bottom(transform t=identity()) {
    return shift(t)+f_bottom;
  } 
  pair left(transform t=identity()) {
    return shift(t)+f_left;
  } 
  pair right(transform t=identity()) {
    return shift(t)+f_right;
  } 
  pair topleft(transform t=identity()) {
    return shift(t)+f_topleft;
  } 
  pair topright(transform t=identity()) {
    return shift(t)+f_topright;
  } 
  pair bottomleft(transform t=identity()) {
    return shift(t)+f_bottomleft;
  } 
  pair bottomright(transform t=identity()) {
    return shift(t)+f_bottomright;
  } 
  
  // Return a frame representing the block.
  frame draw(pen p=currentpen);

  // Store optional label on outgoing edge.
  Label label;

  // Store rectilinear path directions.
  pair[] dirs;

  // Store optional arrow.
  arrowbar arrow=None;
};

// Construct a rectangular block with header and body objects.
block rectangle(object header, object body, pair center=(0,0),
                pen headerpen=mediumgray, pen bodypen=invisible,
                pen drawpen=currentpen,
                real dx=3, real minheaderwidth=minblockwidth,
                real minheaderheight=minblockwidth,
                real minbodywidth=minblockheight,
                real minbodyheight=minblockheight)
{
  frame fbody=body.f;
  frame fheader=header.f;
  pair mheader=min(fheader);
  pair Mheader=max(fheader);
  pair mbody=min(fbody);
  pair Mbody=max(fbody);
  pair bound0=Mheader-mheader;
  pair bound1=Mbody-mbody;
  real width=max(bound0.x,bound1.x);
  pair z0=maxbound((width+2dx,bound0.y+2dx),(minbodywidth,minbodyheight));
  pair z1=maxbound((width+2dx,bound1.y+2dx),(minheaderwidth,minheaderheight));
  path shape=(0,0)--(0,z1.y)--(0,z0.y+z1.y)--(z0.x,z0.y+z1.y)--z1--(z0.x,0)--
    cycle;

  block block;
  block.draw=new frame(pen p) {
    frame block;
    filldraw(block,shift(0,z1.y)*box((0,0),z0),headerpen,drawpen);
    add(block,shift(-0.5*(Mheader+mheader))*fheader,(0,z1.y)+0.5z0);
    filldraw(block,box((0,0),z1),bodypen,drawpen);
    add(block,shift(-0.5*(Mbody+mbody))*fbody,0.5z1);
    return block;
  };
  block.f_position=new pair(real x) {
    return point(shape,x);
  };
  block.f_center=interp(point(shape,0),point(shape,3),0.5);
  block.f_bottomleft=point(shape,0); 
  block.f_bottom=point(shape,5.5); 
  block.f_bottomright=point(shape,5); 
  block.f_right=point(shape,4.5); 
  block.f_topright=point(shape,3); 
  block.f_top=point(shape,2.5); 
  block.f_topleft=point(shape,2); 
  block.f_left=point(shape,0.5); 
  block.center=center;
  block.size=point(shape,3);
  return block;
}

// As above, but without the header.
block rectangle(object body, pair center=(0,0),
                pen fillpen=invisible, pen drawpen=currentpen,
                real dx=3, real minwidth=minblockwidth,
                real minheight=minblockheight)
{
  frame f=body.f;
  pair m=min(f);
  pair M=max(f);
  pair z=maxbound(M-m+dx*(2,2),(minwidth,minheight));
  path shape=box((0,0),z);

  block block;
  block.draw=new frame(pen p) {
    frame block;
    filldraw(block,shape,fillpen,drawpen);
    add(block,shift(-0.5*(M+m))*f,0.5z);
    return block;
  };
  block.f_position=new pair(real x) {
    return point(shape,x);
  };
  block.f_center=0.5*z;
  block.center=center;
  block.size=z;
  block.f_bottomleft=point(shape,0);
  block.f_bottom=point(shape,0.5); 
  block.f_bottomright=point(shape,1); 
  block.f_right=point(shape,1.5); 
  block.f_topright=point(shape,2); 
  block.f_top=point(shape,2.5); 
  block.f_topleft=point(shape,3); 
  block.f_left=point(shape,3.5); 
  return block;
}

block parallelogram(object body, pair center=(0,0),
                    pen fillpen=invisible, pen drawpen=currentpen,
                    real dx=3, real slope=2,
                    real minwidth=minblockwidth,
                    real minheight=minblockheight)
{
  frame f=body.f;
  pair m=min(f);
  pair M=max(f);
  pair bound=maxbound(M-m+dx*(0,2),(minwidth,minheight));

  real skew=bound.y/slope;
  real a=bound.x+skew;
  real b=bound.y;

  path shape=(0,0)--(a,0)--(a+skew,b)--(skew,b)--cycle;

  block block;
  block.draw=new frame(pen p) {
    frame block;
    filldraw(block,shape,fillpen,drawpen);
    add(block,shift(-0.5*(M+m))*f,((a+skew)/2,b/2));
    return block;
  };
  block.f_position=new pair(real x) {
    return point(shape,x);
  };
  block.f_center=((a+skew)/2,b/2);
  block.center=center;
  block.size=(a+skew,b);
  block.f_bottomleft=(0,0);
  block.f_bottom=((a+skew)/2,0);
  block.f_bottomright=(a,0);
  block.f_right=(a+skew/2,b/2);
  block.f_topright=(a+skew,b);
  block.f_top=((a+skew)/2,b);
  block.f_topleft=(skew,b);
  block.f_left=(skew/2,b/2);
  return block;
}

block diamond(object body, pair center=(0,0),
              pen fillpen=invisible, pen drawpen=currentpen,
              real ds=5, real dw=1,
              real height=20, real minwidth=minblockwidth,
              real minheight=minblockheight)
{
  frame f=body.f;
  pair m=min(f);
  pair M=max(f);
  pair bound=maxbound(M-m,(minwidth,minheight));
  
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
    filldraw(block,shape,fillpen,drawpen);
    add(block,shift(-0.5*(M+m))*f,(d,c));
    return block;
  };
  block.f_position=new pair(real x) {
    return point(shape,x);
  };
  block.f_center=(point(shape,1).x,point(shape,0).y);
  block.center=center;
  block.size=(point(shape,0).x,point(shape,1).y);
  block.f_bottomleft=point(shape,2.5); 
  block.f_bottom=point(shape,3); 
  block.f_bottomright=point(shape,3.5); 
  block.f_right=point(shape,0); 
  block.f_topright=point(shape,0.5); 
  block.f_top=point(shape,1); 
  block.f_topleft=point(shape,1.5); 
  block.f_left=point(shape,2); 
  return block;
}

block circle(object body, pair center=(0,0), pen fillpen=invisible,
             pen drawpen=currentpen, real dr=3,
             real mindiameter=mincirclediameter)
{
  frame f=body.f;
  pair m=min(f);
  pair M=max(f);
  real r=max(0.5length(M-m)+dr,0.5mindiameter);
  
  path shape=(0,r)..(r,2r)..(2r,r)..(r,0)..cycle;
  
  block block;
  block.draw=new frame(pen p) {
    frame block;
    filldraw(block,shape,fillpen,drawpen);
    add(block,shift(-0.5*(M+m))*f,(r,r));
    return block;
  };
  block.f_position=new pair(real x) {
    return point(shape,x);
  };
  block.f_center=(r,r);
  block.center=center;
  block.size=(2r,2r);
  block.f_left=point(shape,0);
  block.f_topleft=point(shape,0.5); 
  block.f_top=point(shape,1); 
  block.f_topright=point(shape,1.5); 
  block.f_right=point(shape,2); 
  block.f_bottomright=point(shape,2.5); 
  block.f_bottom=point(shape,3); 
  block.f_bottomleft=point(shape,3.5); 
  return block;
}

block roundrectangle(object body, pair center=(0,0),
                     pen fillpen=invisible, pen drawpen=currentpen,
                     real ds=5, real dw=0, real minwidth=minblockwidth,
                     real minheight=minblockheight)
{
  frame f=body.f;
  pair m=min(f);
  pair M=max(f);
  pair bound=maxbound(M-m,(minwidth,minheight));

  real a=bound.x;
  real b=bound.y;
  
  path shape=(0,ds+dw)--(0,ds+b-dw){up}..{right}
  (ds+dw,2ds+b)--(ds+a-dw,2ds+b){right}..{down}
  (2ds+a,ds+b-dw)--(2ds+a,ds+dw){down}..{left}
  (ds+a-dw,0)--(ds+dw,0){left}..{up}cycle;
  
  block block;
  block.draw=new frame(pen p) {
    frame block;
    filldraw(block,shape,fillpen,drawpen);
    add(block,shift(-0.5*(M+m))*f,(ds,ds)+0.5bound);
    return block;
  };
  block.f_position=new pair(real x) {
    return point(shape,x);
  };
  block.f_center=(ds+0.5a,ds+0.5b);
  block.center=center;
  block.size=(2ds+a,2ds+b);
  block.f_bottomleft=point(shape,7.5); 
  block.f_bottom=point(shape,6.5); 
  block.f_bottomright=point(shape,5.5); 
  block.f_right=point(shape,4.5); 
  block.f_topright=point(shape,3.5); 
  block.f_top=point(shape,2.5); 
  block.f_topleft=point(shape,1.5); 
  block.f_left=point(shape,0.5); 
  return block;
}

block bevel(object body, pair center=(0,0), pen fillpen=invisible,
            pen drawpen=currentpen, real dh=5, real dw=5,
            real minwidth=minblockwidth, real minheight=minblockheight)
{
  frame f=body.f;
  pair m=min(f);
  pair M=max(f);
  pair bound=maxbound(M-m,(minwidth,minheight));

  real a=bound.x;
  real b=0.5bound.y;

  path shape=(2dw+a,b+dh)--(dw+a,2b+2dh)--(dw,2b+2dh)--(0,b+dh)--(dw,0)--
    (dw+a,0)--cycle;
  block block;
  block.draw=new frame(pen p) {
    frame block;
    filldraw(block,shape,fillpen,drawpen);
    add(block,shift(-0.5*(M+m))*f,(0.5bound+(dw,dh)));
    return block;
  };
  block.f_position=new pair(real x) {
    return point(shape,x);
  };
  block.f_center=(dw+0.5a,dh+b);
  block.center=center;
  block.size=(2dw+a,2dh+2b);
  block.f_bottomleft=point(shape,4); 
  block.f_bottom=point(shape,4.5); 
  block.f_bottomright=point(shape,5); 
  block.f_right=point(shape,0); 
  block.f_topright=point(shape,1); 
  block.f_top=point(shape,1.5); 
  block.f_topleft=point(shape,2); 
  block.f_left=point(shape,3); 
  return block;
}

path path(pair point[] ... flowdir dir[])
{
  path line=point[0];
  pair current, prev=point[0];
  for(int i=1; i < point.length; ++i) {
    if(i-1 >= dir.length || dir[i-1] == Horizontal)
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
  pic.add(new void(frame f, transform t) {
      add(f,shift(block.shift(t))*block.draw(p));
    },true);
  pic.addBox(block.center,block.center,
             -0.5*block.size+min(p),0.5*block.size+max(p));
}

typedef block blockconnector(block, block);

blockconnector blockconnector(picture pic, transform t, pen p=currentpen,
                              margin margin=PenMargin)
{
  return new block(block b1, block b2) {
    if(b1.dirs.length == 0) {
      if(abs(b1.center.y-b2.center.y) < sqrtEpsilon) {
        // horizontally aligned
        b1.dirs[0]=b1.center.x < b2.center.x ? right : left;
        blockconnector(pic,t,p,margin)(b1,b2);
      } else if(abs(b1.center.x-b2.center.x) < sqrtEpsilon) {
        // vertically aligned
        b1.dirs[0]=b1.center.y < b2.center.y ? up : down;
        blockconnector(pic,t,p,margin)(b1,b2);
      } else {
        if(abs(b1.center.y-b2.center.y) < abs(b1.center.x-b2.center.x)) {
          b1.dirs[0]=b1.center.x < b2.center.x ? right : left;
          b1.dirs[1]=b1.center.y < b2.center.y ? up : down;
          blockconnector(pic,t,p,margin)(b1,b2);
        } else {
          b1.dirs[0]=b1.center.y < b2.center.y ? up : down;
          b1.dirs[1]=b1.center.x < b2.center.x ? right : left;
          blockconnector(pic,t,p,margin)(b1,b2);
        }
      }
      return b2;
    }

    // compute the link for given directions (and label if any)
    pair[] dirs=copy(b1.dirs); // deep copy
    pair current,prev;
    pair dir=dirs[0];
    if(dir == up) prev=b1.top(t);
    if(dir == down) prev=b1.bottom(t);
    if(dir == left) prev=b1.left(t);
    if(dir == right) prev=b1.right(t);
    path line=prev;
    arrowbar arrow=b1.arrow;

    int i;
    for(i=1; i < dirs.length-1; ++i) {
      if(abs(length(dirs[i-1])-1) < sqrtEpsilon)
        current=prev+t*dirs[i-1]*defaultexcursion;
      else
        current=prev+t*dirs[i-1];

      if(current != prev) {
        line=line--current;
        prev=current;
      }
    }
    dir=dirs[dirs.length-1];
    current=0;
    if(dir == up) current=b2.bottom(t);
    if(dir == down) current=b2.top(t);
    if(dir == left) current=b2.right(t);
    if(dir == right) current=b2.left(t);
    if(abs(dirs[i-1].y) < sqrtEpsilon &&
       abs(prev.x-current.x) > sqrtEpsilon) {
      prev=(current.x,prev.y);
      line=line--prev; // horizontal
    } else if(abs(dirs[i-1].x) < sqrtEpsilon &&
              abs(prev.y-current.y) > sqrtEpsilon) {
      prev=(prev.x,current.y);
      line=line--prev;
    }
    if(current != prev)
      line=line--current;

    draw(pic,b1.label,line,p,arrow,margin);

    b1.label="";
    b1.dirs.delete();
    b1.arrow=None;
    return b2;
  };
}

struct Dir
{
  pair z;
  void operator init(pair z) {this.z=z;}
}

Dir Right=Dir(right);
Dir Left=Dir(left);
Dir Up=Dir(up);
Dir Down=Dir(down);

// Add a label to the current link
block operator --(block b1, Label label)
{
  b1.label=label;
  return b1;
}

// Add a direction to the current link
block operator --(block b1, Dir dir)
{
  b1.dirs.push(dir.z);
  return b1;
}

// Add an arrowbar to the current link
block operator --(block b, arrowbar arrowbar)
{
  b.arrow=arrowbar;
  return b;
}
