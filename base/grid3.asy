// grid3.asy
// Author: Philippe Ivaldi (Grids in 3D)
// http://www.piprime.fr/
// Created: 10 janvier 2007

import graph3;

struct grid3 {
  path3 axea,axeb;
  bounds bds;
  triple dir;
  valuetime vt;
  ticklocate locate;
  void create(picture pic, path3 axea, path3 axeb, path3 axelevel,
              real min, real max, position pos, autoscaleT t) {
    real position=pos.position.x;
    triple level;
    if(pos.relative) {
      position=reltime(axelevel,position);
      level=point(axelevel,position)-point(axelevel,0);
    } else {
      triple v=unit(point(axelevel,1)-point(axelevel,0));
      triple zerolevel=dot(-point(axelevel,0),v)*v;
      level=zerolevel+position*v;
    }
    this.axea=shift(level)*axea;
    this.axeb=shift(level)*axeb;
    bds=autoscale(min,max,t.scale);
    locate=ticklocate(min,max,t,bds.min,bds.max,
                      Dir(point(axeb,0)-point(axea,0)));
  }
};

typedef grid3 grid3routine(picture pic);

triple X(picture pic) {return (pic.userMax().x,pic.userMin().y,pic.userMin().z);}
triple XY(picture pic) {return (pic.userMax().x,pic.userMax().y,pic.userMin().z);}
triple Y(picture pic) {return (pic.userMin().x,pic.userMax().y,pic.userMin().z);}
triple YZ(picture pic) {return (pic.userMin().x,pic.userMax().y,pic.userMax().z);}
triple Z(picture pic) {return (pic.userMin().x,pic.userMin().y,pic.userMax().z);}
triple ZX(picture pic) {return (pic.userMax().x,pic.userMin().y,pic.userMax().z);}

grid3routine XYgrid(position pos=Relative(0)) {
  return new grid3(picture pic) {
    grid3 og;
    triple m=pic.userMin();
    triple M=pic.userMax();
    og.create(pic,m--X(pic),Y(pic)--XY(pic),m--Z(pic),
              m.x,M.x,pos,pic.scale.x);
    return og;
  };
};
grid3routine XYgrid=XYgrid();

grid3routine YXgrid(position pos=Relative(0)) {
  return new grid3(picture pic) {
    grid3 og;
    triple m=pic.userMin();
    triple M=pic.userMax();
    og.create(pic,m--Y(pic),X(pic)--XY(pic),m--Z(pic),
              m.y,M.y,pos,pic.scale.y);
    return og;
  };
};
grid3routine YXgrid=YXgrid();


grid3routine XZgrid(position pos=Relative(0)) {
  return new grid3(picture pic) {
    grid3 og;
    triple m=pic.userMin();
    triple M=pic.userMax();
    og.create(pic,m--X(pic),Z(pic)--ZX(pic),m--Y(pic),
              m.x,M.x,pos,pic.scale.x);
    return og;
  };
};
grid3routine XZgrid=XZgrid();

grid3routine ZXgrid(position pos=Relative(0)) {
  return new grid3(picture pic) {
    grid3 og;
    triple m=pic.userMin();
    triple M=pic.userMax();
    og.create(pic,m--Z(pic),X(pic)--ZX(pic),m--Y(pic),
              m.z,M.z,pos,pic.scale.z);
    return og;
  };
};
grid3routine ZXgrid=ZXgrid();

grid3routine YZgrid(position pos=Relative(0)) {
  return new grid3(picture pic) {
    grid3 og;
    triple m=pic.userMin();
    triple M=pic.userMax();
    og.create(pic,m--Y(pic),Z(pic)--YZ(pic),m--X(pic),
              m.y,M.y,pos,pic.scale.y);
    return og;
  };
};
grid3routine YZgrid=YZgrid();

grid3routine ZYgrid(position pos=Relative(0)) {
  return new grid3(picture pic) {
    grid3 og;
    triple m=pic.userMin();
    triple M=pic.userMax();
    og.create(pic,m--Z(pic),Y(pic)--YZ(pic),m--X(pic),
              m.z,M.z,pos,pic.scale.z);
    return og;
  };
};
grid3routine ZYgrid=ZYgrid();

typedef grid3routine grid3routines[] ;

grid3routines XYXgrid(position pos=Relative(0)) {
  grid3routines ogs=new grid3routine[] {XYgrid(pos),YXgrid(pos)};
  return ogs;
};
grid3routines XYXgrid=XYXgrid();
grid3routines YXYgrid(position pos=Relative(0)) {return XYXgrid(pos);};
grid3routines YXYgrid=XYXgrid();

grid3routines ZXZgrid(position pos=Relative(0)) {
  grid3routines ogs=new grid3routine[] {ZXgrid(pos),XZgrid(pos)};
  return ogs;
};
grid3routines ZXZgrid=ZXZgrid();
grid3routines XZXgrid(position pos=Relative(0)) {return ZXZgrid(pos);};
grid3routines XZXgrid=XZXgrid();

grid3routines ZYZgrid(position pos=Relative(0)) {
  grid3routines ogs=new grid3routine[] {ZYgrid(pos),YZgrid(pos)};
  return ogs;
};
grid3routines ZYZgrid=ZYZgrid();
grid3routines YZYgrid(position pos=Relative(0)) {return ZYZgrid(pos);};
grid3routines YZYgrid=YZYgrid();

grid3routines XY_XZgrid(position posa=Relative(0), position posb=Relative(0)) {
  grid3routines ogs=new grid3routine[] {XYgrid(posa),XZgrid(posb)};
  return ogs;
};
grid3routines XY_XZgrid=XY_XZgrid();

grid3routines YX_YZgrid(position posa=Relative(0), position posb=Relative(0)) {
  grid3routines ogs=new grid3routine[] {YXgrid(posa),YZgrid(posb)};
  return ogs;
};
grid3routines YX_YZgrid=YX_YZgrid();

grid3routines ZX_ZYgrid(position posa=Relative(0), position posb=Relative(0)) {
  grid3routines ogs=new grid3routine[] {ZXgrid(posa),ZYgrid(posb)};
  return ogs;
};
grid3routines ZX_ZYgrid=ZX_ZYgrid();

typedef grid3routines[] grid3routinetype;

grid3routinetype XYZgrid(position pos=Relative(0))
{
  grid3routinetype ogs=new grid3routines[] {YZYgrid(pos),XYXgrid(pos),
                                            XZXgrid(pos)};
  return ogs;
}
grid3routinetype XYZgrid=XYZgrid();

grid3routines operator cast(grid3routine gridroutine) {
  grid3routines og=new grid3routine[] {gridroutine};
  return og;
}

grid3routinetype operator cast(grid3routines gridroutine) {
  grid3routinetype og=new grid3routines[] {gridroutine};
  return og;
}

grid3routinetype operator cast(grid3routine gridroutine) {
  grid3routinetype og=(grid3routinetype)(grid3routines) gridroutine;
  return og;
}

void grid3(picture pic=currentpicture,
           grid3routinetype gridroutine=XYZgrid,
           int N=0, int n=0, real Step=0, real step=0,
           bool begin=true, bool end=true,
           pen pGrid=grey, pen pgrid=lightgrey,
           bool above=false)
{
  for(int j=0; j < gridroutine.length; ++j) {
    grid3routines gridroutinej=gridroutine[j];
    for(int i=0; i < gridroutinej.length; ++i) {
      grid3 gt=gridroutinej[i](pic);
      pic.add(new void(picture f, transform3 t, transform3 T, triple, triple) {
          picture d;
          ticks3 ticks=Ticks3(1,F="%",ticklabel=null,
                              beginlabel=false,endlabel=false,
                              N=N,n=n,Step=Step,step=step,
                              begin=begin,end=end,
                              Size=0,size=0,extend=true,
                              pTick=pGrid,ptick=pgrid);
          ticks(d,t,"",gt.axea,gt.axeb,nullpen,None,NoMargin3,gt.locate,
                gt.bds.divisor,opposite=true,primary=false);
          add(f,t*T*inverse(t)*d);
        },above=above);
      addPath(pic,gt.axea,pGrid);
      addPath(pic,gt.axeb,pGrid);
    }
  }
}

void grid3(picture pic=currentpicture,
           grid3routinetype gridroutine,
           int N=0, int n=0, real Step=0, real step=0,
           bool begin=true, bool end=true,
           pen[] pGrid, pen[] pgrid,
           bool above=false)
{
  if(pGrid.length != gridroutine.length || pgrid.length != gridroutine.length)
    abort("pen array has different length than grid");
  for(int i=0; i < gridroutine.length; ++i) {
    grid3(pic=pic,gridroutine=gridroutine[i],
          N=N,n=n,Step=Step,step=step,
          begin=begin,end=end,
          pGrid=pGrid[i],pgrid=pgrid[i],
          above=above);
  }
}

position top=Relative(1);
position bottom=Relative(0);
position middle=Relative(0.5);

// Structure used to communicate ticks and axis settings to grid3 routines.
struct ticksgridT {
  ticks3 ticks;
  // Other arguments of grid3 are define by ticks and axis settings
  void grid3(picture, bool);
};

typedef ticksgridT ticksgrid();


ticksgrid InOutTicks(Label F="", ticklabel ticklabel=null,
                     bool beginlabel=true, bool endlabel=true,
                     int N=0, int n=0, real Step=0, real step=0,
                     bool begin=true, bool end=true,
                     real Size=0, real size=0,
                     pen pTick=nullpen, pen ptick=nullpen,
                     grid3routinetype gridroutine,
                     pen pGrid=grey, pen pgrid=lightgrey)
{
  return new ticksgridT()
    {
      ticksgridT otg;
      otg.ticks=Ticks3(0,F,ticklabel,beginlabel,endlabel,
                       N,n,Step,step,begin,end,
                       Size,size,false,pTick,ptick);
      otg.grid3=new void(picture pic, bool above) {
        grid3(pic,gridroutine,N,n,Step,step,begin,end,pGrid,pgrid,above);
      };
      return otg;
    };
}

ticksgrid InTicks(Label F="", ticklabel ticklabel=null,
                  bool beginlabel=true, bool endlabel=true,
                  int N=0, int n=0, real Step=0, real step=0,
                  bool begin=true, bool end=true,
                  real Size=0, real size=0,
                  pen pTick=nullpen, pen ptick=nullpen,
                  grid3routinetype gridroutine,
                  pen pGrid=grey, pen pgrid=lightgrey)
{
  return new ticksgridT()
    {
      ticksgridT otg;
      otg.ticks=Ticks3(-1,F,ticklabel,beginlabel,endlabel,N,n,Step,step,
                       begin,end,Size,size,false,pTick,ptick);
      otg.grid3=new void(picture pic, bool above) {
        grid3(pic,gridroutine,N,n,Step,step,begin,end,pGrid,pgrid,above);
      };
      return otg;
    };
}

ticksgrid OutTicks(Label F="", ticklabel ticklabel=null,
                   bool beginlabel=true, bool endlabel=true,
                   int N=0, int n=0, real Step=0, real step=0,
                   bool begin=true, bool end=true,
                   real Size=0, real size=0,
                   pen pTick=nullpen, pen ptick=nullpen,
                   grid3routinetype gridroutine,
                   pen pGrid=grey, pen pgrid=lightgrey)
{
  return new ticksgridT()
    {
      ticksgridT otg;
      otg.ticks=Ticks3(1,F,ticklabel,beginlabel,endlabel,N,n,Step,step,
                       begin,end,Size,size,false,pTick,ptick);
      otg.grid3=new void(picture pic, bool above) {
        grid3(pic,gridroutine,N,n,Step,step,begin,end,pGrid,pgrid,above);
      };
      return otg;
    };
}

void xaxis3(picture pic=currentpicture, Label L="", axis axis=YZZero,
            pen p=currentpen, ticksgrid ticks,
            arrowbar3 arrow=None, bool above=false)
{
  xaxis3(pic,L,axis,p,ticks().ticks,arrow,above);
  ticks().grid3(pic,above);
}

void yaxis3(picture pic=currentpicture, Label L="", axis axis=XZZero,
            pen p=currentpen, ticksgrid ticks,
            arrowbar3 arrow=None, bool above=false)
{
  yaxis3(pic,L,axis,p,ticks().ticks,arrow,above);
  ticks().grid3(pic,above);
}

void zaxis3(picture pic=currentpicture, Label L="", axis axis=XYZero,
            pen p=currentpen, ticksgrid ticks,
            arrowbar3 arrow=None, bool above=false)
{
  zaxis3(pic,L,axis,p,ticks().ticks,arrow,above);
  ticks().grid3(pic,above);
}

/* Example:

   import grid3;

   size(8cm,0);
   currentprojection=orthographic(0.5,1,0.5);

   defaultpen(overwrite(SuppressQuiet));

   scale(Linear, Linear, Log);

   grid3(pic=currentpicture, // picture
   gridroutine=XYZgrid(// grid3routine
   // or grid3routine[] (alias grid3routines)
   // or grid3routines[]:
   // The routine(s) to draw the grid(s):
   // *XYgrid: draw grid from X in direction of Y
   // *YXgrid: draw grid from Y in direction of X, ...
   // *An array of previous values XYgrid, YXgrid, ...
   // *XYXgrid: draw XYgrid and YXgrid grids
   // *YXYgrid: draw XYgrid and YXgrid grids
   // *ZXZgrid: draw ZXgrid and XZgrid grids
   // *YX_YZgrid: draw YXgrid and YZgrid grids
   // *XY_XZgrid: draw XYgrid and XZgrid grids
   // *YX_YZgrid: draw YXgrid and YZgrid grids
   // *An array of previous values XYXgrid,...
   // *XYZgrid: draw XYXgrid, ZYZgrid, XZXgrid grids.
   pos=Relative(0)),
   // the position of the grid relative to the axis
   // perpendicular to the grid; a real number
   // specifies a coordinate relative  to this axis.
   // Aliases: top=Relative(1), middle=Relative(0.5)
   // and bottom=Relative(0).

   // These arguments are similar to those of InOutTicks():
   N=0,                // int
   n=0,                // int
   Step=0,             // real
   step=0,             // real
   begin=true,         // bool
   end=true,           // bool
   pGrid=grey,         // pen
   pgrid=lightgrey,    // pen
   above=false         // bool
   );

   xaxis3(Label("$x$",position=EndPoint,align=S),OutTicks());
   yaxis3(Label("$y$",position=EndPoint,align=S),OutTicks());
   zaxis3(Label("$z$",position=EndPoint,align=(0,0.5)+W),OutTicks());
*/

/* Other examples:

   int N=10, n=2;
   xaxis3(Label("$x$",position=EndPoint,align=S),OutTicks());
   yaxis3(Label("$y$",position=EndPoint,align=S),OutTicks(N=N,n=n));
   zaxis3(Label("$z$",position=EndPoint,align=(0,0.5)+W),OutTicks());
   grid3(N=N,n=n);

   xaxis3(Label("$x$",position=EndPoint,align=S),OutTicks());
   yaxis3(Label("$y$",position=EndPoint,align=S),OutTicks());
   zaxis3(Label("$z$",position=EndPoint,align=(0,0.5)+W),OutTicks());
   grid3(new grid3routines[] {XYXgrid(top),XZXgrid(0)});

   xaxis3(Label("$x$",position=EndPoint,align=S),OutTicks());
   yaxis3(Label("$y$",position=EndPoint,align=S),OutTicks());
   zaxis3(Label("$z$",position=EndPoint,align=(0,0.5)+W),OutTicks());
   grid3(new grid3routines[] {XYXgrid(-0.5),XYXgrid(1.5)},
   pGrid=new pen[] {red,blue},
   pgrid=new pen[] {0.5red,0.5blue});

   // Axes with grids:

   xaxis3(Label("$x$",position=EndPoint,align=S),
   OutTicks(Step=0.5,gridroutine=XYgrid));
   yaxis3(Label("$y$",position=EndPoint,align=S),
   InOutTicks(Label("",align=0.5X),N=8,n=2,gridroutine=YX_YZgrid));
   zaxis3("$z$",OutTicks(ZYgrid));
*/
