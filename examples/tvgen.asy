/* tvgen - draw pm5544-like television test cards.
 * Copyright (C) 2007, 2009, 2012, Servaas Vandenberghe.
 *
 * The tvgen code below is free software: you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with tvgen: see the file COPYING.  If not, write to the
 * Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
 * Boston, MA 02110-1301, USA.
 *
 * tvgen-1.2/tvgen.asy  http://picaros.org/ftp/soft/tvgen-1.2.tgz
 * This asy script generates pm5544-like television test cards.  The image 
 * parameters were derived from a 1990 recording.  The basic parameters 
 * conform to itu-r bt.470, bt.601, and bt.709.  There is no unique image 
 * since local variants exist and parameters have varied over time.  
 */
//papertype="a4";
import plain;
int verbose=settings.verbose/*+2*/;  /* uncomment for debug info */

/* tv dot coordinates --> PS points */
pair tvps(real col, real row, real xd, real yd, int Nv) { 
  real psx, psy; 
  psx=col*xd; psy=(Nv-row)*yd; 
  return (psx,psy); 
}
path tvrect(int lc, int tr, int rc, int br, real xd, real yd, int Nv) {
  real lx, ty, rx, by; 
  pair[] z; 
  
  lx=lc*xd; ty=(Nv-tr)*yd; 
  rx=rc*xd; by=(Nv-br)*yd; 
  /* bl br tr tl */
  z[0]=(lx, by);
  z[1]=(rx, by); 
  z[2]=(rx, ty); 
  z[3]=(lx, ty); 
  
  return z[0]--z[1]--z[2]--z[3]--cycle; 
}

/********************* horizontal castellations ********************/
/* Draw a horizontal red line in the top left and the bottom right
 * castellation.  These testlines disappear if the monitor is not set
 * in a dot-exact mode.  An example is image crop due to overscan.  
 *
 * For 625 line systems any analog-compatible processing removes
 * these red testlines since the first halfline of the odd field and
 * the last halfline of the even field are ignored.  A full 576
 * visible line frame often results via a final copy paste operation.
 */
void castelhor(int colortv, int[] rccoll, int[] rccolr, int cmaxi, int Nh,
	       int topdist, int botdist,
	       pen pdef, real xd, real yd, int Nv) {
  pen pblack, pwhite, pred;
  int i;

  pblack = pdef+gray(0.0);
  pwhite = pdef+gray(1.0);
  pred = pdef+rgb(0.75, 0, 0);

  /** top and bottom: white corners. **/
  for (i=-1; i<=cmaxi; ++i) {
    pen pcast;
    int inext, lc, rc, tr, br;
    path zzc;

    inext = i+1;
    if (inext%2 == 0) {
      pcast = pwhite;
    } else {
      pcast = pblack;
    }

    if (i >= 0) {
      lc = rccolr[i];
    } else {
      lc = 0;
    }
    if (inext <= cmaxi) {
      rc = rccoll[inext];
    } else {
      rc = Nh;
    }

    if (i == 0 && colortv > 0 && topdist > 1) {
      path zzr;
      zzr = tvrect(lc,0, rc,1, xd,yd,Nv);
      fill(zzr, p=pred);
      tr = 1;
    } else {
      tr = 0;
    }
    zzc = tvrect(lc,tr, rc,topdist, xd,yd,Nv);
    fill(zzc, p=pcast);

    if (inext == cmaxi && colortv > 0 && botdist+1 < Nv) {
      path zzr;
      zzr = tvrect(lc,Nv-1, rc,Nv, xd,yd,Nv);
      fill(zzr, p=pred);
      br = Nv-1;
    } else {
      br = Nv;
    }
    zzc = tvrect(lc,botdist, rc,br, xd,yd,Nv);
    fill(zzc, p=pcast);
  }
  
  return;
}

/********************* vertical castellations ********************/
/* The bottom right red rectangle tests for a non causal color FIR 
 * filter in the receiver.  The last 2..4 dots then typically appear 
 * colorless, green, or cyan.  
 *
 * This stems from the fact that the chroma subcarrier is of lower 
 * bandwidth than luma and thus continues after the last active sample.  
 * These trailing (y,u,v) samples result from an abrupt signal to zero 
 * transition and depend on the transmit and receive filters.  Samples 
 * from VHS, system B/G/D/K, system I, or a DVD player output are 
 * different.  Nevertheless, a sharpening filter uses this data and so 
 * adds false color to the last dots.  
 */
void castelver(int colortv, int leftdist, int rightdist, int Nh,
	       int[] rcrowb, int[] rcrowt, int rmaxi,
	       pen pdef, real xd, real yd, int Nv) {
  pen pblack, pwhite;
  int i;

  pblack = pdef+gray(0.0);
  pwhite = pdef+gray(1.0);

  for (i=0; i<rmaxi; ++i) {
    int inext = i+1;
    pen pcastl, pcastr;
    int tr, br;
    path zzc;
  
    if (inext%2 == 0) {
      pcastl = pwhite;
    } else {
      pcastl = pblack;
    }
    if (inext == rmaxi && colortv>0) {
      pcastr = pdef+rgb(0.75,0.0,0);
    } else {
      pcastr = pcastl;
    }

    tr=rcrowb[i];
    br=rcrowt[i+1];    
    zzc=tvrect( 0,tr, leftdist,br, xd,yd,Nv);
    fill(zzc, p=pcastl);
    zzc=tvrect(rightdist,tr, Nh,br, xd,yd,Nv); 
    fill(zzc, p=pcastr);
  }
  return;
}
/********************* image aspect ratio markers ********************/
void rimarkers(real rimage, int Nh, int Nhc, int os, int Nvc, int Nsy,
	       pen pdef, real xd, real yd, int Nv) {
  int[] ridefN={ 4, 16 };
  int[] ridefD={ 3,  9 };
  int i;

  for (i=0; i<2; ++i) {
    real rid=ridefN[i]/ridefD[i];

    if (rimage>rid) {
      int off, offa, offb;

      /* Nhdef=Nh*rid/rimage */
      off=round(Nh*rid/rimage/2);
      offa=off+os;
      offb=off-os;
      // write(offa,offb);

      if (2*offa<Nh) {
        int hy, tr, br;
        path zz;
	
	hy=floor(Nsy/3);
	tr=Nvc-hy;
	br=Nvc+hy;
	
        zz=tvrect(Nhc+offb, tr, Nhc+offa, br, xd,yd,Nv);
        //dot(zz);
        fill(zz, p=pdef);
        zz=tvrect(Nhc-offa, tr, Nhc-offb, br, xd,yd,Nv);
        fill(zz, p=pdef);
      }
    }
  } /* for i */
  return;
}

/************* crosshatch: line pairing, center interlace test *************/
/* There are 2 coordinate systems in use:
 * 1. integer number based for the gridlines
 *   
 *   coff, Nhc, rccoll[], rccolc[], rccolr[] : vertical gridlines,
 *   rcrowc, Nvc : horizontal gridlines,
 *
 * 2. real number based for the center circle
 *
 *   ccenter={ cx=Nh/2, cy=Nv/2} : the true image center,
 *   rcoff rcright rcleft        : offset to ccenter and points on the circle.
 *
 * Both centers coincide if Nh and Nv are even.  
 */
void centerline(int colortv,
		int[] rccoll, int[] rccolc, int[] rccolr, int divsx,
		int Nhc, int os,
		int[] rcrowt, int[] rcrowc, int[] rcrowb, int divsy,
		int Nvc,
		pair ccenter, real[] rcoff, pair[] rcright, pair[] rcleft,
		pen pdef, real xd, real yd, int Nv) {
  pen pblack, pwhite;
  int cmaxi, maxoff, mincol, maxcol;
  int rows, tr, br;
  path zz;

  cmaxi=2*divsx+1;

  pblack=pdef+gray(0.0);
  pwhite=pdef+gray(1.0);

  /* black background for center cross */
  if (colortv > 0) {
    /* black, vertical gridlines redrawn below */
    pair[] z;
    int col;
  
    z[0]=rcright[divsy];

    col = rccolc[divsx+1];
    z[1]=tvps(col,rcrowc[divsy], xd,yd,Nv);
    z[2]=tvps(col,rcrowc[divsy-1], xd,yd,Nv);
    col = rccolc[divsx];
    z[3]=tvps(col,rcrowc[divsy-1], xd,yd,Nv);
    z[4]=tvps(col,rcrowc[divsy], xd,yd,Nv);

    z[5]=rcleft[divsy]; 
    z[6]=rcleft[divsy+1];
    
    z[7]=tvps(col,rcrowc[divsy+1], xd,yd,Nv);
    z[8]=tvps(col,rcrowc[divsy+2], xd,yd,Nv);
    col = rccolc[divsx+1];
    z[9]=tvps(col,rcrowc[divsy+2], xd,yd,Nv);
    z[10]=tvps(col,rcrowc[divsy+1], xd,yd,Nv);

    z[11]=rcright[divsy+1]; 
    fill(z[1]--z[2]--z[3]--z[4] //--z[5]--z[6]
	 --arc(ccenter, z[5], z[6])
	 --z[7]--z[8]--z[9]--z[10] //--z[11]--z[0]
	 --arc(ccenter,z[11], z[0])
	 --cycle, p=pblack);
  } else {
    /* 3 rows of black squares inside the gratings */
    int i, imax = divsy+1;

    for (i=divsy-1; i<=imax; ++i) {  /* all 3 rows */
      int lmaxoff, lmincol, lmaxcol;
      int inext = i+1;
      int tr, br, j;

      /* XXX rcoff is relative to ccenter */
      lmaxoff = min(floor(rcoff[i]), floor(rcoff[inext]));
      lmincol = Nhc-lmaxoff;
      lmaxcol = Nhc+lmaxoff;
      
      /* square top and bottom */
      tr = rcrowb[i];
      br = rcrowt[inext];

      for (j=0; j<cmaxi; ++j) { /* column j */
	int jnext = j+1;
	
	if (lmincol<=rccolc[j] && rccolc[jnext]<=lmaxcol) {
	  /* square is inside circle */
	  int lc, rc;
	  path zzsq;

	  lc = rccolr[j];
	  rc = rccoll[jnext];	  
	  zzsq = tvrect(lc, tr, rc, br, xd,yd,Nv); 
	  fill(zzsq, p=pblack);          /* draw black squares */
	}
      } /* for col j */
    } /* for row i */
  }
  
  /* center cross: vertical and horizontal centerline */
  maxoff = floor(rcoff[divsy]); /* XXX rcoff is relative to ccenter */
  mincol = Nhc-maxoff;
  maxcol = Nhc+maxoff;

  rows=min(Nvc-rcrowc[divsy-1], rcrowc[divsy+2]-Nvc);
  tr=Nvc-rows;
  br=Nvc+rows;
  if (verbose > 1) {
    write("centerline long : rows tr br ", rows, tr, br);
  }
  zz=tvrect(Nhc-os, tr, Nhc+os, br, xd,yd,Nv);
  fill(zz, p=pwhite);
  zz=tvrect(Nhc-maxoff,Nvc-1, Nhc+maxoff,Nvc+1, xd,yd,Nv);
  fill(zz, p=pwhite);

  /* vertical short lines */
  rows=min(Nvc-rcrowc[divsy], rcrowc[divsy+1]-Nvc);
  tr=Nvc-rows;
  br=Nvc+rows;
  if (verbose > 1) {
    write("centerline short: rows tr br ", rows, tr, br);
  }

  if (colortv > 0) {
    int i;
    for (i=0; i<=cmaxi; ++i) {
      int coll, colr;
    
      coll=rccoll[i];
      colr=rccolr[i];

      if (mincol<=coll && colr<=maxcol) {
	path zzv;
	zzv=tvrect(coll, tr, colr, br, xd,yd,Nv); 
	fill(zzv, p=pwhite);
      }
    }
  }
  return;
}
     
/************************ topbw **************************************/
void topbw(int[] coff, int Nhc, int os, int urow, int trow, int brow, 
	   pair ccenter, pair rclt, pair rclb, pair rcrt, pair rcrb, 
	   pen pdef, real xd, real yd, int Nv) {
  pen pblack=pdef+gray(0.0), pwhite=pdef+gray(1.0);
  pair[] ze;
  path zext, zref, zint;
  int off, col, cr;

  off=ceil((coff[2]+coff[3])/2);
  ze[0]=tvps(Nhc+off,trow, xd,yd,Nv);
  ze[1]=rcrt;
  ze[2]=rclt;
  ze[3]=tvps(Nhc-off,trow, xd,yd,Nv);
  ze[4]=tvps(Nhc-off,brow, xd,yd,Nv);
  col=Nhc-coff[2]-os;
  ze[5]=tvps(col,brow, xd,yd,Nv);
  ze[6]=tvps(col,trow, xd,yd,Nv);
  cr=col+3*os;    /* reflection test black pulse */
  zref=tvrect(col,trow, cr,brow, xd,yd,Nv);
  ze[7]=tvps(cr,trow, xd,yd,Nv);
  ze[8]=tvps(cr,brow, xd,yd,Nv);
  ze[9]=tvps(Nhc+off,brow, xd,yd,Nv);
  //dot(ze);

  zext=ze[0] // --ze[1]--ze[2]
    --arc(ccenter, ze[1], ze[2])
    --ze[3]--ze[4]--ze[5]--ze[6]--ze[7]--ze[8]--ze[9]--cycle;

  off=ceil((coff[1]+coff[2])/2);
  zint=tvrect(Nhc-off,urow, Nhc+off,trow, xd,yd,Nv); 

  /* paths are completely resolved; no free endpoint conditions */
  fill(zext^^reverse(zint), p=pwhite);
  fill(zint, p=pblack);
  fill(zref, p=pblack);

  fill(arc(ccenter,rclt,rclb)--ze[4]--ze[3]--cycle, p=pblack);
  fill(arc(ccenter,rcrb,rcrt)--ze[0]--ze[9]--cycle, p=pblack);
  return;
}

/************************ testtone **************************************/
/* x on circle -> return y>=0 
 * in:
 *   x     x-coordinate relative to origin
 *   crad  circle radius in y units, true size=crad*yd
 */
real testcircx(real x, real crad, real xd, real yd) {
  real relx, ph, y;

  relx=x*xd/yd/crad;
  if (relx>1) {
    ph=0;
  } else {
    ph=acos(relx);
  }
  y=crad*sin(ph);         // or (x*xd)^2+(y*yd)^2=(crad*yd)^2

  return y;
}
/* y on circle -> return x>=0 */
real testcircy(real y, real crad, real xd, real yd) {
  real rely, ph, x;

  rely=y/crad;
  if (rely>1) {
    ph=pi/2;
  } else {
    ph=asin(rely);
  }
  x=crad*cos(ph)*yd/xd;         // or (x*xd)^2+(y*yd)^2=(crad*yd)^2

  return x;
}

/* brow>trow && xb>xt */
void testtone(real Tt, int trow, int brow, 
	      real ccol, real crow, real crad,
	      pen pdef, real xd, real yd, int Nv) {
  int blocks, i;
  real yt, xt, yb, xb, Ttt=Tt/2;
  pair ccenter;

  yt=crow-trow;
  xt=testcircy(yt, crad, xd, yd);
  yb=crow-brow;
  xb=testcircy(yb, crad, xd, yd);
  //write('#xt yt\t',xt,yt); write('#xb yb\t',xb,yb);

  ccenter=tvps(ccol,crow, xd,yd,Nv);

  blocks=floor(2*xb/Tt);

  for (i=-blocks-1; i<=blocks; ++i) {
    real tl, tr;
    path zz;

    tl=max(-xb,min(i*Ttt,xb));      /* limit [-xb..xb] */
    tr=max(-xb,min((i+1)*Ttt,xb));

    if (tl<-xt && tr<=-xt || tr>xt && tl>=xt) {   /* top full circle */
      pair[] z;
      real yl, yr;

      yl=testcircx(tl, crad, xd, yd);
      yr=testcircx(tr, crad, xd, yd);

      z[0]=tvps(ccol+tl,brow, xd,yd,Nv);
      z[1]=tvps(ccol+tr,brow, xd,yd,Nv);
      z[2]=tvps(ccol+tr,crow-yr, xd,yd,Nv);
      z[3]=tvps(ccol+tl,crow-yl, xd,yd,Nv);

      zz=z[0]--z[1]--arc(ccenter,z[2],z[3])--cycle;
    } else if(tl<-xt) {       /* tl in circel, tr not, partial */
      pair[] z;
      real yl;

      yl=testcircx(tl, crad, xd, yd);

      z[0]=tvps(ccol+tl,brow, xd,yd,Nv);
      z[1]=tvps(ccol+tr,brow, xd,yd,Nv);
      z[2]=tvps(ccol+tr,trow, xd,yd,Nv);
      z[3]=tvps(ccol-xt,trow, xd,yd,Nv);
      z[4]=tvps(ccol+tl,crow-yl, xd,yd,Nv);

      zz=z[0]--z[1]--z[2]--arc(ccenter,z[3],z[4])--cycle;
    } else if(tr>xt) { /* tr in circle, tl not, partial */
      pair[] z;
      real yr;

      yr=testcircx(tr, crad, xd, yd);

      z[0]=tvps(ccol+tl,brow, xd,yd,Nv);
      z[1]=tvps(ccol+tr,brow, xd,yd,Nv);
      z[2]=tvps(ccol+tr,crow-yr, xd,yd,Nv);
      z[3]=tvps(ccol+xt,trow, xd,yd,Nv);
      z[4]=tvps(ccol+tl,trow, xd,yd,Nv);
      zz=z[0]--z[1]--arc(ccenter,z[2],z[3])--z[4]--cycle;
    } else { /* full block */
      pair[] z;

      z[0]=tvps(ccol+tr,trow, xd,yd,Nv);
      z[1]=tvps(ccol+tl,trow, xd,yd,Nv);
      z[2]=tvps(ccol+tl,brow, xd,yd,Nv);
      z[3]=tvps(ccol+tr,brow, xd,yd,Nv);
      zz=z[0]--z[1]--z[2]--z[3]--cycle;
    } 

    if (tl<tr) {
      if (i%2 == 0) {
        fill(zz, p=pdef+gray(0.0));
      } else {
        fill(zz, p=pdef+gray(0.75));
      }
    }
  }
  return;
}

/************************ color bars *************************************/
void colorbars(int[] coff, int Nhc, int trow, int crow, int brow, 
	       pair ccenter, pair rclt, pair rclb, pair rcrt, pair rcrb, 
	       pen pdef, real xd, real yd, int Nv) {
  real cI=0.75;
  real[] cR={ cI,  0,  0,  cI, cI,  0 };
  real[] cG={ cI, cI, cI,   0,  0,  0 };
  real[] cB={  0, cI,  0,  cI,  0, cI };
  int cmax=2, poff, rows, i;

  rows=brow-trow;
  poff=0;
  for (i=0; i<=cmax; ++i) {
    int off;
    int ii=2*i, il=cmax-i, ir=i+cmax+1;
    path zzl, zzr;
  
    off=ceil((coff[1+ii]+coff[2+ii])/2);
    if (i!=0 && i<cmax) {
      zzr=tvrect(Nhc+poff,trow, Nhc+off,brow, xd,yd,Nv); 
      zzl=tvrect(Nhc-off,trow, Nhc-poff,brow, xd,yd,Nv); 
    } else {
      if(i==0) {
        int col, pcol;
        pair[] zl, zr;

        col=Nhc+off;
        pcol=Nhc+poff;
        zr[0]=tvps(col,trow, xd,yd,Nv);
        zr[1]=tvps(pcol,trow, xd,yd,Nv);
        zr[2]=tvps(pcol,crow, xd,yd,Nv);
        zr[3]=tvps(Nhc+coff[0],crow, xd,yd,Nv);
        zr[4]=tvps(Nhc+coff[0],brow, xd,yd,Nv);
        zr[5]=tvps(col,brow, xd,yd,Nv);
        zzr=zr[0]--zr[1]--zr[2]--zr[3]--zr[4]--zr[5]--cycle;

        col=Nhc-off;
        pcol=Nhc-poff;
        zl[0]=tvps(pcol,trow, xd,yd,Nv);
        zl[1]=tvps(col,trow, xd,yd,Nv);
        zl[2]=tvps(col,brow, xd,yd,Nv);
        zl[3]=tvps(Nhc-coff[0],brow, xd,yd,Nv);
        zl[4]=tvps(Nhc-coff[0],crow, xd,yd,Nv);
        zl[5]=tvps(pcol,crow, xd,yd,Nv);
        zzl=zl[0]--zl[1]--zl[2]--zl[3]--zl[4]--zl[5]--cycle;
      } else {
        int pcol;
        pair[] zl, zr;

        pcol=Nhc+poff;
        zr[0]=tvps(pcol,brow, xd,yd,Nv);
        zr[1]=rcrb;
        zr[2]=rcrt;
        zr[3]=tvps(pcol,trow, xd,yd,Nv);
        zzr=zr[0]--arc(ccenter,zr[1],zr[2])--zr[3]--cycle;

        pcol=Nhc-poff;
        zl[0]=tvps(pcol,trow, xd,yd,Nv);
        zl[1]=rclt;
        zl[2]=rclb;
        zl[3]=tvps(pcol,brow, xd,yd,Nv);
        zzl=zl[0]--arc(ccenter,zl[1],zl[2])--zl[3]--cycle;
      }
    }
    fill(zzr, p=pdef+rgb(cR[ir], cG[ir], cB[ir]));
    fill(zzl, p=pdef+rgb(cR[il], cG[il], cB[il]));

    poff=off;
  }
  return;
}

/************************ test frequencies ****************************/
/* in
 *   theta  rad
 *   freq   1/hdot
 *   step   hdot
 * out
 *   new phase theta
 */
real addphase(real theta, real freq, real step) {
  real cycles, thetaret;
  int coverflow;

  cycles=freq*step;
  coverflow=floor(abs(cycles));  
  if (coverflow>1) {
    thetaret=0;
  } else {
    real dpi=2*pi;

    cycles-=coverflow*sgn(cycles);
    thetaret=theta+cycles*dpi;       /* cycles=(-1 .. 1) */

    if (thetaret>pi) { 
      thetaret-=dpi; 
    } else if (thetaret<-pi) { 
      thetaret-=dpi; 
    }
  }

  //write("addphase: ", step, theta, thetaret);
  return thetaret;
}

void testfreqs(real[] ftones, int[] coff, int Nhc, int trow,int crow,int brow, 
	       pair ccenter, pair rclt, pair rclb, pair rcrt, pair rcrb, 
	       pen pdef, real xd, real yd, int Nv) {
  int[] divc;
  real[] divfl, divfr;
  int i, divs, coffmax, off, divnext;
  real fl, fr, thr, thl;

  /* Segment info for PAL continental test card
   * segment i extends from (divc[i] .. divc[i+1]) with frequency divf[i]
   */
  divs=2;     // the number of segments to the right, total=2*divs+1
  divc[0]=0;
  for (i=0; i<=divs; ++i) {
    int ii=i*2, il=divs-i, ir=divs+i;

    divc[i+1]=ceil((coff[ii]+coff[ii+1])/2);  /* xdot distance to center */

    divfl[i]=ftones[divs-i];
    divfr[i]=ftones[divs+i];
  }
  coffmax=divc[divs+1];

  int trowlim=coff[0];
  int tr;

  tr=crow;

  divnext=0;
  fl=0;
  fr=0;
  thl=0;  /* ={ 0, -pi/2 } : initial angle at center vertical line Nhc */
  thr=thl;
  /* draw a vertical line at off..off+1, use theta for off+1/2 */
  for (off=0; off<coffmax; ++off) {
    real ampl, ampr;
    int col;
    path zz;

    if (off==trowlim) {
      tr=trow;
    }

    if (off == divc[divnext]) {  
      /* switch frequency: cycles=0.5*fcur+0.5*fnext */
      thl=addphase(thl, fl, -0.5);
      thr=addphase(thr, fr,  0.5);
      fl=divfl[divnext];
      fr=divfr[divnext];
      thl=addphase(thl, fl, -0.5);
      thr=addphase(thr, fr,  0.5);

      ++divnext;
      // thl=pi; thr=pi;
      //write(off, fl, fr);
    } else {
      thl=addphase(thl, fl, -1);
      thr=addphase(thr, fr,  1);
      // thl=0; thr=0;
    }

    ampl=(1+sin(thl))/2;
    ampr=(1+sin(thr))/2;
    //write(off, thr, ampr);

    col=Nhc-off-1;
    zz=tvrect(col,tr, col+1,brow, xd,yd,Nv); 
    fill(zz, p=pdef+gray(ampl));
    col=Nhc+off;
    zz=tvrect(col,tr, col+1,brow, xd,yd,Nv); 
    fill(zz, p=pdef+gray(ampr));
  }

  pair[] z;
  z[0]=tvps(Nhc-coffmax,trow, xd,yd,Nv);
  z[1]=tvps(Nhc-coffmax,brow, xd,yd,Nv);
  fill(z[0]--arc(ccenter,rclt,rclb)--z[1]--cycle, p=pdef+gray(0.0));
  z[0]=tvps(Nhc+coffmax,brow, xd,yd,Nv);
  z[1]=tvps(Nhc+coffmax,trow, xd,yd,Nv);
  fill(z[0]--arc(ccenter,rcrb,rcrt)--z[1]--cycle, p=pdef+gray(0.0));
  return;
}

/************************ gray bars **************************************/
void graybars(int[] coff, int Nhc, int trow, int brow, 
	      pair ccenter, pair rclt, pair rclb, pair rcrt, pair rcrb, 
	      pen pdef, real xd, real yd, int Nv) {
  int[] gs={0, 20, 40, 60, 80, 100};
  int cmax=2, poff, i;

  poff=0;
  for (i=0; i<=cmax; ++i) {
    int off;
    int ii=2*i, il=cmax-i, ir=i+cmax+1;
    path zzl, zzr;
  
    off=ceil((coff[1+ii]+coff[2+ii])/2);
    if (i<cmax) {
      zzl=tvrect(Nhc-off,trow, Nhc-poff,brow, xd,yd,Nv); 
      zzr=tvrect(Nhc+poff,trow, Nhc+off,brow, xd,yd,Nv); 
    } else {
      int pcol;
      pair zlt, zlb, zrt, zrb;

      pcol=Nhc-poff;
      zlt=tvps(pcol,trow, xd,yd,Nv);
      zlb=tvps(pcol,brow, xd,yd,Nv);
      zzl=zlt--arc(ccenter,rclt,rclb)--zlb--cycle;

      pcol=Nhc+poff;
      zrb=tvps(pcol,brow, xd,yd,Nv);
      zrt=tvps(pcol,trow, xd,yd,Nv);
      zzr=zrb--arc(ccenter,rcrb,rcrt)--zrt--cycle;
    }
    fill(zzl, p=pdef+gray(gs[il]/100));
    fill(zzr, p=pdef+gray(gs[ir]/100));

    poff=off;
  }
  return;
}

/************************ bottom bw **************************************/
void bottombw(int off, int Nhc, int trow, int brow, 
	      pair ccenter, pair rclt, pair rclb, pair rcrt, pair rcrb, 
	      pen pdef, real xd, real yd, int Nv) {
  int rows;
  pair zt, zb;
  path zz;

  rows=brow-trow;
  zz=tvrect(Nhc-off,trow, Nhc+off,brow, xd,yd,Nv); 
  fill(zz, p=pdef+gray(0.0));

  zt=tvps(Nhc-off,trow, xd,yd,Nv);
  zb=tvps(Nhc-off,brow, xd,yd,Nv);
  fill(zt--arc(ccenter,rclt,rclb)--zb--cycle, p=pdef+gray(1.0));

  zb=tvps(Nhc+off,brow, xd,yd,Nv);
  zt=tvps(Nhc+off,trow, xd,yd,Nv);
  fill(zb--arc(ccenter,rcrb,rcrt)--zt--cycle, p=pdef+gray(1.0));
  return;
}

/************************ bottom circle **************************************/
void bottomcirc(int off, int Nhc, int trow, real cx, real cy, real crad, 
		pair ccenter, pair rclt, pair rcrt,
		pen pdef, real xd, real yd, int Nv) {
  real cI=0.75;
  real xl, yl, xr, yr, phil, phir;
  pair ccleft, ccright;
  pair[] z;

  xl=Nhc-off-cx;
  phil=acos(xl*xd/yd/crad);
  yl=crad*sin(phil);         // or (x*xd)^2+(y*yd)^2=(crad*yd)^2
  ccleft=tvps(cx+xl,cy+yl, xd,yd,Nv);
  //write(xl,yl);

  xr=Nhc+off-cx;
  phir=acos(xr*xd/yd/crad);
  yr=crad*sin(phir); 
  ccright=tvps(cx+xr,cy+yr, xd,yd,Nv);

  //dot(ccright); dot(ccleft);
  // red center
  z[0]=tvps(Nhc-off,trow, xd,yd,Nv);
  z[1]=ccleft;
  z[2]=ccright;
  z[3]=tvps(Nhc+off,trow, xd,yd,Nv);
  fill(z[0]--arc(ccenter,z[1],z[2])--z[3]--cycle, p=pdef+rgb(cI,0,0));

  // yellow
  z[0]=tvps(Nhc-off,trow, xd,yd,Nv);
  z[1]=rclt;
  z[2]=ccleft;
  fill(z[0]--arc(ccenter,z[1],z[2])--cycle, p=pdef+rgb(cI,cI,0));
  z[0]=tvps(Nhc+off,trow, xd,yd,Nv);
  z[1]=ccright;
  z[2]=rcrt;
  fill(z[0]--arc(ccenter,z[1],z[2])--cycle, p=pdef+rgb(cI,cI,0));

  return;
}

/****************************** PAL ears ***********************************/
/* values pro mille
 * left  y      R       G       B 
 *     550     306     674     550
 *     500     363     500     859
 *     500     637     500     141
 *     450     694     326     450
 * right
 *     600     600     684     166
 *     400     400     316     834
 *
 * in: dright=  -1 left ear, +1 right ear
 */
void palears(int[] coff, int[] coffa, int[] coffb, int Nhc, 
	     int[] rcrowt, int[] rcrowb, int Nvc, int divsy, int dright,
	     pen pdef, real xd, real yd, int Nv) {
  /* the amplitude of (u,v) as seen on a vectorscope, 
   * max 0.296 Vn for 100% saturation in W and V ears.
   * cvbs:   0.7*( y +/- |u+jv| ) = -0.24 .. 0.93 V 
   * maxima: ebu 75/0 bars 0.70, bbc 100/25 0.88, 100/0 bars 0.93
   * burst:  0.150 Vcvbs, 21.4 IRE or 0.214 V normalized.
   * luma:   modulated for monochrome compatibility, 1990 version.
   * choice: set amplitude of subcarrier equal to amplitude of colorburst.
   */
  real cI=0.214;

  /* itu-r */
  real wr=0.299, wb=0.114, wg=1-wr-wb;     /* wg=0.587, y=wr*R+wg*G+wb*B */
  real wu=0.493, wv=0.877;                 /* u=wu*(B-y) v=wv*(R-y) */
  /* (u,v) for zero G-y, phase of -34.5 degrees */
  real colu=wu*wg/wb, colv=-wv*wg/wr;      /* for w=(G-y)/0.696 == 0 */

  /* ears:     U==0   W==0   W==0  U==0 */
  real[] cyl={ 0.55,   0.5,   0.5, 0.45 };
  real[] cul={   0,   colu, -colu,    0 };
  real[] cvl={  -1,   colv, -colv,    1 };

  /* ears:     V==0   W==0  W==0   V==0 */
  real[] cyr={ 0.60,   0.5,  0.5,  0.40 };
  real[] cur={  -1,   colu, -colu,    1 };
  real[] cvr={   0,   colv, -colv,    0 };

  real[] cy, cu, cv;
  pair[] z;
  path[] zz;
  int lcol, ccol, cicol, rcol, i;

  if (dright>0) {
    if (verbose > 1)
      write("right ears");
    cy=cyr; cu=cur; cv=cvr;
  } else {
    if (verbose > 1)
      write("left ears");
    cy=cyl; cu=cul; cv=cvl;
  }

  lcol=Nhc+dright*coffa[5];
  ccol=Nhc+dright*coff[6];
  cicol=Nhc+dright*coffa[6];
  rcol=Nhc+dright*coffb[7];

  int urow, trow, crow, brow, arow;
  urow=rcrowb[divsy-5];
  trow=rcrowt[divsy-3];
  crow=Nvc;
  brow=rcrowb[divsy+4];
  arow=rcrowt[divsy+6];

  z[0]=tvps(ccol,urow, xd,yd,Nv);
  z[1]=tvps(ccol,trow, xd,yd,Nv);
  z[2]=tvps(cicol,trow, xd,yd,Nv);
  z[3]=tvps(cicol,crow, xd,yd,Nv);
  z[4]=tvps(rcol,crow, xd,yd,Nv);
  z[5]=tvps(rcol,urow, xd,yd,Nv);
  zz[0]=z[0]--z[1]--z[2]--z[3]--z[4]--z[5]--cycle;

  zz[1]=tvrect(lcol,urow, ccol,trow, xd,yd,Nv);
  zz[2]=tvrect(lcol,brow, ccol,arow, xd,yd,Nv);

  z[0]=tvps(ccol,arow, xd,yd,Nv);
  z[1]=tvps(ccol,brow, xd,yd,Nv);
  z[2]=tvps(cicol,brow, xd,yd,Nv);
  z[3]=tvps(cicol,crow, xd,yd,Nv);
  z[4]=tvps(rcol,crow, xd,yd,Nv);
  z[5]=tvps(rcol,arow, xd,yd,Nv);
  zz[3]=z[0]--z[1]--z[2]--z[3]--z[4]--z[5]--cycle;

  for (i=0; i<4; ++i) {
    real y, u, v, A, ph, By, Ry, Gy, R, G, B;

    y=cy[i];
    u=cu[i];
    v=cv[i];

    A=hypot(u,v);
    ph= (u!=0 || v!=0) ? atan2(v,u) : 0.0;
    if (v>=0) {
      if (ph<0)
	ph=ph+pi;
    } else {
      if (ph>0)
	ph=ph-pi;
    }
    if (A>0) {
      u=u/A*cI;
      v=v/A*cI;
    }

    By=u/wu;
    Ry=v/wv;
    Gy=(-wr*Ry-wb*By)/wg;
    //write(y,Gy,A,ph*180/pi);

    R=Ry+y;
    G=Gy+y;
    B=By+y;
    if (verbose > 1)
      write(y*1000, round(R*1000), round(G*1000), round(B*1000));

    fill(zz[i], p=pdef+rgb(R,G,B));
  }
  return;
}

/****************************** NTSC bars ***********************************/
/* amplitude equals color burst smpte (pm: -V +U)
 *         y   campl  sat       R    G    B 
 * left   0.5  0.21   70%  -I?
 * right  0.5  0.17   60%  +Q?
 */
void ntscbars(int[] rccoll, int[] rccolr, int divsx, 
	      int[] rcrowt, int[] rcrowb, int divsy, int dright,
	      pen pdef, real xd, real yd, int Nv) {
  /* The amplitude of (i,q) as seen on a vectorscope, 
   * max 0.292 Vn for 100% saturation in I==0 ears.
   * burst:    0.143 Vcvbs, 20 IRE or 0.200 V normalized.
   * pedestal: (yp,up,vp)=(p,0,0)+(1-p)*(y,u,v), p=0.075.
   * choice:   equal amplitude for colorburst and subcarrier.
   */
  real campl=0.200/0.925;

  /* wg=0.587, y=wr*R+wg*G+wb*B */
  real wr=0.299, wb=0.114, wg=1-wr-wb;
  /* iT : iq -> RyBy : rotation+scaling */
  real iT11=0.95, iT12=0.62, iT21=-1.11, iT22=1.71;

  /* bars        -2    -1    0     1     2 */
  real[] cyl={ 0.50, 0.50,   1, 0.50, 0.50 };
  real[] cil={    0,    0,   0,   -1,    1 };
  real[] cql={   -1,    1,   0,    0,    0 };
  int[]  indl={  -7,   -8,   0,    8,    7 };
  
  real cy, ci, cq;
  int rmaxi, dri, ind, ibase, lcol, rcol, i;

  rmaxi=2*divsy+1;
  if (dright<-2 || dright>2) {
    dri=2;
  } else {
    dri=2+dright;
  }

  cy=cyl[dri];
  ci=cil[dri];
  cq=cql[dri];
  ind=indl[dri];
  ibase=divsx+ind;
  lcol=rccolr[ibase];
  rcol=rccoll[ibase+1];

  real A, By, Ry, Gy, R, G, B;

  A=hypot(ci,cq);
  if (A>0) {
    ci=ci/A*campl;
    cq=cq/A*campl;
  }
  Ry=iT11*ci+iT12*cq;
  By=iT21*ci+iT22*cq;
  Gy=(-wr*Ry-wb*By)/wg;
  //write(cy,Ry,Gy,By);

  R=Ry+cy;
  G=Gy+cy;
  B=By+cy;
  if (verbose > 1)
    write(ind, cy*1000, round(ci*1000), round(cq*1000),
	  round(R*1000), round(G*1000), round(B*1000));

  for (i=0; i<rmaxi; ++i) {
    path zz;
    int brow, trow, inext=i+1;
    
    if (i>0) {
      trow=rcrowb[i];
    } else {
      trow=floor((rcrowb[i]+rcrowt[inext])/2);
    } 

    if (inext<rmaxi) {
      brow=rcrowt[inext];
    } else {
      brow=floor((rcrowb[i]+rcrowt[inext])/2);
    }

    zz=tvrect(lcol,trow, rcol,brow, xd,yd,Nv);
    fill(zz, p=pdef+rgb(R,G,B));
  }
  
  return;
}

/****************************** main ***********************************/
/* Conversion to bitmap:
 *   EPSPNG='gs -dQUIET -dNOPAUSE -dBATCH -sDEVICE=png16m'
 *   asy -u bsys=2 -u colortv=1 -u os=1 -a Z tvgen
 *   $EPSPNG -r132x144 -g720x576 -sOutputFile=tvgen.png tvgen.eps
 *
 * asy -u bsys=2 -u colortv=1 -u os=1 tvgen
 */
int bsys=2, colortv=1, os=1;

/* bsys: broadcast system
 * bsys  im aspect  Nh
 *    0    4/3       704  guaranteed analog broadcast itu-r bt.470
 *    1    4/3       720  new broadcast, most TV station logos and animations
 *    2    15/11     720  total aperture analog 4/3, 1.37 film DVDs
 *    3    20/11     720  total aperture analog 16/9, 1.85 film DVDs 
 *    4    4/3       768  bsys=0, square dot analog broadcast 
 *    5    4/3       768  bsys=1, square dot cable TV info channel
 *    6    131/96    786  bsys=2, total square dot broadcast camera 
 *    7    16/9      720  new broadcast 16/9, SD from HD-1440 or itu-r bt.709
 *    8    4/3       704  525 analog broadcast itu-r bt.470 711x485
 *    9    4/3       720  525 new broadcast
 *   10    15/11     720  525 total aperture analog broadcast
 *   11    16/9     1920  1250, 1080 square dot at 12.5 frames/second
 *   12    4/3      1600  1250, 1200 square dot at 12.5 frames/second
 * 
 * colortv:
 *    0   monochrome crosshatch,
 *    1   pal ears,
 *    2   ntsc bars, 
 *    3   neither ears nor bars.
 *
 * os: horizontal oversampling, typical values for 13.5MHz:
 *    2   4/3 704*576, 15/11 720*576
 *    4   4/3 720*480
 *    5   4/3 704*480, 15/11 720*480, 4/3 768*576 14.4MHz
 *    8   4/3 720*576, 20/11 720*576
 *   12   704->768 rerastering
 *   16   720->768 rerastering
 */
access settings;
usersetting();

if (bsys<0 || bsys>12 || colortv<0 || colortv>3 || os<=0 || os>16) {
  write("Error: bad user input: bsys, colortv, os=\t", bsys, colortv, os);
  abort("Bad option  -u bsys=N  ?");
}

int[] bNdot=
  {   12,  16,  12,  16,     1,   1,    1,  64,   10,   8,  10,    1,    1 };
int[] bDdot=
  {   11,  15,  11,  11,     1,   1,    1,  45,   11,   9,  11,    1,    1 };
int[] bNh=
  {  704, 720, 720, 720,   768, 768,  786, 720,  704, 720, 720, 1920, 1600 };
int[] bNv=
  {  576, 576, 576, 576,   576, 576,  576, 576,  480, 480, 480, 1080, 1200 };
real[] bfs=
  { 13.5,13.5,13.5,13.5, 14.75,14.4,14.75,13.5, 13.5,13.5,13.5,   36,   30 };
int[] bNsy=
  {   42,  42,  42,  42,    42,  42,   42,  42,   34,  34,  34,   78,   90 };
int[] bNsh=
  {    0,   0,   0,   0,     0,   0,    0,   0,    0,   0,   0,    0,    0 };

/* active lines for a 625 line frame
 *   The number of active video lines decreased around 1997.  
 *     old:  3 run in + 575 visible + 3 run out = 581 lines
 *     new:  6 teletext and WSS + 575 visible 
 *   Hence the image center shifted down by 3 lines.  Thus
 *     old TV + new testcard = bottom is cut off,
 *     new TV + old testcard = top is cut off.
 *
 *   To generate the old testcard either use Nv=582 Nsh=0 or Nv=576 Nsh=3.
 *
 * aspect ratio
 *   rimage=xsize/ysize  rimage=rdot*Nh/Nv
 *   Nh=704 dots
 *   Nv=576 lines
 *   rd=ri*Nv/Nh=4/3*9/11=12/11
 *
 *   Nv: 480=2^5*3*5 576=2^6*3^2  
 *   Nh: 704=2^6*11 720=2^4*3^2*5
 *
 * horizontal line distance for pre 1997 test pattern
 *   top  8 lines, 13 squares of Ny=43 lines, bottom  9 lines
 *   top 12 lines, 13 squares of Ny=42 lines, bottom 18 lines
 *   pairs are from odd even field
 *   Interlace test: Ny must be odd for a cross-hatch without centerline.
 *
 * squares: ly=Nsy, lx=rd*Nsx, lx=ly ==> Nsx=Nsy/rd={ 39.4, 38.5 }
 *   x line width 230 ns -> 3 dots
 *   bottom 2.9us red -> 39.15 dots
 *
 * resolution DPI from image aspect ratio 
 *   Rv=Nv/ly,   ly=4in
 *   ri=Ni/Di,   Ni={ 4, 15, 16}  Di={ 3, 11, 9}
 *   lx=ri*ly
 *
 *   Rh=Nh/lx=Di*(Nh/(Ni*ly))
 *   integer Rh:
 *     Ni=4   ri=4/Di => Nh=k*16
 *     Ni=15 ri=15/Di => Nh=k*60
 *     Ni=16 ri=16/Di => Nh=k*64
 *
 * resolution DPI from dot aspect ratio, general algorithm, 
 *
 *     rd=Nd/Dd=ldx/ldy
 *
 *   assume 1 dot = Nd x Dd square subdots at a resolution of k, in dpi, then
 *
 *     ldx=Nd/k, ldy=Dd/k  ==>  Rh=k/Nd, Rv=k/Dd
 *
 *   choosing k=m*Nd*Dd for integer Rh and Rv gives
 *
 *     ldx=1/(m*Dd), ldy=1/(m*Nd), Rh=m*Dd, Rv=m*Nd
 *
 *   and 
 *
 *     lx=Nh*ldx=Nh/(m*Dd), ly=Nv*ldy=Nv/(m*Nd)
 *
 *   so choose m for the intended height Ly, in inch, as
 *
 *     m=round(Nv/(Ly*Nd))
 *
 *   which limits Ly<=Nv/Nd since Rv>=Nd.
 */
//cm=72/2.540005;
real Ly, ly, lx, ysize, xsize, rimage, xd, yd, pwidth;
int Nd, Dd, m, Nh, Nv, Nshift, Na, Nsy;
real fs, Ttone;

Nd=bNdot[bsys];
Dd=bDdot[bsys]*os;
Nh=bNh[bsys]*os;
Nv=bNv[bsys];

Ly=4;                    // 4 inch vertical size
m=floor(0.5+Nv/(Ly*Nd));
if (m < 1) m=1;
ly=Nv/(m*Nd);
lx=Nh/(m*Dd);

ysize=ly*1inch;
xsize=lx*1inch;
rimage=xsize/ysize;
if (verbose > 1) {
  write("#Nd Dd m ri:\t", Nd, Dd, m, rimage);
}
//size(xsize, ysize, Aspect);  // should not have any effect

Nsy=bNsy[bsys];       // grating size in lines 42,43 or 34,35
Nshift=bNsh[bsys];    // shift image up: pre 1997 =3, 2007 =0 
fs=1e6*bfs[bsys]*os; 
Na=0;          // add 1,0,-1 to height of hor center squares for even Na+Nsy

Ttone=fs/250e3;       // period of ft=250 kHz, fs/ft=54
real[] ftones={0.8e6/fs, 1.8e6/fs, 2.8e6/fs, 3.8e6/fs, 4.8e6/fs};

xd=xsize/Nh;
yd=ysize/Nv;
pwidth=min(abs(xd),abs(yd));

pen pdefault = squarecap+linewidth(pwidth);
pen pblack = pdefault+gray(0.0);
pen pwhite = pdefault+gray(1.0);

/**** calculate grating repeats and size in tv dots ****/
/* horizontal lines */
int divsy, rdisty, Nvc, Nt, Nb, rmaxi;

Nvc=floor(Nv/2)-Nshift;
/* top half picture (Nv-2)/2-(Nsy+Na)/2 dots for divisions of Nsy dots */
divsy=floor(((Nv-2-Na)/Nsy-1)/2);
rdisty=Na+Nsy*(1+2*divsy);
/* first guess free lines top and bottom */
Nt=Nvc-ceil(rdisty/2);
Nb=Nv-Nt-rdisty;
if (verbose > 1) {
  write('#divsy t b: \t', divsy, Nt, Nb);
}
rmaxi=2*divsy+1;

/* Nsyc: center square height 
 *   line pairing test: verify distance of center to top and bot 
 *   distance is odd ==> top=even/odd, cent=odd/even, bot=even/odd
 *
 * Nsyc odd: not possible
 *
 * Nsyc even:
 *   Nsyc/2 odd  --> OK
 *   Nsyc/2 even --> stagger the raster one line upwards
 *
 * rcrowt   top dist of hor line
 * rcrowc   true center for color info, distance to top of image.
 * rcrowb   bot dist of hor line
 *
 * offd = offu-Nsyc
 * Nt = Nvc-(offu+divsy*Nsy);
 * Nb = Nv-( Nvc-(offd-divsy*Nsy) );
 * ==> Nt+Nb = Nv-Nsyc-2*divsy*Nsy
 */
int Nsyc, offu, offd, Nyst=0, i;
int[] rcrowt, rcrowc, rcrowb;

Nsyc=Nsy+Na;
offu=floor(Nsyc/2);
offd=offu-Nsyc;
if (Nsyc%2 != 0) {
  Nyst=1;
} else if (Nsyc%4 == 0) {
  Nyst=1; /* stagger */
}
for (i=0; i<=divsy; ++i) {  
  int iu, id, ou, od, ru, rd;

  iu=divsy-i;
  id=divsy+i+1;

  ou=offu+Nsy*i;
  od=offd-Nsy*i;
  if (verbose > 1) {
    write(ou,od);
  }
  rcrowc[iu]=Nvc-ou;
  rcrowc[id]=Nvc-od;
  
  ru=Nvc-(ou+Nyst);
  rd=Nvc-(od+Nyst);

  rcrowt[iu]=ru-1;
  rcrowb[iu]=ru+1;

  rcrowt[id]=rd-1;
  rcrowb[id]=rd+1;
}
Nt=floor((rcrowt[0]+rcrowb[0])/2);
Nb=Nv-Nt-Nsyc-2*Nsy*divsy;
if (verbose > 1) {
  write('#st t b: \t', Nyst, Nt, Nb);
}

/* vertical lines
 * (Nh-2*os)/2-Nsx/2 dots available for divisions of Nsx dots.
 * At least 5 dots margin left and right ==> use -10*os
 */
real lsq, Nsx, rdistx;
int divsx, Nhc, Nl, Nr, cmaxi;

lsq=Nsy*yd;
Nsx=lsq/xd; /* floating point */
divsx=floor(((Nh-10*os)/Nsx-1)/2);  
Nhc=round(Nh/2);
rdistx=(1+2*divsx)*Nsx;
Nl=Nhc-round(rdistx/2);
if (verbose > 1) {
  write('#divsx Nsx l:\t', divsx, Nsx, Nl);
}
cmaxi=2*divsx+1;

int[] coff, coffl, coffr;
int[] rccoll, rccolc, rccolr;
for (i=0; i<=divsx; ++i) {  
  int off, offl, offr, il, ir;
  real cdist;
  
  cdist=Nsx*(1+2*i);  /* horizontal distance 2 symmetrical vert lines */
  off=round(cdist/2);
  // write(cdist, off);
  offl=off-os;
  offr=off+os;

  coff[i]=off;
  coffl[i]=offl;
  coffr[i]=offr;
  
  if (verbose > 1) {
    write(cdist, off);
  }
  il=divsx-i;
  ir=divsx+i+1;

  rccoll[il]=Nhc-offr;
  rccolc[il]=Nhc-off;
  rccolr[il]=Nhc-offl;

  rccoll[ir]=Nhc+offl;
  rccolc[ir]=Nhc+off;
  rccolr[ir]=Nhc+offr;  
}
Nl=rccolc[0];
Nr=Nh-rccolc[cmaxi];
if (verbose > 1) {
  write('#divsx Nsx l r:\t', divsx, Nsx, Nl, Nr);
}

/**** draw gray background ****/
{ 
  path zz;
  
  //zz=tvrect(0,0, Nh,Nv, xd,yd,Nv);
  /* keep white canvas for castellations */
  zz=tvrect(rccoll[0],rcrowt[0], rccolr[cmaxi],rcrowb[rmaxi], xd,yd,Nv);
  fill(zz, p=pdefault+gray(0.5));
  //dot(zz);
}
/**** draw center circle ****/
real cx, cy, crad;
pair ccenter;
path ccirc;
cx=Nh/2;
cy=Nv/2-Nshift;
crad=6*Nsy;
if (Nv%2 != 0) {
  crad+=0.5;
}
ccenter=tvps(cx,cy, xd,yd,Nv);
ccirc=circle(ccenter, crad*yd);
if (colortv<=0) {
  draw(ccirc, p=pwhite+linewidth(2*yd));
}

/**** draw 2*divsy+2 horizontal gridlines ****/
real[] rcang, rcoff;
pair[] rcright, rcleft;
int i;
for (i=0; i<=rmaxi; ++i) {
  real y, ph, x;
  path zzh;
  pair zd;

  zzh=tvrect(0,rcrowt[i], Nh,rcrowb[i], xd,yd,Nv);
  fill(zzh, p=pwhite);

  y=cy-rcrowc[i];
  if (abs(y)<crad) {
    ph=asin(y/crad);
  } else {
    ph=pi/2;
  }
  rcang[i]=ph;
  x=(crad*cos(ph))*yd/xd;
  rcoff[i]=x;
  zd=tvps(cx+x,cy-y, xd,yd,Nv);
  rcright[i]=zd;
  //dot(zd);
  zd=tvps(cx-x,cy-y, xd,yd,Nv);
  rcleft[i]=zd;
}

/**** draw 2*divsx+2 vertical gridlines ****/
for (i=0; i<=cmaxi; ++i) {
  path zzv;
  zzv=tvrect(rccoll[i],0, rccolr[i],Nv, xd,yd,Nv); 
  fill(zzv, p=pwhite); 
}

/**** castellations ****/
castelhor(colortv, rccoll, rccolr, cmaxi, Nh, rcrowt[0], rcrowb[rmaxi],
	  pdefault, xd, yd, Nv);

castelver(colortv, rccoll[0], rccolr[cmaxi], Nh, rcrowb, rcrowt, rmaxi,
	  pdefault, xd, yd, Nv);

/****** markers for 4/3 aspect ratio ******/
if (rimage>4/3) {
  rimarkers(rimage, Nh, Nhc, os, Nvc, Nsy, pwhite, xd, yd, Nv);
}

/****** line pairing center ******/
centerline(colortv, rccoll, rccolc, rccolr, divsx, Nhc, os,
	   rcrowt, rcrowc, rcrowb, divsy, Nvc,
	   ccenter, rcoff, rcright, rcleft, pdefault, xd, yd, Nv);

if (colortv>0) {
  /* topbw structure */
  topbw(coff, Nhc, os, rcrowc[divsy-5], rcrowc[divsy-4], rcrowc[divsy-3], 
	ccenter, rcleft[divsy-4], rcleft[divsy-3], rcright[divsy-4],
	rcright[divsy-3], pdefault, xd, yd, Nv);

  /* 250 kHz */
  testtone(Ttone, rcrowc[divsy-3], rcrowc[divsy-2], 
           cx, cy, crad, pdefault, xd, yd, Nv);

  /* color bars */ 
  colorbars(coff, Nhc, rcrowc[divsy-2], rcrowc[divsy-1], rcrowc[divsy], 
	    ccenter, rcleft[divsy-2], rcleft[divsy], rcright[divsy-2],
	    rcright[divsy], pdefault, xd, yd, Nv);

  /* test frequencies */
  testfreqs(ftones, coff, Nhc, rcrowc[divsy+1], rcrowc[divsy+2],
	    rcrowc[divsy+3], ccenter, rcleft[divsy+1], rcleft[divsy+3],
	    rcright[divsy+1],rcright[divsy+3], pdefault, xd, yd, Nv);

  /* gray bars */
  graybars(coff, Nhc, rcrowc[divsy+3], rcrowc[divsy+4], ccenter,
	   rcleft[divsy+3], rcleft[divsy+4],
	   rcright[divsy+3], rcright[divsy+4], pdefault, xd,yd,Nv);

  /* PAL ears */
  if (colortv == 1) {
    palears(coff,coffr,coffl, Nhc, rcrowt, rcrowb, Nvc, divsy, -1, 
            pdefault, xd, yd, Nv);
    palears(coff,coffr,coffl, Nhc, rcrowt, rcrowb, Nvc, divsy, 1, 
            pdefault, xd, yd, Nv);
  } else if (colortv == 2) {
    ntscbars(rccoll, rccolr, divsx, rcrowt, rcrowb, divsy, -1, 
             pdefault, xd, yd, Nv);
    ntscbars(rccoll, rccolr, divsx, rcrowt, rcrowb, divsy, 1, 
             pdefault, xd, yd, Nv);
    ntscbars(rccoll, rccolr, divsx, rcrowt, rcrowb, divsy, -2, 
             pdefault, xd, yd, Nv);
    ntscbars(rccoll, rccolr, divsx, rcrowt, rcrowb, divsy, 2, 
             pdefault, xd, yd, Nv);
  }

  /* bottom wh - black - wh */
  bottombw(round((coff[2]+coff[3])/2), Nhc, rcrowc[divsy+4], rcrowc[divsy+5], 
	   ccenter, rcleft[divsy+4], rcleft[divsy+5],
	   rcright[divsy+4], rcright[divsy+5], pdefault, xd, yd, Nv);

  /* bottom yellow red circle */
  bottomcirc(coff[0], Nhc, rcrowc[divsy+5], cx, cy, crad, 
	     ccenter, rcleft[divsy+5], rcright[divsy+5], pdefault, xd, yd, Nv);
}

/********************** set id *********************/
{ /* dpi */
  pair rpos=tvps(Nhc,round((rcrowc[divsy-4]+rcrowc[divsy-5])/2), xd,yd,Nv);
  string iRhor, iRver, ires;
  real Rh, Rv;

  Rh=Nh/xsize*inch;
  Rv=Nv/ysize*inch;
  iRhor=format("%.4gx", Rh);
  iRver=format("%.4gdpi", Rv);
  ires=insert(iRver,0, iRhor);

  /* size info */
  int rowbot=round((rcrowc[divsy+4]+rcrowc[divsy+5])/2);
  pair tpos=tvps(Nhc,rowbot, xd,yd,Nv);
  string ihor, iver, itot, iasp, ifm;
  real asp, fm;

  ihor=format("%ix",Nh);
  iver=format("%i ",Nv);
  itot=insert(iver,0, ihor);
  asp=xsize/ysize;
  iasp=format("%.3g ",asp);
  fm=fs/1e6;
  ifm=format("%.4gMHz",fm);
  itot=insert(iasp,0, itot);
  itot=insert(ifm,0, itot);

  /* size of square */
  int rowNsy, colNsy;
  pair Npos;
  string iNsy;
  pen pbw;

  rowNsy = round((rcrowc[divsy+5]+rcrowc[divsy+6])/2);
  colNsy = round((rccolc[divsx+5]+rccolc[divsx+6])/2);
  Npos = tvps(colNsy,rowNsy, xd,yd,Nv);
  iNsy = format("%i", Nsy);
  
  if (colortv>0) { 
    pbw=pdefault+gray(1.0);
  } else { 
    pbw=pdefault+gray(0.0);
  }
  label(ires, rpos, p=pbw);
  label(itot, tpos, p=pbw);
  label(iNsy, Npos, p=pbw);
  if (verbose > 1)
    write('#res:\t', ires, itot, iNsy);
}
