/* featpost3D.mp translated to asymptote by Jacques Pienaar (2004).
   preliminary version (incomplete)
   
   Mostly the original structures and commands were retained.
   The result is that neither the translator nor the original authors
   would be 100% happy about the result.

% featpost3D.mp
% L. Nobre G., C. Barbarosie and J. Schwaiger
% http://matagalatlante.org
% Copyright (C) 2004
% see also featpost.mp

% This set of macros extends the MetaPost language
% to three dimensions and eases the production of 
% physics diagrams.

% This is free software; you can redistribute it and/or
% modify it under the terms of the GNU General Public License
% as published by the Free Software Foundation; either version 2
% of the License, or (at your option) any later version.

% This set of macros is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
% GNU General Public License for more details.
*/

private import metapost;

public pen background = gray(0.987);

//write("Preloading FeatPost macros, version 0.5(alpha asy port)");

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Global Variables %%%%%%%%%%%%%

  public real RefDist[], HoriZon; 
  public real Spread, Shifts, PrintStep, PageHeight, PageWidth;
  public real MaxFearLimit;

//     V[], L[]p[], F[]p[];
  triple V[], L[][], F[][];
  int NL, npl[], NF, npf[];
    
  public bool ParallelProj, SphericalDistortion, FCD[], ShadowOn;
  public bool OverRidePolyhedricColor;
  public real TDAtiplen, TDAhalftipbase, TDAhalfthick;
  public int TableColors, RopeColors, ActuC,RopeColorSeq[]; 
  public int Nobjects, FC[];
  public pair OriginProjPagePos;
  public path VGAborder;
  public triple f, viewcentr;
  public pen TableC[]; 
  public pen HigColor, SubColor;
  public triple LightSource;
  public string ostr[];

  pair origin = (0,0);

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Default Values %%%%%%%%%%%%%%%

    f = (3,5,4);  // This f is the point of view in 3D
	    
	 viewcentr = (0,0,0);  // This is the aim of the view
	    
    Spread = 140;   // Magnification

    Shifts = 105.00mm;   // Central X and Y Coordinates on paper

    OriginProjPagePos = (105.00mm,148.45mm); 

    ParallelProj = false;   // Kind of perspective 
				                // Can't have both true
    SphericalDistortion = false;   // Kind of lens

    ShadowOn = false;   // Some objects may block the light

    HoriZon = 0;   // and cast a shadow on a horizontal plane at this Z
	    
    VGAborder = (182.05,210.00)--   // This definition assumes
			 (412.05,210.00)--         // that Shifts is 105.00mm
			 (412.05,382.05)--         // Use: gs -r200 and you 
			 (182.05,382.05)--cycle;   // will get few extra pixs

    PrintStep = 5;   // Coarseness, in resolvec

/* Not made use of 
    defaultscale = 0.75;
	 defaultfont = "cmss17";   // This is used by cartaxes
*/

  // And this is used by produce_auto_scale
    PageHeight = currentpicture.xsize;
	 PageWidth = currentpicture.ysize;   

    MaxFearLimit = 64;  // Valid Maximum Distance from Origin
                        // insideviewtriangle
	    
    HigColor = gray(0.85);          // These two colors are used in
	 SubColor = gray(0.35);             // fillfacewithlight
    LightSource = 10*(4,-3,4);   // This also
    OverRidePolyhedricColor = false;   // And also this
	    	    
    TableC[0] = gray(0.85);           // grey
	 TableC[1] = rgb(1,0,0);           // red
	 TableC[2] = rgb( 0.2, 0.2, 1.0 ); // blue
	 TableC[3] = rgb( 1.0, 0.7, 0.0 ); // orange
	 TableC[4] = rgb(0,0.85,0);        // pale green
	 TableC[5] = rgb(0.9,0,0.9);       // magenta
	 TableC[6] = rgb(0,0.85,0.85);     // cyan
    TableC[7] = rgb(0.85,0.85,0);     // yellow
	    
    TableColors = 7;
    ActuC = 5;

    RopeColorSeq[0] = 3;             // 
    RopeColorSeq[1] = 3;             // 
    RopeColorSeq[2] = 1;             // 
    RopeColorSeq[3] = 3;             // ropepattern 
    RopeColorSeq[4] = 3;             // 
    RopeColorSeq[5] = 5;             // 
                                    
    RopeColors = 5;                 

    Nobjects = 0;                  // getready and doitnow

	 TDAtiplen = 0.05;              // tdarrow
	 TDAhalftipbase = 0.02;         // Three-Dimensional
	 TDAhalfthick = 0.01;           // Arrow
	    
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%% Part I:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Very basic:*/
	    
// Colors have three coordinates. Get one. 

    real X(triple A) {
      return A.x;
    }

    real Y(triple A) {
      return A.y;
    }

    real Z(triple A) {
      return A.z;
    }

// The length of a triple.

    real conorm(triple A) { 
       return length(A);  
    }

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%% Triple Calculus:*/
	    
// Calculate the unit triple of a triple (or a point)

    triple N(triple A) {
      return unit(A);
    }

// The Dotproduct of two normalized triples is the cosine of the angle 
// they form.

    real nDotprod(triple A, triple B) {
      return dot(unit(A),unit(B));
    }

// The normalized crossproduct of two triples. 
// Also check getangle below.

    triple ncrossprod(triple A, triple B) { 
        return unit( cross( A, B ) );
    }

// Haahaa! Trigonometry. 

    real getangle(triple A, triple B) {
      real coss, sine;
      coss = dot( A, B );
      sine = conorm( cross( A, B ) );
      return angle((coss, sine));
    }
// Something I need for spatialhalfsfear.

    real getcossine( triple Center, real Radius ) {
	   real a, b;
	   a = conorm( f - Center );
	   b = Radius/a;
	   if (abs(b) >= 1) {
	     write("The point of view f is too close (getcossine).");
	     b = 2;                                           // DANGER!
	   }
	   return b;
    }
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%% Fundamental:*/
	    
// Rigorous Projection. This the kernel of all these lines of code.
// It won't work if R belongs the plane that contains f and that is 
// ortogonal to triple f, unless SphericalDistortion is true.
// f must not be on a line parallel to zz and that contains the
// viewcentr.
 
    pair rp(triple R) {
	   pair projpoi;
      triple v, u;
      real verti, horiz, eta, squarf, radio, ang, lenpl;

      v = unit( (-Y(f-viewcentr), X(f-viewcentr), 0) );
      u = ncrossprod( f-viewcentr, v );

	   horiz = dot( R-viewcentr, v );
	   verti = dot( R-viewcentr, u );

	   if (SphericalDistortion) {
		  if ( horiz != 0 || verti != 0 ) {
	    	    lenpl = sqrt( horiz^2 + verti^2 )*20; //%%%%%%%%%%%%%% DANGER
	    	    ang = getangle( f-R, f-viewcentr );
		    horiz = ang*horiz/lenpl;
		    verti = ang*verti/lenpl;
		    projpoi = (horiz,verti);
        } else {
		    projpoi = origin;
        }
		} 
      else {
	     if (ParallelProj) {
		    eta = 1;
	     } else {
	       squarf = dot( f-viewcentr, f-viewcentr );
		    radio = dot( R-viewcentr, f-viewcentr );
		    eta = 1 - radio/squarf;
		    if (eta < 0.03) {  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DANGER
		        eta = 0.03;
		    }
	     }
	     projpoi = (horiz,verti)/eta;
	   }

      return projpoi*Spread + (Shifts,Shifts);
    }

// Much improved rigorous pseudo-projection algorithm that follows 
// an idea from Cristian Barbarosie. This makes shadows.

    triple cb(triple R) {
      real ve, ho, sc;
      sc = Z(LightSource)-Z(R);
      if ( sc!=0 ) {
        sc = (Z(LightSource)-HoriZon)/sc;
      } else {
        sc = 0;
      }

      ho = (1-sc)*X(LightSource)+sc*X(R);
      ve = (1-sc)*Y(LightSource)+sc*Y(R);
      return (ho, ve, HoriZon);
    }

// And this just projects points rigorously on some generic plane.
    
    triple projectpoint(triple ViewCentr, triple R) {
      real verti, horiz;
      triple v, u, lray;
      
	   lray = LightSource-ViewCentr;
      v = unit( (-Y(lray), X(lray), 0) );
      u = ncrossprod( lray, v );
      // TN : Problem, don't really want to implement a singular 3 var Gauss-Jordan here
//      lray - horiz*v - verti*u = whatever*( LightSource - R );
//      return horiz*v + verti*u + ViewCentr;
      return (0,0,0);
    }
   
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%% Auxiliary:*/
	    
// Projection Size. Meant for objects with size one.
// Used by signalvertex.

    real ps(triple A, real Thicken_Factor) {
      return Thicken_Factor/conorm(A-f)/3;
    }

// Rigorous Projection of a Point. Draws a circle with
// a diameter inversely proportional to the distance of
// that Point from the point of view.

    void signalvertex(triple A, real TF, pen Col) {
      draw(rp(A), linewidth(Spread*ps(A,TF))+Col);
    }

    void signalshadowvertex(triple A, real TF, pen Col) {
	   triple auxc;
	   real auxn;
	   auxc = cb(A);
	   auxn = TF*conorm(f-auxc)/conorm(LightSource-A);
	   signalvertex( auxc, auxn, Col );
    }

// Get the triple that projects onto the resolution

    triple resolvec(triple A, triple B) {
      pair ap, bp;
      real sizel;
      triple returnvec;
      ap = rp(A);
      bp = rp(B);
      sizel = abs( ap - bp );
      if (sizel > 0) {
        returnvec = PrintStep*(B-A)/sizel;
      } else {
        returnvec = 0.3*(B-A);
      }
      return returnvec;
    }

// Movies need a constant frame

    void produce_vga_border() {
	    draw(VGAborder, background);
	    clip(currentpicture, VGAborder);
    }

/* TN : asy does this automatically
    def produce_auto_scale =
        begingroup
            picture storeall, scaleall;
	    real pwidth, pheight;
	    storeall = currentpicture shifted -(center currentpicture);
	    currentpicture := nullpicture;
	    pwidth = xpart ((lrcorner storeall)-(llcorner storeall));
	    pheight = ypart ((urcorner storeall)-(lrcorner storeall));
	    if PageHeight/PageWidth < pheight/pwidth:
	        scaleall = storeall scaled (PageHeight/pheight);
            else:
	        scaleall = storeall scaled (PageWidth/pwidth);
            fi;
	    draw scaleall shifted OriginProjPagePos
        endgroup
    enddef;
*/
    string cstr( triple Cl ) {
      return "(" + (string)X(Cl) + "," + (string)Y(Cl) + "," + (string)Z(Cl) + ")";
    }

    string bstr( bool bv ) {
      if (bv)
        return "true"; 
      else
        return "false"; 
    }
    
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%% Basic Functions:*/
	    
// Get the 2D path of a straight line in beetween 3D points A and B.
// This would add rigor to rigorousdisc, if one would introduce the
// the concept of three-dimensional path. That is not possible now.
// Also this is only interesting when using SphericalDistortion:=true
   
    path pathofstraightline( triple A, triple B ) {
      int k;
      triple mark, stepVec;
	   guide returnp;
	   pair pos[];
      
      stepVec = resolvec(A,B);
	   pos[0] = rp( A );
      k = 1;
      while (true) {
	     mark = A+(k*stepVec);
        if (dot(B-mark,stepVec) <= 0) break;
		  pos[k] = rp( mark );
		  ++k;
	   }
	   pos[k] = rp(B);
	   returnp = pos[0];
      for (int i=1; i <= k; i+=1) {
         returnp = returnp..pos[i];
      }
	   return (path)returnp;
    }

    void drawsegment( picture pic=currentpicture, Label L="",
		      triple A, triple B, pen p=currentpen, 
		      arrowbar arrow=None, arrowbar bar=None,
		      string legend="") 
    {
	   if (SphericalDistortion) {
	     draw( pic, L, pathofstraightline( A, B ), p, arrow, bar, legend);
	   } else {
	     draw( pic, L, rp(A)--rp(B), p, arrow, bar, legend);
      }
    }

//% Cartesian axes with prescribed lengths.

    void cartaxes(real axex, real axey, real axez) {
      triple orig, axxc, ayyc, azzc;
      orig = (0,0,0);
      axxc = (axex,0,0);
      ayyc = (0,axey,0);
      azzc = (0,0,axez);
      // TN : changed draw(..) to drawsegment(..)
      drawsegment(orig,axxc,Arrow);
      drawsegment(orig,ayyc,Arrow);
      drawsegment(orig,azzc,Arrow);
      label("$x$",rp(axxc),S);       //%%%%%%%%%%%%%%%%%%%%%%% 
      label("$y$",rp(ayyc),S);       //   Some  Labels...   %%
      label("$z$",rp(azzc),W);       //%%%%%%%%%%%%%%%%%%%%%%% 
    }

// This is it. Draw an arch beetween two straight lines with a
// common point (Or) in three-dimensional-euclidian-space and 
// place a label near the middle of the arch. Points A and B
// define the lines. The arch is at a distance W from Or. The
// label is S and the position is RelPos (rt,urt,top,ulft,lft,
// llft,bot,lrt). But arches must be smaller than 180 degrees.
/* TN : The RelPos should be (E, NE, N, NW, W, SW, S, SE) */

    void angline(triple A, triple B, triple Or, real W, 
                 string S, pair RelPos) {
      real G;
      triple Dna, Dnb;
      path al;
      G = conorm( W*( unit(A-Or) - unit(B-Or) ) )/2.5; //%%%%%%% BIG DANGER!
      Dna = ncrossprod(ncrossprod(A-Or,B-Or),A-Or);
      Dnb = ncrossprod(ncrossprod(B-Or,A-Or),B-Or);
      al = rp(W*unit(A-Or)+Or).. 
             controls rp(W*unit(A-Or)+Or+G*Dna) 
             and rp(W*unit(B-Or)+Or+G*Dnb)..
             rp(W*unit(B-Or)+Or);
      draw(al);
      label( S, point(al,0.5*length(al)), RelPos );
    }

/* TN : Changed RelPos to correspond to above. */
// As i don't know how to declare variables of type suffix,
// i provide a way to avoid the problem. This time RelPos may
// be 0,1,2,3,4,6,7 or anything else.

    void anglinen(triple A, triple B, triple Or, real W, 
                  string S, pair RelPos) {
      real G;
      triple Dna, Dnb;
      path al;
      pair middlarc;
      G = conorm( W*( N(A-Or) - N(B-Or) ) )/3;
      Dna = ncrossprod(ncrossprod(A-Or,B-Or),A-Or);
      Dnb = ncrossprod(ncrossprod(B-Or,A-Or),B-Or);
      al = rp(W*N(A-Or)+Or)..
              controls rp(W*N(A-Or)+Or+G*Dna) 
              and rp(W*N(B-Or)+Or+G*Dnb)..
                  rp(W*N(B-Or)+Or);
      draw(al);
      middlarc = point(al,0.5*length(al));
      label( S, middlarc, RelPos);
    }

// As a bigger avoidance, replace the arch by a paralellogram.

    void squareangline(triple A, triple B, triple Or, real W) {
      path sal;
      sal = rp(Or)--rp(W*N(A-Or)+Or)-- 
                    rp(W*(N(B-Or)+N(A-Or))+Or)--rp(W*N(B-Or)+Or)--cycle;
      draw(sal);
    }

// Just as we are here we can draw circles. (color,color,real)

    path rigorouscircle(triple CenterPos, triple AngulMom, real Radius) {
      real ind, G;
      triple vec[], Dna, Dnb;
      path al;
      vec[1] = ncrossprod( CenterPos-f, AngulMom);
      for (int ind = 2; ind <= 8;  ind += 2) {
        vec[ind+1] = ncrossprod( vec[ind-1], AngulMom );
        vec[ind] = N( vec[ind-1] + vec[ind+1] ); 
      }
      G = conorm( Radius*( vec[1] - vec[2] ) )/3;
      al = rp(Radius*vec[1]+CenterPos);
      for (int ind = 2; ind <= 8; ind += 1) {
        Dna = ncrossprod(ncrossprod(vec[ind-1],vec[ind]),vec[ind-1]);
        Dnb = ncrossprod(ncrossprod(vec[ind],vec[ind-1]),vec[ind]);
        al = al..controls rp(Radius*vec[ind-1]+CenterPos+G*Dna) 
                 and rp(Radius*vec[ind]+CenterPos+G*Dnb)
               ..rp(Radius*vec[ind]+CenterPos);
      }
	   al = al.. tension atleast 1 ..cycle;
      return al;
    }
    

// 3D arrow.

    void tdarrow(triple FromPos, triple ToTip ) {
      triple basevec, longvec, a, b, c, d, e, g, h;
	   real len;
	   path p;
	   len = conorm( ToTip - FromPos );
      longvec = N( ToTip - FromPos );
      basevec = ncrossprod( FromPos-f, longvec );
	   if (len <= TDAtiplen) {
	      b = basevec*TDAhalftipbase*len/TDAtiplen;
	      c = FromPos+b;
	      e = FromPos-b;
	      p = rp(ToTip)--rp(c)--rp(e)--cycle;
      } else {
	      d = ToTip-longvec*TDAtiplen;
	      a = FromPos+basevec*TDAhalfthick;
	      h = FromPos-basevec*TDAhalfthick;
	      b = d+basevec*TDAhalfthick;
	      g = d-basevec*TDAhalfthick;
	      c = d+basevec*TDAhalftipbase;
	      e = d-basevec*TDAhalftipbase;
	      p = rp(a)--rp(b)--rp(c)--rp(ToTip)--rp(e)--rp(g)--rp(h)--cycle;
	   }
      fill(p,white);
	   draw(p);
    }

//% Draw lines with a better expression of three-dimensionality.

    void emptyline(bool JoinP, real ThickenFactor, pen OutCol, pen InCol,
                   int theN, real EmptyFrac, int sN, 
                   triple LinFunc(real)) {
	   real i, j;
	   if (ShadowOn) {
	     for (int i = 0; i <= theN; i+=1) {
	       signalshadowvertex( LinFunc(i/theN), ThickenFactor, black );
	     }
	   }
	   for (int j = 0; j <= sN-1; j+=1) {
	     signalvertex( LinFunc(j/theN), ThickenFactor, OutCol );
	   }
	   if (JoinP) {
	     for (int j = -sN; j <= 0; j+=1) {
	       signalvertex(LinFunc(j/theN),ThickenFactor*EmptyFrac,InCol);
	     }
	   }
	   for (int i = sN; i <= theN; i+=1) {
	     signalvertex( LinFunc( i/theN ), ThickenFactor, OutCol );
	     for (int j = sN; i >= 0; i-=1) { 
	       signalvertex(LinFunc((i-j)/theN),ThickenFactor*EmptyFrac,InCol);
	     }
	   }
    }

// The next allows you to draw any solid that has no vertices and that has 
// two, exactly two, cyclic edges. In fact, it doesn't need to be a solid. 
// In order to complete the drawing of this solid you have to choose one of
// the edges to be drawn immediatly afterwards.    
    
    path twocyclestogether( path CycleA, path CycleB ) {
	   int i, Leng;
      real TheMargin, TheLengthOfA, TheLengthOfB;
	   path SubPathA, SubPathB, PolygonPath, FinalPath;
	   TheMargin = 0.02;
	   TheLengthOfA = length(CycleA) - TheMargin;
	   TheLengthOfB = length(CycleB) - TheMargin;
	   SubPathA = subpath ( CycleA, 0, TheLengthOfA );
	   SubPathB = subpath ( CycleB, 0, TheLengthOfB );
      /* TN : do not know equivalent */
//	   PolygonPath = makepath(makepen(( SubPathA--SubPathB--cycle )));
      PolygonPath = nullpath;
	   Leng = length(PolygonPath) - 1;
	   FinalPath = point(PolygonPath,0);
		for (int i = 1; i <= Leng; i+=1 )
		    FinalPath = FinalPath -- point(PolygonPath,i);
		FinalPath=FinalPath--cycle;
	   return FinalPath;
    }
    
// Ellipse on the air.

    path ellipticpath(triple CenterPos, triple OneAxe, triple OtherAxe ) {
      real ind;
      triple vec[];
      guide cirath;
      
	   for (int ind=1; ind <= 36; ind += 1) {
	      vec[ind] = CenterPos+OneAxe*Cos(ind*10)+OtherAxe*Sin(ind*10);
	   }
	   cirath = rp( vec[1] );
	   for (int ind=2; ind <= 36; ind += 1) {
		  cirath = cirath .. tension atleast 1 .. rp( vec[ind] );
		}
		cirath = cirath .. tension atleast 1 .. cycle;
      return (path)cirath;
    }
    

// Shadow of an ellipse on the air.

    path ellipticshadowpath(triple CenterPos, 
                            triple OneAxe, triple OtherAxe ) {
      real ind;
      triple vec[];
      guide cirath;
      
	   for (int ind=1; ind <= 36; ind +=1) {
	     vec[ind] = CenterPos+OneAxe*Cos(ind*10)+OtherAxe*Sin(ind*10);
      }
	   cirath = rp( cb( vec[1] ) );
	   for (int ind=2; ind <= 36; ind += 1) {
		  cirath = cirath .. tension atleast 1 .. rp( cb( vec[ind] ) );
      }
      cirath = cirath .. tension atleast 1 .. cycle;
      return (path)cirath;
    }

// It should be possible to attach some text to some plan.
// Unfortunately, this only works correctly when ParallelProj := true;
/* TN : not yet implemented (
  void labelinspace(bool KeepRatio, triple RefPoi, triple BaseVec,
                     triple UpVec, string SomeString) {
    picture labelpic;
	 pair lrc, ulc, llc;
	 transform plak;
	 triple centerc, newbase;
	 real aratio;
	 labelpic = thelabel( SomeString, origin );
	 lrc = lrcorner labelpic;
	 ulc = ulcorner labelpic;
	 llc = llcorner labelpic;
	 aratio = (xpart  lrc - xpart llc)/(ypart ulc - ypart llc);
	 if (KeepRatio)
	   newbase = conorm(UpVec)*aratio*N(BaseVec);
	 else
	   newbase = BaseVec;
	 rp(RefPoi+newbase) = lrc transformed plak;
	 rp(RefPoi+UpVec) = ulc transformed plak;
	 centerc = RefPoi+0.5(newbase+UpVec);
	 rp(RefPoi) = llc transformed plak;
    label( labelpic transformed plak, rp(centerc) )
   }*/

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%% Standard Objects:*/
	    
// And more precisely. The next routines spatialhalfcircle and 
// rigorousfear require circles drawn in a systematic and precise way.

/* TN : Changed the order around, thus in the following 'previous' should
  be 'following' */
// When there are realal problems with the previous routine
// use the following alternative:

    path head_on_circle(triple Pos, real Radius ) {
      real ind;
      triple vecx, vecy, vec[], view;
      guide cirath;
	   view = f-Pos;
      vecx = unit( (-Y(view), X(view), 0) );
      vecy = ncrossprod( view, vecx );
      for (int ind=1; ind <= 36; ind += 1) {
        vec[ind] = vecx*Cos(ind*10) + vecy*Sin(ind*10);
        vec[ind] = Pos + vec[ind]*Radius; 
      }
      cirath = rp( vec[1] );
	   for (int ind=2; ind <= 36; ind += 1) {
		    cirath = cirath .. tension atleast 1 .. rp( vec[ind] );
		}
		cirath = cirath .. tension atleast 1 .. cycle;
      return (path)cirath;
    }

    path goodcirclepath(triple CenterPos, triple AngulMom, real Radius ) {
      real ind, decision;
      triple vecx, vecy, vec[], goodangulmom, view;
      guide cirath;
	   view = f-CenterPos;
	   decision = dot( view, AngulMom );
	   if (decision < 0)
	     goodangulmom = -AngulMom;
	   else
		  goodangulmom = AngulMom;
      vecx = ncrossprod( view, goodangulmom );
	   decision = getangle( view, goodangulmom );
	   if (decision > 0.5) {               //%%%%%%%%%%%%%% DANGER %%%
        vecy = ncrossprod( goodangulmom, vecx );
		  for (int ind=1; ind <= 36; ind += 1) {
		    vec[ind] = vecx*Cos(ind*10) + vecy*Sin(ind*10);
		    vec[ind] = CenterPos + vec[ind]*Radius;
        }
		  cirath = rp( vec[1] );
		  for (int ind=2; ind <= 36; ind += 1) {
		    cirath = cirath .. tension atleast 1 .. rp( vec[ind] );
		  }
		  cirath = cirath .. tension atleast 1 .. cycle;
	   } else {
	     cirath = head_on_circle( CenterPos, Radius );
	   }
      return (path)cirath;
    }

// And its shadow. 

    path circleshadowpath(triple CenterPos, triple AngulMom, real Radius ) {
       real decision;
       triple vecx, vecy, view;
       guide cirath;
	    view = LightSource-CenterPos;
       vecx = ncrossprod( view, AngulMom );
	    decision = getangle( view, AngulMom );
	    if (decision > 0.5) {                  //%%%%%%%%%%%%% DANGER %%%
	      vecy = ncrossprod( AngulMom, vecx );
	      cirath = ellipticshadowpath(CenterPos,vecx*Radius,vecy*Radius);
	    } else {
	      vecx = N( (-Y(view), X(view), 0) );
	      vecy = ncrossprod( view, vecx );
	      cirath = ellipticshadowpath(CenterPos,vecx*Radius,vecy*Radius);
	    }
      return (path)cirath;
    }


// The nearest or the furthest part of a circle returned as a path.
// This function has been set to work for rigorousdisc (next).
// Very tough settings they were.

    path spatialhalfcircle(triple Center, triple AngulMom, real Radius,
                           bool ItsTheNearest ) {
      triple va, vb, vc, cc, vd, ux, uy, pa, pb;
      real nr, cn, valx, valy, valr, choiceang;
	   path auxil, auxih, fcirc, returnp;
	   bool choice;
	   va = Center - f; 
      vb = N( AngulMom ); 
      vc = vb*( dot( va, vb ) );
      cc = f + vc;
	   vd = cc - Center;  // vd := va + vc;
      nr = conorm( vd );
      if (Radius >= nr)
	        returnp = rp( cc );
	   else {
	     valr = Radius*Radius;
	     valx = valr/nr;
	     valy = sqrt( valr - valx*valx ); 
	     ux = N( vd );
	     choiceang = getangle( vc, va );                   //%%%%%%%%%%%%
	     choice = ( choiceang < 89 || choiceang > 91 );    //  DANGER  %%
	     if (choice)                                       //%%%%%%%%%%%%
		    uy = ncrossprod( vc, va );
	     else 
		    uy = ncrossprod( AngulMom, va );
	     pa = valx*ux + valy*uy + Center;
	     pb = pa - 2*valy*uy;
	     if (choice) {
          auxil = rp(interp(Center,pb,1.1))--rp(interp(Center,pb,0.9));
          auxih = rp(interp(Center,pa,1.1))--rp(interp(Center,pa,0.9));
		    fcirc = goodcirclepath( Center, AngulMom, Radius );
		    if (ItsTheNearest)
		      returnp = cutbefore(cutafter(fcirc, auxih), auxil);
		    else
		      returnp = cutbefore(fcirc,auxih)..cutafter(fcirc,auxil);
		  }  else {
		    if (ItsTheNearest) {
		      if (dot( va, AngulMom ) > 0)
		        returnp = rp(pb)--rp(pa);
		      else
		        returnp = rp(pa)--rp(pb);
          } else {
		      if (dot( va, AngulMom ) < 0)
		        returnp = rp(pb)--rp(pa);
		      else
		        returnp = rp(pa)--rp(pb);
          }
        }
      }
	   return returnp;
    }

// Cylinders or tubes ( real, bool, color, real, color ).
// Great stuff. The "disc" in the name comes from the fact that
// when SphericalDistortion := true; the sides of cylinders are
// not drawn correctly (they are straight). And when it is a tube
// you should force the background to be white.
 
    void rigorousdisc(real InRay, bool FullFill, triple BaseCenter, real Radius, triple LenVec) {
      triple va, vb, vc, cc, vd, base;
	   picture holepic;
	   triple vA, cC;
      real nr, vala, valb;
	   bool hashole, istube;
	   path auxil, auxih, halfl, halfh, thehole;
	   path auxili, auxihi, rect, theshadow;

	   va = BaseCenter - f; 
      vb = N( LenVec ); 
      vc = vb*( dot( va, vb ) );
      cc = f + vc;
	   vd = cc - BaseCenter;
      nr = conorm( vd );
	   base = BaseCenter + LenVec;
	   vA = base - f;
	   vala = conorm( va );
	   valb = conorm( vA );
	   if (ShadowOn) {
	     auxil = circleshadowpath( BaseCenter, LenVec, Radius );
	     auxih = circleshadowpath( base, LenVec, Radius );
	     fill(twocyclestogether( auxil, auxih ));
      }
	   auxil = goodcirclepath( base, LenVec, Radius );
	   auxih = goodcirclepath( BaseCenter, LenVec, Radius );
	   istube = false;
	   hashole = false;
	   if (InRay > 0) {
	     istube = true;
	     auxili = goodcirclepath( base, LenVec, InRay );
	     auxihi = goodcirclepath( BaseCenter, LenVec, InRay );
	     hashole = (-1,-1) != intersect(auxili,auxihi);
	     if (hashole) {
	       draw(auxili);
	       draw(auxihi);
	       holepic = currentpicture;
	       clip(holepic, auxili);
	       clip(holepic, auxihi);
	     }
	   }
		//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      if (Radius >= nr) {// THE CASE Radius > nr > InRay IS NOT SUPPORTED %
        if (vala <= valb) {
	       thehole = auxil;
	       auxil = auxih;
	       auxih = thehole;
	     }
	     if (istube) {
          if (vala <= valb) {
	         thehole = auxili;
	         auxili = auxihi;
	         auxihi = thehole;
	       }
	       holepic = currentpicture;
	       clip(holepic, auxihi);
	     }
	     unfill(auxil);
	     draw(auxil);
	     if (istube) {
	       // TN : draw(holepic);
	       draw(auxihi);
	     }
	   } else {
	     cC = base + vd;
        if ( dot( f - cc, f - cC ) <= 0 || !FullFill ) {
	       halfl = spatialhalfcircle(BaseCenter,LenVec,Radius,true);
	       halfh = spatialhalfcircle(base,LenVec,Radius,true);
	       if (FullFill) 
	         rect = halfl--halfh--cycle;
	       else
	         rect = halfl--reverse(halfh)--cycle;
	       unfill(rect);
	       draw(rect);
	     } else if (vala > valb) {
	       halfl = spatialhalfcircle(BaseCenter,LenVec,Radius,true);
	       halfh = spatialhalfcircle(base,LenVec,Radius,false);
	       rect = halfl--halfh--cycle;
	       unfill(rect);
	       draw(rect);
	       if (istube) {
	         if (hashole) {
	           // TN : draw(holepic);
            }
	         draw(auxili);
          }
	       draw(auxil);
	     } else {
	       halfl = spatialhalfcircle(BaseCenter,LenVec,Radius,false);
	       halfh = spatialhalfcircle(base,LenVec,Radius,true);
	       rect = halfl--halfh--cycle;
          unfill(rect);
	       draw(rect);
	       if (istube) {
	         if (hashole) {
	          // TN : draw(holepic);
            }
	         draw(auxihi);
	       }
	       draw(auxih);
        }
      }
    }

// And a cone. The vertex may go anywhere. Choose the full cone border
// (UsualForm=true) or just the nearest part of the base edge (false).
// This is used by tropicalglobe as a generic spatialhalfcircle to
// draw only the in fact visible part of circular lines. Please, don't
// put the vertex too close to the base plan when UsualForm=false.
 
  path rigorouscone(bool UsualForm, triple CenterPos, 
                     triple AngulMom, real Radius, triple VertexPos) {
    path basepath, thesubpath, fullpath, auxpath;
    guide finalpath;
	 path bigcirc;
	 real themargin, newlen, auxt, startt, endt, thelengthofc;
    int i;
	 pair pa, pb, pc, pd, pe;
	 themargin = 0.02; //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DANGER
	 basepath = goodcirclepath( CenterPos, AngulMom, Radius );
	 thelengthofc = length( basepath ) - themargin;
	 thesubpath = subpath ( basepath, 0, thelengthofc );
    /* TN : !equivalent */
//	 fullpath = makepath makepen ( rp(VertexPos)--thesubpath--cycle );
    fullpath = nullpath;
	 pa = interp(rp(CenterPos),rp(VertexPos),0.995);
	 pb = interp(rp(CenterPos),rp(VertexPos),1.005);
	 auxpath = pa--pb;
	 pc = intersect(auxpath, fullpath);
	 if (pc != (-1,-1)) {
	   auxt = pc.y;
	   newlen = length(fullpath);
	   if (UsualForm) {
	     finalpath = point(fullpath, auxt)--point(fullpath,auxt+1);
	     for (real i = auxt+2; i <= auxt+newlen-1; i+=1) {
	        finalpath = finalpath.. tension atleast 1 ..point(fullpath,i);
	     }
	     finalpath=finalpath--cycle;
	   } else {
	     bigcirc = goodcirclepath( CenterPos, AngulMom, 1.005*Radius );
	     pd = intersect(bigcirc, fullpath);
	     pe = intersect( reverse( bigcirc ), fullpath);
	     startt = floor( pd.x );
	     endt = ceil( length( bigcirc ) - pe.x );
	     finalpath = subpath (basepath, startt,endt);
	   }
	 } else {
	   finalpath = rp(VertexPos);
	 }
	 return finalpath;
  }
  
  void verygoodcone(bool BackDash, triple CenterPos, triple AngulMom, real Radius, triple VertexPos) {
	 triple bonevec, sidevec, viewaxe;
	 path thepath, cipath, basepath, thesubpath;
	 real lenpath, thelengthofc, themargin;
	 themargin = 0.02;
	 bonevec = VertexPos - CenterPos;
	 if (dot( bonevec, AngulMom ) < 0)
	   sidevec = -N(AngulMom);
	 else
	   sidevec = N(AngulMom);
	 viewaxe = f-CenterPos;
	 if (ShadowOn) {
	   basepath = circleshadowpath( CenterPos, AngulMom, Radius );
	   thelengthofc = length( basepath ) - themargin;
	   thesubpath = subpath ( basepath, 0, thelengthofc );
      /* TN : !equivalent */
	   // fill makepath makepen ( rp(cb(VertexPos))--thesubpath--cycle );
	 }
	 thepath = rigorouscone(true, CenterPos, AngulMom, Radius, VertexPos);
	 lenpath = length( thepath );
	 if (lenpath!=0) {
	   unfill(thepath);
	   draw( thepath );
	   if (dot( sidevec, viewaxe ) < 0)
	     draw( goodcirclepath( CenterPos, AngulMom, Radius ) );
	   else {
	     if (BackDash)
	       draw(goodcirclepath( CenterPos, AngulMom, Radius ), dashed );
	   }
	 } else {
	   cipath = goodcirclepath( CenterPos, AngulMom, Radius );
	   unfill(cipath);
	   draw(cipath);
	   if (dot( sidevec, viewaxe ) > 0)
	     draw( rp( VertexPos ) );
    }
  }

// Its a sphere, don't fear, but remember that the rigorous projection
// of a sphere is an ellipse. 

    path rigorousfearpath(triple Center, real Radius ) {
      triple ux, uy, newcen;
      real nr, valx, valy, valr;
	   path auxil;
      nr = conorm( Center - f );
	   valr = Radius^2;
	   valx = valr/nr;
	   valy = sqrt( valr - valx^2 );
	   newcen = valx*( f - Center )/nr;
	   auxil = head_on_circle( Center+newcen, valy );
	   return auxil;
    }

    path rigorousfearshadowpath(triple Center, real Radius ) {
      triple ux, uy, newcen;
      real nr, valx, valy, valr, lenr;
	   path auxil, auxih, fcirc, returnp;
	   pair dcenter;
      nr = conorm( Center - LightSource );
	   valr = Radius^2;
	   valx = valr/nr;
	   valy = sqrt( valr - valx^2 );
	   newcen = valx*( LightSource - Center )/nr;
	   auxil = circleshadowpath( Center+newcen, newcen, valy );
	   return auxil;
    }

// It's a globe (without land).

  void tropicalglobe( int NumLats, triple TheCenter, real Radius, triple AngulMom ) {
	 triple viewaxe, globaxe, foc, newcenter;
	 real sinalfa, sinbeta, aux, limicos, stepang, actang;
	 real newradius, lc, i;
	 path cpath, outerpath;
	 bool conditiona, conditionb;
    real lena, lenb;
    
	 if (ShadowOn)
	   fill( rigorousfearshadowpath( TheCenter, Radius ) );
	 viewaxe = f-TheCenter;
	 sinalfa = Radius/conorm( viewaxe );
	 aux = dot( viewaxe, AngulMom );
	 if (aux < 0)
	   globaxe = -N(AngulMom);
	 else
	   globaxe = N(AngulMom);
	 sinbeta = dot( globaxe, N(viewaxe) );
	 aux = sqrt((1-sinalfa^2)*(1-sinbeta^2));
	 limicos = aux - sinalfa*sinbeta;
	 stepang = 180/NumLats;
	 globaxe = globaxe*Radius;
	 outerpath = rigorousfearpath(TheCenter,Radius);
	 unfill( outerpath );
	 draw( outerpath );
    /* TN : not sure correct(until) */
	 for (real actang = 0.5*stepang; actang != 179;  actang+=stepang) {
	   if (Cos(actang) < limicos-0.005) { //%%%%%%%%%%%%%%%%%%%%%%% DANGER
	     newradius = Radius*Sin(actang);
	     newcenter = TheCenter - globaxe*Cos(actang);
	     conditiona = (actang<94) && (actang>86); // DANGER % DANGER VV
	     conditionb = abs(dot(globaxe/Radius,N(f-newcenter)))<0.08;
	     if (conditiona || conditionb) 
	       draw( spatialhalfcircle(newcenter,globaxe,newradius,true) );
	     else {
	       foc = TheCenter - globaxe/Cos(actang);
	       lena = -Radius*Cos(actang);
	       lenb = dot(viewaxe,globaxe/Radius);
	       if ((actang <= 86) || ((lenb<lena) && (actang>=94))) {
		      cpath = rigorouscone(false,newcenter,globaxe,newradius,foc);
		      draw( cpath );
	       } else {
		      cpath = rigorouscone(true,newcenter,globaxe,newradius,foc);
		      lc = length( cpath );
		      if (lc != 0)
		        draw( subpath(cpath, 1, lc-1) );
            else
		        draw( rigorouscircle( newcenter,globaxe,newradius ) );
          }
	     }
      }
    }
  }
	
  void whatisthis(triple CenterPos, triple OneAxe, triple OtherAxe, real CentersDist, real TheFactor ) {
	 path patha, pathb, pathc;
	 triple centersvec;

	 centersvec = CentersDist*ncrossprod( OneAxe, OtherAxe );
	 if (ShadowOn) {
	   patha = ellipticshadowpath( CenterPos,
	           OneAxe,
	           OtherAxe );
	   pathb = ellipticshadowpath( CenterPos+centersvec,
	           TheFactor*OneAxe,
	           TheFactor*OtherAxe );
	   pathc = twocyclestogether( patha, pathb );
	   fill( pathc );
	 }
	 patha = ellipticpath( CenterPos,
	           OneAxe,
	           OtherAxe );
	 pathb = ellipticpath( CenterPos+centersvec,
	           TheFactor*OneAxe,
	           TheFactor*OtherAxe );
	 pathc = twocyclestogether( patha, pathb );
	 unfill( pathc );
	 draw( pathc );
	 if (dot( centersvec, f-CenterPos ) > 0)
	   draw( pathb );
	 else
	   draw( patha );
  }
      
// It is time for a kind of cube. Don't use SphericalDistortion here.
    
    void kindofcube(bool WithDash, bool IsVertex, triple RefP, real AngA,
                    real AngB, real AngC, real LenA, real LenB, real LenC) {
	   triple star, pos[], refv, near, newa, newb, newc;
	   triple veca, vecb, vecc, auxx, auxy, centre, farv;
	   path patw, patb;
	   veca = ( Cos(AngA)*Cos(AngB),
	                  Sin(AngA)*Cos(AngB),
	                  Sin(AngB) );
	   auxx = ( Cos(AngA+90), Sin(AngA+90), 0 );
	   auxy = cross( veca, auxx );
	   vecb = Cos(AngC)*auxx + Sin(AngC)*auxy;
	   vecc = Cos(AngC+90)*auxx + Sin(AngC+90)*auxy;
	   veca = LenA*veca;
	   vecb = LenB*vecb;
	   vecc = LenC*vecc;
	   if (IsVertex) {
	     star = RefP;
	     centre = RefP + 0.5*( veca + vecb + vecc);
	   } else {
	     star = RefP - 0.5*( veca + vecb + vecc);
	     centre = RefP;
	   }
	   pos[1] = star + veca;
	   pos[2] = pos[1] + vecb;
	   pos[3] = pos[2] + vecc;
	   pos[4] = pos[3] - vecb;
	   pos[5] = pos[4] - veca;
	   pos[6] = pos[5] + vecb;
	   pos[7] = pos[6] - vecc;
	   if (ShadowOn) {
	     patw = rp(cb(star))--rp(cb(pos[1]))--rp(cb(pos[2]))
	          --rp(cb(pos[3]))--rp(cb(pos[4]))
	          --rp(cb(pos[5]))--rp(cb(pos[6]))--rp(cb(pos[7]))--cycle;
        /* TN : !equivalent */
	     //patb = makepath makepen patw;
        patb = nullpath;
	     fill( patb );
      }
	   patw = rp(star)--rp(pos[1])--rp(pos[2])--rp(pos[3])--rp(pos[4])
	         --rp(pos[5])--rp(pos[6])--rp(pos[7])--cycle;
      /* TN : !equivalent */
	   // patb := makepath makepen patw;  
      patb = nullpath;
	   unfill( patb );
	   draw( patb );
	   refv = f - centre;
	   if (dot( refv, veca ) > 0)
	      newa = -veca;
	   else
	      newa = veca;
	   if (dot( refv, vecb ) > 0)
	     newb = -vecb;
	   else
	     newb = vecb;
	   if (dot( refv, vecc ) > 0)
	     newc = -vecc;
	   else
	     newc = vecc;
	   near = centre - 0.5*( newa + newb + newc );
	   draw( rp(near)--rp(near+newa) );
	   draw( rp(near)--rp(near+newb) );
	   draw( rp(near)--rp(near+newc) );
	   if (WithDash) {
	     farv = centre + 0.5*( newa + newb + newc );
	     draw( rp(farv)--rp(farv-newa), dashed );
	     draw( rp(farv)--rp(farv-newb), dashed );
	     draw( rp(farv)--rp(farv-newc), dashed );
	   }
    }
    
// Maybe you would like to calculate the angular arguments of kindofcube...
    
    pair getanglepair( triple InVec ) {
	   real alphaone, alphatwo;
	   alphaone = angle( ( X(InVec), Y(InVec) ) );
	   alphatwo = angle( ( sqrt(X(InVec)^2 + Y(InVec)^2), Z(InVec) ) );
	   return (alphaone,alphatwo);
    }
    
// It's a bit late now but the stage must be set.

    void setthestage( real NumberOfSideSquares, real SideSize ) {
	   real i, j, squaresize;
	   path squarepath;
	   triple ca, cb, cc, cd;
	   squaresize = SideSize/(2*NumberOfSideSquares-1);
	   for (real i=-0.5*SideSize; i!=0.5*SideSize; i+=2*squaresize) {
	     for (real j=-0.5*SideSize; j!=0.5*SideSize;  j+=2*squaresize) {
	       ca = (i,j,HoriZon);
	       cb = (i,j+squaresize,HoriZon);
	       cc = (i+squaresize,j+squaresize,HoriZon);
	       cd = (i+squaresize,j,HoriZon);
	       squarepath = rp(ca)--rp(cb)--rp(cc)--rp(cd)--cycle;
	       unfill( squarepath );
	       draw( squarepath );
        }
      }
    }
    
    void setthearena( real NumberOfDiameterCircles, real ArenaDiameter ) {
	   real i, j, circlesize, polar, phi;
	   triple currpos;
	   path cpath;
 	   circlesize = ArenaDiameter/NumberOfDiameterCircles;
	   for (real i=0.5*ArenaDiameter; i!=0.4*circlesize; i+=-circlesize) {
	     polar = floor(6.28318*i/circlesize);
	     for (real j=1; j<= polar; j+=1) {
	       phi = 360*j/polar;
	       currpos = i*(Cos(phi),Sin(phi),HoriZon);
          // TN : changed blue -> (0,0,1)
	       cpath = rigorouscircle( currpos, (0,0,1), 0.25*circlesize);
	       unfill( cpath );
	       draw( cpath );
        }
      }
    }

// And a transparent dome. The angular momentum triple is supposed 
// to point from the concavity of the dome and into outer space.
// The pen can only be changed with a previous drawoptions().

    void spatialhalfsfear(triple Center, triple AngulMom, real Radius ) {
	   path spath, cpath, fpath, rpath, cutp;
	   pair ap, bp, cp, dp, cuti, cute, vp;
	   real auxcos, actcos, actsin, auxsin;
	   picture partoffear;
      triple A, B;
      spath = rigorousfearpath( Center, Radius );
	   auxcos = getcossine( Center, Radius );
	   actcos = dot( N( f - Center ), N( AngulMom ) );
	   actsin = sqrt(1-actcos^2);
	   auxsin = sqrt(1-auxcos^2);
	   if (actsin <= auxcos) {
	     if (actcos >= 0) {
	       cpath = goodcirclepath( Center, AngulMom, Radius );
	       draw( cpath );
	     } else
	       draw( spath );
	   } else {
	     fpath = spatialhalfcircle( Center, AngulMom, Radius, true );
	     rpath = spatialhalfcircle( Center, AngulMom, Radius, false );
	     cuti = point ( rpath, 0 );
	     cute = point ( rpath, length ( rpath ) );
	     ap = interp(cuti,cute,1.1);
	     bp = interp(cute,cuti,1.1);
        /* TN : !equivalent
	     partoffear = nullpicture;
        */
        /* TN : !equivalent
	     addto partoffear doublepath spath;
        */
	     A = ncrossprod( f-Center, ncrossprod( f-Center, AngulMom ) );
	     B = Center + 1.1*Radius*( auxcos*N( f-Center ) + auxsin*A );
	     vp = rp(B) - rp(Center);
	     cp = ap + vp;
	     dp = bp + vp;
	     cutp = ap--cp--dp--bp--cycle;
	     clip( partoffear, cutp );
	     draw( fpath );
	     // TN : draw( partoffear );
	     if (actcos >= 0)
	       draw( rpath );
      }
    }
	      
// Take a donut. 

    void smoothtorus( triple Tcenter, triple Tmoment, real Bray, real Sray ) {
	   triple nearaxe, sideaxe, viewline, circlecenter, circlemoment;
	   real ang, anglim, angstep, distance, coofrac, lr;
      int ind, i;
	   path cpath, apath, ipath, opath, wp, ep;
      guide cguide, aguide, iguide, oguide;
	   pair outerp[], innerp[], refpair;
	   bool cuspcond;
	   picture holepic;
	   triple tmoment;

	   angstep= 4; //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DANGER!

	   viewline = f-Tcenter;
	   if (dot( viewline, Tmoment ) < 0)
	     tmoment = -Tmoment;
	   else
	     tmoment = Tmoment;
	   refpair = unit( rp(Tcenter+tmoment)-rp(Tcenter) );
	   sideaxe = Bray*ncrossprod( tmoment, viewline );
	   nearaxe = Bray*ncrossprod( sideaxe, tmoment );
	   coofrac = dot( viewline, N( tmoment ) )/Sray;

	   if (coofrac <= 1.04 && coofrac >= 1.01) { //%%%%%%%% DANGER!
        // TN : possible conversion problem
	     ind = (int)(360/angstep);
	     anglim = 0.5*angstep;
	     for (int i=1; i <= ind; i+=1) {
	       ang = i*angstep-anglim-180.0;
	       circlecenter= nearaxe*Cos(ang)+sideaxe*Sin(ang)+Tcenter;
	       circlemoment=-nearaxe*Sin(ang)+sideaxe*Cos(ang);
	       cpath=spatialhalfcircle(circlecenter,circlemoment,Sray,true);
	       if (i >= 0.5*ind+1)
	         outerp[i]=point( cpath, 0 );
	       else
	         outerp[i]=point ( cpath, length (cpath) );
        }
	     oguide = outerp[1];
        for (int i=2; i <= ind; i+=1)
          oguide = oguide .. outerp[i];
        oguide = oguide .. cycle;
        opath = oguide;
	     unfill( opath );
	     draw( opath );
      } else if (coofrac < 1.01) {
	     distance = conorm( viewline );
	     lr = Bray + Sray*sqrt( 1^2 - coofrac^2 );
	     anglim = angle( ( lr, sqrt(distance^2 - lr^2 ) ) );
	     ind = 2*floor(anglim/angstep);
	     angstep = 2*anglim/(ind+1);
	     for (int i=0; i <= 0.5*ind-1; i+=1) {
	       ang = i*angstep-anglim;
	       circlecenter= nearaxe*Cos(ang)+sideaxe*Sin(ang)+Tcenter;
	       circlemoment=-nearaxe*Sin(ang)+sideaxe*Cos(ang);
	       cpath=spatialhalfcircle(circlecenter,circlemoment,Sray,true);
	       innerp[i]=point( cpath, 0 );
	       outerp[i]=point( cpath, length (cpath) );
	     }
	     for (int i=(int)(0.5*ind); i <= ind-2; i+=1 ) {
	       ang = (i+2)*angstep-anglim;
	       circlecenter= nearaxe*Cos(ang)+sideaxe*Sin(ang)+Tcenter;
	       circlemoment=-nearaxe*Sin(ang)+sideaxe*Cos(ang);
	       cpath=spatialhalfcircle(circlecenter,circlemoment,Sray,true);
	       outerp[i]=point(cpath, 0);
	       innerp[i]=point(cpath, length(cpath));
	     }
	     if (coofrac > 0.94) {
	       aguide = innerp[0];
          for (int i=1; i <= ind-2; i+=1) {
            aguide = aguide..innerp[i];
          }
          aguide = aguide--cycle;
          apath = (path)aguide;
        } else {
	       aguide = innerp[0];
          for (int i=2; i <= ind-2; i+=1) {
            aguide = aguide .. innerp[i];
          }
	       aguide = aguide .. outerp[ind-2];
          for (int i=ind-3; i >= 0; i-=1) {
            aguide = aguide..outerp[i];
          }
	       aguide = aguide..cycle;
          apath = (path)aguide;
	     }
	     unfill(apath);
	     draw(apath);
	   } else {
	     ind = (int)(360/angstep);
	     anglim = 0.5*angstep;
	     for (int i=1; i <= ind; i+=1) {
	       ang = i*angstep-anglim-180.0;
	       circlecenter= nearaxe*Cos(ang)+sideaxe*Sin(ang)+Tcenter;
	       circlemoment=-nearaxe*Sin(ang)+sideaxe*Cos(ang);
	       cpath=spatialhalfcircle(circlecenter,circlemoment,Sray,true);
	       if (i >= 0.5*ind+1) {
	         outerp[i]=point (cpath,0);
	         innerp[i]=point (cpath, length(cpath));
	       } else {
	         innerp[i]=point (cpath,0);
	         outerp[i]=point (cpath, length(cpath));
	       }
	     }
	     oguide = outerp[1];
        for (int i=2; i <= ind; i+=1)
          oguide = oguide .. outerp[i];
        oguide = oguide .. cycle;
        opath = (path)oguide;
        
	     iguide = innerp[1];
        for (int i=1; i <= ind; i+=1)
          iguide = iguide .. innerp[i];
        iguide = iguide .. cycle;
        ipath = (path)iguide;
        
	     holepic = currentpicture;
	     clip (holepic, ipath);
	     unfill(opath);
	     // TN : draw(holepic);
	     draw(opath);
	     draw(ipath);

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Perhaps there is an analytic way of getting the angle of the cusp point?  %
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	     i = ceil(1+0.5*ind);
	     cuspcond = false;
        while (true) {
	       ++i;
	       if (i > ind-1) break;
	       cuspcond=dot(refpair, innerp[i+1]) < dot(refpair, innerp[i]);
	       if (cuspcond) break;
	     }
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	     if (cuspcond) {
	       write( "Torus shows cusp points." );
	       ep = outerp[ind-i+1]--innerp[ind-i+1];
	       wp = innerp[i]--outerp[i];
	       // TN : unfill( buildcycle(reverse(opath),ep,ipath,wp) );
	       draw( opath );
	       draw( subpath (ipath, i-1,ind-i) );
	     }
	   }
    }

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
//%% Non-standard objects:
	    
    void positivecharge( bool InFactPositive, triple Center, real BallRay ) {
	   triple auxc, axehorf, axeside, viewline, pa, pb, pc, pd;
	   path spath;
	   viewline = f - Center;
	   axehorf = N( ( X(viewline), Y(viewline), 0 ) );
	   axeside = cross( axehorf, (0,0,1) );
	   if (ShadowOn)
	     fill( rigorousfearshadowpath( Center, BallRay ) );
	   spath = rigorousfearpath( Center, BallRay );
	   unfill( spath );
	   draw( spath );
	   auxc = Center + sqrt(3)*axehorf;
	   pa = auxc + axeside;
	   pb = auxc - axeside;
	   angline( pa, pb, Center, BallRay, "", N );
	   if (InFactPositive) {
	     pc = auxc + (0,0,1);
	     pd = auxc - (0,0,1);
	     angline( pc, pd, Center, BallRay, "", N );
	   }
    }

    void simplecar(triple RefP, triple AngCol, triple LenCol, 
                  triple FronWheelCol, triple RearWheelCol ) {
	   triple veca, auxx, auxy, vecb, vecc, viewline, fl, fr, rl, rr, inrefp;
	   real anga, angb, angc, lena, lenb, lenc, auxm, auxn;
	   real fmar, fthi, fray, rmar, rthi, rray;
	   anga = X( AngCol );
	   angb = Y( AngCol );
	   angc = Z( AngCol );
	   lena = X( LenCol );
	   lenb = Y( LenCol );
	   lenc = Z( LenCol );
	   fmar = X( FronWheelCol );
	   fthi = Y( FronWheelCol );
	   fray = Z( FronWheelCol );
	   rmar = X( RearWheelCol );
	   rthi = Y( RearWheelCol );
	   rray = Z( RearWheelCol );
	   veca = ( Cos(anga)*Cos(angb),
	                  Sin(anga)*Cos(angb),
	                  Sin(angb) );
	   auxx = ( Cos(anga+90), Sin(anga+90), 0 );
	   auxy = cross( veca, auxx );
	   vecb = Cos(angc)*auxx + Sin(angc)*auxy;
	   vecc = Cos(angc+90)*auxx + Sin(angc+90)*auxy;
	   viewline = f - RefP;
	   auxm = dot( viewline, veca );
	   auxn = dot( viewline, vecb );
	   inrefp = RefP - 0.5*lenc*vecc;
	   fl = inrefp + (0.5*lena-fmar-fray)*veca + 0.5*lenb*vecb;
	   fr = inrefp + (0.5*lena-fmar-fray)*veca - 0.5*lenb*vecb;
	   rl = inrefp - (0.5*lena-rmar-rray)*veca + 0.5*lenb*vecb;
	   rr = inrefp - (0.5*lena-rmar-rray)*veca - 0.5*lenb*vecb;
	   if (auxn > 0.5*lenb) {
	     if (auxm > 0) {
	       rigorousdisc( 0, true, rr, rray, -rthi*vecb );
	       rigorousdisc( 0, true, fr, fray, -fthi*vecb );
	       kindofcube(false,false,RefP,anga,angb,angc,lena,lenb,lenc);
	       rigorousdisc( 0, true, rl, rray, rthi*vecb );
	       rigorousdisc( 0, true, fl, fray, fthi*vecb );
        } else {
	       rigorousdisc( 0, true, fr, fray, -fthi*vecb );
	       rigorousdisc( 0, true, rr, rray, -rthi*vecb );
	       kindofcube(false,false,RefP,anga,angb,angc,lena,lenb,lenc);
	       rigorousdisc( 0, true, fl, fray, fthi*vecb );
	       rigorousdisc( 0, true, rl, rray, rthi*vecb );
        }
	   } else {
        if (auxn < -0.5*lenb) {
	       if (auxm > 0) {
	         rigorousdisc( 0, true, rl, rray, rthi*vecb );
	         rigorousdisc( 0, true, fl, fray, fthi*vecb );
	         kindofcube(false,false,RefP,anga,angb,angc,lena,lenb,lenc);
	         rigorousdisc( 0, true, rr, rray, -rthi*vecb );
	         rigorousdisc( 0, true, fr, fray, -fthi*vecb );
	       } else {
	         rigorousdisc( 0, true, fl, fray, fthi*vecb );
	         rigorousdisc( 0, true, rl, rray, rthi*vecb );
	         kindofcube(false,false,RefP,anga,angb,angc,lena,lenb,lenc);
	         rigorousdisc( 0, true, fr, fray, -fthi*vecb );
	         rigorousdisc( 0, true, rr, rray, -rthi*vecb );
          }
        } else {
	       if (auxm > 0) {
	         rigorousdisc( 0, true, rl, rray, rthi*vecb );
	         rigorousdisc( 0, true, fl, fray, fthi*vecb );
	         rigorousdisc( 0, true, rr, rray, -rthi*vecb );
	         rigorousdisc( 0, true, fr, fray, -fthi*vecb );
	         kindofcube(false,false,RefP,anga,angb,angc,lena,lenb,lenc);
          } else {
	         rigorousdisc( 0, true, fl, fray, fthi*vecb );
	         rigorousdisc( 0, true, rl, rray, rthi*vecb );
	         rigorousdisc( 0, true, fr, fray, -fthi*vecb );
	         rigorousdisc( 0, true, rr, rray, -rthi*vecb );
	         kindofcube(false,false,RefP,anga,angb,angc,lena,lenb,lenc);
          }
        }
      }
    }
	
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
//%%% Differential Equations:
	    
// Oh! Well... I couldn't do without differential equations.
// The point is that I want to draw tripleial field lines in space.
// Keep it simple: second-order Runge-Kutta method.

    triple fieldlinestep( triple Spos, real Step, triple VecFunc(triple ) ) {
	   triple kone, ktwo;
	   kone = Step*VecFunc( Spos );
	   ktwo = Step*VecFunc( Spos+0.5*kone );
	   return Spos+ktwo;
    }


    path fieldlinepath( int Numb, triple Spos, real Step,
                        triple VecFunc(triple ) ) {
	   real ind;
	   triple prevpos, thispos;
	   guide flpath;
      
	   prevpos = Spos;
	   flpath = rp( Spos );
	   for (int ind=1; ind <= Numb; ind += 1) {
	     thispos = fieldlinestep( prevpos, Step, VecFunc );
	     flpath = flpath .. rp( thispos );
	     prevpos = thispos;
      }
	   return (path)flpath;
    }
    
// Another point is that I want to draw trajectories in space.

    path trajectorypath( int Numb, triple Spos, triple Svel, 
                         real Step, triple VecFunc(triple ) ) {
	   triple prevpos, thispos, prevvel, thisvel;
	   triple rone, rtwo, vone, vtwo;
	   guide flpath;
      
	   prevpos = Spos;
	   prevvel = Svel;
	   flpath = rp( Spos );
	   for (int ind=1; ind <= Numb; ind+=1) {
	     vone = Step*VecFunc( prevpos );
	     rone = Step*prevvel;
	     vtwo = Step*VecFunc( prevpos+0.5*rone );
	     rtwo = Step*( prevvel+0.5*vone );
	     thisvel = prevvel+vtwo;
	     thispos = prevpos+rtwo;
	     flpath = flpath..rp( thispos );
	     prevpos = thispos;
	     prevvel = thisvel;
      }
	   return (path)flpath;
    }

//% And now i stop.

    path magnetictrajectorypath( int Numb, triple Spos, triple Svel, 
                                 real Step, triple VecFunc(triple ) ) {
	   triple prevpos, thispos, prevvel, thisvel;
	   triple rone, rtwo, rthr, rfou, vone, vtwo, vthr, vfou;
	   guide flpath;
      
	   prevpos = Spos;
	   prevvel = Svel;
	   flpath = rp( Spos );
	   for (int ind=1; ind <= Numb; ind+=1 ) {
	     vone = Step*cross( VecFunc( prevpos ), prevvel );
	     rone = Step*prevvel;
	     vtwo = Step*cross( VecFunc( prevpos+0.5*rone ), prevvel+0.5*vone );
	     rtwo = Step*( prevvel+0.5*vone );
	     vthr = Step*cross( VecFunc( prevpos+0.5*rtwo ), prevvel+0.5*vtwo );
	     rthr = Step*( prevvel+0.5*vtwo );
	     vfou = Step*cross( VecFunc( prevpos+rthr ), prevvel+vthr );
	     rfou = Step*( prevvel+vthr );	    
	     thisvel = prevvel+(vtwo+vthr)/3+(vone+vfou)/6;
	     thispos = prevpos+(rtwo+rthr)/3+(rone+rfou)/6;
	     flpath = flpath..rp( thispos );
	     prevpos = thispos;
	     prevvel = thisvel;
      }
	   return (path)flpath;
    }

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%% Part II:
%%%%%%%%%%%%%%%%%%%%%%%%%%%% Advanced 3D-Object Definition Functions %%%%%%%%
% Please check the examples in planpht.mp or the default object below %%%%%*/
 
 // TN : var arg list
  void makeline(int n, int vertices[]) {
    int counter;
    counter = 0;
    for (int ind=vertices[counter]; 
             counter < vertices.length; counter += 1) {
      L[n][counter] = V[ind];
    }
    npl[n] = counter;
    // TN : this explains the 2nd par p 12 of ref manual
    NL = n;
  }
    
    void makeface( int n, int vertices[] ) {
      int counter;
      counter = 0;
      for (int ind=vertices[counter]; 
              counter < vertices.length; counter += 1) {
        F[n][counter] = V[ind];
      }
      npf[n] = counter;
      NF = n;
      FCD[NF] = false;
    }

    void getready( string commstr, triple refpoi ) {
      ++Nobjects;
      ostr[Nobjects] = commstr;
      RefDist[Nobjects] = conorm( f - refpoi );
    }

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Definition of a 3D-Object %%%
// define vertices
/* TN :
    V1 = (1,0,0);
    V2 = (0,0,0);
    V3 = (0,1,0);
    V4 = (-0.3,0.2,1);
    V5 = (1,0,1);
    V6 = (0,1,1);
    V7 = (0,0,2);
    V8 = (-0.5,0.6,1.2);
    V9 = (0.6,-0.5,1.2);
    makeline1(8,9);
    makeface1(1,2,7);
    makeface2(2,3,7);
    makeface3(5,4,6);
*/
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% or the old way below %%%%%%%%%%%%
% define lines
%    NL := 1;     % number of lines
%    npl1 := 2;   % number of vertices of the first line
%    L1p1 := V8;
%    L1p2 := V9;
% define faces
%    NF := 3;     % number of faces
%    npf1 := 3;   % number of vertices of the first face
%    F1p1 := V1;
%    F1p2 := V2;
%    F1p3 := V7;
%    npf2 := 3;   % number of vertices of the second face
%    F2p1 := V2;
%    F2p2 := V3;
%    F2p3 := V7;
%    npf3 := 3;   % number of vertices of the third face
%    F3p1 := V5;
%    F3p2 := V4;
%    F3p3 := V6;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
    
// Flip first argument accordingly to the second

    triple flip(triple A, triple B) {
      triple nv;
      if (dot( A, B) < 0)
        nv = -A;
      else
        nv = A;
      return nv;
    }

// Frontside of a face given by three of its vertices

    triple face(triple A, triple B, triple C) {
      triple nv;
      nv = ncrossprod( A-B, B-C );
      return flip( nv, f-B );
    }

// Center or inside of a face

    triple masscenter(int Nsides, triple Coords[]) {
      real  counter;
      triple mc;
      mc = (0,0,0);
      for (int counter=1; counter <= Nsides; counter+=1) 
        mc = mc + Coords[counter];
      return mc / Nsides;
    }

// Direction of coverability. The trick is here. The condition for visibility
// is that the angle beetween the triple that goes from the side of a face to
// the mark position and the covertriple must be greater than 90 degrees.

    triple cover(triple A, triple B, triple MassCenter) {
            triple nv;
            nv = ncrossprod( A-f, B-f );
            return flip( nv, MassCenter-B );
    }

// O.K., the following macro tests the visibility of a point

    bool themarkisinview(triple Mark, int OwnFace) {
      triple c, faceVec, centerPoint, coverVec;
      bool inview;
      int l, m;
      l = 0;
      while(true) {            // cycle over faces until the mark is covered
        ++l;
        if (l == OwnFace)
          ++l;
        if (l > NF) break;
        faceVec = face(F[l][1],F[l][2],F[l][3]);
        inview = true;
        if (dot(Mark-F[l][1], faceVec) < 0) {
          centerPoint = masscenter(npf[l], F[l]);
          m = 0;
          while(true) {                 // cycle over segments of a face 
            ++m;
            if (m > npf[l]) break;
            if (m < npf[l])
              c = F[l][m+1];
            else
              c = F[l][1];
            coverVec = cover(F[l][m], c, centerPoint);
            inview = dot(Mark-c,coverVec) <= 0;
            if (inview) break;
          }
        }
        if (!inview) break;
      }
      return inview;
    }

// Check for possible intersection or crossing.

    bool maycrossviewplan(triple Ea, triple Eb,
                          triple La, triple Lb) {
      return (abs( dot( cross(Ea-f,Eb-f), La-Lb ) ) > 0.001);
    }

// Calculate the intersection of two sides. This is very nice.

    triple crossingpoint(triple Ea, triple Eb, triple La, triple Lb) {
      triple thecrossing, perpend;
      real exten, aux;
      if ( Ea == Lb || Ea == La )
         thecrossing = Ea;
      else if ( Eb == Lb || Eb == La )
         thecrossing = Eb;
      else {
         perpend = cross( Ea-f, Eb-f );
         if ( conorm( perpend ) == 0 )
           thecrossing = Eb;
         else {
           aux = dot( perpend, f );
           /* TN : 4x4 system of linear equalities
           dot( perpend, thecrossing ) = aux; 
           ( La-Lb )*exten = La-thecrossing; */
         }
      }
      return thecrossing;
    }


// Calculate the intersection of an edge and a face.
    triple crossingpointf(triple Ea, triple Eb, int Fen) {
      triple thecrossing, perpend;
      real exten;
      perpend = cross( F[Fen][1]-F[Fen][2], F[Fen][3]-F[Fen][2] );
      // TN : 4x4
      //dot( perpend, thecrossing ) = dot( perpend, F[Fen]p2 );
      //( Ea-Eb )*exten = Ea-thecrossing; 
      return thecrossing;
    }

// Check for possible intersection of an edge and a face.

    bool maycrossviewplanf(triple Ea, triple Eb, int Fen) 
    {
      triple perpend;
      perpend = cross( F[Fen][1]-F[Fen][2], F[Fen][3]-F[Fen][2] );
      return ( abs( dot( perpend, Ea-Eb ) ) > 0.001 );
    }

// The intersection point must be within the extremes of the segment.

    bool insidedge(triple Point, triple Ea, triple Eb) {
      real fract;
      fract = dot( Point-Ea, Point-Eb );
      return fract < 0;
    }

// Skip edges that are too far away

    bool insideviewsphere(triple Ea, triple Eb, 
                          triple La, triple Lb) {
      triple nearestofline, furthestofedge;
      bool flag;
      real exten;
      nearestofline = La+exten*(Lb-La);
      // TN : dot( nearestofline-f, Lb-La ) = 0;
      if (conorm(Ea-f) < conorm(Eb-f))
        furthestofedge = Eb;
      else
        furthestofedge = Ea;
      if (conorm(nearestofline-f) < conorm(furthestofedge-f))
        flag = true;
      else
        flag = false;
      return flag;
    }


// The intersection point must be within the triangle defined by 
// three points. Really smart.

    bool insidethistriangle(triple Point, triple A, triple B, triple C ) {
            triple arep, area, areb, aret;
            bool flag;
            aret = cross( A-C, B-C );
            arep = flip( cross( C-Point, A-Point ), aret );
            area = flip( cross( A-Point, B-Point ), aret );
            areb = flip( cross( B-Point, C-Point ), aret );
            flag = ( conorm( arep + area + areb ) <= 2*conorm( aret ) ); 
            return flag;
    }

// The intersection point must be within the triangle defined by the
// point of view f and the extremes of some edge. 

    bool insideviewtriangle(triple Point, triple Ea, triple Eb) {
        return insidethistriangle( Point, Ea, Eb, f );
    }

// The intersection point must be within the face

    bool insidethisface(triple Point, int FaN) {
            bool flag;
            int m;
            triple central;
            m = npf[FaN];
            central = masscenter( m, F[FaN] );
            flag = insidethistriangle( Point, 
                              central, F[FaN][m], F[FaN][1] );
            for (int m=2; m <= npf[FaN]; m += 1) {
                if (flag) break;
                flag = insidethistriangle( Point, 
                              central, F[FaN][m-1], F[FaN][m] );
            }
            return flag;
    }

// Draw the visible parts of a straight line in beetween points A and B
// changing the thickness of the line accordingly to the distance from f

    void coarse_line(triple A, triple B, int Facen, real Press, pen Col) {
            int k;
            triple mark, stepVec;
            stepVec = resolvec(A,B); 
            k = 0;
            while (true) {                    // cycle along a whole segment 
                mark = A+(k*stepVec);
                if (dot(B-mark,stepVec) < 0) break;
                if (themarkisinview(mark,Facen))
                    signalvertex(mark, Press, Col);
                ++k;
            }
    }

// Get the 2D rigorous projection path of a face.
// Don't use SphericalDistortion here.

    path facepath(int Facen) {
      guide thispath;
      real counter;
      thispath = rp(F[Facen][1]);
      for (int counter=2; 
               counter <= (npf[Facen]);
               counter += 1) 
        thispath = thispath -- rp(F[Facen][counter]);
      thispath = thispath -- cycle;
      return (path)thispath;
    }

    path faceshadowpath(int Facen) {
      path thispath;
      real counter;
      
      thispath = rp(cb(F[Facen][1]));
      for (int counter=2; counter <= (npf[Facen]); counter += 1)
        thispath = thispath -- rp(cb(F[Facen][counter]));
      thispath = thispath -- cycle;
      return thispath;
    }

// FillDraw a face

    void face_invisible( picture pic=currentpicture, Label L="",
                         int Facen, pen p=currentpen, arrowbar arrow=None,
			 arrowbar bar=None, string legend="" ) {
      path ghost;
      ghost = facepath( Facen );
      unfill( ghost );
      draw(pic, L, ghost, p, arrow, bar, legend);
    }

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
//%%%% Different kinds of renderers:

// Draw only the faces, calculating edge-intersections. 
// Mind-blogging kind of thing.
// Only two constraints: i) faces must be planar and ii) faces must be convex.

    void sharpraytrace() {
//                     ( expr LabelcrossPoints ) 
          int i, j, k, l, counter;
          real swapn;
          triple a, b, c, d, currcross, swapc;
          bool flag, trythis;
          path refpath, otherpath;
          pair intertimes;
          string infolabel;
          triple crosspoin[];
          real sortangle[];
          for (int i=1; i <= NF; i += 1) {                   // scan all faces
             for (int j=1; j <= npf[i]; j += 1) {            // scan all edges
                a = F[i][j];
                if (j != npf[i])
                   b = F[i][j+1];
                else
                   b = F[i][1];
                crosspoin[1] = a;
                counter = 2;
                refpath = rp(a)--rp(b);
                for (int k=1; k <= NF; k += 1) {
                   otherpath = facepath(k);
                   intertimes = intersect( refpath, otherpath );
                   trythis = intertimes.x != 0;
                   if (trythis && (intertimes.x != 1) && (k != i)) {
                      for (int l=1; l <= npf[k]; l += 1) {
                         c = F[k][l];
                         if (l < npf[k])
                            d = F[k][l+1];
                         else
                            d = F[k][1];
                         if (insideviewsphere( a, b, c, d )) {
                           if (maycrossviewplan( a, b, c, d )) {
                             currcross = crossingpoint( a, b, c, d );
                             if (insideviewtriangle( currcross, a, b )) {
                               if (insidedge( currcross, c, d )) {
                                 swapc = cross( a-b, f-currcross);
                                 swapc = cross(swapc,f-currcross);
                                 
                                 triple somepo;
                                 int fract;
                                /* TN :
                                 (b-a)*fract = somepo-a;
                                 dot(swapc,somepo)=dot(swapc,f);
                               */  
                                 if (fract>0 && fract<1) {
                                   crosspoin[counter] = somepo;
                                   ++counter;
                                 }
                               }
                             }
                           }
                         }
                      }
                      if (maycrossviewplanf( a, b, k )) {
                         currcross = crossingpointf( a, b, k );
                         if (insidethisface( currcross, k )) {
                            if (insidedge( currcross, a, b )) {
                               crosspoin[counter] = currcross;
                               ++counter;
                            }
                         }
                      }
                   }
                }
                crosspoin[counter] = b;
                sortangle[1] = 0;
                for (int k=2; k <= counter; k += 1) 
                   sortangle[k] = conorm(crosspoin[k]-a);
                while (true) {
                   flag = true;
                   for (int k=2; k <= counter; k += 1) {
                      if (sortangle[k] < sortangle[k-1]) {
                         swapn = sortangle[k-1];
                         sortangle[k-1] = sortangle[k];
                         sortangle[k] = swapn;
                         swapc = crosspoin[k-1];
                         crosspoin[k-1] = crosspoin[k];
                         crosspoin[k] = swapc;
                         flag = false;
                      }
                   }
                   if (flag) break;
                }
                for (int k=2; k <= counter; k += 1) {
                   swapc = resolvec(crosspoin[k-1],crosspoin[k]);
                   flag = themarkisinview( crosspoin[k-1]+swapc, i );
                   if (flag && themarkisinview( crosspoin[k]-swapc, i )) 
                      draw( rp(crosspoin[k-1])--rp(crosspoin[k]) );
                }
//                if LabelcrossPoints:
//                   for k=1 <= counter:
//                      infolabel:=decimal(i)&","&decimal(j)&","&decimal(k); 
//                      infolabel := "0";
//                      dotlabelrand(infolabel,rp(crosspoin[k]));
//                   endfor;
//                fi;
             }
          }
    }

// Draw three-dimensional lines checking visibility. 

    void lineraytrace(real Press, pen Col) {
            int i, j;
            triple a, b;
            for (int i=1; i <= NL; i += 1) {        // scan all lines
                for (int j=1; j <= npl[i]-1; j += 1) {
                    a = L[i][j];
                    b = L[i][j+1]; 
                    coarse_line( a, b, 0, Press, Col);
                }
            }
    }

// Draw only the faces, rigorously projecting the edges. 

    void faceraytrace(real Press, pen Col) {
            int i, j;
            triple a, b;
            for (int i=1; i <= NF; i += 1) {     // scan all faces
                for (int j=1; j <= npf[i]; j += 1) {
                    a = F[i][j];
                    if (j != npf[i])
                        b = F[i][j+1];
                    else
                        b = F[i][1];
                    coarse_line( a, b, i, Press, Col);
                }
            }
    }


// Fast test for your three-dimensional object

    void draw_all_test( pen Col, bool AlsoDrawLines ) {
	   triple a, b;
	   if (ShadowOn) {
	     for (int i=1; i <= NF; i += 1)
	       fill( faceshadowpath( i ) );
	     if (AlsoDrawLines) {
	       for (int i=1; i <= NL; i += 1) {         // scan all lines
	         for (int j=1; j <= npl[i]-1; j += 1) {
		        a = L[i][j];
		        b = L[i][j+1]; 
		        drawsegment( cb(a), cb(b), Col);
            }
          }
        }
      }
	   for (int i=1; i <= NF; i += 1) {           // scan all faces
	     for (int j=1; j <= npf[i]; j += 1) {
	       a = F[i][j];
	       if (j != npf[i]) 
	         b = F[i][j+1];
	       else
	         b = F[i][1];
	       drawsegment(a, b, Col);
        }
      }
	   if (AlsoDrawLines) {
	     for (int i=1; i <= NL; i += 1) {       // scan all lines
	       for (int j=1; j <= npl[i]-1; j += 1) {
	         a = L[i][j];
	         b = L[i][j+1]; 
	         drawsegment(a, b, Col);
          }
        }
      }
    }

// Don't use SphericalDistortion here.
    // TN
    void fill_faces( picture pic=currentpicture, Label L="",
		     pen p=currentpen, arrowbar arrow=None,
		     arrowbar bar=None, string legend="") 
    {
	   if (ShadowOn) {
	     for (int i=1; i <= NF; i += 1)
          fill(faceshadowpath( i ));
      }
      for (int i=1; i <= NF; i += 1)
          face_invisible( pic, L, i, p, arrow, bar, legend );
    }

  void doitnow() {
    int farone[];
	 for (int i=1; i <= Nobjects; i += 1)
	   farone[i] = i;
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Shell's Method of sorting %%%%%%%%
    int inc, j, vv;
    real v;
	 inc = 1;
    while (true) {
	  inc = 3*inc+1;
	  if (inc > Nobjects) break;
    }
    while (true) {
	   inc = round(inc/3);
	   for (int i=inc+1; i <= Nobjects; i += 1) {
	     v = RefDist[i];
	     vv = farone[i];
	     j = i;
        while (true) {
	      if (!(RefDist[j-inc] > v)) break;
	      RefDist[j] = RefDist[j-inc];
	      farone[j] = farone[j-inc];
	      j = j-inc;
	      if (j <= inc) break;
        }
	     RefDist[j] = v;
	     farone[j] = vv;
      }
	   if (!(inc > 1)) break;
    }
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	 for (int i=Nobjects; i >= 1; i -= 1) {
	   j = farone[i];
	   // TN : scantokens ostr[j];
    }
  }
    
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
//%% Nematic Liquid Crystal wise:

    void generatedirline(int Lin, real Phi, real Theta, 
                        real Long, triple Currpos ) {
            triple longvec;
            npl[Lin] = 2;
            longvec = Long*( Cos(Phi)*Cos(Theta),
                                   Sin(Phi)*Cos(Theta),
                                   Sin(Theta) );
            L[Lin][1] = Currpos-0.5*longvec;
            L[Lin][2] = Currpos+0.5*longvec;
    }

    void generatedirface(int Fen, real Phi, real Theta, 
                         real Long, real Base, triple Currpos ) {
            triple basevec, longvec;
            npf[Fen] = 3;
            longvec = Long*( Cos(Phi)*Cos(Theta),
                                   Sin(Phi)*Cos(Theta),
                                   Sin(Theta) );
            basevec = Base*ncrossprod( Currpos-f, longvec );
            F[Fen][1] = Currpos-0.5*(longvec+basevec);
            F[Fen][2] = Currpos+0.5*longvec;
            F[Fen][3] = Currpos-0.5*(longvec-basevec);
    }

    void generateonebiax(int Lin, real Phi, real Theta, real Long, 
                         real SndDirAngl, real Base, triple Currpos ) {
            triple basevec, longvec, u, v;
            npl[Lin] = 4;
            longvec = Long*( Cos(Phi)*Cos(Theta),
                                   Sin(Phi)*Cos(Theta),
                                   Sin(Theta) );
            v = (-Sin(Phi), Cos(Phi), 0);
            u = ( Cos(Phi)*Cos(Theta+90),
                        Sin(Phi)*Cos(Theta+90),
                        Sin(Theta+90) );
            basevec = Base*( v*Cos(SndDirAngl)+u*Sin(SndDirAngl) );
            L[Lin][1] = Currpos-0.5*longvec;
            L[Lin][2] = Currpos+0.5*basevec;
            L[Lin][3] = Currpos+0.5*longvec;
            L[Lin][4] = Currpos-0.5*basevec;
    }

    void director_invisible( bool SortEmAll, real ThickenFactor, bool CyclicLines ) {
	   real dist[], thisfar, ounum;
      int farone[], j;
	   pen actualpen, outerr, innerr;
	   path direc;
	   actualpen = currentpen;
	   if (SortEmAll) {
	     for (int i=1; i <= NL; i += 1) {                    // scan all lines
	       dist[i] = conorm( masscenter( npl[i], L[i] ) - f );
	       farone[i] = i;
        }
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Shell's Method of sorting %%%%%%%%
	     int inc, vv;
        real v;
	     inc = 1;
	     while(inc <= NL) 
	       inc = 3*inc+1;
	     while(true) {
	       inc = round(inc/3);
	       for (int i=inc+1; i <= NL; i += 1) {
	         v = dist[i];
	         vv= farone[i];
	         j = i;
            while (j <= inc) {
		        if (!(dist[j-inc] > v)) break;
		        dist[j] = dist[j-inc];
		        farone[j] = farone[j-inc];
		        j = j-inc;
            }
	         dist[j] = v;
	         farone[j] = vv;
          }
	       if (!(inc > 1)) break;
        }
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	   } else {
	     for (int i=1; i <= NL; i += 1) 
	       farone[i] = i;
      }
	   for (int i=NL; i >= 1; i -= 1) {                 // draw all pathes
	     j = farone[i];
	     direc = rp( L[j][1] );
	     for (int k=2; k <= npl[j]; k += 1)
	       direc = direc--rp( L[j][k] );
	     if (CyclicLines)
	       direc = direc--cycle;
	     ounum = Spread*ps( masscenter( npl[j], L[j] ), ThickenFactor );
        // TN : pencircle
	     outerr = linewidth(ounum);
	     innerr = linewidth(0.8*ounum); //%% DANGER %%
	     currentpen = outerr;
	     draw(direc,black);
	     currentpen = innerr;
	     draw(direc, background);
      }
	   currentpen = actualpen;
    }

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
//%% Plotting:

    void fillfacewithlight( int FaceN ) {
	   triple perpvec, reflectio, viewvec, inciden, refpos, projincid;
	   pen fcol, lcol, lowcolor;
      triple pa, pb, pc, shiftv;
	   real theangle;
      int j;
	   path ghost;
	   ghost = rp( F[FaceN][1] );
	   for (int j=2; j <= npf[FaceN]; j += 1)
	     ghost = ghost--rp( F[FaceN][j] );
	   ghost = ghost--cycle;
	   if (OverRidePolyhedricColor)
	     unfill(ghost);
	   else {
	     refpos = masscenter( npf[FaceN], F[FaceN] );
	     pa = F[FaceN][1];
	     pb = F[FaceN][2];
	     pc = F[FaceN][3];
	     inciden = LightSource - refpos;
	     viewvec = f - refpos;
	     perpvec = ncrossprod( pa-pb, pb-pc );
	     if (dot( perpvec, (0,0,1) ) < 0) 
	       perpvec = -perpvec;
	     projincid = perpvec*dot( perpvec, inciden );
	     shiftv = inciden - projincid;
	     reflectio = projincid - shiftv;
	     theangle = getangle( reflectio, viewvec );
	     if (FCD[FaceN])
	       lowcolor = TableC[FC[FaceN]];
	     else
	       lowcolor = TableC[0];
	     lcol = interp(lowcolor,HigColor,Cos( theangle ));
/*	     if (dot( viewvec, perpvec ) < 0)
	       fcol = lcol - SubColor;
	     else
	       fcol = lcol;*/
	     fill(ghost, fcol);
      }
	   draw(ghost);
    }

// Define and draw surfaces with a triangular mesh.
// On a hexagonal or triangular area. Without sorting (no need).
   
    void hexagonaltrimesh( bool BeHexa, int theN, real SideSize, 
                           real SurFunc(real, real) ) {
	   real posx, posy, posz, higx, higy,
           stepx, stepy, lowx, lowy;
      int counter, newn, i;
	   triple poi[][];
	   bool bola, bolb, bolc;

	   npf[0] = 3;
	   FCD[0] = true; // this is used in the calls to fillfacewithlight
	   ++ActuC;
	   if (ActuC > TableColors)
	     ActuC = 1;
	   FC[0] = ActuC;    
	   counter = 0;
      stepy = SideSize/theN;
      stepx = 0.5*stepy*sqrt(3);
	   lowy = -0.5*SideSize;
	   lowx = -sqrt(3)*SideSize/6;
	   higy = -lowy;
	   higx = sqrt(3)*SideSize/3;
	   for (int i=0; i <= theN; i += 1) {
	     for (int j=0; j <= theN-i; j += 1) {
          posx = lowx + i*stepx;
          posy = lowy + i*stepx/sqrt(3) + j*stepy;
	    	 posz = SurFunc( posx, posy );
	    	 poi[i][j] = ( posx, posy, posz );
        }
      }
      if (BeHexa) 
	     newn = round((theN+1)/3)+1;
	   else
	     newn = 1;
	   for (int j=newn; j <= theN-newn+1; j += 1) {
	        F[0][1] = poi[0][j-1];
	        F[0][2] = poi[0][j];
	        F[0][3] = poi[1][j-1];
	        fillfacewithlight( 0 );   // see below
      }
	   for (int i=1; i <= theN-1; i += 1) {
	     for (int j=1; j <= theN-i; j += 1) {
		    bola = ( i < newn ) && ( j < newn-i );
		    bolb = ( i < newn ) && ( j > theN-newn+1 );
		    bolc = ( i > theN-newn );
		    if (!(bola || bolb || bolc) ) {
	         F[0][1] = poi[i-1][j];
			   F[0][2] = poi[i][j-1];
			   F[0][3] = poi[i][j];
			   fillfacewithlight( 0 );
			   F[0][1] = poi[i+1][j-1];
			   fillfacewithlight( 0 );
          }
        }
      }
	   i = theN-newn+1;
	   for (int j=1; j <= newn-1; j += 1) {
	     F[0][1] = poi[i-1][j];
		  F[0][2] = poi[i][j-1];
		  F[0][3] = poi[i][j];
		  fillfacewithlight( 0 );
      }
    }
	
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%% Part III (parametric plots and another renderer):
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%                                         %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%   Kindly contributed by Jens Schwaiger  %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%                                         %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/

// dmin_ is the minimal distance of all faces to f;
// dmax_ is the maximal distance of all faces to f;
// (both values are determined in "draw_invisible")
// Facen is the number of the face to be filled;
// ColAtrib is the color used for filling;
// ColAtribone the color used for drawing;
// Colordensity depends on distance of the face from f

    void face_drawfill( int Facen, real dmin_, real dmax_ , pen ColAtrib, pen ColAtribone ) {
	   path ghost;
	   real colfac_;
	   triple ptmp, coltmp_;
	   ghost = rp( F[Facen][1] );
	   for (int j=2; j <= npf[Facen]; j += 1) 
	     ghost = ghost--rp( F[Facen][j] );
	   ghost = ghost--cycle;
	   ptmp = masscenter( npf[Facen], F[Facen] ) - f;
        // 0<=colfac_<=1
	   colfac_ = ( conorm(ptmp)-dmin_ )/( dmax_ - dmin_ );
        // color should be brighter, if distance to f is smaller
	   colfac_ = 1 - colfac_;
        // color should be not to dark, i.e., >=0.1 (and <=1)
	   colfac_ = 0.9colfac_ + 0.1; 
        // now filling and drawing with appropriate color;
	   fill( ghost , colfac_*ColAtrib );
    	draw( ghost , colfac_*ColAtribone );
    }

// Now a much faster faces-only-ray-tracer based upon the unfill
// command and the constraint of non-intersecting faces of similar
// sizes. Faces are sorted by distance from nearest vertex or
// masscenter to the point of view. This routine may be used to 
// draw graphs of 3D surfaces (use masscenters in this case).
//    
// Option=true: test all vertices
// Option=false: test masscenters of faces
// DoJS=true: use face_drawfill by J. Schwaiger
// DoJS=false: use fillfacewithlight by L. Nobre G.
// ColAtrib=color for filling faces
// ColAtribone=color for drawing edges

    void draw_invisible( bool Option, bool DoJS, pen ColAtrib, pen ColAtribone ) {
	   real dist[], thisfar, distmin_, distmax_;
      int farone[], j;
	   triple a, b, ptmp;
	   for (int i=1; i <= NF; i += 1) {      // scan all faces
	     if (Option) {                       // for distances of
	       dist[i] = conorm( F[i][1] - f );   // nearest vertices
	       if (i==1) {
	         distmin_ = dist[1];                // initialisation of
	         distmax_ = dist[1];                // dmin_ and dmax_
          }
	       distmin_ = min( distmin_, dist[i] );
	       distmax_ = max( distmax_, dist[i] );
	       for (int j=2; j <= npf[i]; j += 1) {
	         thisfar = conorm( F[i][j] - f );
	         distmin_ = min( distmin_, thisfar );
	         distmax_ = max( distmax_, thisfar );
	         if (thisfar < dist[i]) 
		        dist[i] = thisfar;
          }
	     } else {                       // for distances of centers of mass
	       dist[i] = conorm( masscenter( npf[i], F[i] ) - f );
	       if (i==1) {
	         distmin_ = dist[1];          // initialisation of
	         distmax_ = dist[1];          // dmin_ and dmax_ in this case 
          }
	       distmin_ = min( distmin_, dist[i] );
	       distmax_ = max( distmax_, dist[i] );
        }
	     farone[i] = i;
      }
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Shell's Method of sorting %%%%%%%%
	   int inc, vv;
      real v;
	   inc = 1;
	   while ( inc <= NF)
	     inc = 3*inc+1;
	   while(true) {
	     inc = round(inc/3);
	     for (int i=inc+1; i <= NF; i += 1) {
	       v = dist[i];
	       vv = farone[i];
	       j = i;
	       while (true) {
	         if (!(dist[j-inc] > v)) break;
	         dist[j] = dist[j-inc];
	         farone[j] = farone[j-inc];
	         j = j-inc;
	         if (j <= inc) break;
          }
	       dist[j] = v;
	       farone[j] = vv;
        }
	     if (!(inc > 1)) break;
      }
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	   for (int i=NF; i >= 1; i -= 1) {                // draw and fill all pathes
	     j = farone[i];
	     if (DoJS) 
	       face_drawfill( j, distmin_, distmax_, ColAtrib, ColAtribone );
	     else
	       fillfacewithlight( j );
      }
    }

// Move to the good range (-1,1).

    real bracket( real low, real poi, real hig ) {
      real zout;
      zout = (2*poi-hig-low)/(hig-low);
      if (zout > 1) zout = 1;
      if (zout < -1) zout = -1;
      return zout;
    }

// Define parametric surfaces with a triangular mesh.
    
    void partrimesh( int nt, int ns, real lowt, real higt,
        real lows, real higs, real lowx, real higx,
        real lowy, real higy, real lowz, real higz,
        real facz, triple parSurFunc(real, real) ) {
	   real posx, posy, posz, poss, post, steps, stept;
      int counter;
	   triple poi[][], tmpaux;
	   counter = NF;                     // <-- NF must be initialized!
	   ++ActuC;
	   if (ActuC > TableColors) 
	     ActuC = 1;
	   steps = ( higs - lows )/ns;
	   stept = ( higt - lowt )/nt;
	   for (int i=0; i <= ns; i += 1) {
	     for (int j=0; j <= nt; j += 1) {
	       poss = lows + i*steps;
	       post = lowt + j*stept;
	       tmpaux = parSurFunc( poss, post );
	       posz = Z(tmpaux); 
	       posx = X(tmpaux);
	       posy = Y(tmpaux);
	       posx = bracket(lowx,posx,higx);      // see below (TN : above )
	       posy = bracket(lowy,posy,higy);      // see below
	       posz = bracket(lowz,posz,higz)/facz; // see below
	       poi[i][j] = ( posx, posy, posz );
        }
      }
	   for (int i=1; i <= ns; i += 1) {
	     for (int j=1; j != nt; j += 1) {
	       for (int k=-1; k <= 0; k += 1) {
	         ++counter;
	         npf[counter] = 3;
	         F[counter][1] = poi[i-1][j-1];
	         F[counter][2] = poi[i+k][j-1-k];
	         F[counter][3] = poi[i][j];
	         FC[counter] = ActuC;
	         FCD[counter] = true;
          }
        }
      }
	   NF = counter;
    }
//% EOF

