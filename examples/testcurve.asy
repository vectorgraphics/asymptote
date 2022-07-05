import three;
import graph;

real NURBStolerance=sqrtEpsilon;

struct NURBSCurveData {
  triple[] controlPoints;
  real[] knots;
  real[] weights;
  int degree=knots.length-controlPoints.length-1;
  void operator init(triple[] controlPoints, real[] knots, real[] weights) {
    this.knots=knots;
    this.controlPoints=controlPoints;
    this.weights=weights;
    this.degree=knots.length-controlPoints.length-1;
  }
}

struct BSpline {
  real[][] controlPoints;
  real[] knots;
  int degree=knots.length-controlPoints.length-1;
  void operator init(real[][] controlPoints, real[] knots) {
    this.knots=knots;
    this.controlPoints=controlPoints;
    this.degree=knots.length-controlPoints.length-1;
  }
}

NURBSCurveData conversion_4DBSpline_to_3DNurbs(BSpline BSpline_4D) {
  NURBSCurveData new_nurb;
  triple[] x=new_nurb.controlPoints;
  real[] weights=new_nurb.weights;
  for(int j=0; j < BSpline_4D.controlPoints.length; ++j) {
    real[] control_point_4D=BSpline_4D.controlPoints[j];
    real weight=control_point_4D[3];
    triple new_3D_control_point=(control_point_4D[0],control_point_4D[1],control_point_4D[2])/weight;
    weights.push(weight);
    x.push(new_3D_control_point);
  }
  new_nurb.degree=BSpline_4D.degree;
  new_nurb.controlPoints=x;
  new_nurb.knots=BSpline_4D.knots;
  new_nurb.weights=weights;

  return new_nurb;
}

BSpline conversion_3DNurbs_to_4DBSpline(NURBSCurveData NURBSS_3D) {
  BSpline new_BSpline_4D;
  real[][] x=new_BSpline_4D.controlPoints;
  for(int j=0; j < NURBSS_3D.controlPoints.length; ++j) {
    triple control_point_3D=NURBSS_3D.controlPoints[j];
    real weight=NURBSS_3D.weights[j];
    real[] new_control_point_4D={control_point_3D.x*weight,control_point_3D.y*weight,control_point_3D.z*weight,weight};
    x.push(new_control_point_4D);
  }
  //   write("NURBSS_3D.degree");
  //   write(NURBSS_3D.degree);
  new_BSpline_4D.degree=NURBSS_3D.degree;
  new_BSpline_4D.controlPoints=x;
  new_BSpline_4D.knots=copy(NURBSS_3D.knots);

  return new_BSpline_4D;
}

real[][] BezierMultiDegreeElevate(real[][] input_control_points, int r) {
  // elevate Bezier curve by degree r
  int d=input_control_points[0].length; // dimension of input control points
  int n=input_control_points.length;
  int p=n-1; // p is the degree of the curve
  int elevated_cp_len=n+r;
  int elevated_cp_len_half=floor((elevated_cp_len-1)/2);
  real[][] bezcoefs=new real[elevated_cp_len][n];
  real[] init_array=array(d,0.0);
  real[][] elevated_cp=array(elevated_cp_len,init_array);
  bezcoefs[0][0]=1.0;
  bezcoefs[elevated_cp_len-1][p]=1.0;
  for(int i=1; i <= elevated_cp_len_half; ++i) {
    real inv=1.0/choose(elevated_cp_len-1,i);
    int mpi=min(p,i);
    for(int j=max(0,i-r); j <= mpi; ++j) {
      bezcoefs[i][j]=inv*choose(p,j)*choose(r,i-j);
    }
  }
  for(int i=elevated_cp_len_half+1; i < elevated_cp_len-1; ++i) {
    int mpi=min(p,i);
    for(int j=max(0,i-r); j <= mpi; ++j) {
      bezcoefs[i][j]=bezcoefs[elevated_cp_len-1-i][p-j];
    }
  }
  elevated_cp[0]=input_control_points[0];
  elevated_cp[elevated_cp_len-1]=input_control_points[p];
  for(int i=1; i < elevated_cp_len-1; ++i) {
    int mpi=min(p,i);
    for(int j=max(0,i-r); j <= mpi; ++j) {
      elevated_cp[i]=elevated_cp[i]+bezcoefs[i][j]*input_control_points[j];
    }
  }

  return elevated_cp;
}

BSpline DegreeElevationCurve(BSpline curve_data, int t) {
  // Degree elevate a curve t times
  int p=curve_data.degree;
  int m=curve_data.knots.length;
  int n=curve_data.controlPoints.length;
  int ph=p+t;
  int ph2=floor(ph/2);
  real[] U=copy(curve_data.knots);
  real[][] Pw=copy(curve_data.controlPoints);

  int mh=ph;
  int kind=ph+1; // knot index
  int r=-1;
  int a=p;
  int b=p+1;
  int cind=1; // control points index
  real ua=U[0];
  real[][] Qw; // new control points array
  real[] Uh; // new knots array
  real[][] bpts=new real[p+1][] ; // pth degree Bezier control points of current segment
  real[][] elevated_bpts=new real[p+t+1][] ; // p+t th degree Bezier control points of current segment
  real[][] Nextbpts=new real[p-1][] ; // leftmost control points of next Bezier Segment
  real[] alphas=new real[p-1]; // interpolation ratios
  Qw[0]=Pw[0];

  // Initialize first knot segment
  for(int i=0; i <= ph; ++i) {
    Uh[i]=ua;
  }
  // Initialize first Bezier control points segment
  for(int i=0; i <= p; ++i) {
    bpts[i]=Pw[i];
  }

  // variables will be used in the b < m loop
  int oldb; // storing b value
  int multi; // storing multiplicity
  int oldr; // storing r value
  real ub; // knot that is next to and different from the knot ua

  while(b < m) {
    // Loop through knot array
    oldb=b;
    while(b < m-1 && U[b] == U[b+1]) {
      ++b;
    }
    multi=b-oldb+1;
    if(b == m-1 && U[b] != U[b-1]) {
      multi=1;
    }
    ub=U[b];
    oldr=r;
    r=p-multi;

    if(r > 0) {
      // Insert knot to get Bezier segment
      real numer=ub-ua;
      for(int i=p ; i > multi; --i) {
        real denominator=U[a+i]-ua;
        alphas[i-multi-1]=numer/denominator;
      }
      for(int j=1; j <= r; ++j) {
        int save=r-j;
        int s=multi+j;
        for(int k=p; k >= s; --k) {
          bpts[k]=alphas[k-s]*bpts[k]+(1.0-alphas[k-s])*bpts[k-1];
        }
        Nextbpts[save]=bpts[p];
      }
    }

    // End of knot inserton

    //Bezier Degree Elevation
    elevated_bpts=BezierMultiDegreeElevate(bpts,t);

    if(a != p) {
      // load the knot ua
      for(int i=0; i < ph; ++i) {
        Uh[kind]=ua;
        ++kind;
      }
    }

    for(int j=1; j <= ph; ++j) {
      // load ctrl pts into Qw
      Qw[cind]=elevated_bpts[j];
      ++cind;
    }
    if(b < n) {
      //Set up for the next pass through loop
      for(int j=0; j < r; ++j) {
        bpts[j]=Nextbpts[j];
      }
      for(int j=r; j <= p; ++j) {
        bpts[j]=Pw[b-p+j];
      }
      a=b;
      ++b;
      ua=ub;
    }
    else{
      //end knot
      for(int i=0; i <= ph; ++i) {
        Uh[kind+i]=ub;
      }
      break;
    }// end of while loop(b < m)
  }
  BSpline new_curve;
  new_curve.controlPoints=Qw;
  //   write("new curve control points");
  //   write(new_curve.controlPoints);
  new_curve.knots=Uh;
  //   write("new curve knots");
  //   write(new_curve.knots);
  return new_curve;
}


real[][] BezDegreeReduce(real[][] bpts) {
  int p=bpts.length-1; // p=num of control_point of reduced Bezier Curve
  real[][] rcpts=new real[p][] ; // reduced control points
  int r=floor((p-1)/2);
  rcpts[0]=bpts[0];
  rcpts[p-1]=bpts[p];
  if(p % 2 == 0) {
    for(int i=1; i <= r; ++i) {
      real alphai=i/p;
      rcpts[i]=(bpts[i]-alphai*rcpts[i-1])/(1-alphai);
    }
    for(int i=p-2; i >= r+1; --i) {
      real alphai1=(i+1)/p;
      rcpts[i]=(bpts[i+1]-(1-alphai1)*rcpts[i+1])/(1-alphai1);
    }
  }
  else{ //p is old
    for(int i=1; i <= r-1; ++i) {
      real alphai=i/p;
      real denominator_inverse=1/(1-alphai);
      rcpts[i]=(bpts[i]-alphai*rcpts[i-1])*denominator_inverse;
    }
    for(int i=p-2; i >= r+1; --i) {
      real alphai1=(i+1)/p;
      real denominator_inverse=1/(1-alphai1);
      rcpts[i]=(bpts[i+1]-(1-alphai1)*rcpts[i+1])*denominator_inverse;
    }
    real alphar=r/p;
    real[] leftp=(bpts[r]-alphar*rcpts[r-1])/(1-alphar);
    real alphar1=(r+1)/p;
    real[] rightp=(bpts[r+1]-(1-alphar1)*rcpts[r+1])/alphar1;
    rcpts[r]=(leftp+rightp)/2;
  }
  return rcpts;
}

BSpline DegreeReduceCurve(BSpline curve_data) {
  // reduce the BSpline curve from degree p to p-1
  int n=curve_data.controlPoints.length;
  int p=curve_data.degree;
  int m=curve_data.knots.length; // num of knots
  real[][] bpts=new real[p+1][] ; // control points of the BSpline curve segment limited according to the degree
  int ph=p-1; // reduced Degree
  int r=-1; // r is the time the knot needs to be inserted or removed
  int a=p;
  int b=p+1; // index for knot counting
  int multi=p; // stores the multiplicity of a knot
  real[][] Pw=curve_data.controlPoints; // control points of the input curve
  real[] U=curve_data.knots;

  BSpline new_curve_data; // BSpline for return
  new_curve_data.degree=p-1;
  real[][] new_curve_cp; // control points of the returning curve
  real[] new_curve_knots;
  int kind=p; //index used to assign new curve knots with values
  int cind=1; //index used to assign new curve control points with values
  new_curve_cp[0]=Pw[0];

  for(int i=0; i <= ph; ++i) {
    new_curve_knots[i]=U[0];
  }
  for(int i=0; i <= p; ++i) {
    bpts[i]=Pw[i];
  }

  int oldr=0; // oldr is used to store the previous r in the loop
  real[] alphalist=new real[p-1]; // interpolation ratio array
  real[][] rbpts=new real[p][] ; // control-points of the degree-reduced BSpline curve segment
  real[][] Nextbpts=new real[p-1][] ; // leftmost control points of the current Bezier segment

  // variables will be used in the b < m loop
  int oldb; // storing b value
  int multi; // storing multiplicity
  int lbz;

  while(b < m) {
    int oldb=b;
    while(b < m-1 && U[b] == U[b+1]) {
      ++b;
    }
    multi=b-oldb+1;
    if(b == m-1 && U[b] != U[b-1]) {
      multi=1;
    }
    oldr=r;
    r=p-multi;

    if(oldr > 0) {
      lbz=floor((oldr+2)/2);
    }
    else{
      lbz=1;
    }

    if(r > 0) {
      real numer=U[b]-U[a];
      for(int k=p; k > multi; --k) {
        alphalist[k-multi-1]=numer/(U[a+k]-U[a]);
      }
      for(int j=1; j <= r; ++j) {
        int save=r-j;
        int s=multi+j;
        for(int k=p; k >= s; --k) {
          bpts[k]=alphalist[k-s]*bpts[k]+(1.0-alphalist[k-s])*bpts[k-1];
        }
        Nextbpts[save]=bpts[p];
      }
    }
    // Degree Reduced Bezier Segment
    rbpts=BezDegreeReduce(bpts);

    // update on the output
    if(a != p) {

      for(int i=0; i < ph; ++i) {
        new_curve_knots[kind]=U[a];
        ++kind;
      }
    }
    for(int i=1; i <= ph; ++i) {
      new_curve_cp[cind]=rbpts[i];
      ++cind;
    }

    //Set next Bezier segment and knots ready for looping
    if(b < n) {
      for(int i=0; i < r; ++i) {
        bpts[i]=Nextbpts[i];
      }
      //         write("old curve control points length");
      //         write(Pw.length);

      for(int i=r; i <= p; ++i) {
        bpts[i]=Pw[b-p+i];
      }// m=n+p +1 \\ n-1
      a=b;
      ++b;
    }
    else{
      for(int i=0; i <= ph; ++i) {
        new_curve_knots[kind+i]=U[b];
      }
      break;
    }
  } //end of(b < m) loop

  new_curve_data.controlPoints=new_curve_cp;
  //   write("new curve control points");
  //   write(new_curve_data.controlPoints);
  new_curve_data.knots=new_curve_knots;
  //   write("new curve knots");
  //   write(new_curve_data.knots);
  return new_curve_data;
}

triple de_casteljau_triple(real t, triple[] coefs) {
  int n=coefs.length;
  triple[] local_coefs=copy(coefs);
  for(int i=1; i < n; ++i) {
    for(int j=0; j < n-i; ++j) {
      local_coefs[j]=local_coefs[j]*(1-t)+local_coefs[j+1]*t;
    }
  }
  return local_coefs[0]; // the point on the curve evaluated with given t
}

real de_casteljau_real(real t, real[] coefs) {
  int n=coefs.length;
  real[] local_coefs=copy(coefs);
  for(int i=1; i < n; ++i) {
    for(int j=0; j < n-i; ++j) {
      local_coefs[j]=local_coefs[j]*(1-t)+local_coefs[j+1]*t;
    }
  }
  return local_coefs[0]; // the point on the curve evaluated with given t
}


triple[] PIA(triple[] data_points, triple[] sample_points) {
  // Progressive Iterative Approximation algorithm to find the adjust vectors
  int n=data_points.length;
  triple[] adjusting_vectors=new triple[n]; // adjusting vectors for return
  for(int i=0; i < n; ++i) {
    adjusting_vectors[i]=data_points[i]-sample_points[i];
  }
  return adjusting_vectors;
}

triple[] conversion_RBezier_to_NRBezier(triple[] RBcontrolPoints, triple[] adjust_controlPoints, triple[] sample_points, real tolerance) {

  triple[] local_control_points=copy(adjust_controlPoints); // local control points will be adjusted according to the adjusting vectors
  triple[] local_sample_points=copy(sample_points); // local sample points will be resampled using the newly generated local control points
  int n=local_control_points.length;
  int k=local_sample_points.length;
  bool approximation_bool=true; // change to false if the approximation error(of one vector) is greater than the allowed_error

  triple[] adjusting_vectors=PIA(RBcontrolPoints, local_sample_points);
  for(int i=0; i < n; ++i) {
    real ad_vector_length=length(adjusting_vectors[i]);
    local_control_points[i]=local_control_points[i]+adjusting_vectors[i];
    if(ad_vector_length > tolerance) {
      approximation_bool=false;
    }
  }

  for(int j=0; j < k; ++j) {
    local_sample_points[j]=de_casteljau_triple(j/(k-1),local_control_points);
  }

  if(approximation_bool == true) {
    return local_control_points; // non-rational Bezier control points are returned
  }
  //write(local_control_points);
  return conversion_RBezier_to_NRBezier(RBcontrolPoints,local_control_points,local_sample_points,tolerance);
}

triple RBezier_evaluation(real t, triple[] control_points, real[] weights) {

  int n=control_points.length;
  triple[] weighted_control_points=new triple[n];

  for(int i=0; i < n; ++i) {
    weighted_control_points[i]=control_points[i]*weights[i];
  }

  triple numerator=de_casteljau_triple(t,weighted_control_points);
  real denominator_inverse=1/de_casteljau_real(t,weights);
  triple point_on_curve=numerator*denominator_inverse;

  return point_on_curve;
}

struct NURBScurve {
  path3[] g;

  NURBSCurveData data;

  void operator init(triple[] control_points, real[] knots, real[] weights) {
    data=NURBSCurveData(control_points,knots,weights);
    BSpline BSpline_4D=conversion_3DNurbs_to_4DBSpline(data);
    int BSpline_degree=BSpline_4D.degree;
    int output_degree=3;
    if(BSpline_degree < output_degree) {
      int t=output_degree-BSpline_degree;
      BSpline_4D=DegreeElevationCurve(BSpline_4D,t);
      int BSpline_degree=BSpline_4D.degree;
    }
    while(BSpline_degree > output_degree) {
      BSpline_4D=DegreeReduceCurve(BSpline_4D);
      BSpline_degree=BSpline_4D.degree;
    }

    NURBSCurveData nurb_3D=conversion_4DBSpline_to_3DNurbs(BSpline_4D);
    // This NURBS curve is composed of several Bezier segments
    int Bezier_first_cp_index=0; // First control point index
    int Bezier_last_cp_index=output_degree; // Last control point index
    triple[] nurb_control_points=nurb_3D.controlPoints;
    real[] nurb_weights=nurb_3D.weights;
    int n=nurb_control_points.length;

    while(Bezier_last_cp_index < n) {
      triple[] current_Bezier_cps=nurb_control_points[Bezier_first_cp_index: Bezier_last_cp_index+1];
      real[] current_Bezier_weights=nurb_weights[Bezier_first_cp_index: Bezier_last_cp_index+1];
      int m=Bezier_last_cp_index-Bezier_first_cp_index+1; // number of control points in the current Bezier segment
      triple[] sample_points=new triple[m];
      bool NR_bool=true;
      for(int i=0; i < m; ++i) {
        if(current_Bezier_weights[i] != 1) {
          NR_bool=false;
        }
      }
      if(NR_bool && Bezier_last_cp_index < n) {
        g.push(current_Bezier_cps[0]..controls current_Bezier_cps[1] and current_Bezier_cps[2]..current_Bezier_cps[3]);
      } else {
        for(int i=0; i <= m-1; ++i) {
          sample_points[i]=RBezier_evaluation(i/(m-1),current_Bezier_cps,current_Bezier_weights);
        }
        write(sample_points);
        real tolerance=NURBStolerance*norm(new triple[][] {current_Bezier_cps});
        triple[] NR_Bezier_control_points=conversion_RBezier_to_NRBezier(current_Bezier_cps,current_Bezier_cps,sample_points,tolerance);
        g.push(NR_Bezier_control_points[0]..controls NR_Bezier_control_points[1] and NR_Bezier_control_points[2]..NR_Bezier_control_points[3]);
        
      }
      Bezier_first_cp_index=Bezier_last_cp_index;
      Bezier_last_cp_index=Bezier_last_cp_index+output_degree;
    }
  }

  void draw(picture pic=currentpicture, pen p=currentpen) {
    draw(pic,g,p);
  }

  triple min3,max3;
  bool havemin3,havemax3;

  triple min() {
    if(havemin3) return min3;
    havemin3=true;
    return min3=min(g);
  }

  triple max() {
    if(havemax3) return max3;
    havemax3=true;
    return max3=max(g);
  }
};

size(10cm);
real[] knot={0,0,0,1,2,3,4,4,6,5,5};
triple[] P=
  {
    (-6,-1,0),
    (-5,2,0),
    (-3,3,0),
    (-1,2,0),
    (0,0,0),
    (3,1,0),
    (3,3,0),
    (1,5,0)
  };

triple[] testPoints={
  ( 4.0,-6.0,6.0),
  (-4.0,1.0,0.0),
  (-1.5,5.0,-6.0),
  ( 0.0,2.0,-2.0),
  ( 1.5,5.0,-6.0),
  ( 4.0,1.0,0.0),
  (-4.0,-6.0,6.0)
};
/* triple[] testPoints ={
   (4,6,0),(7,12,0),(11,6,0),(15,2,0),(20,6,0)
   };
   real[] testKnots={0,0,0,0,0.5,1,1,1,1}; */
real[] testKnots={1,1,1,2,3,4,5,6,6,6};
real[] weights=array(testPoints.length,1.0);
weights[4]=5;
weights[5]=6;
NURBScurve n=NURBScurve(testPoints,testKnots,weights);
n.draw();
dot(n.data.controlPoints,red);
//draw(box(n.min(),n.max()),red);