import three;
import graph;

real NURBStolerance=sqrtEpsilon;

int ceilquotient(int a, int b)
{
  return (a+b-1)#b;
}

int[] removeDuplicates(int[] array){
    // Return if array is empty or contains a single element
    int n=array.length;
    if (n==0||n==1){
        return copy(array);
    }

    int[] temp;

    // Start traversing elements
    int j=0;
    // If current element is not equal to next element
    // then store that current element
    for (int i=0;i<n-1;++i){
        if (array[i]!=array[i+1]){
            temp[j]=array[i];
            ++j;
        }
    }

    // Store the last element as whether it is unique or
    // repeated, it hasn't stored previously
    temp[j]=array[n-1];
    ++j;
    return copy(temp);
}

//de_Casteljau algorithm for triple
triple de_casteljau_triple(real t,triple[] coefs) {
  int n=coefs.length;
  triple[] local_coefs=copy(coefs);
  for(int i=1;i<n;++i) {
    for(int j=0;j<n-i;++j) {
      local_coefs[j]=local_coefs[j]*(1-t)+local_coefs[j+1]*t;
    }
  }
  return local_coefs[0]; // the point on the curve evaluated with given t
}

//de_Casteljau algorithm for real
real de_casteljau_real(real t,real[] coefs) {
  int n=coefs.length;
  real[] local_coefs=copy(coefs);
  for(int i=1;i<n;++i) {
    for(int j=0;j<n-i;++j) {
      local_coefs[j]=local_coefs[j]*(1-t)+local_coefs[j+1]*t;
    }
  }
  return local_coefs[0]; // the point on the curve evaluated with given t
}

// evaluate one point on the Rational Bezier curve
triple RBezier_curve_evaluation(real t,triple[] control_points,real[] weights) {

  int n=control_points.length;
  triple[] weighted_control_points=new triple[n];
  triple point_on_curve;
  for(int i=0;i<n;++i) {
    weighted_control_points[i]=control_points[i]*weights[i];
  }
  triple numerator=de_casteljau_triple(t,weighted_control_points);
  real denominator=de_casteljau_real(t,weights);
  if(denominator!=0){
    point_on_curve=numerator/denominator;
  }
  else{
    point_on_curve=(0,0,0);
  }
  return point_on_curve;
}

struct NURBSCurveData {
  triple[] controlPoints;
  real[] knots;
  real[] weights;
  int degree=knots.length-controlPoints.length-1;
  void operator init(triple[] controlPoints, real[] knots, real[] weights) {
    this.knots=copy(knots);
    this.controlPoints=copy(controlPoints);
    this.weights=copy(weights);
    this.degree=knots.length-controlPoints.length-1;
  }
}

struct BSplineCurveData {
  real[][] controlPoints;
  real[] knots;
  int degree=knots.length-controlPoints.length-1;
  void operator init(real[][] controlPoints, real[] knots) {
    this.knots=copy(knots);
    this.controlPoints=copy(controlPoints);
    this.degree=knots.length-controlPoints.length-1;
  }
}

struct NURBSsurfaceData{
    triple[][] controlPoints;
    real[] U_knot;
    real[] V_knot;
    real[][] weights;
    int U_degree;
    int V_degree;
    void operator init(triple[][] controlPoints, real[] U_knot, real[] V_knot, real[][] weights) {
        this.U_knot=copy(U_knot);
        this.V_knot=copy(V_knot);
        this.controlPoints=copy(controlPoints);
        this.weights=copy(weights);
        this.U_degree=U_knot.length-controlPoints.length-1;
        this.V_degree=V_knot.length-controlPoints[0].length-1;
  }
}

struct BSplineSurfaceData{
    real[][][] controlPoints;
    real[] U_knot;
    real[] V_knot;
    int U_degree;
    int V_degree;
    void operator init(real[][][] controlPoints, real[] U_knot, real[] V_knot) {
        this.U_knot=copy(U_knot);
        this.V_knot=copy(V_knot);
        this.controlPoints=copy(controlPoints);
        this.U_degree=U_knot.length-controlPoints.length-1;
        this.V_degree=V_knot.length-controlPoints[0].length-1;
  }
}

NURBSCurveData conversion_4DBSpline_to_3DNurbs(BSplineCurveData BSpline_4D) {
  NURBSCurveData new_nurb;
  triple[] x=new_nurb.controlPoints;
  real[] weights=new_nurb.weights;
  real[] control_point_4D = new real[4];
  real weight=1;
  triple new_3D_control_point;

  for(int j=0; j<BSpline_4D.controlPoints.length;++j) {
    control_point_4D=copy(BSpline_4D.controlPoints[j]);
    weight=control_point_4D[3];
    if(weight!=0){
      new_3D_control_point=(control_point_4D[0],control_point_4D[1],control_point_4D[2])/weight;
    }
    x.push(new_3D_control_point);
    weights.push(weight);
  }
  new_nurb.degree=BSpline_4D.degree;
  new_nurb.controlPoints=x;
  new_nurb.knots=BSpline_4D.knots;
  new_nurb.weights=weights;

  return new_nurb;
}

BSplineCurveData conversion_3DNurbs_to_4DBSpline(NURBSCurveData NURBS_3D) {
  BSplineCurveData new_BSpline_4D;
  real[][] x=new_BSpline_4D.controlPoints;
  triple control_point_3D;
  real weight;
  for(int j=0; j < NURBS_3D.controlPoints.length; ++j) {
    control_point_3D=NURBS_3D.controlPoints[j];
    weight=NURBS_3D.weights[j];
    real[] new_control_point_4D={control_point_3D.x*weight,control_point_3D.y*weight,control_point_3D.z*weight,weight};
    x.push(copy(new_control_point_4D));
  }
  new_BSpline_4D.degree=NURBS_3D.degree;
  new_BSpline_4D.controlPoints=x;
  new_BSpline_4D.knots=copy(NURBS_3D.knots);

  return new_BSpline_4D;
}

BSplineSurfaceData conversion_3DNurbs_to_4DBSpline(NURBSsurfaceData NURBS_3D) {
    BSplineSurfaceData new_BSpline_4D;
    real[][][] x=new real[NURBS_3D.controlPoints.length][NURBS_3D.controlPoints[0].length][4];
    triple control_point_3D;
    real weight;
    for(int i=0;i<NURBS_3D.controlPoints.length;++i) {
        for(int j=0;j<NURBS_3D.controlPoints[i].length;++j){
            control_point_3D=NURBS_3D.controlPoints[i][j];
            weight=NURBS_3D.weights[i][j];
            real[] new_control_point_4D={control_point_3D.x*weight,control_point_3D.y*weight,control_point_3D.z*weight,weight};
            x[i][j]=copy(new_control_point_4D);
        }
    }
    new_BSpline_4D.controlPoints=copy(x);
    new_BSpline_4D.U_knot=copy(NURBS_3D.U_knot);
    new_BSpline_4D.V_knot=copy(NURBS_3D.V_knot);
    new_BSpline_4D.U_degree=new_BSpline_4D.U_knot.length-new_BSpline_4D.controlPoints.length-1;
    new_BSpline_4D.V_degree=new_BSpline_4D.V_knot.length-new_BSpline_4D.controlPoints[0].length-1;
    return new_BSpline_4D;
}

NURBSsurfaceData conversion_4DBSpline_to_3DNurbs(BSplineSurfaceData BSpline_4D) {
    NURBSsurfaceData nurb_3D;
    triple[][] x=new triple[BSpline_4D.controlPoints.length][BSpline_4D.controlPoints[0].length];
    real[][] weights=new real[BSpline_4D.controlPoints.length][BSpline_4D.controlPoints[0].length];
    real[] control_point_4D;
    real weight=1;
    triple new_3D_control_point;

    for(int i=0;i<BSpline_4D.controlPoints.length;++i) {
        for(int j=0;j<BSpline_4D.controlPoints[i].length;++j){
            control_point_4D=copy(BSpline_4D.controlPoints[i][j]);
            weight=control_point_4D[3];
            if(weight!=0){
                new_3D_control_point=(control_point_4D[0],control_point_4D[1],control_point_4D[2])/weight;
            }
            x[i][j]=new_3D_control_point;
            weights[i][j]=weight;
        }
    }
    nurb_3D.controlPoints=x;
    nurb_3D.U_knot=BSpline_4D.U_knot;
    nurb_3D.V_knot=BSpline_4D.V_knot;
    nurb_3D.weights=weights;
    nurb_3D.U_degree=nurb_3D.U_knot.length-nurb_3D.controlPoints.length-1;
    nurb_3D.V_degree=nurb_3D.V_knot.length-nurb_3D.controlPoints[0].length-1;
    return nurb_3D;
}

real[][] BezierMultiDegreeElevate(real[][] input_control_points, int r) {
  // elevate Bezier curve by degree r
  int d=input_control_points[0].length; // dimension of input control points
  int n=input_control_points.length;
  int p=n-1; // p is the degree of the curve
  int elevated_cp_len=n+r;
  int elevated_cp_len_half=(elevated_cp_len-1)#2;
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

BSplineCurveData DegreeElevationCurve(BSplineCurveData curve_data, int t) {
  // Degree elevate a curve t times
  int p=curve_data.degree;
  int m=curve_data.knots.length;
  int n=curve_data.controlPoints.length;
  int ph=p+t;
  int ph2=ph#2;
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
      for(int i=0;i<ph;++i) {
        Uh[kind]=ua;
        ++kind;
      }
    }

    for(int j=1;j<=ph;++j) {
      // load ctrl pts into Qw
      Qw[cind]=elevated_bpts[j];
      ++cind;
    }
    if(b < n) {
      //Set up for the next pass through loop
      for(int j=0;j<r;++j) {
        bpts[j]=Nextbpts[j];
      }
      for(int j=r;j<=p;++j) {
        bpts[j]=Pw[b-p+j];
      }
      a=b;
      ++b;
      ua=ub;
    }
    else{
      //set end knot values
      for(int i=0;i<=ph;++i) {
        Uh[kind+i]=ub;
      }
      break;
    }// end of while loop(b < m)
  }
  BSplineCurveData new_curve;
  new_curve.controlPoints=Qw;
  new_curve.knots=Uh;
  return new_curve;
}

real[][] BezDegreeReduce(real[][] bpts) {
  int p=bpts.length-1; // p=degree of bts Bezier curve
  real[][] rcpts=new real[p][] ; // reduced control points
  int r=ceilquotient((p-1),2);
  rcpts[0]=bpts[0];
  rcpts[p-1]=bpts[p];
  if(p % 2==0) {
    for(int i=1;i<=r;++i) {
      real alphai=i/p;
      rcpts[i]=(bpts[i]-alphai*rcpts[i-1])/(1-alphai);
    }
    for(int i=p-2;i>=r+1;--i) {
      real alphai1=(i+1)/p;
      rcpts[i]=(bpts[i+1]-(1-alphai1)*rcpts[i+1])/(1-alphai1);
    }
  }
  else{ //p is old
    for(int i=1;i<=r-1;++i) {
      real alphai=i/p;
      real denominator_inverse=1/(1-alphai);
      rcpts[i]=(bpts[i]-alphai*rcpts[i-1])*denominator_inverse;
    }
    for(int i=p-2;i>= r+1;--i) {
      real alphai1=(i+1)/p;
      real denominator_inverse=1/(1-alphai1);
      rcpts[i]=(bpts[i+1]-(1-alphai1)*rcpts[i+1])*denominator_inverse;
    }
    real alphar=r/p;
    real[] leftp=(bpts[r]-alphar*rcpts[r-1])/(1-alphar);
    real alphar1=(r+1)/p;
    real[] rightp=(bpts[r+1]-(1-alphar1)*rcpts[r+1])/alphar1;
    rcpts[r]=0.5*(leftp+rightp);
  }
  return rcpts;
}

BSplineCurveData DegreeReduceCurve(BSplineCurveData curve_data) {
  // reduce the BSpline curve from degree p to p-1
  int n=curve_data.controlPoints.length;
  int p=curve_data.degree;
  int m=curve_data.knots.length;// num of knots
  real[][] bpts=new real[p+1][] ; // control points of the BSpline curve segment limited according to the degree
  int ph=p-1; // reduced Degree
  int r=-1; // r is the time the knot needs to be inserted or removed
  int a=p;
  int b=p+1; // index for knot counting
  int multi=p; // stores the multiplicity of a knot
  real[][] Pw=curve_data.controlPoints; // control points of the input curve
  real[] U=curve_data.knots;

  BSplineCurveData new_curve_data; // BSpline for return
  new_curve_data.degree=p-1;
  real[][] new_curve_cp; // control points of the returning curve
  real[] new_curve_knots;
  int kind=p; //index used to assign new curve knots with values
  int cind=1; //index used to assign new curve control points with values
  new_curve_cp[0]=Pw[0];

  for(int i=0;i<=ph;++i) {
    new_curve_knots[i]=U[0];
  }
  for(int i=0;i<=p;++i) {
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

  while(b<m) {
    int oldb=b;
    while(b<m-1&&U[b]==U[b+1]) {
      ++b;
    }
    multi=b-oldb+1;
    if(b==m-1&&U[b]!=U[b-1]) {
      multi=1;
    }
    oldr=r;
    r=p-multi;

    if(oldr>0) {
      lbz=(oldr+2)#2;
    }
    else{
      lbz=1;
    }

    if(r>0) {
      real numer=U[b]-U[a];
      for(int k=p;k>multi;--k) {
        alphalist[k-multi-1]=numer/(U[a+k]-U[a]);
      }
      for(int j=1;j<=r;++j) {
        int save=r-j;
        int s=multi+j;
        for(int k=p;k>=s;--k) {
          bpts[k]=alphalist[k-s]*bpts[k]+(1.0-alphalist[k-s])*bpts[k-1];
        }
        Nextbpts[save]=bpts[p];
      }
    }
    // Degree Reduced Bezier Segment
    rbpts=BezDegreeReduce(bpts);
    // update on the output
    if(a!=p) {
      for(int i=0;i< ph;++i) {
        new_curve_knots[kind]=U[a];
        ++kind;
      }
    }
    for(int i=1;i<=ph;++i) {
      new_curve_cp[cind]=rbpts[i];
      ++cind;
    }

    //Set next Bezier segment and knots ready for looping
    if(b<n) {
      for(int i=0;i<r;++i) {
        bpts[i]=Nextbpts[i];
      }

      for(int i=r;i<=p;++i) {
        bpts[i]=Pw[b-p+i];
      }// m=n+p +1 \\ n-1
      a=b;
      ++b;
    }
    else{
      for(int i=0;i<=ph;++i) {
        new_curve_knots[kind+i]=U[b];
      }
      break;
    }
  } //end of(b < m) loop

  new_curve_data.controlPoints=new_curve_cp;
  new_curve_data.knots=new_curve_knots;
  return new_curve_data;
}

triple[] PIA(triple[] data_points,triple[] sample_points) {
  // Progressive Iterative Approximation algorithm to find the adjust vectors
  int n=data_points.length;
  triple[] adjusting_vectors=new triple[n]; // adjusting vectors for return
  for(int i=0;i<n;++i) {
    adjusting_vectors[i]=data_points[i]-sample_points[i];
  }
  return adjusting_vectors;
}

triple[] conversion_RBezier_to_NRBezier(triple[] RBcontrolPoints,triple[] adjust_controlPoints,triple[] sample_points,real tolerance) {

  triple[] local_control_points=copy(adjust_controlPoints); // local control points will be adjusted according to the adjusting vectors
  triple[] local_sample_points=copy(sample_points); // local sample points will be resampled using the newly generated local control points
  int n=local_control_points.length;
  int k=local_sample_points.length;
  bool approximation_bool=true; // change to false if the approximation error(of one vector) is greater than the allowed_error

  triple[] adjusting_vectors=PIA(RBcontrolPoints, local_sample_points);
  real ad_vector_length=0;
  for(int i=0;i<n;++i) {
    ad_vector_length=length(adjusting_vectors[i]);
    local_control_points[i]=local_control_points[i]+adjusting_vectors[i];
    if(ad_vector_length>tolerance) {
      approximation_bool=false;
    }
  }
  for(int j=0;j<k;++j) {
    local_sample_points[j]=de_casteljau_triple(j/(k-1),copy(local_control_points));
  }

  if(approximation_bool==true) {
    return local_control_points; // non-rational Bezier control points are returned
  }
  return conversion_RBezier_to_NRBezier(RBcontrolPoints,local_control_points,local_sample_points,tolerance);
}

void DecomposeSurface_U_dir(BSplineSurfaceData BSpline_4D_surface,int p){
    /*  Decompose surface into Bezier strips in u direction */
    /*  Input: BSpline_4D_surface, p */
    /*
        p is the degree elevate to in u-direction
    */
        real[][][] control_points = BSpline_4D_surface.controlPoints;
        int n = control_points.length; //number of control points in u-direction
        int m = control_points[0].length; //number of control points in v-direction

        real[] U = BSpline_4D_surface.U_knot; // U direction knots
        real[] Uh; // Used for storing elevated knots in U direction

        real[][][] Qw=new real[p+1][m][];
        real[][][] Qw_copy=copy(Qw);
        real[][][] Qw_return;

        int a=p;
        int b=p+1;
        int nb=0;
        int kind=p+1;

        for(int i=0; i <= p; ++i) {
            Uh[i]=U[a];
        }

        for(int i=0;i<=p;++i){
            for(int col_num=0;col_num<m;++col_num){
                Qw[i][col_num]=copy(control_points[i][col_num]);
            }
        }
        int U_len=U.length;
        int oldb;
        int multi;
        real[] alphas = new real[p-1]; // interpolation ratios

        while(b<U_len){
            oldb=b;
            while(b<U_len-1 && U[b]==U[b+1]){
                ++b;
            }
            multi=b-oldb+1;
            if(b==U_len-1 && U[b]!=U[b-1]){
                multi=1;
            }
            Qw_copy=copy(Qw);
            if(multi<p){
                real numerator=U[b]-U[a];
                for(int i=p;i>multi;--i){
                    real denominator=U[a+i]-U[a];
                    alphas[i-multi-1]=numerator/denominator;
                }
                for (int j=1;j<=p-multi;++j){
                    int save=p-multi-j;
                    int s=multi+j;
                    for (int k=p;k>=s;--k){
                        real alpha=alphas[k-s];
                        for(int col_num=0;col_num<m;++col_num){
                            Qw[k][col_num]=alpha*copy(Qw[k][col_num])+(1.0-alpha)*copy(Qw[k-1][col_num]);
                        }
                    }
                    Qw_copy = copy(Qw);
                    if (b<n){
                        for(int col_num=0;col_num<m;++col_num){
                            Qw[save][col_num]=copy(Qw_copy[p][col_num]);
                        }
                    }
                }
            }

            if(a != p) {
                // load the knot ua
                for(int i=0;i<p;++i) {
                    Uh[kind]=U[a];
                    ++kind;
                }
            }

            for(int i;i<Qw_copy.length;++i){
                Qw_return.push(copy(Qw_copy[i]));
            }
            if(b<n){
                for(int i=p-multi;i<=p;++i){
                    for(int col_num=0;col_num<m;++col_num){
                        Qw[i][col_num]=copy(control_points[b-p+i][col_num]);
                    }
                }
                a=b;
                ++b;
            }
            else{
                //set end knot values
                for(int i=0;i<=p;++i) {
                    Uh[kind+i]=U[b];
                }
                break;
            }
        }
        BSpline_4D_surface.U_degree=p;
        BSpline_4D_surface.U_knot=Uh;
        BSpline_4D_surface.controlPoints=Qw_return;
        //write("End of Decompose U function");
}

void DecomposeSurface_V_dir(BSplineSurfaceData BSpline_4D_surface,int q){
    /*  Decompose surface into Bezier strips in v direction */
    /*  Input: BSplineSurfaceData BSpline_4D_surface,q*/
    /*
        q is the degree elevated to in v-direction
    */
        real[][][] control_points = BSpline_4D_surface.controlPoints;
        int n = control_points.length; //number of control points in u-direction
        int m = control_points[0].length; //number of control points in v-direction

        real[] V = BSpline_4D_surface.V_knot; // U direction knots
        real[] Vh; // Used for storing elevated knots in U direction

        real[][][] Qw=new real[q+1][n][4];
        real[][][] Qw_copy=copy(Qw);
        real[][][] Qw_return;

        int a=q;
        int b=q+1;
        int nb=0;
        int kind=q+1;

        for(int i=0; i <= q; ++i) {
            Vh[i]=V[a];
        }

        for(int i=0;i<=q;++i){
            for(int row_num=0;row_num<n;++row_num){
                Qw[i][row_num]=copy(control_points[row_num][i]);
            }
        }
        int oldb;
        int multi;
        real[] alphas=new real[q-1]; // interpolation ratios

        while(b<V.length){
            oldb=b;
            while(b<V.length-1&&V[b]==V[b+1]){
                ++b;
            }
            multi=b-oldb+1;
            if(b==V.length-1&&V[b]!=V[b-1]){
                multi=1;
            }
            Qw_copy=copy(Qw);
            if(multi<q){
                real numerator=V[b]-V[a];
                for(int i=q;i>multi;--i){
                    real denominator=V[a+i]-V[a];
                    alphas[i-multi-1]=numerator/denominator;
                }
                for (int j=1;j<=q-multi;++j){
                    int save=q-multi-j;
                    int s=multi+j;
                    for (int k=q;k>=s;--k){
                        real alpha=alphas[k-s];
                        for(int row_num=0;row_num<n;++row_num){
                            Qw[k][row_num] = alpha*copy(Qw[k][row_num])+(1.0-alpha)*copy(Qw[k-1][row_num]);
                        }
                    }
                    Qw_copy=copy(Qw);
                    if (b<m){
                        for(int row_num=0;row_num<n;++row_num){
                            Qw[save][row_num]=copy(Qw_copy[q][row_num]);
                        }
                    }
                }
            }
            if(a != q) {
                // load the knot ua
                for(int i=0;i<q;++i) {
                    Vh[kind]=V[a];
                    ++kind;
                }
            }
            for(int i;i<Qw_copy.length;++i){
                Qw_return.push(Qw_copy[i]);
            }
            if(b<m){
                for(int i=q-multi;i<=q;++i){
                    for(int row_num=0;row_num<n;++row_num){
                        Qw[i][row_num]=copy(control_points[row_num][b-q+i]);
                    }
                }
                a=b;
                ++b;
            }
            else{
                //set end knot values
                for(int i=0;i<=q;++i) {
                    Vh[kind+i]=V[b];
                }
                break;
            }
        }
    // overwrite knot vector V
    BSpline_4D_surface.V_degree=q;
    BSpline_4D_surface.V_knot=Vh;
    // create four dimension transpose for Qw_return
    real[][][] Qw_tranpose = new real[Qw_return[0].length][Qw_return.length][4];
    for(int i=0;i<Qw_return.length;++i){
        for(int j=0;j<Qw_return[i].length;++j){
            Qw_tranpose[j][i]=Qw_return[i][j];
        }
    }
    // overwrite control points
    BSpline_4D_surface.controlPoints=Qw_tranpose;
    //write("End of Decompose V function");
}

void DegreeReduce_U_dir(BSplineSurfaceData BSpline_4D_surface,int output_degree){
     /*
      Input:
        BSpline_4D_surface is the surface data input for degree reduction
        output_degree is the degree of the degree_reduced surface
    */
    real[][][] BS_control_points=copy(BSpline_4D_surface.controlPoints);
    real[][][] reduced_BS_cp=new real[BS_control_points[0].length][][];
    real[] BS_U_knot=BSpline_4D_surface.U_knot;
    int p=BSpline_4D_surface.U_degree;
    for(int j=0;j<BS_control_points[0].length;++j){
      real[][] BS_col=new real[BS_control_points.length][];
      for(int i=0;i<BS_control_points.length;++i){
        BS_col[i]=BS_control_points[i][j];
      }
      BSplineCurveData col_j_BSplineCurve=BSplineCurveData(BS_col,BS_U_knot);
      for(int k=0;k<p-output_degree;++k){
          col_j_BSplineCurve=DegreeReduceCurve(col_j_BSplineCurve);
      }
      reduced_BS_cp[j]=col_j_BSplineCurve.controlPoints;
    }
    real[][][] reduced_BS_cp_tranpose=new real[reduced_BS_cp[0].length][BS_control_points[0].length][];
    for(int j=0;j<BS_control_points[0].length;++j){
      for(int i=0;i<reduced_BS_cp[j].length;++i){
        reduced_BS_cp_tranpose[i][j]=reduced_BS_cp[j][i];
      }
    }
    BSpline_4D_surface.U_degree=output_degree;
    BSpline_4D_surface.controlPoints=reduced_BS_cp_tranpose;
    //BSpline_4D_surface.controlPoints=;
}

void DegreeReduce_V_dir(BSplineSurfaceData BSpline_4D_surface,int output_degree){
      /*
      Input:
        BSpline_4D_surface is the surface data input for degree reduction
        output_degree is the degree of the degree_reduced surface
    */
    real[][][] BS_control_points=copy(BSpline_4D_surface.controlPoints);
    real[] BS_V_knot=copy(BSpline_4D_surface.V_knot);
    int q=BSpline_4D_surface.V_degree;
    for(int i=0;i<BS_control_points.length;++i){
        BSplineCurveData row_i_BSplineCurve=BSplineCurveData(BS_control_points[i],BS_V_knot);
        for(int k=0;k<q-output_degree;++k){
          row_i_BSplineCurve=DegreeReduceCurve(row_i_BSplineCurve);
        }
        BSpline_4D_surface.controlPoints[i]=row_i_BSplineCurve.controlPoints;
    }
    BSpline_4D_surface.U_degree=output_degree;
}

struct NURBScurve{
  path3[] g;

  NURBSCurveData data;

  void operator init(triple[] control_points,real[] knots,real[] weights) {
    data=NURBSCurveData(control_points,knots,weights);
    BSplineCurveData BSpline_4D=conversion_3DNurbs_to_4DBSpline(data);
    int BSpline_degree=BSpline_4D.degree;
    int output_degree=3;
    if(BSpline_degree<output_degree) {
      int t=output_degree-BSpline_degree;
      BSpline_4D=DegreeElevationCurve(BSpline_4D,t);
      int BSpline_degree=BSpline_4D.degree;
    }
    while(BSpline_degree>output_degree) {
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

    while(Bezier_last_cp_index<n) {
      triple[] current_Bezier_cps=nurb_control_points[Bezier_first_cp_index:Bezier_last_cp_index+1];
      real[] current_Bezier_weights=nurb_weights[Bezier_first_cp_index:Bezier_last_cp_index+1];
      int m=Bezier_last_cp_index-Bezier_first_cp_index+1; // number of control points in the current Bezier segment
      triple[] sample_points=new triple[m];
      bool NR_bool=true;
      for(int i=0;i<m;++i) {
        if(current_Bezier_weights[i]!=1) {
          NR_bool=false;
        }
      }
      if(NR_bool&&Bezier_last_cp_index<n) {
        g.push(current_Bezier_cps[0]..controls current_Bezier_cps[1] and current_Bezier_cps[2]..current_Bezier_cps[3]);
      } else {
        for(int i=0;i<m;++i) {
          sample_points[i]=RBezier_curve_evaluation(i/(m-1),current_Bezier_cps,current_Bezier_weights);
        }
        dot(copy(sample_points),yellow);
        real tolerance=NURBStolerance*norm(new triple[][] {current_Bezier_cps});
        triple[] NR_Bezier_control_points=conversion_RBezier_to_NRBezier(current_Bezier_cps,current_Bezier_cps,sample_points,tolerance);
        dot(copy(NR_Bezier_control_points),pink);
        g.push(NR_Bezier_control_points[0]..controls NR_Bezier_control_points[1] and NR_Bezier_control_points[2]..NR_Bezier_control_points[3]);

      }
      Bezier_first_cp_index=Bezier_last_cp_index;
      Bezier_last_cp_index=Bezier_last_cp_index+output_degree;
    }
  }

  void draw(picture pic=currentpicture,pen p=currentpen) {
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
}

struct NURBSsurface{
    surface[] g;

    NURBSsurfaceData data;

    void operator init(triple[][] control_points,real[] U_knot,real[] V_knot,real[][] weights) {
        data = NURBSsurfaceData(control_points,U_knot,V_knot,weights);
        BSplineSurfaceData BSpline_4D = conversion_3DNurbs_to_4DBSpline(data);

        int p=BSpline_4D.U_degree;
        int q=BSpline_4D.V_degree;
        int output_degree=3;

        DecomposeSurface_U_dir(BSpline_4D,p);
        DecomposeSurface_V_dir(BSpline_4D,q);
        // need to code for degree reduction
        if(p>output_degree){
          DegreeReduce_U_dir(BSpline_4D,output_degree);
        }
        if(q>output_degree){
          DegreeReduce_V_dir(BSpline_4D,output_degree);
        }
        NURBSsurfaceData RBezier_3D=conversion_4DBSpline_to_3DNurbs(BSpline_4D);
        // get the rational rows and columns of the control points rational Bezier surfaces
        int[] rational_rows;
        int[] rational_cols;
        for(int i=0;i<RBezier_3D.weights.length;++i){
            for(int j=0;j<RBezier_3D.weights[i].length;++j){
                if(RBezier_3D.weights[i][j]!=1){
                    rational_rows.push(i);
                    rational_cols.push(j);
                }
            }
        }
        rational_rows=removeDuplicates(rational_rows);
        rational_cols=removeDuplicates(rational_cols);
        // convert the rational rows to non-rational Bezier curves
        for(int i=0;i<rational_rows.length;++i){
            triple[] rational_row=copy(RBezier_3D.controlPoints[rational_rows[i]]);
            int row_len=rational_row.length;
            triple[] sample_points = new triple[row_len];
            for(int j=0;j<row_len;++j) {
              sample_points[j]=RBezier_curve_evaluation(j/(row_len-1),rational_row,RBezier_3D.weights[rational_rows[i]]);
            }
            //dot(copy(sample_points),yellow);
            //real tolerance=NURBStolerance*norm(new triple[][] {rational_row});
            real tolerance=sqrtEpsilon*norm(new triple[][] {rational_row});
            RBezier_3D.controlPoints[rational_rows[i]] = conversion_RBezier_to_NRBezier(rational_row,rational_row,sample_points,tolerance);
        }
        // convert the rational columns to non-rational Bezier curves
        int n = RBezier_3D.controlPoints.length;
        for(int h=0;h<rational_cols.length;++h){
            triple[] rational_col = new triple[n];
            real[] col_weights = new real[n];
            for(int i=0;i<n;++i){
                rational_col[i]=RBezier_3D.controlPoints[i][rational_cols[h]];
                col_weights[i]=RBezier_3D.weights[i][rational_cols[h]];
            }
            triple[] sample_points = new triple[n];
            for(int k=0;k<n;++k) {
            sample_points[k]=RBezier_curve_evaluation(k/(n-1),rational_col,col_weights);
            }
            //dot(copy(sample_points),yellow);
            real tolerance=NURBStolerance*norm(new triple[][] {rational_col});
            triple[] non_rational_col = conversion_RBezier_to_NRBezier(rational_col,rational_col,sample_points,tolerance);
            for(int i=0;i<n;++i){
                RBezier_3D.controlPoints[i][rational_cols[h]]=non_rational_col[i];
            }
        }

        triple[][][][] Bezier_surfaces =
          new triple[ceilquotient(n,p+1)]
          [ceilquotient(RBezier_3D.controlPoints[0].length,q+1)][][];

        triple[][] row_trunc;
        triple[][] matrix_trunc;
        for(int i=0;i<ceilquotient(n,p+1);++i){
            row_trunc=RBezier_3D.controlPoints[i*(p+1):(i+1)*(p+1)];
            for(int j=0;j<ceilquotient(RBezier_3D.controlPoints[i].length,q+1);++j){
                for(int k=0;k<row_trunc.length;++k){
                    matrix_trunc[k]=row_trunc[k][j*(q+1):(j+1)*(q+1)];
                }
                Bezier_surfaces[i][j]=copy(matrix_trunc);
            }
        }

        for(int i=0;i<Bezier_surfaces.length;++i){
            for(int j=0;j<Bezier_surfaces[i].length;++j){
                g.push(surface(patch(Bezier_surfaces[i][j])));
            }
        }
    }

    void draw(picture pic=currentpicture,pen p=currentpen) {
        draw(pic,g,p);
    }
}
