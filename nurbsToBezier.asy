import three;
import graph;

/*
Reference:{
  @article{BSpline elevation
   title={{The NURBS Book}},
   author={Les P., Wayne T.},
   year={1995},
   pages={206--225},
   publisher={Springer}
   }
  @article{PIA algorithm
   title={{Conversion of Rational Bezier Curves into Non-Rational Bezier Curves using Progressive Iterative Approximation}},
   author={Anchisa C., Natasha D.},
   year={2013}
   publisher={IEEE}
  }
}
*/
real NURBStolerance=sqrtEpsilon;

int ceilquotient(int a, int b){
  return (a+b-1)#b;
}

real[][][] transpose(real[][][] matrix){
  real[][][] local_matrix=copy(matrix);
  int m=matrix.length;//number of rows
  int n=matrix[0].length;//number of columns
  real[][][] return_matrix=new real[n][m][];
  for(int i=0;i<m;++i){
    for(int j=0;j<n;++j){
      return_matrix[j][i]=local_matrix[i][j];
    }
  }
  return return_matrix;
}

triple de_casteljau(real t,triple[] coefs){
  //de_Casteljau algorithm for triple
  int n=coefs.length;
  triple[] local_coefs=copy(coefs);
  for(int i=1;i<=n-1;++i) {
    for(int j=0;j<=n-i-1;++j) {
      local_coefs[j]=(1-t)*local_coefs[j]+t*local_coefs[j+1];
    }
  }
  return local_coefs[0]; // the point on the curve evaluated with given t
}

triple de_casteljau(real u,real v, triple[][] coefs){
  int m=coefs.length;//number of rows
  int n=coefs[0].length;//number of columns
  triple[] Bezier_cp=new triple[m];
  for(int i=0;i<m;++i){
    Bezier_cp[i]=de_casteljau(v,coefs[i]);
  }

  triple point_return=de_casteljau(u,Bezier_cp);
  return point_return;
}

real de_casteljau(real t,real[] coefs){
  //de_Casteljau algorithm for real
  int n=coefs.length;
  real[] local_coefs=copy(coefs);
  for(int i=1;i<n;++i) {
    for(int j=0;j<n-i;++j) {
      local_coefs[j]=local_coefs[j]*(1-t)+local_coefs[j+1]*t;
    }
  }
  return local_coefs[0]; // the point on the curve evaluated with given t
}

triple RBezier_evaluation(real t, triple[] cp,real[] weights){
  // evaluate one point on the Rational Bezier curve
  int n=cp.length;
  triple[] weighted_cp=new triple[n];
  triple point_on_curve;
  for(int i=0;i<n;++i) {
    weighted_cp[i]=cp[i]*weights[i];
  }
  triple numerator=de_casteljau(t,weighted_cp);
  real denominator=de_casteljau(t,weights);
  if(denominator!=0){
    point_on_curve=numerator/denominator;
  }
  else{
    point_on_curve=(0,0,0);
  }
  return point_on_curve;
}

triple RBezier_evaluation(real u, real v, triple[][] cp, real[][] weights){
  // evaluate one point on the Rational Bezier surface
  int m=cp.length;//number of rows
  int n=cp[0].length;//number of columns
  //triple[][] weighted_cp=new triple[m][n];
  triple[] Bezier_curve_cp=new triple[m];
  real[] weight_berstein=new real[m];
  triple point_on_surface;
  
  for(int i=0;i<m;++i){
    Bezier_curve_cp[i]=RBezier_evaluation(v,cp[i],weights[i]);
    weight_berstein[i]=de_casteljau(v,weights[i]);
  }

  point_on_surface=RBezier_evaluation(u,Bezier_curve_cp,weight_berstein);
  return point_on_surface;
}

struct NURBSCurveData{
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

struct BSplineCurveData{
  real[][] controlPoints;// the fourth entry of a BSpline curve control points is weight
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

NURBSCurveData conversion_4DBSpline_to_3DNurbs(BSplineCurveData BSpline_4D){
  //convert a 4D BSpline curve to a 3D Nurb curve by adding the influence of weight(controlPoints[3])
  NURBSCurveData new_nurb;
  triple[] x=new triple[];
  real[] weights=new real[];
  real[] control_point = new real[4];
  real weight=1;
  triple new_3D_control_point;

  for(int j=0; j<BSpline_4D.controlPoints.length;++j) {
    control_point=copy(BSpline_4D.controlPoints[j]);
    weight=control_point[3];
    if(weight!=0){
      new_3D_control_point=(control_point[0],control_point[1],control_point[2])/weight;
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

BSplineCurveData conversion_3DNurbs_to_4DBSpline(NURBSCurveData NURBS_3D){
  //convert a 3D Nurb curve to a 4D BSpline curve by considering weight as the fourth entry in BSpline(controlPoints[3])
  BSplineCurveData new_BSpline_4D;
  real[][] x=new_BSpline_4D.controlPoints;
  triple control_point_3D;
  real weight;
  for(int j=0; j < NURBS_3D.controlPoints.length; ++j) {
    control_point_3D=NURBS_3D.controlPoints[j];
    weight=NURBS_3D.weights[j];
    real[] new_control_point={control_point_3D.x*weight,control_point_3D.y*weight,control_point_3D.z*weight,weight};
    x.push(copy(new_control_point));
  }
  new_BSpline_4D.degree=NURBS_3D.degree;
  new_BSpline_4D.controlPoints=x;
  new_BSpline_4D.knots=copy(NURBS_3D.knots);

  return new_BSpline_4D;
}

BSplineSurfaceData conversion_3DNurbs_to_4DBSpline(NURBSsurfaceData NURBS_3D){
    //convert a 3D Nurb surface to a 4D BSpline surface by adding the influence of weight(controlPoints[3])
    BSplineSurfaceData new_BSpline_4D;
    real[][][] x=new real[NURBS_3D.controlPoints.length][NURBS_3D.controlPoints[0].length][4];
    triple control_point_3D;
    real weight;
    for(int i=0;i<NURBS_3D.controlPoints.length;++i) {
        for(int j=0;j<NURBS_3D.controlPoints[i].length;++j){
            control_point_3D=NURBS_3D.controlPoints[i][j];
            weight=NURBS_3D.weights[i][j];
            real[] new_control_point={control_point_3D.x*weight,control_point_3D.y*weight,control_point_3D.z*weight,weight};
            x[i][j]=copy(new_control_point);
        }
    }
    new_BSpline_4D.controlPoints=copy(x);
    new_BSpline_4D.U_knot=copy(NURBS_3D.U_knot);
    new_BSpline_4D.V_knot=copy(NURBS_3D.V_knot);
    new_BSpline_4D.U_degree=NURBS_3D.U_degree;
    new_BSpline_4D.V_degree=NURBS_3D.V_degree;
    return new_BSpline_4D;
}

NURBSsurfaceData conversion_4DBSpline_to_3DNurbs(BSplineSurfaceData BSpline_4D){
    //convert a 4D BSpline surface to a 3D Nurb surface by adding the influence of weight(controlPoints[3])
    NURBSsurfaceData nurb_3D;
    triple[][] x=new triple[BSpline_4D.controlPoints.length][BSpline_4D.controlPoints[0].length];
    real[][] weights=new real[BSpline_4D.controlPoints.length][BSpline_4D.controlPoints[0].length];
    real[] control_point=new real[];
    real weight=1;
    triple new_3D_control_point;

    for(int i=0;i<BSpline_4D.controlPoints.length;++i) {
        for(int j=0;j<BSpline_4D.controlPoints[i].length;++j){
            control_point=copy(BSpline_4D.controlPoints[i][j]);
            weight=control_point[3];
            if(weight!=0){
                new_3D_control_point=(control_point[0],control_point[1],control_point[2])/weight;
            }
            x[i][j]=new_3D_control_point;
            weights[i][j]=weight;
        }
    }
    nurb_3D.controlPoints=x;
    nurb_3D.U_knot=BSpline_4D.U_knot;
    nurb_3D.V_knot=BSpline_4D.V_knot;
    nurb_3D.weights=weights;
    nurb_3D.U_degree=BSpline_4D.U_degree;
    nurb_3D.V_degree=BSpline_4D.V_degree;
    return nurb_3D;
}

real[][] BezierMultiDegreeElevate(real[][] input_cp, int r){
  // elevate Bezier curve by degree r, returns control points
  int d=input_cp[0].length; // dimension of input control points
  int n=input_cp.length;
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
  elevated_cp[0]=input_cp[0];
  elevated_cp[elevated_cp_len-1]=input_cp[p];
  for(int i=1; i < elevated_cp_len-1; ++i) {
    int mpi=min(p,i);
    for(int j=max(0,i-r); j <= mpi; ++j) {
      elevated_cp[i]=elevated_cp[i]+bezcoefs[i][j]*input_cp[j];
    }
  }

  return elevated_cp;
}

BSplineCurveData DegreeElevationCurve(BSplineCurveData curve_data, int t){
  // Degree elevate a BSpline curve t times, returns BSplineCurveData data structure
  int p=curve_data.degree;
  int m=curve_data.knots.length;
  int n=curve_data.controlPoints.length;
  int ph=p+t;
  int ph2=floor(ph/2);
  real[] U=copy(curve_data.knots);
  real[][] Pw=copy(curve_data.controlPoints);

  int kind=ph+1; // knot index
  int r=-1;

  int a=0;
  for(int i=0;i<curve_data.knots.length;++i){
    if(curve_data.knots[i]==curve_data.knots[i+1])
      ++a;
    else
      break;
  }
  int b=a+1;
  
  int cind=1; // control points index
  int mpi=0;
  int lbz=0;
  int rbz=0;

  real ua=U[a];
  real[][] Qw; // new control points array
  real[] Uh; // new knots array
  real[][] bpts=new real[p+1][] ; // pth degree Bezier control points of current segment
  real[][] bezalfs=new real[p+t+1][p+1];//coefficients for degree elevating Bezier segments
  real[][] ebpts=new real[p+t+1][] ; // p+t th degree Bezier control points of current segment
  real[][] Nextbpts=new real[p-1][] ; // leftmost control points of next Bezier Segment
  real[] alphas=new real[p-1]; // interpolation ratios
  Qw[0]=Pw[0];
  //initialization of bezalfs
  bezalfs[0][0]=1.0;
  bezalfs[ph][p]=1.0;
  for(int i=1;i<=ph2;++i){
    real inv=1/choose(ph,i);
    mpi=min(p,i);
    for(int j=max(0,i-t);j<=mpi;++j){
      bezalfs[i][j]=inv*choose(p,j)*choose(t,i-j);
    }
  }
  for(int i=ph2+1;i<=ph-1;++i){
    mpi=min(p,i);
    for(int j=max(0,i-t);j<=mpi;++j)
      bezalfs[i][j]=bezalfs[ph-i][p-j];
  }

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
    ub=U[b];
    oldr=r;
    r=p-multi;
    if(oldr>0)
      lbz=floor((oldr+2)/2);
    else
      lbz=1;
    if(r>0)
      rbz=floor(ph-(r+1)/2);
    else
      rbz=ph;
    if(r>0) {
      // Insert knot to get Bezier segment
      real numer=ub-ua;
      for(int i=p ; i > multi; --i) {
        alphas[i-multi-1]=numer/(U[a+i]-ua);
      }
      for(int j=1; j <= r; ++j) {
        int save=r-j;
        int s=multi+j;
        for(int k=p; k >= s; --k) {
          for(int v=0;v<=3;++v)
            bpts[k][v]=alphas[k-s]*bpts[k][v]+(1.0-alphas[k-s])*bpts[k-1][v];
        }
        Nextbpts[save]=bpts[p];
      }
    }
    // End of knot inserton
    //Bezier Degree Elevation
    for(int i=lbz;i<=ph;++i){
      for(int j=0;j<=3;++j)
        ebpts[i][j]=0;
      mpi=min(p,i);
      for(int j=max(0,i-t);j<=mpi;++j){
        for(int k=0;k<=3;++k)
          ebpts[i][k]=ebpts[i][k]+bezalfs[i][j]*bpts[j][k];
      }
    }//end of degree elevating Bezier

    if(a!=p) {
      // load the knot ua
      for(int i=0;i<ph-oldr;++i) {
        Uh[kind]=ub;
        ++kind;
      }
    }

    for(int j=lbz;j<=rbz;++j) {
      // load ctrl pts into Qw
      Qw[cind]=ebpts[j];
      ++cind;
    }
    if(b<n) {
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
      if(ub!=1){
        ++b;
        kind+=ph;
      }
      else
        break;
    }// end of while loop(b < m)
  }
  BSplineCurveData new_curve;
  new_curve.controlPoints=Qw;
  new_curve.knots=Uh;
  return new_curve;
}

real[][] BezDegreeReduce(real[][] bpts){
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
    rcpts[r]=(leftp+rightp)/2;
  }
  return rcpts;
}

BSplineCurveData DegreeReduceCurve(BSplineCurveData curve_data){
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
      lbz=floor((oldr+2)/2);
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

triple[] PIA(triple[] data_points,triple[] sample_points){
  // Progressive Iterative Approximation algorithm to find the adjust vectors(curve)
  int n=data_points.length;
  triple[] adjust_vectors=new triple[n]; // adjusting vectors for return
  for(int i=0;i<n;++i) {
    adjust_vectors[i]=data_points[i]-sample_points[i];
  }
  return adjust_vectors;
}

triple[][] PIA(triple[][] data_points,triple[][] sample_points){
  // Progressive Iterative Approximation algorithm to find the adjust vectors(surface)
  int m=data_points.length;//number of rows
  int n=data_points[0].length;//number of columns
  triple[][] adjust_vectors=new triple[m][n]; // adjust vectors for return
  for(int i=0;i<m;++i)
    for(int j=0;j<n;++j)
      adjust_vectors[i][j]=data_points[i][j]-sample_points[i][j];
  return adjust_vectors;
}

triple[] conversion_RBezier_to_NRBezier(triple[] data_points,triple[] adjust_controlPoints,triple[] sample_points,real tolerance) {
  // conversion from Rational Bezier curve to Non-Rational Bezier curve
  triple[] local_cp=copy(adjust_controlPoints); // local control points will be adjusted according to the adjusting vectors
  triple[] local_sample_points=copy(sample_points); // local sample points will be resampled using the newly generated local control points
  int n=local_cp.length;
  int k=local_sample_points.length;

  bool approximation_bool=true; // change to false if the approximation error(of one vector) is greater than the allowed_error
  triple[] adjust_vectors=PIA(data_points, local_sample_points);

  for(int i=0;i<n;++i) {
    local_cp[i]=local_cp[i]+adjust_vectors[i];
  }
  for(int j=0;j<k;++j) {
    local_sample_points[j]=de_casteljau(j/(k-1),copy(local_cp));
  }
  for(int i=0;i<n;++i){
    if(length(adjust_vectors[i])>tolerance){
      approximation_bool=false;
    }
  }
  if(approximation_bool)
    return local_cp; // non-rational Bezier control points are returned
  return conversion_RBezier_to_NRBezier(data_points,local_cp,local_sample_points,tolerance);
}

triple[][] conversion_RBezier_to_NRBezier(triple[][] data_points,triple[][] adjust_controlPoints,triple[][] sample_points,real tolerance){
  //convert a Rational Bezier surface to a non-rational Bezier surface
  triple[][] local_sp=copy(sample_points);
  int m=adjust_controlPoints.length;//number of rows
  int n=adjust_controlPoints[0].length;
  int u=local_sp.length;
  int v=local_sp[0].length;
  triple[][] local_cp=new triple[m][n];

  bool approximation_bool=true; //change to false if the approximation error(of one vector) is greater than the allowed_error
  triple[][] adjust_vectors=PIA(data_points,local_sp);
  for(int i=0;i<m;++i){
    for(int j=0;j<n;++j){
      local_cp[i][j]=adjust_controlPoints[i][j]+adjust_vectors[i][j];
    }
  }
  
  for(int i=0;i<u;++i){
    for(int j=0;j<v;++j){
      local_sp[i][j]=de_casteljau(i/(u-1),j/(v-1),local_cp);
    }
  }
  for(int i=0;i<u;++i){
    for(int j=0;j<v;++j){
      if(length(adjust_vectors[i][j])>tolerance){
        approximation_bool=false;
        break;
      }
    }
  }
  if(approximation_bool)
    return local_cp;
  
  return conversion_RBezier_to_NRBezier(data_points,local_cp,local_sp,tolerance);
}

void DecomposeSurface_V_dir(BSplineSurfaceData BSpline_4D_surface,int t){
    /*  Decompose surface into Bezier strips in v direction */
    /*  Input: BSpline_4D_surface, t */
    /*
        t is the degree we elevate in v-direction
    */
        real[][][] cp=copy(BSpline_4D_surface.controlPoints);
        real[][][] return_cp=new real[][][];
        int q=BSpline_4D_surface.V_degree;
        int m=cp.length; //number of control points in v-direction
        int n=cp[0].length; //number of control points in u-direction
        int qh=q+t;

        real[] V_knot=copy(BSpline_4D_surface.V_knot);
        BSplineCurveData curve=new BSplineCurveData;
        for(int j=0;j<m;++j){
          curve.controlPoints=cp[j];
          curve.knots=V_knot;
          curve.degree=q;
          curve=DegreeElevationCurve(curve,t);
          return_cp[j]=copy(curve.controlPoints);
        }

        BSpline_4D_surface.V_degree=qh;
        BSpline_4D_surface.V_knot=curve.knots;
        BSpline_4D_surface.controlPoints=copy(return_cp);
        //write("end of V dir decomp");
}

void DecomposeSurface_U_dir(BSplineSurfaceData BSpline_4D_surface,int t){
    /*  Decompose surface into Bezier strips in u direction */
    /*  Input: BSplineSurfaceData BSpline_4D_surface,q*/
    /*
        t is the degree elevate in u-direction
    */
        real[][][] cp=transpose(BSpline_4D_surface.controlPoints);
        int m=cp.length; //number of control points in u-direction
        int n=cp[0].length; //number of control points in v-direction
        real[][][] return_cp=new real[m][][];
        int p=BSpline_4D_surface.U_degree;
        int ph=p+t;

        real[] U_knot=copy(BSpline_4D_surface.U_knot);
        BSplineCurveData curve=new BSplineCurveData;
        for(int j=0;j<m;++j){
          curve.controlPoints=cp[j];
          curve.knots=U_knot;
          curve.degree=p;
          curve=DegreeElevationCurve(curve,t);
          return_cp[j]=copy(curve.controlPoints);
        }
        BSpline_4D_surface.U_degree=ph;
        BSpline_4D_surface.U_knot=curve.knots;
        BSpline_4D_surface.controlPoints=transpose(return_cp);
        //write("end of U dir decomp");
}

void DegreeReduce_V_dir(BSplineSurfaceData BSpline_4D_surface,int output_degree){
      /*
      Input:
        BSpline_4D_surface is the surface data input for degree reduction
        output_degree is the degree of the degree_reduced surface 
    */
    real[][][] BS_cp=copy(BSpline_4D_surface.controlPoints);
    real[] BS_V_knot=copy(BSpline_4D_surface.V_knot);
    int q=BSpline_4D_surface.V_degree;
    for(int i=0;i<BS_cp.length;++i){
        BSplineCurveData row_i_BSplineCurve=BSplineCurveData(BS_cp[i],BS_V_knot);
        for(int k=0;k<q-output_degree;++k){
          row_i_BSplineCurve=DegreeReduceCurve(row_i_BSplineCurve);
        }
        BSpline_4D_surface.controlPoints[i]=row_i_BSplineCurve.controlPoints;
    }
    BSpline_4D_surface.V_degree=output_degree;
}

void DegreeReduce_U_dir(BSplineSurfaceData BSpline_4D_surface,int output_degree){
     /*
      Input:
        BSpline_4D_surface is the surface data input for degree reduction
        output_degree is the degree of the degree_reduced surface
    */
    real[][][] BS_cp=copy(BSpline_4D_surface.controlPoints);
    real[][][] reduced_BS_cp=new real[BS_cp[0].length][][];
    real[] BS_U_knot=BSpline_4D_surface.U_knot;
    int p=BSpline_4D_surface.U_degree;
    for(int j=0;j<BS_cp[0].length;++j){
      real[][] BS_col=new real[BS_cp.length][];
      for(int i=0;i<BS_cp.length;++i){
        BS_col[i]=BS_cp[i][j];
      }
      BSplineCurveData col_j_BSplineCurve=BSplineCurveData(BS_col,BS_U_knot);
      for(int k=0;k<p-output_degree;++k){
          col_j_BSplineCurve=DegreeReduceCurve(col_j_BSplineCurve);
      }
      reduced_BS_cp[j]=col_j_BSplineCurve.controlPoints;
    }
    real[][][] reduced_BS_cp_tranpose=new real[reduced_BS_cp[0].length][BS_cp[0].length][];
    for(int j=0;j<BS_cp[0].length;++j){
      for(int i=0;i<reduced_BS_cp[j].length;++i){
        reduced_BS_cp_tranpose[i][j]=reduced_BS_cp[j][i];
      }
    }
    BSpline_4D_surface.U_degree=output_degree;
    BSpline_4D_surface.controlPoints=reduced_BS_cp_tranpose;
    //BSpline_4D_surface.controlPoints=;
}

struct NURBScurve{
  path3[] g;

  NURBSCurveData data;

  void operator init(triple[] cp,real[] knots,real[] weights) {
    data=NURBSCurveData(cp,knots,weights);
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
    triple[] nurb_cp=nurb_3D.controlPoints;
    real[] nurb_weights=nurb_3D.weights;
    int n=nurb_cp.length;

    while(Bezier_last_cp_index<n) {
      triple[] current_Bezier_cps=nurb_cp[Bezier_first_cp_index:Bezier_last_cp_index+1];
      real[] current_Bezier_weights=nurb_weights[Bezier_first_cp_index:Bezier_last_cp_index+1];
      int m=Bezier_last_cp_index-Bezier_first_cp_index+1; // number of control points in the current Bezier segment
      triple[] data_points=new triple[m];
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
          data_points[i]=RBezier_evaluation(i/(m-1),current_Bezier_cps,current_Bezier_weights);
          sample_points[i]=de_casteljau(i/(m-1),current_Bezier_cps);
        }
        real tolerance=NURBStolerance*norm(new triple[][] {current_Bezier_cps});
        triple[] NR_Bezier_cp=conversion_RBezier_to_NRBezier(data_points,sample_points,sample_points,tolerance);
        g.push(NR_Bezier_cp[0]..controls NR_Bezier_cp[1] and NR_Bezier_cp[2]..NR_Bezier_cp[3]);

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

    void operator init(triple[][] cp,real[] U_knot,real[] V_knot,real[][] weights) {
        data=NURBSsurfaceData(cp,U_knot,V_knot,weights);
        BSplineSurfaceData BSpline_4D=conversion_3DNurbs_to_4DBSpline(data);

        int p=BSpline_4D.U_degree;
        int q=BSpline_4D.V_degree;
        int output_degree=3;
        if(p<output_degree){
          DecomposeSurface_U_dir(BSpline_4D,output_degree-p);
        }
        else{
          DecomposeSurface_U_dir(BSpline_4D,p-output_degree);
        }
        
        if(q<output_degree){
          DecomposeSurface_V_dir(BSpline_4D,output_degree-q);
        }
        else{
          DecomposeSurface_V_dir(BSpline_4D,q-output_degree);
        }
        if(p>output_degree){
          DegreeReduce_U_dir(BSpline_4D,output_degree);
        }
        if(q>output_degree){
          DegreeReduce_V_dir(BSpline_4D,output_degree);
        }
        p=BSpline_4D.U_degree;
        q=BSpline_4D.V_degree;
        NURBSsurfaceData RBezier_3D=conversion_4DBSpline_to_3DNurbs(BSpline_4D);
        int m=RBezier_3D.controlPoints.length;
        int n=RBezier_3D.controlPoints[0].length;
        int sm=1;//sample multiplicity
        triple[][] sample_points=new triple[sm*p+1][sm*q+1];
        triple[][] data_points=new triple[sm*p+1][sm*q+1];
        triple[][] extend_cp=new triple[ceilquotient(m,p+1)*(sm*p+1)][ceilquotient(n,q+1)*(sm*q+1)];
        triple[][][][] Bezier_surfaces =
          new triple[ceilquotient(RBezier_3D.controlPoints.length,p+1)]
          [ceilquotient(RBezier_3D.controlPoints[0].length,q+1)][p+1][q+1];

        triple[][] matrix_trunc=new triple[p+1][q+1];
        real[][] weight_trunc=new real[p+1][q+1];

        for(int i=0;i<ceilquotient(m,p+1);++i){
          for(int j=0;j<ceilquotient(RBezier_3D.controlPoints[i].length,q+1);++j){
            for(int u=0;u<=p;++u){
              for(int v=0;v<=q;++v){
                matrix_trunc[u][v]=RBezier_3D.controlPoints[i*p+u][j*q+v];
                weight_trunc[u][v]=RBezier_3D.weights[i*p+u][j*q+v];
              }
            }
            for(int u=0;u<=sm*p;++u){
              for(int v=0;v<=sm*q;++v){
                data_points[u][v]=RBezier_evaluation(u/(sm*p),v/(sm*q),matrix_trunc,weight_trunc);
                sample_points[u][v]=de_casteljau(u/(sm*p),v/(sm*q),matrix_trunc);
              }
            }
            extend_cp=conversion_RBezier_to_NRBezier(data_points,sample_points,sample_points,0.01);
            for(int u=0;u<=p;++u){
              for(int v=0;v<=q;++v){
                RBezier_3D.controlPoints[i*p+u][j*q+v]=extend_cp[u*sm][v*sm];
              }
            }
            for(int u=0;u<=p;++u){
              for(int v=0;v<=q;++v){
                matrix_trunc[u][v]=RBezier_3D.controlPoints[i*p+u][j*q+v];
                weight_trunc[u][v]=RBezier_3D.weights[i*p+u][j*q+v];
              }
            }
            for(int u=0;u<=sm*p;++u){
              for(int v=0;v<=sm*q;++v){
                sample_points[u][v]=RBezier_evaluation(u/(sm*p),v/(sm*q),matrix_trunc,weight_trunc);
              }
            }
          }
        }
        triple[][] row_trunc;
        for(int i=0;i<ceilquotient(RBezier_3D.controlPoints.length,p+1);++i){
          row_trunc=RBezier_3D.controlPoints[i*p:(i+1)*p+1];
          for(int j=0;j<ceilquotient(RBezier_3D.controlPoints[i].length,q+1);++j){
            for(int k=0;k<row_trunc.length;++k){
                matrix_trunc[k]=row_trunc[k][j*q:(j+1)*q+1];
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