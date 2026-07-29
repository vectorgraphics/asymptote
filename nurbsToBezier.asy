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
  BSplineCurveData result = curve_data;
  for(int step = 0; step < t; ++step) {
    int p = result.degree;
    int m = result.knots.length;
    int n = result.controlPoints.length;
    int ph = p + 1;

    real[] U = copy(result.knots);
    real[][] Pw = copy(result.controlPoints);

    // Find distinct knot values and their multiplicities
    real[] dknots;
    int[] dmults;
    int nd = 0;
    int i = 0;
    while(i < m) {
      int j = i;
      while(j < m && U[j] == U[i]) ++j;
      dknots[nd] = U[i];
      dmults[nd] = j - i;
      ++nd;
      i = j;
    }

    // Build new knot vector: each distinct knot gets multiplicity + 1
    real[] new_knots;
    int nm = 0;
    for(int k = 0; k < nd; ++k) {
      for(int j = 0; j < dmults[k] + 1; ++j) {
        new_knots[nm] = dknots[k];
        ++nm;
      }
    }

    // Special case: pure Bezier
    real[][] Qw;
    if(nd == 2 && dmults[0] == p + 1 && dmults[1] == p + 1) {
      Qw = BezierMultiDegreeElevate(Pw, 1);
      BSplineCurveData new_curve;
      new_curve.controlPoints = Qw;
      real[] special_knots;
      for(int k = 0; k <= ph; ++k) special_knots[k] = U[0];
      for(int k = 0; k <= ph; ++k) special_knots[ph+1+k] = U[m-1];
      new_curve.knots = special_knots;
      new_curve.degree = ph;
      result = new_curve;
      continue;
    }

    // General case: standard single-step elevation
    int cind = 0;
    int cp_n = 0;

    // Handle non-clamped start: if first knot multiplicity < p+1, add blend before first CP
    if(dmults[0] < p + 1 && n > 1) {
      real alpha = dmults[0] / (real)(p + 1);
      real[] new_cp = new real[4];
      for(int v = 0; v <= 3; ++v)
        new_cp[v] = (1 - alpha) * Pw[0][v] + alpha * Pw[1][v];
      Qw[cind] = copy(new_cp);
      ++cind;
    }

    for(int d = 0; d < nd - 1; ++d) {
      int s = dmults[d];

      // Copy s-1 interior CPs (unchanged)
      for(int k = 0; k < s - 1 && cp_n + k < n; ++k) {
        Qw[cind] = copy(Pw[cp_n + k]);
        ++cind;
      }

      // Blend: alpha = s/(p+1), blend P[cp_n+s-1] and P[cp_n+s]
      real alpha = s / (real)(p + 1);
      if(cp_n + s - 1 < n && cp_n + s < n) {
        real[] new_cp = new real[4];
        for(int v = 0; v <= 3; ++v)
          new_cp[v] = (1 - alpha) * Pw[cp_n + s - 1][v] + alpha * Pw[cp_n + s][v];
        Qw[cind] = copy(new_cp);
        ++cind;
      }

      cp_n += s;
    }

    // Last distinct knot: handle non-clamped end
    if(dmults[nd-1] < p + 1 && cp_n < n - 1) {
      // Add blend before last CP
      real alpha = dmults[nd-1] / (real)(p + 1);
      real[] new_cp = new real[4];
      for(int v = 0; v <= 3; ++v)
        new_cp[v] = (1 - alpha) * Pw[n-2][v] + alpha * Pw[n-1][v];
      Qw[cind] = copy(new_cp);
      ++cind;
    }

    // Copy remaining CPs
    while(cp_n < n) {
      Qw[cind] = copy(Pw[cp_n]);
      ++cind;
      ++cp_n;
    }

    // Fallback: if analytical formula didn't produce enough CPs,
    // evaluate at uniform parameter values using de Boor and interpolate
    int expected_n = nm - ph - 1;
    if(cind < expected_n) {
      real[][] fallback_Qw;
      fallback_Qw[0] = copy(Pw[0]);
      fallback_Qw[expected_n-1] = copy(Pw[n-1]);

      for(int fi = 1; fi < expected_n - 1; ++fi) {
        real t_val = fi / (real)(expected_n - 1);
        // Find span: U[span] <= t_val < U[span+1], with span in [p, m-p-2]
        int span = p;
        while(span < m - p - 1 && U[span + p] < t_val + 1e-10) ++span;

        // Initialize local_bpts from active CPs
        real[][] local_bpts = new real[p+1][];
        for(int k = 0; k <= p; ++k) {
          int ci = span - p + k;
          if(ci >= 0 && ci < n)
            local_bpts[k] = copy(Pw[ci]);
          else
            local_bpts[k] = new real[4]; // zero-filled
        }

        // de Boor: for each level, update from high to low k
        for(int lv = 1; lv <= p; ++lv) {
          for(int k = p; k >= lv; --k) {
            int idx_hi = span - lv + 1;
            int idx_lo = span - lv + 1 - k;
            if(idx_hi >= 0 && idx_hi < m && idx_lo >= 0 && idx_lo < m) {
              real denom = U[idx_hi] - U[idx_lo];
              if(denom > 1e-15) {
                real alpha = (t_val - U[idx_lo]) / denom;
                for(int v = 0; v <= 3; ++v)
                  local_bpts[k][v] = alpha * local_bpts[k][v] + (1.0 - alpha) * local_bpts[k-1][v];
              }
            }
          }
        }

        fallback_Qw[fi] = copy(local_bpts[p]);
      }

      Qw = fallback_Qw;
      cind = expected_n;
    }

    BSplineCurveData new_curve;
    new_curve.controlPoints = Qw;
    new_curve.knots = new_knots;
    new_curve.degree = ph;
    result = new_curve;
  }
  return result;
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
  // Uses iterative PIA; only adjusts interior control points (not endpoints)
  triple[] local_cp=copy(adjust_controlPoints);
  int n=local_cp.length;
  int k=data_points.length;

  for(int iter=0;iter<50;++iter) {
    // Resample current non-rational Bezier at sample locations
    triple[] local_sp=new triple[k];
    for(int j=0;j<k;++j) {
      local_sp[j]=de_casteljau((real)j/(k-1),copy(local_cp));
    }

    // Compute adjustment vectors
    triple[] adjust_vectors=PIA(data_points, local_sp);

    // Apply adjustments only to interior control points
    bool converged=true;
    real max_adj=0;
    for(int i=1;i<n-1;++i) {
      local_cp[i]=local_cp[i]+adjust_vectors[i];
      if(length(adjust_vectors[i])>max_adj) max_adj=length(adjust_vectors[i]);
      if(length(adjust_vectors[i])>tolerance) converged=false;
    }
    // Check convergence using all sample points, not just control points
    real max_err=0;
    for(int j=0;j<k;++j) {
      real err=length(data_points[j]-local_sp[j]);
      if(err>max_err) max_err=err;
    }
    if(max_err<=tolerance || max_adj<=tolerance) return local_cp;
  }
  return local_cp; // return best approximation after max iterations
}

triple[][] conversion_RBezier_to_NRBezier(triple[][] data_points,triple[][] adjust_controlPoints,triple[][] sample_points,real tolerance){
  //convert a Rational Bezier surface to a non-rational Bezier surface
  // Uses iterative PIA; only adjusts interior control points (not boundary)
  int m=adjust_controlPoints.length;//number of rows
  int n=adjust_controlPoints[0].length;
  int u=data_points.length;
  int v=data_points[0].length;

  triple[][] local_cp=new triple[m][n];
  for(int i=0;i<m;++i)
    for(int j=0;j<n;++j)
      local_cp[i][j]=adjust_controlPoints[i][j];

  for(int iter=0;iter<50;++iter) {
    // Resample current non-rational Bezier at sample locations
    triple[][] local_sp=new triple[u][v];
    for(int i=0;i<u;++i) {
      for(int j=0;j<v;++j) {
        local_sp[i][j]=de_casteljau((real)i/(u-1),(real)j/(v-1),local_cp);
      }
    }

    // Compute adjustment vectors
    triple[][] adjust_vectors=PIA(data_points,local_sp);

    // Apply adjustments only to interior control points (not boundary)
    bool converged=true;
    real max_adj=0;
    for(int i=1;i<m-1;++i){
      for(int j=1;j<n-1;++j){
        local_cp[i][j]=local_cp[i][j]+adjust_vectors[i][j];
        if(length(adjust_vectors[i][j])>max_adj) max_adj=length(adjust_vectors[i][j]);
        if(length(adjust_vectors[i][j])>tolerance) converged=false;
      }
    }

    // Check convergence using max error over all sample points
    real max_err=0;
    for(int i=0;i<u;++i){
      for(int j=0;j<v;++j){
        real err=length(data_points[i][j]-local_sp[i][j]);
        if(err>max_err) max_err=err;
      }
    }
    if(max_err<=tolerance || max_adj<=tolerance) return local_cp;
  }
  return local_cp;
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

// Knot insertion (Boehm's algorithm): insert knot value 't' into B-spline curve
BSplineCurveData KnotInsertion(BSplineCurveData curve_data, real t) {
  int p = curve_data.degree;
  int n = curve_data.controlPoints.length;
  real[] U = copy(curve_data.knots);
  int m = U.length; // m = n + p + 1

  // Find span k: U[k] <= t < U[k+1], with k in [p, m-p-2]
  // For knot insertion at existing knot values, find the RIGHTMOST span
  // where U[k] = t (so we insert after all existing copies of that knot).
  int k = p;
  while(k < m - p - 2 && U[k + 1] <= t + 1e-12) ++k;
  // Clamp k to valid range
  if(k > m - p - 2) k = m - p - 2;

  // If t already has full multiplicity, return unchanged
  if(k + p + 1 < m && abs(U[k] - t) < 1e-12 && abs(U[k+p+1] - t) < 1e-12) {
    // Check if multiplicity is already p+1
    bool full = true;
    for(int j = k; j <= k + p && j < m; ++j) {
      if(abs(U[j] - t) > 1e-12) { full = false; break; }
    }
    if(full) return curve_data;
  }

  // Boehm's knot insertion formula
  int dim = curve_data.controlPoints[0].length;
  real[][] new_cp = new real[n + 1][];
  for(int i = k - p + 1; i <= k; ++i) {
    real denom = U[i + p] - U[i];
    real alpha;
    if(abs(denom) < 1e-15) alpha = 0;
    else alpha = (t - U[i]) / denom;
    new_cp[i] = new real[dim];
    for(int d = 0; d < dim; ++d) {
      new_cp[i][d] = (1 - alpha) * curve_data.controlPoints[i - 1][d] + alpha * curve_data.controlPoints[i][d];
    }
  }
  // Copy unchanged control points
  for(int i = 0; i <= k - p; ++i)
    new_cp[i] = copy(curve_data.controlPoints[i]);
  for(int i = k + 1; i <= n; ++i)
    new_cp[i] = copy(curve_data.controlPoints[i - 1]);

  // Build new knot vector: insert t at position k+1
  real[] new_knots;
  for(int i = 0; i <= k; ++i) new_knots[i] = U[i];
  new_knots[k + 1] = t;
  for(int i = k + 1; i < m; ++i) new_knots[i + 1] = U[i];

  BSplineCurveData result;
  result.controlPoints = new_cp;
  result.knots = new_knots;
  result.degree = p;
  return result;
}

// Fully decompose a B-spline curve into Bezier segments by inserting knots
BSplineCurveData[] DecomposeToBezier(BSplineCurveData curve_data) {
  // Insert each distinct internal knot until multiplicity = p+1
  int p = curve_data.degree;
  BSplineCurveData current = curve_data;

  real[] U = copy(current.knots);
  int m = U.length;

  // Find distinct knots and their multiplicities
  real[] dknots;
  int[] dmults;
  int nd = 0;
  int i = 0;
  while(i < m) {
    int j = i;
    while(j < m && U[j] == U[i]) ++j;
    dknots[nd] = U[i];
    dmults[nd] = j - i;
    ++nd;
    i = j;
  }

  // For each distinct knot (except first and last), insert until mult = p+1
  for(int d = 1; d < nd - 1; ++d) {
    int needed = p + 1 - dmults[d];
    for(int s = 0; s < needed; ++s) {
      current = KnotInsertion(current, dknots[d]);
    }
  }

  // Now extract Bezier segments.
        // Count segments based on distinct knot values in the valid domain [U[p], U[m-p-2]].
        U = copy(current.knots);
        m = U.length;
        int n = current.controlPoints.length;

        // Find distinct knots within valid domain
        real umin_val = U[p];
        real umax_val = U[m - p - 2];
        real[] seg_knots;  // boundaries of each segment
        int n_segboundaries = 0;
        seg_knots[n_segboundaries] = umin_val;
        ++n_segboundaries;

        int ki = p;
        while(ki < m - p - 2) {
          real kval = U[ki];
          if(kval > umin_val + 1e-12 && kval < umax_val - 1e-12) {
            seg_knots[n_segboundaries] = kval;
            ++n_segboundaries;
            while(ki < m && U[ki] == kval) ++ki;
          } else {
            ++ki;
          }
        }
        seg_knots[n_segboundaries] = umax_val;
        ++n_segboundaries;

        int num_segments = n_segboundaries - 1;
        if(num_segments < 1) num_segments = 1;  // at least one segment

        BSplineCurveData[] segments;
  for(int s = 0; s < num_segments; ++s) {
    BSplineCurveData seg;
    seg.degree = p;
    real[][] seg_cp = new real[p + 1][];

    // Find the Bezier control points for this segment.
    // For a B-spline with degree p and knot vector U:
    // The Bezier CPs for the segment spanning [seg_knots[s], seg_knots[s+1]]
    // are found by running de Boor at the right endpoint of the segment.
    // Equivalently, they're the active CPs at span k where U[k] = seg_knots[s].

    // Find span k for the left boundary of this segment
    real t_left = seg_knots[s];
    int k = p;
    while(k < m - p - 2 && U[k + 1] <= t_left - 1e-12) ++k;

    // The Bezier control points are CP[k-p], CP[k-p+1], ..., CP[k]
    // But only if the knot at seg_knots[s] has multiplicity p+1.
    // If not, we need to compute them via de Boor evaluation.
    int cp_start = k - p;
    for(int i = 0; i <= p; ++i) {
      if(cp_start + i < n)
        seg_cp[i] = copy(current.controlPoints[cp_start + i]);
      else
        seg_cp[i] = new real[current.controlPoints[0].length];
    }
    seg.controlPoints = seg_cp;

    // Segment knot vector: clamped at both ends
    real[] sk;
    for(int i = 0; i <= p; ++i) sk[i] = t_left;
    real t_right = seg_knots[s + 1];
    for(int i = 0; i <= p; ++i) sk[p + 1 + i] = t_right;
    seg.knots = sk;

    segments.push(seg);
  }

  return segments;
}

struct NURBSsurface{
    surface[] g;

    NURBSsurfaceData data;

    void operator init(triple[][] cp,real[] U_knot,real[] V_knot,real[][] weights) {
        data=NURBSsurfaceData(cp,U_knot,V_knot,weights);

        int nu = cp.length;
        int nv = cp[0].length;
        int p = U_knot.length - nu - 1;
        int q = V_knot.length - nv - 1;

        // Convert to 4D homogeneous
        real[][][] Pw = new real[nu][nv][];
        for(int i=0;i<nu;++i)
          for(int j=0;j<nv;++j){
            Pw[i][j] = new real[4];
            Pw[i][j][0] = cp[i][j].x * weights[i][j];
            Pw[i][j][1] = cp[i][j].y * weights[i][j];
            Pw[i][j][2] = cp[i][j].z * weights[i][j];
            Pw[i][j][3] = weights[i][j];
          }

        real[] Uk = copy(U_knot);
        real[] Vk = copy(V_knot);

        // Find segment boundaries: distinct knot values within valid domain [K[deg], K[mk-deg-2]]
        real[] findSegKnots(real[] K, int deg){
          int mk = K.length;
          real[] sk;
          int nsk = 0;
          sk[nsk] = K[deg]; ++nsk;
          real kmax = K[mk - deg - 1];
          int ki = deg;
          while(ki < mk - deg - 1){
            real kval = K[ki];
            if(kval > K[deg]+1e-12 && kval < kmax-1e-12){
              sk[nsk] = kval; ++nsk;
              while(ki < mk && K[ki]==kval) ++ki;
            } else ++ki;
          }
          sk[nsk] = kmax; ++nsk;
          return sk;
        }

        real[] u_segs = findSegKnots(Uk, p);
        real[] v_segs = findSegKnots(Vk, q);
        int n_u = u_segs.length - 1;
        int n_v = v_segs.length - 1;

        // Evaluate B-spline basis at parameter t using Cox-de Boor recursion
        real[] evalBasis(int deg, real t, real[] K, int ncp){
          int mk = K.length;
          real[][] N = new real[deg+1][ncp];
          for(int i=0;i<ncp;++i) N[0][i] = 0;
          // Find active zero-degree basis function (right-closed convention)
          for(int i=0;i<ncp;++i){
            if(t > K[i]-1e-15 && t < K[i+1]+1e-15){ N[0][i]=1; break; }
          }
          if(N[0][ncp-1]==0 && abs(t-K[mk-1])<1e-12) N[0][ncp-1]=1;
          // Cox-de Boor recursion
          for(int r=1;r<=deg;++r)
            for(int i=0;i<ncp;++i){
              real t1=0,t2=0;
              real d1 = K[i+r]-K[i];
              real d2 = (i+1<ncp)?K[i+r+1]-K[i+1]:0;
              if(d1>1e-15) t1=(t-K[i])/d1*N[r-1][i];
              if(i+1<ncp && d2>1e-15) t2=(K[i+r+1]-t)/d2*N[r-1][i+1];
              N[r][i]=t1+t2;
            }
          return copy(N[deg]);
        }

        // Evaluate rational B-spline surface at (u,v)
        triple evalSurf(real u, real v){
          real[] Nu = evalBasis(p, u, Uk, nu);
          real[] Nv = evalBasis(q, v, Vk, nv);
          triple num=(0,0,0); real den=0;
          for(int i=0;i<nu;++i)
            for(int j=0;j<nv;++j){
              real b=Nu[i]*Nv[j];
              num += b*(Pw[i][j][0],Pw[i][j][1],Pw[i][j][2]);
              den += b*Pw[i][j][3];
            }
          return (abs(den)>1e-15)?num/den:(0,0,0);
        }

        // Bernstein polynomial values of degree d at t in [0,1]
        real[] bernVals(int d, real t){
          real[] B = new real[d+1];
          for(int i=0;i<=d;++i){
            real val=1;
            for(int k=0;k<i;++k) val*=t;
            for(int k=0;k<d-i;++k) val*=(1-t);
            real binom=1;
            for(int k=1;k<=i;++k) binom=binom*(d-k+1)/k;
            B[i]=binom*val;
          }
          return B;
        }

        // Invert matrix via Gauss-Jordan elimination
        real[][] invMat(real[][] M, int n){
          real[][] A = new real[n][2*n];
          for(int i=0;i<n;++i)
            for(int j=0;j<2*n;++j) A[i][j]=(j<n)?M[i][j]:((i==j-n)?1:0);
          for(int col=0;col<n;++col){
            int piv=col;
            for(int row=col+1;row<n;++row)
              if(abs(A[row][col])>abs(A[piv][col])) piv=row;
            real[] tmp=copy(A[col]); A[col]=copy(A[piv]); A[piv]=copy(tmp);
            real pv=A[col][col];
            for(int j=0;j<2*n;++j) A[col][j]/=pv;
            for(int row=0;row<n;++row){
              if(row==col) continue;
              real f=A[row][col];
              for(int j=0;j<2*n;++j) A[row][j]-=f*A[col][j];
            }
          }
          real[][] R = new real[n][n];
          for(int i=0;i<n;++i) for(int j=0;j<n;++j) R[i][j]=A[i][j+n];
          return R;
        }

        // Extract Bezier patches via evaluation + matrix inversion.
        // For each B-spline segment, evaluate at interior sample points and solve
        // the Bernstein system to get cubic Bezier control points.
        int targetDeg = 3;
        int nS = targetDeg + 1;

        // Interior sample points (avoid boundaries where basis may degenerate)
        real[] sT;
        for(int i=0;i<nS;++i) sT[i] = (i+0.5)/nS;

        // Build and invert Bernstein matrix
        real[][] BM = new real[nS][nS];
        for(int i=0;i<nS;++i){
          real[] B = bernVals(targetDeg, sT[i]);
          for(int j=0;j<nS;++j) BM[i][j]=B[j];
        }
        real[][] BM_inv = invMat(BM, nS);

        // Process each U/V segment pair, subdividing into smaller patches
        int nsub = 4;
        real[] subT;
        for(int i=0;i<=nsub;++i) subT[i] = (real)i/nsub;

        for(int si=0;si<n_u;++si){
          real uL=u_segs[si], uR=u_segs[si+1];
          for(int sj=0;sj<n_v;++sj){
            real vL=v_segs[sj], vR=v_segs[sj+1];

            for(int su=0;su<nsub;++su){
              real u0 = uL + subT[su]*(uR-uL);
              real u1 = uL + subT[su+1]*(uR-uL);
              for(int sv=0;sv<nsub;++sv){
                real v0 = vL + subT[sv]*(vR-vL);
                real v1 = vL + subT[sv+1]*(vR-vL);

                // Evaluate surface on sample grid for this sub-patch
                triple[][] S = new triple[nS][nS];
                for(int i=0;i<nS;++i)
                  for(int j=0;j<nS;++j){
                    real u=u0+sT[i]*(u1-u0);
                    real v=v0+sT[j]*(v1-v0);
                    S[i][j]=evalSurf(u,v);
                  }

                // Extract Bezier CPs: B = BM_inv * S * (BM_inv)^T
                triple[][] Tm = new triple[nS][nS];
                for(int j=0;j<nS;++j)
                  for(int i=0;i<nS;++i){
                    Tm[i][j]=(0,0,0);
                    for(int k=0;k<nS;++k) Tm[i][j]+=BM_inv[i][k]*S[k][j];
                  }

                triple[][] Bcp = new triple[nS][nS];
                for(int i=0;i<nS;++i)
                  for(int j=0;j<nS;++j){
                    Bcp[i][j]=(0,0,0);
                    for(int k=0;k<nS;++k) Bcp[i][j]+=BM_inv[j][k]*Tm[i][k];
                  }

                // Check if surface is rational
                bool isRational = false;
                {
                  real uC=u0+0.5*(u1-u0), vC=v0+0.5*(v1-v0);
                  triple nPt=evalSurf(uC,vC);
                  real[] Bu=bernVals(targetDeg,0.5), Bv=bernVals(targetDeg,0.5);
                  triple bPt=(0,0,0);
                  for(int i=0;i<nS;++i)
                    for(int j=0;j<nS;++j) bPt+=Bu[i]*Bv[j]*Bcp[i][j];
                  if(length(nPt-bPt)>1e-8) isRational=true;
                }

                triple[][] nr_cp;
                if(isRational){
                  int sm=6, nsu=sm*targetDeg+1, nsv=sm*targetDeg+1;
                  triple[][] dPts=new triple[nsu][nsv];
                  triple[][] sPts=new triple[nsu][nsv];
                  for(int u=0;u<nsu;++u)
                    for(int v=0;v<nsv;++v){
                      real tu=(real)u/(sm*targetDeg), tv=(real)v/(sm*targetDeg);
                      dPts[u][v]=evalSurf(u0+tu*(u1-u0),v0+tv*(v1-v0));
                      sPts[u][v]=de_casteljau(tu,tv,Bcp);
                    }
                  nr_cp=conversion_RBezier_to_NRBezier(dPts,Bcp,sPts,NURBStolerance);
                } else {
                  nr_cp=copy(Bcp);
                }

                g.push(surface(patch(nr_cp)));
              }
            }
          }
        }
    }

    void draw(picture pic=currentpicture,pen p=currentpen) {
      draw(pic,g,p);
    }
}
