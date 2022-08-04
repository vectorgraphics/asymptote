import three;
import graph;
import testcurve;

real NURBStolerance=sqrtEpsilon;

struct NURBSsurfaceData{
    triple[][] controlPoints;
    real[] U_knot;
    real[] V_knot;
    real[][] weights;
    int U_degree;
    int V_degree;
    void operator init(triple[][] controlPoints, real[] U_knot, real[] V_knot, real[][] weights) {
        this.U_knot=U_knot;
        this.V_knot=V_knot;
        this.controlPoints=controlPoints;
        this.weights=weights;
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
        this.U_knot=U_knot;
        this.V_knot=V_knot;
        this.controlPoints=controlPoints;
        this.U_degree=U_knot.length-controlPoints.length-1;
        this.V_degree=V_knot.length-controlPoints[0].length-1;
  }
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

        real[][][] Qw=new real[p+1][m][4];
        real[][][] Qw_copy=copy(Qw);
        real[][][] Qw_return;

        int a=p;
        int b=p+1;
        int nb=0;
        int kind=p+1;

        for(int i=0; i < p; ++i) {
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
        U=Uh;
        control_points=Qw_return;
        write("End of Decompose U function");
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

        for(int i=0; i < q; ++i) {
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
                for(int i=0;i<q;++i) {
                    Vh[kind+i]=V[b];
                }
                break;
            }
        }
    // overwrite knot vector V
    BSpline_4D_surface.V_degree=q;
    V = Vh;
    // create four dimension transpose for Qw_return
    real[][][] Qw_tranpose = new real[Qw_return[0].length][Qw_return.length][4];
    for(int i=0;i<Qw_return.length;++i){
        for(int j=0;j<Qw_return[i].length;++j){
            Qw_tranpose[j][i]=Qw_return[i][j];
        }
    }
    // overwrite control points
    control_points = Qw_tranpose;
    write("End of Decompose V function");
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

struct NURBSsurface{
    surface[] g;

    NURBSsurfaceData data;

    void operator init(triple[][] control_points,real[] U_knot,real[] V_knot,real[][] weights) {
        data = NURBSsurfaceData(control_points,U_knot,V_knot,weights);
        BSplineSurfaceData BSpline_4D = conversion_3DNurbs_to_4DBSpline(data);

        int p=BSpline_4D.U_degree;
        int q=BSpline_4D.V_degree;
        int output_degree=3;

        if(p<output_degree){
            DecomposeSurface_U_dir(BSpline_4D,output_degree);
            p=BSpline_4D.U_degree;
        }
        if(q<output_degree){
            DecomposeSurface_V_dir(BSpline_4D,output_degree);
            q=BSpline_4D.V_degree;
        }
        // need to code for degree reduction
        /* 
        if{p>output_degree}{

        }
        if(q>output_degree){

        } */
        NURBSsurfaceData RBezier_3D=conversion_4DBSpline_to_3DNurbs(BSpline_4D);

        // get the rational rows and columns of the control points rational Bezier surfaces
        int[] rational_rows;
        int[] rational_cols;
        for(int i=0;i<RBezier_3D.weights.length;++i){
            for(int j=0;j<RBezier_3D.weights[i].length;++j){
                if(weights[i][j]!=1){
                    rational_rows.push(i);
                    rational_cols.push(j);
                }
            }
        }
        rational_rows=removeDuplicates(rational_rows);
        rational_cols=removeDuplicates(rational_cols);

        // convert the rational rows to non-rational Bezier curves
        for(int i=0;i<rational_rows.length;++i){
            triple[] rational_row = copy(RBezier_3D.controlPoints[rational_rows[i]]);
            int row_len = rational_row.length;
            triple[] sample_points = new triple[row_len];
            for(int j=0;j<row_len;++j) {
            sample_points[j]=RBezier_evaluation(j/(row_len-1),rational_row,RBezier_3D.weights[rational_rows[i]]);
            }
            //dot(copy(sample_points),yellow);
            real tolerance=NURBStolerance*norm(new triple[][] {rational_row});
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
            sample_points[k]=RBezier_evaluation(k/(n-1),rational_col,col_weights);
            }
            //dot(copy(sample_points),yellow);
            real tolerance=NURBStolerance*norm(new triple[][] {rational_col});
            triple[] non_rational_col = conversion_RBezier_to_NRBezier(rational_col,rational_col,sample_points,tolerance);
            for(int i=0;i<n;++i){
                RBezier_3D.controlPoints[i][rational_cols[h]]=non_rational_col[i];
            }
        }

        triple[][][][] Bezier_surfaces = new triple[ceil(n/(p+1))][ceil(RBezier_3D.controlPoints[0].length/(q+1))][][];

        triple[][] row_trunc;
        triple[][] matrix_trunc;
        for(int i=0;i<ceil(n/(p+1));++i){
            row_trunc=RBezier_3D.controlPoints[i*(p+1):(i+1)*(p+1)];
            for(int j=0;j<ceil(RBezier_3D.controlPoints[i].length/(q+1));++j){
                for(int k=0;k<row_trunc.length;++k){
                    matrix_trunc[k]=row_trunc[k][j*(q+1):(j+1)*(q+1)];
                }
                Bezier_surfaces[i][j]=copy(matrix_trunc);
            }
        }

        for(int i=0;i<Bezier_surfaces.length;++i){
            for(int j=0;j<Bezier_surfaces[i].length;++j){
                draw(surface(patch(Bezier_surfaces[i][j])));
                //g.push(surface(patch(Bezier_surfaces[i][j])));
            }
        }
    }

    void draw(picture pic=currentpicture,pen p=currentpen) {
        draw(pic,g,p);
    }
}

real[] uknot={0,0,0,0,0.5,1,1,1,1};
real[] vknot={0,0,0,0,0.4,0.6,1,1,1,1};

triple[][] P=
  {
   {
    (-31.2061,12.001,6.45082),
    (-31.3952,14.7353,6.53707),
    (-31.5909,21.277,6.70051),
    (-31.4284,25.4933,6.76745),
    (-31.5413,30.3485,6.68777),
    (-31.4896,32.2839,6.58385)
   },
   {
    (-28.279,12.001,7.89625),
    (-28.4187,14.7353,8.00954),
    (-28.5633,21.277,8.22422),
    (-28.4433,25.4933,8.31214),
    (-28.5266,30.3485,8.20749),
    (-28.4885,32.2839,8.07099)
   },
   {
    (-20,12.001,10.0379),
    (-20,14.7353,10.2001),
    (-20,21.277,10.5076),
    (-20,25.4933,10.6335),
    (-20,30.3485,10.4836),
    (-20,32.2839,10.2881)
   },
   {
    (-11.721,12.001,7.84024),
    (-11.5813,14.7353,7.95269),
    (-11.4367,21.277,8.16575),
    (-11.5567,25.4933,8.25302),
    (-11.4734,30.3485,8.14915),
    (-11.5115,32.2839,8.01367)
   },
   {
    (-8.79391,12.001,6.39481),
    (-8.60483,14.7353,6.48022),
    (-8.40905,21.277,6.64204),
    (-8.57158,25.4933,6.70832),
    (-8.45874,30.3485,6.62943),
    (-8.51041,32.2839,6.52653)
   }
  };
real[] uknot={0,0,0,0.5,1,1,1};
real[] vknot={0,0,0,0,0.4,0.6,1,1,1,1};

triple[][] surface2_cp={
    {
    (2.5,2.5,0),
    (1,1,1),
    (4,1,1),
    (2.5,2.5,0)
    },
    {
    (2,2,1),
    (1,1,2),
    (4,1,2),
    (3,2,1)
    },
    {
    (2,3,1),
    (1,4,2),
    (4,4,2),
    (3,3,1)
    },
    {
    (2.5,2.5,0),
    (1,4,1),
    (4,4,1),
    (2.5,2.5,0)
    }
};

real[][] weights = {
    {1,1,1,1},
    {1,2,1,1},
    {1,1,1,5},
    {1,1,1,1}
};

size(10cm);
NURBSsurface surface2=NURBSsurface(surface2_cp,uknot,vknot,weights);
for(int i=0;i<surface2.data.controlPoints.length;++i){
    dot(surface2.data.controlPoints[i],red);
}
//surface2.draw();