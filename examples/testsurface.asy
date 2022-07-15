import three;
import graph;

triple[][][] DecomposeSurface_U_dir(int m,int n,int p,real[] U,triple[][] Pw){
    /*  Decompose surface into Bezier strips in u direction */
    /*  Input: n,p,U,Pw */
    /*  
        m is the number of control points in v-direction
        n is the number of control points in u-direction
        p is the degree in u-direction
        U is the knot vector in u-direction 
        Pw is the control points of the surface
    */
    /*  Output: nb, Qw */
    /*
        Qw is the control points set
    */
    triple[][][] Qw = new triple[floor(n/p)+1][p+1][m];
    int a=p;
    int b=p + 1;
    int nb=0;
    for(int i=0;i<=p;++i){
        for(int col_num=0;col_num<m;++col_num){
            Qw[nb][i][col_num]=Pw[i][col_num];
        }
    }

    int U_len = U.length;
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
                        Qw[nb][k][col_num]=alpha*Qw[nb][k][col_num]+(1.0 - alpha)*Qw[nb][k-1][col_num];
                    }
                }
                if (b<n){
                    for(int col_num=0;col_num<m;++col_num){
                        Qw[nb+1][save][col_num]=Qw[nb][p][col_num];
                    }
                }
            }
        }
        ++nb;
        if(b<n){
            for(int i=p-multi;i<=p;++i){
                for(int col_num=0;col_num<m;++col_num){
                    Qw[nb][i][col_num]=Pw[b-p+i][col_num];
                }
            }
            a=b;
            ++b;
        }
        else{
            // need to change knot vector
            break;
        }
    }
    write("End of Decompose U function");
    return Qw;
}

triple[][][] DecomposeSurface_V_dir(int m,int n,int q,real[] V,triple[][] Pw){
/*  Decompose surface into Bezier strips in v direction */
    /*  Input: m,q,V,Pw*/
    /*
        m is the number of control points in v-direction
        n is the number of control points in u-direction
        q is the degree in v-direction
        V is the knot vector in v-direction 
        Pw is the control points of the surface
    */
    /*  Output: nb, Qw */
    /*
        Qw is the control points set
    */
        triple[][][] Qw=new triple[floor(m/q)+1][q+1][n];
        int V_len=V.length;
        int a=q;
        int b=q+1;
        int nb=0;
        for(int i=0;i<=q;++i){
            for(int row_num=0;row_num<n;++row_num){
                Qw[nb][i][row_num]=Pw[row_num][i];
            }
        }
        
        int oldb;
        int multi;
        real[] alphas=new real[q-1]; // interpolation ratios

        while(b<V_len){
            oldb=b;
            while(b<V_len-1&&V[b]==V[b+1]){
                ++b;
            }
            multi=b-oldb+1;
            if(b==V_len-1 && V[b] != V[b-1]){
                multi = 1;
            }
            if(multi < q){
                real numerator = V[b] - V[a];
                for(int i = q ; i > multi; --i){
                    real denominator = V[a+i]-V[a];
                    alphas[i-multi-1] = numerator/denominator;
                }
                for (int j = 1; j <= q-multi; ++j){
                    int save = q - multi - j;
                    int s = multi + j;
                    for (int k = q; k >= s; --k){
                        real alpha = alphas[k-s];
                        for(int row_num = 0; row_num < n; ++row_num){
                            Qw[nb][k][row_num] = alpha*Qw[nb][k][row_num]+(1.0 - alpha)*Qw[nb][k-1][row_num];
                        }
                    }
                    if (b < m){
                        for(int row_num = 0; row_num < n; ++row_num){
                            Qw[nb+1][save][row_num] = Qw[nb][q][row_num];
                        }
                    }
                }
            }
            ++nb;
            if(b < m){
                for(int i = q - multi;i <= q; ++i){
                    for(int row_num = 0; row_num < n; ++row_num){
                        Qw[nb][i][row_num] = Pw[row_num][b-q+i];
                    }
                }
                a = b;
                ++b;
            }
            else{
                break;
            }
        }
    write("End of Decompose V function");
    return Qw;
}


void drawNURBSsurface(triple[][] cp, real[] U_knot, real[] V_knot, real[][] weights = array(cp.length,array(cp[0].length,1.0))){
    int n = cp.length; // number of control points in u direction
    int p = U_knot.length - n - 1;//degree of u
    int m = cp[0].length; // number of control points in v direction
    int q = V_knot.length - m - 1;//degree of v
    // First apply knot insertion in u direction to get Bezier strips
    triple[][][] Bezier_u_strips = DecomposeSurface_U_dir(m,n,p,U_knot,cp);
    triple[][] U_Qw;
    int U_Qw_ind = 0;

    for(int i=0;i<Bezier_u_strips.length;++i){
        for(int j=0;j<Bezier_u_strips[i].length;++j){
            U_Qw[U_Qw_ind]=copy(Bezier_u_strips[i][j]);
            ++U_Qw_ind;
        }
    }

    n=U_Qw.length;
    m=U_Qw[0].length;
    int rowcount=0;
    triple[][][] Bezier_v_strips=DecomposeSurface_V_dir(m,n,q,V_knot,U_Qw);
    triple[][] V_Qw=new triple[Bezier_v_strips[0][0].length][Bezier_v_strips[0].length*Bezier_v_strips.length];

    for(int i=0;i<Bezier_v_strips.length;++i){
        for(int j=0;j<Bezier_v_strips[i].length;++j){
            for(int k=0;k<Bezier_v_strips[i][j].length;++k){
                V_Qw[k][j+rowcount]=Bezier_v_strips[i][j][k];
            }
        }
        rowcount+=Bezier_v_strips[i].length;
    }

    triple[][][][] Bezier_surfaces = new triple[ceil(V_Qw.length/(p+1))][ceil(V_Qw[0].length/(q+1))][][];

    triple[][] row_trunc;
    triple[][] matrix_trunc;
    for(int i=0;i<ceil(V_Qw.length/(p+1));++i){
        row_trunc=V_Qw[i*(p+1):(i+1)*(p+1)];
        for(int j=0;j<ceil(V_Qw[i].length/(q+1));++j){
            for(int k=0;k<row_trunc.length;++k){
                matrix_trunc[k]=row_trunc[k][j*(q+1):(j+1)*(q+1)];
            }
            Bezier_surfaces[i][j]=copy(matrix_trunc);
        }
    }

    for(int i=0;i<Bezier_surfaces.length;++i){
        for(int j=0;j<Bezier_surfaces[i].length;++j){
            for(int row=0;row<Bezier_surfaces[i][j].length;++row){
                for(int col=0;col<Bezier_surfaces[i][j][row].length;++col){
                    dot(Bezier_surfaces[i][j][row][col],red);
                }
            }
            surface s=surface(patch(Bezier_surfaces[i][j]));
            draw(s,green);
        }
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



size(10cm);
drawNURBSsurface(P,uknot,vknot);