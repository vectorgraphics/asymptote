import three;
import graph;

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


struct NURBSsurface{
    surface[] g;

    NURBSsurfaceData data;

    // Mesh data built during init, drawn by draw()
    triple[] vertices;
    triple[] normals;
    int[][] triangles;
    int[][] nindices;

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

        // Find segment boundaries: distinct knot values within valid domain [K[deg], K[mk-deg-1]]
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

        // Evaluate B-spline basis at parameter t using Cox-de Boor recursion
        real[] evalBasis(int deg, real t, real[] K, int ncp){
          int mk = K.length;

          // Handle full-multiplicity boundary knots: when t coincides with a
          // boundary knot that has multiplicity >= deg+1, the Cox-de Boor
          // recursion collapses all basis functions to zero (all denominators
          // are zero). Return the appropriate unit vector directly.
          bool leftBoundary = true;
          for(int i=0;i<=deg && i<mk;++i)
            if(abs(K[i]-t)>1e-12){ leftBoundary=false; break; }
          if(leftBoundary && ncp>0){
            real[] result = new real[ncp];
            for(int i=0;i<ncp;++i) result[i]=0;
            result[0]=1;
            return result;
          }

          bool rightBoundary = true;
          int startR = mk - deg - 1;
          if(startR < 0) startR = 0;
          for(int i=startR;i<mk;++i)
            if(abs(K[i]-t)>1e-12){ rightBoundary=false; break; }
          if(rightBoundary && ncp>0){
            real[] result = new real[ncp];
            for(int i=0;i<ncp;++i) result[i]=0;
            result[ncp-1]=1;
            return result;
          }

          // Standard Cox-de Boor recursion
          real[][] N = new real[deg+1][ncp];
          for(int i=0;i<ncp;++i) N[0][i] = 0;
          for(int i=0;i<ncp;++i){
            if(t > K[i]-1e-15 && t < K[i+1]+1e-15){ N[0][i]=1; break; }
          }
          if(N[0][ncp-1]==0 && abs(t-K[mk-1])<1e-12) N[0][ncp-1]=1;
          for(int r=1;r<=deg;++r)
            for(int i=0;i<ncp;++i){
              real t1=0,t2=0;
              real d1 = K[i+r]-K[i];
              real d2 = (i+1<ncp)?K[i+r+1]-K[i+1]:0;
              if(d1>1e-15) t1=(t-K[i])/d1*N[r-1][i];
              if(i+1<ncp && d2>1e-15) t2=(K[i+r+1]-t)/d2*N[r-1][i+1];
              N[r][i]=t1+t2;
            }

          // Safety check: if all basis functions are zero (shouldn't happen
          // after boundary fix), fall back to nearest non-zero knot span
          real bsum = 0;
          for(int i=0;i<ncp;++i) bsum += N[deg][i];
          if(bsum < 1e-12){
            for(int i=0;i<ncp;++i) N[deg][i]=0;
            // Find the span where t belongs and activate that basis function
            for(int i=0;i<ncp;++i){
              if(t >= K[i]-1e-12 && t <= K[i+1]+1e-12){
                N[deg][i]=1; break;
              }
            }
            if(N[deg][ncp-1]==0) N[deg][ncp-1]=1;
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

        // Derivative of B-spline basis functions using standard formula:
        // N_i'^p(t) = p * (N_i^(p-1)(t) - N_(i+1)^(p-1)(t)) / (K_(i+p) - K_(i+1))
        real[] evalBasisDeriv(int deg, real t, real[] K, int ncp){
          real[] Nprev = evalBasis(deg-1, t, K, ncp);
          real[] Nderiv;
          for(int i=0;i<ncp;++i) Nderiv[i] = 0; // explicit init
          for(int i=0;i<ncp;++i){
            real denom = K[i+deg]-K[i+1];
            if(abs(denom) > 1e-15){
              real num = 0;
              if(i < ncp) num += Nprev[i];
              if(i+1 < ncp) num -= Nprev[i+1];
              Nderiv[i] = deg * num / denom;
            }
          }
          return Nderiv;
        }

        // Compute exact surface normal from NURBS derivatives (quotient rule)
        // Returns normal via cross(du, dv) — CCW convention
        triple evalNormal(real u, real v){
          real[] Nu = evalBasis(p, u, Uk, nu);
          real[] Nv = evalBasis(q, v, Vk, nv);
          real[] Ndu = evalBasisDeriv(p, u, Uk, nu);
          real[] Ndv = evalBasisDeriv(q, v, Vk, nv);

          // A = sum(Nu[i]*Nv[j] * Pw[i][j]), w = sum(Nu[i]*Nv[j] * weight)
          triple A=(0,0,0); real wt=0;
          triple Adu=(0,0,0); real wdu=0;
          triple Adv=(0,0,0); real wdv=0;

          for(int i=0;i<nu;++i)
            for(int j=0;j<nv;++j){
              triple Pwi = (Pw[i][j][0],Pw[i][j][1],Pw[i][j][2]);
              real sw = Pw[i][j][3];
              Adu += Ndu[i]*Nv[j]*Pwi; wdu += Ndu[i]*Nv[j]*sw;
              Adv += Nu[i]*Ndv[j]*Pwi; wdv += Nu[i]*Ndv[j]*sw;
              A    += Nu[i]*Nv[j]*Pwi; wt  += Nu[i]*Nv[j]*sw;
            }

          triple pos = (abs(wt)>1e-15)?A/wt:(0,0,0);
          // Quotient rule: d(pos)/du = (Adu*wt - A*wdu) / wt^2
          triple du = (abs(wt)>1e-15)?(Adu*wt - A*wdu)/(wt*wt):(0,0,0);
          triple dv = (abs(wt)>1e-15)?(Adv*wt - A*wdv)/(wt*wt):(0,0,0);

          // At poles, both tangents vanish — use radial direction
          if(length(du) < 1e-10 || length(dv) < 1e-10){
            return unit(pos);
          }

          // CCW convention: cross(du, dv) gives the correct normal direction
          triple n = cross(du, dv);
          if(length(n) < 1e-10) return unit(pos);
          return unit(n);
        }

        // Recursively subdivide and build triangle mesh
        int maxDepth = 4;

        void subdivide(real uL, real vL, real uR, real vR, int depth){
          // Evaluate the 4 corners and center
          triple c00 = evalSurf(uL, vL);
          triple c10 = evalSurf(uR, vL);
          triple c01 = evalSurf(uL, vR);
          triple c11 = evalSurf(uR, vR);
          triple cm  = evalSurf(0.5*(uL+uR), 0.5*(vL+vR));

          // Flatness test: use plane through c00, c10, c01 and measure
          // deviation of c11 and cm from this plane
          triple e1 = c10 - c00;
          triple e2 = c01 - c00;
          triple pln = cross(e1, e2);
          real plnLen = length(pln);

          real maxd = 0;
          if(plnLen > 1e-15){
            pln = unit(pln);
            real d11 = abs(dot(c11 - c00, pln));
            real dcm = abs(dot(cm - c00, pln));
            maxd = max(d11, dcm);
          } else {
            // Degenerate quad (all corners collinear or coincident)
            maxd = max(length(c11-cm), length(c00-cm));
          }

          // Flatness threshold: use relative to surface size
          real sz = max(length(c10-c00), length(c01-c00));
          if(depth >= maxDepth || maxd < 0.001 * sz){
            int base = vertices.length;

            // Add vertices with exact NURBS normals
            triple n00 = evalNormal(uL, vL);
            triple n10 = evalNormal(uR, vL);
            triple n11 = evalNormal(uR, vR);
            triple n01 = evalNormal(uL, vR);
            triple nm  = evalNormal(0.5*(uL+uR), 0.5*(vL+vR));

            vertices.push(c00); normals.push(n00);
            vertices.push(c10); normals.push(n10);
            vertices.push(c11); normals.push(n11);
            vertices.push(c01); normals.push(n01);
            vertices.push(cm);  normals.push(nm);

            // 4 triangles around center (vertex index base+4)
            int[] t0={base, base+1, base+4};
            int[] t1={base+1, base+2, base+4};
            int[] t2={base+2, base+3, base+4};
            int[] t3={base+3, base, base+4};
            triangles.push(t0);
            triangles.push(t1);
            triangles.push(t2);
            triangles.push(t3);

            // Normal indices match vertex indices
            nindices.push(t0);
            nindices.push(t1);
            nindices.push(t2);
            nindices.push(t3);
          } else {
            // Subdivide into 4 sub-quads
            real um = 0.5*(uL+uR), vm = 0.5*(vL+vR);
            subdivide(uL, vL, um, vm, depth+1);
            subdivide(um, vL, uR, vm, depth+1);
            subdivide(uL, vm, um, vR, depth+1);
            subdivide(um, vm, uR, vR, depth+1);
          }
        }

        // Process each B-spline segment
        for(int si=0;si<u_segs.length-1;++si)
          for(int sj=0;sj<v_segs.length-1;++sj)
            subdivide(u_segs[si], v_segs[sj], u_segs[si+1], v_segs[sj+1], 0);
    }

    void draw(picture pic=currentpicture,pen p=currentpen) {
      // Draw the triangle mesh with exact NURBS normals and user-specified pen
      draw(pic, vertices, triangles, normals, nindices, material(p));
    }
}
