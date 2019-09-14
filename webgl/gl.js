// Render Bezier patches via subdivision with WebGL.
// Author: John C. Bowman, Supakorn "Jamie" Rassameemasmuang

var gl;

var canvasWidth,canvasHeight;
var halfCanvWidth,halfCanvHeight;

var pixel=0.75; // Adaptive rendering constant.
var BezierFactor=0.4;
var FillFactor=0.1;
var Zoom;
var Zoom0;
const zoomStep=0.1;
var zoomFactor=1.05;
var zoomPinchFactor=10;
var zoomPinchCap=100;
var shiftHoldDistance=20;
var shiftWaitTime=200; // ms
var vibrateTime=25; // ms
var lastzoom;
var H; // maximum camera view half-height

var Fuzz2=1000*Number.EPSILON;
var Fuzz4=Fuzz2*Fuzz2;
var third=1.0/3.0;

var P=[]; // Array of Bezier patches, triangles, and curves
var M=[]; // Array of materials
var Centers=[]; // Array of billboard centers

var rotMat=mat4.create();
var pMatrix=mat4.create(); // projection matrix
var vMatrix=mat4.create(); // view matrix
var T=mat4.create(); // Offscreen transformation matrix

var pvmMatrix=mat4.create(); // projection view matrix
var normMat=mat3.create();
var vMatrix3=mat3.create(); // 3x3 view matrix
var rotMats=mat4.create();

var zmin,zmax;
var center={x:0,y:0,z:0};
var size2;
var b,B; // Scene min,max bounding box corners
var shift={
  x:0,y:0
};

var viewParam = {
  xmin:0,xmax:0,
  ymin:0,ymax:0,
  zmin:0,zmax:0
};

var positionBuffer;
var materialBuffer;
var colorBuffer;
var indexBuffer;

var redraw=true;
var remesh=true;
var mouseDownOrTouchActive=false;
var lastMouseX=null;
var lastMouseY=null;
var touchID=null;

class Material {
  constructor(diffuse,emissive,specular,shininess,metallic,fresnel0) {
    this.diffuse=diffuse;
    this.emissive=emissive;
    this.specular=specular;
    this.shininess=shininess;
    this.metallic=metallic;
    this.fresnel0=fresnel0;
  }

  setUniform(program,stringLoc,index=null) {
    var getLoc;
    if (index === null)
      getLoc =
        param => gl.getUniformLocation(program,stringLoc+"."+param);
    else
      getLoc =
        param => gl.getUniformLocation(program,stringLoc+"["+index+"]."+param);

    gl.uniform4fv(getLoc("diffuse"),new Float32Array(this.diffuse));
    gl.uniform4fv(getLoc("emissive"),new Float32Array(this.emissive));
    gl.uniform4fv(getLoc("specular"),new Float32Array(this.specular));

    gl.uniform1f(getLoc("shininess"),this.shininess);
    gl.uniform1f(getLoc("metallic"),this.metallic);
    gl.uniform1f(getLoc("fresnel0"),this.fresnel0);
  }
}

var enumPointLight=1;
var enumDirectionalLight=2;

class Light {
  constructor(type,lightColor,brightness,customParam) {
    this.type=type;
    this.lightColor=lightColor;
    this.brightness=brightness;
    this.customParam=customParam;
  }

  setUniform(program,stringLoc,index) {
    var getLoc=
        param => gl.getUniformLocation(program,stringLoc+"["+index+"]."+param);

    gl.uniform1i(getLoc("type"),this.type);
    gl.uniform3fv(getLoc("color"),new Float32Array(this.lightColor));
    gl.uniform1f(getLoc("brightness"),this.brightness);
    gl.uniform4fv(getLoc("parameter"),new Float32Array(this.customParam));
  }
}

function initGL(canvas) {
  try {
    gl=canvas.getContext("webgl");
    gl.viewportWidth=canvas.width;
    gl.viewportHeight=canvas.height;
  } catch(e) {}
  if (!gl)
    alert("Could not initialize WebGL");
}

function getShader(gl,id,options=[]) {
  var shaderScript=document.getElementById(id);
  if(!shaderScript)
    return null;

  var str=`#version 100
  precision highp float;
  const int nLights=${lights.length};
  const int nMaterials=${M.length};
  const int nCenters=${Math.max(Centers.length,1)};\n`

  if(orthographic)
    str += `#define ORTHOGRAPHIC\n`;

  options.forEach(s => str += `#define `+s+`\n`);

  var k=shaderScript.firstChild;
  while(k) {
    if(k.nodeType == 3)
      str += k.textContent;
    k=k.nextSibling;
  }
  var shader;
  if(shaderScript.type == "x-shader/x-fragment")
    shader = gl.createShader(gl.FRAGMENT_SHADER);
  else if (shaderScript.type == "x-shader/x-vertex")
    shader = gl.createShader(gl.VERTEX_SHADER);
  else
    return null;

  gl.shaderSource(shader,str);
  gl.compileShader(shader);
  if(!gl.getShaderParameter(shader,gl.COMPILE_STATUS)) {
    alert(gl.getShaderInfoLog(shader));
    return null;
  }
  return shader;
}


function drawBuffer(data,shader,indices=data.indices)
{
  let normal=shader != noNormalShader;
  setUniforms(shader);

  gl.bindBuffer(gl.ARRAY_BUFFER,positionBuffer);
  gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(data.vertices),
                gl.STATIC_DRAW);
  gl.vertexAttribPointer(shader.vertexPositionAttribute,
                         3,gl.FLOAT,false,normal ? 24 : 12,0);
  if(normal)
    gl.vertexAttribPointer(shader.vertexNormalAttribute,
                           3,gl.FLOAT,false,24,12);

  gl.bindBuffer(gl.ARRAY_BUFFER,materialBuffer);
  gl.bufferData(gl.ARRAY_BUFFER,new Int16Array(data.materials),
                gl.STATIC_DRAW);
  gl.vertexAttribPointer(shader.vertexMaterialAttribute,
                         1,gl.SHORT,false,4,0);
  gl.vertexAttribPointer(shader.vertexCenterAttribute,
                         1,gl.SHORT,false,4,2);

  if(shader == colorShader || shader == transparentShader) {
    gl.bindBuffer(gl.ARRAY_BUFFER,colorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER,new Uint8Array(data.colors),
                  gl.STATIC_DRAW);
    gl.vertexAttribPointer(shader.vertexColorAttribute,
                           4,gl.UNSIGNED_BYTE,true,0,0);
  }

  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,indexBuffer);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,
                indexExt ? new Uint32Array(indices) :
                new Uint16Array(indices),gl.STATIC_DRAW);

  gl.drawElements(shader == noNormalShader ? gl.LINES : gl.TRIANGLES,
                  indices.length,
                  indexExt ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT,0);
}

class vertexData {
  constructor() {
    this.clear();
  }
  clear() {
    this.vertices=[];
    this.materials=[];
    this.colors=[];
    this.indices=[];

    this.nvertices=0;
  }

  // material vertex 
  vertex(v,n) {
    this.vertices.push(v[0]);
    this.vertices.push(v[1]);
    this.vertices.push(v[2]);
    this.vertices.push(n[0]);
    this.vertices.push(n[1]);
    this.vertices.push(n[2]);
    this.materials.push(materialIndex);
    this.materials.push(centerIndex);
    return this.nvertices++;
  }

  // colored vertex
  Vertex(v,n,c=[0,0,0,0]) {
    this.vertices.push(v[0]);
    this.vertices.push(v[1]);
    this.vertices.push(v[2]);
    this.vertices.push(n[0]);
    this.vertices.push(n[1]);
    this.vertices.push(n[2]);
    this.materials.push(materialIndex);
    this.materials.push(centerIndex);
    this.colors.push(c[0]);
    this.colors.push(c[1]);
    this.colors.push(c[2]);
    this.colors.push(c[3]);
    return this.nvertices++;
  }

   // material vertex without normal
  vertex1(v) {
    this.vertices.push(v[0]);
    this.vertices.push(v[1]);
    this.vertices.push(v[2]);
    this.materials.push(materialIndex);
    this.materials.push(centerIndex);
    return this.nvertices++;
  }

 copy(data) {
    this.vertices=data.vertices.slice();
    this.materials=data.materials.slice();
    this.colors=data.colors.slice();
    this.indices=data.indices.slice();
    this.nvertices=data.nvertices;
  }

  append(data) {
    append(this.vertices,data.vertices);
    append(this.materials,data.materials);
    append(this.colors,data.colors);
    appendOffset(this.indices,data.indices,this.nvertices);
    this.nvertices += data.nvertices;
  }
}

var data=new vertexData();

var materialOn=new vertexData();     // Onscreen material data
var materialOff=new vertexData();    // Partially offscreen material data

var colorOn=new vertexData();        // Onscreen color data
var colorOff=new vertexData();       // Partially offscreen color data

var transparentOn=new vertexData();  // Onscreen transparent data
var transparentOff=new vertexData(); // Partially offscreen transparent data

var material1On=new vertexData();     // Onscreen material 1D data
var material1Off=new vertexData();    // Partially offscreen material 1D data

var materialIndex;
var centerIndex;

// efficiently append array b onto array a
function append(a,b)
{
  let n=a.length;
  a.length += b.length;
  for(let i=0, m=b.length; i < m; ++i)
    a[n+i]=b[i];
}

// efficiently append array b onto array a
function appendOffset(a,b,o)
{
  let n=a.length;
  a.length += b.length;
  for(let i=0, m=b.length; i < m; ++i)
    a[n+i]=b[i]+o;
}

class BezierPatch {
  /**
   * Constructor for Bezier Patch
   * @param {*} controlpoints array of 16 control points
   * @param {*} CenterIndex center index of billboard labels (or 0)
   * @param {*} MaterialIndex material index (>= 0)
   * @param {*} Minimum bounding box corner
   * @param {*} Maximum bounding box corner
   * @param {*} colors array of 4 RGBA color arrays
   */
  constructor(controlpoints,CenterIndex,MaterialIndex,Min,Max,color) {
    this.controlpoints=controlpoints;
    this.Min=Min;
    this.Max=Max;
    this.color=color;
    this.CenterIndex=CenterIndex;
    this.transparent=color ?
      color[0][3]+color[1][3]+color[2][3]+color[3][3] < 1020 :
      M[MaterialIndex].diffuse[3] < 1.0;
    this.MaterialIndex=this.transparent ?
      (color ? -1-MaterialIndex : 1+MaterialIndex) : MaterialIndex;
    this.vertex=(this.color || this.transparent) ?
      data.Vertex.bind(data) : data.vertex.bind(data);
    this.L2norm();
  }

  // Approximate bounds by bounding box of control polyhedron.
  offscreen(n,v) {
    let x,y,z;
    let X,Y,Z;

    X=x=v[0][0];
    Y=y=v[0][1];
    Z=z=v[0][2];
    
    for(let i=1; i < n; ++i) {
      let V=v[i];
      if(V[0] < x) x=V[0];
      else if(V[0] > X) X=V[0];
      if(V[1] < y) y=V[1];
      else if(V[1] > Y) Y=V[1];
    }

    if(X >= this.x && x <= this.X &&
       Y >= this.y && y <= this.Y)
      return false;

    return this.Offscreen=true;
  }

  OffScreen() {
    centerIndex=this.CenterIndex;
    materialIndex=this.MaterialIndex;

    let b=[viewParam.xmin,viewParam.ymin,viewParam.zmin];
    let B=[viewParam.xmax,viewParam.ymax,viewParam.zmax];

    let s,m,M;

    if(orthographic) {
      m=b;
      M=B;
      s=1.0;
    } else {
      let perspective=1.0/B[2];
      let f=this.Min[2]*perspective;
      let F=this.Max[2]*perspective;
      m=[Math.min(f*b[0],F*b[0]),Math.min(f*b[1],F*b[1]),b[2]];
      M=[Math.max(f*B[0],F*B[0]),Math.max(f*B[1],F*B[1]),B[2]];
      s=Math.max(f,F);
    }

    [this.x,this.y,this.X,this.Y]=new bbox2(m,M).bounds();

    if(centerIndex == 0 &&
       (this.Max[0] < this.x || this.Min[0] > this.X ||
        this.Max[1] < this.y || this.Min[1] > this.Y)) {
      return this.Offscreen=true;
    }

    let res=pixel*Math.hypot(s*(B[0]-b[0]),s*(B[1]-b[1]))/size2;
    this.res2=res*res;
    this.Epsilon=FillFactor*res;
    return this.Offscreen=false;
  }

// Render a Bezier patch via subdivision.
  L2norm() {
    let p=this.controlpoints;
    let p0=p[0];
    this.epsilon=0;
    let n=p.length;
    for(let i=1; i < n; ++i)
      this.epsilon=Math.max(this.epsilon,
        abs2([p[i][0]-p0[0],p[i][1]-p0[1],p[i][2]-p0[2]]));
    this.epsilon *= Fuzz4;
  }

  render() {
    if(this.OffScreen()) return;

    let p=this.controlpoints;
    if(p.length == 10) return this.render3();
    
    let p0=p[0];
    let p3=p[3];
    let p12=p[12];
    let p15=p[15];

    let n0=this.normal(p3,p[2],p[1],p0,p[4],p[8],p12);
    if(iszero(n0)) {
      n0=this.normal(p3,p[2],p[1],p0,p[13],p[14],p15);
      if(iszero(n0)) n0=this.normal(p15,p[11],p[7],p3,p[4],p[8],p12);
    }

    let n1=this.normal(p0,p[4],p[8],p12,p[13],p[14],p15);
    if(iszero(n1)) {
      n1=this.normal(p0,p[4],p[8],p12,p[11],p[7],p3);
      if(iszero(n1)) n1=this.normal(p3,p[2],p[1],p0,p[13],p[14],p15);
    }

    let n2=this.normal(p12,p[13],p[14],p15,p[11],p[7],p3);
    if(iszero(n2)) {
      n2=this.normal(p12,p[13],p[14],p15,p[2],p[1],p0);
      if(iszero(n2)) n2=this.normal(p0,p[4],p[8],p12,p[11],p[7],p3);
    }

    let n3=this.normal(p15,p[11],p[7],p3,p[2],p[1],p0);
    if(iszero(n3)) {
      n3=this.normal(p15,p[11],p[7],p3,p[4],p[8],p12);
      if(iszero(n3)) n3=this.normal(p12,p[13],p[14],p15,p[2],p[1],p0);
    }

    if(this.color) {
      let c0=this.color[0];
      let c1=this.color[1];
      let c2=this.color[2];
      let c3=this.color[3];

      let i0=data.Vertex(p0,n0,c0);
      let i1=data.Vertex(p12,n1,c1);
      let i2=data.Vertex(p15,n2,c2);
      let i3=data.Vertex(p3,n3,c3);

      this.Render(p,i0,i1,i2,i3,p0,p12,p15,p3,false,false,false,false,
                  c0,c1,c2,c3);
    } else {
      let i0=this.vertex(p0,n0);
      let i1=this.vertex(p12,n1);
      let i2=this.vertex(p15,n2);
      let i3=this.vertex(p3,n3);

      this.Render(p,i0,i1,i2,i3,p0,p12,p15,p3,false,false,false,false);
    }
    if(data.nvertices > 0) this.append();
  }

  append() {
    if(this.transparent) {
      if(this.Offscreen)
        transparentOff.append(data);
      else
        transparentOn.append(data);
    } else {
      if(this.color) {
        if(this.Offscreen)
          colorOff.append(data);
        else
          colorOn.append(data);
      } else {
        if(this.Offscreen)
          materialOff.append(data);
        else
          materialOn.append(data);
      }
    }
    data.clear();
  }

  Render(p,I0,I1,I2,I3,P0,P1,P2,P3,flat0,flat1,flat2,flat3,C0,C1,C2,C3) {
    if(this.Distance(p) < this.res2) { // Bezier patch is flat
      let P=[P0,P1,P2,P3];
      if(!this.offscreen(4,P)) {
        data.indices.push(I0);
        data.indices.push(I1);
        data.indices.push(I2);
        
        data.indices.push(I0);
        data.indices.push(I2);
        data.indices.push(I3);
      }
    } else {
      if(this.offscreen(16,p)) return;

    /* Control points are indexed as follows:
         
       Coordinate
       +-----
        Index
         

       03    13    23    33
       +-----+-----+-----+
       |3    |7    |11   |15
       |     |     |     |
       |02   |12   |22   |32
       +-----+-----+-----+
       |2    |6    |10   |14
       |     |     |     |
       |01   |11   |21   |31
       +-----+-----+-----+
       |1    |5    |9    |13
       |     |     |     |
       |00   |10   |20   |30
       +-----+-----+-----+
       0     4     8     12
         

       Subdivision:
       P refers to a corner
       m refers to a midpoint
       s refers to a subpatch
         
                m2
       +--------+--------+
       |P3      |      P2|
       |        |        |
       |   s3   |   s2   |
       |        |        |
       |        |m4      |
     m3+--------+--------+m1
       |        |        |
       |        |        |
       |   s0   |   s1   |
       |        |        |
       |P0      |      P1|
       +--------+--------+
                m0
    */
    
      // Subdivide patch:

      let p0=p[0];
      let p3=p[3];
      let p12=p[12];
      let p15=p[15];

      let c0=new Split3(p0,p[1],p[2],p3);
      let c1=new Split3(p[4],p[5],p[6],p[7]);
      let c2=new Split3(p[8],p[9],p[10],p[11]);
      let c3=new Split3(p12,p[13],p[14],p15);

      let c4=new Split3(p0,p[4],p[8],p12);
      let c5=new Split3(c0.m0,c1.m0,c2.m0,c3.m0);
      let c6=new Split3(c0.m3,c1.m3,c2.m3,c3.m3);
      let c7=new Split3(c0.m5,c1.m5,c2.m5,c3.m5);
      let c8=new Split3(c0.m4,c1.m4,c2.m4,c3.m4);
      let c9=new Split3(c0.m2,c1.m2,c2.m2,c3.m2);
      let c10=new Split3(p3,p[7],p[11],p15);

      let s0=[p0,c0.m0,c0.m3,c0.m5,c4.m0,c5.m0,c6.m0,c7.m0,
              c4.m3,c5.m3,c6.m3,c7.m3,c4.m5,c5.m5,c6.m5,c7.m5];
      let s1=[c4.m5,c5.m5,c6.m5,c7.m5,c4.m4,c5.m4,c6.m4,c7.m4,
              c4.m2,c5.m2,c6.m2,c7.m2,p12,c3.m0,c3.m3,c3.m5];
      let s2=[c7.m5,c8.m5,c9.m5,c10.m5,c7.m4,c8.m4,c9.m4,c10.m4,
              c7.m2,c8.m2,c9.m2,c10.m2,c3.m5,c3.m4,c3.m2,p15];
      let s3=[c0.m5,c0.m4,c0.m2,p3,c7.m0,c8.m0,c9.m0,c10.m0,
              c7.m3,c8.m3,c9.m3,c10.m3,c7.m5,c8.m5,c9.m5,c10.m5];

      let m4=s0[15];

      let n0=this.normal(s0[0],s0[4],s0[8],s0[12],s0[13],s0[14],s0[15]);
      if(iszero(n0)) {
        n0=this.normal(s0[0],s0[4],s0[8],s0[12],s0[11],s0[7],s0[3]);
        if(iszero(n0))
          n0=this.normal(s0[3],s0[2],s0[1],s0[0],s0[13],s0[14],s0[15]);
      }

      let n1=this.normal(s1[12],s1[13],s1[14],s1[15],s1[11],s1[7],s1[3]);
      if(iszero(n1)) {
        n1=this.normal(s1[12],s1[13],s1[14],s1[15],s1[2],s1[1],s1[0]);
        if(iszero(n1))
          n1=this.normal(s1[0],s1[4],s1[8],s1[12],s1[11],s1[7],s1[3]);
      }

      let n2=this.normal(s2[15],s2[11],s2[7],s2[3],s2[2],s2[1],s2[0]);
      if(iszero(n2)) {
        n2=this.normal(s2[15],s2[11],s2[7],s2[3],s2[4],s2[8],s2[12]);
        if(iszero(n2))
          n2=this.normal(s2[12],s2[13],s2[14],s2[15],s2[2],s2[1],s2[0]);
      }

      let n3=this.normal(s3[3],s3[2],s3[1],s3[0],s3[4],s3[8],s3[12]);
      if(iszero(n3)) {
        n3=this.normal(s3[3],s3[2],s3[1],s3[0],s3[13],s3[14],s3[15]);
        if(iszero(n3))
          n3=this.normal(s3[15],s3[11],s3[7],s3[3],s3[4],s3[8],s3[12]);
      }

      let n4=this.normal(s2[3],s2[2],s2[1],m4,s2[4],s2[8],s2[12]);

      let e=this.Epsilon;

      // A kludge to remove subdivision cracks, only applied the first time
      // an edge is found to be flat before the rest of the subpatch is.
      let m0=[0.5*(P0[0]+P1[0]),
              0.5*(P0[1]+P1[1]),
              0.5*(P0[2]+P1[2])];
      if(!flat0) {
        if((flat0=Straightness(p0,p[4],p[8],p12) < this.res2)) {
          let r=unit(this.derivative(s1[0],s1[1],s1[2],s1[3]));
          m0=[m0[0]-e*r[0],m0[1]-e*r[1],m0[2]-e*r[2]];
        }
        else m0=s0[12];
      }

      let m1=[0.5*(P1[0]+P2[0]),
              0.5*(P1[1]+P2[1]),
              0.5*(P1[2]+P2[2])];
      if(!flat1) {
        if((flat1=Straightness(p12,p[13],p[14],p15) < this.res2)) {
          let r=unit(this.derivative(s2[12],s2[8],s2[4],s2[0]));
          m1=[m1[0]-e*r[0],m1[1]-e*r[1],m1[2]-e*r[2]];
        }
        else m1=s1[15];
      }

      let m2=[0.5*(P2[0]+P3[0]),
              0.5*(P2[1]+P3[1]),
              0.5*(P2[2]+P3[2])];
      if(!flat2) {
        if((flat2=Straightness(p15,p[11],p[7],p3) < this.res2)) {
          let r=unit(this.derivative(s3[15],s2[14],s2[13],s1[12]));
          m2=[m2[0]-e*r[0],m2[1]-e*r[1],m2[2]-e*r[2]];
        }
        else m2=s2[3];
      }
      
      let m3=[0.5*(P3[0]+P0[0]),
              0.5*(P3[1]+P0[1]),
              0.5*(P3[2]+P0[2])];
      if(!flat3) {
        if((flat3=Straightness(p0,p[1],p[2],p3) < this.res2)) {
          let r=unit(this.derivative(s0[3],s0[7],s0[11],s0[15]));
          m3=[m3[0]-e*r[0],m3[1]-e*r[1],m3[2]-e*r[2]];
        }
        else m3=s3[0];
      }
      
      if(C0) {
        let c0=Array(4);
        let c1=Array(4);
        let c2=Array(4);
        let c3=Array(4);
        let c4=Array(4);
        for(let i=0; i < 4; ++i) {
          c0[i]=0.5*(C0[i]+C1[i]);
          c1[i]=0.5*(C1[i]+C2[i]);
          c2[i]=0.5*(C2[i]+C3[i]);
          c3[i]=0.5*(C3[i]+C0[i]);
          c4[i]=0.5*(c0[i]+c2[i]);
        }

        let i0=data.Vertex(m0,n0,c0);
        let i1=data.Vertex(m1,n1,c1);
        let i2=data.Vertex(m2,n2,c2);
        let i3=data.Vertex(m3,n3,c3);
        let i4=data.Vertex(m4,n4,c4);

        this.Render(s0,I0,i0,i4,i3,P0,m0,m4,m3,flat0,false,false,flat3,
                    C0,c0,c4,c3);
        this.Render(s1,i0,I1,i1,i4,m0,P1,m1,m4,flat0,flat1,false,false,
                    c0,C1,c1,c4);
        this.Render(s2,i4,i1,I2,i2,m4,m1,P2,m2,false,flat1,flat2,false,
                    c4,c1,C2,c2);
        this.Render(s3,i3,i4,i2,I3,m3,m4,m2,P3,false,false,flat2,flat3,
                    c3,c4,c2,C3);
      } else {
        let i0=this.vertex(m0,n0);
        let i1=this.vertex(m1,n1);
        let i2=this.vertex(m2,n2);
        let i3=this.vertex(m3,n3);
        let i4=this.vertex(m4,n4);

        this.Render(s0,I0,i0,i4,i3,P0,m0,m4,m3,flat0,false,false,flat3);
        this.Render(s1,i0,I1,i1,i4,m0,P1,m1,m4,flat0,flat1,false,false);
        this.Render(s2,i4,i1,I2,i2,m4,m1,P2,m2,false,flat1,flat2,false);
        this.Render(s3,i3,i4,i2,I3,m3,m4,m2,P3,false,false,flat2,flat3);
      }
    }
  }

// Render a Bezier triangle via subdivision.
  render3() {
    this.Res2=BezierFactor*BezierFactor*this.res2;
    let p=this.controlpoints;

    let p0=p[0];
    let p6=p[6];
    let p9=p[9];

    let n0=this.normal(p9,p[5],p[2],p0,p[1],p[3],p6);
    let n1=this.normal(p0,p[1],p[3],p6,p[7],p[8],p9);    
    let n2=this.normal(p6,p[7],p[8],p9,p[5],p[2],p0);
    
    if(this.color) {
      let c0=this.color[0];
      let c1=this.color[1];
      let c2=this.color[2];

      let i0=data.Vertex(p0,n0,c0);
      let i1=data.Vertex(p6,n1,c1);
      let i2=data.Vertex(p9,n2,c2);
    
      this.Render3(p,i0,i1,i2,p0,p6,p9,false,false,false,c0,c1,c2);

    } else {
      let i0=this.vertex(p0,n0);
      let i1=this.vertex(p6,n1);
      let i2=this.vertex(p9,n2);

      this.Render3(p,i0,i1,i2,p0,p6,p9,false,false,false);
    }
    if(data.nvertices > 0) this.append();
  }

  Render3(p,I0,I1,I2,P0,P1,P2,flat0,flat1,flat2,C0,C1,C2) {
    if(this.Distance3(p) < this.Res2) { // Bezier triangle is flat
      let P=[P0,P1,P2];
      if(!this.offscreen(3,P)) {
        data.indices.push(I0);
        data.indices.push(I1);
        data.indices.push(I2);
      }
    } else {
      if(this.offscreen(10,p)) return;

    /* Control points are indexed as follows:

       Coordinate
        Index

                                  030
                                   9
                                   /\
                                  /  \
                                 /    \
                                /      \
                               /        \
                          021 +          + 120
                           5 /            \ 8
                            /              \
                           /                \
                          /                  \
                         /                    \
                    012 +          +           + 210
                     2 /          111           \ 7
                      /            4             \
                     /                            \
                    /                              \
                   /                                \
                  /__________________________________\
                003         102           201        300
                 0           1             3          6


       Subdivision:
                                   P2
                                   030
                                   /\
                                  /  \
                                 /    \
                                /      \
                               /        \
                              /    up    \
                             /            \
                            /              \
                        p1 /________________\ p0
                          /\               / \
                         /  \             /   \
                        /    \           /     \
                       /      \  center /       \
                      /        \       /         \
                     /          \     /           \
                    /    left    \   /    right    \
                   /              \ /               \
                  /________________V_________________\
                003               p2                300
                P0                                    P1
    */
      
      // Subdivide triangle:

      let l003=p[0];
      let p102=p[1];
      let p012=p[2];
      let p201=p[3];
      let p111=p[4];
      let p021=p[5];
      let r300=p[6];
      let p210=p[7];
      let p120=p[8];
      let u030=p[9];

      let u021=[0.5*(u030[0]+p021[0]),
                0.5*(u030[1]+p021[1]),
                0.5*(u030[2]+p021[2])];
      let u120=[0.5*(u030[0]+p120[0]),
                0.5*(u030[1]+p120[1]),
                0.5*(u030[2]+p120[2])];

      let p033=[0.5*(p021[0]+p012[0]),
                0.5*(p021[1]+p012[1]),
                0.5*(p021[2]+p012[2])];
      let p231=[0.5*(p120[0]+p111[0]),
                0.5*(p120[1]+p111[1]),
                0.5*(p120[2]+p111[2])];
      let p330=[0.5*(p120[0]+p210[0]),
                0.5*(p120[1]+p210[1]),
                0.5*(p120[2]+p210[2])];

      let p123=[0.5*(p012[0]+p111[0]),
                0.5*(p012[1]+p111[1]),
                0.5*(p012[2]+p111[2])];

      let l012=[0.5*(p012[0]+l003[0]),
                0.5*(p012[1]+l003[1]),
                0.5*(p012[2]+l003[2])];
      let p312=[0.5*(p111[0]+p201[0]),
                0.5*(p111[1]+p201[1]),
                0.5*(p111[2]+p201[2])];
      let r210=[0.5*(p210[0]+r300[0]),
                0.5*(p210[1]+r300[1]),
                0.5*(p210[2]+r300[2])];

      let l102=[0.5*(l003[0]+p102[0]),
                0.5*(l003[1]+p102[1]),
                0.5*(l003[2]+p102[2])];
      let p303=[0.5*(p102[0]+p201[0]),
                0.5*(p102[1]+p201[1]),
                0.5*(p102[2]+p201[2])];
      let r201=[0.5*(p201[0]+r300[0]),
                0.5*(p201[1]+r300[1]),
                0.5*(p201[2]+r300[2])];

      let u012=[0.5*(u021[0]+p033[0]),
                0.5*(u021[1]+p033[1]),
                0.5*(u021[2]+p033[2])];
      let u210=[0.5*(u120[0]+p330[0]),
                0.5*(u120[1]+p330[1]),
                0.5*(u120[2]+p330[2])];
      let l021=[0.5*(p033[0]+l012[0]),
                0.5*(p033[1]+l012[1]),
                0.5*(p033[2]+l012[2])];
      let p4xx=[0.5*p231[0]+0.25*(p111[0]+p102[0]),
                0.5*p231[1]+0.25*(p111[1]+p102[1]),
                0.5*p231[2]+0.25*(p111[2]+p102[2])];
      let r120=[0.5*(p330[0]+r210[0]),
                0.5*(p330[1]+r210[1]),
                0.5*(p330[2]+r210[2])];
      let px4x=[0.5*p123[0]+0.25*(p111[0]+p210[0]),
                0.5*p123[1]+0.25*(p111[1]+p210[1]),
                0.5*p123[2]+0.25*(p111[2]+p210[2])];
      let pxx4=[0.25*(p021[0]+p111[0])+0.5*p312[0],
                0.25*(p021[1]+p111[1])+0.5*p312[1],
                0.25*(p021[2]+p111[2])+0.5*p312[2]];
      let l201=[0.5*(l102[0]+p303[0]),
                0.5*(l102[1]+p303[1]),
                0.5*(l102[2]+p303[2])];
      let r102=[0.5*(p303[0]+r201[0]),
                0.5*(p303[1]+r201[1]),
                0.5*(p303[2]+r201[2])];

      let l210=[0.5*(px4x[0]+l201[0]),
                0.5*(px4x[1]+l201[1]),
                0.5*(px4x[2]+l201[2])]; // =c120
      let r012=[0.5*(px4x[0]+r102[0]),
                0.5*(px4x[1]+r102[1]),
                0.5*(px4x[2]+r102[2])]; // =c021
      let l300=[0.5*(l201[0]+r102[0]),
                0.5*(l201[1]+r102[1]),
                0.5*(l201[2]+r102[2])]; // =r003=c030

      let r021=[0.5*(pxx4[0]+r120[0]),
                0.5*(pxx4[1]+r120[1]),
                0.5*(pxx4[2]+r120[2])]; // =c012
      let u201=[0.5*(u210[0]+pxx4[0]),
                0.5*(u210[1]+pxx4[1]),
                0.5*(u210[2]+pxx4[2])]; // =c102
      let r030=[0.5*(u210[0]+r120[0]),
                0.5*(u210[1]+r120[1]),
                0.5*(u210[2]+r120[2])]; // =u300=c003

      let u102=[0.5*(u012[0]+p4xx[0]),
                0.5*(u012[1]+p4xx[1]),
                0.5*(u012[2]+p4xx[2])]; // =c201
      let l120=[0.5*(l021[0]+p4xx[0]),
                0.5*(l021[1]+p4xx[1]),
                0.5*(l021[2]+p4xx[2])]; // =c210
      let l030=[0.5*(u012[0]+l021[0]),
                0.5*(u012[1]+l021[1]),
                0.5*(u012[2]+l021[2])]; // =u003=c300

      let l111=[0.5*(p123[0]+l102[0]),
                0.5*(p123[1]+l102[1]),
                0.5*(p123[2]+l102[2])];
      let r111=[0.5*(p312[0]+r210[0]),
                0.5*(p312[1]+r210[1]),
                0.5*(p312[2]+r210[2])];
      let u111=[0.5*(u021[0]+p231[0]),
                0.5*(u021[1]+p231[1]),
                0.5*(u021[2]+p231[2])];
      let c111=[0.25*(p033[0]+p330[0]+p303[0]+p111[0]),
                0.25*(p033[1]+p330[1]+p303[1]+p111[1]),
                0.25*(p033[2]+p330[2]+p303[2]+p111[2])];

      let l=[l003,l102,l012,l201,l111,l021,l300,l210,l120,l030]; // left
      let r=[l300,r102,r012,r201,r111,r021,r300,r210,r120,r030]; // right
      let u=[l030,u102,u012,u201,u111,u021,r030,u210,u120,u030]; // up
      let c=[r030,u201,r021,u102,c111,r012,l030,l120,l210,l300]; // center

      let n0=this.normal(l300,r012,r021,r030,u201,u102,l030);
      let n1=this.normal(r030,u201,u102,l030,l120,l210,l300);
      let n2=this.normal(l030,l120,l210,l300,r012,r021,r030);
      
      let e=this.Epsilon;

      // A kludge to remove subdivision cracks, only applied the first time
      // an edge is found to be flat before the rest of the subpatch is.

      let m0=[0.5*(P1[0]+P2[0]),
              0.5*(P1[1]+P2[1]),
              0.5*(P1[2]+P2[2])];
      if(!flat0) {
        if((flat0=Straightness(r300,p210,p120,u030) < this.res2)) {
          let r=unit(this.sumderivative(c[0],c[2],c[5],c[9],c[1],c[3],c[6]));
          m0=[m0[0]-e*r[0],m0[1]-e*r[1],m0[2]-e*r[2]];
        }
        else m0=r030;
      }


      let m1=[0.5*(P2[0]+P0[0]),
              0.5*(P2[1]+P0[1]),
              0.5*(P2[2]+P0[2])];
      if(!flat1) {
        if((flat1=Straightness(l003,p012,p021,u030) < this.res2)) {
          let r=unit(this.sumderivative(c[6],c[3],c[1],c[0],c[7],c[8],c[9]));
          m1=[m1[0]-e*r[0],m1[1]-e*r[1],m1[2]-e*r[2]];
        }
        else m1=l030;
      }

      let m2=[0.5*(P0[0]+P1[0]),
              0.5*(P0[1]+P1[1]),
              0.5*(P0[2]+P1[2])];
      if(!flat2) {
        if((flat2=Straightness(l003,p102,p201,r300) < this.res2)) {
          let r=unit(this.sumderivative(c[9],c[8],c[7],c[6],c[5],c[2],c[0]));
          m2=[m2[0]-e*r[0],m2[1]-e*r[1],m2[2]-e*r[2]];
        }
        else m2=l300;
      }

      if(C0) {
        let c0=Array(4);
        let c1=Array(4);
        let c2=Array(4);
        for(let i=0; i < 4; ++i) {
          c0[i]=0.5*(C1[i]+C2[i]);
          c1[i]=0.5*(C2[i]+C0[i]);
          c2[i]=0.5*(C0[i]+C1[i]);
        }
        
        let i0=data.Vertex(m0,n0,c0);
        let i1=data.Vertex(m1,n1,c1);
        let i2=data.Vertex(m2,n2,c2);
        
        this.Render3(l,I0,i2,i1,P0,m2,m1,false,flat1,flat2,C0,c2,c1);
        this.Render3(r,i2,I1,i0,m2,P1,m0,flat0,false,flat2,c2,C1,c0);
        this.Render3(u,i1,i0,I2,m1,m0,P2,flat0,flat1,false,c1,c0,C2);
        this.Render3(c,i0,i1,i2,m0,m1,m2,false,false,false,c0,c1,c2);
      } else {
        let i0=this.vertex(m0,n0);
        let i1=this.vertex(m1,n1);
        let i2=this.vertex(m2,n2);
        
        this.Render3(l,I0,i2,i1,P0,m2,m1,false,flat1,flat2);
        this.Render3(r,i2,I1,i0,m2,P1,m0,flat0,false,flat2);
        this.Render3(u,i1,i0,I2,m1,m0,P2,flat0,flat1,false);
        this.Render3(c,i0,i1,i2,m0,m1,m2,false,false,false);
      }
    }
  }

  // Check the flatness of a Bezier patch
  Distance(p) {
    let p0=p[0];
    let p3=p[3];
    let p12=p[12];
    let p15=p[15];

    // Check the flatness of a patch.
    let d=Distance2(p15,p0,this.normal(p3,p[2],p[1],p0,p[4],p[8],p12));
    
    // Determine how straight the edges are.
    d=Math.max(d,Straightness(p0,p[1],p[2],p3));
    d=Math.max(d,Straightness(p0,p[4],p[8],p12));
    d=Math.max(d,Straightness(p3,p[7],p[11],p15));
    d=Math.max(d,Straightness(p12,p[13],p[14],p15));
    
    // Determine how straight the interior control curves are.
    d=Math.max(d,Straightness(p[4],p[5],p[6],p[7]));
    d=Math.max(d,Straightness(p[8],p[9],p[10],p[11]));
    d=Math.max(d,Straightness(p[1],p[5],p[9],p[13]));
    return Math.max(d,Straightness(p[2],p[6],p[10],p[14]));
  }

  // Check the flatness of a Bezier triangle
  Distance3(p) {
    let p0=p[0];
    let p4=p[4];
    let p6=p[6];
    let p9=p[9];

    // Check how far the internal point is from the centroid of the vertices.
    let d=abs2([(p0[0]+p6[0]+p9[0])*third-p4[0],
                (p0[1]+p6[1]+p9[1])*third-p4[1],
                (p0[2]+p6[2]+p9[2])*third-p4[2]]);

    // Determine how straight the edges are.
    d=Math.max(d,Straightness(p0,p[1],p[3],p6));
    d=Math.max(d,Straightness(p0,p[2],p[5],p9));
    return Math.max(d,Straightness(p6,p[7],p[8],p9));
  }

  derivative(p0,p1,p2,p3) {
    let lp=[p1[0]-p0[0],p1[1]-p0[1],p1[2]-p0[2]];
    if(abs2(lp) > this.epsilon)
      return lp;
    
    let lpp=bezierPP(p0,p1,p2);
    if(abs2(lpp) > this.epsilon)
      return lpp;
    
    return bezierPPP(p0,p1,p2,p3);
  }

  sumderivative(p0,p1,p2,p3,p4,p5,p6) {
    let d0=this.derivative(p0,p1,p2,p3);
    let d1=this.derivative(p0,p4,p5,p6);
    return [d0[0]+d1[0],d0[1]+d1[1],d0[2]+d1[2]];
  }
  
  normal(left3,left2,left1,middle,right1,right2,right3) {
    let ux=right1[0]-middle[0];
    let uy=right1[1]-middle[1];
    let uz=right1[2]-middle[2];
    let vx=left1[0]-middle[0];
    let vy=left1[1]-middle[1];
    let vz=left1[2]-middle[2];
    let n=[uy*vz-uz*vy,
           uz*vx-ux*vz,
           ux*vy-uy*vx];
    if(abs2(n) > this.epsilon)
      return unit(n);

    let lp=[vx,vy,vz];
    let rp=[ux,uy,uz];
    let lpp=bezierPP(middle,left1,left2);
    let rpp=bezierPP(middle,right1,right2);
    let a=cross(rpp,lp);
    let b=cross(rp,lpp);
    n=[a[0]+b[0],
       a[1]+b[1],
       a[2]+b[2]];
    if(abs2(n) > this.epsilon)
      return unit(n);

    let lppp=bezierPPP(middle,left1,left2,left3);
    let rppp=bezierPPP(middle,right1,right2,right3);
    a=cross(rpp,lpp);
    b=cross(rp,lppp);
    let c=cross(rppp,lp);
    let d=cross(rppp,lpp);
    let e=cross(rpp,lppp);
    let f=cross(rppp,lppp);
    return unit([9*a[0]+3*(b[0]+c[0]+d[0]+e[0])+f[0],
                 9*a[1]+3*(b[1]+c[1]+d[1]+e[1])+f[1],
                 9*a[2]+3*(b[2]+c[2]+d[2]+e[2])+f[2]]);
  }
}

class BezierCurve {
  constructor(controlpoints,CenterIndex,MaterialIndex,Min,Max) {
    this.controlpoints=controlpoints;
    this.Min=Min;
    this.Max=Max;
    this.CenterIndex=CenterIndex;
    this.MaterialIndex=MaterialIndex;
  }

  // Approximate bounds by bounding box of control polyhedron.
  offscreen(n,v) {
    let x,y,z;
    let X,Y,Z;

    X=x=v[0][0];
    Y=y=v[0][1];
    Z=z=v[0][2];
    
    for(let i=1; i < n; ++i) {
      let V=v[i];
      if(V[0] < x) x=V[0];
      else if(V[0] > X) X=V[0];
      if(V[1] < y) y=V[1];
      else if(V[1] > Y) Y=V[1];
    }

    if(X >= this.x && x <= this.X &&
       Y >= this.y && y <= this.Y)
      return false;

    return this.Offscreen=true;
  }

  OffScreen() {
    centerIndex=this.CenterIndex;
    materialIndex=this.MaterialIndex;

    let b=[viewParam.xmin,viewParam.ymin,viewParam.zmin];
    let B=[viewParam.xmax,viewParam.ymax,viewParam.zmax];

    let s,m,M;

    if(orthographic) {
      m=b;
      M=B;
      s=1.0;
    } else {
      let perspective=1.0/B[2];
      let f=this.Min[2]*perspective;
      let F=this.Max[2]*perspective;
      m=[Math.min(f*b[0],F*b[0]),Math.min(f*b[1],F*b[1]),b[2]];
      M=[Math.max(f*B[0],F*B[0]),Math.max(f*B[1],F*B[1]),B[2]];
      s=Math.max(f,F);
    }

    [this.x,this.y,this.X,this.Y]=new bbox2(m,M).bounds();

    if(centerIndex == 0 &&
       (this.Max[0] < this.x || this.Min[0] > this.X ||
        this.Max[1] < this.y || this.Min[1] > this.Y)) {
      return this.Offscreen=true;
    }

    let res=pixel*Math.hypot(s*(B[0]-b[0]),s*(B[1]-b[1]))/size2;
    this.res2=res*res;
    return this.Offscreen=false;
  }

  render() {
    if(this.OffScreen()) return;

    let p=this.controlpoints;
    
    var i0=data.vertex1(p[0]);
    var i3=data.vertex1(p[3]);
    
    this.Render(p,i0,i3);
    if(data.nvertices > 0) this.append();
  }
  
  append() {
    if(this.Offscreen)
      material1Off.append(data);
    else
      material1On.append(data);
    data.clear();
  }

  Render(p,I0,I1) {
    let p0=p[0];
    let p1=p[1];
    let p2=p[2];
    let p3=p[3];

    if(Straightness(p0,p1,p2,p3) < this.res2) { // Segment is flat
      let P=[p0,p3];
      if(!this.offscreen(2,P)) {
        data.indices.push(I0);
        data.indices.push(I1);
      }
    } else { // Segment is not flat
      if(this.offscreen(4,p)) return;

      let m0=[0.5*(p0[0]+p1[0]),0.5*(p0[1]+p1[1]),0.5*(p0[2]+p1[2])];
      let m1=[0.5*(p1[0]+p2[0]),0.5*(p1[1]+p2[1]),0.5*(p1[2]+p2[2])];
      let m2=[0.5*(p2[0]+p3[0]),0.5*(p2[1]+p3[1]),0.5*(p2[2]+p3[2])];
      let m3=[0.5*(m0[0]+m1[0]),0.5*(m0[1]+m1[1]),0.5*(m0[2]+m1[2])];
      let m4=[0.5*(m1[0]+m2[0]),0.5*(m1[1]+m2[1]),0.5*(m1[2]+m2[2])];
      let m5=[0.5*(m3[0]+m4[0]),0.5*(m3[1]+m4[1]),0.5*(m3[2]+m4[2])];
      
      let s0=[p0,m0,m3,m5];
      let s1=[m5,m4,m2,p3];
      
      let i0=data.vertex1(m5);
      
      this.Render(s0,I0,i0);
      this.Render(s1,i0,I1);
    }
  }
}

function home()
{
  mat4.identity(rotMat);
  initProjection();
  setProjection();
  updatevMatrix();
  redraw=true;
}

function initShader(options)
{
  var fragmentShader=getShader(gl,"fragment",options);
  var vertexShader=getShader(gl,"vertex",options);
  var shader=gl.createProgram();

  gl.attachShader(shader,vertexShader);
  gl.attachShader(shader,fragmentShader);
  gl.linkProgram(shader);
  if (!gl.getProgramParameter(shader,gl.LINK_STATUS)) {
    alert("Could not initialize shaders");
  }

  return shader;
}

class Split3 {
  constructor(z0,c0,c1,z1) {
    this.m0=[0.5*(z0[0]+c0[0]),0.5*(z0[1]+c0[1]),0.5*(z0[2]+c0[2])];
    let m1=[0.5*(c0[0]+c1[0]),0.5*(c0[1]+c1[1]),0.5*(c0[2]+c1[2])];
    this.m2=[0.5*(c1[0]+z1[0]),0.5*(c1[1]+z1[1]),0.5*(c1[2]+z1[2])];
    this.m3=[0.5*(this.m0[0]+m1[0]),0.5*(this.m0[1]+m1[1]),
             0.5*(this.m0[2]+m1[2])];
    this.m4=[0.5*(m1[0]+this.m2[0]),0.5*(m1[1]+this.m2[1]),
             0.5*(m1[2]+this.m2[2])];
    this.m5=[0.5*(this.m3[0]+this.m4[0]),0.5*(this.m3[1]+this.m4[1]),
             0.5*(this.m3[2]+this.m4[2])];
  }
}

function iszero(v)
{
  return v[0] == 0.0 && v[1] == 0.0 && v[2] == 0.0;
}

function unit(v)
{
  var norm=Math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
  norm=(norm != 0.0) ? 1.0/norm : 1.0;
  return [v[0]*norm,v[1]*norm,v[2]*norm];
}

function abs2(v)
{
  return v[0]*v[0]+v[1]*v[1]+v[2]*v[2];
}

function dot(u,v)
{
  return u[0]*v[0]+u[1]*v[1]+u[2]*v[2];
}

function cross(u,v)
{
  return [u[1]*v[2]-u[2]*v[1],
          u[2]*v[0]-u[0]*v[2],
          u[0]*v[1]-u[1]*v[0]];
}

// Return one-sixth of the second derivative of the Bezier curve defined
// by a,b,c,d at 0. 
function bezierPP(a,b,c)
{
  return [a[0]+c[0]-2.0*b[0],
          a[1]+c[1]-2.0*b[1],
          a[2]+c[2]-2.0*b[2]];
}

// Return one-third of the third derivative of the Bezier curve defined by
// a,b,c,d at 0.
function bezierPPP(a,b,c,d)
{
  return [d[0]-a[0]+3.0*(b[0]-c[0]),
          d[1]-a[1]+3.0*(b[1]-c[1]),
          d[2]-a[2]+3.0*(b[2]-c[2])];
}

/**
 * @Return the maximum distance squared of points c0 and c1 from 
 * the respective internal control points of z0--z1.
*/
function Straightness(z0,c0,c1,z1)
{
  var v=[third*(z1[0]-z0[0]),third*(z1[1]-z0[1]),third*(z1[2]-z0[2])];
  return Math.max(abs2([c0[0]-v[0]-z0[0],c0[1]-v[1]-z0[1],c0[2]-v[2]-z0[2]]),
    abs2([z1[0]-v[0]-c1[0],z1[1]-v[1]-c1[1],z1[2]-v[2]-c1[2]]));
}

/**
 * @Return the perpendicular distance squared of a point z from the plane
 * through u with unit normal n.
 */
function Distance2(z,u,n)
{
  var d=dot([z[0]-u[0],z[1]-u[1],z[2]-u[2]],n);
  return d*d;
}

class bbox2 {
  constructor(m,M) {
    var V=[];
    vec3.transformMat4(V,m,T);
    this.x=this.X=V[0];
    this.y=this.Y=V[1];
    this.transformed([m[0],m[1],M[2]]);
    this.transformed([m[0],m[1],M[2]]);
    this.transformed([m[0],M[1],m[2]]);
    this.transformed([m[0],M[1],M[2]]);
    this.transformed([M[0],m[1],m[2]]);
    this.transformed([M[0],m[1],M[2]]);
    this.transformed([M[0],M[1],m[2]]);
    this.transformed(M);
  }

  bounds() {
    return [this.x,this.y,this.X,this.Y];
  }

  transformed(v) {
    var V=[];
    vec3.transformMat4(V,v,T);
    if(V[0] < this.x) this.x=V[0];
    else if(V[0] > this.X) this.X=V[0];
    if(V[1] < this.y) this.y=V[1];
    else if(V[1] > this.Y) this.Y=V[1];
  }
}

/**
 * Perform a change of basis
 * @param {*} out Out Matrix
 * @param {*} conjMatrix Conjugate Matrix
 * @param {*} mat Matrix
 * 
 * @Return the matrix (conjMatrix) * mat * (conjMatrix)^{-1} 
 */
function mat4COB(out,conjMatrix,mat) {
  var cjMatInv=mat4.create();
  mat4.invert(cjMatInv,conjMatrix);

  mat4.multiply(out,mat,cjMatInv);
  mat4.multiply(out,conjMatrix,out);

  return out;
}

function getTargetOrigMat() {
  var translMat=mat4.create();
  mat4.fromTranslation(translMat,[center.x,center.y,center.z])
  return translMat;
}

function COBTarget(out,mat) {
  return mat4COB(out,getTargetOrigMat(),mat);
}

function setUniforms(shader)
{
  gl.useProgram(shader);

  shader.vertexPositionAttribute=
    gl.getAttribLocation(shader,"position");
  gl.enableVertexAttribArray(shader.vertexPositionAttribute);

  if(shader != noNormalShader) {
    shader.vertexNormalAttribute=
      gl.getAttribLocation(shader,"normal");
    gl.enableVertexAttribArray(shader.vertexNormalAttribute);
  }

  shader.vertexMaterialAttribute=
    gl.getAttribLocation(shader,"materialIndex");
  gl.enableVertexAttribArray(shader.vertexMaterialAttribute);

  shader.vertexCenterAttribute=
    gl.getAttribLocation(shader,"centerIndex");
  gl.enableVertexAttribArray(shader.vertexCenterAttribute);

  shader.pvMatrixUniform=gl.getUniformLocation(shader,"projViewMat");
  shader.vmMatrixUniform=gl.getUniformLocation(shader,"viewMat");
  shader.normMatUniform=gl.getUniformLocation(shader,"normMat");

  if(shader == colorShader || shader == transparentShader) {
    shader.vertexColorAttribute=
      gl.getAttribLocation(shader,"color");
    gl.enableVertexAttribArray(shader.vertexColorAttribute);
  }

  for(let i=0; i < M.length; ++i)
    M[i].setUniform(shader,"objMaterial",i);

  for(let i=0; i < lights.length; ++i)
    lights[i].setUniform(shader,"objLights",i);

  for(let i=0; i < Centers.length; ++i)
    gl.uniform3fv(gl.getUniformLocation(shader,"Centers["+i+"]"),Centers[i]);

  mat4.invert(T,vMatrix);
  mat4.multiply(pvmMatrix,pMatrix,vMatrix);
  mat3.fromMat4(vMatrix3,vMatrix);
  mat3.invert(normMat,vMatrix3);

  gl.uniformMatrix4fv(shader.pvMatrixUniform,false,pvmMatrix);
  gl.uniformMatrix4fv(shader.vmMatrixUniform,false,vMatrix);
  gl.uniformMatrix3fv(shader.normMatUniform,false,normMat);
}

function handleMouseDown(event) {
  mouseDownOrTouchActive=true;
  lastMouseX=event.clientX;
  lastMouseY=event.clientY;
}

var pinch=false;
var pinchStart;

function pinchDistance(touches)
{
  return Math.hypot(
    touches[0].pageX-touches[1].pageX,
    touches[0].pageY-touches[1].pageY);
}


var touchStartTime;

function handleTouchStart(evt) {
  evt.preventDefault();
  var touches=evt.targetTouches;
  swipe=rotate=pinch=false;
  if(zooming) return;

  if(touches.length == 1 && !mouseDownOrTouchActive) {
    touchStartTime=new Date().getTime();
    touchId=touches[0].identifier;
    lastMouseX=touches[0].pageX,
    lastMouseY=touches[0].pageY;
  }

  if(touches.length == 2 && !mouseDownOrTouchActive) {
    touchId=touches[0].identifier;
    pinchStart=pinchDistance(touches);
    pinch=true;
  }
}

function handleMouseUpOrTouchEnd(event) {
  mouseDownOrTouchActive=false;
}

function rotateScene(lastX,lastY,rawX,rawY,factor) {
    let [angle,axis]=arcballLib.arcball([lastX,-lastY],[rawX,-rawY]);

    mat4.fromRotation(rotMats,angle*2.0*factor/lastzoom,axis);
    mat4.multiply(rotMat,rotMats,rotMat);
}

function shiftScene(lastX,lastY,rawX,rawY) {
  let zoominv=1/lastzoom;
  shift.x += (rawX-lastX)*zoominv*canvasWidth/2;
  shift.y -= (rawY-lastY)*zoominv*canvasHeight/2;
}

function panScene(lastX,lastY,rawX,rawY) {
  if (orthographic) {
    shiftScene(lastX,lastY,rawX,rawY);
  } else {
    center.x += (rawX-lastX)*(viewParam.xmax-viewParam.xmin);
    center.y -= (rawY-lastY)*(viewParam.ymax-viewParam.ymin);
  }
}

function updatevMatrix() {
  COBTarget(vMatrix,rotMat);
  mat4.translate(vMatrix,vMatrix,[center.x,center.y,0]);
  mat4.invert(T,vMatrix);
}

function capzoom() 
{
  var maxzoom=Math.sqrt(Number.MAX_VALUE);
  var minzoom=1.0/maxzoom;
  if(Zoom <= minzoom) Zoom=minzoom;
  if(Zoom >= maxzoom) Zoom=maxzoom;
  
  if(Zoom != lastzoom) remesh=true;
  lastzoom=Zoom;
}

function zoomImage(diff) {
  let stepPower=zoomStep*halfCanvHeight*diff;
  const limit=Math.log(0.1*Number.MAX_VALUE)/Math.log(zoomFactor);

  if(Math.abs(stepPower) < limit) {
    Zoom *= zoomFactor**stepPower;
    capzoom();
  }
}

/**
 * Mouse Drag Zoom
 * @param {*} lastX unused
 * @param {*} lastY 
 * @param {*} rawX unused
 * @param {*} rawY 
 */
function zoomScene(lastX,lastY,rawX,rawY) {
  zoomImage(lastY-rawY);
}

// mode:
const DRAGMODE_ROTATE=1;
const DRAGMODE_SHIFT=2;
const DRAGMODE_ZOOM=3;
const DRAGMODE_PAN=4
function processDrag(newX,newY,mode,factor=1) {
  let dragFunc;
  switch (mode) {
    case DRAGMODE_ROTATE:
      dragFunc=rotateScene;
      break;
    case DRAGMODE_SHIFT:
      dragFunc=shiftScene;
      break;
    case DRAGMODE_ZOOM:
      dragFunc=zoomScene;
      break;
    case DRAGMODE_PAN:
      dragFunc=panScene;
      break;
    default:
      dragFunc=(_a,_b,_c,_d) => {};
      break;
  }

  let lastX=(lastMouseX-halfCanvWidth)/halfCanvWidth;
  let lastY=(lastMouseY-halfCanvHeight)/halfCanvHeight;
  let rawX=(newX-halfCanvWidth)/halfCanvWidth;
  let rawY=(newY-halfCanvHeight)/halfCanvHeight;

  dragFunc(lastX,lastY,rawX,rawY,factor);

  lastMouseX=newX;
  lastMouseY=newY;

  setProjection();
  updatevMatrix();
  redraw=true;
}

function handleKey(event) {
  var keycode=event.key;
  var axis=[];
  switch(keycode) {
  case "x":
    axis=[1,0,0];
    break;
  case "y":
    axis=[0,1,0];
    break;
  case "z":
    axis=[0,0,1];
    break;
  case "h":
    home();
    break;
  default:
    break;
  }

  if(axis.length > 0) {
    mat4.rotate(rotMat,rotMat,0.1,axis);
    updatevMatrix();
    redraw=true;
  }
}

function handleMouseWheel(event) {
  if (event.deltaY < 0.0) {
    Zoom *= zoomFactor;
  } else {
    Zoom /= zoomFactor;
  }
  capzoom();
  setProjection();
  updatevMatrix();

  redraw=true;
}

function handleMouseMove(event) {
  if(!mouseDownOrTouchActive) {
    return;
  }

  var newX=event.clientX;
  var newY=event.clientY;

  let mode;
  if(event.getModifierState("Control")) {
    mode=DRAGMODE_SHIFT;
  } else if(event.getModifierState("Shift")) {
    mode=DRAGMODE_ZOOM;
  } else if(event.getModifierState("Alt")) {
    mode=DRAGMODE_PAN;
  } else {
    mode=DRAGMODE_ROTATE;
  }

  processDrag(newX,newY,mode);
}

var zooming=false;
var swipe=false;
var rotate=false;

function handleTouchMove(evt) {
  evt.preventDefault();
  if(zooming) return;
  var touches=evt.targetTouches;

  if(!pinch && touches.length == 1 && touchId == touches[0].identifier) {
    var newX=touches[0].pageX;
    var newY=touches[0].pageY;
    var dx=newX-lastMouseX;
    var dy=newY-lastMouseY;
    var stationary=dx*dx+dy*dy <= shiftHoldDistance*shiftHoldDistance;
    if(stationary) {
      if(!swipe && !rotate &&
         new Date().getTime()-touchStartTime > shiftWaitTime) {
        if(navigator.vibrate)
          window.navigator.vibrate(vibrateTime);
        swipe=true;
      }
    }
    if(swipe)
      processDrag(newX,newY,DRAGMODE_SHIFT);
    else if(!stationary) {
      rotate=true;
      var newX=touches[0].pageX;
      var newY=touches[0].pageY;
      processDrag(newX,newY,DRAGMODE_ROTATE,0.5);
    }
  }

  if(pinch && !swipe &&
     touches.length == 2 && touchId == touches[0].identifier) {
    let distance=pinchDistance(touches);
    let diff=distance-pinchStart;
    zooming=true;
    diff *= zoomPinchFactor;
    if(diff > zoomPinchCap) diff=zoomPinchCap;
    if(diff < -zoomPinchCap) diff=-zoomPinchCap;
    zoomImage(diff/size2);
    pinchStart=distance;
    swipe=rotate=zooming=false;
    setProjection();
    updatevMatrix();
    redraw=true;
  }
}

var indexExt;

// Create buffers for the patch and its subdivisions.
function setBuffer()
{
  positionBuffer=gl.createBuffer();
  materialBuffer=gl.createBuffer();
  colorBuffer=gl.createBuffer();
  indexBuffer=gl.createBuffer();
  indexExt=gl.getExtension("OES_element_index_uint");
}

function transformVertices(vertices)
{
  var Tz0=vMatrix[2];
  var Tz1=vMatrix[6];
  var Tz2=vMatrix[10];
  zbuffer.length=vertices.length;
  for(var i=0; i < vertices.length; ++i) {
    var i6=6*i;
    zbuffer[i]=Tz0*vertices[i6]+Tz1*vertices[i6+1]+Tz2*vertices[i6+2];
  }
}

var zbuffer=[];

function draw()
{
  material1Off.clear();
  materialOff.clear();
  colorOff.clear();
  transparentOff.clear();

  if(remesh) {
    material1On.clear();
    materialOn.clear();
    colorOn.clear();
    transparentOn.clear();
  }

  P.forEach(function(p) {
    if(remesh || p.Offscreen)
      p.render();
  });

  if(material1On.indices.length > 0)
    drawBuffer(material1On,noNormalShader);
  
  if(material1Off.indices.length > 0)
    drawBuffer(material1Off,noNormalShader);

  if(materialOn.indices.length > 0)
    drawBuffer(materialOn,materialShader);
  
  if(materialOff.indices.length > 0)
    drawBuffer(materialOff,materialShader);

  if(colorOn.indices.length > 0)
    drawBuffer(colorOn,colorShader);

  if(colorOff.indices.length > 0)
    drawBuffer(colorOff,colorShader);

  data.copy(transparentOn);
  data.append(transparentOff);
  let indices=data.indices;
  if(indices.length > 0) {
    transformVertices(data.vertices);
    
    let n=indices.length/3;
    let triangles=Array(n).fill().map((_,i)=>i);

    triangles.sort(function(a,b) {
      let a3=3*a;
      Ia=indices[a3];
      Ib=indices[a3+1];
      Ic=indices[a3+2];

      let b3=3*b;
      IA=indices[b3];
      IB=indices[b3+1];
      IC=indices[b3+2];

      return zbuffer[Ia]+zbuffer[Ib]+zbuffer[Ic] < 
        zbuffer[IA]+zbuffer[IB]+zbuffer[IC] ? -1 : 1;
    });

    let Indices=Array(indices.length);

    for(let i=0; i < n; ++i) {
      let i3=3*i;
      let t=3*triangles[i];
      Indices[3*i]=indices[t];
      Indices[3*i+1]=indices[t+1];
      Indices[3*i+2]=indices[t+2];
    }

    drawBuffer(data,transparentShader,Indices);
    data.clear();
  }

  remesh=false;
}

function tick() {
  requestAnimationFrame(tick);
  if(redraw) {
    draw();
    redraw=false;
  }
}

function setDimensions(width=canvasWidth,height=canvasHeight,X=0,Y=0) {
  let Aspect=width/height;
  let zoominv=1.0/lastzoom;
  let xshift=X/width*lastzoom
  let yshift=Y/height*lastzoom

  if (orthographic) {
    let xsize=B[0]-b[0];
    let ysize=B[1]-b[1];
    if (xsize < ysize*Aspect) {
      let r=0.5*ysize*Aspect*zoominv;
      let X0=2*r*xshift;
      let Y0=ysize*zoominv*yshift;
      viewParam.xmin=-r-X0;
      viewParam.xmax=r-X0;
      viewParam.ymin=b[1]*zoominv-Y0;
      viewParam.ymax=B[1]*zoominv-Y0;
    } else {
      let r=0.5*xsize/(Aspect*Zoom);
      let X0=xsize*zoominv*xshift;
      let Y0=2*r*yshift;
      viewParam.xmin=b[0]*zoominv-X0;
      viewParam.xmax=B[0]*zoominv-X0;
      viewParam.ymin=-r-Y0;
      viewParam.ymax=r-Y0;
    }
  } else {
      let r=H*zoominv;
      let rAspect=r*Aspect;
      let X0=2*rAspect*xshift;
      let Y0=2*r*yshift;
      viewParam.xmin=-rAspect-X0;
      viewParam.xmax=rAspect-X0;
      viewParam.ymin=-r-Y0;
      viewParam.ymax=r-Y0;
  }
}

function setProjection() {
  setDimensions(canvasWidth,canvasHeight,shift.x,shift.y);
  let f=orthographic ? mat4.ortho : mat4.frustum;
  f(pMatrix,viewParam.xmin,viewParam.xmax,
    viewParam.ymin,viewParam.ymax,
    -viewParam.zmax,-viewParam.zmin);
}

function initProjection() {
  H=-Math.tan(0.5*angle)*B[2];

  center={x:0,y:0,z:0.5*(b[2]+B[2])};
  lastzoom=Zoom=Zoom0;

  viewParam={
    xmin:b[0],xmax:B[0],
    ymin:b[1],ymax:B[1],
    zmin:b[2],zmax:B[2]
  };
  shift={
    x:0,y:0
  };
  setProjection();
}

var materialShader,colorShader,noNormalShader,transparentShader;

function webGLStart()
{
  var canvas=document.getElementById("Asymptote");

  canvas.width=canvasWidth;
  canvas.height=canvasHeight;

  halfCanvWidth=canvasWidth/2.0;
  halfCanvHeight=canvasHeight/2.0;

  initProjection();
  initGL(canvas);

  gl.clearColor(1.0,1.0,1.0,1.0);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);
  gl.enable(gl.DEPTH_TEST);
  gl.viewport(0,0,gl.viewportWidth,gl.viewportHeight);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  noNormalShader=initShader();
  materialShader=initShader(["NORMAL"]);
  colorShader=initShader(["NORMAL","COLOR"]);
  transparentShader=initShader(["NORMAL","TRANSPARENT"]);

  setBuffer();

  canvas.onmousedown=handleMouseDown;
  document.onmouseup=handleMouseUpOrTouchEnd;
  document.onmousemove=handleMouseMove;
  canvas.onkeydown=handleKey;
  document.onwheel=handleMouseWheel;

  canvas.addEventListener("touchstart",handleTouchStart,false);
  canvas.addEventListener("touchend",handleMouseUpOrTouchEnd,false);
  canvas.addEventListener("touchcancel",handleMouseUpOrTouchEnd,false);
  canvas.addEventListener("touchleave",handleMouseUpOrTouchEnd,false);
  canvas.addEventListener("touchmove",handleTouchMove,false);
  document.addEventListener("keydown",handleKey,false);

  tick();
}
