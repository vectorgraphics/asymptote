let P=[]; // Array of Bezier patches, triangles, curves, and pixels
let Materials=[]; // Array of materials
let Lights=[]; // Array of lights
let Centers=[]; // Array of billboard centers
let Background=[1,1,1,1]; // Background color

let canvasWidth,canvasHeight;

let absolute=false;

let b,B; // Scene min,max bounding box corners (3-tuples)
let angle; // Field of view angle
let Zoom0; // Initial zoom
let viewportmargin; // Margin around viewport (2-tuple)
let viewportshift=[0,0]; // Viewport shift (for perspective projection)

let zoomFactor;
let zoomPinchFactor;
let zoomPinchCap;
let zoomStep;

let shiftHoldDistance;
let shiftWaitTime;
let vibrateTime;

let embedded; // Is image embedded within another window?

let canvas; // Rendering canvas
let gl; // WebGL rendering context
let alpha; // Is background opaque?

let offscreen; // Offscreen rendering canvas for embedded images
let context; // 2D context for copying embedded offscreen images

let nlights=0; // Number of lights compiled in shader
let Nmaterials=2; // Maximum number of materials compiled in shader

let materials=[]; // Subset of Materials passed as uniforms
let maxMaterials; // Limit on number of materials allowed in shader

let halfCanvasWidth,halfCanvasHeight;

let pixel=0.75; // Adaptive rendering constant.
let FillFactor=0.1;
let Zoom;

let maxViewportWidth=window.innerWidth;
let maxViewportHeight=window.innerHeight;

const windowTrim=10;
let resizeStep=1.2;

let lastzoom;
let H; // maximum camera view half-height

let third=1/3;

let rotMat=mat4.create();
let projMat=mat4.create(); // projection matrix
let viewMat=mat4.create(); // view matrix

let projViewMat=mat4.create(); // projection view matrix
let normMat=mat3.create();
let viewMat3=mat3.create(); // 3x3 view matrix
let cjMatInv=mat4.create();
let T=mat4.create(); // Temporary matrix

let zmin,zmax;
let center={x:0,y:0,z:0};
let size2;
let ArcballFactor;
let shift={
  x:0,y:0
};

let viewParam = {
  xmin:0,xmax:0,
  ymin:0,ymax:0,
  zmin:0,zmax:0
};

let positionBuffer;
let materialBuffer;
let colorBuffer;
let indexBuffer;

let remesh=true;
let wireframe=0;
let mouseDownOrTouchActive=false;
let lastMouseX=null;
let lastMouseY=null;
let touchID=null;

// Indexed triangles:
let Positions=[];
let Normals=[];
let Colors=[];
let Indices=[];

class Material {
  constructor(diffuse,emissive,specular,shininess,metallic,fresnel0) {
    this.diffuse=diffuse;
    this.emissive=emissive;
    this.specular=specular;
    this.shininess=shininess;
    this.metallic=metallic;
    this.fresnel0=fresnel0;
  }

  setUniform(program,index) {
    let getLoc=
        param => gl.getUniformLocation(program,"Materials["+index+"]."+param);

    gl.uniform4fv(getLoc("diffuse"),new Float32Array(this.diffuse));
    gl.uniform4fv(getLoc("emissive"),new Float32Array(this.emissive));
    gl.uniform4fv(getLoc("specular"),new Float32Array(this.specular));

    gl.uniform4f(getLoc("parameters"),this.shininess,this.metallic,
                 this.fresnel0,0);
  }
}

let enumPointLight=1;
let enumDirectionalLight=2;

class Light {
  constructor(direction,color) {
    this.direction=direction;
    this.color=color;
  }

  setUniform(program,index) {
    let getLoc=
        param => gl.getUniformLocation(program,"Lights["+index+"]."+param);

    gl.uniform3fv(getLoc("direction"),new Float32Array(this.direction));
    gl.uniform3fv(getLoc("color"),new Float32Array(this.color));
  }
}

function initShaders()
{
  let maxUniforms=gl.getParameter(gl.MAX_VERTEX_UNIFORM_VECTORS);
  maxMaterials=Math.floor((maxUniforms-14)/4);
  Nmaterials=Math.min(Math.max(Nmaterials,Materials.length),maxMaterials);

  pixelShader=initShader(["WIDTH"]);
  materialShader=initShader(["NORMAL"]);
  colorShader=initShader(["NORMAL","COLOR"]);
  transparentShader=initShader(["NORMAL","COLOR","TRANSPARENT"]);
}

function deleteShaders()
{
  gl.deleteProgram(transparentShader);
  gl.deleteProgram(colorShader);
  gl.deleteProgram(materialShader);
  gl.deleteProgram(pixelShader);
}

// Create buffers for the patch and its subdivisions.
function setBuffers()
{
  positionBuffer=gl.createBuffer();
  materialBuffer=gl.createBuffer();
  colorBuffer=gl.createBuffer();
  indexBuffer=gl.createBuffer();
}

function noGL() {
  if (!gl)
    alert("Could not initialize WebGL");
}

function saveAttributes()
{
  let a=window.top.document.asygl[alpha];

  a.gl=gl;
  a.nlights=Lights.length;
  a.Nmaterials=Nmaterials;
  a.maxMaterials=maxMaterials;

  a.pixelShader=pixelShader;
  a.materialShader=materialShader;
  a.colorShader=colorShader;
  a.transparentShader=transparentShader;
}

function restoreAttributes()
{
  let a=window.top.document.asygl[alpha];

  gl=a.gl;
  nlights=a.nlights;
  Nmaterials=a.Nmaterials;
  maxMaterials=a.maxMaterials;

  pixelShader=a.pixelShader;
  materialShader=a.materialShader;
  colorShader=a.colorShader;
  transparentShader=a.transparentShader;
}

let indexExt;

function initGL()
{
  alpha=Background[3] < 1;

  if(embedded) {
    let p=window.top.document;

    if(p.asygl == null)
      p.asygl=Array(2);
  
    context=canvas.getContext("2d");
    offscreen=p.offscreen;
    if(!offscreen) {
      offscreen=p.createElement("canvas");
      p.offscreen=offscreen;
    }

    if(!p.asygl[alpha] || !p.asygl[alpha].gl) {
      gl=offscreen.getContext("webgl",{alpha:alpha});
      if(!gl) noGL();
      initShaders();
      p.asygl[alpha]={};
      saveAttributes();
    } else {
      restoreAttributes();
      if((Lights.length != nlights) ||
         Math.min(Materials.length,maxMaterials) > Nmaterials) {
        initShaders();
        saveAttributes();
      }
    }
  } else {
    gl=canvas.getContext("webgl",{alpha:alpha});
    if(!gl) noGL();
    initShaders();
  }

  setBuffers();
  indexExt=gl.getExtension("OES_element_index_uint");

  TRIANGLES=gl.TRIANGLES;
  material0Data=new vertexBuffer(gl.POINTS);
  material1Data=new vertexBuffer(gl.LINES);
  materialData=new vertexBuffer();
  colorData=new vertexBuffer();
  transparentData=new vertexBuffer();
  triangleData=new vertexBuffer();
}

function getShader(gl,shaderScript,type,options=[])
{
  let str=`#version 100
#ifdef GL_FRAGMENT_PRECISION_HIGH
  precision highp float;
#else
  precision mediump float;
#endif
  #define nlights ${wireframe == 0 ? Lights.length : 0}\n
  const int Nlights=${Math.max(Lights.length,1)};\n
  #define Nmaterials ${Nmaterials}\n`;

  if(orthographic)
    str += `#define ORTHOGRAPHIC\n`;

  options.forEach(s => str += `#define `+s+`\n`);

  let shader=gl.createShader(type);
  gl.shaderSource(shader,str+shaderScript);
  gl.compileShader(shader);
  if(!gl.getShaderParameter(shader,gl.COMPILE_STATUS)) {
    alert(gl.getShaderInfoLog(shader));
    return null;
  }
  return shader;
}

function drawBuffer(data,shader,indices=data.indices)
{
  if(data.indices.length == 0) return;
  
  let normal=shader != pixelShader;

  setUniforms(data,shader);

  gl.bindBuffer(gl.ARRAY_BUFFER,positionBuffer);
  gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(data.vertices),
                gl.STATIC_DRAW);
  gl.vertexAttribPointer(positionAttribute,3,gl.FLOAT,false,
                         normal ? 24 : 16,0);
  if(normal && Lights.length > 0)
    gl.vertexAttribPointer(normalAttribute,3,gl.FLOAT,false,24,12);
  else if(pixel)
    gl.vertexAttribPointer(widthAttribute,1,gl.FLOAT,false,16,12);

  gl.bindBuffer(gl.ARRAY_BUFFER,materialBuffer);
  gl.bufferData(gl.ARRAY_BUFFER,new Int16Array(data.materialIndices),
                gl.STATIC_DRAW);
  gl.vertexAttribPointer(materialAttribute,1,gl.SHORT,false,2,0);

  if(shader == colorShader || shader == transparentShader) {
    gl.bindBuffer(gl.ARRAY_BUFFER,colorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER,new Uint8Array(data.colors),
                  gl.STATIC_DRAW);
    gl.vertexAttribPointer(colorAttribute,4,gl.UNSIGNED_BYTE,true,0,0);
  }

  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,indexBuffer);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,
                indexExt ? new Uint32Array(indices) :
                new Uint16Array(indices),gl.STATIC_DRAW);

  gl.drawElements(normal ? (wireframe ? gl.LINES : data.type) : gl.POINTS,
                  indices.length,
                  indexExt ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT,0);
}

let TRIANGLES;

class vertexBuffer {
  constructor(type) {
    this.type=type ? type : TRIANGLES;
    this.clear();
  }
  clear() {
    this.vertices=[];
    this.materialIndices=[];
    this.colors=[];
    this.indices=[];
    this.nvertices=0;

    this.materials=[];
    this.materialTable=[];
  }

  // material vertex 
  vertex(v,n) {
    this.vertices.push(v[0]);
    this.vertices.push(v[1]);
    this.vertices.push(v[2]);
    this.vertices.push(n[0]);
    this.vertices.push(n[1]);
    this.vertices.push(n[2]);
    this.materialIndices.push(materialIndex);
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
    this.materialIndices.push(materialIndex);
    this.colors.push(c[0]);
    this.colors.push(c[1]);
    this.colors.push(c[2]);
    this.colors.push(c[3]);
    return this.nvertices++;
  }

  // material vertex with width and without normal 
  vertex0(v,width) {
    this.vertices.push(v[0]);
    this.vertices.push(v[1]);
    this.vertices.push(v[2]);
    this.vertices.push(width);
    this.materialIndices.push(materialIndex);
    return this.nvertices++;
  }

  // indexed colored vertex 
  iVertex(i,v,n,c=[0,0,0,0]) {
    let i6=6*i;
    this.vertices[i6]=v[0];
    this.vertices[i6+1]=v[1];
    this.vertices[i6+2]=v[2];
    this.vertices[i6+3]=n[0];
    this.vertices[i6+4]=n[1];
    this.vertices[i6+5]=n[2];
    this.materialIndices[i]=materialIndex;
    let i4=4*i;
    this.colors[i4]=c[0];
    this.colors[i4+1]=c[1];
    this.colors[i4+2]=c[2];
    this.colors[i4+3]=c[3];
    this.indices.push(i);
  }

  append(data) {
    append(this.vertices,data.vertices);
    append(this.materialIndices,data.materialIndices);
    append(this.colors,data.colors);
    appendOffset(this.indices,data.indices,this.nvertices);
    this.nvertices += data.nvertices;
  }
}

let material0Data;    // pixels
let material1Data;    // material Bezier curves
let materialData;     // material Bezier patches & triangles
let colorData;        // colored Bezier patches & triangles
let transparentData;  // transparent patches & triangles
let triangleData;     // opaque indexed triangles

let materialIndex;

// efficiently append array b onto array a
function append(a,b)
{
  let n=a.length;
  let m=b.length;
  a.length += m;
  for(let i=0; i < m; ++i)
    a[n+i]=b[i];
}

// efficiently append array b onto array a
function appendOffset(a,b,o)
{
  let n=a.length;
  let m=b.length;
  a.length += b.length;
  for(let i=0; i < m; ++i)
    a[n+i]=b[i]+o;
}

class Geometry {
  constructor() {
    this.data=new vertexBuffer();
    this.Onscreen=false;
    this.m=[];
  }

  // Is 2D bounding box formed by projecting 3d points in vector v offscreen?
  offscreen(v) {
    let m=projViewMat;
    let v0=v[0];
    let x=v0[0], y=v0[1], z=v0[2];
    let f=1/(m[3]*x+m[7]*y+m[11]*z+m[15]);
    this.x=this.X=(m[0]*x+m[4]*y+m[8]*z+m[12])*f;
    this.y=this.Y=(m[1]*x+m[5]*y+m[9]*z+m[13])*f;
    for(let i=1, n=v.length; i < n; ++i) {
      let vi=v[i];
      let x=vi[0], y=vi[1], z=vi[2];
      let f=1/(m[3]*x+m[7]*y+m[11]*z+m[15]);
      let X=(m[0]*x+m[4]*y+m[8]*z+m[12])*f;
      let Y=(m[1]*x+m[5]*y+m[9]*z+m[13])*f;
      if(X < this.x) this.x=X;
      else if(X > this.X) this.X=X;
      if(Y < this.y) this.y=Y;
      else if(Y > this.Y) this.Y=Y;
    }
    let eps=1e-2;
    let min=-1-eps;
    let max=1+eps;
    if(this.X < min || this.x > max || this.Y < min || this.y > max) {
      this.Onscreen=false;
      return true;
    }
    return false;
  }

  T(v) {
    let c0=this.c[0];
    let c1=this.c[1];
    let c2=this.c[2];
    let x=v[0]-c0;
    let y=v[1]-c1;
    let z=v[2]-c2;
    return [x*normMat[0]+y*normMat[3]+z*normMat[6]+c0,
            x*normMat[1]+y*normMat[4]+z*normMat[7]+c1,
            x*normMat[2]+y*normMat[5]+z*normMat[8]+c2];
  }

  Tcorners(m,M) {
    return [this.T(m),this.T([m[0],m[1],M[2]]),this.T([m[0],M[1],m[2]]),
            this.T([m[0],M[1],M[2]]),this.T([M[0],m[1],m[2]]),
            this.T([M[0],m[1],M[2]]),this.T([M[0],M[1],m[2]]),this.T(M)];
  }

  setMaterial(data,draw) {
    if(data.materialTable[this.MaterialIndex] == null) {
      if(data.materials.length >= Nmaterials)
        draw();
      data.materialTable[this.MaterialIndex]=data.materials.length;
      data.materials.push(Materials[this.MaterialIndex]);
    }
    materialIndex=data.materialTable[this.MaterialIndex];
  }

  render() {
    this.setMaterialIndex();

    // First check if re-rendering is required
    let v;
    if(this.CenterIndex == 0)
      v=corners(this.Min,this.Max);
    else {
      this.c=Centers[this.CenterIndex-1];
      v=this.Tcorners(this.Min,this.Max);
    }

    if(this.offscreen(v)) { // Fully offscreen
      this.data.clear();
      return;
    }

    let p=this.controlpoints;
    let P;

    if(this.CenterIndex == 0) {
      if(!remesh && this.Onscreen) {
        // Fully onscreen; no need to re-render
        this.append();
        return;
      }
      P=p;
    } else { // Transform billboard labels
      let n=p.length;
      P=Array(n);
      for(let i=0; i < n; ++i)
        P[i]=this.T(p[i]);
    }

    let s=orthographic ? 1 : this.Min[2]/B[2];
    let res=pixel*Math.hypot(s*(viewParam.xmax-viewParam.xmin),
                             s*(viewParam.ymax-viewParam.ymin))/size2;
    this.res2=res*res;
    this.Epsilon=FillFactor*res;
    
    this.data.clear();
    this.Onscreen=true;
    this.process(P);
  }
}

class BezierPatch extends Geometry {
  /**
   * Constructor for Bezier Patch
   * @param {*} controlpoints array of 16 control points
   * @param {*} CenterIndex center index of billboard labels (or 0)
   * @param {*} MaterialIndex material index (>= 0)
   * @param {*} Min bounding box corner
   * @param {*} Max bounding box corner
   * @param {*} colors array of 4 RGBA color arrays
   */
  constructor(controlpoints,CenterIndex,MaterialIndex,Min,Max,color) {
    super();
    this.controlpoints=controlpoints;
    this.Min=Min;
    this.Max=Max;
    this.color=color;
    this.CenterIndex=CenterIndex;
    let n=controlpoints.length;
    if(color) {
      let sum=color[0][3]+color[1][3]+color[2][3];
      this.transparent=(n == 16 || n == 4) ?
                        sum+color[3][3] < 1020 : sum < 765;
    } else
      this.transparent=Materials[MaterialIndex].diffuse[3] < 1;
    this.MaterialIndex=MaterialIndex;

    this.vertex=this.transparent ? this.data.Vertex.bind(this.data) :
      this.data.vertex.bind(this.data);
    this.L2norm(this.controlpoints);
  }

  setMaterialIndex() {
    if(this.transparent)
      this.setMaterial(transparentData,drawTransparent);
    else {
      if(this.color)
        this.setMaterial(colorData,drawColor);
      else
        this.setMaterial(materialData,drawMaterial);
    }
  }

// Render a Bezier patch via subdivision.
  L2norm(p) {
    let p0=p[0];
    this.epsilon=0;
    let n=p.length;
    for(let i=1; i < n; ++i)
      this.epsilon=Math.max(this.epsilon,
        abs2([p[i][0]-p0[0],p[i][1]-p0[1],p[i][2]-p0[2]]));
    this.epsilon *= Number.EPSILON
  }

  processTriangle(p) {
    let p0=p[0];
    let p1=p[1];
    let p2=p[2];
    let n=unit(cross([p1[0]-p0[0],p1[1]-p0[1],p1[2]-p0[2]],
                     [p2[0]-p0[0],p2[1]-p0[1],p2[2]-p0[2]]));
    if(!this.offscreen([p0,p1,p2])) {
      let i0,i1,i2;
      if(this.color) {
        i0=this.data.Vertex(p0,n,this.color[0]);
        i1=this.data.Vertex(p1,n,this.color[1]);
        i2=this.data.Vertex(p2,n,this.color[2]);
      } else {
        i0=this.vertex(p0,n);
        i1=this.vertex(p1,n);
        i2=this.vertex(p2,n);
      }

      if(wireframe == 0) {
        this.data.indices.push(i0);
        this.data.indices.push(i1);
        this.data.indices.push(i2);
      } else {
        this.data.indices.push(i0);
        this.data.indices.push(i1);
        this.data.indices.push(i1);
        this.data.indices.push(i2);
        this.data.indices.push(i2);
        this.data.indices.push(i0);
      }

      this.append();
    }
  }

  processQuad(p) {
    let p0=p[0];
    let p1=p[1];
    let p2=p[2];
    let p3=p[3];
    let n1=cross([p1[0]-p0[0],p1[1]-p0[1],p1[2]-p0[2]],
                 [p2[0]-p1[0],p2[1]-p1[1],p2[2]-p1[2]]);
    let n2=cross([p2[0]-p3[0],p2[1]-p3[1],p2[2]-p3[2]],
                 [p3[0]-p0[0],p3[1]-p0[1],p3[2]-p0[2]]);
    let n=unit([n1[0]+n2[0],n1[1]+n2[1],n1[2]+n2[2]]);
    if(!this.offscreen([p0,p1,p2,p3])) {
      let i0,i1,i2,i3;
      if(this.color) {
        i0=this.data.Vertex(p0,n,this.color[0]);
        i1=this.data.Vertex(p1,n,this.color[1]);
        i2=this.data.Vertex(p2,n,this.color[2]);
        i3=this.data.Vertex(p3,n,this.color[3]);
      } else {
        i0=this.vertex(p0,n);
        i1=this.vertex(p1,n);
        i2=this.vertex(p2,n);
        i3=this.vertex(p3,n);
      }

      if(wireframe == 0) {
        this.data.indices.push(i0);
        this.data.indices.push(i1);
        this.data.indices.push(i2);

        this.data.indices.push(i0);
        this.data.indices.push(i2);
        this.data.indices.push(i3);
      } else {
        this.data.indices.push(i0);
        this.data.indices.push(i1);
        this.data.indices.push(i1);
        this.data.indices.push(i2);
        this.data.indices.push(i2);
        this.data.indices.push(i3);
        this.data.indices.push(i3);
        this.data.indices.push(i0);
      }

      this.append();
    }
  }

  curve(p,a,b,c,d) {
    new BezierCurve([p[a],p[b],p[c],p[d]],0,materialIndex,
                    this.Min,this.Max).render();
  }

  process(p) {
    if(this.transparent && wireframe != 1)
      // Override materialIndex to encode color vs material
      materialIndex=this.color ? -1-materialIndex : 1+materialIndex;

    if(p.length == 10) return this.process3(p);
    if(p.length == 3) return this.processTriangle(p);
    if(p.length == 4) return this.processQuad(p);
    
    if(wireframe == 1) {
      this.curve(p,0,4,8,12);
      this.curve(p,12,13,14,15);
      this.curve(p,15,11,7,3);
      this.curve(p,3,2,1,0);
      return;
    }

    let p0=p[0];
    let p3=p[3];
    let p12=p[12];
    let p15=p[15];

    let n0=this.normal(p3,p[2],p[1],p0,p[4],p[8],p12);
    if(abs2(n0) < this.epsilon) {
      n0=this.normal(p3,p[2],p[1],p0,p[13],p[14],p15);
      if(abs2(n0) < this.epsilon)
        n0=this.normal(p15,p[11],p[7],p3,p[4],p[8],p12);
    }

    let n1=this.normal(p0,p[4],p[8],p12,p[13],p[14],p15);
    if(abs2(n1) < this.epsilon) {
      n1=this.normal(p0,p[4],p[8],p12,p[11],p[7],p3);
      if(abs2(n1) < this.epsilon)
        n1=this.normal(p3,p[2],p[1],p0,p[13],p[14],p15);
    }

    let n2=this.normal(p12,p[13],p[14],p15,p[11],p[7],p3);
    if(abs2(n2) < this.epsilon) {
      n2=this.normal(p12,p[13],p[14],p15,p[2],p[1],p0);
      if(abs2(n2) < this.epsilon)
        n2=this.normal(p0,p[4],p[8],p12,p[11],p[7],p3);
    }

    let n3=this.normal(p15,p[11],p[7],p3,p[2],p[1],p0);
    if(abs2(n3) < this.epsilon) {
      n3=this.normal(p15,p[11],p[7],p3,p[4],p[8],p12);
      if(abs2(n3) < this.epsilon)
        n3=this.normal(p12,p[13],p[14],p15,p[2],p[1],p0);
    }

    if(this.color) {
      let c0=this.color[0];
      let c1=this.color[1];
      let c2=this.color[2];
      let c3=this.color[3];

      let i0=this.data.Vertex(p0,n0,c0);
      let i1=this.data.Vertex(p12,n1,c1);
      let i2=this.data.Vertex(p15,n2,c2);
      let i3=this.data.Vertex(p3,n3,c3);

      this.Render(p,i0,i1,i2,i3,p0,p12,p15,p3,false,false,false,false,
                  c0,c1,c2,c3);
    } else {
      let i0=this.vertex(p0,n0);
      let i1=this.vertex(p12,n1);
      let i2=this.vertex(p15,n2);
      let i3=this.vertex(p3,n3);

      this.Render(p,i0,i1,i2,i3,p0,p12,p15,p3,false,false,false,false);
    }
    if(this.data.indices.length > 0) this.append();
  }

  append() {
    if(this.transparent)
      transparentData.append(this.data);
    else if(this.color)
      colorData.append(this.data);
    else
      materialData.append(this.data);
  }

  Render(p,I0,I1,I2,I3,P0,P1,P2,P3,flat0,flat1,flat2,flat3,C0,C1,C2,C3) {
    let d=this.Distance(p);
    if(d[0] < this.res2 && d[1] < this.res2) { // Bezier patch is flat
      if(!this.offscreen([P0,P1,P2])) {
        if(wireframe == 0) {
          this.data.indices.push(I0);
          this.data.indices.push(I1);
          this.data.indices.push(I2);
        } else {
          this.data.indices.push(I0);
          this.data.indices.push(I1);
          this.data.indices.push(I1);
          this.data.indices.push(I2);
        }
      }        
      if(!this.offscreen([P0,P2,P3])) {
        if(wireframe == 0) {
          this.data.indices.push(I0);
          this.data.indices.push(I2);
          this.data.indices.push(I3);
        } else {
          this.data.indices.push(I2);
          this.data.indices.push(I3);
          this.data.indices.push(I3);
          this.data.indices.push(I0);
        }
      }
    } else {
  // Approximate bounds by bounding box of control polyhedron.
      if(this.offscreen(p)) return;

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

      */
         
      let p0=p[0];
      let p3=p[3];
      let p12=p[12];
      let p15=p[15];

      if(d[0] < this.res2) { // flat in horizontal direction; split vertically
        /*
       P refers to a corner
       m refers to a midpoint
       s refers to a subpatch

       +--------+--------+
       |P3             P2|
       |                 |
       |       s1        |
       |                 |
       |                 |
    m1 +-----------------+ m0
       |                 |
       |                 |
       |       s0        |
       |                 |
       |P0             P1|
       +-----------------+

        */

        let c0=new Split3(p0,p[1],p[2],p3);
        let c1=new Split3(p[4],p[5],p[6],p[7]);
        let c2=new Split3(p[8],p[9],p[10],p[11]);
        let c3=new Split3(p12,p[13],p[14],p15);

        let s0=[p0  ,c0.m0,c0.m3,c0.m5,
                p[4],c1.m0,c1.m3,c1.m5,
                p[8],c2.m0,c2.m3,c2.m5,
                p12 ,c3.m0,c3.m3,c3.m5];

        let s1=[c0.m5,c0.m4,c0.m2,p3,
                c1.m5,c1.m4,c1.m2,p[7],
                c2.m5,c2.m4,c2.m2,p[11],
                c3.m5,c3.m4,c3.m2,p15];

        let n0=this.normal(s0[12],s0[13],s0[14],s0[15],s0[11],s0[7],s0[3]);
        if(abs2(n0) <= this.epsilon) {
          n0=this.normal(s0[12],s0[13],s0[14],s0[15],s0[2],s0[1],s0[0]);
          if(abs2(n0) <= this.epsilon)
            n0=this.normal(s0[0],s0[4],s0[8],s0[12],s0[11],s0[7],s0[3]);
        }

        let n1=this.normal(s1[3],s1[2],s1[1],s1[0],s1[4],s1[8],s1[12]);
        if(abs2(n1) <= this.epsilon) {
          n1=this.normal(s1[3],s1[2],s1[1],s1[0],s1[13],s1[14],s1[15]);
          if(abs2(n1) <= this.epsilon)
            n1=this.normal(s1[15],s1[11],s1[7],s1[3],s1[4],s1[8],s1[12]);
        }

        let e=this.Epsilon;

        // A kludge to remove subdivision cracks, only applied the first time
        // an edge is found to be flat before the rest of the subpatch is.

        let m0=[0.5*(P1[0]+P2[0]),
                0.5*(P1[1]+P2[1]),
                0.5*(P1[2]+P2[2])];
        if(!flat1) {
          if((flat1=Straightness(p12,p[13],p[14],p15) < this.res2)) {
            let r=unit(this.differential(s1[12],s1[8],s1[4],s1[0]));
            m0=[m0[0]-e*r[0],m0[1]-e*r[1],m0[2]-e*r[2]];
          }
          else m0=s0[15];
        }

        let m1=[0.5*(P3[0]+P0[0]),
                0.5*(P3[1]+P0[1]),
                0.5*(P3[2]+P0[2])];
        if(!flat3) {
          if((flat3=Straightness(p0,p[1],p[2],p3) < this.res2)) {
            let r=unit(this.differential(s0[3],s0[7],s0[11],s0[15]));
            m1=[m1[0]-e*r[0],m1[1]-e*r[1],m1[2]-e*r[2]];
          }
          else m1=s1[0];
        }

        if(C0) {
          let c0=Array(4);
          let c1=Array(4);
          for(let i=0; i < 4; ++i) {
            c0[i]=0.5*(C1[i]+C2[i]);
            c1[i]=0.5*(C3[i]+C0[i]);
          }

          let i0=this.data.Vertex(m0,n0,c0);
          let i1=this.data.Vertex(m1,n1,c1);

          this.Render(s0,I0,I1,i0,i1,P0,P1,m0,m1,flat0,flat1,false,flat3,
                      C0,C1,c0,c1);
          this.Render(s1,i1,i0,I2,I3,m1,m0,P2,P3,false,flat1,flat2,flat3,
                      c1,c0,C2,C3);
        } else {
          let i0=this.vertex(m0,n0);
          let i1=this.vertex(m1,n1);

          this.Render(s0,I0,I1,i0,i1,P0,P1,m0,m1,flat0,flat1,false,flat3);
          this.Render(s1,i1,i0,I2,I3,m1,m0,P2,P3,false,flat1,flat2,flat3);
        }
        return;
      }

      if(d[1] < this.res2) { // flat in vertical direction; split horizontally
        /*
          P refers to a corner
          m refers to a midpoint
          s refers to a subpatch

                   m1
          +--------+--------+
          |P3      |      P2|
          |        |        |
          |        |        |
          |        |        |
          |        |        |
          |   s0   |   s1   |
          |        |        |
          |        |        |
          |        |        |
          |        |        |
          |P0      |      P1|
          +--------+--------+
                   m0

        */

        let c0=new Split3(p0,p[4],p[8],p12);
        let c1=new Split3(p[1],p[5],p[9],p[13]);
        let c2=new Split3(p[2],p[6],p[10],p[14]);
        let c3=new Split3(p3,p[7],p[11],p15);

        let s0=[p0,p[1],p[2],p3,
                c0.m0,c1.m0,c2.m0,c3.m0,
                c0.m3,c1.m3,c2.m3,c3.m3,
                c0.m5,c1.m5,c2.m5,c3.m5];

        let s1=[c0.m5,c1.m5,c2.m5,c3.m5,
                c0.m4,c1.m4,c2.m4,c3.m4,
                c0.m2,c1.m2,c2.m2,c3.m2,
                p12,p[13],p[14],p15];

        let n0=this.normal(s0[0],s0[4],s0[8],s0[12],s0[13],s0[14],s0[15]);
        if(abs2(n0) <= this.epsilon) {
          n0=this.normal(s0[0],s0[4],s0[8],s0[12],s0[11],s0[7],s0[3]);
          if(abs2(n0) <= this.epsilon)
            n0=this.normal(s0[3],s0[2],s0[1],s0[0],s0[13],s0[14],s0[15]);
        }

        let n1=this.normal(s1[15],s1[11],s1[7],s1[3],s1[2],s1[1],s1[0]);
        if(abs2(n1) <= this.epsilon) {
          n1=this.normal(s1[15],s1[11],s1[7],s1[3],s1[4],s1[8],s1[12]);
          if(abs2(n1) <= this.epsilon)
            n1=this.normal(s1[12],s1[13],s1[14],s1[15],s1[2],s1[1],s1[0]);
        }

        let e=this.Epsilon;

        // A kludge to remove subdivision cracks, only applied the first time
        // an edge is found to be flat before the rest of the subpatch is.

        let m0=[0.5*(P0[0]+P1[0]),
                0.5*(P0[1]+P1[1]),
                0.5*(P0[2]+P1[2])];
        if(!flat0) {
          if((flat0=Straightness(p0,p[4],p[8],p12) < this.res2)) {
            let r=unit(this.differential(s1[0],s1[1],s1[2],s1[3]));
            m0=[m0[0]-e*r[0],m0[1]-e*r[1],m0[2]-e*r[2]];
          }
          else m0=s0[12];
        }

        let m1=[0.5*(P2[0]+P3[0]),
                0.5*(P2[1]+P3[1]),
                0.5*(P2[2]+P3[2])];
        if(!flat2) {
          if((flat2=Straightness(p15,p[11],p[7],p3) < this.res2)) {
            let r=unit(this.differential(s0[15],s0[14],s0[13],s0[12]));
            m1=[m1[0]-e*r[0],m1[1]-e*r[1],m1[2]-e*r[2]];
          }
          else m1=s1[3];
        }

        if(C0) {
          let c0=Array(4);
          let c1=Array(4);
          for(let i=0; i < 4; ++i) {
            c0[i]=0.5*(C0[i]+C1[i]);
            c1[i]=0.5*(C2[i]+C3[i]);
          }

          let i0=this.data.Vertex(m0,n0,c0);
          let i1=this.data.Vertex(m1,n1,c1);

          this.Render(s0,I0,i0,i1,I3,P0,m0,m1,P3,flat0,false,flat2,flat3,
                      C0,c0,c1,C3);
          this.Render(s1,i0,I1,I2,i1,m0,P1,P2,m1,flat0,flat1,flat2,false,
                      c0,C1,C2,c1);
        } else {
          let i0=this.vertex(m0,n0);
          let i1=this.vertex(m1,n1);

          this.Render(s0,I0,i0,i1,I3,P0,m0,m1,P3,flat0,false,flat2,flat3);
          this.Render(s1,i0,I1,I2,i1,m0,P1,P2,m1,flat0,flat1,flat2,false);
        }
        return;
      }

      /*
       Horizontal and vertical subdivision:
       P refers to a corner
       m refers to a midpoint
       s refers to a subpatch
         
                m2
       +--------+--------+
       |P3      |      P2|
       |        |        |
       |   s3   |   s2   |
       |        |        |
       |        | m4     |
    m3 +--------+--------+ m1
       |        |        |
       |        |        |
       |   s0   |   s1   |
       |        |        |
       |P0      |      P1|
       +--------+--------+
                m0
    */

      // Subdivide patch:

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
      if(abs2(n0) < this.epsilon) {
        n0=this.normal(s0[0],s0[4],s0[8],s0[12],s0[11],s0[7],s0[3]);
        if(abs2(n0) < this.epsilon)
          n0=this.normal(s0[3],s0[2],s0[1],s0[0],s0[13],s0[14],s0[15]);
      }

      let n1=this.normal(s1[12],s1[13],s1[14],s1[15],s1[11],s1[7],s1[3]);
      if(abs2(n1) < this.epsilon) {
        n1=this.normal(s1[12],s1[13],s1[14],s1[15],s1[2],s1[1],s1[0]);
        if(abs2(n1) < this.epsilon)
          n1=this.normal(s1[0],s1[4],s1[8],s1[12],s1[11],s1[7],s1[3]);
      }

      let n2=this.normal(s2[15],s2[11],s2[7],s2[3],s2[2],s2[1],s2[0]);
      if(abs2(n2) < this.epsilon) {
        n2=this.normal(s2[15],s2[11],s2[7],s2[3],s2[4],s2[8],s2[12]);
        if(abs2(n2) < this.epsilon)
          n2=this.normal(s2[12],s2[13],s2[14],s2[15],s2[2],s2[1],s2[0]);
      }

      let n3=this.normal(s3[3],s3[2],s3[1],s3[0],s3[4],s3[8],s3[12]);
      if(abs2(n3) < this.epsilon) {
        n3=this.normal(s3[3],s3[2],s3[1],s3[0],s3[13],s3[14],s3[15]);
        if(abs2(n3) < this.epsilon)
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
          let r=unit(this.differential(s1[0],s1[1],s1[2],s1[3]));
          m0=[m0[0]-e*r[0],m0[1]-e*r[1],m0[2]-e*r[2]];
        }
        else m0=s0[12];
      }

      let m1=[0.5*(P1[0]+P2[0]),
              0.5*(P1[1]+P2[1]),
              0.5*(P1[2]+P2[2])];
      if(!flat1) {
        if((flat1=Straightness(p12,p[13],p[14],p15) < this.res2)) {
          let r=unit(this.differential(s2[12],s2[8],s2[4],s2[0]));
          m1=[m1[0]-e*r[0],m1[1]-e*r[1],m1[2]-e*r[2]];
        }
        else m1=s1[15];
      }

      let m2=[0.5*(P2[0]+P3[0]),
              0.5*(P2[1]+P3[1]),
              0.5*(P2[2]+P3[2])];
      if(!flat2) {
        if((flat2=Straightness(p15,p[11],p[7],p3) < this.res2)) {
          let r=unit(this.differential(s3[15],s3[14],s3[13],s3[12]));
          m2=[m2[0]-e*r[0],m2[1]-e*r[1],m2[2]-e*r[2]];
        }
        else m2=s2[3];
      }
      
      let m3=[0.5*(P3[0]+P0[0]),
              0.5*(P3[1]+P0[1]),
              0.5*(P3[2]+P0[2])];
      if(!flat3) {
        if((flat3=Straightness(p0,p[1],p[2],p3) < this.res2)) {
          let r=unit(this.differential(s0[3],s0[7],s0[11],s0[15]));
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

        let i0=this.data.Vertex(m0,n0,c0);
        let i1=this.data.Vertex(m1,n1,c1);
        let i2=this.data.Vertex(m2,n2,c2);
        let i3=this.data.Vertex(m3,n3,c3);
        let i4=this.data.Vertex(m4,n4,c4);

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
  process3(p) {
    if(wireframe == 1) {
      this.curve(p,0,1,3,6);
      this.curve(p,6,7,8,9);
      this.curve(p,9,5,2,0);
      return;
    }

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

      let i0=this.data.Vertex(p0,n0,c0);
      let i1=this.data.Vertex(p6,n1,c1);
      let i2=this.data.Vertex(p9,n2,c2);
    
      this.Render3(p,i0,i1,i2,p0,p6,p9,false,false,false,c0,c1,c2);

    } else {
      let i0=this.vertex(p0,n0);
      let i1=this.vertex(p6,n1);
      let i2=this.vertex(p9,n2);

      this.Render3(p,i0,i1,i2,p0,p6,p9,false,false,false);
    }
    if(this.data.indices.length > 0) this.append();
  }

  Render3(p,I0,I1,I2,P0,P1,P2,flat0,flat1,flat2,C0,C1,C2) {
    if(this.Distance3(p) < this.res2) { // Bezier triangle is flat
      if(!this.offscreen([P0,P1,P2])) {
        if(wireframe == 0) {
          this.data.indices.push(I0);
          this.data.indices.push(I1);
          this.data.indices.push(I2);
        } else {
          this.data.indices.push(I0);
          this.data.indices.push(I1);
          this.data.indices.push(I1);
          this.data.indices.push(I2);
          this.data.indices.push(I2);
          this.data.indices.push(I0);
        }
      }
    } else {
  // Approximate bounds by bounding box of control polyhedron.
      if(this.offscreen(p)) return;

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
          let r=unit(this.sumdifferential(c[0],c[2],c[5],c[9],c[1],c[3],c[6]));
          m0=[m0[0]-e*r[0],m0[1]-e*r[1],m0[2]-e*r[2]];
        }
        else m0=r030;
      }


      let m1=[0.5*(P2[0]+P0[0]),
              0.5*(P2[1]+P0[1]),
              0.5*(P2[2]+P0[2])];
      if(!flat1) {
        if((flat1=Straightness(l003,p012,p021,u030) < this.res2)) {
          let r=unit(this.sumdifferential(c[6],c[3],c[1],c[0],c[7],c[8],c[9]));
          m1=[m1[0]-e*r[0],m1[1]-e*r[1],m1[2]-e*r[2]];
        }
        else m1=l030;
      }

      let m2=[0.5*(P0[0]+P1[0]),
              0.5*(P0[1]+P1[1]),
              0.5*(P0[2]+P1[2])];
      if(!flat2) {
        if((flat2=Straightness(l003,p102,p201,r300) < this.res2)) {
          let r=unit(this.sumdifferential(c[9],c[8],c[7],c[6],c[5],c[2],c[0]));
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
        
        let i0=this.data.Vertex(m0,n0,c0);
        let i1=this.data.Vertex(m1,n1,c1);
        let i2=this.data.Vertex(m2,n2,c2);
        
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

    // Check the horizontal flatness.
    let h=Flatness(p0,p12,p3,p15);
    // Check straightness of the horizontal edges and interior control curves.
    h=Math.max(Straightness(p0,p[4],p[8],p12));
    h=Math.max(h,Straightness(p[1],p[5],p[9],p[13]));
    h=Math.max(h,Straightness(p3,p[7],p[11],p15));
    h=Math.max(h,Straightness(p[2],p[6],p[10],p[14]));

    // Check the vertical flatness.
    let v=Flatness(p0,p3,p12,p15);
    // Check straightness of the vertical edges and interior control curves.
    v=Math.max(v,Straightness(p0,p[1],p[2],p3));
    v=Math.max(v,Straightness(p[4],p[5],p[6],p[7]));
    v=Math.max(v,Straightness(p[8],p[9],p[10],p[11]));
    v=Math.max(v,Straightness(p12,p[13],p[14],p15));

    return [h,v];
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

  // Return the differential of the Bezier curve p0,p1,p2,p3 at 0.
  differential(p0,p1,p2,p3) {
    let p=[3*(p1[0]-p0[0]),3*(p1[1]-p0[1]),3*(p1[2]-p0[2])];
    if(abs2(p) > this.epsilon)
      return p;
    
    p=bezierPP(p0,p1,p2);
    if(abs2(p) > this.epsilon)
      return p;
    
    return bezierPPP(p0,p1,p2,p3);
  }

  sumdifferential(p0,p1,p2,p3,p4,p5,p6) {
    let d0=this.differential(p0,p1,p2,p3);
    let d1=this.differential(p0,p4,p5,p6);
    return [d0[0]+d1[0],d0[1]+d1[1],d0[2]+d1[2]];
  }
  
  normal(left3,left2,left1,middle,right1,right2,right3) {
    let ux=3*(right1[0]-middle[0]);
    let uy=3*(right1[1]-middle[1]);
    let uz=3*(right1[2]-middle[2]);
    let vx=3*(left1[0]-middle[0]);
    let vy=3*(left1[1]-middle[1]);
    let vz=3*(left1[2]-middle[2]);

    let n=[uy*vz-uz*vy,
           uz*vx-ux*vz,
           ux*vy-uy*vx];
    if(abs2(n) > this.epsilon)
      return n;

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
      return n;

    let lppp=bezierPPP(middle,left1,left2,left3);
    let rppp=bezierPPP(middle,right1,right2,right3);

    a=cross(rp,lppp);
    b=cross(rppp,lp);
    let c=cross(rpp,lpp);

    n=[a[0]+b[0]+c[0],
       a[1]+b[1]+c[1],
       a[2]+b[2]+c[2]];
    if(abs2(n) > this.epsilon)
      return n;

    a=cross(rppp,lpp);
    b=cross(rpp,lppp);

    n=[a[0]+b[0],
       a[1]+b[1],
       a[2]+b[2]];
    if(abs2(n) > this.epsilon)
      return n;

    return cross(rppp,lppp);
  }
}

class BezierCurve extends Geometry {
  constructor(controlpoints,CenterIndex,MaterialIndex,Min,Max) {
    super();
    this.controlpoints=controlpoints;
    this.Min=Min;
    this.Max=Max;
    this.CenterIndex=CenterIndex;
    this.MaterialIndex=MaterialIndex;
  }

  setMaterialIndex() {
    this.setMaterial(material1Data,drawMaterial1);
  }

  processLine(p) {
    let p0=p[0];
    let p1=p[1];
    if(!this.offscreen([p0,p1])) {
      let n=[0,0,1];
      this.data.indices.push(this.data.vertex(p0,n));
      this.data.indices.push(this.data.vertex(p1,n));
      this.append();
    }
  }

  process(p) {
    if(p.length == 2) return this.processLine(p);

    let p0=p[0];
    let p1=p[1];
    let p2=p[2];
    let p3=p[3];

    let n0=this.normal(bezierP(p0,p1),bezierPP(p0,p1,p2));
    let n1=this.normal(bezierP(p2,p3),bezierPP(p3,p2,p1));
    
    let i0=this.data.vertex(p0,n0);
    let i3=this.data.vertex(p3,n1);
    
    this.Render(p,i0,i3);
    if(this.data.indices.length > 0) this.append();
  }
  
  append() {
    material1Data.append(this.data);
  }

  Render(p,I0,I1) {
    let p0=p[0];
    let p1=p[1];
    let p2=p[2];
    let p3=p[3];

    if(Straightness(p0,p1,p2,p3) < this.res2) { // Segment is flat
      if(!this.offscreen([p0,p3])) {
        this.data.indices.push(I0);
        this.data.indices.push(I1);
      }
    } else { // Segment is not flat
      if(this.offscreen(p)) return;

      let m0=[0.5*(p0[0]+p1[0]),0.5*(p0[1]+p1[1]),0.5*(p0[2]+p1[2])];
      let m1=[0.5*(p1[0]+p2[0]),0.5*(p1[1]+p2[1]),0.5*(p1[2]+p2[2])];
      let m2=[0.5*(p2[0]+p3[0]),0.5*(p2[1]+p3[1]),0.5*(p2[2]+p3[2])];
      let m3=[0.5*(m0[0]+m1[0]),0.5*(m0[1]+m1[1]),0.5*(m0[2]+m1[2])];
      let m4=[0.5*(m1[0]+m2[0]),0.5*(m1[1]+m2[1]),0.5*(m1[2]+m2[2])];
      let m5=[0.5*(m3[0]+m4[0]),0.5*(m3[1]+m4[1]),0.5*(m3[2]+m4[2])];
      
      let s0=[p0,m0,m3,m5];
      let s1=[m5,m4,m2,p3];
      
      let n0=this.normal(bezierPh(p0,p1,p2,p3),bezierPPh(p0,p1,p2,p3));
      let i0=this.data.vertex(m5,n0);
      
      this.Render(s0,I0,i0);
      this.Render(s1,i0,I1);
    }
  }

  normal(bP,bPP) {
    let bPbP=dot(bP,bP);
    let bPbPP=dot(bP,bPP);
    return [bPbP*bPP[0]-bPbPP*bP[0],
            bPbP*bPP[1]-bPbPP*bP[1],
            bPbP*bPP[2]-bPbPP*bP[2]];
  }
}

class Pixel extends Geometry {
  constructor(controlpoint,width,MaterialIndex,Min,Max) {
    super();
    this.controlpoint=controlpoint;
    this.width=width;
    this.CenterIndex=0;
    this.MaterialIndex=MaterialIndex;
    this.Min=Min;
    this.Max=Max;
  }

  setMaterialIndex() {
    this.setMaterial(material0Data,drawMaterial0);
  }

  process(p) {
    this.data.indices.push(this.data.vertex0(this.controlpoint,this.width));
    this.append();
  }
  
  append() {
    material0Data.append(this.data);
  }
}

class Triangles extends Geometry {
  constructor(MaterialIndex,Min,Max) {
    super();
    this.CenterIndex=0;
    this.MaterialIndex=MaterialIndex;
    this.Min=Min;
    this.Max=Max;
    this.Positions=Positions;
    this.Normals=Normals;
    this.Colors=Colors;
    this.Indices=Indices;
    Positions=[];
    Normals=[];
    Colors=[];
    Indices=[];
    this.transparent=Materials[MaterialIndex].diffuse[3] < 1;
  }
    
  setMaterialIndex() {
    if(this.transparent)
      this.setMaterial(transparentData,drawTransparent);
    else
      this.setMaterial(triangleData,drawTriangle);
  }

  process(p) {
    // Override materialIndex to encode color vs material
      materialIndex=this.Colors.length > 0 ?
      -1-materialIndex : 1+materialIndex;

    for(let i=0, n=this.Indices.length; i < n; ++i) {
      let index=this.Indices[i];
      let PI=index[0];
      let P0=this.Positions[PI[0]];
      let P1=this.Positions[PI[1]];
      let P2=this.Positions[PI[2]];
      if(!this.offscreen([P0,P1,P2])) {
        let NI=index.length > 1 ? index[1] : PI;
        if(!NI || NI.length == 0) NI=PI;
        if(this.Colors.length > 0) {
          let CI=index.length > 2 ? index[2] : PI;
          if(!CI || CI.length == 0) CI=PI;
          let C0=this.Colors[CI[0]];
          let C1=this.Colors[CI[1]];
          let C2=this.Colors[CI[2]];
          this.transparent |= C0[3]+C1[3]+C2[3] < 765;
          if(wireframe == 0) {
            this.data.iVertex(PI[0],P0,this.Normals[NI[0]],C0);
            this.data.iVertex(PI[1],P1,this.Normals[NI[1]],C1);
            this.data.iVertex(PI[2],P2,this.Normals[NI[2]],C2);
          } else {
            this.data.iVertex(PI[0],P0,this.Normals[NI[0]],C0);
            this.data.iVertex(PI[1],P1,this.Normals[NI[1]],C1);
            this.data.iVertex(PI[1],P1,this.Normals[NI[1]],C1);
            this.data.iVertex(PI[2],P2,this.Normals[NI[2]],C2);
            this.data.iVertex(PI[2],P2,this.Normals[NI[2]],C2);
            this.data.iVertex(PI[0],P0,this.Normals[NI[0]],C0);
          }
        } else {
          if(wireframe == 0) {
            this.data.iVertex(PI[0],P0,this.Normals[NI[0]]);
            this.data.iVertex(PI[1],P1,this.Normals[NI[1]]);
            this.data.iVertex(PI[2],P2,this.Normals[NI[2]]);
          } else {
            this.data.iVertex(PI[0],P0,this.Normals[NI[0]]);
            this.data.iVertex(PI[1],P1,this.Normals[NI[1]]);
            this.data.iVertex(PI[1],P1,this.Normals[NI[1]]);
            this.data.iVertex(PI[2],P2,this.Normals[NI[2]]);
            this.data.iVertex(PI[2],P2,this.Normals[NI[2]]);
            this.data.iVertex(PI[0],P0,this.Normals[NI[0]]);
          }
        }
      }
    }
    this.data.nvertices=this.Positions.length;
    if(this.data.indices.length > 0) this.append();
  }

  append() {
    if(this.transparent)
      transparentData.append(this.data);
    else
      triangleData.append(this.data);
  }
}

function home()
{
  mat4.identity(rotMat);
  initProjection();
  setProjection();
  remesh=true;
  draw();
}

let positionAttribute=0;
let normalAttribute=1;
let materialAttribute=2;
let colorAttribute=3;
let widthAttribute=4;

function initShader(options=[])
{
  let vertexShader=getShader(gl,vertex,gl.VERTEX_SHADER,options);
  let fragmentShader=getShader(gl,fragment,gl.FRAGMENT_SHADER,options);
  let shader=gl.createProgram();

  gl.attachShader(shader,vertexShader);
  gl.attachShader(shader,fragmentShader);
  gl.bindAttribLocation(shader,positionAttribute,"position");
  gl.bindAttribLocation(shader,normalAttribute,"normal");
  gl.bindAttribLocation(shader,materialAttribute,"materialIndex");
  gl.bindAttribLocation(shader,colorAttribute,"color");
  gl.bindAttribLocation(shader,widthAttribute,"width");
  gl.linkProgram(shader);
  if (!gl.getProgramParameter(shader,gl.LINK_STATUS)) {
    alert("Could not initialize shaders");
  }

  return shader;
}

class Split3 {
  constructor(z0,c0,c1,z1) {
    this.m0=[0.5*(z0[0]+c0[0]),0.5*(z0[1]+c0[1]),0.5*(z0[2]+c0[2])];
    let m1_0=0.5*(c0[0]+c1[0]);
    let m1_1=0.5*(c0[1]+c1[1]);
    let m1_2=0.5*(c0[2]+c1[2]);
    this.m2=[0.5*(c1[0]+z1[0]),0.5*(c1[1]+z1[1]),0.5*(c1[2]+z1[2])];
    this.m3=[0.5*(this.m0[0]+m1_0),0.5*(this.m0[1]+m1_1),
             0.5*(this.m0[2]+m1_2)];
    this.m4=[0.5*(m1_0+this.m2[0]),0.5*(m1_1+this.m2[1]),
             0.5*(m1_2+this.m2[2])];
    this.m5=[0.5*(this.m3[0]+this.m4[0]),0.5*(this.m3[1]+this.m4[1]),
             0.5*(this.m3[2]+this.m4[2])];
  }
}

function unit(v)
{
  let norm=1/(Math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]) || 1);
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

// Return one-third of the first derivative of the Bezier curve defined
// by a,b,c,d at t=0.
function bezierP(a,b)
{
  return [b[0]-a[0],
          b[1]-a[1],
          b[2]-a[2]];
}

// Return one-half of the second derivative of the Bezier curve defined
// by a,b,c,d at t=0.
function bezierPP(a,b,c)
{
  return [3*(a[0]+c[0])-6*b[0],
          3*(a[1]+c[1])-6*b[1],
          3*(a[2]+c[2])-6*b[2]];
}

// Return one-sixth of the third derivative of the Bezier curve defined by
// a,b,c,d at t=0.
function bezierPPP(a,b,c,d)
{
  return [d[0]-a[0]+3*(b[0]-c[0]),
          d[1]-a[1]+3*(b[1]-c[1]),
          d[2]-a[2]+3*(b[2]-c[2])];
}

// Return four-thirds of the first derivative of the Bezier curve defined by
// a,b,c,d at t=1/2.
function bezierPh(a,b,c,d)
{
  return [c[0]+d[0]-a[0]-b[0],
          c[1]+d[1]-a[1]-b[1],
          c[2]+d[2]-a[2]-b[2]];
}

// Return two-thirds of the second derivative of the Bezier curve defined by
// a,b,c,d at t=1/2.
function bezierPPh(a,b,c,d)
{
  return [3*a[0]-5*b[0]+c[0]+d[0],
          3*a[1]-5*b[1]+c[1]+d[1],
          3*a[2]-5*b[2]+c[2]+d[2]];
}

/**
 * Return the maximum distance squared of points c0 and c1 from 
 * the respective internal control points of z0--z1.
*/
function Straightness(z0,c0,c1,z1)
{
  let v=[third*(z1[0]-z0[0]),third*(z1[1]-z0[1]),third*(z1[2]-z0[2])];
  return Math.max(abs2([c0[0]-v[0]-z0[0],c0[1]-v[1]-z0[1],c0[2]-v[2]-z0[2]]),
    abs2([z1[0]-v[0]-c1[0],z1[1]-v[1]-c1[1],z1[2]-v[2]-c1[2]]));
}

// Return one ninth of the relative flatness squared of a--b and c--d.
function Flatness(a,b,c,d)
{
  let u=[b[0]-a[0],b[1]-a[1],b[2]-a[2]];
  let v=[d[0]-c[0],d[1]-c[1],d[2]-c[2]];
  return Math.max(abs2(cross(u,unit(v))),abs2(cross(v,unit(u))))/9;
}

// Return the vertices of the box containing 3d points m and M.
function corners(m,M)
{
  return [m,[m[0],m[1],M[2]],[m[0],M[1],m[2]],[m[0],M[1],M[2]],
          [M[0],m[1],m[2]],[M[0],m[1],M[2]],[M[0],M[1],m[2]],M];
}

function minbound(v) {
  return [
    Math.min(v[0][0],v[1][0],v[2][0],v[3][0],v[4][0],v[5][0],v[6][0],v[7][0]),
    Math.min(v[0][1],v[1][1],v[2][1],v[3][1],v[4][1],v[5][1],v[6][1],v[7][1]),
    Math.min(v[0][2],v[1][2],v[2][2],v[3][2],v[4][2],v[5][2],v[6][2],v[7][2])
  ];
}

function maxbound(v) {
  return [
    Math.max(v[0][0],v[1][0],v[2][0],v[3][0],v[4][0],v[5][0],v[6][0],v[7][0]),
    Math.max(v[0][1],v[1][1],v[2][1],v[3][1],v[4][1],v[5][1],v[6][1],v[7][1]),
    Math.max(v[0][2],v[1][2],v[2][2],v[3][2],v[4][2],v[5][2],v[6][2],v[7][2])
  ];
}

/**
 * Perform a change of basis
 * @param {*} out Out Matrix
 * @param {*} mat Matrix
 * 
 * Compute the matrix (translMatrix) * mat * (translMatrix)^{-1} 
 */

function COBTarget(out,mat)
{
  mat4.fromTranslation(T,[center.x,center.y,center.z])
  mat4.invert(cjMatInv,T);
  mat4.multiply(out,mat,cjMatInv);
  mat4.multiply(out,T,out);
}

function setUniforms(data,shader)
{
  let pixel=shader == pixelShader;

  gl.useProgram(shader);

  gl.enableVertexAttribArray(positionAttribute);

  if(pixel)
    gl.enableVertexAttribArray(widthAttribute);

  let normals=!pixel && Lights.length > 0;
  if(normals)
    gl.enableVertexAttribArray(normalAttribute);

  gl.enableVertexAttribArray(materialAttribute);

  shader.projViewMatUniform=gl.getUniformLocation(shader,"projViewMat");
  shader.viewMatUniform=gl.getUniformLocation(shader,"viewMat");
  shader.normMatUniform=gl.getUniformLocation(shader,"normMat");

  if(shader == colorShader || shader == transparentShader)
    gl.enableVertexAttribArray(colorAttribute);

  if(normals) {
    for(let i=0; i < Lights.length; ++i)
      Lights[i].setUniform(shader,i);
  }

  for(let i=0; i < data.materials.length; ++i)
      data.materials[i].setUniform(shader,i);

  gl.uniformMatrix4fv(shader.projViewMatUniform,false,projViewMat);
  gl.uniformMatrix4fv(shader.viewMatUniform,false,viewMat);
  gl.uniformMatrix3fv(shader.normMatUniform,false,normMat);
}

function handleMouseDown(event)
{
  if(!zoomEnabled)
    enableZoom();
  mouseDownOrTouchActive=true;
  lastMouseX=event.clientX;
  lastMouseY=event.clientY;
}

let pinch=false;
let pinchStart;

function pinchDistance(touches)
{
  return Math.hypot(
    touches[0].pageX-touches[1].pageX,
    touches[0].pageY-touches[1].pageY);
}


let touchStartTime;

function handleTouchStart(event)
{
  event.preventDefault();
  if(!zoomEnabled)
    enableZoom();
  let touches=event.targetTouches;
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

function handleMouseUpOrTouchEnd(event)
{
  mouseDownOrTouchActive=false;
}

function rotateScene(lastX,lastY,rawX,rawY,factor)
{
  if(lastX == rawX && lastY == rawY) return;
  let [angle,axis]=arcball([lastX,-lastY],[rawX,-rawY]);

  mat4.fromRotation(T,2*factor*ArcballFactor*angle/lastzoom,axis);
  mat4.multiply(rotMat,T,rotMat);
}

function shiftScene(lastX,lastY,rawX,rawY)
{
  let zoominv=1/lastzoom;
  shift.x += (rawX-lastX)*zoominv*halfCanvasWidth;
  shift.y -= (rawY-lastY)*zoominv*halfCanvasHeight;
}

function panScene(lastX,lastY,rawX,rawY)
{
  if (orthographic) {
    shiftScene(lastX,lastY,rawX,rawY);
  } else {
    center.x += (rawX-lastX)*(viewParam.xmax-viewParam.xmin);
    center.y -= (rawY-lastY)*(viewParam.ymax-viewParam.ymin);
  }
}

function updateViewMatrix()
{
  COBTarget(viewMat,rotMat);
  mat4.translate(viewMat,viewMat,[center.x,center.y,0]);
  mat3.fromMat4(viewMat3,viewMat);
  mat3.invert(normMat,viewMat3);
  mat4.multiply(projViewMat,projMat,viewMat);
}

function capzoom() 
{
  let maxzoom=Math.sqrt(Number.MAX_VALUE);
  let minzoom=1/maxzoom;
  if(Zoom <= minzoom) Zoom=minzoom;
  if(Zoom >= maxzoom) Zoom=maxzoom;
  
  if(Zoom != lastzoom) remesh=true;
  lastzoom=Zoom;
}

function zoomImage(diff)
{
  let stepPower=zoomStep*halfCanvasHeight*diff;
  const limit=Math.log(0.1*Number.MAX_VALUE)/Math.log(zoomFactor);

  if(Math.abs(stepPower) < limit) {
    Zoom *= zoomFactor**stepPower;
    capzoom();
  }
}

function normMouse(v)
{
  let v0=v[0];
  let v1=v[1];
  let norm=Math.hypot(v0,v1);
  if(norm > 1) {
    denom=1/norm;
    v0 *= denom;
    v1 *= denom;
  }
  return [v0,v1,Math.sqrt(Math.max(1-v1*v1-v0*v0,0))];
}

function arcball(oldmouse,newmouse)
{
  let oldMouse=normMouse(oldmouse);
  let newMouse=normMouse(newmouse);
  let Dot=dot(oldMouse,newMouse);
  if(Dot > 1) Dot=1;
  else if(Dot < -1) Dot=-1;
  return [Math.acos(Dot),unit(cross(oldMouse,newMouse))]
}

/**
 * Mouse Drag Zoom
 * @param {*} lastX unused
 * @param {*} lastY 
 * @param {*} rawX unused
 * @param {*} rawY 
 */
function zoomScene(lastX,lastY,rawX,rawY)
{
  zoomImage(lastY-rawY);
}

// mode:
const DRAGMODE_ROTATE=1;
const DRAGMODE_SHIFT=2;
const DRAGMODE_ZOOM=3;
const DRAGMODE_PAN=4
function processDrag(newX,newY,mode,factor=1)
{
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

  let lastX=(lastMouseX-halfCanvasWidth)/halfCanvasWidth;
  let lastY=(lastMouseY-halfCanvasHeight)/halfCanvasHeight;
  let rawX=(newX-halfCanvasWidth)/halfCanvasWidth;
  let rawY=(newY-halfCanvasHeight)/halfCanvasHeight;

  dragFunc(lastX,lastY,rawX,rawY,factor);

  lastMouseX=newX;
  lastMouseY=newY;

  setProjection();
  draw();
}

let zoomEnabled=0;

function enableZoom()
{
  zoomEnabled=1;
  canvas.addEventListener("wheel",handleMouseWheel,false);
}

function disableZoom()
{
  zoomEnabled=0;
  canvas.removeEventListener("wheel",handleMouseWheel,false);
}

function handleKey(event)
{
  let ESC=27;

  if(!zoomEnabled)
    enableZoom();

  if(embedded && zoomEnabled && event.keyCode == ESC) {
    disableZoom();
    return;
  }

  let keycode=event.key;
  let axis=[];

  switch(keycode) {
  case 'x':
    axis=[1,0,0];
    break;
  case 'y':
    axis=[0,1,0];
    break;
  case 'z':
    axis=[0,0,1];
    break;
  case 'h':
    home();
    break;
  case 'm':
    ++wireframe;
    if(wireframe == 3) wireframe=0;
    if(wireframe != 2) {
      if(!embedded)
        deleteShaders();
      initShaders();
    }
    remesh=true;
    draw();
    break;
  case '+':
  case '=':
  case '>':
    expand();
    break;
  case '-':
  case '_':
  case '<':
    shrink();
    break;
  default:
    break;
  }

  if(axis.length > 0) {
    mat4.rotate(rotMat,rotMat,0.1,axis);
    updateViewMatrix();
    draw();
  }
}

function handleMouseWheel(event)
{
  event.preventDefault();
  
  if (event.deltaY < 0) {
    Zoom *= zoomFactor;
  } else {
    Zoom /= zoomFactor;
  }
  capzoom();
  setProjection();

  draw();
}

function handleMouseMove(event)
{
  if(!mouseDownOrTouchActive) {
    return;
  }

  let newX=event.clientX;
  let newY=event.clientY;

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

let zooming=false;
let swipe=false;
let rotate=false;

function handleTouchMove(event)
{
  event.preventDefault();
  if(zooming) return;
  let touches=event.targetTouches;

  if(!pinch && touches.length == 1 && touchId == touches[0].identifier) {
    let newX=touches[0].pageX;
    let newY=touches[0].pageY;
    let dx=newX-lastMouseX;
    let dy=newY-lastMouseY;
    let stationary=dx*dx+dy*dy <= shiftHoldDistance*shiftHoldDistance;
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
      let newX=touches[0].pageX;
      let newY=touches[0].pageY;
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
    draw();
  }
}

let zbuffer=[];

function transformVertices(vertices)
{
  let Tz0=viewMat[2];
  let Tz1=viewMat[6];
  let Tz2=viewMat[10];
  zbuffer.length=vertices.length;
  for(let i=0; i < vertices.length; ++i) {
    let i6=6*i;
    zbuffer[i]=Tz0*vertices[i6]+Tz1*vertices[i6+1]+Tz2*vertices[i6+2];
  }
}

function drawMaterial0()
{
  drawBuffer(material0Data,pixelShader);
  material0Data.clear();
}

function drawMaterial1()
{
  drawBuffer(material1Data,materialShader);
  material1Data.clear();
}

function drawMaterial()
{
  drawBuffer(materialData,materialShader);
  materialData.clear();
}

function drawColor()
{
  drawBuffer(colorData,colorShader);
  colorData.clear();
}

function drawTriangle()
{
  drawBuffer(triangleData,transparentShader);
  triangleData.clear();
}

function drawTransparent()
{
  let indices=transparentData.indices;
  if(wireframe > 0) {
    drawBuffer(transparentData,transparentShader,indices);
    transparentData.clear();
    return;
  }
  if(indices.length > 0) {
    transformVertices(transparentData.vertices);
    
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

    gl.depthMask(false); // Enable transparency
    drawBuffer(transparentData,transparentShader,Indices);
    gl.depthMask(true); // Disable transparency
  }
  transparentData.clear();
}

function drawBuffers()
{
  drawMaterial0();
  drawMaterial1();
  drawMaterial();
  drawColor();
  drawTriangle();
  drawTransparent();
}

function draw()
{
  if(embedded) {
    offscreen.width=canvas.width;
    offscreen.height=canvas.height;
    setViewport();
  }

  gl.clearColor(Background[0],Background[1],Background[2],Background[3]);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  for(let i=0; i < P.length; ++i)
    P[i].render();

  drawBuffers();

  if(embedded) {
    context.clearRect(0,0,canvas.width,canvas.height);
    context.drawImage(offscreen,0,0);
  }

  if(wireframe == 0) remesh=false;
}

function setDimensions(width,height,X,Y)
{
  let Aspect=width/height;
  let zoominv=1/lastzoom;
  let xshift=(X/width+viewportshift[0])*lastzoom;
  let yshift=(Y/height+viewportshift[1])*lastzoom;

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

function setProjection()
{
  setDimensions(canvasWidth,canvasHeight,shift.x,shift.y);
  let f=orthographic ? mat4.ortho : mat4.frustum;
  f(projMat,viewParam.xmin,viewParam.xmax,
    viewParam.ymin,viewParam.ymax,
    -viewParam.zmax,-viewParam.zmin);
  updateViewMatrix();
}

function initProjection()
{
  H=-Math.tan(0.5*angle)*B[2];

  center.x=center.y=0;
  center.z=0.5*(b[2]+B[2]);
  lastzoom=Zoom=Zoom0;

  viewParam.zmin=b[2];
  viewParam.zmax=B[2];

  shift.x=shift.y=0;
}

function setViewport()
{
  gl.viewportWidth=canvasWidth;
  gl.viewportHeight=canvasHeight;
  gl.viewport(0,0,gl.viewportWidth,gl.viewportHeight);
  gl.scissor(0,0,gl.viewportWidth,gl.viewportHeight);
}

function setCanvas()
{
  canvas.width=canvasWidth;
  canvas.height=canvasHeight;
  if(embedded) {
    offscreen.width=canvasWidth;
    offscreen.height=canvasHeight;
  }
  size2=Math.hypot(canvasWidth,canvasHeight);
  halfCanvasWidth=0.5*canvasWidth;
  halfCanvasHeight=0.5*canvasHeight;
}

function setsize(w,h)
{
  if(w > maxViewportWidth)
    w=maxViewportWidth;

  if(h > maxViewportHeight)
    h=maxViewportHeight;

  shift.x *= w/canvasWidth;
  shift.y *= h/canvasHeight;

  canvasWidth=w;
  canvasHeight=h;
  setCanvas();
  setViewport();
  home();
}

function expand() 
{
  setsize(canvasWidth*resizeStep+0.5,canvasHeight*resizeStep+0.5);
}

function shrink() 
{
  setsize(Math.max((canvasWidth/resizeStep+0.5),1),
          Math.max((canvasHeight/resizeStep+0.5),1));
}

let pixelShader,materialShader,colorShader,transparentShader;

function webGLInit()
{
  canvas=document.getElementById("Asymptote");
  embedded=window.top.document != document;

  initGL();

  if(absolute && !embedded) {
    canvasWidth *= window.devicePixelRatio;
    canvasHeight *= window.devicePixelRatio;
  } else {
    canvas.width=Math.max(window.innerWidth-windowTrim,windowTrim);
    canvas.height=Math.max(window.innerHeight-windowTrim,windowTrim);

    let Aspect=canvasWidth/canvasHeight;
    if(canvas.width > canvas.height*Aspect) 
      canvas.width=Math.min(canvas.height*Aspect,canvas.width);
    else 
      canvas.height=Math.min(canvas.width/Aspect,canvas.height);

    if(canvas.width > 0) 
      canvasWidth=canvas.width;

    if(canvas.height > 0) 
      canvasHeight=canvas.height;
  }

  setCanvas();

  ArcballFactor=1+8*Math.hypot(viewportmargin[0],viewportmargin[1])/size2;

  viewportshift[0] /= Zoom0;
  viewportshift[1] /= Zoom0;

  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);
  gl.enable(gl.DEPTH_TEST);
  gl.enable(gl.SCISSOR_TEST);

  setViewport();
  home();

  canvas.onmousedown=handleMouseDown;
  document.onmouseup=handleMouseUpOrTouchEnd;
  document.onmousemove=handleMouseMove;
  canvas.onkeydown=handleKey;

  if(!embedded)
    enableZoom();
  canvas.addEventListener("touchstart",handleTouchStart,false);
  canvas.addEventListener("touchend",handleMouseUpOrTouchEnd,false);
  canvas.addEventListener("touchcancel",handleMouseUpOrTouchEnd,false);
  canvas.addEventListener("touchleave",handleMouseUpOrTouchEnd,false);
  canvas.addEventListener("touchmove",handleTouchMove,false);
  document.addEventListener("keydown",handleKey,false);
}

let listen=false;

class Align {
  constructor(center,dir) {
    this.center=center;
    if(dir) {
      let theta=dir[0];
      let phi=dir[1];

      this.ct=Math.cos(theta);
      this.st=Math.sin(theta);
      this.cp=Math.cos(phi);
      this.sp=Math.sin(phi);
    }
  }

  T0(v) {
    return [v[0]+this.center[0],v[1]+this.center[1],v[2]+this.center[2]];
  }

  T(v) {
    let x=v[0];
    let Y=v[1];
    let z=v[2];
    let X=x*this.ct+z*this.st;
    return [X*this.cp-Y*this.sp+this.center[0],
            X*this.sp+Y*this.cp+this.center[1],
            -x*this.st+z*this.ct+this.center[2]];
  };
}

function Tcorners(T,m,M) {
  let v=[T(m),T([m[0],m[1],M[2]]),T([m[0],M[1],m[2]]),
         T([m[0],M[1],M[2]]),T([M[0],m[1],m[2]]),
         T([M[0],m[1],M[2]]),T([M[0],M[1],m[2]]),T(M)];
  return [minbound(v),maxbound(v)];
}

// draw a sphere of radius r about center
// (or optionally a hemisphere symmetric about direction dir)
function sphere(center,r,CenterIndex,MaterialIndex,dir)
{
  let b=0.524670512339254;
  let c=0.595936986722291;
  let d=0.954967051233925;
  let e=0.0820155480083437;
  let f=0.996685028842544;
  let g=0.0549670512339254;
  let h=0.998880711874577;
  let i=0.0405017186586849;

  let octant=[[
    [1,0,0],
    [1,0,b],
    [c,0,d],
    [e,0,f],

    [1,a,0],
    [1,a,b],
    [c,a*c,d],
    [e,a*e,f],

    [a,1,0],
    [a,1,b],
    [a*c,c,d],
    [a*e,e,f],

    [0,1,0],
    [0,1,b],
    [0,c,d],
    [0,e,f]
  ],[
    [e,0,f],
    [e,a*e,f],
    [g,0,h],
    [a*e,e,f],
    [i,i,1],
    [0.05*a,0,1],
    [0,e,f],
    [0,g,h],
    [0,0.05*a,1],
    [0,0,1]
  ]];

  let rx,ry,rz;
  let A=new Align(center,dir);
  let s,t,z;

  if(dir) {
    s=1;
    z=0;
    t=A.T.bind(A);
  } else {
    s=-1;
    z=-r;
    t=A.T0.bind(A);
  }

  function T(V) {
    let p=Array(V.length);
    for(let i=0; i < V.length; ++i) {
      let v=V[i];
      p[i]=t([rx*v[0],ry*v[1],rz*v[2]]);
    }
    return p;
  }

  let v=Tcorners(t,[-r,-r,z],[r,r,r]);
  let Min=v[0], Max=v[1];
  for(let i=-1; i <= 1; i += 2) {
    rx=i*r;
    for(let j=-1; j <= 1; j += 2) {
      ry=j*r;
      for(let k=s; k <= 1; k += 2) {
        rz=k*r;
        for(let m=0; m < 2; ++m)
          P.push(new BezierPatch(T(octant[m]),CenterIndex,MaterialIndex,
                                 Min,Max));
      }
    }
  }
}

let a=4/3*(Math.sqrt(2)-1);

// draw a disk of radius r aligned in direction dir
function disk(center,r,CenterIndex,MaterialIndex,dir)
{
  let b=1-2*a/3;

  let unitdisk=[
    [1,0,0],
    [1,-a,0],
    [a,-1,0],
    [0,-1,0],

    [1,a,0],
    [b,0,0],
    [0,-b,0],
    [-a,-1,0],

    [a,1,0],
    [0,b,0],
    [-b,0,0],
    [-1,-a,0],

    [0,1,0],
    [-a,1,0],
    [-1,a,0],
    [-1,0,0]
  ];

  let A=new Align(center,dir);

  function T(V) {
    let p=Array(V.length);
    for(let i=0; i < V.length; ++i) {
      let v=V[i];
      p[i]=A.T([r*v[0],r*v[1],0]);
    }
    return p;
  }

  let v=Tcorners(A.T.bind(A),[-r,-r,0],[r,r,0]);
  P.push(new BezierPatch(T(unitdisk),CenterIndex,MaterialIndex,v[0],v[1]));
}

// draw a cylinder with circular base of radius r about center and height h
// aligned in direction dir
function cylinder(center,r,h,CenterIndex,MaterialIndex,dir,core)
{
  let unitcylinder=[
    [1,0,0],
    [1,0,1/3],
    [1,0,2/3],
    [1,0,1],

    [1,a,0],
    [1,a,1/3],
    [1,a,2/3],
    [1,a,1],

    [a,1,0],
    [a,1,1/3],
    [a,1,2/3],
    [a,1,1],

    [0,1,0],
    [0,1,1/3],
    [0,1,2/3],
    [0,1,1]
  ];

  let rx,ry,rz;
  let A=new Align(center,dir);

  function T(V) {
    let p=Array(V.length);
    for(let i=0; i < V.length; ++i) {
      let v=V[i];
      p[i]=A.T([rx*v[0],ry*v[1],h*v[2]]);
    }
    return p;
  }

  let v=Tcorners(A.T.bind(A),[-r,-r,0],[r,r,h]);
  let Min=v[0], Max=v[1];

  for(let i=-1; i <= 1; i += 2) {
    rx=i*r;
    for(let j=-1; j <= 1; j += 2) {
      ry=j*r;
      P.push(new BezierPatch(T(unitcylinder),CenterIndex,MaterialIndex,
                             Min,Max));
    }
  }

  if(core) {
    let Center=A.T([0,0,h]);
    P.push(new BezierCurve([center,Center],CenterIndex,MaterialIndex,
                           center,Center));
  }
}

function rmf(z0,c0,c1,z1,t)
{
  class Rmf {
    constructor(p,r,t) {
      this.p=p;
      this.r=r;
      this.t=t;
      this.s=cross(t,r);
    }
  }

  // Return a unit vector perpendicular to a given unit vector v.
  function perp(v)
  {
    let u=cross(v,[0,1,0]);
    let norm=Number.EPSILON*abs2(v);
    if(abs2(u) > norm) return unit(u);
    u=cross(v,[0,0,1]);
    return (abs2(u) > norm) ? unit(u) : [1,0,0];
  }

  let norm=Number.EPSILON*Math.max(abs2(z0),abs2(c0),abs2(c1),
                                abs2(z1));

// Special case of dir for t in (0,1].
  function dir(t) {
    if(t == 1) {
      let dir=[z1[0]-c1[0],
               z1[1]-c1[1],
               z1[2]-c1[2]];
      if(abs2(dir) > norm) return unit(dir);
      dir=[2*c1[0]-c0[0]-z1[0],
           2*c1[1]-c0[1]-z1[1],
           2*c1[2]-c0[2]-z1[2]];
      if(abs2(dir) > norm) return unit(dir);
      return [z1[0]-z0[0]+3*(c0[0]-c1[0]),
              z1[1]-z0[1]+3*(c0[1]-c1[1]),
              z1[2]-z0[2]+3*(c0[2]-c1[2])];
    }
    let a=[z1[0]-z0[0]+3*(c0[0]-c1[0]),
           z1[1]-z0[1]+3*(c0[1]-c1[1]),
           z1[2]-z0[2]+3*(c0[2]-c1[2])];
    let b=[2*(z0[0]+c1[0])-4*c0[0],
           2*(z0[1]+c1[1])-4*c0[1],
           2*(z0[2]+c1[2])-4*c0[2]];
    let c=[c0[0]-z0[0],c0[1]-z0[1],c0[2]-z0[2]];
    let t2=t*t;
    let dir=[a[0]*t2+b[0]*t+c[0],
             a[1]*t2+b[1]*t+c[1],
             a[2]*t2+b[2]*t+c[2]];
    if(abs2(dir) > norm) return unit(dir);
    t2=2*t;
    dir=[a[0]*t2+b[0],
         a[1]*t2+b[1],
         a[2]*t2+b[2]];
    if(abs2(dir) > norm) return unit(dir);
    return unit(a);
  }

  let R=Array(t.length);
  let T=[c0[0]-z0[0],
         c0[1]-z0[1],
         c0[2]-z0[2]];
  if(abs2(T) < norm) {
    T=[z0[0]-2*c0[0]+c1[0],
       z0[1]-2*c0[1]+c1[1],
       z0[2]-2*c0[2]+c1[2]];
    if(abs2(T) < norm)
      T=[z1[0]-z0[0]+3*(c0[0]-c1[0]),
         z1[1]-z0[1]+3*(c0[1]-c1[1]),
         z1[2]-z0[2]+3*(c0[2]-c1[2])];
  }
  T=unit(T);
  let Tp=perp(T);
  R[0]=new Rmf(z0,Tp,T);
  for(let i=1; i < t.length; ++i) {
    let Ri=R[i-1];
    let s=t[i];
    let onemt=1-s;
    let onemt2=onemt*onemt;
    let onemt3=onemt2*onemt;
    let s3=3*s;
    onemt2 *= s3;
    onemt *= s3*s;
    let t3=s*s*s;
    let p=[
      onemt3*z0[0]+onemt2*c0[0]+onemt*c1[0]+t3*z1[0],
      onemt3*z0[1]+onemt2*c0[1]+onemt*c1[1]+t3*z1[1],
      onemt3*z0[2]+onemt2*c0[2]+onemt*c1[2]+t3*z1[2]];
    let v1=[p[0]-Ri.p[0],p[1]-Ri.p[1],p[2]-Ri.p[2]];
    if(v1[0] != 0 || v1[1] != 0 || v1[2] != 0) {
      let r=Ri.r;
      let u1=unit(v1);
      let ti=Ri.t;
      let dotu1ti=dot(u1,ti)
      let tp=[ti[0]-2*dotu1ti*u1[0],
              ti[1]-2*dotu1ti*u1[1],
              ti[2]-2*dotu1ti*u1[2]];
      ti=dir(s);
      let dotu1r2=2*dot(u1,r);
      let rp=[r[0]-dotu1r2*u1[0],r[1]-dotu1r2*u1[1],r[2]-dotu1r2*u1[2]];
      let u2=unit([ti[0]-tp[0],ti[1]-tp[1],ti[2]-tp[2]]);
      let dotu2rp2=2*dot(u2,rp);
      rp=[rp[0]-dotu2rp2*u2[0],rp[1]-dotu2rp2*u2[1],rp[2]-dotu2rp2*u2[2]];
      R[i]=new Rmf(p,unit(rp),unit(ti));
    } else
      R[i]=R[i-1];
  }
  return R;
}

// draw a tube of width w using control points v
function tube(v,w,CenterIndex,MaterialIndex,Min,Max,core)
{
  let Rmf=rmf(v[0],v[1],v[2],v[3],[0,1/3,2/3,1]);

  let aw=a*w;
  let arc=[[w,0],[w,aw],[aw,w],[0,w]];

  function f(a,b,c,d) {
    let s=Array(16);
    for(let i=0; i < 4; ++i) {
      let R=Rmf[i];

      let R0=R.r[0], R1=R.s[0];
      let T0=R0*a+R1*b;
      let T1=R0*c+R1*d;

      R0=R.r[1]; R1=R.s[1];
      let T4=R0*a+R1*b;
      let T5=R0*c+R1*d;

      R0=R.r[2]; R1=R.s[2];
      let T8=R0*a+R1*b;
      let T9=R0*c+R1*d;

      let w=v[i];
      let w0=w[0]; w1=w[1]; w2=w[2];
      for(let j=0; j < 4; ++j) {
        let u=arc[j];
        let x=u[0], y=u[1];
        s[4*i+j]=[T0*x+T1*y+w0,
                  T4*x+T5*y+w1,
                  T8*x+T9*y+w2];
      }
    }
    P.push(new BezierPatch(s,CenterIndex,MaterialIndex,Min,Max));
  }

  f(1,0,0,1);
  f(0,-1,1,0);
  f(-1,0,0,-1);
  f(0,1,-1,0);

  if(core)
    P.push(new BezierCurve(v,CenterIndex,MaterialIndex,Min,Max));
}

function webGLStart()
{
  if(window.innerWidth == 0 || window.innerHeight == 0) {
    if(!listen) {
      listen=true;
      window.addEventListener("resize",webGLStart,false);
    }
  } else {
    if(listen) {
      window.removeEventListener("resize",webGLStart,false);
      listen=false;
    }
    webGLInit();
  }
}
