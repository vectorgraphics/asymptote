// Render Bezier patches via subdivision with WebGL.
// Author: John C. Bowman, Supakorn "Jamie" Rassameemasmuang

var gl;

var epsilon;
var pixel=1.0; // Adaptive rendering constant.
var BezierFactor=0.4;
var FillFactor=0.1;
var Zoom=1;
const zoomStep=0.1;
var zoomFactor=1.05;
var Zoom0=1; // FIXME
var lastzoom;

var Fuzz2=1000*Number.EPSILON;
var Fuzz4=Fuzz2*Fuzz2;
var third=1.0/3.0;

var P=[]; // Array of patches
var M=[]; // Array of materials

var rotMat=mat4.create();
var rMatrix=mat4.create();
var pMatrix=mat4.create();
var vMatrix=mat4.create();
var normMatrix=mat4.create();
var mMatrix=mat4.create();
var T=mat4.create(); // Offscreen transformation matrix

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


class Material {
  constructor(diffuse, emissive, specular, shininess, metallic, fresnel0) {
    this.diffuse = diffuse;
    this.emissive = emissive;
    this.specular = specular;
    this.shininess = shininess;
    this.metallic = metallic;
    this.fresnel0 = fresnel0;
  }

  setUniform(program, stringLoc, index = null) {
    var getLoc;
    if (index === null) {
      getLoc =
        param => gl.getUniformLocation(program, stringLoc + "." + param);
    } else {
      getLoc =
        param => gl.getUniformLocation(program, stringLoc + "[" + index + "]." + param);
    }

    gl.uniform4fv(getLoc("diffuse"), new Float32Array(this.diffuse));
    gl.uniform4fv(getLoc("emissive"), new Float32Array(this.emissive));
    gl.uniform4fv(getLoc("specular"), new Float32Array(this.specular));

    gl.uniform1f(getLoc("shininess"), this.shininess);
    gl.uniform1f(getLoc("metallic"), this.metallic);
    gl.uniform1f(getLoc("fresnel0"), this.fresnel0);
  }
}

var enumPointLight = 1;
var enumDirectionalLight = 2;

class Light {
  constructor(type, lightColor, brightness, customParam) {
    this.type = type;
    this.lightColor = lightColor;
    this.brightness = brightness;
    this.customParam = customParam;
  }

  setUniform(program, stringLoc, index) {
    var getLoc =
        param => gl.getUniformLocation(program, stringLoc + "[" + index + "]." + param);

    gl.uniform1i(getLoc("type"), this.type);
    gl.uniform3fv(getLoc("color"), new Float32Array(this.lightColor));
    gl.uniform1f(getLoc("brightness"), this.brightness);
    gl.uniform4fv(getLoc("parameter"), new Float32Array(this.customParam));
  }
}

function initGL(canvas) {
  try {
    gl = canvas.getContext("webgl");
    gl.viewportWidth = canvas.width;
    gl.viewportHeight = canvas.height;
  } catch (e) {}
  if (!gl) {
    alert("Could not initialize WebGL");
  }
}

function getShader(gl, id) {
  var shaderScript = document.getElementById(id);
  if (!shaderScript) {
    return null;
  }

  var str = `#version 100
  precision highp float;
  const int nLights=${lights.length};
  const int nMaterials=${M.length};
  `
  if (orthographic) {
    str += `
    #define ORTHOGRAPHIC
    `;
  }

  var k = shaderScript.firstChild;
  while (k) {
    if (k.nodeType == 3) {
      str += k.textContent;
    }
    k = k.nextSibling;
  }
  var shader;
  if (shaderScript.type == "x-shader/x-fragment") {
    shader = gl.createShader(gl.FRAGMENT_SHADER);
  } else if (shaderScript.type == "x-shader/x-vertex") {
    shader = gl.createShader(gl.VERTEX_SHADER);
  } else {
    return null;
  }
  gl.shaderSource(shader, str);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    alert(gl.getShaderInfoLog(shader));
    return null;
  }
  return shader;
}

class DrawableObject {
  draw(remesh=false) {}
}

class AvertexStructure {
  addVertex() {}
  drawBuffer() {}
  clearBuffer() {}
}

class StdVertexStructure extends AvertexStructure {
  constructor() {
    super()

    this.vertices = [];
    this.materials = [];
    this.colors = [];
    this.normals = [];
    this.indices = [];
    this.nvertices = 0;

    this.fArrVertices = null;
    this.fArrColors = null;
    this.fArrNormals = null;
    this.iArrIndices=null;
    this.fArrMaterials=null;

    this.arraysInitialized=false;
    this.nvertices = 0;
  }

  addVertex(vertData) {
    vertData.position.forEach( coord => 
      this.vertices.push(coord)
    );
    vertData.normals.forEach( coord => 
      this.normals.push(coord)
    );

    this.colors.push(0.0);
    this.colors.push(0.0);
    this.colors.push(0.0);
    this.colors.push(1.0);
    
    this.materials.push(vertData.materialIndex);

    this.arraysInitialized=false;

    return this.nvertices++;
  }

  createArrays() {
    this.fArrVertices=new Float32Array(this.vertices);
    this.fArrColors=new Float32Array(this.colors);
    this.fArrNormals=new Float32Array(this.normals);
    this.iArrIndices=indexExt ? new Uint32Array(this.indices) : new Uint16Array(this.indices);
    this.fArrMaterials=new Float32Array(this.materials);

    this.arraysInitialized=true;
  }

  drawBuffer() {
    if (!this.arraysInitialized) {
      this.createArrays();
    }

    copyFloatBuffer(VertexBuffer,this.fArrVertices,shaderProgram.vertexPositionAttribute, this.nvertices);
//   copyFloatBuffer(ColorBuffer,this.fArrColors,shaderProgram.vertexColorAttribute, this.nvertices);
    copyFloatBuffer(NormalBuffer,this.fArrNormals,shaderProgram.vertexNormalAttribute, this.nvertices);

    if (shaderProgram.vertexMaterialIndexAttribute !== -1) {
      gl.bindBuffer(gl.ARRAY_BUFFER, MaterialIndexBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, this.fArrMaterials, gl.STATIC_DRAW);
      gl.vertexAttribPointer(shaderProgram.vertexMaterialIndexAttribute,
      MaterialIndexBuffer.itemSize, gl.FLOAT, false, 0, 0);
    }
    MaterialIndexBuffer.numItems = this.nvertices;

    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,this.iArrIndices,gl.STATIC_DRAW);
    indexBuffer.numItems = this.indices.length;

    gl.drawElements(gl.TRIANGLES, indexBuffer.numItems,
                    indexExt ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT, 0);
    }
    
  clearBuffer() {
    this.vertices=[];
    this.colors=[];
    this.normals=[];
    this.indices=[];
    this.materials=[];

    this.nvertices=0;

    this.fArrVertices=null;
    this.fArrColors=null;
    this.fArrNormals=null;
    this.iArrIndices=null;
    this.iArrMaterials=null;
    
    this.arraysInitialized=false;
  }
}

class GeometryDrawable extends DrawableObject {

  /**
   * @param {*} materialIndex Index of Material
   * @param {*} vertexstructure Vertex Structure (Color, transparent, etc)
   */
  constructor(materialIndex, vertexstructure=StdVertexStructure) {
    super();
    this.Offscreen=true;
    this.rendered=false;
    this.materialIndex=materialIndex;
    this.vertexstructures=new vertexstructure();
  }
  
  clearBuffer() {
    this.vertexstructures.clearBuffer();
    this.rendered=false;
  }

  drawBuffer() {
    this.vertexstructures.drawBuffer();
  }

  render() {}

  draw(remesh=false) {
    if(remesh || this.Offscreen)
      this.clearBuffer();

    if(!this.rendered)
      this.render();

    this.drawBuffer()
  }

}

function copyFloatBuffer(buf, data, attrib, nverts) {
  if (attrib !== -1) {
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    gl.vertexAttribPointer(attrib, buf.itemSize, gl.FLOAT, false, 0, 0);
    }
    buf.numItems=nverts;
}


class BezierPatch extends GeometryDrawable {
  /**
   * Constructor for Bezier Patch
   * @param {*} controlpoints array of 16 control points.
   * @param {*} materialIndex material index
   * @param {*} Minimum bounding box corner
   * @param {*} Maximum bounding box corner
   */
  constructor(controlpoints,materialIndex,Min,Max,
              vertexstructure=StdVertexStructure) {
    super(materialIndex,vertexstructure);
    this.controlpoints=controlpoints;
    this.Min=Min;
    this.Max=Max;
  }

  addVertex(pos, color, normal) {
    return this.vertexstructures.addVertex({
      position: pos,
      color: color,
      normals: normal, 
      materialIndex: this.materialIndex
    });
  }

  pushIndices(ind) {
    this.vertexstructures.indices.push(ind);
  }

  // Approximate bounds by bounding box of control polyhedron.
  offscreen(n,v) {
    var [x,y,z,X,Y,Z]=boundstriples(n,v);

    if(X >= this.x && x <= this.X &&
       Y >= this.y && y <= this.Y &&
       Z >= this.z && z <= this.Z)
      return false;
    return this.Offscreen=true;
  }

  render() {
    this.Offscreen=false;
    var f,F,s;

    var b=[viewParam.xmin,viewParam.ymin,viewParam.zmin];
    var B=[viewParam.xmax,viewParam.ymax,viewParam.zmax];

    if(orthographic) {
      this.m=b;
      this.M=B;
      s=1.0;
    } else {
      var perspective=1.0/B[2];
      f=this.Min[2]*perspective;
      F=this.Max[2]*perspective;
      this.m=[Math.min(f*b[0],F*b[0]),Math.min(f*b[1],F*b[1]),b[2]];
      this.M=[Math.max(f*B[0],F*B[0]),Math.max(f*B[1],F*B[1]),B[2]];
      s=Math.max(f,F);
    }

   [this.x,this.y,this.z,this.X,this.Y,this.Z]=bboxtransformed(this.m,this.M);

    var billboard=false;
    if(!billboard && (this.Max[0] < this.x || this.Min[0] > this.X ||
                      this.Max[1] < this.y || this.Min[1] > this.Y ||
                      this.Max[2] < this.z || this.Min[2] > this.Z)) {
      this.Offscreen=true;
      return;
    }

    let p=this.controlpoints;
    var res=pixel*Math.hypot(s*(B[0]-b[0]),s*(B[1]-b[1]))/size2;
    this.res2=res*res;
    this.Epsilon=FillFactor*res;

    var p0=p[0];
    epsilon=0;
    for(var i=1; i < 16; ++i)
      epsilon=Math.max(epsilon,
        abs2([p[i][0]-p0[0],p[i][1]-p0[1],p[i][2]-p0[2]]));
    epsilon *= Fuzz4;

    var p3=p[3];
    var p12=p[12];
    var p15=p[15];

    var n0=normal(p3,p[2],p[1],p0,p[4],p[8],p12);
    if(iszero(n0)) {
      n0=normal(p3,p[2],p[1],p0,p[13],p[14],p15);
      if(iszero(n0)) n0=normal(p15,p[11],p[7],p3,p[4],p[8],p12);
    }

    var n1=normal(p0,p[4],p[8],p12,p[13],p[14],p15);
    if(iszero(n1)) {
      n1=normal(p0,p[4],p[8],p12,p[11],p[7],p3);
      if(iszero(n1)) n1=normal(p3,p[2],p[1],p0,p[13],p[14],p15);
    }

    var n2=normal(p12,p[13],p[14],p15,p[11],p[7],p3);
    if(iszero(n2)) {
      n2=normal(p12,p[13],p[14],p15,p[2],p[1],p0);
      if(iszero(n2)) n2=normal(p0,p[4],p[8],p12,p[11],p[7],p3);
    }

    var n3=normal(p15,p[11],p[7],p3,p[2],p[1],p0);
    if(iszero(n3)) {
      n3=normal(p15,p[11],p[7],p3,p[4],p[8],p12);
      if(iszero(n3)) n3=normal(p12,p[13],p[14],p15,p[2],p[1],p0);
    }

    var c0=color(n0);
    var c1=color(n1);
    var c2=color(n2);
    var c3=color(n3);
    
    var i0=this.addVertex(p0,c0,n0);
    var i1=this.addVertex(p12,c1,n1);
    var i2=this.addVertex(p15,c2,n2);
    var i3=this.addVertex(p3,c3,n3);

    this.Render(p,i0,i1,i2,i3,p0,p12,p15,p3,false,false,false,false,
           c0,c1,c2,c3);

    this.rendered=true;
  }

  Render(p,I0,I1,I2,I3,P0,P1,P2,P3,flat0,flat1,flat2,flat3,C0,C1,C2,C3) {
    if(this.Distance(p) < this.res2) { // Patch is flat
      var P=[P0,P1,P2,P3];
      if(!this.offscreen(4,P)) {
        this.pushIndices(I0);
        this.pushIndices(I1);
        this.pushIndices(I2);
        
        this.pushIndices(I0);
        this.pushIndices(I2);
        this.pushIndices(I3);
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

      var p0=p[0];
      var p3=p[3];
      var p12=p[12];
      var p15=p[15];

      var c0=new Split3(p0,p[1],p[2],p3);
      var c1=new Split3(p[4],p[5],p[6],p[7]);
      var c2=new Split3(p[8],p[9],p[10],p[11]);
      var c3=new Split3(p12,p[13],p[14],p15);

      var c4=new Split3(p0,p[4],p[8],p12);
      var c5=new Split3(c0.m0,c1.m0,c2.m0,c3.m0);
      var c6=new Split3(c0.m3,c1.m3,c2.m3,c3.m3);
      var c7=new Split3(c0.m5,c1.m5,c2.m5,c3.m5);
      var c8=new Split3(c0.m4,c1.m4,c2.m4,c3.m4);
      var c9=new Split3(c0.m2,c1.m2,c2.m2,c3.m2);
      var c10=new Split3(p3,p[7],p[11],p15);

      var s0=[p0,c0.m0,c0.m3,c0.m5,c4.m0,c5.m0,c6.m0,c7.m0,
              c4.m3,c5.m3,c6.m3,c7.m3,c4.m5,c5.m5,c6.m5,c7.m5];
      var s1=[c4.m5,c5.m5,c6.m5,c7.m5,c4.m4,c5.m4,c6.m4,c7.m4,
              c4.m2,c5.m2,c6.m2,c7.m2,p12,c3.m0,c3.m3,c3.m5];
      var s2=[c7.m5,c8.m5,c9.m5,c10.m5,c7.m4,c8.m4,c9.m4,c10.m4,
              c7.m2,c8.m2,c9.m2,c10.m2,c3.m5,c3.m4,c3.m2,p15];
      var s3=[c0.m5,c0.m4,c0.m2,p3,c7.m0,c8.m0,c9.m0,c10.m0,
              c7.m3,c8.m3,c9.m3,c10.m3,c7.m5,c8.m5,c9.m5,c10.m5];

      var m4=s0[15];

      var n0=normal(s0[0],s0[4],s0[8],s0[12],s0[13],s0[14],s0[15]);
      if(iszero(n0)) {
        n0=normal(s0[0],s0[4],s0[8],s0[12],s0[11],s0[7],s0[3]);
        if(iszero(n0)) n0=normal(s0[3],s0[2],s0[1],s0[0],s0[13],s0[14],s0[15]);
      }

      var n1=normal(s1[12],s1[13],s1[14],s1[15],s1[11],s1[7],s1[3]);
      if(iszero(n1)) {
        n1=normal(s1[12],s1[13],s1[14],s1[15],s1[2],s1[1],s1[0]);
        if(iszero(n1)) n1=normal(s1[0],s1[4],s1[8],s1[12],s1[11],s1[7],s1[3]);
      }

      var n2=normal(s2[15],s2[11],s2[7],s2[3],s2[2],s2[1],s2[0]);
      if(iszero(n2)) {
        n2=normal(s2[15],s2[11],s2[7],s2[3],s2[4],s2[8],s2[12]);
        if(iszero(n2)) n2=normal(s2[12],s2[13],s2[14],s2[15],s2[2],s2[1],s2[0]);
      }

      var n3=normal(s3[3],s3[2],s3[1],s3[0],s3[4],s3[8],s3[12]);
      if(iszero(n3)) {
        n3=normal(s3[3],s3[2],s3[1],s3[0],s3[13],s3[14],s3[15]);
        if(iszero(n3)) n3=normal(s3[15],s3[11],s3[7],s3[3],s3[4],s3[8],s3[12]);
      }

      var n4=normal(s2[3],s2[2],s2[1],m4,s2[4],s2[8],s2[12]);

      var e=this.Epsilon;

      // A kludge to remove subdivision cracks, only applied the first time
      // an edge is found to be flat before the rest of the subpatch is.
      var m0=[0.5*(P0[0]+P1[0]),0.5*(P0[1]+P1[1]),0.5*(P0[2]+P1[2])];
      if(!flat0) {
        if((flat0=Straightness(p0,p[4],p[8],p12) < this.res2)) {
          var r=unit(derivative(s1[0],s1[1],s1[2],s1[3]));
          m0=[m0[0]-e*r[0],m0[1]-e*r[1],m0[2]-e*r[2]];
        }
        else m0=s0[12];
      }

      var m1=[0.5*(P1[0]+P2[0]),0.5*(P1[1]+P2[1]),0.5*(P1[2]+P2[2])];
      if(!flat1) {
        if((flat1=Straightness(p12,p[13],p[14],p15) < this.res2)) {
          var r=unit(derivative(s2[12],s2[8],s2[4],s2[0]));
          m1=[m1[0]-e*r[0],m1[1]-e*r[1],m1[2]-e*r[2]];
        }
        else m1=s1[15];
      }

      var m2=[0.5*(P2[0]+P3[0]),0.5*(P2[1]+P3[1]),0.5*(P2[2]+P3[2])];
      if(!flat2) {
        if((flat2=Straightness(p15,p[11],p[7],p3) < this.res2)) {
          var r=unit(derivative(s3[15],s2[14],s2[13],s1[12]));
          m2=[m2[0]-e*r[0],m2[1]-e*r[1],m2[2]-e*r[2]];
        }
        else m2=s2[3];
      }
      
      var m3=[0.5*(P3[0]+P0[0]),0.5*(P3[1]+P0[1]),0.5*(P3[2]+P0[2])];
      if(!flat3) {
        if((flat3=Straightness(p0,p[1],p[2],p3) < this.res2)) {
          var r=unit(derivative(s0[3],s0[7],s0[11],s0[15]));
          m3=[m3[0]-e*r[0],m3[1]-e*r[1],m3[2]-e*r[2]];
        }
        else m3=s3[0];
      }

      {
        var c0=color(n0);
        var c1=color(n1);
        var c2=color(n2);
        var c3=color(n3);
        var c4=color(n4);

        var i0=this.addVertex(m0,c0,n0);
        var i1=this.addVertex(m1,c1,n1);
        var i2=this.addVertex(m2,c2,n2);
        var i3=this.addVertex(m3,c3,n3);
        var i4=this.addVertex(m4,c4,n4);

        this.Render(s0,I0,i0,i4,i3,P0,m0,m4,m3,flat0,false,false,flat3,
                    C0,c0,c4,c3);
        this.Render(s1,i0,I1,i1,i4,m0,P1,m1,m4,flat0,flat1,false,false,
                    c0,C1,c1,c4);
        this.Render(s2,i4,i1,I2,i2,m4,m1,P2,m2,false,flat1,flat2,false,
                    c4,c1,C2,c2);
        this.Render(s3,i3,i4,i2,I3,m3,m4,m2,P3,false,false,flat2,flat3,
                    c3,c4,c2,C3);
      }
    }
  }

  Distance(p) {
    var p0=p[0];
    var p3=p[3];
    var p12=p[12];
    var p15=p[15];

    // Check the flatness of the quad.
    var d=Distance2(p15,p0,normal(p3,p[2],p[1],p0,p[4],p[8],p12));
    
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
}

function resetCamera()
{
  mat4.identity(rotMat);
  initProjection();
  setProjection();
  updatevMatrix();
  redraw=true;
}

var shaderProgram;

function initShaders() {
  var fragmentShader = getShader(gl, "shader-fs");
  var vertexShader = getShader(gl, "shader-vs");
  shaderProgram = gl.createProgram();

  gl.attachShader(shaderProgram, vertexShader);
  gl.attachShader(shaderProgram, fragmentShader);
  gl.linkProgram(shaderProgram);
  if (!gl.getProgramParameter(shaderProgram, gl.LINK_STATUS)) {
    alert("Could not initialize shaders");
  }
  gl.useProgram(shaderProgram);

  shaderProgram.vertexPositionAttribute = gl.getAttribLocation(shaderProgram, "aVertexPosition");
  gl.enableVertexAttribArray(shaderProgram.vertexPositionAttribute);
//  shaderProgram.vertexColorAttribute = gl.getAttribLocation(shaderProgram, "aVertexColor");
//  gl.enableVertexAttribArray(shaderProgram.vertexColorAttribute);
  shaderProgram.vertexNormalAttribute = gl.getAttribLocation(shaderProgram, "aVertexNormal");
  gl.enableVertexAttribArray(shaderProgram.vertexNormalAttribute);
  shaderProgram.vertexMaterialIndexAttribute = gl.getAttribLocation(shaderProgram, "aVertexMaterialIndex");
  gl.enableVertexAttribArray(shaderProgram.vertexMaterialIndexAttribute);


  shaderProgram.pvMatrixUniform=gl.getUniformLocation(shaderProgram,"uPVMatrix");
  shaderProgram.vmMatrixUniform=gl.getUniformLocation(shaderProgram,"uVMMatrix");
  shaderProgram.normMatUniform=gl.getUniformLocation(shaderProgram, "uNormMatrix");
  shaderProgram.useColorUniform = gl.getUniformLocation(shaderProgram, "useColor");

}

class Split3 {
  constructor(z0,c0,c1,z1) {
    this.m0=new Array(3);
    this.m2=new Array(3);
    this.m3=new Array(3);
    this.m4=new Array(3);
    this.m5=new Array(3);
    for(var i=0; i < 3; ++i) {
      this.m0[i]=0.5*(z0[i]+c0[i]);
      var m1=0.5*(c0[i]+c1[i]);
      this.m2[i]=0.5*(c1[i]+z1[i]);
      this.m3[i]=0.5*(this.m0[i]+m1);
      this.m4[i]=0.5*(m1+this.m2[i]);
      this.m5[i]=0.5*(this.m3[i]+this.m4[i]);
    }
  }
}

function iszero(v)
{
  return v[0] == 0.0 && v[1] == 0.0 && v[2] == 0.0;
}

function unit(v)
{
  var norm=Math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
  norm=(norm != 0.0) ? 1 / norm : 1;
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

function derivative(p0,p1,p2,p3)
{
  var lp=[p1[0]-p0[0],p1[1]-p0[1],p1[2]-p0[2]];
  if(abs2(lp) > epsilon)
    return lp;
  
  var lpp=bezierPP(p0,p1,p2);
  if(abs2(lpp) > epsilon)
    return lpp;
  
  return bezierPPP(p0,p1,p2,p3);
}

function normal(left3,left2,left1,middle,right1,right2,right3)
{
  var ux=right1[0]-middle[0];
  var uy=right1[1]-middle[1];
  var uz=right1[2]-middle[2];
  var vx=left1[0]-middle[0];
  var vy=left1[1]-middle[1];
  var vz=left1[2]-middle[2];
  var n=[uy*vz-uz*vy,
         uz*vx-ux*vz,
         ux*vy-uy*vx];
  if(abs2(n) > epsilon)
    return unit(n);

  var lp=[vx,vy,vz];
  var rp=[ux,uy,uz];
  var lpp=bezierPP(middle,left1,left2);
  var rpp=bezierPP(middle,right1,right2);
  var a=cross(rpp,lp);
  var b=cross(rp,lpp);
  n=[a[0]+b[0],
     a[1]+b[1],
     a[2]+b[2]];
  if(abs2(n) > epsilon)
    return unit(n);

  var lppp=bezierPPP(middle,left1,left2,left3);
  var rppp=bezierPPP(middle,right1,right2,right3);
  a=cross(rpp,lpp);
  b=cross(rp,lppp);
  var c=cross(rppp,lp);
  var d=cross(rppp,lpp);
  var e=cross(rpp,lppp);
  var f=cross(rppp,lppp);
  return unit([9*a[0]+3*(b[0]+c[0]+d[0]+e[0])+f[0],
               9*a[1]+3*(b[1]+c[1]+d[1]+e[1])+f[1],
               9*a[2]+3*(b[2]+c[2]+d[2]+e[2])+f[2]]);
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

function bboxtransformed(m,M)
{
  var m0=[], m1=[], m2=[], m3=[], m4=[], m5=[], m6=[], m7=[];
  vec3.transformMat4(m0,m,T);
  vec3.transformMat4(m1,[m[0],m[1],M[2]],T);
  vec3.transformMat4(m2,[m[0],M[1],m[2]],T);
  vec3.transformMat4(m3,[m[0],M[1],M[2]],T);
  vec3.transformMat4(m4,[M[0],m[1],m[2]],T);
  vec3.transformMat4(m5,[M[0],m[1],M[2]],T);
  vec3.transformMat4(m6,[M[0],M[1],m[2]],T);
  vec3.transformMat4(m7,M,T);
  return boundstriples(8,[m0,m1,m2,m3,m4,m5,m6,m7]);
}

function boundstriples(n,v)
{
  var x,y,z;
  var X,Y,Z;

  X=x=v[0][0];
  Y=y=v[0][1];
  Z=z=v[0][2];
    
  for(var i=1; i < n; ++i)
    [x,y,z,X,Y,Z]=boundstriple(x,y,z,X,Y,Z,v[i]);
  return [x,y,z,X,Y,Z];
}

function boundstriple(x,y,z,X,Y,Z,v)
{
  [x,X]=bounds(x,X,v[0]);
  [y,Y]=bounds(y,Y,v[1]);
  [z,Z]=bounds(z,Z,v[2]);
  return [x,y,z,X,Y,Z];
}

function bounds(x,X,v)
{
  if(v < x) x=v;
  else if(v > X) X=v;
  return [x,X];
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

function COBTarget(out, mat) {
  return mat4COB(out, getTargetOrigMat(), mat);
}

function inversetranspose(out,mat) {
  mat4.invert(out,mat);
  mat4.transpose(out,out);
  return out;
}

function setUniforms() {
  var msMatrix = mat4.create();
  COBTarget(msMatrix, mMatrix);

  var vmMatrix=mat4.create();
  mat4.multiply(vmMatrix,vMatrix,msMatrix);
  mat4.invert(T,vmMatrix);

  var pvmMatrix=mat4.create();
  mat4.multiply(pvmMatrix,pMatrix,vmMatrix);

  var mNormMatrix=mat4.create();
  inversetranspose(mNormMatrix,mMatrix);

  var vmNormMatrix=mat4.create();
  mat4.multiply(vmNormMatrix,normMatrix,mNormMatrix)

  gl.uniformMatrix4fv(shaderProgram.pvMatrixUniform,false,pvmMatrix);
  gl.uniformMatrix4fv(shaderProgram.vmMatrixUniform,false,vmMatrix);
  gl.uniformMatrix4fv(shaderProgram.normMatUniform,false,vmNormMatrix);
  
  for (let i=0; i < M.length; ++i)
    M[i].setUniform(shaderProgram, "objMaterial", i);

  for (let i=0; i<lights.length; ++i)
    lights[i].setUniform(shaderProgram, "objLights", i);

  // for now, if we simulate headlamp. Can also specify custom lights later on...
  gl.uniform1i(shaderProgram.useColorUniform, 0);

}

/* Buffers */
var VertexBuffer;
var ColorBuffer;
var NormalBuffer;

var pMatrix = mat4.create();
var mMatrix = mat4.create();

var redraw = true;
var remesh=true;
var mouseDownOrTouchActive = false;
var lastMouseX = null;
var lastMouseY = null;
var touchID = null;

function handleMouseDown(event) {
  mouseDownOrTouchActive = true;
  lastMouseX = event.clientX;
  lastMouseY = event.clientY;
}

function handleTouchStart(evt) {
  evt.preventDefault();
  var touches = evt.targetTouches;

  if (touches.length == 1 && !mouseDownOrTouchActive) {
    touchId = touches[0].identifier;
    lastMouseX = touches[0].pageX,
    lastMouseY = touches[0].pageY;
  }
}

function handleMouseUpOrTouchEnd(event) {
  mouseDownOrTouchActive = false;
}

var canvasWidth;
var canvasHeight;

function rotateScene (lastX, lastY, rawX, rawY) {
    let [angle, axis] = arcballLib.arcball([lastX, -lastY], [rawX, -rawY]);

    if (isNaN(angle) || isNaN(axis[0]) ||
      isNaN(axis[1]) || isNaN(axis[2])) {
      console.error("Angle or axis NaN!");
      return;
    }

    var rotMats = mat4.create();
    mat4.fromRotation(rotMats, angle, axis);
    mat4.multiply(rotMat, rotMats, rotMat);
}

function shiftScene(lastX, lastY, rawX, rawY) {
  let xTransl = (rawX - lastX);
  let yTransl = (rawY - lastY);
  // equivalent to shift function in asy.
  let zoominv=1/lastzoom;
  shift.x+=xTransl*zoominv*canvasWidth/2;
  shift.y-=yTransl*zoominv*canvasHeight/2;
}

function panScene(lastX, lastY, rawX, rawY) {
  let xTransl = (rawX - lastX);
  let yTransl = (rawY - lastY);
  if (orthographic) {
    shiftScene(lastX,lastY,rawX,rawY);
  } else {
    center.x+=xTransl*(viewParam.xmax-viewParam.xmin)/2;
    center.y-=yTransl*(viewParam.ymax-viewParam.ymin)/2;
  }
}

function updatevMatrix() {
  COBTarget(vMatrix,rotMat);
  mat4.translate(vMatrix,vMatrix,[center.x,center.y,0]);
  inversetranspose(normMatrix, vMatrix);
}

function updateTmatrix() {
  // has to be called every iteration modelmatrix is changed. 
  var msMatrix = mat4.create();
  COBTarget(msMatrix, mMatrix);

  mat4.multiply(T,vMatrix,msMatrix);
  mat4.invert(T,T);
}

function capzoom() 
{
  var maxzoom=Math.sqrt(Number.MAX_VALUE);
  var minzoom=1.0/maxzoom;
  if(Zoom <= minzoom) Zoom=minzoom;
  if(Zoom >= maxzoom) Zoom=maxzoom;
  
  remesh=true;
  if(Zoom != lastzoom) remesh=true;
  lastzoom=Zoom;
}

/**
 * Mouse Drag Zoom
 * @param {*} lastX unused
 * @param {*} lastY 
 * @param {*} rawX unused
 * @param {*} rawY 
 */
function zoomScene(lastX, lastY, rawX, rawY) {
  let halfCanvHeight=0.5*canvasHeight;
  let stepPower=zoomStep*halfCanvHeight*(lastY-rawY);
  const limit=Math.log(0.1*Number.MAX_VALUE)/Math.log(zoomFactor);

  if(Math.abs(stepPower) < limit) {
    Zoom *= zoomFactor**stepPower;
    capzoom();
  }
}

// mode:
const DRAGMODE_ROTATE=1;
const DRAGMODE_SHIFT=2;
const DRAGMODE_ZOOM=3;
const DRAGMODE_PAN=4
function processDrag(newX, newY, mode, touch=false) {
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
      dragFunc = (_a, _b, _c, _d) => {};
      break;
  }

  let halfCanvWidth = canvasWidth / 2;
  let halfCanvHeight = canvasHeight / 2;

  let lastX = (lastMouseX - halfCanvWidth) / halfCanvWidth;
  let lastY = (lastMouseY - halfCanvHeight) / halfCanvHeight;
  let rawX = (newX - halfCanvWidth) / halfCanvWidth;
  let rawY = (newY - halfCanvHeight) / halfCanvHeight;

  dragFunc(lastX, lastY, rawX, rawY);

  lastMouseX = newX;
  lastMouseY = newY;

  updatevMatrix();
  setProjection();
  redraw = true;
}

function handleKey(key) {
  var keycode = key.key;
  var rotate = true;
  var axis = [0, 0, 1];
  switch (keycode) {
  case "w":
    axis = [-1, 0, 0];
    break;
  case "d":
    axis = [0, 1, 0];
    break;
  case "a":
    axis = [0, -1, 0];
    break;
  case "s":
    axis = [1, 0, 0];
    break;
  case "h":
    resetCamera();
    break;
  default:
    rotate = false;
    break;
  }

  if (rotate) {
    mat4.rotate(rotationMatrix, rotationMatrix, 0.1, axis);
    redraw = true;
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

  redraw=true;
}

function handleMouseMove(event) {
  if (!mouseDownOrTouchActive) {
    return;
  }

  var newX = event.clientX;
  var newY = event.clientY;

  let mode;
  if (event.getModifierState("Control")) {
    mode = DRAGMODE_SHIFT;
  } else if (event.getModifierState("Shift")) {
    mode = DRAGMODE_ZOOM;
  } else if (event.getModifierState("Alt")) {
    mode = DRAGMODE_PAN;
  } else {
    mode = DRAGMODE_ROTATE;
  }

  processDrag(newX, newY, mode, false);
}

function handleTouchMove(evt) {
  evt.preventDefault();
  var touches = evt.targetTouches;

  if (touches.length == 1 && touchId == touches[0].identifier) {
    var newX = touches[0].pageX;
    var newY = touches[0].pageY;
    processDrag(newX, newY, DRAGMODE_ROTATE, true);
  }
}

// Prepare canvas for drawing
function sceneSetup() {
  gl.viewport(0, 0, gl.viewportWidth, gl.viewportHeight);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
}

var indexExt;

// Create buffer data for the patch and its subdivisions to be pushed to the graphics card
//Takes as an argument the array of vertices that define the patch to be drawn 
// Using the vertex position buffer of the above function,draw patch.
function setBuffer() {
  VertexBuffer = gl.createBuffer();
  VertexBuffer.itemSize = 3;

//  ColorBuffer = gl.createBuffer();
//  ColorBuffer.itemSize = 4;

  NormalBuffer = gl.createBuffer();
  NormalBuffer.itemSize = 3;

  MaterialIndexBuffer = gl.createBuffer();
  MaterialIndexBuffer.itemSize = 1;

  indexBuffer = gl.createBuffer();
  indexBuffer.itemSize = 1;

  setUniforms();
  indexExt = gl.getExtension("OES_element_index_uint");
}

// Return color associated with unit normal vector n.
function color(n) {
  return [0,0,0,1];
}

function draw() {
  sceneSetup();
  setBuffer();
  updateTmatrix();
  
  P.forEach(p => p.draw(remesh));

  remesh=false;
}

var forceredraw = false;
var lasttime;
var newtime;

function tick() {
  requestAnimationFrame(tick);
  lasttime = newtime;
  newtime = performance.now();
  // invariant: every time this loop is called, lasttime stores the
  // last time processloop was called. 
  processloop(newtime - lasttime);
  draw();
}

function tickNoRedraw() {
  requestAnimationFrame(tickNoRedraw);
  if (redraw) {
    draw();
    redraw = false;
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
      let H=-Math.tan(0.5*angle)*B[2];
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
  let fn=orthographic ? mat4.ortho : mat4.frustum;
  fn(pMatrix,viewParam.xmin,viewParam.xmax,
     viewParam.ymin,viewParam.ymax,
     -viewParam.zmax,-viewParam.zmin);
}

function initProjection() {
  center={x:0,y:0,z:0.5*(b[2]+B[2])};
  lastzoom=Zoom=Zoom0;
  let fn=orthographic ? mat4.ortho : mat4.frustum;
  fn(pMatrix,b[0],B[0],b[1],B[1],-B[2],-b[2]);

  viewParam={
    xmin:b[0],xmax:B[0],
    ymin:b[1],ymax:B[1],
    zmin:b[2],zmax:B[2]
  };
  shift={
    x:0,y:0
  };
}

function webGLStart() {
  var canvas = document.getElementById("Asymptote");

  canvas.width=canvasWidth;
  canvas.height=canvasHeight;

  initProjection();
  initGL(canvas);
  initShaders();
  gl.clearColor(1.0,1.0,1.0,1.0);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);
  gl.enable(gl.DEPTH_TEST);

  canvas.onmousedown = handleMouseDown;
  document.onmouseup = handleMouseUpOrTouchEnd;
  document.onmousemove = handleMouseMove;
  canvas.onkeydown = handleKey;
  document.onwheel = handleMouseWheel;

  var supportsPassive=false;
  try {
    var opts=Object.defineProperty({},'passive',{
      get: function() {supportsPassive = true;}
    });
    window.addEventListener("testPassive",null,opts);
    window.removeEventListener("testPassive",null,opts);
  } catch (e) {}

  canvas.addEventListener("touchstart", handleTouchStart,
                          supportsPassive ? {passive:true} : false);
  canvas.addEventListener("touchend", handleMouseUpOrTouchEnd, false);
  canvas.addEventListener("touchcancel", handleMouseUpOrTouchEnd, false);
  canvas.addEventListener("touchleave", handleMouseUpOrTouchEnd, false);
  canvas.addEventListener("touchmove", handleTouchMove,
                          supportsPassive ? {passive:true} : false);

  newtime = performance.now();

  pMatrixInit=new Float32Array(pMatrix);

  if (forceredraw) {
    tick();
  } else {
    tickNoRedraw();
  }
}
