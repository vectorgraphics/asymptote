// Contains code from http: //learningwebgl.com/blog/ ? p=28#triangle-vertex-positions 
// modified to produce a subdivision algorithm for rendering Bezier
// patches with WebGL
var gl;

function initGL(canvas) {
    try {
        gl=canvas.getContext("experimental-webgl");
        gl.viewportWidth=canvas.width;
        gl.viewportHeight=canvas.height;
    } catch(e) {}
    if(!gl) {
        alert("Could not initialise WebGL, sorry : -(");
    }
}

function getShader(gl, id) {
    var shaderScript=document.getElementById(id);
    if(!shaderScript) {
        return null;
    }
    var str="";
    var k=shaderScript.firstChild;
    while(k) {
        if(k.nodeType == 3) {
            str += k.textContent;
        }
        k=k.nextSibling;
    }
    var shader;
    if(shaderScript.type == "x-shader/x-fragment") {
        shader=gl.createShader(gl.FRAGMENT_SHADER);
    } else if(shaderScript.type == "x-shader/x-vertex") {
        shader=gl.createShader(gl.VERTEX_SHADER);
    } else {
        return null;
    }
    gl.shaderSource(shader,str);
    gl.compileShader(shader);
    if(!gl.getShaderParameter(shader,gl.COMPILE_STATUS)) {
        alert(gl.getShaderInfoLog(shader));
        return null;
    }
    return shader;
}

var shaderProgram;

function initShaders() {
    var fragmentShader=getShader(gl,"shader-fs");
    var vertexShader=getShader(gl,"shader-vs");
    shaderProgram=gl.createProgram();
    gl.attachShader(shaderProgram,vertexShader);
    gl.attachShader(shaderProgram,fragmentShader);
    gl.linkProgram(shaderProgram);
    if(!gl.getProgramParameter(shaderProgram,gl.LINK_STATUS)) {
        alert("Could not initialise shaders");
    }
    gl.useProgram(shaderProgram);
    shaderProgram.vertexPositionAttribute=gl.getAttribLocation(shaderProgram,"aVertexPosition");
    gl.enableVertexAttribArray(shaderProgram.vertexPositionAttribute);
    shaderProgram.vertexColorAttribute=gl.getAttribLocation(shaderProgram,"aVertexColor");
    gl.enableVertexAttribArray(shaderProgram.vertexColorAttribute);
    shaderProgram.pMatrixUniform=gl.getUniformLocation(shaderProgram,"uPMatrix");
    shaderProgram.mvMatrixUniform=gl.getUniformLocation(shaderProgram,"uMVMatrix");
}
var mvMatrix=mat4.create();
var pMatrix=mat4.create();

function setMatrixUniforms() {
    gl.uniformMatrix4fv(shaderProgram.pMatrixUniform,false,pMatrix);
    gl.uniformMatrix4fv(shaderProgram.mvMatrixUniform,false,mvMatrix);
}

var VertexBuffer;
var ColorBuffer;

var vertices=new Array();
var colors=new Array();
var indices=new Array();
var nvertices=0;

var mvMatrix=mat4.create();
var translOffset=[0,0,0];
var zoomFactor=1;

var mvMatrixStack=[] ;
var pMatrix=mat4.create();

var localRotation=mat4.create();

function mvPushMatrix() {
    var copy=mat4.create();
    mat4.set(mvMatrix,copy);
    mvMatrixStack.push(copy);
}

function mvPopMatrix() {
    if(mvMatrixStack.length == 0) {
        throw "Invalid popMatrix!";
    }
    mvMatrix=mvMatrixStack.pop();
}

function degToRad(degrees) {
    return degrees*Math.PI/180;
}

var redraw=true;
var mouseDownOrTouchActive=false;
var lastMouseX=null;
var lastMouseY=null;
var touchID=null;


var center=[0,0,1];
var centerInv=[0,0,-1];

var rotationMatLocal=mat4.create();
var rotationMatrix=mat4.create();
mat4.identity(rotationMatrix);

function handleMouseDown(event) {
    mouseDownOrTouchActive=true;
    lastMouseX=event.clientX;
    lastMouseY=event.clientY;
}

function handleTouchStart(evt) {
    evt.preventDefault();
    var touches=evt.targetTouches;

    if(touches.length == 1 && !mouseDownOrTouchActive) {
        touchId=touches[0].identifier;
        lastMouseX=touches[0].pageX,
        lastMouseY=touches[0].pageY;
    }
}

function handleMouseUpOrTouchEnd(event) {
    mouseDownOrTouchActive=false;
}

function processDrag(newX, newY, pan=false) {
    let lastX=(lastMouseX-400)/400;
    let lastY=(lastMouseY-400)/400;

    let rawX=(newX-400)/400;
    let rawY=(newY-400)/400;

    if(!pan) {
        let [angle,axis]=arcballLib.arcball([lastX,-lastY],[rawX,-rawY]);
        let tmpMatrix=mat4.create()
        mat4.identity(tmpMatrix)
        mat4.rotate(tmpMatrix,tmpMatrix,angle,[axis[0],axis[1],axis[2]])
        mat4.multiply(rotationMatrix,tmpMatrix,rotationMatrix);
    } else {
        translOffset[0] +=(rawX-lastX);
        translOffset[1] -=(rawY-lastY);
    }
    lastMouseX=newX;
    lastMouseY=newY;
    redraw=true;
}

function handleKey(key) {
    var keycode=key.key;
    var rotate=true;
    var axis=[0,0,1];
    switch(keycode) {
    case "w": 
        axis=[-1,0,0];
        break;
    case "d": 
        axis=[0,1,0];
        break;
    case "a": 
        axis=[0,-1,0];
        break;
    case "s": 
        axis=[1,0,0];
        break;

    default: 
        rotate=false;
        break;
    }

    if(rotate) {
        mat4.rotate(rotationMatrix,rotationMatrix,0.1,axis);
        redraw=true;
    }

}

function handleMouseWheel(event) {
    zoomFactor -= event.deltaY/100;

    if(zoomFactor < 0) {
        zoomFactor=0;
    } else if(zoomFactor > 100) {
        zoomFactor=100;
    }

    res=zoomFactor*0.001;
    redraw=true;
}

function handleMouseMove(event) {
    if(!mouseDownOrTouchActive) {
        return;
    }

    var newX=event.clientX;
    var newY=event.clientY;

    processDrag(newX,newY,event.getModifierState("Alt"));
}

function handleTouchMove(evt) {
    evt.preventDefault();
    var touches=evt.targetTouches;

    if(touches.length == 1 && touchId == touches[0].identifier) {
        var newX=touches[0].pageX;
        var newY=touches[0].pageY;
        processDrag(newX,newY);
    }
}

// Prepare canvas for drawing
function sceneSetup() {
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    mat4.perspective(pMatrix,45,gl.viewportWidth/gl.viewportHeight,0.1,100.0);
    mat4.identity(mvMatrix);

    mat4.translate(mvMatrix,mvMatrix,centerInv);
    mat4.multiply(mvMatrix,mvMatrix,rotationMatrix);
    mat4.scale(mvMatrix,mvMatrix,[zoomFactor,zoomFactor,zoomFactor]);
    mat4.translate(mvMatrix,mvMatrix,center);

    mat4.translate(mvMatrix,mvMatrix,translOffset)
}

var indexExt;

// Create buffer data for the patch and its subdivisions to be pushed to the graphics card
//Takes as an argument the array of vertices that define the patch to be drawn 
// Using the vertex position buffer of the above function,draw patch.
function setBuffer() {
    VertexBuffer=gl.createBuffer();
    VertexBuffer.itemSize=3;

    ColorBuffer=gl.createBuffer();
    ColorBuffer.itemSize=4;

    indexBuffer=gl.createBuffer();
    indexBuffer.itemSize=1;
}

function drawBuffer() {
    gl.bindBuffer(gl.ARRAY_BUFFER,VertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(vertices),gl.STATIC_DRAW);
    gl.vertexAttribPointer(shaderProgram.vertexPositionAttribute,
                           VertexBuffer.itemSize,gl.FLOAT,false,0,0);
    VertexBuffer.numItems=nvertices;

    gl.bindBuffer(gl.ARRAY_BUFFER,ColorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(colors),gl.STATIC_DRAW);
    gl.vertexAttribPointer(shaderProgram.vertexColorAttribute,
                           ColorBuffer.itemSize,gl.FLOAT,false,0,0);
    ColorBuffer.numItems=nvertices;

    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,indexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,
                  indexExt ? new Uint32Array(indices) : new Uint16Array(indices),
                  gl.STATIC_DRAW);
    indexBuffer.numItems=indices.length;
    gl.drawElements(gl.TRIANGLES,indexBuffer.numItems,
                    indexExt ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT,0);
    vertices=[];
    colors=[];
    indices=[];
    nvertices=0;
}

var pixel=1.0; // Adaptive rendering constant.
var FillFactor=0.1;
var BezierFactor=0.4;
//var res=0.0005; // Temporary
var res=0.001; // Temporary
var res2=res*res;
var Epsilon=0.1*res;
var epsilon=0;
var Fuzz=1000*Number.EPSILON;
var Fuzz2=Fuzz*Fuzz;

function Split3(z0, c0, c1, z1) {
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

function unit(v) {
    var norm=Math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
    norm=(norm != 0) ? 1/norm : 1;
    return [v[0]*norm,v[1]*norm,v[2]*norm];
}

// Store the vertex v and its color vector c in the buffer.
function vertex(v,c) {
    vertices.push(v[0]);
    vertices.push(v[1]);
    vertices.push(v[2]);

    colors.push(c[0]);
    colors.push(c[1]);
    colors.push(c[2]);
    colors.push(c[3]);
    return nvertices++;
}

function abs2(v) {
    return v[0]*v[0]+v[1]*v[1]+v[2]*v[2];
}

function dot(u, v) {
    return u[0]*v[0]+u[1]*v[1]+u[2]*v[2];
}

function cross(u, v) {
    return [u[1]*v[2]-u[2]*v[1],
            u[2]*v[0]-u[0]*v[2],
            u[0]*v[1]-u[1]*v[0]
           ];
}

function normal(left3, left2, left1, middle, right1, right2, right3) {
    var u0=right1[0]-middle[0];
    var v0=left1[0]-middle[0];
    var u1=right1[1]-middle[1];
    var v1=left1[1]-middle[1];
    var u2=right1[2]-middle[2];
    var v2=left1[2]-middle[2];
    var n=[
        u1*v2-u2*v1,
        u2*v0-u0*v2,
        u0*v1-u1*v0
    ];
    if(abs2(n) > epsilon)
        return unit(n);

    var lp=[v0,v1,v2];
    var rp=[u0,u1,u2];
    var lpp=[middle[0]+left2[0]-2*left1[0],
             middle[1]+left2[1]-2*left1[1],
             middle[2]+left2[2]-2*left1[2]
            ];
    var rpp=[middle[0]+right2[0]-2*right1[0],
             middle[1]+right2[1]-2*right1[1],
             middle[2]+right2[2]-2*right1[2]
            ];
    var a=cross(rpp,lp);
    var b=cross(rp,lpp);
    n=[a[0]+b[0],
       a[1]+b[1],
       a[2]+b[2]
      ];
    if(abs2(n) > epsilon)
        return unit(n);

    var lppp=[left3[0]-middle[0]+3*(left1[0]-left2[0]),
              left3[1]-middle[1]+3*(left1[1]-left2[1]),
              left3[2]-middle[2]+3*(left1[2]-left2[2])
             ];
    var rppp=[right3[0]-middle[0]+3*(right1[0]-right2[0]),
              right3[1]-middle[1]+3*(right1[1]-right2[1]),
              right3[2]-middle[2]+3*(right1[2]-right2[2])
             ];
    a=cross(rpp,lpp);
    b=cross(rp,lppp);
    var c=cross(rppp,lp);
    var d=cross(rppp,lpp);
    var e=cross(rpp,lppp);
    var f=cross(rppp,lppp);
    return unit([9*a[0]+3*(b[0]+c[0]+d[0]+e[0])+f[0],
                 9*a[1]+3*(b[1]+c[1]+d[1]+e[1])+f[1],
                 9*a[2]+3*(b[2]+c[2]+d[2]+e[2])+f[2]
                ]);
}

// return the maximum distance squared of points c0 and c1 from 
// the respective internal control points of z0--z1.
function Straightness(z0,c0,c1,z1)
{
    var third=1.0/3.0;
    var v=[third*(z1[0]-z0[0]),third*(z1[1]-z0[1]),third*(z1[2]-z0[2])];
    return Math.max(abs2([c0[0]-v[0]-z0[0],c0[1]-v[1]-z0[1],c0[2]-v[2]-z0[2]]),
                    abs2([z1[0]-v[0]-c1[0],z1[1]-v[1]-c1[1],z1[2]-v[2]-c1[2]]));
}

// return the maximum perpendicular distance squared of points c0 and c1
// from z0--z1.
function Distance1(z0, c0, c1, z1) {
    var Z0=[c0[0]-z0[0],c0[1]-z0[1],c0[2]-z0[2]];
    var Q=unit([z1[0]-z0[0],z1[1]-z0[1],z1[2]-z0[2]]);
    var Z1=[c1[0]-z0[0],c1[1]-z0[1],c1[2]-z0[2]];
    var p0=dot(Z0,Q);
    var p1=dot(Z1,Q);
    return Math.max(abs2([Z0[0]-p0*Q[0],Z0[1]-p0*Q[1],Z0[2]-p0*Q[2]]),
                    abs2([Z1[0]-p1*Q[0],Z1[1]-p1*Q[1],Z1[2]-p1*Q[2]]));
}

// return the perpendicular distance squared of a point z from the plane
// through u with unit normal n.
function Distance2(z, u, n) {
    var d=dot([z[0]-u[0],z[1]-u[1],z[2]-u[2]],n);
    return d*d;
}

function Distance(p) {
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

var k=1;
// Return color associated with unit normal vector n.
function color(n) {
    var Ldotn=L[0]*n[0]+L[1]*n[1]+L[2]*n[2];
    if(Ldotn < 0) Ldotn=0;
    var p=[emissive[0]+ambient[0]*Ambient[0]+Ldotn*diffuse[0]*Diffuse[0],
           emissive[1]+ambient[1]*Ambient[1]+Ldotn*diffuse[1]*Diffuse[1],
           emissive[2]+ambient[2]*Ambient[2]+Ldotn*diffuse[2]*Diffuse[2]
          ];
    var s=shininess*128;
    var H=unit([L[0],L[1],L[2]+1]);
    var f=Math.pow(H[0]*n[0]+H[1]*n[1]+H[2]*n[2],s);

    if(Ldotn > 0) // Phong-Blinn model of specular reflection
        p=[p[0]+f*specular[0]*Specular[0],p[1]+f*specular[1]*Specular[1],
           p[2]+f*specular[2]*Specular[2]
          ];

    return [p[0],p[1],p[2],1];
}

function render(p, I0, I1, I2, I3, P0, P1, P2, P3, flat0, flat1, flat2, flat3,
                C0, C1, C2, C3) {
    if(Distance(p) < res2) { // Patch is flat
        indices.push(I0);
        indices.push(I1);
        indices.push(I2);

        indices.push(I0);
        indices.push(I2);
        indices.push(I3);
        return;
    }

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
            c4.m3,c5.m3,c6.m3,c7.m3,c4.m5,c5.m5,c6.m5,c7.m5
           ];
    var s1=[c4.m5,c5.m5,c6.m5,c7.m5,c4.m4,c5.m4,c6.m4,c7.m4,
            c4.m2,c5.m2,c6.m2,c7.m2,p12,c3.m0,c3.m3,c3.m5
           ];
    var s2=[c7.m5,c8.m5,c9.m5,c10.m5,c7.m4,c8.m4,c9.m4,c10.m4,
            c7.m2,c8.m2,c9.m2,c10.m2,c3.m5,c3.m4,c3.m2,p15
           ];
    var s3=[c0.m5,c0.m4,c0.m2,p3,c7.m0,c8.m0,c9.m0,c10.m0,
            c7.m3,c8.m3,c9.m3,c10.m3,c7.m5,c8.m5,c9.m5,c10.m5
           ];

    var m4=s0[15];

    var n0=normal(s0[0],s0[4],s0[8],s0[12],s0[13],s0[14],s0[15]);
    if(n0 == 0.0) {
        n0=normal(s0[0],s0[4],s0[8],s0[12],s0[11],s0[7],s0[3]);
        if(n0 == 0.0) n0=normal(s0[3],s0[2],s0[1],s0[0],s0[13],s0[14],s0[15]);
    }

    var n1=normal(s1[12],s1[13],s1[14],s1[15],s1[11],s1[7],s1[3]);
    if(n1 == 0.0) {
        n1=normal(s1[12],s1[13],s1[14],s1[15],s1[2],s1[1],s1[0]);
        if(n1 == 0.0) n1=normal(s1[0],s1[4],s1[8],s1[12],s1[11],s1[7],s1[3]);
    }

    var n2=normal(s2[15],s2[11],s2[7],s2[3],s2[2],s2[1],s2[0]);
    if(n2 == 0.0) {
        n2=normal(s2[15],s2[11],s2[7],s2[3],s2[4],s2[8],s2[12]);
        if(n2 == 0.0) n2=normal(s2[12],s2[13],s2[14],s2[15],s2[2],s2[1],s2[0]);
    }

    var n3=normal(s3[3],s3[2],s3[1],s3[0],s3[4],s3[8],s3[12]);
    if(n3 == 0.0) {
        n3=normal(s3[3],s3[2],s3[1],s3[0],s3[13],s3[14],s3[15]);
        if(n3 == 0.0) n3=normal(s3[15],s3[11],s3[7],s3[3],s3[4],s3[8],s3[12]);
    }

    var n4=normal(s2[3],s2[2],s2[1],m4,s2[4],s2[8],s2[12]);

    var m0,m1,m2,m3;

    // A kludge to remove subdivision cracks, only applied the first time
    // an edge is found to be flat before the rest of the subpatch is.
    if(flat0)
        m0=[0.5*(P0[0]+P1[0]),0.5*(P0[1]+P1[1]),0.5*(P0[2]+P1[2])];
    else {
        if((flat0=Distance1(p0,p[4],p[8],p12) < res2)) {
            var u=s0[12];
            var v=s2[3];
            var e=unit([u[0]-v[0],u[1]-v[1],u[2]-v[2]]);
            m0=[0.5*(P0[0]+P1[0])+Epsilon*e[0],0.5*(P0[1]+P1[1])+Epsilon*e[1],
                0.5*(P0[2]+P1[2])+Epsilon*e[2]
               ];
        } else
            m0=s0[12];
    }

    if(flat1)
        m1=[0.5*(P1[0]+P2[0]),0.5*(P1[1]+P2[1]),0.5*(P1[2]+P2[2])];
    else {
        if((flat1=Distance1(p12,p[13],p[14],p15) < res2)) {
            var u=s1[15];
            var v=s3[0];
            var e=unit([u[0]-v[0],u[1]-v[1],u[2]-v[2]]);
            m1=[0.5*(P1[0]+P2[0])+Epsilon*e[0],0.5*(P1[1]+P2[1])+Epsilon*e[1],
                0.5*(P1[2]+P2[2])+Epsilon*e[2]
               ];
        } else
            m1=s1[15];
    }

    if(flat2)
        m2=[0.5*(P2[0]+P3[0]),0.5*(P2[1]+P3[1]),0.5*(P2[2]+P3[2])];
    else {
        if((flat2=Distance1(p15,p[11],p[7],p3) < res2)) {
            var u=s2[3];
            var v=s0[12];
            var e=unit([u[0]-v[0],u[1]-v[1],u[2]-v[2]]);
            m2=[0.5*(P2[0]+P3[0])+Epsilon*e[0],0.5*(P2[1]+P3[1])+Epsilon*e[1],
                0.5*(P2[2]+P3[2])+Epsilon*e[2]
               ];
        } else
            m2=s2[3];
    }

    if(flat3)
        m3=[0.5*(P3[0]+P0[0]),0.5*(P3[1]+P0[1]),0.5*(P3[2]+P0[2])];
    else {
        if((flat3=Distance1(p3,p[2],p[1],p0) < res2)) {
            var u=s3[0];
            var v=s1[15];
            var e=unit([u[0]-v[0],u[1]-v[1],u[2]-v[2]]);
            m3=[0.5*(P3[0]+P0[0])+Epsilon*e[0],
                0.5*(P3[1]+P0[1])+Epsilon*e[1],
                0.5*(P3[2]+P0[2])+Epsilon*e[2]
               ];
        } else
            m3=s3[0];
    }

    //  document.write(n0);      
    //  document.write(" < br > ");

    {
        /*
          var c0=new Array(4);
          var c1=new Array(4);
          var c2=new Array(4);
          var c3=new Array(4);
          var c4=new Array(4);
          
          for(var i=0; i < 4; ++i) {
          c0[i]=0.5*(C0[i]+C1[i]);
          c1[i]=0.5*(C1[i]+C2[i]);
          c2[i]=0.5*(C2[i]+C3[i]);
          c3[i]=0.5*(C3[i]+C0[i]);
          c4[i]=0.5*(c0[i]+c2[i]);
          }
       */

        var c0=color(n0);
        var c1=color(n1);
        var c2=color(n2);
        var c3=color(n3);
        var c4=color(n4);


        var i0=vertex(m0,c0);
        var i1=vertex(m1,c1);
        var i2=vertex(m2,c2);
        var i3=vertex(m3,c3);
        var i4=vertex(m4,c4);

        render(s0,I0,i0,i4,i3,P0,m0,m4,m3,flat0,false,false,flat3,
               C0,c0,c4,c3);
        render(s1,i0,I1,i1,i4,m0,P1,m1,m4,flat0,flat1,false,false,
               c0,C1,c1,c4);
        render(s2,i4,i1,I2,i2,m4,m1,P2,m2,false,flat1,flat2,false,
               c4,c1,C2,c2);
        render(s3,i3,i4,i2,I3,m3,m4,m2,P3,false,false,flat2,flat3,
               c3,c4,c2,C3);
    }
}


function draw(p) {
    var p0=p[0];
    var p3=p[3];
    var p12=p[12];
    var p15=p[15];

    epsilon=0;
    for(var i=1; i < 16; ++i)
        epsilon=Math.max(epsilon,
                         abs2([p[i][0]-p0[0],p[i][1]-p0[1],p[i][2]-p0[2]]));
    epsilon *= Fuzz2;

    var n0=normal(p3,p[2],p[1],p0,p[4],p[8],p12);
    var n1=normal(p0,p[4],p[8],p12,p[13],p[14],p15);
    var n2=normal(p12,p[13],p[14],p15,p[11],p[7],p3);
    var n3=normal(p15,p[11],p[7],p3,p[2],p[1],p0);

    var c0=color(n0);
    var c1=color(n1);
    var c2=color(n2);
    var c3=color(n3);

    var i0=vertex(p0,c0);
    var i1=vertex(p12,c1);
    var i2=vertex(p15,c2);
    var i3=vertex(p3,c3);

    render(p,i0,i1,i2,i3,p0,p12,p15,p3,false,false,false,false,
           c0,c1,c2,c3);

    drawBuffer();
}

var p=[];

function tick() {
    requestAnimationFrame(tick);
    if(redraw) {
        sceneSetup();
        mat4.translate(mvMatrix,mvMatrix,[-0.5,-0.5,-1.5]);
        setMatrixUniforms();
        for(var i=0; i < p.length; ++i)
            draw(p[i]);
        redraw=false;
    }
}

function webGLStart() {
    var canvas=document.getElementById("Asymptote");
    initGL(canvas);
    initShaders();
    gl.clearColor(0.0,0.0,0.0,1.0);
    gl.enable(gl.DEPTH_TEST);

    indexExt=gl.getExtension("OES_element_index_uint");

    gl.viewport(0,0,gl.viewportWidth,gl.viewportHeight);
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

    tick();
}


// Lighting parameters
var L = [0.447735768366173, 0.497260947684137, 0.743144825477394];
var Ambient = [0.1, 0.1, 0.1];
var Diffuse = [0.8, 0.8, 0.8, 1];
var Specular = [0.7, 0.7, 0.7, 1];
var specularfactor = 3;

// Material parameters
var emissive = [0, 0, 0, 1];
var ambient = [1, 1, 1, 1];
var diffuse = [0, 1, 0, 1];
var specular = [0.75, 0.75, 0.75, 1];
var shininess = 0.5;
