let vec4=glmat.vec4;
let mat4=glmat.mat4;
function up(delta) {
  return mat4.fromValues(1,0,0,0,
    0,1,0,0,
    0,0,1,0,
    0,delta,0,1);
}

setf(function(v,delta) {
  let out=[0,0,0,0];
  let V=[v[0]+5*Math.sin(delta),v[1]+delta,v[2],1];
  vec4.transformMat4(out,V,up(delta));
  return out;
  });
