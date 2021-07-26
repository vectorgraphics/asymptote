# Draft V3D Standard
# Compress with gzip -9

unsigned int version
unsigned int type [material,transform,element
                         vertex,
#MATERIAL ELEMENTS:
                         line,curve,triangle,bezierTriangle,quad,bezierPatch,
                         disk,cylinder,tube,sphere,
                         animations,
                         2d elements,

#COLORED:
                         vertex,
                         line,curve,triangle,bezierTriangle,quad,bezierPatch]
MATERIAL:
index
float[4] diffuse
float[4] emissive
float[4] specular
float shininess
float metallic
float fresnel0

TRANSFORM:
index
double[16] 4x4 array

VERTEX:
index
double x,y,z

COLORED VERTEX:
index
double x,y,z
float r,g,b,a

MATERIAL ELEMENT:
index
material index
double control point
...

ELEMENT
element index
material index
transform index
