# Draft V3D Standard
# Compress with gzip -9

unsigned int version
unsigned short int type [material,transform,element
                         line,curve,triangle,bezierTriangle,quad,bezierPatch, #MATERIAL ELEMENT
                         line,curve,triangle,bezierTriangle,quad,bezierPatch, #COLORED ELEMENT
                         disk,cylinder,tube,sphere, #PRIMITIVE 
                         animations,
                         2d elements,
                         
MATERIAL:
index
float[4] diffuse
float[4] emissive
float[4] specular
float[4] shininess
float[4] metallic
float[4] fresnel0

TRANSFORM:
index
double[16] 4x4 array

MATERIAL ELEMENT:
index
material index
double control point
...

ELEMENT
element index
material index
transform index
