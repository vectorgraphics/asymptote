// Supakorn "Jamie" Rassameemasuang <jamievlin@outlook.com> and
// John C. Bowman

import three;
import v3dtypes;
import v3dheadertypes;

struct triangleGroup
{
  triple[] positions;
  triple[] normals;
  pen[] colors;

  int[][] positionIndices;
  int[][] normalIndices;
  int[][] colorIndices;
}

struct pixel
{
  triple point;
  real width;
}

struct CameraInformation
{
  int canvasWidth;
  int canvasHeight;
  bool absolute;

  triple b;
  triple B;
  bool orthographic;
  real angle;
  real Zoom0;
  pair viewportshift;
  pair viewportmargin;

  light light;

  void setCameraInfo()
  {
    size(canvasWidth,canvasHeight);
    triple center=0.5*(b.z+B.z)*Z;

    if(orthographic)
      currentprojection=orthographic(Z,target=center,Zoom0,
                                     viewportshift=viewportshift);
    else
      currentprojection=perspective(Z,Y,target=center,Zoom0,
                                    degrees(2.0*atan(tan(0.5*angle)/Zoom0)),
                                    viewportshift=viewportshift,
                                    autoadjust=false);
    light.specular=light.diffuse;
    currentlight=light;
  }
}

transform3 Align(real polar, real azimuth)
{
  return align(dir(degrees(polar),degrees(azimuth)));
}

struct v3dfile
{
  file xdrfile;
  int version;
  bool hasCameraInfo=false;
  CameraInformation info;
  bool singleprecision=false;

  triple[] centers;
  material[] materials;

  surface[][][] surfaces;
  path3[][][] paths3;
  triangleGroup[][][] triangles;
  pixel[][] pixels;

  void initSurface(int center, int material) {
    if(!surfaces.initialized(center))
      surfaces[center]=new surface[][];
    if(!surfaces[center].initialized(material))
      surfaces[center][material]=new surface[] {new surface};
  }

  void initPath3(int center, int material) {
    if(!paths3.initialized(center))
      paths3[center]=new path3[][];
    if(!paths3[center].initialized(material))
      paths3[center][material]=new path3[];
  }

  void initTriangleGroup(int center, int material) {
    if (!triangles.initialized(center))
      triangles[center]=new triangleGroup[][];
    if(!triangles[center].initialized(material))
      triangles[center][material]=new triangleGroup[];
  }

  void initPixel(int material) {
    if(!pixels.initialized(material))
      pixels[material]=new pixel[];
  }

  void surface(int center, int material, patch p) {
    initSurface(center,material);
    surfaces[center][material][0].s.push(p);
  }

  void primitive(int center, int material, surface s) {
    initSurface(center,material);
    surfaces[center][material].push(s);
  }

  void path3(int center, int material, path3 p) {
    initPath3(center,material);
    paths3[center][material].push(p);
  }

  void triangleGroup(int center, int material, triangleGroup g) {
    initTriangleGroup(center,material);
    triangles[center][material].push(g);
  }

  void pixel(int material, pixel P) {
    initPixel(material);
    pixels[material].push(P);
  }

  void operator init(string name)
  {
    xdrfile=input(name, mode="xdrgz");
    version=xdrfile;

    int doubleprecision=xdrfile;
    singleprecision=doubleprecision == 0;
    xdrfile.singlereal(singleprecision);
  }

  int getType()
  {
    return xdrfile;
  }

  void setCameraInfo()
  {
    if(hasCameraInfo)
      info.setCameraInfo();
  }

  pen[] readColorData(int size=4, bool alpha=true)
  {
    xdrfile.singlereal(true);

    xdrfile.dimension(alpha ? 4 : 3);
    pen[] newPen=new pen[size];
    for (int i=0; i < size; ++i)
      {
        newPen[i]=alpha ? rgba(xdrfile) : rgb((real[]) xdrfile);
      }

    xdrfile.singlereal(singleprecision);

    return newPen;
  }

  CameraInformation processHeader()
  {
    CameraInformation ci;

    int entryCount=xdrfile;
    for (int i=0;i<entryCount;++i)
      {
        int headerKey=xdrfile;
        int headerSz=xdrfile;

        if (headerKey == v3dheadertypes.canvasWidth)
          {
            ci.canvasWidth=xdrfile;
          }
        else if (headerKey == v3dheadertypes.canvasHeight)
          {
            ci.canvasHeight=xdrfile;
          }
        else if (headerKey == v3dheadertypes.absolute)
          {
            int val=xdrfile;
            ci.absolute=(val != 0);
          }
        else if (headerKey == v3dheadertypes.minBound)
          {
            ci.b=xdrfile;
          }
        else if (headerKey == v3dheadertypes.maxBound)
          {
            ci.B=xdrfile;
          }
        else if (headerKey == v3dheadertypes.orthographic)
          {
            int val=xdrfile;
            ci.orthographic=(val != 0);
          }
        else if (headerKey == v3dheadertypes.angleOfView)
          {
            ci.angle=xdrfile;
          }
        else if (headerKey == v3dheadertypes.initialZoom)
          {
            ci.Zoom0=xdrfile;
          }
        else if (headerKey==v3dheadertypes.viewportShift)
          {
            ci.viewportshift=xdrfile;
          }
        else if (headerKey==v3dheadertypes.viewportMargin)
          {
            ci.viewportmargin=xdrfile;
          }
        else if (headerKey==v3dheadertypes.background)
          {
            ci.light.background=readColorData(1)[0];
          }
        else if (headerKey==v3dheadertypes.light)
          {
            triple position=xdrfile;
            ci.light.position.push(position);
            ci.light.diffuse.push(rgba(readColorData(1,false)[0]));
          }
        else
          {
            xdrfile.dimension(headerSz);
            int[] _dmp=xdrfile;
          }
      }
    return ci;
  }

  material readMaterial() {
    xdrfile.dimension(4);
    xdrfile.singlereal(true);

    pen diffusePen=rgba(xdrfile);
    pen emissivePen=rgba(xdrfile);
    pen specularPen=rgba(xdrfile);

    xdrfile.dimension(3);
    real[] params=xdrfile;
    real shininess=params[0];
    real metallic=params[1];
    real F0=params[2];

    xdrfile.singlereal(singleprecision);

    return material(diffusePen,emissivePen,specularPen,1,shininess,
                    metallic,F0);
  }

  triple[][] readRawPatchData() {
    triple[][] val;
    xdrfile.dimension(4,4);
    val=xdrfile;
    return val;
  }

  triple[][] readRawTriangleData() {
    triple[][] val;

    for(int i=0; i < 4; ++i) {
      xdrfile.dimension(i+1);
      triple[] v=xdrfile;
      val.push(v);
    }
    return val;
  }

  void readBezierPatch() {
    triple[][] val=readRawPatchData();
    int center=xdrfile;
    int material=xdrfile;
    surface(center,material,patch(val));
  }

  void readBezierTriangle() {
    triple[][] val=readRawTriangleData();
    int center=xdrfile;
    int material=xdrfile;
    surface(center,material,patch(val,triangular=true));
  }

  triple[] readCenters() {
    int centerCount=xdrfile;
    xdrfile.dimension(centerCount);
    triple[] centersFetched;
    if (centerCount>0)
      centersFetched=xdrfile;
    return centersFetched;
  }

  void readBezierPatchColor() {
    triple[][] val=readRawPatchData();
    int center=xdrfile;
    int material=xdrfile;
    pen[] colors=readColorData(4);
    surface(center,material,patch(val,colors=colors));
  }

  void readBezierTriangleColor() {
    triple[][] val=readRawTriangleData();
    int center=xdrfile;
    int material=xdrfile;
    pen[] colors=readColorData(3);
    surface(center,material,patch(val,triangular=true,colors=colors));
  }

  void readSphere() {
    triple origin=xdrfile;
    real radius=xdrfile;

    int center=xdrfile;
    int material=xdrfile;

    surface s=shift(origin)*scale3(radius)*unitsphere;
    s.primitive=unitsphere.primitive;
    primitive(center,material,s);
  }

  void readHemisphere() {
    triple origin=xdrfile;
    real radius=xdrfile;

    int center=xdrfile;
    int material=xdrfile;

    real polar=xdrfile;
    real azimuth=xdrfile;

    surface s=shift(origin)*Align(polar,azimuth)*scale3(radius)*unithemisphere;
    s.primitive=unithemisphere.primitive;
    primitive(center,material,s);
  }

  void readDisk() {
    triple origin=xdrfile;
    real radius=xdrfile;

    int center=xdrfile;
    int material=xdrfile;

    real polar=xdrfile;
    real azimuth=xdrfile;

    surface s=shift(origin)*Align(polar,azimuth)*scale3(radius)*unitdisk;
    s.primitive=unitdisk.primitive;
    primitive(center,material,s);
  }

  void readPolygon(int n)
  {
    xdrfile.dimension(n);
    triple[] val=xdrfile;

    int center=xdrfile;
    int material=xdrfile;
    surface(center,material,patch(val));
  }

  void readPolygonColor(int n)
  {
    xdrfile.dimension(n);
    triple[] val=xdrfile;

    int center=xdrfile;
    int material=xdrfile;

    pen[] colors=readColorData(n);
    surface(center,material,patch(val,colors=colors));
  }

  void readCylinder() {
    triple origin=xdrfile;
    real radius=xdrfile;
    real height=xdrfile;

    int center=xdrfile;
    int material=xdrfile;

    real polar=xdrfile;
    real azimuth=xdrfile;

    int core=xdrfile;

    transform3 T=shift(origin)*Align(polar,azimuth)*
      scale(radius,radius,height);
    if(core != 0)
      path3(center,material,T*(O--Z));

    surface s=T*unitcylinder;
    s.primitive=unitcylinder.primitive;
    primitive(center,material,s);
  }

  void readTube() {
    xdrfile.dimension(4);
    triple[] g=xdrfile;

    real width=xdrfile;
    int center=xdrfile;
    int material=xdrfile;

    int core=xdrfile;

    if(core != 0)
      path3(center,material,g[0]..controls g[1] and g[2]..g[3]);

    surface s=tube(g[0],g[1],g[2],g[3],width);
    s.primitive=primitive(drawTube(g,width,info.b,info.B),
                          new bool(transform3 t) {
                            return unscaled(t,X,Y);
                          });
    primitive(center,material,s);
  }

  void readCurve() {
    xdrfile.dimension(4);
    triple[] points=xdrfile;

    int center=xdrfile;
    int material=xdrfile;

    path3(center,material,
          points[0]..controls points[1] and points[2]..points[3]);
  }

  void readLine() {
    xdrfile.dimension(2);
    triple[] points=xdrfile;

    int center=xdrfile;
    int material=xdrfile;

    path3(center,material,points[0]--points[1]);
  }

  void readTriangles() {
    triangleGroup g;

    int nI=xdrfile;

    int nP=xdrfile;
    xdrfile.dimension(nP);
    g.positions=xdrfile;

    int nN=xdrfile;
    xdrfile.dimension(nN);
    g.normals=xdrfile;

    int explicitNI=xdrfile;

    int nC=xdrfile;
    int explicitCI;
    if (nC > 0) {
      g.colors=readColorData(nC);
      explicitCI=xdrfile;
    }

    g.positionIndices=new int[nI][3];
    g.normalIndices=new int[nI][3];
    int[][] colorIndices;
    if (nC > 0)
      g.colorIndices=new int[nI][3];

    for (int i=0; i < nI; ++i) {
      xdrfile.dimension(3);
      g.positionIndices[i]=xdrfile;
      g.normalIndices[i]=explicitNI != 0 ? xdrfile :
        g.positionIndices[i];
      g.colorIndices[i]=nC > 0 && explicitCI != 0 ? xdrfile :
        g.positionIndices[i];
    }
    int center=xdrfile;
    int material=xdrfile;
    triangleGroup(center,material,g);
  }

  void readPixel() {
    pixel P;
    P.point=xdrfile;
    P.width=xdrfile;
    int material=xdrfile;
    pixel(material,P);
  }

  void process()
  {
    static bool processed;
    if(processed) return;

    while (!eof(xdrfile))
      {
        int ty=getType();
        if(ty == v3dtypes.header)
          {
            hasCameraInfo=true;
            info=processHeader();
          }
        else if(ty == v3dtypes.material)
          {
            materials.push(readMaterial());
          }
        else if(ty == v3dtypes.bezierPatch)
          {
            readBezierPatch();
          }
        else if(ty == v3dtypes.bezierTriangle)
          {
            readBezierTriangle();
          }
        else if(ty == v3dtypes.bezierPatchColor)
          {
            readBezierPatchColor();
          }
        else if(ty == v3dtypes.bezierTriangleColor)
          {
            readBezierTriangleColor();
          }
        else if(ty == v3dtypes.quad)
          {
            readPolygon(4);
          }
        else if(ty == v3dtypes.quadColor)
          {
            readPolygonColor(4);
          }
        else if(ty == v3dtypes.triangle)
          {
            readPolygon(3);
          }
        else if(ty == v3dtypes.triangleColor)
          {
            readPolygonColor(3);
          }
        else if(ty == v3dtypes.sphere)
          {
            readSphere();
          }
        else if(ty == v3dtypes.halfSphere)
          {
            readHemisphere();
          }
        else if(ty == v3dtypes.cylinder)
          {
            readCylinder();
          }
        else if(ty == v3dtypes.tube)
          {
            readTube();
          }
        else if(ty == v3dtypes.disk)
          {
            readDisk();
          }
        else if(ty == v3dtypes.curve)
          {
            readCurve();
          }
        else if(ty == v3dtypes.line)
          {
            readLine();
          }
        else if(ty == v3dtypes.triangles)
          {
            readTriangles();
          }
        else if(ty == v3dtypes.centers)
          {
            centers=readCenters();
          }
        else if(ty == v3dtypes.pixel)
          {
            readPixel();
          }
        else
          {
            //  abort("Unknown type:"+string(ty));
          }
      }
    processed=true;
  }
}

void importv3d(string name)
{
  if(name == stripextension(name)) name += ".v3d";
  v3dfile xf=v3dfile(name);
  xf.process();
  xf.setCameraInfo();

  for(int c=0; c < xf.surfaces.length; ++c) {
    triple center=c > 0 ? xf.centers[c-1] : O;
    render r=render(interaction(c == 0 ? Embedded : Billboard,center=center));
    surface[][] S=xf.surfaces[c];
    for(int m=0; m < S.length; ++m)
      if(S.initialized(m))
        draw(S[m],xf.materials[m],r);
  }

  for(int c=0; c < xf.paths3.length; ++c) {
    triple center=c > 0 ? xf.centers[c-1] : O;
    render r=render(interaction(c == 0 ? Embedded : Billboard,center=center));
    path3[][] G=xf.paths3[c];
    for(int m=0; m < G.length; ++m)
      if(G.initialized(m)) {
        material material=material(xf.materials[m]);
        material.p[0] += thin();
        draw(G[m],material,currentlight,r);
      }
  }

  for(int c=0;c < xf.triangles.length; ++c) {
    triple center=c > 0 ? xf.centers[c-1] : O;
    render r=render(interaction(c == 0 ? Embedded : Billboard,center=center));
    triangleGroup[][] groups=xf.triangles[c];
    for(int m=0; m < groups.length; ++m) {
      if(groups.initialized(m)) {
        material material=xf.materials[m];
        triangleGroup[] triangleGroups=groups[m];
        for(triangleGroup g : triangleGroups) {
          if(g.colors.length > 0)
            draw(g.positions,g.positionIndices,g.normals,g.normalIndices,
                 g.colors,g.colorIndices,r);
          else
            draw(g.positions,g.positionIndices,g.normals,g.normalIndices,
                 material,r);
        }
      }
    }
  }

  for(int m=0; m < xf.pixels.length; ++m) {
    if(xf.pixels.initialized(m)) {
      material material=xf.materials[m];
      pixel[] pixels=xf.pixels[m];
      for(pixel P : pixels)
        pixel(P.point,material.p[0],P.width);
    }
  }
}
