// module importv3d;
// Supakorn "Jamie" Rassameemasuang <rassamee@ualberta.ca>

import three;

struct v3dtypes
{
  int other=0;
  int material_=1;
  int transform_=2;
  int element=3;
  int centers=4;
  int header=5;

  int line=64;
  int triangle=65;
  int quad=66;
  int curve=128;

  int bezierTriangle=129;
  int bezierPatch=130;

  int lineColor=192;
  int triangleColor=193;
  int quadColor=194;

  int curveColor=256;
  int bezierTriangleColor=257;
  int bezierPatchColor=258;

  int triangles=512; // specify nP;nN;nC

  //primitives
  int disk=1024;
  int cylinder=1025;
  int tube=1026;
  int sphere=1027;
  int halfSphere=1028;

  int animation=2048;
  int pixels_=4096;
};
v3dtypes v3dtype;

struct indicesModes
{
    int P=0;
    int PN=1;
    int PC=2;
    int PNC=3;
}
indicesModes indicesMode;

struct _v3dheadertypes
{
  int other_=0;
  int canvasWidth=1;
  int canvasHeight=2;
  int absolute=3;
  int box1=4;
  int box2=5;
  int orthographic=6;
  int angle_=7;
  int viewportmargin=8;
  int light=9;
  int background=10;
  int viewportMargin=11;
  int zoomFactor=12;
  int zoomPinchFactor=13;
  int zoomPinchCap=14;
  int zoomStep=15;
  int shiftHoldDistance=16;
  int shiftWaitTime=17;
  int vibrateTime=18;
};

_v3dheadertypes v3dheadertypes;

struct v3dPatchData
{
    patch p;
    int matId;
    int centerIdx;
}

struct v3dPixelInfo
{
    triple point;
    real width;
}

struct v3dPixelInfoGroup
{
    v3dPixelInfo vpi;
    int matId;
}

struct v3dSingleSurface
{
    surface s;
    int matId;
    int centerIdx;
}

struct v3dPath
{
    path3 p;
    int matId;
    int centerIdx;
}

struct v3dTrianglesData
{
    triple[] positions;
    triple[] normals;

    int[][] posIndices;
    int[][] normIndices;

    pen[] colors;
    int[][] colorIndices;

    material m;
}

struct v3dTrianglesCollection
{
    triple[] positions;
    triple[] normals;

    int[][] posIndices;
    int[][] normIndices;

    v3dTrianglesData tov3dTrianglesData()
    {
        v3dTrianglesData vtd;
        vtd.positions=positions;
        vtd.normals=normals;
        vtd.posIndices=posIndices;
        vtd.normIndices=normIndices;

        vtd.colors=null;
        vtd.colorIndices=null;
        return vtd;
    }
}


struct v3dColorTrianglesCollection
{
    v3dTrianglesCollection base;
    pen[] colors;
    int[][] colorIndices;

    v3dTrianglesData tov3dTrianglesData()
    {
        v3dTrianglesData vtd;
        vtd.positions=base.positions;
        vtd.normals=base.normals;
        vtd.posIndices=base.posIndices;
        vtd.normIndices=base.normIndices;

        vtd.colors=colors;
        vtd.colorIndices=colorIndices;
        return vtd;
    }

    base.tov3dTrianglesData=tov3dTrianglesData;
}

v3dTrianglesCollection operator cast(v3dColorTrianglesCollection vctc)
{
    return vctc.base;
}

struct v3dTriangleGroup
{
    v3dTrianglesCollection c;
    int matId;
}

struct CameraInformation
{
    int canvasWidth;
    int canvasHeight;
    bool absolute;

    triple b1;
    triple b2;
    bool orthographic;
    real angle;
}

transform3 Align(real polar, real azimuth)
{
  return align(dir(degrees(polar),degrees(azimuth)));
}

struct v3dSurfaceData
{
    bool hasCenter;
    triple center;
    material m;
    surface s;
}

struct v3dPathData
{
    bool hasCenter;
    triple center;
    material m;
    path3 p;
}

struct v3dPixelData
{
    triple point;
    real width;
    material m;
}

struct v3dfile
{
    file _xdrfile;
    int fileversion;
    surface[][] surf=new surface[][];
    path3[][][] paths=new path3[][][];
    v3dTrianglesCollection[][] triangles=new v3dTrianglesCollection[][];
    v3dPixelInfo[][] pixels=new v3dPixelInfo[][];
    bool hasCameraInfo=false;
    CameraInformation info;

    material[] materials=new material[];
    triple[] centers;
    bool processed=false;

    void operator init(string name)
    {
        _xdrfile=input(name, mode="xdr");
        fileversion=_xdrfile;
    }

    int getType()
    {
        return _xdrfile;
    }

    CameraInformation processHeader()
    {
        CameraInformation ci;

        int entryCount=_xdrfile;
        for (int i=0;i<entryCount;++i)
        {
            int headerKey=_xdrfile;
            int headerSz=_xdrfile;

            if (headerKey == v3dheadertypes.canvasWidth)
            {
                ci.canvasWidth=_xdrfile;
            }
            else if (headerKey == v3dheadertypes.canvasHeight)
            {
                ci.canvasHeight=_xdrfile;
            }
            else if (headerKey == v3dheadertypes.absolute)
            {
                int val=_xdrfile;
                ci.absolute=(val != 0);
            }
            else if (headerKey == v3dheadertypes.box1)
            {
                ci.b1=_xdrfile;
            }
            else if (headerKey == v3dheadertypes.box2)
            {
                ci.b2=_xdrfile;
            }
            else if (headerKey == v3dheadertypes.orthographic)
            {
                int val=_xdrfile;
                ci.orthographic=(val != 0);
            }
            else if (headerKey == v3dheadertypes.angle_)
            {
                ci.angle=_xdrfile;
            }
            else
            {
                _xdrfile.dimension(headerSz);
                int[] _dmp=_xdrfile;
            }
        }
        return ci;
    }

    material readMaterial()
    {
        _xdrfile.dimension(4);
        _xdrfile.singlereal(true);

        pen diffusePen=rgba(_xdrfile);
        pen emissivePen=rgba(_xdrfile);
        pen specularPen=rgba(_xdrfile);
        real[] params=_xdrfile;

        _xdrfile.singlereal(false);

        real shininess=params[0];
        real metallic=params[1];
        real F0=params[2];

        return material(diffusePen,emissivePen,specularPen,1.0,shininess,metallic,F0);
    }

    pen[] readColorData(int size=4)
    {
        _xdrfile.singlereal(true);

        _xdrfile.dimension(4);
        pen[] newPen=new pen[size];
        for (int i=0;i<size;++i)
        {
            newPen[i]=rgba(_xdrfile);
        }

        _xdrfile.singlereal(false);

        return newPen;
    }

    triple[][] readRawPatchData()
    {
        triple[][] val=new triple[4][4];
        _xdrfile.dimension(4,4);
        val=_xdrfile;
        return val;
    }

    triple[][] readRawTriangleData()
    {
        triple[][] val=new triple[][];

        for (int i=0;i<4;++i)
        {
            triple subval[] = new triple[i+1];
            for (int j=0;j<=i;++j)
            {
                subval[j]=_xdrfile;
            }
            val.push(subval);
        }
        return val;
    }

    v3dPatchData readBezierPatch()
    {
        triple[][] val=readRawPatchData();
        int centerIdx=_xdrfile;
        int matIdx=_xdrfile;

        triple Min=_xdrfile;
        triple Max=_xdrfile;

        v3dPatchData vpd;
        vpd.p=patch(val);
        vpd.matId=matIdx;
        vpd.centerIdx=centerIdx;
        return vpd;
    }

    v3dPatchData readBezierTriangle()
    {
        triple[][] val=readRawTriangleData();
        int centerIdx=_xdrfile;
        int matIdx=_xdrfile;

        triple Min=_xdrfile;
        triple Max=_xdrfile;

        v3dPatchData vpd;
        vpd.p=patch(val,triangular=true);
        vpd.matId=matIdx;
        vpd.centerIdx=centerIdx;
        return vpd;
    }

    triple[] readCenters()
    {
        int centerCount=_xdrfile;
        _xdrfile.dimension(centerCount);
        triple[] centersFetched;
        if (centerCount>0)
            centersFetched=_xdrfile;
        return centersFetched;
    }

    v3dPatchData readBezierPatchColor()
    {
        triple[][] val=readRawPatchData();
        int centerIdx=_xdrfile;
        int matIdx=_xdrfile;
        pen[] colData=readColorData(4);

        triple Min=_xdrfile;
        triple Max=_xdrfile;

        v3dPatchData vpd;
        vpd.p=patch(val,colors=colData);
        vpd.matId=matIdx;
        vpd.centerIdx=centerIdx;
        return vpd;
    }

    v3dPatchData readBezierTriangleColor()
    {
        triple[][] val=readRawTriangleData();
        int centerIdx=_xdrfile;
        int matIdx=_xdrfile;
        pen[] colData=readColorData(3);

        triple Min=_xdrfile;
        triple Max=_xdrfile;

        v3dPatchData vpd;
        vpd.p=patch(val,triangular=true,colors=colData);
        vpd.matId=matIdx;
        vpd.centerIdx=centerIdx;
        return vpd;
    }

    v3dSingleSurface readSphere()
    {
        triple center=_xdrfile;
        real radius=_xdrfile;

        int centerIdx=_xdrfile;
        int matIdx=_xdrfile;

        v3dSingleSurface vss;
        vss.s=shift(center)*scale3(radius)*unitsphere;
        vss.matId=matIdx;
        vss.centerIdx=centerIdx;
        return vss;
    }

    v3dSingleSurface readHalfSphere()
    {
        triple center=_xdrfile;
        real radius=_xdrfile;

        int centerIdx=_xdrfile;
        int matIdx=_xdrfile;

        real polar=_xdrfile;
        real azimuth=_xdrfile;

        v3dSingleSurface vss;
        vss.s=shift(center)*Align(polar,azimuth)*scale3(radius)*unithemisphere;
        vss.matId=matIdx;
        vss.centerIdx=centerIdx;
        return vss;
    }

    v3dSingleSurface readCylinder()
    {
        triple center=_xdrfile;
        real radius=_xdrfile;
        real height=_xdrfile;

        int centerIdx=_xdrfile;
        int matIdx=_xdrfile;

        real polar=_xdrfile;
        real azimuth=_xdrfile;

        int core=_xdrfile;

        v3dSingleSurface vss;
        vss.s=shift(center)*Align(polar,azimuth)*scale(radius,radius,height)*unitcylinder;
        vss.matId=matIdx;
        vss.centerIdx=centerIdx;

        return vss;
    }

    v3dSingleSurface readDisk()
    {
        triple center=_xdrfile;
        real radius=_xdrfile;

        int centerIdx=_xdrfile;
        int matIdx=_xdrfile;

        real polar=_xdrfile;
        real azimuth=_xdrfile;

        v3dSingleSurface vss;
        vss.s=shift(center)*Align(polar,azimuth)*scale3(radius)*unitdisk;
        vss.matId=matIdx;
        vss.centerIdx=centerIdx;

        return vss;
    }

    v3dPatchData readQuad()
    {
        triple[] val;
        _xdrfile.dimension(4);
        val=_xdrfile;

        int centerIdx=_xdrfile;
        int matIdx=_xdrfile;

        triple Min=_xdrfile;
        triple Max=_xdrfile;

        v3dPatchData vpd;
        vpd.p=patch(val);
        vpd.matId=matIdx;
        vpd.centerIdx=centerIdx;

        return vpd;
    }

    v3dPatchData readQuadColor()
    {
        triple[] val;
        _xdrfile.dimension(4);
        val=_xdrfile;

        int centerIdx=_xdrfile;
        int matIdx=_xdrfile;

        pen[] colData=readColorData(4);

        triple Min=_xdrfile;
        triple Max=_xdrfile;

        v3dPatchData vpd;
        vpd.p=patch(val,colors=colData);
        vpd.matId=matIdx;
        vpd.centerIdx=centerIdx;

        return vpd;
    }

    v3dPatchData readTriangle()
    {
        _xdrfile.dimension(3);
        triple[] val=_xdrfile;

        int centerIdx=_xdrfile;
        int matIdx=_xdrfile;

        triple Min=_xdrfile;
        triple Max=_xdrfile;

        v3dPatchData vpd;
        vpd.p=patch(val[0]--val[1]--val[2]--cycle);
        vpd.matId=matIdx;
        vpd.centerIdx=centerIdx;

        return vpd;
    }

    v3dPatchData readTriangleColor()
    {
        triple[] val;
        _xdrfile.dimension(3);
        val=_xdrfile;

        int centerIdx=_xdrfile;
        int matIdx=_xdrfile;

        pen[] colData=readColorData(3);

        triple Min=_xdrfile;
        triple Max=_xdrfile;

        v3dPatchData vpd;
        vpd.p=patch(val[0]--val[1]--val[2]--cycle,colors=colData);
        vpd.matId=matIdx;
        vpd.centerIdx=centerIdx;

        return vpd;
    }

    void addToPathData(v3dPath vp)
    {
        if (!paths.initialized(vp.centerIdx))
        {
            paths[vp.centerIdx]=new path3[][];
        }
        if (!surf[vp.centerIdx].initialized(vp.matId))
        {
            paths[vp.centerIdx][vp.matId]=new path3[];
        }
        paths[vp.centerIdx][vp.matId].push(vp.p);
    }

    void addToTriangleData(v3dTriangleGroup vp)
    {
        if (!triangles.initialized(vp.matId))
        {
            triangles[vp.matId]=new v3dTrianglesCollection[];
        }
        triangles[vp.matId].push(vp.c);
    }

    void addToPixelData(v3dPixelInfoGroup vpig)
    {
        if (!pixels.initialized(vpig.matId))
        {
            pixels[vpig.matId]=new v3dPixelInfo[];
        }
        pixels[vpig.matId].push(vpig.vpi);
    }

    v3dSingleSurface readTube()
    {
        triple[] g=new triple[4];
        _xdrfile.dimension(4);
        g=_xdrfile;

        real width=_xdrfile;
        int centerIdx=_xdrfile;
        int matIdx=_xdrfile;
        triple Min=_xdrfile;
        triple Max=_xdrfile;

        int core=_xdrfile;

        if (core != 0)
        {
            v3dPath vp;
            vp.p=g[0]..g[1]..g[2]..g[3];
            vp.matId=matIdx;
            vp.centerIdx=centerIdx;
            addToPathData(vp);
        }

        v3dSingleSurface vss;
        vss.s=tube(g[0],g[1],g[2],g[3],width);
        vss.matId=matIdx;
        vss.centerIdx=centerIdx;
        return vss;
    }

    v3dPath readCurve()
    {
        _xdrfile.dimension(4);
        triple[] points=new triple[4];
        points=_xdrfile;

        int centerIdx=_xdrfile;
        int matIdx=_xdrfile;
        triple Min=_xdrfile;
        triple Max=_xdrfile;

        v3dPath vp;
        vp.p=points[0]..points[1]..points[2]..points[3];
        vp.matId=matIdx;
        vp.centerIdx=centerIdx;
        return vp;
    }

    v3dPath readLine()
    {
        _xdrfile.dimension(2);
        triple[] points=new triple[2];
        points=_xdrfile;

        int centerIdx=_xdrfile;
        int matIdx=_xdrfile;
        triple Min=_xdrfile;
        triple Max=_xdrfile;

        v3dPath vp;
        vp.p=points[0]--points[1];
        vp.matId=matIdx;
        vp.centerIdx=centerIdx;
        return vp;
    }

    v3dTriangleGroup readTriangles()
    {
        v3dTriangleGroup vtg;

        int nP=_xdrfile;
        _xdrfile.dimension(nP);
        vtg.c.positions=new triple[nP];
        vtg.c.positions=_xdrfile;

        int nN=_xdrfile;
        _xdrfile.dimension(nN);
        vtg.c.normals=new triple[nN];
        vtg.c.normals=_xdrfile;

        int nC=_xdrfile;
        pen[] colors;
        int[][] colorIndices;
        if (nC > 0)
        {
            colors=readColorData(nC);
        }

        int nI=_xdrfile;
        vtg.c.posIndices=new int[nI][3];
        vtg.c.normIndices=new int[nI][3];
        if (nC > 0)
        {
            colorIndices=new int[nI][3];
        }

        for (int i=0;i<nI;++i)
        {
            _xdrfile.dimension(3);
            int mode=_xdrfile;
            vtg.c.posIndices[i]=_xdrfile;

            if (mode == indicesMode.PN)
            {
                vtg.c.normIndices[i]=_xdrfile;
                if (nC > 0)
                {
                    colorIndices[i]=vtg.c.posIndices[i];
                }
            }
            else if (mode == indicesMode.PC)
            {
                vtg.c.normIndices[i]=vtg.c.posIndices[i];
                if (nC > 0)
                {
                    colorIndices[i]=_xdrfile;
                }
            }
            else if (mode == indicesMode.PNC)
            {
                vtg.c.normIndices[i]=_xdrfile;
                if (nC > 0)
                {
                    colorIndices[i]=_xdrfile;
                }
            }
            else
            {
                vtg.c.normIndices[i]=vtg.c.posIndices[i];
                if (nC > 0)
                {
                    colorIndices[i]=vtg.c.posIndices[i];
                }
            }
        }

        int matId=_xdrfile;
        triple Min=_xdrfile;
        triple Max=_xdrfile;
        vtg.matId=_xdrfile;

        if (nC > 0)
        {
            v3dColorTrianglesCollection vctc;
            vctc.base=vtg.c;
            vctc.colors=colors;
            vctc.colorIndices=colorIndices;

            vtg.c=vctc;
        }
        return vtg;
    }

    v3dPixelInfoGroup readPixel()
    {
        v3dPixelInfoGroup vpig;
        vpig.vpi.point=_xdrfile;
        vpig.vpi.width=_xdrfile;
        vpig.matId=_xdrfile;

        triple Min=_xdrfile;
        triple Max=_xdrfile;

        return vpig;
    }

    void addToSurfaceData(v3dSingleSurface vp)
    {
        if (!surf.initialized(vp.centerIdx))
        {
            surf[vp.centerIdx]=new surface[];
        }
        if (!surf[vp.centerIdx].initialized(vp.matId))
        {
            surface s;
            surf[vp.centerIdx][vp.matId]=s;
        }
        surf[vp.centerIdx][vp.matId].append(vp.s);
    }

    void addToSurfaceData(v3dPatchData vp)
    {
        if (!surf.initialized(vp.centerIdx))
        {
            surf[vp.centerIdx]=new surface[];
        }
        if (!surf[vp.centerIdx].initialized(vp.matId))
        {
            surface s;
            surf[vp.centerIdx][vp.matId]=s;
        }
        surf[vp.centerIdx][vp.matId].push(vp.p);
    }

    surface[][] process()
    {
        if (processed)
        {
            return surf;
        }

        while (!eof(_xdrfile))
        {
            int ty=getType();
            if (ty == v3dtype.header)
            {
                hasCameraInfo=true;
                info=this.processHeader();
            }
            else if (ty == v3dtype.material_)
            {
                materials.push(this.readMaterial());
            }
            else if (ty == v3dtype.bezierPatch)
            {
                addToSurfaceData(this.readBezierPatch());
            }
            else if (ty == v3dtype.bezierTriangle)
            {
                addToSurfaceData(this.readBezierTriangle());
            }
            else if (ty == v3dtype.bezierPatchColor)
            {
                addToSurfaceData(this.readBezierPatchColor());
            }
            else if (ty == v3dtype.bezierTriangleColor)
            {
                addToSurfaceData(this.readBezierTriangleColor());
            }
            else if (ty == v3dtype.quad)
            {
                addToSurfaceData(this.readQuad());
            }
            else if (ty == v3dtype.quadColor)
            {
                addToSurfaceData(this.readQuadColor());
            }
            else if (ty == v3dtype.triangle)
            {
                addToSurfaceData(this.readTriangle());
            }
            else if (ty == v3dtype.triangleColor)
            {
                addToSurfaceData(this.readTriangleColor());
            }
            else if (ty == v3dtype.sphere)
            {
                addToSurfaceData(this.readSphere());
            }
            else if (ty == v3dtype.halfSphere)
            {
                addToSurfaceData(this.readHalfSphere());
            }
            else if (ty == v3dtype.cylinder)
            {
                addToSurfaceData(this.readCylinder());
            }
            else if (ty == v3dtype.tube)
            {
                addToSurfaceData(this.readTube());
            }
            else if (ty == v3dtype.disk)
            {
                addToSurfaceData(this.readDisk());
            }
            else if (ty == v3dtype.curve)
            {
                addToPathData(this.readCurve());
            }
            else if (ty == v3dtype.line)
            {
                addToPathData(this.readLine());
            }
            else if (ty == v3dtype.triangles)
            {
                addToTriangleData(this.readTriangles());
            }
            else if (ty == v3dtype.centers)
            {
                centers=this.readCenters();
            }
            else if (ty == v3dtype.pixels_)
            {
                addToPixelData(this.readPixel());
            }
            else
            {
                // report error?
            }
        }

        processed=true;
        return surf;
    }

    v3dSurfaceData[] generateSurfaceList()
    {
        if (!processed)
        {
            process();
        }

        v3dSurfaceData[] vsdFinal;
        for (int i=0;i<surf.length;++i)
        {
            if (surf.initialized(i))
            {
                for (int j=0;j<surf[i].length;++j)
                {
                    if (surf[i].initialized(j))
                    {
                        v3dSurfaceData vsd;
                        vsd.s=surf[i][j];
                        vsd.m=materials[j];
                        vsd.hasCenter=i > 0;
                        if(vsd.hasCenter)
                          vsd.center=centers[i-1];
                        vsdFinal.push(vsd);
                    }
                }
            }
        }
        return vsdFinal;
    }

    v3dPathData[] generatePathList()
    {
        if (!processed)
        {
            process();
        }

        v3dPathData[] vpdFinal;
        for (int i=0;i<paths.length;++i)
        {
            if (paths.initialized(i))
            {
                for (int j=0;j<paths[i].length;++j)
                {
                    if (paths[i].initialized(j))
                    {
                        for (path3 p: paths[i][j])
                        {
                            v3dPathData vpd;
                            vpd.p=p;
                            vpd.m=materials[j];
                            vpd.hasCenter=i > 0;
                            if(vpd.hasCenter)
                              vpd.center=centers[i-1];
                            vpdFinal.push(vpd);
                        }
                    }
                }
            }
        }
        return vpdFinal;
    }

    v3dPixelData[] generatePixelList()
    {
        if (!processed)
        {
            process();
        }

        v3dPixelData[] vpdFinal;
        for (int j=0;j<pixels.length;++j)
        {
            if (pixels.initialized(j))
            {
                for (v3dPixelInfo pi: pixels[j])
                {
                    v3dPixelData vpd;
                    vpd.point=pi.point;
                    vpd.width=pi.width;
                    vpd.m=materials[j];
                    vpdFinal.push(vpd);
                }
            }
        }
        return vpdFinal;
    }

    v3dTrianglesData[] generateTrianglesList()
    {
        if (!processed)
        {
            process();
        }

        v3dTrianglesData[] vtdFinal;
        for (int j=0;j<triangles.length;++j)
        {
            if (triangles.initialized(j))
            {
                for (v3dTrianglesCollection pi: triangles[j])
                {
                    v3dTrianglesData vtd=pi.tov3dTrianglesData();
                    vtd.m=materials[j];
                    vtdFinal.push(vtd);
                }
            }
        }
        return vtdFinal;
    }
};

void _test_fn_importv3d(string name)
{
  v3dfile xf=v3dfile(name);
  v3dSurfaceData[] vsd=xf.generateSurfaceList();
  for(v3dSurfaceData vs : vsd)
    draw(vs.s,vs.m,render(interaction(vs.hasCenter ? Billboard : Embedded,center=vs.center)));
}

// _test_fn_importv3d("sph.v3d");
