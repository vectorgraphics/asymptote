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
};
v3dtypes v3dtype;

struct v3dPatchData
{
    patch p;
    int matId;
    int centerIdx;
}

struct v3dfile
{
    file _xdrfile;
    int fileversion;
    surface[][] surf=new surface[][];

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

    v3dPatchData readBezierPatch()
    {
        triple[][] val=new triple[4][4];
        _xdrfile.singlereal(false);
        _xdrfile.dimension(4,4);
        val=_xdrfile;

        int centerIdx=_xdrfile;
        int matIdx=_xdrfile;

        v3dPatchData vpd;
        vpd.p=patch(val);
        vpd.matId=matIdx;
        vpd.centerIdx=centerIdx;
        return vpd;
    }

    v3dPatchData readBezierTriangle()
    {
        _xdrfile.singlereal(false);
        _xdrfile.dimension(1);


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
        int centerIdx=_xdrfile;
        int matIdx=_xdrfile;

        v3dPatchData vpd;
        vpd.p=patch(val,triangular=true);
        vpd.matId=matIdx;
        vpd.centerIdx=centerIdx;
        return vpd;
    }

    triple[] readCenters()
    {
        _xdrfile.singlereal(false);
        _xdrfile.dimension(1);
        int centerCount=_xdrfile;

        _xdrfile.dimension(centerCount);
        triple[] centersFetched=new triple[centerCount];
        if (centerCount>0)
            centersFetched=_xdrfile;
        return centersFetched;
    }

    /*
    v3dSurfaceData readBezierPatchColor()
    {
        triple[][] val;
        _xdrfile.singlereal(false);
        _xdrfile.dimension(4);
        for (int i=0;i<4;++i)
        {
            val.push(_xdrfile);
        }

        pen[] col;
        _xdrfile.dimension(4);
        for (int i=0;i<4;++i)
        {
            pen[] coli;
            for (int i=0; i<4; ++i)
            {
                coli.push(rgba(_xdrfile));
            }
            col.push(coli);
        }

        int centerIdx=_xdrfile;
        int matIdx=_xdrfile;

        v3dSurfaceData sd = v3dSurfaceData(surface(new triple[][][] {val}), mats[matIdx], centerIdx, col);
        return sd;
    }

    v3dSurfaceData readBezierTriangleColor()
    {
        triple[] val;
        _xdrfile.singlereal(false);
        _xdrfile.dimension(1);
        for (int i=0;i<10;++i)
        {
            val.push(_xdrfile);
        }

        pen[] col;
        _xdrfile.dimension(4);
        for (int i=0;i<10;++i)
        {
            col.push(rgba(_xdrfile));
        }

        int centerIdx=_xdrfile;
        int matIdx=_xdrfile;


        v3dSurfaceData sd = v3dSurfaceData(surface(val), mats[matIdx], centerIdx, col);
        return sd;
    }

*/

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
            if (ty == v3dtype.material_)
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
            else if (ty == v3dtype.centers)
            {
                centers=this.readCenters();
            }
        }

        processed=true;
        return surf;
    }
};

void _test_fn_importv3d()
{
    v3dfile xf=v3dfile("BezierTriangle.v3d");
    surface[][] arrays = xf.process();
    for (int i=0;i<arrays.length;++i)
    {
        for (int j=0;j<arrays[i].length;++j)
        {
            draw(arrays[i][j], xf.materials[j]);
        }
    }
}
_test_fn_importv3d();