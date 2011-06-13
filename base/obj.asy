// A module for reading simple obj files with groups.
// Authors: Jens Schwaiger and John Bowman
//
// Here simple means that : 
//
// 1) all vertex statements should come before the face statements;
//
// 2) face informations with respect to texture and/or normal vectors are
// ignored;
//
// 3) face statements only contain positive numbers(no relative positions).
//
// The reading process only takes into account lines starting with "v" or
// "f" or "g"(group).

import three;

struct obj {
  surface s;
  material[] surfacepen;
  pen[] meshpen;

  path3[][] read(string datafile, bool verbose=false) {
    file in=input(datafile).word().line();
    triple[] vert;
    path3[][] g;
    g[0]=new path3[] ;
    string[] G;
    void Vertex(real x,real y ,real z) {vert.push((x,y,z));}
    void Face(int[] vertnr, int groupnr) {
      guide3 gh;
      for(int i=0; i < vertnr.length; ++i)
        gh=gh--vert[vertnr[i]-1];
      gh=gh--cycle;
      g[groupnr].push(gh);
    }
    if(verbose) write("Reading data from "+datafile+".");
    int groupnr;
    while(true) {
      string[] str=in;
      if(str.length == 0) break;
      str=sequence(new string(int i) {return split(str[i],"/")[0];},str.length);
      if(str[0] == "g" && str.length > 1) {
        int tst=find(G == str[1]);
        if(tst == -1) {
          G.push(str[1]);
          groupnr=G.length-1;
          g[groupnr]=new path3[] ;
        }
        if(tst > -1) groupnr=tst;
      }
      if(str[0] == "v") Vertex((real) str[1],(real) str[2],(real) str[3]);
      if(str[0] == "f") {
        int[] vertnr;
        for(int i=1; i < str.length; ++i) vertnr[i-1]=(int) str[i];
        Face(vertnr,groupnr);
      }
      if(eof(in)) break;
    }
    close(in);
    if(verbose) {
      write("Number of groups: ",G.length);
      write("Groups and their names:");
      write(G);
      write("Reading done.");
      write("Number of faces contained in the groups: ");
      for(int j=0; j < G.length; ++j)
        write(G[j],": ",(string) g[j].length);
    }
    return g;
  }

  void operator init(path3[][] g, material[] surfacepen, pen[] meshpen) {
    for(int i=0; i < g.length; ++i) {
      path3[] gi=g[i];
      for(int j=0; j < gi.length; ++j) {
        // Treat all faces as planar to avoid subdivision cracks.
        surface sij=surface(gi[j],planar=true);
        s.append(sij);
        this.surfacepen.append(array(sij.s.length,surfacepen[i]));
        this.meshpen.append(array(sij.s.length,meshpen[i]));
      }
    }
  }

  void operator init(string datafile, bool verbose=false,
                     material[] surfacepen, pen[] meshpen=nullpens) {
    operator init(read(datafile,verbose),surfacepen,meshpen);
  }

  void operator init(string datafile, bool verbose=false,
                     material surfacepen, pen meshpen=nullpen) {
    material[] surfacepen={surfacepen};
    pen[] meshpen={meshpen};
    surfacepen.cyclic=true;
    meshpen.cyclic=true;
    operator init(read(datafile,verbose),surfacepen,meshpen);
  }
}

obj operator * (transform3 T, obj o)
{
  obj ot;
  ot.s=T*o.s;
  ot.surfacepen=copy(o.surfacepen);
  ot.meshpen=copy(o.meshpen);
  return ot;
}

void draw(picture pic=currentpicture, obj o, light light=currentlight)
{
  draw(pic,o.s,o.surfacepen,o.meshpen,light);
}
