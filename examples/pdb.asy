import three;
import cpkcolors;

// A sample Protein Data Bank file for this example is available from
// http://ndbserver.rutgers.edu/files/ftp/NDB/coordinates/na-biol/100d.pdb1

currentlight=White;
//currentlight=nolight;

defaultrender.merge=true;  // Fast low-quality rendering
//defaultrender.merge=false; // Slow high-quality rendering
bool pixel=false; // Set to true to draw dots as pixels.
real width=6;

size(200);
currentprojection=perspective(30,30,15);

pen chainpen=green;
pen hetpen=purple;

string filename="100d.pdb1";
//string filename=getstring("filename");

string prefix=stripextension(filename);
file data=input(filename);

pen color(string e) 
{
  e=replace(e," ","");
  int n=length(e);
  if(n < 1) return currentpen;
  if(n > 1) e=substr(e,0,1)+downcase(substr(e,1,n-1));
  int index=find(Element == e);
  if(index < 0) return currentpen;
  return rgb(Hexcolor[index]);
}	

// ATOM
string[] name,altLoc,resName,chainID,iCode,element,charge;
int[] serial,resSeq;
real[][] occupancy,tempFactor;

bool newchain=true;

struct bond 
{
  int i,j;
  void operator init(int i, int j) {
    this.i=i;
    this.j=j;
  }
}

bond[] bonds;

struct atom 
{
  string name;
  triple v;
  void operator init(string name, triple v) {
    this.name=name;
    this.v=v;
  }
}

struct chain
{
  int[] serial;
  atom[] a;
}

int[] serials;
chain[] chains;
atom[] atoms;

while(true) {
  string line=data;
  if(eof(data)) break;
  string record=replace(substr(line,0,6)," ","");
  if(record == "TER") {newchain=true; continue;}
  bool ATOM=record == "ATOM";
  bool HETATOM=record == "HETATM";
  int serial;

  atom a;
  if(ATOM || HETATOM) {
    serial=(int) substr(line,6,5);
    a.name=substr(line,76,2);
    a.v=((real) substr(line,30,8),
	 (real) substr(line,38,8),
	 (real) substr(line,46,8));
  }
  if(ATOM) {
    if(newchain) {
      chains.push(new chain);
      newchain=false;
    }
    chain c=chains[chains.length-1];
    c.serial.push(serial);
    c.a.push(a);
    continue;
  }
  if(HETATOM) {
    serials.push(serial);
    atoms.push(a);
  }
  if(record == "CONECT") {
    int k=0;
    int i=(int) substr(line,6,5);
    while(true) {
      string s=replace(substr(line,11+k,5)," ","");
     if(s == "") break;
      k += 5;
      int j=(int) s;
      if(j <= i) continue;
      bonds.push(bond(i,j));
     }
  }
}

write("Number of atomic chains: ",chains.length);

int natoms;
begingroup3("chained");
for(chain c : chains) {
  for(int i=0; i < c.a.length-1; ++i)
    draw(c.a[i].v--c.a[i+1].v,chainpen,currentlight);
  for(atom a : c.a)
    if(pixel)
      pixel(a.v,color(a.name),width);
    else
      dot(a.v,color(a.name),currentlight);
  natoms += c.a.length;
}
endgroup3();

write("Number of chained atoms: ",natoms);
write("Number of hetero atoms: ",atoms.length);

begingroup3("hetero");
for(atom h : atoms)
  if(pixel)
    pixel(h.v,color(h.name),width);
  else
    dot(h.v,color(h.name),currentlight);
endgroup3();

write("Number of hetero bonds: ",bonds.length);

begingroup3("bonds");
for(bond b : bonds) {
  triple v(int i) {return atoms[find(serials == i)].v;}
  draw(v(b.i)--v(b.j),hetpen,currentlight);
}
endgroup3();

string options;
string viewfilename=prefix+".views";

if(!error(input(viewfilename,check=false)))
  options="3Dviews="+viewfilename;

shipout(prefix,options=options);
currentpicture.erase();
