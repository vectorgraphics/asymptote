restricted bool inXasyMode=false;
bool diagnostics=false;
void report(string text)
{
 if(diagnostics)
  write(text);
}
void report(transform t)
{
 if(diagnostics)
  write(t);
}
void report(int i)
{
 if(diagnostics)
  write(i);
}
void initXasyMode()
{
    // TODO: Figure out what's wrong...
    //size(0,0);
    inXasyMode=true;
}

void exitXasyMode()
{
  inXasyMode=false;
}
private picture[] tempStore;
private picture newPic;
void startScript()
{
  /*
  tempStore.push(currentpicture.copy());
  newPic=new picture;
  currentpicture=newPic;
  */
}

void endScript()
{
  /*
  if(tempStore.length < 1) {
    abort("endScript() without matching beginScript()");
  } else {
    currentpicture=tempStore.pop();
    add(currentpicture,newPic.fit(),group=false);
  }
  shipped=false;
  */
}

// TODO: Replace this with something more elegant.
// John's code - just a working temporary.
struct keytransform {
  string key;
  transform T;
  void operator init(string key) {
    this.key=key;
  }
  void operator init(string key, transform T) {
    this.key=key;
    this.T=T;
  }
}

bool operator < (keytransform a, keytransform b) {
  return a.key < b.key;
}

struct map {
  keytransform[] M;
  void add(string key, transform T) {
    keytransform m=keytransform(key,T);
    int i=search(M,m,operator <);
    M.insert(i+1,m);
  }
  transform lookup(string key) {
    int i=search(M,keytransform(key),operator <);
    if(i >= 0 && M[i].key == key) return M[i].T;
    return identity();
  }
}


struct indexedTransform {
  int index;
  transform t;
  bool active;
  void operator init(int index, transform t, bool active=true) {
    this.index=index;
    this.t=t;
    this.active=active;
  }
}
/* 
struct keyedTransform {
    string key;
    transform transf;
    bool active;

    void operator init(string key, transform t, bool active=true) {
        this.key = key;
        this.transf = t;
        this.active = active;
    }
}

struct hashedTransformMap {
    struct transact {
      transform t;
      bool active;
      void operator init(transform t, bool active=true) {
        this.t=t;
        this.active=active;
      }
      void operator init(indexedTransform i){
        this.t=i.t;
        this.active=i.active;
      }
      void operator init() {
        this.t=identity();
        this.active=true;
      }
    }

    // private hashMap (somethigng? ) hashmap;

    transform getKey(string key) {
        return (0, 0, 1, 0, 0, 1);
    }

    void setKey(string key, transform t, bool active=true) {

    }

    bool empty() {
        return true;
    }
} */

struct framedTransformStack {
  struct transact {
    transform t;
    bool active;
    void operator init(transform t, bool active=true) {
      this.t=t;
      this.active=active;
    }
    void operator init(indexedTransform i){
      this.t=i.t;
      this.active=i.active;
    }
    void operator init() {
      this.t=identity();
      this.active=true;
    }
  }
  private transact[] stack;
  // private int[] frames;
  private int stackBase=0;
  transform pop() {
    if(stack.length == 0)
      return identity();
    else {
      transform popped=stack[0].t;
      stack.delete(0);
      report("Popped");
      report(popped);
      return popped;
    }
  }

  transform pop0() {
    if(stack.length == 0)
      return identity();
    else {
      static transform zerotransform=(0,0,0,0,0,0);
      transform popped=stack[0].active ? stack[0].t : zerotransform;
      stack.delete(0);
      report("Popped");
      report(popped);
      return popped;
    }
  }

  void push(transform t, bool Active=true) {
    report("Pushed");
    report(t);
    stack.push(transact(t,Active));
  }

  void add(... indexedTransform[] tList) {
    transact[] toPush;
    for(int a=0; a < tList.length; ++a)
      toPush[tList[a].index]=transact(tList[a]);
    for(int a=0; a < toPush.length; ++a)
      if(!toPush.initialized(a))
        toPush[a]=transact();
    report("Added");
    report(toPush.length);
    stack.append(toPush);
  }

  bool empty() {
    return stack.length == 0;
  }
}

framedTransformStack xformStack;
map xformMap;

void deconstruct(picture pic=currentpicture, real magnification=1)
{
  deconstruct(pic.fit(),currentpatterns,magnification,xformStack.pop);
}
