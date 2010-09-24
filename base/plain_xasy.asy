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
  size(0,0);
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
  tempStore.push(currentpicture.copy());
  newPic=new picture;
  currentpicture=newPic;
}

void endScript()
{
  if(tempStore.length < 1) {
    abort("endScript() without matching beginScript()");
  } else {
    currentpicture=tempStore.pop();
    add(currentpicture,newPic.fit(),group=false);
  }
  shipped=false;
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
  private int[] frames;
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

void deconstruct(picture pic=currentpicture, real magnification=1)
{
  deconstruct(pic.fit(),currentpatterns,magnification,xformStack.pop);
}
