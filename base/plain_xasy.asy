restricted bool inXasyMode = false;
bool diagnostics = false;
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
  inXasyMode = true;
  settings.deconstruct = 0;
}

void exitXasyMode()
{
  inXasyMode = false;
}
private picture[] tempStore;
private picture newPic;
void startScript()
{
  tempStore.push(currentpicture.copy());
  newPic = new picture;
  currentpicture = newPic;
}

void endScript()
{
  if(tempStore.length < 1)
  {
    write("Error: endscript() without matching beginScript()");
  } else {
    currentpicture = tempStore.pop();
    add(currentpicture,newPic.fit(),group=false);
  }
  shipped = false;
}

struct indexedTransform {
  int index;
  transform t;
  static indexedTransform indexedTransform(int index,transform t) {
    indexedTransform nt=new indexedTransform;
    nt.index=index;
    nt.t=t;
    return nt;
  }
}

struct framedTransformStack {
  private transform[] stack;
  private int[] frames;
  private int stackBase=0;
  transform pop() {
    if(stack.length == 0)
      return identity();
    else
    {
      transform popped = stack[0];
      stack.delete(0);
      report("Popped");
      report(popped);
      return popped;
    }
  }

  void push(transform t) {
    report("Pushed");
    report(t);
    stack.push(t);
  }

  void add(... indexedTransform[] tList) {
    transform[] toPush;
    for(int a=0; a < tList.length; ++a)
      toPush[tList[a].index]=tList[a].t;
    for(int a=0; a < toPush.length; ++a)
      if(!toPush.initialized(a))
	toPush[a]=identity();
    report("Added");
    report(toPush.length);
    stack.append(toPush);
  }
}

framedTransformStack xformStack;

void deconstruct(string prefix="out", picture pic=currentpicture,
		 real magnification=1.0, bool countonly=false)
{
  settings.deconstruct=magnification;
  deconstruct(prefix,pic.fit(),patterns,xformStack.pop,countonly);
}
