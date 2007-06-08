restricted bool inXasyMode = false;
private bool diagnostics = false;
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
  inXasyMode = true;
}

void exitXasyMode()
{

}
void startScript()
{
}

void endScript()
{
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
    if(!inXasyMode) return identity();
    //transform popped = (stack.length > stackBase) ? stack.pop() : identity();
    transform popped = stack[0];
    stack.delete(0);
    report("Popped");
    report(popped);
    return popped;
  }

  transform push(transform t) {
    report("Pushed");
    report(t);
    return stack.push(t);
  }

  void enterFrame() {
    return;
    frames.push(stackBase);
    stackBase=stack.length;
    report("entered frame");
  }

  void leaveFrame() {
    return;
    if(stackBase < stack.length)
      stack.delete(stackBase,stack.length-1);
    stackBase=(stack.length > 0) ? frames.pop() : 0;
    report("left frame");
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
