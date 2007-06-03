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
    return(stack.length > stackBase) ? stack.pop() : identity();
  }

  transform push(transform t) {
    return stack.push(t);
  }

  void enterFrame() {
    frames.push(stackBase);
    stackBase=stack.length;
  }

  void leaveFrame() {
    if(stackBase < stack.length)
      stack.delete(stackBase,stack.length-1);
    stackBase=(stack.length > 0) ? frames.pop() : 0;
  }

  void add(... indexedTransform[] tList) {
    transform[] toPush;
    for(int a=0; a < tList.length; ++a)
      toPush[tList[a].index]=tList[a].t;
    for(int a=0; a < toPush.length; ++a)
      if(!toPush.initialized(a))
	toPush[a]=identity();
    stack.append(toPush);
  }
}

framedTransformStack xformStack;
