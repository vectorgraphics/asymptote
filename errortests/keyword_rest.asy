// Keyword and rest errors.
{
  int f(string s="" ... int[] a);
  f(1,2,3 ... new int[] {4,5,6}, "hi");
  f(1,2,3 ... new int[] {4,5,6} ... new int[] {7,8,9});
  f(... new int[] {4,5,6}, "hi");
  f(... new int[] {4,5,6} ... new int[] {7,8,9});
}
{
  int f(... int[] x, int y);
  int g(... int[] x, int y) { return 7; }
  int f(string s ... int[] x, int y);
  int g(string s ... int[] x, int y) { return 7; }
  int f(int keyword x, int y);
  int g(int keyword x, int y) { return 7; }
  int f(int keyword x, int y, string z);
  int g(int keyword x, int y, string z) { return 7; }
  int f(real t, int keyword x, int y);
  int g(real t, int keyword x, int y) { return 7; }
  int f(real t, int keyword x, int y, string z);
  int g(real t, int keyword x, int y, string z) { return 7; }
}
{
  int f(int notkeyword x);
  int g(int notkeyword x) { return 7; }
  int f(real w, int notkeyword x);
  int g(real w, int notkeyword x) { return 7; }
  int f(real w, int keyword y, int notkeyword x);
  int g(real w, int keyword y, int notkeyword x) { return 7; }
  int f(real w, int notkeyword y, int keyword x);
  int g(real w, int notkeyword y, int keyword x) { return 7; }
  int f(int notkeyword x, int y, string z);
  int g(int notkeyword x, int y, string z) { return 7; }
  int f(int notkeyword x, int y);
  int g(int notkeyword x, int y) { return 7; }
  int f(int notkeyword x, int y, string z);
  int g(int notkeyword x, int y, string z) { return 7; }
  int f(real t, int notkeyword x, int y);
  int g(real t, int notkeyword x, int y) { return 7; }
  int f(real t, int notkeyword x, int y, string z);
  int g(real t, int notkeyword x, int y, string z) { return 7; }
}
