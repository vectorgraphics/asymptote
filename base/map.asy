// Create a struct <name> parameterized by types <key> and <value>,
// that maps keys to values, defaulting to the value in <default>.
void mapTemplate(string name, string key, string value, string default)
{
  type(key,"Key");
  type(value,"Value");
  eval("Value default="+default,true);

  eval("
  struct keyValue {
    Key key;
    Value T;
    void operator init(Key key) {
      this.key=key;
    }
    void operator init(Key key, Value T) {
      this.key=key;
      this.T=T;
    }
  }

  struct map {
    keyValue[] M;
    bool operator < (keyValue a, keyValue b) {return a.key < b.key;}

    void add(Key key, Value T) {
      keyValue m=keyValue(key,T);
      M.insert(search(M,m,operator <)+1,m);
    }
    Value lookup(Key key) {
      int i=search(M,keyValue(key),operator <);
      if(i >= 0 && M[i].key == key) return M[i].T;
      return default;
    }
  }
",true);

  type("map",name);
}

