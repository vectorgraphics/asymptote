typedef import(Key, Value);

struct keyValue {
  Key key;
  Value value;
  void operator init(Key key) {
    this.key=key;
  }
  void operator init(Key key, Value value) {
    this.key=key;
    this.value=value;
  }
}

// Map keys to values, defaulting to the value default.

struct map {
  keyValue[] M;
  int Default;

  void operator init(Value Default) {
    this.Default=Default;
  }

  bool operator < (keyValue a, keyValue b) {return a.key < b.key;}

  void add(Key key, Value value) {
    keyValue m=keyValue(key,value);
    M.insert(search(M,m,operator <)+1,m);
  }
  Value lookup(Key key) {
    int i=search(M,keyValue(key),operator <);
    if(i >= 0 && M[i].key == key) return M[i].value;
    return Default;
  }
}
