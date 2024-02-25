from pureset(int) access
    Set_T as set_int,
    operator cast;

struct Set_smallPositiveInt {
  bool[] buffer = new bool[];

  int size() {
    return sum(buffer);
  }

  bool empty() {
    return all(!buffer);
  }

  bool contains(int item) {
    if (item < 0 || item >= buffer.length) {
      return false;
    }
    return buffer[item];
  }

  bool insert(int item) {
    if (item < 0) {
      return false;
    }
    while (item >= buffer.length) {
      buffer.push(false);
    }
    if (buffer[item]) {
      return false;
    }
    buffer[item] = true;
    return true;
  }

  int replace(int item) {
    if (item < 0) {
      return -1;
    }
    while (item >= buffer.length) {
      buffer.push(false);
    }
    if (buffer[item]) {
      return item;
    }
    buffer[item] = true;
    return -1;
  }

  int get(int item) {
    if (item < 0 || item >= buffer.length) {
      return -1;
    }
    if (buffer[item]) {
      return item;
    }
    return -1;
  }

  bool delete(int item) {
    if (item < 0 || item >= buffer.length) {
      return false;
    }
    if (buffer[item]) {
      buffer[item] = false;
      return true;
    }
    return false;
  }

  void foreach(bool process(int item)) {
    for (int i = 0; i < buffer.length; ++i) {
      if (buffer[i]) {
        if (!process(i)) {
          return;
        }
      }
    }
  }

}

Set_int operator cast(Set_smallPositiveInt set) {
  Set_int result = new Set_int;
  result.size = set.size;
  result.empty = set.empty;
  result.contains = set.contains;
  result.insert = set.insert;
  result.replace = set.replace;
  result.get = set.get;
  result.delete = set.delete;
  result.foreach = set.foreach;
  return result;
}