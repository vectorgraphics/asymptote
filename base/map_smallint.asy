typedef import(T);

from genericpair(K=int, V=T) access
    Pair_K_V as Pair_int_T,
    operator >>,
    alias;

from puremap(K=int, V=T) access
    Map_K_V as Map_int_T;

struct Map_smallint_T {
  private T[] buffer;
  private T emptyresponse;
  private bool isEmptyResponse(T response);
  private int size = 0;
  int size() { return size; }
  bool contains(int key) {
    return key >= 0 && key < buffer.length && !isEmptyResponse(buffer[key]);
  }
  T get(int key) {
    if (key < 0 || key >= buffer.length) return emptyresponse;
    return buffer[key];
  }
  void forEach(bool process(int key, T value)) {
    for (int i = 0; i < buffer.length; ++i) {
      if (!isEmptyResponse(buffer[i]) && !process(i, buffer[i])) return;
    }
  }
  T put(Pair_int_T kv) {
    unravel kv;
    if (k < 0) return emptyresponse;
    while (k >= buffer.length) {
      buffer.push(emptyresponse);
    }
    T response = buffer[k];
    buffer[k] = v;
    if (isEmptyResponse(response) && !isEmptyResponse(v)) ++size;
    else if (!isEmptyResponse(response) && isEmptyResponse(v)) --size;
    return response;
  }
  T pop(int key) {
    if (key < 0 || key >= buffer.length) return emptyresponse;
    T response = buffer[key];
    buffer[key] = emptyresponse;
    if (!isEmptyResponse(response)) --size;
    return response;
  }

  void operator init(T emptyresponse, bool isEmptyResponse(T)) {
    this.emptyresponse = emptyresponse;
    this.isEmptyResponse = isEmptyResponse;
    size = 0;
  }
}

Map_int_T operator cast(Map_smallint_T map) {
  Map_int_T result = new Map_int_T;
  result.size = map.size;
  result.contains = map.contains;
  result.get = map.get;
  result.forEach = map.forEach;
  result.put = map.put;
  result.pop = map.pop;
  return result;
}

Map_int_T makeMapSmallint(T emptyresponse, bool isEmptyResponse(T)) {
  return Map_smallint_T(emptyresponse, isEmptyResponse);
}