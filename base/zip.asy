typedef import(T);

T[][] zip(...T[][] arrays) {
  return transpose(arrays);
}