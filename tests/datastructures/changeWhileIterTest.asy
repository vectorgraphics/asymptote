import TestLib;

StartTest('Change collections.map values while iterating');
{
  from collections.map(K=string, V=int) access
      Map_K_V as Map_string_int,
      NaiveMap_K_V as NaiveMap_string_int;
  Map_string_int map = NaiveMap_string_int(0);
  map['a'] = 1;
  map['b'] = 2;
  map['c'] = 3;
  map['d'] = 4;
  for (string key : map) {
    int _ = map[key];  // Okay to read values of existing keys.
    assert(map['e'] == 0);  // Okay to read non-existent keys (given nullValue).
    map[key] = 5;  // Okay to change values of existing keys.
    map['b'] = 6;  // Okay to change values of existing keys.
    map['e'] = 0;  // Okay to soft-delete non-existent keys.
    // Uncommenting any of the following should cause errors:
    // map['e'] = 7;  // Should cause errors since we are adding a key.
    // map['b'] = 0;  // Should cause errors since we are deleting a key.
    // map.delete(it.get());  // Not allowed to delete keys while iterating.
  }
}
EndTest();

StartTest('Change collections.hashmap values while iterating');
{
  using ktype = string;
  // The following can be uncommented to specify a custom hash function:
  // struct wrapped_string {
  //   string s;
  //   int hash() { return 1; }
  //   autounravel wrapped_string operator cast(string s) {
  //     wrapped_string result = new wrapped_string;
  //     result.s = s;
  //     return result;
  //   }
  //   autounravel bool operator ==(wrapped_string a, wrapped_string b) {
  //     return a.s == b.s;
  //   }
  //   autounravel bool operator !=(wrapped_string a, wrapped_string b) {
  //     return a.s != b.s;
  //   }
  // }
  // using ktype = wrapped_string;
  from collections.hashmap(K=ktype, V=int) access
      HashMap_K_V as HashMap_string_int;
  var map = HashMap_string_int(0);
  map['a'] = 1;
  map['b'] = 2;
  map['c'] = 3;
  map['d'] = 4;
  for (string key : map) {
    int _ = map[key];  // Okay to read values of existing keys.
    assert(map['e'] == 0);  // Okay to read non-existent keys (given nullValue).
    map[key] = 5;  // Okay to change values of existing keys.
    map['b'] = 6;  // Okay to change values of existing keys.
    map['e'] = 0;  // Okay to soft-delete non-existent keys.
    // Uncommenting any of the following should cause errors:
    // map['e'] = 7;  // Should cause errors since we are adding a key.
    // map['b'] = 0;  // Should cause errors since we are deleting a key.
    // map.delete(key);  // Not allowed to delete keys while iterating.
  }
}
EndTest();