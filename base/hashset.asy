typedef import(T);

from pureset(T) access Set_T, operator cast;
access linkedlist(T) as list_T;

int bitWidth(int x) {
  return CLZ(0) - CLZ(x);
}

struct HashSet_T {
  // These should not change once set in the constructor.
  private int hash(T, int bits) = null;
  private bool equiv(T, T) = null;
  private T emptyresponse;

  // These fields are mutable.
  private list_T.L[] buckets = null;
  private int bits = 0;
  private int size = 0;
  private int numChanges = 0;  // Detect concurrent modification.
  
  void operator init(
      int hash(T, int bits), 
      bool equiv(T, T),
      T emptyresponse,
      int initialBuckets = 16
  ) {
    assert(initialBuckets > 0);
    this.hash = hash;
    this.emptyresponse = emptyresponse;
    this.equiv = equiv;
    // Need enough bits to represent 0..initialBuckets-1
    this.bits = bitWidth(initialBuckets - 1);
    int numBuckets = 2 ^ this.bits;
    this.buckets = new list_T.L[numBuckets];
    for (int i = 0; i < numBuckets; ++i) {
      this.buckets[i] = list_T.make();
    }
  }

  int size() {
    return size;
  }

  bool contains(T item) {
    int bucket = hash(item, bits);
    for (list_T.Iter it = buckets[bucket].iterator(); it.hasNext();) {
      if (equiv(it.next(), item)) {
        return true;
      }
    }
    return false;
  }

  T get(T item) {
    int bucket = hash(item, bits);
    for (list_T.Iter it = buckets[bucket].iterator(); it.hasNext();) {
      T candidate = it.next();
      if (equiv(candidate, item)) {
        return candidate;
      }
    }
    return emptyresponse;
  }

  bool delete(T item) {
    ++numChanges;
    int bucket = hash(item, bits);
    for (list_T.Iter it = buckets[bucket].iterator(); it.hasNext();) {
      if (equiv(it.next(), item)) {
        it.delete();
        --size;
        return true;
      }
    }
    return false;
  }

  void forEach(bool process(T)) {
    int numChanges = this.numChanges;
    for (list_T.L bucket in buckets) {
      for (list_T.Iter it = bucket.iterator(); it.hasNext();) {
        T item = it.next();
        bool keepGoing = process(item);
        assert(this.numChanges == numChanges,
               'Concurrent modification detected');
        if (!keepGoing) {
          return;
        }
      }
    }
  }

  T[] elements() {
    T[] result = new T[size];
    int i = -1;
    forEach(new bool(T item) {
      result[++i] = item;
      return true;
    });
    return result;
  }

  private void addUnsafe(T item) {
    int bucket = hash(item, bits);
    buckets[bucket].insertAtBeginning(item);
  }

  private void raiseCapacity() {
    ++numChanges;
    T[] items = elements();
    int numBuckets = 2 ^ ++bits;
    buckets = new list_T.L[numBuckets];
    for (int i = 0; i < numBuckets; ++i) {
      buckets[i] = list_T.make();
    }
    for (T item in items) {
      addUnsafe(item);
    }
  }

  bool add(T item) {
    ++numChanges;
    list_T.L bucket = buckets[hash(item, bits)];
    for (list_T.Iter it = bucket.iterator(); it.hasNext();) {
      if (equiv(it.next(), item)) {
        return false;
      }
    }
    bucket.insertAtBeginning(item);
    ++size;
    if (size > buckets.length) {
      raiseCapacity();
    }
    return true;
  }

  T update(T item) {
    ++numChanges;
    list_T.L bucket = buckets[hash(item, bits)];
    for (list_T.Iter it = bucket.iterator(); it.hasNext();) {
      T candidate = it.next();
      if (equiv(candidate, item)) {
        it.delete();
        bucket.insertAtBeginning(item);
        return candidate;
      }
    }
    bucket.insertAtBeginning(item);
    ++size;
    if (size > buckets.length) {
      raiseCapacity();
    }
    return emptyresponse;
  }

}

Set_T operator cast(HashSet_T hashSet) {
  Set_T result = new Set_T;
  result.size = hashSet.size;
  result.contains = hashSet.contains;
  result.add = hashSet.add;
  result.update = hashSet.update;
  result.get = hashSet.get;
  result.delete = hashSet.delete;
  result.forEach = hashSet.forEach;
  return result;
}

T[] operator cast(HashSet_T hashSet) {
  return hashSet.elements();
}