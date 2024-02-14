typedef import(T);

struct LinkedIterator_T {
  T next();
  bool hasNext();
  void delete();
}
typedef LinkedIterator_T Iter;  // for qualified access

struct List_T {
  int size();
  void add(T elem);
  void insertAtBeginning(T elem);
  LinkedIterator_T iterator();
}
typedef List_T L;  // for qualified access

struct LinkedList_T {
  struct Node {
    T data;
    Node next;
    void operator init(T data, Node next=null) {
      this.data = data;
      this.next = next;
    }
  };

  private Node head = null;
  private Node tail = null;
  private int size = 0;
  private int numChanges = 0;

  int size() {
    return size;
  }

  void add(T data) {
    Node newNode = Node(data);
    if (head == null) {
      head = newNode;
      tail = newNode;
    } else {
      tail.next = newNode;
      tail = newNode;
    }
    ++size;
    ++numChanges;
  }

  void insertAtBeginning(T data) {
    Node newNode = Node(data, head);
    head = newNode;
    ++size;
    ++numChanges;
  }

  LinkedIterator_T iterator() {
    Node next = head;
    Node current = null;
    Node previous = null;
    LinkedIterator_T it = new LinkedIterator_T;
    int it_numChanges = numChanges;
    bool canDelete() {
      return current != null;
    }
    it.next = new T() {
      assert(next != null, "No more elements in the list");
      assert(it_numChanges == numChanges, "Concurrent modification detected");
      // Advance previous, current, next:
      if (current != null) {
        previous = current;
      }
      current = next;
      next = next.next;
      return current.data;
    };
    it.hasNext = new bool() {
      assert(it_numChanges == numChanges, "Concurrent modification detected");
      return next != null;
    };
    it.delete = new void() {
      assert(it_numChanges == numChanges, "Concurrent modification detected");
      assert(canDelete(), "No element to delete");
      assert(size > 0, "Bug in iterator");
      if (current == tail) {
        tail = previous;
      }
      if (previous != null) {
        previous.next = next;
        current = null;
      } else {
        assert(current == head, "Bug in iterator");
        head = next;
        current = null;
      }
      --size;
      ++numChanges;
      ++it_numChanges;
    };
    return it;
  }
}
typedef LinkedList_T Linked;

List_T makeLinked(T[] initialData) {
  List_T list = new List_T;
  LinkedList_T linked = new LinkedList_T;
  list.size = linked.size;
  list.add = linked.add;
  list.insertAtBeginning = linked.insertAtBeginning;
  list.iterator = linked.iterator;
  for (int i = initialData.length - 1; i >= 0; --i) {
    list.insertAtBeginning(initialData[i]);
  }
  return list;
}

L make() {  // for qualified access
  return makeLinked(new T[0]);
}

struct NaiveList_T {
  T[] data = new T[0];

  int size() {
    return data.length;
  }

  void add(T elem) {
    data.push(elem);
  }

  void insertAtBeginning(T elem) {
    data.insert(0, elem);
  }

  LinkedIterator_T iterator() {
    int i = 0;
    int[] lastSeen = new int[];
    LinkedIterator_T it = new LinkedIterator_T;
    it.next = new T() {
      assert(i < data.length, "No more elements in the list");
      T retv = data[i];
      if (lastSeen.length > 0) {
        lastSeen[0] = i;
      } else {
        lastSeen.push(i);
      }
      assert(lastSeen.length == 1);
      ++i;
      return retv;
    };
    it.hasNext = new bool() {
      return i < data.length;
    };
    it.delete = new void() {
      assert(lastSeen.length == 1, "No element to delete");
      assert(lastSeen.pop() == --i);
      data.delete(i);
    };
    return it;
  }
}
typedef NaiveList_T Naive;  // for qualified access

List_T makeNaive(T[] initialData) {
  List_T list = new List_T;
  NaiveList_T naive = new NaiveList_T;
  list.size = naive.size;
  list.add = naive.add;
  list.insertAtBeginning = naive.insertAtBeginning;
  list.iterator = naive.iterator;
  for (int i = initialData.length - 1; i >= 0; --i) {
    list.insertAtBeginning(initialData[i]);
  }
  return list;
}

L makeNaive() {  // for qualified access
  return makeNaive(new T[0]);
}

T[] toArray(List_T list) {
  T[] retv = new T[];
  for (Iter it = list.iterator(); it.hasNext(); ) {
    retv.push(it.next());
  }
  return retv;
}