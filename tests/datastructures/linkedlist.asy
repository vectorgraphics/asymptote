typedef import(T);

struct LinkedIterator_T {
  T next();
  bool hasNext();
  void delete();
}

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
    if (head == null) {
      head = Node(data);
      tail = head;
    } else {
      tail.next = Node(data);
      tail = tail.next;
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
    Node previous = new Node;
    previous.next = head;
    Node extraNode = previous;  // This Node is not actually in the list. Remember it for bug checks.
    LinkedIterator_T it = new LinkedIterator_T;
    int it_numChanges = numChanges;
    bool canDelete = false;
    it.next = new T() {
      assert(next != null, "No more elements in the list");
      assert(it_numChanges == numChanges, "Concurrent modification detected");
      assert(next == previous.next, "Bug in iterator");
      T retv = next.data;
      if (next.next != null) {  // If we're not at the end of the list, advance previous.
        previous = next;
      }
      next = next.next;
      canDelete = true;
      assert(next != extraNode, "Bug in iterator");
      return retv;
    };
    it.hasNext = new bool() {
      assert(it_numChanges == numChanges, "Concurrent modification detected");
      return next != null;
    };
    it.delete = new void() {
      assert(it_numChanges == numChanges, "Concurrent modification detected");
      assert(canDelete, "No element to delete");
      assert(previous != null, "Bug in iterator");
      assert(size > 0, "Bug in iterator");
      if (size == 1) {
        // Delete the only element in the list.
        head = null;
        tail = null;
      } else if (next == null) {
        // Delete the tail.
        assert(previous != extraNode, "Bug in iterator");
        tail = previous;  // This works because we did not advance previous when we reached the end of the list.
        tail.next = null;
      } else {
        assert(previous != extraNode, "Bug in iterator");
        // Copy next to previous.
        previous.data = next.data;
        previous.next = next.next;

        // Advance next.
        next = next.next;
      }
      
      --size;
      ++numChanges;
      ++it_numChanges;
      canDelete = false;
    };
    return it;
  }
}