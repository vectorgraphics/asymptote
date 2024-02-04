typedef import(T);

// This is supposed to be an interface. We should probably import it from
// somewhere outside the test folder. Also we should decide on a style for
// naming interfaces.
struct Queue_T {
  void push(T value);
  T peek();
  T pop();
  int size();
}

Queue_T makeNaiveQueue(T[] initialData) {
  Queue_T queue = new Queue_T;
  T[] data = new T[0];
  data.append(initialData);
  queue.push = new void(T value) {
    data.push(value);
  };
  queue.peek = new T() {
    return data[0];
  };
  queue.pop = new T() {
    T retv = data[0];
    data.delete(0);
    return retv;
  };
  queue.size = new int() {
    return data.length;
  };
  return queue;
}

struct ArrayQueue_T {
  T[] data = new T[8];
  data.cyclic = true;
  int start = 0;
  int size = 0;

  private void resize() {
    T[] newData = new T[data.length * 2];
    newData.cyclic = true;
    newData[:size] = data[start : start+size];
    data = newData;
    start = 0;
  }

  void operator init(T[] initialData) {
    if (initialData.length == 0 || alias(initialData, null)) {
      return;
    }
    desiredLength = data.length;
    // TODO: Do this computation using CLZ.
    while (desiredLength < initialData.length) {
      desiredLength *= 2;
    }
    if (desiredLength != data.length) {
      data = new T[desiredLength];
      data.cyclic = true;
    }
    size = initialData.length;
    data[:size] = initialData;
  }

  void push(T value) {
    if (size == data.length) {
      resize();
    }
    data[start+size] = value;
    ++size;
  }

  T peek() {
    return data[start];
  }

  T pop() {
    T retv = data[start];
    ++start;
    --size;
    return retv;
  }

  int size() {
    return size;
  }
}

Queue_T cast(ArrayQueue_T queue) {
  Queue_T queue_ = new Queue_T;
  queue_.push = queue.push;
  queue_.peek = queue.peek; 
  queue_.pop = queue.pop;
  queue_.size = queue.size;
  return queue_;
}

Queue_T makeArrayQueue(T[] initialData /*specify type for overloading*/) {
  return ArrayQueue_T(initialData);
}

struct LinkedQueue_T {
  struct Node {
    T value;
    Node next;
  }
  Node head;
  Node tail;
  int size = 0;

  void push(T value) {
    Node node = new Node;
    node.value = value;
    if (size == 0) {
      head = node;
      tail = node;
    } else {
      tail.next = node;
      tail = node;
    }
    ++size;
  }

  T peek() {
    return head.value;
  }

  T pop() {
    T retv = head.value;
    head = head.next;
    --size;
    return retv;
  }

  int size() {
    return size;
  }
}

Queue_T cast(LinkedQueue_T queue) {
  Queue_T queue_ = new Queue_T;
  queue_.push = queue.push;
  queue_.peek = queue.peek; 
  queue_.pop = queue.pop;
  queue_.size = queue.size;
  return queue_;
}

Queue_T makeLinkedQueue(T[] initialData) {
  var queue = new LinkedQueue_T;
  for (T value in initialData) {
    queue.push(value);
  }
  return queue;
}

// Specify a "default" queue implementation.
Queue_T makeQueue(T[]) = makeArrayQueue;