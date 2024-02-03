typedef import(T);

// This is supposed to be an interface. We should probably import it from
// somewhere outside the test folder. Also we should decide on a style for
// naming interfaces.
struct Queue {
  void push(T value);
  T peek();
  T pop();
  int size();
}

Queue makeNaiveQueue(T /*specify type for overloading*/) {
  Queue queue = new Queue;
  T[] data = new T[0];
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

struct ArrayQueue {
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

Queue cast(ArrayQueue queue) {
  Queue queue_ = new Queue;
  queue_.push = queue.push;
  queue_.peek = queue.peek; 
  queue_.pop = queue.pop;
  queue_.size = queue.size;
  return queue_;
}

Queue makeArrayQueue(T /*specify type for overloading*/) {
  return new ArrayQueue;
}

struct LinkedQueue {
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

Queue cast(LinkedQueue queue) {
  Queue queue_ = new Queue;
  queue_.push = queue.push;
  queue_.peek = queue.peek; 
  queue_.pop = queue.pop;
  queue_.size = queue.size;
  return queue_;
}

Queue makeLinkedQueue(T /*specify type for overloading*/) {
  return new LinkedQueue;
}

// Specify a "default" queue implementation.
Queue makeQueue(T /*specify type for overloading*/) = makeArrayQueue;