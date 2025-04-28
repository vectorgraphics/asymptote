typedef import(T);

from collections.iter(T=T) access Iter_T, Iterable_T;

struct Queue_T {
  void push(T value);
  T peek();
  T pop();
  int size();
  Iter_T operator iter();
  autounravel Iterable_T operator cast(Queue_T queue) {
    return Iterable_T(queue.operator iter);
  }
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
  queue.operator iter = new Iter_T() {
    return Iter_T(data);
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

  Iter_T operator iter() {
    int i = 0;
    Iter_T result;
    result.advance = new void() {
      ++i;
    };
    result.get = new T() {
      return data[start+i];
    };
    result.valid = new bool() {
      return i < size;
    };
    return result;
  }

  void operator init(T[] initialData) {
    if (initialData.length == 0 || alias(initialData, null)) {
      return;
    }
    int desiredLength = data.length;
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

  autounravel Iterable_T operator cast(ArrayQueue_T queue) {
    return Iterable_T(queue.operator iter);
  }

  autounravel Queue_T operator cast(ArrayQueue_T queue) {
    Queue_T queue_ = new Queue_T;
    queue_.push = queue.push;
    queue_.peek = queue.peek; 
    queue_.pop = queue.pop;
    queue_.size = queue.size;
    queue_.operator iter = queue.operator iter;
    return queue_;
  }


}

Queue_T makeArrayQueue(T[] initialData /*specify type for overloading*/) {
  return ArrayQueue_T(initialData);
}

struct LinkedQueue_T {
  struct Node {
    T value;
    Node next;
  }
  Node head = null;
  Node tail = null;
  int size = 0;

  Iter_T operator iter() {
    Node node = head;
    Iter_T result;
    result.advance = new void() {
      node = node.next;
    };
    result.get = new T() {
      return node.value;
    };
    result.valid = new bool() {
      return node != null;
    };
    return result;
  }

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

  autounravel Queue_T operator cast(LinkedQueue_T queue) {
    Queue_T queue_ = new Queue_T;
    queue_.push = queue.push;
    queue_.peek = queue.peek; 
    queue_.pop = queue.pop;
    queue_.size = queue.size;
    queue_.operator iter = queue.operator iter;
    return queue_;
  }

  autounravel Iterable_T operator cast(LinkedQueue_T queue) {
    return Iterable_T(queue.operator iter);
  }

}

Queue_T makeLinkedQueue(T[] initialData) {
  var queue = new LinkedQueue_T;
  for (T value : initialData) {
    queue.push(value);
  }
  return queue;
}

// Specify a "default" queue implementation.
Queue_T makeQueue(T[] initialData) = makeArrayQueue;