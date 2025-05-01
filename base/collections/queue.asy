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
  int version = 0;
  queue.push = new void(T value) {
    ++version;
    data.push(value);
  };
  queue.peek = new T() {
    return data[0];
  };
  queue.pop = new T() {
    ++version;
    T retv = data[0];
    data.delete(0);
    return retv;
  };
  queue.size = new int() {
    return data.length;
  };
  queue.operator iter = new Iter_T() {
    int expectedVersion = version;
    int i = 0;
    Iter_T result;
    result.advance = new void() {
      assert(expectedVersion == version, 'Iterator undermined.');
      ++i;
    };
    result.get = new T() {
      assert(expectedVersion == version, 'Iterator undermined.');
      return data[i];
    };
    result.valid = new bool() {
      assert(expectedVersion == version, 'Iterator undermined.');
      return i < data.length;
    };
    return result;
  };
  return queue;
}

struct ArrayQueue_T {
  T[] data = new T[8];
  data.cyclic = true;
  int start = 0;
  int size = 0;
  // The version is used to detect if the queue has been modified
  // while iterating over it.
  int version = 0;

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
    int currentVersion = version;
    result.advance = new void() {
      assert(currentVersion == version, 'Iterator undermined.');
      ++i;
    };
    result.get = new T() {
      assert(currentVersion == version, 'Iterator undermined.');
      return data[start+i];
    };
    result.valid = new bool() {
      assert(currentVersion == version, 'Iterator undermined.');
      return i < size;
    };
    return result;
  }

  void operator init(T[] initialData) {
    if (alias(initialData, null) || initialData.length == 0) {
      return;
    }
    int desiredLength = data.length;
    // Possible optimization: next power of 2 is 2^(CLZ(0) - CLZ(initialData.length - 1))
    // The following code is more readable with fewer edge cases.
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
    ++version;
  }

  T peek() {
    assert(size > 0);
    return data[start];
  }

  T pop() {
    assert(size > 0);
    T retv = data[start];
    ++start;
    --size;
    ++version;
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
  int version = 0;

  Iter_T operator iter() {
    Node node = head;
    Iter_T result;
    int expectedVersion = version;
    result.advance = new void() {
      assert(expectedVersion == version, 'Iterator undermined.');
      node = node.next;
    };
    result.get = new T() {
      assert(expectedVersion == version, 'Iterator undermined.');
      return node.value;
    };
    result.valid = new bool() {
      assert(expectedVersion == version, 'Iterator undermined.');
      return node != null;
    };
    return result;
  }

  void push(T value) {
    ++version;
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
    ++version;
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