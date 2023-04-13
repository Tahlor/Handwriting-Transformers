import threading
import queue

class Daemon(threading.Thread):
    def __init__(self, data_iterator, buffer_size=5000):
        super().__init__()
        self.buffer_size = buffer_size
        self.queue = queue.Queue(maxsize=buffer_size)
        self.stop_event = threading.Event()
        self.data_iterator = data_iterator

    def run(self):
        # This is the function that runs in the background thread.
        for item in self.data_iterator:
            # Add the item to the queue. This will block if the queue is full.
            self.queue.put(item)
            if self.stop_event.is_set():
                return

    def stop(self):
        self.stop_event.set()
