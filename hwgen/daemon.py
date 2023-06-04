import threading
import queue
import logging

logging.basicConfig(level=logging.DEBUG, format='(%(threadName)-9s) %(message)s',)

class Daemon(threading.Thread):
    def __init__(self, data_iterator, buffer_size=5000):
        super().__init__()
        self.buffer_size = buffer_size
        self.queue = queue.Queue(maxsize=buffer_size)
        self.stop_event = threading.Event()
        self.data_iterator = data_iterator
        self.iterations = 0

    def run(self):
        for item in self.data_iterator:
            try:
                # Add the item to the queue. This will block if the queue is full.
                self.iterations += 1
                self.queue.put(item)
                size = self.queue.qsize()
                if size % 100 == 0:
                    print("QUEUE SIZE: ", size)
                if self.stop_event.is_set():
                    print("Daemon stopped!")
                    print(f"Iterations: {self.iterations} Queue size: {size}")
                    return
            except Exception as e:
                print("Daemon error!")
                print(f"Iterations: {self.iterations} Queue size: {size}")
                print(e)

    def stop(self):
        self.stop_event.set()



import multiprocessing
import threading

class Daemon2(multiprocessing.Process):  # Changed from threading.Thread to multiprocessing.Process
    def __init__(self, data_iterator, buffer_size=5000):
        super().__init__()
        self.buffer_size = buffer_size
        self.queue = multiprocessing.Queue(maxsize=buffer_size)  # Changed from queue.Queue to multiprocessing.Queue
        self.stop_event = multiprocessing.Event()  # Changed from threading.Event to multiprocessing.Event
        self.data_iterator = data_iterator

    def run(self):
        # This is the function that runs in the background process.
        for item in self.data_iterator:
            # Add the item to the queue. This will block if the queue is full.
            self.queue.put(item)
            size = self.queue.qsize()
            if size % 100 == 0:
                print("QUEUE SIZE: ", size)
            if self.stop_event.is_set():
                return

    def stop(self):
        self.stop_event.set()


import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue, Event
import torch.cuda as cuda

class Daemon3(Process):
    def __init__(self, data_iterator, buffer_size=5000):
        super().__init__()
        self.buffer_size = buffer_size
        self.queue = Queue(maxsize=buffer_size)
        self.stop_event = Event()
        self.data_iterator = data_iterator

    def run(self):
        # Move to CUDA before starting the loop
        cuda.set_device(0)
        # This is the function that runs in the background thread.
        for item in self.data_iterator:
            # Add the item to the queue. This will block if the queue is full.
            self.queue.put(item)
            size = self.queue.qsize()
            if size % 100 == 0:
                print("QUEUE SIZE: ", size)
            if self.stop_event.is_set():
                return

    def stop(self):
        self.stop_event.set()


# Change to 'spawn' start method
mp.set_start_method('spawn', force=True)
