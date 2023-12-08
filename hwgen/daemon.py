import multiprocessing
import threading
import queue
import logging

logging.basicConfig(level=logging.DEBUG, format='(%(threadName)-9s) %(message)s',)

class DaemonBrokenMaybe(threading.Thread):
    def __init__(self, data_iterator, buffer_size=5000):
        super().__init__()
        self.buffer_size = buffer_size
        self.queue = queue.Queue(maxsize=buffer_size)
        self.stop_event = threading.Event()
        self.data_iterator = data_iterator
        self.iterations = 0

    def run(self):
        while not self.stop_event.is_set():
            try:
                item = next(self.data_iterator, None)
                if item is None:
                    break  # End of iterator
                self.iterations += 1
                self.queue.put(item)  # This will block if the queue is full
                size = self.queue.qsize()
                if size % 100 == 0:
                    print("QUEUE SIZE: ", size)
            except Exception as e:
                print("Daemon error!")
                print(f"Iterations: {self.iterations} Queue size: {self.queue.qsize()}")
                print(e)
                break

    def stop(self):
        self.stop_event.set()

    def exit_cleanly(self):
        while not self.queue.empty():
            self.queue.get()
            self.queue.task_done()
        #self.queue.join()
        #self.join()

class Daemon(threading.Thread):
    def __init__(self, data_iterator, buffer_size=5000):
        super().__init__()
        self.buffer_size = buffer_size
        self.queue = queue.Queue(maxsize=buffer_size)
        self.stop_event = threading.Event()
        self.data_iterator = data_iterator
        self.iterations = 0

    def run(self):
        try:
            for item in self.data_iterator:
                try:
                    if self.stop_event.is_set():
                        break
                    self.iterations += 1
                    self.queue.put(item)  # This will block if the queue is full
                    size = self.queue.qsize()
                    if size % 100 == 0:
                        print("QUEUE SIZE: ", size)
                except Exception as e:
                    print("Daemon error!")
                    print(f"Iterations: {self.iterations} Queue size: {self.queue.qsize()}")
                    print(e)
        except Exception as e:
            print(e)
        finally:
            self.cleanup()

    def stop(self):
        self.stop_event.set()
        self.empty_queue()  # Call to empty the queue

    def empty_queue(self):
        # Empty the queue
        while not self.queue.empty():
            self.queue.get()
            self.queue.task_done()

    def cleanup(self):
        # Close the data iterator if it's a generator
        if hasattr(self.data_iterator, 'close'):
            self.data_iterator.close()

        print(f"Cleaning up generator")


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
