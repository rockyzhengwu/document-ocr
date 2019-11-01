import multiprocessing
import threading
import time
from multiprocessing import Queue

import numpy as np

class GeneratorEnqueuer():
  def __init__(self, generator,wait_time=0.05, random_seed=None):
    self.wait_time = wait_time
    self._generator = generator
    self._threads = []
    self._stop_event = None
    self.queue = None
    self.random_seed = random_seed

  def start(self, workers=1, max_queue_size=10):
    def data_generator_task():
      while not self._stop_event.is_set():
        try:
          generator_output = next(self._generator)
          self.queue.put(generator_output)
        except Exception as e:
          print(e)
          self._stop_event.set()
          raise
    try:
      self.queue = Queue(maxsize=max_queue_size)
      self._stop_event = multiprocessing.Event()
      for _ in range(workers):
        np.random.seed(self.random_seed)
        p = multiprocessing.Process(target=data_generator_task)
        p.daemon = True
        if self.random_seed is not None:
          self.random_seed += 1
        self._threads.append(p)
        p.start()
    except Exception as e:
      print(e)
      p.stop()
      raise

  def is_running(self):
    return self._stop_event is not None and not self._stop_event.is_set()

  def stop(self, timeout=None):
    if self.is_running():
      self._stop_event.set()

    for thread in self._threads:
      if thread.is_alive():
        thread.terminate()

    if self.queue is not None:
      self.queue.close()

    self._threads = []
    self._stop_event = None
    self.queue = None

  def get(self):
    while self.is_running():
      if not self.queue.empty():
        inputs = self.queue.get()
        if inputs is not None:
          yield inputs
      else:
        time.sleep(self.wait_time)
