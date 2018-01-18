"""

 Main class for asynchronous tasks (multi-threading jobs)
 
 This class is the general version intended for an arbitrary number
 of CPUs (specified in the task constructor).
 
 This class is a direct derivative of Pytorch's dataloader defined here
 https://github.com/pytorch/pytorch/blob/master/torch/utils/data/dataloader.py
 Therefore, all this code can be seen as an extension of their multithread
 capabilities. However, here we extend this process so that it can perform
 any function required (not only data loading).
 
 We also made two major modifications
     * Asynchronous returns are not limited to Torch tensors
     ()

 Author : Philippe Esling
          <esling@ircam.fr>

 Version : 0.9

"""

###################################
# Imports
import sys
import torch as torch
import torch.multiprocessing as multiprocessing
from torch.utils.data.sampler import BatchSampler
import collections
import traceback
import threading
import numpy as np
if sys.version_info[0] == 2:
    import Queue as queue
else:
    import queue

_use_shared_memory = False
"""Whether to use shared memory in default_collate"""

class ExceptionWrapper(object):
    "Wraps an exception plus traceback to communicate across threads"

    def __init__(self, exc_info):
        self.exc_type = exc_info[0]
        self.exc_msg = "".join(traceback.format_exception(*exc_info))

def _worker_loop(funPointer, dataIn, options, index_queue, data_queue, collate_fn):
    """
    This is one of our main change to Pytorch multiprocessing interface.
    Here we allow any type of operation through the use of a function pointer
    """
    global _use_shared_memory
    _use_shared_memory = True

    torch.set_num_threads(1)
    while True:
        r = index_queue.get()
        if r is None:
            data_queue.put(None)
            break
        idx, batch_indices = r
        try:
            samples = collate_fn([funPointer(i, dataIn, options) for i in batch_indices])
        except Exception:
            data_queue.put((idx, ExceptionWrapper(sys.exc_info())))
        else:
            data_queue.put((idx, samples))

def defaultCollate(batch):
    """
    Puts each data field into a tensor with outer dimension batch size.
    Second wide modification is that we do not really want to force Torch
    tensors as output of our multi-threading. We will mostly keep numpy 
    outputs (except if the function pointer outputs clearly Torch tensors)
    """
    if torch.is_tensor(batch[0]):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif type(batch[0]).__module__ == 'numpy':
        elem = batch[0]
        if type(elem).__name__ == 'ndarray':
            return np.stack(batch, axis=0)
        if elem.shape == ():  # scalars
            return np.array(batch)
    elif isinstance(batch[0], int):
        return np.array(batch)
    elif isinstance(batch[0], float):
        return np.array(batch)
    elif isinstance(batch[0], str):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: defaultCollate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [defaultCollate(samples) for samples in transposed]
    # Raise a type error if any unkown type is found
    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                     .format(type(batch[0]))))

class asynchronousIterator(object):
    "Iterates once over the DataLoader's dataset, as specified by the sampler"

    def __init__(self, loader):
        self.importFunc = loader.importFunc
        self.collateFunc = loader.collateFunc
        self.numWorkers = loader.numWorkers
        self.dataIn = loader.dataIn
        self.options = loader.options
        self.sampler = loader.sampler
        self.doneEvent = threading.Event()
        # Take an iteration of the batch sampler
        self.sample_iter = iter(self.sampler)

        if self.numWorkers > 0:
            self.index_queue = multiprocessing.SimpleQueue()
            self.data_queue = multiprocessing.SimpleQueue()
            self.batches_outstanding = 0
            self.shutdown = False
            self.send_idx = 0
            self.rcvd_idx = 0
            self.reorder_dict = {}

            self.workers = [
                multiprocessing.Process(
                    target=_worker_loop,
                    args=(self.importFunc, self.dataIn, self.options, self.index_queue, self.data_queue, self.collateFunc))
                for _ in range(self.numWorkers)]

            for w in self.workers:
                w.daemon = True  # ensure that the worker exits on process exit
                w.start()
            # prime the prefetch loop
            for _ in range(2 * self.numWorkers):
                self._put_indices()

    def __len__(self):
        return len(self.sampler)

    def __next__(self):
        # same-process loading
        if self.numWorkers == 0:
            indices = next(self.sample_iter)  # may raise StopIteration
            batch = self.collateFunc([self.importFunc(i, self.dataIn, self.options) for i in indices])
            return batch

        # check if the next sample has already been generated
        if self.rcvd_idx in self.reorder_dict:
            batch = self.reorder_dict.pop(self.rcvd_idx)
            return self._process_next_batch(batch)

        if self.batches_outstanding == 0:
            self._shutdown_workers()
            raise StopIteration

        while True:
            assert (not self.shutdown and self.batches_outstanding > 0)
            idx, batch = self.data_queue.get()
            self.batches_outstanding -= 1
            if idx != self.rcvd_idx:
                # store out-of-order samples
                self.reorder_dict[idx] = batch
                continue
            return self._process_next_batch(batch)

    next = __next__  # Python 2 compatibility

    def __iter__(self):
        return self

    def _put_indices(self):
        assert self.batches_outstanding < 2 * self.numWorkers
        indices = next(self.sample_iter, None)
        if indices is None:
            return
        self.index_queue.put((self.send_idx, indices))
        self.batches_outstanding += 1
        self.send_idx += 1

    def _process_next_batch(self, batch):
        self.rcvd_idx += 1
        self._put_indices()
        if isinstance(batch, ExceptionWrapper):
            raise batch.exc_type(batch.exc_msg)
        return batch

    def __getstate__(self):
        # TODO: add limited pickling support for sharing an iterator
        # across multiple threads for HOGWILD.
        # Probably the best way to do this is by moving the sample pushing
        # to a separate thread and then just sharing the data queue
        # but signalling the end is tricky without a non-blocking API
        raise NotImplementedError("DataLoaderIterator cannot be pickled")

    def _shutdown_workers(self):
        if not self.shutdown:
            self.shutdown = True
            self.doneEvent.set()
            for _ in self.workers:
                self.index_queue.put(None)

    def __del__(self):
        if self.numWorkers > 0:
            self._shutdown_workers()


class AsynchronousTask(object):
    """
    Asynchronous task generalizes the notion of DataLoader.
    We will still use the multiprocessing capabilities, however this
    should be completely abstract from specific data loading process.
    Allows single- or multi-process iterators over the dataset.
    
    Arguments:
    ----------
    importFunc : function pointer
        Pointer to the function that is expected to do one thread work
    numWorkers : int (default: 0)
        How many subprocesses to use (0 = only use the main process)             
    batchSize : int (default: 1)
        How many samples per batch to load
    shuffle : bool (default: False):
        Have the data reshuffled at every epoch.
    collateFunc (callable, optional)
        Function pointer that merges a list of samples to form a mini-batch.
    """

    def __init__(self, importFunc, numWorkers=0, batchSize=1, shuffle=False, collateFunc=defaultCollate):
        self.importFunc = importFunc
        self.batchSize = batchSize
        self.numWorkers = numWorkers
        self.collateFunc = collateFunc
        self.shuffle = shuffle
    
    def createTask(self, dataIn, options):
        if self.shuffle:
            sampler = iter(np.random.permutation(len(dataIn)))
        else:
            sampler = iter(range(len(dataIn)))
        # Save samplers
        self.sampler = BatchSampler(sampler, self.batchSize, False) 
        self.dataIn = dataIn
        self.options = options

    def __iter__(self):
        return asynchronousIterator(self)

    def __len__(self):
        return len(self.batch_sampler)