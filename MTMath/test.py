import time
import threading
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# globals in worker processes
worker_slot = None
tick_queue = None


def init_worker(slot_queue, queue):
    # this runs once in each worker process
    global worker_slot, tick_queue
    worker_slot = slot_queue.get()  # grab one fixed slot (1…max_workers)
    tick_queue = queue  # shared queue for step-ticks


def worker(n_steps: int):
    # run in worker process, sending one tick per iteration
    for _ in range(n_steps):
        time.sleep(0.001)  # your real work here
        tick_queue.put((worker_slot, 1))
    # no close/reset here, just finish


def listener(total_tasks, n_steps, queue, max_workers):
    # runs in main thread: owns all tqdm calls
    outer = tqdm(total=total_tasks, desc="Overall", position=0)
    # pre-create one bar per worker-slot
    bars = {
        slot: tqdm(total=n_steps, desc=f"Worker-{slot}", position=slot, leave=True)
        for slot in range(1, max_workers + 1)
    }
    counts = {slot: 0 for slot in bars}
    done = 0

    while done < total_tasks:
        slot, delta = queue.get()
        counts[slot] += delta
        bars[slot].update(delta)
        if counts[slot] >= n_steps:
            # reset that bar in place for its next job
            bars[slot].reset()
            counts[slot] = 0
            outer.update(1)
            done += 1

    outer.close()
    for b in bars.values():
        b.close()


if __name__ == "__main__":
    total_tasks = 30
    steps_per_task = 1000
    max_workers = 4

    mgr = multiprocessing.Manager()
    slot_queue = mgr.Queue()
    queue = mgr.Queue()

    # fill slot_queue with slots 1…max_workers
    for s in range(1, max_workers + 1):
        slot_queue.put(s)

    # start listener thread
    t = threading.Thread(
        target=listener,
        args=(total_tasks, steps_per_task, queue, max_workers),
        daemon=True,
    )
    t.start()

    # launch pool; each process runs init_worker once
    with ProcessPoolExecutor(
        max_workers=max_workers, initializer=init_worker, initargs=(slot_queue, queue)
    ) as exe:
        # schedule all your tasks
        for _ in range(total_tasks):
            exe.submit(worker, steps_per_task)

    t.join()
