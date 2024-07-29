from time import sleep
from tqdm import tqdm, trange
from concurrent.futures import ThreadPoolExecutor

L = list(range(9))


def progresser(n):
    interval = 0.001 / (n + 2)
    total = 5000
    text = f"#{n}, est. {interval * total:<04.2}s"
    for _ in trange(total, desc=text):
        sleep(interval)
    if n == 6:
        tqdm.write("n == 6 completed.")
        tqdm.write("`tqdm.write()` is thread-safe in py3!")


if __name__ == "__main__":
    with ThreadPoolExecutor() as p:
        p.map(progresser, L)
