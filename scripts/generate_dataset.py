from polytope_core.builder import PolytopeBuilder as pb
import numpy as np
from collections import defaultdict
from pathlib import Path

np.random.seed(4444)

BUCKETS = [
    (4, 8),
    (8, 16),
    (16, 32),
    (32, 64),
    (64, 128),
    (128, 256),
]
MAX_SAMPLE_POINTS = 2000
BUCKET_VOLUME = 500

bucket_count = defaultdict(int)
all_bucket_dirs = [f"{lo}-{hi - 1}" for lo, hi in BUCKETS]


def get_bucket_dir(num_facets):
    for lo, hi in BUCKETS:
        if lo <= num_facets < hi:
            return f"{lo}-{hi - 1}"
    return None


for name, gen in [
    ("normal", pb.random_normal),
    ("uniform", pb.random_uniform),
    ("exponential", pb.random_exponential),
    ("uniform_on_hypersphere", pb.uniform_on_hypersphere),
]:
    while True:
        num_points = int(np.exp(np.random.uniform(np.log(5), np.log(10 * 256))))
        polytope = gen(num_points)
        num_facets = len(polytope.simplices)

        bucket_dir = get_bucket_dir(num_facets)
        if bucket_dir is None:
            continue
        if bucket_count[bucket_dir] >= BUCKET_VOLUME:
            continue

        outputfile = (
            Path("../polytopes_dataset")
            / name
            / f"{bucket_dir}_facets"
            / f"polytope_{bucket_count[bucket_dir]}.json"
        )
        pb.serialize(polytope, outputfile)
        bucket_count[bucket_dir] += 1

        if all(bucket_count[bd] == BUCKET_VOLUME for bd in all_bucket_dirs):
            break
