from polytope_core.builder import PolytopeBuilder as pb
import numpy as np
from collections import defaultdict
from pathlib import Path

np.random.seed(4444)

BUCKET_WIDTH = 5
MIN_FACETS = 5
MAX_FACETS = 100
MAX_SAMPLE_POINTS = int(
    MAX_FACETS / 2
)  # Works fine for 100 facets, but propabley doesn't scale
BUCKET_VOLUME = 1000


bucket_count = defaultdict(int)
buckets = range(MIN_FACETS, MAX_FACETS, BUCKET_WIDTH)

bucket_dict = {}
all_bucket_dirs = []
for bucket in buckets:
    bucket_dir = f"{bucket}-{bucket + BUCKET_WIDTH - 1}"
    all_bucket_dirs.append(bucket_dir)
    for i in range(BUCKET_WIDTH):
        bucket_dict[bucket + i] = bucket_dir


while True:
    num_points = np.random.randint(5, MAX_SAMPLE_POINTS)
    polytope = pb.random_normal(num_points)
    num_facets = len(polytope.simplices)
    if num_facets >= MAX_FACETS or num_facets < 5:
        continue

    bucket_dir = bucket_dict[num_facets]
    if bucket_count[bucket_dir] < BUCKET_VOLUME:
        outputfile = (
            Path("polytopes_dataset")
            / "normal_distribution"
            / f"{bucket_dir}_facets"
            / f"polytope_{bucket_count[bucket_dir]}.json"
        )
        pb.serialize(polytope, outputfile)
        bucket_count[bucket_dir] += 1

        if all(
            bucket_count[bucket_dir] == BUCKET_VOLUME for bucket_dir in all_bucket_dirs
        ):
            break
