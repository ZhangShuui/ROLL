import os

import ray


@ray.remote(num_gpus=1)
def get_visible_gpus():
    return ray.get_gpu_ids()


@ray.remote(num_gpus=1)
def get_node_rank():
    return int(os.environ.get("NODE_RANK", "0"))
