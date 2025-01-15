import numpy as np
import pyarrow as pa
import robosuite as suite
from dora import Node


def ndarray_to_pyarrow(arr: np.ndarray) -> pa.StructArray:
    """Convert a numpy array to a PyArrow struct array containing shape, dtype, and data.

    Args:
        arr: Input numpy array

    Returns:
        PyArrow struct array containing:
        - shape: List of dimensions
        - dtype: Original numpy dtype as string
        - data: Flattened array data with original dtype preserved
    """
    return pa.StructArray.from_arrays(
        [
            pa.array([list(arr.shape)], type=pa.list_(pa.int64())),
            pa.array([str(arr.dtype)], type=pa.string()),
            pa.array([arr.ravel()]),
        ],
        names=["shape", "dtype", "data"],
    )


def main():
    node = Node()

    # create environment instance
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
    )

    # reset the environment and send the initial observation
    obs = env.reset()
    node.send_output("observations", ndarray_to_pyarrow(obs["agentview_image"]))

    for _ in range(10000):
        action = np.random.randn(*env.action_spec[0].shape) * 0.1
        obs, reward, done, _ = env.step(action)
        node.send_output("observations", ndarray_to_pyarrow(obs["agentview_image"]))

        if done:
            obs = env.reset()
            node.send_output("observations", ndarray_to_pyarrow(obs["agentview_image"]))


if __name__ == "__main__":
    main()
