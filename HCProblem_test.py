from itertools import permutations
from DatasetCreator.loadGraphDatasets.HCPDatasetGenerator import HCProblem, plot
import numpy as np
import jax.numpy as jnp
from GraphWithMeta import GraphWithMeta
import jax.numpy as jnp
from jax import random
import jax

def aug_solution(solution: np.ndarray, room_perm: list[int], problem: HCProblem) -> np.ndarray:
    """Augment the solution nodes according to the given room permutation."""
    augmented_solution = solution.copy()
    room_perm = np.array(room_perm)
    augmented_solution[problem.OFFSET_CABINETS:problem.OFFSET_THINGS_CABINETS] = room_perm[augmented_solution[problem.OFFSET_CABINETS:problem.OFFSET_THINGS_CABINETS]]
    augmented_solution[problem.OFFSET_ROOMS:problem.OFFSET_CABINETS] = np.argsort(room_perm)[augmented_solution[problem.OFFSET_ROOMS:problem.OFFSET_CABINETS]]
    return augmented_solution


def sample_permutations(key, N, M):
    keys = random.split(key, N)
    perm_fn = lambda k: random.permutation(k, M)
    return jax.vmap(perm_fn)(keys)

def aug_solution_jax_old(in_solution: jnp.array, room_perm: jnp.array, problem: GraphWithMeta) -> jnp.array:
    in_solution = in_solution.astype(jnp.int32)
    solution = jnp.swapaxes(in_solution, 0, 2)
    plot(None, problem.globals["node_types"], "before_aug.png", solution_nodes=solution[:, 0, 0, 0, 0], meta_graph=problem)
    # Broadcast room permutations across node/time/device axes; last two dims are (basis, rooms)
    room_perm = jnp.asarray(room_perm, dtype=jnp.int32)
    broadcast_shape = [1] * solution.ndim
    broadcast_shape[-2] = room_perm.shape[0]
    broadcast_shape[-1] = room_perm.shape[1]
    room_perm_broadcast = room_perm.reshape(broadcast_shape)

    cabinets_slice = slice(problem.meta["offset_cabinets"], problem.meta["offset_things_cabinets"])
    cabinets = solution[cabinets_slice]
    permuted_cabinets = jnp.take_along_axis(room_perm_broadcast, cabinets, axis=-1)
    augmented_solution = solution.at[cabinets_slice].set(permuted_cabinets)

    rooms_slice = slice(problem.meta["offset_rooms"], problem.meta["offset_cabinets"])
    rooms = solution[rooms_slice]
    
    def scatter_rooms(x, p):
        # x: [R, ...], p: [R] old->new
        out = jnp.empty_like(x)
        return out.at[p].set(x)                    # rp'[new] = rp[old]

    rooms_prime = jax.vmap(scatter_rooms)(rooms, room_perm)
    augmented_solution = augmented_solution.at[rooms_slice].set(rooms_prime)
    plot(None, problem.globals["node_types"], "after_aug.png", solution_nodes=augmented_solution[:, 0, 0, 0, 0], meta_graph=problem)
    augmented_solution = jnp.swapaxes(augmented_solution, 0, 2)
    return augmented_solution

def aug_solution_jax(in_solution: jnp.array, room_perm: jnp.array, problem: GraphWithMeta) -> jnp.array:
    in_solution = in_solution.astype(jnp.int32)
    solution = jnp.swapaxes(in_solution, 0, 2)
    plot(None, problem.globals["node_types"], "before_aug.png", solution_nodes=solution[:, 0, 0, 0, 0], meta_graph=problem)
    # Broadcast room permutations across node/time/device axes; last two dims are (basis, rooms)
    room_perm = jnp.asarray(room_perm, dtype=jnp.int32)
    broadcast_shape = [1] * solution.ndim
    broadcast_shape[-2] = room_perm.shape[0]
    broadcast_shape[-1] = room_perm.shape[1]
    room_perm_broadcast = room_perm.reshape(broadcast_shape)

    # cabinet->room: value remap old_room -> new_room
    cabinets_slice = slice(problem.meta["offset_cabinets"], problem.meta["offset_things_cabinets"])
    cabinets = solution[cabinets_slice]
    permuted_cabinets = jnp.take_along_axis(room_perm_broadcast, cabinets, axis=-1)
    augmented_solution = solution.at[cabinets_slice].set(permuted_cabinets)

    # room->person: permute *entries* along the room-node axis (axis=0), per basis
    rooms_slice = slice(problem.meta["offset_rooms"], problem.meta["offset_cabinets"])
    rooms = solution[rooms_slice]  # shape: [R, ..., B, ...] with basis at axis -2

    # Move basis to front so we can vmap per basis
    rooms_b = jnp.moveaxis(rooms, -2, 0)          # [B, R, ...]
    perm_b = room_perm                             # [B, R] where perm_b[b, old] = new

    def scatter_rooms(x, p):
        # x: [R, ...], p: [R] old->new
        out = jnp.empty_like(x)
        return out.at[p].set(x)                    # rp'[new] = rp[old]

    rooms_b_prime = jax.vmap(scatter_rooms)(rooms_b, perm_b)   # [B, R, ...]
    rooms_prime = jnp.moveaxis(rooms_b_prime, 0, -2)           # back to [R, ..., B, ...]

    augmented_solution = augmented_solution.at[rooms_slice].set(rooms_prime)
    plot(None, problem.globals["node_types"], "after_aug.png", solution_nodes=augmented_solution[:, 0, 0, 0, 0], meta_graph=problem)
    augmented_solution = jnp.swapaxes(augmented_solution, 0, 2)
    return augmented_solution

if __name__ == "__main__":

    problem = HCProblem(5, 10, 50, 5)
    for j, room_perm in enumerate(permutations(range(5))):
        solution = aug_solution(problem.globals["solution_nodes"], room_perm, problem)
        plot(None, problem.globals["node_types"], f"test_plot_0_aug_{j}.png", solution_nodes=solution, meta_graph=problem.meta_graph)
	
