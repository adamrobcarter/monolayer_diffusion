import scipy
import numpy as np
from numba import njit, prange
import json
import os
import sys
# sys.path.append("./lubrication/")
# sys.path.append(os.path.expanduser("~/libmobility_diffusion/lubrication/"))
# ##### FIXME DO NOT COMMIT
# from Lubrication import Lubrication

STORE_PATH = 'store'

def write_row_binary(out_dir, row, N, out_dtype="float32") -> None:
    pos_file = out_dir + "colloids.bin"
    meta_file = out_dir + "binary_metadata.json"

    row = np.array(row, dtype=out_dtype)
    if not os.path.exists(meta_file):
        metadata = {
            "row_size": row.size,
            "N": N,
            "n_rows": 1,  # account for row about to be written
            "dtype": out_dtype,
        }
        with open(meta_file, "w") as f:
            json.dump(metadata, f, indent=4)
    else:
        with open(meta_file, "r") as f:
            metadata = json.load(f)
            metadata["n_rows"] += 1
            metadata["time"] = str(np.datetime64("now")) # might be useful for comparing simulation speeds
        with open(meta_file, "w") as f:
            json.dump(metadata, f, indent=4)

    with open(pos_file, "ab") as f:
        row.tofile(f)


def read_binary_file(dir) -> np.ndarray:
    pos_file = dir + "colloids.bin"
    meta_file = dir + "binary_metadata.json"

    with open(meta_file, "r") as f:
        metadata = json.load(f)

    n_rows = metadata["n_rows"]
    row_size = metadata["row_size"]
    dtype = np.dtype(metadata["dtype"])

    data = np.fromfile(pos_file, dtype=dtype)
    data = data.reshape((n_rows, row_size))
    return data


def create_solvers(solver_name, Lx, Ly, a, eta, lub_cut, tol=1e-1, includeAngular=True, wall=False, wall_sep=None):
    assert solver_name == 'Self', "currenty we only support Self solver"

    blob_fname = None # set this if you want to use lubrication
    
    if solver_name == "NBody":
        from libMobility import NBody

        if wall == 'single_wall':
            solver = NBody("open", "open", "single_wall")
            solver.setParameters(wallHeight=0.0)
            blob_fname = "./lubrication/resistance_coeffs/res_scalars_blob_nbody_wall_trans.csv"
            # blob_fname = "resistance_coeffs/res_scalars_blob_nbody_wall_trans.csv"
        elif wall == 'open':
            solver = NBody("open", "open", "open")
            solver.setParameters()
        
    elif solver_name == "DPStokes":
        assert wall
        from libMobility import DPStokes

        if wall == 'single_wall':
            zmax = 6 * a
            # quote docs
            # Even in open mode (Z periodicity set to open) the values of zmin and zmax are still required.
            # The algorithm needs to define a grid in the z direction, and these values define the extents of that grid.
            # The code will fail if a position outside of these extents is used.

            solver = DPStokes("periodic", "periodic", "single_wall")
            solver.setParameters(
                zmin=0.0, zmax=zmax, Lx=Lx, Ly=Ly, allowChangingBoxSize=False
            )
            blob_fname = "/home/cartera/libmobility_diffusion/lubrication/resistance_coeffs/res_scalars_blob_dpstokes_wall_trans.csv"
            ### FIX, DO NOT COMMIT

        elif wall == 'open':

            solver = DPStokes("periodic", "periodic", "open")
            solver.setParameters(
                zmin=-3*a, zmax=3*a, Lx=Lx, Ly=Ly, allowChangingBoxSize=False
            )

        elif wall == 'two_walls':
            zmax = wall_sep
            solver = DPStokes("periodic", "periodic", "two_walls")
            solver.setParameters(
                zmin=0.0, zmax=zmax, Lx=Lx, Ly=Ly, allowChangingBoxSize=False
            )
            
        else:
            assert False, f"Unknown wall option: {wall}"
        # blob_fname = "resistance_coeffs/res_scalars_blob_dpstokes_wall_trans.csv"

    elif solver_name == "Self":
        from libMobility import SelfMobility

        solver = SelfMobility("open", "open", "open")
        # solver.setParameters()


    else:
        raise ValueError(f"Unknown solver name: {solver_name}")

    solver.initialize(viscosity=eta, hydrodynamicRadius=a, tolerance=tol, includeAngular=includeAngular)
    """includeAngular=True
        got this again in the two walls btw with an even smaller timestep RuntimeError: [Lanczos] Unknown error (found NaN in result guess)
        try turning on torques but not using any torques
        how it's set up is that'll change the force kernel to be more accurate
    """
    return solver


def get_simulation_dir(solver, N, L, dt, t_final, t_save, wall, mg, disable_lubrication, a) -> str:
    dirFound = False
    runNumber = 0
    dir = ""

    if t_final % 1 == 0:
        t_final = int(t_final)
    if t_save % 1 == 0:
        t_save = int(t_save)

    if wall:
        wall_str = f"_{wall}"
    else:
        wall_str = "_open"

    extra = ''
    if mg == 0:
        extra += '_nograv'
    if disable_lubrication:
        extra += '_nolub'

    while not dirFound:
        dir = f"{STORE_PATH}/cartera/libmobility_diffusion/solver_{solver}_N_{N}_L_{int(L)}{wall_str}_dt_{dt*1000:.0f}_t_{t_final}_{t_save}_a_{a}{extra}_run_{runNumber}/"
        if os.path.isdir(dir):
            print(f"Directory {dir} already exists, trying again...")
            runNumber += 1
            continue
        os.makedirs(dir, exist_ok=False)
        dirFound = True
    return dir


def save_params_json(params, out_dir=None):
    if out_dir is not None:
        fname = out_dir + "params.json"
    else:
        fname = "params.json"

    params["job_started_at"] = str(np.datetime64("now"))
    with open(fname, "w") as f:
        json.dump(params, f, indent=4)
    print("Saved parameters to params.json")


def build_neighbor_list(r_vectors, r_cut, L, eps=0.0):

    r_vectors = periodize_r_vecs(r_vectors, L, r_vectors.shape[0])

    # TODO benchmark the balanced_tree and compact_node options
    r_tree = scipy.spatial.cKDTree(
        r_vectors, boxsize=L, balanced_tree=False, compact_nodes=False
    ) # boxsize=L builds in periodic images

    pairs = r_tree.query_ball_point(
        r_vectors, r_cut, return_sorted=False, workers=1, eps=eps
    )  # eps has a large effect on performance and can affect accuracy if set incorrectly

    offsets = np.cumsum([0] + [len(p) for p in pairs], dtype=int)
    list_of_neighbors = np.fromiter(
        (item for sublist in pairs for item in sublist), dtype=int
    )
    return offsets, list_of_neighbors


@njit(parallel=True, fastmath=True)
def periodize_r_vecs(r_vecs_np, L, Nb):
    r_vecs = np.copy(r_vecs_np)
    # r_vecs = np.reshape(r_vecs, (Nb, 3))
    for k in prange(Nb):
        for i in range(3):
            if L[i] > 0:
                while r_vecs[k, i] < 0:
                    r_vecs[k, i] += L[i]
                while r_vecs[k, i] > L[i]:
                    r_vecs[k, i] -= L[i]
    return r_vecs
