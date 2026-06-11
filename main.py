import numpy as np
from pathlib import Path
from numba import njit, prange
from functools import partial
from copy import deepcopy
import time
import argparse
import tqdm
import place_colloids
import scipy
import libMobility
import os
import utils


def main(a, mg, Lx, Ly, kbt, eta, phi, bool_attrac, bool_attrac_wall, range_attrac, D_e, w, r_e, fact_wall, dt, t_final, t_save, solver_name, z_trap_width=None, z_trap_position=None,
         wall=None, wall_sep=0.0, initial_distribution='flat', gravity=True,
         theta=0):
    
    assert wall in ['open', 'single_wall', 'two_walls']
    # assert wall != 'two_walls', 'temporarily disabling due to weird stuff in the sterics function'

    if wall == 'open' and gravity:
        assert z_trap_width, 'if the z dimension is open with no wall and gravity, you must use a potential to confine in z'
        assert z_trap_position is not None, f'z_trap_position was {z_trap_position}'

    if wall == 'two_walls':
        print('wall sep', wall_sep)
        assert wall_sep, f'wall_sep was {wall_sep}'
        assert wall_sep is not None, f'wall_sep was {wall_sep}'
        assert np.isfinite(wall_sep), f'wall_sep was {wall_sep}'
    else:
        assert not wall_sep
    print('wall: ', wall)

    assert np.isscalar(eta)
    assert np.isscalar(a)
    assert np.isscalar(kbt)
    assert np.isscalar(Lx)
    assert np.isscalar(Ly)
    assert np.isscalar(mg), f'mg should be a scalar, but was {mg} of type {type(mg)}'
    
    L = np.array([Lx, Ly, 0])

    h_g = a + (kbt / mg)
    print("gravity height: ", h_g - a)

    if kbt > 0:
        tau = 6 * np.pi * eta * a**3 / kbt
        print(f"diffusion time: {tau}, dt: {dt}")
    n_steps = int(np.ceil(t_final / dt))
    print(f"Number of steps: {n_steps}")
    assert t_save / dt % 1 == 0, 't_save was not a multiple of dt'
    n_save = int(t_save / dt)
    print(f"saving every: {n_save}")

    kbt_J = kbt / 1e18
    print(f'T = {kbt_J / 1.38e-23:.0f}K')
    a_m = a * 1e-6
    D0_m = kbt_J / (6 * np.pi * eta * a_m)
    D0 = D0_m * 1e12
    Dc_nohydro = D0 * (1 + phi) / (1 - phi)**3
    print(f'theoretical, no hydro Dself = {D0:.3g}um/s2, Dcoll = {Dc_nohydro:.3g}um/s2')
    min_k = 2 * np.pi / max(L[0], L[1])
    max_decay_time = 1 / ( Dc_nohydro * min_k**2 )
    print(f't_final = {t_final/max_decay_time:.3g} * max_decay_time')

    # sterics stuff
    firm_delta = 1e-2
    debye_length = 2.0 * a * firm_delta / np.log(10.0)
    n_cutoff = 4  # number of debye lengths to include in the cutoff
    r_cut = 2 * a + n_cutoff * debye_length + range_attrac
    U_0 = 4 * kbt
    nlist_buffer = 3.0  # in units of blob radius

    # with z_trap_width=a/2, use dt=0.02

    lub_cutoff = 1e-2
    solver = utils.create_solvers(
        solver_name, Lx, Ly, a, eta, lub_cut=lub_cutoff, tol=1e-1, wall=wall, wall_sep=wall_sep
    )

    wall_to_place_colloids = wall
    if wall == 'two_walls':
        wall_to_place_colloids = 'single_wall' # placing colloids with two walls is hard, so we'll just place them with one wall and then add the second wall later

    if initial_distribution == 'uneven':
        r_vecs = place_colloids.place_colloids_uneven(phi, L, a, mg, kbt, U_0, firm_delta, debye_length, wall=wall_to_place_colloids, wall_sep=wall_sep)
    elif initial_distribution == 'stripe':
        r_vecs = place_colloids.place_colloids_stripe(phi, L, a, mg, kbt, U_0, firm_delta, debye_length, wall=wall_to_place_colloids, wall_sep=wall_sep)
    elif initial_distribution == 'sin':
        r_vecs = place_colloids.place_colloids_sin(phi, L, a, mg, kbt, U_0, firm_delta, debye_length, wall=wall_to_place_colloids, wall_sep=wall_sep)
    elif initial_distribution == 'flat':
        r_vecs = place_colloids.place_colloids_brennan(phi, L, a, mg, kbt, U_0, firm_delta, debye_length, wall=wall_to_place_colloids, wall_sep=wall_sep, z_trap_width=z_trap_width)
    else:
        raise Exception(f'initial distribution "{initial_distribution}" not recognised')
    
    N = r_vecs.shape[0]
    print(f"z_min = {np.min(r_vecs[:, 2]) / a}a")
    print("packing fraction:", N * np.pi * a**2 / (Lx * Ly))
    print('r vecs shape: ', r_vecs.shape)
    if wall == 'single_wall':
        assert np.all(r_vecs[:, 2] > 0.95*a), f'r_vecs[:, 2].min() = {r_vecs[:, 2].min()/a}a'
    if wall == 'two_walls':
        assert np.all(r_vecs[:, 2] > 0.95*a), f'r_vecs[:, 2].min() = {r_vecs[:, 2].min()/a}a'
        assert np.all(r_vecs[:, 2] < wall_sep - 0.95*a), f'r_vecs[:, 2].max() = {r_vecs[:, 2].max()/a}a'

    output_dir = utils.get_simulation_dir(solver=solver_name, N=N, L=Lx, dt=dt, t_final=t_final, t_save=t_save, wall=wall, mg=mg, disable_lubrication=False, a=a)
    # output_dir = "TEMP/"
    print(f"Output directory: {output_dir}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    F_calc = partial(
        calc_force,
        solver_name=solver_name,
        kbt=kbt,
        L=L,
        a=a,
        repulsion_strength=U_0,
        debye_length=debye_length,
        delta=firm_delta,
        mg=mg,
        N=N,
        bool_attrac=bool_attrac, 
        bool_attrac_wall=bool_attrac_wall,
        range_attrac=range_attrac, 
        D_e=D_e, 
        w=w, 
        r_e=r_e,
        fact_wall=fact_wall,
        z_trap_width=z_trap_width,
        z_trap_position=z_trap_position,
        wall=wall,
        wall_sep=wall_sep,
        gravity=gravity,
        theta=theta,
    )

    offsets, neighbor_list = utils.build_neighbor_list(
        r_vecs, r_cut + nlist_buffer * a, L
    )
    x0 = deepcopy(r_vecs)

    params = {
        "N_colloids": N,
        "Lx": Lx,
        "Ly": Ly,
        "a": a,
        "mg": mg,
        "kbt": kbt,
        "eta": eta,
        "solver_name": solver_name,
        "phi": phi,
        "diffusion_time": tau,
        "debye_length": debye_length,
        "U_0": U_0,
        "firm_delta": firm_delta,
        "dt": dt,
        "T_final": t_final,
        "n_steps": n_steps,
        "r_cut": r_cut,
        "n_cutoff": n_cutoff,
        "neighbor_list_buffer": nlist_buffer,
        "t_save": t_save,
        "n_save": n_save,
        "wall": wall,
        'initial_distribution': initial_distribution,
        'bool_attrac': bool_attrac, 
        'bool_attrac_wall': bool_attrac_wall,
        'range_attrac': range_attrac, 
        'D_e': D_e, 
        'w': w, 
        'r_e': r_e,
        'fact_wall' : fact_wall,
    }
    if z_trap_width:
        params["z_trap_width"] = z_trap_width
        params["z_trap_position"] = z_trap_position
        print(f'using z trap. width {z_trap_width/a:.1f}a, height {z_trap_position/a:.1f}a')
    if wall_sep:
        params["wall_sep"] = wall_sep
    utils.save_params_json(params, out_dir=output_dir)

    t_time = 20
    n_time = int(np.ceil(t_time / dt))
    t_current = 0.0
    step_start = time.time()
    rng = np.random.default_rng()

    r_vecs = r_vecs.flatten()

    # initialise v_prev
    solver.setPositions(r_vecs)
    forces = F_calc(r_vecs=r_vecs, offsets=offsets, neighbor_list=neighbor_list)
    assert np.isfinite(forces).all()
    v_prev, _ = solver.Mdot(forces=forces)
    assert np.isfinite(v_prev).all()

    for step in tqdm.trange(n_steps, mininterval=1.0, desc="Simulation progress"):

        assert np.all(r_vecs[0::3] > -0.5*Lx), f'r_vecs[0::3].min() = {r_vecs[0::3].min()}' # for NBody, we can have x < 0, but not too far
        assert np.all(r_vecs[0::3] <  1.5*Lx), f'r_vecs[0::3].max() = {r_vecs[0::3].max()}'
        assert np.all(r_vecs[1::3] > -0.5*Ly), f'r_vecs[1::3].min() = {r_vecs[1::3].min()}'
        assert np.all(r_vecs[1::3] <  1.5*Ly), f'r_vecs[1::3].max() = {r_vecs[1::3].max()}'

        if wall == 'single_wall':
            assert np.all(r_vecs[2::3] > 0), f'z min = {r_vecs[2::3].min()/a:.2f}a, particles are overlapping the wall' # note this should actually be a, but the particles can be slightly inside the wall
        if wall == 'two_walls':
            assert np.all(r_vecs[2::3] > 0), f'z min = {r_vecs[2::3].min()/a:.2f}a, particles are overlapping the wall'
            assert np.all(r_vecs[2::3] < wall_sep), f'z max = {r_vecs[2::3].max()/a:.2f}a (wall_sep = {wall_sep/a:.2f}a)'


        # no wall => no lubrication
        assert np.isfinite(r_vecs).all()
        solver.setPositions(r_vecs)

        forces = F_calc(r_vecs=r_vecs, offsets=offsets, neighbor_list=neighbor_list)
        assert np.isfinite(forces).all()

        t0 = time.time()
        v_det, _ = solver.Mdot(forces=forces)
        t1 = time.time()
        sqrt_m, _ = solver.sqrtMdotW()
        t2 = time.time()
        div_m, _ = solver.divM()
        t3 = time.time()
        t_mdot = t1 - t0
        t_sqrt = t2 - t1
        t_div = t3 - t2
        t_total = t3 - t0
        # print(f'{t_mdot/t_total:.2%} Mdot, {t_sqrt/t_total:.2%} sqrtM, {t_div/t_total:.2%} divM, total {t_total*1000:.1f}ms')

        r_vecs += np.sqrt(2 * kbt * dt) * sqrt_m  # stochastic velocity

        v_det += kbt * div_m.reshape(v_det.shape)
        r_vecs += dt * (1.5 * v_det - 0.5 * v_prev).reshape(r_vecs.shape)  # deterministic velocity

        v_prev = v_det

        t_current += dt

        if solver_name == 'Self' or solver_name == 'DPStokes':
            # enforce periodic BCs
            r_vecs[0::3] = r_vecs[0::3] % L[0]
            r_vecs[1::3] = r_vecs[1::3] % L[1]

        max_delta_pos = np.max(
            np.linalg.vector_norm(r_vecs.reshape((-1, 3)) - x0, axis=1)
        )
        if max_delta_pos > nlist_buffer * a:
            temp_r = r_vecs.reshape((-1, 3))
            # print("rebuilding neighbor list")
            offsets, neighbor_list = utils.build_neighbor_list(
                temp_r, r_cut + nlist_buffer * a, L
            )
            x0 = deepcopy(temp_r)

        # if step % n_time == 0:
        #     end = time.time()
        #     elapsed = end - step_start
        #     print(f"time to simulate {n_time*dt} seconds: {elapsed:.2f}s")
        #     step_start = time.time()

        if step % n_save == 0:
            # print(f"saving at step {step}")
            row = np.concatenate(([t_current], r_vecs.flatten()))
            utils.write_row_binary(output_dir, row, N)
            # print(np.max(z_vals) / a, np.min(z_vals) / a)
            # print("percent overlapping wall", 100 * np.sum(r_vecs[2::3] < a) / N)

    print('Done.')
    return output_dir



def calc_force(
    r_vecs,
    solver_name,
    kbt,
    offsets,
    neighbor_list,
    L,
    a,
    repulsion_strength,
    debye_length,
    delta,
    mg,
    N,
    bool_attrac, 
    bool_attrac_wall,
    range_attrac, 
    D_e, 
    w, 
    r_e,
    fact_wall,
    z_trap_width=None,
    z_trap_position=None,
    wall=None,
    wall_sep=0.0,
    gravity=True,
    theta=0,
):

    r_vecs = np.reshape(r_vecs, (-1, 3))

    # the below 3 lines are in a different version of the code, should they be here?
    # set L to zero for non-periodic sterics
    # if solver_name == "NBody":
    #     L = np.array([0.0, 0.0, 0.0])


    forces = np.zeros((N, 3))
    forces += blob_blob_sterics(
        r_vectors=r_vecs,
        L=L,
        a=a,
        repulsion_strength=repulsion_strength,
        debye_length=debye_length,
        delta=delta,
        list_of_neighbors=neighbor_list,
        offsets=offsets,
        bool_attrac=bool_attrac,
        bool_attrac_wall=bool_attrac_wall, 
        range_attrac=range_attrac, 
        D_e=D_e, 
        w=w, 
        r_e=r_e,
        fact_wall=fact_wall,
        wall=wall,
        wall_sep=wall_sep
    )

    if "NBody" in solver_name:
        forces += blob_external_force_xy_potential_confinement_numba(
            r_vectors=r_vecs, blob_radius=a, kT=kbt, potential_width=L
        )

    if z_trap_width:
        assert z_trap_position is not None
        forces += blob_external_force_z_confinement_numba(r_vecs, kbt, z0=z_trap_position, z_trap_width=z_trap_width)

    if gravity:
        forces[:, 2] += -mg * np.cos(np.radians(theta))
        forces[:, 0] += -mg * np.sin(np.radians(theta))

    return forces


@njit(parallel=True, fastmath=True)
def blob_external_force_xy_potential_confinement_numba(
    r_vectors, blob_radius, kT, potential_width
):
    """
    This function computes the force on a blob in a confinement potential.
    The potential has a flat bottom for 0 <= x < Lx and 0 <= y < Ly and increases quadratically outside
    """
    assert np.isfinite(kT)

    N = r_vectors.size // 3
    r_vectors = r_vectors.reshape((N, 3))
    f = np.zeros((N, 3))
    
    prefactor = 2 * kT / blob_radius**2

    for i in prange(N):
        r = r_vectors[i, :]

        if r[0] < 0:
            f[i, 0] += prefactor * (abs(r[0]))
        elif r[0] > potential_width[0]:
            f[i, 0] += prefactor * (potential_width[0] - r[0])

        if r[1] < 0:
            f[i, 1] += prefactor * (abs(r[1]))
        elif r[1] > potential_width[1]:
            f[i, 1] += prefactor * (potential_width[1] - r[1])

    return f

@njit(fastmath=True)
def blob_external_force_z_confinement_numba(r_vectors, kT, z0, z_trap_width):
    #potential (z-z0)^2 * k_s/2
    #force 2(z-z0) * k_s/2
    # k_s = k_B T / z_trap_width**2    # https://link.aps.org/doi/10.1103/PhysRevE.95.012602 eq 1, z_trap_width === delta
    #only used if there is no wall

    N = r_vectors.size // 3
    r_vectors = r_vectors.reshape((N, 3))
    f = np.zeros((N, 3))

    f[:, 2] = - 2 * (r_vectors[:, 2] - z0) * kT / z_trap_width**2 / 2

    return f

@njit(parallel=True, fastmath=True)
def blob_blob_sterics(
    r_vectors,
    L,
    a,
    repulsion_strength,
    debye_length,
    delta,
    list_of_neighbors,
    offsets,
    bool_attrac, 
    bool_attrac_wall,
    range_attrac, 
    D_e, 
    w, 
    r_e,
    fact_wall,
    wall=None,
    wall_sep=0.0
):
    """
    if bool_attrac==False:
    The force is derived from the potential

    U(r) = U0 + U0 * (2*a-r)/b   if z<2*a
    U(r) = U0 * exp(-(r-2*a)/b)  iz z>=2*a

    with
    eps = potential strength
    r_norm = distance between blobs
    b = Debye length
    a = blob_radius


    if bool_attrac==True:
    The force is derived from the potential
    U(r) = D_e_J * ( 1 - exp(-w(r-r_e)) )**2 

    with r the distance between the centers of the blobs

    Which means the force (projected on the radial vector) is :
    F(r) = -2*w*D_e_J * ( 1 - exp(-w(r-r_e) ) * exp(-w(r-r_e))
    """

    N = r_vectors.size // 3
    force = np.zeros((N, 3))

    for i in prange(N):
        # for j in range(N):
        for kk in range(offsets[i + 1] - offsets[i]):
            j = list_of_neighbors[offsets[i] + kk]

            if i == j:
                continue

            dr = np.zeros(3)
            for k in range(3):
                dr[k] = r_vectors[j, k] - r_vectors[i, k]
                #part that take into account the boundary conditions to calculate distances
                if L[k] > 0:
                    dr[k] -= (
                        int(dr[k] / L[k] + 0.5 * (int(dr[k] > 0) - int(dr[k] < 0)))
                        * L[k]
                    )

            # Compute force
            r_norm = np.sqrt(dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2])

            offset = 2.0 * a * (1 - delta)
            temp_r = max(r_norm, 1.0e-12)
            inv_r_norm = 1 / temp_r
            
            #colloid-colloid interaction
            if bool_attrac==False:
                #steric interaction between colloids
                if r_norm > offset:
                    prefactor = (
                        -(repulsion_strength / debye_length)
                        * np.exp(-(r_norm - offset) / debye_length)
                        * inv_r_norm
                    )
                else:
                    prefactor = -(repulsion_strength / debye_length) * inv_r_norm

                force[i] += prefactor * dr

            else:
                #Morse interaction between colloids
                r_e_centers=r_e+2*a
                prefactor = 2*w*(D_e*kbt)*(1-np.exp(-w*(r_norm-r_e_centers)))*np.exp(-w*(r_norm-r_e_centers))
                force[i] += prefactor * dr *inv_r_norm


        #wall interaction
        if wall == 'single_wall':
            h = r_vectors[i, 2]
            if bool_attrac_wall==False:
                # wall sterics
                force[i, :] += wall_forces(a, repulsion_strength, debye_length, delta, h)
            else:
                # wall Morse interaction
                force[i, :] += wall_forces_attrac(a, w, D_e, r_e, fact_wall, delta, h)

        elif wall == 'two_walls':
            # assert False
            # bottom wall
            h = r_vectors[i, 2]
            force[i, :] += wall_forces(a, repulsion_strength, debye_length, delta, h)

            # top wall
            h_from_top = wall_sep - r_vectors[i, 2]
            force[i, :] -= wall_forces(a, repulsion_strength, debye_length, delta, h_from_top)

    return force

@njit(fastmath=True)
def wall_forces(a, repulsion_strength, debye_length, delta, h):
    force = np.zeros(3)
    offset = a * (1 - delta)
    if h < offset:  # bottom wall
        force[2] += repulsion_strength / debye_length
    else:
        force[2] += (repulsion_strength / debye_length) * np.exp(
                    -(h - offset) / debye_length
                )
    return force

@njit(fastmath=True)
def wall_forces_attrac(a, w, D_e, r_e, fact_wall,delta, h):
    # we treat the repulsion with the wall as a repulsion between two particles but with a factor of "fact_wall" (Morse Potential)
    r_e_centers2= r_e+2*a
    r_virtual = a +h #repulsion as if there were a virtual particlein the wall (of radius a)
    force = np.zeros(3)
    force[2] += -fact_wall*2*w*D_e*kbt*(1-np.exp(-w*(r_virtual-r_e_centers2)))*np.exp(-w*(r_virtual-r_e_centers2))
    return force

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--a',               type=float, default=1.395)
    parser.add_argument('--phi',             type=float, default=0.114)

    parser.add_argument('--bool_attrac',     type=bool,  default=True) #decides wether or not we add the depletion attraction
    parser.add_argument('--bool_attrac_wall',type=bool,  default=True) #decides wether or not we add the depletion attraction with the wall (avoids stacking at high packing fraction)
    parser.add_argument('--range_attrac',    type=float, default=1.395*1.5) #it decides where we cut the potential
    parser.add_argument('--D_e',             type=float, default=2) #well depth, in kbt 
    parser.add_argument('--w',               type=float, default=6) #controls the width of the well
    parser.add_argument('--r_e',             type=float, default=0.16) #equilibrium bond distance between surfaces of colloids
    parser.add_argument('--fact_wall',       type=float, default=1/0.6) #factor of intensity of interaction with the wall in comparison to the one between particles

    parser.add_argument('--L',               type=float, default=640)
    parser.add_argument('--dt',              type=float, default=0.1)
    parser.add_argument('--t_final',         type=float, default=60.0 * 60 * 1)
    parser.add_argument('--t_save',          type=float, default=0.1)
    parser.add_argument('--solver',          type=str,   default='Self')
    parser.add_argument('--wall',            type=str,   default='single_wall')
    parser.add_argument('--wall_sep',        type=float, default=None) # this is for two_walls iirc
    parser.add_argument('--z_width',         type=float, default=None)
    parser.add_argument('--z_position',      type=float, default=None)
    parser.add_argument('--initial_dist',    type=str, default='flat')
    parser.add_argument('--nograv', action='store_true')
    args = parser.parse_args()

    mg = 0.0592  # m*g, in pN
    if args.nograv:
        mg = 0
    else:
        g = scipy.constants.g # m/s2
        delta_rho = 1510 - 970 # kg/m3. 1510 from Soft Matter SI, 970 from Eleanor in Slack
        V = (4/3) * np.pi * (args.a * 1e-6)**3  # m3
        mg = delta_rho * V * g * 1e12  # pN
        print('mg', mg)
    
    kbt = 0.0041419464  # aJ
    # T = kbt / 1e18 / scipy.constants.Boltzmann # K
    # T = 273.15 + 21
    # kbt = scipy.constants.Boltzmann * T * 1e18 # aJ
    # assert 0.003 < kbt < 0.005
    if args.a < 2.5/2:
        eta = 1e-3
    else:
        eta = 1.75e-3 # Pa s = kg m / s. From Eleanor in Slack

    main(
        a = args.a,
        mg = mg,
        Lx = args.L,
        Ly = args.L,
        kbt = kbt,
        eta = eta,
        phi = args.phi,
        bool_attrac = args.bool_attrac,
        bool_attrac_wall = args.bool_attrac_wall,
        range_attrac = args.range_attrac,
        D_e = args.D_e,
        w = args.w,
        r_e =args.r_e,
        fact_wall=args.fact_wall,
        dt = args.dt,
        t_final = args.t_final,
        t_save = args.t_save,
        solver_name = args.solver,
        wall = args.wall,
        wall_sep = args.wall_sep * args.a if args.wall_sep else 0.0,
        z_trap_position = args.z_position * args.a if args.z_position is not None else None,
        z_trap_width = args.z_width * args.a if args.z_width is not None else None,
        initial_distribution = args.initial_dist,
    )