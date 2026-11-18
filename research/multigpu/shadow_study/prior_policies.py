"""Offline candidates for the validated local backend; no simulator changes.

These are explicit, uncalibrated demand proxies, not fitted runtime predictors.
Only route plans and topology are read: speed/position/state in snapshots are
ignored. The caller supplies an expected simultaneous ROAD population. A future
forecast can replace these features without changing the ownership interface.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

POLICIES = ("work_balanced", "boundary_aware")


@dataclass
class Features:
    source: np.ndarray
    target: np.ndarray
    lanes: np.ndarray
    visits: np.ndarray
    road_population: np.ndarray
    transition_per_step: np.ndarray
    owned_population: np.ndarray
    node_count: int
    expected_active: float
    dt: float


def build_features(geometry, cars, expected_active=None, dt=.25):
    """Stationary free-flow residence proxy, not a queue/occupancy forecast.

    lambda = expected_active / sum(route road travel times). Then road e has
    lambda * visits[e] * length[e]/speed[e] predicted residents. This conserves
    the supplied population. It ignores departures, signals and connector time;
    finite-horizon correctness of this proxy is NOT claimed.
    """
    roads = geometry["roads"]
    active = float(len(cars) if expected_active is None else expected_active)
    if not roads or not math.isfinite(active) or active <= 0 or not math.isfinite(dt) or dt <= 0:
        raise ValueError("Positive finite population, timestep and nonempty roads required")
    source = np.asarray([r["a"] for r in roads], dtype=np.int64)
    target = np.asarray([r["b"] for r in roads], dtype=np.int64)
    if min(source.min(), target.min()) < 0:
        raise ValueError("Negative junction ID")
    n = int(max(source.max(), target.max())) + 1
    lanes = np.asarray([r["lanes"] for r in roads], dtype=np.float64)
    length = np.asarray([r["length"] for r in roads], dtype=np.float64)
    speed = np.asarray([r["speed"] for r in roads], dtype=np.float64)
    if any(not np.isfinite(x).all() or np.any(x <= 0) for x in (lanes, length, speed)):
        raise ValueError("Invalid road dimensions/speed")
    visits = np.zeros(len(roads), dtype=np.float64)
    for car in cars:
        size = int(car[2])
        if not 1 <= size <= 32:
            raise ValueError("Invalid route length")
        route = np.asarray(car[6:6+size], dtype=np.int64)
        if np.any(route < 0) or np.any(route >= len(roads)):
            raise ValueError("Invalid route road")
        np.add.at(visits, route, 1)
    residence = visits * length / speed
    if residence.sum() <= 0:
        raise ValueError("No route demand")
    rate = active / residence.sum()
    population = rate * residence
    transitions = rate * visits * dt
    owned = np.bincount(target, weights=population, minlength=n)
    return Features(source, target, lanes, visits, population, transitions, owned, n, active, dt)


def describe(owners, features, parts):
    """Consistent proxy diagnostics for old and new ownership maps.

    The backend owns road vehicles at the destination junction and ghosts them
    at the source. Two refreshes/step send 432-byte records; reservation exchange
    sends one 24-byte record/lane/step, even for unvisited cut roads. Migration
    bytes are a separate rate proxy; they can overlap refresh records in the real
    transport. Therefore the sum is a heuristic objective, not predicted wire
    bytes. Pool capacities and full device memory require separate measurement.
    """
    owner = np.asarray(owners, dtype=np.int64)
    if len(owner) != features.node_count or set(owner) != set(range(parts)):
        raise ValueError("Invalid/empty partition map")
    a, b = owner[features.source], owner[features.target]
    cut = a != b
    owned = np.bincount(owner, weights=features.owned_population, minlength=parts)
    ghost = np.bincount(a[cut], weights=features.road_population[cut], minlength=parts)
    refresh = float(2 * 432 * features.road_population[cut].sum())
    reservation = float(24 * features.lanes[cut].sum())
    migration = float(432 * features.transition_per_step[cut].sum())
    # Directed pairs of connected ranks, not individual roads or CUDA launches.
    pairs = set(zip(a[cut].tolist(), b[cut].tolist()))
    pairs |= {(b, a) for a, b in list(pairs)}
    return dict(
        expected_owned=owned.tolist(), expected_ghosts=ghost.tolist(),
        expected_visible=(owned + ghost).tolist(),
        owned_max_mean=float(owned.max() / owned.mean()),
        route_cut=float(features.visits[cut].sum()), cut_roads=int(cut.sum()),
        ghost_refresh_byte_proxy_per_step=refresh,
        reservation_bytes_per_step=reservation,
        migration_byte_proxy_per_step=migration,
        boundary_objective=refresh + reservation + migration,
        directed_neighbor_pairs=len(pairs),
        predicted_runtime_ms=None, measured_speedup=None,
    )


def partition(geometry, cars, policy, parts=3, seed=2710, *, expected_active=None,
              imbalance=.05, dt=.25, features=None):
    """Same ownership-map contract as local_scaling.policies.partition.

    work_balanced changes METIS vertex weights to residence-based road work;
    boundary_aware additionally changes edges to refresh/reservation/migration
    proxies. No future traces or cost coefficients fitted to evaluation runs.
    """
    import pymetis

    started = time.perf_counter()
    if policy not in POLICIES or not math.isfinite(imbalance) or not 0 < imbalance <= 1:
        raise ValueError("Invalid policy or imbalance")
    f = features if features is not None else build_features(geometry, cars, expected_active, dt)
    if not 1 <= parts <= f.node_count:
        raise ValueError("Invalid partition count")
    # Work quantization gives the mean vertex ~1000 integer units. Minimum 1
    # retains empty topology without asserting a measured cost for that topology.
    scale = 1000 * f.node_count / f.expected_active
    vertex = np.maximum(1, np.rint(f.owned_population * scale)).astype(np.int64)
    edge = f.visits if policy == "work_balanced" else (
        2 * 432 * f.road_population + 24 * f.lanes + 432 * f.transition_per_step)
    graph = {}
    for a, b, weight in zip(f.source, f.target, edge):
        a, b = int(a), int(b)
        if a != b:
            key = tuple(sorted((a, b)))
            graph[key] = graph.get(key, 0.) + float(weight)
    # Preserve every real topology edge; an unvisited edge has minimum weight 1.
    norm = 1000 / max(1., float(np.mean(list(graph.values())))) if graph else 1.
    adj = [[] for _ in range(f.node_count)]
    for (a, b), weight in sorted(graph.items()):
        w = max(1, int(round(weight * norm)))
        adj[a].append((b, w)); adj[b].append((a, w))
    xadj, neighbors, weights = [0], [], []
    for row in adj:
        for b, w in sorted(row):
            neighbors.append(b); weights.append(w)
        xadj.append(len(neighbors))
    if max(sum(vertex.tolist()), sum(weights)) >= 2**31 - 1:
        raise ValueError("METIS integer weights overflow; rescale explicitly")
    if parts == 1:
        owners, cut = [0] * f.node_count, 0
    else:
        cut, owners = pymetis.part_graph(
            parts, xadj=xadj, adjncy=neighbors, vweights=vertex.tolist(),
            eweights=weights, recursive=False,
            options=pymetis.Options(seed=seed, ufactor=max(1, round(1000*imbalance))))
        owners = list(owners)
    stats = describe(owners, f, parts)
    quantized = np.bincount(owners, weights=vertex, minlength=parts)
    return owners, dict(
        policy=policy, parts=parts, seed=seed,
        construction_ms=1000*(time.perf_counter()-started),
        includes_feature_construction=features is None,
        future_trajectory_access=False, forecast="uncalibrated stationary route-residence proxy",
        expected_road_population=f.expected_active, dt=f.dt,
        solver="METIS multilevel k-way with weighted vertices",
        imbalance_target=1+imbalance,
        achieved_quantized_max_mean=float(quantized.max()/quantized.mean()),
        balance_target_met=bool(quantized.max()/quantized.mean() <= 1+imbalance+1e-9),
        metis_cut=int(cut), statistics=stats,
        limitations=["No calibrated junction, connector or topology cost model",
                     "No device memory budget or online pool-growth guarantee",
                     "Proxy scores are not runtime or finite-horizon traffic predictions"],
    )


def synchronized_step_ms(stage_rank_ms, fixed_ms=0.):
    """Sum of slowest-rank costs at EACH barrier, not max of rank totals.

    Input must include pack/list/transport costs and be measured or independently
    calibrated for the candidate map. This function does not infer them from
    occupancy/scoreboard percentages or assume asynchronous overlap exists.
    """
    stages = np.asarray(stage_rank_ms, dtype=float)
    if (stages.ndim != 2 or not stages.size or not np.isfinite(stages).all()
            or np.any(stages < 0) or not math.isfinite(fixed_ms) or fixed_ms < 0):
        raise ValueError("Invalid nonnegative stage/rank durations")
    return float(stages.max(axis=1).sum() + fixed_ms)


def switch_decision(current_ms, candidate_ms, horizon_steps, switch_ms,
                    uncertainty_ms=0., margin=.10):
    """A gate only: no online ownership migration is implemented here.

    Compare paired windows, charging all forecast/partition/migration costs in
    switch_ms, plus a conservative TOTAL horizon uncertainty allowance.
    """
    old, new = np.asarray(current_ms, dtype=float), np.asarray(candidate_ms, dtype=float)
    steps = np.asarray(horizon_steps, dtype=float)
    if (old.ndim != 1 or not old.size or old.shape != new.shape or old.shape != steps.shape
            or any(not np.isfinite(x).all() for x in (old, new, steps))
            or np.any(old < 0) or np.any(new < 0) or np.any(steps <= 0)
            or not all(math.isfinite(x) for x in (switch_ms, uncertainty_ms, margin))
            or min(switch_ms, uncertainty_ms, margin) < 0 or margin >= 1):
        raise ValueError("Invalid forecast horizon or costs")
    baseline = float(old @ steps)
    gross = float((old-new) @ steps)
    net = gross-switch_ms-uncertainty_ms
    return dict(switch=net > margin*baseline, baseline_ms=baseline,
                gross_saving_ms=gross, conservative_net_saving_ms=net,
                required_saving_ms=margin*baseline)
