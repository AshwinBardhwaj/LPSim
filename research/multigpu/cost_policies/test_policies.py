"""Contract tests for forecasts and selection; physical gates run separately."""
import unittest
import numpy as np
from policies import build_features, partition, describe, synchronized_step_ms, switch_decision


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.geo = {'roads': [dict(a=i, b=(i+1)%12, lanes=2,
                                   length=10*(i+1), speed=10) for i in range(12)]}
        self.cars = [[1, 0, 3, -1, 0, 0, i, (i+1)%12, (i+2)%12] for i in range(12)]

    def test_population_conservation_and_residence(self):
        f = build_features(self.geo, self.cars, 120)
        self.assertAlmostEqual(f.road_population.sum(), 120)
        self.assertAlmostEqual(f.owned_population.sum(), 120)
        self.assertAlmostEqual(f.road_population[-1]/f.road_population[0], 12)

    def test_not_conditioned_on_vehicle_states_or_storage_order(self):
        altered = [c[:] for c in reversed(self.cars)]
        for c in altered:
            c[0], c[1], c[3], c[4], c[5] = 0, 2, 11, 5, 3
        a, b = build_features(self.geo, self.cars), build_features(self.geo, altered)
        np.testing.assert_array_equal(a.owned_population, b.owned_population)
        np.testing.assert_array_equal(a.visits, b.visits)

    def test_ownership_ghost_direction_and_empty_road_reservations(self):
        geo = {'roads': [dict(a=0, b=1, lanes=2, length=10, speed=10),
                         dict(a=1, b=0, lanes=1, length=10, speed=10)]}
        f = build_features(geo, [[1, 0, 1, -1, 0, 0, 0]], 10)
        d = describe([0, 1], f, 2)
        self.assertEqual(d['expected_owned'], [0., 10.])
        self.assertEqual(d['expected_ghosts'], [10., 0.])
        self.assertEqual(d['ghost_refresh_byte_proxy_per_step'], 8640.)
        self.assertEqual(d['reservation_bytes_per_step'], 72.)

    def test_valid_deterministic_maps(self):
        for policy in ('work_balanced', 'boundary_aware'):
            for parts in (1, 2, 3):
                a, _ = partition(self.geo, self.cars, policy, parts)
                b, _ = partition(self.geo, self.cars, policy, parts)
                self.assertEqual(a, b)
                self.assertEqual(set(a), set(range(parts)))

    def test_barrier_critical_path_not_total_per_rank(self):
        self.assertEqual(synchronized_step_ms([[10, 1], [1, 10]], fixed_ms=2), 22)

    def test_switch_requires_amortizing_costs_and_uncertainty(self):
        self.assertFalse(switch_decision([3], [2], [10], 12)['switch'])
        self.assertFalse(switch_decision([3], [2], [10], 1, uncertainty_ms=7)['switch'])
        self.assertTrue(switch_decision([3], [2], [100], 12)['switch'])

    def test_invalid_input_rejected(self):
        with self.assertRaises(ValueError):
            build_features(self.geo, self.cars, -1)
        with self.assertRaises(ValueError):
            describe([0]*12, build_features(self.geo, self.cars), 2)
        with self.assertRaises(ValueError):
            synchronized_step_ms([[float('nan')]])
        with self.assertRaises(ValueError):
            switch_decision([3], [2, 1], [10], 1)


if __name__ == '__main__':
    unittest.main()
