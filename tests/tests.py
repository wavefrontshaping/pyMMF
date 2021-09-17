import unittest
import numpy as np
import pyMMF

class TestRadialSolver(unittest.TestCase):

    def test_GRIN(self):
        print('Testing radial solver with GRIN fiber')
        data = np.load('GRIN_test_radial.npz', allow_pickle = True)
        print('>>', data)
        params = data['params'][()]
        M0_benchmark = data['M0']
        
        # Recompute modes
        profile = pyMMF.IndexProfile(
            npoints = params['n_points_modes'], 
            areaSize = params['areaSize']
        )
        profile.initParabolicGRIN(
            n1 = params['n1'], 
            a = params['radius'], 
            NA = params['NA']
        )

        solver = pyMMF.propagationModeSolver()
        solver.setIndexProfile(profile)
        solver.setWL(params['wl'])
        modes = solver.solve(mode = params['mode'],
                            curvature = params['curvature'],
                            r_max = params['r_max'], 
                            dh = params['dh'], 
                            min_radius_bc = params['min_radius_bc'],
                            change_bc_radius_step = params['change_bc_radius_step'],
                            N_beta_coarse = params['N_beta_coarse'], 
                            degenerate_mode = params['degenerate_mode'],
                            )
        
        M0_test = modes.getModeMatrix()
        print('\nComputation done\n')
        print('-'*25)
        
        
        print('Checking the number of modes')
        self.assertEqual(M0_test.shape[1], M0_benchmark.shape[1])
        
        
        print('Checking all the mode profiles')
        for m0, m1 in zip(M0_test.transpose(), M0_benchmark.transpose()):
            np.testing.assert_array_almost_equal(m0, m1)
        
if __name__ == '__main__':
    unittest.main()