import unittest
import numpy as np
from energy_alignment import solve, apply

class AlignmentTests(unittest.TestCase):
    def test_recover_element_offsets_with_different_sizes(self):
        rows=[];energies={}
        for i,(al,ti) in enumerate([(2,0),(0,7),(3,4),(50,50)]):
            energies[str(i)]=-8*(al+ti)
            rows.append(dict(split='train',geometry_id=i,n_Al=al,n_Ti=ti,n_atoms=al+ti,energy_eV=energies[str(i)]+al*.25-ti*1.7))
        c=solve(rows,energies)['coefficients_eV_per_atom']
        np.testing.assert_allclose([c['Al'],c['Ti']],[.25,-1.7],atol=1e-14)
        self.assertNotIn('test',str(rows))

    def test_equal_frame_weight_not_atom_weight(self):
        rows=[dict(split='train',geometry_id=i,n_Al=n if i<2 else 0,n_Ti=n if i==2 else 0,n_atoms=n,energy_eV=e*n) for i,(n,e) in enumerate([(1,1),(100,3),(2,5)])]
        c=solve(rows,dict.fromkeys(['0','1','2'],0))['coefficients_eV_per_atom']
        np.testing.assert_allclose([c['Al'],c['Ti']],[2,5])

    def test_reject_validation_or_unidentifiable_design(self):
        row=dict(split='validate',geometry_id=0,n_Al=1,n_Ti=1,n_atoms=2,energy_eV=-1)
        with self.assertRaises(AssertionError): solve([row],{'0':0})
        row['split']='train'
        with self.assertRaises(AssertionError): solve([row],{'0':0})

    def test_only_total_energy_changes_and_raw_cache_is_untouched(self):
        p=dict(symbols=['Ti','Al','Ti'],energy_eV=-10.,energy_offset_applied_eV=0.,forces_eV_A=[[1,2,3]]*3,virial_eV=list(range(9)))
        a=apply(p,dict(Al=.5,Ti=-2))
        self.assertEqual(a['energy_eV'],-13.5)
        self.assertEqual(p['energy_eV'],-10.)
        self.assertEqual(a['forces_eV_A'],p['forces_eV_A'])
        self.assertEqual(a['virial_eV'],p['virial_eV'])
        with self.assertRaises(AssertionError): apply(a,dict(Al=.5,Ti=-2))

if __name__=='__main__': unittest.main()
