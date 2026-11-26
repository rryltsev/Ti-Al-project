"""Small scientific checks; synthetic values never enter production outputs."""
import unittest
import tempfile
import json
from unittest.mock import patch
import analyze
from pathlib import Path
import numpy as np
from analyze import metric, readiness, compare, parity, alignment_figure, XC


class MetricsTests(unittest.TestCase):
    def test_versioned_checkpoint_cache_cannot_mix_models(self):
        sid='synthetic-fixed'
        bundle={'records':[{'metadata':{'structure_id':sid,'poscar_sha256':'geometry'},'symbols':['Ti']}]}
        with tempfile.TemporaryDirectory() as folder, patch.object(analyze,'ROOT',Path(folder)):
            manifest={}
            for directory,model,energy in [('PBE','old-model',-5.),('PBE-15000','new-model',-6.)]:
                path=analyze.prediction_path('PBE',sid,directory);path.parent.mkdir(parents=True)
                path.write_text(json.dumps({'structure_id':sid,'functional':'PBE','model_sha256':model,
                    'poscar_sha256':'geometry','symbols':['Ti'],'energy_offset_applied_eV':0,
                    'energy_eV':energy,'forces_eV_A':[[0.,0.,0.]],'virial_eV':np.zeros((3,3)).tolist()}))
                manifest[path.relative_to(folder).as_posix()]=analyze.sha(path)
            pred=analyze.load_predictions(bundle,'PBE','new-model',manifest,[],'PBE-15000')
            self.assertEqual(pred[sid]['energy_eV'],-6.)
            with self.assertRaises(AssertionError):
                analyze.load_predictions(bundle,'PBE','new-model',manifest,[])
            with self.assertRaises(AssertionError):
                analyze.prediction_path('PBE',sid,'../PBE')

    def test_component_weighting_and_structure_weighting_differ(self):
        m=metric([np.zeros(3), np.full(6,2.0)])
        self.assertAlmostEqual(m['mae'],4/3)
        self.assertAlmostEqual(m['rmse'],np.sqrt(8/3))
        self.assertAlmostEqual(m['equal_structure_mae'],1)
        self.assertAlmostEqual(m['equal_structure_rmse'],np.sqrt(2))
        self.assertEqual(m['n_structures'],2)

    def test_bootstrap_preserves_per_structure_correlations(self):
        small=metric([[-1],[3]])
        repeated=metric([np.full(1000,-1.),np.full(1000,3.)])
        self.assertEqual(small['rmse_ci95'], repeated['rmse_ci95'])
        self.assertEqual(small['mae_ci95'], repeated['mae_ci95'])

    def test_absolute_offsets_are_not_fitted_away(self):
        m=metric([[10],[10]],[[0],[1]])
        self.assertEqual(m['rmse'],10)
        self.assertEqual(m['bias'],10)

    def test_empty_species_has_no_fake_zero(self):
        self.assertIsNone(metric([[],[]]))

    def test_small_liquid_only_subset_cannot_trigger_stop(self):
        r={'common_ids':list(range(6)), 'coverage':{'liquid':{'completed':6,'composition_bins_ready':[0,1,2], 'composition_bins_planned':[0,1,2,3,4]}}, 'metrics':None}
        decision=readiness(r,[])
        self.assertFalse(decision['candidate_for_discussion'])
        self.assertFalse(decision['automatic_stop'])

    def test_paired_comparison_and_plot_export(self):
        records=[]; predictions={xc:{} for xc in XC}
        for i in range(2):
            sid=f'synthetic-{i}'
            label={'energy_eV':-10.+i, 'forces_eV_A':[[.1,0.,0.],[-.1,0.,0.]], 'stress_eV_A3':np.zeros((3,3)).tolist(), 'volume_A3':20.}
            records.append({'metadata':{'structure_id':sid,'family':'liquid','x_Al':.5}, 'symbols':['Al','Ti'], 'labels':{xc:label for xc in XC}})
            for xc in XC:
                predictions[xc][sid]={'energy_eV':label['energy_eV']+.02,'forces_eV_A':label['forces_eV_A'],'virial_eV':np.zeros((3,3)).tolist()}
        values={}; metrics={}
        for xc in XC:
            values[xc],metrics[xc]=compare(records,predictions,xc)
            self.assertAlmostEqual(metrics[xc]['all']['energy_meV_atom']['rmse'],10.)
            self.assertEqual(metrics[xc]['all']['force_eV_A']['rmse'],0.)
            self.assertEqual(metrics[xc]['all']['stress_GPa']['rmse'],0.)
        with tempfile.TemporaryDirectory() as folder:
            parity(values,metrics,Path(folder),2,values,metrics,values,metrics)
            alignment_figure(values,metrics,values,metrics,Path(folder),2)
            self.assertGreater((Path(folder)/'parity.png').stat().st_size,1000)
            self.assertTrue((Path(folder)/'parity.pdf').read_bytes().startswith(b'%PDF'))
            self.assertTrue((Path(folder)/'energy_alignment.pdf').read_bytes().startswith(b'%PDF'))

    def test_one_pretrained_prediction_against_three_reference_energies(self):
        label=lambda e: {'energy_eV':e,'forces_eV_A':[[0.,0.,0.],[0.,0.,0.]],
                         'stress_eV_A3':np.zeros((3,3)).tolist(),'volume_A3':20.}
        records=[{'metadata':{'structure_id':'fixed','family':'liquid','x_Al':.5},
                  'symbols':['Al','Ti'],'labels':dict(zip(XC,[label(-10),label(-11),label(-14)]))}]
        shared={'fixed':{'energy_eV':-9.,'forces_eV_A':[[0.,0.,0.],[0.,0.,0.]],
                         'virial_eV':np.zeros((3,3)).tolist()}}
        predictions={xc:shared for xc in XC}
        for xc,expected in zip(XC,[500.,1000.,2500.]):
            values,metrics=compare(records,predictions,xc)
            self.assertEqual(values[0]['e_pred'],-4.5)
            self.assertEqual(metrics['all']['energy_meV_atom']['rmse'],expected)


if __name__=='__main__':
    unittest.main()
