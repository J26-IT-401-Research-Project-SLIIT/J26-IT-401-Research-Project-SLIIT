"""Small hand-written fixtures only. No fitting, raw processing or dataset access."""
import ast
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import pp1_training as t


def fixture():
    features=[{'session_id':'tiny_a',**{k:'1.0' for fields in t.FEATURES.values() for k in fields}}]
    targets=[{'session_id':'tiny_a','TS':'0','BS':'1','RS':'2','GI':'3','demo_MPS':'0.375'}]
    splits=[{'session_id':'tiny_a','split':'train'}]
    return features,targets,splits,[dict(r) for r in splits]


class Tests(unittest.TestCase):
    def test_metrics_fixed_five_classes(self):
        m=t.metrics([0,1,2,3,4],[0,2,2,1,4])
        self.assertEqual(m['exact_accuracy'],.6)
        self.assertEqual(m['within_1_accuracy'],.8)
        self.assertEqual(m['MAE'],.6)
        self.assertEqual(m['class_support'],[1]*5)
        self.assertAlmostEqual(m['macro_f1'],(1+0+2/3+0+1)/5)
        self.assertEqual(m['balanced_accuracy'],.6)
        self.assertEqual(len(m['confusion_matrix']),5)

    def test_missing_class_not_dropped_from_macro_f1(self):
        m=t.metrics([0,0],[0,0],True,20)
        self.assertEqual(m['macro_f1'],.2)
        self.assertEqual(m['balanced_accuracy'],1)
        self.assertEqual(m['unsupported_true_classes'],[1,2,3,4])
        self.assertIsNone(m['uncertainty']['class_recall_wilson_95'][1])

    def test_bad_predictions_rejected(self):
        for values in ([5],[1.5],[float('nan')]):
            with self.assertRaises(ValueError): t.metrics([0],values)
        with self.assertRaises(ValueError): t.metrics([0],[])

    def test_cv_rare_policy(self):
        self.assertEqual(t.cv_folds([0]*3+[1]*3),3)
        self.assertEqual(t.cv_folds([0]*3+[1]*2),2)
        self.assertEqual(t.cv_folds([0]*3+[1]),0)
        with self.assertRaises(ValueError): t.cv_folds([0,0])

    def test_missing_class_encoding_preserves_ordinal_distances(self):
        classes,codes=t.encode_labels([0,2,4,2])
        self.assertEqual(codes,[0,1,2,1])
        self.assertEqual(t.decode_labels(classes,codes),[0,2,4,2])
        self.assertEqual(t.metrics([0],[t.decode_labels(classes,[2])[0]])['MAE'],4)

    def test_cv_tiebreak(self):
        a={'mean_fold_MAE':.2,'mean_fold_macro_f1':.4,'configuration_index':0}
        b={**a,'configuration_index':1}
        self.assertLess(t.cv_rank(a),t.cv_rank(b))
        self.assertLess(t.cv_rank({**b,'mean_fold_macro_f1':.5}),t.cv_rank(a))

    def test_validation_tiebreak(self):
        a={'MAE':.2,'macro_f1':.4,'algorithm':'RF'}
        b={**a,'algorithm':'SVM'}
        self.assertLess(t.validation_rank(a,['RF','SVM','XGB','CatBoost']),t.validation_rank(b,['RF','SVM','XGB','CatBoost']))

    def test_allowlisted_fixture(self):
        f,y,s=t.validate_tables(*fixture())
        self.assertEqual(set(s),{'tiny_a'})

    def test_leakage_column_rejected(self):
        data=fixture(); data[0][0]['profile']='healthy_like'
        with self.assertRaises(ValueError): t.validate_tables(*data)

    def test_changed_split_rejected(self):
        data=fixture(); data[2][0]['split']='test'
        with self.assertRaises(ValueError): t.validate_tables(*data)

    def test_duplicate_id_rejected(self):
        with self.assertRaises(ValueError): t.by_id([{'session_id':'a'},{'session_id':'a'}])

    def test_invalid_reference_mps_rejected(self):
        data=fixture(); data[1][0]['demo_MPS']='.9'
        with self.assertRaises(ValueError): t.validate_tables(*data)

    def test_nonfinite_feature_rejected(self):
        data=fixture(); data[0][0]['TS_tremor_rms_ms2']='nan'
        with self.assertRaises(ValueError): t.validate_tables(*data)

    def test_system_mps_and_missing_prediction(self):
        tables=t.validate_tables(*fixture())
        p={d:{'tiny_a':v} for d,v in zip(t.DOMAINS,[0,1,2,3])}
        rows,m=t.system_report(tables,['tiny_a'],p,20)
        self.assertEqual(m['MPS_MAE'],0)
        self.assertEqual(rows[0]['predicted_demo_MPS'],.375)
        del p['GI']['tiny_a']
        _,m=t.system_report(tables,['tiny_a'],p,20)
        self.assertEqual(m['complete_session_coverage'],0)
        self.assertIsNone(m['MPS_MAE'])

    def test_bootstrap_reproducible(self):
        self.assertEqual(t.mean_interval([0,.5,1],replicates=20),t.mean_interval([0,.5,1],replicates=20))
        lower,upper=t.wilson(1,1)
        self.assertLess(lower,.3); self.assertAlmostEqual(upper,1)

    def test_approval_gates(self):
        for fn in (t.train,t.validate):
            with self.assertRaises(PermissionError): fn('unused')
        with self.assertRaises(PermissionError): t.final_test('unused','unused')

    def test_fit_and_predict_runtime_gate_before_ml_import(self):
        with patch.object(t,'require_colab',side_effect=RuntimeError('local blocked')):
            with self.assertRaises(RuntimeError): t.fit_candidate('RF',{},None,None,1)
            with self.assertRaises(RuntimeError): t.predict({},None)

    def test_test_claim_cannot_repeat_after_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            t.claim_final_test(tmp,'fixture','run1')
            with self.assertRaises(FileExistsError): t.claim_final_test(tmp,'fixture','run2')

    def test_hash_mutation_and_exclusive_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'file.json'; t.save_json(p,{'x':1}); hashes={'file.json':t.sha(p)}
            t.verify_hashes(tmp,hashes)
            with self.assertRaises(FileExistsError): t.save_json(p,{'x':2})
            p.write_text('{}')
            with self.assertRaises(ValueError): t.verify_hashes(tmp,hashes)

    def test_unsafe_paths(self):
        for path in ['../x','/abs','a/../../x','C:/x','a\\x']:
            with self.assertRaises(ValueError): t.safe_relative(Path.cwd(),path)

    def test_search_budget_and_defaults(self):
        c=t.read_json(t.PACKAGE/'configs/search_spaces.json')
        self.assertEqual(set(c['spaces']),{'RF','SVM','XGB','CatBoost'})
        self.assertTrue(all(1<=len(v)<=4 for v in c['spaces'].values()))
        self.assertEqual(c['default_configuration_index'],0)

    def test_training_matrix_uses_only_requested_split(self):
        # Two invented table rows; array construction only, no fitting.
        features,targets,splits,plan=fixture()
        features.append({**features[0],'session_id':'tiny_b'})
        targets.append({**targets[0],'session_id':'tiny_b'})
        splits.append({'session_id':'tiny_b','split':'test'}); plan.append(dict(splits[-1]))
        tables=t.validate_tables(features,targets,splits,plan)
        ids,x,y=t.matrix(tables,'TS','train')
        self.assertEqual(ids,['tiny_a']); self.assertEqual(x.shape,(1,1))


if __name__=='__main__': unittest.main()
