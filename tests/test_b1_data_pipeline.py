import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.io import savemat

from prism.data.b1_dataset import (B1DataError, B1Dataset, arrays_to_tensors,
                                  load_raw_mat, load_srgb_png, resolve_project_path,
                                  validate_pair)
from prism.data.b1_packing import (apply_dihedral, pack_bayer, sample_augmentation, transform_cfa_pattern,
                                   transform_paired, unpack_bayer)


class BayerPackingTests(unittest.TestCase):
    def test_exact_channel_order_and_inverse(self):
        mosaic=np.arange(8*10,dtype=np.float32).reshape(8,10)
        packed=pack_bayer(mosaic)
        self.assertEqual(packed.shape,(4,4,5))
        for channel,expected in enumerate((mosaic[0::2,0::2],mosaic[0::2,1::2],mosaic[1::2,0::2],mosaic[1::2,1::2])):
            np.testing.assert_array_equal(packed[channel],expected)
        np.testing.assert_array_equal(unpack_bayer(packed),mosaic)

    def test_cfa_values_stay_in_fixed_spatial_positions(self):
        levels={'R':7.0,'G':3.0,'B':1.0}
        for pattern in ('BGGR','GRBG','RGGB'):
            raw=np.tile(np.array([levels[c] for c in pattern],dtype=np.float32).reshape(2,2),(3,4))
            np.testing.assert_array_equal(pack_bayer(raw)[:,0,0],[levels[c] for c in pattern])
            np.testing.assert_array_equal(unpack_bayer(pack_bayer(raw)),raw)

    def test_odd_dimensions_and_bad_packed_shapes_rejected(self):
        with self.assertRaisesRegex(ValueError,'even'):
            pack_bayer(np.zeros((5,4),dtype=np.float32))
        with self.assertRaisesRegex(ValueError,'shape'):
            unpack_bayer(np.zeros((3,2,2),dtype=np.float32))

    def test_all_paired_dihedral_ops_preserve_coordinates_and_cfa_phase(self):
        raw=np.arange(8*8,dtype=np.float32).reshape(8,8)
        target=np.stack((raw,raw+100,raw+200),axis=2).astype(np.float32)
        names=('identity','rot90','rot180','rot270','flip_horizontal','flip_vertical','transpose','anti_transpose')
        for op in names:
            moved_raw,moved_rgb=transform_paired(raw,target,op)
            np.testing.assert_array_equal(moved_rgb[:,:,0],moved_raw)
            np.testing.assert_array_equal(moved_rgb[:,:,1],moved_raw+100)
            np.testing.assert_array_equal(unpack_bayer(pack_bayer(moved_raw)),moved_raw)
            tile=np.array(list('BGGR')).reshape(2,2)
            expected=''.join(apply_dihedral(tile,op).reshape(-1).tolist())
            self.assertEqual(transform_cfa_pattern('BGGR',op),expected)
        with self.assertRaisesRegex(ValueError,'unsupported paired augmentation'):
            apply_dihedral(raw,'resize')

    def test_transform_pair_rejects_misaligned_dimensions(self):
        with self.assertRaisesRegex(ValueError,'dimensions differ'):
            transform_paired(np.zeros((4,4)),np.zeros((3,4,3)),'identity')

    def test_seeded_augmentation_sequence_is_reproducible(self):
        first=np.random.default_rng(991);second=np.random.default_rng(991)
        self.assertEqual([sample_augmentation(first) for _ in range(20)],
                         [sample_augmentation(second) for _ in range(20)])


class RawAndTargetTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
    def tearDown(self): self.tmp.cleanup()

    def test_raw_loading_preserves_negative_and_overrange_float32_exactly(self):
        raw=np.array([[-0.25,0.0],[0.5,1.75]],dtype=np.float32)
        path=self.root/'valid.MAT';savemat(path,{'data':raw})
        actual=load_raw_mat(path,(2,2))
        self.assertEqual(actual.dtype,np.float32)
        np.testing.assert_array_equal(actual,raw)
        self.assertLess(float(actual.min()),0.0);self.assertGreater(float(actual.max()),1.0)

    def test_invalid_mat_key_dimensions_dtype_and_nonfinite_fail(self):
        cases=[({'other':np.zeros((2,2),np.float32)},'data'),
               ({'data':np.zeros((2,2,2),np.float32)},'2D'),
               ({'data':np.zeros((2,2),np.float64)},'float32'),
               ({'data':np.array([[0,np.nan],[1,2]],np.float32)},'non-finite')]
        for i,(payload,message) in enumerate(cases):
            path=self.root/('bad{}.MAT'.format(i));savemat(path,payload)
            with self.subTest(message=message),self.assertRaisesRegex(B1DataError,message):load_raw_mat(path)

    def test_target_rgb_channel_order_and_scaling(self):
        rgb=np.array([[[255,0,128],[1,2,3]],[[10,20,30],[40,50,60]]],dtype=np.uint8)
        path=self.root/'rgb.PNG';Image.fromarray(rgb).save(path)
        actual=load_srgb_png(path,(2,2))
        np.testing.assert_array_equal(actual,rgb)
        x,y=arrays_to_tensors(pack_bayer(np.arange(16,dtype=np.float32).reshape(4,4)),actual)
        self.assertEqual(tuple(y.shape),(3,2,2))
        np.testing.assert_allclose(y[:,0,0].numpy(),rgb[0,0].astype(np.float32)/255)
        self.assertTrue(y.is_contiguous());self.assertEqual(y.dtype,torch.float32)
        self.assertEqual(float(y.max()),1.0);self.assertEqual(float(y.min()),0.0)

    def test_target_mode_shape_and_pair_errors(self):
        path=self.root/'gray.png';Image.fromarray(np.zeros((2,2),np.uint8)).save(path)
        with self.assertRaisesRegex(B1DataError,'RGB mode'):load_srgb_png(path)
        rgb=np.zeros((3,2,3),np.uint8);raw=np.zeros((2,2),np.float32)
        with self.assertRaisesRegex(B1DataError,'dimensions differ'):validate_pair(raw,rgb,'scene:idx')
        with self.assertRaisesRegex(B1DataError,'unable to decode'):
            load_srgb_png(self.root/'missing.png')
        with self.assertRaisesRegex(B1DataError,'unable to decode'):
            load_raw_mat(self.root/'missing.MAT')

    def test_manifest_path_traversal_rejected(self):
        with self.assertRaisesRegex(B1DataError,'escapes'):
            resolve_project_path(self.root,'../../etc/passwd')
        with self.assertRaisesRegex(B1DataError,'project-relative'):
            resolve_project_path(self.root,'/etc/passwd')


class FrozenSplitTests(unittest.TestCase):
    def test_frozen_split_counts_stable_ids_and_disjointness(self):
        expected={'train':192,'validation':24,'internal_test':14,'official_test':928}
        refs={}
        for split,count in expected.items():
            ds=B1Dataset(split=split,verify_hashes=False,augment=False)
            self.assertEqual(len(ds),count)
            ids=[row['sample_id'] for row in ds._rows]
            self.assertEqual(len(ids),len(set(ids)))
            refs[split]={path for row in ds._rows for path in row['files'].values()}
            ds.close()
        for left in refs:
            for right in refs:
                if left<right:self.assertFalse(refs[left]&refs[right],(left,right))

    def test_tampered_frozen_manifest_fails_before_loading(self):
        source=Path(__file__).resolve().parents[1]/'experiments/P3/B1_SPLIT_MANIFEST.csv'
        with tempfile.TemporaryDirectory() as temp:
            bad=Path(temp)/'split.csv';bad.write_bytes(source.read_bytes()+b'\n')
            with self.assertRaisesRegex(B1DataError,'SHA-256 mismatch'):
                B1Dataset(split='train',split_manifest=bad,verify_hashes=False,augment=False)


if __name__=='__main__':unittest.main()
