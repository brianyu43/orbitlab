import dataclasses
import unittest
from common import np
from object_world import ObjectState,render,rotate_scene,translate_scene,edit,sample_scene,overlapping,state_slots


class ObjectWorldChecks(unittest.TestCase):
    def setUp(self):
        self.scene=(ObjectState(0,0,1,18,21,6,0),ObjectState(1,2,4,44,41,8,3))

    def test_exact_image_and_mask_rotation_including_occlusion_and_boundaries(self):
        scenes=[self.scene,tuple(dataclasses.replace(v,x=31+v.id*3,y=30) for v in self.scene),
                (ObjectState(0,1,2,3,6,8,2),)]
        for scene in scenes:
            base=render(scene)
            for k in range(4):
                rotated=render(rotate_scene(scene,k))
                for key in ['image','segmentation']:np.testing.assert_array_equal(rotated[key],np.rot90(base[key],k))
                for key in ['amodal','visible']:np.testing.assert_array_equal(rotated[key],np.rot90(base[key],k,axes=(1,2)))

    def test_local_edits_preserve_other_object_and_color_rotation_commute(self):
        source=render(self.scene)
        for op,value in [('color',3),('rotate',1),('translate',(3,0))]:
            edited=edit(self.scene,0,op,value);after=render(edited)
            self.assertEqual(edited[1],self.scene[1])
            np.testing.assert_array_equal(after['amodal'][1],source['amodal'][1])
            np.testing.assert_array_equal(after['visible'][1],source['visible'][1])
        a=edit(edit(self.scene,0,'color',3),0,'rotate',1)
        b=edit(edit(self.scene,0,'rotate',1),0,'color',3)
        self.assertEqual(a,b)
        np.testing.assert_array_equal(render(a)['image'],render(b)['image'])

    def test_global_rotation_and_world_translation_do_not_commute(self):
        a=rotate_scene(translate_scene(self.scene,3,0),1)
        b=translate_scene(rotate_scene(self.scene,1),3,0)
        self.assertNotEqual(a,b)
        self.assertFalse(np.array_equal(render(a)['image'],render(b)['image']))
        # The corresponding rotated translation restores covariance.
        self.assertEqual(a,translate_scene(rotate_scene(self.scene,1),0,-3))

    def test_scene_generation_factor_split_and_capacity(self):
        for split,n in [('train',2),('ood',2),('count3',3),('count4',4)]:
            scene=sample_scene(17,942301,split,n)
            self.assertEqual(scene,sample_scene(17,942301,split,n))
            self.assertFalse(overlapping(scene));self.assertEqual(len(scene),n)
            self.assertEqual(state_slots(scene).shape,(4,16))
            self.assertEqual(int(state_slots(scene)[:,-1].sum()),n)
            if split=='ood':self.assertEqual(scene[0].shape,scene[0].color)
            else:self.assertTrue(all(v.shape!=v.color for v in scene))


if __name__=='__main__':unittest.main()
