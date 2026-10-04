"""Analytical support and volume tests for the independent refinement patch."""
import unittest
import numpy as np
from compton_cartesian_patch import CartesianPatch, patch_specs, refined_patch_specs
from generate_compton_a_guard import axes
from compton_boundary_quadrature import cell_quadrature, rotate_to_detector


class PatchTests(unittest.TestCase):
    def test_affine_interpolation_on_nonuniform_axes_and_faces(self):
        xyz = (np.array([-2., 0., 3.]), np.array([246., 249., 258.]), np.array([-60., -58.5, -54.]))
        xx, yy, zz = np.meshgrid(*xyz, indexing='ij')
        field = (1000+2*xx+3*yy-4*zz).transpose(2, 1, 0)
        points = np.array([[-2, 246, -60], [3, 258, -54], [1.2, 253, -59]])
        interp = CartesianPatch(xyz)
        np.testing.assert_allclose(interp.evaluate(field, interp.cache(points)),
                                   1000+2*points[:, 0]+3*points[:, 1]-4*points[:, 2])
        with self.assertRaises(ValueError): interp.cache(np.array([[3.001, 250, -57]]))

    def test_full_near_side_cell_integral_is_exact_for_affine_A(self):
        cell = (249., 255., -np.pi/140, np.pi/140)
        nodes, w = cell_quadrature(cell, 1.5, 24, 8, 8, ellipse=False)
        nodes = rotate_to_detector(nodes, 15)
        xyz = axes(patch_specs()[1]); interp = CartesianPatch(xyz)
        xx, yy, zz = np.meshgrid(*xyz, indexing='ij')
        field = (1000+xx+2*yy+3*zz).transpose(2, 1, 0)
        value = np.dot(interp.evaluate(field, interp.cache(nodes)), w)
        np.testing.assert_allclose(value, np.dot(1000+nodes[:, 0]+2*nodes[:, 1]+3*nodes[:, 2], w), rtol=1e-13)
        np.testing.assert_allclose(w.sum(), (255**2-249**2)*np.pi/140*3, rtol=1e-13)

    def test_held_out_points_detect_curvature_hidden_by_total_sum(self):
        xyz = axes(patch_specs()[1]); interp = CartesianPatch(tuple(a[::2] for a in xyz))
        xx, yy, zz = np.meshgrid(*(a[::2] for a in xyz), indexing='ij')
        field = (1+(yy-252)**2).transpose(2, 1, 0)
        points = np.array([[0., 253.5, 0.]])
        self.assertGreater(float(interp.evaluate(field, interp.cache(points))[0]), 1+1.5**2)

    def test_refined_boxes_cover_whole_cells_at_all_three_axial_positions(self):
        cell = (249., 255., -np.pi/140, np.pi/140)
        for z, spec in zip((-58.5, 1.5, 58.5), refined_patch_specs()):
            nodes, _ = cell_quadrature(cell, z, 32, 12, 12, ellipse=False)
            CartesianPatch(axes(spec)).cache(rotate_to_detector(nodes, 15))


if __name__ == '__main__': unittest.main()
