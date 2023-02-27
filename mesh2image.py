import open3d as o3d
import open3d.visualization.rendering as rendering
import numpy as np


def render_mesh(mesh, gt_axis=None):
    renderer = rendering.OffscreenRenderer(640, 640)

    # material
    white = rendering.MaterialRecord()
    white.base_color = [1.0, 1.0, 1.0, 1.0]
    white.shader = "defaultLit"

    red = rendering.MaterialRecord()
    red.base_color = [1.0, 0.0, 0.0, 1.0]

    green = rendering.MaterialRecord()
    green.base_color = [0.0, 1.0, 0.0, 1.0]

    blue = rendering.MaterialRecord()
    blue.base_color = [0.0, 0.0, 1.0, 1.0]

    l_red = rendering.MaterialRecord()
    l_red.base_color = [0.87, 0.0, 0.87, 1.0]

    l_green = rendering.MaterialRecord()
    l_green.base_color = [0.87, 0.87, 0.0, 1.0]

    l_blue = rendering.MaterialRecord()
    l_blue.base_color = [0.0, 0.87, 0.87, 1.0]


    global_rotation = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]


    # mesh
    mesh.rotate(global_rotation, center=(0, 0, 0))
    renderer.scene.add_geometry("mesh", mesh, white)

    # camera
    renderer.setup_camera(60.0, [0, 0, 0], [2]*3, [0, 0, 1])

    # arrows
    pred_arrow_params = dict(cylinder_radius=0.05, cone_radius=0.08, cylinder_height=1.4, cone_height=0.2)
    gt_arrow_params = dict(cylinder_radius=0.03, cone_radius=0.05, cylinder_height=1.4, cone_height=0.2)

    arrow_x = o3d.geometry.TriangleMesh.create_arrow(**pred_arrow_params)
    arrow_y = o3d.geometry.TriangleMesh.create_arrow(**pred_arrow_params)
    arrow_z = o3d.geometry.TriangleMesh.create_arrow(**pred_arrow_params)

    arrow_x.rotate([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], center=(0, 0, 0))
    arrow_y.rotate([[0, 1, 0], [0, 0, 1], [1, 0, 0]], center=(0, 0, 0))

    arrow_x.rotate(global_rotation, center=(0, 0, 0))
    arrow_y.rotate(global_rotation, center=(0, 0, 0))
    arrow_z.rotate(global_rotation, center=(0, 0, 0))

    renderer.scene.add_geometry("axis_x", arrow_x, blue)
    renderer.scene.add_geometry("axis_y", arrow_y, red)
    renderer.scene.add_geometry("axis_z", arrow_z, green)

    if gt_axis is not None:
        arrow_gt_x = o3d.geometry.TriangleMesh.create_arrow(**gt_arrow_params)
        arrow_gt_y = o3d.geometry.TriangleMesh.create_arrow(**gt_arrow_params)
        arrow_gt_z = o3d.geometry.TriangleMesh.create_arrow(**gt_arrow_params)

        arrow_gt_x.rotate([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], center=(0, 0, 0))
        arrow_gt_y.rotate([[0, 1, 0], [0, 0, 1], [1, 0, 0]], center=(0, 0, 0))

        arrow_gt_x.rotate(gt_axis, center=(0, 0, 0))
        arrow_gt_y.rotate(gt_axis, center=(0, 0, 0))
        arrow_gt_z.rotate(gt_axis, center=(0, 0, 0))

        arrow_gt_x.rotate(global_rotation, center=(0, 0, 0))
        arrow_gt_y.rotate(global_rotation, center=(0, 0, 0))
        arrow_gt_z.rotate(global_rotation, center=(0, 0, 0))

        renderer.scene.add_geometry("axis_gt_x", arrow_gt_x, l_blue)
        renderer.scene.add_geometry("axis_gt_y", arrow_gt_y, l_red)
        renderer.scene.add_geometry("axis_gt_z", arrow_gt_z, l_green)


    img = renderer.render_to_image()
    return img



if __name__ == "__main__":
    name = 'test'
    mesh = o3d.io.read_triangle_mesh(name + '.ply')
    img = render_mesh(mesh=mesh, gt_axis=np.array([[0.6, 0.8, 0], [0.8, -0.6, 0], [0, 0, -1]]))
    o3d.io.write_image('test.png', img)
