import numpy as np
import open3d as o3d


def create_open3d_point_cloud(point_cloud):
    indices = np.all(np.isfinite(point_cloud.xyzs()), axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud.xyzs()[:, indices].T)

    if point_cloud.has_rgbs():
        pcd.colors = o3d.utility.Vector3dVector(point_cloud.rgbs()[:, indices].T
                                                / 255.)

    # TODO(russt): add normals (and test) if needed


#    if point_cloud.has_normals():
#        pcd.normals = o3d.uility.Vector3dVector(
#            point_cloud.normals()[:, indices].T)

    return pcd


def create_open3d_rgbd_image(color_image, depth_image):
    color_image = o3d.geometry.Image(np.copy(
        color_image.data[:, :, :3]))  # No alpha
    depth_image = o3d.geometry.Image(np.squeeze(np.copy(depth_image.data)))
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=color_image,
        depth=depth_image,
        depth_scale=1.0,
        depth_trunc=3.0,
        convert_rgb_to_intensity=False)
    return rgbd_image
