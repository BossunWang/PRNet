import trimesh
import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import pyrender
import numpy as np
import time
import cv2
import tqdm
import copy
from pyrender.constants import (OPEN_GL_MAJOR, OPEN_GL_MINOR, TEXT_PADDING, DEFAULT_SCENE_SCALE,
                        DEFAULT_Z_FAR, DEFAULT_Z_NEAR, RenderFlags, TextAlign)
from pyrender.camera import PerspectiveCamera, OrthographicCamera
from pyrender.trackball import Trackball

def compute_initial_camera_pose(scene):
    centroid = scene.centroid
    scale = scene.scale
    if scale == 0.0:
        scale = DEFAULT_SCENE_SCALE

    s2 = 1.0 / np.sqrt(2.0)
    cp = np.eye(4)
    cp[:3, :3] = np.array([
        [0.0, -s2, s2],
        [1.0, 0.0, 0.0],
        [0.0, s2, s2]
    ])
    hfov = np.pi / 6.0
    dist = scale / (2.0 * np.tan(hfov))
    cp[:3, 3] = dist * np.array([1.0, 0.0, 1.0]) + centroid

    return cp


def render_scene(filepath, scale_value, rotate_radius, light_intensity):
    fuze_trimesh = trimesh.load(filepath)
    mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)

    scene = pyrender.Scene(bg_color=(0, 0, 0))
    scene.add(mesh)

    pose = compute_initial_camera_pose(scene)
    zfar = max(scene.scale * 10.0, DEFAULT_Z_FAR)
    if scene.scale == 0:
        znear = DEFAULT_Z_NEAR
    else:
        znear = min(scene.scale / 10.0, DEFAULT_Z_NEAR)

    # cam = PerspectiveCamera(
    #     yfov=np.pi / 3.0, znear=znear, zfar=zfar
    # )

    xmag = ymag = scene.scale / scale_value
    if scene.scale == 0:
        xmag = ymag = 1.0
    cam = OrthographicCamera(
                xmag=xmag, ymag=ymag,
                znear=znear,
                zfar=zfar
    )

    scale = scene.scale
    if scale == 0.0:
        scale = DEFAULT_SCENE_SCALE
    centroid = scene.centroid
    viewport_size = (112, 112)
    trackball = Trackball(
        pose, viewport_size, scale, centroid
    )

    rand_sign = 1 if np.random.random() < 0.5 else -1
    random_pitch = np.random.randint(9.0, 36.0,size=1) * rand_sign

    trackball.rotate(rotate_radius, trackball._n_pose[:3, 0].flatten())
    trackball.rotate(np.pi / random_pitch, trackball._n_pose[:3, 1].flatten())
    trackball.scroll(scale_value)    
    pose = trackball._pose

    scene.add(cam, pose=pose)
    light = pyrender.light.DirectionalLight(color=np.ones(3), intensity=light_intensity)
    scene.add(light, pose=pose)

    try:
        r = pyrender.OffscreenRenderer(112, 112)
        color, depth = r.render(scene)
        r.delete()  
    except Exception as e:
        print(e)
        return None, None  

    return color, depth


def save_image(filepath, property, render_img):
    render_img = cv2.cvtColor(render_img, cv2.COLOR_BGR2RGB)
    render_img = cv2.rotate(render_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    render_img = cv2.flip(render_img, 1)

    new_path = filepath.replace('.obj', property + '.jpg')
    cv2.imwrite(new_path, render_img)


def render(data_dir, target_dir):
    for dir, dirs, files in tqdm.tqdm(os.walk(data_dir)):
        for file in files:
            filepath = os.path.join(dir, file)
            if filepath.endswith(".obj"):
                dir = dir.replace("\\", "/")
                filepath = dir + '/' + file
                print(filepath)
                new_dir = dir.replace(data_dir, target_dir)
                # print(new_dir + '/' + file)
                if not os.path.isdir(new_dir):
                    os.mkdir(new_dir)

                color_image1, _ = render_scene(filepath, 5.0, 0.0, 6.0)
                color_image2, _ = render_scene(filepath, 5.0, -np.pi / 9.0, 6.0)

                if color_image1 or color_image2 is None:
                    continue

                new_path = filepath.replace(data_dir, target_dir)
                save_image(new_path, '_60_', color_image1)
                save_image(new_path, '_40_', color_image2)


def main():
    render('../face_dataset/ms1m_align_112_objects', '../face_dataset/ms1m_align_112_render_images')


if __name__ == '__main__':
    main()

