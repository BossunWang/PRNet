import numpy as np
import os
from glob import glob
import scipy.io as sio
from skimage.io import imread, imsave
from skimage.transform import rescale, resize
from time import time
import argparse
import ast

from api import PRN

from utils.estimate_pose import estimate_pose
from utils.rotate_vertices import frontalize
from utils.render_app import get_visibility, get_uv_mask, get_depth_image
from utils.write import write_obj_with_colors, write_obj_with_texture
from render import render_scene, save_image 

def main(args):
    if args.isShow or args.isTexture:
        import cv2
        from utils.cv_plot import plot_kpt, plot_vertices, plot_pose_box

    # ---- init PRN
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu # GPU number, -1 for CPU
    prn = PRN(is_dlib = args.isDlib)

    # ------------- load data
    image_folder = args.inputDir
    save_folder = args.outputDir
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    # types = ('*.jpg', '*.png')
    # image_path_list= []
    # for files in types:
    #     image_path_list.extend(glob(os.path.join(image_folder, files)))
    # total_num = len(image_path_list)

    for dir, dirs, files in sorted(os.walk(image_folder)):
        for file in files:
            image_path = os.path.join(dir, file)
            dir = dir.replace("\\", "/")
            new_dir = dir.replace(image_folder, save_folder)
            if not os.path.isdir(new_dir):
                os.mkdir(new_dir)

            name = image_path.replace(image_folder, save_folder)
            print('data path:', name)

            # read image
            image = imread(image_path)
            [h, w, c] = image.shape
            if c>3:
                image = image[:,:,:3]

            # the core: regress position map
            if args.isDlib:
                max_size = max(image.shape[0], image.shape[1])
                if max_size> 1000:
                    image = rescale(image, 1000./max_size)
                    image = (image*255).astype(np.uint8)
                pos = prn.process(image) # use dlib to detect face
            else:
                # if image.shape[0] == image.shape[1]:
                #     image = resize(image, (256,256))
                #     pos = prn.net_forward(image/255.) # input image has been cropped to 256x256
                # else:
                box = np.array([0, image.shape[1]-1, 0, image.shape[0]-1]) # cropped with bounding box
                pos = prn.process(image, box)

            image = image/255.
            if pos is None:
                continue

            if args.is3d or args.isMat or args.isPose or args.isShow:
                # 3D vertices
                vertices = prn.get_vertices(pos)
                if args.isFront:
                    save_vertices = frontalize(vertices)
                else:
                    save_vertices = vertices.copy()
                save_vertices[:,1] = h - 1 - save_vertices[:,1]

            if args.isImage:
                imsave(name, image)

            if args.is3d:
                # corresponding colors
                colors = prn.get_colors(image, vertices)

                if args.isTexture:
                    if args.texture_size != 256:
                        pos_interpolated = resize(pos, (args.texture_size, args.texture_size), preserve_range = True)
                    else:
                        pos_interpolated = pos.copy()
                    texture = cv2.remap(image, pos_interpolated[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
                    if args.isMask:
                        vertices_vis = get_visibility(vertices, prn.triangles, h, w)
                        uv_mask = get_uv_mask(vertices_vis, prn.triangles, prn.uv_coords, h, w, prn.resolution_op)
                        uv_mask = resize(uv_mask, (args.texture_size, args.texture_size), preserve_range = True)
                        texture = texture*uv_mask[:,:,np.newaxis]
                    write_obj_with_texture(name.replace('.jpg', '.obj'), save_vertices, prn.triangles, texture, prn.uv_coords/prn.resolution_op)#save 3d face with texture(can open with meshlab)
                else:
                    write_obj_with_colors(name.replace('.jpg', '.obj'), save_vertices, prn.triangles, colors) #save 3d face(can open with meshlab)
                
                # filepath = name.replace('.jpg', '.obj')
                # filepath = filepath.replace("\\", "/")
                # print('filepath:', filepath)
                # new_dir = dir.replace(args.inputDir, args.renderDir)
                # # print(new_dir + '/' + file)
                # if not os.path.isdir(new_dir):
                #     os.mkdir(new_dir)

                # color_image1, _ = render_scene(filepath, 4.0, 0.0, 3.0)
                # color_image2, _ = render_scene(filepath, 4.0, np.pi / 9.0, 3.0)

                # if color_image1 is None or color_image2 is None:
                #     continue

                # new_path = filepath.replace(args.outputDir, args.renderDir)
                # # print('new_path:', new_path)
                # save_image(new_path, '_40_', color_image1)
                # save_image(new_path, '_60_', color_image2)

                # os.remove(name.replace('.jpg', '.obj'))

            if args.isDepth:
                depth_image = get_depth_image(vertices, prn.triangles, h, w, True)
                depth = get_depth_image(vertices, prn.triangles, h, w)
                imsave(os.path.join(name.replace('.jpg', '_depth.jpg')), depth_image)
                sio.savemat(name.replace('.jpg', '_depth.mat'), {'depth': depth})

            if args.isMat:
                sio.savemat(name.replace('.jpg', '_mesh.mat'),
                            {'vertices': vertices, 'colors': colors, 'triangles': prn.triangles})

            if args.isKpt or args.isShow:
                # get landmarks
                kpt = prn.get_landmarks(pos)
                np.savetxt(name.replace('.jpg', '_kpt.txt'), kpt)

            if args.isPose or args.isShow:
                # estimate pose
                camera_matrix, pose = estimate_pose(vertices)

                np.savetxt(name.replace('.jpg', '_pose.txt'), pose)
                np.savetxt(name.replace('.jpg', '_camera_matrix.txt'), camera_matrix)

                np.savetxt(name.replace('.jpg', '_pose.txt'), pose)

            if args.isShow:
                # ---------- Plot
                image_pose = plot_pose_box(image, camera_matrix, kpt)
                cv2.imshow('sparse alignment', plot_kpt(image, kpt))
                cv2.imshow('dense alignment', plot_vertices(image, vertices))
                cv2.imshow('pose', plot_pose_box(image, camera_matrix, kpt))
                cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network')

    parser.add_argument('-i', '--inputDir', default='TestImages/', type=str,
                        help='path to the input directory, where input images are stored.')
    parser.add_argument('-o', '--outputDir', default='TestImages/results', type=str,
                        help='path to the output directory, where results(obj,txt files) will be stored.')
    parser.add_argument('-r', '--renderDir', default='TestImages/results', type=str,
                        help='path to the render directory, where results(jpg files) will be stored.')
    parser.add_argument('--gpu', default='0', type=str,
                        help='set gpu id, -1 for CPU')
    parser.add_argument('--isDlib', default=True, type=ast.literal_eval,
                        help='whether to use dlib for detecting face, default is True, if False, the input image should be cropped in advance')
    parser.add_argument('--is3d', default=True, type=ast.literal_eval,
                        help='whether to output 3D face(.obj). default save colors.')
    parser.add_argument('--isMat', default=False, type=ast.literal_eval,
                        help='whether to save vertices,color,triangles as mat for matlab showing')
    parser.add_argument('--isKpt', default=False, type=ast.literal_eval,
                        help='whether to output key points(.txt)')
    parser.add_argument('--isPose', default=False, type=ast.literal_eval,
                        help='whether to output estimated pose(.txt)')
    parser.add_argument('--isShow', default=False, type=ast.literal_eval,
                        help='whether to show the results with opencv(need opencv)')
    parser.add_argument('--isImage', default=False, type=ast.literal_eval,
                        help='whether to save input image')
    # update in 2017/4/10
    parser.add_argument('--isFront', default=False, type=ast.literal_eval,
                        help='whether to frontalize vertices(mesh)')
    # update in 2017/4/25
    parser.add_argument('--isDepth', default=False, type=ast.literal_eval,
                        help='whether to output depth image')
    # update in 2017/4/27
    parser.add_argument('--isTexture', default=False, type=ast.literal_eval,
                        help='whether to save texture in obj file')
    parser.add_argument('--isMask', default=False, type=ast.literal_eval,
                        help='whether to set invisible pixels(due to self-occlusion) in texture as 0')
    # update in 2017/7/19
    parser.add_argument('--texture_size', default=256, type=int,
                        help='size of texture map, default is 256. need isTexture is True')
    main(parser.parse_args())
