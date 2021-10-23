''' 
Generate 2d maps representing different attributes(colors, depth, pncc, etc)
: render attributes to image space.
'''
import os, sys
import numpy as np
import scipy.io as sio
from skimage import io
from time import time
import matplotlib.pyplot as plt

sys.path.append('..')
import face3d
from face3d import mesh

# ------------------------------ load mesh data
C = sio.loadmat('Data/example1.mat')
vertices = C['vertices']; colors = C['colors']; triangles = C['triangles']
colors = colors/np.max(colors)

# ------------------------------ modify vertices(transformation. change position of obj)
# scale. target size=200 for example
s = 180/(np.max(vertices[:,1]) - np.min(vertices[:,1]))
# rotate 30 degree for example
R = mesh.transform.angle2matrix([0, 30, 0]) 
# no translation. center of obj:[0,0]
t = [0, 0, 0]
transformed_vertices = mesh.transform.similarity_transform(vertices, s, R, t)
camera_vertices = mesh.transform.lookat_camera(transformed_vertices, eye = [0, 0, 250], at = np.array([0, 0, 0]), up = np.array([0,1,0]))
ortho_vertices = mesh.transform.orthographic_project(camera_vertices)
persp_vertices = mesh.transform.perspective_project(camera_vertices, 30, )

# ------------------------------ render settings(to 2d image)
# set h, w of rendering
h = w = 256
# change to image coords for rendering
image_vertices = mesh.transform.to_image(ortho_vertices, h, w)
image_vertices_ = mesh.transform.to_image(persp_vertices, h, w, True)

## --- start
save_folder = 'results/image_map'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

## 0. color map
attribute = colors
color_image = mesh.render.render_colors(image_vertices, triangles, attribute, h, w, c=3)
color_image_ = mesh.render.render_colors(image_vertices_, triangles, attribute, h, w, c=3)
io.imsave('{}/color.jpg'.format(save_folder), np.concatenate([np.squeeze(color_image), np.squeeze(color_image_)], axis=1))

## 1. depth map
z = image_vertices[:,2:]
z = z - np.min(z)
z = z/np.max(z)
attribute = z
depth_image = mesh.render.render_colors(image_vertices, triangles, attribute, h, w, c=1)
# io.imsave('{}/depth.jpg'.format(save_folder), np.squeeze(depth_image))
z = image_vertices_[:,2:]
z = z - np.min(z)
z = z/np.max(z)
attribute = z
depth_image_ = mesh.render.render_colors(image_vertices_, triangles, attribute, h, w, c=1)

io.imsave('{}/depth.jpg'.format(save_folder), np.concatenate([np.squeeze(depth_image), np.squeeze(depth_image_)], axis=1))

## 2. pncc in 'Face Alignment Across Large Poses: A 3D Solution'. for dense correspondences 
pncc = face3d.morphable_model.load.load_pncc_code('Data/BFM/Out/pncc_code.mat')
attribute = pncc
pncc_image = mesh.render.render_colors(image_vertices, triangles, attribute, h, w, c=3)
pncc_image_ = mesh.render.render_colors(image_vertices_, triangles, attribute, h, w, c=3)
# io.imsave('{}/pncc.jpg'.format(save_folder), np.squeeze(pncc_image))
io.imsave('{}/pncc.jpg'.format(save_folder), np.concatenate([np.squeeze(pncc_image), np.squeeze(pncc_image_)], axis=1))

## 3. uv coordinates in 'DenseReg: Fully convolutional dense shape regression in-the-wild'. for dense correspondences
uv_coords = face3d.morphable_model.load.load_uv_coords('Data/BFM/Out/BFM_UV.mat') #
attribute = uv_coords # note that: original paper used quantized coords, here not
uv_coords_image = mesh.render.render_colors(image_vertices, triangles, attribute, h, w, c=2) # two channels: u, v
# add one channel for show
uv_coords_image = np.concatenate((np.zeros((h, w, 1)), uv_coords_image), 2)
uv_coords_image_ = mesh.render.render_colors(image_vertices_, triangles, attribute, h, w, c=2) # two channels: u, v
# add one channel for show
uv_coords_image_ = np.concatenate((np.zeros((h, w, 1)), uv_coords_image_), 2)
# io.imsave('{}/uv_coords.jpg'.format(save_folder), np.squeeze(uv_coords_image))
io.imsave('{}/uv_coords.jpg'.format(save_folder), np.concatenate([np.squeeze(uv_coords_image), np.squeeze(uv_coords_image_)], axis=1))

