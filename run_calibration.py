# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import cv2
import matplotlib.pyplot as plt


from estimater import *
from datareader import *
import argparse
import yaml
import csv
import paramiko

with open('./configs/config.yml', 'r') as file:
  config = yaml.safe_load(file)

WHITE_MASK=str(config['object_detection']['white_mask'])=='True'

parser = argparse.ArgumentParser() # set arguments 
# code_dir = os.path.dirname(os.path.realpath(__file__))
# code_dir = '/home/jacknaimer/SchoelligLabProjects/FoundationPose'
code_dir = config['workspace']['workspace_global_path']
parser.add_argument('--mesh_file', type=str, default=code_dir+config['object_detection']['mesh_file'])
parser.add_argument('--test_scene_dir', type=str, default=code_dir+config['camera']['rgbd_frames_files'])
parser.add_argument('--est_refine_iter', type=int, default=5)
parser.add_argument('--track_refine_iter', type=int, default=2)
parser.add_argument('--debug', type=int, default=1)
parser.add_argument('--debug_dir', type=str, default=f'{code_dir}./debug')
args = parser.parse_args()

set_logging_format()
set_seed(0)

mesh = trimesh.load(args.mesh_file) # gets the mesh for the object (from args)

# sets debug dirs and args
debug = args.debug
debug_dir = args.debug_dir
os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam') 

# orients axes to the mesh of the objects based on the best fitting minimum volume convex hull around the mesh
to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

# Does something
scorer = ScorePredictor() # init scorer for score prediction
refiner = PoseRefinePredictor() # init this for refined pose predictions/scoring or something
glctx = dr.RasterizeCudaContext() # stuff for cuda optimization
est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
logging.info("estimator initialization done")

shorter_side = config['object_detection']['shorter_side']
if shorter_side == 'None': shorter_side = None
else: shorter_side = int(shorter_side)
reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=shorter_side, zfar=np.inf) # was shorter_side = None 
color_copy = None
color = reader.get_color(0) # get the ith color frame
color_copy = color.copy()

def get_block_locks(center):
  global color_copy
  for i in range(len(reader.color_files)):
    logging.info(f'i:{i}')
    color = reader.get_color(i) # get the ith color frame
    depth = reader.get_depth(i) # get the ith depth frame
    if i==0: # if the first frame then do detection pose estimation instead of tracking pose estimation
      if WHITE_MASK:
        # mask = np.ones((720, 1280), dtype=bool)
        height, width = 720, 1280
        if str(config['camera']['crop_size'])!=None:
          crop_size = int(config['camera']['crop_size'])
          height, width = crop_size, crop_size
        mask = np.zeros((height, width), dtype=bool)

        center_x, center_y = center[0], center[1]

        radius = 40
        for y in range(height):
          for x in range(width):
            if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2:
              mask[y, x] = True



      else:
        mask = reader.get_mask(0).astype(bool) # mask = get_mask, maybe of area around object
      print('RESTISTERED12345\n\n')
      pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter) # Get the 6d rotation/translation of the object from the color and depth frames and the mask

      point3D = np.append(pose[:-1, 3], 1)  # Ensure point3D is in homogeneous coordinates (3D -> 4D)
      pixel_homogeneous = reader.K @ point3D[:3]  # Only use the first three components (X, Y, Z)
      pixel = pixel_homogeneous[:2] / pixel_homogeneous[2]  # Convert from homogeneous to image coordinates

      cv2.circle(color_copy, (int(pixel[0]), int(pixel[1])), radius=10, color=(0, 255, 0), thickness=-1)  # Draws a green circle at the center
      return(list(pose[:-1, 3]))

      if debug>=3:
        m = mesh.copy()
        m.apply_transform(pose)
        m.export(f'{debug_dir}/model_tf.obj')
        xyz_map = depth2xyzmap(depth, reader.K)
        valid = depth>=0.1
        pcd = toOpen3dCloud(xyz_map[valid], color[valid])
        o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
    else: # if not the first frame
      pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter) # do tracking pose estimation instead of simple pose estimation

    os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
    np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4,4))

    if debug>=1:
      center_pose = pose@np.linalg.inv(to_origin)
      vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
      cv2.imshow('1', vis[...,::-1])
      cv2.waitKey(1)


    if debug>=2:
      os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
      imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)

        
coordinates = [
  (112, 130, "Red"),
  (255, 128, "Orange"),
  (405, 128, "Yellow"),
  (88, 237, "LightBlue"),
  (252, 240, "LightYellow"),
  (422, 236, "LightGreen"),
  (51, 389, "Pink"),
  (247, 387, "Blue"),
  (449, 392, "White")
]

matrix = []
for coordinate in coordinates:
  matrix.append(get_block_locks(coordinate))

print()
print(matrix)


# Save the matrix to a CSV file
csv_file = 'matrix.csv'
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(matrix)

# SSH/SCP parameters
hostname = "10.157.163.52"
username = "lsy_franka"
password = "franka"
remote_path = "/home/lsy_franka/Documents/matrix.csv"  # Change to the desired remote path

# Establish SSH connection
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(hostname, username=username, password=password)

# SCP the file
sftp = ssh.open_sftp()
sftp.put(csv_file, remote_path)
sftp.close()
ssh.close()

print("File transferred successfully")



cv2.imshow('colordisplay', color_copy)
cv2.waitKey(0)  # Waits indefinitely until a key is pressed
cv2.destroyAllWindows()  # Closes the window once a key is pressed