import paramiko
from io import StringIO, BytesIO
import json
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
parser.add_argument('--test_scene_dir', type=str, default=code_dir+config['camera']['inference_frames_files'])
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

import time

def wait_for_unlock():
    file_path = "/home/jacknaimer/SchoelligLabProjects/FoundationPoseTest/data/live_run/lock.txt"
    while True:
        try:
            with open(file_path, 'r') as file:
                content = file.read().strip()  # Read the file and strip any whitespace
                if content == '0':
                    return  # Exit the function if the content is '0'
        except FileNotFoundError:
            break
        except Exception:
            break
        time.sleep(0.000001)  # Sleep for 1 millisecond before checking again


color_copy = None
print('starting spin cycle')
wait_for_unlock()
print('done spin cycle')
color = reader.get_color(0) # get the ith color frame
color_copy = color.copy()



def get_block_locks(bounding_box):
  global color_copy
  for i in range(1):#range(len(reader.color_files)):
    logging.info(f'i:{i}')
    wait_for_unlock()
    color = reader.get_color(i) # get the ith color frame
    depth = reader.get_depth(i) # get the ith depth frame
    if i==0: # if the first frame then do detection pose estimation instead of tracking pose estimation
      # mask = np.ones((720, 1280), dtype=bool)

      crop_size = int(config['camera']['crop_size'])
      height, width = crop_size, crop_size

      mask = np.zeros((height, width), dtype=bool)

      for y in range(height):
        for x in range(width):
          # if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2:
          if bounding_box[0]<=x<=bounding_box[2] and bounding_box[1]<=y<=bounding_box[3]:
            mask[y, x] = True



      # else:
      #   mask = reader.get_mask(0).astype(bool) # mask = get_mask, maybe of area around object
      print('RESTISTERED12345\n\n')
      pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter) # Get the 6d rotation/translation of the object from the color and depth frames and the mask
      print('pose')
      print(pose)
      print('done')
      print('now2d')
      point3D = np.append(pose[:-1, 3], 1)  # The 1x4 (standard python list) of the 3d coordinates, not pose
      # print(point3D)
      # print(reader.K)
      # assert False
      pixel_homogeneous = reader.K @ point3D[:3]  # Only use the first three components (X, Y, Z)
      pixel = pixel_homogeneous[:2] / pixel_homogeneous[2]  # Convert from homogeneous to image coordinates

      cv2.circle(color_copy, (int(pixel[0]), int(pixel[1])), radius=10, color=(0, 255, 0), thickness=-1)  # Draws a green circle at the center
      # return(list(pose[:-1, 3]))


      ### START AXIS VIS
      # Axis length in meters
      axis_length = 0.1
      center_3D = np.append(pose[:-1, 3], 1)  # The 1x4 (standard python list) of the 3d coordinates
      pixel_homogeneous = reader.K @ center_3D[:3]
      center_pixel = pixel_homogeneous[:2] / pixel_homogeneous[2]
      # Create axis vectors
      x_axis = center_3D[:3] + pose[:3, 0] * axis_length
      y_axis = center_3D[:3] + pose[:3, 1] * axis_length
      z_axis = center_3D[:3] + pose[:3, 2] * axis_length

      # Transform axes points
      x_pixel_homogeneous = reader.K @ x_axis
      y_pixel_homogeneous = reader.K @ y_axis
      z_pixel_homogeneous = reader.K @ z_axis

      # Convert from homogeneous to image coordinates
      x_pixel = x_pixel_homogeneous[:2] / x_pixel_homogeneous[2]
      y_pixel = y_pixel_homogeneous[:2] / y_pixel_homogeneous[2]
      z_pixel = z_pixel_homogeneous[:2] / z_pixel_homogeneous[2]

      # Draw axes
      cv2.line(color_copy, (int(center_pixel[0]), int(center_pixel[1])), (int(x_pixel[0]), int(x_pixel[1])), (255, 0, 0), thickness=2)  # X-axis in red
      cv2.line(color_copy, (int(center_pixel[0]), int(center_pixel[1])), (int(y_pixel[0]), int(y_pixel[1])), (0, 255, 0), thickness=2)  # Y-axis in green
      cv2.line(color_copy, (int(center_pixel[0]), int(center_pixel[1])), (int(z_pixel[0]), int(z_pixel[1])), (0, 0, 255), thickness=2)  # Z-axis in blue
      ### END AXIS VIS
      
      return pose

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

def load_bounding_boxes():

  # Set up the SSH client and connect
  client = paramiko.SSHClient()
  client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
  client.connect('10.157.163.52', username='lsy_franka', password='franka')  # Reminder: this is insecure!

  # Use SFTP to retrieve the file
  sftp = client.open_sftp()
  remote_path = '/home/lsy_franka/jackdata/colors.json'
  buff = BytesIO()  # Create an in-memory file-like object
  sftp.getfo(remote_path, buff)  # Download file into the buffer
  sftp.close()
  client.close()

  # Read JSON data from buffer
  buff.seek(0)  # Reset buffer position
  data_received = buff.getvalue()  # Retrieve string
  json_data = json.loads(data_received)  # Parse JSON data

  return json_data

json_data = load_bounding_boxes()

matrix = []
for color_key in json_data.keys():
  for bounding_box in json_data[color_key]:
    pose_estimate = get_block_locks(bounding_box)
    # if pose_estimate!="ERROR":
      # matrix.append(get_block_locks(coordinate))
    matrix.append([color_key, pose_estimate])

print()
print(matrix)


# Save the matrix to a CSV file
csv_file = 'inference_matrix.csv'
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(matrix)

# SSH/SCP parameters
hostname = "10.157.163.52"
username = "lsy_franka"
password = "franka"
remote_path = "/home/lsy_franka/Documents/inference_matrix.csv"  # Change to the desired remote path

# Establish SSH connection
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(hostname, username=username, password=password)

# SCP the file
sftp = ssh.open_sftp()
sftp.put(csv_file, remote_path)
sftp.close()
ssh.close()

print("File transferred successfully: Press '1' to close picture")


color_copy = cv2.cvtColor(color_copy, cv2.COLOR_BGR2RGB)
cv2.imshow('colordisplay', color_copy)
cv2.waitKey(0)  # Waits indefinitely until a key is pressed
cv2.destroyAllWindows()  # Closes the window once a key is pressed