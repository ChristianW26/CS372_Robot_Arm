import numpy as np
import quaternion as quat
import cv2
import aruco_markers as am
from scipy.spatial.transform import Rotation

# Constants
cam_idx = 0
dict_name = 'DICT_4X4_50'
marker_id = 0
marker_length = .025

cam_yaw = np.deg2rad(50)
cam_pitch = np.deg2rad(-30 - 90)
t_c_o_o = [.2697, .0784, .3299]

# Initialize camera and detector objects
camera = am.camera.cvCamera(cam_idx)
print('Using camera: ', camera.name)
detector = am.load_detector(dict_name)
pose_detector = am.SingleMarkerPoseEstimation(camera, marker_length)

# Capture image
img = camera.read()
cv2.imwrite('test_image.png', img)
corners, ids, _ = detector.detectMarkers(img)

if ids is None:
    raise TypeError('No marker detected')
elif len(ids) != 1: 
    raise ValueError(f'Detected {len(ids):d} markers\nExpecting just one marker')
elif ids[0] != marker_id: 
    raise ValueError(f'Detected marker id: {ids[0]:d}\nExpected marker id:{marker_id:d}')

# Draw detected marker
img = cv2.aruco.drawDetectedMarkers(img, (corners[0],), np.array([ids[0]]))
cv2.imwrite('test_image.png', img)

# Estimate marker pose
pose = pose_detector.estimate_marker_pose(corners[0])
q_b_c = np.quaternion(*pose.rotation.as_quat(scalar_first=True))
t_b_c_c = pose.translation

# Print camera frame info
# euler_b_c = Rotation.from_quat(quat.as_float_array(q_b_c), scalar_first=True).as_euler('ZXY', degrees=True)
# print(f'Local Translation (cm): x={t_b_c_c[0]*100:.1f}, y={t_b_c_c[1]*100:.1f}, z={t_b_c_c[2]*100:.1f}')
# print(f'Local Rotation (deg): yaw={euler_b_c[0]:.1f}, pitch={euler_b_c[1]:.1f}, roll={euler_b_c[2]:.1f}')

# Rotation from origin to camera frame 
q_c_o = np.quaternion(np.cos(cam_yaw/2), 0, 0, np.sin(np.sin(cam_yaw/2))) * np.quaternion(np.cos(cam_pitch/2), np.sin(cam_pitch/2), 0, 0)

# Convert box pose from camera frame to origin 
q_b_o = q_c_o * q_b_c
t_b_c_o = quat.as_vector_part(q_c_o * quat.from_vector_part(t_b_c_c) * q_c_o.conjugate())
t_b_o_o = t_c_o_o+t_b_c_o

# Print final pose
euler_b_o = Rotation.from_quat(quat.as_float_array(q_b_o), scalar_first=True).as_euler('ZXY', degrees=True)
print(f'Rotation (deg): yaw={euler_b_o[0]:.1f}, pitch={euler_b_o[1]:.1f}, roll={euler_b_o[2]:.1f}')
print(f'Translation (cm): x={t_b_o_o[0]*100:.1f}, y={t_b_o_o[1]*100:.1f}, z={t_b_o_o[2]*100:.1f}')