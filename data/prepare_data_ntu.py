import os
import numpy as np
from glob import glob
from tqdm import tqdm

def process_ntu():
    skeleton_dir = 'data/nturgb+d_skeletons'
    files = glob(os.path.join(skeleton_dir, '*.skeleton'))
    if len(files) == 0:
        print(f"No skeleton files found in {skeleton_dir}.")
        return

    # Map from 25 NTU joints to 17 NW-UCLA-like joints
    # 0: Pelvis, 1: RHip, 2: RKnee, 3: RAnkle, 4: RFoot, 5: LHip, 6: LKnee, 7: LAnkle, 8: LFoot
    # 9: SpineShoulder, 10: Head, 11: LShoulder, 12: LElbow, 13: LWrist, 14: RShoulder, 15: RElbow, 16: RWrist
    ntu_to_nwucla = [0, 16, 17, 18, 19, 12, 13, 14, 15, 20, 3, 4, 5, 6, 8, 9, 10]

    output_3d = {}
    output_2d = {}
    
    cameras = {}

    train_subjects = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
    
    res_w = 1920
    res_h = 1080
    
    dummy_cam = {
        'orientation': np.array([1, 0, 0, 0], dtype=np.float32), # Identity quaternion [w, x, y, z] -> [1, 0, 0, 0]
        'translation': np.array([0, 0, 0], dtype=np.float32),
        'res_w': res_w,
        'res_h': res_h,
        'azimuth': 0
    }
    
    for f in tqdm(files):
        basename = os.path.basename(f)
        # S001C001P001R001A001.skeleton
        subject_id = int(basename[9:12])
        camera_id = int(basename[5:8])
        action_id = int(basename[17:20])
        
        prefix = 'Train' if subject_id in train_subjects else 'Validate'
        subject_name = f"{prefix}/S{subject_id}"
        action_name = f"A{action_id:03d}"
        
        if subject_name not in output_3d:
            output_3d[subject_name] = {}
            output_2d[subject_name] = {}
            cameras[subject_name] = [dummy_cam] # For NTU, all cameras are basically the same in our camera space since we treat them as independent sequences or just keep them in their own local camera space.
            # Actually, VideoPose3D expects cameras as an array if there are multiple cameras per action.
            # But the 'camera_id' in NTU is C001, C002, C003. We can just store them in a list.
            cameras[subject_name] = [
                dict(dummy_cam, id='C001'),
                dict(dummy_cam, id='C002'),
                dict(dummy_cam, id='C003')
            ]
            
        if action_name not in output_3d[subject_name]:
            output_3d[subject_name][action_name] = {'positions_3d': [None, None, None], 'cameras': cameras[subject_name]}
            output_2d[subject_name][action_name] = [None, None, None]
            
        with open(f, 'r') as fp:
            num_frames = int(fp.readline().strip())
            
            seq_3d = []
            seq_2d = []
            
            for _ in range(num_frames):
                num_bodies = int(fp.readline().strip())
                frame_3d = []
                frame_2d = []
                
                # If there are multiple bodies, we just take the first tracked body to simplify.
                found_body = False
                for b in range(num_bodies):
                    body_info = fp.readline().strip().split()
                    num_joints = int(fp.readline().strip())
                    
                    joints_3d = []
                    joints_2d = []
                    for j in range(num_joints):
                        joint_info = fp.readline().strip().split()
                        if not found_body:
                            joints_3d.append([float(joint_info[0]), float(joint_info[1]), float(joint_info[2])])
                            joints_2d.append([float(joint_info[3]), float(joint_info[4])])
                    
                    if not found_body:
                        # Map to 17 joints
                        frame_3d = [joints_3d[i] for i in ntu_to_nwucla]
                        frame_2d = [joints_2d[i] for i in ntu_to_nwucla]
                        found_body = True
                
                if not found_body:
                    # If no body found (e.g. tracking lost), we copy the previous frame or use zeros
                    if len(seq_3d) > 0:
                        frame_3d = seq_3d[-1]
                        frame_2d = seq_2d[-1]
                    else:
                        frame_3d = [[0,0,0]] * 17
                        frame_2d = [[0,0]] * 17
                        
                seq_3d.append(frame_3d)
                seq_2d.append(frame_2d)
                
            cam_idx = camera_id - 1
            output_3d[subject_name][action_name]['positions_3d'][cam_idx] = np.array(seq_3d, dtype=np.float32)
            output_2d[subject_name][action_name][cam_idx] = np.array(seq_2d, dtype=np.float32)
            
    # Cleanup empty camera slots where a specific action was not recorded from all 3 cameras
    for subj in output_3d.keys():
        for act in output_3d[subj].keys():
            valid_cams = [i for i, x in enumerate(output_3d[subj][act]['positions_3d']) if x is not None]
            
            output_3d[subj][act]['positions_3d'] = [output_3d[subj][act]['positions_3d'][i] for i in valid_cams]
            output_3d[subj][act]['cameras'] = [output_3d[subj][act]['cameras'][i] for i in valid_cams]
            
            output_2d[subj][act] = [output_2d[subj][act][i] for i in valid_cams]
            
    print('Saving 3D data...')
    np.savez('data/data_3d_ntu.npz', positions_3d=output_3d, cameras=cameras)
    
    print('Saving 2D data...')
    metadata = {
        'num_joints': 17,
        'keypoints_symmetry': [
            [5, 6, 7, 8, 11, 12, 13],
            [1, 2, 3, 4, 14, 15, 16]
        ]
    }
    np.savez('data/data_2d_ntu_gt.npz', positions_2d=output_2d, metadata=metadata)

if __name__ == '__main__':
    process_ntu()
