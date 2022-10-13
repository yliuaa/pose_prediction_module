#!/usr/bin/env python

import rospy
import torch
import numpy as np
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from predictor_pkg.training.experiment import model_factory
import predictor_pkg.utils.utils as utils   

'''
<Pose prediction script>
Predictor folder is installed as ROS module named predictor_pkg

(assumed rate 1 Hz)
Listen to topic 'human_traj' with type JointTrajectory(
    header: empty
    joint_names: order of indices as in H36M,
    points: frames of poses of size (frame_num=25, joint_num=21, exp_coordinate_dim=3)
)

Accumulate a 50-frame pose sequence, reformat, then predict a trajectory of 25 frame

Publish to topic 'prediction_traj' in type JointTrajectory(
    header: empty
    joint_names: order of indices as in H36M,
    points: frames of poses of size (frame_num, joint_num=21, exp_coordinate_dim=3)
)
'''
_MAJOR_JOINTS = [
    '0', '1', '2', '3', '4', '6', '7', '8', '9', '11', '12', '13', '14', '16', '17', '18', '19', '24', '25', '26', '27'
]
_JOINT_NUM = 21

class PoseListener:
    def __init__(self):
        # prepare predictor & pose queue
        self.model = model_factory()
        self.pose_sequence = torch.tensor([])
        
        # prepare node
        rospy.init_node('traj_listener', anonymous=True)
        self.predict_publisher = rospy.Publisher('prediction_traj', JointTrajectory, queue_size=10)
        rospy.Subscriber('human_traj', JointTrajectory, self.callback)
        rospy.spin()

    def callback(self, data):
        sequence = data.points
        points = np.asarray([pose.positions for pose in sequence])

        # raw input reformatting: (frame_num, _JOINT_NUM*3) -> (frame_num, _JOINT_NUM, 3)
        points = points.reshape(points.shape[0], _JOINT_NUM, 3)
        # use helper function provided in POTR for conversion expmap(dim=3) -> rotmat(dim=9)
        rotmat_points = utils.expmap_to_rotmat(points)
        rotmat_tensor = torch.Tensor(np.asarray(rotmat_points))
        print(rotmat_tensor.shape)

        # concatenate to saved pose_sequence
        self.pose_sequence = torch.concat((self.pose_sequence, rotmat_tensor), 0)
        if len(self.pose_sequence) >= 50:
            print("Input queue filled -- Generating prediction")
            traj = self.predict(self.pose_sequence)
            self.predict_publisher.publish(traj)
            self.pose_sequence = torch.Tensor([])


    '''
        Model inputs:
        1. encoder_input of size (1, frame_num-1, _JOINT_NUM*dim)
            Note: last observation is discarded
        2. decoder_input(query) of size (1, prediction_frame, _JOINT_NUM*dim)
            Note: this is created by duplicating the last observation
    '''
    @torch.no_grad()
    def predict(self, input_sequence):
        encoder_input = input_sequence.view(1, -1, _JOINT_NUM*9)[:, :-1, :]
        decoder_input = encoder_input[0, -1, :].repeat(1, 25, 1)
        decoder_pred = self.model._model(
            encoder_input, decoder_input)[0][-1]

        pred = decoder_pred.view(decoder_pred.shape[1], _JOINT_NUM, 9).numpy()
        print(pred.shape)
        trajectory_points = utils.rotmat_to_expmap(pred)

        # form JointTrajectory 
        traj_plist = []
        for point in trajectory_points:
            traj_point = JointTrajectoryPoint()
            traj_point.positions = point
            traj_plist.append(traj_point)
        # Prediction in rotation matrix format 
        pred_trajectory =  JointTrajectory()
        pred_trajectory.points = traj_plist
        pred_trajectory.joint_names = _MAJOR_JOINTS
        return pred_trajectory


if __name__ == '__main__':
    listener = PoseListener()
