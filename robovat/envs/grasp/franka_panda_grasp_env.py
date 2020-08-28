"""This is an example environment for grasping given a specific pose. This environment assumes one object in the environment only.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import random
import os.path

import gym
import numpy as np

from robovat.envs import arm_env
from robovat.envs.grasp.grasp_2d import Grasp2D
from robovat.math import Pose
from robovat.math import get_transform
from robovat.observations import camera_obs
from robovat.reward_fns.grasp_reward import GraspReward
from robovat.utils.logging import logger
from robovat.utils.yaml_config import YamlConfig


GRASPABLE_NAME = 'graspable'


class FrankaPandaGraspEnv(arm_env.ArmEnv):
    """An example franka panda grasping environment."""

    def __init__(self,
                 simulator=None,
                 config=None,
                 debug=True):
        """Initialize.

        Args:
            simulator: Instance of the simulator.
            config: Environment configuration.
            debug: True if it is debugging mode, False otherwise.
        """
        self._simulator = simulator
        self._config = config or self.default_config
        self._debug = debug

        # Camera.
        self.camera = self._create_camera(
            height=self.config.KINECT2.DEPTH.HEIGHT,
            width=self.config.KINECT2.DEPTH.WIDTH,
            intrinsics=self.config.KINECT2.DEPTH.INTRINSICS,
            translation=self.config.KINECT2.DEPTH.TRANSLATION,
            rotation=self.config.KINECT2.DEPTH.ROTATION)

        # Graspable object.
        if self.is_simulation:
            self.graspable = None
            self.graspable_path = None
            self.graspable_pose = None
            self.all_graspable_paths = []
            self.graspable_index = 0

            for pattern in self.config.SIM.GRASPABLE.PATHS:
                if not os.path.isabs(pattern):
                    pattern = os.path.join(self.simulator.assets_dir, pattern)

                if pattern[-4:] == '.txt':
                    with open(pattern, 'r') as f:
                        paths = [line.rstrip('\n') for line in f]
                else:
                    paths = glob.glob(pattern)

                print(pattern)

                self.all_graspable_paths += paths

            self.all_graspable_paths.sort()
            num_graspable_paths = len(self.all_graspable_paths)
            assert num_graspable_paths > 0, (
                'Found no graspable objects at %s'
                % (self.config.SIM.GRASPABLE.PATHS))
            logger.debug('Found %d graspable objects.', num_graspable_paths)

        super(FrankaPandaGraspEnv, self).__init__(
            simulator=self.simulator,
            config=self.config,
            debug=self.debug)

    @property
    def default_config(self):
        """Load the default configuration file."""
        config_path = os.path.join('configs', 'envs', 'grasp_4dof_env.yaml')
        assert os.path.exists(config_path), (
                'Default configuration file %s does not exist' % (config_path))
        return YamlConfig(config_path).as_easydict()

    def _create_observations(self):
        """Create observations.

        Returns:
            List of observations.
        """
        return [
            camera_obs.CameraObs(
                name=self.config.OBSERVATION.TYPE,
                camera=self.camera,
                modality=self.config.OBSERVATION.TYPE,
                max_visible_distance_m=None),
            camera_obs.CameraIntrinsicsObs(
                name='intrinsics',
                camera=self.camera),
            camera_obs.CameraTranslationObs(
                name='translation',
                camera=self.camera),
            camera_obs.CameraRotationObs(
                name='rotation',
                camera=self.camera)
        ]

    def _create_reward_fns(self):
        """Initialize reward functions.

        Returns:
            List of reward functions.
        """
        if self.simulator is None:
            raise NotImplementedError(
                'Need to implement the real-world grasping reward.'
            )

        return [
            GraspReward(
                name='grasp_reward',
                end_effector_name=self.config.SIM.ARM.ARM_NAME,
                graspable_name=GRASPABLE_NAME)
        ]

    def _create_action_space(self):
        """Create the action space.

        Returns:
            The action space.
        """
        if self.config.ACTION.TYPE == 'CUBOID':
            low = self.config.ACTION.CUBOID.LOW + [0.0]
            high = self.config.ACTION.CUBOID.HIGH + [2 * np.pi]
            return gym.spaces.Box(
                    low=np.array(low),
                    high=np.array(high),
                    dtype=np.float32)
        elif self.config.ACTION.TYPE == 'IMAGE':
            height = self.camera.height
            width = self.camera.width
            return gym.spaces.Box(
                low=np.array([0, 0, 0, 0, -(2*24 - 1)]),
                high=np.array([width, height, width, height, 2*24 - 1]),
                dtype=np.float32)
        else:
            raise ValueError

    def _reset_scene(self):
        """Reset the scene in simulation or the real world.
        """

        # Configure debug visualizer
        if self._debug and self.is_simulation:
            visualizer_info = self.simulator.physics.get_debug_visualizer_info(['yaw', 'pitch', 'dist', 'target'])

            # Adjust the camera view to your own need.
            visualizer_info['dist'] = 1.5
            visualizer_info['yaw'] = 50
            visualizer_info['pitch'] = -20

            self.simulator.physics.reset_debug_visualizer(camera_distance=visualizer_info['dist'],
                                                         camera_yaw=visualizer_info['yaw'],
                                                         camera_pitch=visualizer_info['pitch'],
                                                         camera_target_position=visualizer_info['target'])
        
        super(FrankaPandaGraspEnv, self)._reset_scene()

        # Reload graspable object.
        if self.config.SIM.GRASPABLE.RESAMPLE_N_EPISODES:
            if (self.num_episodes %
                    self.config.SIM.GRASPABLE.RESAMPLE_N_EPISODES == 0):
                self.graspable_path = None

        if self.graspable_path is None:
            if self.config.SIM.GRASPABLE.USE_RANDOM_SAMPLE:
                self.graspable_path = random.choice(
                    self.all_graspable_paths)
            else:
                self.graspable_index = ((self.graspable_index + 1) %
                                        len(self.all_graspable_paths))
                self.graspable_path = (
                    self.all_graspable_paths[self.graspable_index])

        pose = Pose.uniform(x=self.config.SIM.GRASPABLE.POSE.X,
                            y=self.config.SIM.GRASPABLE.POSE.Y,
                            z=self.config.SIM.GRASPABLE.POSE.Z,
                            roll=self.config.SIM.GRASPABLE.POSE.ROLL,
                            pitch=self.config.SIM.GRASPABLE.POSE.PITCH,
                            yaw=self.config.SIM.GRASPABLE.POSE.YAW)
        pose = get_transform(source=self.table_pose).transform(pose)
        scale = np.random.uniform(*self.config.SIM.GRASPABLE.SCALE)
        logger.info('Loaded the graspable object from %s with scale %.2f...',
                    self.graspable_path, scale)
        self.graspable = self.simulator.add_body(self.graspable_path, pose, scale=scale, name=GRASPABLE_NAME, baseMass=0.1)
        # self.graspable = self.simulator.add_body(self.graspable_path, pose, scale=scale * 0.01, name=GRASPABLE_NAME, collisionFrameOrientation=[-0.5, -0.5, 0.5, 0.5], visualFrameOrientation=[-0.5, -0.5, 0.5, 0.5], baseMass=0.1)
        
        logger.debug('Waiting for graspable objects to be stable...')
        # Change some properties of the object compensating for
        # artifacts due to simulation
        self.table.set_dynamics(contact_damping=100., contact_stiffness=100.)
        self.graspable.set_dynamics(
            lateral_friction=1.,
            rolling_friction=0.001,
            spinning_friction=0.001,
            contact_damping=100.,
            contact_stiffness=100.)
        self.simulator.wait_until_stable(self.graspable)

        # Reset camera.
        self._reset_camera(
            self.camera,
            intrinsics=self.config.KINECT2.DEPTH.INTRINSICS,
            translation=self.config.KINECT2.DEPTH.TRANSLATION,
            rotation=self.config.KINECT2.DEPTH.ROTATION,
            intrinsics_noise=self.config.KINECT2.DEPTH.INTRINSICS_NOISE,
            translation_noise=self.config.KINECT2.DEPTH.TRANSLATION_NOISE,
            rotation_noise=self.config.KINECT2.DEPTH.ROTATION_NOISE)

    def _reset_robot(self):
        """Reset the robot in simulation or the real world.
        """
        super(FrankaPandaGraspEnv, self)._reset_robot()
        self.robot.reset(self.config.ARM.OFFSTAGE_POSITIONS)
        
    def _execute_action(self, action):
        """Execute the grasp action.

        Args:
            action: A 4-DoF grasp defined in the image space or the 3D space.
        """
        if self.config.ACTION.TYPE == 'CUBOID':
            x, y, z, angle = action
        elif self.config.ACTION.TYPE == 'IMAGE':
            grasp = Grasp2D.from_vector(action, camera=self.camera)
            x, y, z, angle = grasp.as_4dof()
        else:
            raise ValueError(
                'Unrecognized action type: %r' % (self.config.ACTION.TYPE))

        start = Pose(
            [[x, y, z + self.config.ARM.FINGER_TIP_OFFSET], [0, np.pi, angle]]
        )

        phase = 'initial'

        # Handle the simulation robustness.
        if self.is_simulation:
            num_action_steps = 0

        while(phase != 'done'):

            if self.is_simulation:
                self.simulator.step()
                if phase == 'start':
                    num_action_steps += 1

            if self._is_phase_ready(phase, num_action_steps):
                phase = self._get_next_phase(phase)
                logger.debug('phase: %s', phase)

                if phase == 'overhead':
                    self.robot.move_to_joint_positions(
                        self.config.ARM.OVERHEAD_POSITIONS)
                    # self.robot.grip(0)

                elif phase == 'prestart':
                    prestart = start.copy()
                    prestart.z = self.config.ARM.GRIPPER_SAFE_HEIGHT
                    self.robot.move_to_gripper_pose(prestart)

                elif phase == 'start':
                    self.robot.move_to_gripper_pose(start, straight_line=True)

                    # Prevent problems caused by unrealistic frictions.
                    if self.is_simulation:
                        self.robot.l_finger_tip.set_dynamics(
                            lateral_friction=0.001,
                            spinning_friction=0.001)
                        self.robot.r_finger_tip.set_dynamics(
                            lateral_friction=0.001,
                            spinning_friction=0.001)
                        self.table.set_dynamics(
                            lateral_friction=100)

                elif phase == 'end':
                    self.robot.grip(1)

                elif phase == 'postend':
                    postend = self.robot.end_effector.pose
                    postend.z = self.config.ARM.GRIPPER_SAFE_HEIGHT
                    self.robot.move_to_gripper_pose(
                        postend, straight_line=True)

                    # Prevent problems caused by unrealistic frictions.
                    if self.is_simulation:
                        self.robot.l_finger_tip.set_dynamics(
                            lateral_friction=100,
                            rolling_friction=10,
                            spinning_friction=10)
                        self.robot.r_finger_tip.set_dynamics(
                            lateral_friction=100,
                            rolling_friction=10,
                            spinning_friction=10)
                        self.table.set_dynamics(
                            lateral_friction=1)

    def _get_next_phase(self, phase):
        """Get the next phase of the current phase.

        Args:
            phase: A string variable.

        Returns:
            The next phase as a string variable.
        """
        phase_list = ['initial',
                      'overhead',
                      'prestart',
                      'start',
                      'end',
                      'postend',
                      'done']

        if phase in phase_list:
            i = phase_list.index(phase)
            if i == len(phase_list):
                raise ValueError('phase %r does not have a next phase.')
            else:
                return phase_list[i + 1]
        else:
            raise ValueError('Unrecognized phase: %r' % phase)

    def _is_phase_ready(self, phase, num_action_steps):
        """Check if the current phase is ready.

        Args:
            phase: A string variable.
            num_action_steps: Number of steps in the `start` phase.

        Returns:
            The boolean value indicating if the current phase is ready.
        """
        if self.is_simulation:
            if phase == 'start':
                if num_action_steps >= self.config.SIM.MAX_ACTION_STEPS:
                    logger.debug('The grasping motion is stuck.')
                    return True

            if phase == 'start' or phase == 'end':
                if self.simulator.check_contact(self.robot.arm, self.table):
                    logger.debug('The gripper contacts the table')
                    return True

        if self.robot.is_limb_ready() and self.robot.is_gripper_ready():
            return True
        else:
            return False

    # The following is an example of decomposed functions from
    # self._execute_action to better illustrate how the codebased
    # works to control the robot
        
    def execute_moving_action(self, target_pose, straight_line=True):
        """Execute the gripper to move to a target psoe.

        Args:
           target_pose (Pose)
           straight_line (bool, optional): interpolate linearly between start pose and target pose
        """
        phase = 'initial'

        if self.is_simulation:
            num_action_steps = 0

        while (phase != 'done'):
            if self.is_simulation:
                self.simulator.step()
                if phase == 'start':
                    num_action_steps += 1

                if self._is_moving_phase_ready(phase, num_action_steps):
                    phase = self._get_moving_next_phase(phase)
                    logger.debug(f'phase: {phase}')

                    if phase == 'start':
                        self.robot.move_to_gripper_pose(target_pose, straight_line=True)
                        
    def _get_moving_next_phase(self, phase):
        """Get the next phase of the current phase in moving action.

        Args:
            phase: A string variable.

        Returns:
            The next phase as a string variable.
        """
        phase_list = ['initial',
                      'start',
                      'end',
                      'done']

        if phase in phase_list:
            i = phase_list.index(phase)
            if i == len(phase_list):
                raise ValueError('phase %r does not have a next phase.')
            else:
                return phase_list[i + 1]
        else:
            raise ValueError('Unrecognized phase: %r' % phase)

    def _is_moving_phase_ready(self, phase, num_action_steps):
        """Check if the current phase is ready.

        Args:
            phase: A string variable.
            num_action_steps: Number of steps in the `start` phase.

        Returns:
            The boolean value indicating if the current phase is ready.
        """
        if self.is_simulation:
            if phase == 'start':
                if num_action_steps >= self.config.SIM.MAX_ACTION_STEPS:
                    logger.debug('The grasping motion is stuck.')
                    return True

            if phase == 'start' or phase == 'end':
                if self.simulator.check_contact(self.robot.arm, self.table):
                    logger.debug('The gripper contacts the table')
                    return True

        if self.robot.is_limb_ready() and self.robot.is_gripper_ready():
            self.robot.arm.reset_targets()
            return True
        else:
            return False
        
    def execute_grasping_action(self, target_grasp_pose):
        """Execute the grasping action given a target grasp pose

        Args:
           target_grasp_pose (Pose)
        """
        phase = 'initial'

        if self.is_simulation:
            num_action_steps = 0

        pregrasp_pose = target_grasp_pose.copy()
        pregrasp_pose.z += 0.1
            
        while (phase != 'done'):
            if self.is_simulation:
                self.simulator.step()
                if phase == 'pregrasp':
                    num_action_steps += 1

                if self._is_grasping_phase_ready(phase, num_action_steps):
                    phase = self._get_grasping_next_phase(phase)
                    logger.debug(f'phase: {phase}')
                    self.robot.grip(0)
                    if phase == 'pregrasp':
                        self.robot.move_to_gripper_pose(pregrasp_pose)
                    elif phase == 'grasp':
                        self.robot.move_to_gripper_pose(target_grasp_pose, straight_line=True)
                        if self.is_simulation:
                            self.robot.l_finger_tip.set_dynamics(
                                lateral_friction=0.001,
                                spinning_friction=0.001)
                            self.robot.r_finger_tip.set_dynamics(
                                lateral_friction=0.001,
                                spinning_friction=0.001)
                            self.table.set_dynamics(
                                lateral_friction=100)

                    elif phase == 'end':
                        self.robot.grip(1)
                    elif phase == 'postend':
                        self.robot.move_to_gripper_pose(pregrasp_pose, straight_line=True)
                        if self.is_simulation:
                            self.robot.l_finger_tip.set_dynamics(
                                lateral_friction=100,
                                rolling_friction=10,
                                spinning_friction=10)
                            self.robot.r_finger_tip.set_dynamics(
                                lateral_friction=100,
                                rolling_friction=10,
                                spinning_friction=10)
                            self.table.set_dynamics(
                                lateral_friction=1)

    def execute_gentle_grasping_action(self, target_grasp_pose):
        """Execute a more gentle grasping action given a target grasp pose

        Args:
           target_grasp_pose (Pose)
        """
        phase = 'initial'

        if self.is_simulation:
            num_action_steps = 0

        pregrasp_pose = target_grasp_pose.copy()
        pregrasp_pose.z += 0.1
            
        while (phase != 'done'):
            if self.is_simulation:
                self.simulator.step()
                if phase == 'pregrasp':
                    num_action_steps += 1

                if self._is_grasping_phase_ready(phase, num_action_steps):
                    phase = self._get_grasping_next_phase(phase)
                    logger.debug(f'phase: {phase}')
                    self.robot.grip(0)
                    if phase == 'pregrasp':
                        self.robot.move_to_gripper_pose(pregrasp_pose)
                    elif phase == 'grasp':
                        self.robot.move_to_gripper_pose(target_grasp_pose, straight_line=True)
                        if self.is_simulation:
                            self.robot.l_finger_tip.set_dynamics(
                                lateral_friction=0.001,
                                spinning_friction=0.001)
                            self.robot.r_finger_tip.set_dynamics(
                                lateral_friction=0.001,
                                spinning_friction=0.001)
                            self.table.set_dynamics(
                                lateral_friction=100)

                    elif phase == 'end':
                        value = 0.2
                        stop_l_finger = False
                        stop_r_finger = False
                        # This is an example of grasping gently
                        while value < 1.0 and not (stop_l_finger or stop_r_finger):
                            self.robot.grip(value)
                            while not self.robot.is_gripper_ready():
                                if self.simulator.check_contact(self.robot.l_finger_tip, self.graspable):
                                    stop_l_finger = True
                                    self.robot.stop_l_finger()
                                if self.simulator.check_contact(self.robot.r_finger_tip, self.graspable):
                                    stop_r_finger = True
                                    self.robot.stop_r_finger()

                                self.simulator.step()                                    
                                if stop_l_finger and stop_r_finger:
                                    while self.robot.is_gripper_ready():
                                        self.simulator.step()
                                    break
                            value += 0.1

                    elif phase == 'postend':
                        self.robot.move_to_gripper_pose(pregrasp_pose, straight_line=True)
                        if self.is_simulation:
                            self.robot.l_finger_tip.set_dynamics(
                                lateral_friction=100,
                                rolling_friction=10,
                                spinning_friction=10)
                            self.robot.r_finger_tip.set_dynamics(
                                lateral_friction=100,
                                rolling_friction=10,
                                spinning_friction=10)
                            self.table.set_dynamics(
                                lateral_friction=1)
                            
    def _get_grasping_next_phase(self, phase):
        """Get the next phase of the current phase in grasping action.

        Args:
            phase: A string variable.

        Returns:
            The next phase as a string variable.        
        """
        phase_list = ['initial',
                      'pregrasp',
                      'grasp',
                      'end',
                      'postend',
                      'done']
        if phase in phase_list:
            i = phase_list.index(phase)
            if i == len(phase_list):
                raise ValueError('phase %r does not have a next phase.')
            else:
                return phase_list[i + 1]
        else:
            raise ValueError('Unrecognized phase: %r' % phase)

    def _is_grasping_phase_ready(self, phase, num_action_steps):
        """Check if the current grasping phase is ready

        Args:
           phase: A string variable
           num_action_steps: Number of steps in the `pregrasp` phase

        Returns:
           The boolean value indicating if the current phase is ready.
        """
        if self.is_simulation:
            if phase == 'pregrasp':
                if num_action_steps >= self.config.SIM.MAX_ACTION_STEPS:
                    logger.debug('The grasping motion is stuck.')
                    return True

            if phase == 'pregrasp' or phase == 'end':
                if self.simulator.check_contact(self.robot.arm, self.table):
                    logger.debug('The gripper contacts the table')
                    return True

        if self.robot.is_limb_ready() and self.robot.is_gripper_ready():
            self.robot.arm.reset_targets()
            return True
        else:
            return False

            
            
