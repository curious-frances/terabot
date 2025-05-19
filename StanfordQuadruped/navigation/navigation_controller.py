# navigation/navigation_controller.py
import numpy as np
from src.Controller import Controller
from src.State import State, BehaviorState

class NavigationController:
    def __init__(self, config, controller):
        self.config = config
        self.controller = controller
        self.state = State()
        
        # Navigation parameters
        self.target_center_threshold = 50  # pixels from center to consider "centered"
        self.safe_distance = 1.0  # meters, minimum distance to maintain
        self.max_forward_speed = 0.5  # maximum forward speed
        self.max_turn_speed = 0.3  # maximum turn speed
        
        # PID controller parameters
        self.kp_center = 0.001  # proportional gain for centering
        self.kp_distance = 0.1  # proportional gain for distance control
        
    def calculate_control_commands(self, frame_width, frame_height, bbox):
        """
        Calculate control commands based on object detection bbox
        bbox format: [x1, y1, x2, y2] in pixels
        """
        # Calculate center of bounding box
        bbox_center_x = (bbox[0] + bbox[2]) / 2
        bbox_center_y = (bbox[1] + bbox[3]) / 2
        
        # Calculate frame center
        frame_center_x = frame_width / 2
        frame_center_y = frame_height / 2
        
        # Calculate errors
        x_error = bbox_center_x - frame_center_x
        y_error = bbox_center_y - frame_center_y
        
        # Calculate bbox size for distance estimation
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        bbox_size = max(bbox_width, bbox_height)
        
        # Initialize command
        command = type('Command', (), {})()
        command.activate_event = 0
        command.trot_event = 1  # Enable trotting gait
        command.hop_event = 0
        
        # Proportional control for turning
        command.yaw_rate = np.clip(-x_error * self.kp_center, 
                                 -self.max_turn_speed, 
                                 self.max_turn_speed)
        
        # Forward/backward control based on bbox size
        target_size = 300  # pixels, adjust based on desired distance
        size_error = target_size - bbox_size
        command.pitch = np.clip(size_error * self.kp_distance, 
                              -self.max_forward_speed, 
                              self.max_forward_speed)
        
        # Keep roll and height constant
        command.roll = 0.0
        command.height = 0.0
        
        return command
    
    def navigate_to_target(self, frame_width, frame_height, bbox):
        """
        Main navigation loop
        Returns True if target is centered, False otherwise
        """
        command = self.calculate_control_commands(frame_width, frame_height, bbox)
        
        # Run the controller with the calculated commands
        self.controller.run(self.state, command)
        
        # Check if we're close enough to center
        bbox_center_x = (bbox[0] + bbox[2]) / 2
        frame_center_x = frame_width / 2
        if abs(bbox_center_x - frame_center_x) < self.target_center_threshold:
            return True  # Target is centered
        return False  # Need to continue navigation