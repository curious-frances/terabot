# navigation/navigation_controller.py
import numpy as np
from enum import Enum
from src.Controller import Controller
from src.State import State, BehaviorState
import logging
import time

class NavigationState(Enum):
    SEARCHING = "searching"
    APPROACHING = "approaching"
    CENTERING = "centering"
    REACHED = "reached"
    ERROR = "error"

class NavigationController:
    def __init__(self, config, controller):
        self.config = config
        self.controller = controller
        self.state = State()
        self.nav_state = NavigationState.SEARCHING
        
        # Navigation parameters
        self.target_center_threshold = 50  # pixels from center to consider "centered"
        self.safe_distance = 1.0  # meters, minimum distance to maintain
        self.max_forward_speed = 0.5  # maximum forward speed
        self.max_turn_speed = 0.3  # maximum turn speed
        self.min_confidence = 0.5  # minimum detection confidence
        
        # PID controller parameters
        self.kp_center = 0.001  # proportional gain for centering
        self.kp_distance = 0.1  # proportional gain for distance control
        self.ki_center = 0.0001  # integral gain for centering
        self.kd_center = 0.0005  # derivative gain for centering
        
        # State variables
        self.center_error_integral = 0
        self.last_center_error = 0
        self.last_update_time = time.time()
        
        # Safety limits
        self.max_approach_speed = 0.3
        self.min_approach_distance = 0.5  # meters
        self.max_approach_distance = 3.0  # meters
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def calculate_control_commands(self, frame_width, frame_height, bbox, confidence):
        """
        Calculate control commands based on object detection bbox and confidence
        bbox format: [x1, y1, x2, y2] in pixels
        """
        if confidence < self.min_confidence:
            self.logger.warning(f"Low confidence detection: {confidence}")
            return self._create_safe_command()
            
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
        
        # Update state machine
        self._update_navigation_state(x_error, bbox_size)
        
        # Calculate control commands based on state
        command = self._create_base_command()
        
        if self.nav_state == NavigationState.SEARCHING:
            # Slow rotation to search for target
            command.yaw_rate = self.max_turn_speed * 0.5
            command.pitch = 0
            
        elif self.nav_state == NavigationState.APPROACHING:
            # PID control for turning
            current_time = time.time()
            dt = current_time - self.last_update_time
            
            # Update PID terms
            self.center_error_integral += x_error * dt
            derivative = (x_error - self.last_center_error) / dt if dt > 0 else 0
            
            # Calculate control outputs
            yaw_output = (self.kp_center * x_error + 
                         self.ki_center * self.center_error_integral +
                         self.kd_center * derivative)
            
            command.yaw_rate = np.clip(yaw_output, -self.max_turn_speed, self.max_turn_speed)
            
            # Forward/backward control based on bbox size
            target_size = 300  # pixels, adjust based on desired distance
            size_error = target_size - bbox_size
            command.pitch = np.clip(size_error * self.kp_distance, 
                                  -self.max_approach_speed, 
                                  self.max_approach_speed)
            
            # Update state variables
            self.last_center_error = x_error
            self.last_update_time = current_time
            
        elif self.nav_state == NavigationState.CENTERING:
            # Fine-tune position
            command.yaw_rate = np.clip(-x_error * self.kp_center * 0.5, 
                                     -self.max_turn_speed * 0.3, 
                                     self.max_turn_speed * 0.3)
            command.pitch = 0
            
        elif self.nav_state == NavigationState.REACHED:
            # Stop movement
            command.yaw_rate = 0
            command.pitch = 0
            
        else:  # ERROR state
            return self._create_safe_command()
        
        return command
    
    def _update_navigation_state(self, x_error, bbox_size):
        """Update the navigation state based on current conditions"""
        if abs(x_error) < self.target_center_threshold:
            if bbox_size > 300:  # Target is close
                self.nav_state = NavigationState.REACHED
            else:
                self.nav_state = NavigationState.CENTERING
        else:
            if bbox_size < 100:  # Target is far
                self.nav_state = NavigationState.SEARCHING
            else:
                self.nav_state = NavigationState.APPROACHING
    
    def _create_base_command(self):
        """Create a base command with default values"""
        command = type('Command', (), {})()
        command.activate_event = 0
        command.trot_event = 1
        command.hop_event = 0
        command.roll = 0.0
        command.height = 0.0
        command.yaw_rate = 0.0
        command.pitch = 0.0
        return command
    
    def _create_safe_command(self):
        """Create a safe command that stops the robot"""
        command = self._create_base_command()
        command.trot_event = 0  # Disable trotting
        return command
    
    def navigate_to_target(self, frame_width, frame_height, bbox, confidence):
        """
        Main navigation loop
        Returns True if target is reached, False otherwise
        """
        command = self.calculate_control_commands(frame_width, frame_height, bbox, confidence)
        
        # Run the controller with the calculated commands
        try:
            self.controller.run(self.state, command)
        except Exception as e:
            self.logger.error(f"Controller error: {str(e)}")
            self.nav_state = NavigationState.ERROR
            return False
        
        return self.nav_state == NavigationState.REACHED