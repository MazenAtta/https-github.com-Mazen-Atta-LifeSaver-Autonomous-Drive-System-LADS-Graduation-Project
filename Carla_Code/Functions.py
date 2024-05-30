from typing import List
#all imports
import carla
import sys
sys.path.append('C:\CARLA_0.9.14\WindowsNoEditor\PythonAPI\carla')
from agents.navigation.basic_agent import BasicAgent

import cv2 
import time 
import numpy as np 
#import pygame
import math
import random
import threading,time
import time
import paho.mqtt.client as mqtt
from Functions import *
#from pub import #publish_msg

# camera mount offset on the car - you can tweak these to have the car in view or not
CAMERA_POS_X = .9
CAMERA_POS_Y = 0
CAMERA_POS_Z = 1.6
CAMERA_ROT_ROLL = 0
CAMERA_ROT_PITCH = 0
CAMERA_ROT_YAW = 0


# camera mount offset on the car - you can tweak these to have the car in view or not
OBSTACLE_POS_X = 0
OBSTACLE_POS_Y = .9
OBSTACLE_POS_Z = 1
OBSTACLE_ROT_ROLL = 0
OBSTACLE_ROT_PITCH = 0
OBSTACLE_ROT_YAW = 90

# connect to the sim
client = carla.Client('localhost', 2000)
# setting up the map
client.load_world("Town04")
world = client.get_world()
settings = world.get_settings()
settings.no_rendering_mode = True
world.apply_settings


hospitals_locations = []
AllSpawndVechilesPositions = []
near_location = []
vehicle_blueprints = world.get_blueprint_library().filter('*vehicle*')
world = client.get_world()
spawn_points = world.get_map().get_spawn_points()
h1 = carla.Transform(carla.Location(x=280.313599, y=-197.952637, z=1.565966))

banned_vehciles = ['vehicle.carlamotors.carlacola', 'vehicle.tesla.cybertruck', 'vehicle.carlamotors.european_hgv', 'vehicle.carlamotors.firetruck',
                   'vehicle.mercedes.sprinter', 'vehicle.volkswagen.t2', 'vehicle.volkswagen.t2_2021', 'vehicle.mitsubishi.fusorosa', 'vehicle.ford.ambulance']


ParkState = 2

def monitor_file():
    global ParkState
    while True:
        # Read Parking State From file
        try:
            file = open(
                f'D:\Projects\Project 3\Code\carla_simulation\hparkState.txt', 'r')
            ParkState = int(file.readline())
            print(f'ParkState = {ParkState}')
            file.close()
        except:
            # creat the file
            print("Error cant open file")

        time.sleep(1)  # Adjust the interval as needed

# monitor_thread = threading.Thread(target=monitor_file)
# monitor_thread.start()


def spawn_traffic(num_vehicles=50, spawned_positions=[]):
    """
    Spawns a specified number of vehicles in the simulation.

    Args:
    - num_vehicles: int, the number of vehicles to spawn (default is 50)
    - spawned_positions: list, a list of positions where vehicles have already been spawned (default is an empty list)
    """
    new_spawn_points = spawn_points[1:]
    allowed_vehicle_blueprints = [v for v in vehicle_blueprints if v.id not in banned_vehciles]
    
    for x in range(0, num_vehicles):
        temp_loc = random.choice(new_spawn_points)
        temp_vech = random.choice(allowed_vehicle_blueprints)
        
        if temp_loc not in spawned_positions and temp_vech.id not in banned_vehciles:
            spawned_positions.append(temp_loc)
            try:
                temp_vehicle = world.try_spawn_actor(temp_vech, temp_loc)
                temp_vehicle.set_autopilot(True)
            except Exception as e:
                print(f"Error spawning actor: {e}")


def obstacle_callback(event, data_dict):
    '''
    Process the obstacle event and update the data dictionary with information about the detected obstacle.

    Parameters:
    - event: CARLA obstacle event
    - data_dict: dictionary to store obstacle information

    Returns:
    None
    '''
    if event.other_actor is not None and isinstance(data_dict, dict) and 'static' not in event.other_actor.type_id:
        if 'obstacle' not in data_dict:
            data_dict['obstacle'] = []
        data_dict['obstacle'].append({'type_id': event.other_actor.type_id, 'frame': event.frame})
        # Adding logging statement
        print(f"Obstacle detected: {event.other_actor.type_id} at frame {event.frame}")

def check_for_obstacles(data_dic):
    '''
    This function takes the data_dic dictionoray
    checks if there are any vehicles detected by the obstacle sensor
    returns True or False based on the value
    '''
    if data_dic['obstacle']:
        obstacles = data_dic['obstacle'][0]
        if 'vehicle' in obstacles['transform']:
            print('test')
            return True
    else:
        return False

def lane_inv_callback(event, data_dict):
    '''
    Process the lane invasion event and update the data dictionary with the type of crossed lane markings.

    Parameters:
    - event: CARLA lane invasion event
    - data_dict: dictionary to store lane invasion information

    Returns:
    None
    '''
    if hasattr(event, 'crossed_lane_markings') and all(hasattr(x, 'type') for x in event.crossed_lane_markings):
        lane_types = set()
        for x in event.crossed_lane_markings:
            lane_types.add(x.type)
        
        text = [x.type for x in event.crossed_lane_markings]

        data_dict['type_of_crossed_lane_marking'] = ' and '.join(text)
        data_dict['lane_invasion'] = True


def check_for_crossed_lanetype(data_dic):
    '''
    This function takes data_dic dictionary 
    checks the type of crossed line 
    and returns itm
    '''
    lane_type = data_dic['crossed_line_type']
    if lane_type == "'Solid'":
        last_crossed_line = 'Solid'
        return 'Solid'
    elif lane_type == "'Broken'":
        last_crossed_line = 'Solid'
        return 'Broken'
    else:
        return None


def camera_callback(image, data_dict):
    """
    Process the camera image and store it in the data dictionary.

    Parameters:
    - image: CARLA camera image
    - data_dict: dictionary to store processed image

    Returns:
    None
    """
    if image is None or data_dict is None or not hasattr(image, 'raw_data') or not hasattr(image, 'height') or not hasattr(image, 'width'):
        return
    
    data_dict["processed_image"] = image.raw_data.reshape(image.height, image.width, 4)

def sumMatrix(A: List[int], B: List[int]) -> List[int]:
    """
    Sums two matrices element-wise.

    Args:
        A (list): The first matrix.
        B (list): The second matrix.

    Returns:
        list: The sum of the two matrices.
    """
    if len(A) != len(B):
        raise ValueError("Input lists must have the same length")
    A = np.array(A)
    B = np.array(B)
    sum_result = A + B
    return sum_result.tolist()

def process_image(image):

    car_state = 'Center'
    pt1_sum_ri = (0, 0)
    pt2_sum_ri = (0, 0)
    pt1_avg_ri = (0, 0)
    count_posi_num_ri = 0

    pt1_sum_le = (0, 0)
    pt2_sum_le = (0, 0)
    pt1_avg_le = (0, 0)
    count_posi_num_le = 0

    test_im = image.copy()
    test_im = test_im.reshape((image.shape[0], image.shape[1], 4))
    test_im = test_im[:, :, :3]

    size_im = cv2.resize(test_im, dsize=(640, 480))  # VGA resolution

    roi = size_im[240:480, 108:532]  # [380:430, 330:670]   [y:y+b, x:x+a]
    roi_im = cv2.resize(roi, (424, 240))  # (a of x, b of y)

    if roi_im.dtype != np.uint8:
        roi_im = np.uint8(roi_im)
        size_im = np.uint8(size_im)
    
    Blur_im = cv2.bilateralFilter(roi_im, d=-1, sigmaColor=3, sigmaSpace=3)

    edges = cv2.Canny(Blur_im, 50, 100)

    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180.0, threshold=25, minLineLength=10, maxLineGap=20)

    if lines is not None:
        N = lines.shape[0]
        for line in range(N):
            x1 = lines[line][0][0]
            y1 = lines[line][0][1]
            x2 = lines[line][0][2]
            y2 = lines[line][0][3]

            if x2 == x1:
                a = 1
            else:
                a = x2 - x1

            b = y2 - y1

            radi = b / a

            theta_atan = math.atan(radi) * 180.0 / math.pi

            pt1_ri = (x1 + 108, y1 + 240)
            pt2_ri = (x2 + 108, y2 + 240)
            pt1_le = (x1 + 108, y1 + 240)
            pt2_le = (x2 + 108, y2 + 240)

            if theta_atan > 30.0 and theta_atan < 80.0:
                count_posi_num_ri += 1
                pt1_sum_ri = sumMatrix(pt1_ri, pt1_sum_ri)
                pt2_sum_ri = sumMatrix(pt2_ri, pt2_sum_ri)

            if theta_atan < -30.0 and theta_atan > -80.0:
                count_posi_num_le += 1
                pt1_sum_le = sumMatrix(pt1_le, pt1_sum_le)
                pt2_sum_le = sumMatrix(pt2_le, pt2_sum_le)

    if count_posi_num_ri != 0:
        pt1_avg_ri = pt1_sum_ri // np.array(count_posi_num_ri)
        pt2_avg_ri = pt2_sum_ri // np.array(count_posi_num_ri)
    else:
        pt1_avg_ri = (0, 0)
        pt2_avg_ri = (0, 0)

    if count_posi_num_le != 0:
        pt1_avg_le = pt1_sum_le // np.array(count_posi_num_le)
        pt2_avg_le = pt2_sum_le // np.array(count_posi_num_le)
    else:
        pt1_avg_le = (0, 0)
        pt2_avg_le = (0, 0)

    x1_avg_ri, y1_avg_ri = pt1_avg_ri
    x2_avg_ri, y2_avg_ri = pt2_avg_ri

    if x2_avg_ri != x1_avg_ri:
        a_avg_ri = ((y2_avg_ri - y1_avg_ri) / (x2_avg_ri - x1_avg_ri))
        b_avg_ri = (y2_avg_ri - (a_avg_ri * x2_avg_ri))
    else:
        a_avg_ri = 0
        b_avg_ri = 0

    pt2_y2_fi_ri = 480

    if a_avg_ri > 0:
        pt2_x2_fi_ri = int((pt2_y2_fi_ri - b_avg_ri) // a_avg_ri)
    else:
        pt2_x2_fi_ri = 0

    pt2_fi_ri = (pt2_x2_fi_ri, pt2_y2_fi_ri)

    x1_avg_le, y1_avg_le = pt1_avg_le
    x2_avg_le, y2_avg_le = pt2_avg_le

    if x2_avg_le != x1_avg_le:
        a_avg_le = ((y2_avg_le - y1_avg_le) / (x2_avg_le - x1_avg_le))
        b_avg_le = (y2_avg_le - (a_avg_le * x2_avg_le))
    else:
        a_avg_le = 0
        b_avg_le = 0

    pt1_y1_fi_le = 480
    if a_avg_le < 0:
        pt1_x1_fi_le = int((pt1_y1_fi_le - b_avg_le) // a_avg_le)
    else:
        pt1_x1_fi_le = 0

    pt1_fi_le = (pt1_x1_fi_le, pt1_y1_fi_le)

    cv2.line(size_im, tuple(pt1_avg_ri), tuple(pt2_fi_ri), (0, 255, 0), 2)  # right lane
    cv2.line(size_im, tuple(pt1_fi_le), tuple(pt2_avg_le), (0, 255, 0), 2)  # left lane
    cv2.line(size_im, (320, 480), (320, 360), (0, 228, 255), 1)  # middle lane

    FCP_img = np.zeros(shape=(480, 640, 3), dtype=np.uint8) + 0
    FCP = np.array([pt2_avg_le, pt1_fi_le, pt2_fi_ri, pt1_avg_ri])
    cv2.fillConvexPoly(FCP_img, FCP, color=(255, 242, 213))  # BGR

    alpha = 0.9
    size_im = cv2.addWeighted(size_im, alpha, FCP_img, 1 - alpha, 0)

    lane_center_y_ri = 360
    if a_avg_ri > 0:
        lane_center_x_ri = int((lane_center_y_ri - b_avg_ri) // a_avg_ri)
    else:
        lane_center_x_ri = 0

    lane_center_y_le = 360
    if a_avg_le < 0:
        lane_center_x_le = int((lane_center_y_le - b_avg_le) // a_avg_le)
    else:
        lane_center_x_le = 0

    cv2.line(size_im, (lane_center_x_le, lane_center_y_le - 10), (lane_center_x_le, lane_center_y_le + 10),
             (0, 228, 255), 1)
    cv2.line(size_im, (lane_center_x_ri, lane_center_y_ri - 10), (lane_center_x_ri, lane_center_y_ri + 10),
             (0, 228, 255), 1)

    lane_center_x = ((lane_center_x_ri - lane_center_x_le) // 2) + lane_center_x_le
    cv2.line(size_im, (lane_center_x, lane_center_y_ri - 10), (lane_center_x, lane_center_y_le + 10),
             (0, 228, 255), 1)

    text_left = 'Turn Left'
    text_right = 'Turn Right'
    text_center = 'Center'
    text_non = ''
    org = (320, 440)
    font = cv2.FONT_HERSHEY_SIMPLEX

    if 0 < lane_center_x <= 318:
        cv2.putText(size_im, text_left, org, font, 0.7, (0, 0, 255), 2)
        car_state = 'Left'
    elif 318 < lane_center_x < 322:
        cv2.putText(size_im, text_center, org, font, 0.7, (0, 0, 255), 2)
        car_state = 'Center'
    elif lane_center_x >= 322:
        cv2.putText(size_im, text_right, org, font, 0.7, (0, 0, 255), 2)
        car_state = 'Right'
    elif lane_center_x == 0:
        cv2.putText(size_im, text_non, org, font, 0.7, (0, 0, 255), 2)
        car_state = 'Non'

    count_posi_num_ri = 0
    pt1_sum_ri = (0, 0)
    pt2_sum_ri = (0, 0)
    pt1_avg_ri = (0, 0)
    pt2_avg_ri = (0, 0)

    count_posi_num_le = 0
    pt1_sum_le = (0, 0)
    pt2_sum_le = (0, 0)
    pt1_avg_le = (0, 0)

    return size_im,car_state,lane_center_x


def normalize_data(data):
    min_val = 0
    max_val = 1
    normalized = (data - min_val) / (max_val - min_val)
    return normalized


def get_angle(car, wp):
    '''
    this function returns degrees between the car's direction 
    and direction to a selected waypoint
    '''
    vehicle_pos = car.get_transform()
    car_x = vehicle_pos.location.x
    car_y = vehicle_pos.location.y
    wp_x = wp.transform.location.x
    wp_y = wp.transform.location.y

    # vector to waypoint
    x = (wp_x - car_x)/((wp_y - car_y)**2 + (wp_x - car_x)**2)**0.5
    y = (wp_y - car_y)/((wp_y - car_y)**2 + (wp_x - car_x)**2)**0.5

    # car vector
    car_vector = vehicle_pos.get_forward_vector()
    degrees = math.degrees(np.arctan2(
        y, x) - np.arctan2(car_vector.y, car_vector.x))
    # extra checks on predicted angle when values close to 360 degrees are returned
    if degrees < -300:
        degrees = degrees + 360
    elif degrees > 300:
        degrees = degrees - 360
    return degrees


def change_lane(vehicle, direction, distance=10.0, duration=3.0, frequency=20):
    """
    Gradually changes the vehicle's lane towards the target waypoint in the specified direction.

    Args:
        vehicle (carla.Vehicle): The vehicle to control.
        direction (str): The direction to change the lane, either 'left' or 'right'.
        distance (float): The distance ahead in the target lane to set the target waypoint.
        duration (float): Duration over which the lane change should be completed (in seconds).
        frequency (int): How often to update the vehicle's control (in Hz).

    Returns:
        None
    """
    map = vehicle.get_world().get_map()
    current_waypoint = map.get_waypoint(vehicle.get_location())

    if direction == 'left':
        target_waypoint = current_waypoint.get_left_lane()
    elif direction == 'right':
        target_waypoint = current_waypoint.get_right_lane()
    else:
        raise ValueError("Direction should be either 'left' or 'right'")



    # Move the target waypoint ahead by the specified distance
    target_waypoint = target_waypoint.next(
        distance)[0] if target_waypoint.next(distance) else target_waypoint

    start_time = time.time()
    current_time = start_time
    end_time = start_time + duration

    while current_time < end_time:
        image, car_state, lane_center = process_image(sensor_data['image'])
        cv2.imshow('RGB Camera', image)
        cv2.waitKey(1)
        elapsed_time = current_time - start_time
        progress = elapsed_time / duration

        # Calculate the intermediate target location between the current vehicle location and the target waypoint
        current_location = vehicle.get_location()
        target_location = target_waypoint.transform.location

        # Interpolate between the current location and the target location based on progress
        intermediate_x = (1 - progress) * current_location.x + \
            progress * target_location.x
        intermediate_y = (1 - progress) * current_location.y + \
            progress * target_location.y
        intermediate_location = carla.Location(
            x=intermediate_x, y=intermediate_y)

        # Calculate the steering angle needed to head towards the intermediate location
        direction_vector = carla.Location(
            intermediate_x - current_location.x, intermediate_y - current_location.y)
        vehicle_transform = vehicle.get_transform()
        forward_vector = vehicle_transform.get_forward_vector()
        dot_product = forward_vector.x * direction_vector.x + \
            forward_vector.y * direction_vector.y
        det = forward_vector.x * direction_vector.y - \
            forward_vector.y * direction_vector.x
        angle = math.atan2(det, dot_product)

        # Convert angle to steering command
        # Assuming max steering angle is pi/4 radians
        steer_command = angle / (math.pi / 4)
        # Clamp value between -1 and 1
        steer_command = max(min(steer_command, 1.0), -1.0)

        if direction == 'right':
            steer_command = abs(steer_command)
        elif direction == 'left':
            # Negative value for steering to left
            steer_command = -abs(steer_command)
        # Apply control to the vehicle
        vehicle.apply_control(carla.VehicleControl(
            throttle=0.5, steer=steer_command))  # Adjust throttle as needed

        # Wait for the next update
        # time.sleep(1.0 / frequency)
        current_time = time.time()
        CarSpeed = vehicle.get_velocity().length()
        steer_angle = vehicle.get_control().steer
        #publish_msg(str(CarSpeed), 'esp32/CarSpeed')
        #publish_msg(str(steer_angle), 'esp32/CarSteer')

    # Optionally, straighten the vehicle after completing the lane change
    vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.0))
    if not target_waypoint or target_waypoint.lane_type != carla.LaneType.Driving:
        return target_waypoint.lane_type

def calculate_steering_angle(image, lane_center_x):
    image_center_x = image.shape[1] // 2
    
    # Calculate the difference in x-coordinates
    dx = lane_center_x - image_center_x
    
    # Calculate the angle in radians, then convert to degrees
    angle_radians = math.atan2(dx, image.shape[0])  # Assuming the y-coordinate difference is the height of the image
    angle_degrees = math.degrees(angle_radians)
    
    # Normalize the angle to the range [-1, 1] for CARLA steering control
    # Assuming the maximum steering angle is approximately 45 degrees (which is a common assumption)
    max_steering_angle_radians = math.radians(70)
    steering_value = angle_radians / max_steering_angle_radians
    
    # Clamp the steering value to the range [0, 1] to ensure it's valid for vehicle control
    steering_value = normalize_data(steering_value)
    print(steering_value)
    return abs(steering_value)



vehicle_bp = world.get_blueprint_library().filter('vehicle.mercedes.coupe_2020')
start_point = spawn_points[0]
vehicle = world.try_spawn_actor(vehicle_bp[0], start_point)
basic_vehicle = BasicAgent(vehicle)
spectator = world.get_spectator()
spectator.set_transform(vehicle.get_transform())

# Getting the needed blueprints
gnss_sensor_bp = world.get_blueprint_library().find('sensor.other.gnss')
lane_invasion_sensor_bp = world.get_blueprint_library().find(
    'sensor.other.lane_invasion')
obstacle_sensor_bp = world.get_blueprint_library().find('sensor.other.obstacle')
camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')

# Spawning Obstacle sensor in simulation and setting its position and attributes
obstacle_sensor_bp.set_attribute('hit_radius', '.5')
obstacle_sensor_bp.set_attribute('only_dynamics', 'True')
obstacle_sensor_bp.set_attribute('distance', '1')
obstacle_init_trans = carla.Transform(carla.Location(x=OBSTACLE_POS_X, y=OBSTACLE_POS_Y, z=OBSTACLE_POS_Z),
                                      carla.Rotation(roll=OBSTACLE_ROT_ROLL, pitch=OBSTACLE_ROT_PITCH, yaw=OBSTACLE_ROT_YAW))
obstacle = world.spawn_actor(
    obstacle_sensor_bp, obstacle_init_trans, attach_to=vehicle)

# Spawning Lane Invasion sensor in simulation
lane_inv = world.spawn_actor(
    lane_invasion_sensor_bp, carla.Transform(), attach_to=vehicle)

# Spawning Camera in simulation and setting its position and attributes
camera_bp.set_attribute('image_size_x', '640')
camera_bp.set_attribute('image_size_y', '360')
camera_bp.set_attribute('fov', '90')  # default = 90
camera_init_trans = carla.Transform(
    carla.Location(z=CAMERA_POS_Z, x=CAMERA_POS_X))
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)
image_w = camera_bp.get_attribute('image_size_x').as_int()
image_h = camera_bp.get_attribute('image_size_y').as_int()


# Listening for incoming data
sensor_data = {'image': np.zeros((image_h, image_w, 4)),
               'crossed_line_type': '',
               'lane_invasion': False,
               'obstacle': []}

camera.listen(lambda image: camera_callback(image, sensor_data))
obstacle.listen(lambda event: obstacle_callback(event, sensor_data))
lane_inv.listen(lambda event: lane_inv_callback(event, sensor_data))

cv2.namedWindow('RGB Camera', cv2.WINDOW_AUTOSIZE)