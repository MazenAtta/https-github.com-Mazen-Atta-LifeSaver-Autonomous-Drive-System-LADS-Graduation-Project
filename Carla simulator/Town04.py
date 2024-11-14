import carla
import sys
import glob
# The added path depends on where the CARLA binaries are stored
try:
    sys.path.append(glob.glob('C:\\CARLA_0.9.14\\WindowsNoEditor\\PythonAPI\\carla')[0])
except IndexError:
    pass

import math
import numpy as np
import random
import pygame
import time
import keyboard


from agents.navigation.controller import VehiclePIDController
from agents.navigation.basic_agent import Agent
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.local_planner import LocalPlanner
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO





VEHICLE_VEL = 5

object_id = {
    "None": 0,
    "Buildings": 3,
    "Fences": 5,
    "Other": 22,
    "Pedestrians": 12,
    "Poles": 6,
    "RoadLines": 24,
    "Roads": 1,
    "Sidewalks": 2,
    "TrafficSigns": 8,
    "Vegetation": 9,
    "Car": 14,
    "Walls": 4,
    "Sky": 11,
    "Ground": 25,
    "Bridge": 26,
    "RailTrack": 27,
    "GuardRail": 28,
    "TrafficLight": 7,
    "Static": 20,
    "Dynamic": 21,
    "Water": 23,
    "Terrain": 10,
    "Truck": 15,
    "Motorcycle": 18,
    "Bicycle": 19,
    "Bus": 16,
    "Rider": 13,
    "Train": 17,
    "Any": 255
}
# Assuming the Agent class is already defined above
parking_sequence ={
    "Safe to Park" : 1,
    "Follow Lane" : 2,
    "Stop the Car" :3,
    "Change Lane" : 4
    
}
class ManeuverAgent(BasicAgent):
    def __init__(self, world, bp, actor_list, vel_ref=VEHICLE_VEL, max_throt=.75, max_brake=.3, max_steer=.8):
        self.world = world
        self.max_throt = max_throt
        self.max_brake = max_brake
        self.max_steer = max_steer
        self.vehicle = None
        self.bp = bp 
        self.max_retries = 10
        self.retry_count = 0
        self.grp = None
        self.hop_resolution = 2.0
        self.left_lane_count = 0
        self.right_lane_count = 0
        self.left_most_lanetype = None
        self.right_most_lanetype = None
        self.parking_state = None
        
        while self.vehicle is None and self.retry_count < self.max_retries:
            spawn_points = world.get_map().get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.vehicle = world.try_spawn_actor(self.bp, spawn_point)
            self.retry_count += 1
        
        if self.vehicle is None:
            raise RuntimeError("Failed to spawn vehicle after multiple retries.")
        else:
            actor_list.append(self.vehicle)
            self._vehicle = self.vehicle  # Set the vehicle for the base Agent class
            super().__init__(self._vehicle)

        self.spectator = None
        for _ in range(self.max_retries):
            try:
                self.spectator = world.get_spectator()
                break
            except RuntimeError:
                pass
        
        if self.spectator is None:
            raise RuntimeError("Failed to retrieve spectator after multiple retries.")
        

        dt = 1.0 / 20.0
        args_lateral_dict = {'K_P': 1.95, 'K_I': 0.05, 'K_D': 0.2, 'dt': dt}
        args_longitudinal_dict = {'K_P': 1.0, 'K_I': 0.05, 'K_D': 0, 'dt': dt}
        offset = 0
        
        self.controller = VehiclePIDController(self.vehicle,
                                               args_lateral=args_lateral_dict,
                                               args_longitudinal=args_longitudinal_dict,
                                               offset=offset,
                                               max_throttle=max_throt,
                                               max_brake=max_brake,
                                               max_steering=max_steer)
        self.vel_ref = vel_ref
        self.waypointsList = []
        self.current_pos = self.vehicle.get_transform().location
        self.past_pos = self.vehicle.get_transform().location
    
    
        self.lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
        self.lidar_bp.set_attribute('range', '50')
        self.lidar_bp.set_attribute('rotation_frequency', '10')
        self.lidar_bp.set_attribute('channels', '32')
        self.lidar_transform = carla.Transform(carla.Location(z=2.0))
        self.lidar_sensor = None
        self.lidar_data = None
        
      
    def dist2Waypoint(self, waypoint):
        vehicle_transform = self.vehicle.get_transform()
        vehicle_x = vehicle_transform.location.x
        vehicle_y = vehicle_transform.location.y
        waypoint_x = waypoint.transform.location.x
        waypoint_y = waypoint.transform.location.y
        return math.sqrt((vehicle_x - waypoint_x)**2 + (vehicle_y - waypoint_y)**2)
    
    def go2Waypoint(self, waypoint, draw_waypoint=True, threshold=4.0):
        if draw_waypoint:
            self.world.debug.draw_string(waypoint.transform.location, 'O', draw_shadow=False,
                                         color=carla.Color(r=255, g=0, b=0), life_time=10.0,
                                         persistent_lines=True)
        
        current_pos_np = np.array([self.current_pos.x, self.current_pos.y])
        past_pos_np = np.array([self.past_pos.x, self.past_pos.y])
        waypoint_np = np.array([waypoint.transform.location.x, waypoint.transform.location.y])
        vec2wp = waypoint_np - current_pos_np
        motion_vec = current_pos_np - past_pos_np
        dot = np.dot(vec2wp, motion_vec)
        if dot >= 0:
            while self.dist2Waypoint(waypoint) > threshold:
                control_signal = self.controller.run_step(self.vel_ref, waypoint)
                self.vehicle.apply_control(control_signal)
                self.update_spectator()

    def getLaneWaypoints(self, offset=2*VEHICLE_VEL, separation=0.3):
        self.waypointsList = []
        current_waypoint = self.world.get_map().get_waypoint(self.vehicle.get_location())
        self.waypointsList = current_waypoint.next(offset)[0].next_until_lane_end(separation)
      
    def getLeftLaneWaypoints(self, offset=2*VEHICLE_VEL, separation=0.3):
        self.waypointsList = []
        current_waypoint = self.world.get_map().get_waypoint(self.vehicle.get_location())
        left_lane = current_waypoint.get_left_lane()

        if left_lane is not None and left_lane.lane_id * current_waypoint.lane_id > 0:
            next_waypoints = left_lane.next(offset)
            if next_waypoints:
                self.waypointsList = next_waypoints[0].next_until_lane_end(separation)
            else:
                self.waypointsList = current_waypoint.next(offset)[0].next_until_lane_end(separation)
                self.waypointsList = [wp.get_left_lane() for wp in self.waypointsList if wp.get_left_lane() is not None]
    
        elif left_lane is not None and left_lane.lane_id * current_waypoint.lane_id < 0:
            previous_waypoints = left_lane.previous(offset)
            if previous_waypoints:
                self.waypointsList = previous_waypoints[0].previous_until_lane_start(separation)
            else:
                self.waypointsList = current_waypoint.previous(offset)[0].previous_until_lane_start(separation)
                self.waypointsList = [wp.get_left_lane() for wp in self.waypointsList if wp.get_left_lane() is not None]
       
    def getRightLaneWaypoints(self, offset=2*VEHICLE_VEL, separation=0.3):
        self.waypointsList = []
        current_waypoint = self.world.get_map().get_waypoint(self.vehicle.get_location())
        right_lane = current_waypoint.get_right_lane()

        if right_lane is not None:
            next_waypoints = right_lane.next(offset)
            if next_waypoints:
                self.waypointsList = next_waypoints[0].next_until_lane_end(separation)
            else:
                self.waypointsList = current_waypoint.next_until_lane_end(separation)
                self.waypointsList = [wp.get_right_lane() for wp in self.waypointsList if wp.get_right_lane() is not None]
    
    def follow_lane(self, waypoint_count = 0):
            
        self.getLaneWaypoints()
        counter = 0
        if waypoint_count == 0:
            waypoint_count = len(self.waypointsList)
        
        
        for i in range(len(self.waypointsList) - 1):
            self.current_pos = self.vehicle.get_location()
            self.go2Waypoint(self.waypointsList[i])
            self.past_pos = self.current_pos
            self.update_spectator()
            
            if counter < waypoint_count:
                counter += 1
                continue
            else:
                break
    
    def do_left_lane_change(self, waypoint_count = 0):
        
        self.getLeftLaneWaypoints()
        counter = 0
        if waypoint_count == 0:
            waypoint_count = len(self.waypointsList)
        
        
        for i in range(len(self.waypointsList) - 1):
            self.current_pos = self.vehicle.get_location()
            self.go2Waypoint(self.waypointsList[i])
            self.past_pos = self.current_pos
            self.update_spectator()
            
            if counter < waypoint_count:
                counter += 1
                continue
            else:
                break
            
    def do_right_lane_change(self, waypoint_count = 0):
        
        self.getRightLaneWaypoints()
        
        counter = 0
        if waypoint_count == 0:
            waypoint_count = len(self.waypointsList)
        
        
        
        for i in range(len(self.waypointsList) - 1):
            self.current_pos = self.vehicle.get_location()
            self.go2Waypoint(self.waypointsList[i])
            self.past_pos = self.current_pos
            self.update_spectator()

            if counter < waypoint_count:
                counter += 1
                continue
            else:
                break

    def check_for_lane(self):
        waypoint = self.world.get_map().get_waypoint(self.vehicle.get_location())  
        self.left_lane_count = 0
        self.right_lane_count = 0
        self.left_most_lanetype = None
        self.right_most_lanetype = None
        def check_driving_direction(waypoint):
            if waypoint.get_left_lane() is not None:
                if waypoint.get_left_lane().lane_id * waypoint.lane_id > 0:
                    return True
                else:
                    return False
        
        def process_lane(current_waypoint, lane_count=0, direction='None'):
            last_lanetype = None

            while current_waypoint is not None:
                lane_count += 1
                last_lanetype = current_waypoint.lane_type
                if direction == 'left':
                    next_waypoint = current_waypoint.get_left_lane()
                    if next_waypoint and check_driving_direction(next_waypoint):
                        current_waypoint = next_waypoint
                    else:
                        break
                elif direction == 'right':
                    next_waypoint = current_waypoint.get_right_lane()
                    if next_waypoint:
                        current_waypoint = next_waypoint
                    else:
                        break
                else:
                    raise ValueError('Direction must be "left" or "right"')

            return lane_count, last_lanetype
        
        
        if waypoint.get_left_lane() is not None and check_driving_direction(waypoint):
            self.left_lane_count, self.left_most_lanetype = process_lane(waypoint.get_left_lane(), self.left_lane_count, 'left')
        self.right_lane_count, self.right_most_lanetype = process_lane(waypoint.get_right_lane(), self.right_lane_count, 'right')
        
        if self.world.get_map().name == 'Carla/Maps/Town04':
            if self.left_lane_count + self.right_lane_count < 5 and self.left_lane_count + self.right_lane_count >= 2:
                self.left_lane_count+=1
                self.left_most_lanetype = carla.LaneType.Shoulder
                if self.left_lane_count + self.right_lane_count < 5:
                    self.left_lane_count+=1
    
    def check_distance(self, direction):
        # Get the vehicle transform
        transform = self.vehicle.get_transform()
        location = transform.location
        right_vector = transform.get_right_vector()
        start_location = transform.location + carla.Location(z=0.5)
        
        # Define the end points for the ray cast to the left and right
        # Calculate end locations for left and right rays
        if direction == 'left':
            end_location =  start_location + carla.Location(x=-right_vector.x * 300, y=-right_vector.y * 300, z=0.5)
        elif direction == 'right':
            end_location = start_location + carla.Location(x=right_vector.x * 300, y=right_vector.y * 300, z=0.5)

        hits = self.world.cast_ray(start_location, end_location)
        non_zero_hits = [hit for hit in hits if start_location.distance(hit.location) != 0]
        
        distance = math.sqrt((start_location.x - non_zero_hits[0].location.x)**2 + (start_location.y - non_zero_hits[0].location.y)**2)
        if distance > 4.5:
            return True
        else:
            return False
    
    
    def lidar_callback(self, data):
        self.lidar_data = data

    def check_cars(self, listen_times=5, direction="left"):
        if self.lidar_sensor is None:
            self.lidar_sensor = self.world.spawn_actor(self.lidar_bp, self.lidar_transform, attach_to=self.vehicle)
            self.lidar_sensor.listen(self.lidar_callback)

        car_detected = False
        car_distance = 0
        for _ in range(listen_times):
            if self.lidar_data is not None:
                for detection in self.lidar_data:
                    if detection.object_tag == object_id["Car"]:
                        
                        x, y = detection.point.x, detection.point.y
                        if y < 0:  # Car is behind the vehicle
                            if direction == "left" and x < 0:
                                car_detected = True
                                car_distance = math.sqrt((x ** 2) + (y ** 2))
                                return car_detected, car_distance

                            elif direction == "right" and x > 0:
                                car_detected = True
                                car_distance = math.sqrt((x ** 2) + (y ** 2))
                                return car_detected, car_distance

        if self.lidar_sensor:
            self.lidar_sensor.stop()
            self.lidar_sensor.destroy()
            self.lidar_sensor = None

        return car_detected, car_distance

    def check_trafficsigns(self, listen_times=5, direction="left"):
        if self.lidar_sensor is None:
            self.lidar_sensor = self.world.spawn_actor(self.lidar_bp, self.lidar_transform, attach_to=self.vehicle)
            self.lidar_sensor.listen(self.lidar_callback)

        traffic_sign_detected = False
        traffic_sign_distance = 0
        for _ in range(listen_times):
            if self.lidar_data is not None:
                for detection in self.lidar_data:
                    if detection.object_tag == object_id["TrafficSigns"]:
                        
                        x, y = detection.point.x, detection.point.y
                        if y > 0:  # Sign is in front of the vehicle
                            if direction == "left" and x < 0:
                                traffic_sign_detected = True
                                traffic_sign_distance = math.sqrt((x ** 2) + (y ** 2))
                                return traffic_sign_detected, traffic_sign_distance

                            elif direction == "right" and x > 0:
                                traffic_sign_detected = True
                                traffic_sign_distance = math.sqrt((x ** 2) + (y ** 2))
                                return traffic_sign_detected, traffic_sign_distance

        if self.lidar_sensor:
            self.lidar_sensor.stop()
            self.lidar_sensor.destroy()
            self.lidar_sensor = None

        return traffic_sign_detected, traffic_sign_distance
    
    def check_lanechange(self, direction, waypoint):
        self.parking_state = None
        if direction == 'left':
            left_waypoint =  waypoint.get_left_lane()
            
            if left_waypoint != None and left_waypoint.lane_type in (carla.LaneType.Shoulder, carla.LaneType.Parking):
                distance = self.check_distance(direction)
                if distance:
                    trafficsign , trafficsign_distance = self.check_trafficsigns(1000,direction)
                    car, car_distance = self.check_cars(1000, direction)
                    if not trafficsign and not car:
                        self.parking_state = parking_sequence["Safe to Park"]
                    elif trafficsign or car:
                        self.parking_state = parking_sequence["Follow Lane"]
                else:
                    self.parking_state = parking_sequence["Stop the Car"]
                    
            elif left_waypoint != None and left_waypoint.lane_type == carla.LaneType.Driving:
                self.parking_state = parking_sequence["Change Lane"]

        elif direction == 'right':
            right_waypoint =  waypoint.get_right_lane()
            
            if right_waypoint != None and right_waypoint.lane_type in (carla.LaneType.Shoulder, carla.LaneType.Parking):
                distance = self.check_distance(direction)
                if distance:
                    trafficsign , trafficsign_distance = self.check_trafficsigns(1000,direction)
                    if not trafficsign:
                        self.parking_state = parking_sequence["Safe to Park"]
                    elif trafficsign:
                        self.parking_state = parking_sequence["Follow Lane"]
                else:
                    self.parking_state = parking_sequence["Stop the Car"]
                    
            elif right_waypoint != None and right_waypoint.lane_type == carla.LaneType.Driving:
                self.parking_state = parking_sequence["Change Lane"]
        
    def park_vehicle(self):
        while True:
            self.check_for_lane()
            lane = [self.left_lane_count,self.left_most_lanetype,self.right_lane_count,self.right_most_lanetype]
            waypoint = self.world.get_map().get_waypoint(self.vehicle.get_location())  

            if lane[0] + lane[2] < 2 or waypoint.lane_type in (carla.LaneType.Shoulder, carla.LaneType.Parking):
                print("Single lane detected Stopping the car")
                self.vehicle.apply_control(carla.VehicleControl(steer =0.0,brake=1, hand_brake=True))
                break
            
            if lane[1] in (carla.LaneType.Shoulder, carla.LaneType.Parking) and lane[3] in (carla.LaneType.Shoulder, carla.LaneType.Parking):
                if lane[0] > lane[2] and waypoint.lane_type not in (carla.LaneType.Shoulder, carla.LaneType.Parking):
                    self.check_lanechange('right',waypoint)
                    if self.parking_state == parking_sequence["Safe to Park"] or self.parking_state == parking_sequence["Change Lane"]:
                        print("Performing right lane change")
                        self.do_right_lane_change(waypoint_count= 20)
                    elif self.parking_state == parking_sequence["Follow Lane"]:
                        print("Following the lane")
                        self.follow_lane(waypoint_count= 20)
                    elif self.parking_state == parking_sequence["Stop the Car"] or self.parking_state == None:
                        print("Stopping the car")
                        self.vehicle.apply_control(carla.VehicleControl(steer =0.0,brake=1, hand_brake=True))
                        break
                    
                elif lane[0] < lane[2] and waypoint.lane_type not in (carla.LaneType.Shoulder, carla.LaneType.Parking):
                    self.check_lanechange('left',waypoint)
                    if self.parking_state == parking_sequence["Safe to Park"] or self.parking_state == parking_sequence["Change Lane"]:
                        print("Performing left lane change")
                        self.do_left_lane_change(waypoint_count= 20)
                    elif self.parking_state == parking_sequence["Follow Lane"]:
                        print("Following the lane")
                        self.follow_lane(waypoint_count= 20)
                    elif self.parking_state == parking_sequence["Stop the Car"] or self.parking_state == None:
                        print("Stopping the car")
                        self.vehicle.apply_control(carla.VehicleControl(steer =0.0,brake=1, hand_brake=True))
                        break
                
            elif lane[1] in (carla.LaneType.Shoulder, carla.LaneType.Parking) and waypoint.lane_type not in (carla.LaneType.Shoulder, carla.LaneType.Parking):
                self.check_lanechange('left',waypoint)
                if self.parking_state == parking_sequence["Safe to Park"] or self.parking_state == parking_sequence["Change Lane"]:
                    print("Performing left lane change")
                    self.do_left_lane_change(waypoint_count= 20)
                elif self.parking_state == parking_sequence["Follow Lane"]:
                    print("Following the lane")
                    self.follow_lane(waypoint_count= 20)
                elif self.parking_state == parking_sequence["Stop the Car"] or self.parking_state == None:
                    print("Stopping the car")
                    self.vehicle.apply_control(carla.VehicleControl(steer =0.0, brake=1, hand_brake=True))
                    break
                             
                        
            elif lane[3] in (carla.LaneType.Shoulder, carla.LaneType.Parking) and waypoint.lane_type not in (carla.LaneType.Shoulder, carla.LaneType.Parking):
                self.check_lanechange('right',waypoint)
                if self.parking_state == parking_sequence["Safe to Park"] or self.parking_state == parking_sequence["Change Lane"]:
                    print("Performing right lane change")
                    self.do_right_lane_change(waypoint_count= 20)
                elif self.parking_state == parking_sequence["Follow Lane"]:
                    print("Following the lane")
                    self.follow_lane(waypoint_count= 20)
                elif self.parking_state == parking_sequence["Stop the Car"] or self.parking_state == None:
                    print("Stopping the car")
                    self.vehicle.apply_control(carla.VehicleControl(brake=1, hand_brake=True))
                    break
            if waypoint.lane_type in (carla.LaneType.Shoulder, carla.LaneType.Parking):
                print("Stopping the car")
                self.vehicle.apply_control(carla.VehicleControl(brake=1, hand_brake=True))
                break
              
    def update_spectator(self):
        new_yaw = math.radians(self.vehicle.get_transform().rotation.yaw)
        spectator_transform = self.vehicle.get_transform()
        spectator_transform.location += carla.Location(x=-10*math.cos(new_yaw), y=-10*math.sin(new_yaw), z=5.0)
        
        self.spectator.set_transform(spectator_transform)
        self.world.tick()
        
    def draw_waypoints(self):
        for waypoint in self.waypointsList:
            self.world.debug.draw_string(waypoint.transform.location, 'O', draw_shadow=False,
                                         color=carla.Color(r=255, g=0, b=0), life_time=10.0,
                                         persistent_lines=True)


# Additional functions for setting up traffic, and parking
def spawn_traffic(world, spawn_points, blueprint_library):
    banned_vehicles = ['vehicle.carlamotors.carlacola', 'vehicle.tesla.cybertruck', 'vehicle.carlamotors.european_hgv', 'vehicle.carlamotors.firetruck', 'vehicle.mercedes.sprinter', 'vehicle.volkswagen.t2', 'vehicle.volkswagen.t2_2021', 'vehicle.mitsubishi.fusorosa', 'vehicle.ford.ambulance']
    all_spawned_vehicles_positions = []
    new_spawn_points = spawn_points[1:]
    actor_list = []
    for _ in range(50):
        temp_loc = random.choice(new_spawn_points)
        temp_vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))
        
        # Check if the selected blueprint is a vehicle and not in the banned list
        if temp_vehicle_bp.has_attribute('role_name') and temp_vehicle_bp.id not in banned_vehicles:
            # Spawn the vehicle
            temp_vehicle = world.try_spawn_actor(temp_vehicle_bp, temp_loc)
            if temp_vehicle:
                all_spawned_vehicles_positions.append(temp_loc)
                actor_list.append(temp_vehicle)
                # Check if the spawned actor is a vehicle before setting autopilot
                if 'vehicle' in temp_vehicle.type_id:
                    temp_vehicle.set_autopilot(True)
                else:
                    print(f"Spawned actor {temp_vehicle.type_id} is not a vehicle, skipping autopilot.")
        else:
            print(f"Skipping spawn of blueprint {temp_vehicle_bp.id}.")
    return actor_list

def spawn_hosbitals(vehicle_blueprints, world,actor_list):
    
    hospitals_locations = []
    
    h1 = carla.Transform(carla.Location(x=338.789948, y=-180.229202, z=1.300000), carla.Rotation(0,90,0)) #carla.Rotation(0,90,0)
    h2 = carla.Transform(carla.Location(x=232.315262, y=-241.054062, z=1.300000), carla.Rotation(0,0,0))

    hospitals_locations.append(h1)
    hospitals_locations.append(h2)
    
    ambulance_bp  = vehicle_blueprints.filter('vehicle.ford.ambulance')[0]

    hospital1 = world.try_spawn_actor(ambulance_bp, h1)
    hostpital2 = world.try_spawn_actor(ambulance_bp, h2)
    actor_list.append(hospital1)
    actor_list.append(hostpital2)
    
    return hospitals_locations

def get_nearest_hospital(vehicle, hospitals_locations , carla_map):
    
    def calculate_road_distance(carla_map, start_location, end_location):
        start_waypoint = carla_map.get_waypoint(start_location)
        end_waypoint = carla_map.get_waypoint(end_location)
        distance = start_waypoint.transform.location.distance(end_waypoint.transform.location)
        return distance
    
    near_location=[]
    ego_loc = vehicle.get_location()

    for index,location in enumerate(hospitals_locations) :
        temp_h = location.location
        distance = calculate_road_distance(carla_map,ego_loc,temp_h)
        near_location.append(distance)

    near_hos_loc = min(near_location)

    for index,loc in enumerate(near_location) :
        if near_hos_loc == near_location[index]:
            return [index,loc]
           
def move_to_hospital(Maneuver_Agent, hospitals_locations, Map):
    state = 2
    loc = 5
    while True:
        if state == 1 and loc > 20:
            near_location_ind,loc = get_nearest_hospital(Maneuver_Agent.vehicle, hospitals_locations, Map)
            Maneuver_Agent.set_destination(hospitals_locations[near_location_ind].location)

        if state == 2:
            near_location,loc = get_nearest_hospital(Maneuver_Agent.vehicle, hospitals_locations, Map)
            Maneuver_Agent.set_destination(hospitals_locations[near_location].location)
            state = 0
        Maneuver_Agent.update_spectator()
        control = Maneuver_Agent.run_step()
        
        Maneuver_Agent.vehicle.apply_control(control)
        if Maneuver_Agent.done():
            Maneuver_Agent.vehicle.apply_control(carla.VehicleControl(brake=1, hand_brake=True))
            print("Reached hospital.")
            break
        


# Main CARLA simulation environment
def main():
    parking_mode = False
    Hospital_mode = False
    auto_driving = True
    actor_list = []
    
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    client.load_world("Town04")
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    hospitals_locations = spawn_hosbitals(blueprint_library, world, actor_list)
    actor_list = spawn_traffic(world, spawn_points, blueprint_library) 
    

    
    vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
    Maneuver_Agent = ManeuverAgent(world, vehicle_bp,actor_list)
        
    destination = random.choice(spawn_points).location
    Maneuver_Agent.set_destination(destination)
    
    

        # Main simulation loop
    while True:
        Maneuver_Agent.update_spectator()
        
        if keyboard.is_pressed('e'):
            parking_mode = True
            auto_driving = False
        elif keyboard.is_pressed('a'):
            parking_mode = False
            auto_driving = True

        if not parking_mode and auto_driving:
            control = Maneuver_Agent.run_step()
            Maneuver_Agent.vehicle.apply_control(control)


            if Maneuver_Agent.done():
                destination = random.choice(spawn_points).location
                Maneuver_Agent.set_destination(destination)
        
        if parking_mode and not auto_driving:
                Maneuver_Agent.park_vehicle()                
                current_time = time.time()
                Hospital_mode = True
                parking_mode = False
            
        if Hospital_mode == True and time.time() > current_time + 5:
            print('Hospital')
            move_to_hospital(Maneuver_Agent, hospitals_locations,world.get_map())
            Hospital_mode == False
                


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

