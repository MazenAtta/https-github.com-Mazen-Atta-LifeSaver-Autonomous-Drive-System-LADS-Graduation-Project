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
import keyboard
from agents.navigation.controller import VehiclePIDController
from agents.navigation.basic_agent import Agent
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.local_planner import LocalPlanner
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO


VEHICLE_VEL = 5

# Assuming the Agent class is already defined above

class ManeuverAgent(BasicAgent):
    def __init__(self, world, bp, actor_list, vel_ref=VEHICLE_VEL, max_throt=1, max_brake=0.3, max_steer=.8):
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
    
    def dist2Waypoint(self, waypoint):
        vehicle_transform = self.vehicle.get_transform()
        vehicle_x = vehicle_transform.location.x
        vehicle_y = vehicle_transform.location.y
        waypoint_x = waypoint.transform.location.x
        waypoint_y = waypoint.transform.location.y
        return math.sqrt((vehicle_x - waypoint_x)**2 + (vehicle_y - waypoint_y)**2)
    
    def go2Waypoint(self, waypoint, draw_waypoint=True, threshold=0.3):
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
                



    def getLeftLaneWaypoints(self, offset=2*VEHICLE_VEL, separation=0.3):
        current_waypoint = self.world.get_map().get_waypoint(self.vehicle.get_location())
        left_lane = current_waypoint.get_left_lane()
        if left_lane is not None:
            self.waypointsList = left_lane.previous(offset)[0].previous_until_lane_start(separation)

    def getRightLaneWaypoints(self, offset=2*VEHICLE_VEL, separation=0.3):
        current_waypoint = self.world.get_map().get_waypoint(self.vehicle.get_location())
        right_lane = current_waypoint.get_right_lane()
        if right_lane is not None:
            self.waypointsList = right_lane.next(offset)[0].next_until_lane_end(separation)
    
    def do_left_lane_change(self):
        self.getLeftLaneWaypoints()
        counter = 0

        for i in range(len(self.waypointsList) - 1):
            self.current_pos = self.vehicle.get_location()
            self.go2Waypoint(self.waypointsList[i])
            self.past_pos = self.current_pos
            self.update_spectator()
            if counter == 10:
                print("Left lane change done")
                break
            else:
                counter += 1
        return True

    def do_right_lane_change(self):
        self.getRightLaneWaypoints()
        counter = 0

        for i in range(len(self.waypointsList) - 1):
            self.current_pos = self.vehicle.get_location()
            self.go2Waypoint(self.waypointsList[i])
            self.past_pos = self.current_pos
            self.update_spectator()
            if counter == 10:
                print("Right lane change done")
                break
            else:
                counter += 1
        return True

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

# Additional functions for setting up sensors, traffic, and parking
def setup_pygame():
    pygame.init()
    display = pygame.display.set_mode((800, 600))
    pygame.display.set_caption('CARLA Simulator')
    return display

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
    
    h1 = carla.Transform(carla.Location(x=-39.757774, y=146.348129, z=1.300000)) #carla.Rotation(0,90,0)
    h2 = carla.Transform(carla.Location(x=75.081476, y=5.813677, z=1.300000), carla.Rotation(0,180,0))
    
    hospitals_locations.append(h1)
    hospitals_locations.append(h2)
    
    ambulance_bp  = vehicle_blueprints.filter('vehicle.ford.ambulance')[0]

    hospital1 = world.try_spawn_actor(ambulance_bp, h1)
    hostpital2 = world.try_spawn_actor(ambulance_bp, h2)
    actor_list.append(hospital1)
    actor_list.append(hostpital2)
    
    return hospitals_locations

def get_nearest_hospital(vehicle, hospitals_locations , map):
    def calculate_road_distance(map, start_location, end_location):
        start_waypoint = map.get_waypoint(start_location)
        end_waypoint = map.get_waypoint(end_location)
        distance = start_waypoint.transform.location.distance(end_waypoint.transform.location)
        return distance
    
    near_location=[]
    ego_loc = vehicle.get_location()

    for index,location in enumerate(hospitals_locations) :
        temp_h = location.location
        distance = calculate_road_distance(map,ego_loc,temp_h)
        near_location.append(distance)

    near_hos_loc = min(near_location)

    for index,loc in enumerate(near_location) :
        if near_hos_loc == near_location[index]:
            return [index,loc]

def park_vehicle(Maneuver_Agent, world):
    waypoint = world.get_map().get_waypoint(Maneuver_Agent.vehicle.get_location())

    while True:
        lane_state_list = check_for_lane(world,waypoint)
        
        if waypoint.lane_type in (carla.LaneType.Shoulder, carla.LaneType.Parking):
            print("Reached parking lane.")
            Maneuver_Agent.vehicle.apply_control(carla.VehicleControl(brake=1, hand_brake=True))
            break

        if lane_state_list[0] + lane_state_list[2] == 1:
            print("Single lane detected. Stopping vehicle.")
            Maneuver_Agent.vehicle.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
            break

        elif lane_state_list[1] in (carla.LaneType.Shoulder, carla.LaneType.Parking) and lane_state_list[3] in (carla.LaneType.Shoulder, carla.LaneType.Parking):
            if lane_state_list[0] > lane_state_list[2] and waypoint.lane_type not in (carla.LaneType.Shoulder, carla.LaneType.Parking):
                print("Performing right lane change")
                for _ in range(lane_state_list[2]):
                    Maneuver_Agent.do_right_lane_change()
                print("Reached parking lane.")
                Maneuver_Agent.vehicle.apply_control(carla.VehicleControl(brake=1, hand_brake=True))
                break

            elif lane_state_list[0] < lane_state_list[2] and waypoint.lane_type not in (carla.LaneType.Shoulder, carla.LaneType.Parking):
                print("Performing left lane change")
                for _ in range(lane_state_list[0]):
                    Maneuver_Agent.do_left_lane_change()
                print("Reached parking lane.")
                Maneuver_Agent.vehicle.apply_control(carla.VehicleControl(brake=1, hand_brake=True))
                break
            
            elif lane_state_list[1] in (carla.LaneType.Shoulder, carla.LaneType.Parking) and waypoint.lane_type not in (carla.LaneType.Shoulder, carla.LaneType.Parking):
                print("Performing left lane change")
                for _ in range(lane_state_list[0]):
                    Maneuver_Agent.do_left_lane_change()
                    
                print("Reached parking lane.")
                Maneuver_Agent.vehicle.apply_control(carla.VehicleControl(brake=1, hand_brake=True))
                break                    
                    
        elif lane_state_list[3] in (carla.LaneType.Shoulder, carla.LaneType.Parking) and waypoint.lane_type not in (carla.LaneType.Shoulder, carla.LaneType.Parking):
            print("Performing right lane change")
            for _ in range(lane_state_list[2]):
                Maneuver_Agent.do_right_lane_change()
            print("Reached parking lane.")
            Maneuver_Agent.vehicle.apply_control(carla.VehicleControl(brake=1, hand_brake=True))
            break  
    return True          

def check_for_lane(world,waypoint):
    left_most_lanetype = None
    right_most_lanetype = None
    left_lane_count = 0
    right_lane_count = 0    

    def check_pitch(waypoint):
        if waypoint.get_left_lane() is not None:
            return waypoint.get_left_lane().transform.rotation.pitch != waypoint.transform.rotation.pitch + 360 and waypoint.get_left_lane().transform.rotation.pitch != waypoint.transform.rotation.pitch - 360
    
    def process_lane(current_waypoint, lane_count=0, direction='None'):
        last_lanetype = None

        while current_waypoint is not None:
            if current_waypoint.lane_type == carla.LaneType.Driving:
                lane_count += 1
                last_lanetype = current_waypoint.lane_type
                if direction == 'left':
                    next_waypoint = current_waypoint.get_left_lane()
                    if next_waypoint and check_pitch(next_waypoint):
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
            else:
                if current_waypoint.lane_type in (carla.LaneType.Shoulder, carla.LaneType.Parking):
                    lane_count += 1
                    last_lanetype = current_waypoint.lane_type
                break

        return lane_count, last_lanetype
    
    
    def detect_intersection_merges_splits(waypoint):
        """
        Detects intersections, merges, and splits at the given waypoint.

        Args:
            waypoint (carla.Waypoint): The current waypoint to analyze.

        Returns:
            dict: A dictionary with the following keys:
                'is_intersection': True if at an intersection, False otherwise.
                'is_merge': True if at a merge, False otherwise.
                'is_split': True if at a split, False otherwise.
        """
        results = {
            'is_intersection': False,
            'is_merge': False,
            'is_split': False
        }

        left_lane = waypoint.get_left_lane()
        right_lane = waypoint.get_right_lane()

        # Check for intersection
        # An intersection is where multiple waypoints converge or diverge.
        if len(waypoint.next(10)) > 1:
            results['is_intersection'] = True

        # Check for merge
        # A merge occurs when there are more lanes behind than ahead.
        if left_lane is not None and right_lane is not None:
            left_count, _ = process_lane(left_lane, 0, 'left')
            right_count, _ = process_lane(right_lane, 0, 'right')
            if (left_count + right_count) < 2:
                results['is_merge'] = True

        # Check for split
        # A split occurs when there are more lanes ahead than behind.
        if waypoint.get_left_lane() is not None or waypoint.get_right_lane() is not None:
            current_left_count, _ = process_lane(waypoint, 0, 'left')
            current_right_count, _ = process_lane(waypoint, 0, 'right')

            next_waypoints = waypoint.next(10)
            if next_waypoints:
                next_wp = next_waypoints[0]
                next_left_count, _ = process_lane(next_wp, 0, 'left')
                next_right_count, _ = process_lane(next_wp, 0, 'right')
                if (next_left_count + next_right_count) > (current_left_count + current_right_count):
                    results['is_split'] = True

        return results

    if waypoint.get_left_lane() is not None and check_pitch(waypoint):
        left_lane_count, left_most_lanetype = process_lane(waypoint.get_left_lane(), left_lane_count, 'left')
    right_lane_count, right_most_lanetype = process_lane(waypoint.get_right_lane(), right_lane_count, 'right')
    
    if world.get_map().name == 'Carla/Maps/Town04':
        if left_lane_count + right_lane_count < 5 and left_lane_count + right_lane_count >= 2:
            left_lane_count+=1
            left_most_lanetype = carla.LaneType.Shoulder
            if left_lane_count + right_lane_count < 5:
                left_lane_count+=1
                
    if world.get_map().name == 'Carla/Maps/Town10HD_Opt':
        if left_lane_count + right_lane_count == 3 and left_most_lanetype == carla.LaneType.Shoulder:
            left_lane_count-=1
            left_most_lanetype = carla.LaneType.Driving

    results  = detect_intersection_merges_splits(waypoint)
    return left_lane_count, left_most_lanetype, right_lane_count, right_most_lanetype, results


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
        
        control = Maneuver_Agent.run_step()
        Maneuver_Agent.vehicle.apply_control(control)
        if Maneuver_Agent.done():
            Maneuver_Agent.vehicle.apply_control(carla.VehicleControl(brake=1, hand_brake=True))
            print("Reached hospital.")
            break
        

# Main CARLA simulation environment
def main():
    parking_mode = False
    park_state = False
    actor_list = []
    
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        client.load_world("Town02")
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
        Map =  world.get_map()
        display = setup_pygame()
        hospitals_locations = spawn_hosbitals(blueprint_library, world, actor_list)
        #actor_list = spawn_traffic(world, spawn_points, blueprint_library) 
        
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        Maneuver_Agent = ManeuverAgent(world, vehicle_bp,actor_list)
            
        destination = random.choice(spawn_points).location
        Maneuver_Agent.set_destination(destination)

            
            # Main simulation loop
        while True:
            Maneuver_Agent.update_spectator()
            if not parking_mode:
                try:
                    control = Maneuver_Agent.run_step()
                    Maneuver_Agent.vehicle.apply_control(control)
                except AttributeError as e:
                    print(f"Error during agent's run_step: {e}")
                    continue

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        parking_mode = True
                        park_state = park_vehicle(Maneuver_Agent, world)
                        
                    elif event.key == pygame.K_e :
                        print('Hospital')
                        move_to_hospital(Maneuver_Agent, hospitals_locations, Map)

                
                pygame.display.flip()
    finally:
        
        for actor in actor_list:
            actor.destroy()
        pygame.quit()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
