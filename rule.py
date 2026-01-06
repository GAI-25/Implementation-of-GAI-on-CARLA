import carla
import math
import logging
from experta import *

class CompleteCarlaRuleEngine(KnowledgeEngine):
    def __init__(self, world, vehicle, 
                 fraction=0.8, 
                 red_light_dist=20,
                 lane_offset_threshold=0.8,
                 lane_keep_strength=0.35,
                 safe_margin=0.8,
                 check_distance=10.0,
                 debug=True):
        super().__init__()
        self.world = world
        self.vehicle = vehicle
        self.fraction = fraction
        self.red_light_dist = red_light_dist
        self.lane_offset_threshold = lane_offset_threshold
        self.lane_keep_strength = lane_keep_strength
        self.safe_margin = safe_margin
        self.check_distance = check_distance
        self.debug = debug
        
        self.control = carla.VehicleControl()
        
        self._setup_logging()
        
        self.base_distance_threshold = 6.0
        self.vehicle_size_factor = 3.0
        self.pedestrian_safe_dist = 2.0
        self.detection_radius = 30.0
        self.forward_fov = 60.0
        
        self.reset()
    
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO if self.debug else logging.WARNING,
            format='%(message)s',
            force=True
        )
        logging.getLogger('experta').setLevel(logging.WARNING)
    
    def get_speed_limit(self):
        actors = self.world.get_actors().filter('traffic.speed_limit*')
        vehicle_location = self.vehicle.get_location()
        
        min_dist = float('inf')
        current_limit = 30.0
        
        for actor in actors:
            dist = vehicle_location.distance(actor.get_location())
            if dist < min_dist and dist < 30:
                min_dist = dist
                try:
                    current_limit = float(actor.type_id.split('.')[-1])
                except:
                    continue
        return current_limit
    
    def get_current_speed(self):
        velocity = self.vehicle.get_velocity()
        return 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    
    def get_traffic_light_info(self):
        traffic_light = self.vehicle.get_traffic_light()
        
        if traffic_light is not None:
            state = traffic_light.get_state()
            dist = self.vehicle.get_location().distance(traffic_light.get_location())
            return state, dist
        
        return None, float('inf')
    
    def get_off_lane_info(self):
        map_ = self.world.get_map()
        loc = self.vehicle.get_location()
        
        wp = map_.get_waypoint(loc, project_to_road=False)
        
        is_off_lane = False
        lane_type = None

        if wp is None:
            is_off_lane = True
            lane_type = "None"
        elif wp.lane_type != carla.LaneType.Driving:
            is_off_lane = True
            lane_type = str(wp.lane_type)
        
        return is_off_lane, lane_type
    
    def get_emergency_brake_info(self):
        ego = self.vehicle
        ego_wp = self.world.get_map().get_waypoint(ego.get_location())
        ego_forward = ego.get_transform().get_forward_vector()
        ego_loc = ego.get_location()
        
        bounding_box = ego.bounding_box.extent
        vehicle_size_factor = (bounding_box.x + bounding_box.y) / 2.0
        distance_threshold = self.base_distance_threshold + vehicle_size_factor * self.vehicle_size_factor
        
        min_vehicle_dist = float('inf')
        min_pedestrian_dist = float('inf')
        
        vehicles = self.world.get_actors().filter('vehicle.*')
        for v in vehicles:
            if v.id == ego.id:
                continue
            
            target_loc = v.get_location()
            v_wp = self.world.get_map().get_waypoint(target_loc)
            
            if v_wp.lane_id != ego_wp.lane_id:
                continue
            
            diff = target_loc - ego_loc
            distance = diff.length()
            
            forward_dot = (ego_forward.x * diff.x + 
                          ego_forward.y * diff.y + 
                          ego_forward.z * diff.z)
            if forward_dot > 0 and distance < min_vehicle_dist:
                min_vehicle_dist = distance
        
        pedestrians = self.world.get_actors().filter('walker.pedestrian.*')
        for p in pedestrians:
            target_loc = p.get_location()
            p_wp = self.world.get_map().get_waypoint(target_loc, project_to_road=False)
            
            if p_wp is not None and p_wp.lane_type == carla.LaneType.Sidewalk:
                continue
            
            diff = target_loc - ego_loc
            distance = diff.length()
            
            forward_dot = (ego_forward.x * diff.x + 
                          ego_forward.y * diff.y + 
                          ego_forward.z * diff.z)
            if forward_dot > 0 and distance < min_pedestrian_dist:
                min_pedestrian_dist = distance
        
        return min_vehicle_dist, min_pedestrian_dist, distance_threshold
    
    def get_lane_safety_info(self):
        location = self.vehicle.get_location()
        waypoint = self.world.get_map().get_waypoint(location)
        
        offset = waypoint.transform.location.distance(location)
        
        hit_left, hit_right = False, False
        ego_location = self.vehicle.get_location()
        ego_waypoint = self.world.get_map().get_waypoint(ego_location)
        ego_lane_id = ego_waypoint.lane_id
        
        vehicles = self.world.get_actors().filter('vehicle.*')
        for v in vehicles:
            if v.id == self.vehicle.id:
                continue
            loc = v.get_location()
            wp = self.world.get_map().get_waypoint(loc)
            dist = loc.distance(ego_location)
            if dist < self.check_distance and wp.road_id == ego_waypoint.road_id:
                lane_diff = wp.lane_id - ego_lane_id
                if lane_diff == 1:
                    hit_right = True
                elif lane_diff == -1:
                    hit_left = True
        
        ego_loc = self.vehicle.get_location()
        ego_wp = self.world.get_map().get_waypoint(ego_loc, project_to_road=True, 
                                                   lane_type=carla.LaneType.Driving)
        lateral_offset = ego_wp.transform.get_right_vector().dot(ego_loc - ego_wp.transform.location)
        lane_width_half = ego_wp.lane_width * 0.5
        
        return {
            'lane_offset': offset,
            'lane_center': waypoint.transform.location,
            'hit_left': hit_left,
            'hit_right': hit_right,
            'lateral_offset': lateral_offset,
            'lane_width_half': lane_width_half,
            'ego_steer': self.control.steer
        }
        
    def get_pedestrians_on_sidewalk_in_front(self):
        vehicle_loc = self.vehicle.get_location()
        vehicle_forward = self.vehicle.get_transform().get_forward_vector()
        
        pedestrians = self.world.get_actors().filter('walker.pedestrian.*')
        has_pedestrian = False
        min_distance = float('inf')
        
        for ped in pedestrians:
            ped_loc = ped.get_location()
            
            distance = vehicle_loc.distance(ped_loc)
            if distance > self.detection_radius:
                continue
            
            map_ = self.world.get_map()
            ped_wp = map_.get_waypoint(ped_loc, project_to_road=False)
            
            if ped_wp is None or ped_wp.lane_type != carla.LaneType.Sidewalk:
                continue
            
            relative_vector = ped_loc - vehicle_loc
            relative_length = relative_vector.length()
            
            if relative_length > 0:
                relative_vector_normalized = relative_vector / relative_length
                forward_dot = (vehicle_forward.x * relative_vector_normalized.x + 
                              vehicle_forward.y * relative_vector_normalized.y + 
                              vehicle_forward.z * relative_vector_normalized.z)

                angle = math.degrees(math.acos(max(min(forward_dot, 1.0), -1.0)))

                if angle <= self.forward_fov / 2:
                    has_pedestrian = True
                    if distance < min_distance:
                        min_distance = distance
        
        return has_pedestrian, min_distance
    
    def update_and_run(self, control):
        self.control = control
        
        speed = self.get_current_speed()
        speed_limit = self.get_speed_limit() * self.fraction
        light_state, light_dist = self.get_traffic_light_info()
        is_off_lane, lane_type = self.get_off_lane_info()
        min_vehicle_dist, min_pedestrian_dist, distance_threshold = self.get_emergency_brake_info()
        lane_info = self.get_lane_safety_info()
        has_pedestrian, min_distance = self.get_pedestrians_on_sidewalk_in_front()
        
        self.reset()
        facts = {
            'speed': speed,
            'speed_limit': speed_limit,
            'light_state': light_state,
            'light_distance': light_dist if light_state else float('inf'),
            'is_off_lane': is_off_lane,
            'off_lane_type': lane_type,
            'min_vehicle_dist': min_vehicle_dist,
            'min_pedestrian_dist': min_pedestrian_dist,
            'distance_threshold': distance_threshold,
            'lane_offset': lane_info['lane_offset'],
            'hit_left': lane_info['hit_left'],
            'hit_right': lane_info['hit_right'],
            'lateral_offset': lane_info['lateral_offset'],
            'lane_width_half': lane_info['lane_width_half'],
            'raw_throttle': control.throttle,
            'raw_brake': control.brake,
            'raw_steer': control.steer,
            'has_sidewalk_pedestrian': has_pedestrian,
            'pedestrian_distance': min_distance
        }
        
        self.declare(Fact(**facts))

        self.run()
        
        return self.control
  
    
    @Rule(Fact(speed=MATCH.s, speed_limit=MATCH.limit), salience=200)
    def speed_control_rule(self, s, limit):
        if s > limit + 2:
            self.control.throttle = 0.0
            self.control.brake = 0.3
            print(f'limit speed: {limit:.1f} km/h - control speed rule 1')
    
    
    @Rule(Fact(min_pedestrian_dist=MATCH.ped_dist), salience=190)
    def emergency_brake_pedestrian_rule(self, ped_dist):
        if ped_dist < self.pedestrian_safe_dist:
            self.control.throttle = 0.0
            self.control.brake = 1.0
            print(f"[Safety] Emergency brake for PEDESTRIAN at {ped_dist:.2f} m")
          
            
    @Rule(Fact(has_sidewalk_pedestrian=True))
    def rule_simple_brake(self):
        self.control.throttle = 0.0
        self.control.brake = 1.0
        print(f"Emergency brake for PEDESTRIAN at {dist:.1f} m")
    
    
    @Rule(Fact(
        min_vehicle_dist=MATCH.veh_dist,
        distance_threshold=MATCH.threshold
    ), salience=180)
    def emergency_brake_vehicle_rule(self, veh_dist, threshold):
        if veh_dist < threshold:
            self.control.throttle = 0.0
            self.control.brake = 1.0
            print(f"[Safety] Emergency brake SAME LANE VEHICLE at {veh_dist:.2f} m")


    @Rule(Fact(
        lateral_offset=MATCH.lateral,
        lane_width_half=MATCH.width_half
    ), salience=170)
    def avoid_side_collision_rule(self, lateral, width_half):
        if abs(lateral) > width_half * self.safe_margin:
            correction = -math.copysign(self.lane_keep_strength, lateral)
            self.control.steer = self.control.steer * 0.3 + correction * 0.7
            print(f"[Safety] Side boundary correction steer={self.control.steer:.3f}")
          
            
    @Rule(Fact(
        hit_left=MATCH.left,
        hit_right=MATCH.right,
        raw_steer=MATCH.steer_val
    ), salience=160)
    def lane_occupied_control_rule(self, left, right, steer_val):
        if left and steer_val < 0:
            self.control.steer = min(steer_val, 0.0)
            print("[Safety] Left lane occupied → disable left turn")
        
        if right and steer_val > 0:
            self.control.steer = max(steer_val, 0.0)
            print("[Safety] Right lane occupied → disable right turn")
    
    
    @Rule(Fact(lane_offset=MATCH.offset), salience=150)
    def lane_safety_recenter_rule(self, offset):
        if offset > self.lane_offset_threshold:
            self.control.steer *= 0.5
            print(f"[Safety] Lane boundary near, soft recentering (offset: {offset:.2f}m)")
    
    
    @Rule(Fact(is_off_lane=True, off_lane_type=MATCH.lane_type), salience=140)
    def brake_if_off_lane_rule(self, lane_type):
        self.control.throttle = 0.0
        self.control.brake = 1.0
        
        if lane_type == "None":
            print("[Safety] Vehicle OFF ROAD — FULL BRAKE")
        else:
            print(f"[Safety] Off Driving Lane ({lane_type}) — FULL BRAKE")
    
    
    @Rule(Fact(
        light_state=carla.TrafficLightState.Red,
        light_distance=MATCH.dist
    ), salience=100)
    def red_light_stop_rule(self, dist):
        if dist < self.red_light_dist:
            self.control.throttle = 0.0
            self.control.brake = 1.0
            print(f'red light stop rule (distance: {dist:.1f}m)')
    

    @Rule(Fact(
        raw_throttle=MATCH.throttle,
        raw_brake=MATCH.brake,
        raw_steer=MATCH.steer
    ), salience=0)
    def clip_control_rule(self, throttle, brake, steer):
        self.control.throttle = max(0.0, min(self.control.throttle, 1.0))
        self.control.brake = max(0.0, min(self.control.brake, 1.0))
        self.control.steer = max(-1.0, min(self.control.steer, 0.8))
        
        if self.control.throttle > 0.1 and self.control.brake > 0.1:
            self.control.throttle = 0.0
            print('clip_control rule 1')
        
        if self.control.throttle < 0:
            self.control.throttle = 0.0
            print('clip_control rule 2')
        
        if self.control.brake < 0:
            self.control.brake = 0.0
            print('clip_control rule 3')