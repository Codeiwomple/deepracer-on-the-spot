import numpy as np
from scipy.spatial import KDTree
from shapely.geometry import Point, LineString


class Reward:
    def __init__(self):
        self.verbose = 2  # Verbosity level for output, 0 for none, 1 for lap level, 2 for segment level, 3 for action level.

        # Number of segments/ milestones to split the track into
        self.num_segments = 10
        # Proportion of record steps to give a partial reward 1.1 = within 10% of the segent record
        self.segment_reward_threshold = 1.1

        # Segment variables
        self.previous_segment = None
        self.prev_progress = 0
        self.segment_steps = 0
        self.segment_speeds = []
        self.segment_step_reward = 0
        self.segment_time_reward = 0

        self.segment_step_record = [31, 35, 39, 37, 44, 44, 41, 41, 40, 31]
        self.segment_time_record = [
            17.955008870692957,
            17.302909051584297,
            19.91142706102811,
            17.667287456541203,
            25.799775522097995,
            22.663374979248445,
            24.2728184553661,
            20.15755379526728,
            17.977986139421116,
            18.678387354731765,
        ]

        self.lap_metrics = {
            "current_steps": 0,
            "current_speeds": [],
            "step_record": 451,
            "time_record": 232.49258833503316,
        }

        # Output/ log variables
        self.segment_totals = {
            "step_reward": 0,
            "time_reward": 0,
        }

    def log(self, verbosity, message):
        """Function to log/ print statements depending on verbosity level"""
        if verbosity >= self.verbose:
            print(message)

    def get_current_segment(self, closest_index, waypoints):
        """Function to return the cars current segment"""

        # TODO Find a way to include this calculation in the initialisation function
        # Divide the waypoint as per the number of segments
        waypoints_per_segment = len(waypoints) / self.num_segments
        milestones = [
            int(i * waypoints_per_segment) for i in range(self.num_segments + 1)
        ]

        # Determine current segment using the 'next' milestone
        for i, milestone in enumerate(milestones):
            if closest_index < milestone:
                return i
        return None

    def calculate_segment_time(self):
        # Calculate avg speed per step for segment
        # Average speed of segment
        avg_speed = np.mean(self.segment_speeds)
        time = len(self.segment_speeds) / avg_speed

        return time

    def handle_segment_change(self, current_segment, progress):
        """Function to calculate the reward when the car passes a milestone"""
        reward = 0
        # If the previous segment is None the car has started part way through
        # So no segment reward should be given: set to inf so no record recorded
        if self.previous_segment is None:
            self.log(2, "The car started part way through this segment")

            self.segment_steps = np.inf
            self.reset_lap_metrics()

        # Agent went off track during the segment
        elif self.off_track:
            self.log(2, "The agent went off track in this segment")
            self.segment_steps = np.inf
            self.reset_lap_metrics()

        # Agent was reset in the same segemnt
        # If the car has completed the segemnt in an impossible progress score
        elif progress <= (100 / self.num_segments) - 1:
            self.log(
                2,
                "Impossible record: The agent may have been reset into the same segemnt",
            )

            self.segment_steps = np.inf
            self.reset_lap_metrics()

        # If the car has jumped a segment
        elif current_segment != self.previous_segment + 1:
            if current_segment == (self.previous_segment + 1) % self.num_segments:
                # New lap! TODO
                self.log(2, "Car has started a new lap!")

                self.segment_complete(current_segment)
                self.segment_steps = 0

                self.log(2, "The current segment records are:")
                self.log(2, self.segment_step_record)
            else:
                # Agent has been reset
                self.log(2, "The agent has been reset")
                # Set steps to inf to stop a record for partial completion
                self.segment_steps = np.inf
                self.reset_lap_metrics()

        # Car completed the segemnt properly
        else:
            self.segment_complete(current_segment)
            self.segment_steps = 0

        self.off_track = False

    def segment_complete(self, current_segment):
        """Function handles the proper completion of a segment"""

        self.segment_step_reward = 10  # Default reward for completing segment

        # Partial reward for completion within a certain percentage of the record
        partial_reward_10pc = self.segment_time_record[current_segment - 2] * 1.1
        partial_reward_5pc = self.segment_time_record[current_segment - 2] * 1.05

        print(f"Segment {current_segment} Complete!")
        seg_time = self.calculate_segment_time()

        # Time rewards
        if seg_time <= self.segment_time_record[current_segment - 2]:
            self.segment_time_record[current_segment - 2] = seg_time
            print(f"New time reward for segment {current_segment}")
            print(self.segment_time_record)
            self.segment_time_reward = 100

        elif seg_time <= partial_reward_5pc:
            print(f"Partial time reward for segment (5pc) {current_segment}")
            self.segment_time_reward = 50

        elif seg_time <= partial_reward_10pc:
            print(f"Partial time reward for segment (10pc) {current_segment}")
            self.segment_time_reward = 25

        # Partial reward for completion within a certain percentage of the record
        partial_reward_10pc = self.segment_step_record[current_segment - 2] * 1.1
        partial_reward_5pc = self.segment_step_record[current_segment - 2] * 1.05

        # Step rewards
        if self.segment_steps <= self.segment_step_record[current_segment - 2]:
            # Update the record
            self.segment_step_record[current_segment - 2] = self.segment_steps

            self.log(2, f"New step reward for segment {current_segment}")
            self.log(2, self.segment_step_record)

            self.segment_step_reward = 100

        elif self.segment_steps <= partial_reward_5pc:
            self.log(2, f"Partial step reward for segment (5pc) {current_segment}")
            self.log(2, self.segment_step_record)

            self.segment_step_reward = 50

        elif self.segment_steps <= partial_reward_10pc:
            self.log(2, f"Partial step reward for segment (10pc) {current_segment}")
            self.log(2, self.segment_step_record)

            self.segment_step_reward = 25

    def reset_lap_metrics(self):
        self.lap_metrics["current_steps"] = 0
        self.lap_metrics["current_speeds"] = []

    def reward_function(self, params):
        # DeepRacer input parameters
        x = params["x"]
        y = params["y"]
        heading = params["heading"]
        speed = params["speed"]
        steps = params["steps"]
        progress = params["progress"]
        is_offtrack = params["is_offtrack"]
        waypoints = params["waypoints"]
        closest_index = params["closest_waypoints"][0]

        car_loc = Point(x, y)
        current_segment = self.get_current_segment(closest_index, waypoints)

        self.segment_speeds.append(speed)
        step_reward = 0
        time_reward = 0
        segment_reward = 0
        lap_reward = 0

        if is_offtrack:
            self.off_track = True

        """Segment Complete"""

        if current_segment != self.previous_segment:
            self.handle_segment_change(current_segment, progress)

            step_reward = self.segment_step_reward
            time_reward = self.segment_time_reward

            # Update the log
            self.segment_totals["step_reward"] = step_reward
            self.segment_totals["time_reward"] = time_reward

            self.log(2, "Rewards for this segment: ")
            self.log(2, self.segment_totals)

            print("Lap metrics: ")
            print(self.lap_metrics)

            # Update tracking and logging variables
            for k in self.segment_totals:
                self.segment_totals[k] = 0

            self.segment_step_reward = 0
            self.segment_time_reward = 0
            self.segment_speeds = []

        self.previous_segment = current_segment

        segment_reward = step_reward + time_reward

        """Update the tracking variable at the end of step"""
        self.segment_steps += 1
        self.lap_metrics["current_steps"] += 1
        self.lap_metrics["current_speeds"].append(speed)

        """Lap Complete"""
        if progress == 100:
            lap_reward = 100

            print("100 Progress")

            # Lap step rewards
            partial_reward_10pc = self.lap_metrics["step_record"] * 1.1
            partial_reward_5pc = self.lap_metrics["step_record"] * 1.05

            if self.lap_metrics["current_steps"] <= self.lap_metrics["step_record"]:
                print("Lap completed in record number of steps!!!")
                print(self.lap_metrics["current_steps"])
                self.lap_metrics["step_record"] = self.lap_metrics["current_steps"]

                lap_reward += 500

            elif self.lap_metrics["current_steps"] <= partial_reward_5pc:
                print("Partial lap step reward (5pc)")
                lap_reward += 250
            elif self.lap_metrics["current_steps"] <= partial_reward_10pc:
                print("Partial lap step reward (10pc)")
                lap_reward += 100

            avg_speed = np.mean(self.lap_metrics["current_speeds"])
            time = self.lap_metrics["current_steps"] / avg_speed

            partial_reward_10pc = self.lap_metrics["time_record"] * 1.1
            partial_reward_5pc = self.lap_metrics["time_record"] * 1.05

            if time <= self.lap_metrics["time_record"]:
                print(f"Record Lap Time!!! {time}")
                self.lap_metrics["time_record"] = time
                lap_reward += 500

            elif time <= partial_reward_5pc:
                print(f"Partial lap time reward (5pc)")
                lap_reward += 250
            elif time <= partial_reward_10pc:
                print(f"Partial lap time reward (10pc)")
                lap_reward += 100

            self.reset_lap_metrics()

        if progress < self.prev_progress:
            # Agent restarted
            print(f"Progress was reset")
            self.reset_lap_metrics()

        self.prev_progress = progress

        reward = segment_reward + lap_reward

        if is_offtrack:
            reward = -20

        return float(reward)


reward_object = Reward()


def reward_function(params):
    return reward_object.reward_function(params)
