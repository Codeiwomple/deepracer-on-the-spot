import numpy as np
from scipy.spatial import KDTree
from shapely.geometry import Point, LineString


class Reward:
    def __init__(self, race_line, inner_border, outer_border):
        self.verbose = 2  # Verbosity level for output, 0 for none, 1 for lap level, 2 for segment level, 3 for action level.

        # Reward weightings
        self.heading_weight = 2
        self.segment_step_reward_weight = 1
        self.partial_segment_reward_weight = (
            0.5  # Proportion of segment reward for getting close to the record
        )
        self.smoothness_weight = 1

        # Configurations
        # Number of segments/ milestones to split the track into
        self.num_segments = 10
        # Size of smooth driving buffer. Larger will encourage smoother steering but car may not be very responsive
        self.smoothing_buffer_size = 3
        # Proportion of record steps to give a partial reward 1.1 = within 10% of the segent record
        self.segment_reward_threshold = 1.1
        # Cutoff/ max diff/ threshold for heading reward e.g. value of 10 will mean heading has to be within 10 deg for a reward
        self.heading_threshold = 60
        # The endpoint of the gradient: The reward gradient will be calculated off this value instead of the threshold.
        # It simply means the car wont get close to 0 for being near the threshold. This value need to be bigger than the threshold.
        self.heading_gradient = 60
        # Range of the steering angles as defined in the action space
        self.steering_angle_range = 60
        # Number of steps before checking for step_progress_reward. 50 would check progress after every 50 steps.
        self.progress_steps = 30

        self.segment_time_reward = 0

        # Track geometries
        self.race_line = race_line
        self.race_line_tree = KDTree(race_line)

        # Segment variables
        self.previous_segment = None
        self.segment_steps = 0
        self.segemnt_speeds = []
        self.segment_step_record = [
            31,
            35,
            39,
            38,
            44,
            45,
            41,
            41,
            40,
            31,
        ]  # [np.inf] * self.num_segments  # Update this from the logs after training. Ensure the size matches number of segments. Or guess or use np.inf to start
        self.segment_time_record = [np.inf] * self.num_segments
        self.segment_reward = 0

        # Action space variables
        self.max_speed = 4
        self.min_speed = 1.3
        self.max_steering_angle = 30

        self.off_track = False

        # Buffer for smooth steering
        self.smoothing_buffer = [0] * self.smoothing_buffer_size
        self.buffer_index = 0

        # Output/ log variables
        self.segment_totals = {
            "HR": 0,
            "SSR": 0,
            "SMR": 0,
            "STR": 0
            "Total": 0,
        }

        self.segment_speeds = []

    def log(self, verbosity, message):
        """Function to log/ print statements depending on verbosity level"""
        if verbosity >= self.verbose:
            print(message)

    def buffer_add(self, value):
        """Function to add an item to the buffer. It maintains a constant size by incrimenting the index as each item is added"""
        self.smoothing_buffer[self.buffer_index] = value
        self.buffer_index = (self.buffer_index + 1) % self.smoothing_buffer_size

    def buffer_range(self):
        """Function returns the range of values in the buffer"""
        return max(self.smoothing_buffer) - min(self.smoothing_buffer)

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

        print(f"Speeds: {self.segment_speeds}")
        print(f"Avg: {avg_speed}")
        print(f"Steps (len) {len(self.segment_speeds)}")
        print(f"Time: {time}")

        return time

    def handle_segment_change(self, current_segment, progress):
        """Function to calculate the reward when the car passes a milestone"""
        reward = 0
        # If the previous segment is None the car has started part way through
        # So no segment reward should be given: set to inf so no record recorded
        if self.previous_segment is None:
            self.log(2, "The car started part way through this segment")

            self.segment_steps = np.inf

        # Agent went off track during the segment
        elif self.off_track:
            self.log(2, "The agent went off track in this segment")
            self.segment_steps = np.inf

        # Agent was reset in the same segemnt
        # If the car has completed the segemnt in an impossible progress score
        elif progress <= (100 / self.num_segments) - 1:
            self.log(
                2,
                "Impossible record: The agent may have been reset into the same segemnt",
            )

            self.segment_steps = np.inf

        # If the car has jumped a segment
        elif current_segment != self.previous_segment + 1:
            if current_segment == (self.previous_segment + 1) % self.num_segments:
                # New lap! TODO
                self.log(2, "Car has started a new lap!")

                reward = self.segment_complete(current_segment)
                self.segment_steps = 0

                self.log(2, "The current segment records are:")
                self.log(2, self.segment_step_record)
            else:
                # Agent has been reset
                self.log(2, "The agent has been reset")
                # Set steps to inf to stop a record for partial completion
                self.segment_steps = np.inf

        # Car completed the segemnt properly
        else:
            reward = self.segment_complete(current_segment)
            self.segment_steps = 0

        self.off_track = False
        return reward

    def segment_complete(self, current_segment):
        """Function handles the proper completion of a segment"""

        # Partial reward for completion within a certain percentage of the record
        partial_reward = (
            self.segment_time_record[current_segment - 2]
            * self.segment_reward_threshold
        )

        print(f"Segment {current_segment} Complete!")
        seg_time = self.calculate_segment_time()
        if seg_time <= self.segment_time_record[current_segment - 2]:
            self.segment_time_record[current_segment - 2] = seg_time
            print("NEW SEGMENT TIME RECORD")
            print(self.segment_time_record)
            self.segment_time_reward = 50

        if seg_time <= partial_reward:
            print("Partial time reward")
            self.segment_time_reward = 25

        # If car has performed a record segment
        if self.segment_steps <= self.segment_step_record[current_segment - 2]:
            # Update the record
            self.segment_step_record[current_segment - 2] = self.segment_steps

            self.log(2, "New record for segment")
            self.log(2, self.segment_step_record)

            return self.segment_reward

        # Partial reward for completion within a certain percentage of the record
        partial_reward = (
            self.segment_step_record[current_segment - 2]
            * self.segment_reward_threshold
        )

        if self.segment_steps <= partial_reward:
            self.log(2, "Partial segment reward")
            self.log(2, self.segment_step_record)

            return self.segment_reward * self.partial_segment_reward_weight

        return 0

    def heading_reward(self, direction, heading):
        """
        Function calculates the reward for the car based on its heading compared to the heading of the racing line.
        This is based on a linear gradient defined in the configuration variables in the init function
        """
        # Calculate the difference between the track direction and the heading direction of the car
        direction_diff = abs(direction - heading)
        if direction_diff > 180:
            direction_diff = 360 - direction_diff

        # Reward the car based on how close it is to the correct heading. If it doesnt meet the threshold, give it nothing.
        if direction_diff >= self.heading_threshold:
            heading_reward = 0
        else:
            # If the car is within the threshold for a reward, allocate based on the gradient variable
            heading_reward = (
                self.heading_gradient - direction_diff
            ) / self.heading_gradient

        return heading_reward

    def opposite_heading(self, heading):
        """Function return the opposite of the given heading"""
        if heading > 0:
            return heading - 180
        else:
            return heading + 180

        return 0

    def calculate_heading_reward(self, car_loc, heading):
        """
        Function calculates the heading reward.
        As it uses the closes 2 waypoints to calculate the heading of the racingline it is possible that one of the points is behind the car.
        In this case the heading may be 180deg out. It is fair to assume the car will not be heading 180deg in the wrong direction
        so the reward for both headings is calculated and the max retuned.
        """

        """Instead of using the direction of the RL, use the direction to the RL. Specifically 2 points ahead of car current loc..."""
        closest_points_i = self.race_line_tree.query([car_loc.x, car_loc.y], k=1)[1]
        cp = race_line[(closest_points_i + 2) % len(race_line)]

        # Calculate the direction in radians, arctan2(dy, dx), the result is (-pi, pi) in radians
        direction = np.arctan2(cp[1] - car_loc.y, cp[0] - car_loc.x)

        # Convert to degree
        direction = np.degrees(direction)

        # Calculate the reward for the current and opposite headings
        # This will ensure the reward is track direction agnostic if the clostest 2 race line point are in the wrong direction to race
        # Safe to assume the car wont be heading in the oppsoite direction for any significant ammount of time
        heading_reward = self.heading_reward(direction, heading)
        opposite_heading_reward = self.heading_reward(
            direction, self.opposite_heading(heading)
        )

        return direction, max(heading_reward, opposite_heading_reward)

    def calculate_smoothness_reward(self, steering_angle):
        """
        Function calculates a reward based on the range of the last n steering angles.
        The smaller the range, the larger the reward. This will discourage the car from zigzagging
        and steer more efficiently
        """
        # Add to buffer
        self.buffer_add(steering_angle)

        return 1 - (self.buffer_range() / self.steering_angle_range)

    def reward_function(self, params):
        # DeepRacer input parameters
        x = params["x"]
        y = params["y"]
        track_width = params["track_width"]
        heading = params["heading"]
        progress = params["progress"]
        speed = params["speed"]
        steps = params["steps"]
        steering_angle = params["steering_angle"]
        is_offtrack = params["is_offtrack"]
        waypoints = params["waypoints"]
        closest_index = params["closest_waypoints"][0]

        self.segment_speeds.append(speed)

        # Car location
        car_loc = Point(x, y)

        # Maintain segment level off-track variable
        if is_offtrack:
            self.off_track = True

        """Action level rewards"""
        heading_reward = self.calculate_heading_reward(car_loc, heading)[1]
        smoothness_reward = self.calculate_smoothness_reward(steering_angle)

        """Segment level rewards"""
        # Calculate which segment of the track the car is currently in
        current_segment = self.get_current_segment(closest_index, waypoints)

        step_reward = 0

        # If the car has changed segment
        if current_segment != self.previous_segment:
            step_reward = self.handle_segment_change(current_segment, progress)

            # Update the log
            self.segment_totals["SSR"] = step_reward * self.segment_step_reward_weight

            self.log(2, "Rewards for this segment: ")
            self.log(2, self.segment_totals)

            # Update tracking and logging variables
            for k in self.segment_totals:
                self.segment_totals[k] = 0

            self.segment_reward = 0
            self.segment_speeds = []

        self.previous_segment = current_segment

        """Calculate the reward for this step"""

        # Heading reward: reward the car for deading in the direction of the racing line
        HR = self.heading_weight * heading_reward
        # Segment step reward: Reward for completeing each segment in a minimum number of steps
        SSR = self.segment_step_reward_weight * step_reward
        # Smoothness reward: reward the car for driving smoothly and not changing direction irratically
        SMR = self.smoothness_weight * smoothness_reward

        STR = self.segment_time_reward

        reward = HR + SSR + SMR

        # Update the logging variables
        self.segment_totals["HR"] += HR
        self.segment_totals["SMR"] += SMR
        self.segment_totals["STR"] += STR
        self.segment_totals["Total"] += reward

        """Update the tracking variable at the end of step"""
        self.segment_steps += 1
        self.segment_reward += reward
        self.segment_time_reward = 0

        if is_offtrack:
            reward = -5

        if progress == 100:
            reward += 100

        return float(reward)


## Copy and past variables from race line calc
race_line = np.array(
    [
        [-0.12527623, -3.1533136],
        [0.02440991, -3.24644196],
        [0.13473509, -3.31520499],
        [0.28019151, -3.40586448],
        [0.53597136, -3.56528997],
        [0.79175642, -3.72470641],
        [1.0475527, -3.88410199],
        [1.30336004, -4.04347742],
        [1.5591625, -4.20286155],
        [1.81490648, -4.36235297],
        [2.07058543, -4.52196288],
        [2.32639319, -4.68133747],
        [2.58210292, -4.84089117],
        [2.8379686, -4.99958508],
        [3.094709, -5.15481916],
        [3.35289132, -5.30399479],
        [3.61292166, -5.44455683],
        [3.87502634, -5.57407016],
        [4.13925362, -5.69023872],
        [4.40547461, -5.79093236],
        [4.67338798, -5.87419053],
        [4.94252656, -5.93811591],
        [5.21223889, -5.98084729],
        [5.48165693, -6.00053669],
        [5.74963306, -5.99523813],
        [6.01461416, -5.96286526],
        [6.2753743, -5.90517536],
        [6.53081578, -5.82336824],
        [6.77988006, -5.71835568],
        [7.02150023, -5.5909056],
        [7.25457669, -5.44174195],
        [7.47794791, -5.27159596],
        [7.69037758, -5.08129291],
        [7.89054307, -4.87183604],
        [8.07701828, -4.64449651],
        [8.24831367, -4.40099011],
        [8.4029343, -4.1436369],
        [8.53959533, -3.87516931],
        [8.65724951, -3.59895544],
        [8.75409566, -3.3180238],
        [8.82884064, -3.03507321],
        [8.88083123, -2.7521989],
        [8.90973355, -2.47110337],
        [8.91540024, -2.19323561],
        [8.89772203, -1.91989383],
        [8.8565174, -1.65231683],
        [8.79140394, -1.39179087],
        [8.70183033, -1.1397262],
        [8.5869529, -0.8978125],
        [8.45062284, -0.66563957],
        [8.29579227, -0.44273569],
        [8.12536993, -0.22843998],
        [7.94234577, -0.0218986],
        [7.74977618, 0.17801476],
        [7.55062949, 0.37275895],
        [7.34765856, 0.564144],
        [7.14873985, 0.7493342],
        [6.95787351, 0.93931933],
        [6.78216355, 1.1381704],
        [6.62823648, 1.34932316],
        [6.50275695, 1.57556628],
        [6.41328107, 1.81888878],
        [6.35291698, 2.07433341],
        [6.31548834, 2.33847026],
        [6.29412741, 2.6083363],
        [6.28127255, 2.88113707],
        [6.26531792, 3.15756448],
        [6.24042862, 3.4309714],
        [6.20113178, 3.69806842],
        [6.14302328, 3.95498046],
        [6.06295856, 4.1978227],
        [5.95894516, 4.42291664],
        [5.82982176, 4.62656067],
        [5.67507751, 4.804672],
        [5.49483162, 4.95237046],
        [5.29003785, 5.06339195],
        [5.06339841, 5.12925905],
        [4.82407107, 5.15605111],
        [4.57662486, 5.14317459],
        [4.32558008, 5.08998594],
        [4.07572991, 4.99603664],
        [3.83233406, 4.86166522],
        [3.6012028, 4.68851783],
        [3.38855001, 4.48065446],
        [3.20050274, 4.24566553],
        [3.04379742, 3.99472452],
        [2.92127365, 3.73615557],
        [2.83200739, 3.47407472],
        [2.77453878, 3.21103353],
        [2.74783864, 2.94884284],
        [2.75137727, 2.68899062],
        [2.78518998, 2.43288092],
        [2.8538105, 2.18289516],
        [2.95261805, 1.9398036],
        [3.07743961, 1.70372083],
        [3.22413611, 1.47423599],
        [3.38827513, 1.25043782],
        [3.56496717, 1.03095051],
        [3.74894941, 0.81404784],
        [3.92768357, 0.59653445],
        [4.09033132, 0.37551606],
        [4.2272036, 0.15010956],
        [4.33083284, -0.07878787],
        [4.39599877, -0.30861393],
        [4.41909345, -0.53539798],
        [4.3975918, -0.75384401],
        [4.32951605, -0.95707208],
        [4.21318059, -1.13572812],
        [4.04824595, -1.27597729],
        [3.85194567, -1.38018053],
        [3.63062237, -1.44601003],
        [3.39014224, -1.47194498],
        [3.13618975, -1.45748532],
        [2.87433572, -1.40374869],
        [2.60969102, -1.31407009],
        [2.34622988, -1.19424482],
        [2.08609544, -1.05208366],
        [1.82925349, -0.89654608],
        [1.57377015, -0.73666516],
        [1.31845481, -0.57653313],
        [1.06342844, -0.41596917],
        [0.8088374, -0.25475447],
        [0.55483032, -0.09266677],
        [0.30155709, 0.07051807],
        [0.04916847, 0.23502547],
        [-0.20218341, 0.40108273],
        [-0.45234294, 0.56892178],
        [-0.70115122, 0.73877785],
        [-0.94844002, 0.91089358],
        [-1.19401934, 1.08551164],
        [-1.43759941, 1.26282489],
        [-1.6787059, 1.44330812],
        [-1.91307896, 1.63056601],
        [-2.13110747, 1.82836672],
        [-2.3228072, 2.03731055],
        [-2.48065704, 2.25597194],
        [-2.59996936, 2.48174726],
        [-2.67804096, 2.71150162],
        [-2.71285076, 2.94181256],
        [-2.70210251, 3.16876182],
        [-2.64003434, 3.38653372],
        [-2.533707, 3.59141404],
        [-2.38832686, 3.78106325],
        [-2.20854589, 3.95420762],
        [-1.99934008, 4.11085735],
        [-1.76595137, 4.2531955],
        [-1.56255566, 4.41617996],
        [-1.39593819, 4.59798132],
        [-1.27068712, 4.79494071],
        [-1.19005781, 5.00249594],
        [-1.15714543, 5.21539864],
        [-1.17572402, 5.42717263],
        [-1.25103186, 5.62842815],
        [-1.370755, 5.81408861],
        [-1.53107097, 5.97961179],
        [-1.72834668, 6.12099614],
        [-1.95847474, 6.2348867],
        [-2.21612022, 6.31935312],
        [-2.49463595, 6.37482792],
        [-2.7866988, 6.40481421],
        [-3.08562632, 6.4157888],
        [-3.38679743, 6.41631508],
        [-3.6881907, 6.41572797],
        [-3.98932902, 6.41441355],
        [-4.28980541, 6.41051731],
        [-4.58908542, 6.40222533],
        [-4.88654435, 6.38778697],
        [-5.18147544, 6.36553357],
        [-5.47310259, 6.33389846],
        [-5.76059546, 6.29144044],
        [-6.04307976, 6.23685272],
        [-6.31966332, 6.16900583],
        [-6.5894756, 6.08700441],
        [-6.85165393, 5.99012462],
        [-7.10533518, 5.8777818],
        [-7.34959747, 5.74942945],
        [-7.5836199, 5.60482219],
        [-7.80640075, 5.4435709],
        [-8.01689002, 5.26540242],
        [-8.21384237, 5.07001584],
        [-8.39572039, 4.85709722],
        [-8.56061589, 4.62644754],
        [-8.70606219, 4.37817197],
        [-8.82894276, 4.11327574],
        [-8.92553455, 3.83467449],
        [-8.99212848, 3.54818379],
        [-9.02653425, 3.26125912],
        [-9.02862162, 2.9798899],
        [-8.99938847, 2.70786792],
        [-8.93998014, 2.44772964],
        [-8.85129246, 2.2015542],
        [-8.7339257, 1.97135345],
        [-8.58789543, 1.75952401],
        [-8.41274157, 1.56916874],
        [-8.21407507, 1.39840511],
        [-7.99591324, 1.24541331],
        [-7.76197407, 1.10783656],
        [-7.51598203, 0.98267052],
        [-7.2617309, 0.86632787],
        [-7.0031715, 0.75468988],
        [-6.73210386, 0.63359108],
        [-6.46238108, 0.50908628],
        [-6.19394995, 0.38131784],
        [-5.92675918, 0.25042162],
        [-5.66075907, 0.11652827],
        [-5.39590097, -0.02023511],
        [-5.13213265, -0.15973322],
        [-4.8694023, -0.30183243],
        [-4.60764667, -0.44637114],
        [-4.34679392, -0.59316689],
        [-4.08677687, -0.74204921],
        [-3.82751088, -0.89280339],
        [-3.56891508, -1.0452189],
        [-3.31090117, -1.19904805],
        [-3.0534002, -1.35403932],
        [-2.79636803, -1.50989646],
        [-2.53975528, -1.66625683],
        [-2.28356726, -1.822825],
        [-2.02792001, -1.97941469],
        [-1.77318654, -2.13575554],
        [-1.51926004, -2.29177911],
        [-1.26708036, -2.44699557],
        [-1.01785228, -2.60065923],
        [-0.7737104, -2.75144674],
        [-0.53844994, -2.89700275],
        [-0.31867121, -3.03322406],
        [-0.12527623, -3.1533136],
    ]
)
outer_boarder = np.array(
    [
        [-3.68053126e-01, -3.63035529e00],
        [-2.57727087e-01, -3.69911695e00],
        [-1.47401921e-01, -3.76788000e00],
        [-1.95116096e-03, -3.85853601e00],
        [2.53834099e-01, -4.01796484e00],
        [5.09644806e-01, -4.17739677e00],
        [7.65475392e-01, -4.33681393e00],
        [1.02129304e00, -4.49619579e00],
        [1.27699602e00, -4.65551805e00],
        [1.53254700e00, -4.81488895e00],
        [1.78804195e00, -4.97438383e00],
        [2.04396009e00, -5.13427496e00],
        [2.30079007e00, -5.29414082e00],
        [2.55820704e00, -5.45344877e00],
        [2.81470895e00, -5.61127377e00],
        [3.06684303e00, -5.76747322e00],
        [3.31418610e00, -5.92580986e00],
        [3.56101894e00, -6.08912086e00],
        [3.82542610e00, -6.26652193e00],
        [4.11856604e00, -6.43571186e00],
        [4.40839100e00, -6.57659912e00],
        [4.69329309e00, -6.71229219e00],
        [5.02673101e00, -6.85041523e00],
        [5.41120386e00, -6.93608522e00],
        [5.81020308e00, -6.95463419e00],
        [6.20096111e00, -6.89607906e00],
        [6.56121302e00, -6.78289604e00],
        [6.89623499e00, -6.62559891e00],
        [7.20249605e00, -6.43204308e00],
        [7.47425699e00, -6.21789598e00],
        [7.73169279e00, -5.98560715e00],
        [7.95581722e00, -5.73728895e00],
        [8.17048740e00, -5.48877096e00],
        [8.36558247e00, -5.22188711e00],
        [8.54966545e00, -4.95493603e00],
        [8.71820927e00, -4.67622519e00],
        [8.87621784e00, -4.39562798e00],
        [9.02134705e00, -4.10662603e00],
        [9.15531254e00, -3.81496406e00],
        [9.27873898e00, -3.51549697e00],
        [9.38785458e00, -3.21326494e00],
        [9.48920536e00, -2.90203094e00],
        [9.56993484e00, -2.58554411e00],
        [9.64210320e00, -2.26113391e00],
        [9.68724632e00, -1.92446899e00],
        [9.71319294e00, -1.57501197e00],
        [9.69250870e00, -1.19616795e00],
        [9.60158062e00, -7.95500696e-01],
        [9.42020988e00, -4.29894805e-01],
        [9.19977570e00, -1.21217802e-01],
        [8.94366741e00, 1.48547903e-01],
        [8.66300583e00, 3.78196597e-01],
        [8.37007904e00, 5.70723295e-01],
        [8.07631302e00, 7.31566727e-01],
        [7.79066896e00, 8.71411502e-01],
        [7.51322889e00, 1.00109899e00],
        [7.25383282e00, 1.12192798e00],
        [7.02050114e00, 1.24747205e00],
        [6.82309008e00, 1.37305903e00],
        [6.68053293e00, 1.50264800e00],
        [6.58681202e00, 1.63980806e00],
        [6.53631592e00, 1.79569697e00],
        [6.52256107e00, 1.95913506e00],
        [6.56724882e00, 2.16439295e00],
        [6.64084196e00, 2.43580198e00],
        [6.72045898e00, 2.72521996e00],
        [6.80450201e00, 3.03200889e00],
        [6.87325621e00, 3.36171508e00],
        [6.91973209e00, 3.70935702e00],
        [6.92645311e00, 4.07928085e00],
        [6.87769079e00, 4.46334219e00],
        [6.75621223e00, 4.84785223e00],
        [6.55821419e00, 5.20336580e00],
        [6.30164385e00, 5.50704479e00],
        [5.99628592e00, 5.76274395e00],
        [5.61757803e00, 5.96727419e00],
        [5.17274380e00, 6.06219196e00],
        [4.73771381e00, 6.02995396e00],
        [4.35028505e00, 5.91127205e00],
        [4.01155281e00, 5.73974180e00],
        [3.71134806e00, 5.53678608e00],
        [3.44410896e00, 5.31201410e00],
        [3.20326996e00, 5.07654095e00],
        [2.98025608e00, 4.83073378e00],
        [2.76889801e00, 4.56963396e00],
        [2.58272004e00, 4.30192518e00],
        [2.40294409e00, 4.03196621e00],
        [2.23452401e00, 3.73761892e00],
        [2.08755708e00, 3.43099689e00],
        [1.96146798e00, 3.10994601e00],
        [1.85764098e00, 2.74516201e00],
        [1.82836902e00, 2.28433609e00],
        [1.98039901e00, 1.81912303e00],
        [2.22617006e00, 1.47830999e00],
        [2.47222495e00, 1.22508001e00],
        [2.71115088e00, 1.00718999e00],
        [2.94401193e00, 8.05365205e-01],
        [3.17197704e00, 6.08047009e-01],
        [3.39975095e00, 4.10840690e-01],
        [3.62514305e00, 2.15593904e-01],
        [3.83076191e00, 3.39083485e-02],
        [4.00585985e00, -1.51811898e-01],
        [4.15348101e00, -3.35260004e-01],
        [4.24856520e00, -5.00228286e-01],
        [4.28837204e00, -6.40144527e-01],
        [4.27510309e00, -7.84991205e-01],
        [4.22484493e00, -9.36013877e-01],
        [4.14302683e00, -1.05669796e00],
        [4.02946806e00, -1.14459205e00],
        [3.88111711e00, -1.20200801e00],
        [3.70311093e00, -1.22451103e00],
        [3.50612593e00, -1.20851696e00],
        [3.30810905e00, -1.15713704e00],
        [3.10486388e00, -1.05997205e00],
        [2.87469411e00, -9.21564400e-01],
        [2.62379503e00, -7.63668299e-01],
        [2.36730909e00, -6.03686392e-01],
        [2.11176896e00, -4.44107205e-01],
        [1.85609698e00, -2.84471095e-01],
        [1.60043502e00, -1.24843299e-01],
        [1.34476399e00, 3.47820111e-02],
        [1.08907902e00, 1.94405496e-01],
        [8.33430111e-01, 3.53983790e-01],
        [5.77388823e-01, 5.13869703e-01],
        [3.24244708e-01, 6.71252191e-01],
        [8.55606422e-02, 8.24279130e-01],
        [-1.48309693e-01, 9.92498577e-01],
        [-3.93986702e-01, 1.16814995e00],
        [-6.39197707e-01, 1.34343696e00],
        [-8.84403884e-01, 1.51871800e00],
        [-1.12976801e00, 1.69409204e00],
        [-1.36514699e00, 1.86190403e00],
        [-1.58308697e00, 2.03347206e00],
        [-1.80320597e00, 2.22239804e00],
        [-2.01919794e00, 2.40741205e00],
        [-2.21501398e00, 2.59850407e00],
        [-2.39407897e00, 2.78631401e00],
        [-2.52639198e00, 2.96096992e00],
        [-2.59563804e00, 3.08409309e00],
        [-2.59708500e00, 3.17830300e00],
        [-2.56177902e00, 3.28315711e00],
        [-2.48526311e00, 3.37716198e00],
        [-2.34754801e00, 3.46233010e00],
        [-2.13393593e00, 3.54479098e00],
        [-1.86433697e00, 3.63065195e00],
        [-1.56859004e00, 3.72530890e00],
        [-1.24187005e00, 3.84353495e00],
        [-8.85666788e-01, 4.03361607e00],
        [-5.54301322e-01, 4.33397198e00],
        [-3.17104399e-01, 4.73831606e00],
        [-2.24081695e-01, 5.16992092e00],
        [-2.28367701e-01, 5.61410999e00],
        [-3.90636712e-01, 6.07388783e00],
        [-7.08950520e-01, 6.44505596e00],
        [-1.08017898e00, 6.66378784e00],
        [-1.43800497e00, 6.80054188e00],
        [-1.79711294e00, 6.87769413e00],
        [-2.13312602e00, 6.91753817e00],
        [-2.45208406e00, 6.93951797e00],
        [-2.77351904e00, 6.95481396e00],
        [-3.09001589e00, 6.95226622e00],
        [-3.39012194e00, 6.94970512e00],
        [-3.68911409e00, 6.94852781e00],
        [-3.98937798e00, 6.94866896e00],
        [-4.29082918e00, 6.94876194e00],
        [-4.59223795e00, 6.94884014e00],
        [-4.89360094e00, 6.94891787e00],
        [-5.19484901e00, 6.94903278e00],
        [-5.50444889e00, 6.94922209e00],
        [-5.82302284e00, 6.93953085e00],
        [-6.13703012e00, 6.91923809e00],
        [-6.45354986e00, 6.89370489e00],
        [-6.78037977e00, 6.85233879e00],
        [-7.10893297e00, 6.79064608e00],
        [-7.44267988e00, 6.70504713e00],
        [-7.78967619e00, 6.57822418e00],
        [-8.15595818e00, 6.36876106e00],
        [-8.44700146e00, 6.07251501e00],
        [-8.65973282e00, 5.78200483e00],
        [-8.83617115e00, 5.49078083e00],
        [-8.98964214e00, 5.19805908e00],
        [-9.12648392e00, 4.90312719e00],
        [-9.25098038e00, 4.60294390e00],
        [-9.36123657e00, 4.29697514e00],
        [-9.45790005e00, 3.98879194e00],
        [-9.54247475e00, 3.67649388e00],
        [-9.61382484e00, 3.35779190e00],
        [-9.67077637e00, 3.02281308e00],
        [-9.69693184e00, 2.68605995e00],
        [-9.70755291e00, 2.34293795e00],
        [-9.67522717e00, 1.96704602e00],
        [-9.57044888e00, 1.56964898e00],
        [-9.34760094e00, 1.16813302e00],
        [-9.01883316e00, 8.72172177e-01],
        [-8.69456768e00, 6.77367687e-01],
        [-8.36755276e00, 5.27595401e-01],
        [-8.02625275e00, 4.15253311e-01],
        [-7.68873310e00, 3.42162699e-01],
        [-7.37096691e00, 2.93136299e-01],
        [-7.06798506e00, 2.51712799e-01],
        [-6.79538488e00, 2.15095803e-01],
        [-6.55630684e00, 1.58102095e-01],
        [-6.32537985e00, 7.50722364e-02],
        [-6.10422087e00, -2.96782795e-02],
        [-5.88530684e00, -1.66274801e-01],
        [-5.64315081e00, -3.34116489e-01],
        [-5.38260078e00, -5.07009685e-01],
        [-5.11993122e00, -6.73908472e-01],
        [-4.85968113e00, -8.33420396e-01],
        [-4.60405111e00, -9.90500689e-01],
        [-4.34939480e00, -1.14818501e00],
        [-4.09463692e00, -1.30712497e00],
        [-3.83910489e00, -1.46685696e00],
        [-3.58316112e00, -1.62663901e00],
        [-3.32717490e00, -1.78623903e00],
        [-3.07128096e00, -1.94563699e00],
        [-2.81551194e00, -2.10498405e00],
        [-2.55976200e00, -2.26435494e00],
        [-2.30400491e00, -2.42376399e00],
        [-2.04822397e00, -2.58319092e00],
        [-1.79243505e00, -2.74261808e00],
        [-1.53664505e00, -2.90204000e00],
        [-1.28085697e00, -3.06145692e00],
        [-1.02507198e00, -3.22087193e00],
        [-7.69287527e-01, -3.38028693e00],
        [-5.13505280e-01, -3.53970098e00],
        [-3.68053126e-01, -3.63035529e00],
    ]
)
inner_boarder = np.array(
    [
        [0.19622072, -2.72500554],
        [0.3065469, -2.79376698],
        [0.41687209, -2.86252998],
        [0.56233418, -2.95319295],
        [0.81810862, -3.11261511],
        [1.07386804, -3.27201605],
        [1.32963002, -3.43139005],
        [1.58542705, -3.59075904],
        [1.84132898, -3.75020504],
        [2.09726596, -3.90981698],
        [2.35312891, -4.06954193],
        [2.6084609, -4.22906685],
        [2.86336493, -4.38773394],
        [3.118433, -4.54558802],
        [3.37516594, -4.70355606],
        [3.63536906, -4.86478806],
        [3.89608908, -5.03168917],
        [4.15294313, -5.20160294],
        [4.3900528, -5.36139297],
        [4.61826086, -5.49317884],
        [4.87065792, -5.61515713],
        [5.12985706, -5.73890877],
        [5.35062981, -5.83397388],
        [5.55318117, -5.87877417],
        [5.75517702, -5.88925409],
        [5.95936394, -5.85699606],
        [6.17305422, -5.78921795],
        [6.38297176, -5.69038486],
        [6.58561182, -5.56168795],
        [6.78643322, -5.40244389],
        [6.97710085, -5.23151493],
        [7.15585899, -5.03150415],
        [7.33555508, -4.82472992],
        [7.49556589, -4.604527],
        [7.65372896, -4.37583113],
        [7.79681206, -4.13855314],
        [7.93451881, -3.89436293],
        [8.05978012, -3.64461708],
        [8.17728615, -3.38890409],
        [8.28360844, -3.13107896],
        [8.37892246, -2.86668396],
        [8.46462154, -2.60488796],
        [8.53236961, -2.33750892],
        [8.59184456, -2.07399607],
        [8.62627125, -1.81314003],
        [8.64647007, -1.56221402],
        [8.63543129, -1.33986104],
        [8.59661102, -1.15340805],
        [8.50731945, -0.98188621],
        [8.37734318, -0.80067849],
        [8.21822071, -0.63362288],
        [8.03214931, -0.48208401],
        [7.82128811, -0.3440949],
        [7.58607006, -0.21591701],
        [7.33044004, -0.09100879],
        [7.06178617, 0.03452676],
        [6.7749548, 0.1686497],
        [6.47766685, 0.32910699],
        [6.16920805, 0.5301463],
        [5.87017012, 0.80883712],
        [5.62881279, 1.17044497],
        [5.49051714, 1.58505797],
        [5.4575181, 2.02034307],
        [5.52809811, 2.40569997],
        [5.61156797, 2.71626091],
        [5.69183207, 3.00804496],
        [5.7675252, 3.28249311],
        [5.82185793, 3.54233789],
        [5.85609007, 3.791394],
        [5.86111498, 4.02343607],
        [5.83530903, 4.23640299],
        [5.77640581, 4.42590094],
        [5.68133211, 4.59579611],
        [5.54891109, 4.75109816],
        [5.39337778, 4.8826499],
        [5.2455678, 4.96743822],
        [5.10308409, 4.99766922],
        [4.93907022, 4.98232794],
        [4.75175381, 4.92289686],
        [4.55381393, 4.82103777],
        [4.35510588, 4.68611622],
        [4.16141796, 4.52237415],
        [3.97172904, 4.336586],
        [3.79007196, 4.13628292],
        [3.62215209, 3.92930508],
        [3.46469212, 3.70176792],
        [3.3101151, 3.47062492],
        [3.17909789, 3.24179292],
        [3.0655601, 3.0048821],
        [2.97108412, 2.76536298],
        [2.90679502, 2.55192804],
        [2.88726091, 2.41399789],
        [2.92608809, 2.312819],
        [3.03825593, 2.17010403],
        [3.21321011, 1.99254596],
        [3.41970205, 1.80469799],
        [3.64239597, 1.61179101],
        [3.87020302, 1.41461003],
        [4.09814787, 1.217255],
        [4.32835913, 1.01781094],
        [4.57346916, 0.7997089],
        [4.81197691, 0.54692823],
        [5.03936481, 0.25910759],
        [5.23791599, -0.101171],
        [5.35059214, -0.54139012],
        [5.32086992, -0.99578577],
        [5.17856884, -1.41400397],
        [4.91991711, -1.78779602],
        [4.56036282, -2.06991005],
        [4.15025282, -2.23430109],
        [3.7324779, -2.29090691],
        [3.33006501, -2.26068902],
        [2.94555306, -2.16043997],
        [2.60378408, -2.00176907],
        [2.31680894, -1.83086503],
        [2.0574491, -1.66772401],
        [1.80248499, -1.50869298],
        [1.54673803, -1.34898496],
        [1.29110098, -1.18937099],
        [1.03545201, -1.02975094],
        [0.77980441, -0.87014031],
        [0.5241614, -0.71054298],
        [0.26846161, -0.55093271],
        [0.01321426, -0.39154199],
        [-0.2455764, -0.2306165],
        [-0.51399148, -0.05810441],
        [-0.76973021, 0.1253769],
        [-1.01440799, 0.300313],
        [-1.25958002, 0.47557229],
        [-1.50475895, 0.65083408],
        [-1.74980104, 0.8259778],
        [-2.00508499, 1.00835705],
        [-2.26036596, 1.20924199],
        [-2.497823, 1.41272402],
        [-2.73925304, 1.62027597],
        [-2.97467208, 1.84951603],
        [-3.21005511, 2.09911203],
        [-3.43807602, 2.40698695],
        [-3.630337, 2.82436109],
        [-3.65351391, 3.32669711],
        [-3.50002098, 3.79086304],
        [-3.204, 4.16550207],
        [-2.83925509, 4.40905523],
        [-2.49380589, 4.54905987],
        [-2.18924499, 4.64677095],
        [-1.91084194, 4.73571777],
        [-1.67049003, 4.8204422],
        [-1.49451995, 4.90960693],
        [-1.37978804, 5.00971985],
        [-1.31275403, 5.12138987],
        [-1.28538406, 5.27809095],
        [-1.280321, 5.43674994],
        [-1.31434, 5.54018879],
        [-1.38657403, 5.62110806],
        [-1.53623903, 5.69938517],
        [-1.73996794, 5.77736902],
        [-1.96999097, 5.82499409],
        [-2.23178506, 5.85530901],
        [-2.51420999, 5.87452793],
        [-2.79453611, 5.88822079],
        [-3.08080912, 5.8855052],
        [-3.38347292, 5.88292503],
        [-3.68726301, 5.88172913],
        [-3.98979092, 5.88186884],
        [-4.29113102, 5.88196182],
        [-4.59251404, 5.88203907],
        [-4.89394283, 5.88211823],
        [-5.19548893, 5.88223314],
        [-5.4886899, 5.88253784],
        [-5.77231407, 5.87393713],
        [-6.05983686, 5.85523415],
        [-6.34402323, 5.83254194],
        [-6.61511517, 5.79841709],
        [-6.87844992, 5.74904108],
        [-7.12874985, 5.68548298],
        [-7.34630299, 5.60792398],
        [-7.50391912, 5.52442312],
        [-7.62953281, 5.38708878],
        [-7.77151823, 5.19112682],
        [-7.90684986, 4.96692514],
        [-8.03299618, 4.72594595],
        [-8.1497097, 4.47420311],
        [-8.25618458, 4.21765709],
        [-8.35022545, 3.95650697],
        [-8.43389988, 3.68964791],
        [-8.5068779, 3.42036104],
        [-8.56724072, 3.15108895],
        [-8.61201954, 2.89205003],
        [-8.63172626, 2.62775207],
        [-8.64108467, 2.36953998],
        [-8.62343502, 2.14536405],
        [-8.57776737, 1.96034801],
        [-8.51438427, 1.83432603],
        [-8.39197636, 1.73537195],
        [-8.19870186, 1.62191999],
        [-7.97809696, 1.52076602],
        [-7.74709415, 1.44488096],
        [-7.49555302, 1.39132702],
        [-7.21776104, 1.34887803],
        [-6.92350197, 1.30868399],
        [-6.59886789, 1.26364005],
        [-6.25167799, 1.18048406],
        [-5.91602898, 1.06020904],
        [-5.59352589, 0.90693992],
        [-5.30028582, 0.72580957],
        [-5.04455423, 0.54891539],
        [-4.80160284, 0.38769919],
        [-4.55515289, 0.2311272],
        [-4.30170918, 0.07582727],
        [-4.04398298, -0.08254288],
        [-3.78624701, -0.24213441],
        [-3.5295651, -0.40227279],
        [-3.27390289, -0.5620864],
        [-3.01848888, -0.72153711],
        [-2.7629509, -0.88085711],
        [-2.50720692, -1.04016197],
        [-2.25135899, -1.19955897],
        [-1.99552, -1.35898495],
        [-1.73971498, -1.51842403],
        [-1.48393703, -1.67785001],
        [-1.228163, -1.83726799],
        [-0.97238493, -1.99668205],
        [-0.71660161, -2.15609598],
        [-0.46081591, -2.31551099],
        [-0.20502999, -2.47492695],
        [0.05075799, -2.63434505],
        [0.19622072, -2.72500554],
    ]
)


reward_object = Reward(race_line, inner_boarder, outer_boarder)


def reward_function(params):
    return reward_object.reward_function(params)
