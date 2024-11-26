import torch


def extract_kinetic_features(positions: torch.Tensor):
    """
    :param positions: t j c
    :return:
    """
    assert len(positions.shape) == 3  # (seq_len, n_joints, 3)
    features = KineticFeatures(positions)
    kinetic_feature_vector = []
    for i in range(positions.shape[1]):
        feature_vector = torch.tensor(
            [
                features.average_kinetic_energy_horizontal(i),
                features.average_kinetic_energy_vertical(i),
                features.average_energy_expenditure(i),
            ]
        )
        kinetic_feature_vector.extend(feature_vector)
    kinetic_feature_vector = torch.stack(kinetic_feature_vector)
    return kinetic_feature_vector


class KineticFeatures:
    def __init__(self, positions, fps=60, up_vec="y", sliding_window=2):
        self.positions = torch.tensor(positions, dtype=torch.float32)  # Convert positions to tensor

        self.frame_time = 1.0 / fps
        self.up_vec = up_vec
        self.sliding_window = sliding_window

    def average_kinetic_energy(self, joint):
        average_kinetic_energy = 0
        for i in range(1, len(self.positions)):
            average_velocity = calc_average_velocity(
                self.positions, i, joint, self.sliding_window, self.frame_time
            )
            average_kinetic_energy += average_velocity ** 2
        average_kinetic_energy = average_kinetic_energy / (len(self.positions) - 1.0)
        return average_kinetic_energy

    def average_kinetic_energy_horizontal(self, joint) -> float:
        val = 0
        for i in range(1, len(self.positions)):
            average_velocity = calc_average_velocity_horizontal(
                self.positions,
                i,
                joint,
                self.sliding_window,
                self.frame_time,
                self.up_vec,
            )
            val += average_velocity ** 2
        val = val / (len(self.positions) - 1.0)
        return val

    def average_kinetic_energy_vertical(self, joint):
        val = 0
        for i in range(1, len(self.positions)):
            average_velocity = calc_average_velocity_vertical(
                self.positions,
                i,
                joint,
                self.sliding_window,
                self.frame_time,
                self.up_vec,
            )
            val += average_velocity ** 2
        val = val / (len(self.positions) - 1.0)
        return val

    def average_energy_expenditure(self, joint):
        val = 0.0
        for i in range(1, len(self.positions)):
            val += calc_average_acceleration(
                self.positions, i, joint, self.sliding_window, self.frame_time
            )
        val = val / (len(self.positions) - 1.0)
        return val


def distance_between_points(a, b):
    return torch.norm(torch.tensor(a, dtype=torch.float32) - torch.tensor(b, dtype=torch.float32))


def distance_from_plane(a, b, c, p, threshold):
    a, b, c, p = map(lambda x: torch.tensor(x, dtype=torch.float32), [a, b, c, p])
    ba = b - a
    ca = c - a
    cross = torch.cross(ca, ba)

    pa = p - a
    return (torch.dot(cross, pa) / torch.norm(cross)) > threshold


def distance_from_plane_normal(n1, n2, a, p, threshold):
    normal = torch.tensor(n2, dtype=torch.float32) - torch.tensor(n1, dtype=torch.float32)
    pa = torch.tensor(p, dtype=torch.float32) - torch.tensor(a, dtype=torch.float32)
    return (torch.dot(normal, pa) / torch.norm(normal)) > threshold


def angle_within_range(j1, j2, k1, k2, angle_range):
    j = torch.tensor(j2, dtype=torch.float32) - torch.tensor(j1, dtype=torch.float32)
    k = torch.tensor(k2, dtype=torch.float32) - torch.tensor(k1, dtype=torch.float32)

    angle = torch.acos(torch.clamp(torch.dot(j, k) / (torch.norm(j) * torch.norm(k)), -1.0, 1.0))
    angle_degrees = torch.degrees(angle)

    return angle_range[0] < angle_degrees < angle_range[1]


def velocity_direction_above_threshold(
        j1, j1_prev, j2, j2_prev, p, p_prev, threshold, time_per_frame=1 / 120.0
):
    j1, j1_prev, j2, p, p_prev = map(lambda x: torch.tensor(x, dtype=torch.float32), [j1, j1_prev, j2, p, p_prev])
    velocity = p - j1 - (p_prev - j1_prev)
    direction = j2 - j1

    velocity_along_direction = torch.dot(velocity, direction) / torch.norm(direction)
    velocity_along_direction /= time_per_frame
    return velocity_along_direction > threshold


def velocity_direction_above_threshold_normal(
        j1, j1_prev, j2, j3, p, p_prev, threshold, time_per_frame=1 / 120.0
):
    j1, j1_prev, j2, j3, p, p_prev = map(lambda x: torch.tensor(x, dtype=torch.float32),
                                         [j1, j1_prev, j2, j3, p, p_prev])
    velocity = p - j1 - (p_prev - j1_prev)
    j31 = j3 - j1
    j21 = j2 - j1
    direction = torch.cross(j31, j21)

    velocity_along_direction = torch.dot(velocity, direction) / torch.norm(direction)
    velocity_along_direction /= time_per_frame
    return velocity_along_direction > threshold


def velocity_above_threshold(p, p_prev, threshold, time_per_frame=1 / 120.0):
    p, p_prev = map(lambda x: torch.tensor(x, dtype=torch.float32), [p, p_prev])
    velocity = torch.norm(p - p_prev) / time_per_frame
    return velocity > threshold


def calc_average_velocity(positions, i, joint_idx, sliding_window, frame_time):
    current_window = 0
    average_velocity = torch.zeros_like(positions[0][joint_idx])
    for j in range(-sliding_window, sliding_window + 1):
        if i + j - 1 < 0 or i + j >= len(positions):
            continue
        average_velocity += (positions[i + j][joint_idx] - positions[i + j - 1][joint_idx])
        current_window += 1
    return torch.norm(average_velocity / (current_window * frame_time))


def calc_average_acceleration(positions, i, joint_idx, sliding_window, frame_time):
    current_window = 0
    average_acceleration = torch.zeros_like(positions[0][joint_idx])
    for j in range(-sliding_window, sliding_window + 1):
        if i + j - 1 < 0 or i + j + 1 >= len(positions):
            continue
        v2 = (positions[i + j + 1][joint_idx] - positions[i + j][joint_idx]) / frame_time
        v1 = (positions[i + j][joint_idx] - positions[i + j - 1][joint_idx]) / frame_time
        average_acceleration += (v2 - v1) / frame_time
        current_window += 1
    return torch.norm(average_acceleration / current_window)


def calc_average_velocity_horizontal(positions, i, joint_idx, sliding_window, frame_time, up_vec="z"):
    current_window = 0
    average_velocity = torch.zeros_like(positions[0][joint_idx])
    for j in range(-sliding_window, sliding_window + 1):
        if i + j - 1 < 0 or i + j >= len(positions):
            continue
        average_velocity += (positions[i + j][joint_idx] - positions[i + j - 1][joint_idx])
        current_window += 1
    if up_vec == "y":
        average_velocity = torch.tensor([average_velocity[0], average_velocity[2]])
    elif up_vec == "z":
        average_velocity = torch.tensor([average_velocity[0], average_velocity[1]])
    else:
        raise NotImplementedError
    return torch.norm(average_velocity / (current_window * frame_time))


def calc_average_velocity_vertical(positions, i, joint_idx, sliding_window, frame_time, up_vec):
    current_window = 0
    average_velocity = torch.zeros_like(positions[0][joint_idx])
    for j in range(-sliding_window, sliding_window + 1):
        if i + j - 1 < 0 or i + j >= len(positions):
            continue
        average_velocity += (positions[i + j][joint_idx] - positions[i + j - 1][joint_idx])
        current_window += 1
    if up_vec == "y":
        average_velocity = torch.tensor([average_velocity[1]])
    elif up_vec == "z":
        average_velocity = torch.tensor([average_velocity[2]])
    else:
        raise NotImplementedError
    return torch.norm(average_velocity / (current_window * frame_time))
