"""PyCUDA MPPI controller for the B2 omnidirectional SE(2) nominal model."""

from __future__ import annotations

import numpy as np

import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from pycuda import gpuarray

from mppi_controller.core.omni_b2 import OmniB2


CUDA_SOURCE = r"""
__device__ float clamp_value(float value, float limit) {
    if (value > limit) {
        return limit;
    }
    if (value < -limit) {
        return -limit;
    }
    return value;
}

__device__ float clamp_range(float value, float lower, float upper) {
    if (value > upper) {
        return upper;
    }
    if (value < lower) {
        return lower;
    }
    return value;
}

__device__ float angle_diff(float a, float b) {
    const float pi = 3.14159265358979323846f;
    return fmodf(a - b + pi, 2.0f * pi) - pi;
}

__device__ float barrier_distance(
    int cbf_type,
    float robot_x,
    float robot_y,
    float robot_vx,
    float robot_vy,
    float obs_x,
    float obs_y,
    float obs_vx,
    float obs_vy,
    float obs_radius,
    float robot_radius,
    float safety_dist,
    float atau
) {
    float safe_radius = obs_radius + robot_radius + safety_dist;
    float dx = obs_x - robot_x;
    float dy = obs_y - robot_y;
    float dvx = obs_vx - robot_vx;
    float dvy = obs_vy - robot_vy;
    float dot = dx * dvx + dy * dvy;
    float dist = sqrtf(dx * dx + dy * dy);

    if (cbf_type == 1 && dot < 0.0f && dist > 1e-6f) {
        float cos_v = dot / dist;
        float tau = 0.1f * dist / (cos_v + 0.001f);
        if (fabsf(tau) > atau) {
            return dist + cos_v * atau - safe_radius;
        }
        return dist + cos_v * fabsf(tau) - safe_radius;
    }

    if (cbf_type == 2 && dot < 0.0f) {
        float px = dx + dvx * atau;
        float py = dy + dvy * atau;
        return sqrtf(px * px + py * py) - safe_radius;
    }

    return dist - safe_radius;
}

__device__ float path_progress_value(float px, float py, const float *path, int path_count) {
    if (path_count < 2) {
        return 0.0f;
    }
    float best_sq = 3.402823466e+38f;
    float best_progress = 0.0f;
    float cumulative = 0.0f;
    for (int idx = 0; idx < path_count - 1; ++idx) {
        int start_offset = idx * 2;
        int end_offset = (idx + 1) * 2;
        float sx = path[start_offset];
        float sy = path[start_offset + 1];
        float ex = path[end_offset];
        float ey = path[end_offset + 1];
        float vx = ex - sx;
        float vy = ey - sy;
        float segment_len_sq = vx * vx + vy * vy;
        float segment_len = sqrtf(segment_len_sq);
        float t = 0.0f;
        if (segment_len_sq > 1e-9f) {
            t = ((px - sx) * vx + (py - sy) * vy) / segment_len_sq;
            t = fminf(fmaxf(t, 0.0f), 1.0f);
        }
        float proj_x = sx + t * vx;
        float proj_y = sy + t * vy;
        float dx = px - proj_x;
        float dy = py - proj_y;
        float sq = dx * dx + dy * dy;
        if (sq < best_sq) {
            best_sq = sq;
            best_progress = cumulative + t * segment_len;
        }
        cumulative += segment_len;
    }
    return best_progress;
}

__device__ float path_tracking_step_cost(
    float px,
    float py,
    const float *path,
    int path_count,
    float path_tracking_weight,
    float path_tracking_tolerance
) {
    if (path_count < 2 || path_tracking_weight <= 0.0f) {
        return 0.0f;
    }
    float min_sq = 3.402823466e+38f;
    for (int idx = 0; idx < path_count - 1; ++idx) {
        int start_offset = idx * 2;
        int end_offset = (idx + 1) * 2;
        float sx = path[start_offset];
        float sy = path[start_offset + 1];
        float ex = path[end_offset];
        float ey = path[end_offset + 1];
        float vx = ex - sx;
        float vy = ey - sy;
        float segment_len_sq = vx * vx + vy * vy;
        float t = 0.0f;
        if (segment_len_sq > 1e-9f) {
            t = ((px - sx) * vx + (py - sy) * vy) / segment_len_sq;
            t = fminf(fmaxf(t, 0.0f), 1.0f);
        }
        float proj_x = sx + t * vx;
        float proj_y = sy + t * vy;
        float dx = px - proj_x;
        float dy = py - proj_y;
        min_sq = fminf(min_sq, dx * dx + dy * dy);
    }
    float excess = sqrtf(fmaxf(min_sq, 0.0f)) - path_tracking_tolerance;
    if (excess <= 0.0f) {
        return 0.0f;
    }
    return path_tracking_weight * excess * excess;
}

__device__ float local_costmap_point_term(
    float px,
    float py,
    float start_x,
    float start_y,
    const float *costmap_data,
    const unsigned char *costmap_unknown,
    int costmap_width,
    int costmap_height,
    float costmap_origin_x,
    float costmap_origin_y,
    float costmap_resolution,
    float costmap_power,
    float costmap_unknown_cost,
    float costmap_max_cost,
    float unknown_clear_radius,
    float unknown_clear_value
) {
    int ix = (int)floorf((px - costmap_origin_x) / costmap_resolution);
    int iy = (int)floorf((py - costmap_origin_y) / costmap_resolution);
    float sampled = costmap_unknown_cost;
    bool sampled_unknown = true;
    if (ix >= 0 && ix < costmap_width && iy >= 0 && iy < costmap_height) {
        int flat_idx = iy * costmap_width + ix;
        sampled = costmap_data[flat_idx];
        sampled_unknown = costmap_unknown[flat_idx] != 0;
    }
    if (sampled_unknown && unknown_clear_radius > 0.0f) {
        float dx = px - start_x;
        float dy = py - start_y;
        if (sqrtf(dx * dx + dy * dy) <= unknown_clear_radius) {
            sampled = unknown_clear_value;
        }
    }
    float normalized = fminf(fmaxf(sampled, 0.0f), costmap_max_cost) / fmaxf(costmap_max_cost, 1e-6f);
    return powf(normalized, fmaxf(costmap_power, 0.1f));
}

extern "C" __global__ void omni_costs(
    const float *initial_state,
    const float *controls,
    const float *previous_control,
    const float *goal,
    const float *progress_goal,
    const float *obstacles,
    const float *path,
    const float *costmap_data,
    const unsigned char *costmap_unknown,
    float *costs,
    int num_samples,
    int horizon_steps,
    int obstacle_count,
    float dt,
    float min_vx,
    float max_vx,
    float max_vy,
    float max_wz,
    float goal_xy_weight,
    float yaw_weight,
    float control_weight,
    float smooth_weight,
    float lateral_weight,
    float yaw_rate_weight,
    float accel_weight,
    float jerk_weight,
    float obstacle_weight,
    float obstacle_collision_weight,
    float obstacle_soft_weight,
    float obstacle_influence_dist,
    float cbf_weight,
    float cbf_alpha,
    int cbf_type,
    float atau,
    float robot_radius,
    float safety_dist,
    float max_ax,
    float max_ay,
    float max_awz,
    float velocity_lag_beta,
    float goal_progress_weight,
    float heading_to_goal_weight,
    float heading_to_goal_min_distance,
    int path_count,
    float path_tracking_weight,
    float path_tracking_tolerance,
    float path_progress_weight,
    int costmap_enabled,
    int costmap_width,
    int costmap_height,
    float costmap_origin_x,
    float costmap_origin_y,
    float costmap_resolution,
    float costmap_weight,
    float costmap_power,
    float costmap_unknown_cost,
    float costmap_max_cost,
    float unknown_clear_radius,
    float unknown_clear_value,
    int footprint_enabled,
    float footprint_radius,
    int footprint_sample_count
) {
    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= num_samples) {
        return;
    }

    float x = initial_state[0];
    float y = initial_state[1];
    float theta = initial_state[2];
    float robot_world_vx = initial_state[3] * cosf(theta) - initial_state[4] * sinf(theta);
    float robot_world_vy = initial_state[3] * sinf(theta) + initial_state[4] * cosf(theta);
    float prev_u0 = previous_control[0];
    float prev_u1 = previous_control[1];
    float prev_u2 = previous_control[2];
    float prev_real0 = initial_state[3];
    float prev_real1 = initial_state[4];
    float prev_real2 = initial_state[5];
    float prev_prev_real0 = prev_real0;
    float prev_prev_real1 = prev_real1;
    float prev_prev_real2 = prev_real2;
    float max_du0 = max_ax * dt;
    float max_du1 = max_ay * dt;
    float max_du2 = max_awz * dt;
    float start_x = x;
    float start_y = y;
    float start_goal_dx = progress_goal[0] - x;
    float start_goal_dy = progress_goal[1] - y;
    float start_goal_distance = sqrtf(start_goal_dx * start_goal_dx + start_goal_dy * start_goal_dy);
    float start_path_progress = path_progress_value(x, y, path, path_count);
    float cost = 0.0f;

    for (int step = 0; step < horizon_steps; ++step) {
        int offset = (sample * horizon_steps + step) * 3;
        float cmd0 = clamp_range(controls[offset], min_vx, max_vx);
        float cmd1 = clamp_value(controls[offset + 1], max_vy);
        float cmd2 = clamp_value(controls[offset + 2], max_wz);

        cost += control_weight * (cmd0 * cmd0 + cmd1 * cmd1 + cmd2 * cmd2);
        float du0 = cmd0 - prev_u0;
        float du1 = cmd1 - prev_u1;
        float du2 = cmd2 - prev_u2;
        cost += smooth_weight * (du0 * du0 + du1 * du1 + du2 * du2);

        float lag0 = velocity_lag_beta * prev_real0 + (1.0f - velocity_lag_beta) * cmd0;
        float lag1 = velocity_lag_beta * prev_real1 + (1.0f - velocity_lag_beta) * cmd1;
        float lag2 = velocity_lag_beta * prev_real2 + (1.0f - velocity_lag_beta) * cmd2;
        float real_du0 = clamp_value(lag0 - prev_real0, max_du0);
        float real_du1 = clamp_value(lag1 - prev_real1, max_du1);
        float real_du2 = clamp_value(lag2 - prev_real2, max_du2);
        float vx = clamp_value(prev_real0 + real_du0, max_vx);
        float vy = clamp_value(prev_real1 + real_du1, max_vy);
        float wz = clamp_value(prev_real2 + real_du2, max_wz);

        float ax = (vx - prev_real0) / dt;
        float ay = (vy - prev_real1) / dt;
        float awz = (wz - prev_real2) / dt;
        cost += accel_weight * (ax * ax + ay * ay + awz * awz);
        float jerk0 = vx - 2.0f * prev_real0 + prev_prev_real0;
        float jerk1 = vy - 2.0f * prev_real1 + prev_prev_real1;
        float jerk2 = wz - 2.0f * prev_real2 + prev_prev_real2;
        cost += jerk_weight * (jerk0 * jerk0 + jerk1 * jerk1 + jerk2 * jerk2);
        cost += lateral_weight * vy * vy;
        cost += yaw_rate_weight * wz * wz;

        float old_x = x;
        float old_y = y;
        float cos_theta = cosf(theta);
        float sin_theta = sinf(theta);
        x += (vx * cos_theta - vy * sin_theta) * dt;
        y += (vx * sin_theta + vy * cos_theta) * dt;
        theta += wz * dt;

        if (heading_to_goal_weight > 0.0f) {
            float heading_dx = progress_goal[0] - x;
            float heading_dy = progress_goal[1] - y;
            float heading_distance = sqrtf(heading_dx * heading_dx + heading_dy * heading_dy);
            if (heading_distance > heading_to_goal_min_distance) {
                float target_yaw = atan2f(heading_dy, heading_dx);
                float heading_error = angle_diff(theta, target_yaw);
                cost += heading_to_goal_weight * heading_error * heading_error;
            }
        }

        cost += path_tracking_step_cost(x, y, path, path_count, path_tracking_weight, path_tracking_tolerance);

        if (
            costmap_enabled != 0
            && costmap_width > 0
            && costmap_height > 0
            && costmap_resolution > 0.0f
            && costmap_weight > 0.0f
        ) {
            float step_term = local_costmap_point_term(
                x,
                y,
                start_x,
                start_y,
                costmap_data,
                costmap_unknown,
                costmap_width,
                costmap_height,
                costmap_origin_x,
                costmap_origin_y,
                costmap_resolution,
                costmap_power,
                costmap_unknown_cost,
                costmap_max_cost,
                unknown_clear_radius,
                unknown_clear_value
            );
            if (footprint_enabled != 0 && footprint_radius > 0.0f && footprint_sample_count >= 4) {
                int capped_count = min(footprint_sample_count, 64);
                for (int fp_idx = 0; fp_idx < capped_count; ++fp_idx) {
                    float angle = 6.28318530717958647692f * ((float)fp_idx) / ((float)capped_count);
                    float ox = footprint_radius * cosf(angle);
                    float oy = footprint_radius * sinf(angle);
                    float sample_x = x + cosf(theta) * ox - sinf(theta) * oy;
                    float sample_y = y + sinf(theta) * ox + cosf(theta) * oy;
                    float term = local_costmap_point_term(
                        sample_x,
                        sample_y,
                        start_x,
                        start_y,
                        costmap_data,
                        costmap_unknown,
                        costmap_width,
                        costmap_height,
                        costmap_origin_x,
                        costmap_origin_y,
                        costmap_resolution,
                        costmap_power,
                        costmap_unknown_cost,
                        costmap_max_cost,
                        unknown_clear_radius,
                        unknown_clear_value
                    );
                    step_term = fmaxf(step_term, term);
                }
            }
            cost += costmap_weight * step_term;
        }

        for (int obs_index = 0; obs_index < obstacle_count; ++obs_index) {
            int obs_offset = obs_index * 7;
            float ox = obstacles[obs_offset];
            float oy = obstacles[obs_offset + 1];
            float radius = obstacles[obs_offset + 2];
            float obs_vx = obstacles[obs_offset + 5];
            float obs_vy = obstacles[obs_offset + 6];

            float dx = x - ox;
            float dy = y - oy;
            float clearance = sqrtf(dx * dx + dy * dy) - radius - robot_radius;
            float margin = safety_dist - clearance;
            if (margin > 0.0f) {
                cost += obstacle_weight * margin * margin;
            }
            if (clearance < 0.0f && obstacle_collision_weight > 0.0f) {
                float overlap = -clearance;
                cost += obstacle_collision_weight * (overlap + overlap * overlap);
            }
            if (
                obstacle_soft_weight > 0.0f
                && obstacle_influence_dist > safety_dist
                && clearance > safety_dist
                && clearance < obstacle_influence_dist
            ) {
                float soft_margin = obstacle_influence_dist - clearance;
                cost += obstacle_soft_weight * soft_margin * soft_margin;
            }

            if (cbf_weight > 0.0f) {
                float old_obs_x = ox;
                float old_obs_y = oy;
                float next_obs_x = ox + obs_vx * dt;
                float next_obs_y = oy + obs_vy * dt;
                float next_robot_world_vx = vx * cos_theta - vy * sin_theta;
                float next_robot_world_vy = vx * sin_theta + vy * cos_theta;
                float h = barrier_distance(
                    cbf_type,
                    old_x,
                    old_y,
                    robot_world_vx,
                    robot_world_vy,
                    old_obs_x,
                    old_obs_y,
                    obs_vx,
                    obs_vy,
                    radius,
                    robot_radius,
                    safety_dist,
                    atau
                );
                float h_next = barrier_distance(
                    cbf_type,
                    x,
                    y,
                    next_robot_world_vx,
                    next_robot_world_vy,
                    next_obs_x,
                    next_obs_y,
                    obs_vx,
                    obs_vy,
                    radius,
                    robot_radius,
                    safety_dist,
                    atau
                );
                float cbf_violation = cbf_type == 3 ? -h : -h_next + cbf_alpha * h;
                if (cbf_violation > 0.0f) {
                    cost += cbf_weight * cbf_violation;
                }
            }
        }

        prev_u0 = cmd0;
        prev_u1 = cmd1;
        prev_u2 = cmd2;
        prev_prev_real0 = prev_real0;
        prev_prev_real1 = prev_real1;
        prev_prev_real2 = prev_real2;
        prev_real0 = vx;
        prev_real1 = vy;
        prev_real2 = wz;
        robot_world_vx = vx * cos_theta - vy * sin_theta;
        robot_world_vy = vx * sin_theta + vy * cos_theta;
    }

    float xy0 = x - goal[0];
    float xy1 = y - goal[1];
    float yaw_error = angle_diff(theta, goal[2]);
    cost += goal_xy_weight * (xy0 * xy0 + xy1 * xy1);
    cost += yaw_weight * yaw_error * yaw_error;
    if (goal_progress_weight > 0.0f) {
        float progress_dx = x - progress_goal[0];
        float progress_dy = y - progress_goal[1];
        float final_distance = sqrtf(progress_dx * progress_dx + progress_dy * progress_dy);
        cost += -goal_progress_weight * (start_goal_distance - final_distance);
    }
    if (path_progress_weight > 0.0f && path_count >= 2) {
        float final_path_progress = path_progress_value(x, y, path, path_count);
        cost += -path_progress_weight * (final_path_progress - start_path_progress);
    }
    costs[sample] = cost;
}
"""


class MppiOmniCuda:
    def __init__(
        self,
        *,
        dt: float,
        horizon_steps: int,
        num_samples: int,
        lambda_: float,
        noise_std: np.ndarray,
        max_vx: float,
        max_vy: float,
        max_wz: float,
        min_vx: float | None = None,
        goal_xy_weight: float = 5.0,
        yaw_weight: float = 0.2,
        control_weight: float = 0.01,
        smooth_weight: float = 0.2,
        obstacle_weight: float = 25.0,
        obstacle_collision_weight: float = 0.0,
        obstacle_soft_weight: float = 0.0,
        obstacle_influence_dist: float = 0.0,
        max_ax: float = 1000.0,
        max_ay: float = 1000.0,
        max_awz: float = 1000.0,
        velocity_lag_beta: float = 0.0,
        lateral_weight: float = 0.0,
        yaw_rate_weight: float = 0.0,
        accel_weight: float = 0.0,
        jerk_weight: float = 0.0,
        path_tracking_weight: float = 0.0,
        path_tracking_tolerance: float = 0.3,
        path_progress_weight: float = 0.0,
        goal_progress_weight: float = 0.0,
        heading_to_goal_weight: float = 0.0,
        heading_to_goal_min_distance: float = 0.3,
        update_smoothing_alpha: float | list[float] | tuple[float, ...] | np.ndarray = 0.0,
        goal_change_reset_distance: float = 0.75,
        goal_change_reset_yaw: float = 1.0,
        cbf_weight: float = 0.0,
        cbf_alpha: float = 0.1,
        cbf_type: int = 0,
        atau: float = 0.0,
        robot_radius: float = 0.6,
        safety_dist: float = 0.3,
        draw_num_traj: int = 150,
        seed: int | None = None,
        block_dim: int = 128,
    ) -> None:
        self.dt = float(dt)
        self.horizon_steps = int(horizon_steps)
        self.num_samples = int(num_samples)
        self.lambda_ = float(lambda_)
        self.noise_std = np.asarray(noise_std, dtype=np.float32)
        self.max_control = np.asarray([max_vx, max_vy, max_wz], dtype=np.float32)
        self.min_control = np.asarray(
            [min_vx if min_vx is not None else -max_vx, -max_vy, -max_wz],
            dtype=np.float32,
        )
        self.goal_xy_weight = float(goal_xy_weight)
        self.yaw_weight = float(yaw_weight)
        self.control_weight = float(control_weight)
        self.smooth_weight = float(smooth_weight)
        self.obstacle_weight = float(obstacle_weight)
        self.obstacle_collision_weight = float(obstacle_collision_weight)
        self.obstacle_soft_weight = float(obstacle_soft_weight)
        self.obstacle_influence_dist = float(obstacle_influence_dist)
        self.max_accel = np.asarray([max_ax, max_ay, max_awz], dtype=np.float32)
        self.velocity_lag_beta = float(np.clip(velocity_lag_beta, 0.0, 1.0))
        self.lateral_weight = float(lateral_weight)
        self.yaw_rate_weight = float(yaw_rate_weight)
        self.accel_weight = float(accel_weight)
        self.jerk_weight = float(jerk_weight)
        self.path_tracking_weight = float(path_tracking_weight)
        self.path_tracking_tolerance = float(path_tracking_tolerance)
        self.path_progress_weight = float(path_progress_weight)
        self.goal_progress_weight = float(goal_progress_weight)
        self.heading_to_goal_weight = float(heading_to_goal_weight)
        self.heading_to_goal_min_distance = float(max(heading_to_goal_min_distance, 0.0))
        self.update_smoothing_alpha = self._control_axis_alpha(update_smoothing_alpha)
        self.goal_change_reset_distance = float(max(goal_change_reset_distance, 0.0))
        self.goal_change_reset_yaw = float(max(goal_change_reset_yaw, 0.0))
        self.cbf_weight = float(cbf_weight)
        self.cbf_alpha = float(cbf_alpha)
        self.cbf_type = int(cbf_type)
        self.atau = float(atau)
        self.robot_radius = float(robot_radius)
        self.safety_dist = float(safety_dist)
        self.draw_num_traj = min(int(draw_num_traj), self.num_samples)
        self.block_dim = int(block_dim)
        self.rng = np.random.default_rng(seed)
        self.nominal_u = np.zeros((self.horizon_steps, 3), dtype=np.float32)
        self.previous_control = np.zeros(3, dtype=np.float32)
        self._has_nominal_update = False
        self._last_goal_for_nominal_reset: np.ndarray | None = None
        self.model = OmniB2(self.dt, max_vx, max_vy, max_wz)
        self.module = SourceModule(CUDA_SOURCE, no_extern_c=True)
        self.cost_kernel = self.module.get_function("omni_costs")

    @classmethod
    def from_config(
        cls,
        config: dict,
        seed: int | None = None,
        **overrides,
    ) -> "MppiOmniCuda":
        sim = config["simulation"]
        mppi = config["mppi"]
        robot = config["robot"]
        cbf = config["cbf"]
        sampling_rate = float(sim["sampling_rate"])
        dt = 1.0 / sampling_rate
        horizon_steps = int(float(sim["time_horizon"]) * sampling_rate)
        cbf_enabled = bool(cbf.get("enabled", True))
        cbf_weight = float(overrides.get("cbf_weight", mppi.get("cbf_weight", 0.0)))
        if not cbf_enabled:
            cbf_weight = 0.0
        return cls(
            dt=dt,
            horizon_steps=horizon_steps,
            num_samples=int(mppi["num_trajectories"]),
            lambda_=float(mppi["lambda"]),
            noise_std=np.asarray(mppi["std_normal"], dtype=np.float32),
            max_vx=float(robot["max_vx"]),
            max_vy=float(robot["max_vy"]),
            max_wz=float(robot["max_wz"]),
            min_vx=float(overrides.get("min_vx", robot.get("min_vx", -float(robot["max_vx"])))),
            goal_xy_weight=float(overrides.get("goal_xy_weight", mppi["weights"][0])),
            yaw_weight=float(overrides.get("yaw_weight", mppi["weights"][2])),
            control_weight=float(overrides.get("control_weight", mppi.get("control_weight", 0.01))),
            smooth_weight=float(overrides.get("smooth_weight", mppi.get("smooth_weight", 0.2))),
            obstacle_weight=float(overrides.get("obstacle_weight", mppi.get("obstacle_weight", 25.0))),
            obstacle_collision_weight=float(
                overrides.get("obstacle_collision_weight", mppi.get("obstacle_collision_weight", 0.0))
            ),
            obstacle_soft_weight=float(
                overrides.get("obstacle_soft_weight", mppi.get("obstacle_soft_weight", 0.0))
            ),
            obstacle_influence_dist=float(
                overrides.get("obstacle_influence_dist", mppi.get("obstacle_influence_dist", 0.0))
            ),
            max_ax=float(overrides.get("max_ax", robot.get("max_ax", 1000.0))),
            max_ay=float(overrides.get("max_ay", robot.get("max_ay", 1000.0))),
            max_awz=float(overrides.get("max_awz", robot.get("max_awz", 1000.0))),
            velocity_lag_beta=float(overrides.get("velocity_lag_beta", robot.get("velocity_lag_beta", 0.0))),
            lateral_weight=float(overrides.get("lateral_weight", mppi.get("lateral_weight", 0.0))),
            yaw_rate_weight=float(overrides.get("yaw_rate_weight", mppi.get("yaw_rate_weight", 0.0))),
            accel_weight=float(overrides.get("accel_weight", mppi.get("accel_weight", 0.0))),
            jerk_weight=float(overrides.get("jerk_weight", mppi.get("jerk_weight", 0.0))),
            path_tracking_weight=float(
                overrides.get("path_tracking_weight", mppi.get("path_tracking_weight", 0.0))
            ),
            path_tracking_tolerance=float(
                overrides.get("path_tracking_tolerance", mppi.get("path_tracking_tolerance", 0.3))
            ),
            path_progress_weight=float(
                overrides.get("path_progress_weight", mppi.get("path_progress_weight", 0.0))
            ),
            goal_progress_weight=float(
                overrides.get("goal_progress_weight", mppi.get("goal_progress_weight", 0.0))
            ),
            heading_to_goal_weight=float(
                overrides.get("heading_to_goal_weight", mppi.get("heading_to_goal_weight", 0.0))
            ),
            heading_to_goal_min_distance=float(
                overrides.get(
                    "heading_to_goal_min_distance",
                    mppi.get("heading_to_goal_min_distance", 0.3),
                )
            ),
            update_smoothing_alpha=overrides.get("update_smoothing_alpha", mppi.get("update_smoothing_alpha", 0.0)),
            goal_change_reset_distance=float(
                overrides.get("goal_change_reset_distance", mppi.get("goal_change_reset_distance", 0.75))
            ),
            goal_change_reset_yaw=float(
                overrides.get("goal_change_reset_yaw", mppi.get("goal_change_reset_yaw", 1.0))
            ),
            cbf_weight=cbf_weight,
            cbf_alpha=float(overrides.get("cbf_alpha", cbf.get("dcbf_alpha", 0.1))),
            cbf_type=int(overrides.get("cbf_type", cbf.get("type", 0))),
            atau=float(overrides.get("atau", cbf.get("atau", 0.0))),
            robot_radius=float(robot["radius"]),
            safety_dist=float(robot["safety_dist"]),
            draw_num_traj=int(mppi["draw_num_traj"]),
            seed=seed,
        )

    def compute_control(self, state: np.ndarray, cost_params):
        goal = np.asarray(cost_params[3], dtype=np.float32)
        self._reset_nominal_on_goal_change(goal)
        obstacles = np.asarray(cost_params[4], dtype=np.float32).reshape(-1, 7)
        path = self._path_from_cost_params(cost_params)
        costmap = self._costmap_from_cost_params(cost_params)
        progress_goal = self._progress_goal_from_cost_params(cost_params, goal)
        previous_nominal = self.nominal_u.copy()
        noise = self.rng.normal(
            loc=0.0,
            scale=self.noise_std,
            size=(self.num_samples, self.horizon_steps, 3),
        ).astype(np.float32)
        candidates = np.clip(
            self.nominal_u[None, :, :] + noise,
            self.min_control,
            self.max_control,
        ).astype(np.float32)
        costs = self.trajectory_cost_batch(
            state,
            candidates,
            goal,
            obstacles,
            path=path,
            costmap=costmap,
            progress_goal=progress_goal,
        )
        min_cost = float(np.min(costs))
        weights = np.exp(-(costs - min_cost) / max(self.lambda_, 1e-6))
        normalizer = float(np.sum(weights))
        if not np.isfinite(normalizer) or normalizer <= 0.0:
            weights = np.full(self.num_samples, 1.0 / self.num_samples, dtype=np.float32)
            normalizer = 1.0
        else:
            weights = weights / normalizer
        updated_nominal = np.tensordot(weights, candidates, axes=(0, 0)).astype(np.float32)
        self.nominal_u = self._smooth_nominal_update(updated_nominal, previous_nominal)
        self.nominal_u = np.clip(self.nominal_u, self.min_control, self.max_control)
        command = self.nominal_u[0].copy()
        control = self._apply_velocity_response(state, command).astype(np.float32)
        optimal_u = self.nominal_u.copy()
        sample_u = candidates[: self.draw_num_traj].copy()
        self.previous_control = command.copy()
        self._shift_nominal_controls()
        return control, optimal_u, sample_u, normalizer, min_cost

    def _reset_nominal_on_goal_change(self, goal: np.ndarray) -> None:
        goal_pose = np.asarray(goal, dtype=np.float32).reshape(-1)[:3].copy()
        if goal_pose.shape[0] < 3:
            return
        previous_goal = self._last_goal_for_nominal_reset
        self._last_goal_for_nominal_reset = goal_pose
        if previous_goal is None:
            return
        xy_delta = float(np.linalg.norm(goal_pose[:2] - previous_goal[:2]))
        yaw_delta = abs(self._angle_diff(float(goal_pose[2]), float(previous_goal[2])))
        should_reset = (
            self.goal_change_reset_distance > 0.0
            and xy_delta > self.goal_change_reset_distance
        ) or (
            self.goal_change_reset_yaw > 0.0
            and yaw_delta > self.goal_change_reset_yaw
        )
        if should_reset:
            self.nominal_u.fill(0.0)
            self._has_nominal_update = False

    def _smooth_nominal_update(self, updated_nominal: np.ndarray, previous_nominal: np.ndarray) -> np.ndarray:
        updated = np.asarray(updated_nominal, dtype=np.float32)
        if not self._has_nominal_update or not np.any(self.update_smoothing_alpha > 0.0):
            self._has_nominal_update = True
            return updated
        alpha = self.update_smoothing_alpha.reshape(1, 3)
        self._has_nominal_update = True
        return (alpha * np.asarray(previous_nominal, dtype=np.float32) + (1.0 - alpha) * updated).astype(np.float32)

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        return float((a - b + np.pi) % (2.0 * np.pi) - np.pi)

    @staticmethod
    def _control_axis_alpha(value: float | list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
        alpha = np.asarray(value, dtype=np.float32).reshape(-1)
        if alpha.size == 1:
            alpha = np.full(3, float(alpha[0]), dtype=np.float32)
        elif alpha.size != 3:
            raise ValueError("update_smoothing_alpha must be a scalar or a three-element control-axis list")
        return np.clip(alpha, 0.0, 0.95).astype(np.float32)

    @staticmethod
    def _path_from_cost_params(cost_params) -> np.ndarray:
        if len(cost_params) <= 6 or cost_params[6] is None:
            return np.empty((0, 2), dtype=np.float32)
        return np.asarray(cost_params[6], dtype=np.float32).reshape(-1, 2)

    @staticmethod
    def _costmap_from_cost_params(cost_params) -> dict | None:
        if len(cost_params) <= 7:
            return None
        candidate = cost_params[7]
        if not isinstance(candidate, dict) or not bool(candidate.get("enabled", False)):
            return None
        return candidate

    @staticmethod
    def _progress_goal_from_cost_params(cost_params, fallback: np.ndarray) -> np.ndarray:
        if len(cost_params) <= 8 or cost_params[8] is None:
            return np.asarray(fallback, dtype=np.float32)
        return np.asarray(cost_params[8], dtype=np.float32).reshape(-1)

    def trajectory_cost_batch(
        self,
        initial_state: np.ndarray,
        controls: np.ndarray,
        goal: np.ndarray,
        obstacles: np.ndarray,
        path: np.ndarray | None = None,
        costmap: dict | None = None,
        progress_goal: np.ndarray | None = None,
    ) -> np.ndarray:
        controls = np.clip(np.asarray(controls, dtype=np.float32), self.min_control, self.max_control)
        controls = np.ascontiguousarray(controls)
        num_samples = int(controls.shape[0])
        progress_goal_arr = np.asarray(goal if progress_goal is None else progress_goal, dtype=np.float32)
        obstacles = np.ascontiguousarray(np.asarray(obstacles, dtype=np.float32).reshape(-1, 7))
        obstacle_count = int(len(obstacles))
        if obstacle_count == 0:
            obstacles = np.zeros((1, 7), dtype=np.float32)
        path_arr = np.ascontiguousarray(np.asarray(path if path is not None else [], dtype=np.float32).reshape(-1, 2))
        path_count = int(len(path_arr))
        if path_count == 0:
            path_arr = np.zeros((1, 2), dtype=np.float32)
        costmap_data = np.zeros(1, dtype=np.float32)
        costmap_unknown = np.zeros(1, dtype=np.uint8)
        costmap_enabled = 0
        costmap_width = 0
        costmap_height = 0
        costmap_origin = np.zeros(2, dtype=np.float32)
        costmap_resolution = 0.0
        costmap_weight = 0.0
        costmap_power = 2.0
        costmap_unknown_cost = 100.0
        costmap_max_cost = 100.0
        unknown_clear_radius = 0.0
        unknown_clear_value = 0.0
        footprint_enabled = 0
        footprint_radius = 0.0
        footprint_sample_count = 0
        if costmap and bool(costmap.get("enabled", False)):
            width = int(costmap.get("width", 0))
            height = int(costmap.get("height", 0))
            data = np.asarray(costmap.get("data", []), dtype=np.float32).reshape(-1)
            resolution = float(costmap.get("resolution", 0.0))
            if width > 0 and height > 0 and resolution > 0.0 and data.size == width * height:
                costmap_enabled = 1
                costmap_width = width
                costmap_height = height
                costmap_data = np.ascontiguousarray(data.astype(np.float32, copy=False))
                unknown = np.asarray(costmap.get("unknown_mask", np.zeros(data.size, dtype=bool)), dtype=np.uint8).reshape(-1)
                if unknown.size != data.size:
                    unknown = np.zeros(data.size, dtype=np.uint8)
                costmap_unknown = np.ascontiguousarray(unknown)
                costmap_origin = np.asarray(costmap.get("origin", [0.0, 0.0]), dtype=np.float32).reshape(2)
                costmap_resolution = resolution
                costmap_weight = float(costmap.get("weight", 0.0))
                costmap_power = float(costmap.get("power", 2.0))
                costmap_unknown_cost = float(costmap.get("unknown_cost", 100.0))
                costmap_max_cost = max(float(costmap.get("max_cost", 100.0)), 1e-6)
                unknown_clear_radius = float(costmap.get("unknown_clear_radius", 0.0))
                unknown_clear_value = float(costmap.get("unknown_clear_value", 0.0))
                footprint_enabled = 1 if bool(costmap.get("footprint_enabled", False)) else 0
                footprint_radius = float(costmap.get("footprint_radius", 0.0)) + float(
                    costmap.get("footprint_safety_margin", 0.0)
                )
                footprint_sample_count = int(costmap.get("footprint_sample_count", 8))
        costs_gpu = gpuarray.empty(num_samples, dtype=np.float32)
        grid_dim = ((num_samples + self.block_dim - 1) // self.block_dim, 1, 1)
        self.cost_kernel(
            cuda.In(np.asarray(initial_state, dtype=np.float32)),
            cuda.In(controls),
            cuda.In(np.asarray(self.previous_control, dtype=np.float32)),
            cuda.In(np.asarray(goal, dtype=np.float32)),
            cuda.In(progress_goal_arr),
            cuda.In(obstacles),
            cuda.In(path_arr),
            cuda.In(costmap_data),
            cuda.In(costmap_unknown),
            costs_gpu,
            np.int32(num_samples),
            np.int32(self.horizon_steps),
            np.int32(obstacle_count),
            np.float32(self.dt),
            np.float32(self.min_control[0]),
            np.float32(self.max_control[0]),
            np.float32(self.max_control[1]),
            np.float32(self.max_control[2]),
            np.float32(self.goal_xy_weight),
            np.float32(self.yaw_weight),
            np.float32(self.control_weight),
            np.float32(self.smooth_weight),
            np.float32(self.lateral_weight),
            np.float32(self.yaw_rate_weight),
            np.float32(self.accel_weight),
            np.float32(self.jerk_weight),
            np.float32(self.obstacle_weight),
            np.float32(self.obstacle_collision_weight),
            np.float32(self.obstacle_soft_weight),
            np.float32(self.obstacle_influence_dist),
            np.float32(self.cbf_weight),
            np.float32(self.cbf_alpha),
            np.int32(self.cbf_type),
            np.float32(self.atau),
            np.float32(self.robot_radius),
            np.float32(self.safety_dist),
            np.float32(self.max_accel[0]),
            np.float32(self.max_accel[1]),
            np.float32(self.max_accel[2]),
            np.float32(self.velocity_lag_beta),
            np.float32(self.goal_progress_weight),
            np.float32(self.heading_to_goal_weight),
            np.float32(self.heading_to_goal_min_distance),
            np.int32(path_count),
            np.float32(self.path_tracking_weight),
            np.float32(self.path_tracking_tolerance),
            np.float32(self.path_progress_weight),
            np.int32(costmap_enabled),
            np.int32(costmap_width),
            np.int32(costmap_height),
            np.float32(costmap_origin[0]),
            np.float32(costmap_origin[1]),
            np.float32(costmap_resolution),
            np.float32(costmap_weight),
            np.float32(costmap_power),
            np.float32(costmap_unknown_cost),
            np.float32(costmap_max_cost),
            np.float32(unknown_clear_radius),
            np.float32(unknown_clear_value),
            np.int32(footprint_enabled),
            np.float32(footprint_radius),
            np.int32(footprint_sample_count),
            block=(self.block_dim, 1, 1),
            grid=grid_dim,
        )
        return costs_gpu.get()

    def _shift_nominal_controls(self) -> None:
        self.nominal_u[:-1] = self.nominal_u[1:]
        self.nominal_u[-1] = 0.0

    def _apply_velocity_response(self, state: np.ndarray, command: np.ndarray) -> np.ndarray:
        prev_real = np.asarray(state, dtype=np.float32)[3:]
        command = np.clip(np.asarray(command, dtype=np.float32), self.min_control, self.max_control)
        lagged = self.velocity_lag_beta * prev_real + (1.0 - self.velocity_lag_beta) * command
        delta = np.clip(lagged - prev_real, -self.max_accel * self.dt, self.max_accel * self.dt)
        return np.clip(prev_real + delta, self.min_control, self.max_control)
