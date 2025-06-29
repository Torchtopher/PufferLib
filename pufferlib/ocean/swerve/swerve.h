#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include "raylib.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_GREEN = (Color){0, 187, 0, 255};
const Color PUFF_YELLOW = (Color){187, 187, 0, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};

#define FIELD_WIDTH 10.0f
#define FIELD_HEIGHT 10.0f
#define ROBOT_RADIUS 0.3f
#define MAX_VEL 2.0f
#define MAX_OMEGA 3.14f
#define DT 0.02f
#define COLLISION_PENALTY -1.0f
#define GOAL_REWARD 0.2f
#define DISTANCE_REWARD_SCALE 0.1f
#define COLLISION_AVOIDANCE_REWARD_SCALE 0.2f
#define SAFE_DISTANCE 1.0f
#define PROGRESS_REWARD_SCALE 0.05f
#define ANGULAR_PROGRESS_REWARD_SCALE 0.03f

// Only use floats!
typedef struct {
    float perf; // Recommended 0-1 normalized single real number perf metric
    float score; // Recommended unnormalized single real number perf metric
    float episode_return; // Recommended metric: sum of agent rewards over episode
    float episode_length; // Recommended metric: number of steps of agent episode
    // Any extra fields you add here may be exported to Python in binding.c
    float n; // Required as the last field 
} Log;

typedef struct {
    float x, y, theta;
    float vx, vy, omega;
    int ticks_since_reward;
    float prev_distance;        // Previous distance to goal for reward shaping
    float prev_angle_diff;      // Previous angle difference to goal for angular reward shaping
} Robot;

typedef struct {
    float x, y, theta; // Goal now includes orientation
} Goal;

typedef struct {
    Log log;                     // Required field
    float* observations;         // Required field. Changed to float for continuous obs
    float* actions;              // Required field. Changed to float for continuous actions
    float* rewards;              // Required field
    unsigned char* terminals;    // Required field
    Robot* robots;               // Dynamic array of swerve drive robots
    Goal* goals;                 // Dynamic array of goal positions with orientations
    int num_agents;              // Number of agents (robots)
    int episode_steps;
    int max_episode_steps;
} Swerve;

void init(Swerve* env) {
    env->robots = calloc(env->num_agents, sizeof(Robot));
    env->goals = calloc(env->num_agents, sizeof(Goal));
}

float distance(float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    return sqrtf(dx*dx + dy*dy);
}

bool check_collision(Robot* r1, Robot* r2) {
    float dist = distance(r1->x, r1->y, r2->x, r2->y);
    return dist < (2 * ROBOT_RADIUS);
}

bool check_goal_reached(Robot* robot, Goal* goal) {
    float pos_dist = distance(robot->x, robot->y, goal->x, goal->y);
    
    // Check position tolerance
    if (pos_dist > ROBOT_RADIUS) {
        return false;
    }
    
    // Check orientation tolerance (within 15 degrees = 0.26 radians)
    float angle_diff = fabs(robot->theta - goal->theta);
    
    // Handle angle wraparound (e.g., difference between 0.1 and 6.2 radians)
    if (angle_diff > M_PI) {
        angle_diff = 2 * M_PI - angle_diff;
    }
    
    return angle_diff < 0.13f; // 7 degrees tolerance
}

void generate_random_goals(Swerve* env) {
    // Generate goals that are far apart from each other
    for (int i = 0; i < env->num_agents; i++) {
        bool valid_goal = false;
        int attempts = 0;
        
        while (!valid_goal && attempts < 100) {
            env->goals[i].x = ((float)rand() / RAND_MAX - 0.5f) * FIELD_WIDTH * 0.8f;
            env->goals[i].y = ((float)rand() / RAND_MAX - 0.5f) * FIELD_HEIGHT * 0.8f;
            env->goals[i].theta = ((float)rand() / RAND_MAX) * 2 * M_PI; // Random orientation
            
            // Check if this goal is far enough from other goals
            valid_goal = true;
            for (int j = 0; j < i; j++) {
                if (distance(env->goals[i].x, env->goals[i].y, env->goals[j].x, env->goals[j].y) < 2.5f) {
                    valid_goal = false;
                    break;
                }
            }
            attempts++;
        }
    }
}

void c_reset(Swerve* env) {
    // Reset robots to random starting positions
    for (int i = 0; i < env->num_agents; i++) {
        bool valid_position = false;
        int attempts = 0;
        
        while (!valid_position && attempts < 100) {
            // Random position within field bounds
            env->robots[i].x = ((float)rand() / RAND_MAX - 0.5f) * FIELD_WIDTH * 0.8f;
            env->robots[i].y = ((float)rand() / RAND_MAX - 0.5f) * FIELD_HEIGHT * 0.8f;
            env->robots[i].theta = ((float)rand() / RAND_MAX) * 2 * M_PI; // Random orientation
            
            // Check if this robot is far enough from other robots
            valid_position = true;
            for (int j = 0; j < i; j++) {
                if (distance(env->robots[i].x, env->robots[i].y, env->robots[j].x, env->robots[j].y) < 1.5f) {
                    valid_position = false;
                    break;
                }
            }
            attempts++;
        }
        
        // Reset velocities and tracking
        env->robots[i].vx = 0.0f;
        env->robots[i].vy = 0.0f;
        env->robots[i].omega = 0.0f;
        env->robots[i].ticks_since_reward = 0;
        
        // Reset rewards and terminals for this agent
        env->rewards[i] = 0.0f;
        env->terminals[i] = 0;
    }
    
    // Generate new goals
    generate_random_goals(env);
    
    // Store initial distances and angle differences for reward shaping
    for (int i = 0; i < env->num_agents; i++) {
        env->robots[i].prev_distance = distance(env->robots[i].x, env->robots[i].y, env->goals[i].x, env->goals[i].y);
        
        // Calculate initial angle difference
        float angle_diff = env->goals[i].theta - env->robots[i].theta;
        while (angle_diff > M_PI) angle_diff -= 2 * M_PI;
        while (angle_diff < -M_PI) angle_diff += 2 * M_PI;
        env->robots[i].prev_angle_diff = fabs(angle_diff);
    }
    
    env->episode_steps = 0;
    
    update_observations(env);
}

void c_step(Swerve* env) {
    env->episode_steps++;
    
    // Zero out rewards and terminals at start of step (critical for training)
    for (int i = 0; i < env->num_agents; i++) {
        env->rewards[i] = 0.0f;
        env->terminals[i] = 0;
        env->robots[i].ticks_since_reward++;
    }
    
    // Apply actions to robots - each agent has 3 actions: [x_vel, y_vel, omega]
    for (int i = 0; i < env->num_agents; i++) {
        Robot* robot = &env->robots[i];
        
        // Extract actions for this agent
        float vx_cmd = env->actions[i * 3 + 0];
        float vy_cmd = env->actions[i * 3 + 1];
        float omega_cmd = env->actions[i * 3 + 2];
        
        // Clamp actions to valid ranges
        vx_cmd = fmaxf(-MAX_VEL, fminf(MAX_VEL, vx_cmd));
        vy_cmd = fmaxf(-MAX_VEL, fminf(MAX_VEL, vy_cmd));
        omega_cmd = fmaxf(-MAX_OMEGA, fminf(MAX_OMEGA, omega_cmd));
        
        // Update robot velocities (simple model)
        robot->vx = vx_cmd;
        robot->vy = vy_cmd;
        robot->omega = omega_cmd;
        
        // Update robot position and orientation
        robot->x += robot->vx * DT;
        robot->y += robot->vy * DT;
        robot->theta += robot->omega * DT;
        
        // Keep theta in [0, 2*PI]
        while (robot->theta < 0) robot->theta += 2 * M_PI;
        while (robot->theta >= 2 * M_PI) robot->theta -= 2 * M_PI;
        
        // Keep robots within field bounds
        robot->x = fmaxf(-FIELD_WIDTH/2, fminf(FIELD_WIDTH/2, robot->x));
        robot->y = fmaxf(-FIELD_HEIGHT/2, fminf(FIELD_HEIGHT/2, robot->y));
    }
    
    // Process each agent individually
    for (int i = 0; i < env->num_agents; i++) {
        Robot* robot = &env->robots[i];
        Goal* goal = &env->goals[i];
        
        // Check for collisions with other robots - collision affects this agent
        bool collision = false;
        for (int j = 0; j < env->num_agents; j++) {
            if (i != j && check_collision(robot, &env->robots[j])) {
                collision = true;
                break;
            }
        }
        
        if (collision) {
            env->rewards[i] += COLLISION_PENALTY * 20; // Strong penalty for collision
            env->terminals[i] = 1; // This agent's episode ends
            env->log.score += COLLISION_PENALTY; // Penalize score for collision
            env->log.episode_return += COLLISION_PENALTY * 20;
            continue; // Skip other processing for this agent
        }
        
        // Add collision avoidance rewards - incentivize maintaining safe distances
        float collision_avoidance_reward = 0.0f;
        for (int j = 0; j < env->num_agents; j++) {
            if (i != j) {
                float dist = distance(robot->x, robot->y, env->robots[j].x, env->robots[j].y);
                
                // Reward for maintaining safe distance, penalty for getting too close
                if (dist < SAFE_DISTANCE) {
                    // Penalty increases as robots get closer
                    float danger_level = (SAFE_DISTANCE - dist) / SAFE_DISTANCE;
                    collision_avoidance_reward -= danger_level * COLLISION_AVOIDANCE_REWARD_SCALE;
                } else if (dist < SAFE_DISTANCE * 1.5f) {
                    // Small reward for maintaining good distance
                    collision_avoidance_reward += 0.01f;
                }
            }
        }
        env->rewards[i] += collision_avoidance_reward;
        
        // Distance-based reward shaping
        float current_distance = distance(robot->x, robot->y, goal->x, goal->y);
        float distance_reward = (robot->prev_distance - current_distance) * DISTANCE_REWARD_SCALE;
        env->rewards[i] += distance_reward;
        robot->prev_distance = current_distance;
        
        // Angular progress reward - reward for getting orientation closer to goal
        float angle_diff = goal->theta - robot->theta;
        while (angle_diff > M_PI) angle_diff -= 2 * M_PI;
        while (angle_diff < -M_PI) angle_diff += 2 * M_PI;
        float current_angle_diff = fabs(angle_diff);
        
        float angular_reward = (robot->prev_angle_diff - current_angle_diff) * ANGULAR_PROGRESS_REWARD_SCALE;
        env->rewards[i] += angular_reward;
        robot->prev_angle_diff = current_angle_diff;
        
        // Additional progress reward based on velocity toward goal
        float dx_to_goal = goal->x - robot->x;
        float dy_to_goal = goal->y - robot->y;
        float dist_to_goal = sqrtf(dx_to_goal * dx_to_goal + dy_to_goal * dy_to_goal);
        
        if (dist_to_goal > 0.1f) { // Avoid division by zero
            // Normalize direction vector to goal
            float dx_normalized = dx_to_goal / dist_to_goal;
            float dy_normalized = dy_to_goal / dist_to_goal;
            
            // Dot product of velocity with direction to goal
            float velocity_toward_goal = robot->vx * dx_normalized + robot->vy * dy_normalized;
            
            // Reward for moving toward goal, scaled by distance (closer = more reward)
            float distance_scale = 1.0f / (1.0f + dist_to_goal * 0.2f); // Diminishes with distance
            env->rewards[i] += velocity_toward_goal * PROGRESS_REWARD_SCALE * distance_scale;
        }
        
        // Goal reaching reward
        if (check_goal_reached(robot, goal)) {
            env->rewards[i] += GOAL_REWARD;
            env->log.perf += 1.0f;
            env->log.score += 1.0f;
            env->log.episode_length += robot->ticks_since_reward;
            env->log.episode_return += GOAL_REWARD;
            env->log.n++;
            robot->ticks_since_reward = 0;
            
            // Generate a new goal for this robot
            int attempts = 0;
            do {
                goal->x = ((float)rand() / RAND_MAX - 0.5f) * FIELD_WIDTH * 0.8f;
                goal->y = ((float)rand() / RAND_MAX - 0.5f) * FIELD_HEIGHT * 0.8f;
                goal->theta = ((float)rand() / RAND_MAX) * 2 * M_PI;
                attempts++;
            } while (distance(goal->x, goal->y, robot->x, robot->y) < 2.0f && attempts < 50);
            
            // Update the distance tracking for the new goal
            robot->prev_distance = distance(robot->x, robot->y, goal->x, goal->y);
            
            // Update angle difference tracking for the new goal
            angle_diff = goal->theta - robot->theta;
            while (angle_diff > M_PI) angle_diff -= 2 * M_PI;
            while (angle_diff < -M_PI) angle_diff += 2 * M_PI;
            robot->prev_angle_diff = fabs(angle_diff);
        }
        
        // Episode timeout for this agent
        if (env->episode_steps >= env->max_episode_steps) {
            env->terminals[i] = 1;
        }
        
        // Reset agent if stuck (no reward for too long)
        if (robot->ticks_since_reward >= 500) {
            robot->x = ((float)rand() / RAND_MAX - 0.5f) * FIELD_WIDTH * 0.8f;
            robot->y = ((float)rand() / RAND_MAX - 0.5f) * FIELD_HEIGHT * 0.8f;
            robot->theta = ((float)rand() / RAND_MAX) * 2 * M_PI;
            robot->ticks_since_reward = 0;
        }
        
        // Update per-agent episode stats
        env->log.episode_return += env->rewards[i];
    }
    
    update_observations(env);
}

void update_observations(Swerve* env) {
    // Enhanced observation space: 14 observations per robot
    // Each robot observes:
    // - Relative position to own goal (2): dx, dy
    // - Relative angle to own goal orientation (1): angle_diff
    // - Own velocity (3): vx, vy, omega
    // - Enhanced info about other robots (8): (num_agents-1) other robots * (dx, dy, dvx, dvy) each
    
    for (int i = 0; i < env->num_agents; i++) {
        int obs_offset = i * 14;
        Robot* robot = &env->robots[i];
        Goal* goal = &env->goals[i];
        
        // Relative position to own goal (normalized)
        float dx_goal = (goal->x - robot->x) / FIELD_WIDTH;
        float dy_goal = (goal->y - robot->y) / FIELD_HEIGHT;
        env->observations[obs_offset + 0] = dx_goal;
        env->observations[obs_offset + 1] = dy_goal;
        
        // Relative angle to goal orientation (normalized to [-1, 1])
        float angle_diff = goal->theta - robot->theta;
        // Normalize angle difference to [-pi, pi]
        while (angle_diff > M_PI) angle_diff -= 2 * M_PI;
        while (angle_diff < -M_PI) angle_diff += 2 * M_PI;
        env->observations[obs_offset + 2] = angle_diff / M_PI;
        
        // Own velocity (normalized)
        env->observations[obs_offset + 3] = robot->vx / MAX_VEL;
        env->observations[obs_offset + 4] = robot->vy / MAX_VEL;
        env->observations[obs_offset + 5] = robot->omega / MAX_OMEGA;
        
        // Enhanced info about other robots (position + velocity)
        int other_idx = 0;
        for (int j = 0; j < env->num_agents && other_idx < 2; j++) {
            if (j != i) {
                float dx_other = (env->robots[j].x - robot->x) / FIELD_WIDTH;
                float dy_other = (env->robots[j].y - robot->y) / FIELD_HEIGHT;
                float dvx_other = (env->robots[j].vx - robot->vx) / MAX_VEL;
                float dvy_other = (env->robots[j].vy - robot->vy) / MAX_VEL;
                
                env->observations[obs_offset + 6 + other_idx * 4] = dx_other;
                env->observations[obs_offset + 7 + other_idx * 4] = dy_other;
                env->observations[obs_offset + 8 + other_idx * 4] = dvx_other;
                env->observations[obs_offset + 9 + other_idx * 4] = dvy_other;
                other_idx++;
            }
        }
        
        // If there are fewer than 2 other robots, pad with zeros
        while (other_idx < 2) {
            env->observations[obs_offset + 6 + other_idx * 4] = 0.0f;
            env->observations[obs_offset + 7 + other_idx * 4] = 0.0f;
            env->observations[obs_offset + 8 + other_idx * 4] = 0.0f;
            env->observations[obs_offset + 9 + other_idx * 4] = 0.0f;
            other_idx++;
        }
    }
}

void c_render(Swerve* env) {
    if (!IsWindowReady()) {
        InitWindow(1080, 720, "PufferLib Swerve Environment");
        SetTargetFPS(50);
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);
    
    // Scale factor for rendering
    float scale_x = 1080.0f / FIELD_WIDTH;
    float scale_y = 720.0f / FIELD_HEIGHT;
    float center_x = 540.0f;
    float center_y = 360.0f;
    
    // Draw field boundaries
    DrawRectangleLines(center_x - FIELD_WIDTH * scale_x / 2, center_y - FIELD_HEIGHT * scale_y / 2,
                      FIELD_WIDTH * scale_x, FIELD_HEIGHT * scale_y, PUFF_WHITE);
    
    // Draw goals with orientation indicators
    Color goal_colors[] = {PUFF_CYAN, PUFF_GREEN, {255, 165, 0, 255}, PUFF_RED, PUFF_YELLOW}; // Support up to 5 colors
    for (int i = 0; i < env->num_agents; i++) {
        float goal_x = center_x + env->goals[i].x * scale_x;
        float goal_y = center_y + env->goals[i].y * scale_y;
        
        // Draw goal circle
        Color color = goal_colors[i % 5]; // Cycle through colors if more than 5 agents
        DrawCircle(goal_x, goal_y, ROBOT_RADIUS * scale_x, color);
        
        // Draw goal orientation arrow
        float goal_dir_x = goal_x + cosf(env->goals[i].theta) * ROBOT_RADIUS * scale_x;
        float goal_dir_y = goal_y + sinf(env->goals[i].theta) * ROBOT_RADIUS * scale_y;
        DrawLine(goal_x, goal_y, goal_dir_x, goal_dir_y, PUFF_WHITE);
        DrawCircle(goal_dir_x, goal_dir_y, 2, PUFF_WHITE);
    }
    
    // Draw robots as squares
    Color robot_colors[] = {PUFF_CYAN, PUFF_GREEN, {255, 165, 0, 255}, PUFF_RED, PUFF_YELLOW}; // Support up to 5 colors
    for (int i = 0; i < env->num_agents; i++) {
        Robot* robot = &env->robots[i];
        
        float robot_x = center_x + robot->x * scale_x;
        float robot_y = center_y + robot->y * scale_y;
        float robot_size = ROBOT_RADIUS * scale_x * 2; // Make square size based on robot diameter
        
        // Draw robot body as square
        Color color = robot_colors[i % 5]; // Cycle through colors if more than 5 agents
        DrawRectangle(robot_x - robot_size/2, robot_y - robot_size/2, robot_size, robot_size, color);
        
        // Draw robot orientation arrow
        float dir_x = robot_x + cosf(robot->theta) * ROBOT_RADIUS * scale_x;
        float dir_y = robot_y + sinf(robot->theta) * ROBOT_RADIUS * scale_y;
        DrawLine(robot_x, robot_y, dir_x, dir_y, PUFF_WHITE);
        
        // Draw a small circle at the front to show direction more clearly
        DrawCircle(dir_x, dir_y, 3, PUFF_WHITE);
    }
    
    // Draw UI text
    DrawText(TextFormat("Swerve Environment - %d robots with position+orientation goals", env->num_agents), 10, 10, 20, PUFF_WHITE);
    DrawText(TextFormat("Episode: %d/%d", env->episode_steps, env->max_episode_steps), 10, 40, 16, PUFF_WHITE);
    DrawText(TextFormat("Perf: %.1f, Score: %.1f", env->log.perf, env->log.score), 10, 60, 16, PUFF_WHITE);
    DrawText(TextFormat("Episode Return: %.1f", env->log.episode_return), 10, 80, 16, PUFF_WHITE);
    
    EndDrawing();
}

void c_close(Swerve* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
    // Free allocated memory
    if (env->robots) {
        free(env->robots);
        env->robots = NULL;
    }
    if (env->goals) {
        free(env->goals);
        env->goals = NULL;
    }
}
