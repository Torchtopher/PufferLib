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
#define GOAL_REWARD 1.0f
#define DISTANCE_REWARD_SCALE 0.1f

// Only use floats!
typedef struct {
    float episode_return;
    float episode_length;
    float collisions;
    float goals_reached;
    float score;
    float n; // Required as the last field 
} Log;

typedef struct {
    float x, y, theta;
    float vx, vy, omega;
} Robot;

typedef struct {
    float x, y;
} Goal;

typedef struct {
    Log log;                     // Required field
    float* observations;         // Required field. Changed to float for continuous obs
    float* actions;              // Required field. Changed to float for continuous actions
    float* rewards;              // Required field
    unsigned char* terminals;    // Required field
    Robot robots[2];            // Two swerve drive robots
    Goal goals[2];              // Two goal positions
    float prev_distances[2];    // Previous distances to goals for reward shaping
    int episode_steps;
    int max_episode_steps;
} Swerve;

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
    float dist = distance(robot->x, robot->y, goal->x, goal->y);
    return dist < ROBOT_RADIUS;
}

void generate_random_goals(Swerve* env) {
    // Generate two goals that are far apart
    do {
        env->goals[0].x = ((float)rand() / RAND_MAX - 0.5f) * FIELD_WIDTH * 0.8f;
        env->goals[0].y = ((float)rand() / RAND_MAX - 0.5f) * FIELD_HEIGHT * 0.8f;
        env->goals[1].x = ((float)rand() / RAND_MAX - 0.5f) * FIELD_WIDTH * 0.8f;
        env->goals[1].y = ((float)rand() / RAND_MAX - 0.5f) * FIELD_HEIGHT * 0.8f;
    } while (distance(env->goals[0].x, env->goals[0].y, env->goals[1].x, env->goals[1].y) < 3.0f);
}

void c_reset(Swerve* env) {
    // Reset robots to starting positions
    env->robots[0].x = -2.0f;
    env->robots[0].y = 0.0f;
    env->robots[0].theta = 0.0f;
    env->robots[0].vx = 0.0f;
    env->robots[0].vy = 0.0f;
    env->robots[0].omega = 0.0f;
    
    env->robots[1].x = 2.0f;
    env->robots[1].y = 0.0f;
    env->robots[1].theta = 0.0f;
    env->robots[1].vx = 0.0f;
    env->robots[1].vy = 0.0f;
    env->robots[1].omega = 0.0f;
    
    // Generate new goals
    generate_random_goals(env);
    
    // Store initial distances for reward shaping
    env->prev_distances[0] = distance(env->robots[0].x, env->robots[0].y, env->goals[0].x, env->goals[0].y);
    env->prev_distances[1] = distance(env->robots[1].x, env->robots[1].y, env->goals[1].x, env->goals[1].y);
    
    env->episode_steps = 0;
    
    // Update observations
    // Robot 1: [x, y, theta, vx, vy, omega, goal_x, goal_y, other_robot_x, other_robot_y, other_goal_x, other_goal_y]
    env->observations[0] = env->robots[0].x / FIELD_WIDTH;
    env->observations[1] = env->robots[0].y / FIELD_HEIGHT;
    env->observations[2] = env->robots[0].theta / (2 * M_PI);
    env->observations[3] = env->robots[0].vx / MAX_VEL;
    env->observations[4] = env->robots[0].vy / MAX_VEL;
    env->observations[5] = env->robots[0].omega / MAX_OMEGA;
    env->observations[6] = env->goals[0].x / FIELD_WIDTH;
    env->observations[7] = env->goals[0].y / FIELD_HEIGHT;
    env->observations[8] = env->robots[1].x / FIELD_WIDTH;
    env->observations[9] = env->robots[1].y / FIELD_HEIGHT;
    env->observations[10] = env->goals[1].x / FIELD_WIDTH;
    env->observations[11] = env->goals[1].y / FIELD_HEIGHT;
    
    // Robot 2: [x, y, theta, vx, vy, omega, goal_x, goal_y, other_robot_x, other_robot_y, other_goal_x, other_goal_y]
    env->observations[12] = env->robots[1].x / FIELD_WIDTH;
    env->observations[13] = env->robots[1].y / FIELD_HEIGHT;
    env->observations[14] = env->robots[1].theta / (2 * M_PI);
    env->observations[15] = env->robots[1].vx / MAX_VEL;
    env->observations[16] = env->robots[1].vy / MAX_VEL;
    env->observations[17] = env->robots[1].omega / MAX_OMEGA;
    env->observations[18] = env->goals[1].x / FIELD_WIDTH;
    env->observations[19] = env->goals[1].y / FIELD_HEIGHT;
    env->observations[20] = env->robots[0].x / FIELD_WIDTH;
    env->observations[21] = env->robots[0].y / FIELD_HEIGHT;
    env->observations[22] = env->goals[0].x / FIELD_WIDTH;
    env->observations[23] = env->goals[0].y / FIELD_HEIGHT;
    
    // Reset rewards and terminals
    env->rewards[0] = 0.0f;
    env->rewards[1] = 0.0f;
    env->terminals[0] = 0;
    env->terminals[1] = 0;
}

void c_step(Swerve* env) {
    env->episode_steps++;
    
    // Apply actions to robots
    for (int i = 0; i < 2; i++) {
        Robot* robot = &env->robots[i];
        
        // Extract actions: [x_vel, y_vel, omega] for each robot
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
    
    // Calculate rewards
    env->rewards[0] = 0.0f;
    env->rewards[1] = 0.0f;
    env->terminals[0] = 0;
    env->terminals[1] = 0;
    
    // Check for collisions
    bool collision = check_collision(&env->robots[0], &env->robots[1]);
    if (collision) {
        env->rewards[0] += COLLISION_PENALTY;
        env->rewards[1] += COLLISION_PENALTY;
        env->log.collisions += 1.0f;
        env->terminals[0] = 1;
        env->terminals[1] = 1;
    }
    
    // Check goal reaching and distance-based rewards
    for (int i = 0; i < 2; i++) {
        float current_distance = distance(env->robots[i].x, env->robots[i].y, env->goals[i].x, env->goals[i].y);
        
        // Distance-based reward shaping
        float distance_reward = (env->prev_distances[i] - current_distance) * DISTANCE_REWARD_SCALE;
        env->rewards[i] += distance_reward;
        env->prev_distances[i] = current_distance;
        
        // Goal reaching reward
        if (check_goal_reached(&env->robots[i], &env->goals[i])) {
            env->rewards[i] += GOAL_REWARD;
            env->log.goals_reached += 1.0f;
            env->terminals[0] = 1;
            env->terminals[1] = 1;
        }
    }
    
    // Episode timeout
    if (env->episode_steps >= env->max_episode_steps) {
        env->terminals[0] = 1;
        env->terminals[1] = 1;
    }
    
    // Update episode stats
    env->log.episode_return += env->rewards[0] + env->rewards[1];
    env->log.episode_length = (float)env->episode_steps;
    env->log.score = env->log.goals_reached - env->log.collisions;
    
    // Update observations
    env->observations[0] = env->robots[0].x / FIELD_WIDTH;
    env->observations[1] = env->robots[0].y / FIELD_HEIGHT;
    env->observations[2] = env->robots[0].theta / (2 * M_PI);
    env->observations[3] = env->robots[0].vx / MAX_VEL;
    env->observations[4] = env->robots[0].vy / MAX_VEL;
    env->observations[5] = env->robots[0].omega / MAX_OMEGA;
    env->observations[6] = env->goals[0].x / FIELD_WIDTH;
    env->observations[7] = env->goals[0].y / FIELD_HEIGHT;
    env->observations[8] = env->robots[1].x / FIELD_WIDTH;
    env->observations[9] = env->robots[1].y / FIELD_HEIGHT;
    env->observations[10] = env->goals[1].x / FIELD_WIDTH;
    env->observations[11] = env->goals[1].y / FIELD_HEIGHT;
    
    env->observations[12] = env->robots[1].x / FIELD_WIDTH;
    env->observations[13] = env->robots[1].y / FIELD_HEIGHT;
    env->observations[14] = env->robots[1].theta / (2 * M_PI);
    env->observations[15] = env->robots[1].vx / MAX_VEL;
    env->observations[16] = env->robots[1].vy / MAX_VEL;
    env->observations[17] = env->robots[1].omega / MAX_OMEGA;
    env->observations[18] = env->goals[1].x / FIELD_WIDTH;
    env->observations[19] = env->goals[1].y / FIELD_HEIGHT;
    env->observations[20] = env->robots[0].x / FIELD_WIDTH;
    env->observations[21] = env->robots[0].y / FIELD_HEIGHT;
    env->observations[22] = env->goals[0].x / FIELD_WIDTH;
    env->observations[23] = env->goals[0].y / FIELD_HEIGHT;
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
    
    // Draw goals
    DrawCircle(center_x + env->goals[0].x * scale_x, center_y + env->goals[0].y * scale_y, 
               ROBOT_RADIUS * scale_x, PUFF_RED);
    DrawCircle(center_x + env->goals[1].x * scale_x, center_y + env->goals[1].y * scale_y, 
               ROBOT_RADIUS * scale_x, PUFF_YELLOW);
    
    // Draw robots
    for (int i = 0; i < 2; i++) {
        Robot* robot = &env->robots[i];
        Color robot_color = (i == 0) ? PUFF_CYAN : PUFF_GREEN;
        
        float robot_x = center_x + robot->x * scale_x;
        float robot_y = center_y + robot->y * scale_y;
        
        // Draw robot body
        DrawCircle(robot_x, robot_y, ROBOT_RADIUS * scale_x, robot_color);
        
        // Draw robot orientation
        float dir_x = robot_x + cosf(robot->theta) * ROBOT_RADIUS * scale_x;
        float dir_y = robot_y + sinf(robot->theta) * ROBOT_RADIUS * scale_y;
        DrawLine(robot_x, robot_y, dir_x, dir_y, PUFF_WHITE);
    }
    
    // Draw UI text
    DrawText("Swerve Environment - Blue/Green robots, Red/Yellow goals", 10, 10, 20, PUFF_WHITE);
    DrawText(TextFormat("Episode: %d/%d", env->episode_steps, env->max_episode_steps), 10, 40, 16, PUFF_WHITE);
    DrawText(TextFormat("Goals: %.0f, Collisions: %.0f", env->log.goals_reached, env->log.collisions), 10, 60, 16, PUFF_WHITE);
    DrawText(TextFormat("Score: %.0f", env->log.score), 10, 80, 16, PUFF_WHITE);
    
    EndDrawing();
}

void c_close(Swerve* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}
