#include "swerve.h"

int main() {
    Swerve env = {0};
    
    // Initialize observation and action buffers
    float observations[24] = {0}; // 12 observations per agent * 2 agents
    float actions[6] = {0};       // 3 actions per agent * 2 agents  
    float rewards[2] = {0};       // 1 reward per agent
    unsigned char terminals[2] = {0}; // 1 terminal per agent
    
    env.observations = observations;
    env.actions = actions;
    env.rewards = rewards;
    env.terminals = terminals;
    env.max_episode_steps = 1000;
    
    c_reset(&env);
    c_render(&env);
    
    while (!WindowShouldClose()) {
        // Simple keyboard controls for testing
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            // Manual control for robot 1
            if (IsKeyDown(KEY_A) || IsKeyDown(KEY_LEFT)) {
                env.actions[0] = -1.0f; // move left
                env.actions[1] = 0.0f;
            } else if (IsKeyDown(KEY_D) || IsKeyDown(KEY_RIGHT)) {
                env.actions[0] = 1.0f; // move right
                env.actions[1] = 0.0f;
            } else if (IsKeyDown(KEY_W) || IsKeyDown(KEY_UP)) {
                env.actions[0] = 0.0f;
                env.actions[1] = -1.0f; // move up
            } else if (IsKeyDown(KEY_S) || IsKeyDown(KEY_DOWN)) {
                env.actions[0] = 0.0f;
                env.actions[1] = 1.0f; // move down
            } else if (IsKeyDown(KEY_Q)) {
                env.actions[2] = -1.0f; // rotate left
            } else if (IsKeyDown(KEY_E)) {
                env.actions[2] = 1.0f; // rotate right
            } else {
                env.actions[0] = 0.0f;
                env.actions[1] = 0.0f;
                env.actions[2] = 0.0f;
            }
            
            // Manual control for robot 2 with arrow keys
            if (IsKeyDown(KEY_J)) {
                env.actions[3] = -1.0f; // move left
                env.actions[4] = 0.0f;
            } else if (IsKeyDown(KEY_L)) {
                env.actions[3] = 1.0f; // move right
                env.actions[4] = 0.0f;
            } else if (IsKeyDown(KEY_I)) {
                env.actions[3] = 0.0f;
                env.actions[4] = -1.0f; // move up
            } else if (IsKeyDown(KEY_K)) {
                env.actions[3] = 0.0f;
                env.actions[4] = 1.0f; // move down
            } else if (IsKeyDown(KEY_U)) {
                env.actions[5] = -1.0f; // rotate left
            } else if (IsKeyDown(KEY_O)) {
                env.actions[5] = 1.0f; // rotate right
            } else {
                env.actions[3] = 0.0f;
                env.actions[4] = 0.0f;
                env.actions[5] = 0.0f;
            }
        } else {
            // Random actions for both robots
            for (int i = 0; i < 6; i++) {
                env.actions[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
            }
        }
        
        c_step(&env);
        c_render(&env);
        
        // Reset if episode is done
        if (env.terminals[0] || env.terminals[1]) {
            printf("Episode finished! Goals: %.0f, Collisions: %.0f, Score: %.0f\n", 
                   env.log.goals_reached, env.log.collisions, env.log.score);
            c_reset(&env);
        }
    }
    
    c_close(&env);
    return 0;
}
