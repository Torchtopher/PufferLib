#include "swerve.h"

#define Env Swerve 
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    // Check if max_episode_steps is provided, otherwise use default
    PyObject* steps_val = PyDict_GetItemString(kwargs, "max_episode_steps");
    if (steps_val != NULL) {
        env->max_episode_steps = unpack(kwargs, "max_episode_steps");
    } else {
        env->max_episode_steps = 1000;  // default value
    }
    env->episode_steps = 0;
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "collisions", log->collisions);
    assign_to_dict(dict, "goals_reached", log->goals_reached);
    assign_to_dict(dict, "score", log->score);
    return 0;
}
