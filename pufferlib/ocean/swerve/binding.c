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
    
    // Check if num_agents is provided, otherwise use default
    PyObject* agents_val = PyDict_GetItemString(kwargs, "num_agents");
    if (agents_val != NULL) {
        env->num_agents = unpack(kwargs, "num_agents");
    } else {
        env->num_agents = 3;  // default value
    }
    
    env->episode_steps = 0;
    init(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    return 0;
}
