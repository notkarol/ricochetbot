#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct move_t {
  int64_t robot;
  int64_t action;
  int64_t src_x;
  int64_t src_y;
  int64_t dst_x;
  int64_t dst_y;
  int64_t robot_order_i;
} move_t;

static const char *MARKERS[35] = {"wu", "wd", "wl", "wr", "b", "g", "r", "y", "k",
				  "bi", "bd", "gi", "gd", "ri", "rd", "yi", "yd",
				  "bo", "bv", "bs", "bh", "go", "gv", "gs", "gh",
				  "ro", "rv", "rs", "rh", "yo", "yv", "ys", "yh",
				  "kp", ""};
static int64_t WALL_MASK = 15;
static int64_t ROBOT_MASK = 496;
static int64_t REDIRECT_MASK = 130560;
static int64_t TARGET_MASK = 17179738112;
static int64_t BLOCK_DOWN = 497;
static int64_t BLOCK_UP = 498;
static int64_t BLOCK_RIGHT = 500;
static int64_t BLOCK_LEFT = 504;
static int64_t ROBOT_OFFSET = 4;
static int64_t MOVE_UP = 0;
static int64_t MOVE_LEFT = 1;
static int64_t MOVE_DOWN = 2;
static int64_t MOVE_RIGHT = 3;
static int64_t N_ACTIONS = 4;
static int64_t GRID_WIDTH = 16;

static int64_t left(const int64_t* grid, int64_t robot, int64_t src_x, int64_t src_y,
		    int64_t* dst_x, int64_t* dst_y) {
  *dst_x = src_x;
  *dst_y = src_y;
  while (*dst_x > 0 && (grid[(*dst_y) * GRID_WIDTH + (*dst_x - 1)] & BLOCK_LEFT) == 0) {
    (*dst_x)--;
  }
  return src_x == *dst_x;
}

static int64_t up(const int64_t* grid, int64_t robot, int64_t src_x, int64_t src_y,
		  int64_t* dst_x, int64_t* dst_y) {
  *dst_x = src_x;
  *dst_y = src_y;
  while (*dst_y > 0 && (grid[(*dst_y - 1) * GRID_WIDTH + (*dst_x)] & BLOCK_UP) == 0) {
    (*dst_y)--;
  }
  return src_y == *dst_y;
}

static int64_t right(const int64_t* grid, int64_t robot, int64_t src_x, int64_t src_y,
		     int64_t* dst_x, int64_t* dst_y) {
  *dst_x = src_x;
  *dst_y = src_y;
  while (*dst_x < 15 && (grid[(*dst_y) * GRID_WIDTH + (*dst_x + 1)] & BLOCK_RIGHT) == 0) {
    (*dst_x)++;
  }
  return src_x == *dst_x;
}

static int64_t down(const int64_t* grid, int64_t robot, int64_t src_x, int64_t src_y,
		    int64_t* dst_x, int64_t* dst_y) {
  *dst_x = src_x;
  *dst_y = src_y;
  while (*dst_y > 0 && (grid[(*dst_y + 1) * GRID_WIDTH + (*dst_x)] & BLOCK_DOWN) == 0) {
    (*dst_y)++;
  }
  return src_y == *dst_y;
}

static int64_t move_robot(int64_t *grid, int64_t *robots, move_t *move) {

  // Find an available move
  int64_t rc = -1;
  move->src_x = robots[move->robot * 2];
  move->src_y = robots[move->robot * 2 + 1];
  if (move->action == MOVE_UP) {
    rc = up(grid, move->robot, move->src_x, move->src_y, &move->dst_x, &move->dst_y);
  } else if (move->action == MOVE_LEFT) {
    rc = left(grid, move->robot, move->src_x, move->src_y, &move->dst_x, &move->dst_y);
  } else if (move->action == MOVE_DOWN) {
    rc = down(grid, move->robot, move->src_x, move->src_y, &move->dst_x, &move->dst_y);
  } else if (move->action == MOVE_RIGHT) {
    rc = right(grid, move->robot, move->src_x, move->src_y, &move->dst_x, &move->dst_y);
  }
  if (rc) {
    return rc;
  }

  // If the move results in a new location, do it
  int64_t robot_writer = 1 << (ROBOT_OFFSET + move->robot);
  int64_t robot_eraser = ~robot_writer;  
  grid[move->src_y * GRID_WIDTH + move->src_x] &= robot_eraser;
  grid[move->dst_y * GRID_WIDTH + move->dst_x] |= robot_writer;
  robots[move->robot * 2] = move->dst_x;
  robots[move->robot * 2 + 1] = move->dst_y;
  return 0;
}

static void unmove_robot(int64_t *grid, int64_t *robots, move_t* move) {
  int64_t robot_writer = 1 << (ROBOT_OFFSET + move->robot);
  int64_t robot_eraser = ~robot_writer;
  grid[move->dst_y * GRID_WIDTH + move->dst_x] &= robot_eraser;
  grid[move->src_y * GRID_WIDTH + move->src_x] |= robot_writer;
  robots[move->robot * 2] = move->src_x;
  robots[move->robot * 2 + 1] = move->src_y;
}


static PyObject *ricochet_markers(PyObject *self) {
  int64_t len = sizeof(MARKERS) / sizeof(char*);
  PyObject *list = PyList_New(len);
  for (int64_t i = 0; i < len; i++) {
    PyList_SetItem(list, i, Py_BuildValue("s", MARKERS[i]));
  }
  return list;
}

static PyObject *ricochet_wall_mask(PyObject *self) {
  return Py_BuildValue("l", WALL_MASK);
}

static PyObject *ricochet_robot_mask(PyObject *self) {
  return Py_BuildValue("l", ROBOT_MASK);
}

static PyObject *ricochet_redirect_mask(PyObject *self) {
  return Py_BuildValue("l", REDIRECT_MASK);
}

static PyObject *ricochet_target_mask(PyObject *self) {
  return Py_BuildValue("l", TARGET_MASK);
}

static char ricochet_solve_docstring[] = "usage: solve(grid, robots, target_robot, target_x, target_y, max_depth)";
static PyObject *ricochet_solve(PyObject *self, PyObject *args) {

  // Process Arguments
  PyArrayObject *grid_obj, *robots_obj;
  int64_t target_robot, target_x, target_y, max_depth;
  if (!PyArg_ParseTuple(args, "OOllll", &grid_obj, &robots_obj, &target_robot,
			&target_x, &target_y, &max_depth))
    return NULL;

  // Extract the grid data so that we can manipulate it in our search
  int64_t* grid = PyArray_DATA(grid_obj);
  
  // Extract the robots array so we can manipulate it in our search
  int64_t* robots = PyArray_DATA(robots_obj);
  int64_t n_robots = PyArray_DIM(robots_obj, 0);

  // Robot order is a heuristic optimization that will always search for solutions that
  // move the target robot. So initialize it, and then do moves for the other robots
  int64_t robot_order[n_robots];
  robot_order[0] = (target_robot == 4 ? n_robots - 1 : target_robot);
  for (int i = 0; i < n_robots; ++i) {
    if (i != target_robot) {
      robot_order[i + (i < target_robot)] = i;
    }
  }
  for (int i = 0; i < n_robots; ++i) {
    printf("%i %li\n", i, robot_order[i]);
  }

  // Initialize moves arrays to store the progress we make and the shortest solution
  move_t moves[max_depth + 1], out_moves[max_depth + 1];
  int64_t n_moves = 0, n_out_moves = -1;
  moves[0].robot = target_robot;
  moves[0].action = -1;
  moves[0].robot_order_i = 0;

  // Dive depth-first to try every combination of moves
  int64_t solution_found = 0;
  while (n_moves >= 0) {

    // If we're beyond our depth or out of robots/actions to try for this combination, go back
    if (n_moves >= max_depth) {
      if (n_moves > 0) {
	unmove_robot(grid, robots, &(moves[--n_moves]));
	continue;
      }
      break;
    }    
    
    // Verify a robot-action pair isn't out of bounds
    moves[n_moves].action++;
    if (moves[n_moves].action >= N_ACTIONS) {
      moves[n_moves].robot = robot_order[moves[n_moves].robot_order_i];
      moves[n_moves].action = 0;
      moves[n_moves].robot_order_i++;
    }
    
    // If we're out of moves, go back
    if (moves[n_moves].robot_order_i >= n_robots) {
      if (--n_moves >= 0) unmove_robot(grid, robots, &(moves[n_moves]));
      continue;
    }

    // Try to move, if we can't skip
    if (move_robot(grid, robots, &(moves[n_moves])))
      continue;
    n_moves++;

    // If we're done, print it
    if (target_robot < 4) {
      solution_found = (robots[target_robot * 2] == target_x
			&& robots[target_robot * 2 + 1] == target_y);
    }
    else {
      for (int i = 0; i < n_robots; ++i) {
	solution_found |= robots[i * 2] == target_x && robots[i * 2 + 1] == target_y;
      }
    }

    // If we found a solution, save it
    if (solution_found) {
      solution_found = 0;
      printf("Found a solution [%li]\n", n_moves);

      // Store the shortest solution
      if (n_out_moves < 0 || n_moves < n_out_moves) {
	n_out_moves = n_moves;
	memcpy(&out_moves, &moves, n_out_moves * sizeof(move_t));
      }

      // Update the acceptable depth
      max_depth = n_moves - 1;
      continue;
    }
    
    moves[n_moves].robot_order_i = 0;
    moves[n_moves].robot = robot_order[moves[n_moves].robot_order_i];
    moves[n_moves].action = -1;    
  }

  // If we didn't find anything, do None
  if (n_out_moves < 0) {
    return Py_BuildValue("");
  }

  // Build output
  PyObject *moves_list = PyList_New(n_out_moves);
  for (int64_t i = 0; i < n_out_moves; i++) {
    PyObject *move_list = PyList_New(5);
    PyList_SetItem(move_list, 0, Py_BuildValue("i", out_moves[i].robot));
    PyList_SetItem(move_list, 1, Py_BuildValue("i", out_moves[i].src_x));
    PyList_SetItem(move_list, 2, Py_BuildValue("i", out_moves[i].src_y));
    PyList_SetItem(move_list, 3, Py_BuildValue("i", out_moves[i].dst_x));
    PyList_SetItem(move_list, 4, Py_BuildValue("i", out_moves[i].dst_y));
    PyList_SetItem(moves_list, i, move_list);
  }
  return moves_list;
  
}

static struct PyMethodDef module_methods[] = {
  {"solve", (PyCFunction) ricochet_solve, METH_VARARGS, ricochet_solve_docstring},
  {"markers", (PyCFunction) ricochet_markers, METH_NOARGS, NULL},
  {"wall_mask", (PyCFunction) ricochet_wall_mask, METH_NOARGS, NULL},
  {"robot_mask", (PyCFunction) ricochet_robot_mask, METH_NOARGS, NULL},
  {"redirect_mask", (PyCFunction) ricochet_redirect_mask, METH_NOARGS, NULL},
  {"target_mask", (PyCFunction) ricochet_target_mask, METH_NOARGS, NULL},
  {NULL}
};

static struct PyModuleDef ricochet =
  {
    PyModuleDef_HEAD_INIT,
    "ricohet",
    "usage: TBD", 
    -1, 
    module_methods
  };

PyMODINIT_FUNC PyInit_ricochet(void)
{
  import_array();
  return PyModule_Create(&ricochet);
}
