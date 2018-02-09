#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct move_t {
  int64_t robot_i;
  int64_t action;
  int64_t src_x;
  int64_t src_y;
  int64_t dst_x;
  int64_t dst_y;
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


static int64_t left(const int64_t* grid, int64_t robot,
		    int64_t src_x, int64_t src_y,
		    int64_t* dst_x, int64_t* dst_y) {
  *dst_x = src_x;
  *dst_y = src_y;
  while (*dst_x > 0 && (grid[(*dst_y) * 16 + (*dst_x - 1)] & BLOCK_LEFT) == 0) {
    (*dst_x)--;
  }
  return src_x == *dst_x;
}

static int64_t up(const int64_t* grid, int64_t robot,
		  int64_t src_x, int64_t src_y,
		  int64_t* dst_x, int64_t* dst_y) {
  *dst_x = src_x;
  *dst_y = src_y;
  while (*dst_y > 0 && (grid[(*dst_y - 1) * 16 + (*dst_x)] & BLOCK_UP) == 0) {
    (*dst_y)--;
  }
  return src_y == *dst_y;
}

static int64_t right(const int64_t* grid, int64_t robot,
		     int64_t src_x, int64_t src_y,
		     int64_t* dst_x, int64_t* dst_y) {
  *dst_x = src_x;
  *dst_y = src_y;
  while (*dst_x < 15 && (grid[(*dst_y) * 16 + (*dst_x + 1)] & BLOCK_RIGHT) == 0) {
    (*dst_x)++;
  }
  return src_x == *dst_x;
}

static int64_t down(const int64_t* grid, int64_t robot,
		    int64_t src_x, int64_t src_y,
		    int64_t* dst_x, int64_t* dst_y) {
  *dst_x = src_x;
  *dst_y = src_y;
  while (*dst_y > 0 && (grid[(*dst_y + 1) * 16 + (*dst_x)] & BLOCK_DOWN) == 0) {
    (*dst_y)++;
  }
  return src_y == *dst_y;
}


static int64_t move_robot(int64_t *grid, int64_t *robots, move_t *move) {
  // Find an available move
  move->src_x = robots[move->robot_i * 2];
  move->src_y = robots[move->robot_i * 2 + 1];
  int64_t rc = -1;
  if (move->action == MOVE_UP) {
    rc = up(grid, move->robot_i, move->src_x, move->src_y, &move->dst_x, &move->dst_y);
  } else if (move->action == MOVE_LEFT) {
    rc = left(grid, move->robot_i, move->src_x, move->src_y, &move->dst_x, &move->dst_y);
  } else if (move->action == MOVE_DOWN) {
    rc = down(grid, move->robot_i, move->src_x, move->src_y, &move->dst_x, &move->dst_y);
  } else if (move->action == MOVE_RIGHT) {
    rc = right(grid, move->robot_i, move->src_x, move->src_y, &move->dst_x, &move->dst_y);
  }
  if (rc) {
    return rc;
  }
  
  int64_t robot_writer = 1 << (ROBOT_OFFSET + move->robot_i);
  int64_t robot_eraser = ~robot_writer;  
  grid[move->src_y * 16 + move->src_x] &= robot_eraser;
  grid[move->dst_y * 16 + move->dst_x] |= robot_writer;

  grid[move->src_y * 16 + move->src_x] &= robot_eraser;
  grid[move->dst_y * 16 + move->dst_x] |= robot_writer;
  robots[move->robot_i * 2] = move->dst_x;
  robots[move->robot_i * 2 + 1] = move->dst_y;
  return 0;
}

static void unmove_robot(int64_t *grid, int64_t *robots, move_t* move) {
  int64_t robot_writer = 1 << (ROBOT_OFFSET + move->robot_i);
  int64_t robot_eraser = ~robot_writer;
  grid[move->dst_y * 16 + move->dst_x] &= robot_eraser;
  grid[move->src_y * 16 + move->src_x] |= robot_writer;
  robots[move->robot_i * 2] = move->src_x;
  robots[move->robot_i * 2 + 1] = move->src_y;
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

  // Prepare Grid
  int64_t* grid = PyArray_DATA(grid_obj);
  int64_t n_actions = 4;
  
  // Initialize robots
  int64_t* robots = PyArray_DATA(robots_obj);
  int64_t n_robots = PyArray_DIM(robots_obj, 0);

  // Initialize moves
  move_t moves[max_depth];
  moves[0].robot_i = 0;
  moves[0].action = -1;
  int64_t n_moves = 0;

  /* moves[n_moves].robot_i = 2; */
  /* moves[n_moves].action = MOVE_UP; */
  /* move_robot(grid, robots, &(moves[n_moves])); */
  /* n_moves++; */
  
  /* moves[n_moves].robot_i = 3; */
  /* moves[n_moves].action = MOVE_LEFT; */
  /* move_robot(grid, robots, &(moves[n_moves])); */
  /* n_moves++; */

  /* moves[n_moves].robot_i = 3; */
  /* moves[n_moves].action = MOVE_UP; */
  /* move_robot(grid, robots, &(moves[n_moves])); */
  /* n_moves++; */

  /* moves[n_moves].robot_i = 3; */
  /* moves[n_moves].action = MOVE_RIGHT; */
  /* move_robot(grid, robots, &(moves[n_moves])); */
  /* n_moves++; */
  

  // Try every combination
  while (n_moves >= 0) {
    // If we're beyond our depth or out of robots/actions to try for this combination, go back
    if (n_moves >= max_depth) {
      n_moves--;
      if (n_moves < 0)
	break;
      unmove_robot(grid, robots, &(moves[n_moves]));
      continue;
    }

    // Verify a robot-action pair isn't out of bounds
    moves[n_moves].action++;
    if (moves[n_moves].action >= n_actions) {
      moves[n_moves].robot_i++;
      moves[n_moves].action = 0;
    }
    
    // If we're out of moves, go back
    if (moves[n_moves].robot_i >= n_robots) {
      n_moves--;
      if (n_moves < 0)
	break;
      unmove_robot(grid, robots, &(moves[n_moves]));
      continue;
    }

    // Try to move, if we can't skip
    if (move_robot(grid, robots, &(moves[n_moves])))
      continue;

    // Update for a successful move
    n_moves++;
    moves[n_moves].robot_i = 0;
    moves[n_moves].action = -1;

    // If we're done, print it
    if (robots[target_robot * 2] == target_x && robots[target_robot * 2 + 1] == target_y) {
      printf("Found a solution [%li]\n", n_moves);
      for (int i = 0; i < n_moves; ++i) {
	printf("%li %li %li\n", i, moves[i].robot_i, moves[i].action);
      }
      max_depth = n_moves - 1;
      n_moves--;
      if (n_moves < 0)
	break;
      unmove_robot(grid, robots, &(moves[n_moves]));
      continue;
    }    
  }

  // Store output and return number of moves
  return Py_BuildValue("l", n_moves);
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
