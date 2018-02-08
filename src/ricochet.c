#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct move {
  int64_t robot_i;
  int64_t action;
  int64_t src_x;
  int64_t src_y;
  int64_t dst_x;
  int64_t dst_y;
} move;


typedef struct robot {
  int64_t x;
  int64_t y;
} robot;

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

static void move_robot(int64_t *grid, int64_t robot_i, robot *robots,
		       int64_t src_x, int64_t src_y,
		       int64_t dst_x, int64_t dst_y) {
      int64_t robot_writer = 1 << (ROBOT_OFFSET + robot_i);
      int64_t robot_eraser = ~robot_writer;
      grid[src_y * 16 + src_x] &= robot_eraser;
      grid[dst_y * 16 + dst_x] |= robot_writer;
      robots[robot_i].x = dst_x;
      robots[robot_i].y = dst_y;
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

static char ricochet_solve_docstring[] = "usage: solve(grid, robots, target)";
static PyObject *ricochet_solve(PyObject *self, PyObject *args) {

  // Process Arguments
  PyArrayObject *grid_obj, *robots_obj;
  int64_t target_robot, target_x, target_y, max_depth;
  if (!PyArg_ParseTuple(args, "OOllll", &grid_obj, &robots_obj, &target_robot,
			&target_x, &target_y, &max_depth))
    return NULL;

  // Prepare Grid
  int64_t* grid = PyArray_DATA(grid_obj);

  // Initialize robots
  int64_t* robots_onedim = PyArray_DATA(robots_obj);
  int64_t n_robots = PyArray_DIM(robots_obj, 0);
  robot robots[n_robots];
  for (int64_t i = 0; i < n_robots; ++i) {
    robots[i].x = robots_onedim[i * 2];
    robots[i].y = robots_onedim[i * 2 + 1];
    printf("(%li, %li)\n", robots[i].x, robots[i].y);
  }

  // Initialize moves
  move moves[max_depth];
  moves[0].robot_i = 0;
  moves[0].action = -1;
  int64_t n_moves = 0;

  for (int y = 0; y < 16; ++y) {
    for (int x = 0; x < 16; ++x) {
      printf("%12i", grid[y * 16 + x]);
    }
    printf("\n");
  }
  
  
  // Try every combination
  while (n_moves >= 0) {
    //printf("a %li\n", n_moves);
    // If we're beyond our depth or out of robots/actions to try for this combination, go back
    if (n_moves == max_depth || (moves[n_moves].robot_i >= (n_robots - 1)
				 && moves[n_moves].action >= 3)) {
      --n_moves;
      move_robot(grid, moves[n_moves].robot_i, robots,
		 moves[n_moves].dst_x, moves[n_moves].dst_y,
		 moves[n_moves].src_x, moves[n_moves].src_y);
      continue;
    }

    //printf("b %li\n", n_moves);
    // If we're done, print it
    if (moves[n_moves].robot_i == target_robot
	&& robots[target_robot].x == target_x
	&& robots[target_robot].y == target_y) {
      printf("Found a solution [%li]\n", n_moves);
      max_depth = n_moves;
      --n_moves;
      move_robot(grid, moves[n_moves].robot_i, robots,
		 moves[n_moves].dst_x, moves[n_moves].dst_y,
		 moves[n_moves].src_x, moves[n_moves].src_y);
      continue;
    }

    //printf("c %li\n", n_moves);
    // Find the next move to make
    while (n_moves >= 0) {

      //printf("d %li\n", n_moves);
      // Verify a robot-action pair isn't out of bounds
      ++moves[n_moves].action;
      if (moves[n_moves].action >= 4) {
	++moves[n_moves].robot_i;
	moves[n_moves].action = 0;
      }
      //printf("e %li\n", n_moves);
      if (moves[n_moves].robot_i >= n_robots) {
	--n_moves;
	if (n_moves < 0)
	  break;
	move_robot(grid, moves[n_moves].robot_i, robots,
		   moves[n_moves].dst_x, moves[n_moves].dst_y,
		   moves[n_moves].src_x, moves[n_moves].src_y);
	continue;
      }
      
      //printf("f %li\n", n_moves);
      // Find an available move
      moves[n_moves].src_x = robots[moves[n_moves].robot_i].x;
      moves[n_moves].src_y = robots[moves[n_moves].robot_i].y;	
      if (moves[n_moves].action == 0) {
	if (up(grid, moves[n_moves].robot_i, moves[n_moves].src_x, moves[n_moves].src_y,
	       &moves[n_moves].dst_x, &moves[n_moves].dst_y))
	  continue;
      }    
      else if (moves[n_moves].action == 1) {
	if (down(grid, moves[n_moves].robot_i, moves[n_moves].src_x, moves[n_moves].src_y,
	       &moves[n_moves].dst_x, &moves[n_moves].dst_y))
	  continue;
      }    
      else if (moves[n_moves].action == 2) {
	if (left(grid, moves[n_moves].robot_i, moves[n_moves].src_x, moves[n_moves].src_y,
	       &moves[n_moves].dst_x, &moves[n_moves].dst_y))
	  continue;
      }
      else if (moves[n_moves].action == 3) {
	if (right(grid, moves[n_moves].robot_i, moves[n_moves].src_x, moves[n_moves].src_y,
		  &moves[n_moves].dst_x, &moves[n_moves].dst_y))
	  continue;
      }

      //printf("g %li\n", n_moves);      
      // If one works, move it
      move_robot(grid, moves[n_moves].robot_i, robots,
		 moves[n_moves].src_x, moves[n_moves].src_y,
		 moves[n_moves].dst_x, moves[n_moves].dst_y);
      ++n_moves;
      moves[n_moves].robot_i = 0;
      moves[n_moves].action = -1;
      break;
    }
  }
  for (int64_t i = 0; i < n_robots; ++i) {
    printf("(%li, %li)\n", robots[i].x, robots[i].y);
    robots_onedim[i * 2] = robots[i].x;
    robots_onedim[i * 2 + 1] = robots[i].y;
  }  // Return output
  for (int y = 0; y < 16; ++y) {
    for (int x = 0; x < 16; ++x) {
      printf("%12i", grid[y * 16 + x]);
    }
    printf("\n");
  }
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
