#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct move_t {
  int64_t robot_order_i;
  int64_t robot;
  int64_t action;
  int64_t src_x;
  int64_t src_y;
  int64_t dst_x;
  int64_t dst_y;
} move_t;

static const char *MARKERS[35] = {"wu", "wd", "wl", "wr", "b", "g", "r", "y", "k",
				  "bi", "bd", "gi", "gd", "ri", "rd", "yi", "yd",
				  "bo", "b^", "bs", "bh", "go", "g^", "gs", "gh",
				  "ro", "r^", "rs", "rh", "yo", "y^", "ys", "yh",
				  "kp", ""};
static int64_t WALL_MASK = 15;
static int64_t ROBOT_MASK = 496;
static int64_t REDIRECT_MASK = 130560;
static int64_t TARGET_MASK = 17179738112;
static int64_t REDIRECT_INCLINE = 85;
static int64_t ROBOT_OFFSET = 4;
static int64_t REDIRECT_OFFSET = 9;
static int64_t N_ACTIONS = 4;
static int64_t GRID_WIDTH = 16;
static int64_t BLOCK[4] = {131058, 131064, 131057, 131060};
static int64_t WALL[4] = {2, 8, 1, 4};
static int64_t SOLVE_DFS = 1;
static int64_t SOLVE_GRAPH = 2;

static int64_t redirect_direction(int64_t tile, int64_t robot) {
  int64_t redirects = tile >> REDIRECT_OFFSET;
  // If the robot is the same color as the redirect, just send the robot through
  if (robot < 4 && ((redirects >> (robot * 2)) & 3)) {
    return 0;
  }
  if ((redirects & REDIRECT_INCLINE) > 0) {
    return 1;
  }
  return -1;
}

// up left down right
static int64_t find_move(const int64_t *grid, int64_t robot, int64_t action,
			 int64_t src_x, int64_t src_y, int64_t *dst_x, int64_t *dst_y) {
  *dst_x = src_x;
  *dst_y = src_y;
  int64_t dx = ((action % 2) == 1) * pow(-1, action < 2);
  int64_t dy = ((action % 2) == 0) * pow(-1, action < 2);
  while ((dy >= 0 || *dst_y > 0) && (dy <= 0 || *dst_y < 15) &&
	 (dx >= 0 || *dst_x > 0) && (dx <= 0 || *dst_x < 15) && 
	 (grid[(*dst_y + dy) * GRID_WIDTH + (*dst_x + dx)] & BLOCK[action]) == 0) {
    (*dst_x) += dx;
    (*dst_y) += dy;
  }
  // Handle redirect
  if ((dy >= 0 || *dst_y > 0) && (dy <= 0 || *dst_y < 15) &&
      (dx >= 0 || *dst_x > 0) && (dx <= 0 || *dst_x < 15) &&
      grid[(*dst_y + dy) * GRID_WIDTH + (*dst_x + dx)] & REDIRECT_MASK &&
      (grid[(*dst_y + dy) * GRID_WIDTH + (*dst_x + dx)] & WALL[action]) == 0) {
    (*dst_x) += dx;
    (*dst_y) += dy;
    int64_t rd = redirect_direction(grid[(*dst_y) * GRID_WIDTH + (*dst_x)], robot);
    action = (action + (int64_t) pow(rd, 1 + (action >= 2))) % 4;
    return find_move(grid, robot, action, *dst_x, *dst_y, dst_x, dst_y);
  }
  return src_y == *dst_y && src_x == *dst_x;
}

static int64_t move_robot(int64_t *grid, int64_t *robots, move_t *move) {

  // Find an available move
  move->src_x = robots[move->robot * 2];
  move->src_y = robots[move->robot * 2 + 1];
  int64_t rc = find_move(grid, move->robot, move->action, move->src_x, move->src_y,
			 &move->dst_x, &move->dst_y);
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

static void move_robots(int64_t *grid, int64_t *robots, move_t* moves, int64_t n_moves) {
  if (n_moves < 0) {
    return;
  }
  for (int64_t i = 0; i < n_moves; i++) {
    move_robot(grid, robots, &(moves[i]));
  }
}

static PyObject* build_moves_list(const move_t* moves, int64_t n_moves) {
  if (n_moves < 0) {
    return Py_BuildValue("");
  }
  PyObject *moves_list = PyList_New(n_moves);
  for (int64_t i = 0; i < n_moves; i++) {
    PyObject *move_list = PyList_New(6);
    PyList_SetItem(move_list, 0, Py_BuildValue("i", moves[i].robot));
    PyList_SetItem(move_list, 1, Py_BuildValue("i", moves[i].action));
    PyList_SetItem(move_list, 2, Py_BuildValue("i", moves[i].src_x));
    PyList_SetItem(move_list, 3, Py_BuildValue("i", moves[i].src_y));
    PyList_SetItem(move_list, 4, Py_BuildValue("i", moves[i].dst_x));
    PyList_SetItem(move_list, 5, Py_BuildValue("i", moves[i].dst_y));
    PyList_SetItem(moves_list, i, move_list);
  }
  return moves_list; 
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

static int64_t is_solution(int64_t *grid, int64_t *robots, int64_t n_robots,
			   int64_t target_robot, int64_t target_x, int64_t target_y) {
  if (target_robot < 4) {
    return robots[target_robot * 2] == target_x && robots[target_robot * 2 + 1] == target_y;
  }
  
  int64_t solution_found = 0;
  for (int i = 0; i < n_robots; ++i) {
    solution_found |= robots[i * 2] == target_x && robots[i * 2 + 1] == target_y;
  }
  return solution_found;
}

static void dfs_solver(int64_t *grid, int64_t *robots, move_t *moves, int64_t *max_depth,
		       move_t *out_moves, int64_t *n_out_moves,
		       int64_t *robot_order, int64_t n_robots,
		       int64_t target_robot, int64_t target_x, int64_t target_y) {
  int64_t n_moves = 0;
  moves[0].robot_order_i = 0;
  moves[0].robot = robot_order[0];
  moves[0].action = -1;

  // Dive depth-first to try every combination of moves
  while (n_moves >= 0) {

    // Verify a robot-action pair isn't out of bounds
    moves[n_moves].action++;
    if (moves[n_moves].action >= N_ACTIONS) {
      moves[n_moves].robot_order_i++;
      if (moves[n_moves].robot_order_i >= n_robots) {
	if (--n_moves >= 0) unmove_robot(grid, robots, &(moves[n_moves]));
	continue;
      }
      moves[n_moves].robot = robot_order[moves[n_moves].robot_order_i];
      moves[n_moves].action = 0;
    }
    
    // Try to move, if we can't skip
    if (move_robot(grid, robots, &(moves[n_moves]))) {
      continue;
    }

    // Skip if we already been in this location
    if (n_moves >= 1 && moves[n_moves - 1].robot == moves[n_moves].robot &&
	moves[n_moves - 1].src_x == moves[n_moves].dst_x &&
	moves[n_moves - 1].src_y == moves[n_moves].dst_y) {
      unmove_robot(grid, robots, &(moves[n_moves]));
      continue;
    }

    // Otherwise register this move and go on
    n_moves++;

    // If we found a solution, save it
    if (is_solution(grid, robots, n_robots, target_robot, target_x, target_y)) {

      // If this is the first solution or shorter than any other, store it
      if (*n_out_moves < 0 || n_moves < *n_out_moves) {
	*n_out_moves = n_moves;
	memcpy(out_moves, moves, *n_out_moves * sizeof(move_t));
      }

      // Update the acceptable depth
      *max_depth = n_moves - 1;
    }

    // If we're beyond our depth or out of robots/actions to try for this combination, go back
    if (n_moves >= *max_depth) {
      if (n_moves > 0) {
	unmove_robot(grid, robots, &(moves[--n_moves]));
	continue;
      }
      break;
    }
    
    moves[n_moves].robot_order_i = 0;
    moves[n_moves].robot = robot_order[moves[n_moves].robot_order_i];
    moves[n_moves].action = -1;    
  }
}

static void dfs_driver(int64_t *grid, int64_t *robots, move_t *moves, int64_t *max_depth,
		       move_t *out_moves, int64_t *n_out_moves,
		       int64_t *robot_order, int64_t n_robots,
		       int64_t target_robot, int64_t target_x, int64_t target_y) {
  int64_t n = 1;
  dfs_solver(grid, robots, moves, max_depth, out_moves, n_out_moves,
	     robot_order, n, target_robot, target_x, target_y);
  // Solve for every combination of 2 robots
  n = 2;
  for (int i = 0; i < n_robots; ++i) {
    if (i != robot_order[0]) {
      robot_order[1] = i;
      dfs_solver(grid, robots, moves, max_depth, out_moves, n_out_moves,
		 robot_order, n, target_robot, target_x, target_y);
    }
  }
    
  // Solve for every combination of 3 robots
  n = 3;
  for (int i = 0; i < n_robots; ++i) {
    if (i != robot_order[0]) {
      robot_order[1] = i;
      for (int j = i + 1; j < n_robots; ++j) {
	if (j != robot_order[0]) {
	  robot_order[2] = j;
	  dfs_solver(grid, robots, moves, max_depth, out_moves, n_out_moves,
		     robot_order, n, target_robot, target_x, target_y);
	}
      }
    }
  }
	
  // Solve for n robots
  n = n_robots;
  for (int i = 0; i < n_robots; ++i)
    if (i != target_robot)
      robot_order[i + (i < target_robot)] = i;
  dfs_solver(grid, robots, moves, max_depth, out_moves, n_out_moves,
	     robot_order, n, target_robot, target_x, target_y);
}

static void graph_driver(int64_t *grid, int64_t *robots, move_t *moves, int64_t *max_depth,
			 move_t *out_moves, int64_t *n_out_moves,
			 int64_t *robot_order, int64_t n_robots,
			 int64_t target_robot, int64_t target_x, int64_t target_y) {
  
}

static char ricochet_solve_docstring[] = "usage: solve(grid, robots, target_robot, target_x, target_y, max_depth)";
static PyObject *ricochet_solve(PyObject *self, PyObject *args) {

  // Process Arguments
  PyArrayObject *grid_obj, *robots_obj;
  int64_t target_robot, target_x, target_y, max_depth, solver;
  if (!PyArg_ParseTuple(args, "OOlllll", &grid_obj, &robots_obj, &target_robot,
			&target_x, &target_y, &max_depth, &solver))
    return NULL;

  // Extract the grid data so that we can manipulate it in our search
  int64_t *grid = (int64_t*) PyArray_DATA(grid_obj);
  
  // Extract the robots array so we can manipulate it in our search
  int64_t *robots = (int64_t*) PyArray_DATA(robots_obj);
  int64_t n_robots = PyArray_DIM(robots_obj, 0);

  // Initialize moves arrays to store the progress we make and the shortest solution
  move_t *moves = (move_t*) malloc(max_depth * sizeof(move_t));
  move_t *out_moves = (move_t*) malloc(max_depth * sizeof(move_t));
  int64_t n_out_moves = -1;
  
  // Robot order is a heuristic optimization that will always search for solutions that
  // move the target robot. So initialize it, and then do moves for the other robots
  int64_t *robot_order = (int64_t*) malloc(n_robots * sizeof(int64_t));
  robot_order[0] = (target_robot == 4 ? n_robots - 1 : target_robot);

  // Solve for Depth first
  if (solver == SOLVE_DFS) {
    dfs_driver(grid, robots, moves, &max_depth, out_moves, &n_out_moves,
	       robot_order, n_robots, target_robot, target_x, target_y);
  }
  else if (solver == SOLVE_GRAPH) {
    graph_driver(grid, robots, moves, &max_depth, out_moves, &n_out_moves,
		 robot_order, n_robots, target_robot, target_x, target_y);
  }
  
  move_robots(grid, robots, out_moves, n_out_moves);
  PyObject *list = build_moves_list(out_moves, n_out_moves);
  free(moves);
  free(out_moves);
  free(robot_order);
  return list;
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
