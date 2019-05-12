#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct move_t {
  int robot_order_i;
  int robot;
  int action;
  int src_x;
  int src_y;
  int dst_x;
  int dst_y;
} move_t;

typedef struct board_t {
  float *robot_grid;
  float *target_grid;
  float *wall_h_grid;
  float *wall_v_grid;
  int *robot_list;
  move_t *move_list;
  int n_moves;
  int n_robots;
} board_t;

static int N_ACTIONS = 4;
static int GRID_WIDTH = 16;
static int SOLVE_DFS = 1;
static int SOLVE_GRAPH = 2;


int avail_up(const board_t* board, const move_t* move) {
  int src_pos = move->dst_y * GRID_WIDTH + move->dst_x;
  int dst_pos = (move->dst_y - 1) * GRID_WIDTH + move->dst_x;
  return (move->dst_y > 0
	  && board->wall_h_grid[src_pos] == 0
	  && board->robot_grid[dst_pos] == 0);
}

int avail_down(const board_t* board, const move_t* move) {
  int src_pos = move->dst_y * GRID_WIDTH + move->dst_x;
  int dst_pos = (move->dst_y + 1) * GRID_WIDTH + move->dst_x;
  return (move->dst_y < (GRID_WIDTH - 1)
	  && board->wall_h_grid[src_pos] == 0
	  && board->robot_grid[dst_pos] == 0);
}

int avail_left(const board_t* board, const move_t* move) {
  int src_pos = move->dst_y * GRID_WIDTH + move->dst_x;
  int dst_pos = move->dst_y * GRID_WIDTH + move->dst_x - 1;
  return (move->dst_x > 0
	  && board->wall_v_grid[src_pos] == 0
	  && board->robot_grid[dst_pos] == 0);
}

int avail_right(const board_t* board, const move_t* move) {
  int src_pos = move->dst_y * GRID_WIDTH + move->dst_x;
  int dst_pos = move->dst_y * GRID_WIDTH + move->dst_x + 1;
  return (move->dst_x < (GRID_WIDTH - 1)
	  && board->wall_v_grid[src_pos] == 0
	  && board->robot_grid[dst_pos] == 0);
}

int can_move(const board_t *board, move_t *move) {
  move->dst_x = move->src_x;
  move->dst_y = move->src_y;
  if (move->action == 0) {
    while (avail_up(board, move))
      move->dst_y--;
  }
  else if (move->action == 1) {
    while (avail_left(board, move))
      move->dst_x--;
  }
  else if (move->action == 2) {
    while (avail_down(board, move))
      move->dst_y++;
  }
  else if (move->action == 3) {
    while (avail_right(board, move))
      move->dst_x++;
  }
  return move->src_y == move->dst_y && move->src_x == move->dst_x;
}

int move_robot(board_t *board, move_t *move) {
  move->src_x = board->robot_list[move->robot * 2];
  move->src_y = board->robot_list[move->robot * 2 + 1];

  // First make sure we can move
  int rc = can_move(board, move);
  if (rc) {
    return rc;
  }

  // Execute the move
  int src_pos = move->src_y * GRID_WIDTH + move->src_x;
  int dst_pos = move->dst_y * GRID_WIDTH + move->dst_x;
  board->robot_grid[dst_pos] = board->robot_grid[src_pos];
  board->robot_grid[src_pos] = 0;
  board->robot_list[move->robot * 2] = move->dst_x;
  board->robot_list[move->robot * 2 + 1] = move->dst_y;
  return 0;
}

void unmove_robot(board_t *board, move_t *move) {
  int src_pos = move->src_y * GRID_WIDTH + move->src_x;
  int dst_pos = move->dst_y * GRID_WIDTH + move->dst_x;
  board->robot_grid[src_pos] = board->robot_grid[dst_pos];
  board->robot_grid[dst_pos] = 0;
  board->robot_list[move->robot * 2] = move->src_x;
  board->robot_list[move->robot * 2 + 1] = move->src_y;
}

void execute_move_list(board_t *board, move_t* move_list, int n_moves) {
  for (int i = 0; i < n_moves; i++) {
    move_robot(board, &(move_list[i]));
  }
}

PyObject* build_move_list(const move_t* move_list, int n_moves) {
  if (n_moves < 0) {
    return Py_BuildValue("");
  }
  PyObject *move_list_obj = PyList_New(n_moves);
  for (int i = 0; i < n_moves; i++) {
    PyObject *move_obj = PyList_New(6);
    PyList_SetItem(move_obj, 0, Py_BuildValue("i", move_list[i].robot));
    PyList_SetItem(move_obj, 1, Py_BuildValue("i", move_list[i].action));
    PyList_SetItem(move_obj, 2, Py_BuildValue("i", move_list[i].src_x));
    PyList_SetItem(move_obj, 3, Py_BuildValue("i", move_list[i].src_y));
    PyList_SetItem(move_obj, 4, Py_BuildValue("i", move_list[i].dst_x));
    PyList_SetItem(move_obj, 5, Py_BuildValue("i", move_list[i].dst_y));
    PyList_SetItem(move_list_obj, i, move_obj);
  }
  return move_list_obj; 
}
 
static int is_solution(board_t *board, int n_robots, int target_robot, int target_x, int target_y) {
  if (target_robot < 4) {
    return (board->robot_list[target_robot * 2] == target_x
	    && board->robot_list[target_robot * 2 + 1] == target_y);
  }
  
  int solution_found = 0;
  for (int i = 0; i < n_robots; ++i) {
    solution_found |= board->robot_list[i * 2] == target_x && board->robot_list[i * 2 + 1] == target_y;
  }
  return solution_found;
}

static void dfs_solver(board_t *board, int *max_depth, move_t *out_move_list,
		       float *n_out_moves, float *robot_order, int n_robots,
		       int target_robot, int target_x, int target_y) {
  out_move_list[0].robot_order_i = 0;
  out_move_list[0].robot = robot_order[0];
  out_move_list[0].action = -1;

  // Dive depth-first to try every combination of moves
  int n_moves = 0;
  while (n_moves >= 0) {

    // Verify a robot-action pair isn't out of bounds
    out_move_list[n_moves].action++;
    if (out_move_list[n_moves].action >= N_ACTIONS) {
      out_move_list[n_moves].robot_order_i++;
      if (out_move_list[n_moves].robot_order_i >= n_robots) {
	if (--n_moves >= 0) unmove_robot(board, &(out_move_list[n_moves]));
	continue;
      }
      out_move_list[n_moves].robot = robot_order[out_move_list[n_moves].robot_order_i];
      out_move_list[n_moves].action = 0;
    }
    
    // Try to move, if we can't skip
    if (move_robot(board, &(out_move_list[n_moves]))) {
      continue;
    }

    // Skip if we already been in this location
    if (n_moves >= 1 && out_move_list[n_moves - 1].robot == out_move_list[n_moves].robot &&
	out_move_list[n_moves - 1].src_x == out_move_list[n_moves].dst_x &&
	out_move_list[n_moves - 1].src_y == out_move_list[n_moves].dst_y) {
      unmove_robot(board, &(out_move_list[n_moves]));
      continue;
    }

    // Otherwise register this move and go on
    n_moves++;

    // If we found a solution, save it
    if (is_solution(board, n_robots, target_robot, target_x, target_y)) {

      // If this is the first solution or shorter than any other, store it
      if (*n_out_moves < 0 || n_moves < *n_out_moves) {
	*n_out_moves = n_moves;
	memcpy(out_out_move_list, out_move_list, *n_out_moves * sizeof(move_t));
      }

      // Update the acceptable depth
      *max_depth = n_moves - 1;
    }

    // If we're beyond our depth or out of robots/actions to try for this combination, go back
    if (n_moves >= *max_depth) {
      if (n_moves > 0) {
	unmove_robot(board, &(out_move_list[--n_moves]));
	continue;
      }
      break;
    }
    
    out_move_list[n_moves].robot_order_i = 0;
    out_move_list[n_moves].robot = robot_order[out_move_list[n_moves].robot_order_i];
    out_move_list[n_moves].action = -1;    
  }
}

static void dfs_driver(board_t *board, float *max_depth,
		       move_t *out_move_list, float *n_out_moves,
		       float *robot_order, int n_robots,
		       int target_robot, int target_x, int target_y) {
  int n = 1;
  dfs_solver(board, max_depth, out_move_list, n_out_moves,
	     robot_order, n, target_robot, target_x, target_y);

  // Solve for every combination of 2 robots
  n = 2;
  for (int i = 0; i < n_robots; ++i) {
    if (i != robot_order[0]) {
      robot_order[1] = i;
      dfs_solver(board, max_depth, out_move_list, n_out_moves,
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
	  dfs_solver(board, max_depth, out_move_list, n_out_moves,
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
  dfs_solver(board, max_depth, out_move_list, n_out_moves,
	     robot_order, n, target_robot, target_x, target_y);
}

static char ricochet_solve_docstring[] = "usage: solve(robot_grid, target_grid, wall_h_grid, wall_v_grid, robot_list, target_robot, target_x, target_y, max_depth)";
static PyObject *ricochet_solve(PyObject *self, PyObject *args) {

  // Process Arguments
  PyArrayObject *robot_grid_obj, *target_grid_obj, *wall_h_grid, *wall_v_grid, *robot_list_obj;
  int target_robot, target_x, target_y, max_depth, solver;
  if (!PyArg_ParseTuple(args, "OOOOOOOOOlllll",
			&robot_grid_obj, &target_grid_obj, &wall_h_grid_obj, &wall_v_grid_obj,
		        &robot_list_obj, &target_robot,	&target_x, &target_y, &max_depth, &solver))
    return NULL;


  // Prepare the board
  board.robot_grid = (float*) PyArray_DATA(robot_grid_obj);
  board.target_grid = (float*) PyArray_DATA(target_grid_obj);
  board.wall_h_grid = (float*) PyArray_DATA(wall_h_grid_obj);
  board.wall_v_grid = (float*) PyArray_DATA(wall_v_grid_obj);
  board.robot_list = (float*) PyArray_DATA(robot_list_obj);
  board.n_robots = PyArray_DIM(robot_list_obj, 0);
  board.move_list = (move_t*) malloc(max_depth * sizeof(move_t));
  board.n_moves = 0;

  // Prepare output
  move_t *out_move_list = (move_t*) malloc(max_depth * sizeof(move_t));
  int n_out_moves = -1;
  
  // Robot order is a heuristic optimization that will always search for solutions that
  // move the target robot. So initialize it, and then do moves for the other robots
  float *robot_order = (float*) malloc(n_robots * sizeof(float));
  robot_order[0] = (target_robot == 4 ? n_robots - 1 : target_robot);

  // Solve for Depth first
  if (solver == SOLVE_DFS) {
    dfs_driver(&board, move_list, &max_depth, out_move_list, &n_out_move_list,
	       robot_order, n_robots, target_robot, target_x, target_y);
  }
  
  execute_move_list(&board, out_move_list, n_out_moves);
  PyObject *list = build_move_list(out_move_list, n_out_moves);
  free(board.move_list);
  free(out_move_list);
  free(robot_order);
  return list;
}

static struct PyMethodDef module_methods[] = {
  {"solve", (PyCFunction) ricochet_solve, METH_VARARGS, ricochet_solve_docstring},
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
