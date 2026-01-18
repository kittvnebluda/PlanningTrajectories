# Planning Trajectories Course

## Requirements

Python 3.13.7 was used, required packages are in `requirements.txt`.

## Usage

Once you have installed the required packages (see `requirements.txt`),
you can run any lab or debug scenario directly from the command line.

### Running Labs

The `platra` package provides several lab exercises. Use the following syntax:

```bash
python -m platra <lab> [task]
```

**Examples:**

* **Lab 1 (Trajectory Planning):**

```bash
python -m platra lab1
```

* **Lab 2 (Tracking / Teleoperation):**

```bash
python -m platra lab2 tracking
python -m platra lab2 teleop
```

* **Lab 3 (Trajectory Stabilization / 3D):**

```bash
python -m platra lab3 2d
python -m platra lab3 2d_spiral
python -m platra lab3 3d
```

### Running Debug Scenarios

Debug scenarios allow testing specific modules or cases, such as parking maneuvers.

```bash
python -m platra debug <module> <task_id>
```

**Example: Parking**

```bash
python -m platra debug parking 0
```

* `<module>` — the debug module to run (`parking` for parking scenarios).
* `<task_id>` — index of the scenario to run (integer starting from `0`).

## TODOs

* Diagonal parts of a trajectory are more thin than vertical or horizontal
part, fix it.
* Automatic creation of launch arguments with decorator for every class
that extends Laboratory abstract class.
* Poetry
