# Computer vision

> _Collection of code samples for Computer Vision classes_ 👁️


## Prerequisite
As I was developing the apps on Windows on WSL
and using [`pipenv`](https://pipenv.pypa.io/en/latest/) as a virtualenv management tool, several steps are required to be accomplished:

### Installing `pipenv`

As from Python 3.12 or Ubuntu 24.04 running simply `pip install` to install systemwide Python packages will result in throwing error of this kind:
```
error: externally-managed-environment

× This environment is externally managed
╰─> To install Python packages system-wide, try apt install
    python3-xyz, where xyz is the package you are trying to
    install.
```

The solution I took was to run:
```bash
pip install pipenv --break-system-packages
```

## Opening windows from WSL

By default there is no way to open external windows from WSL directly so `x11-apps` should be installed:

```bash
sudo apt-get install x11-apps
```

To test if everything is working fine run:
```bash
xeyes
```

If not refer to this [StackOverflow questions](https://stackoverflow.com/questions/65939167/problem-using-opencv-in-wsl-when-opening-windows).

## Folder structure
Every exercise is in its separate folder.

Every "exercise" folder has the following structure:
```
.
├── subtasks/
│   ├── subtask1.py
│   ├── subtask2.py
│   └── ...
├── data/
├── dist/
├── __init__.py
├── __main__.py
├── task.py
└── README.md
```
* `data/` folder contains everything related to the exercise, for example pictures and images 
* `dist/` is the output folder for result - ignored by **Git**
* `task.py` the main logic of the hands-on indepentent task
* `__main__.py` itended to invoke `task.py` when called outside as a module
* `subtasks/` is a additional folder containing code for various usage - learning, testing, experimenting