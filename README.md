# Usage

1. Install Julia (mac):
```
curl -fsSL https://install.julialang.org | sh
```
2. start Julia in multithreaded mode by running this command IN THE DIRECTORY WHERE YOU CLONED THIS REPO:
```
julia -t 10 --project=.
```
3. Press the `]` key to enter Package mode and run this command to install all required dependencies for the project (this may take a while):
```
] instantiate
```
4. Press the backspace key to exit package mode
