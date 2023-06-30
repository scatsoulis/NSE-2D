# Incompressible N-S solver: Finite Differences

The purpose of this is to provide a functional-based outline to the Ae232a Fall 2021 project. The code will be outlined in Python, but the project may be completed in any language

## Usage notes

The primary file to control the solve will be ```RunSolver.py```.

Optionally, users may fill out test functions and run any other of the files. These functions are not necessarily required, but they provide useful hooks to run your code and diagnose errors. The main testing loop, in files outside of ```RunSolver.py``` will be indicated by:

```
if __name__ == "__main__":
```

When developing a large (or complex) code-base, test-driven development (TDD) is a quintessential development technique and I recommend implementing it in your own software development workflow.

## Variable definition

It is important to only define global variables once (as opposed to in each seperate files), in a functional code like this, that requires defining the variable only in the primary file (e.g. ```RunSolve.py```) and passing the variable to each function as needed. By writing your code in this way, you will ensure that you do not introduce bugs due to not changing the variables in every file.

## Documentation

In the code, all functions and files will include Summary notes and Input/Output notes. In developing a code-base, especially in the non-type checking language like python, it is important to ensure that variable names **and** types are clearly defined. This helps the code developer (that is, you) to not get lost in variable names and prevents bugs from occuring with types not being exact. Again, doing this is not required, but getting in the habit of doing this will benefit you throughout your career.

## Git usage

Another fundamental software development tool is *git*. While not required for this project, it's usage will benefit you throughout your career (academic or otherwise). On a basic level, git allows users to track changes, commit old work, and to go back to old versions of your code -- hence git being called a "version control". A wonderful guide on the basic of git may be found [here](https://rogerdudler.github.io/git-guide/)

## Acknowledgements

The outline of this code follows that of Jane Bae for the Matlab version. The purpose of this version is to aid in the usage of Python for users that are not familiar to the language and to aid in cultivating good software-development techniques.
