# Contributing Guide

## Developer Setup

Install the library with the `[dev,doc]` options to install the test and
documentation building suite.

```Python
pip install git+https://github.com/Teoroo-CMC/PiNN.git[dev,doc]
pytest # to run all the tests
mkdocs serve # build a live documentation
```

## Pull Request Checklist

Contributions to the main repo `teoroo-cmc/pinn` shall proceed with a pull
request (PR), which will be reviewed by at least one member of the PiNN team.
Track your development with a fork and start contributing by opening a PR. PR
can be used to discussed new features without implementation. 

The default GitHub actions runs the tests automatically, and builds a versioned
documentation for each fork. Below is a checklist before merging commits to the
master branch:

### Code Quality

- [ ] Test should pass;
- [ ] New code should be tested at least with an integration test (e.g. a layer
      should be used in a test of the potential model);
- [ ] Components with non-trivial expected outputs should be tested with simple
      cases (e.g. neighbor list or equivarient tensor products).

### Documentation

- [ ] Documentation build should pass;
- [ ] New layers should be documented, include equations when necessary;
- [ ] Citations should be added as a Bibtex entry.


### Hygiene Rules

- [ ] **Rebase changes** new changes should always be rebased on the latest
      master branch before merged, see [interactive
      rebase](https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History) if you
      need to rewrite history;
- [ ] Follow [PEP 440](https://peps.python.org/pep-0440/) for versioning.
