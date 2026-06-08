# Third-Party Notices

This project uses TabReD as a Git submodule:

- https://github.com/yandex-research/tabred

TabReD is located at `tabred/`. No TabReD source files are modified by this
project. The TabReD license and upstream attribution are kept in
`tabred/LICENSE`.

Local runtime setup may create `tabred/data -> ../data` so that TabReD's
existing `:data/...` paths can read datasets stored in this repository's
ignored `data/` directory. This symlink is a runtime setup detail, not a TabReD
source-code modification.

TabReD datasets are downloaded from Kaggle or competition pages. Dataset use is
governed by the terms of the respective dataset/competition, not by this
repository.
