# niche-VI archive (now scVIVA in scvi-tools)

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/YosefLab/niche-VI/test.yaml?branch=main
[link-tests]: https://github.com/YosefLab/niche-VI/actions/workflows/test.yml
[badge-docs]: https://img.shields.io/readthedocs/niche-VI

⚠️ Attention! DEPRECATED, repo to reproduce the results from the scVIVA paper, now maintained into [scvi-tools](https://github.com/scverse/scvi-tools).

## Installation

You need to have Python 3.9 or newer installed on your system. If you don't have
Python installed, we recommend installing [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge).

There are several alternative options to install niche-VI:

<!--
1) Install the latest release of `niche-VI` from `PyPI <https://pypi.org/project/niche-VI/>`_:

```bash
pip install niche-VI
```
-->

1. Install the latest development version:

```bash
pip install git+https://github.com/YosefLab/niche-VI.git@main
```

## Citation
> @article{levy2025scviva,
  title={scVIVA: a probabilistic framework for representation of cells and their environments in spatial transcriptomics},
  author={Levy, Nathan and Ingelfinger, Florian and Bakulin, Artemii and Cinnirella, Giacomo and Boyeau, Pierre and Nadler, Boaz and Ergen, Can and Yosef, Nir},
  journal={bioRxiv},
  pages={2025--06},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
