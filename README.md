# scviva-legacy (formerly niche-VI)

> **Archived.** This is the original research implementation of scVIVA, kept to reproduce the
> results of the scVIVA paper. scVIVA is now maintained in
> [scVIVA-Tools](https://scviva-tools.org/) ([GitHub](https://github.com/YosefLab/scviva-tools)),
> installed with `pip install scviva-tools` and used as `scviva.SCVIVA`. Use that for new work.
> This repository receives no further development.

The code that produces the paper's figures is in
[scviva_paper](https://github.com/LevyNat/scviva_paper). It imports this package as `nichevi`.

## Which version to install

All paper results were produced with commit
[`34a85af`](https://github.com/YosefLab/scviva-legacy/commit/34a85af59a441d69ed7aeb90d4dda98f3dcbf047)
("iLISI computation"). Install that exact commit:

```bash
pip install git+https://github.com/YosefLab/scviva-legacy.git@34a85af59a441d69ed7aeb90d4dda98f3dcbf047
```

The repository was renamed from `niche-VI`. GitHub redirects the old URL, so
`git+https://github.com/YosefLab/niche-VI.git@...` still works. The Python import name is still
`nichevi`, and the distribution name is still `niche-VI`. Neither was renamed, because the trained
checkpoints are loaded through this package and its classes.

The paper environment also pinned scvi-tools to development commit `3e275ff` (installed from git)
and scib-metrics to the 0.5.7 release. The full environment is `envs/scvi.yml` in the
reproducibility repo.

Python 3.12 was used. The package declares `>=3.9`.



## Citation

```bibtex
@article{levy2025scviva,
  title={scVIVA: a probabilistic framework for representation of cells and their environments in spatial transcriptomics},
  author={Levy, Nathan and Ingelfinger, Florian and Bakulin, Artemii and Cinnirella, Giacomo and Boyeau, Pierre and Nadler, Boaz and Ergen, Can and Yosef, Nir},
  journal={bioRxiv},
  pages={2025--06},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```
