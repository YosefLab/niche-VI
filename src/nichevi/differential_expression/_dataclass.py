from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from rich import print
from sklearn.gaussian_process import GaussianProcessClassifier


@dataclass
class DifferentialExpressionResults:
    """Dataclass for storing the results of the differential expression analysis, including the GP classifier"""

    gpc: GaussianProcessClassifier
    g1_g2: pd.DataFrame
    g1_n1: pd.DataFrame
    n1_g2: pd.DataFrame
    n1_n2: Optional[pd.DataFrame] = field(default=None)
    n1_index: Optional[np.array] = field(default=None)
    n2_index: Optional[np.array] = field(default=None)

    def gpc_info(self):
        """Print the log marginal likelihood value and the kernel of the Gaussian Process Classifier"""
        print("Training score: ", self.gpc.train_score_)
        print("Marginal likelihood: ", self.gpc.log_marginal_likelihood_value_)
        print("Kernel: ", self.gpc.kernel_)
