import pandas as pd
import numpy as np


def adata_to_df(adata):
    X = adata.X
    if isinstance(X, np.ndarray):
        pass
    elif hasattr(X, "toarray"):
        X = X.toarray()
    elif hasattr(X, "A"):
        X = X.A
    else:
        X = np.asarray(X)

    df = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)
    return df