import os
import pickle
from abc import ABC, abstractmethod
from itertools import product
from typing import Optional, List, Dict, Any

import numpy as np
from sklearn.model_selection import cross_val_score

# Third-party helpers that already exist in the project
from dataloader_and_utils import load_dataset, mean_percent_error


class BaseBatteryModelTrainer(ABC):
    """Abstract base class that encapsulates the shared workflow for all models.

    Sub-classes only have to provide:
        * the sklearn model constructor via ``_build_model``
        * a parameter grid via ``_get_param_grid``
        * any custom prediction logic via ``_predict`` (e.g. undoing log-scales)
    """

    #: default cycles to train on
    N_CYCLES = np.array([20, 30, 40, 50, 60, 70, 80, 90, 100])

    #: feature subset from Severson et al.
    WHICH_FEATURES = [2, 3, 4, 21, 22, 24, 25, 39, 40, 48, 49, 63, 65]

    def __init__(
        self,
        results_root: Optional[str] = None,
        use_log_features: bool = True,
        use_all_features: bool = False,
    ) -> None:
        self.use_log_features = use_log_features
        self.use_all_features = use_all_features
        self.which_features = self.WHICH_FEATURES.copy()
        self.results_dir = (
            results_root
            or os.path.join(os.path.dirname(__file__), "results")
        )
        os.makedirs(self.results_dir, exist_ok=True)

        # will be filled during training
        self.min_rmse: np.ndarray = np.zeros_like(self.N_CYCLES, dtype=float)
        self.min_mpe: np.ndarray = np.zeros_like(self.N_CYCLES, dtype=float)
        self.training_mpe: np.ndarray = np.zeros_like(self.N_CYCLES, dtype=float)
        self.trained_models: list = []

    # ---------------------------------------------------------------------
    # Hooks that *might* need customisation
    # ---------------------------------------------------------------------
    @abstractmethod
    def _get_param_grid(self) -> list[dict]:
        """Return a list of parameter dictionaries to iterate over."""

    @abstractmethod
    def _build_model(self, **params):
        """Return an *unfitted* sklearn regressor instance."""

    def _predict(self, model, X, y_orig):  # noqa: N802 — keep sklearn naming
        """Hook for custom prediction logic.

        By default simply calls ``model.predict`` and returns the result.  The
        original *y* is provided in case the subclass needs to undo
        transformations (e.g. log-scaling).
        """
        return model.predict(X)

    # ------------------------------------------------------------------
    # Core workflow — *rarely* overridden in subclasses
    # ------------------------------------------------------------------
    def _data_path(self, n_cycle: int) -> str:
        postfix = "_log.csv" if self.use_log_features else ".csv"
        return os.path.join("./training", f"cycles_2TO{n_cycle}{postfix}")

    def _load_data(self, n_cycle: int):
        file_name = self._data_path(n_cycle)
        return load_dataset(
        file_name,          # csv_path
        False,              # add_intercept = False
        self.use_all_features,
        self.which_features,
        )

    def train(self):
        """Run the full training / evaluation loop for all ``N_CYCLES``."""
        for i, n_cycle in enumerate(self.N_CYCLES):
            X, y, feature_names = self._load_data(n_cycle)

            best_rmse = np.inf
            best_mpe = np.inf
            best_params = None
            best_model = None

            for params in self._get_param_grid():
                model = self._build_model(**params)
                model.fit(X, y)

                # Training set evaluation (for reference only)
                y_hat_train = self._predict(model, X, y)
                residuals = y_hat_train - y
                train_rmse = np.sqrt(np.mean(residuals**2))

                # 5-fold CV on RMSE
                mse_scores = -cross_val_score(
                    model,
                    X,
                    y,
                    cv=5,
                    scoring="neg_mean_squared_error",
                    n_jobs=-1,
                )
                rmse = np.sqrt(mse_scores.mean())

                # 5-fold CV on MPE (log-space so that MPE is comparable across
                # algorithms that may use log-targets internally)
                mpe_scores = cross_val_score(
                    model,
                    X,
                    np.log10(y),
                    cv=5,
                    scoring=mean_percent_error,
                    n_jobs=-1,
                )
                mpe = mpe_scores.mean()

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_mpe = mpe
                    best_params = params
                    best_model = model

            # Store metrics
            self.min_rmse[i] = best_rmse
            self.min_mpe[i] = best_mpe

            # Re-fit best model on full data for this *n_cycle* setting
            best_model = self._build_model(**best_params)
            best_model.fit(X, y)
            self.trained_models.append(best_model)

            # Training MPE for diagnostic
            y_hat_full = self._predict(best_model, X, y)
            self.training_mpe[i] = (
                np.abs(y_hat_full - y) / y
            ).mean() * 100

        self._save_results()

    # ------------------------------------------------------------------
    # Helpers that subclasses can override
    # ------------------------------------------------------------------
    def _file_prefix(self) -> str:
        """Return filename prefix used when persisting artifacts."""
        # default behaviour: old <algorithm>Trainer -> lowercase algorithm name
        return self.__class__.__name__.replace("Trainer", "").lower()

    # ------------------------------------------------------------------
    # Saving helpers
    # ------------------------------------------------------------------
    def _save_results(self):
        stem = self._file_prefix()
        with open(
            os.path.join(self.results_dir, f"{stem}_trained_models.pkl"), "wb"
        ) as fh:
            pickle.dump(self.trained_models, fh)
        with open(
            os.path.join(self.results_dir, f"{stem}_crossvalid_percenterror.pkl"),
            "wb",
        ) as fh:
            pickle.dump(self.min_mpe, fh)
        with open(
            os.path.join(self.results_dir, f"{stem}_training_percenterror.pkl"),
            "wb",
        ) as fh:
            pickle.dump(self.training_mpe, fh)


# ---------------------------------------------------------------------------
# Concrete subclasses
# ---------------------------------------------------------------------------

from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor  # noqa: E402
from sklearn.linear_model import ElasticNetCV  # noqa: E402


class AdaBoostTrainer(BaseBatteryModelTrainer):
    def _file_prefix(self) -> str:
        return "AB"

    # ------------------------------------------------------------------
    # Persist additional artefacts for legacy plotting script
    # ------------------------------------------------------------------
    def _save_results(self):
        super()._save_results()  # save default files first

        # Compute extra outputs expected by legacy script
        test_mpe = np.zeros_like(self.N_CYCLES, dtype=float)
        predicted_cycle_lives_full = None
        train_predicted_cycle_lives_full = None

        for i, n_cycle in enumerate(self.N_CYCLES):
            # Load train and test datasets
            train_path = f"./training/cycles_2TO{n_cycle}{'_log' if self.use_log_features else ''}.csv"
            test_path = f"./testing/cycles_2TO{n_cycle}{'_log' if self.use_log_features else ''}.csv"

            X_train, y_train, _ = load_dataset(train_path, False, self.use_all_features, self.which_features)
            X_test, y_test, _ = load_dataset(test_path, False, self.use_all_features, self.which_features)

            model = self.trained_models[i]
            y_pred_test = model.predict(X_test)
            y_pred_train = model.predict(X_train)

            test_mpe[i] = (np.abs(y_pred_test - y_test) / y_test).mean() * 100

            # Save detailed predictions only for the final cycle (100) to match original usage
            if n_cycle == 100:
                predicted_cycle_lives_full = y_pred_test
                train_predicted_cycle_lives_full = y_pred_train

        # Cross‑validation mpe already stored as self.min_mpe
        data_tuple = (
            predicted_cycle_lives_full,
            train_predicted_cycle_lives_full,
            self.training_mpe,
            self.min_mpe,
            test_mpe,
        )
        with open(os.path.join(self.results_dir, f"{self._file_prefix()}_data.pkl"), "wb") as fh:
            pickle.dump(data_tuple, fh)

    def _get_param_grid(self):
        n_trees = [1000]
        lr = [0.1]
        return [
            {"n_estimators": n, "learning_rate": eta} for n, eta in product(n_trees, lr)
        ]

    def _build_model(self, **params):
        return AdaBoostRegressor(**params, random_state=0)


class RandomForestTrainer(BaseBatteryModelTrainer):
    def _file_prefix(self) -> str:
        return "RF"

    # ------------------------------------------------------------------
    # Persist additional artefacts for legacy plotting script
    # ------------------------------------------------------------------
    def _save_results(self):
        super()._save_results()  # save default files first

        # Compute extra outputs expected by legacy script
        test_mpe = np.zeros_like(self.N_CYCLES, dtype=float)
        predicted_cycle_lives_full = None
        train_predicted_cycle_lives_full = None

        for i, n_cycle in enumerate(self.N_CYCLES):
            # Load train and test datasets
            train_path = f"./training/cycles_2TO{n_cycle}{'_log' if self.use_log_features else ''}.csv"
            test_path = f"./testing/cycles_2TO{n_cycle}{'_log' if self.use_log_features else ''}.csv"

            X_train, y_train, _ = load_dataset(train_path, False, self.use_all_features, self.which_features)
            X_test, y_test, _ = load_dataset(test_path, False, self.use_all_features, self.which_features)

            model = self.trained_models[i]
            y_pred_test = model.predict(X_test)
            y_pred_train = model.predict(X_train)

            test_mpe[i] = (np.abs(y_pred_test - y_test) / y_test).mean() * 100

            # Save detailed predictions only for the final cycle (100) to match original usage
            if n_cycle == 100:
                predicted_cycle_lives_full = y_pred_test
                train_predicted_cycle_lives_full = y_pred_train

        # Cross‑validation mpe already stored as self.min_mpe
        data_tuple = (
            predicted_cycle_lives_full,
            train_predicted_cycle_lives_full,
            self.training_mpe,
            self.min_mpe,
            test_mpe,
        )
        with open(os.path.join(self.results_dir, f"{self._file_prefix()}_data.pkl"), "wb") as fh:
            pickle.dump(data_tuple, fh)

    def _get_param_grid(self):
        n_trees = [100, 1000]
        depths = [1, 5, 20, 200, None]
        return [
            {"n_estimators": n, "max_depth": d} for n, d in product(n_trees, depths)
        ]

    def _build_model(self, **params):
        return RandomForestRegressor(
            **params,
            max_features="sqrt",
            random_state=0,
            n_jobs=-1,
        )


class ElasticNetTrainer(BaseBatteryModelTrainer):
    def _file_prefix(self) -> str:
        return "enet"

    def __init__(
        self,
        results_root: Optional[str] = None,
        use_log_features: bool = True,
        use_all_features: bool = False,
        use_log_cycle_life: bool = True,
    ) -> None:
        super().__init__(results_root, use_log_features, use_all_features)
        self.use_log_cycle_life = use_log_cycle_life
        # additional storage specific to ENet
        self.optimal_l1_ratio = np.zeros_like(self.N_CYCLES, dtype=float)
        self.optimal_alpha = np.zeros_like(self.N_CYCLES, dtype=float)
        self.norm_coeffs: Optional[np.ndarray] = None

    # ElasticNetCV chooses its own params internally, so the grid is a noop
    def _get_param_grid(self):
        return [{}]

    def _build_model(self, **params):
        l1_ratio = [0.01, 0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1]
        return ElasticNetCV(
            l1_ratio=l1_ratio,
            cv=5,
            fit_intercept=True,
            max_iter=60_000,
            random_state=0,
            **params,
        )

    # Override full training loop because of log-target option and coefficient export
    def train(self):  # noqa: C901 — complex but contained to subclass
        X_cache, feature_names_cache = None, None
        self.norm_coeffs = np.zeros((len(self.WHICH_FEATURES), len(self.N_CYCLES)))

        for i, n_cycle in enumerate(self.N_CYCLES):
            X, y, feature_names = self._load_data(n_cycle)
            if X_cache is None:
                X_cache, feature_names_cache = X.copy(), feature_names.copy()

            model = self._build_model()
            y_fit = np.log10(y) if self.use_log_cycle_life else y
            model.fit(X, y_fit)
            y_pred = 10 ** model.predict(X) if self.use_log_cycle_life else model.predict(X)

            self.trained_models.append(model)
            residuals = y_pred - y
            self.min_rmse[i] = np.sqrt(np.mean(residuals**2))
            self.min_mpe[i] = (np.abs(residuals) / y).mean() * 100
            self.training_mpe[i] = (np.abs(residuals) / y).mean() * 100

            self.optimal_l1_ratio[i] = model.l1_ratio_
            self.optimal_alpha[i] = model.alpha_
            self.norm_coeffs[:, i] = model.coef_ * X.std(axis=0)

        # coeff export to CSV
        import pandas as pd  # local import to avoid pandas dependency if not used

        which = [i - 2 for i in self.WHICH_FEATURES] if not self.use_all_features else slice(None)
        df = pd.DataFrame(
            self.norm_coeffs,
            columns=self.N_CYCLES,
            index=[feature_names_cache[i] for i in which],
        )
        df.to_csv(os.path.join(self.results_dir, "enet_norm_coeffs.csv"))

        self._save_results()

    # ElasticNet uses model.predict internally with log-scale, so no override needed.


if __name__ == "__main__":
    AdaBoostTrainer().train()
    RandomForestTrainer().train()
    ElasticNetTrainer().train()
