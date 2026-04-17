"""
Data Science / ML Layer -- Financial Entropy Agent
Phase Space Regime Classification via Full-Covariance GMM.
Plane 1 (Price): [WPE, SPE_Z] -> Raw features -> Full GMM -> Deterministic/Transitional/Stochastic.
Plane 2 (Volume): [Shannon, SampEn] -> PowerTransform -> Full GMM -> Consensus/Dispersed/Erratic.
Note: Plane 1 uses NO preprocessing — raw entropy topology is preserved by design.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import PowerTransformer
from scipy.stats import normaltest

# NOTE: PowerTransformer va normaltest van duoc import cho VolumeRegimeClassifier.
# EntropyPhaseSpaceClassifier (Plane 1) KHONG dung PowerTransformer.

logger = logging.getLogger(__name__)


# ==============================================================================
# REFIT CADENCE FOR ROLLING SPE_Z STREAMING INFERENCE
# ==============================================================================
# Khi SPE_Z chuyen sang rolling 504d, distribution cua features troi theo
# thoi gian. Refit GMM moi 21 phien (~ 1 thang giao dich) tren cua so
# rolling 504 ngay gan nhat de centroid cluster bam sat dynamics hien tai.
REFIT_INTERVAL: int = 21
ROLLING_FIT_WINDOW: int = 504


# ==============================================================================
# REGIME LABELS
# ==============================================================================
REGIME_NAMES: dict[int, str] = {
    0: "Deterministic",   # Lowest entropy — HIGH RISK (strong trend, crash/rally)
    1: "Transitional",   # Mid entropy    — MODERATE RISK (phase transition)
    2: "Stochastic",     # Highest entropy — LOW RISK (random walk, normal market)
}


# ==============================================================================
# HYSTERESIS DEFAULTS (Schmitt-trigger filter for regime labels)
# ==============================================================================
# Calibrated by scripts/calibrate_hysteresis.py on VNINDEX, post-2020 only
# (microstructure shift makes pre-2020 bars non-comparable for flip-rate
# tuning). Achievable target band on VNINDEX: 4-10 regime flips/year. The
# raw GMM emits ~28-30 flips/yr; these defaults compress that to ~8 flips/yr
# (roughly one structural change every six weeks) while preserving 83% bar-
# level agreement with the raw classifier.
HYSTERESIS_DELTA_HARD: float = 0.60   # min posterior margin for instant flip
HYSTERESIS_DELTA_SOFT: float = 0.35   # min posterior margin for sustained flip
HYSTERESIS_T_PERSIST:  int   = 8      # consecutive bars required for soft flip


# ==============================================================================
# HYSTERESIS GMM WRAPPER (Schmitt trigger over GMM posteriors)
# ==============================================================================
class HysteresisGMMWrapper:
    """
    Schmitt-trigger filter over a fitted GMM's posterior probabilities.

    Goal: suppress single-bar regime flips ("flicker") that come from noise
    near cluster boundaries, while still permitting:
      - INSTANT flips when the new regime's posterior dominates the held
        regime by `delta_hard` (a hard, decisive penetration);
      - DELAYED flips when the new regime leads by at least `delta_soft`
        for `t_persist` consecutive bars (sustained drift).

    The wrapper is pure post-processing: it never touches the underlying GMM
    fit. Pass any classifier with a `.gmm.predict_proba()` method (the price
    EntropyPhaseSpaceClassifier and VolumeRegimeClassifier both qualify) and
    a label-mapping function that converts raw cluster indices -> semantic
    regime labels (so hysteresis operates in semantic-label space).

    Streaming usage:
        wrapper = HysteresisGMMWrapper(price_clf)
        for x_t in feature_stream:
            label_t = wrapper.step(x_t)

    Batch usage:
        labels = wrapper.transform(features_matrix)

    State (`held_label`, `pending_label`, `pending_count`) persists across
    `step()` calls — reset between independent series via `.reset()`.
    """

    def __init__(
        self,
        base_clf,
        delta_hard: float = HYSTERESIS_DELTA_HARD,
        delta_soft: float = HYSTERESIS_DELTA_SOFT,
        t_persist: int = HYSTERESIS_T_PERSIST,
        label_map: dict[int, int] | None = None,
    ) -> None:
        if delta_hard < delta_soft:
            raise ValueError("delta_hard must be >= delta_soft")
        if t_persist < 1:
            raise ValueError("t_persist must be >= 1")

        self.base_clf = base_clf
        self.delta_hard = float(delta_hard)
        self.delta_soft = float(delta_soft)
        self.t_persist = int(t_persist)

        # Cluster-index -> semantic-label map. Defaults to the base classifier's
        # _cluster_to_regime if it exists; otherwise identity.
        if label_map is None:
            label_map = getattr(base_clf, "_cluster_to_regime", None) or {}
        self._label_map = dict(label_map)

        self.reset()

    def reset(self) -> None:
        """Clear streaming state — call between independent series."""
        self.held_label: int | None = None
        self.pending_label: int | None = None
        self.pending_count: int = 0

    def _gmm(self):
        return getattr(self.base_clf, "gmm", self.base_clf)

    def _semantic(self, cluster_idx: int) -> int:
        return self._label_map.get(int(cluster_idx), int(cluster_idx))

    def _aggregate_semantic(self, proba_raw: np.ndarray) -> np.ndarray:
        """
        Aggregate raw cluster posteriors into semantic-label posteriors.
        Accepts (n_clusters,) for a single bar or (T, n_clusters) for a batch.
        """
        proba_raw = np.atleast_2d(proba_raw)  # (T, n_clusters)
        n_clusters = proba_raw.shape[1]
        if self._label_map:
            n_labels = max(self._label_map.values()) + 1
        else:
            n_labels = n_clusters
        proba_sem = np.zeros((proba_raw.shape[0], int(n_labels)), dtype=np.float64)
        for cluster_idx in range(n_clusters):
            proba_sem[:, self._semantic(cluster_idx)] += proba_raw[:, cluster_idx]
        return proba_sem

    def _step_with_proba(self, proba: np.ndarray) -> int:
        """State-machine core; takes pre-computed semantic posteriors."""
        top_label = int(np.argmax(proba))

        if self.held_label is None:
            self.held_label = top_label
            self.pending_label = None
            self.pending_count = 0
            return self.held_label

        if top_label == self.held_label:
            self.pending_label = None
            self.pending_count = 0
            return self.held_label

        margin = float(proba[top_label] - proba[self.held_label])

        if margin >= self.delta_hard:
            self.held_label = top_label
            self.pending_label = None
            self.pending_count = 0
            return self.held_label

        if margin >= self.delta_soft:
            if self.pending_label == top_label:
                self.pending_count += 1
            else:
                self.pending_label = top_label
                self.pending_count = 1
            if self.pending_count >= self.t_persist:
                self.held_label = top_label
                self.pending_label = None
                self.pending_count = 0
            return self.held_label

        self.pending_label = None
        self.pending_count = 0
        return self.held_label

    def step(self, x_t: np.ndarray) -> int:
        """
        Process one observation, return the (post-hysteresis) semantic label.

        Decision tree per bar:
          1. If no label is held yet, adopt the argmax label.
          2. Compute margin = P(top) - P(held). If top == held, hold.
          3. If margin >= delta_hard -> flip immediately, clear pending.
          4. Elif margin >= delta_soft -> increment pending counter on the
             same candidate. After t_persist consecutive bars, flip.
          5. Else (or candidate changed) -> reset the pending counter.
        """
        x_t = np.asarray(x_t, dtype=np.float64).ravel()
        proba_raw = self._gmm().predict_proba(x_t.reshape(1, -1))[0]
        proba = self._aggregate_semantic(proba_raw)[0]
        return self._step_with_proba(proba)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply hysteresis to a (T x n_features) matrix in chronological order.
        Returns a (T,) array of semantic labels. Resets state before processing.

        Performance: posteriors are computed in a SINGLE vectorized
        predict_proba() call on the full matrix; only the sequential state
        machine (which has loop-carried dependencies) iterates in Python.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        self.reset()
        proba_raw = self._gmm().predict_proba(X)            # (T, n_clusters)
        proba_sem = self._aggregate_semantic(proba_raw)     # (T, n_labels)
        out = np.empty(X.shape[0], dtype=np.int64)
        for i in range(X.shape[0]):
            out[i] = self._step_with_proba(proba_sem[i])
        return out


# ==============================================================================
# ENTROPY PHASE SPACE CLASSIFIER (PLANE 1: [WPE, SPE_Z])
# ==============================================================================
class EntropyPhaseSpaceClassifier:
    """
    Plane 1 GMM Classifier trong Raw Entropy Phase Space.

    Pipeline:
        1. Feed RAW [WPE, SPE_Z] truc tiep vao GMM (KHONG transform).
        2. GaussianMixture(n=3, covariance_type='full')
           -> Moi cluster co ma tran hiep phuong sai RIENG (2x2).
              Full covariance tu dong xu ly scale khac nhau giua features.
        3. Semantic sorting: sum of centroid means (WPE_mean + SPE_Z_mean).
           Lowest combined entropy -> Stable, Mid -> Fragile, Highest -> Chaos.

    Tai sao KHONG dung PowerTransformer?
        - WPE bounded [0, 1], SPE_Z la Z-score (mean~0, std~1).
        - PowerTransform ep du lieu thanh Gaussian blob, pha huy
          topological boundaries tu nhien cua entropy metrics.
        - Full-covariance GMM du suc xu ly varying scales giua features.
        - Khac Plane 2 (Volume): Vol_Shannon/Vol_SampEn co phan phoi lech
          (right-skewed) nen Plane 2 can Yeo-Johnson truoc GMM.
          Plane 1 entropy metrics da co phan phoi phu hop GMM truc tiep.
    """

    def __init__(
        self,
        n_components: int = 3,
        random_state: int = 42,
    ) -> None:
        self.n_components = n_components
        self.gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",  # Moi cluster co covariance rieng
            n_init=10,
            max_iter=500,
            random_state=random_state,
        )
        self._cluster_to_regime: dict[int, int] = {}
        self.X_fitted: np.ndarray | None = None
        self.last_fit_date = None  # streaming refit cadence anchor

    def fit(self, features: np.ndarray) -> "EntropyPhaseSpaceClassifier":
        """
        Fit Full GMM truc tiep tren raw [WPE, SPE_Z].
        """
        self.X_fitted = np.asarray(features, dtype=np.float64)
        self.gmm.fit(self.X_fitted)
        self._map_clusters_by_combined_entropy()
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict semantic regime labels (0=Stable, 1=Fragile, 2=Chaos)."""
        raw_labels = self.gmm.predict(features)
        mapped = np.array([self._cluster_to_regime.get(l, l) for l in raw_labels])
        return mapped

    def fit_predict(self, features: np.ndarray) -> np.ndarray:
        """Fit + predict trong 1 buoc."""
        self.fit(features)
        raw_labels = self.gmm.predict(self.X_fitted)
        mapped = np.array([self._cluster_to_regime.get(l, l) for l in raw_labels])
        return mapped

    def get_regime_name(self, label: int) -> str:
        """Chuyen semantic label thanh ten regime."""
        return REGIME_NAMES.get(label, f"Unknown_{label}")

    def get_ellipse_params(self, cluster_idx: int, n_std: float = 2.0) -> dict:
        """
        Tinh tham so ellipse (95% confidence) cho cluster trong raw space.
        Voi covariance_type='full', gmm.covariances_ la (n_components, n_features, n_features).
        Moi cluster co ellipse RIENG voi hinh dang khac nhau.
        Returns: {"center": (cx, cy), "width": w, "height": h, "angle": theta_deg}
        """
        mean = self.gmm.means_[cluster_idx]
        cov = self.gmm.covariances_[cluster_idx]

        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width = 2 * n_std * np.sqrt(eigenvalues[0])
        height = 2 * n_std * np.sqrt(eigenvalues[1])
        return {
            "center": (float(mean[0]), float(mean[1])),
            "width": float(width),
            "height": float(height),
            "angle": float(angle),
        }

    def fit_or_refit(
        self,
        features_df: pd.DataFrame,
        current_date,
        refit_interval: int = REFIT_INTERVAL,
        rolling_window: int = ROLLING_FIT_WINDOW,
    ) -> np.ndarray:
        """
        Streaming inference helper: refit GMM moi `refit_interval` phien
        tren cua so rolling `rolling_window` ngay gan nhat (rolling SPE_Z
        distribution troi theo thoi gian, centroid co dinh se lech).

        Tra ve nhan regime cho dong cuoi cung cua features_df.
        """
        features_df = features_df.dropna()
        if features_df.empty:
            return np.array([], dtype=int)

        need_refit = self.last_fit_date is None or (
            (pd.Timestamp(current_date) - pd.Timestamp(self.last_fit_date)).days
            >= refit_interval
        )
        if need_refit:
            window_df = features_df.iloc[-rolling_window:]
            self.fit(window_df.values)
            self.last_fit_date = current_date

        latest = features_df.iloc[[-1]].values
        return self.predict(latest)

    def _map_clusters_by_combined_entropy(self) -> None:
        """
        Map GMM cluster indices -> semantic regime labels (0,1,2)
        dua tren tong cac centroid means (WPE_mean + SPE_Z_mean).
        Lowest combined entropy -> 0 (Stable).
        Highest combined entropy -> 2 (Chaos).
        """
        combined = self.gmm.means_.sum(axis=1)  # WPE + SPE_Z per cluster
        sorted_indices = np.argsort(combined)
        self._cluster_to_regime = {
            int(sorted_indices[i]): i for i in range(self.n_components)
        }


# ==============================================================================
# CONVENIENCE: FIT + PREDICT (FUNCTIONAL API) -- PRICE PLANE
# ==============================================================================
def fit_predict_regime(
    features: np.ndarray,
    n_components: int = 3,
) -> tuple[np.ndarray, EntropyPhaseSpaceClassifier]:
    """
    Ham tien ich: tao EntropyPhaseSpaceClassifier, fit va predict.
    Input:  features (N x 2 array: [WPE, SPE_Z])
    Output: (labels, fitted_classifier)
    Labels duoc quyet dinh boi Full GMM trong Entropy Phase Space.
    """
    clf = EntropyPhaseSpaceClassifier(n_components=n_components)
    labels = clf.fit_predict(features)
    return labels, clf


# ==============================================================================
# VOLUME REGIME LABELS (PLANE 2: LIQUIDITY STRUCTURE)
# ==============================================================================
VOLUME_REGIME_NAMES: dict[int, str] = {
    0: "Consensus Flow",
    1: "Dispersed Flow",
    2: "Erratic/Noisy Flow",
}


# ==============================================================================
# VOLUME REGIME CLASSIFIER (PLANE 2)
# ==============================================================================
class VolumeRegimeClassifier:
    """
    GMM Unsupervised cho Volume Entropy Plane.
    Features dau vao: [Vol_Shannon, Vol_SampEn].
    Mapping: sap xep cluster theo tong centroid (Shannon + SampEn).
    """

    def __init__(self, n_components: int = 3, random_state: int = 42) -> None:
        self.n_components = n_components
        self.power_tf = PowerTransformer(method="yeo-johnson", standardize=True)
        self.gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            n_init=10,
            random_state=random_state,
        )
        self._cluster_to_regime: dict[int, str] = {}

    def fit(self, features: np.ndarray) -> "VolumeRegimeClassifier":
        """Fit PowerTransformer + GMM tren [Vol_Shannon, Vol_SampEn] (N x 2)."""
        self.X_transformed = self.power_tf.fit_transform(features)
        self.gmm.fit(self.X_transformed)
        self._map_clusters(features)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict volume regime labels."""
        X_tf = self.power_tf.transform(features)
        return self.gmm.predict(X_tf)

    def fit_predict(self, features: np.ndarray) -> np.ndarray:
        """Fit + predict trong 1 buoc."""
        self.fit(features)
        return self.gmm.predict(self.X_transformed)

    def get_regime_name(self, label: int) -> str:
        """Chuyen cluster index thanh ten volume regime."""
        return self._cluster_to_regime.get(label, f"Unknown_{label}")

    def _map_clusters(self, original_features: np.ndarray) -> None:
        """Mapping: centroid sort tren original space (sum of Shannon + SampEn)."""
        labels = self.gmm.predict(self.X_transformed)
        centroids = np.array([
            original_features[labels == k].mean(axis=0)
            for k in range(self.n_components)
        ])
        sort_key = centroids.sum(axis=1)
        sorted_indices = np.argsort(sort_key)

        regime_order = list(VOLUME_REGIME_NAMES.values())
        self._cluster_to_regime = {
            int(sorted_indices[i]): regime_order[min(i, len(regime_order) - 1)]
            for i in range(self.n_components)
        }


# ==============================================================================
# CONVENIENCE: FIT + PREDICT (FUNCTIONAL API) -- VOLUME PLANE
# ==============================================================================
def fit_predict_volume_regime(
    features: np.ndarray,
    n_components: int = 3,
) -> tuple[np.ndarray, VolumeRegimeClassifier]:
    """
    Ham tien ich: tao VolumeRegimeClassifier, fit va predict.
    Input:  features (N x 2 array: [Vol_Shannon, Vol_SampEn])
    Output: (labels, fitted_classifier)
    """
    clf = VolumeRegimeClassifier(n_components=n_components)
    labels = clf.fit_predict(features)
    return labels, clf


# ==============================================================================
# TESTING BLOCK
# ==============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="  [%(levelname)s] %(message)s")
    np.random.seed(42)

    print("=" * 60)
    print("TEST 1: Plane 1 -- Raw Entropy Phase Space (Full GMM, NO transform)")
    print("=" * 60)

    # Tao 3 cum gia lap: WPE (X, [0,1]) va SPE_Z (Y, z-score)
    stable = np.column_stack([
        np.random.beta(5, 8, 300) * 0.3 + 0.3,   # WPE ~0.3-0.6
        np.random.randn(300) * 0.4 - 0.8          # SPE_Z low (near -1)
    ])
    fragile = np.column_stack([
        np.random.beta(5, 5, 200) * 0.2 + 0.6,    # WPE ~0.6-0.8
        np.random.randn(200) * 0.5                 # SPE_Z mid (near 0)
    ])
    chaos = np.column_stack([
        np.random.beta(8, 3, 100) * 0.15 + 0.82,  # WPE ~0.82-0.97
        np.random.randn(100) * 0.6 + 0.8           # SPE_Z high (near +1)
    ])

    fake_features = np.vstack([stable, fragile, chaos])
    print(f"  Feature matrix shape : {fake_features.shape}")
    print(f"  Columns semantics    : [WPE (raw, [0,1]), SPE_Z (raw, z-score)]")

    labels, clf = fit_predict_regime(fake_features, n_components=3)

    print(f"\n  GMM covariance_type  : {clf.gmm.covariance_type}")
    print(f"  PowerTransformer     : NONE (raw features)")

    print(f"\n  GMM Centroids (Raw Phase Space):")
    for i in range(3):
        regime_idx = clf._cluster_to_regime.get(i, i)
        name = clf.get_regime_name(regime_idx)
        mean = clf.gmm.means_[i]
        combined = mean.sum()
        print(f"    Cluster {i} -> {name:15s} : WPE={mean[0]:.3f}, SPE_Z={mean[1]:+.3f}, sum={combined:+.3f}")

    print(f"\n  Combined-entropy sorting check:")
    combined_scores = clf.gmm.means_.sum(axis=1)
    print(f"    Combined scores: {[f'{s:+.3f}' for s in combined_scores]}")

    print(f"\n  Predicted labels     : {np.unique(labels)}")
    print(f"  Label distribution   :")
    for lbl in np.unique(labels):
        name = clf.get_regime_name(lbl)
        count = (labels == lbl).sum()
        print(f"    {lbl} -> {name:25s} (n={count})")

    # Ellipse params (full covariance in RAW space)
    print(f"\n  95% Confidence Ellipses (FULL, RAW space):")
    for i in range(3):
        e = clf.get_ellipse_params(i, n_std=2.0)
        regime_idx = clf._cluster_to_regime.get(i, i)
        name = clf.get_regime_name(regime_idx)
        print(f"    {name:15s}: center=({e['center'][0]:.3f},{e['center'][1]:+.3f}), w={e['width']:.3f}, h={e['height']:.3f}, angle={e['angle']:.1f}")

    print()
    print("=" * 60)
    print("TEST 2: Plane 2 -- Volume Regime (PowerTransform + GMM)")
    print("=" * 60)

    consensus = np.random.randn(50, 2) * 0.05 + np.array([0.65, 0.8])
    dispersed = np.random.randn(50, 2) * 0.05 + np.array([0.85, 1.2])
    erratic = np.random.randn(50, 2) * 0.05 + np.array([0.95, 2.5])

    vol_features = np.vstack([consensus, dispersed, erratic])
    print(f"  Feature matrix shape : {vol_features.shape}")

    vol_labels, vol_clf = fit_predict_volume_regime(vol_features, n_components=3)

    print(f"\n  Predicted labels     : {np.unique(vol_labels)}")
    for lbl in np.unique(vol_labels):
        name = vol_clf.get_regime_name(lbl)
        count = (vol_labels == lbl).sum()
        print(f"    {lbl} -> {name:25s} (n={count})")
