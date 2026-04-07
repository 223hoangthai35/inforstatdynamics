"""
Data Science / ML Layer -- Financial Entropy Agent
Standardized Shock Space: PowerTransform + Tied-Covariance GMM Regime Classification.
Plane 1 (Price): [WPE, Momentum_Entropy_Flux] -> PowerTransform -> Tied GMM -> Stable/Fragile/Chaos.
Plane 2 (Volume): [Shannon, SampEn] -> PowerTransform -> GMM -> Consensus/Dispersed/Erratic.
"""

import logging
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import PowerTransformer
from scipy.stats import normaltest

logger = logging.getLogger(__name__)


# ==============================================================================
# REGIME LABELS
# ==============================================================================
REGIME_NAMES: dict[int, str] = {
    0: "Stable",
    1: "Fragile",
    2: "Chaos",
}


# ==============================================================================
# STANDARDIZED SHOCK SPACE CLASSIFIER (PLANE 1: KINEMATIC)
# ==============================================================================
class KinematicRegimeClassifier:
    """
    Plane 1 GMM Classifier trong Standardized Shock Space.

    Pipeline:
        1. PowerTransformer(yeo-johnson) -> Gaussianize skewed [WPE, Flux]
        2. D'Agostino normality test -> validate Gaussian assumption
        3. GaussianMixture(n=3, covariance_type='tied', means_init along X)
           -> 3 clusters CHIA CHUNG 1 ma tran hiep phuong sai.
              Ngan chan 1 cluster ve ellipse khong lo boc quanh cluster khac.
        4. Centroid sorting: argsort(means_[:, 0]) -> Stable/Fragile/Chaos.

    Tai sao covariance_type='tied'?
        Voi 'full', moi cluster co covariance rieng (2x2). Cluster co
        variance Y lon se ve ellipse doc boc concentric quanh cluster khac.
        Voi 'tied', ca 3 clusters chia chung 1 ma tran covariance duy nhat.
        -> Moi cluster co CUNG hinh dang ellipse, chi khac vi tri centroid.
        -> Ket hop voi means_init doc theo X-axis, GMM buoc phai cat
           LEFT-TO-RIGHT thay vi tao topology concentric.
    """

    def __init__(
        self,
        n_components: int = 3,
        random_state: int = 42,
    ) -> None:
        self.n_components = n_components
        self.power_tf = PowerTransformer(method="yeo-johnson", standardize=True)
        # means_init: cam co doc truc X trong transformed space (mean~0, std~1)
        self._means_init = np.array([[-1.5, 0.0], [0.0, 0.0], [1.5, 0.0]])
        self.gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="tied",  # 3 clusters chia chung 1 covariance matrix
            means_init=self._means_init,
            max_iter=500,
            random_state=random_state,
        )
        self._cluster_to_regime: dict[int, int] = {}
        self.X_transformed: np.ndarray | None = None
        self.normality_pvalue: float | None = None

    def fit(self, features: np.ndarray) -> "KinematicRegimeClassifier":
        """
        Fit pipeline: PowerTransform -> Normality Test -> Tied GMM.
        """
        # Step 1: PowerTransform
        self.X_transformed = self.power_tf.fit_transform(features)

        # Step 2: D'Agostino normality test tren WPE Shock axis
        try:
            stat, p_value = normaltest(self.X_transformed[:, 0])
            self.normality_pvalue = float(p_value)
            if p_value < 0.01:
                logger.warning(
                    f"Normality Test Warning: WPE Shock p-value = {p_value:.4e}. "
                    f"Data might still have heavy tails after PowerTransform."
                )
            else:
                logger.info(f"Normality Test Passed: WPE Shock p-value = {p_value:.4f}.")
        except Exception:
            self.normality_pvalue = None

        # Step 3: Fit Tied GMM
        self.gmm.fit(self.X_transformed)
        self._map_clusters_by_centroid()
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Transform raw features -> PowerTransformed space (for scatter plot)."""
        return self.power_tf.transform(features)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict semantic regime labels (0=Stable, 1=Fragile, 2=Chaos)."""
        X_tf = self.power_tf.transform(features)
        raw_labels = self.gmm.predict(X_tf)
        mapped = np.array([self._cluster_to_regime.get(l, l) for l in raw_labels])
        return mapped

    def fit_predict(self, features: np.ndarray) -> np.ndarray:
        """Fit + predict trong 1 buoc."""
        self.fit(features)
        raw_labels = self.gmm.predict(self.X_transformed)
        mapped = np.array([self._cluster_to_regime.get(l, l) for l in raw_labels])
        return mapped

    def get_regime_name(self, label: int) -> str:
        """Chuyen semantic label thanh ten regime."""
        return REGIME_NAMES.get(label, f"Unknown_{label}")

    def get_gmm_proba(self, features: np.ndarray) -> np.ndarray:
        """Soft GMM probabilities."""
        X_tf = self.power_tf.transform(features)
        return self.gmm.predict_proba(X_tf)

    def get_ellipse_params(self, cluster_idx: int, n_std: float = 2.0) -> dict:
        """
        Tinh tham so ellipse (95% confidence) cho cluster trong transformed space.
        Voi covariance_type='tied', gmm.covariances_ la 1 ma tran (2,2) duy nhat
        dung chung cho ca 3 clusters. Moi cluster chi khac vi tri centroid.
        Returns: {"center": (cx, cy), "width": w, "height": h, "angle": theta_deg}
        """
        mean = self.gmm.means_[cluster_idx]
        # 'tied' -> covariances_ la (n_features, n_features), KHONG phai (n_components, n_f, n_f)
        cov = self.gmm.covariances_

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

    def _map_clusters_by_centroid(self) -> None:
        """
        Map GMM cluster indices -> semantic regime labels (0,1,2)
        dua tren sorted X-axis centroids.
        Lowest mean X -> 0 (Stable), Middle -> 1 (Fragile), Highest -> 2 (Chaos).
        """
        means_x = self.gmm.means_[:, 0]
        sorted_indices = np.argsort(means_x)
        self._cluster_to_regime = {
            int(sorted_indices[i]): i for i in range(self.n_components)
        }


# ==============================================================================
# CONVENIENCE: FIT + PREDICT (FUNCTIONAL API) -- PRICE PLANE
# ==============================================================================
def fit_predict_regime(
    features: np.ndarray,
    n_components: int = 3,
) -> tuple[np.ndarray, KinematicRegimeClassifier]:
    """
    Ham tien ich: tao KinematicRegimeClassifier, fit va predict.
    Input:  features (N x 2 array: [WPE, Momentum_Entropy_Flux])
    Output: (labels, fitted_classifier)
    Labels duoc quyet dinh boi Tied GMM trong Standardized Shock Space.
    """
    clf = KinematicRegimeClassifier(n_components=n_components)
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
    print("TEST 1: Plane 1 -- Standardized Shock Space (Tied GMM)")
    print("=" * 60)

    # Tao 3 cum gia lap voi phan phoi skewed (realistic WPE)
    stable = np.column_stack([
        np.random.beta(5, 8, 300) * 0.3 + 0.3,   # ~0.3-0.6 (left-skewed)
        np.random.randn(300) * 0.5                 # Low flux
    ])
    fragile = np.column_stack([
        np.random.beta(5, 5, 200) * 0.2 + 0.6,    # ~0.6-0.8
        np.random.randn(200) * 1.5                 # Medium flux
    ])
    chaos = np.column_stack([
        np.random.beta(8, 3, 100) * 0.15 + 0.82,  # ~0.82-0.97
        np.random.randn(100) * 3.0                 # High flux
    ])

    fake_features = np.vstack([stable, fragile, chaos])
    print(f"  Feature matrix shape : {fake_features.shape}")
    print(f"  Columns semantics    : [WPE, Momentum_Entropy_Flux]")

    labels, clf = fit_predict_regime(fake_features, n_components=3)

    print(f"\n  Normality test p-value (WPE Shock): {clf.normality_pvalue:.4f}")
    print(f"  GMM covariance_type: {clf.gmm.covariance_type}")
    print(f"  Shared covariance matrix:")
    print(f"    {clf.gmm.covariances_}")

    print(f"\n  GMM Centroids (Transformed Shock Space):")
    for i in range(3):
        regime_idx = clf._cluster_to_regime.get(i, i)
        name = clf.get_regime_name(regime_idx)
        mean = clf.gmm.means_[i]
        print(f"    Cluster {i} -> {name:15s} : X={mean[0]:+.3f}, Y={mean[1]:+.3f}")

    print(f"\n  Left-to-Right check:")
    x_means = clf.gmm.means_[:, 0]
    sorted_x = np.sort(x_means)
    is_monotonic = all(sorted_x[i] < sorted_x[i+1] for i in range(len(sorted_x)-1))
    print(f"    Sorted X: {[f'{x:+.3f}' for x in sorted_x]}")
    print(f"    Monotonically increasing: {is_monotonic}")

    print(f"\n  Predicted labels     : {np.unique(labels)}")
    print(f"  Label distribution   :")
    for lbl in np.unique(labels):
        name = clf.get_regime_name(lbl)
        count = (labels == lbl).sum()
        print(f"    {lbl} -> {name:25s} (n={count})")

    # Ellipse params (shared shape, different centers)
    print(f"\n  95% Confidence Ellipses (TIED = same shape, different center):")
    for i in range(3):
        e = clf.get_ellipse_params(i, n_std=2.0)
        regime_idx = clf._cluster_to_regime.get(i, i)
        name = clf.get_regime_name(regime_idx)
        print(f"    {name:15s}: center=({e['center'][0]:+.2f},{e['center'][1]:+.2f}), w={e['width']:.2f}, h={e['height']:.2f}, angle={e['angle']:.1f}")

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
