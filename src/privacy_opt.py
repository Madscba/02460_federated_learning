import torch
from opacus import PrivacyEngine
from opacus.privacy_engine import privacy_analysis

DEFAULT_ALPHAS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

class DP_SGD():
    def __init__(
            self,
            net,
            learning_rate: float = 0.001,
            momentum: float = 0.90,
            sample_rate: float = None,
            alphas: list = DEFAULT_ALPHAS,
            max_grad_norm: float = None,
            noise_multiplier: float = None,
            target_delta: float = 1e-6,
            lib: bool = False):

        self.net = net
        self.steps = 0
        self.alphas = alphas
        self.sample_rate = sample_rate
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.target_delta = target_delta
        self.lib = lib
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=learning_rate, momentum=momentum)
        self.privacy_engine = self.set_privacy_engine()

    def set_privacy_engine(self, privacy_engine=None):
        if self.lib:
            privacy_engine = PrivacyEngine(self.net,
                                           sample_rate=self.sample_rate,
                                           target_delta=self.target_delta,
                                           max_grad_norm=self.max_grad_norm,
                                           noise_multiplier=self.noise_multiplier,
                                           accountant='gdp'
                                           )
            privacy_engine.attach(self.optimizer)
        return privacy_engine

    def step(self):
        self.steps += 1
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_renyi_divergence(self):
        rdp = torch.tensor(
            privacy_analysis.compute_rdp(
                self.sample_rate, self.noise_multiplier, 1, self.alphas
            )
        )
        return rdp

    def get_privacy_spent(self, target_delta: float = None):
        """
        Computes the (epsilon, delta) privacy budget spent so far.

        This method converts from an (alpha, epsilon)-DP guarantee for all alphas that
        the ``PrivacyEngine`` was initialized with. It returns the best epsilon.

        Args:
            target_delta: The Target delta. If None, it will default to the privacy
                engine's target delta.

        Returns:
            Pair of epsilon and optimal order alpha.
        """
        if target_delta is None:
            if self.target_delta is None:
                raise ValueError(
                    "If self.target_delta is not specified, target_delta should be set as argument to get_privacy_spent."
                )
            target_delta = self.target_delta
        rdp = self.get_renyi_divergence() * self.steps
        eps, best_alpha = privacy_analysis.get_privacy_spent(
            self.alphas, rdp, target_delta
        )
        return float(eps)