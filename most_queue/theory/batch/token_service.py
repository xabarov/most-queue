"""Finite token-length laws and Yang--Jiao--Xu (2024) batch service costs.

arXiv:2407.05347v1, equations (2)--(5), (18), (26). Lengths here count ALL
output tokens; decode length is total minus the first token. Elastic batches
are gated: no admissions until the last member completes.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import lognorm, poisson, truncnorm

from most_queue.theory.batch.dynamic_batching import _positive_integer


class _FiniteCounts:
    """Finite FFT count law with positive tail sums and stable quantiles.

    Inverse CDF must not return index zero when accumulated roundoff puts
    its final entry just below 1-q. Survival is summed from the right, not
    obtained by subtracting the CDF from one.
    """

    def __init__(self, masses):
        self.masses = masses
        self.cumulative = np.minimum(1.0, np.cumsum(masses))
        self.cumulative[-1] = 1.0
        self.survival = np.concatenate((np.cumsum(masses[:0:-1])[::-1], [0.0]))

    def pmf(self, counts):
        """Probability at integer counts, zero outside support."""
        k = np.asarray(counts)
        valid = (k >= 0) & (k < len(self.masses)) & (k == np.floor(k))
        indices = np.clip(k, 0, len(self.masses) - 1).astype(int)
        return np.where(valid, self.masses[indices], 0.0)

    def cdf(self, counts):
        """P(A<=counts), with an exactly normalized endpoint."""
        k = np.asarray(counts)
        indices = np.clip(np.floor(k), 0, len(self.masses) - 1).astype(int)
        return np.where(k < 0, 0.0, np.where(k >= len(self.masses) - 1, 1.0, self.cumulative[indices]))

    def sf(self, counts):
        """P(A>counts), retaining small positive probabilities."""
        k = np.asarray(counts)
        indices = np.clip(np.floor(k), 0, len(self.masses) - 1).astype(int)
        return np.where(k < 0, 1.0, np.where(k >= len(self.masses) - 1, 0.0, self.survival[indices]))

    def ppf(self, probability):
        """Left inverse CDF on [0,1]."""
        return float(np.searchsorted(self.cumulative, probability, side="left"))

    def isf(self, probability):
        """Left inverse survival, with no cancellation of rare tail mass."""
        return float(np.searchsorted(-self.survival, -probability, side="left"))

    def mean(self):
        """Count expectation."""
        return float(self.masses @ np.arange(len(self.masses)))

    def var(self):
        """Count variance."""
        return float(self.masses @ (np.arange(len(self.masses)) - self.mean()) ** 2)


class TokenLengthDistribution:
    """An immutable finite PMF of positive integer total output lengths.

    Arbitrary sparse empirical laws are supported. Clipping moves mass to
    the limit (censoring), never renormalizes a truncated density.
    """

    def __init__(self, lengths, probabilities, max_tokens=None):
        values, weights = np.asarray(lengths), np.asarray(probabilities, dtype=float)
        if values.ndim != 1 or weights.shape != values.shape or values.size == 0:
            raise ValueError("lengths/probabilities must be non-empty equal-length vectors")
        if not np.all(np.isfinite(values)) or np.any(values < 1) or np.any(values != np.floor(values)):
            raise ValueError("total lengths must be finite positive integers")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0) or abs(weights.sum() - 1) > 1e-12:
            raise ValueError("probabilities must be non-negative and sum to one")
        if max_tokens is not None:
            max_tokens = _positive_integer(max_tokens, "max_tokens")
        clipped = values if max_tokens is None else np.minimum(values, max_tokens)
        support, indices = np.unique(clipped[weights > 0], return_inverse=True)
        mass = np.bincount(indices, weights=weights[weights > 0])
        self.lengths, self.probabilities = support.astype(float), mass / mass.sum()
        self.clipped_fraction = float(weights[values > max_tokens].sum()) if max_tokens is not None else 0.0
        self.utility = float(weights @ (clipped / values))
        self.lengths.setflags(write=False)
        self.probabilities.setflags(write=False)

    @classmethod
    def uniform(cls, max_tokens):
        """Discrete uniform total length 1,...,max_tokens."""
        size = _positive_integer(max_tokens, "max_tokens")
        if size > 100_000:
            raise MemoryError("token support exceeds 100000")
        return cls(np.arange(1, size + 1), np.full(size, 1 / size))

    @classmethod
    def from_cdf(cls, cdf, max_tokens):
        """Law of min(max(1, ceil(X)), max_tokens) from a nonnegative X CDF.

        No silent tail truncation: the entire upper tail becomes a cap atom.
        The exact continuous utility in the tail is not recoverable from
        this censored PMF; utility is therefore NaN for this factory.
        """
        size = _positive_integer(max_tokens, "max_tokens")
        if size > 100_000:
            raise MemoryError("token support exceeds 100000")
        cumulative = np.asarray(cdf(np.arange(1, size)), dtype=float)
        if cumulative.shape != (size - 1,) or not np.all(np.isfinite(cumulative)):
            raise ValueError("CDF must return finite values on the token grid")
        mass = np.diff(np.concatenate(([0.0], cumulative, [1.0])))
        result = cls(np.arange(1, size + 1), mass)
        result.clipped_fraction = float(1 - cdf(size))
        if not 0 <= result.clipped_fraction <= 1:
            raise ValueError("invalid CDF at max_tokens")
        result.utility = float("nan")
        return result

    @classmethod
    def clipped_lognormal(cls, log_mean, log_std, max_tokens):
        """Ceiled lognormal output lengths with an explicit cap atom."""
        if (
            not np.isfinite(log_mean)
            or not np.isfinite(log_std)
            or log_std <= 0
            or log_mean >= np.log(np.finfo(float).max)
        ):
            raise ValueError("lognormal parameters must have finite scale and positive std")
        return cls.from_cdf(lognorm(s=log_std, scale=np.exp(log_mean)).cdf, max_tokens)

    @classmethod
    def clipped_normal(cls, mean, std, max_tokens):
        """Nonnegative truncated-normal variate, then ceiling and clipping."""
        if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
            raise ValueError("normal parameters must be finite with positive std")
        return cls.from_cdf(truncnorm(-mean / std, np.inf, loc=mean, scale=std).cdf, max_tokens)

    def maximum_pmf(self, batch_size):
        """PMF of max(total length) among iid batch members, using stable powers."""
        size = _positive_integer(batch_size, "batch_size")
        cumulative = np.minimum(1.0, np.cumsum(self.probabilities))
        upper = cumulative**size
        mass = np.diff(np.concatenate(([0.0], upper)))
        return mass / mass.sum()

    def moments(self, decode=False):
        """First two raw moments; decode=True subtracts the first output token."""
        values = self.lengths - int(decode)
        return float(self.probabilities @ values), float(self.probabilities @ values**2)

    def sample(self, generator, size):
        """Sample actual individual total output lengths, not their batch maximum."""
        return generator.choice(self.lengths, size=size, p=self.probabilities)


@dataclass(frozen=True)
class TokenBatchService:
    """Cost k1*b+k2+(k3*b+k4)*max(decode lengths), or elastic shrinking.

    k1>0; other coefficients >=0. No KV-memory ceiling, prefill overlap,
    length-dependent admission or arrivals inside the running batch.
    """

    tokens: TokenLengthDistribution
    k1: float
    k2: float
    k3: float
    k4: float
    elastic: bool = False
    max_fft_points: int = 262_144

    def __post_init__(self):
        coefficients = np.asarray([self.k1, self.k2, self.k3, self.k4])
        if not isinstance(self.tokens, TokenLengthDistribution):
            raise ValueError("tokens must be a TokenLengthDistribution")
        if not np.all(np.isfinite(coefficients)) or self.k1 <= 0 or np.any(coefficients < 0):
            raise ValueError("require finite k1>0, k2/k3/k4>=0")
        _positive_integer(self.max_fft_points, "max_fft_points")
        if not isinstance(self.elastic, (bool, np.bool_)):
            raise ValueError("elastic must be boolean")

    @property
    def envelope(self):
        """Pathwise H<=a*b+c, hence global first/second moment certificates."""
        limit = self.tokens.lengths[-1] - 1
        a, c = self.k1 + self.k3 * limit, self.k2 + self.k4 * limit
        return (a, c), (a * a, 2 * a * c, c * c)

    def moments(self, batch_size):
        """Exact conditional batch-duration first/second moments of the finite law."""
        size = _positive_integer(batch_size, "batch_size")
        n, p = self.tokens.lengths - 1, self.tokens.probabilities
        max_p = self.tokens.maximum_pmf(size)
        max_mean, max_second = float(max_p @ n), float(max_p @ n**2)
        pre = self.k1 * size + self.k2
        if not self.elastic:
            step = self.k3 * size + self.k4
            return pre + step * max_mean, pre**2 + 2 * pre * step * max_mean + step**2 * max_second
        mean, second = self.tokens.moments(decode=True)
        sum_mean = size * mean
        sum_second = size * (second - mean**2) + sum_mean**2
        # E[S*M] = b E[N_1 max(N_1,M_{b-1})]; N_1 is independent of M_{b-1}.
        if size == 1:
            sum_max = second
        else:
            other_p = self.tokens.maximum_pmf(size - 1)
            above = np.cumsum((other_p * n)[::-1])[::-1] - other_p * n
            conditional = n * np.cumsum(other_p) + above
            sum_max = size * float(p @ (n * conditional))
        duration_mean = pre + self.k3 * sum_mean + self.k4 * max_mean
        duration_second = (
            pre**2
            + 2 * pre * (self.k3 * sum_mean + self.k4 * max_mean)
            + self.k3**2 * sum_second
            + 2 * self.k3 * self.k4 * sum_max
            + self.k4**2 * max_second
        )
        return duration_mean, duration_second

    def job_mean(self, batch_size):
        """Completion offset for a random batch member, accounting for early release."""
        size = _positive_integer(batch_size, "batch_size")
        if not self.elastic:
            return self.moments(size)[0]
        n, p = self.tokens.lengths - 1, self.tokens.probabilities
        mean = float(p @ n)
        # E[min(N_1,N_2)] via the survival function on a possibly sparse support.
        jumps = np.diff(np.concatenate(([0.0], n)))
        survival = 1 - np.concatenate(([0.0], np.cumsum(p)[:-1]))
        pair_min = float(jumps @ survival**2)
        return self.k1 * size + self.k2 + self.k3 * (mean + (size - 1) * pair_min) + self.k4 * mean

    def counts(self, batch_size, rate):
        """Mixed-Poisson arrival counts via the conditional Laplace transform.

        FFT inversion of the count PGF is numerical, not a phase-type fit.
        A pathwise maximum duration bounds aliasing probability by 1e-15;
        max_fft_points limits work before allocating the transform.
        """
        size = _positive_integer(batch_size, "batch_size")
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError("rate must be finite and positive")
        (a, c), _ = self.envelope
        high = poisson(rate * (a * size + c)).isf(1e-15)
        if not np.isfinite(high) or high > self.max_fft_points - 2:
            raise MemoryError("arrival-count FFT exceeds max_fft_points")
        points = 1 << max(4, int(np.ceil(np.log2(high + 2))))
        if points > self.max_fft_points:
            raise MemoryError("arrival-count FFT exceeds max_fft_points")
        z = np.exp(2j * np.pi * np.arange(points) / points)
        t = rate * (z - 1)
        n, p = self.tokens.lengths - 1, self.tokens.probabilities
        pre = self.k1 * size + self.k2
        transform = np.zeros(points, dtype=complex)
        if self.elastic:
            partial = np.zeros(points, dtype=complex)
            previous = np.zeros(points, dtype=complex)
            for length, mass in zip(n, p):
                partial += mass * np.exp(t * self.k3 * length)
                power = partial**size
                transform += np.exp(t * self.k4 * length) * (power - previous)
                previous = power
        else:
            for length, mass in zip(n, self.tokens.maximum_pmf(size)):
                transform += mass * np.exp(t * (self.k3 * size + self.k4) * length)
        transform *= np.exp(t * pre)
        masses = np.fft.fft(transform).real / points
        if masses.min() < -1e-11:
            raise RuntimeError("count PGF inversion produced negative probabilities")
        masses = np.maximum(masses, 0)
        masses /= masses.sum()
        distribution = _FiniteCounts(masses)
        return [(1.0, distribution)]
