"""User-supplied conditional service laws for Inoue's all-waiting rule.

The callbacks supply E[H|b], E[H²|b] and the mixed-Poisson arrival-count
distribution. A global moment envelope is a caller-supplied certificate, not
something a finite-state solver can prove by sampling a few batch sizes.
Source of the embedded-chain/reward identities: Inoue (2021), arXiv:1912.06322.
"""

import numpy as np

from most_queue.theory.batch.dynamic_batching import DynamicBatchingCalc


class CallbackBatchingCalc(DynamicBatchingCalc):
    """General non-affine service with explicit sufficient finite-mean checks.

    moments(b) returns (E[H|b], E[H²|b]); counts(b, rate) returns a list of
    (weight, count_distribution). Distributions implement pmf/cdf/sf/ppf/isf,
    mean/var (SciPy's frozen count distributions work directly).
    job_mean(b), if supplied, is the mean completion offset of a random job
    in this batch, permitting early release but no intra-batch admission.
    """

    moments_callback = counts_callback = job_callback = None
    second_bound = None
    _components_cache = None
    _cached_count_entries = 0

    def set_servers(self, moments, counts, *, mean_bound, second_bound, job_mean=None):
        # pylint: disable=arguments-differ,too-many-arguments
        """Set callbacks and global bounds E[H|b]<=a*b+c, E[H²|b]<=u*b²+v*b+w.

        Bounds must hold for ALL positive integers b. lambda*a<1 and
        lambda²*u<1 suffice for a stationary finite waiting mean. These
        conservative conditions can reject a law with sharper unprovided
        bounds. Callback consistency is checked at every visited size.
        """
        if not callable(moments) or not callable(counts) or (job_mean is not None and not callable(job_mean)):
            raise ValueError("moments/counts/job_mean must be callable")
        mean = np.asarray(mean_bound, dtype=float)
        second = np.asarray(second_bound, dtype=float)
        if mean.shape != (2,) or second.shape != (3,):
            raise ValueError("mean_bound must have 2 entries, second_bound 3")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(second)) or min(*mean, *second) < 0:
            raise ValueError("moment envelope coefficients must be finite and non-negative")
        self.moments_callback, self.counts_callback, self.job_callback = moments, counts, job_mean
        self.alpha, self.tau0 = mean
        self.second_bound = second
        self.kind = "callback"
        self.is_servers_set = True
        self._cache = None
        self._components_cache = {}
        self._cached_count_entries = 0
        return self

    def set_sources(self, rate):
        """Set Poisson rate and invalidate conditional count caches."""
        super().set_sources(rate)
        self._components_cache = {}
        self._cached_count_entries = 0
        return self

    def _check_domain(self):
        self._check_ready()
        if self.kind != "callback":
            return super()._check_domain()
        if self.rate * self.alpha >= 1 or self.rate**2 * self.second_bound[0] >= 1:
            raise ValueError("finite-mean certificate requires lambda*a<1 and lambda²*u<1")
        return None

    def _batch_service_moments(self, sizes):
        if self.kind != "callback":
            return super()._batch_service_moments(sizes)
        pairs = np.asarray([self.moments_callback(int(size)) for size in sizes], dtype=float)
        if pairs.shape != (len(sizes), 2) or not np.all(np.isfinite(pairs)):
            raise ValueError("service moments must be finite pairs")
        means, seconds = pairs.T
        mean_cap = self.alpha * sizes + self.tau0
        u, v, w = self.second_bound
        if (
            np.any(means <= 0)
            or np.any(seconds < means**2 * (1 - 1e-12))
            or np.any(means > mean_cap * (1 + 1e-12))
            or np.any(seconds > (u * sizes**2 + v * sizes + w) * (1 + 1e-12))
        ):
            raise ValueError("invalid service moments or violated global moment envelope")
        return means, seconds

    def _batch_job_service_means(self, sizes, means):
        if self.kind != "callback" or self.job_callback is None:
            return means
        jobs = np.asarray([self.job_callback(int(size)) for size in sizes], dtype=float)
        if jobs.shape != means.shape or not np.all(np.isfinite(jobs)) or np.any(jobs < 0):
            raise ValueError("job completion means must be finite and non-negative")
        if np.any(jobs > means * (1 + 1e-12)):
            raise ValueError("job completion cannot exceed the batch completion mean")
        return jobs

    def _batch_count_components(self, size, mean):
        if self.kind != "callback":
            return super()._batch_count_components(size, mean)
        if size not in self._components_cache:
            parts = list(self.counts_callback(size, self.rate))
            weights = np.asarray([weight for weight, _ in parts], dtype=float)
            if not parts or not np.all(np.isfinite(weights)) or np.any(weights < 0) or abs(weights.sum() - 1) > 1e-12:
                raise ValueError("count-mixture weights must form a probability distribution")
            count_mean = sum(weight * dist.mean() for weight, dist in parts)
            count_second = sum(weight * (dist.var() + dist.mean() ** 2) for weight, dist in parts)
            service_second = self.moments_callback(size)[1]
            expected = np.array([self.rate * mean, self.rate * mean + self.rate**2 * service_second])
            if not np.allclose([count_mean, count_second], expected, rtol=1e-9, atol=1e-11):
                raise ValueError("count distribution is inconsistent with service moments")
            probes = np.arange(5)
            for _, dist in parts:
                masses = np.asarray(dist.pmf(probes))
                if not np.all(np.isfinite(masses)) or np.any(masses < 0) or np.any(masses > 1):
                    raise ValueError("invalid count PMF")
                if not np.allclose(dist.cdf(probes) + dist.sf(probes), 1, atol=1e-12):
                    raise ValueError("count CDF and survival must be complementary")
            entries = sum(np.size(getattr(dist, "masses", getattr(dist, "xk", []))) for _, dist in parts)
            if self._cached_count_entries + entries > 2 * self.max_nonzeros:
                raise MemoryError("cached finite count laws exceed the entry budget")
            self._cached_count_entries += entries
            self._components_cache[size] = parts
        return self._components_cache[size]
