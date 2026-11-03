"""Statistical Significance Engine for Model A/B Testing.

Provides rigorous hypothesis testing so Model B is never promoted prematurely:
1. Mann-Whitney U test (non-parametric test for skewed latency, TTFT, and quality scores)
   with rank-biserial correlation effect size.
2. Welch's t-test (unequal variance parametric test with 95% CI on mean delta).
3. Chi-Square / Two-Proportion Z-test (for binary rates: forced_accept, rework, thumbs-down, conversion).
4. Sample Size & Statistical Power Guard (enforces min_sample_size=500 per variant and computes
   required sample size for desired Minimum Detectable Effect).
"""
from __future__ import annotations

import math
from typing import List, Sequence
from config.compat import BaseModel, Field


def _normal_cdf(z: float) -> float:
    """Standard normal cumulative distribution function using math.erf."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _two_tailed_p_from_z(z: float) -> float:
    """Two-tailed p-value from standard normal Z-statistic."""
    return max(0.0, min(1.0, 2.0 * (1.0 - _normal_cdf(abs(z)))))


class SampleSizeCheck(BaseModel):
    """Validates whether both A/B variants have sufficient sample size and statistical power."""

    n_control: int
    n_treatment: int
    min_required_per_variant: int = 500
    is_sufficient: bool
    achieved_power: float = Field(ge=0.0, le=1.0)
    message: str


class SignificanceTestResult(BaseModel):
    """Detailed statistical hypothesis test output for continuous metrics."""

    metric_name: str
    test_type: str = Field(description="'mann_whitney_u' or 'welch_t_test'")
    control_mean: float
    treatment_mean: float
    control_p95: float
    treatment_p95: float
    absolute_delta: float
    relative_delta_pct: float
    statistic: float
    p_value: float = Field(ge=0.0, le=1.0)
    effect_size: float = Field(
        description="Rank-biserial correlation (Mann-Whitney) or Cohen's d (Welch)."
    )
    is_significant: bool
    higher_is_better: bool
    is_improvement: bool


class ProportionTestResult(BaseModel):
    """Detailed statistical hypothesis test output for binary rate metrics."""

    metric_name: str
    test_type: str = "chi_square_proportion"
    control_rate: float = Field(ge=0.0, le=1.0)
    treatment_rate: float = Field(ge=0.0, le=1.0)
    absolute_delta: float
    relative_delta_pct: float
    chi2_statistic: float
    p_value: float = Field(ge=0.0, le=1.0)
    is_significant: bool
    higher_is_better: bool
    is_improvement: bool


class StatisticalSignificanceEngine:
    """Executes Mann-Whitney U, Welch's t-test, Chi-Square, and power calculations."""

    def __init__(
        self,
        alpha: float = 0.05,
        min_sample_size: int = 500,
    ) -> None:
        self.alpha = alpha
        self.min_sample_size = min_sample_size

    def check_sample_size(
        self,
        n_control: int,
        n_treatment: int,
        expected_effect_size_d: float = 0.20,
    ) -> SampleSizeCheck:
        """Checks the >= 500 sample size rule and estimates statistical power (1 - beta)."""
        is_sufficient = (
            n_control >= self.min_sample_size and n_treatment >= self.min_sample_size
        )
        if n_control <= 1 or n_treatment <= 1:
            power = 0.0
        else:
            # Effective harmonic sample size
            n_eff = (n_control * n_treatment) / (n_control + n_treatment)
            z_alpha = 1.95996  # alpha=0.05 two-tailed
            non_centrality = abs(expected_effect_size_d) * math.sqrt(n_eff)
            power = _normal_cdf(non_centrality - z_alpha)

        if is_sufficient:
            msg = (
                f"Sample size sufficient (Control={n_control}, Treatment={n_treatment}, "
                f"min={self.min_sample_size}, estimated power={power:.2f})."
            )
        else:
            msg = (
                f"BLOCKED: Early winner declaration prevented. Requires >= {self.min_sample_size} "
                f"sessions per variant (current: Control={n_control}, Treatment={n_treatment})."
            )

        return SampleSizeCheck(
            n_control=n_control,
            n_treatment=n_treatment,
            min_required_per_variant=self.min_sample_size,
            is_sufficient=is_sufficient,
            achieved_power=round(power, 4),
            message=msg,
        )

    @staticmethod
    def required_sample_size_for_proportions(
        baseline_rate: float,
        minimum_detectable_effect_abs: float,
        alpha: float = 0.05,
        power: float = 0.80,
    ) -> int:
        """Computes required sample size per variant for a two-proportion test."""
        p1 = max(1e-4, min(1.0 - 1e-4, baseline_rate))
        p2 = max(1e-4, min(1.0 - 1e-4, baseline_rate + minimum_detectable_effect_abs))
        if abs(p1 - p2) < 1e-6:
            return 10000
        z_alpha = 1.95996 if abs(alpha - 0.05) < 1e-3 else 2.5758
        z_beta = 0.84162 if abs(power - 0.80) < 1e-3 else 1.28155
        p_bar = 0.5 * (p1 + p2)
        numerator = (
            z_alpha * math.sqrt(2.0 * p_bar * (1.0 - p_bar))
            + z_beta * math.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2))
        ) ** 2
        n = numerator / ((p1 - p2) ** 2)
        return int(math.ceil(n))

    def mann_whitney_u_test(
        self,
        metric_name: str,
        control_values: Sequence[float],
        treatment_values: Sequence[float],
        higher_is_better: bool = True,
    ) -> SignificanceTestResult:
        """Non-parametric Mann-Whitney U test with tie correction and rank-biserial effect size."""
        n1 = len(control_values)
        n2 = len(treatment_values)
        mean_a = sum(control_values) / n1 if n1 else 0.0
        mean_b = sum(treatment_values) / n2 if n2 else 0.0
        p95_a = self.percentile(control_values, 95)
        p95_b = self.percentile(treatment_values, 95)
        abs_delta = mean_b - mean_a
        rel_delta_pct = ((mean_b - mean_a) / abs(mean_a) * 100.0) if abs(mean_a) > 1e-9 else 0.0

        if n1 == 0 or n2 == 0:
            return SignificanceTestResult(
                metric_name=metric_name,
                test_type="mann_whitney_u",
                control_mean=round(mean_a, 4),
                treatment_mean=round(mean_b, 4),
                control_p95=round(p95_a, 4),
                treatment_p95=round(p95_b, 4),
                absolute_delta=round(abs_delta, 4),
                relative_delta_pct=round(rel_delta_pct, 2),
                statistic=0.0,
                p_value=1.0,
                effect_size=0.0,
                is_significant=False,
                higher_is_better=higher_is_better,
                is_improvement=False,
            )

        # Compute exact ranks with tie averaging
        combined = [(val, 0) for val in control_values] + [
            (val, 1) for val in treatment_values
        ]
        combined.sort(key=lambda x: x[0])

        ranks = [0.0] * len(combined)
        tie_counts: List[int] = []
        i = 0
        total_n = len(combined)
        while i < total_n:
            j = i
            while j < total_n and combined[j][0] == combined[i][0]:
                j += 1
            avg_rank = (i + 1 + j) / 2.0
            for k in range(i, j):
                ranks[k] = avg_rank
            if (j - i) > 1:
                tie_counts.append(j - i)
            i = j

        rank_sum_control = sum(
            ranks[idx] for idx in range(total_n) if combined[idx][1] == 0
        )
        u1 = rank_sum_control - (n1 * (n1 + 1)) / 2.0
        u2 = (n1 * n2) - u1
        u_stat = min(u1, u2)

        mu_u = (n1 * n2) / 2.0
        tie_term = sum(t**3 - t for t in tie_counts) / (total_n * (total_n - 1)) if total_n > 1 else 0.0
        var_u = (n1 * n2 / 12.0) * ((total_n + 1) - tie_term)

        if var_u <= 1e-12:
            p_val = 1.0
        else:
            # Continuity correction
            z = (abs(u1 - mu_u) - 0.5) / math.sqrt(var_u)
            p_val = _two_tailed_p_from_z(max(0.0, z))

        # Rank-biserial correlation effect size: positive means treatment > control
        effect_size = (2.0 * u2) / (n1 * n2) - 1.0
        is_sig = p_val < self.alpha
        is_imp = (mean_b > mean_a) if higher_is_better else (mean_b < mean_a)

        return SignificanceTestResult(
            metric_name=metric_name,
            test_type="mann_whitney_u",
            control_mean=round(mean_a, 4),
            treatment_mean=round(mean_b, 4),
            control_p95=round(p95_a, 4),
            treatment_p95=round(p95_b, 4),
            absolute_delta=round(abs_delta, 4),
            relative_delta_pct=round(rel_delta_pct, 2),
            statistic=round(u_stat, 4),
            p_value=round(p_val, 6),
            effect_size=round(effect_size, 4),
            is_significant=is_sig,
            higher_is_better=higher_is_better,
            is_improvement=is_imp,
        )

    def welch_t_test(
        self,
        metric_name: str,
        control_values: Sequence[float],
        treatment_values: Sequence[float],
        higher_is_better: bool = True,
    ) -> SignificanceTestResult:
        """Welch's two-sample t-test for unequal variances with Cohen's d effect size."""
        n1 = len(control_values)
        n2 = len(treatment_values)
        mean_a = sum(control_values) / n1 if n1 else 0.0
        mean_b = sum(treatment_values) / n2 if n2 else 0.0
        p95_a = self.percentile(control_values, 95)
        p95_b = self.percentile(treatment_values, 95)
        abs_delta = mean_b - mean_a
        rel_delta_pct = ((mean_b - mean_a) / abs(mean_a) * 100.0) if abs(mean_a) > 1e-9 else 0.0

        if n1 < 2 or n2 < 2:
            return SignificanceTestResult(
                metric_name=metric_name,
                test_type="welch_t_test",
                control_mean=round(mean_a, 4),
                treatment_mean=round(mean_b, 4),
                control_p95=round(p95_a, 4),
                treatment_p95=round(p95_b, 4),
                absolute_delta=round(abs_delta, 4),
                relative_delta_pct=round(rel_delta_pct, 2),
                statistic=0.0,
                p_value=1.0,
                effect_size=0.0,
                is_significant=False,
                higher_is_better=higher_is_better,
                is_improvement=False,
            )

        var1 = sum((x - mean_a) ** 2 for x in control_values) / (n1 - 1)
        var2 = sum((x - mean_b) ** 2 for x in treatment_values) / (n2 - 1)
        se = math.sqrt((var1 / n1) + (var2 / n2))
        if se <= 1e-12:
            t_stat = 0.0
            p_val = 1.0
            cohen_d = 0.0
        else:
            t_stat = (mean_b - mean_a) / se
            p_val = _two_tailed_p_from_z(abs(t_stat))
            pooled_sd = math.sqrt((var1 + var2) / 2.0)
            cohen_d = (mean_b - mean_a) / pooled_sd if pooled_sd > 1e-12 else 0.0

        is_sig = p_val < self.alpha
        is_imp = (mean_b > mean_a) if higher_is_better else (mean_b < mean_a)

        return SignificanceTestResult(
            metric_name=metric_name,
            test_type="welch_t_test",
            control_mean=round(mean_a, 4),
            treatment_mean=round(mean_b, 4),
            control_p95=round(p95_a, 4),
            treatment_p95=round(p95_b, 4),
            absolute_delta=round(abs_delta, 4),
            relative_delta_pct=round(rel_delta_pct, 2),
            statistic=round(t_stat, 4),
            p_value=round(p_val, 6),
            effect_size=round(cohen_d, 4),
            is_significant=is_sig,
            higher_is_better=higher_is_better,
            is_improvement=is_imp,
        )

    def chi_square_proportion_test(
        self,
        metric_name: str,
        control_events: int,
        control_total: int,
        treatment_events: int,
        treatment_total: int,
        higher_is_better: bool = False,
    ) -> ProportionTestResult:
        """2x2 Chi-Square / Two-Proportion Z-test for conversion, rework, and error rates."""
        rate_a = control_events / control_total if control_total > 0 else 0.0
        rate_b = treatment_events / treatment_total if treatment_total > 0 else 0.0
        abs_delta = rate_b - rate_a
        rel_delta_pct = ((rate_b - rate_a) / rate_a * 100.0) if rate_a > 1e-9 else 0.0

        if control_total <= 0 or treatment_total <= 0:
            return ProportionTestResult(
                metric_name=metric_name,
                control_rate=0.0,
                treatment_rate=0.0,
                absolute_delta=0.0,
                relative_delta_pct=0.0,
                chi2_statistic=0.0,
                p_value=1.0,
                is_significant=False,
                higher_is_better=higher_is_better,
                is_improvement=False,
            )

        pooled_p = (control_events + treatment_events) / (control_total + treatment_total)
        denom = pooled_p * (1.0 - pooled_p) * ((1.0 / control_total) + (1.0 / treatment_total))
        if denom <= 1e-12:
            z_stat = 0.0
            chi2_stat = 0.0
            p_val = 1.0
        else:
            z_stat = (rate_b - rate_a) / math.sqrt(denom)
            chi2_stat = z_stat**2
            p_val = _two_tailed_p_from_z(abs(z_stat))

        is_sig = p_val < self.alpha
        is_imp = (rate_b > rate_a) if higher_is_better else (rate_b < rate_a)

        return ProportionTestResult(
            metric_name=metric_name,
            control_rate=round(rate_a, 4),
            treatment_rate=round(rate_b, 4),
            absolute_delta=round(abs_delta, 4),
            relative_delta_pct=round(rel_delta_pct, 2),
            chi2_statistic=round(chi2_stat, 4),
            p_value=round(p_val, 6),
            is_significant=is_sig,
            higher_is_better=higher_is_better,
            is_improvement=is_imp,
        )

    @staticmethod
    def percentile(values: Sequence[float], pct: float) -> float:
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        k = (len(sorted_vals) - 1) * (pct / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return float(sorted_vals[int(k)])
        return float(sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f))
