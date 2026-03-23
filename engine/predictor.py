# -*- coding: utf-8 -*-
"""
WatchDog – Predictive Analytics Engine.

Forecasts crowd trends using historical tracking data:
  • Short-term prediction  – next 5-15 minutes crowd count
  • Trend classification   – rising / stable / declining
  • Peak detection         – identify historical peak times
  • Zone occupancy forecast – predict when zones will exceed capacity
  • Anomaly probability    – likelihood of anomaly based on patterns

Uses simple statistical models (no external ML framework needed).
"""

import time
import math
import numpy as np
from collections import deque
from datetime import datetime


class CrowdPredictor:
    """Time-series crowd trend forecasting."""

    def __init__(self, history_size: int = 360):
        # Store (timestamp, count) tuples — ~30 min at 5s intervals
        self._history: deque = deque(maxlen=history_size)
        self._hourly_patterns: dict[int, list] = {}  # hour → [counts]
        self._daily_peaks: list[dict] = []

    def record(self, count: int, timestamp: float | None = None):
        """Record a crowd count observation."""
        if timestamp is None:
            timestamp = time.time()
        self._history.append((timestamp, count))

        # Track hourly patterns
        hour = datetime.fromtimestamp(timestamp).hour
        if hour not in self._hourly_patterns:
            self._hourly_patterns[hour] = []
        self._hourly_patterns[hour].append(count)
        # Keep last 100 per hour
        if len(self._hourly_patterns[hour]) > 100:
            self._hourly_patterns[hour] = self._hourly_patterns[hour][-100:]

    def get_trend(self) -> dict:
        """
        Analyze current crowd trend.

        Returns dict with:
          - trend: "rising" | "stable" | "declining"
          - rate: change per minute
          - current: current count
          - predicted_5min: forecast 5 min ahead
          - predicted_15min: forecast 15 min ahead
          - confidence: 0-1
          - peak_today: highest count today
          - avg_today: average count today
        """
        if len(self._history) < 5:
            return {
                "trend": "insufficient_data",
                "rate": 0, "current": 0,
                "predicted_5min": 0, "predicted_15min": 0,
                "confidence": 0, "peak_today": 0, "avg_today": 0,
                "history": [],
            }

        times = [h[0] for h in self._history]
        counts = [h[1] for h in self._history]
        current = counts[-1]

        # Linear regression on recent data (last 2 min)
        cutoff = time.time() - 120
        recent_t = [t for t in times if t >= cutoff]
        recent_c = [counts[i] for i, t in enumerate(times) if t >= cutoff]

        if len(recent_t) < 3:
            recent_t = times[-10:]
            recent_c = counts[-10:]

        slope, intercept = self._linear_fit(recent_t, recent_c)
        rate_per_min = slope * 60  # count change per minute

        # Trend classification
        if rate_per_min > 0.5:
            trend = "rising"
        elif rate_per_min < -0.5:
            trend = "declining"
        else:
            trend = "stable"

        # Predictions
        now = time.time()
        pred_5 = max(0, round(intercept + slope * (now + 300)))
        pred_15 = max(0, round(intercept + slope * (now + 900)))

        # Confidence based on data consistency
        if len(recent_c) > 3:
            std = np.std(recent_c)
            mean = max(1, np.mean(recent_c))
            cv = std / mean  # coefficient of variation
            confidence = max(0, min(1, 1 - cv))
        else:
            confidence = 0.3

        # Today's stats
        today_start = datetime.now().replace(hour=0, minute=0, second=0).timestamp()
        today_counts = [c for t, c in self._history if t >= today_start]
        peak_today = max(today_counts) if today_counts else 0
        avg_today = round(np.mean(today_counts), 1) if today_counts else 0

        # Compact history for chart (last 60 points)
        history_points = []
        for t, c in list(self._history)[-60:]:
            history_points.append({
                "t": datetime.fromtimestamp(t).strftime("%H:%M:%S"),
                "v": c,
            })

        return {
            "trend": trend,
            "rate": round(rate_per_min, 2),
            "current": current,
            "predicted_5min": pred_5,
            "predicted_15min": pred_15,
            "confidence": round(confidence, 2),
            "peak_today": peak_today,
            "avg_today": avg_today,
            "history": history_points,
        }

    def get_hourly_forecast(self) -> list[dict]:
        """Return expected crowd levels for each hour based on historical patterns."""
        forecast = []
        for hour in range(24):
            data = self._hourly_patterns.get(hour, [])
            if data:
                avg = round(np.mean(data), 1)
                peak = max(data)
                low = min(data)
            else:
                avg = peak = low = 0
            forecast.append({
                "hour": hour,
                "label": f"{hour:02d}:00",
                "avg": avg,
                "peak": peak,
                "low": low,
            })
        return forecast

    def get_risk_assessment(self) -> dict:
        """
        Assess crowd risk level based on current trends.

        Returns:
          - risk_level: "low" | "moderate" | "high" | "critical"
          - factors: list of contributing factors
          - score: 0-100
        """
        trend = self.get_trend()
        factors = []
        score = 0

        # Count-based risk
        current = trend["current"]
        if current >= 15:
            score += 40
            factors.append(f"Very high crowd count ({current})")
        elif current >= 10:
            score += 25
            factors.append(f"High crowd count ({current})")
        elif current >= 6:
            score += 10
            factors.append(f"Moderate crowd count ({current})")

        # Trend-based risk
        rate = trend["rate"]
        if rate > 2:
            score += 30
            factors.append(f"Rapidly rising ({rate:+.1f}/min)")
        elif rate > 0.5:
            score += 15
            factors.append(f"Rising trend ({rate:+.1f}/min)")

        # Prediction-based risk
        if trend["predicted_5min"] > current * 1.5 and current > 3:
            score += 20
            factors.append(f"Predicted surge to {trend['predicted_5min']} in 5min")

        # Historical pattern risk
        current_hour = datetime.now().hour
        hourly_data = self._hourly_patterns.get(current_hour, [])
        if hourly_data and current > np.percentile(hourly_data, 90):
            score += 10
            factors.append("Above 90th percentile for this hour")

        score = min(100, score)

        if score >= 70:
            level = "critical"
        elif score >= 45:
            level = "high"
        elif score >= 20:
            level = "moderate"
        else:
            level = "low"

        return {
            "risk_level": level,
            "score": score,
            "factors": factors,
        }

    @staticmethod
    def _linear_fit(x_vals, y_vals):
        """Simple linear regression. Returns (slope, intercept)."""
        n = len(x_vals)
        if n < 2:
            return 0, y_vals[-1] if y_vals else 0

        x = np.array(x_vals, dtype=float)
        y = np.array(y_vals, dtype=float)

        x_mean = np.mean(x)
        y_mean = np.mean(y)

        num = np.sum((x - x_mean) * (y - y_mean))
        den = np.sum((x - x_mean) ** 2)

        if den == 0:
            return 0, y_mean

        slope = num / den
        intercept = y_mean - slope * x_mean
        return slope, intercept
