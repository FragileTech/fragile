import os

from holoviews.streams import Pipe, Buffer
from streamz.dataframe import DataFrame
from streamz import Stream
import holoviews as hv
import hvplot.pandas
import hvplot.streamz
import numpy as np
import pandas as pd

from fragile.core.walkers import Walkers


class MetricWalkers(Walkers):
    def __init__(self, walkers: Walkers, plot_interval: int = 100, max_iters: int = 1500):
        self.walkers = walkers
        self.max_iters = max_iters
        self.vr_eff_hist = np.zeros(self.max_iters + 1)
        self.reward_hist = np.zeros(self.max_iters + 1)
        self.clone_pct_hist = np.zeros(self.max_iters + 1)
        self.clone_eff = 0
        self._max_ent = (2 - (1 / self.n) ** (1 / self.n)) ** self.n
        self._plot_iter = 0
        self.plot_interval = plot_interval
        self.stream = Stream()
        example = pd.DataFrame({"vr_eff": [], "reward": [], "clone_eff": []})
        self.buffer_df = DataFrame(stream=self.stream, example=example)

    def __getattr__(self, item):
        return getattr(self.walkers, item)

    @classmethod
    def from_walkers_class(
        cls, walkers, plot_interval: int = 100, max_iters: int = 1500, *args, **kwargs
    ):
        return MetricWalkers(
            walkers(*args, **kwargs), plot_interval=plot_interval, max_iters=max_iters
        )

    @property
    def df(self) -> pd.DataFrame:
        cols = ["vr_eff", "reward", "clone_pct"]
        df = pd.DataFrame(columns=cols)
        df["vr_eff"] = self.vr_eff_hist
        df["reward"] = self.reward_hist
        df["clone_pct"] = self.clone_pct_hist
        # df["n_Walkers"] = self.n
        return df

    @property
    def metrics(self):
        cols = [
            "n_walkers",
            "n_dims",
            "best_reward",
            "eff_mean",
            "eff_std",
            "clone_mean",
            "clone_std",
        ]
        df = self.df
        metrics = pd.DataFrame(columns=cols)
        metrics["n_walkers"] = [self.n]
        metrics["n_dims"] = [self.env_states.observs.shape[1]]
        metrics["best_reward"] = [df["reward"].min() if self.minimize else df["reward"].max()]
        metrics["eff_mean"] = [df["vr_eff"].mean()]
        metrics["eff_std"] = [df["vr_eff"].std()]
        metrics["clone_mean"] = [df["clone_pct"].mean()]
        metrics["clone_std"] = [df["clone_pct"].std()]
        return metrics

    def to_csv(self, name, save_values: bool = False):
        self_df = self.df if save_values else self.metrics
        try:
            df = pd.read_csv(name, index_col=0)
            if (df.columns.values == self_df.columns.values).all():
                df = pd.concat([df, self_df])
                df.to_csv(name)
            else:
                self_df.to_csv(name)
        except Exception as e:
            self_df.to_csv(name)

    def __repr__(self):
        msg = "Mean Efficiency: {:.4f}\n{}"
        msg = msg.format(
            np.mean(self.vr_eff_hist[: self.n_iters]), super(MetricWalkers, self).__repr__()
        )
        return msg

    def calculate_virtual_reward(self):
        super(MetricWalkers, self).calculate_virtual_reward()
        self.vr_eff_hist[self.n_iters - 1] = self.efficiency
        self.reward_hist[self.n_iters - 1] = float(self.states.best_reward_found)

    def plot_best_evolution(self):
        return self.buffer_df.hvplot(  # self.buffer_df.hvplot(y=["vr_eff", "clone_eff"]) +
            y=["reward"]
        )
        curve_dmap = hv.DynamicMap(hv.Curve, streams=[self.buffer_df])
        curve_dmap = curve_dmap.opts(
            tools=["hover"], title="Best value found", xlabel="iteration", ylabel="Best value"
        )
        return curve_dmap

    def update_clone_probs(self):
        super(MetricWalkers, self).update_clone_probs()
        self.clone_eff = 1 - self.states.will_clone.sum() / self.n
        self.clone_pct_hist[self.n_iters - 1] = self.clone_eff
        if self.n_iters % self.plot_interval == 0:
            df = pd.DataFrame(
                {
                    "vr_eff": [self.efficiency],
                    "reward": float(self.states.best_reward_found),
                    "clone_eff": [self.clone_eff],
                },
                index=[self.n_iters],
            )
            self.stream.emit(df)
