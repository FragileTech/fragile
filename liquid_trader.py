import param
import panel as pn
import pandas as pd
import numpy as np
from typing import Dict, Any
from hydra.utils import get_class

from circuit_rl.config_ui.detector import Detector
from circuit_rl.config_ui.lsm_config import LiquidStateMachineConfig
from circuit_rl.liquid_trader import LiquidStateMachine

from circuit_rl.config_ui.core import (
    DashboardConfig,
    INPUT_WIDTH,
    SLIDER_WIDTH,
)


class TradeMomentConfig(DashboardConfig):
    """Configuration for trade moment analysis parameters"""

    analysis_window_ms = param.Number(
        default=10.0,
        doc="TradeMomentConfig: Analysis window size in milliseconds",
    )

    analysis_window_shift_ms = param.Number(
        default=10.0,
        doc="TradeMomentConfig: Shift between analysis windows in milliseconds",
    )

    neuron_group = param.String(
        default="excitatory",
        doc="TradeMomentConfig: Neuron group to analyze for trade moments",
    )
    minimal_trade_moment_distance = param.String(
        default=None,
        doc="Minimum distance between trade moments",
    )
    detector = param.ClassSelector(class_=Detector, default=None)
    detector_type = param.String("maximal", doc="Type of detector")
    _target_ = param.String(
        default="circuit_rl.liquid_trader.TradeMomentConfig"
    )
    _n_widget_columns = 5

    def __init__(self, detector=None, **params):
        if detector is None:
            detector = Detector()
        super().__init__(detector=detector, **params)

    @property
    def widgets(self):
        return {
            "analysis_window_ms": {
                "type": pn.widgets.FloatInput,
                "width": int(INPUT_WIDTH * 1.5),
                "format": "0.00",
            },
            "analysis_window_shift_ms": {
                "type": pn.widgets.FloatInput,
                "width": int(INPUT_WIDTH * 1.5),
                "format": "0.00",
            },
            "detector_type": {
                "type": pn.widgets.Select,
                "options": ["maximal", "random"],
                "width": int(INPUT_WIDTH * 1.3),
            },
            # "detector": {
            #     "type": pn.widgets.TextAreaInput,
            #     "placeholder": self.detector,
            #     "height": 80,
            #     "width": int(INPUT_WIDTH * 4),
            # },
            "neuron_group": {
                "type": pn.widgets.TextInput,
                # "options": ["excitatory", "inhibitory"],
                "width": int(INPUT_WIDTH * 1.3),
            },
            "minimal_trade_moment_distance": {
                "type": pn.widgets.FloatInput,
                "width": int(INPUT_WIDTH * 1.3),
                "placeholder": "None",
                "name": "Min moment distance",
            },
        }

    def to_dict(self, parameters=None) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization"""
        return {
            "_target_": self._target_,
            "analysis_window_ms": self.analysis_window_ms,
            "analysis_window_shift_ms": self.analysis_window_shift_ms,
            "detector_type": self.detector_type,
            # "detector": {
            #     "_target_": "builtins.eval",
            #     "_args_": [self.detector],
            # },
            "detector": self.detector.to_dict(),
            "neuron_group": self.neuron_group,
            "minimal_trade_moment_distance": self.minimal_trade_moment_distance,
        }

    def set_values(self, d: Dict[str, Any], parameters=None) -> None:
        """Set values from a dictionary"""
        if parameters is None:
            parameters = [
                "analysis_window_ms",
                "analysis_window_shift_ms",
                "neuron_group",
                "minimal_trade_moment_distance",
                "detector_type",
            ]
        super().set_values(d, parameters=parameters)
        if "detector" in d:
            self.detector.set_values(d["detector"])
        # detector = self.parse_lambda_function("detector", d)
        # if detector is not None:
        #     self.detector = detector

    @property
    def widget_parameters(self):
        return [
            "analysis_window_ms",
            "analysis_window_shift_ms",
            "neuron_group",
            "minimal_trade_moment_distance",
            "detector_type",
        ]

    def __panel__(self, parameters=None):
        params = super().__panel__(parameters=parameters)
        # detector = pn.Param(
        #     self,
        #     parameters=["detector", "detector_type"],
        #     show_name=False,
        #     widgets=self.widgets,
        #     default_layout=pn.Column,
        # )
        return params


class PriceNewsTicks(DashboardConfig):
    """Configuration for PriceNewsTicks parameters"""

    _target_ = "circuit_rl.liquid_trader.input.PriceNewsTicks.from_file"
    file = param.Filename(
        default="./spike_data.parquet",
        doc="Path to the spike data file",
        check_exists=False,
    )
    input_freq_hz = param.Number(
        default=3.0,
        doc="Input frequency in Hz",
        bounds=(0.001, None),
        step=0.01,
    )
    max_symbols = param.Integer(
        default=None,
        bounds=(1, None),
        step=1,
        doc="Maximum number of symbols to include in the data",
    )
    mode = param.String(
        default="direct",
        doc="Mode of the spike train generation (`direct`, `fuzzy_news` or `fuzzy_all`)",
    )
    min_frequency = param.Number(
        default=0.001,
        bounds=(0.001, None),
        allow_None=True,
        step=0.01,
        doc="Minimum frequency of the input neurons (Hz)",
    )
    max_frequency = param.Number(
        default=4,
        allow_None=True,
        bounds=(0.001, None),
        step=0.01,
        doc="Maximum frequency of the input neurons (Hz)",
    )
    # Replace decay_rate_news with proportion and duration
    decay_rate_news_proportion = param.Number(
        default=0.1,
        step=0.1,
        bounds=(0.01, 1),
        doc="Proportion remaining after time period for news memory",
    )
    decay_rate_news_duration = param.Number(
        default=300,
        step=1.0,
        bounds=(0.01, None),
        doc="Duration in seconds for news memory decay",
    )
    # Replace decay_rate_price with proportion and duration
    decay_rate_price_proportion = param.Number(
        default=0.1,
        bounds=(0.01, 1),
        step=0.1,
        doc="Proportion remaining after time period for price memory",
    )
    decay_rate_price_duration = param.Number(
        default=1,
        step=1.0,
        bounds=(0.01, None),
        doc="Duration in seconds for price memory decay",
    )
    _n_widget_columns = 5

    # decay_rate_flow_proportion = param.Number(
    #     default=0.1,
    #     bounds=(0.01, 1),
    #     step=0.1,
    #     doc="Proportion remaining after time period for flow memory",
    # )
    # decay_rate_flow_duration = param.Number(
    #     default=30,
    #     step=1.0,
    #     bounds=(0.01, None),
    #     doc="Duration in seconds for flow memory decay",
    # )

    def __init__(self, **params):
        super().__init__(**params)
        self._param_panel = None
        self.price_decay = None
        # Create dedicated toggle for mode selection
        self._mode_toggle = pn.widgets.RadioButtonGroup(
            name="Mode",
            options={
                "Direct": "direct",
                "Fuzzy News": "fuzzy_news",
                "Fuzzy All": "fuzzy_all",
            },
            value=self.mode,
            button_type="primary",
        )
        self._mode_toggle_col = pn.Row(
            pn.pane.Markdown("#### Mode:"), self._mode_toggle
        )
        self._mode_toggle.param.watch(self._update_mode, "value")

    @staticmethod
    def compute_decay_rate(duration, proportion):
        """
        Compute the decay rate based on duration and proportion.
        The formula is derived from the exponential decay function.
        """
        return -np.log(proportion) / duration

    def _update_mode(self, event=None):
        if event is not None:
            self.mode = event.new
        if self.mode == "direct":
            self.price_decay.visible = False
            self.news_decay.visible = False
        elif self.mode == "fuzzy_news":
            self.price_decay.visible = False
            self.news_decay.visible = True
        elif self.mode == "fuzzy_all":
            self.price_decay.visible = True
            self.news_decay.visible = True

    @property
    def dict_param_names(self) -> list[str]:
        return [
            "_target_",
            "file",
            "input_freq_hz",
            "max_symbols",
            "mode",
            "min_frequency",
            "max_frequency",
        ]

    def to_dict(self, parameters=None) -> Dict[str, Any]:
        if parameters is None:
            parameters = [
                x
                for x in self.dict_param_names
                if x not in {"min_frequency", "max_frequency"}
            ]
        data = super().to_dict(parameters=parameters)
        if self.mode in ["fuzzy_news", "fuzzy_all"]:
            fuzzy_data = {
                "min_frequency": self.min_frequency,
                "max_frequency": self.max_frequency,
                "news_decay_rate": {
                    "_target_": "circuit_rl.config_utils.compute_decay_rate",
                    "proportion": self.decay_rate_news_proportion,
                    "duration": self.decay_rate_news_duration,
                },
            }
        else:
            fuzzy_data = {
                "min_frequency": self.min_frequency,
                "max_frequency": self.max_frequency,
                "news_decay_rate": None,
            }
        data = {**data, **fuzzy_data}

        if self.mode == "fuzzy_all":
            data["price_decay_rate"] = {
                "_target_": "circuit_rl.config_utils.compute_decay_rate",
                "proportion": self.decay_rate_price_proportion,
                "duration": self.decay_rate_price_duration,
            }
        else:
            data["price_decay_rate"] = None
        return data

    def set_values(self, d, parameters=None) -> None:
        """Set values from a dictionary."""
        super().set_values(d, parameters=parameters)
        # Update toggle buttons to match if they exist
        if "mode" in d and hasattr(self, "_mode_toggle"):
            self._mode_toggle.value = d["mode"]

        if "news_decay_rate" in d and d["news_decay_rate"] is not None:
            self.decay_rate_news_proportion = d["news_decay_rate"].get(
                "proportion", self.decay_rate_news_proportion
            )
            self.decay_rate_news_duration = d["news_decay_rate"].get(
                "duration", self.decay_rate_news_duration
            )
        if "price_decay_rate" in d and d["price_decay_rate"] is not None:
            self.decay_rate_price_proportion = d["price_decay_rate"].get(
                "proportion", self.decay_rate_price_proportion
            )
            self.decay_rate_price_duration = d["price_decay_rate"].get(
                "duration", self.decay_rate_price_duration
            )
        # if "decay_rate_flow" in d:
        #     self.decay_rate_flow_proportion = d["decay_rate_flow"].get(
        #         "proportion", self.decay_rate_flow_proportion
        #     )
        #     self.decay_rate_flow_duration = d["decay_rate_flow"].get(
        #         "duration", self.decay_rate_flow_duration
        #     )

    @property
    def widgets(self):
        input_width = int(INPUT_WIDTH * 1.15)
        return {
            "max_symbols": {
                "type": pn.widgets.IntInput,
                "width": input_width,
            },
            "input_freq_hz": {
                "type": pn.widgets.FloatInput,
                "width": input_width,
                "start": 0.001,
                "step": 0.01,
                "name": "Input frequency (Hz)",
                "format": "0.00",
            },
            "min_frequency": {
                "type": pn.widgets.FloatInput,
                "start": 0.001,
                "step": 0.01,
                "format": "0.00",
                "name": "Min frequency (Hz)",
                "width": input_width,
            },
            "max_frequency": {
                "type": pn.widgets.FloatInput,
                "start": 0.001,
                "step": 0.01,
                "format": "0.00",
                "name": "Max frequency (Hz)",
                "width": input_width,
            },
            "decay_rate_news_proportion": {
                "type": pn.widgets.EditableFloatSlider,
                "start": 0.01,
                "end": 1.0,
                "step": 0.1,
                "name": "Proportion remaining",
                "width": SLIDER_WIDTH,
            },
            "decay_rate_news_duration": {
                "type": pn.widgets.FloatInput,
                "start": 0.01,
                "step": 1.0,
                "format": "0.",
                "name": "Duration (seconds)",
                "width": INPUT_WIDTH,
            },
            "decay_rate_price_proportion": {
                "type": pn.widgets.EditableFloatSlider,
                "start": 0.01,
                "end": 1.0,
                "step": 0.1,
                "format": "0.00",
                "name": "Proportion remaining",
                "width": SLIDER_WIDTH,
            },
            "decay_rate_price_duration": {
                "type": pn.widgets.FloatInput,
                "start": 0.01,
                "step": 1.0,
                "format": "0.",
                "name": "Duration (seconds)",
                "width": INPUT_WIDTH,
            },
            # "decay_rate_flow_proportion": {
            #     "type": pn.widgets.EditableFloatSlider,
            #     "start": 0.01,
            #     "end": 1.0,
            #     "step": 0.01,
            #     "width": SLIDER_WIDTH,
            # },
            # "decay_rate_flow_duration": {
            #     "type": pn.widgets.FloatInput,
            #     "width": INPUT_WIDTH,
            # },
        }

    def __panel__(self, parameters=None):
        """Render the fuzzy spike train parameters in a Panel layout."""
        # Create frequency parameters section
        ticks_widgets = super().__panel__(
            parameters=[
                "input_freq_hz",
                "max_symbols",
                "file",
                "min_frequency",
                "max_frequency",
            ]
        )

        # Create decay rate sections with headers and parameter rows
        self.news_decay = pn.Row(
            pn.Column(
                pn.pane.Markdown("### News Memory Decay"),
                super().__panel__(
                    parameters=[
                        "decay_rate_news_duration",
                        "decay_rate_news_proportion",
                    ]
                ),
            ),
            pn.Column(width=10),
        )
        self.price_decay = pn.Column(
            pn.pane.Markdown("### Price Memory Decay"),
            super().__panel__(
                parameters=[
                    "decay_rate_price_duration",
                    "decay_rate_price_proportion",
                ]
            ),
        )
        # self.flow_decay = pn.Column(
        #     pn.pane.Markdown("### Flow Memory Decay"),
        #     super().__panel__(
        #         parameters=[
        #             "decay_rate_flow_duration",
        #             "decay_rate_flow_proportion",
        #         ]
        #     ),
        # )

        param_panel = pn.Row(
            self._mode_toggle_col,
            self.news_decay,
            self.price_decay,
            # self.flow_decay,
        )
        self._param_panel = param_panel
        self._update_mode()
        return pn.Column(ticks_widgets, param_panel)


class CausalityFeatures(DashboardConfig):
    window_ms = param.Number(
        default=100.0,
        doc="CausalityFeatures: Window size in milliseconds for causality features",
        bounds=(0.1, None),
        step=0.1,
    )
    _target_ = "circuit_rl.liquid_trader.readout.features.CausalityFeatures"

    @property
    def widgets(self):
        return {
            "window_ms": {
                "type": pn.widgets.FloatInput,
                "width": INPUT_WIDTH,
                "format": "0.00",
                "step": 0.1,
                "start": 0.1,
            },
        }

    def __panel__(self, parameters=None):
        params = super().__panel__(parameters)
        return pn.Column(
            pn.pane.Markdown("### Causality Features"),
            params,
        )


class VoltageFeatures(DashboardConfig):
    neuron_group = param.String(
        default="excitatory",
        doc="VoltageFeatures: Neuron group to analyze for voltage features",
    )
    _target_ = "circuit_rl.liquid_trader.readout.features.VoltageFeatures"

    @property
    def widgets(self):
        return {
            "neuron_group": {
                "type": pn.widgets.TextInput,
                # "options": ["excitatory", "inhibitory"],
                "width": int(INPUT_WIDTH * 1.3),
            },
        }

    def __panel__(self, parameters=None):
        params = super().__panel__(parameters)
        return pn.Column(
            pn.pane.Markdown("### Voltage Features"),
            params,
        )


class LiquidTrader(DashboardConfig):
    circuit = param.ClassSelector(class_=LiquidStateMachine)
    input = param.ClassSelector(class_=PriceNewsTicks)
    trade_moments = param.ClassSelector(class_=TradeMomentConfig)
    _target_ = "circuit_rl.liquid_trader.LiquidTrader"

    def __init__(self, circuit=None, input=None, trade_moments=None, **params):
        if circuit is None:
            circuit = LiquidStateMachine()
        if input is None:
            input = PriceNewsTicks()
        if trade_moments is None:
            trade_moments = TradeMomentConfig()
        super().__init__(
            circuit=circuit, input=input, trade_moments=trade_moments, **params
        )

    def instantiate(self, **kwargs):
        """Instantiate the LiquidTrader with the provided parameters"""
        input_instance = self.input.instantiate(**kwargs)
        kwargs["num_inputs"] = input_instance.num_inputs
        circuit_instance = self.circuit.instantiate(**kwargs)
        trade_moments_instance = self.trade_moments.instantiate(**kwargs)
        return get_class(self._target_)(
            circuit=circuit_instance,
            input=input_instance,
            trade_moments=trade_moments_instance,
        )

    def create_inputs_and_moments_panel(self):
        return pn.Row(
            pn.pane.Markdown("### Input params"),
            self.input.__panel__(),
            pn.pane.Markdown("### Trade moments params"),
            self.trade_moments.__panel__(),
        )

    def create_recurrent_col_panel(self):
        return pn.Column(
            pn.pane.Markdown("### Recurrent Column params"),
            self.circuit.config.recurrent_column_ui(),
        )

    def create_input_config_panel(self):
        return pn.Column(
            pn.pane.Markdown("### Input Configuration"),
            self.circuit.config.input_config.__panel__(),
        )


class SimplexFeatures(DashboardConfig):
    # trader = param.ClassSelector(class_=LiquidTrader, default=None)
    window_ms = param.Number(
        default=100.0,
        doc="SimplexFeatures: Window size in milliseconds for simplex features",
        bounds=(0.1, None),
        step=0.1,
    )
    # selection = param.ClassSelector(
    #     class_=RandomSimplexSelection,
    #     default=None,
    #     doc="SimplexFeatures: Selection of simplex features",
    # )

    # neuron_group = param.String(
    #     default="excitatory",
    #     doc="SimplexFeatures: Neuron group to analyze for simplex features",
    # )
    percent_per_dimension: str = param.List(
        default=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0],
        item_type=float,
        doc="SimplexFeatures: Probabilities to select a simplex per dimension as a list of floats",
    )
    # analog = param.Boolean(
    #     default=False,
    #     doc="SimplexFeatures: Use analog features",
    # )
    _target_ = "circuit_rl.liquid_trader.readout.features.SimplexFeatures"
    _n_widget_columns = 3

    # def __init__(self, selection=None, trader=None, **kwargs):
    #     if selection is None:
    #         selection = RandomSimplexSelection()
    #     if trader is None:
    #         trader = LiquidTrader()
    #     super().__init__(selection=selection, trader=trader, **kwargs)

    @property
    def widgets(self):
        return {
            "window_ms": {
                "type": pn.widgets.FloatInput,
                "width": INPUT_WIDTH,
                "format": "0.00",
                "step": 0.1,
                "start": 0.1,
            },
            "analog": {
                "type": pn.widgets.Toggle,
                "width": INPUT_WIDTH,
            },
            # "neuron_group": {
            #     "type": pn.widgets.TextInput,
            #     # "options": ["excitatory", "inhibitory"],
            #     "width": int(INPUT_WIDTH * 1.3),
            # },
            # "percent_per_dimension": {
            #     "type": pn.widgets.LiteralInput,
            #     "width": 500,
            # },
        }

    @property
    def widget_parameters(self):
        return ["window_ms", "percent_per_dimension"]

    def __panel__(self, parameters=None):
        if parameters is None:
            parameters = self.widget_parameters
        params = super().__panel__(parameters)
        return pn.Column(
            pn.pane.Markdown("### Simplex Features Configuration"),
            params,
        )


class FeaturesConfig(DashboardConfig):
    features = param.List(
        default=None,
        item_type=DashboardConfig,
        doc="List of features to extract from the liquid trader",
    )
    _target_ = "circuit_rl.liquid_trader.readout.features.FeaturesConfig"

    def __init__(self, features=None, **params):
        self.simplex_features = SimplexFeatures()
        self.causality_features = CausalityFeatures()
        self.voltage_features = VoltageFeatures()
        features = [
            self.simplex_features,
            self.causality_features,
            self.voltage_features,
        ]
        super().__init__(features=features, **params)

    def to_dict(self, parameters=None) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization"""
        return {
            "_target_": self._target_,
            "features": [feature.to_dict() for feature in self.features],
        }

    def __panel__(self, parameters=None):
        return pn.Column(
            pn.pane.Markdown("### Features Configuration"),
            pn.Row(
                *[
                    feature.__panel__()
                    for feature in self.features
                    if feature is not None
                ]
            ),
        )

    def set_values(self, d: Dict[str, Any], parameters=None) -> None:
        """Set values from a dictionary"""
        if parameters is None:
            parameters = ["_target_"]
        super().set_values(d, parameters=parameters)
        my_features = d.get("features", None)
        for feature, feat_dict in zip(self.features, my_features):
            if isinstance(feature, DashboardConfig):
                feature.set_values(feat_dict, parameters=parameters)


class ForecastingConfig(DashboardConfig):
    features = param.ClassSelector(
        class_=FeaturesConfig,
        default=FeaturesConfig(),
        doc="ForecastingConfig: Features for forecasting",
    )
    horizons = param.List(
        default=[1, 10, 20, 30, 40, 50, 60],
        item_type=int,
        doc="ForecastingConfig: Forecasting horizons as a list of int",
    )
    tick_data_folder = param.Path(
        default="./bidask_full_down_sampled",
        check_exists=False,
        doc="ForecastingConfig: Path to the tick data folder",
    )
    market_impacts_bps = param.List(
        default=[0.5],
        item_type=float,
        doc="ForecastingConfig: Market impacts in basis points",
    )
    delay = param.Number(
        default=10.0,
        bounds=(0.0, None),
        step=0.1,
        doc="ForecastingConfig: Delay in milliseconds",
    )
    add_bias = param.Boolean(
        default=False,
        doc="ForecastingConfig: Add bias to the forecasting",
    )
    _target_ = "circuit_rl.liquid_trader.readout.ForecastingConfig"
    _n_widget_columns = 3

    def __init__(self, features=None, **params):
        if features is None:
            features = FeaturesConfig()
        # self.tick_data_widget = FileInputWidget(
        #     self, "tick_data_folder", "Tick Data Folder",
        # )
        super().__init__(**params)

    @property
    def widget_parameters(self):
        return [
            "horizons",
            "delay",
            "market_impacts_bps",
            "tick_data_folder",
            "add_bias",
        ]

    @property
    def widgets(self):
        return {
            "delay": {
                "type": pn.widgets.FloatInput,
                "width": INPUT_WIDTH,
                "format": "0.00",
            },
            "add_bias": {
                "type": pn.widgets.Toggle,
                "width": INPUT_WIDTH,
            },
            # "horizons": {
            #     "type": pn.widgets.TextAreaInput,
            #     "placeholder": str(self.horizons),
            #     "height": 80,
            #     "width": int(INPUT_WIDTH * 4),
            # },
            # "market_impacts_bps": {
            #     "type": pn.widgets.TextAreaInput,
            #     "placeholder": str(self.market_impacts_bps),
            #     "height": 80,
            #     "width": int(INPUT_WIDTH * 4),
            # },
        }

    def to_dict(self, parameters=None):
        data = super().to_dict(parameters)
        # self.tick_data_folder = self.tick_data_widget.file_path
        horizons = [
            {"_target_": "pandas.Timedelta", "seconds": h}
            for h in self.horizons
        ]
        data["horizons"] = horizons
        data["tick_data_folder"] = {
            "_target_": "pathlib.Path",
            "_args_": [self.tick_data_folder],
        }
        data["delay"] = {
            "_target_": "pandas.Timedelta",
            "milliseconds": self.delay,
        }

        return data

    @property
    def dict_param_names(self) -> list[str]:
        return list(set(super().dict_param_names) - {"horizons"})

    def set_values(self, d: Dict[str, Any], parameters=None) -> None:
        """Set values from a dictionary"""
        # if parameters is None:
        #     parameters = list(set(self.dict_param_names) - {"horizons"})
        horizons = d.pop("horizons", None)
        if parameters is None:
            parameters = list(
                set(self.dict_param_names)
                - {"features", "tick_data_folder", "delay"}
            )
        for key in parameters:
            if key in d:
                setattr(self, key, d[key])
        if "tick_data_folder" in d:
            if isinstance(d["tick_data_folder"], dict):
                self.tick_data_folder = d["tick_data_folder"]["_args_"][0]
                # self.tick_data_widget.file_path = self.tick_data_folder
                # self.tick_data_widget.file_selector.value = [
                #     self.tick_data_folder
                # ]
        if "delay" in d and isinstance(d["delay"], dict):
            self.delay = d["delay"]["milliseconds"]
        features = d.pop("features", None)
        if features is not None:
            self.features.set_values(features)
        if horizons is not None:
            new_horizons = []
            for h in horizons:
                try:
                    if isinstance(h, dict) and "seconds" in h:
                        value = int(h["seconds"])
                    else:
                        value = int(h)
                    new_horizons.append(value)
                except (TypeError, ValueError) as e:
                    print(f"Error processing horizon value {h}: {e}")
            self.horizons = new_horizons

    def __panel__(self, parameters=None):
        params = super().__panel__(parameters)

        return pn.Column(
            pn.pane.Markdown("### Forecasting Configuration"),
            # self.tick_data_widget,
            params,
            pn.Row(
                *[
                    feature.__panel__()
                    for feature in self.features.features
                    if feature is not None
                ]
            ),
        )


class TradingPeriod(DashboardConfig):
    warmup_start = param.Date(
        default=pd.to_datetime("2024-08-08 9:30:00"),
        doc="Training warmup start time",
    )
    train_start = param.Date(
        default=pd.to_datetime("2024-08-08 9:35:00"), doc="Training start time"
    )
    # train_end = param.Date(
    #     default=pd.to_datetime("2024-08-08 10:00:00"), doc="Training end time"
    # )
    # test_warmup_start = param.Date(
    #     default=pd.to_datetime("2024-08-08 9:55:00"),
    #     doc="Testing warmup start time",
    # )
    test_start = param.Date(
        default=pd.to_datetime("2024-08-08 10:00:00"), doc="Testing start time"
    )
    test_end = param.Date(
        default=pd.to_datetime("2024-08-08 10:25:00"), doc="Testing end time"
    )
    _target_ = param.String(default="circuit_rl.liquid_trader.TradingPeriod")

    def __panel__(self, parameters=None):
        return pn.Row(
            pn.Card(
                pn.Column(
                    pn.pane.Markdown("#### Training Period"),
                    pn.Param(
                        self.param,
                        parameters=[
                            "warmup_start",
                            "train_start",
                            # "train_end",
                        ],
                        show_name=False,
                        widgets={
                            "warmup_start": {
                                "type": pn.widgets.DatetimePicker,
                                "width": int(INPUT_WIDTH * 2),
                            },
                            "train_start": {
                                "type": pn.widgets.DatetimePicker,
                                "width": int(INPUT_WIDTH * 2),
                            },
                            "train_end": {
                                "type": pn.widgets.DatetimePicker,
                                "width": int(INPUT_WIDTH * 2),
                            },
                        },
                    ),
                ),
                title="Training Time Period",
                width=250,
            ),
            pn.Card(
                pn.Column(
                    pn.pane.Markdown("#### Testing Period"),
                    pn.Param(
                        self.param,
                        parameters=[
                            # "test_warmup_start",
                            "test_start",
                            "test_end",
                        ],
                        show_name=False,
                        widgets={
                            "test_start": {
                                "type": pn.widgets.DatetimePicker,
                                "width": int(INPUT_WIDTH * 2),
                            },
                            "test_end": {
                                "type": pn.widgets.DatetimePicker,
                                "width": int(INPUT_WIDTH * 2),
                            },
                        },
                    ),
                ),
                title="Testing Time Period",
                width=250,
            ),
        )

    def _to_dict(self, parameters=None):
        """Convert to dictionary with proper datetime instantiation targets for hydra."""
        datetime_format = "%Y-%m-%d %H:%M:%S"
        return {
            "_target_": self._target_,
            "warmup_start": {
                "_target_": "pandas.to_datetime",
                "_args_": [self.warmup_start.strftime(datetime_format)],
            },
            "train_start": {
                "_target_": "pandas.to_datetime",
                "_args_": [self.train_start.strftime(datetime_format)],
            },
            # "train_end": {
            #     "_target_": "pandas.to_datetime",
            #     "_args_": [self.train_end.strftime(datetime_format)],
            # },
            # "test_warmup_start": {
            #     "_target_": "pandas.to_datetime",
            #     "_args_": [self.test_warmup_start.strftime(datetime_format)],
            # },
            "test_start": {
                "_target_": "pandas.to_datetime",
                "_args_": [self.test_start.strftime(datetime_format)],
            },
            "test_end": {
                "_target_": "pandas.to_datetime",
                "_args_": [self.test_end.strftime(datetime_format)],
            },
        }

    def to_dict(self, parameters=None):
        """Convert to dictionary with proper datetime instantiation targets for hydra."""
        datetime_format = "%Y-%m-%d %H:%M:%S"
        if isinstance(self.warmup_start, str):
            warmup_start = pd.to_datetime(self.warmup_start)
        else:
            warmup_start = self.warmup_start
        if isinstance(self.train_start, str):
            train_start = pd.to_datetime(self.train_start)
        else:
            train_start = self.train_start
        if isinstance(self.test_start, str):
            test_start = pd.to_datetime(self.test_start)
        else:
            test_start = self.test_start
        if isinstance(self.test_end, str):
            test_end = pd.to_datetime(self.test_end)
        else:
            test_end = self.test_end
        return {
            "_target_": self._target_,
            "warmup_start": warmup_start.strftime(datetime_format),
            "train_start": train_start.strftime(datetime_format),
            # "train_end": {
            #     "_target_": "pandas.to_datetime",
            #     "_args_": [self.train_end.strftime(datetime_format)],
            # },
            # "test_warmup_start": {
            #     "_target_": "pandas.to_datetime",
            #     "_args_": [self.test_warmup_start.strftime(datetime_format)],
            # },
            "test_start": test_start.strftime(datetime_format),
            "test_end": test_end.strftime(datetime_format),
        }

    def set_values(self, d, parameters=None) -> None:
        """Set values from a dictionary."""
        for key in [
            "warmup_start",
            "train_start",
            # "train_end",
            # "test_warmup_start",
            "test_start",
            "test_end",
        ]:
            if key in d:
                # Handle both string and dictionary formats for datetimes
                if isinstance(d[key], dict) and "_args_" in d[key]:
                    # When loading from config with _target_ and _args_
                    datetime_str = d[key]["_args_"][0]
                    setattr(self, key, pd.to_datetime(datetime_str))
                else:
                    # When loading directly as string
                    setattr(self, key, pd.to_datetime(d[key]))
            if "_target_" in d:
                self._target_ = d["_target_"]

    def instantiate(self, **kwargs):
        datetime_format = "%Y-%m-%d %H:%M:%S"
        data = self.to_dict()
        for key in [
            "warmup_start",
            "train_start",
            # "train_end",
            # "test_warmup_start",
            "test_start",
            "test_end",
        ]:
            value = (
                pd.to_datetime(data[key])
                if isinstance(data[key], str)
                else data[key]
            )
            data[key] = value.strftime(datetime_format)
        return super().instantiate(**data, _convert_="all")


class LiquidTraderConfig(DashboardConfig):
    output_folder = param.String(
        default=None,
        doc="Path to the output folder",
    )
    config_file = param.String(
        default=None,
        doc="Path to the config file",
    )
    input_data = param.ClassSelector(
        class_=PriceNewsTicks,
        default=None,
        doc="Input data for the trader",
    )
    trade_moment = param.ClassSelector(
        class_=TradeMomentConfig,
        default=None,
        doc="Trade moment to evaluate",
    )
    period = param.ClassSelector(
        class_=TradingPeriod,
        default=None,
        doc="Trading period for evaluation",
    )
    forecasting = param.ClassSelector(
        class_=ForecastingConfig,
        default=None,
        doc="Forecasting configuration",
    )
    lsm_config = param.ClassSelector(
        class_=LiquidStateMachineConfig,
        default=None,
        doc="Liquid state machine configuration",
    )
    interactive_plots = param.Boolean(
        default=False,
        doc="Use Pythia for forecasting",
    )
    _target_ = param.String(
        default="circuit_rl.liquid_trader.LiquidTraderConfig"
    )

    def __init__(
        self,
        input_data=None,
        trade_moment=None,
        period=None,
        forecasting=None,
        lsm_config=None,
        **params,
    ):
        if input_data is None:
            input_data = PriceNewsTicks()
        if trade_moment is None:
            trade_moment = TradeMomentConfig()
        if period is None:
            period = TradingPeriod()
        if forecasting is None:
            forecasting = ForecastingConfig()
        if lsm_config is None:
            lsm_config = LiquidStateMachineConfig()
        super().__init__(
            input_data=input_data,
            trade_moment=trade_moment,
            period=period,
            forecasting=forecasting,
            lsm_config=lsm_config,
            **params,
        )

    @property
    def widgets(self):
        return {
            "interactive_plots": {
                "type": pn.widgets.Toggle,
                "width": INPUT_WIDTH,
            },
        }

    @property
    def widget_parameters(self):
        return [
            "output_folder",
            "config_file",
            "interactive_plots",
        ]

    def create_inputs_and_moments_panel(self):
        return pn.Row(
            pn.Column(
                pn.pane.Markdown("### Input params"),
                self.input_data.__panel__(),
                pn.pane.Markdown("### Trade moments params"),
                self.trade_moment.__panel__(),
            ),
            self.trade_moment.detector.__panel__(),
        )

    def __panel__(self, parameters=None):
        return pn.Column(
            pn.pane.Markdown("### Trader Evaluation"),
            self.trade_moment.__panel__(),
            pn.pane.Markdown("#### Trader Evaluation Configuration"),
            self.forecasting.__panel__(),
            pn.pane.Markdown("####  Period Configuration"),
            self.period.__panel__(),
        )

    def to_dict(self, parameters=None) -> dict[str, Any]:
        """Convert configuration to dictionary for serialization"""
        data = super().to_dict(parameters=parameters)
        output_folder = data.pop("output_folder", None)
        output_file = data.pop("config_file", None)
        data["output"] = {
            "output_folder": output_folder,
            "config_file": output_file,
            "_target_": "circuit_rl.liquid_trader.OutputData",
        }
        return data

    def set_values(self, d, parameters=None) -> None:
        """Set values from a dictionary"""
        parameters = parameters or {
            x
            for x in self.dict_param_names
            if x not in {"output_folder", "config_file"}
        }
        super().set_values(d, parameters=parameters)
        if "output" in d:
            output = d["output"]
            self.output_folder = output.get(
                "output_folder", self.output_folder
            )
            self.config_file = output.get("config_file", self.config_file)
