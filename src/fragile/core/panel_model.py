from typing import Any

from hydra.utils import get_object, instantiate
import panel as pn
import param


# Initialize Panel extensions
pn.extension("tabulator", notifications=True, throttled=True)

# Global constants for UI consistency
INPUT_WIDTH = 165
SLIDER_WIDTH = 200


class FileInputWidget(param.Parameterized):
    """A widget for selecting a file with a nice UI presentation."""

    file_path = param.String(default="", doc="Path to the selected file")
    widget_name = param.String(default="File", doc="Name of the widget")

    def __init__(
        self,
        ref,
        ref_name,
        widget_name="File",
        only_files=False,
        file_path=None,
        file_pattern="*",
        directory=".",
        **params,
    ):
        self._ref = ref
        self._ref_name = ref_name
        if file_path is None:
            file_path = str(getattr(ref, ref_name))
        else:
            file_path = str(file_path)
            setattr(ref, ref_name, file_path)
        super().__init__(
            widget_name=widget_name, file_path=file_path, **params
        )

        self.file_selector = pn.widgets.FileSelector(
            only_files=only_files,
            value=[file_path],
            file_pattern=file_pattern,
            directory=str(directory),
        )
        self.file_selector.param.watch(self._update_file_path, "value")
        text = """<a href="https://panel.holoviz.org/reference/widgets/FileSelector.html#display" target="_blank" rel="noopener noreferrer" style="color: #176bbf; text-decoration: none; font-weight: bold;">
    How to use the file explorer
</a>"""
        self.tooltip = pn.widgets.TooltipIcon(value=text)
        self._header = pn.Row(
            pn.pane.Markdown(f"### {self.widget_name}: {self.file_path}"),
            self.tooltip,
        )
        self._card = None

    def _update_file_path(self, event):
        """Update the file path and card header when a file is selected."""
        if event.new and len(event.new) > 0:
            new_value = str(event.new[0])
            self.file_path = new_value
            self._header[0].object = (
                f"### {self.widget_name}: {self.file_path}"
            )
            setattr(self._ref, self._ref_name, new_value)

    def __panel__(self):
        """Render the file input widget in a collapsed card."""
        self._card = pn.Card(
            self.file_selector, header=self._header, collapsed=True
        )
        return self._card


class PanelModel(param.Parameterized):
    _max_widget_width = param.Integer(
        default=800, bounds=(0, None), doc="Max widget width"
    )
    _n_widget_columns = param.Integer(
        default=2, bounds=(1, None), doc="Number of widget columns"
    )
    _target_ = param.String(doc="Target class for instantiation")

    @property
    def dict_param_names(self) -> list[str]:
        """Names of the config parameters that this class is tracking."""
        return list(
            set(list(self.param))
            - {"_max_widget_width", "_n_widget_columns", "name"}
        )

    @property
    def widgets(self) -> dict[str, dict]:
        """Widget configurations for the parameters."""
        return {}

    @property
    def widget_parameters(self) -> list[str]:
        """List of parameter names to be displayed as widgets."""
        ignore_params = {"_target_", "_max_widget_width", "_n_widget_columns"}
        return list(set(self.dict_param_names) - ignore_params)

    @property
    def default_layout(self):
        """Define the default layout for the parameters."""

        def new_class(cls, **kwargs):
            "Creates a new class which overrides parameter defaults."
            return type(type(cls).__name__, (cls,), kwargs)

        names = list(self.widget_parameters)
        ncols = (
            min(len(names), self._n_widget_columns)
        )
        return new_class(
            pn.GridBox, ncols=ncols, max_width=self._max_widget_width
        )

    @staticmethod
    def parse_lambda_function(k, d):
        if k in d:
            if isinstance(d[k], dict) and "_args_" in d[k]:
                # When loading from config with _target_ and _args_
                return d[k]["_args_"][0]
            if callable(d[k]):
                # When loading directly as callable
                import inspect

                return inspect.getsource(d[k]).strip()

    @classmethod
    def from_dict(cls, d, parameters=None) -> "PanelModel":
        """Create a PanelModel from a dictionary."""
        instance = cls()
        instance.set_values(d, parameters=parameters)
        return instance

    @classmethod
    def process_widgets(cls, widgets):
        """Process widgets to ensure consistent naming and structure."""
        new_widgets = {}
        for k, v in widgets.items():
            widget_data = v if isinstance(v, dict) else {"type": v}
            if "name" not in widget_data:
                widget_data["name"] = k.replace("_", " ").capitalize()
            new_widgets[k] = widget_data
        return new_widgets

    def instantiate(self, **kwargs):
        """Instantiate the class with the current parameter values."""
        d = self.to_dict()

        def _is_config(k):
            is_panel_model = isinstance(getattr(self, k), PanelModel)
            is_list_of_conf = isinstance(getattr(self, k), list) and all(
                isinstance(item, PanelModel) for item in getattr(self, k)
            )
            return is_panel_model or is_list_of_conf

        config_params = {
            k: (
                getattr(self, k).instantiate(**kwargs)
                if not isinstance(getattr(self, k), list)
                else [item.instantiate(**kwargs) for item in getattr(self, k)]
            )
            for k in d.keys()
            if _is_config(k)
        }
        raw_params = {
            k: v for k, v in d.items() if not _is_config(k) and k != "_target_"
        }

        all_params = {
            **instantiate(raw_params, _convert_="all"),
            **config_params,
            **kwargs,
        }

        if "_target_" in d:
            return get_object(self._target_)(**all_params)
        return all_params

    def to_dict(self, parameters=None) -> dict[str, Any]:
        """Convert the parameter values to a dictionary."""
        if parameters is None:
            parameters = self.dict_param_names
        return {
            param: (
                getattr(self, param).to_dict()
                if hasattr(getattr(self, param), "to_dict")
                else getattr(self, param)
            )
            for param in parameters
        }

    def set_values(self, d, parameters=None) -> None:
        """Set values from a dictionary."""
        if parameters is None:
            parameters = self.dict_param_names
        for key in parameters:
            if key in d and hasattr(getattr(self, key), "set_values"):
                try:
                    getattr(self, key).set_values(d[key])
                except Exception as e:
                    print(

                            f"Error setting values for {self.__class__.__name__} "
                            f"{getattr(self, key)} {key} of {d.keys()} with value {d[key]}: {e}"

                    )
                    raise e
            elif key in d:
                setattr(self, key, d[key])

    def __panel__(self, parameters=None):
        """Render the LIF neuron parameters in a single column layout."""
        if parameters is None:
            parameters = self.widget_parameters
        return pn.Param(
            self,
            show_name=False,
            parameters=parameters,
            widgets=self.process_widgets(self.widgets),
            default_layout=self.default_layout,
        )
