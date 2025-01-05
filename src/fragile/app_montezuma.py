from functools import partial
import threading
import time

import holoviews as hv
from holoviews.streams import Pipe
import numpy as np
import pandas as pd
import panel as pn
import param
import plangym
from plangym.utils import process_frame

from fragile.montezuma import aggregate_visits, FractalTree
from fragile.shaolin.stream_plots import Image, RGB


hv.extension("bokeh")
pn.extension("tabulator", theme="dark")


class FaiRunner(param.Parameterized):
    is_running = param.Boolean(default=False)

    def __init__(self, fai, n_steps, plot=None, report_interval=100):
        super().__init__()
        self.reset_btn = pn.widgets.Button(icon="restore", button_type="primary")
        self.play_btn = pn.widgets.Button(icon="player-play", button_type="primary")
        self.pause_btn = pn.widgets.Button(icon="player-pause", button_type="primary")
        self.step_btn = pn.widgets.Button(name="Step", button_type="primary")
        self.progress = pn.indicators.Progress(
            name="Progress", value=0, width=600, max=n_steps, bar_color="primary"
        )
        self.sleep_val = pn.widgets.FloatInput(value=0.0, width=60)
        self.report_interval = pn.widgets.IntInput(value=report_interval)
        self.table = pn.widgets.Tabulator()
        self.fai = fai
        self.n_steps = n_steps
        self.curr_step = 0
        self.plot = plot
        self.thread = None
        self.erase_coef_val = pn.widgets.FloatInput(value=0.05, width=60, name="erase")

    @param.depends("erase_coef_val.value")
    def update_erase_coef(self):
        self.fai.erase_coef = self.erase_coef_val.value

    @param.depends("reset_btn.value")
    def on_reset_click(self):
        self.fai.reset()
        self.curr_step = 0
        self.progress.value = 1
        self.curr_step = 0
        self.play_btn.disabled = False
        self.pause_btn.disabled = True
        self.step_btn.disabled = False
        self.is_running = False
        self.progress.bar_color = "primary"
        summary = pd.DataFrame(self.fai.summary(), index=[0])
        self.table.value = summary
        if self.plot is not None:
            self.plot.reset(self.fai)
            self.plot.send(self.fai)

    @param.depends("play_btn.value")
    def on_play_click(self):
        self.play_btn.disabled = True
        self.pause_btn.disabled = False
        self.is_running = True
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self.run)
            self.thread.start()

    @param.depends("pause_btn.clicks")
    def on_pause_click(self):
        self.is_running = False
        self.play_btn.disabled = False
        self.pause_btn.disabled = True
        if self.thread is not None:
            self.thread.join()

    @param.depends("step_btn.value")
    def on_step_click(self):
        self.take_single_step()

    def take_single_step(self):
        self.fai.step_tree()
        self.curr_step += 1
        self.progress.value = self.curr_step
        if self.curr_step >= self.n_steps:
            self.is_running = False
            self.progress.bar_color = "success"
            self.step_btn.disabled = True
            self.play_btn.disabled = True
            self.pause_btn.disabled = True

        if self.fai.oobs.sum().cpu().item() == self.fai.n_walkers - 1:
            self.is_running = False
            self.progress.bar_color = "danger"

        if self.fai.iteration % self.report_interval.value == 0:
            summary = pd.DataFrame(self.fai.summary(), index=[0])
            self.table.value = summary
            if self.plot is not None:
                self.plot.send(self.fai)

    def run(self):
        while self.is_running:
            self.take_single_step()
            time.sleep(self.sleep_val.value)

    def __panel__(self):
        # pn.state.add_periodic_callback(self.run, period=20)

        return pn.Column(
            self.table,
            self.progress,
            pn.Row(
                self.play_btn,
                self.pause_btn,
                self.reset_btn,
                self.step_btn,
                pn.pane.Markdown("**Sleep**"),
                self.sleep_val,
                self.report_interval,
                self.erase_coef_val,
            ),
            self.on_play_click,
            self.on_pause_click,
            self.on_reset_click,
            self.on_step_click,
            self.update_erase_coef,
            # self.run,
        )


PYRAMID = [
    [-1, -1, -1, 0, 1, 2, -1, -1, -1],
    [-1, -1, 3, 4, 5, 6, 7, -1, -1],
    [-1, 8, 9, 10, 11, 12, 13, 14, -1],
    [15, 16, 17, 18, 19, 20, 21, 22, 23],
]
EMPTY_ROOMS = [
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 6),
    (0, 7),
    (0, 8),
    (1, 0),
    (1, 1),
    (1, 7),
    (1, 8),
    (2, 0),
    (2, 8),
]


def get_rooms_xy(pyramid=None) -> np.ndarray:
    """Get the tuple that encodes the provided room."""
    pyramid = pyramid if pyramid is not None else PYRAMID
    n_rooms = max(max(row) for row in pyramid) + 1
    rooms_xy = []
    for room in range(n_rooms):
        for y, loc in enumerate(PYRAMID):
            if room in loc:
                room_xy = [loc.index(room), y]
                rooms_xy.append(room_xy)
                break
    return np.array(rooms_xy)


def get_pyramid_layout(room_h=160, room_w=160, channels=3, pyramid=None, empty_rooms=None):
    pyramid = pyramid if pyramid is not None else PYRAMID
    ph, pw = len(pyramid), len(pyramid[0])
    all_rooms = np.zeros((room_h * ph, room_w * pw, channels))
    return set_empty_rooms(all_rooms, empty_rooms, height=room_h, width=room_w)


def set_empty_rooms(all_rooms, empty_rooms=None, height=160, width=160):
    empty_rooms = empty_rooms if empty_rooms is not None else EMPTY_ROOMS
    val = np.array([255, 255, 255], dtype=np.uint8)
    for i, j in empty_rooms:
        all_rooms[i * height : (i + 1) * height, j * width : (j + 1) * width] = val
    return all_rooms


def draw_rooms(env, pyramid_layout=None, height=160, width=160):
    pyramid_layout = pyramid_layout if pyramid_layout is not None else get_pyramid_layout()
    rooms = env.rooms
    for n_room, room in rooms.items():
        i, j = env.get_room_xy(n_room)
        coord_x, coord_x1 = j * width, (j + 1) * width
        coord_y, coord_y1 = i * height, (i + 1) * height
        pyramid_layout[coord_x:coord_x1, coord_y:coord_y1, :] = room
    return pyramid_layout


def to_pyramid_coords(observ, room_xy, width=160, height=160):
    x, y, room = (
        observ[:, 0].astype(np.int64),
        observ[:, 1].astype(np.int64),
        observ[:, 2].astype(np.int64),
    )
    room_coords = room_xy[room]
    offset_coords = room_coords * np.array([width, height])
    return np.array([x, y]).T + offset_coords


def to_plot_coords(room_coords, width=160, height=160):
    plot_x = (room_coords[:, 0]) / (width - 1) - 0.5
    plot_y = ((height - 1) - room_coords[:, 1]) / (height - 1) - 0.5
    return plot_x, plot_y


def draw_pyramid(data, pyramid_layout=None):
    return hv.RGB(draw_rooms(data, pyramid_layout)).opts(
        width=1440, height=640, xaxis=None, yaxis=None
    )


def draw_tree_pyramid(data, max_x: int = 1440, max_y: int = 640, room_xy=None):
    room_xy = room_xy if room_xy is not None else get_rooms_xy()
    if not data:
        return hv.Segments(bgcolor=None) * hv.Scatter(bgcolor=None)
    observ = data.observ.cpu().numpy().astype(np.int64)
    room_coords = to_pyramid_coords(observ, room_xy)
    parents = data.parent.cpu().numpy()
    room_coords = room_coords.astype(np.float64)
    room_coords[:, 0] /= float(data.env.gym_env._x_repeat)
    room_coords = room_coords.astype(np.int64)
    plot_x, plot_y = to_plot_coords(room_coords, width=max_x, height=max_y)
    segs = plot_x[parents], plot_y[parents], plot_x, plot_y
    edges = hv.Segments(segs).opts(line_color="white", bgcolor=None)
    nodes = hv.Scatter((plot_x, plot_y)).opts(
        size=2, bgcolor=None, color="red", line_width=0.01, xaxis=None, yaxis=None
    )
    return edges * nodes


def draw_tree_best_room(data, width=160, height=160):
    if not data:
        return hv.Segments(bgcolor=None) * hv.Scatter(bgcolor=None)
    room_coords = data.observ.cpu().numpy().astype(np.int64)
    room = room_coords[:, 2][data.cum_reward.argmax().cpu().item()]
    room_ix = room_coords[:, 2] == room
    parents = data.parent.cpu().numpy()[room_ix]
    room_coords = room_coords.astype(np.float64)
    room_coords[:, 0] /= float(data.env.gym_env._x_repeat)
    room_coords = room_coords.astype(np.int64)
    room_coords = room_coords[room_ix]
    plot_x, plot_y = to_plot_coords(room_coords, width=width, height=height)
    segs = plot_x[parents], plot_y[parents], plot_x, plot_y
    edges = hv.Segments(segs).opts(line_color="black", bgcolor=None)
    nodes = hv.Scatter((plot_x, plot_y)).opts(size=2, bgcolor=None, color="red")
    return edges * nodes


class MontezumaDisplay:
    def __init__(
        self,
    ):
        self.best_rgb = RGB()
        self.room_grey = Image(cmap="greys")
        self.visits_image = Image(alpha=0.7, xaxis=None, yaxis=None, cmap="fire", bgcolor=None)
        self.visits = np.zeros((24, 160, 160), dtype=np.int32) * np.nan
        self.rooms = np.zeros((24, 160, 160))
        self.visited_rooms = []
        self.pipe_tree = Pipe()
        self.room_pipe = Pipe()
        self._curr_best = -1
        # self.tree_best_room = hv.DynamicMap(draw_tree_best_room, streams=[self.pipe_tree])
        self.tree_pyramid = hv.DynamicMap(
            partial(draw_tree_pyramid, room_xy=get_rooms_xy()), streams=[self.pipe_tree]
        )
        self.pyramid = hv.DynamicMap(draw_pyramid, streams=[self.room_pipe])

    def reset(self, fai):  # noqa: ARG002
        self.visited_rooms = []
        self.visits = np.zeros((24, 160, 160), dtype=np.int32) * np.nan
        self.rooms = np.zeros((24, 160, 160))

    def send(self, fai):
        best_ix = fai.cum_reward.argmax().cpu().item()
        best_rgb = fai.rgb[best_ix]
        if best_ix != self._curr_best:
            self.best_rgb.send(best_rgb)
            self._curr_best = best_ix

        observ = fai.observ.cpu().numpy().astype(np.float64)
        observ[:, 0] /= int(fai.env.gym_env._x_repeat)
        observ = observ.astype(np.int64)
        room_ix = observ[:, 2]
        for ix in np.unique(room_ix):
            if ix not in self.visited_rooms:
                self.visited_rooms.append(ix)
                self.room_pipe.send(fai.env.gym_env)
                batch_ix = np.argmax(room_ix == ix)
                self.rooms[ix] = process_frame(fai.rgb[batch_ix][50:], mode="L").copy()
        best_room_ix = room_ix[best_ix]
        self.room_grey.send(self.rooms[best_room_ix])
        visits = fai.visits[best_room_ix][None]
        visits = aggregate_visits(visits, block_size=fai.agg_block_size, upsample=True)[0]
        visits[visits == 0] = np.nan
        self.visits_image.send(visits)
        self.pipe_tree.send(fai)

    def __panel__(self):
        return pn.Column(
            pn.Row(
                self.best_rgb.plot,
                self.room_grey.plot * self.visits_image.plot,
                # self.room_grey.plot * self.tree_best_room,
            ),
            self.pyramid * self.tree_pyramid,
        )


def main():
    env = plangym.make(
        "PlanMontezuma-v0",
        obs_type="coords",
        return_image=True,
        frameskip=3,
        check_death=True,
        episodic_life=False,
    )  # , n_workers=10, ray=True)

    n_walkers = 10000
    plot = MontezumaDisplay()
    fai = FractalTree(
        max_walkers=n_walkers, env=env, device="cpu", min_leafs=100, start_walkers=100
    )
    runner = FaiRunner(fai, 1000000, plot=plot)
    pn.panel(pn.Column(runner, plot)).servable()


main()
