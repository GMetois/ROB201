import arcade
import time
from typing import Optional, Tuple, Dict, Union, Type
import cv2

from spg.agent.controller.controller import Command, Controller
from spg.playground import Playground
from spg.view import TopDownView

from place_bot.utils.constants import FRAME_RATE
from place_bot.entities.robot_abstract import RobotAbstract
from place_bot.entities.keyboard_controller import KeyboardController
from place_bot.utils.fps_display import FpsDisplay
from place_bot.simu_world.world_abstract import WorldAbstract
from place_bot.utils.mouse_measure import MouseMeasure
from place_bot.utils.screen_recorder import ScreenRecorder
from place_bot.utils.visu_noises import VisuNoises


class Simulator(TopDownView):
    def __init__(
            self,
            the_world: WorldAbstract,
            size: Optional[Tuple[int, int]] = None,
            center: Tuple[float, float] = (0, 0),
            zoom: float = 1,
            display_uid: bool = False,
            draw_transparent: bool = False,
            draw_interactive: bool = False,
            draw_lidar: bool = False,
            use_keyboard: bool = False,
            use_mouse_measure: bool = False,
            enable_visu_noises: bool = False,
            filename_video_capture: str = None
    ) -> None:
        super().__init__(
            the_world.playground,
            size,
            center,
            zoom,
            display_uid,
            draw_transparent,
            draw_interactive,
        )

        self._playground.window.set_size(*self._size)
        self._playground.window.set_visible(True)

        self._the_world = the_world
        self._robot = self._the_world.robot

        self._robot_commands: Union[Dict[RobotAbstract, Dict[Union[str, Controller], Command]], Type[None]] = None
        if self._robot:
            self._robot_commands = {}

        self._playground.window.on_draw = self.on_draw
        self._playground.window.on_update = self.on_update
        self._playground.window.on_key_press = self.on_key_press
        self._playground.window.on_key_release = self.on_key_release
        self._playground.window.on_mouse_motion = self.on_mouse_motion
        self._playground.window.on_mouse_press = self.on_mouse_press
        self._playground.window.on_mouse_release = self.on_mouse_release
        self._playground.window.set_update_rate(FRAME_RATE)

        self._draw_lidar = draw_lidar
        self._use_keyboard = use_keyboard
        self._use_mouse_measure = use_mouse_measure
        self._enable_visu_noises = enable_visu_noises

        self._elapsed_time = 0
        self._start_real_time = time.time()
        self._real_time_elapsed = 0

        self._terminate = False

        self.fps_display = FpsDisplay(period_display=2)
        self._keyboardController = KeyboardController()
        self._mouse_measure = MouseMeasure(playground_size=self._playground.size)
        self._visu_noises = VisuNoises(playground_size=self._playground.size, robot=self._robot)

        self.recorder = ScreenRecorder(self._size[0], self._size[1], fps=30, out_file=filename_video_capture)

    def run(self):
        self._playground.window.run()

    def on_draw(self):
        self._playground.window.clear()
        self._fbo.use()
        self.draw()

    def on_update(self, delta_time):
        self._elapsed_time += 1

        if self._elapsed_time < 5:
            self._playground.step(commands=self._robot_commands)
            return

        # COMPUTE COMMANDS
        command = self._robot.control()
        if self._use_keyboard:
            command = self._keyboardController.control()

        self._robot_commands[self._robot] = command

        if self._robot:
            self._robot.display()

        self._playground.step(commands=self._robot_commands)

        self._visu_noises.update(enable=self._enable_visu_noises)

        end_real_time = time.time()
        self._real_time_elapsed = (end_real_time - self._start_real_time)

        # Capture the frame
        self.recorder.capture_frame(self)

        self.fps_display.update(display=False)

        if self._terminate:
            self.recorder.end_recording()
            arcade.close_window()

    def get_playground_image(self):
        self.update()
        # The image should be flip and the color channel permuted
        image = cv2.flip(self.get_np_img(), 0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def draw(self, force=False):
        arcade.start_render()
        self.update_sprites(force)

        self._playground.window.use()
        self._playground.window.clear(self._background)

        if self._draw_lidar:
            for robot in self._playground.agents:
                robot.lidar().draw()

        self._mouse_measure.draw(enable=self._use_mouse_measure)
        self._visu_noises.draw(enable=self._enable_visu_noises)

        self._transparent_sprites.draw(pixelated=True)
        self._interactive_sprites.draw(pixelated=True)
        self._visible_sprites.draw(pixelated=True)
        self._traversable_sprites.draw(pixelated=True)

    def on_key_press(self, key, modifiers):
        """Called whenever a key is pressed."""
        self._keyboardController.on_key_press(key, modifiers)

        if key == arcade.key.Q:
            self._terminate = True

        if key == arcade.key.R:
            self._playground.reset()
            self._visu_noises.reset()

        if key == arcade.key.L:
            self._draw_lidar = not self._draw_lidar

    def on_key_release(self, key, modifiers):
        self._keyboardController.on_key_release(key, modifiers)

    # Creating function to check the position of the mouse
    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int):
        self._mouse_measure.on_mouse_motion(x, y, dx, dy)

    # Creating function to check the mouse clicks
    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int):
        self._mouse_measure.on_mouse_press(x, y, button, enable=self._use_mouse_measure)

    def on_mouse_release(self, x: int, y: int, button: int, modifiers: int):
        self._mouse_measure.on_mouse_release(x, y, button, enable=self._use_mouse_measure)

    @property
    def elapsed_time(self):
        return self._elapsed_time

    @property
    def real_time_elapsed(self):
        return self._real_time_elapsed
