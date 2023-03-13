import arcade


class KeyboardController:
    def __init__(self):
        self._command = {"forward": 0.0,
                         "rotation": 0.0}

    def on_key_press(self, key, modifiers):
        """Called whenever a key is pressed."""
        if self._command:

            if key == arcade.key.UP:
                self._command["forward"] = 1.0
            elif key == arcade.key.DOWN:
                self._command["forward"] = -1.0

            if key == arcade.key.LEFT:
                self._command["rotation"] = 1.0
            elif key == arcade.key.RIGHT:
                self._command["rotation"] = -1.0

    def on_key_release(self, key, modifiers):
        """Called whenever a key is released."""
        if self._command:

            if key == arcade.key.UP:
                self._command["forward"] = 0
            elif key == arcade.key.DOWN:
                self._command["forward"] = 0

            if key == arcade.key.LEFT:
                self._command["rotation"] = 0
            elif key == arcade.key.RIGHT:
                self._command["rotation"] = 0

    def control(self):
        return self._command
