# This class is used to store the last key pressed and reset it
class KeyPressed:
    def __init__(self):
        self.last_key = None
        self.unresolved_key_press = False

    def on_key_event(self, event):
        self.last_key = event.name
        self.unresolved_key_press = True

    def get_last_key(self):
        return self.last_key

    def resolve_key_press(self):
        self.unresolved_key_press = False

    def reset_key(self):
        self.last_key = None