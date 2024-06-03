# This class is used to store the last key pressed and reset it
class KeyPressed:
    def __init__(self):
        self.last_key = None

    def on_key_event(self, event):
        self.last_key = event.name

    def get_last_key(self):
        return self.last_key

    def reset_key(self):
        self.last_key = None