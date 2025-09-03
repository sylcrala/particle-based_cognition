from textual.app import ComposeResult
from textual.widgets import Label, Button, Hori
from textual.screen import Screen

from apis.api_registry import api

class QuitScreen(Screen):
    def __init__(self):
        self.title = "Quit"
        self.quit_confirm_question = Label("Are you sure you want to quit?")
        self.yes_button = Button("Yes", on_click=self.handle_yes)
        self.no_button = Button("No", on_click=self.handle_no)

    def compose(self):
        yield self.quit_confirm_question
        yield self.yes_button
        yield self.no_button

    def handle_yes(self):
        if api.handle_shutdown():
            self.app.exit()

    def handle_no(self):
        self.app.pop_screen()
