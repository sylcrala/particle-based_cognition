from textual.containers import Vertical, Horizontal
from textual.widgets import Static, RichLog, Button, Input, ContentSwitcher
from textual.app import ComposeResult

from apis.api_registry import api

class TodoList(Vertical):
    CSS = """
    #content-switcher {
        text-align: center;
        color: cyan;
        border: solid white;
        padding: 1;
        margin-bottom: 1;
    }
    #TodoList {
        height: 100%;
        padding: 1;
    }
    
    #todo-title {
        height: 3;
        text-align: center;
        color: cyan;
        border: solid white;
        padding: 1;
        margin-bottom: 1;
    }

    #todo-container {
        height: 100%;
    }

    #task-viewer {
        height: 3;
        text-align: center;
        color: cyan;
        border: solid white;
        padding: 1;
        margin-bottom: 1;
    }
    
    #task-management {
        width: 100%;
        border: solid green;
        padding: 1;
    }

    #task-viewer-btn {
        margin: 0 1;
        min-width: 15;
    }
    
    #task-management-btn {
        margin-bottom: 1;
        width: 100%;.;
        max-width: 20;
    }

    #task-viewer-buttons {
        margin-bottom: 1;
        justify-content: center;
        align-items: center;
        max-height: 3;
        display: flex;
    }
        

    #task-list {
        border: solid yellow;
        height: 90%;
        margin-top: 1;
        padding: 1;
    }
    """


    def __init__(self):
        super().__init__()
        self.title = "To-Do List"
        self.id = "todo"
        self.todo_list_api = api.get_api("todo_list")
        self.logger = api.get_api("logger")
        self.pending_task = None

    def compose(self) -> ComposeResult:
        with ContentSwitcher(initial = "main", id = "content-switcher"):
            with Vertical(id = "main"):
                yield Static("To-Do List", id="todo-title")
                with Vertical(id="task-viewer"):
                    with Horizontal(id="task-viewer-buttons"):
                        yield Button("All", id="view-all-btn", classes="task-viewer-btn", variant="primary")
                        yield Button("Pending", id="view-incomplete-btn", classes="task-viewer-btn", variant="default")
                        yield Button("Done", id="view-completed-btn", classes="task-viewer-btn", variant="success")
                    with Vertical(id="task-list"):
                        yield self.TaskViewer(self.todo_list_api)
                    with Horizontal(id="task-management", classes="task-management"):
                        yield Button("Add Task", id="add-task-btn", classes="task-management-btn", variant="primary")
                        yield Button("Complete Task", id="complete-task-btn", classes="task-management-btn", variant="success")
                        yield Button("Remove Task", id="remove-task-btn", classes="task-management-btn", variant="error")
            with Vertical(id="new-task-popup"):
                yield Static("What task would you like to add?")
                yield Input(placeholder="Enter task description", id="new-task-input")
                with Horizontal():
                    yield Button("Add Task", id="confirm-add-task-btn", variant="primary")
                    yield Button("Cancel", id="cancel-add-task-btn", variant="error")

    def switch_content(self, content_id: str):
        """Switch the content area to show different content"""
        try:
            content_switcher = self.query_one("#content-switcher", ContentSwitcher)
            content_switcher.current = content_id

        except Exception as e:
            self.logger.log(f"Issue while switching content: {e}", "ERROR", "switch_content()", "TodoList")

    class TaskViewer(RichLog):
        def __init__(self, todo_api):
            super().__init__()
            self.id = "task-list"
            self.highlight = True
            self.markup = True
            self.todo_api = todo_api
            
        async def on_mount(self):
            """Load and display tasks when mounted"""
            self.refresh_tasks()
            
        def refresh_tasks(self, filter_type="all"):
            """Refresh the task display"""
            if not self.todo_api:
                self.write("[red]Todo API not available[/red]")
                return
                
            tasks = self.todo_api.get_tasks()
            self.clear()
            
            if not tasks:
                self.write("[dim]No tasks yet. Add one to get started![/dim]")
                return
                
            # Filter tasks based on type
            if filter_type == "incomplete":
                tasks = [t for t in tasks if not t.get("completed", False)]
            elif filter_type == "completed":
                tasks = [t for t in tasks if t.get("completed", False)]
            
            if not tasks:
                self.write(f"[dim]No {filter_type} tasks found[/dim]")
                return
                
            # Display tasks
            for task in tasks:
                status = "✅" if task.get("completed", False) else "⏳"
                created = task.get("created_at", "Unknown")
                if hasattr(created, 'strftime'):
                    date_str = created.strftime("%m/%d %H:%M")
                else:
                    date_str = str(created)
                    
                task_text = task.get("task", "No description")
                color = "green" if task.get("completed") else "white"
                
                self.write(f"[{color}]{status} [{task['id']}] {task_text}[/] [dim]({date_str})[/]")

    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses in the todo widget"""
        task_viewer = self.query_one("#task-list", self.TaskViewer)
        
        if event.button.id == "view-all-btn":
            task_viewer.refresh_tasks("all")

        if event.button.id == "view-incomplete-btn":
            task_viewer.refresh_tasks("incomplete")

        if event.button.id == "view-completed-btn":
            task_viewer.refresh_tasks("completed")

        if event.button.id == "add-task-btn":
            # For now, add a sample task - you can make this interactive later
            if self.todo_list_api:
                self.switch_content("new-task-popup")

        if event.button.id == "complete-task-btn":
            # Complete first incomplete task as example
            if self.todo_list_api:
                tasks = self.todo_list_api.get_tasks()
                incomplete_tasks = [t for t in tasks if not t.get("completed", False)]
                if incomplete_tasks:
                    self.todo_list_api.complete_task(incomplete_tasks[0]["id"])
                    task_viewer.refresh_tasks()
                    self.notify("Task completed!")
                else:
                    self.notify("No incomplete tasks to complete")

        if event.button.id == "remove-task-btn":
            # Remove last task as example
            if self.todo_list_api:
                tasks = self.todo_list_api.get_tasks()
                if tasks:
                    self.todo_list_api.delete_task(tasks[-1]["id"])
                    task_viewer.refresh_tasks()
                    self.notify("Task removed!")
                else:
                    self.notify("No tasks to remove")

        if event.button.id == "confirm-add-task-btn":
            try:
                task_input = self.query_one("#new-task-input", Input)
                self.todo_list_api.add_task(task_input.value)
                self.switch_content("main")

                task_viewer.refresh_tasks()
                self.notify("Task added!")
                self.logger.log(f"Added new task: {task_input.value}", "INFO", "tui_todo", "TodoList")
            except Exception as e:
                self.logger.log(f"Error adding task: {e}", "ERROR", "tui_todo", "TodoList")

        if event.button.id == "cancel-add-task-btn":
            self.switch_content("main")
            
