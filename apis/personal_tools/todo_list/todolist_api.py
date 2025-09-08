"""
todo list API
"""
import datetime as dt

from apis.api_registry import api

class TodoList():
    def __init__(self):
        self.tasks = []

    def add_task(self, task):
        entry = {
            "id": len(self.tasks) + 1,
            "task": task,
            "created_at": dt.datetime.now(),
            "completed": False
        }
        self.tasks.append(entry)

    def get_tasks(self):
        return self.tasks
    
    def get_incomplete_tasks(self):
        return [task for task in self.tasks if not task.get("completed", False)]
    
    def get_completed_tasks(self):
        return [task for task in self.tasks if task.get("completed", False)]

    def complete_task(self, task_id):
        for task in self.tasks:
            if task["id"] == task_id:
                task["completed"] = True
                return task
        return None

    def delete_task(self, task_id):
        for i, task in enumerate(self.tasks):
            if task["id"] == task_id:
                del self.tasks[i]
                return task
        return None
    
api.register_api("todo_list", TodoList(), permissions=["read", "write"], user_only=True)