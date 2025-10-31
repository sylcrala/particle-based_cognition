"""
Particle-based Cognition Engine - todolist API module
Copyright (C) 2025 sylcrala

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version, subject to the additional terms 
specified in TERMS.md.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License and TERMS.md for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

Additional terms apply per TERMS.md. See also ETHICS.md.
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
    
api.register_api("todo_list", TodoList())