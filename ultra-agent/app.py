"""Simple Flask REST API for managing todos with in-memory storage."""

from flask import Flask, request, jsonify, render_template

app = Flask(__name__)


@app.route("/")
def index() -> str:
    """Serve the frontend template."""
    return render_template("index.html")


# In-memory storage for todos
todos: list[dict] = []
current_id = 1


@app.route("/api/todos", methods=["GET"])
def list_todos() -> tuple:
    """Return a list of all todos."""
    return jsonify(todos), 200


@app.route("/api/todos/completed", methods=["GET"])
def list_completed_todos() -> tuple:
    """Return a list of only completed todos."""
    completed_todos = [t for t in todos if t["completed"]]
    return jsonify(completed_todos), 200


@app.route("/api/todos/incomplete", methods=["GET"])
def list_incomplete_todos() -> tuple:
    """Return a list of only incomplete todos."""
    incomplete_todos = [t for t in todos if not t["completed"]]
    return jsonify(incomplete_todos), 200


@app.route("/api/todos", methods=["POST"])
def create_todo() -> tuple:
    """Create a new todo from JSON payload."""
    global current_id
    data = request.get_json(silent=True)
    if not data or "title" not in data:
        return jsonify({"error": "'title' is required"}), 400

    todo = {"id": current_id, "title": data["title"], "completed": False}
    current_id += 1
    todos.append(todo)
    return jsonify(todo), 201


@app.route("/api/todos/<int:todo_id>", methods=["GET"])
def get_todo(todo_id: int) -> tuple:
    """Return a todo by id, or 404 if not found."""
    todo = next((t for t in todos if t["id"] == todo_id), None)
    if todo is None:
        return jsonify({"error": "Todo not found"}), 404
    return jsonify(todo), 200


@app.route("/api/todos/<int:todo_id>", methods=["PUT"])
def update_todo(todo_id: int) -> tuple:
    """Update a todo's title or completed status by id."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid data"}), 400

    todo = next((t for t in todos if t["id"] == todo_id), None)
    if todo is None:
        return jsonify({"error": "Todo not found"}), 404

    if "title" in data:
        todo["title"] = data["title"]
    if "completed" in data:
        todo["completed"] = data["completed"]

    return jsonify(todo), 200


@app.route("/api/todos/<int:todo_id>", methods=["DELETE"])
def delete_todo(todo_id: int) -> tuple:
    """Delete a todo by id, returning 204 on success."""
    global todos
    todo = next((t for t in todos if t["id"] == todo_id), None)
    if todo is None:
        return jsonify({"error": "Todo not found"}), 404

    todos = [t for t in todos if t["id"] != todo_id]
    return "", 204


@app.route("/api/todos/search", methods=["GET"])
def search_todos() -> tuple:
    """Search todos by query parameter q (case-insensitive) and/or completed status."""
    query = request.args.get("q", "", type=str)
    completed_str = request.args.get("completed", "", type=str)

    if not query and not completed_str:
        return jsonify([]), 200

    matched = todos

    if query:
        matched = [
            t for t in matched if query.lower() in t["title"].lower()
        ]

    if completed_str:
        completed_bool = completed_str.lower() in ("true", "1", "yes")
        matched = [t for t in matched if t["completed"] is completed_bool]

    return jsonify(matched), 200



@app.route("/api/todos/count", methods=["GET"])
def todo_count() -> tuple:
    """Return the total count of todos."""
    return jsonify({"count": len(todos)}), 200


@app.route("/api/todos/stats", methods=["GET"])
def todos_stats() -> tuple:
    """Return statistics about todos: total count, completed count, incomplete count."""
    total = len(todos)
    completed = sum(1 for t in todos if t["completed"])
    incomplete = total - completed
    return jsonify({
        "total": total,
        "completed": completed,
        "incomplete": incomplete,
    }), 200


@app.route("/api/todos/bulk", methods=["DELETE"])
def bulk_delete_todos() -> tuple:
    """Delete multiple todos by ID."""
    global todos
    data = request.get_json(silent=True)
    if not data or "ids" not in data or not isinstance(data["ids"], list):
        return jsonify({"error": "'ids' must be a list"}), 400

    ids_to_delete = data["ids"]
    original_count = len(todos)
    todos = [t for t in todos if t["id"] not in ids_to_delete]
    deleted_count = original_count - len(todos)

    return jsonify(
        {
            "message": f"Successfully deleted {deleted_count} todos.",
            "deleted_ids": ids_to_delete,
        }
    ), 200


@app.route("/api/todos/bulk/update", methods=["POST"])
def bulk_update_todos() -> tuple:
    """Update multiple todos by ID with a single payload."""
    global todos
    data = request.get_json(silent=True)
    if not data or "ids" not in data or "updates" not in data or not isinstance(data["ids"], list) or not isinstance(data["updates"], dict):
        return jsonify({"error": "Invalid payload. Requires 'ids' (list) and 'updates' (dict)."}), 400

    ids_to_update = data["ids"]
    updates = data["updates"]
    updated_count = 0

    for todo in todos:
        if todo["id"] in ids_to_update:
            updated = False
            for key, value in updates.items():
                if key in todo and todo[key] != value:
                    todo[key] = value
                    updated = True

            if updated:
                updated_count += 1

    return jsonify({
        "message": f"Successfully updated {updated_count} todos.",
        "updated_ids": [t["id"] for t in todos if t["id"] in ids_to_update],
    }), 200


if __name__ == "__main__":
    app.run(debug=True)