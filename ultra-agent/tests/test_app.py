"""Unit tests for the Flask REST API using pytest and Flask's test client."""

import pytest


@pytest.fixture(autouse=True)
def reset_app_state() -> None:
    """Reset the app's in-memory state before each test."""
    import app

    # Reset global state
    app.todos.clear()
    app.current_id = 1

    yield


@pytest.fixture
def client() -> object:
    """Create a test client for the Flask app."""
    from app import app

    # Configure the app for testing
    app.config["TESTING"] = True

    with app.test_client() as client:
        yield client


def test_list_todos_empty(client) -> None:
    """GET /api/todos should return an empty list when no todos exist."""
    response = client.get("/api/todos")
    assert response.status_code == 200
    assert response.get_json() == []


def test_create_todo(client) -> None:
    """POST /api/todos should create a new todo and return 201."""
    response = client.post("/api/todos", json={"title": "Buy milk"})
    assert response.status_code == 201
    data = response.get_json()
    assert data is not None
    assert data["title"] == "Buy milk"
    assert "id" in data
    assert data["completed"] is False


def test_create_todo_missing_title(client) -> None:
    """POST /api/todos without a title should return 400."""
    response = client.post("/api/todos", json={})
    assert response.status_code == 400


def test_get_todo(client) -> None:
    """GET /api/todos/<id> should return the todo if found."""
    client.post("/api/todos", json={"title": "Read book"})
    response = client.get("/api/todos/1")
    assert response.status_code == 200
    data = response.get_json()
    assert data is not None
    assert data["title"] == "Read book"  # Note: title is stored as "title"
    assert data["id"] == 1


def test_get_todo_not_found(client) -> None:
    """GET /api/todos/<id> should return 404 when todo does not exist."""
    response = client.get("/api/todos/999")
    assert response.status_code == 404


def test_update_todo(client) -> None:
    """PUT /api/todos/<id> should update the todo's title."""
    client.post("/api/todos", json={"title": "Original"})
    response = client.put("/api/todos/1", json={"title": "Updated"})
    assert response.status_code == 200
    data = response.get_json()
    assert data is not None
    assert data["title"] == "Updated"


def test_update_todo_missing_data(client) -> None:
    """PUT /api/todos/<id> without data should return 400."""
    client.post("/api/todos", json={"title": "Someone"})
    response = client.put("/api/todos/1", json={})
    assert response.status_code == 400


def test_update_todo_completed(client) -> None:
    """PUT /api/todos/<id> should update the completed status."""
    client.post("/api/todos", json={"title": "Task"})
    response = client.put("/api/todos/1", json={"completed": True})
    assert response.status_code == 200
    data = response.get_json()
    assert data is not None
    assert data["completed"] is True


def test_delete_todo(client) -> None:
    """DELETE /api/todos/<id> should delete the todo and return 204."""
    client.post("/api/todos", json={"title": "ToBeDeleted"})
    response = client.delete("/api/todos/1")
    assert response.status_code == 204


def test_delete_todo_not_found(client) -> None:
    """DELETE /api/todos/<id> for a non-existent todo should return 404."""
    response = client.delete("/api/todos/999")
    assert response.status_code == 404


def test_list_todos_after_creation(client) -> None:
    """GET /api/todos should return the created todo(s)."""
    client.post("/api/todos", json={"title": "First"})
    client.post("/api/todos", json={"title": "Second"})
    response = client.get("/api/todos")
    assert response.status_code == 200
    data = response.get_json()
    assert len(data) == 2
    assert data[0]["title"] == "First"
    assert data[1]["title"] == "Second"


def test_search_todos(client) -> None:
    """GET /api/todos/search?q=<query> should return todos whose title contains the query string (case-insensitive)."""
    client.post("/api/todos", json={"title": "Alice"})
    client.post("/api/todos", json={"title": "Bob"})
    client.post("/api/todos", json={"title": "Alicia"})

    # Search with "ali" (case-insensitive, should match "Alice" and "Alicia")
    response = client.get("/api/todos/search?q=ali")
    assert response.status_code == 200
    data = response.get_json()
    assert len(data) == 2
    names = {u["title"] for u in data}
    assert names == {"Alice", "Alicia"}

    # Search with "bob" (lowercase, should match "Bob")
    response = client.get("/api/todos/search?q=bob")
    assert response.status_code == 200
    data = response.get_json()
    assert len(data) == 1


def test_bulk_delete_todos(client) -> None:
    """DELETE /api/todos/bulk should delete multiple todos by ID and return 200."""
    # Setup: Create 3 todos
    client.post("/api/todos", json={"title": "Todo 1"})
    client.post("/api/todos", json={"title": "Todo 2"})
    client.post("/api/todos", json={"title": "Todo 3"})

    # Verify initial state (3 todos)
    response = client.get("/api/todos")
    assert response.status_code == 200
    assert len(response.get_json()) == 3

    # Action: Bulk delete IDs 1 and 3
    ids_to_delete = [1, 3]
    response = client.delete("/api/todos/bulk", json={"ids": ids_to_delete})

    # Verification
    assert response.status_code == 200
    data = response.get_json()
    assert data["message"] == "Successfully deleted 2 todos."
    assert data["deleted_ids"] == ids_to_delete

    # Verify final state (1 todo remaining: Todo 2)
    response = client.get("/api/todos")
    assert response.status_code == 200
    data = response.get_json()
    assert len(data) == 1
    assert data[0]["title"] == "Todo 2"

    # Test bulk delete with non-existent IDs (should still work for existing ones)
    ids_to_delete_mixed = [1, 999]  # 1 is gone, 999 doesn't exist
    response = client.delete("/api/todos/bulk", json={"ids": ids_to_delete_mixed})
    assert response.status_code == 200
    data = response.get_json()
    assert data["message"] == "Successfully deleted 0 todos."
    assert data["deleted_ids"] == ids_to_delete_mixed

    # Verify final state (1 todo remaining: Todo 2)
    response = client.get("/api/todos")
    assert response.status_code == 200
    data = response.get_json()
    assert len(data) == 1
    assert data[0]["title"] == "Todo 2"


def test_bulk_update_todos(client) -> None:
    """POST /api/todos/bulk/update should update multiple todos by ID."""
    # Setup: Create 3 todos
    client.post("/api/todos", json={"title": "Task A"})
    client.post("/api/todos", json={"title": "Task B"})
    client.post("/api/todos", json={"title": "Task C"})

    # Initial state check (3 todos)
    response = client.get("/api/todos")
    assert response.status_code == 200
    assert len(response.get_json()) == 3

    # Action: Bulk update IDs 1 and 3 to completed=True
    ids_to_update = [1, 3]
    updates = {"completed": True}
    response = client.post("/api/todos/bulk/update", json={"ids": ids_to_update, "updates": updates})

    # Verification
    assert response.status_code == 200
    data = response.get_json()
    assert data["message"] == "Successfully updated 2 todos."
    assert data["updated_ids"] == ids_to_update

    # Verify state: Todo 1 and 3 are completed, Todo 2 is not
    response = client.get("/api/todos")
    assert response.status_code == 200
    todos = response.get_json()

    # Check specific todos
    todo1 = next(t for t in todos if t["id"] == 1)
    todo2 = next(t for t in todos if t["id"] == 2)
    todo3 = next(t for t in todos if t["id"] == 3)

    assert todo1["completed"] is True
    assert todo2["completed"] is False
    assert todo3["completed"] is True

    # Action: Bulk update ID 2 title
    ids_to_update_2 = [2]
    updates_2 = {"title": "Updated Task B"}
    response = client.post("/api/todos/bulk/update", json={"ids": ids_to_update_2, "updates": updates_2})
    assert response.status_code == 200

    # Verify state again
    response = client.get("/api/todos")
    todos = response.get_json()
    todo2_updated = next(t for t in todos if t["id"] == 2)
    assert todo2_updated["title"] == "Updated Task B"


def test_list_completed_todos(client) -> None:
    """GET /api/todos/completed should return only todos marked as completed."""
    # Setup: Create a todo and mark it as completed
    client.post("/api/todos", json={"title": "Completed Task"})  # ID 1
    response_update = client.put("/api/todos/1", json={"completed": True})
    assert response_update.status_code == 200

    # Setup: Create an incomplete todo
    client.post("/api/todos", json={"title": "Incomplete Task"})  # ID 2

    # Action: Get completed todos
    response = client.get("/api/todos/completed")

    # Verification
    assert response.status_code == 200
    data = response.get_json()
    assert len(data) == 1
    assert data[0]["title"] == "Completed Task"
    assert data[0]["completed"] is True


def test_list_incomplete_todos(client) -> None:
    """GET /api/todos/incomplete should return only todos marked as incomplete."""
    # Setup: Create a todo and mark it as completed
    client.post("/api/todos", json={"title": "Completed Task"})  # ID 1
    response_update = client.put("/api/todos/1", json={"completed": True})
    assert response_update.status_code == 200

    # Setup: Create an incomplete todo
    client.post("/api/todos", json={"title": "Incomplete Task"})  # ID 2

    # Action: Get incomplete todos
    response = client.get("/api/todos/incomplete")

    # Verification
    assert response.status_code == 200
    data = response.get_json()
    assert len(data) == 1
    assert data[0]["title"] == "Incomplete Task"


def test_todo_count(client) -> None:
    """GET /api/todos/count should return the total number of todos."""
    # Test with no todos
    response = client.get("/api/todos/count")
    assert response.status_code == 200
    assert response.get_json() == {"count": 0}

    # Setup: Create 3 todos
    client.post("/api/todos", json={"title": "T1"})
    client.post("/api/todos", json={"title": "T2"})
    client.post("/api/todos", json={"title": "T3"})

    # Test with 3 todos
    response = client.get("/api/todos/count")
    assert response.status_code == 200
    assert response.get_json() == {"count": 3}

    # Action: Delete one todo
    client.delete("/api/todos/1")

    # Test with 2 todos
    response = client.get("/api/todos/count")
    assert response.status_code == 200
    assert response.get_json() == {"count": 2}