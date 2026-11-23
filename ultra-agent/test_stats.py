import sys
sys.path.insert(0, ".")

import app


def test_stats_empty():
    """Test stats endpoint with no todos."""
    # Reset global state
    app.todos.clear()
    app.current_id = 1

    client = app.app.test_client()
    response = client.get("/api/todos/stats")
    assert response.status_code == 200
    data = response.get_json()
    assert data == {"total": 0, "completed": 0, "incomplete": 0}


def test_stats_with_todos():
    """Test stats endpoint with todos."""
    # Reset global state
    app.todos.clear()
    app.current_id = 1

    client = app.app.test_client()

    # Create 3 todos, mark one as completed
    client.post("/api/todos", json={"title": "Task 1"})  # id=1, completed=False
    client.post("/api/todos", json={"title": "Task 2"})  # id=2, completed=False
    client.post("/api/todos", json={"title": "Task 3"})  # id=3, completed=False
    client.put("/api/todos/2", json={"completed": True})  # mark task 2 as completed

    response = client.get("/api/todos/stats")
    assert response.status_code == 200
    data = response.get_json()
    assert data == {"total": 3, "completed": 1, "incomplete": 2}


def test_stats_all_completed():
    """Test stats when all todos are completed."""
    app.todos.clear()
    app.current_id = 1

    client = app.app.test_client()
    client.post("/api/todos", json={"title": "Task 1"})
    client.post("/api/todos", json={"title": "Task 2"})
    client.put("/api/todos/1", json={"completed": True})
    client.put("/api/todos/2", json={"completed": True})

    response = client.get("/api/todos/stats")
    assert response.status_code == 200
    data = response.get_json()
    assert data == {"total": 2, "completed": 2, "incomplete": 0}, f"Got {data}"


if __name__ == "__main__":
    test_stats_empty()
    print("PASS: test_stats_empty")
    test_stats_with_todos()
    print("PASS: test_stats_with_todos")
    test_stats_all_completed()
    print("PASS: test_stats_all_completed")
    print("All tests passed!")