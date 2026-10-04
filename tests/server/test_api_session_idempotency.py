# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0


async def test_message_and_commit_http_retry_contract(client):
    created = await client.post(
        "/api/v1/sessions",
        json={"memory_policy": {"memory_types": [], "working_memory": {"enabled": False}}},
    )
    assert created.status_code == 200, created.text
    base = "/api/v1/sessions/" + created.json()["result"]["session_id"]
    body = {
        "messages": [
            {"role": "user", "content": "hello", "source_message_ids": ["one"]},
            {"role": "assistant", "content": "world", "source_message_ids": ["two"]},
        ]
    }
    first = await client.post(base + "/messages/batch", json=body)
    assert first.status_code == 200, first.text
    replay = await client.post(base + "/messages/batch", json=body)
    assert replay.status_code == 200, replay.text
    assert first.json()["result"]["added"] == 2
    assert replay.json()["result"]["added"] == 0
    assert first.json()["result"]["message_ids"] == replay.json()["result"]["message_ids"]

    conflict = await client.post(
        base + "/messages/batch",
        json={
            "messages": [
                {"role": "user", "content": "must not append"},
                {"role": "user", "content": "changed", "source_message_ids": ["one"]},
            ]
        },
    )
    assert conflict.status_code == 409, conflict.text
    commit = await client.post(base + "/commit", json={"idempotency_key": "request-one"})
    assert commit.status_code == 200, commit.text
    receipt = commit.json()["result"]
    assert receipt["archived"]
    after_archive = await client.post(base + "/messages/batch", json=body)
    assert after_archive.status_code == 200, after_archive.text
    assert after_archive.json()["result"]["added"] == 0
    assert after_archive.json()["result"]["message_ids"] == first.json()["result"]["message_ids"]

    normal = await client.post(base + "/messages", json={"role": "user", "content": "new"})
    assert normal.status_code == 200, normal.text
    retried = await client.post(base + "/commit", json={"idempotency_key": "request-one"})
    assert retried.json()["result"] == receipt
    changed = await client.post(
        base + "/commit", json={"idempotency_key": "request-one", "keep_recent_count": 1}
    )
    assert changed.status_code == 409, changed.text
    status = await client.get(base + "/commit-status", params={"idempotency_key": "request-one"})
    assert status.status_code == 200, status.text
    assert status.json()["result"]["receipt"] == receipt
    assert status.json()["result"]["archive_state"] in {"pending", "completed"}
    missing = await client.get(base + "/commit-status", params={"idempotency_key": "unknown"})
    assert missing.status_code == 404, missing.text
    current = await client.get(base)
    assert current.json()["result"]["message_count"] == 1
