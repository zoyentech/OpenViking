# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

"""Retry contracts against real Session code, a durable store and actual mutexes."""

import asyncio
import json
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.message import Message, TextPart, ToolPart
from openviking.server.identity import RequestContext, Role
from openviking.session.session import Session
from openviking_cli.exceptions import ConflictError, NotFoundError
from openviking_cli.session.user_id import UserIdentifier


class ProcessCrash(BaseException):
    """Bypass exception cleanup to model losing a worker at a persistence boundary."""


class Store:
    def __init__(self):
        self.files = {}
        self.locks = defaultdict(asyncio.Lock)
        self.reads = []
        self.listings = 0
        self.fault = None
        self.fault_error = ProcessCrash
        self._async_agfs = SimpleNamespace(
            pathlock_acquire_exact=self.acquire, pathlock_release=self.release
        )

    def _uri_to_path(self, uri, ctx=None):
        return f"{ctx.account_id}:{uri}"

    async def acquire(self, path, **kwargs):
        await self.locks[path].acquire()
        return path

    async def release(self, lease):
        self.locks[lease].release()

    def crash(self, operation, uri, content, after):
        if self.fault and self.fault(operation, uri, content, after):
            self.fault = None
            raise self.fault_error(uri)

    async def read_file(self, uri, ctx=None):
        self.reads.append(uri)
        path = self._uri_to_path(uri, ctx)
        if path not in self.files:
            raise FileNotFoundError(uri)
        return self.files[path]

    async def write_file(self, uri, content, ctx=None, lease_ref=None):
        self.crash("write", uri, content, False)
        self.files[self._uri_to_path(uri, ctx)] = content
        self.crash("write", uri, content, True)

    async def append_file(self, uri, content, ctx=None):
        self.crash("append", uri, content, False)
        self.files[self._uri_to_path(uri, ctx)] += content
        self.crash("append", uri, content, True)

    async def exists(self, uri, ctx=None):
        path = self._uri_to_path(uri, ctx)
        return path in self.files or any(key.startswith(path + "/") for key in self.files)

    async def stat(self, uri, ctx=None, **kwargs):
        if not await self.exists(uri, ctx):
            raise FileNotFoundError(uri)
        return {}

    async def mkdir(self, uri, ctx=None, **kwargs):
        pass

    async def ls(self, uri, ctx=None):
        self.listings += 1
        prefix = self._uri_to_path(uri, ctx) + "/"
        return [
            {"name": name}
            for name in sorted(
                {
                    path[len(prefix) :].split("/")[0]
                    for path in self.files
                    if path.startswith(prefix)
                }
            )
        ]


@pytest.fixture
def env(monkeypatch):
    store = Store()
    jobs = []

    async def enqueue(queue, message):
        store.crash("enqueue", "queue", message, False)
        jobs.append(message)
        store.crash("enqueue", "queue", message, True)

    tracker = SimpleNamespace(
        create=AsyncMock(),
        fail=AsyncMock(),
        has_work=lambda task_id: any(job["task_id"] == task_id for job in jobs),
    )
    monkeypatch.setattr("openviking.session.session._enabled_memory_types", lambda: set())
    monkeypatch.setattr("openviking.service.task_tracker.get_task_tracker", lambda: tracker)
    monkeypatch.setattr(
        "openviking.storage.queuefs.get_queue_manager", lambda: SimpleNamespace(enqueue=enqueue)
    )

    def session(user="alice", account="account-a", api_key=None):
        ctx = RequestContext(
            user=UserIdentifier(account_id=account, user_id=user),
            role=Role.USER,
            api_key=api_key,
        )
        return Session(viking_fs=store, session_id="retry", ctx=ctx)

    return store, jobs, session


def spec(text="hello", source="source-1", **kwargs):
    return {
        "role": "user",
        "parts": [TextPart(text)],
        "source_message_ids": [source] if source else None,
        **kwargs,
    }


@pytest.mark.asyncio
async def test_concurrent_append_and_mixed_unkeyed_messages(env):
    store, _, new = env
    first, second = new(), new()
    results = await asyncio.gather(
        first.add_messages_async([spec(), spec(source=None)]),
        second.add_messages_async([spec(), spec(source=None)]),
    )
    assert results[0][0].id == results[1][0].id
    assert results[0][1].id != results[1][1].id
    reloaded = new()
    await reloaded.load()
    assert len(reloaded.messages) == 3
    assert results[0].added + results[1].added == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        {"parts": [TextPart("changed")]},
        {"role": "assistant"},
        {"peer_id": "other"},
        {"turn_id": "turn"},
        {"message_kind": "tool_transport"},
        {"created_at": "2026-01-01T00:00:00Z"},
        {"source_message_ids": ["source-1", "extra"]},
    ],
)
async def test_payload_conflict_rejects_entire_batch(env, change):
    _, _, new = env
    session = new()
    original = await session.add_messages_async([spec()])
    with pytest.raises(ConflictError):
        await new().add_messages_async([spec("must not be appended", "new"), spec(**change)])
    check = new()
    await check.load()
    assert [message.id for message in check.messages] == [original[0].id]


@pytest.mark.asyncio
async def test_archived_retry_uses_index_and_survives_index_rebuild(env):
    store, _, new = env
    session = new()
    original = await session.add_messages_async([spec()])
    first = await session.commit_async()
    await session.add_messages_async([spec("unrelated", "unrelated")])
    second = await session.commit_async()
    store.reads.clear()
    listings = store.listings
    replay = await new().add_messages_async([spec()])
    assert [message.id for message in replay] == [original[0].id]
    assert f"{second['archive_uri']}/messages.jsonl" not in store.reads
    assert f"{first['archive_uri']}/messages.jsonl" in store.reads
    assert store.listings == listings
    del store.files[store._uri_to_path(session._source_index.uri, session.ctx)]
    assert (await new().add_messages_async([spec()]))[0].id == original[0].id
    assert store.listings == listings + 1


@pytest.mark.asyncio
async def test_split_tool_group_deduplicates_across_partial_retention(env):
    _, _, new = env
    payload = spec(
        parts=[
            ToolPart(tool_id="one", tool_name="read", tool_output="one"),
            ToolPart(tool_id="two", tool_name="read", tool_output="two"),
        ]
    )
    session = new()
    original = await session.add_messages_async([payload])
    assert len(original) == 2
    await session.commit_async(keep_recent_count=1)
    retry = new()
    replay = await retry.add_messages_async([payload])
    assert [message.id for message in replay] == [message.id for message in original]
    assert replay.added == 0
    assert len(retry.messages) == 1


@pytest.mark.asyncio
async def test_append_ack_lost_before_meta_write(env):
    store, _, new = env
    store.fault = lambda op, uri, content, after: uri.endswith("/.meta.json") and not after
    with pytest.raises(ProcessCrash):
        await new().add_messages_async([spec()])
    replay = new()
    result = await replay.add_messages_async([spec()])
    assert result.added == 0
    assert len(replay.messages) == 1
    assert replay.meta.pending_tokens == replay.messages[0].estimated_tokens


@pytest.mark.asyncio
async def test_concurrent_commit_receipts_survive_later_messages_and_key_rotation(env):
    _, jobs, new = env
    await new().add_messages_async([spec()])
    first, second = await asyncio.gather(
        new(api_key="old").commit_async(idempotency_key="commit-1"),
        new(api_key="new").commit_async(idempotency_key="commit-1"),
    )
    assert first == second
    assert len(jobs) == 1
    later = await new().add_messages_async([spec("later", None)])
    assert await new().commit_async(idempotency_key="commit-1") == first
    check = new()
    await check.load()
    assert [message.id for message in check.messages] == [later[0].id]
    with pytest.raises(ConflictError):
        await new().commit_async(idempotency_key="commit-1", keep_recent_count=1)
    with pytest.raises(NotFoundError):
        await new(user="bob").get_commit_status("commit-1")
    with pytest.raises(NotFoundError):
        await new(account="other").get_commit_status("commit-1")


@pytest.mark.asyncio
@pytest.mark.parametrize("keep", [0, 100])
async def test_skipped_commit_is_also_idempotent(env, keep):
    store, jobs, new = env
    session = new()
    await store.write_file(f"{session.uri}/messages.jsonl", "", ctx=session.ctx)
    if keep:
        await session.add_messages_async([spec()])
    first = await session.commit_async(idempotency_key="skip", keep_recent_count=keep)
    assert first["status"] == "skipped"
    await new().add_messages_async([spec("later", "later")])
    assert await new().commit_async(idempotency_key="skip", keep_recent_count=keep) == first
    assert not jobs


def crash_point(stage, after):
    def match(op, uri, content, when):
        if when != after:
            return False
        if stage == "enqueue":
            return op == "enqueue"
        if stage in ("reserve", "finish") and uri.endswith("/.commit-receipts.json"):
            return next(iter(json.loads(content).values()))["finished"] == (stage == "finish")
        if stage in ("intent", "ready") and "/history/" in uri and uri.endswith("/.meta.json"):
            return json.loads(content)["phase1"]["status"] == (
                "ready" if stage == "ready" else "preparing"
            )
        if stage == "raw":
            return "/history/" in uri and uri.endswith("/messages.jsonl")
        if stage == "index":
            return uri.endswith("/.source-message-index.json")
        if stage == "root":
            return "/history/" not in uri and uri.endswith("/messages.jsonl")
        if stage == "meta":
            return "/history/" not in uri and uri.endswith("/.meta.json")
        if stage == "reset":
            return uri.endswith("/.done") and json.loads(content).get("context_reset")
        return False

    return match


@pytest.mark.asyncio
@pytest.mark.parametrize("after", [False, True])
@pytest.mark.parametrize(
    "stage",
    ["reserve", "intent", "raw", "enqueue", "index", "root", "meta", "ready", "reset", "finish"],
)
async def test_commit_process_crash_never_recuts_or_loses_messages(env, stage, after):
    store, jobs, new = env
    session = new()
    original = await session.add_messages_async([spec(), spec("ordinary", None)])
    store.fault = crash_point(stage, after)
    with pytest.raises(ProcessCrash):
        await session.commit_async(idempotency_key="crash", reset_context=True)
    retry = new()
    recovered = await retry.commit_async(idempotency_key="crash", reset_context=True)
    assert recovered["status"] in {"accepted", "failed"}
    assert len(jobs) <= 1
    assert await new().commit_async(idempotency_key="crash", reset_context=True) == recovered
    state = await new().get_commit_status("crash")
    if recovered["status"] == "failed":
        assert state["archive_state"] == "failed"
    else:
        assert recovered["reset_archive_uri"].endswith("archive_002")
    # Union of durable raw files always contains all original IDs, including
    # ordinary messages that cannot be reconstructed by source-ID replay.
    durable_ids = {
        json.loads(line)["id"]
        for path, content in store.files.items()
        if path.endswith("/messages.jsonl")
        for line in content.splitlines()
        if line.strip()
    }
    assert {message.id for message in original} <= durable_ids
    later = await new().add_messages_async([spec("new normal message", None)])
    assert await new().commit_async(idempotency_key="crash", reset_context=True) == recovered
    check = new()
    await check.load()
    assert later[0].id in {message.id for message in check.messages}


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["root", "meta", "ready", "reset", "finish"])
async def test_storage_response_lost_after_apply_recovers_success(env, stage):
    store, jobs, new = env
    await new().add_messages_async([spec()])
    store.fault_error = OSError
    store.fault = crash_point(stage, True)
    with pytest.raises(OSError):
        await new().commit_async(idempotency_key="lost-storage-response", reset_context=True)
    result = await new().commit_async(idempotency_key="lost-storage-response", reset_context=True)
    assert result["status"] == "accepted"
    assert result["archived"]
    assert len(jobs) == 1
    assert result["archive_uri"] == jobs[0]["archive_uri"]


@pytest.mark.asyncio
async def test_status_returns_summary_and_durable_receipt_without_task_tracker(env):
    store, _, new = env
    session = new()
    await session.add_messages_async([spec()])
    receipt = await session.commit_async(idempotency_key="status")
    assert not (await new().get_commit_status("status"))["summary_ready"]
    await store.write_file(f"{receipt['archive_uri']}/.done", "{}", ctx=session.ctx)
    await store.write_file(
        f"{receipt['archive_uri']}/.overview.md", "# Context\nremember this", ctx=session.ctx
    )
    state = await new().get_commit_status("status")
    assert state["receipt"] == receipt
    assert state["archive_state"] == "completed"
    assert state["summary_ready"]
    assert "remember this" in state["overview"]


@pytest.mark.asyncio
async def test_corrupt_or_unreadable_index_cannot_accept_duplicate(env):
    store, _, new = env
    session = new()
    await session.add_messages_async([spec()])
    await session.commit_async()
    await store.write_file(session._source_index.uri, "broken json", ctx=session.ctx)
    with pytest.raises(ValueError):
        await new().add_messages_async([spec()])
    check = new()
    await check.load()
    assert not check.messages


@pytest.mark.asyncio
async def test_checkpoint_sources_remain_cumulative_provenance(env):
    _, _, new = env
    session = new()
    await session.add_messages_async([spec(source="a1")])
    await session.add_messages_async(
        [
            spec("checkpoint one", message_kind="checkpoint", source_message_ids=["a1"]),
            spec("checkpoint two", message_kind="checkpoint", source_message_ids=["a1", "a2"]),
        ]
    )
    await session.commit_async()
    added = await session.add_messages_async([spec("a2", "a2")])
    assert added.added == 1
    assert (await session.add_messages_async([spec(source="a1")])).added == 0


@pytest.mark.asyncio
async def test_in_memory_externalization_and_per_call_counts(monkeypatch):
    session = Session(viking_fs=None)
    externalize = AsyncMock()
    monkeypatch.setattr(session._tool_outputs, "externalize_group", externalize)
    first = await session.add_messages_async([spec()])
    replay = await session.add_messages_async([spec()])
    assert first.added == 1 and replay.added == 0
    assert first[0].id == replay[0].id
    externalize.assert_awaited_once()


@pytest.mark.asyncio
async def test_same_session_concurrent_counts_are_receipt_local(env):
    _, _, new = env
    session = new()
    results = await asyncio.gather(*[session.add_messages_async([spec()]) for _ in range(8)])
    assert sorted(result.added for result in results) == [0] * 7 + [1]
    assert len({result[0].id for result in results}) == 1


@pytest.mark.asyncio
async def test_legacy_source_id_reuse_fails_closed(env):
    store, _, new = env
    session = new()
    legacy = Message(
        id="legacy", role="user", parts=[TextPart("hello")], source_message_ids=["source-1"]
    )
    await store.write_file(
        f"{session.uri}/messages.jsonl", legacy.to_jsonl() + "\n", ctx=session.ctx
    )
    with pytest.raises(ConflictError):
        await session.add_messages_async([spec()])


@pytest.mark.asyncio
async def test_real_router_and_service_http_contract(env, monkeypatch):
    import httpx
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    from openviking.server.auth import get_session_request_context
    from openviking.server.models import ERROR_CODE_TO_HTTP_STATUS
    from openviking.server.routers import sessions
    from openviking.service.session_service import SessionService
    from openviking_cli.exceptions import OpenVikingError

    store, _, new = env
    service = SessionService()
    service._viking_fs = store
    monkeypatch.setattr(service, "get_agent_evolution_enabled", AsyncMock(return_value=False))
    monkeypatch.setattr(service, "_get_user_memory_policy", AsyncMock(return_value=None))
    monkeypatch.setattr(service, "maybe_schedule_auto_commit", AsyncMock())
    monkeypatch.setattr(sessions, "get_service", lambda: SimpleNamespace(sessions=service))
    app = FastAPI()
    app.include_router(sessions.router)
    app.dependency_overrides[get_session_request_context] = lambda: new().ctx

    @app.exception_handler(OpenVikingError)
    async def handle_error(request, exc):
        return JSONResponse(
            status_code=ERROR_CODE_TO_HTTP_STATUS[exc.code], content={"error": exc.code}
        )

    base = "/api/v1/sessions/retry"
    payload = {"messages": [{"role": "user", "content": "hello", "source_message_ids": ["one"]}]}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        first = await client.post(base + "/messages/batch", json=payload)
        assert first.status_code == 200, first.text
        replay = await client.post(base + "/messages/batch", json=payload)
        assert replay.status_code == 200, replay.text
        assert replay.json()["result"]["added"] == 0
        assert replay.json()["result"]["message_ids"] == first.json()["result"]["message_ids"]
        bad = await client.post(
            base + "/messages",
            json={"role": "user", "content": "changed", "source_message_ids": ["one"]},
        )
        assert bad.status_code == 409, bad.text
        commit = await client.post(base + "/commit", json={"idempotency_key": "key"})
        assert commit.status_code == 200, commit.text
        retried = await client.post(base + "/commit", json={"idempotency_key": "key"})
        assert retried.json()["result"] == commit.json()["result"]
        conflict = await client.post(
            base + "/commit", json={"idempotency_key": "key", "keep_recent_count": 1}
        )
        assert conflict.status_code == 409, conflict.text
        status = await client.get(base + "/commit-status", params={"idempotency_key": "key"})
        assert status.status_code == 200, status.text
        assert status.json()["result"]["receipt"] == commit.json()["result"]
        missing = await client.get(base + "/commit-status", params={"idempotency_key": "missing"})
        assert missing.status_code == 404, missing.text
