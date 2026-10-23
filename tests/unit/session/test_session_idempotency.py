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
from openviking_cli.exceptions import ConflictError
from openviking_cli.session.user_id import UserIdentifier


class ProcessCrash(BaseException):
    """Bypass exception cleanup to model losing a worker at a persistence boundary."""


class Store:
    def __init__(self):
        self.files = {}
        self.locks = defaultdict(asyncio.Lock)
        self.reads = []
        self.writes = []
        self.listings = 0
        self.fault = None
        self.fault_error = ProcessCrash
        self.yield_message_reads = False
        self._async_agfs = SimpleNamespace(
            pathlock_acquire_exact=self.acquire,
            pathlock_acquire_exact_batch=self.acquire_batch,
            pathlock_release=self.release,
        )

    def _uri_to_path(self, uri, ctx=None):
        return f"{ctx.account_id}:{uri}"

    async def acquire(self, path, **kwargs):
        await self.locks[path].acquire()
        return path

    async def acquire_batch(self, paths, **kwargs):
        leases = []
        for path in sorted(paths):
            leases.append(await self.acquire(path))
        return leases

    async def release(self, lease):
        leases = lease if isinstance(lease, list) else [lease]
        for path in reversed(leases):
            self.locks[path].release()

    def crash(self, operation, uri, content, after):
        if self.fault and self.fault(operation, uri, content, after):
            self.fault = None
            raise self.fault_error(uri)

    async def read_file(self, uri, ctx=None):
        self.reads.append(uri)
        path = self._uri_to_path(uri, ctx)
        if path not in self.files:
            raise FileNotFoundError(uri)
        content = self.files[path]
        if self.yield_message_reads and uri.endswith("/messages.jsonl"):
            await asyncio.sleep(0)
        return content

    async def write_file(self, uri, content, ctx=None, lease_ref=None):
        if isinstance(content, bytes):
            content = content.decode("utf-8")
        self.writes.append(uri)
        self.crash("write", uri, content, False)
        self.files[self._uri_to_path(uri, ctx)] = content
        self.crash("write", uri, content, True)

    async def append_file(self, uri, content, ctx=None):
        self.writes.append(uri)
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

    async def has_work(task_id, **kwargs):
        return any(job["task_id"] == task_id for job in jobs)

    tracker = SimpleNamespace(
        create=AsyncMock(),
        get=AsyncMock(return_value=None),
        fail=AsyncMock(),
        has_work=has_work,
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
async def test_message_source_ids_remain_idempotent_across_archives(env):
    store, _, new = env
    session = new()
    original = await session.add_messages_async([spec()])
    first = await session.commit_async()
    store.reads.clear()
    listings = store.listings
    replay = await new().add_messages_async([spec()])
    assert replay.added == 0
    assert replay[0].id == original[0].id
    assert f"{first['archive_uri']}/messages.jsonl" in store.reads

    latest = await session.add_messages_async([spec("latest", "latest")])
    second = await session.commit_async()
    store.reads.clear()
    replay_after_archive = await new().add_messages_async([spec()])
    assert replay_after_archive.added == 0
    assert replay_after_archive[0].id == original[0].id
    assert f"{first['archive_uri']}/messages.jsonl" in store.reads
    assert f"{second['archive_uri']}/messages.jsonl" in store.reads
    assert (await new().add_messages_async([spec("latest", "latest")]))[0].id == latest[0].id
    assert store.listings > listings
    assert not any(uri.endswith("/.source-message-index.json") for uri in store.writes)
    assert not any(uri.endswith("/.source-message-index.json") for uri in store.reads)


@pytest.mark.asyncio
async def test_archive_lookup_fails_closed_when_history_listing_is_unavailable(env, monkeypatch):
    store, _, new = env
    session = new()
    original = await session.add_messages_async([spec()])
    await session.commit_async()

    async def fail_listing(uri, ctx=None):
        raise OSError("history listing unavailable")

    monkeypatch.setattr(store, "ls", fail_listing)
    with pytest.raises(OSError, match="history listing unavailable"):
        await new().add_messages_async([spec()])

    check = new()
    await check.load()
    assert not check.messages
    assert (
        original[0].id
        in store.files[
            store._uri_to_path(f"{session.uri}/history/archive_001/messages.jsonl", session.ctx)
        ]
    )


@pytest.mark.asyncio
async def test_archive_lookup_fails_closed_on_malformed_history_entry(env):
    store, _, new = env
    session = new()
    await session.add_messages_async([spec()])
    await session.commit_async()
    await store.write_file(
        f"{session.uri}/history/archive_broken/messages.jsonl",
        "{}\n",
        ctx=session.ctx,
    )

    with pytest.raises(ValueError, match="invalid session archive entry"):
        await new().add_messages_async([spec()])

    check = new()
    await check.load()
    assert not check.messages


@pytest.mark.asyncio
async def test_retry_fails_closed_when_newer_archive_is_corrupt(env):
    store, _, new = env
    session = new()
    original = await session.add_messages_async([spec()])
    await session.commit_async()
    await session.add_messages_async([spec("newer", "newer")])
    newest = await session.commit_async()
    await store.write_file(
        f"{newest['archive_uri']}/messages.jsonl", "broken json", ctx=session.ctx
    )

    with pytest.raises(ValueError, match="invalid message record"):
        await new().add_messages_async([spec()])

    check = new()
    await check.load()
    assert not check.messages
    older_archive_uri = f"{session.uri}/history/archive_001/messages.jsonl"
    assert original[0].id in store.files[store._uri_to_path(older_archive_uri, session.ctx)]


@pytest.mark.asyncio
async def test_source_ids_stay_unique_after_multiple_archives_and_concurrent_retries(env):
    _, _, new = env
    original = (await new().add_messages_async([spec()]))[0]

    for index in range(1, 4):
        result = await new().commit_async()
        assert result["archived"]
        if index < 3:
            await new().add_messages_async([spec(f"later-{index}", f"later-{index}")])

    first, second = await asyncio.gather(
        new().add_messages_async([spec()]),
        new().add_messages_async([spec()]),
    )

    assert first.added + second.added == 0
    assert first[0].id == second[0].id == original.id
    with pytest.raises(ConflictError):
        await new().add_messages_async([spec("changed", "source-1")])


@pytest.mark.asyncio
async def test_lost_append_response_remains_idempotent_after_archive_rollover(env):
    store, _, new = env
    session = new()
    root_uri = f"{session.uri}/messages.jsonl"

    def lose_response(operation, uri, content, after):
        return operation == "write" and uri == root_uri and after

    store.fault = lose_response
    with pytest.raises(ProcessCrash):
        await new().add_messages_async([spec()])

    persisted = new()
    await persisted.load()
    original_id = persisted.messages[0].id
    for index in range(1, 4):
        assert (await new().commit_async())["archived"]
        if index < 3:
            await new().add_messages_async([spec(f"later-{index}", f"later-{index}")])

    replay = await new().add_messages_async([spec()])
    assert replay.added == 0
    assert replay[0].id == original_id


@pytest.mark.asyncio
async def test_source_ids_remain_idempotent_across_context_reset(env):
    store, _, new = env
    session = new()
    old = await session.add_messages_async([spec("old", "old")])
    oldest = await session.commit_async()
    latest = await session.add_messages_async([spec()])
    reset = await session.commit_async(idempotency_key="reset", reset_context=True)
    store.reads.clear()
    replay = await new().add_messages_async([spec()])
    assert replay.added == 0
    assert replay[0].id == latest[0].id
    raw_reads = [
        uri for uri in store.reads if "/history/" in uri and uri.endswith("/messages.jsonl")
    ]
    assert f"{reset['archive_uri']}/messages.jsonl" in raw_reads
    assert f"{oldest['archive_uri']}/messages.jsonl" in store.reads
    old_replay = await new().add_messages_async([spec("old", "old")])
    assert old_replay.added == 0
    assert old_replay[0].id == old[0].id
    assert not any(uri.endswith("/.source-message-index.json") for uri in store.writes)


@pytest.mark.asyncio
async def test_retry_scans_prior_archives_past_reset_boundaries(env):
    store, _, new = env
    session = new()
    original = await session.add_messages_async([spec()])
    first = await session.commit_async()
    # Consecutive empty boundaries can exist in imported history. The read
    # bound must hold even when neither of the two candidates contains raw.
    for index in (2, 3):
        await store.write_file(
            f"{session.uri}/history/archive_{index:03d}/.done",
            json.dumps({"context_reset": True}),
            ctx=session.ctx,
        )
    meta_uri = f"{session.uri}/.meta.json"
    meta = json.loads(await store.read_file(meta_uri, ctx=session.ctx))
    meta["commit_count"] = 3
    await store.write_file(meta_uri, json.dumps(meta), ctx=session.ctx)
    store.reads.clear()
    listings = store.listings
    replay = await new().add_messages_async([spec()])
    assert replay.added == 0
    assert replay[0].id == original[0].id
    raw_reads = [
        uri for uri in store.reads if "/history/" in uri and uri.endswith("/messages.jsonl")
    ]
    assert raw_reads == [f"{first['archive_uri']}/messages.jsonl"]
    assert store.listings > listings


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
    assert await new(user="bob")._commit_receipts.get("commit-1") is None
    assert await new(account="other")._commit_receipts.get("commit-1") is None


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


@pytest.mark.asyncio
async def test_commit_receipts_keep_eight_and_expired_key_starts_new_attempt(env):
    _, jobs, new = env
    receipts = []
    for index in range(9):
        await new().add_messages_async([spec(f"message {index}", None)])
        receipts.append(await new().commit_async(idempotency_key=f"commit-{index}"))
        if index == 7:
            # Reading/replaying the oldest retained key does not refresh its
            # first-write order before the ninth reservation is created.
            assert await new().commit_async(idempotency_key="commit-0") == receipts[0]
    assert len(await new()._commit_receipts.read()) == 8
    assert await new()._commit_receipts.get("commit-0") is None
    for index in range(1, 9):
        assert (await new()._commit_receipts.get(f"commit-{index}"))["result"] == receipts[index]

    await new().add_messages_async([spec("new content after key expired", None)])
    restarted = await new().commit_async(idempotency_key="commit-0")
    assert restarted["status"] == "accepted"
    assert restarted["task_id"] != receipts[0]["task_id"]
    assert restarted["archive_uri"] != receipts[0]["archive_uri"]
    assert len(jobs) == 10
    assert len(await new()._commit_receipts.read()) == 8
    assert await new()._commit_receipts.get("commit-1") is None


@pytest.mark.asyncio
async def test_receipt_capacity_never_evicts_an_unfinished_reservation(env):
    store, _, new = env
    session = new()
    receipts = session._commit_receipts
    lease = await store.acquire(store._uri_to_path(session.uri, session.ctx))
    try:
        await receipts.save("unfinished", "pending-hash", {"status": "accepted"}, finished=False)
        for index in range(7):
            await receipts.save(
                f"finished-{index}", "done-hash", {"status": "skipped"}, finished=True
            )
        await receipts.save("newest", "new-hash", {"status": "accepted"}, finished=False)
        entries = await receipts.read()
        assert len(entries) == 8
        assert not (await receipts.get("unfinished"))["finished"]
        assert not (await receipts.get("newest"))["finished"]
        assert await receipts.get("finished-0") is None
        assert await receipts.get("finished-1") is not None
    finally:
        await store.release(lease)


@pytest.mark.asyncio
async def test_new_key_resolves_old_reservation_under_lock_before_capacity_eviction(
    env, monkeypatch
):
    store, jobs, new = env
    for index in range(7):
        await new().add_messages_async([spec(f"finished {index}", None)])
        await new().commit_async(idempotency_key=f"finished-{index}")
    await new().add_messages_async([spec("pending publication", None)])
    store.fault = crash_point("ready", True)
    with pytest.raises(ProcessCrash):
        await new().commit_async(idempotency_key="unfinished")

    session = new()
    assert len(await session._commit_receipts.read()) == 8
    assert not (await session._commit_receipts.get("unfinished"))["finished"]
    original_save = session._commit_receipts.save
    observed = []

    async def save_after_recovery(key, request_hash, result, *, finished):
        if key == "newest" and not finished:
            previous = await session._commit_receipts.get("unfinished")
            assert previous["finished"]
            assert previous["result"]["status"] == "accepted"
            assert store.locks[store._uri_to_path(session.uri, session.ctx)].locked()
            observed.append(True)
        await original_save(key, request_hash, result, finished=finished)

    monkeypatch.setattr(session._commit_receipts, "save", save_after_recovery)
    newest = await session.commit_async(idempotency_key="newest")
    assert newest["status"] == "skipped"
    assert observed == [True]
    assert len(jobs) == 8
    assert len(await session._commit_receipts.read()) == 8
    assert (await session._commit_receipts.get("unfinished"))["result"]["status"] == "accepted"
    assert await session._commit_receipts.get("finished-0") is None


def crash_point(stage, after):
    def match(op, uri, content, when):
        if when != after:
            return False
        if stage == "enqueue":
            return op == "enqueue"
        if stage in ("reserve", "finish") and uri.endswith("/.commit-receipts.json"):
            return list(json.loads(content).values())[-1]["finished"] == (stage == "finish")
        if stage in ("intent", "ready") and "/history/" in uri and uri.endswith("/.meta.json"):
            return json.loads(content)["phase1"]["status"] == (
                "ready" if stage == "ready" else "preparing"
            )
        if stage == "raw":
            return "/history/" in uri and uri.endswith("/messages.jsonl")
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
    ["reserve", "intent", "raw", "enqueue", "root", "meta", "ready", "reset", "finish"],
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
    if recovered["status"] == "failed":
        archive_uri = recovered["archive_uri"]
        assert await store.exists(f"{archive_uri}/.failed.json", ctx=session.ctx)
        assert not await store.exists(f"{archive_uri}/.done", ctx=session.ctx)
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
@pytest.mark.parametrize("failure", ["missing", "corrupt", "unreadable"])
async def test_latest_archive_read_failure_does_not_fall_back_or_accept_message(
    env, monkeypatch, failure
):
    store, _, new = env
    session = new()
    await session.add_messages_async([spec()])
    old = await session.commit_async()
    await session.add_messages_async([spec("latest", "latest")])
    latest = await session.commit_async()
    raw_uri = f"{latest['archive_uri']}/messages.jsonl"
    if failure == "missing":
        del store.files[store._uri_to_path(raw_uri, session.ctx)]
        expected = FileNotFoundError
    elif failure == "corrupt":
        await store.write_file(raw_uri, "broken json", ctx=session.ctx)
        expected = ValueError
    else:
        read_file = store.read_file

        async def unreadable(uri, ctx=None):
            if uri == raw_uri:
                raise OSError("archive unavailable")
            return await read_file(uri, ctx=ctx)

        monkeypatch.setattr(store, "read_file", unreadable)
        expected = OSError
    store.reads.clear()
    with pytest.raises(expected):
        await new().add_messages_async([spec()])
    assert f"{old['archive_uri']}/messages.jsonl" not in store.reads
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
    store, _, new = env
    session = new()
    await store.write_file(f"{session.uri}/messages.jsonl", "", ctx=session.ctx)
    store.yield_message_reads = True
    results = await asyncio.gather(*[new().add_messages_async([spec()]) for _ in range(8)])
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
        archived_replay = await client.post(base + "/messages/batch", json=payload)
        assert archived_replay.status_code == 200, archived_replay.text
        assert archived_replay.json()["result"]["added"] == 0
        assert (
            archived_replay.json()["result"]["message_ids"] == first.json()["result"]["message_ids"]
        )
        retried = await client.post(base + "/commit", json={"idempotency_key": "key"})
        assert retried.json()["result"] == commit.json()["result"]
        conflict = await client.post(
            base + "/commit", json={"idempotency_key": "key", "keep_recent_count": 1}
        )
        assert conflict.status_code == 409, conflict.text

        # Eight newer commit attempts evict the original key even when they
        # skipped an empty session; retry access does not extend its lifetime.
        for index in range(8):
            newer = await client.post(base + "/commit", json={"idempotency_key": f"newer-{index}"})
            assert newer.status_code == 200, newer.text
            assert newer.json()["result"]["status"] == "skipped"
        added = await client.post(
            base + "/messages", json={"role": "user", "content": "after expiry"}
        )
        assert added.status_code == 200, added.text
        restarted = await client.post(base + "/commit", json={"idempotency_key": "key"})
        assert restarted.status_code == 200, restarted.text
        assert restarted.json()["result"]["task_id"] != commit.json()["result"]["task_id"]
