# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Session-scoped request identities. The caller must hold the session path lock."""

import hashlib
import json
from typing import Any, Dict, List, Optional

from openviking.message import Message
from openviking.session.archive_store import is_storage_not_found
from openviking_cli.exceptions import ConflictError, InvalidArgumentError


class MessageWriteResult(list):
    """List-compatible write receipt; counts belong to this call, not the Session."""

    def __init__(self, messages: List[Message], added: int):
        super().__init__(messages)
        self.added = added


def delivery_source_ids(message: Message) -> List[str]:
    # Checkpoints use this field for cumulative provenance, not delivery IDs.
    return (message.source_message_ids or []) if message.message_kind != "checkpoint" else []


def fingerprint(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def validate_key(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise InvalidArgumentError(
            "Idempotency identifiers must be nonblank strings of at most 256 characters"
        )
    return value


def validate_source_ids(value: Optional[List[str]]) -> Optional[List[str]]:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) > 100:
        raise InvalidArgumentError("source_message_ids must be a list of at most 100 identifiers")
    for item in value:
        validate_key(item)
    if len(set(value)) != len(value):
        raise InvalidArgumentError("source_message_ids must not contain duplicates")
    return list(value)


def select_message_groups(
    groups: List[List[Message]], existing: List[Message]
) -> tuple[List[List[Message]], List[Message]]:
    """Validate the whole batch before accepting any group, including split tool results.

    Each source ID identifies one input payload (possibly several physical messages).
    Source sets may neither partially overlap nor be reassigned to another payload.
    Legacy source IDs without a verifiable fingerprint fail closed on reuse.
    """
    by_source: Dict[str, Dict[str, Message]] = {}
    for message in existing:
        for source_id in delivery_source_ids(message):
            by_source.setdefault(source_id, {})[message.id] = message
    new_groups = []
    result = []
    for group in groups:
        first = group[0]
        sources = set(delivery_source_ids(first))
        matches = {
            message.id: message
            for source_id in sources
            for message in by_source.get(source_id, {}).values()
        }
        if matches:
            identity = first.source_message_identity
            old = list(matches.values())
            if (
                any(set(message.source_message_ids or []) != sources for message in old)
                or any(
                    not message.source_message_identity
                    or message.source_message_identity.get("payload_hash")
                    != identity["payload_hash"]
                    for message in old
                )
                or len({message.source_message_identity.get("group_id") for message in old}) != 1
                or len(old) != identity["group_size"]
            ):
                raise ConflictError(
                    "source_message_ids already identify a different or unverifiable message payload"
                )
            result.extend(
                sorted(old, key=lambda message: message.source_message_identity["group_index"])
            )
        else:
            new_groups.append(group)
            result.extend(group)
            for message in group:
                for source_id in sources:
                    by_source.setdefault(source_id, {})[message.id] = message
    return new_groups, result


class CommitReceipts:
    """One durable receipt file per session, independent of expiring task records.

    Reserve before Phase 1 side effects, finish before releasing the same session
    lock. An interrupted reservation is retained: replay must never allocate a
    second archive. Archive/QueueFS state remains the authority for progress.
    No credentials or raw payloads are stored in this file.
    """

    def __init__(self, fs: Any, ctx: Any, session_uri: str):
        self.fs, self.ctx = fs, ctx
        self.uri = f"{session_uri}/.commit-receipts.json"

    async def read(self) -> Dict[str, Any]:
        try:
            content = await self.fs.read_file(self.uri, ctx=self.ctx)
        except Exception as exc:
            if is_storage_not_found(exc):
                return {}
            raise
        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError("Invalid session commit receipts")
        return data

    async def write(self, entries: Dict[str, Any]) -> None:
        await self.fs.write_file(self.uri, json.dumps(entries, ensure_ascii=False), ctx=self.ctx)

    async def get(self, key: str, request_hash: Optional[str] = None) -> Optional[Dict[str, Any]]:
        entry = (await self.read()).get(fingerprint(validate_key(key)))
        if entry is not None and request_hash is not None and entry["request_hash"] != request_hash:
            raise ConflictError("idempotency_key already identifies a different commit request")
        return entry

    async def save(
        self, key: str, request_hash: str, result: Dict[str, Any], *, finished: bool
    ) -> None:
        entries = await self.read()
        entries[fingerprint(validate_key(key))] = {
            "request_hash": request_hash,
            "result": result,
            "finished": finished,
        }
        await self.write(entries)


class SourceMessageIndex:
    """Rebuildable source-to-archive pointers; raw messages are authoritative.

    Index an archive before clearing its messages from live storage. This order
    makes a crash safe without a second message payload store or a transaction
    spanning two files. A missing index is bootstrapped once from archive raw.
    """

    def __init__(self, fs: Any, ctx: Any, session_uri: str, archives: Any):
        self.fs, self.ctx, self.archives = fs, ctx, archives
        self.uri = f"{session_uri}/.source-message-index.json"

    @staticmethod
    def add(data: Dict[str, Any], archive_uri: str, messages: List[Message]) -> None:
        for message in messages:
            for source_id in delivery_source_ids(message):
                refs = data["sources"].setdefault(source_id, [])
                if archive_uri not in refs:
                    refs.append(archive_uri)

    async def load(self, commit_count: int) -> Dict[str, Any]:
        try:
            data = json.loads(await self.fs.read_file(self.uri, ctx=self.ctx))
        except Exception as exc:
            if not is_storage_not_found(exc):
                raise
            data = {"through": -1, "sources": {}}
        if data["through"] < commit_count:
            for ref in await self.archives.list_refs(strict=True):
                if ref["index"] <= data["through"]:
                    continue
                try:
                    messages = await self.archives.read_messages(ref["archive_uri"])
                except Exception as exc:
                    if not is_storage_not_found(exc):
                        raise
                    if await self.archives.is_context_reset_archive(ref["archive_uri"]):
                        continue
                    # Missing historical raw is not evidence of an unused ID.
                    # Fail closed, including while rebuilding a missing index.
                    raise
                self.add(data, ref["archive_uri"], messages)
            data["through"] = max(0, commit_count)
            await self.save(data)
        return data

    async def save(self, data: Dict[str, Any]) -> None:
        await self.fs.write_file(self.uri, json.dumps(data, ensure_ascii=False), ctx=self.ctx)

    async def index_archive(
        self, archive_uri: str, messages: List[Message], commit_count: int
    ) -> None:
        data = await self.load(commit_count)
        self.add(data, archive_uri, messages)
        data["through"] = max(data["through"], self.archives.archive_index_from_uri(archive_uri))
        await self.save(data)

    async def lookup(self, source_ids: set[str], commit_count: int) -> List[Message]:
        data = await self.load(commit_count)
        refs = {uri for source_id in source_ids for uri in data["sources"].get(source_id, [])}
        result = []
        for uri in sorted(refs):
            result.extend(await self.archives.read_messages(uri))
        return result
