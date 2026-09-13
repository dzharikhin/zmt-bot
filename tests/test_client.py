import asyncio
from concurrent.futures.process import BrokenProcessPool
from types import SimpleNamespace

import persistqueue
from telethon.errors import BotMethodInvalidError
from telethon.tl.types import DocumentAttributeAudio

import bot_utils
import client


class FakeQueue:
    def __init__(self, cmd):
        self._ready = [cmd] if cmd is not None else []
        self.calls = {"nack": 0, "ack": 0, "ack_failed": 0}

    def get_nowait(self):
        if self._ready:
            return self._ready.pop(0)
        raise persistqueue.exceptions.Empty()

    def nack(self, cmd):
        self.calls["nack"] += 1
        self._ready.append(cmd)
        return 1

    def ack(self, cmd):
        self.calls["ack"] += 1

    def ack_failed(self, cmd):
        self.calls["ack_failed"] += 1


class FakeBot:
    def __init__(self, connected=True):
        self.connected = connected
        self.messages = []

    def is_connected(self):
        return self.connected

    async def send_message(self, user_id, text, **kwargs):
        self.messages.append((user_id, text))


def run_handler_for(handler, seconds):
    async def run():
        task = asyncio.create_task(handler)
        await asyncio.sleep(seconds)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(run())


def patch_train_queue(monkeypatch, tmp_path, queue):
    queue_path = tmp_path / "train-queue.db"
    queue_path.touch()
    monkeypatch.setattr(
        client.config, "get_train_queue_path", lambda user_id: queue_path
    )
    monkeypatch.setattr(client, "get_or_create_train_queue", lambda user_id: queue)


def patch_estimate_queue(monkeypatch, tmp_path, queue):
    queue_path = tmp_path / "estimate-queue.db"
    queue_path.touch()
    monkeypatch.setattr(
        client.config, "get_estimate_queue_path", lambda user_id: queue_path
    )
    monkeypatch.setattr(client, "get_or_create_estimate_queue", lambda user_id: queue)


def test_wait_for_connectivity_waits_for_reconnect():
    bot = FakeBot(connected=False)

    async def reconnect_later():
        await asyncio.sleep(0.02)
        bot.connected = True

    async def scenario():
        await asyncio.gather(
            client.wait_for_connectivity(
                bot, poll_interval=0.005, settle_interval=0.005
            ),
            reconnect_later(),
        )

    asyncio.run(scenario())
    assert bot.is_connected()


def test_wait_for_connectivity_returns_fast_when_connected():
    bot = FakeBot(connected=True)
    asyncio.run(client.wait_for_connectivity(bot, floor_interval=0.005))
    assert bot.is_connected()


def test_handle_non_recoverable_survives_notification_failure():
    queue = FakeQueue({"message_id": 1})
    cmd = queue.get_nowait()

    class BrokenNotifyBot:
        async def send_message(self, *args, **kwargs):
            raise ConnectionError("Cannot send requests while disconnected")

    asyncio.run(
        client.handle_non_recoverable(
            BrokenNotifyBot(),
            cmd,
            RuntimeError("boom"),
            queue,
            42,
            "cannot train model",
        )
    )
    assert queue.calls["ack_failed"] == 1


def test_train_queue_retries_connection_error(monkeypatch, tmp_path):
    cmd = {
        "message_id": 7178,
        "forced": False,
        "limit": None,
        "latest_message_links": [],
    }
    queue = FakeQueue(cmd)
    patch_train_queue(monkeypatch, tmp_path, queue)
    bot = FakeBot()

    calls = {"prepare": 0, "wait": 0}

    async def failing_prepare_model(*args, **kwargs):
        calls["prepare"] += 1
        raise ConnectionError("Connection to Telegram failed 5 time(s)")

    async def fake_wait(bot_client):
        calls["wait"] += 1
        await asyncio.sleep(0)

    monkeypatch.setattr(client, "prepare_model", failing_prepare_model)
    monkeypatch.setattr(client, "wait_for_connectivity", fake_wait)

    run_handler_for(
        client.handle_train_queue_tasks(42, bot),
        seconds=0.1,
    )

    assert calls["prepare"] >= 1
    assert calls["wait"] == calls["prepare"]
    assert queue.calls["nack"] == calls["prepare"]
    assert queue.calls["ack_failed"] == 0
    assert queue.calls["ack"] == 0
    assert bot.messages == []


def test_train_queue_resets_broken_training_executor(monkeypatch, tmp_path):
    cmd = {
        "message_id": 7178,
        "forced": False,
        "limit": None,
        "latest_message_links": [],
    }
    queue = FakeQueue(cmd)
    patch_train_queue(monkeypatch, tmp_path, queue)

    resets = {"training": 0, "estimation": 0}
    monkeypatch.setattr(
        client.config,
        "reset_training_executor",
        lambda: resets.update(training=resets["training"] + 1),
    )
    monkeypatch.setattr(
        client.config,
        "reset_estimation_executor",
        lambda: resets.update(estimation=resets["estimation"] + 1),
    )

    async def failing_prepare_model(*args, **kwargs):
        raise BrokenProcessPool()

    async def fake_wait(bot_client):
        await asyncio.sleep(0)

    monkeypatch.setattr(client, "prepare_model", failing_prepare_model)
    monkeypatch.setattr(client, "wait_for_connectivity", fake_wait)

    run_handler_for(
        client.handle_train_queue_tasks(42, FakeBot()),
        seconds=0.1,
    )

    assert resets["training"] >= 1
    assert resets["estimation"] == 0
    assert queue.calls["ack_failed"] == 0


def test_train_queue_marks_non_transient_failure(monkeypatch, tmp_path):
    cmd = {
        "message_id": 7178,
        "forced": False,
        "limit": None,
        "latest_message_links": [],
    }
    queue = FakeQueue(cmd)
    patch_train_queue(monkeypatch, tmp_path, queue)
    bot = FakeBot()

    async def failing_prepare_model(*args, **kwargs):
        raise ValueError("no channels initialized")

    async def fake_wait(bot_client):
        raise AssertionError("must not wait on non-transient failure")

    monkeypatch.setattr(client, "prepare_model", failing_prepare_model)
    monkeypatch.setattr(client, "wait_for_connectivity", fake_wait)

    run_handler_for(
        client.handle_train_queue_tasks(42, bot),
        seconds=0.2,
    )

    assert queue.calls["ack_failed"] == 1
    assert queue.calls["nack"] == 0
    assert len(bot.messages) == 1
    assert "Failed to execute" in bot.messages[0][1]


def test_train_queue_sends_success_notification(monkeypatch, tmp_path):
    cmd = {
        "message_id": 7178,
        "forced": False,
        "limit": None,
        "latest_message_links": [],
    }
    queue = FakeQueue(cmd)
    patch_train_queue(monkeypatch, tmp_path, queue)
    bot = FakeBot()

    async def fake_prepare_model(*args, **kwargs):
        return None

    fake_model = client.config.Model(
        model_id=7178,
        pickle_file_path=tmp_path / "model.pkl",
        liked_tracks_count=1376,
        disliked_tracks_count=603,
        metrics_source="cross_validated",
        outliers_removed_liked=12,
        outliers_removed_disliked=5,
        include_liked_tp=0.8,
        include_liked_tn=0.85,
        include_liked_fp=0.15,
        include_liked_fn=0.2,
        exclude_disliked_tp=0.9,
        exclude_disliked_tn=0.32,
        exclude_disliked_fp=0.68,
        exclude_disliked_fn=0.1,
    )

    monkeypatch.setattr(client, "prepare_model", fake_prepare_model)
    monkeypatch.setattr(
        client.config, "get_model", lambda user_id, model_id: fake_model
    )

    run_handler_for(
        client.handle_train_queue_tasks(42, bot),
        seconds=0.2,
    )

    assert queue.calls["ack"] == 1
    assert queue.calls["ack_failed"] == 0
    assert len(bot.messages) == 1
    text = bot.messages[0][1]
    assert text.startswith("Successfully trained model 7178: cross_validated")
    assert "liked tracks: 1376 (outliers removed: 12)" in text
    assert "disliked tracks: 603 (outliers removed: 5)" in text
    assert "include_liked: tp=0.80 tn=0.85 fp=0.15 fn=0.20" in text
    assert "exclude_disliked: tp=0.90 tn=0.32 fp=0.68 fn=0.10" in text


def test_estimate_queue_retries_connection_error(monkeypatch, tmp_path):
    cmd = {
        "chat_id": -1002439736204,
        "message_id": 9610,
        "model_id": 6177,
        "model_type": "INCLUDE_LIKED",
        "channel_name": "estimation",
    }
    queue = FakeQueue(cmd)
    patch_estimate_queue(monkeypatch, tmp_path, queue)
    bot = FakeBot()

    calls = {"estimate": 0, "wait": 0}

    async def failing_estimate(*args, **kwargs):
        calls["estimate"] += 1
        raise ConnectionError("Connection to Telegram failed 5 time(s)")

    async def fake_wait(bot_client):
        calls["wait"] += 1
        await asyncio.sleep(0)

    monkeypatch.setattr(client, "estimate", failing_estimate)
    monkeypatch.setattr(client, "wait_for_connectivity", fake_wait)

    run_handler_for(
        client.handle_estimate_queue_tasks(42, bot),
        seconds=0.1,
    )

    assert calls["estimate"] >= 1
    assert calls["wait"] == calls["estimate"]
    assert queue.calls["nack"] == calls["estimate"]
    assert queue.calls["ack_failed"] == 0
    assert bot.messages == []


def test_estimate_queue_marks_bot_method_invalid_as_failed(monkeypatch, tmp_path):
    cmd = {
        "chat_id": -1002439736204,
        "message_id": 9610,
        "model_id": 6177,
        "model_type": "INCLUDE_LIKED",
        "channel_name": "estimation",
    }
    queue = FakeQueue(cmd)
    patch_estimate_queue(monkeypatch, tmp_path, queue)
    bot = FakeBot()

    async def failing_estimate(*args, **kwargs):
        raise BotMethodInvalidError(None)

    async def fake_wait(bot_client):
        raise AssertionError("must not wait on non-transient failure")

    monkeypatch.setattr(client, "estimate", failing_estimate)
    monkeypatch.setattr(client, "wait_for_connectivity", fake_wait)

    run_handler_for(
        client.handle_estimate_queue_tasks(42, bot),
        seconds=0.2,
    )

    assert queue.calls["ack_failed"] == 1
    assert queue.calls["nack"] == 0
    assert len(bot.messages) == 1


def make_audio_attr(performer=None, title=None, duration=120):
    return DocumentAttributeAudio(duration=duration, title=title, performer=performer)


def make_estimate_message(text="new track", attributes=(), channel_id=2439736204):
    return SimpleNamespace(
        message=text,
        forward=None,
        id=555,
        input_chat=SimpleNamespace(channel_id=channel_id),
        media=SimpleNamespace(document=SimpleNamespace(attributes=list(attributes))),
    )


def patch_not_recommended(monkeypatch, message):
    async def fake_estimate(*args, **kwargs):
        return False

    async def fake_get_message(chat_id, msg_id, bot_client):
        return message

    monkeypatch.setattr(client, "estimate", fake_estimate)
    monkeypatch.setattr(client, "get_message", fake_get_message)


def test_get_track_label_with_both_tags():
    message = make_estimate_message(
        attributes=[make_audio_attr(performer="Boards of Canada", title="Roygbiv")]
    )
    assert bot_utils.get_track_label(message) == "Boards of Canada - Roygbiv"


def test_get_track_label_partial_tags():
    message = make_estimate_message(attributes=[make_audio_attr(title="OnlyTitle")])
    assert bot_utils.get_track_label(message) == "OnlyTitle"
    message = make_estimate_message(attributes=[make_audio_attr(performer="OnlyAct")])
    assert bot_utils.get_track_label(message) == "OnlyAct"


def test_get_track_label_missing_or_empty():
    assert bot_utils.get_track_label(make_estimate_message()) is None
    message = make_estimate_message(
        attributes=[make_audio_attr(title="", performer=" ")]
    )
    assert bot_utils.get_track_label(message) is None
    assert bot_utils.get_track_label(None) is None
    assert bot_utils.get_track_label(SimpleNamespace(media=None)) is None


def _estimate_cmd():
    return {
        "chat_id": -1002439736204,
        "message_id": 555,
        "model_id": 6177,
        "model_type": "INCLUDE_LIKED",
        "channel_name": "estimation",
    }


def test_estimate_queue_not_recommended_includes_track_label(monkeypatch, tmp_path):
    queue = FakeQueue(_estimate_cmd())
    patch_estimate_queue(monkeypatch, tmp_path, queue)
    bot = FakeBot()
    message = make_estimate_message(
        attributes=[make_audio_attr(performer="Artist", title="Track")]
    )
    patch_not_recommended(monkeypatch, message)

    run_handler_for(client.handle_estimate_queue_tasks(42, bot), seconds=0.2)

    assert bot.messages == [
        (
            42,
            "[estimation] Rated as not recommended: Artist - Track\n"
            "https://t.me/c/2439736204/555",
        )
    ]
    assert queue.calls["ack"] == 1


def test_estimate_queue_not_recommended_link_caption_with_label(monkeypatch, tmp_path):
    queue = FakeQueue(_estimate_cmd())
    patch_estimate_queue(monkeypatch, tmp_path, queue)
    bot = FakeBot()
    message = make_estimate_message(
        text="https://t.me/some_channel/12",
        attributes=[make_audio_attr(performer="Artist", title="Track")],
    )
    patch_not_recommended(monkeypatch, message)

    run_handler_for(client.handle_estimate_queue_tasks(42, bot), seconds=0.2)

    assert bot.messages == [
        (
            42,
            "[estimation] Rated as not recommended: Artist - Track\n"
            "https://t.me/some_channel/12",
        )
    ]


def test_estimate_queue_not_recommended_without_label_unchanged(monkeypatch, tmp_path):
    queue = FakeQueue(_estimate_cmd())
    patch_estimate_queue(monkeypatch, tmp_path, queue)
    bot = FakeBot()
    message = make_estimate_message()
    patch_not_recommended(monkeypatch, message)

    run_handler_for(client.handle_estimate_queue_tasks(42, bot), seconds=0.2)

    assert bot.messages == [
        (42, "[estimation] Rated as not recommended: https://t.me/c/2439736204/555")
    ]
