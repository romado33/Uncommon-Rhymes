import logging

from rhyme_rarity.utils.telemetry import StructuredTelemetry, TelemetryLogger


def test_structured_telemetry_emits_logging_events(caplog):
    telemetry = StructuredTelemetry()
    listener = TelemetryLogger()
    telemetry.add_listener(listener)

    caplog.set_level(logging.INFO, logger="rhyme_rarity.utils.telemetry")

    telemetry.start_trace("test-trace")
    with telemetry.timer("phase"):
        pass
    telemetry.increment("search.completed")
    telemetry.annotate("result.total", 3)

    messages = [record.message for record in caplog.records]
    assert any("Telemetry trace_started: test-trace" in message for message in messages)
    assert any("Telemetry timer_started: phase" in message for message in messages)
    assert any("Telemetry timing: phase" in message for message in messages)
    assert any("Telemetry counter: search.completed" in message for message in messages)
    assert any("Telemetry metadata: result.total" in message for message in messages)
