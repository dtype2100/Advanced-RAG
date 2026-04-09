"""Unit tests for the clarification policy."""

from __future__ import annotations

from app.rag.policies.clarification_policy import needs_clarification


def test_no_missing_slots_no_clarification():
    analysis = {"missing_slots": [], "is_ambiguous": False}
    assert needs_clarification(analysis) is False


def test_missing_time_slot_triggers_clarification():
    analysis = {"missing_slots": ["time_period"], "is_ambiguous": False}
    assert needs_clarification(analysis) is True


def test_missing_location_triggers_clarification():
    analysis = {"missing_slots": ["location"], "is_ambiguous": True}
    assert needs_clarification(analysis) is True
