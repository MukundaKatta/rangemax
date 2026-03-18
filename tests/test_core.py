"""Tests for Rangemax."""
from src.core import Rangemax
def test_init(): assert Rangemax().get_stats()["ops"] == 0
def test_op(): c = Rangemax(); c.process(x=1); assert c.get_stats()["ops"] == 1
def test_multi(): c = Rangemax(); [c.process() for _ in range(5)]; assert c.get_stats()["ops"] == 5
def test_reset(): c = Rangemax(); c.process(); c.reset(); assert c.get_stats()["ops"] == 0
def test_service_name(): c = Rangemax(); r = c.process(); assert r["service"] == "rangemax"
