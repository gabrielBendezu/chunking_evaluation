# === tests/test_parsing.py ===
from __future__ import annotations

import pendulum

from tests.conftest import assert_date
from tests.conftest import assert_datetime
from tests.conftest import assert_duration
from tests.conftest import assert_time


def test_parse() -> None:
    text = "2016-10-16T12:34:56.123456+01:30"

    dt = pendulum.parse(text)

    assert isinstance(dt, pendulum.DateTime)
    assert_datetime(dt, 2016, 10, 16, 12, 34, 56, 123456)
    assert dt.tz is not None
    assert dt.tz.name == "+01:30"
    assert dt.offset == 5400

    text = "2016-10-16"

    dt = pendulum.parse(text)

    assert isinstance(dt, pendulum.DateTime)
    assert_datetime(dt, 2016, 10, 16, 0, 0, 0, 0)
    assert dt.offset == 0

    with pendulum.travel_to(pendulum.datetime(2015, 11, 12), freeze=True):
        text = "12:34:56.123456"

        dt = pendulum.parse(text)

    assert isinstance(dt, pendulum.DateTime)
    assert_datetime(dt, 2015, 11, 12, 12, 34, 56, 123456)
    assert dt.offset == 0


def test_parse_with_timezone() -> None:
    text = "2016-10-16T12:34:56.123456"

    dt = pendulum.parse(text, tz="Europe/Paris")
    assert isinstance(dt, pendulum.DateTime)
    assert_datetime(dt, 2016, 10, 16, 12, 34, 56, 123456)
    assert dt.tz is not None
    assert dt.tz.name == "Europe/Paris"
    assert dt.offset == 7200


def test_parse_exact() -> None:
    text = "2016-10-16T12:34:56.123456+01:30"

    dt = pendulum.parse(text, exact=True)

    assert isinstance(dt, pendulum.DateTime)
    assert_datetime(dt, 2016, 10, 16, 12, 34, 56, 123456)
    assert dt.offset == 5400

    text = "2016-10-16"

    dt = pendulum.parse(text, exact=True)

    assert isinstance(dt, pendulum.Date)
    assert_date(dt, 2016, 10, 16)

    text = "12:34:56.123456"

    dt = pendulum.parse(text, exact=True)

    assert isinstance(dt, pendulum.Time)
    assert_time(dt, 12, 34, 56, 123456)

    text = "13:00"

    dt = pendulum.parse(text, exact=True)

    assert isinstance(dt, pendulum.Time)
    assert_time(dt, 13, 0, 0)


def test_parse_duration() -> None:
    text = "P2Y3M4DT5H6M7S"

    duration = pendulum.parse(text)

    assert isinstance(duration, pendulum.Duration)
    assert_duration(duration, 2, 3, 0, 4, 5, 6, 7)

    text = "P2W"

    duration = pendulum.parse(text)

    assert isinstance(duration, pendulum.Duration)
    assert_duration(duration, 0, 0, 2, 0, 0, 0, 0)


def test_parse_interval() -> None:
    text = "2008-05-11T15:30:00Z/P1Y2M10DT2H30M"

    interval = pendulum.parse(text)

    assert isinstance(interval, pendulum.Interval)
    assert_datetime(interval.start, 2008, 5, 11, 15, 30, 0, 0)
    assert interval.start.offset == 0
    assert_datetime(interval.end, 2009, 7, 21, 18, 0, 0, 0)
    assert interval.end.offset == 0

    text = "P1Y2M10DT2H30M/2008-05-11T15:30:00Z"

    interval = pendulum.parse(text)

    assert isinstance(interval, pendulum.Interval)
    assert_datetime(interval.start, 2007, 3, 1, 13, 0, 0, 0)
    assert interval.start.offset == 0
    assert_datetime(interval.end, 2008, 5, 11, 15, 30, 0, 0)
    assert interval.end.offset == 0

    text = "2007-03-01T13:00:00Z/2008-05-11T15:30:00Z"

    interval = pendulum.parse(text)

    assert isinstance(interval, pendulum.Interval)
    assert_datetime(interval.start, 2007, 3, 1, 13, 0, 0, 0)
    assert interval.start.offset == 0
    assert_datetime(interval.end, 2008, 5, 11, 15, 30, 0, 0)
    assert interval.end.offset == 0


def test_parse_now() -> None:
    assert pendulum.parse("now").timezone_name == "UTC"
    assert (
        pendulum.parse("now", tz="America/Los_Angeles").timezone_name
        == "America/Los_Angeles"
    )

    dt = pendulum.parse("now", tz="local")
    assert dt.timezone_name == "America/Toronto"

    mock_now = pendulum.yesterday()

    with pendulum.travel_to(mock_now, freeze=True):
        assert pendulum.parse("now") == mock_now


def test_parse_with_utc_timezone() -> None:
    dt = pendulum.parse("2020-02-05T20:05:37.364951Z")

    assert dt.to_iso8601_string() == "2020-02-05T20:05:37.364951Z"


# === tests/conftest.py ===
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import pendulum


if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def setup() -> Iterator[None]:
    pendulum.set_local_timezone(pendulum.timezone("America/Toronto"))

    yield

    pendulum.set_locale("en")
    pendulum.set_local_timezone()
    pendulum.week_starts_at(pendulum.WeekDay.MONDAY)
    pendulum.week_ends_at(pendulum.WeekDay.SUNDAY)


def assert_datetime(
    d: pendulum.DateTime,
    year: int,
    month: int,
    day: int,
    hour: int | None = None,
    minute: int | None = None,
    second: int | None = None,
    microsecond: int | None = None,
) -> None:
    assert year == d.year
    assert month == d.month
    assert day == d.day

    if hour is not None:
        assert hour == d.hour

    if minute is not None:
        assert minute == d.minute

    if second is not None:
        assert second == d.second

    if microsecond is not None:
        assert microsecond == d.microsecond


def assert_date(d: pendulum.Date, year: int, month: int, day: int) -> None:
    assert year == d.year
    assert month == d.month
    assert day == d.day


def assert_time(
    t: pendulum.Time,
    hour: int,
    minute: int,
    second: int,
    microsecond: int | None = None,
) -> None:
    assert hour == t.hour
    assert minute == t.minute
    assert second == t.second

    if microsecond is not None:
        assert microsecond == t.microsecond


def assert_duration(
    dur: pendulum.Duration,
    years: int | None = None,
    months: int | None = None,
    weeks: int | None = None,
    days: int | None = None,
    hours: int | None = None,
    minutes: int | None = None,
    seconds: int | None = None,
    microseconds: int | None = None,
) -> None:
    expected = {}
    actual = {}

    if years is not None:
        expected["years"] = dur.years
        actual["years"] = years

    if months is not None:
        expected["months"] = dur.months
        actual["months"] = months

    if weeks is not None:
        expected["weeks"] = dur.weeks
        actual["weeks"] = weeks

    if days is not None:
        expected["days"] = dur.remaining_days
        actual["days"] = days

    if hours is not None:
        expected["hours"] = dur.hours
        actual["hours"] = hours

    if minutes is not None:
        expected["minutes"] = dur.minutes
        actual["minutes"] = minutes

    if seconds is not None:
        expected["seconds"] = dur.remaining_seconds
        actual["seconds"] = seconds

    if microseconds is not None:
        expected["microseconds"] = dur.microseconds
        actual["microseconds"] = microseconds

    assert expected == actual


# === tests/__init__.py ===


# === tests/test_helpers.py ===
from __future__ import annotations

from datetime import datetime

import pytest
import pytz

import pendulum

from pendulum import timezone
from pendulum.helpers import PreciseDiff
from pendulum.helpers import days_in_year
from pendulum.helpers import precise_diff
from pendulum.helpers import week_day


def assert_diff(
    diff: PreciseDiff,
    years: int = 0,
    months: int = 0,
    days: int = 0,
    hours: int = 0,
    minutes: int = 0,
    seconds: int = 0,
    microseconds: int = 0,
) -> None:
    assert diff.years == years
    assert diff.months == months
    assert diff.days == days
    assert diff.hours == hours
    assert diff.minutes == minutes
    assert diff.seconds == seconds
    assert diff.microseconds == microseconds


def test_precise_diff() -> None:
    dt1 = datetime(2003, 3, 1, 0, 0, 0)
    dt2 = datetime(2003, 1, 31, 23, 59, 59)

    diff = precise_diff(dt1, dt2)
    assert_diff(diff, months=-1, seconds=-1)

    diff = precise_diff(dt2, dt1)
    assert_diff(diff, months=1, seconds=1)

    dt1 = datetime(2012, 3, 1, 0, 0, 0)
    dt2 = datetime(2012, 1, 31, 23, 59, 59)

    diff = precise_diff(dt1, dt2)
    assert_diff(diff, months=-1, seconds=-1)
    assert diff.total_days == -30

    diff = precise_diff(dt2, dt1)
    assert_diff(diff, months=1, seconds=1)

    dt1 = datetime(2001, 1, 1)
    dt2 = datetime(2003, 9, 17, 20, 54, 47, 282310)

    diff = precise_diff(dt1, dt2)
    assert_diff(
        diff,
        years=2,
        months=8,
        days=16,
        hours=20,
        minutes=54,
        seconds=47,
        microseconds=282310,
    )

    dt1 = datetime(2017, 2, 17, 16, 5, 45, 123456)
    dt2 = datetime(2018, 2, 17, 16, 5, 45, 123256)

    diff = precise_diff(dt1, dt2)
    assert_diff(
        diff, months=11, days=30, hours=23, minutes=59, seconds=59, microseconds=999800
    )

    # DST
    tz = timezone("America/Toronto")
    dt1 = tz.datetime(2017, 3, 7)
    dt2 = tz.datetime(2017, 3, 13)

    diff = precise_diff(dt1, dt2)
    assert_diff(diff, days=6, hours=0)


def test_precise_diff_timezone() -> None:
    paris = pendulum.timezone("Europe/Paris")
    toronto = pendulum.timezone("America/Toronto")

    dt1 = paris.datetime(2013, 3, 31, 1, 30)
    dt2 = paris.datetime(2013, 4, 1, 1, 30)

    diff = precise_diff(dt1, dt2)
    assert_diff(diff, days=1, hours=0)
    assert diff.total_days == 1

    dt2 = toronto.datetime(2013, 4, 1, 1, 30)

    diff = precise_diff(dt1, dt2)
    assert_diff(diff, days=1, hours=5)
    assert diff.total_days == 1

    # pytz
    paris_pytz = pytz.timezone("Europe/Paris")
    toronto_pytz = pytz.timezone("America/Toronto")

    dt1 = paris_pytz.localize(datetime(2013, 3, 31, 1, 30))
    dt2 = paris_pytz.localize(datetime(2013, 4, 1, 1, 30))

    diff = precise_diff(dt1, dt2)
    assert_diff(diff, days=1, hours=0)
    assert diff.total_days == 1

    dt2 = toronto_pytz.localize(datetime(2013, 4, 1, 1, 30))

    diff = precise_diff(dt1, dt2)
    assert_diff(diff, days=1, hours=5)

    # Issue238
    dt1 = timezone("UTC").datetime(2018, 6, 20, 1, 30)
    dt2 = timezone("Europe/Paris").datetime(2018, 6, 20, 3, 40)  # UTC+2
    diff = precise_diff(dt1, dt2)
    assert_diff(diff, minutes=10)
    assert diff.total_days == 0


def test_week_day() -> None:
    assert week_day(2017, 6, 2) == 5
    assert week_day(2017, 1, 1) == 7


def test_days_in_years() -> None:
    assert days_in_year(2017) == 365
    assert days_in_year(2016) == 366


def test_locale() -> None:
    dt = pendulum.datetime(2000, 11, 10, 12, 34, 56, 123456)
    pendulum.set_locale("fr")

    assert pendulum.get_locale() == "fr"

    assert dt.format("MMMM") == "novembre"
    assert dt.date().format("MMMM") == "novembre"


def test_set_locale_invalid() -> None:
    with pytest.raises(ValueError):
        pendulum.set_locale("invalid")


@pytest.mark.parametrize(
    "locale", ["DE", "pt-BR", "pt-br", "PT-br", "PT-BR", "pt_br", "PT_BR", "PT_BR"]
)
def test_set_locale_malformed_locale(locale: str) -> None:
    pendulum.set_locale(locale)

    pendulum.set_locale("en")


def test_week_starts_at() -> None:
    pendulum.week_starts_at(pendulum.SATURDAY)

    dt = pendulum.now().start_of("week")
    assert dt.day_of_week == pendulum.SATURDAY
    assert dt.date().day_of_week == pendulum.SATURDAY


def test_week_starts_at_invalid_value() -> None:
    with pytest.raises(ValueError):
        pendulum.week_starts_at(-1)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        pendulum.week_starts_at(11)  # type: ignore[arg-type]


def test_week_ends_at() -> None:
    pendulum.week_ends_at(pendulum.SATURDAY)

    dt = pendulum.now().end_of("week")
    assert dt.day_of_week == pendulum.SATURDAY
    assert dt.date().day_of_week == pendulum.SATURDAY


def test_week_ends_at_invalid_value() -> None:
    with pytest.raises(ValueError):
        pendulum.week_ends_at(-1)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        pendulum.week_ends_at(11)  # type: ignore[arg-type]


# === tests/test_main.py ===
from __future__ import annotations

from datetime import date
from datetime import datetime
from datetime import time

import pytz

from dateutil import tz

import pendulum

from pendulum import _safe_timezone
from pendulum import timezone
from pendulum.tz.timezone import Timezone


def test_instance_with_naive_datetime_defaults_to_utc() -> None:
    now = pendulum.instance(datetime.now())
    assert now.timezone_name == "UTC"


def test_instance_with_aware_datetime() -> None:
    now = pendulum.instance(datetime.now(timezone("Europe/Paris")))
    assert now.timezone_name == "Europe/Paris"


def test_instance_with_aware_datetime_pytz() -> None:
    now = pendulum.instance(datetime.now(pytz.timezone("Europe/Paris")))
    assert now.timezone_name == "Europe/Paris"


def test_instance_with_aware_datetime_any_tzinfo() -> None:
    dt = datetime(2016, 8, 7, 12, 34, 56, tzinfo=tz.gettz("Europe/Paris"))
    now = pendulum.instance(dt)
    assert now.timezone_name == "+02:00"


def test_instance_with_date() -> None:
    dt = pendulum.instance(date(2022, 12, 23))

    assert isinstance(dt, pendulum.Date)


def test_instance_with_naive_time() -> None:
    dt = pendulum.instance(time(12, 34, 56, 123456))

    assert isinstance(dt, pendulum.Time)


def test_instance_with_aware_time() -> None:
    dt = pendulum.instance(time(12, 34, 56, 123456, tzinfo=timezone("Europe/Paris")))

    assert isinstance(dt, pendulum.Time)
    assert isinstance(dt.tzinfo, Timezone)
    assert dt.tzinfo.name == "Europe/Paris"


def test_safe_timezone_with_tzinfo_objects() -> None:
    tz = _safe_timezone(pytz.timezone("Europe/Paris"))

    assert isinstance(tz, Timezone)
    assert tz.name == "Europe/Paris"


# === tests/datetime/test_construct.py ===
from __future__ import annotations

import os

from datetime import datetime

import pytest

import pendulum

from pendulum import DateTime
from pendulum import timezone
from pendulum.utils._compat import PYPY
from tests.conftest import assert_datetime


if not PYPY:
    import time_machine
else:
    time_machine = None


@pytest.fixture(autouse=True)
def _setup():
    yield

    if os.getenv("TZ"):
        del os.environ["TZ"]


def test_creates_an_instance_default_to_utcnow():
    now = pendulum.now("UTC")
    p = pendulum.datetime(
        now.year, now.month, now.day, now.hour, now.minute, now.second
    )
    assert now.timezone_name == p.timezone_name

    assert_datetime(p, now.year, now.month, now.day, now.hour, now.minute, now.second)


def test_setting_timezone():
    tz = "Australia/Brisbane"
    dtz = timezone(tz)
    dt = datetime.utcnow()
    offset = dtz.convert(dt).utcoffset().total_seconds() / 3600

    p = pendulum.datetime(dt.year, dt.month, dt.day, tz=dtz)
    assert p.timezone_name == tz
    assert p.offset_hours == int(offset)


def test_setting_timezone_with_string():
    tz = "Australia/Brisbane"
    dtz = timezone(tz)
    dt = datetime.utcnow()
    offset = dtz.convert(dt).utcoffset().total_seconds() / 3600

    p = pendulum.datetime(dt.year, dt.month, dt.day, tz=tz)
    assert p.timezone_name == tz
    assert p.offset_hours == int(offset)


def test_today():
    today = pendulum.today()
    assert isinstance(today, DateTime)


def test_tomorrow():
    now = pendulum.now().start_of("day")
    tomorrow = pendulum.tomorrow()
    assert isinstance(tomorrow, DateTime)
    assert now.diff(tomorrow).in_days() == 1


def test_yesterday():
    now = pendulum.now().start_of("day")
    yesterday = pendulum.yesterday()

    assert isinstance(yesterday, DateTime)
    assert now.diff(yesterday, False).in_days() == -1


def test_now():
    now = pendulum.now("America/Toronto")
    in_paris = pendulum.now("Europe/Paris")

    assert now.hour != in_paris.hour


if time_machine:

    @time_machine.travel("2016-03-27 00:30:00Z", tick=False)
    def test_now_dst_off():
        utc = pendulum.now("UTC")
        in_paris = pendulum.now("Europe/Paris")
        in_paris_from_utc = utc.in_tz("Europe/Paris")
        assert in_paris.hour == 1
        assert not in_paris.is_dst()
        assert in_paris.isoformat() == in_paris_from_utc.isoformat()

    @time_machine.travel("2016-03-27 01:30:00Z", tick=False)
    def test_now_dst_transitioning_on():
        utc = pendulum.now("UTC")
        in_paris = pendulum.now("Europe/Paris")
        in_paris_from_utc = utc.in_tz("Europe/Paris")
        assert in_paris.hour == 3
        assert in_paris.is_dst()
        assert in_paris.isoformat() == in_paris_from_utc.isoformat()

    @time_machine.travel("2016-10-30 00:30:00Z", tick=False)
    def test_now_dst_on():
        utc = pendulum.now("UTC")
        in_paris = pendulum.now("Europe/Paris")
        in_paris_from_utc = utc.in_tz("Europe/Paris")
        assert in_paris.hour == 2
        assert in_paris.is_dst()
        assert in_paris.isoformat() == in_paris_from_utc.isoformat()

    @time_machine.travel("2016-10-30 01:30:00Z", tick=False)
    def test_now_dst_transitioning_off():
        utc = pendulum.now("UTC")
        in_paris = pendulum.now("Europe/Paris")
        in_paris_from_utc = utc.in_tz("Europe/Paris")
        assert in_paris.hour == 2
        assert not in_paris.is_dst()
        assert in_paris.isoformat() == in_paris_from_utc.isoformat()


def test_now_with_fixed_offset():
    now = pendulum.now(6)

    assert now.timezone_name == "+06:00"


def test_create_with_no_transition_timezone():
    dt = pendulum.now("Etc/UTC")

    assert dt.timezone_name == "Etc/UTC"


def test_create_maintains_microseconds():
    d = pendulum.datetime(2016, 11, 12, 2, 9, 39, 594000, tz="America/Panama")
    assert_datetime(d, 2016, 11, 12, 2, 9, 39, 594000)

    d = pendulum.datetime(2316, 11, 12, 2, 9, 39, 857, tz="America/Panama")
    assert_datetime(d, 2316, 11, 12, 2, 9, 39, 857)


def test_second_inaccuracy_on_past_datetimes():
    dt = pendulum.datetime(1901, 12, 13, 0, 0, 0, 555555, tz="US/Central")

    assert_datetime(dt, 1901, 12, 13, 0, 0, 0, 555555)


def test_local():
    local = pendulum.local(2018, 2, 2, 12, 34, 56, 123456)

    assert_datetime(local, 2018, 2, 2, 12, 34, 56, 123456)
    assert local.timezone_name == "America/Toronto"


# === tests/datetime/test_start_end_of.py ===
from __future__ import annotations

import pytest

import pendulum

from tests.conftest import assert_datetime


def test_start_of_second():
    d = pendulum.now()
    new = d.start_of("second")
    assert isinstance(new, pendulum.DateTime)
    assert_datetime(new, d.year, d.month, d.day, d.hour, d.minute, d.second, 0)


def test_end_of_second():
    d = pendulum.now()
    new = d.end_of("second")
    assert isinstance(new, pendulum.DateTime)
    assert_datetime(new, d.year, d.month, d.day, d.hour, d.minute, d.second, 999999)


def test_start_of_minute():
    d = pendulum.now()
    new = d.start_of("minute")
    assert isinstance(new, pendulum.DateTime)
    assert_datetime(new, d.year, d.month, d.day, d.hour, d.minute, 0, 0)


def test_end_of_minute():
    d = pendulum.now()
    new = d.end_of("minute")
    assert isinstance(new, pendulum.DateTime)
    assert_datetime(new, d.year, d.month, d.day, d.hour, d.minute, 59, 999999)


def test_start_of_hour():
    d = pendulum.now()
    new = d.start_of("hour")
    assert isinstance(new, pendulum.DateTime)
    assert_datetime(new, d.year, d.month, d.day, d.hour, 0, 0, 0)


def test_end_of_hour():
    d = pendulum.now()
    new = d.end_of("hour")
    assert isinstance(new, pendulum.DateTime)
    assert_datetime(new, d.year, d.month, d.day, d.hour, 59, 59, 999999)


def test_start_of_day():
    d = pendulum.now()
    new = d.start_of("day")
    assert isinstance(new, pendulum.DateTime)
    assert_datetime(new, d.year, d.month, d.day, 0, 0, 0, 0)


def test_end_of_day():
    d = pendulum.now()
    new = d.end_of("day")
    assert isinstance(new, pendulum.DateTime)
    assert_datetime(new, d.year, d.month, d.day, 23, 59, 59, 999999)


def test_start_of_month_is_fluid():
    d = pendulum.now()
    assert isinstance(d.start_of("month"), pendulum.DateTime)


def test_start_of_month_from_now():
    d = pendulum.now()
    new = d.start_of("month")
    assert_datetime(new, d.year, d.month, 1, 0, 0, 0, 0)


def test_start_of_month_from_last_day():
    d = pendulum.datetime(2000, 1, 31, 2, 3, 4)
    new = d.start_of("month")
    assert_datetime(new, 2000, 1, 1, 0, 0, 0, 0)


def test_start_of_year_is_fluid():
    d = pendulum.now()
    new = d.start_of("year")
    assert isinstance(new, pendulum.DateTime)


def test_start_of_year_from_now():
    d = pendulum.now()
    new = d.start_of("year")
    assert_datetime(new, d.year, 1, 1, 0, 0, 0, 0)


def test_start_of_year_from_first_day():
    d = pendulum.datetime(2000, 1, 1, 1, 1, 1)
    new = d.start_of("year")
    assert_datetime(new, 2000, 1, 1, 0, 0, 0, 0)


def test_start_of_year_from_last_day():
    d = pendulum.datetime(2000, 12, 31, 23, 59, 59)
    new = d.start_of("year")
    assert_datetime(new, 2000, 1, 1, 0, 0, 0, 0)


def test_end_of_month_is_fluid():
    d = pendulum.now()
    assert isinstance(d.end_of("month"), pendulum.DateTime)


def test_end_of_month():
    d = pendulum.datetime(2000, 1, 1, 2, 3, 4).end_of("month")
    new = d.end_of("month")
    assert_datetime(new, 2000, 1, 31, 23, 59, 59)


def test_end_of_month_from_last_day():
    d = pendulum.datetime(2000, 1, 31, 2, 3, 4)
    new = d.end_of("month")
    assert_datetime(new, 2000, 1, 31, 23, 59, 59)


def test_end_of_year_is_fluid():
    d = pendulum.now()
    assert isinstance(d.end_of("year"), pendulum.DateTime)


def test_end_of_year_from_now():
    d = pendulum.now().end_of("year")
    new = d.end_of("year")
    assert_datetime(new, d.year, 12, 31, 23, 59, 59, 999999)


def test_end_of_year_from_first_day():
    d = pendulum.datetime(2000, 1, 1, 1, 1, 1)
    new = d.end_of("year")
    assert_datetime(new, 2000, 12, 31, 23, 59, 59, 999999)


def test_end_of_year_from_last_day():
    d = pendulum.datetime(2000, 12, 31, 23, 59, 59, 999999)
    new = d.end_of("year")
    assert_datetime(new, 2000, 12, 31, 23, 59, 59, 999999)


def test_start_of_decade_is_fluid():
    d = pendulum.now()
    assert isinstance(d.start_of("decade"), pendulum.DateTime)


def test_start_of_decade_from_now():
    d = pendulum.now()
    new = d.start_of("decade")
    assert_datetime(new, d.year - d.year % 10, 1, 1, 0, 0, 0, 0)


def test_start_of_decade_from_first_day():
    d = pendulum.datetime(2000, 1, 1, 1, 1, 1)
    new = d.start_of("decade")
    assert_datetime(new, 2000, 1, 1, 0, 0, 0, 0)


def test_start_of_decade_from_last_day():
    d = pendulum.datetime(2009, 12, 31, 23, 59, 59)
    new = d.start_of("decade")
    assert_datetime(new, 2000, 1, 1, 0, 0, 0, 0)


def test_end_of_decade_is_fluid():
    d = pendulum.now()
    assert isinstance(d.end_of("decade"), pendulum.DateTime)


def test_end_of_decade_from_now():
    d = pendulum.now()
    new = d.end_of("decade")
    assert_datetime(new, d.year - d.year % 10 + 9, 12, 31, 23, 59, 59, 999999)


def test_end_of_decade_from_first_day():
    d = pendulum.datetime(2000, 1, 1, 1, 1, 1)
    new = d.end_of("decade")
    assert_datetime(new, 2009, 12, 31, 23, 59, 59, 999999)


def test_end_of_decade_from_last_day():
    d = pendulum.datetime(2009, 12, 31, 23, 59, 59, 999999)
    new = d.end_of("decade")
    assert_datetime(new, 2009, 12, 31, 23, 59, 59, 999999)


def test_start_of_century_is_fluid():
    d = pendulum.now()
    assert isinstance(d.start_of("century"), pendulum.DateTime)


def test_start_of_century_from_now():
    d = pendulum.now()
    new = d.start_of("century")
    assert_datetime(new, d.year - d.year % 100 + 1, 1, 1, 0, 0, 0, 0)


def test_start_of_century_from_first_day():
    d = pendulum.datetime(2001, 1, 1, 1, 1, 1)
    new = d.start_of("century")
    assert_datetime(new, 2001, 1, 1, 0, 0, 0, 0)


def test_start_of_century_from_last_day():
    d = pendulum.datetime(2100, 12, 31, 23, 59, 59)
    new = d.start_of("century")
    assert_datetime(new, 2001, 1, 1, 0, 0, 0, 0)


def test_end_of_century_is_fluid():
    d = pendulum.now()
    assert isinstance(d.end_of("century"), pendulum.DateTime)


def test_end_of_century_from_now():
    now = pendulum.now()
    d = now.end_of("century")
    assert_datetime(d, now.year - now.year % 100 + 100, 12, 31, 23, 59, 59, 999999)


def test_end_of_century_from_first_day():
    d = pendulum.datetime(2001, 1, 1, 1, 1, 1)
    new = d.end_of("century")
    assert_datetime(new, 2100, 12, 31, 23, 59, 59, 999999)


def test_end_of_century_from_last_day():
    d = pendulum.datetime(2100, 12, 31, 23, 59, 59, 999999)
    new = d.end_of("century")
    assert_datetime(new, 2100, 12, 31, 23, 59, 59, 999999)


def test_average_is_fluid():
    d = pendulum.now().average()
    assert isinstance(d, pendulum.DateTime)


def test_average_from_same():
    d1 = pendulum.datetime(2000, 1, 31, 2, 3, 4)
    d2 = pendulum.datetime(2000, 1, 31, 2, 3, 4).average(d1)
    assert_datetime(d2, 2000, 1, 31, 2, 3, 4)


def test_average_from_greater():
    d1 = pendulum.datetime(2000, 1, 1, 1, 1, 1, tz="local")
    d2 = pendulum.datetime(2009, 12, 31, 23, 59, 59, tz="local").average(d1)
    assert_datetime(d2, 2004, 12, 31, 12, 30, 30)


def test_average_from_lower():
    d1 = pendulum.datetime(2009, 12, 31, 23, 59, 59, tz="local")
    d2 = pendulum.datetime(2000, 1, 1, 1, 1, 1, tz="local").average(d1)
    assert_datetime(d2, 2004, 12, 31, 12, 30, 30)


def start_of_with_invalid_unit():
    with pytest.raises(ValueError):
        pendulum.now().start_of("invalid")


def end_of_with_invalid_unit():
    with pytest.raises(ValueError):
        pendulum.now().end_of("invalid")


def test_start_of_with_transition():
    d = pendulum.datetime(2013, 10, 27, 23, 59, 59, tz="Europe/Paris")
    assert d.offset == 3600
    assert d.start_of("month").offset == 7200
    assert d.start_of("day").offset == 7200
    assert d.start_of("year").offset == 3600


def test_start_of_on_date_before_transition():
    d = pendulum.datetime(2013, 10, 27, 0, 59, 59, tz="UTC").in_timezone("Europe/Paris")
    assert d.offset == 7200
    assert d.start_of("minute").offset == 7200
    assert d.start_of("hour").offset == 7200
    assert d.start_of("day").offset == 7200
    assert d.start_of("month").offset == 7200
    assert d.start_of("year").offset == 3600


def test_start_of_on_date_after_transition():
    d = pendulum.datetime(2013, 10, 27, 1, 59, 59, tz="UTC").in_timezone("Europe/Paris")
    assert d.offset == 3600
    assert d.start_of("minute").offset == 3600
    assert d.start_of("hour").offset == 3600
    assert d.start_of("day").offset == 7200
    assert d.start_of("month").offset == 7200
    assert d.start_of("year").offset == 3600


def test_end_of_with_transition():
    d = pendulum.datetime(2013, 3, 31, tz="Europe/Paris")
    assert d.offset == 3600
    assert d.end_of("month").offset == 7200
    assert d.end_of("day").offset == 7200
    assert d.end_of("year").offset == 3600


def test_end_of_on_date_before_transition():
    d = pendulum.datetime(2013, 10, 27, 0, 0, 0, tz="UTC").in_timezone("Europe/Paris")
    assert d.offset == 7200
    assert d.end_of("minute").offset == 7200
    assert d.end_of("hour").offset == 7200
    assert d.end_of("day").offset == 3600
    assert d.end_of("month").offset == 3600
    assert d.end_of("year").offset == 3600


def test_end_of_on_date_after_transition():
    d = pendulum.datetime(2013, 10, 27, 1, 0, 0, tz="UTC").in_timezone("Europe/Paris")
    assert d.offset == 3600
    assert d.end_of("minute").offset == 3600
    assert d.end_of("hour").offset == 3600
    assert d.end_of("day").offset == 3600
    assert d.end_of("month").offset == 3600
    assert d.end_of("year").offset == 3600


# === tests/datetime/test_getters.py ===
from __future__ import annotations

import struct

import pytest

import pendulum

from pendulum import DateTime
from pendulum import timezone
from tests.conftest import assert_date
from tests.conftest import assert_time


def test_year():
    d = pendulum.datetime(1234, 5, 6, 7, 8, 9)
    assert d.year == 1234


def test_month():
    d = pendulum.datetime(1234, 5, 6, 7, 8, 9)
    assert d.month == 5


def test_day():
    d = pendulum.datetime(1234, 5, 6, 7, 8, 9)
    assert d.day == 6


def test_hour():
    d = pendulum.datetime(1234, 5, 6, 7, 8, 9)
    assert d.hour == 7


def test_minute():
    d = pendulum.datetime(1234, 5, 6, 7, 8, 9)
    assert d.minute == 8


def test_second():
    d = pendulum.datetime(1234, 5, 6, 7, 8, 9)
    assert d.second == 9


def test_microsecond():
    d = pendulum.datetime(1234, 5, 6, 7, 8, 9)
    assert d.microsecond == 0

    d = pendulum.datetime(1234, 5, 6, 7, 8, 9, 101112)
    assert d.microsecond == 101112


def test_tzinfo():
    d = pendulum.now()
    assert d.tzinfo.name == timezone("America/Toronto").name


def test_day_of_week():
    d = pendulum.datetime(2012, 5, 7, 7, 8, 9)
    assert d.day_of_week == pendulum.MONDAY


def test_day_of_year():
    d = pendulum.datetime(2012, 5, 7)
    assert d.day_of_year == 128


def test_days_in_month():
    d = pendulum.datetime(2012, 5, 7)
    assert d.days_in_month == 31


def test_timestamp():
    d = pendulum.datetime(1970, 1, 1, 0, 0, 0)
    assert d.timestamp() == 0
    assert d.add(minutes=1, microseconds=123456).timestamp() == 60.123456


def test_float_timestamp():
    d = pendulum.datetime(1970, 1, 1, 0, 0, 0, 123456)
    assert d.float_timestamp == 0.123456


def test_int_timestamp():
    d = pendulum.datetime(1970, 1, 1, 0, 0, 0)
    assert d.int_timestamp == 0
    assert d.add(minutes=1, microseconds=123456).int_timestamp == 60


@pytest.mark.skipif(
    struct.calcsize("P") * 8 == 32, reason="Test only available for 64bit systems"
)
def test_int_timestamp_accuracy():
    d = pendulum.datetime(3000, 10, 1, 12, 23, 10, 999999)

    assert d.int_timestamp == 32527311790


def test_timestamp_with_transition():
    d_pre = pendulum.datetime(2012, 10, 28, 2, 0, tz="Europe/Warsaw", fold=0)
    d_post = pendulum.datetime(2012, 10, 28, 2, 0, tz="Europe/Warsaw", fold=1)

    # the difference between the timestamps before and after is equal to one hour
    assert d_post.timestamp() - d_pre.timestamp() == pendulum.SECONDS_PER_HOUR
    assert d_post.float_timestamp - d_pre.float_timestamp == (pendulum.SECONDS_PER_HOUR)
    assert d_post.int_timestamp - d_pre.int_timestamp == pendulum.SECONDS_PER_HOUR


def test_age():
    d = pendulum.now()
    assert d.age == 0
    assert d.add(years=1).age == -1
    assert d.subtract(years=1).age == 1


def test_local():
    assert pendulum.datetime(2012, 1, 1, tz="America/Toronto").is_local()
    assert pendulum.datetime(2012, 1, 1, tz="America/New_York").is_local()
    assert not pendulum.datetime(2012, 1, 1, tz="UTC").is_local()
    assert not pendulum.datetime(2012, 1, 1, tz="Europe/London").is_local()


def test_utc():
    assert not pendulum.datetime(2012, 1, 1, tz="America/Toronto").is_utc()
    assert not pendulum.datetime(2012, 1, 1, tz="Europe/Paris").is_utc()
    assert pendulum.datetime(2012, 1, 1, tz="UTC").is_utc()
    assert pendulum.datetime(2012, 1, 1, tz=0).is_utc()
    assert not pendulum.datetime(2012, 1, 1, tz=5).is_utc()
    # There is no time difference between Greenwich Mean Time
    # and Coordinated Universal Time
    assert pendulum.datetime(2012, 1, 1, tz="GMT").is_utc()


def test_is_dst():
    assert not pendulum.datetime(2012, 1, 1, tz="America/Toronto").is_dst()
    assert pendulum.datetime(2012, 7, 1, tz="America/Toronto").is_dst()


def test_offset_with_dst():
    assert pendulum.datetime(2012, 1, 1, tz="America/Toronto").offset == -18000


def test_offset_no_dst():
    assert pendulum.datetime(2012, 6, 1, tz="America/Toronto").offset == -14400


def test_offset_for_gmt():
    assert pendulum.datetime(2012, 6, 1, tz="GMT").offset == 0


def test_offset_hours_with_dst():
    assert pendulum.datetime(2012, 1, 1, tz="America/Toronto").offset_hours == -5


def test_offset_hours_no_dst():
    assert pendulum.datetime(2012, 6, 1, tz="America/Toronto").offset_hours == -4


def test_offset_hours_for_gmt():
    assert pendulum.datetime(2012, 6, 1, tz="GMT").offset_hours == 0


def test_offset_hours_float():
    assert pendulum.datetime(2012, 6, 1, tz=9.5).offset_hours == 9.5


def test_is_leap_year():
    assert pendulum.datetime(2012, 1, 1).is_leap_year()
    assert not pendulum.datetime(2011, 1, 1).is_leap_year()


def test_is_long_year():
    assert pendulum.datetime(2015, 1, 1).is_long_year()
    assert not pendulum.datetime(2016, 1, 1).is_long_year()


def test_week_of_month():
    assert pendulum.datetime(2012, 9, 30).week_of_month == 5
    assert pendulum.datetime(2012, 9, 28).week_of_month == 5
    assert pendulum.datetime(2012, 9, 20).week_of_month == 4
    assert pendulum.datetime(2012, 9, 8).week_of_month == 2
    assert pendulum.datetime(2012, 9, 1).week_of_month == 1
    assert pendulum.datetime(2020, 1, 1).week_of_month == 1
    assert pendulum.datetime(2020, 1, 7).week_of_month == 2
    assert pendulum.datetime(2020, 1, 14).week_of_month == 3


def test_week_of_year_first_week():
    assert pendulum.datetime(2012, 1, 1).week_of_year == 52
    assert pendulum.datetime(2012, 1, 2).week_of_year == 1


def test_week_of_year_last_week():
    assert pendulum.datetime(2012, 12, 30).week_of_year == 52
    assert pendulum.datetime(2012, 12, 31).week_of_year == 1


def test_week_of_month_edge_case():
    assert pendulum.datetime(2020, 1, 1).week_of_month == 1
    assert pendulum.datetime(2020, 1, 7).week_of_month == 2
    assert pendulum.datetime(2020, 1, 14).week_of_month == 3
    assert pendulum.datetime(2023, 1, 1).week_of_month == 1
    assert pendulum.datetime(2023, 1, 31).week_of_month == 6


def test_timezone():
    d = pendulum.datetime(2000, 1, 1, tz="America/Toronto")
    assert d.timezone.name == "America/Toronto"

    d = pendulum.datetime(2000, 1, 1, tz=-5)
    assert d.timezone.name == "-05:00"


def test_tz():
    d = pendulum.datetime(2000, 1, 1, tz="America/Toronto")
    assert d.tz.name == "America/Toronto"

    d = pendulum.datetime(2000, 1, 1, tz=-5)
    assert d.tz.name == "-05:00"


def test_timezone_name():
    d = pendulum.datetime(2000, 1, 1, tz="America/Toronto")
    assert d.timezone_name == "America/Toronto"

    d = pendulum.datetime(2000, 1, 1, tz=-5)
    assert d.timezone_name == "-05:00"


def test_is_future():
    with pendulum.travel_to(DateTime(2000, 1, 1)):
        d = pendulum.now()
        assert not d.is_future()
        d = d.add(days=1)
        assert d.is_future()


def test_is_past():
    with pendulum.travel_to(DateTime(2000, 1, 1), freeze=True):
        d = pendulum.now()
        assert not d.is_past()
        d = d.subtract(days=1)
        assert d.is_past()


def test_date():
    dt = pendulum.datetime(2016, 10, 20, 10, 40, 34, 123456)
    d = dt.date()
    assert isinstance(d, pendulum.Date)
    assert_date(d, 2016, 10, 20)


def test_time():
    dt = pendulum.datetime(2016, 10, 20, 10, 40, 34, 123456)
    t = dt.time()
    assert isinstance(t, pendulum.Time)
    assert_time(t, 10, 40, 34, 123456)


@pytest.mark.parametrize(
    "date, expected",
    [
        (pendulum.Date(2000, 1, 1), 1),
        (pendulum.Date(2000, 1, 3), 2),
        (pendulum.Date(2019, 12, 29), 5),
        (pendulum.Date(2019, 12, 30), 6),
        (pendulum.Date(2019, 12, 31), 6),
        (pendulum.Date(2020, 1, 7), 2),
        (pendulum.Date(2020, 1, 14), 3),
        (pendulum.Date(2021, 1, 1), 1),
        (pendulum.Date(2021, 1, 2), 1),
        (pendulum.Date(2021, 1, 9), 2),
        (pendulum.Date(2021, 1, 10), 2),
        (pendulum.Date(2021, 1, 11), 3),
        (pendulum.Date(2021, 1, 15), 3),
        (pendulum.Date(2021, 1, 16), 3),
        (pendulum.Date(2021, 1, 17), 3),
        (pendulum.Date(2021, 1, 23), 4),
        (pendulum.Date(2021, 1, 31), 5),
        (pendulum.Date(2021, 12, 19), 3),
        (pendulum.Date(2021, 12, 25), 4),
        (pendulum.Date(2021, 12, 26), 4),
        (pendulum.Date(2021, 12, 29), 5),
        (pendulum.Date(2021, 12, 30), 5),
        (pendulum.Date(2021, 12, 31), 5),
        (pendulum.Date(2022, 1, 1), 1),
        (pendulum.Date(2022, 1, 3), 2),
        (pendulum.Date(2022, 1, 10), 3),
        (pendulum.Date(2023, 1, 1), 1),
        (pendulum.Date(2023, 1, 2), 2),
        (pendulum.Date(2029, 12, 31), 6),
    ],
)
def test_week_of_month_negative(date, expected):
    assert date.week_of_month == expected


# === tests/datetime/test_day_of_week_modifiers.py ===
from __future__ import annotations

import pytest

import pendulum

from pendulum.exceptions import PendulumException
from tests.conftest import assert_datetime


def test_start_of_week():
    d = pendulum.datetime(1980, 8, 7, 12, 11, 9).start_of("week")
    assert_datetime(d, 1980, 8, 4, 0, 0, 0)


def test_start_of_week_from_week_start():
    d = pendulum.datetime(1980, 8, 4).start_of("week")
    assert_datetime(d, 1980, 8, 4, 0, 0, 0)


def test_start_of_week_crossing_year_boundary():
    d = pendulum.datetime(2014, 1, 1).start_of("week")
    assert_datetime(d, 2013, 12, 30, 0, 0, 0)


def test_end_of_week():
    d = pendulum.datetime(1980, 8, 7, 12, 11, 9).end_of("week")
    assert_datetime(d, 1980, 8, 10, 23, 59, 59)


def test_end_of_week_from_week_end():
    d = pendulum.datetime(1980, 8, 10).end_of("week")
    assert_datetime(d, 1980, 8, 10, 23, 59, 59)


def test_end_of_week_crossing_year_boundary():
    d = pendulum.datetime(2013, 12, 31).end_of("week")
    assert_datetime(d, 2014, 1, 5, 23, 59, 59)


def test_next():
    d = pendulum.datetime(1975, 5, 21).next()
    assert_datetime(d, 1975, 5, 28, 0, 0, 0)


def test_next_monday():
    d = pendulum.datetime(1975, 5, 21).next(pendulum.MONDAY)
    assert_datetime(d, 1975, 5, 26, 0, 0, 0)


def test_next_saturday():
    d = pendulum.datetime(1975, 5, 21).next(5)
    assert_datetime(d, 1975, 5, 24, 0, 0, 0)


def test_next_keep_time():
    d = pendulum.datetime(1975, 5, 21, 12).next()
    assert_datetime(d, 1975, 5, 28, 0, 0, 0)

    d = pendulum.datetime(1975, 5, 21, 12).next(keep_time=True)
    assert_datetime(d, 1975, 5, 28, 12, 0, 0)


def test_next_invalid():
    dt = pendulum.datetime(1975, 5, 21, 12)

    with pytest.raises(ValueError):
        dt.next(7)


def test_previous():
    d = pendulum.datetime(1975, 5, 21).previous()
    assert_datetime(d, 1975, 5, 14, 0, 0, 0)


def test_previous_monday():
    d = pendulum.datetime(1975, 5, 21).previous(pendulum.MONDAY)
    assert_datetime(d, 1975, 5, 19, 0, 0, 0)


def test_previous_saturday():
    d = pendulum.datetime(1975, 5, 21).previous(5)
    assert_datetime(d, 1975, 5, 17, 0, 0, 0)


def test_previous_keep_time():
    d = pendulum.datetime(1975, 5, 21, 12).previous()
    assert_datetime(d, 1975, 5, 14, 0, 0, 0)

    d = pendulum.datetime(1975, 5, 21, 12).previous(keep_time=True)
    assert_datetime(d, 1975, 5, 14, 12, 0, 0)


def test_previous_invalid():
    dt = pendulum.datetime(1975, 5, 21, 12)

    with pytest.raises(ValueError):
        dt.previous(7)


def test_first_day_of_month():
    d = pendulum.datetime(1975, 11, 21).first_of("month")
    assert_datetime(d, 1975, 11, 1, 0, 0, 0)


def test_first_wednesday_of_month():
    d = pendulum.datetime(1975, 11, 21).first_of("month", pendulum.WEDNESDAY)
    assert_datetime(d, 1975, 11, 5, 0, 0, 0)


def test_first_friday_of_month():
    d = pendulum.datetime(1975, 11, 21).first_of("month", 4)
    assert_datetime(d, 1975, 11, 7, 0, 0, 0)


def test_last_day_of_month():
    d = pendulum.datetime(1975, 12, 5).last_of("month")
    assert_datetime(d, 1975, 12, 31, 0, 0, 0)


def test_last_tuesday_of_month():
    d = pendulum.datetime(1975, 12, 1).last_of("month", pendulum.TUESDAY)
    assert_datetime(d, 1975, 12, 30, 0, 0, 0)


def test_last_friday_of_month():
    d = pendulum.datetime(1975, 12, 5).last_of("month", 4)
    assert_datetime(d, 1975, 12, 26, 0, 0, 0)


def test_nth_of_month_outside_scope():
    d = pendulum.datetime(1975, 6, 5)

    with pytest.raises(PendulumException):
        d.nth_of("month", 6, pendulum.MONDAY)


def test_nth_of_month_outside_year():
    d = pendulum.datetime(1975, 12, 5)

    with pytest.raises(PendulumException):
        d.nth_of("month", 55, pendulum.MONDAY)


def test_nth_of_month_first():
    d = pendulum.datetime(1975, 12, 5).nth_of("month", 1, pendulum.MONDAY)

    assert_datetime(d, 1975, 12, 1, 0, 0, 0)


def test_2nd_monday_of_month():
    d = pendulum.datetime(1975, 12, 5).nth_of("month", 2, pendulum.MONDAY)

    assert_datetime(d, 1975, 12, 8, 0, 0, 0)


def test_3rd_wednesday_of_month():
    d = pendulum.datetime(1975, 12, 5).nth_of("month", 3, 2)

    assert_datetime(d, 1975, 12, 17, 0, 0, 0)


def test_first_day_of_quarter():
    d = pendulum.datetime(1975, 11, 21).first_of("quarter")
    assert_datetime(d, 1975, 10, 1, 0, 0, 0)


def test_first_wednesday_of_quarter():
    d = pendulum.datetime(1975, 11, 21).first_of("quarter", pendulum.WEDNESDAY)
    assert_datetime(d, 1975, 10, 1, 0, 0, 0)


def test_first_friday_of_quarter():
    d = pendulum.datetime(1975, 11, 21).first_of("quarter", 4)
    assert_datetime(d, 1975, 10, 3, 0, 0, 0)


def test_first_of_quarter_from_a_day_that_will_not_exist_in_the_first_month():
    d = pendulum.datetime(2014, 5, 31).first_of("quarter")
    assert_datetime(d, 2014, 4, 1, 0, 0, 0)


def test_last_day_of_quarter():
    d = pendulum.datetime(1975, 8, 5).last_of("quarter")
    assert_datetime(d, 1975, 9, 30, 0, 0, 0)


def test_last_tuesday_of_quarter():
    d = pendulum.datetime(1975, 8, 5).last_of("quarter", pendulum.TUESDAY)
    assert_datetime(d, 1975, 9, 30, 0, 0, 0)


def test_last_friday_of_quarter():
    d = pendulum.datetime(1975, 8, 5).last_of("quarter", pendulum.FRIDAY)
    assert_datetime(d, 1975, 9, 26, 0, 0, 0)


def test_last_day_of_quarter_that_will_not_exist_in_the_last_month():
    d = pendulum.datetime(2014, 5, 31).last_of("quarter")
    assert_datetime(d, 2014, 6, 30, 0, 0, 0)


def test_nth_of_quarter_outside_scope():
    d = pendulum.datetime(1975, 1, 5)

    with pytest.raises(PendulumException):
        d.nth_of("quarter", 20, pendulum.MONDAY)


def test_nth_of_quarter_outside_year():
    d = pendulum.datetime(1975, 1, 5)

    with pytest.raises(PendulumException):
        d.nth_of("quarter", 55, pendulum.MONDAY)


def test_nth_of_quarter_first():
    d = pendulum.datetime(1975, 12, 5).nth_of("quarter", 1, pendulum.MONDAY)

    assert_datetime(d, 1975, 10, 6, 0, 0, 0)


def test_nth_of_quarter_from_a_day_that_will_not_exist_in_the_first_month():
    d = pendulum.datetime(2014, 5, 31).nth_of("quarter", 2, pendulum.MONDAY)
    assert_datetime(d, 2014, 4, 14, 0, 0, 0)


def test_2nd_monday_of_quarter():
    d = pendulum.datetime(1975, 8, 5).nth_of("quarter", 2, pendulum.MONDAY)
    assert_datetime(d, 1975, 7, 14, 0, 0, 0)


def test_3rd_wednesday_of_quarter():
    d = pendulum.datetime(1975, 8, 5).nth_of("quarter", 3, 2)
    assert_datetime(d, 1975, 7, 16, 0, 0, 0)


def test_first_day_of_year():
    d = pendulum.datetime(1975, 11, 21).first_of("year")
    assert_datetime(d, 1975, 1, 1, 0, 0, 0)


def test_first_wednesday_of_year():
    d = pendulum.datetime(1975, 11, 21).first_of("year", pendulum.WEDNESDAY)
    assert_datetime(d, 1975, 1, 1, 0, 0, 0)


def test_first_friday_of_year():
    d = pendulum.datetime(1975, 11, 21).first_of("year", 4)
    assert_datetime(d, 1975, 1, 3, 0, 0, 0)


def test_last_day_of_year():
    d = pendulum.datetime(1975, 8, 5).last_of("year")
    assert_datetime(d, 1975, 12, 31, 0, 0, 0)


def test_last_tuesday_of_year():
    d = pendulum.datetime(1975, 8, 5).last_of("year", pendulum.TUESDAY)
    assert_datetime(d, 1975, 12, 30, 0, 0, 0)


def test_last_friday_of_year():
    d = pendulum.datetime(1975, 8, 5).last_of("year", 4)
    assert_datetime(d, 1975, 12, 26, 0, 0, 0)


def test_nth_of_year_outside_scope():
    d = pendulum.datetime(1975, 1, 5)

    with pytest.raises(PendulumException):
        d.nth_of("year", 55, pendulum.MONDAY)


def test_nth_of_year_first():
    d = pendulum.datetime(1975, 12, 5).nth_of("year", 1, pendulum.MONDAY)

    assert_datetime(d, 1975, 1, 6, 0, 0, 0)


def test_2nd_monday_of_year():
    d = pendulum.datetime(1975, 8, 5).nth_of("year", 2, pendulum.MONDAY)
    assert_datetime(d, 1975, 1, 13, 0, 0, 0)


def test_2rd_wednesday_of_year():
    d = pendulum.datetime(1975, 8, 5).nth_of("year", 3, pendulum.WEDNESDAY)
    assert_datetime(d, 1975, 1, 15, 0, 0, 0)


def test_7th_thursday_of_year():
    d = pendulum.datetime(1975, 8, 31).nth_of("year", 7, pendulum.THURSDAY)
    assert_datetime(d, 1975, 2, 13, 0, 0, 0)


def test_first_of_invalid_unit():
    d = pendulum.datetime(1975, 8, 5)

    with pytest.raises(ValueError):
        d.first_of("invalid")


def test_last_of_invalid_unit():
    d = pendulum.datetime(1975, 8, 5)

    with pytest.raises(ValueError):
        d.last_of("invalid")


def test_nth_of_invalid_unit():
    d = pendulum.datetime(1975, 8, 5)

    with pytest.raises(ValueError):
        d.nth_of("invalid", 3, pendulum.MONDAY)


# === tests/datetime/test_comparison.py ===
from __future__ import annotations

from datetime import datetime

import pytz

import pendulum

from tests.conftest import assert_datetime


def test_equal_to_true():
    d1 = pendulum.datetime(2000, 1, 1, 1, 2, 3)
    d2 = pendulum.datetime(2000, 1, 1, 1, 2, 3)
    d3 = datetime(2000, 1, 1, 1, 2, 3, tzinfo=pendulum.UTC)

    assert d2 == d1
    assert d3 == d1


def test_equal_to_false():
    d1 = pendulum.datetime(2000, 1, 1, 1, 2, 3)
    d2 = pendulum.datetime(2000, 1, 2, 1, 2, 3)
    d3 = datetime(2000, 1, 2, 1, 2, 3, tzinfo=pendulum.UTC)

    assert d2 != d1
    assert d3 != d1


def test_equal_with_timezone_true():
    d1 = pendulum.datetime(2000, 1, 1, 12, 0, 0, tz="America/Toronto")
    d2 = pendulum.datetime(2000, 1, 1, 9, 0, 0, tz="America/Vancouver")
    d3 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=pendulum.timezone("America/Toronto"))

    assert d2 == d1
    assert d3 == d1


def test_equal_with_timezone_false():
    d1 = pendulum.datetime(2000, 1, 1, tz="America/Toronto")
    d2 = pendulum.datetime(2000, 1, 1, tz="America/Vancouver")
    d3 = datetime(2000, 1, 1, tzinfo=pendulum.timezone("America/Toronto"))

    assert d2 != d1
    assert d3 == d1


def test_not_equal_to_true():
    d1 = pendulum.datetime(2000, 1, 1, 1, 2, 3)
    d2 = pendulum.datetime(2000, 1, 2, 1, 2, 3)
    d3 = datetime(2000, 1, 2, 1, 2, 3, tzinfo=pendulum.UTC)

    assert d2 != d1
    assert d3 != d1


def test_not_equal_to_false():
    d1 = pendulum.datetime(2000, 1, 1, 1, 2, 3)
    d2 = pendulum.datetime(2000, 1, 1, 1, 2, 3)
    d3 = datetime(2000, 1, 1, 1, 2, 3, tzinfo=pendulum.UTC)

    assert d2 == d1
    assert d3 == d1


def test_not_equal_with_timezone_true():
    d1 = pendulum.datetime(2000, 1, 1, tz="America/Toronto")
    d2 = pendulum.datetime(2000, 1, 1, tz="America/Vancouver")
    d3 = datetime(2000, 1, 1, tzinfo=pendulum.timezone("America/Toronto"))

    assert d2 != d1
    assert d3 == d1


def test_not_equal_to_none():
    d1 = pendulum.datetime(2000, 1, 1, 1, 2, 3)

    assert d1 is not None


def test_greater_than_true():
    d1 = pendulum.datetime(2000, 1, 1)
    d2 = pendulum.datetime(1999, 12, 31)
    d3 = datetime(1999, 12, 31, tzinfo=pendulum.UTC)

    assert d1 > d2
    assert d1 > d3


def test_greater_than_false():
    d1 = pendulum.datetime(2000, 1, 1)
    d2 = pendulum.datetime(2000, 1, 2)
    d3 = datetime(2000, 1, 2, tzinfo=pendulum.UTC)

    assert not d1 > d2
    assert not d1 > d3


def test_greater_than_with_timezone_true():
    d1 = pendulum.datetime(2000, 1, 1, 12, 0, 0, tz="America/Toronto")
    d2 = pendulum.datetime(2000, 1, 1, 8, 59, 59, tz="America/Vancouver")
    d3 = pytz.timezone("America/Vancouver").localize(datetime(2000, 1, 1, 8, 59, 59))

    assert d1 > d2
    assert d1 > d3


def test_greater_than_with_timezone_false():
    d1 = pendulum.datetime(2000, 1, 1, 12, 0, 0, tz="America/Toronto")
    d2 = pendulum.datetime(2000, 1, 1, 9, 0, 1, tz="America/Vancouver")
    d3 = pytz.timezone("America/Vancouver").localize(datetime(2000, 1, 1, 9, 0, 1))

    assert not d1 > d2
    assert not d1 > d3


def test_greater_than_or_equal_true():
    d1 = pendulum.datetime(2000, 1, 1)
    d2 = pendulum.datetime(1999, 12, 31)
    d3 = datetime(1999, 12, 31, tzinfo=pendulum.UTC)

    assert d1 >= d2
    assert d1 >= d3


def test_greater_than_or_equal_true_equal():
    d1 = pendulum.datetime(2000, 1, 1)
    d2 = pendulum.datetime(2000, 1, 1)
    d3 = datetime(2000, 1, 1, tzinfo=pendulum.UTC)

    assert d1 >= d2
    assert d1 >= d3


def test_greater_than_or_equal_false():
    d1 = pendulum.datetime(2000, 1, 1)
    d2 = pendulum.datetime(2000, 1, 2)
    d3 = datetime(2000, 1, 2, tzinfo=pendulum.UTC)

    assert not d1 >= d2
    assert not d1 >= d3


def test_greater_than_or_equal_with_timezone_true():
    d1 = pendulum.datetime(2000, 1, 1, 12, 0, 0, tz="America/Toronto")
    d2 = pendulum.datetime(2000, 1, 1, 8, 59, 59, tz="America/Vancouver")
    d3 = pytz.timezone("America/Vancouver").localize(datetime(2000, 1, 1, 8, 59, 59))

    assert d1 >= d2
    assert d1 >= d3


def test_greater_than_or_equal_with_timezone_false():
    d1 = pendulum.datetime(2000, 1, 1, 12, 0, 0, tz="America/Toronto")
    d2 = pendulum.datetime(2000, 1, 1, 9, 0, 1, tz="America/Vancouver")
    d3 = pytz.timezone("America/Vancouver").localize(datetime(2000, 1, 1, 9, 0, 1))

    assert not d1 >= d2
    assert not d1 >= d3


def test_less_than_true():
    d1 = pendulum.datetime(2000, 1, 1)
    d2 = pendulum.datetime(2000, 1, 2)
    d3 = datetime(2000, 1, 2, tzinfo=pendulum.UTC)

    assert d1 < d2
    assert d1 < d3


def test_less_than_false():
    d1 = pendulum.datetime(2000, 1, 2)
    d2 = pendulum.datetime(2000, 1, 1)
    d3 = datetime(2000, 1, 1, tzinfo=pendulum.UTC)

    assert not d1 < d2
    assert not d1 < d3


def test_less_than_with_timezone_true():
    d1 = pendulum.datetime(2000, 1, 1, 8, 59, 59, tz="America/Vancouver")
    d2 = pendulum.datetime(2000, 1, 1, 12, 0, 0, tz="America/Toronto")
    d3 = pytz.timezone("America/Toronto").localize(datetime(2000, 1, 1, 12, 0, 0))

    assert d1 < d2
    assert d1 < d3


def test_less_than_with_timezone_false():
    d1 = pendulum.datetime(2000, 1, 1, 9, 0, 1, tz="America/Vancouver")
    d2 = pendulum.datetime(2000, 1, 1, 12, 0, 0, tz="America/Toronto")
    d3 = pytz.timezone("America/Toronto").localize(datetime(2000, 1, 1, 12, 0, 0))

    assert not d1 < d2
    assert not d1 < d3


def test_less_than_or_equal_true():
    d1 = pendulum.datetime(2000, 1, 1)
    d2 = pendulum.datetime(2000, 1, 2)
    d3 = datetime(2000, 1, 2, tzinfo=pendulum.UTC)

    assert d1 <= d2
    assert d1 <= d3


def test_less_than_or_equal_true_equal():
    d1 = pendulum.datetime(2000, 1, 1)
    d2 = pendulum.datetime(2000, 1, 1)
    d3 = datetime(2000, 1, 1, tzinfo=pendulum.UTC)

    assert d1 <= d2
    assert d1 <= d3


def test_less_than_or_equal_false():
    d1 = pendulum.datetime(2000, 1, 2)
    d2 = pendulum.datetime(2000, 1, 1)
    d3 = datetime(2000, 1, 1, tzinfo=pendulum.UTC)

    assert not d1 <= d2
    assert not d1 <= d3


def test_less_than_or_equal_with_timezone_true():
    d1 = pendulum.datetime(2000, 1, 1, 8, 59, 59, tz="America/Vancouver")
    d2 = pendulum.datetime(2000, 1, 1, 12, 0, 0, tz="America/Toronto")
    d3 = pytz.timezone("America/Toronto").localize(datetime(2000, 1, 1, 12, 0, 0))

    assert d1 <= d2
    assert d1 <= d3


def test_less_than_or_equal_with_timezone_false():
    d1 = pendulum.datetime(2000, 1, 1, 9, 0, 1, tz="America/Vancouver")
    d2 = pendulum.datetime(2000, 1, 1, 12, 0, 0, tz="America/Toronto")
    d3 = pytz.timezone("America/Toronto").localize(datetime(2000, 1, 1, 12, 0, 0))

    assert not d1 <= d2
    assert not d1 <= d3


def test_is_anniversary():
    with pendulum.travel_to(pendulum.now()):
        d = pendulum.now()
        an_anniversary = d.subtract(years=1)
        assert an_anniversary.is_anniversary()
        not_an_anniversary = d.subtract(days=1)
        assert not not_an_anniversary.is_anniversary()
        also_not_an_anniversary = d.add(days=2)
        assert not also_not_an_anniversary.is_anniversary()

    d1 = pendulum.datetime(1987, 4, 23)
    d2 = pendulum.datetime(2014, 9, 26)
    d3 = pendulum.datetime(2014, 4, 23)
    assert not d2.is_anniversary(d1)
    assert d3.is_anniversary(d1)


def test_is_birthday():  # backward compatibility
    with pendulum.travel_to(pendulum.now()):
        d = pendulum.now()
        an_anniversary = d.subtract(years=1)
        assert an_anniversary.is_birthday()
        not_an_anniversary = d.subtract(days=1)
        assert not not_an_anniversary.is_birthday()
        also_not_an_anniversary = d.add(days=2)
        assert not also_not_an_anniversary.is_birthday()

    d1 = pendulum.datetime(1987, 4, 23)
    d2 = pendulum.datetime(2014, 9, 26)
    d3 = pendulum.datetime(2014, 4, 23)
    assert not d2.is_birthday(d1)
    assert d3.is_birthday(d1)


def test_closest():
    instance = pendulum.datetime(2015, 5, 28, 12, 0, 0)
    dt1 = pendulum.datetime(2015, 5, 28, 11, 0, 0)
    dt2 = pendulum.datetime(2015, 5, 28, 14, 0, 0)
    closest = instance.closest(dt1, dt2)
    assert closest == dt1

    closest = instance.closest(dt2, dt1)
    assert closest == dt1

    dts = [
        pendulum.datetime(2015, 5, 28, 16, 0, 0) + pendulum.duration(hours=x)
        for x in range(4)
    ]
    closest = instance.closest(*dts)
    assert closest == dts[0]

    closest = instance.closest(*(dts[::-1]))
    assert closest == dts[0]


def test_closest_with_datetime():
    instance = pendulum.datetime(2015, 5, 28, 12, 0, 0)
    dt1 = datetime(2015, 5, 28, 11, 0, 0)
    dt2 = datetime(2015, 5, 28, 14, 0, 0)
    closest = instance.closest(dt1, dt2)
    assert_datetime(closest, 2015, 5, 28, 11, 0, 0)

    dts = [
        pendulum.datetime(2015, 5, 28, 16, 0, 0) + pendulum.duration(hours=x)
        for x in range(4)
    ]
    closest = instance.closest(dt1, dt2, *dts)

    assert_datetime(closest, 2015, 5, 28, 11, 0, 0)


def test_closest_with_equals():
    instance = pendulum.datetime(2015, 5, 28, 12, 0, 0)
    dt1 = pendulum.datetime(2015, 5, 28, 12, 0, 0)
    dt2 = pendulum.datetime(2015, 5, 28, 14, 0, 0)
    closest = instance.closest(dt1, dt2)
    assert closest == dt1


def test_farthest():
    instance = pendulum.datetime(2015, 5, 28, 12, 0, 0)
    dt1 = pendulum.datetime(2015, 5, 28, 11, 0, 0)
    dt2 = pendulum.datetime(2015, 5, 28, 14, 0, 0)
    farthest = instance.farthest(dt1, dt2)
    assert farthest == dt2

    farthest = instance.farthest(dt2, dt1)
    assert farthest == dt2

    dts = [
        pendulum.datetime(2015, 5, 28, 16, 0, 0) + pendulum.duration(hours=x)
        for x in range(4)
    ]
    farthest = instance.farthest(*dts)
    assert farthest == dts[-1]

    farthest = instance.farthest(*(dts[::-1]))
    assert farthest == dts[-1]

    f = pendulum.datetime(2010, 1, 1, 0, 0, 0)
    assert f == instance.farthest(f, *(dts))


def test_farthest_with_datetime():
    instance = pendulum.datetime(2015, 5, 28, 12, 0, 0)
    dt1 = datetime(2015, 5, 28, 11, 0, 0, tzinfo=pendulum.UTC)
    dt2 = datetime(2015, 5, 28, 14, 0, 0, tzinfo=pendulum.UTC)
    farthest = instance.farthest(dt1, dt2)
    assert_datetime(farthest, 2015, 5, 28, 14, 0, 0)

    dts = [
        pendulum.datetime(2015, 5, 28, 16, 0, 0) + pendulum.duration(hours=x)
        for x in range(4)
    ]
    farthest = instance.farthest(dt1, dt2, *dts)

    assert_datetime(farthest, 2015, 5, 28, 19, 0, 0)


def test_farthest_with_equals():
    instance = pendulum.datetime(2015, 5, 28, 12, 0, 0)
    dt1 = pendulum.datetime(2015, 5, 28, 12, 0, 0)
    dt2 = pendulum.datetime(2015, 5, 28, 14, 0, 0)
    farthest = instance.farthest(dt1, dt2)
    assert farthest == dt2

    dts = [
        pendulum.datetime(2015, 5, 28, 16, 0, 0) + pendulum.duration(hours=x)
        for x in range(4)
    ]
    farthest = instance.farthest(dt1, dt2, *dts)
    assert farthest == dts[-1]


def test_is_same_day():
    dt1 = pendulum.datetime(2015, 5, 28, 12, 0, 0)
    dt2 = pendulum.datetime(2015, 5, 29, 12, 0, 0)
    dt3 = pendulum.datetime(2015, 5, 28, 12, 0, 0)
    dt4 = datetime(2015, 5, 28, 12, 0, 0, tzinfo=pendulum.UTC)
    dt5 = datetime(2015, 5, 29, 12, 0, 0, tzinfo=pendulum.UTC)

    assert not dt1.is_same_day(dt2)
    assert dt1.is_same_day(dt3)
    assert dt1.is_same_day(dt4)
    assert not dt1.is_same_day(dt5)


def test_comparison_to_unsupported():
    dt1 = pendulum.now()

    assert dt1 != "test"
    assert dt1 not in ["test"]


# === tests/datetime/__init__.py ===


# === tests/datetime/test_add.py ===
from __future__ import annotations

from datetime import timedelta

import pytest

import pendulum

from tests.conftest import assert_datetime


def test_add_years_positive():
    assert pendulum.datetime(1975, 1, 1).add(years=1).year == 1976


def test_add_years_zero():
    assert pendulum.datetime(1975, 1, 1).add(years=0).year == 1975


def test_add_years_negative():
    assert pendulum.datetime(1975, 1, 1).add(years=-1).year == 1974


def test_add_months_positive():
    assert pendulum.datetime(1975, 12, 1).add(months=1).month == 1


def test_add_months_zero():
    assert pendulum.datetime(1975, 12, 1).add(months=0).month == 12


def test_add_months_negative():
    assert pendulum.datetime(1975, 12, 1).add(months=-1).month == 11


def test_add_month_with_overflow():
    assert pendulum.datetime(2012, 1, 31).add(months=1).month == 2


def test_add_days_positive():
    assert pendulum.datetime(1975, 5, 31).add(days=1).day == 1


def test_add_days_zero():
    assert pendulum.datetime(1975, 5, 31).add(days=0).day == 31


def test_add_days_negative():
    assert pendulum.datetime(1975, 5, 31).add(days=-1).day == 30


def test_add_weeks_positive():
    assert pendulum.datetime(1975, 5, 21).add(weeks=1).day == 28


def test_add_weeks_zero():
    assert pendulum.datetime(1975, 5, 21).add(weeks=0).day == 21


def test_add_weeks_negative():
    assert pendulum.datetime(1975, 5, 21).add(weeks=-1).day == 14


def test_add_hours_positive():
    assert pendulum.datetime(1975, 5, 21, 0, 0, 0).add(hours=1).hour == 1


def test_add_hours_zero():
    assert pendulum.datetime(1975, 5, 21, 0, 0, 0).add(hours=0).hour == 0


def test_add_hours_negative():
    assert pendulum.datetime(1975, 5, 21, 0, 0, 0).add(hours=-1).hour == 23


def test_add_minutes_positive():
    assert pendulum.datetime(1975, 5, 21, 0, 0, 0).add(minutes=1).minute == 1


def test_add_minutes_zero():
    assert pendulum.datetime(1975, 5, 21, 0, 0, 0).add(minutes=0).minute == 0


def test_add_minutes_negative():
    assert pendulum.datetime(1975, 5, 21, 0, 0, 0).add(minutes=-1).minute == 59


def test_add_seconds_positive():
    assert pendulum.datetime(1975, 5, 21, 0, 0, 0).add(seconds=1).second == 1


def test_add_seconds_zero():
    assert pendulum.datetime(1975, 5, 21, 0, 0, 0).add(seconds=0).second == 0


def test_add_seconds_negative():
    assert pendulum.datetime(1975, 5, 21, 0, 0, 0).add(seconds=-1).second == 59


def test_add_timedelta():
    delta = timedelta(days=6, seconds=45, microseconds=123456)
    d = pendulum.datetime(2015, 3, 14, 3, 12, 15, 654321)

    d = d + delta
    assert d.day == 20
    assert d.minute == 13
    assert d.second == 0
    assert d.microsecond == 777777


def test_add_duration():
    duration = pendulum.duration(
        years=2, months=3, days=6, seconds=45, microseconds=123456
    )
    d = pendulum.datetime(2015, 3, 14, 3, 12, 15, 654321)

    d = d + duration
    assert d.year == 2017
    assert d.month == 6
    assert d.day == 20
    assert d.hour == 3
    assert d.minute == 13
    assert d.second == 0
    assert d.microsecond == 777777


def test_addition_invalid_type():
    d = pendulum.datetime(2015, 3, 14, 3, 12, 15, 654321)

    with pytest.raises(TypeError):
        d + 3

    with pytest.raises(TypeError):
        3 + d


def test_add_to_fixed_timezones():
    dt = pendulum.parse("2015-03-08T01:00:00-06:00")
    dt = dt.add(weeks=1)
    dt = dt.add(hours=1)

    assert_datetime(dt, 2015, 3, 15, 2, 0, 0)
    assert dt.timezone_name == "-06:00"
    assert dt.offset == -6 * 3600


def test_add_time_to_new_transition_skipped():
    dt = pendulum.datetime(2013, 3, 31, 1, 59, 59, 999999, tz="Europe/Paris")

    assert_datetime(dt, 2013, 3, 31, 1, 59, 59, 999999)
    assert dt.timezone_name == "Europe/Paris"
    assert dt.offset == 3600
    assert not dt.is_dst()

    dt = dt.add(microseconds=1)

    assert_datetime(dt, 2013, 3, 31, 3, 0, 0, 0)
    assert dt.timezone_name == "Europe/Paris"
    assert dt.offset == 7200
    assert dt.is_dst()

    dt = pendulum.datetime(2013, 3, 10, 1, 59, 59, 999999, tz="America/New_York")

    assert_datetime(dt, 2013, 3, 10, 1, 59, 59, 999999)
    assert dt.timezone_name == "America/New_York"
    assert dt.offset == -5 * 3600
    assert not dt.is_dst()

    dt = dt.add(microseconds=1)

    assert_datetime(dt, 2013, 3, 10, 3, 0, 0, 0)
    assert dt.timezone_name == "America/New_York"
    assert dt.offset == -4 * 3600
    assert dt.is_dst()

    dt = pendulum.datetime(1957, 4, 28, 1, 59, 59, 999999, tz="America/New_York")

    assert_datetime(dt, 1957, 4, 28, 1, 59, 59, 999999)
    assert dt.timezone_name == "America/New_York"
    assert dt.offset == -5 * 3600
    assert not dt.is_dst()

    dt = dt.add(microseconds=1)

    assert_datetime(dt, 1957, 4, 28, 3, 0, 0, 0)
    assert dt.timezone_name == "America/New_York"
    assert dt.offset == -4 * 3600
    assert dt.is_dst()


def test_add_time_to_new_transition_skipped_big():
    dt = pendulum.datetime(2013, 3, 31, 1, tz="Europe/Paris")

    assert_datetime(dt, 2013, 3, 31, 1, 0, 0, 0)
    assert dt.timezone_name == "Europe/Paris"
    assert dt.offset == 3600
    assert not dt.is_dst()

    dt = dt.add(weeks=1)

    assert_datetime(dt, 2013, 4, 7, 1, 0, 0, 0)
    assert dt.timezone_name == "Europe/Paris"
    assert dt.offset == 7200
    assert dt.is_dst()


def test_add_time_to_new_transition_repeated():
    dt = pendulum.datetime(2013, 10, 27, 1, 59, 59, 999999, tz="Europe/Paris")
    dt = dt.add(hours=1)

    assert_datetime(dt, 2013, 10, 27, 2, 59, 59, 999999)
    assert dt.timezone_name == "Europe/Paris"
    assert dt.offset == 7200
    assert dt.is_dst()

    dt = dt.add(microseconds=1)

    assert_datetime(dt, 2013, 10, 27, 2, 0, 0, 0)
    assert dt.timezone_name == "Europe/Paris"
    assert dt.offset == 3600
    assert not dt.is_dst()

    dt = pendulum.datetime(2013, 11, 3, 0, 59, 59, 999999, tz="America/New_York")
    dt = dt.add(hours=1)

    assert_datetime(dt, 2013, 11, 3, 1, 59, 59, 999999)
    assert dt.timezone_name == "America/New_York"
    assert dt.offset == -4 * 3600
    assert dt.is_dst()

    dt = dt.add(microseconds=1)

    assert_datetime(dt, 2013, 11, 3, 1, 0, 0, 0)
    assert dt.timezone_name == "America/New_York"
    assert dt.offset == -5 * 3600
    assert not dt.is_dst()


def test_add_time_to_new_transition_repeated_big():
    dt = pendulum.datetime(2013, 10, 27, 1, tz="Europe/Paris")

    assert_datetime(dt, 2013, 10, 27, 1, 0, 0, 0)
    assert dt.timezone_name == "Europe/Paris"
    assert dt.offset == 7200
    assert dt.is_dst()

    dt = dt.add(weeks=1)

    assert_datetime(dt, 2013, 11, 3, 1, 0, 0, 0)
    assert dt.timezone_name == "Europe/Paris"
    assert dt.offset == 3600
    assert not dt.is_dst()


def test_add_duration_across_transition():
    dt = pendulum.datetime(2017, 3, 11, 10, 45, tz="America/Los_Angeles")
    new = dt + pendulum.duration(hours=24)

    assert_datetime(new, 2017, 3, 12, 11, 45)


def test_add_duration_across_transition_days():
    dt = pendulum.datetime(2017, 3, 11, 10, 45, tz="America/Los_Angeles")
    new = dt + pendulum.duration(days=1)

    assert_datetime(new, 2017, 3, 12, 10, 45)

    dt = pendulum.datetime(2023, 11, 5, 0, 0, tz="America/Chicago")
    new = dt + pendulum.duration(days=1)

    assert_datetime(new, 2023, 11, 6, 0, 0)


def test_interval_over_midnight_tz():
    start = pendulum.datetime(2018, 2, 25, tz="Europe/Paris")
    end = start.add(hours=1)
    interval = end - start
    new_end = start + interval

    assert new_end == end


# === tests/datetime/test_behavior.py ===
from __future__ import annotations

import pickle
import zoneinfo

from copy import deepcopy
from datetime import date
from datetime import datetime
from datetime import time

import pytest

import pendulum

from pendulum import timezone
from pendulum.tz.timezone import Timezone


@pytest.fixture
def p():
    return pendulum.datetime(2016, 8, 27, 12, 34, 56, 123456, tz="Europe/Paris")


@pytest.fixture
def p1(p):
    return p.in_tz("America/New_York")


@pytest.fixture
def dt():
    tz = timezone("Europe/Paris")

    return tz.convert(datetime(2016, 8, 27, 12, 34, 56, 123456))


def test_timetuple(p, dt):
    assert dt.timetuple() == p.timetuple()


def test_utctimetuple(p, dt):
    assert dt.utctimetuple() == p.utctimetuple()


def test_date(p, dt):
    assert p.date() == dt.date()


def test_time(p, dt):
    assert p.time() == dt.time()


def test_timetz(p, dt):
    assert p.timetz() == dt.timetz()


def test_astimezone(p, dt, p1):
    assert p.astimezone(p1.tzinfo) == dt.astimezone(p1.tzinfo)


def test_ctime(p, dt):
    assert p.ctime() == dt.ctime()


def test_isoformat(p, dt):
    assert p.isoformat() == dt.isoformat()


def test_utcoffset(p, dt):
    assert p.utcoffset() == dt.utcoffset()


def test_tzname(p, dt):
    assert p.tzname() == dt.tzname()


def test_dst(p, dt):
    assert p.dst() == dt.dst()


def test_toordinal(p, dt):
    assert p.toordinal() == dt.toordinal()


def test_weekday(p, dt):
    assert p.weekday() == dt.weekday()


def test_isoweekday(p, dt):
    assert p.isoweekday() == dt.isoweekday()


def test_isocalendar(p, dt):
    assert p.isocalendar() == dt.isocalendar()


def test_fromtimestamp():
    p = pendulum.DateTime.fromtimestamp(0, pendulum.UTC)
    dt = datetime.fromtimestamp(0, pendulum.UTC)

    assert p == dt


def test_utcfromtimestamp():
    p = pendulum.DateTime.utcfromtimestamp(0)
    dt = datetime.utcfromtimestamp(0)

    assert p == dt


def test_fromordinal():
    assert datetime.fromordinal(730120) == pendulum.DateTime.fromordinal(730120)


def test_combine():
    p = pendulum.DateTime.combine(date(2016, 1, 1), time(1, 2, 3, 123456))
    dt = datetime.combine(date(2016, 1, 1), time(1, 2, 3, 123456))

    assert p == dt


def test_hash(p, dt):
    assert hash(p) == hash(dt)

    dt1 = pendulum.datetime(2016, 8, 27, 12, 34, 56, 123456, tz="Europe/Paris")
    dt2 = pendulum.datetime(2016, 8, 27, 12, 34, 56, 123456, tz="Europe/Paris")
    dt3 = pendulum.datetime(2016, 8, 27, 12, 34, 56, 123456, tz="America/Toronto")

    assert hash(dt1) == hash(dt2)
    assert hash(dt1) != hash(dt3)


def test_pickle():
    dt1 = pendulum.datetime(2016, 8, 27, 12, 34, 56, 123456, tz="Europe/Paris")
    s = pickle.dumps(dt1)
    dt2 = pickle.loads(s)

    assert dt1 == dt2


def test_pickle_with_integer_tzinfo():
    dt1 = pendulum.datetime(2016, 8, 27, 12, 34, 56, 123456, tz=0)
    s = pickle.dumps(dt1)
    dt2 = pickle.loads(s)

    assert dt1 == dt2


def test_proper_dst():
    dt = pendulum.datetime(1941, 7, 1, tz="Europe/Amsterdam")
    native_dt = datetime(1941, 7, 1, tzinfo=zoneinfo.ZoneInfo("Europe/Amsterdam"))

    assert dt.dst() == native_dt.dst()


def test_deepcopy():
    dt = pendulum.datetime(1941, 7, 1, tz="Europe/Amsterdam")

    assert dt == deepcopy(dt)


def test_deepcopy_on_transition():
    dt = pendulum.datetime(2023, 11, 5, 1, 0, 0, tz="US/Pacific")
    clone = deepcopy(dt)

    assert dt == clone
    assert dt.offset == clone.offset


def test_pickle_timezone():
    dt1 = pendulum.timezone("Europe/Amsterdam")
    s = pickle.dumps(dt1)
    dt2 = pickle.loads(s)

    assert isinstance(dt2, Timezone)

    dt1 = pendulum.timezone("UTC")
    s = pickle.dumps(dt1)
    dt2 = pickle.loads(s)

    assert isinstance(dt2, Timezone)


# === tests/datetime/test_from_format.py ===
from __future__ import annotations

import pytest

import pendulum

from tests.conftest import assert_datetime


def test_from_format_returns_datetime():
    d = pendulum.from_format("1975-05-21 22:32:11", "YYYY-MM-DD HH:mm:ss")
    assert_datetime(d, 1975, 5, 21, 22, 32, 11)
    assert isinstance(d, pendulum.DateTime)
    assert d.timezone_name == "UTC"


def test_from_format_rejects_extra_text():
    with pytest.raises(ValueError):
        pendulum.from_format("1975-05-21 22:32:11 extra text", "YYYY-MM-DD HH:mm:ss")


def test_from_format_with_timezone_string():
    d = pendulum.from_format(
        "1975-05-21 22:32:11", "YYYY-MM-DD HH:mm:ss", tz="Europe/London"
    )
    assert_datetime(d, 1975, 5, 21, 22, 32, 11)
    assert d.timezone_name == "Europe/London"


def test_from_format_with_timezone():
    d = pendulum.from_format(
        "1975-05-21 22:32:11",
        "YYYY-MM-DD HH:mm:ss",
        tz=pendulum.timezone("Europe/London"),
    )
    assert_datetime(d, 1975, 5, 21, 22, 32, 11)
    assert d.timezone_name == "Europe/London"


def test_from_format_with_square_bracket_in_timezone():
    with pytest.raises(ValueError, match="^String does not match format"):
        pendulum.from_format(
            "1975-05-21 22:32:11 Eu[rope/London",
            "YYYY-MM-DD HH:mm:ss z",
        )


def test_from_format_with_escaped_elements():
    d = pendulum.from_format("1975-05-21T22:32:11+00:00", "YYYY-MM-DD[T]HH:mm:ssZ")
    assert_datetime(d, 1975, 5, 21, 22, 32, 11)
    assert d.timezone_name == "+00:00"


def test_from_format_with_escaped_elements_valid_tokens():
    d = pendulum.from_format("1975-05-21T22:32:11.123Z", "YYYY-MM-DD[T]HH:mm:ss.SSS[Z]")
    assert_datetime(d, 1975, 5, 21, 22, 32, 11)
    assert d.timezone_name == "UTC"


def test_from_format_with_millis():
    d = pendulum.from_format("1975-05-21 22:32:11.123456", "YYYY-MM-DD HH:mm:ss.SSSSSS")
    assert_datetime(d, 1975, 5, 21, 22, 32, 11, 123456)


def test_from_format_with_padded_day():
    d = pendulum.from_format("Apr  2 12:00:00 2020 GMT", "MMM DD HH:mm:ss YYYY z")
    assert_datetime(d, 2020, 4, 2, 12)


def test_from_format_with_invalid_padded_day():
    with pytest.raises(ValueError):
        pendulum.from_format("Apr   2 12:00:00 2020 GMT", "MMM DD HH:mm:ss YYYY z")


@pytest.mark.parametrize(
    "text,fmt,expected,now",
    [
        ("2014-4", "YYYY-Q", "2014-10-01T00:00:00+00:00", None),
        ("12-02-1999", "MM-DD-YYYY", "1999-12-02T00:00:00+00:00", None),
        ("12-02-1999", "DD-MM-YYYY", "1999-02-12T00:00:00+00:00", None),
        ("12/02/1999", "DD/MM/YYYY", "1999-02-12T00:00:00+00:00", None),
        ("12_02_1999", "DD_MM_YYYY", "1999-02-12T00:00:00+00:00", None),
        ("12:02:1999", "DD:MM:YYYY", "1999-02-12T00:00:00+00:00", None),
        ("2-2-99", "D-M-YY", "1999-02-02T00:00:00+00:00", None),
        ("99", "YY", "1999-01-01T00:00:00+00:00", None),
        ("300-1999", "DDD-YYYY", "1999-10-27T00:00:00+00:00", None),
        ("12-02-1999 2:45:10", "DD-MM-YYYY h:m:s", "1999-02-12T02:45:10+00:00", None),
        ("12-02-1999 12:45:10", "DD-MM-YYYY h:m:s", "1999-02-12T12:45:10+00:00", None),
        ("12:00:00", "HH:mm:ss", "2015-11-12T12:00:00+00:00", None),
        ("12:30:00", "HH:mm:ss", "2015-11-12T12:30:00+00:00", None),
        ("00:00:00", "HH:mm:ss", "2015-11-12T00:00:00+00:00", None),
        ("00:30:00 1", "HH:mm:ss S", "2015-11-12T00:30:00.100000+00:00", None),
        ("00:30:00 12", "HH:mm:ss SS", "2015-11-12T00:30:00.120000+00:00", None),
        ("00:30:00 123", "HH:mm:ss SSS", "2015-11-12T00:30:00.123000+00:00", None),
        ("1234567890", "X", "2009-02-13T23:31:30+00:00", None),
        ("1234567890123", "x", "2009-02-13T23:31:30.123000+00:00", None),
        ("2016-10-06", "YYYY-MM-DD", "2016-10-06T00:00:00+00:00", None),
        ("Tuesday", "dddd", "2015-11-10T00:00:00+00:00", None),
        ("Monday", "dddd", "2018-01-29T00:00:00+00:00", "2018-02-02"),
        ("Mon", "ddd", "2018-01-29T00:00:00+00:00", "2018-02-02"),
        ("Mo", "dd", "2018-01-29T00:00:00+00:00", "2018-02-02"),
        ("0", "d", "2018-01-29T00:00:00+00:00", "2018-02-02"),
        ("6", "d", "2018-02-04T00:00:00+00:00", "2018-02-02"),
        ("1", "E", "2018-01-29T00:00:00+00:00", "2018-02-02"),
        ("March", "MMMM", "2018-03-01T00:00:00+00:00", "2018-02-02"),
        ("Mar", "MMM", "2018-03-01T00:00:00+00:00", "2018-02-02"),
        (
            "Thursday 25th December 1975 02:15:16 PM",
            "dddd Do MMMM YYYY hh:mm:ss A",
            "1975-12-25T14:15:16+00:00",
            None,
        ),
        (
            "Thursday 25th December 1975 02:15:16 PM -05:00",
            "dddd Do MMMM YYYY hh:mm:ss A Z",
            "1975-12-25T14:15:16-05:00",
            None,
        ),
        (
            "1975-12-25T14:15:16 America/Guayaquil",
            "YYYY-MM-DDTHH:mm:ss z",
            "1975-12-25T14:15:16-05:00",
            None,
        ),
        (
            "1975-12-25T14:15:16 America/New_York",
            "YYYY-MM-DDTHH:mm:ss z",
            "1975-12-25T14:15:16-05:00",
            None,
        ),
        (
            "1975-12-25T14:15:16 Africa/Porto-Novo",
            "YYYY-MM-DDTHH:mm:ss z",
            "1975-12-25T14:15:16+01:00",
            None,
        ),
        (
            "1975-12-25T14:15:16 Etc/GMT+0",
            "YYYY-MM-DDTHH:mm:ss z",
            "1975-12-25T14:15:16+00:00",
            None,
        ),
        (
            "1975-12-25T14:15:16 W-SU",
            "YYYY-MM-DDTHH:mm:ss z",
            "1975-12-25T14:15:16+03:00",
            None,
        ),
        ("190022215", "YYDDDDHHmm", "2019-01-02T22:15:00+00:00", None),
    ],
)
def test_from_format(text, fmt, expected, now):
    now = pendulum.datetime(2015, 11, 12) if now is None else pendulum.parse(now)

    with pendulum.travel_to(now, freeze=True):
        assert pendulum.from_format(text, fmt).isoformat() == expected


@pytest.mark.parametrize(
    "text,fmt,expected",
    [
        ("lundi", "dddd", "2018-01-29T00:00:00+00:00"),
        ("lun.", "ddd", "2018-01-29T00:00:00+00:00"),
        ("lu", "dd", "2018-01-29T00:00:00+00:00"),
        ("mars", "MMMM", "2018-03-01T00:00:00+00:00"),
        ("mars", "MMM", "2018-03-01T00:00:00+00:00"),
    ],
)
def test_from_format_with_locale(text, fmt, expected):
    now = pendulum.datetime(2018, 2, 2)

    with pendulum.travel_to(now, freeze=True):
        formatted = pendulum.from_format(text, fmt, locale="fr").isoformat()
        assert formatted == expected


@pytest.mark.parametrize(
    "text,fmt,locale",
    [
        ("23:00", "hh:mm", "en"),
        ("23:00 am", "HH:mm a", "en"),
        ("invalid", "dddd", "en"),
        ("invalid", "ddd", "en"),
        ("invalid", "dd", "en"),
        ("invalid", "MMMM", "en"),
        ("invalid", "MMM", "en"),
    ],
)
def test_from_format_error(text, fmt, locale):
    now = pendulum.datetime(2018, 2, 2)

    with pendulum.travel_to(now, freeze=True), pytest.raises(ValueError):
        pendulum.from_format(text, fmt, locale=locale)


def test_strptime():
    d = pendulum.DateTime.strptime("1975-05-21 22:32:11", "%Y-%m-%d %H:%M:%S")
    assert_datetime(d, 1975, 5, 21, 22, 32, 11)
    assert isinstance(d, pendulum.DateTime)
    assert d.timezone_name == "UTC"


def test_from_format_2_digit_year():
    """
    Complies with open group spec for 2 digit years
    https://pubs.opengroup.org/onlinepubs/9699919799/

    "If century is not specified, then values in the range [69,99] shall
    refer to years 1969 to 1999 inclusive, and values in the
    range [00,68] shall refer to years 2000 to 2068 inclusive."
    """
    d = pendulum.from_format("00", "YY")
    assert d.year == 2000

    d = pendulum.from_format("68", "YY")
    assert d.year == 2068

    d = pendulum.from_format("69", "YY")
    assert d.year == 1969

    d = pendulum.from_format("99", "YY")
    assert d.year == 1999


# === tests/datetime/test_diff.py ===
from __future__ import annotations

from datetime import datetime

import pytest

import pendulum


def test_diff_in_years_positive():
    dt = pendulum.datetime(2000, 1, 1)
    assert dt.diff(dt.add(years=1)).in_years() == 1


def test_diff_in_years_negative_with_sign():
    dt = pendulum.datetime(2000, 1, 1)
    assert dt.diff(dt.subtract(years=1), False).in_years() == -1


def test_diff_in_years_negative_no_sign():
    dt = pendulum.datetime(2000, 1, 1)
    assert dt.diff(dt.subtract(years=1)).in_years() == 1


def test_diff_in_years_vs_default_now():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert pendulum.now().subtract(years=1).diff().in_years() == 1


def test_diff_in_years_ensure_is_truncated():
    dt = pendulum.datetime(2000, 1, 1)
    assert dt.diff(dt.add(years=1).add(months=7)).in_years() == 1


def test_diff_in_months_positive():
    dt = pendulum.datetime(2000, 1, 1)
    assert dt.diff(dt.add(years=1).add(months=1)).in_months() == 13


def test_diff_in_months_negative_with_sign():
    dt = pendulum.datetime(2000, 1, 1)
    assert dt.diff(dt.subtract(years=1).add(months=1), False).in_months() == -11


def test_diff_in_months_negative_no_sign():
    dt = pendulum.datetime(2000, 1, 1)
    assert dt.diff(dt.subtract(years=1).add(months=1)).in_months() == 11


def test_diff_in_months_vs_default_now():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert pendulum.now().subtract(years=1).diff().in_months() == 12


def test_diff_in_months_ensure_is_truncated():
    dt = pendulum.datetime(2000, 1, 1)
    assert dt.diff(dt.add(months=1).add(days=16)).in_months() == 1


def test_diff_in_days_positive():
    dt = pendulum.datetime(2000, 1, 1)
    assert dt.diff(dt.add(years=1)).in_days() == 366


def test_diff_in_days_negative_with_sign():
    dt = pendulum.datetime(2000, 1, 1)
    assert dt.diff(dt.subtract(years=1), False).in_days() == -365


def test_diff_in_days_negative_no_sign():
    dt = pendulum.datetime(2000, 1, 1)
    assert dt.diff(dt.subtract(years=1)).in_days() == 365


def test_diff_in_days_vs_default_now():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert pendulum.now().subtract(weeks=1).diff().in_days() == 7


def test_diff_in_days_ensure_is_truncated():
    dt = pendulum.datetime(2000, 1, 1)
    assert dt.diff(dt.add(days=1).add(hours=13)).in_days() == 1


def test_diff_in_weeks_positive():
    dt = pendulum.datetime(2000, 1, 1)
    assert dt.diff(dt.add(years=1)).in_weeks() == 52


def test_diff_in_weeks_negative_with_sign():
    dt = pendulum.datetime(2000, 1, 1)
    assert dt.diff(dt.subtract(years=1), False).in_weeks() == -52


def test_diff_in_weeks_negative_no_sign():
    dt = pendulum.datetime(2000, 1, 1)
    assert dt.diff(dt.subtract(years=1)).in_weeks() == 52


def test_diff_in_weeks_vs_default_now():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert pendulum.now().subtract(weeks=1).diff().in_weeks() == 1


def test_diff_in_weeks_ensure_is_truncated():
    dt = pendulum.datetime(2000, 1, 1)
    assert dt.diff(dt.add(weeks=1).subtract(days=1)).in_weeks() == 0


def test_diff_in_hours_positive():
    dt = pendulum.datetime(2000, 1, 1)
    assert dt.diff(dt.add(days=1).add(hours=2)).in_hours() == 26


def test_diff_in_hours_negative_with_sign():
    dt = pendulum.datetime(2000, 1, 1)
    assert dt.diff(dt.subtract(days=1).add(hours=2), False).in_hours() == -22


def test_diff_in_hours_negative_no_sign():
    dt = pendulum.datetime(2000, 1, 1)
    assert dt.diff(dt.subtract(days=1).add(hours=2)).in_hours() == 22


def test_diff_in_hours_vs_default_now():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 15), freeze=True):
        assert pendulum.now().subtract(days=2).diff().in_hours() == 48


def test_diff_in_hours_ensure_is_truncated():
    dt = pendulum.datetime(2000, 1, 1)
    assert dt.diff(dt.add(hours=1).add(minutes=31)).in_hours() == 1


def test_diff_in_minutes_positive():
    dt = pendulum.datetime(2000, 1, 1)
    assert dt.diff(dt.add(hours=1).add(minutes=2)).in_minutes() == 62


def test_diff_in_minutes_positive_big():
    dt = pendulum.datetime(2000, 1, 1)
    assert dt.diff(dt.add(hours=25).add(minutes=2)).in_minutes() == 1502


def test_diff_in_minutes_negative_with_sign():
    dt = pendulum.datetime(2000, 1, 1)
    assert dt.diff(dt.subtract(hours=1).add(minutes=2), False).in_minutes() == -58


def test_diff_in_minutes_negative_no_sign():
    dt = pendulum.datetime(2000, 1, 1)
    assert dt.diff(dt.subtract(hours=1).add(minutes=2)).in_minutes() == 58


def test_diff_in_minutes_vs_default_now():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert pendulum.now().subtract(hours=1).diff().in_minutes() == 60


def test_diff_in_minutes_ensure_is_truncated():
    dt = pendulum.datetime(2000, 1, 1)
    assert dt.diff(dt.add(minutes=1).add(seconds=59)).in_minutes() == 1


def test_diff_in_seconds_positive():
    dt = pendulum.datetime(2000, 1, 1)
    assert dt.diff(dt.add(minutes=1).add(seconds=2)).in_seconds() == 62


def test_diff_in_seconds_positive_big():
    dt = pendulum.datetime(2000, 1, 1)
    assert dt.diff(dt.add(hours=2).add(seconds=2)).in_seconds() == 7202


def test_diff_in_seconds_negative_with_sign():
    dt = pendulum.datetime(2000, 1, 1)
    assert dt.diff(dt.subtract(minutes=1).add(seconds=2), False).in_seconds() == -58


def test_diff_in_seconds_negative_no_sign():
    dt = pendulum.datetime(2000, 1, 1)
    assert dt.diff(dt.subtract(minutes=1).add(seconds=2)).in_seconds() == 58


def test_diff_in_seconds_vs_default_now():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert pendulum.now().subtract(hours=1).diff().in_seconds() == 3600


def test_diff_in_seconds_ensure_is_truncated():
    dt = pendulum.datetime(2000, 1, 1)
    assert dt.diff(dt.add(seconds=1.9)).in_seconds() == 1


def test_diff_in_seconds_with_timezones():
    dt_ottawa = pendulum.datetime(2000, 1, 1, 13, tz="America/Toronto")
    dt_vancouver = pendulum.datetime(2000, 1, 1, 13, tz="America/Vancouver")
    assert dt_ottawa.diff(dt_vancouver).in_seconds() == 3 * 60 * 60


def test_diff_for_humans_now_and_second():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert pendulum.now().diff_for_humans() == "a few seconds ago"


def test_diff_for_humans_now_and_second_with_timezone():
    van_now = pendulum.now("America/Vancouver")
    here_now = van_now.in_timezone(pendulum.now().timezone)

    with pendulum.travel_to(here_now, freeze=True):
        assert here_now.diff_for_humans() == "a few seconds ago"


def test_diff_for_humans_now_and_seconds():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert (
            pendulum.now().subtract(seconds=2).diff_for_humans() == "a few seconds ago"
        )


def test_diff_for_humans_now_and_nearly_minute():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert pendulum.now().subtract(seconds=59).diff_for_humans() == "59 seconds ago"


def test_diff_for_humans_now_and_minute():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert pendulum.now().subtract(minutes=1).diff_for_humans() == "1 minute ago"


def test_diff_for_humans_now_and_minutes():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert pendulum.now().subtract(minutes=2).diff_for_humans() == "2 minutes ago"


def test_diff_for_humans_now_and_nearly_hour():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert pendulum.now().subtract(minutes=59).diff_for_humans() == "59 minutes ago"


def test_diff_for_humans_now_and_hour():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert pendulum.now().subtract(hours=1).diff_for_humans() == "1 hour ago"


def test_diff_for_humans_now_and_hours():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert pendulum.now().subtract(hours=2).diff_for_humans() == "2 hours ago"


def test_diff_for_humans_now_and_nearly_day():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert pendulum.now().subtract(hours=23).diff_for_humans() == "23 hours ago"


def test_diff_for_humans_now_and_day():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert pendulum.now().subtract(days=1).diff_for_humans() == "1 day ago"


def test_diff_for_humans_now_and_days():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert pendulum.now().subtract(days=2).diff_for_humans() == "2 days ago"


def test_diff_for_humans_now_and_nearly_week():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert pendulum.now().subtract(days=6).diff_for_humans() == "6 days ago"


def test_diff_for_humans_now_and_week():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert pendulum.now().subtract(weeks=1).diff_for_humans() == "1 week ago"


def test_diff_for_humans_now_and_weeks():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert pendulum.now().subtract(weeks=2).diff_for_humans() == "2 weeks ago"


def test_diff_for_humans_now_and_nearly_month():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert pendulum.now().subtract(weeks=3).diff_for_humans() == "3 weeks ago"


def test_diff_for_humans_now_and_month():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert pendulum.now().subtract(weeks=4).diff_for_humans() == "4 weeks ago"
        assert pendulum.now().subtract(months=1).diff_for_humans() == "1 month ago"


def test_diff_for_humans_now_and_months():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert pendulum.now().subtract(months=2).diff_for_humans() == "2 months ago"


def test_diff_for_humans_now_and_nearly_year():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert pendulum.now().subtract(months=11).diff_for_humans() == "11 months ago"


def test_diff_for_humans_now_and_year():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert pendulum.now().subtract(years=1).diff_for_humans() == "1 year ago"


def test_diff_for_humans_now_and_years():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert pendulum.now().subtract(years=2).diff_for_humans() == "2 years ago"


def test_diff_for_humans_now_and_future_second():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert pendulum.now().add(seconds=1).diff_for_humans() == "in a few seconds"


def test_diff_for_humans_now_and_future_seconds():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert pendulum.now().add(seconds=2).diff_for_humans() == "in a few seconds"


def test_diff_for_humans_now_and_nearly_future_minute():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert pendulum.now().add(seconds=59).diff_for_humans() == "in 59 seconds"


def test_diff_for_humans_now_and_future_minute():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert pendulum.now().add(minutes=1).diff_for_humans() == "in 1 minute"


def test_diff_for_humans_now_and_future_minutes():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert pendulum.now().add(minutes=2).diff_for_humans() == "in 2 minutes"


def test_diff_for_humans_now_and_nearly_future_hour():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert pendulum.now().add(minutes=59).diff_for_humans() == "in 59 minutes"


def test_diff_for_humans_now_and_future_hour():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert pendulum.now().add(hours=1).diff_for_humans() == "in 1 hour"


def test_diff_for_humans_now_and_future_hours():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert pendulum.now().add(hours=2).diff_for_humans() == "in 2 hours"


def test_diff_for_humans_now_and_nearly_future_day():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert pendulum.now().add(hours=23).diff_for_humans() == "in 23 hours"


def test_diff_for_humans_now_and_future_day():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert pendulum.now().add(days=1).diff_for_humans() == "in 1 day"


def test_diff_for_humans_now_and_future_days():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert pendulum.now().add(days=2).diff_for_humans() == "in 2 days"


def test_diff_for_humans_now_and_nearly_future_week():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert pendulum.now().add(days=6).diff_for_humans() == "in 6 days"


def test_diff_for_humans_now_and_future_week():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert pendulum.now().add(weeks=1).diff_for_humans() == "in 1 week"


def test_diff_for_humans_now_and_future_weeks():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert pendulum.now().add(weeks=2).diff_for_humans() == "in 2 weeks"


def test_diff_for_humans_now_and_nearly_future_month():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert pendulum.now().add(weeks=3).diff_for_humans() == "in 3 weeks"


def test_diff_for_humans_now_and_future_month():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert pendulum.now().add(weeks=4).diff_for_humans() == "in 4 weeks"
        assert pendulum.now().add(months=1).diff_for_humans() == "in 1 month"


def test_diff_for_humans_now_and_future_months():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert pendulum.now().add(months=2).diff_for_humans() == "in 2 months"


def test_diff_for_humans_now_and_nearly_future_year():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert pendulum.now().add(months=11).diff_for_humans() == "in 11 months"


def test_diff_for_humans_now_and_future_year():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert pendulum.now().add(years=1).diff_for_humans() == "in 1 year"


def test_diff_for_humans_now_and_future_years():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert pendulum.now().add(years=2).diff_for_humans() == "in 2 years"


def test_diff_for_humans_other_and_second():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert (
            pendulum.now().diff_for_humans(pendulum.now().add(seconds=1))
            == "a few seconds before"
        )


def test_diff_for_humans_other_and_seconds():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert (
            pendulum.now().diff_for_humans(pendulum.now().add(seconds=2))
            == "a few seconds before"
        )


def test_diff_for_humans_other_and_nearly_minute():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert (
            pendulum.now().diff_for_humans(pendulum.now().add(seconds=59))
            == "59 seconds before"
        )


def test_diff_for_humans_other_and_minute():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert (
            pendulum.now().diff_for_humans(pendulum.now().add(minutes=1))
            == "1 minute before"
        )


def test_diff_for_humans_other_and_minutes():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert (
            pendulum.now().diff_for_humans(pendulum.now().add(minutes=2))
            == "2 minutes before"
        )


def test_diff_for_humans_other_and_nearly_hour():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert (
            pendulum.now().diff_for_humans(pendulum.now().add(minutes=59))
            == "59 minutes before"
        )


def test_diff_for_humans_other_and_hour():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert (
            pendulum.now().diff_for_humans(pendulum.now().add(hours=1))
            == "1 hour before"
        )


def test_diff_for_humans_other_and_hours():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert (
            pendulum.now().diff_for_humans(pendulum.now().add(hours=2))
            == "2 hours before"
        )


def test_diff_for_humans_other_and_nearly_day():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert (
            pendulum.now().diff_for_humans(pendulum.now().add(hours=23))
            == "23 hours before"
        )


def test_diff_for_humans_other_and_day():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert (
            pendulum.now().diff_for_humans(pendulum.now().add(days=1)) == "1 day before"
        )


def test_diff_for_humans_other_and_days():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert (
            pendulum.now().diff_for_humans(pendulum.now().add(days=2))
            == "2 days before"
        )


def test_diff_for_humans_other_and_nearly_week():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert (
            pendulum.now().diff_for_humans(pendulum.now().add(days=6))
            == "6 days before"
        )


def test_diff_for_humans_other_and_week():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert (
            pendulum.now().diff_for_humans(pendulum.now().add(weeks=1))
            == "1 week before"
        )


def test_diff_for_humans_other_and_weeks():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert (
            pendulum.now().diff_for_humans(pendulum.now().add(weeks=2))
            == "2 weeks before"
        )


def test_diff_for_humans_other_and_nearly_month():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert (
            pendulum.now().diff_for_humans(pendulum.now().add(weeks=3))
            == "3 weeks before"
        )


def test_diff_for_humans_other_and_month():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert (
            pendulum.now().diff_for_humans(pendulum.now().add(weeks=4))
            == "4 weeks before"
        )
        assert (
            pendulum.now().diff_for_humans(pendulum.now().add(months=1))
            == "1 month before"
        )


def test_diff_for_humans_other_and_months():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert (
            pendulum.now().diff_for_humans(pendulum.now().add(months=2))
            == "2 months before"
        )


def test_diff_for_humans_other_and_nearly_year():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert (
            pendulum.now().diff_for_humans(pendulum.now().add(months=11))
            == "11 months before"
        )


def test_diff_for_humans_other_and_year():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert (
            pendulum.now().diff_for_humans(pendulum.now().add(years=1))
            == "1 year before"
        )


def test_diff_for_humans_other_and_years():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert (
            pendulum.now().diff_for_humans(pendulum.now().add(years=2))
            == "2 years before"
        )


def test_diff_for_humans_other_and_future_second():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert (
            pendulum.now().diff_for_humans(pendulum.now().subtract(seconds=1))
            == "a few seconds after"
        )


def test_diff_for_humans_other_and_future_seconds():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert (
            pendulum.now().diff_for_humans(pendulum.now().subtract(seconds=2))
            == "a few seconds after"
        )


def test_diff_for_humans_other_and_nearly_future_minute():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert (
            pendulum.now().diff_for_humans(pendulum.now().subtract(seconds=59))
            == "59 seconds after"
        )


def test_diff_for_humans_other_and_future_minute():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert (
            pendulum.now().diff_for_humans(pendulum.now().subtract(minutes=1))
            == "1 minute after"
        )


def test_diff_for_humans_other_and_future_minutes():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert (
            pendulum.now().diff_for_humans(pendulum.now().subtract(minutes=2))
            == "2 minutes after"
        )


def test_diff_for_humans_other_and_nearly_future_hour():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert (
            pendulum.now().diff_for_humans(pendulum.now().subtract(minutes=59))
            == "59 minutes after"
        )


def test_diff_for_humans_other_and_future_hour():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert (
            pendulum.now().diff_for_humans(pendulum.now().subtract(hours=1))
            == "1 hour after"
        )


def test_diff_for_humans_other_and_future_hours():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert (
            pendulum.now().diff_for_humans(pendulum.now().subtract(hours=2))
            == "2 hours after"
        )


def test_diff_for_humans_other_and_nearly_future_day():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert (
            pendulum.now().diff_for_humans(pendulum.now().subtract(hours=23))
            == "23 hours after"
        )


def test_diff_for_humans_other_and_future_day():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert (
            pendulum.now().diff_for_humans(pendulum.now().subtract(days=1))
            == "1 day after"
        )


def test_diff_for_humans_other_and_future_days():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert (
            pendulum.now().diff_for_humans(pendulum.now().subtract(days=2))
            == "2 days after"
        )


def test_diff_for_humans_other_and_nearly_future_week():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert (
            pendulum.now().diff_for_humans(pendulum.now().subtract(days=6))
            == "6 days after"
        )


def test_diff_for_humans_other_and_future_week():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert (
            pendulum.now().diff_for_humans(pendulum.now().subtract(weeks=1))
            == "1 week after"
        )


def test_diff_for_humans_other_and_future_weeks():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert (
            pendulum.now().diff_for_humans(pendulum.now().subtract(weeks=2))
            == "2 weeks after"
        )


def test_diff_for_humans_other_and_nearly_future_month():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert (
            pendulum.now().diff_for_humans(pendulum.now().subtract(weeks=3))
            == "3 weeks after"
        )


def test_diff_for_humans_other_and_future_month():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert (
            pendulum.now().diff_for_humans(pendulum.now().subtract(weeks=4))
            == "4 weeks after"
        )
        assert (
            pendulum.now().diff_for_humans(pendulum.now().subtract(months=1))
            == "1 month after"
        )


def test_diff_for_humans_other_and_future_months():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert (
            pendulum.now().diff_for_humans(pendulum.now().subtract(months=2))
            == "2 months after"
        )


def test_diff_for_humans_other_and_nearly_future_year():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert (
            pendulum.now().diff_for_humans(pendulum.now().subtract(months=11))
            == "11 months after"
        )


def test_diff_for_humans_other_and_future_year():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert (
            pendulum.now().diff_for_humans(pendulum.now().subtract(years=1))
            == "1 year after"
        )


def test_diff_for_humans_other_and_future_years():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert (
            pendulum.now().diff_for_humans(pendulum.now().subtract(years=2))
            == "2 years after"
        )


def test_diff_for_humans_absolute_seconds():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert (
            pendulum.now().diff_for_humans(pendulum.now().subtract(seconds=59), True)
            == "59 seconds"
        )
        assert (
            pendulum.now().diff_for_humans(pendulum.now().add(seconds=59), True)
            == "59 seconds"
        )


def test_diff_for_humans_absolute_minutes():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert (
            pendulum.now().diff_for_humans(pendulum.now().subtract(minutes=30), True)
            == "30 minutes"
        )
        assert (
            pendulum.now().diff_for_humans(pendulum.now().add(minutes=30), True)
            == "30 minutes"
        )


def test_diff_for_humans_absolute_hours():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert (
            pendulum.now().diff_for_humans(pendulum.now().subtract(hours=3), True)
            == "3 hours"
        )
        assert (
            pendulum.now().diff_for_humans(pendulum.now().add(hours=3), True)
            == "3 hours"
        )


def test_diff_for_humans_absolute_days():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert (
            pendulum.now().diff_for_humans(pendulum.now().subtract(days=2), True)
            == "2 days"
        )
        assert (
            pendulum.now().diff_for_humans(pendulum.now().add(days=2), True) == "2 days"
        )


def test_diff_for_humans_absolute_weeks():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert (
            pendulum.now().diff_for_humans(pendulum.now().subtract(weeks=2), True)
            == "2 weeks"
        )
        assert (
            pendulum.now().diff_for_humans(pendulum.now().add(weeks=2), True)
            == "2 weeks"
        )


def test_diff_for_humans_absolute_months():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert (
            pendulum.now().diff_for_humans(pendulum.now().subtract(months=2), True)
            == "2 months"
        )
        assert (
            pendulum.now().diff_for_humans(pendulum.now().add(months=2), True)
            == "2 months"
        )


def test_diff_for_humans_absolute_years():
    with pendulum.travel_to(pendulum.datetime(2012, 1, 1, 1, 2, 3), freeze=True):
        assert (
            pendulum.now().diff_for_humans(pendulum.now().subtract(years=1), True)
            == "1 year"
        )
        assert (
            pendulum.now().diff_for_humans(pendulum.now().add(years=1), True)
            == "1 year"
        )


def test_diff_for_humans_accuracy():
    now = pendulum.now("utc")

    with pendulum.travel_to(now.add(microseconds=200), freeze=True):
        assert now.add(years=1).diff_for_humans(absolute=True) == "1 year"
        assert now.add(months=11).diff_for_humans(absolute=True) == "11 months"
        assert now.add(days=27).diff_for_humans(absolute=True) == "4 weeks"
        assert now.add(years=1, months=3).diff_for_humans(absolute=True) == "1 year"
        assert now.add(years=1, months=8).diff_for_humans(absolute=True) == "2 years"

    # DST
    now = pendulum.datetime(2017, 3, 7, tz="America/Toronto")
    with pendulum.travel_to(now, freeze=True):
        assert now.add(days=6).diff_for_humans(absolute=True) == "6 days"


def test_subtraction():
    d = pendulum.naive(2016, 7, 5, 12, 32, 25, 0)
    future_dt = datetime(2016, 7, 5, 13, 32, 25, 0)
    future = d.add(hours=1)

    assert (future - d).total_seconds() == 3600
    assert (future_dt - d).total_seconds() == 3600


def test_subtraction_aware_naive():
    dt = pendulum.datetime(2016, 7, 5, 12, 32, 25, 0)
    future_dt = datetime(2016, 7, 5, 13, 32, 25, 0)

    with pytest.raises(TypeError):
        future_dt - dt

    future_dt = pendulum.naive(2016, 7, 5, 13, 32, 25, 0)

    with pytest.raises(TypeError):
        future_dt - dt


def test_subtraction_with_timezone():
    dt = pendulum.datetime(2013, 3, 31, 1, 59, 59, 999999, tz="Europe/Paris")
    post = dt.add(microseconds=1)

    assert (post - dt).total_seconds() == 1e-06

    dt = pendulum.datetime(
        2013,
        10,
        27,
        2,
        59,
        59,
        999999,
        tz="Europe/Paris",
        fold=0,
    )
    post = dt.add(microseconds=1)

    assert (post - dt).total_seconds() == 1e-06


# === tests/datetime/test_timezone.py ===
from __future__ import annotations

import pendulum

from tests.conftest import assert_datetime


def test_in_timezone():
    d = pendulum.datetime(2015, 1, 15, 18, 15, 34)
    now = pendulum.datetime(2015, 1, 15, 18, 15, 34)
    assert d.timezone_name == "UTC"
    assert_datetime(d, now.year, now.month, now.day, now.hour, now.minute)

    d = d.in_timezone("Europe/Paris")
    assert d.timezone_name == "Europe/Paris"
    assert_datetime(d, now.year, now.month, now.day, now.hour + 1, now.minute)


def test_in_tz():
    d = pendulum.datetime(2015, 1, 15, 18, 15, 34)
    now = pendulum.datetime(2015, 1, 15, 18, 15, 34)
    assert d.timezone_name == "UTC"
    assert_datetime(d, now.year, now.month, now.day, now.hour, now.minute)

    d = d.in_tz("Europe/Paris")
    assert d.timezone_name == "Europe/Paris"
    assert_datetime(d, now.year, now.month, now.day, now.hour + 1, now.minute)


def test_astimezone():
    d = pendulum.datetime(2015, 1, 15, 18, 15, 34)
    now = pendulum.datetime(2015, 1, 15, 18, 15, 34)
    assert d.timezone_name == "UTC"
    assert_datetime(d, now.year, now.month, now.day, now.hour, now.minute)

    d = d.astimezone(pendulum.timezone("Europe/Paris"))
    assert d.timezone_name == "Europe/Paris"
    assert_datetime(d, now.year, now.month, now.day, now.hour + 1, now.minute)


# === tests/datetime/test_strings.py ===
from __future__ import annotations

import pytest

import pendulum


def test_to_string():
    d = pendulum.datetime(1975, 12, 25, 0, 0, 0, 0, tz="local")
    assert str(d) == "1975-12-25 00:00:00-05:00"
    d = pendulum.datetime(1975, 12, 25, 0, 0, 0, 123456, tz="local")
    assert str(d) == "1975-12-25 00:00:00.123456-05:00"


def test_to_date_string():
    d = pendulum.datetime(1975, 12, 25, 14, 15, 16)

    assert d.to_date_string() == "1975-12-25"


def test_to_formatted_date_string():
    d = pendulum.datetime(1975, 12, 25, 14, 15, 16)

    assert d.to_formatted_date_string() == "Dec 25, 1975"


def test_to_timestring():
    d = pendulum.datetime(1975, 12, 25, 14, 15, 16)

    assert d.to_time_string() == "14:15:16"


def test_to_atom_string():
    d = pendulum.datetime(1975, 12, 25, 14, 15, 16, tz="local")
    assert d.to_atom_string() == "1975-12-25T14:15:16-05:00"


def test_to_cookie_string():
    d = pendulum.datetime(1975, 12, 25, 14, 15, 16, tz="local")
    assert d.to_cookie_string() == "Thursday, 25-Dec-1975 14:15:16 EST"


def test_to_iso8601_string():
    d = pendulum.datetime(1975, 12, 25, 14, 15, 16, tz="local")
    assert d.to_iso8601_string() == "1975-12-25T14:15:16-05:00"


def test_to_iso8601_string_utc():
    d = pendulum.datetime(1975, 12, 25, 14, 15, 16)
    assert d.to_iso8601_string() == "1975-12-25T14:15:16Z"


def test_to_iso8601_extended_string():
    d = pendulum.datetime(1975, 12, 25, 14, 15, 16, 123456, tz="local")
    assert d.to_iso8601_string() == "1975-12-25T14:15:16.123456-05:00"


def test_to_rfc822_string():
    d = pendulum.datetime(1975, 12, 25, 14, 15, 16, tz="local")
    assert d.to_rfc822_string() == "Thu, 25 Dec 75 14:15:16 -0500"


def test_to_rfc850_string():
    d = pendulum.datetime(1975, 12, 25, 14, 15, 16, tz="local")
    assert d.to_rfc850_string() == "Thursday, 25-Dec-75 14:15:16 EST"


def test_to_rfc1036_string():
    d = pendulum.datetime(1975, 12, 25, 14, 15, 16, tz="local")
    assert d.to_rfc1036_string() == "Thu, 25 Dec 75 14:15:16 -0500"


def test_to_rfc1123_string():
    d = pendulum.datetime(1975, 12, 25, 14, 15, 16, tz="local")
    assert d.to_rfc1123_string() == "Thu, 25 Dec 1975 14:15:16 -0500"


def test_to_rfc2822_string():
    d = pendulum.datetime(1975, 12, 25, 14, 15, 16, tz="local")
    assert d.to_rfc2822_string() == "Thu, 25 Dec 1975 14:15:16 -0500"


def test_to_rfc3339_string():
    d = pendulum.datetime(1975, 12, 25, 14, 15, 16, tz="local")
    assert d.to_rfc3339_string() == "1975-12-25T14:15:16-05:00"


def test_to_rfc3339_extended_string():
    d = pendulum.datetime(1975, 12, 25, 14, 15, 16, 123456, tz="local")
    assert d.to_rfc3339_string() == "1975-12-25T14:15:16.123456-05:00"


def test_to_rss_string():
    d = pendulum.datetime(1975, 12, 25, 14, 15, 16, tz="local")
    assert d.to_rss_string() == "Thu, 25 Dec 1975 14:15:16 -0500"


def test_to_w3c_string():
    d = pendulum.datetime(1975, 12, 25, 14, 15, 16, tz="local")
    assert d.to_w3c_string() == "1975-12-25T14:15:16-05:00"


def test_to_string_invalid():
    d = pendulum.datetime(1975, 12, 25, 14, 15, 16, tz="local")

    with pytest.raises(ValueError):
        d._to_string("invalid")


def test_repr():
    d = pendulum.datetime(1975, 12, 25, 14, 15, 16, tz="local")
    expected = f"DateTime(1975, 12, 25, 14, 15, 16, tzinfo={d.tzinfo!r})"
    assert repr(d) == expected

    d = pendulum.datetime(1975, 12, 25, 14, 15, 16, 123456, tz="local")
    expected = f"DateTime(1975, 12, 25, 14, 15, 16, 123456, tzinfo={d.tzinfo!r})"
    assert repr(d) == expected


def test_format_with_locale():
    d = pendulum.datetime(1975, 12, 25, 14, 15, 16, tz="local")
    expected = "jeudi 25e jour de décembre 1975 02:15:16 PM -05:00"
    assert d.format("dddd Do [jour de] MMMM YYYY hh:mm:ss A Z", locale="fr") == expected


def test_strftime():
    d = pendulum.datetime(1975, 12, 25, 14, 15, 16, tz="local")
    assert d.strftime("%d") == "25"


def test_for_json():
    d = pendulum.datetime(1975, 12, 25, 14, 15, 16, tz="local")
    assert d.for_json() == "1975-12-25T14:15:16-05:00"


def test_format():
    d = pendulum.datetime(1975, 12, 25, 14, 15, 16, tz="Europe/Paris")
    assert f"{d}" == "1975-12-25 14:15:16+01:00"
    assert f"{d:YYYY}" == "1975"
    assert f"{d:%Y}" == "1975"
    assert f"{d:%H:%M %d.%m.%Y}" == "14:15 25.12.1975"


# === tests/datetime/test_create_from_timestamp.py ===
from __future__ import annotations

import pendulum

from pendulum import timezone
from tests.conftest import assert_datetime


def test_create_from_timestamp_returns_pendulum():
    d = pendulum.from_timestamp(pendulum.datetime(1975, 5, 21, 22, 32, 5).timestamp())
    assert_datetime(d, 1975, 5, 21, 22, 32, 5)
    assert d.timezone_name == "UTC"


def test_create_from_timestamp_with_timezone_string():
    d = pendulum.from_timestamp(0, "America/Toronto")
    assert d.timezone_name == "America/Toronto"
    assert_datetime(d, 1969, 12, 31, 19, 0, 0)


def test_create_from_timestamp_with_timezone():
    d = pendulum.from_timestamp(0, timezone("America/Toronto"))
    assert d.timezone_name == "America/Toronto"
    assert_datetime(d, 1969, 12, 31, 19, 0, 0)


# === tests/datetime/test_replace.py ===
from __future__ import annotations

import pendulum

from tests.conftest import assert_datetime


def test_replace_tzinfo_dst_off():
    utc = pendulum.datetime(2016, 3, 27, 0, 30)  # 30 min before DST turning on
    in_paris = utc.in_tz("Europe/Paris")

    assert_datetime(in_paris, 2016, 3, 27, 1, 30, 0)

    in_paris = in_paris.replace(second=1)

    assert_datetime(in_paris, 2016, 3, 27, 1, 30, 1)
    assert not in_paris.is_dst()
    assert in_paris.offset == 3600
    assert in_paris.timezone_name == "Europe/Paris"


def test_replace_tzinfo_dst_transitioning_on():
    utc = pendulum.datetime(2016, 3, 27, 1, 30)  # In middle of turning on
    in_paris = utc.in_tz("Europe/Paris")

    assert_datetime(in_paris, 2016, 3, 27, 3, 30, 0)

    in_paris = in_paris.replace(second=1)

    assert_datetime(in_paris, 2016, 3, 27, 3, 30, 1)
    assert in_paris.is_dst()
    assert in_paris.offset == 7200
    assert in_paris.timezone_name == "Europe/Paris"


def test_replace_tzinfo_dst_on():
    utc = pendulum.datetime(2016, 10, 30, 0, 30)  # 30 min before DST turning off
    in_paris = utc.in_tz("Europe/Paris")

    assert_datetime(in_paris, 2016, 10, 30, 2, 30, 0)

    in_paris = in_paris.replace(second=1)

    assert_datetime(in_paris, 2016, 10, 30, 2, 30, 1)
    assert in_paris.is_dst()
    assert in_paris.offset == 7200
    assert in_paris.timezone_name == "Europe/Paris"


def test_replace_tzinfo_dst_transitioning_off():
    utc = pendulum.datetime(2016, 10, 30, 1, 30)  # In the middle of turning off
    in_paris = utc.in_tz("Europe/Paris")

    assert_datetime(in_paris, 2016, 10, 30, 2, 30, 0)

    in_paris = in_paris.replace(second=1)

    assert_datetime(in_paris, 2016, 10, 30, 2, 30, 1)
    assert not in_paris.is_dst()
    assert in_paris.offset == 3600
    assert in_paris.timezone_name == "Europe/Paris"


# === tests/datetime/test_fluent_setters.py ===
from __future__ import annotations

from datetime import datetime

import pendulum

from tests.conftest import assert_datetime


def test_fluid_year_setter():
    d = pendulum.now()
    new = d.set(year=1995)
    assert isinstance(new, datetime)
    assert new.year == 1995
    assert d.year != new.year


def test_fluid_month_setter():
    d = pendulum.datetime(2016, 7, 2, 0, 41, 20)
    new = d.set(month=11)
    assert isinstance(new, datetime)
    assert new.month == 11
    assert d.month == 7


def test_fluid_day_setter():
    d = pendulum.datetime(2016, 7, 2, 0, 41, 20)
    new = d.set(day=9)
    assert isinstance(new, datetime)
    assert new.day == 9
    assert d.day == 2


def test_fluid_hour_setter():
    d = pendulum.datetime(2016, 7, 2, 0, 41, 20)
    new = d.set(hour=5)
    assert isinstance(new, datetime)
    assert new.hour == 5
    assert d.hour == 0


def test_fluid_minute_setter():
    d = pendulum.datetime(2016, 7, 2, 0, 41, 20)
    new = d.set(minute=32)
    assert isinstance(new, datetime)
    assert new.minute == 32
    assert d.minute == 41


def test_fluid_second_setter():
    d = pendulum.datetime(2016, 7, 2, 0, 41, 20)
    new = d.set(second=49)
    assert isinstance(new, datetime)
    assert new.second == 49
    assert d.second == 20


def test_fluid_microsecond_setter():
    d = pendulum.datetime(2016, 7, 2, 0, 41, 20, 123456)
    new = d.set(microsecond=987654)
    assert isinstance(new, datetime)
    assert new.microsecond == 987654
    assert d.microsecond == 123456


def test_fluid_setter_keeps_timezone():
    d = pendulum.datetime(2016, 7, 2, 0, 41, 20, 123456, tz="Europe/Paris")
    new = d.set(microsecond=987654)
    assert_datetime(new, 2016, 7, 2, 0, 41, 20, 987654)


def test_fluid_timezone_setter():
    d = pendulum.datetime(2016, 7, 2, 0, 41, 20)
    new = d.set(tz="Europe/Paris")
    assert isinstance(new, datetime)
    assert new.timezone_name == "Europe/Paris"
    assert new.tzinfo.name == "Europe/Paris"


def test_fluid_on():
    d = pendulum.datetime(2016, 7, 2, 0, 41, 20)
    new = d.on(1995, 11, 9)
    assert isinstance(new, datetime)
    assert new.year == 1995
    assert new.month == 11
    assert new.day == 9
    assert d.year == 2016
    assert d.month == 7
    assert d.day == 2


def test_fluid_on_with_transition():
    d = pendulum.datetime(2013, 3, 31, 0, 0, 0, 0, tz="Europe/Paris")
    new = d.on(2013, 4, 1)
    assert isinstance(new, datetime)
    assert new.year == 2013
    assert new.month == 4
    assert new.day == 1
    assert new.offset == 7200
    assert d.year == 2013
    assert d.month == 3
    assert d.day == 31
    assert d.offset == 3600


def test_fluid_at():
    d = pendulum.datetime(2016, 7, 2, 0, 41, 20)
    new = d.at(5, 32, 49, 123456)
    assert isinstance(new, datetime)
    assert new.hour == 5
    assert new.minute == 32
    assert new.second == 49
    assert new.microsecond == 123456
    assert d.hour == 0
    assert d.minute == 41
    assert d.second == 20
    assert d.microsecond == 0


def test_fluid_at_partial():
    d = pendulum.datetime(2016, 7, 2, 0, 41, 20)
    new = d.at(10)

    assert_datetime(new, 2016, 7, 2, 10, 0, 0, 0)

    new = d.at(10, 30)

    assert_datetime(new, 2016, 7, 2, 10, 30, 0, 0)

    new = d.at(10, 30, 45)

    assert_datetime(new, 2016, 7, 2, 10, 30, 45, 0)


def test_fluid_at_with_transition():
    d = pendulum.datetime(2013, 3, 31, 0, 0, 0, 0, tz="Europe/Paris")
    new = d.at(2, 30, 0)
    assert isinstance(new, datetime)
    assert new.hour == 3
    assert new.minute == 30
    assert new.second == 0


def test_replace_tzinfo_dst_off():
    d = pendulum.datetime(2016, 3, 27, 0, 30)  # 30 min before DST turning on
    new = d.replace(tzinfo=pendulum.timezone("Europe/Paris"))

    assert_datetime(new, 2016, 3, 27, 0, 30)
    assert not new.is_dst()
    assert new.offset == 3600
    assert new.timezone_name == "Europe/Paris"


def test_replace_tzinfo_dst_transitioning_on():
    d = pendulum.datetime(2016, 3, 27, 1, 30)  # In middle of turning on
    new = d.replace(tzinfo=pendulum.timezone("Europe/Paris"))

    assert_datetime(new, 2016, 3, 27, 1, 30)
    assert not new.is_dst()
    assert new.offset == 3600
    assert new.timezone_name == "Europe/Paris"


def test_replace_tzinfo_dst_on():
    d = pendulum.datetime(2016, 10, 30, 0, 30)  # 30 min before DST turning off
    new = d.replace(tzinfo=pendulum.timezone("Europe/Paris"))

    assert_datetime(new, 2016, 10, 30, 0, 30)
    assert new.is_dst()
    assert new.offset == 7200
    assert new.timezone_name == "Europe/Paris"


def test_replace_tzinfo_dst_transitioning_off():
    d = pendulum.datetime(2016, 10, 30, 1, 30)  # In the middle of turning off
    new = d.replace(tzinfo=pendulum.timezone("Europe/Paris"))

    assert_datetime(new, 2016, 10, 30, 1, 30)
    assert new.is_dst()
    assert new.offset == 7200
    assert new.timezone_name == "Europe/Paris"


# === tests/datetime/test_sub.py ===
from __future__ import annotations

from datetime import timedelta

import pytest

import pendulum

from tests.conftest import assert_datetime


def test_sub_years_positive():
    assert pendulum.datetime(1975, 1, 1).subtract(years=1).year == 1974


def test_sub_years_zero():
    assert pendulum.datetime(1975, 1, 1).subtract(years=0).year == 1975


def test_sub_years_negative():
    assert pendulum.datetime(1975, 1, 1).subtract(years=-1).year == 1976


def test_sub_months_positive():
    assert pendulum.datetime(1975, 12, 1).subtract(months=1).month == 11


def test_sub_months_zero():
    assert pendulum.datetime(1975, 12, 1).subtract(months=0).month == 12


def test_sub_months_negative():
    assert pendulum.datetime(1975, 12, 1).subtract(months=-1).month == 1


def test_sub_days_positive():
    assert pendulum.datetime(1975, 5, 31).subtract(days=1).day == 30


def test_sub_days_zero():
    assert pendulum.datetime(1975, 5, 31).subtract(days=0).day == 31


def test_sub_days_negative():
    assert pendulum.datetime(1975, 5, 31).subtract(days=-1).day == 1


def test_sub_weeks_positive():
    assert pendulum.datetime(1975, 5, 21).subtract(weeks=1).day == 14


def test_sub_weeks_zero():
    assert pendulum.datetime(1975, 5, 21).subtract(weeks=0).day == 21


def test_sub_weeks_negative():
    assert pendulum.datetime(1975, 5, 21).subtract(weeks=-1).day == 28


def test_sub_hours_positive():
    assert pendulum.datetime(1975, 5, 21, 0, 0, 0).subtract(hours=1).hour == 23


def test_sub_hours_zero():
    assert pendulum.datetime(1975, 5, 21, 0, 0, 0).subtract(hours=0).hour == 0


def test_sub_hours_negative():
    assert pendulum.datetime(1975, 5, 21, 0, 0, 0).subtract(hours=-1).hour == 1


def test_sub_minutes_positive():
    assert pendulum.datetime(1975, 5, 21, 0, 0, 0).subtract(minutes=1).minute == 59


def test_sub_minutes_zero():
    assert pendulum.datetime(1975, 5, 21, 0, 0, 0).subtract(minutes=0).minute == 0


def test_sub_minutes_negative():
    assert pendulum.datetime(1975, 5, 21, 0, 0, 0).subtract(minutes=-1).minute == 1


def test_sub_seconds_positive():
    assert pendulum.datetime(1975, 5, 21, 0, 0, 0).subtract(seconds=1).second == 59


def test_sub_seconds_zero():
    assert pendulum.datetime(1975, 5, 21, 0, 0, 0).subtract(seconds=0).second == 0


def test_sub_seconds_negative():
    assert pendulum.datetime(1975, 5, 21, 0, 0, 0).subtract(seconds=-1).second == 1


def test_subtract_timedelta():
    delta = timedelta(days=6, seconds=16, microseconds=654321)
    d = pendulum.datetime(2015, 3, 14, 3, 12, 15, 777777)

    d = d - delta
    assert d.day == 8
    assert d.minute == 11
    assert d.second == 59
    assert d.microsecond == 123456


def test_subtract_duration():
    duration = pendulum.duration(
        years=2, months=3, days=6, seconds=16, microseconds=654321
    )
    d = pendulum.datetime(2015, 3, 14, 3, 12, 15, 777777)

    d = d - duration
    assert d.year == 2012
    assert d.month == 12
    assert d.day == 8
    assert d.hour == 3
    assert d.minute == 11
    assert d.second == 59
    assert d.microsecond == 123456


def test_subtract_time_to_new_transition_skipped():
    dt = pendulum.datetime(2013, 3, 31, 3, 0, 0, 0, tz="Europe/Paris")

    assert_datetime(dt, 2013, 3, 31, 3, 0, 0, 0)
    assert dt.timezone_name == "Europe/Paris"
    assert dt.offset == 7200
    assert dt.is_dst()

    dt = dt.subtract(microseconds=1)

    assert_datetime(dt, 2013, 3, 31, 1, 59, 59, 999999)
    assert dt.timezone_name == "Europe/Paris"
    assert dt.offset == 3600
    assert not dt.is_dst()

    dt = pendulum.datetime(2013, 3, 10, 3, 0, 0, 0, tz="America/New_York")

    assert_datetime(dt, 2013, 3, 10, 3, 0, 0, 0)
    assert dt.timezone_name == "America/New_York"
    assert dt.offset == -4 * 3600
    assert dt.is_dst()

    dt = dt.subtract(microseconds=1)

    assert_datetime(dt, 2013, 3, 10, 1, 59, 59, 999999)
    assert dt.timezone_name == "America/New_York"
    assert dt.offset == -5 * 3600
    assert not dt.is_dst()

    dt = pendulum.datetime(1957, 4, 28, 3, 0, 0, 0, tz="America/New_York")

    assert_datetime(dt, 1957, 4, 28, 3, 0, 0, 0)
    assert dt.timezone_name == "America/New_York"
    assert dt.offset == -4 * 3600
    assert dt.is_dst()

    dt = dt.subtract(microseconds=1)

    assert_datetime(dt, 1957, 4, 28, 1, 59, 59, 999999)
    assert dt.timezone_name == "America/New_York"
    assert dt.offset == -5 * 3600
    assert not dt.is_dst()


def test_subtract_time_to_new_transition_skipped_big():
    dt = pendulum.datetime(2013, 3, 31, 3, 0, 0, 0, tz="Europe/Paris")

    assert_datetime(dt, 2013, 3, 31, 3, 0, 0, 0)
    assert dt.timezone_name == "Europe/Paris"
    assert dt.offset == 7200
    assert dt.is_dst()

    dt = dt.subtract(days=1)

    assert_datetime(dt, 2013, 3, 30, 3, 0, 0, 0)
    assert dt.timezone_name == "Europe/Paris"
    assert dt.offset == 3600
    assert not dt.is_dst()


def test_subtract_time_to_new_transition_repeated():
    dt = pendulum.datetime(2013, 10, 27, 2, 0, 0, 0, tz="Europe/Paris")

    assert_datetime(dt, 2013, 10, 27, 2, 0, 0, 0)
    assert dt.timezone_name == "Europe/Paris"
    assert dt.offset == 3600
    assert not dt.is_dst()

    dt = dt.subtract(microseconds=1)

    assert_datetime(dt, 2013, 10, 27, 2, 59, 59, 999999)
    assert dt.timezone_name == "Europe/Paris"
    assert dt.offset == 7200
    assert dt.is_dst()

    dt = pendulum.datetime(2013, 11, 3, 1, 0, 0, 0, tz="America/New_York")

    assert_datetime(dt, 2013, 11, 3, 1, 0, 0, 0)
    assert dt.timezone_name == "America/New_York"
    assert dt.offset == -5 * 3600
    assert not dt.is_dst()

    dt = dt.subtract(microseconds=1)

    assert_datetime(dt, 2013, 11, 3, 1, 59, 59, 999999)
    assert dt.timezone_name == "America/New_York"
    assert dt.offset == -4 * 3600
    assert dt.is_dst()


def test_subtract_time_to_new_transition_repeated_big():
    dt = pendulum.datetime(2013, 10, 27, 2, 0, 0, 0, tz="Europe/Paris")

    assert_datetime(dt, 2013, 10, 27, 2, 0, 0, 0)
    assert dt.timezone_name == "Europe/Paris"
    assert dt.offset == 3600
    assert not dt.is_dst()

    dt = dt.subtract(days=1)

    assert_datetime(dt, 2013, 10, 26, 2, 0, 0, 0)
    assert dt.timezone_name == "Europe/Paris"
    assert dt.offset == 7200
    assert dt.is_dst()


def test_subtract_invalid_type():
    d = pendulum.datetime(1975, 5, 21, 0, 0, 0)

    with pytest.raises(TypeError):
        d - "ab"

    with pytest.raises(TypeError):
        "ab" - d


def test_subtract_negative_over_dls_transitioning_off():
    just_before_dls_ends = pendulum.datetime(
        2019, 11, 3, 1, 30, tz="US/Pacific", fold=0
    )
    plus_10_hours = just_before_dls_ends + timedelta(hours=10)
    minus_neg_10_hours = just_before_dls_ends - timedelta(hours=-10)

    # 1:30-0700 becomes 10:30-0800
    assert plus_10_hours.hour == 10
    assert minus_neg_10_hours.hour == 10
    assert just_before_dls_ends.is_dst()
    assert not plus_10_hours.is_dst()
    assert not minus_neg_10_hours.is_dst()


# === tests/datetime/test_naive.py ===
from __future__ import annotations

import pendulum

from tests.conftest import assert_datetime


def test_naive():
    dt = pendulum.naive(2018, 2, 2, 12, 34, 56, 123456)

    assert_datetime(dt, 2018, 2, 2, 12, 34, 56, 123456)
    assert dt.tzinfo is None
    assert dt.timezone is None
    assert dt.timezone_name is None


def test_naive_add():
    dt = pendulum.naive(2013, 3, 31, 1, 30)
    new = dt.add(hours=1)

    assert_datetime(new, 2013, 3, 31, 2, 30)


def test_naive_subtract():
    dt = pendulum.naive(2013, 3, 31, 1, 30)
    new = dt.subtract(hours=1)

    assert_datetime(new, 2013, 3, 31, 0, 30)


def test_naive_in_timezone():
    dt = pendulum.naive(2013, 3, 31, 1, 30)
    new = dt.in_timezone("Europe/Paris")

    assert_datetime(new, 2013, 3, 31, 1, 30)
    assert new.timezone_name == "Europe/Paris"


def test_naive_in_timezone_dst():
    dt = pendulum.naive(2013, 3, 31, 2, 30)
    new = dt.in_timezone("Europe/Paris")

    assert_datetime(new, 2013, 3, 31, 3, 30)
    assert new.timezone_name == "Europe/Paris"


def test_add():
    dt = pendulum.naive(2013, 3, 31, 2, 30)
    new = dt.add(days=3)

    assert_datetime(new, 2013, 4, 3, 2, 30)


def test_subtract():
    dt = pendulum.naive(2013, 3, 31, 2, 30)
    new = dt.subtract(days=3)

    assert_datetime(new, 2013, 3, 28, 2, 30)


def test_to_strings():
    dt = pendulum.naive(2013, 3, 31, 2, 30)

    assert dt.isoformat() == "2013-03-31T02:30:00"
    assert dt.to_iso8601_string() == "2013-03-31T02:30:00"
    assert dt.to_rfc3339_string() == "2013-03-31T02:30:00"
    assert dt.to_atom_string() == "2013-03-31T02:30:00"
    assert dt.to_cookie_string() == "Sunday, 31-Mar-2013 02:30:00 "


def test_naive_method():
    dt = pendulum.datetime(2018, 2, 2, 12, 34, 56, 123456)
    dt = dt.naive()

    assert_datetime(dt, 2018, 2, 2, 12, 34, 56, 123456)
    assert dt.tzinfo is None
    assert dt.timezone is None
    assert dt.timezone_name is None


# === tests/interval/test_construct.py ===
from __future__ import annotations

from datetime import datetime

import pendulum

from tests.conftest import assert_datetime


def test_with_datetimes():
    dt1 = datetime(2000, 1, 1)
    dt2 = datetime(2000, 1, 31)
    p = pendulum.interval(dt1, dt2)

    assert isinstance(p.start, pendulum.DateTime)
    assert isinstance(p.end, pendulum.DateTime)
    assert_datetime(p.start, 2000, 1, 1)
    assert_datetime(p.end, 2000, 1, 31)


def test_with_pendulum():
    dt1 = pendulum.DateTime(2000, 1, 1)
    dt2 = pendulum.DateTime(2000, 1, 31)
    p = pendulum.interval(dt1, dt2)

    assert_datetime(p.start, 2000, 1, 1)
    assert_datetime(p.end, 2000, 1, 31)


def test_inverted():
    dt1 = pendulum.DateTime(2000, 1, 1)
    dt2 = pendulum.DateTime(2000, 1, 31)
    p = pendulum.interval(dt2, dt1)

    assert_datetime(p.start, 2000, 1, 31)
    assert_datetime(p.end, 2000, 1, 1)


def test_inverted_and_absolute():
    dt1 = pendulum.DateTime(2000, 1, 1)
    dt2 = pendulum.DateTime(2000, 1, 31)
    p = pendulum.interval(dt2, dt1, True)

    assert_datetime(p.start, 2000, 1, 1)
    assert_datetime(p.end, 2000, 1, 31)


def test_accuracy():
    dt1 = pendulum.DateTime(2000, 11, 20)
    dt2 = pendulum.DateTime(2000, 11, 25)
    dt3 = pendulum.DateTime(2016, 11, 5)
    p1 = pendulum.interval(dt1, dt3)
    p2 = pendulum.interval(dt2, dt3)

    assert p1.years == 15
    assert p1.in_years() == 15
    assert p1.months == 11
    assert p1.in_months() == 191
    assert p1.days == 5829
    assert p1.remaining_days == 2
    assert p1.in_days() == 5829

    assert p2.years == 15
    assert p2.in_years() == 15
    assert p2.months == 11
    assert p2.in_months() == 191
    assert p2.days == 5824
    assert p2.remaining_days == 4
    assert p2.in_days() == 5824


def test_dst_transition():
    start = pendulum.datetime(2017, 3, 7, tz="America/Toronto")
    end = start.add(days=6)
    interval = end - start

    assert interval.days == 5
    assert interval.seconds == 82800

    assert interval.remaining_days == 6
    assert interval.hours == 0
    assert interval.remaining_seconds == 0

    assert interval.in_days() == 6
    assert interval.in_hours() == 5 * 24 + 23


def test_timedelta_behavior():
    dt1 = pendulum.DateTime(2000, 11, 20, 1)
    dt2 = pendulum.DateTime(2000, 11, 25, 2)
    dt3 = pendulum.DateTime(2016, 11, 5, 3)

    p1 = pendulum.interval(dt1, dt3)
    p2 = pendulum.interval(dt2, dt3)
    it1 = p1.as_timedelta()
    it2 = p2.as_timedelta()

    assert it1.total_seconds() == p1.total_seconds()
    assert it2.total_seconds() == p2.total_seconds()
    assert it1.days == p1.days
    assert it2.days == p2.days
    assert it1.seconds == p1.seconds
    assert it2.seconds == p2.seconds
    assert it1.microseconds == p1.microseconds
    assert it2.microseconds == p2.microseconds


def test_different_timezones_same_time():
    dt1 = pendulum.datetime(2013, 3, 31, 1, 30, tz="Europe/Paris")
    dt2 = pendulum.datetime(2013, 4, 1, 1, 30, tz="Europe/Paris")
    interval = dt2 - dt1

    assert interval.in_words() == "1 day"
    assert interval.in_hours() == 23

    dt1 = pendulum.datetime(2013, 3, 31, 1, 30, tz="Europe/Paris")
    dt2 = pendulum.datetime(2013, 4, 1, 1, 30, tz="America/Toronto")
    interval = dt2 - dt1

    assert interval.in_words() == "1 day 5 hours"
    assert interval.in_hours() == 29


# === tests/interval/__init__.py ===


# === tests/interval/test_behavior.py ===
from __future__ import annotations

import pickle

from datetime import timedelta

import pendulum


def test_pickle():
    dt1 = pendulum.datetime(2016, 11, 18)
    dt2 = pendulum.datetime(2016, 11, 20)

    p = pendulum.interval(dt1, dt2)
    s = pickle.dumps(p)
    p2 = pickle.loads(s)

    assert p.start == p2.start
    assert p.end == p2.end
    assert p.invert == p2.invert

    p = pendulum.interval(dt2, dt1)
    s = pickle.dumps(p)
    p2 = pickle.loads(s)

    assert p.start == p2.start
    assert p.end == p2.end
    assert p.invert == p2.invert

    p = pendulum.interval(dt2, dt1, True)
    s = pickle.dumps(p)
    p2 = pickle.loads(s)

    assert p.start == p2.start
    assert p.end == p2.end
    assert p.invert == p2.invert


def test_comparison_to_timedelta():
    dt1 = pendulum.datetime(2016, 11, 18)
    dt2 = pendulum.datetime(2016, 11, 20)

    interval = dt2 - dt1

    assert interval < timedelta(days=4)


def test_equality_to_timedelta():
    dt1 = pendulum.datetime(2016, 11, 18)
    dt2 = pendulum.datetime(2016, 11, 20)

    interval = dt2 - dt1

    assert interval == timedelta(days=2)


def test_inequality():
    dt1 = pendulum.datetime(2016, 11, 18)
    dt2 = pendulum.datetime(2016, 11, 20)
    dt3 = pendulum.datetime(2016, 11, 22)

    interval1 = dt2 - dt1
    interval2 = dt3 - dt2
    interval3 = dt3 - dt1

    assert interval1 != interval2
    assert interval1 != interval3


# === tests/interval/test_hashing.py ===
from __future__ import annotations

import pendulum


def test_intervals_with_same_duration_and_different_dates():
    day1 = pendulum.DateTime(2018, 1, 1)
    day2 = pendulum.DateTime(2018, 1, 2)
    day3 = pendulum.DateTime(2018, 1, 2)

    interval1 = day2 - day1
    interval2 = day3 - day2

    assert interval1 != interval2
    assert len({interval1, interval2}) == 2


def test_intervals_with_same_dates():
    interval1 = pendulum.DateTime(2018, 1, 2) - pendulum.DateTime(2018, 1, 1)
    interval2 = pendulum.DateTime(2018, 1, 2) - pendulum.DateTime(2018, 1, 1)

    assert interval1 == interval2
    assert len({interval1, interval2}) == 1


# === tests/interval/test_in_words.py ===
from __future__ import annotations

import pendulum


def test_week():
    start_date = pendulum.datetime(2012, 1, 1)
    interval = pendulum.interval(start=start_date, end=start_date.add(weeks=1))
    assert interval.in_words() == "1 week"


def test_week_and_day():
    start_date = pendulum.datetime(2012, 1, 1)
    interval = pendulum.interval(start=start_date, end=start_date.add(weeks=1, days=1))
    assert interval.in_words() == "1 week 1 day"


def test_all():
    start_date = pendulum.datetime(2012, 1, 1)
    interval = pendulum.interval(
        start=start_date,
        end=start_date.add(years=1, months=1, days=1, seconds=1, microseconds=1),
    )
    assert interval.in_words() == "1 year 1 month 1 day 1 second"


def test_in_french():
    start_date = pendulum.datetime(2012, 1, 1)
    interval = pendulum.interval(
        start=start_date,
        end=start_date.add(years=1, months=1, days=1, seconds=1, microseconds=1),
    )
    assert interval.in_words(locale="fr") == "1 an 1 mois 1 jour 1 seconde"


def test_singular_negative_values():
    start_date = pendulum.datetime(2012, 1, 1)
    interval = pendulum.interval(start=start_date, end=start_date.subtract(days=1))
    assert interval.in_words() == "-1 day"


def test_separator():
    start_date = pendulum.datetime(2012, 1, 1)
    interval = pendulum.interval(
        start=start_date,
        end=start_date.add(years=1, months=1, days=1, seconds=1, microseconds=1),
    )
    assert interval.in_words(separator=", ") == "1 year, 1 month, 1 day, 1 second"


def test_subseconds():
    start_date = pendulum.datetime(2012, 1, 1)
    interval = pendulum.interval(
        start=start_date, end=start_date.add(microseconds=123456)
    )
    assert interval.in_words() == "0.12 second"


def test_subseconds_with_seconds():
    start_date = pendulum.datetime(2012, 1, 1)
    interval = pendulum.interval(
        start=start_date, end=start_date.add(seconds=12, microseconds=123456)
    )
    assert interval.in_words() == "12 seconds"


def test_zero_interval():
    start_date = pendulum.datetime(2012, 1, 1)
    interval = pendulum.interval(start=start_date, end=start_date)
    assert interval.in_words() == "0 microseconds"


# === tests/interval/test_arithmetic.py ===
from __future__ import annotations

import pendulum

from tests.conftest import assert_duration


def test_multiply():
    dt1 = pendulum.DateTime(2016, 8, 7, 12, 34, 56)
    dt2 = dt1.add(days=6, seconds=34)
    it = pendulum.interval(dt1, dt2)
    mul = it * 2
    assert isinstance(mul, pendulum.Duration)
    assert_duration(mul, 0, 0, 1, 5, 0, 1, 8)

    dt1 = pendulum.DateTime(2016, 8, 7, 12, 34, 56)
    dt2 = dt1.add(days=6, seconds=34)
    it = pendulum.interval(dt1, dt2)
    mul = it * 2
    assert isinstance(mul, pendulum.Duration)
    assert_duration(mul, 0, 0, 1, 5, 0, 1, 8)


def test_divide():
    dt1 = pendulum.DateTime(2016, 8, 7, 12, 34, 56)
    dt2 = dt1.add(days=2, seconds=34)
    it = pendulum.interval(dt1, dt2)
    mul = it / 2
    assert isinstance(mul, pendulum.Duration)
    assert_duration(mul, 0, 0, 0, 1, 0, 0, 17)

    dt1 = pendulum.DateTime(2016, 8, 7, 12, 34, 56)
    dt2 = dt1.add(days=2, seconds=35)
    it = pendulum.interval(dt1, dt2)
    mul = it / 2
    assert isinstance(mul, pendulum.Duration)
    assert_duration(mul, 0, 0, 0, 1, 0, 0, 17)


def test_floor_divide():
    dt1 = pendulum.DateTime(2016, 8, 7, 12, 34, 56)
    dt2 = dt1.add(days=2, seconds=34)
    it = pendulum.interval(dt1, dt2)
    mul = it // 2
    assert isinstance(mul, pendulum.Duration)
    assert_duration(mul, 0, 0, 0, 1, 0, 0, 17)

    dt1 = pendulum.DateTime(2016, 8, 7, 12, 34, 56)
    dt2 = dt1.add(days=2, seconds=35)
    it = pendulum.interval(dt1, dt2)
    mul = it // 3
    assert isinstance(mul, pendulum.Duration)
    assert_duration(mul, 0, 0, 0, 0, 16, 0, 11)


# === tests/interval/test_range.py ===
from __future__ import annotations

import pendulum

from pendulum.interval import Interval
from tests.conftest import assert_datetime


def test_range():
    dt1 = pendulum.datetime(2000, 1, 1, 12, 45, 37)
    dt2 = pendulum.datetime(2000, 1, 31, 12, 45, 37)

    p = Interval(dt1, dt2)
    r = list(p.range("days"))

    assert len(r) == 31
    assert_datetime(r[0], 2000, 1, 1, 12, 45, 37)
    assert_datetime(r[-1], 2000, 1, 31, 12, 45, 37)


def test_range_no_overflow():
    dt1 = pendulum.datetime(2000, 1, 1, 12, 45, 37)
    dt2 = pendulum.datetime(2000, 1, 31, 11, 45, 37)

    p = Interval(dt1, dt2)
    r = list(p.range("days"))

    assert len(r) == 30
    assert_datetime(r[0], 2000, 1, 1, 12, 45, 37)
    assert_datetime(r[-1], 2000, 1, 30, 12, 45, 37)


def test_range_inverted():
    dt1 = pendulum.datetime(2000, 1, 1, 12, 45, 37)
    dt2 = pendulum.datetime(2000, 1, 31, 12, 45, 37)

    p = Interval(dt2, dt1)
    r = list(p.range("days"))

    assert len(r) == 31
    assert_datetime(r[-1], 2000, 1, 1, 12, 45, 37)
    assert_datetime(r[0], 2000, 1, 31, 12, 45, 37)


def test_iter():
    dt1 = pendulum.datetime(2000, 1, 1, 12, 45, 37)
    dt2 = pendulum.datetime(2000, 1, 31, 12, 45, 37)

    p = Interval(dt1, dt2)
    i = 0
    for dt in p:
        assert isinstance(dt, pendulum.DateTime)
        i += 1

    assert i == 31


def test_contains():
    dt1 = pendulum.datetime(2000, 1, 1, 12, 45, 37)
    dt2 = pendulum.datetime(2000, 1, 31, 12, 45, 37)

    p = pendulum.interval(dt1, dt2)
    dt = pendulum.datetime(2000, 1, 7)
    assert dt in p


def test_not_contains():
    dt1 = pendulum.datetime(2000, 1, 1, 12, 45, 37)
    dt2 = pendulum.datetime(2000, 1, 31, 12, 45, 37)

    p = pendulum.interval(dt1, dt2)
    dt = pendulum.datetime(2000, 1, 1, 11, 45, 37)
    assert dt not in p


def test_contains_with_datetime():
    dt1 = pendulum.datetime(2000, 1, 1, 12, 45, 37)
    dt2 = pendulum.datetime(2000, 1, 31, 12, 45, 37)

    p = pendulum.interval(dt1, dt2)
    dt = pendulum.datetime(2000, 1, 7)
    assert dt in p


def test_range_months_overflow():
    dt1 = pendulum.datetime(2016, 1, 30, tz="America/Sao_Paulo")
    dt2 = dt1.add(months=4)

    p = pendulum.interval(dt1, dt2)
    r = list(p.range("months"))

    assert_datetime(r[0], 2016, 1, 30, 0, 0, 0)
    assert_datetime(r[-1], 2016, 5, 30, 0, 0, 0)


def test_range_with_dst():
    dt1 = pendulum.datetime(2016, 10, 14, tz="America/Sao_Paulo")
    dt2 = dt1.add(weeks=1)

    p = pendulum.interval(dt1, dt2)
    r = list(p.range("days"))

    assert_datetime(r[0], 2016, 10, 14, 0, 0, 0)
    assert_datetime(r[2], 2016, 10, 16, 1, 0, 0)
    assert_datetime(r[-1], 2016, 10, 21, 0, 0, 0)


def test_range_amount():
    dt1 = pendulum.datetime(2016, 10, 14, tz="America/Sao_Paulo")
    dt2 = dt1.add(weeks=1)

    p = pendulum.interval(dt1, dt2)
    r = list(p.range("days", 2))

    assert len(r) == 4
    assert_datetime(r[0], 2016, 10, 14, 0, 0, 0)
    assert_datetime(r[1], 2016, 10, 16, 1, 0, 0)
    assert_datetime(r[2], 2016, 10, 18, 0, 0, 0)
    assert_datetime(r[3], 2016, 10, 20, 0, 0, 0)


# === tests/interval/test_add_subtract.py ===
from __future__ import annotations

import pendulum


def test_dst_add():
    start = pendulum.datetime(2017, 3, 7, tz="America/Toronto")
    end = start.add(days=6)
    interval = end - start
    new_end = start + interval

    assert new_end == end


def test_dst_add_non_variable_units():
    start = pendulum.datetime(2013, 3, 31, 1, 30, tz="Europe/Paris")
    end = start.add(hours=1)
    interval = end - start
    new_end = start + interval

    assert new_end == end


def test_dst_subtract():
    start = pendulum.datetime(2017, 3, 7, tz="America/Toronto")
    end = start.add(days=6)
    interval = end - start
    new_start = end - interval

    assert new_start == start


def test_naive_subtract():
    start = pendulum.naive(2013, 3, 31, 1, 30)
    end = start.add(hours=1)
    interval = end - start
    new_end = start + interval

    assert new_end == end


def test_negative_difference_subtract():
    start = pendulum.datetime(2018, 5, 28, 12, 34, 56, 123456)
    end = pendulum.datetime(2018, 1, 1)

    interval = end - start
    new_end = start + interval

    assert new_end == end


# === tests/date/test_construct.py ===
from __future__ import annotations

from pendulum import Date
from tests.conftest import assert_date


def test_construct():
    d = Date(2016, 10, 20)

    assert_date(d, 2016, 10, 20)


def test_today():
    d = Date.today()

    assert isinstance(d, Date)


# === tests/date/test_start_end_of.py ===
from __future__ import annotations

import pytest

import pendulum

from pendulum import Date
from tests.conftest import assert_date


def test_start_of_day():
    d = Date.today()
    new = d.start_of("day")
    assert isinstance(new, Date)
    assert_date(new, d.year, d.month, d.day)


def test_end_of_day():
    d = Date.today()
    new = d.end_of("day")
    assert isinstance(new, Date)
    assert_date(new, d.year, d.month, d.day)


def test_start_of_week():
    d = Date(2016, 10, 20)
    new = d.start_of("week")
    assert isinstance(new, Date)
    assert_date(new, d.year, d.month, 17)


def test_end_of_week():
    d = Date(2016, 10, 20)
    new = d.end_of("week")
    assert isinstance(new, Date)
    assert_date(new, d.year, d.month, 23)


def test_start_of_month_is_fluid():
    d = Date.today()
    assert isinstance(d.start_of("month"), Date)


def test_start_of_month_from_now():
    d = Date.today()
    new = d.start_of("month")
    assert_date(new, d.year, d.month, 1)


def test_start_of_month_from_last_day():
    d = Date(2000, 1, 31)
    new = d.start_of("month")
    assert_date(new, 2000, 1, 1)


def test_start_of_year_is_fluid():
    d = Date.today()
    new = d.start_of("year")
    assert isinstance(new, Date)


def test_start_of_year_from_now():
    d = Date.today()
    new = d.start_of("year")
    assert_date(new, d.year, 1, 1)


def test_start_of_year_from_first_day():
    d = Date(2000, 1, 1)
    new = d.start_of("year")
    assert_date(new, 2000, 1, 1)


def test_start_of_year_from_last_day():
    d = Date(2000, 12, 31)
    new = d.start_of("year")
    assert_date(new, 2000, 1, 1)


def test_end_of_month_is_fluid():
    d = Date.today()
    assert isinstance(d.end_of("month"), Date)


def test_end_of_month_from_now():
    d = Date.today().start_of("month")
    new = d.start_of("month")
    assert_date(new, d.year, d.month, 1)


def test_end_of_month():
    d = Date(2000, 1, 1).end_of("month")
    new = d.end_of("month")
    assert_date(new, 2000, 1, 31)


def test_end_of_month_from_last_day():
    d = Date(2000, 1, 31)
    new = d.end_of("month")
    assert_date(new, 2000, 1, 31)


def test_end_of_year_is_fluid():
    d = Date.today()
    assert isinstance(d.end_of("year"), Date)


def test_end_of_year_from_now():
    d = Date.today().end_of("year")
    new = d.end_of("year")
    assert_date(new, d.year, 12, 31)


def test_end_of_year_from_first_day():
    d = Date(2000, 1, 1)
    new = d.end_of("year")
    assert_date(new, 2000, 12, 31)


def test_end_of_year_from_last_day():
    d = Date(2000, 12, 31)
    new = d.end_of("year")
    assert_date(new, 2000, 12, 31)


def test_start_of_decade_is_fluid():
    d = Date.today()
    assert isinstance(d.start_of("decade"), Date)


def test_start_of_decade_from_now():
    d = Date.today()
    new = d.start_of("decade")
    assert_date(new, d.year - d.year % 10, 1, 1)


def test_start_of_decade_from_first_day():
    d = Date(2000, 1, 1)
    new = d.start_of("decade")
    assert_date(new, 2000, 1, 1)


def test_start_of_decade_from_last_day():
    d = Date(2009, 12, 31)
    new = d.start_of("decade")
    assert_date(new, 2000, 1, 1)


def test_end_of_decade_is_fluid():
    d = Date.today()
    assert isinstance(d.end_of("decade"), Date)


def test_end_of_decade_from_now():
    d = Date.today()
    new = d.end_of("decade")
    assert_date(new, d.year - d.year % 10 + 9, 12, 31)


def test_end_of_decade_from_first_day():
    d = Date(2000, 1, 1)
    new = d.end_of("decade")
    assert_date(new, 2009, 12, 31)


def test_end_of_decade_from_last_day():
    d = Date(2009, 12, 31)
    new = d.end_of("decade")
    assert_date(new, 2009, 12, 31)


def test_start_of_century_is_fluid():
    d = Date.today()
    assert isinstance(d.start_of("century"), Date)


def test_start_of_century_from_now():
    d = Date.today()
    new = d.start_of("century")
    assert_date(new, d.year - d.year % 100 + 1, 1, 1)


def test_start_of_century_from_first_day():
    d = Date(2001, 1, 1)
    new = d.start_of("century")
    assert_date(new, 2001, 1, 1)


def test_start_of_century_from_last_day():
    d = Date(2100, 12, 31)
    new = d.start_of("century")
    assert_date(new, 2001, 1, 1)


def test_end_of_century_is_fluid():
    d = Date.today()
    assert isinstance(d.end_of("century"), Date)


def test_end_of_century_from_now():
    now = Date.today()
    d = now.end_of("century")
    assert_date(d, now.year - now.year % 100 + 100, 12, 31)


def test_end_of_century_from_first_day():
    d = Date(2001, 1, 1)
    new = d.end_of("century")
    assert_date(new, 2100, 12, 31)


def test_end_of_century_from_last_day():
    d = Date(2100, 12, 31)
    new = d.end_of("century")
    assert_date(new, 2100, 12, 31)


def test_average_is_fluid():
    d = Date.today().average()
    assert isinstance(d, Date)


def test_average_from_same():
    d1 = pendulum.date(2000, 1, 31)
    d2 = pendulum.date(2000, 1, 31).average(d1)
    assert_date(d2, 2000, 1, 31)


def test_average_from_greater():
    d1 = pendulum.date(2000, 1, 1)
    d2 = pendulum.date(2009, 12, 31).average(d1)
    assert_date(d2, 2004, 12, 31)


def test_average_from_lower():
    d1 = pendulum.date(2009, 12, 31)
    d2 = pendulum.date(2000, 1, 1).average(d1)
    assert_date(d2, 2004, 12, 31)


def test_start_of():
    d = pendulum.date(2013, 3, 31)

    with pytest.raises(ValueError):
        d.start_of("invalid")


def test_end_of_invalid_unit():
    d = pendulum.date(2013, 3, 31)

    with pytest.raises(ValueError):
        d.end_of("invalid")


# === tests/date/test_getters.py ===
from __future__ import annotations

import pendulum


def test_year():
    d = pendulum.Date(1234, 5, 6)
    assert d.year == 1234


def test_month():
    d = pendulum.Date(1234, 5, 6)
    assert d.month == 5


def test_day():
    d = pendulum.Date(1234, 5, 6)
    assert d.day == 6


def test_day_of_week():
    d = pendulum.Date(2012, 5, 7)
    assert d.day_of_week == pendulum.MONDAY


def test_day_of_year():
    d = pendulum.Date(2015, 12, 31)
    assert d.day_of_year == 365
    d = pendulum.Date(2016, 12, 31)
    assert d.day_of_year == 366


def test_days_in_month():
    d = pendulum.Date(2012, 5, 7)
    assert d.days_in_month == 31


def test_age():
    d = pendulum.Date.today()
    assert d.age == 0
    assert d.add(years=1).age == -1
    assert d.subtract(years=1).age == 1


def test_is_leap_year():
    assert pendulum.Date(2012, 1, 1).is_leap_year()
    assert not pendulum.Date(2011, 1, 1).is_leap_year()


def test_is_long_year():
    assert pendulum.Date(2015, 1, 1).is_long_year()
    assert not pendulum.Date(2016, 1, 1).is_long_year()


def test_week_of_month():
    assert pendulum.Date(2012, 9, 30).week_of_month == 5
    assert pendulum.Date(2012, 9, 28).week_of_month == 5
    assert pendulum.Date(2012, 9, 20).week_of_month == 4
    assert pendulum.Date(2012, 9, 8).week_of_month == 2
    assert pendulum.Date(2012, 9, 1).week_of_month == 1
    assert pendulum.date(2020, 1, 1).week_of_month == 1
    assert pendulum.date(2020, 1, 7).week_of_month == 2
    assert pendulum.date(2020, 1, 14).week_of_month == 3


def test_week_of_year_first_week():
    assert pendulum.Date(2012, 1, 1).week_of_year == 52
    assert pendulum.Date(2012, 1, 2).week_of_year == 1


def test_week_of_year_last_week():
    assert pendulum.Date(2012, 12, 30).week_of_year == 52
    assert pendulum.Date(2012, 12, 31).week_of_year == 1


def test_is_future():
    d = pendulum.Date.today()
    assert not d.is_future()
    d = d.add(days=1)
    assert d.is_future()


def test_is_past():
    d = pendulum.Date.today()
    assert not d.is_past()
    d = d.subtract(days=1)
    assert d.is_past()


# === tests/date/test_day_of_week_modifiers.py ===
from __future__ import annotations

import pytest

import pendulum

from pendulum.exceptions import PendulumException
from tests.conftest import assert_date


def test_start_of_week():
    d = pendulum.date(1980, 8, 7).start_of("week")
    assert_date(d, 1980, 8, 4)


def test_start_of_week_from_week_start():
    d = pendulum.date(1980, 8, 4).start_of("week")
    assert_date(d, 1980, 8, 4)


def test_start_of_week_crossing_year_boundary():
    d = pendulum.date(2014, 1, 1).start_of("week")
    assert_date(d, 2013, 12, 30)


def test_end_of_week():
    d = pendulum.date(1980, 8, 7).end_of("week")
    assert_date(d, 1980, 8, 10)


def test_end_of_week_from_week_end():
    d = pendulum.date(1980, 8, 10).end_of("week")
    assert_date(d, 1980, 8, 10)


def test_end_of_week_crossing_year_boundary():
    d = pendulum.date(2013, 12, 31).end_of("week")
    assert_date(d, 2014, 1, 5)


def test_next():
    d = pendulum.date(1975, 5, 21).next()
    assert_date(d, 1975, 5, 28)


def test_next_monday():
    d = pendulum.date(1975, 5, 21).next(pendulum.MONDAY)
    assert_date(d, 1975, 5, 26)


def test_next_saturday():
    d = pendulum.date(1975, 5, 21).next(5)
    assert_date(d, 1975, 5, 24)


def test_next_invalid():
    dt = pendulum.date(1975, 5, 21)

    with pytest.raises(ValueError):
        dt.next(7)


def test_previous():
    d = pendulum.date(1975, 5, 21).previous()
    assert_date(d, 1975, 5, 14)


def test_previous_monday():
    d = pendulum.date(1975, 5, 21).previous(pendulum.MONDAY)
    assert_date(d, 1975, 5, 19)


def test_previous_saturday():
    d = pendulum.date(1975, 5, 21).previous(5)
    assert_date(d, 1975, 5, 17)


def test_previous_invalid():
    dt = pendulum.date(1975, 5, 21)

    with pytest.raises(ValueError):
        dt.previous(7)


def test_first_day_of_month():
    d = pendulum.date(1975, 11, 21).first_of("month")
    assert_date(d, 1975, 11, 1)


def test_first_wednesday_of_month():
    d = pendulum.date(1975, 11, 21).first_of("month", pendulum.WEDNESDAY)
    assert_date(d, 1975, 11, 5)


def test_first_friday_of_month():
    d = pendulum.date(1975, 11, 21).first_of("month", 4)
    assert_date(d, 1975, 11, 7)


def test_last_day_of_month():
    d = pendulum.date(1975, 12, 5).last_of("month")
    assert_date(d, 1975, 12, 31)


def test_last_tuesday_of_month():
    d = pendulum.date(1975, 12, 1).last_of("month", pendulum.TUESDAY)
    assert_date(d, 1975, 12, 30)


def test_last_friday_of_month():
    d = pendulum.date(1975, 12, 5).last_of("month", 4)
    assert_date(d, 1975, 12, 26)


def test_nth_of_month_outside_scope():
    d = pendulum.date(1975, 6, 5)

    with pytest.raises(PendulumException):
        d.nth_of("month", 6, pendulum.MONDAY)


def test_nth_of_month_outside_year():
    d = pendulum.date(1975, 12, 5)

    with pytest.raises(PendulumException):
        d.nth_of("month", 55, pendulum.MONDAY)


def test_nth_of_month_first():
    d = pendulum.date(1975, 12, 5).nth_of("month", 1, pendulum.MONDAY)

    assert_date(d, 1975, 12, 1)


def test_2nd_monday_of_month():
    d = pendulum.date(1975, 12, 5).nth_of("month", 2, pendulum.MONDAY)

    assert_date(d, 1975, 12, 8)


def test_3rd_wednesday_of_month():
    d = pendulum.date(1975, 12, 5).nth_of("month", 3, 2)

    assert_date(d, 1975, 12, 17)


def test_first_day_of_quarter():
    d = pendulum.date(1975, 11, 21).first_of("quarter")
    assert_date(d, 1975, 10, 1)


def test_first_wednesday_of_quarter():
    d = pendulum.date(1975, 11, 21).first_of("quarter", pendulum.WEDNESDAY)
    assert_date(d, 1975, 10, 1)


def test_first_friday_of_quarter():
    d = pendulum.date(1975, 11, 21).first_of("quarter", 4)
    assert_date(d, 1975, 10, 3)


def test_first_of_quarter_from_a_day_that_will_not_exist_in_the_first_month():
    d = pendulum.date(2014, 5, 31).first_of("quarter")
    assert_date(d, 2014, 4, 1)


def test_last_day_of_quarter():
    d = pendulum.date(1975, 8, 5).last_of("quarter")
    assert_date(d, 1975, 9, 30)


def test_last_tuesday_of_quarter():
    d = pendulum.date(1975, 8, 5).last_of("quarter", pendulum.TUESDAY)
    assert_date(d, 1975, 9, 30)


def test_last_friday_of_quarter():
    d = pendulum.date(1975, 8, 5).last_of("quarter", pendulum.FRIDAY)
    assert_date(d, 1975, 9, 26)


def test_last_day_of_quarter_that_will_not_exist_in_the_last_month():
    d = pendulum.date(2014, 5, 31).last_of("quarter")
    assert_date(d, 2014, 6, 30)


def test_nth_of_quarter_outside_scope():
    d = pendulum.date(1975, 1, 5)

    with pytest.raises(PendulumException):
        d.nth_of("quarter", 20, pendulum.MONDAY)


def test_nth_of_quarter_outside_year():
    d = pendulum.date(1975, 1, 5)

    with pytest.raises(PendulumException):
        d.nth_of("quarter", 55, pendulum.MONDAY)


def test_nth_of_quarter_first():
    d = pendulum.date(1975, 12, 5).nth_of("quarter", 1, pendulum.MONDAY)

    assert_date(d, 1975, 10, 6)


def test_nth_of_quarter_from_a_day_that_will_not_exist_in_the_first_month():
    d = pendulum.date(2014, 5, 31).nth_of("quarter", 2, pendulum.MONDAY)
    assert_date(d, 2014, 4, 14)


def test_2nd_monday_of_quarter():
    d = pendulum.date(1975, 8, 5).nth_of("quarter", 2, pendulum.MONDAY)
    assert_date(d, 1975, 7, 14)


def test_3rd_wednesday_of_quarter():
    d = pendulum.date(1975, 8, 5).nth_of("quarter", 3, 2)
    assert_date(d, 1975, 7, 16)


def test_first_day_of_year():
    d = pendulum.date(1975, 11, 21).first_of("year")
    assert_date(d, 1975, 1, 1)


def test_first_wednesday_of_year():
    d = pendulum.date(1975, 11, 21).first_of("year", pendulum.WEDNESDAY)
    assert_date(d, 1975, 1, 1)


def test_first_friday_of_year():
    d = pendulum.date(1975, 11, 21).first_of("year", 4)
    assert_date(d, 1975, 1, 3)


def test_last_day_of_year():
    d = pendulum.date(1975, 8, 5).last_of("year")
    assert_date(d, 1975, 12, 31)


def test_last_tuesday_of_year():
    d = pendulum.date(1975, 8, 5).last_of("year", pendulum.TUESDAY)
    assert_date(d, 1975, 12, 30)


def test_last_friday_of_year():
    d = pendulum.date(1975, 8, 5).last_of("year", 4)
    assert_date(d, 1975, 12, 26)


def test_nth_of_year_outside_scope():
    d = pendulum.date(1975, 1, 5)

    with pytest.raises(PendulumException):
        d.nth_of("year", 55, pendulum.MONDAY)


def test_nth_of_year_first():
    d = pendulum.date(1975, 12, 5).nth_of("year", 1, pendulum.MONDAY)

    assert_date(d, 1975, 1, 6)


def test_2nd_monday_of_year():
    d = pendulum.date(1975, 8, 5).nth_of("year", 2, pendulum.MONDAY)
    assert_date(d, 1975, 1, 13)


def test_2rd_wednesday_of_year():
    d = pendulum.date(1975, 8, 5).nth_of("year", 3, pendulum.WEDNESDAY)
    assert_date(d, 1975, 1, 15)


def test_7th_thursday_of_year():
    d = pendulum.date(1975, 8, 31).nth_of("year", 7, pendulum.THURSDAY)
    assert_date(d, 1975, 2, 13)


def test_first_of_invalid_unit():
    d = pendulum.date(1975, 8, 5)

    with pytest.raises(ValueError):
        d.first_of("invalid", 3)


def test_last_of_invalid_unit():
    d = pendulum.date(1975, 8, 5)

    with pytest.raises(ValueError):
        d.last_of("invalid", 3)


def test_nth_of_invalid_unit():
    d = pendulum.date(1975, 8, 5)

    with pytest.raises(ValueError):
        d.nth_of("invalid", 3, pendulum.MONDAY)


# === tests/date/test_comparison.py ===
from __future__ import annotations

from datetime import date

import pendulum

from tests.conftest import assert_date


def test_equal_to_true():
    d1 = pendulum.Date(2000, 1, 1)
    d2 = pendulum.Date(2000, 1, 1)
    d3 = date(2000, 1, 1)

    assert d2 == d1
    assert d3 == d1


def test_equal_to_false():
    d1 = pendulum.Date(2000, 1, 1)
    d2 = pendulum.Date(2000, 1, 2)
    d3 = date(2000, 1, 2)

    assert d1 != d2
    assert d1 != d3


def test_not_equal_to_true():
    d1 = pendulum.Date(2000, 1, 1)
    d2 = pendulum.Date(2000, 1, 2)
    d3 = date(2000, 1, 2)

    assert d1 != d2
    assert d1 != d3


def test_not_equal_to_false():
    d1 = pendulum.Date(2000, 1, 1)
    d2 = pendulum.Date(2000, 1, 1)
    d3 = date(2000, 1, 1)

    assert d2 == d1
    assert d3 == d1


def test_not_equal_to_none():
    d1 = pendulum.Date(2000, 1, 1)

    assert d1 is not None


def test_greater_than_true():
    d1 = pendulum.Date(2000, 1, 1)
    d2 = pendulum.Date(1999, 12, 31)
    d3 = date(1999, 12, 31)

    assert d1 > d2
    assert d1 > d3


def test_greater_than_false():
    d1 = pendulum.Date(2000, 1, 1)
    d2 = pendulum.Date(2000, 1, 2)
    d3 = date(2000, 1, 2)

    assert not d1 > d2
    assert not d1 > d3


def test_greater_than_or_equal_true():
    d1 = pendulum.Date(2000, 1, 1)
    d2 = pendulum.Date(1999, 12, 31)
    d3 = date(1999, 12, 31)

    assert d1 >= d2
    assert d1 >= d3


def test_greater_than_or_equal_true_equal():
    d1 = pendulum.Date(2000, 1, 1)
    d2 = pendulum.Date(2000, 1, 1)
    d3 = date(2000, 1, 1)

    assert d1 >= d2
    assert d1 >= d3


def test_greater_than_or_equal_false():
    d1 = pendulum.Date(2000, 1, 1)
    d2 = pendulum.Date(2000, 1, 2)
    d3 = date(2000, 1, 2)

    assert not d1 >= d2
    assert not d1 >= d3


def test_less_than_true():
    d1 = pendulum.Date(2000, 1, 1)
    d2 = pendulum.Date(2000, 1, 2)
    d3 = date(2000, 1, 2)

    assert d1 < d2
    assert d1 < d3


def test_less_than_false():
    d1 = pendulum.Date(2000, 1, 2)
    d2 = pendulum.Date(2000, 1, 1)
    d3 = date(2000, 1, 1)

    assert not d1 < d2
    assert not d1 < d3


def test_less_than_or_equal_true():
    d1 = pendulum.Date(2000, 1, 1)
    d2 = pendulum.Date(2000, 1, 2)
    d3 = date(2000, 1, 2)

    assert d1 <= d2
    assert d1 <= d3


def test_less_than_or_equal_true_equal():
    d1 = pendulum.Date(2000, 1, 1)
    d2 = pendulum.Date(2000, 1, 1)
    d3 = date(2000, 1, 1)

    assert d1 <= d2
    assert d1 <= d3


def test_less_than_or_equal_false():
    d1 = pendulum.Date(2000, 1, 2)
    d2 = pendulum.Date(2000, 1, 1)
    d3 = date(2000, 1, 1)

    assert not d1 <= d2
    assert not d1 <= d3


def test_is_anniversary():
    d = pendulum.Date.today()
    an_anniversary = d.subtract(years=1)
    assert an_anniversary.is_anniversary()
    not_an_anniversary = d.subtract(days=1)
    assert not not_an_anniversary.is_anniversary()
    also_not_an_anniversary = d.add(days=2)
    assert not also_not_an_anniversary.is_anniversary()

    d1 = pendulum.Date(1987, 4, 23)
    d2 = pendulum.Date(2014, 9, 26)
    d3 = pendulum.Date(2014, 4, 23)
    assert not d2.is_anniversary(d1)
    assert d3.is_anniversary(d1)


def test_is_birthday():  # backward compatibility
    d = pendulum.Date.today()
    an_anniversary = d.subtract(years=1)
    assert an_anniversary.is_birthday()
    not_an_anniversary = d.subtract(days=1)
    assert not not_an_anniversary.is_birthday()
    also_not_an_anniversary = d.add(days=2)
    assert not also_not_an_anniversary.is_birthday()

    d1 = pendulum.Date(1987, 4, 23)
    d2 = pendulum.Date(2014, 9, 26)
    d3 = pendulum.Date(2014, 4, 23)
    assert not d2.is_birthday(d1)
    assert d3.is_birthday(d1)


def test_closest():
    instance = pendulum.Date(2015, 5, 28)
    dt1 = pendulum.Date(2015, 5, 27)
    dt2 = pendulum.Date(2015, 5, 30)
    closest = instance.closest(dt1, dt2)
    assert closest == dt1

    closest = instance.closest(dt2, dt1)
    assert closest == dt1


def test_closest_with_date():
    instance = pendulum.Date(2015, 5, 28)
    dt1 = date(2015, 5, 27)
    dt2 = date(2015, 5, 30)
    closest = instance.closest(dt1, dt2)
    assert isinstance(closest, pendulum.Date)
    assert_date(closest, 2015, 5, 27)


def test_closest_with_equals():
    instance = pendulum.Date(2015, 5, 28)
    dt1 = pendulum.Date(2015, 5, 28)
    dt2 = pendulum.Date(2015, 5, 30)
    closest = instance.closest(dt1, dt2)
    assert closest == dt1


def test_farthest():
    instance = pendulum.Date(2015, 5, 28)
    dt1 = pendulum.Date(2015, 5, 27)
    dt2 = pendulum.Date(2015, 5, 30)
    closest = instance.farthest(dt1, dt2)
    assert closest == dt2

    closest = instance.farthest(dt2, dt1)
    assert closest == dt2


def test_farthest_with_date():
    instance = pendulum.Date(2015, 5, 28)
    dt1 = date(2015, 5, 27)
    dt2 = date(2015, 5, 30)
    closest = instance.farthest(dt1, dt2)
    assert isinstance(closest, pendulum.Date)
    assert_date(closest, 2015, 5, 30)


def test_farthest_with_equals():
    instance = pendulum.Date(2015, 5, 28)
    dt1 = pendulum.Date(2015, 5, 28)
    dt2 = pendulum.Date(2015, 5, 30)
    closest = instance.farthest(dt1, dt2)
    assert closest == dt2


def test_is_same_day():
    dt1 = pendulum.Date(2015, 5, 28)
    dt2 = pendulum.Date(2015, 5, 29)
    dt3 = pendulum.Date(2015, 5, 28)
    dt4 = date(2015, 5, 28)
    dt5 = date(2015, 5, 29)

    assert not dt1.is_same_day(dt2)
    assert dt1.is_same_day(dt3)
    assert dt1.is_same_day(dt4)
    assert not dt1.is_same_day(dt5)


def test_comparison_to_unsupported():
    dt1 = pendulum.Date.today()

    assert dt1 != "test"
    assert dt1 not in ["test"]


# === tests/date/__init__.py ===


# === tests/date/test_add.py ===
from __future__ import annotations

from datetime import timedelta

import pytest

import pendulum

from tests.conftest import assert_date


def test_add_years_positive():
    assert pendulum.date(1975, 1, 1).add(years=1).year == 1976


def test_add_years_zero():
    assert pendulum.date(1975, 1, 1).add(years=0).year == 1975


def test_add_years_negative():
    assert pendulum.date(1975, 1, 1).add(years=-1).year == 1974


def test_add_months_positive():
    assert pendulum.date(1975, 12, 1).add(months=1).month == 1


def test_add_months_zero():
    assert pendulum.date(1975, 12, 1).add(months=0).month == 12


def test_add_months_negative():
    assert pendulum.date(1975, 12, 1).add(months=-1).month == 11


def test_add_month_with_overflow():
    assert pendulum.Date(2012, 1, 31).add(months=1).month == 2


def test_add_days_positive():
    assert pendulum.Date(1975, 5, 31).add(days=1).day == 1


def test_add_days_zero():
    assert pendulum.Date(1975, 5, 31).add(days=0).day == 31


def test_add_days_negative():
    assert pendulum.Date(1975, 5, 31).add(days=-1).day == 30


def test_add_weeks_positive():
    assert pendulum.Date(1975, 5, 21).add(weeks=1).day == 28


def test_add_weeks_zero():
    assert pendulum.Date(1975, 5, 21).add(weeks=0).day == 21


def test_add_weeks_negative():
    assert pendulum.Date(1975, 5, 21).add(weeks=-1).day == 14


def test_add_timedelta():
    delta = timedelta(days=18)
    d = pendulum.date(2015, 3, 14)

    new = d + delta
    assert isinstance(new, pendulum.Date)
    assert_date(new, 2015, 4, 1)


def test_add_duration():
    duration = pendulum.duration(years=2, months=3, days=18)
    d = pendulum.Date(2015, 3, 14)

    new = d + duration
    assert_date(new, 2017, 7, 2)


def test_addition_invalid_type():
    d = pendulum.date(2015, 3, 14)

    with pytest.raises(TypeError):
        d + 3

    with pytest.raises(TypeError):
        3 + d


# === tests/date/test_behavior.py ===
from __future__ import annotations

import pickle

from datetime import date

import pytest

import pendulum


@pytest.fixture()
def p():
    return pendulum.Date(2016, 8, 27)


@pytest.fixture()
def d():
    return date(2016, 8, 27)


def test_timetuple(p, d):
    assert p.timetuple() == d.timetuple()


def test_ctime(p, d):
    assert p.ctime() == d.ctime()


def test_isoformat(p, d):
    assert p.isoformat() == d.isoformat()


def test_toordinal(p, d):
    assert p.toordinal() == d.toordinal()


def test_weekday(p, d):
    assert p.weekday() == d.weekday()


def test_isoweekday(p, d):
    assert p.isoweekday() == d.isoweekday()


def test_isocalendar(p, d):
    assert p.isocalendar() == d.isocalendar()


def test_fromtimestamp():
    assert pendulum.Date.fromtimestamp(0) == date.fromtimestamp(0)


def test_fromordinal():
    assert pendulum.Date.fromordinal(730120) == date.fromordinal(730120)


def test_hash():
    d1 = pendulum.Date(2016, 8, 27)
    d2 = pendulum.Date(2016, 8, 27)
    d3 = pendulum.Date(2016, 8, 28)

    assert hash(d2) == hash(d1)
    assert hash(d1) != hash(d3)


def test_pickle():
    d1 = pendulum.Date(2016, 8, 27)
    s = pickle.dumps(d1)
    d2 = pickle.loads(s)

    assert isinstance(d2, pendulum.Date)
    assert d2 == d1


# === tests/date/test_diff.py ===
from __future__ import annotations

from datetime import date

import pytest

import pendulum


@pytest.fixture
def today():
    return pendulum.Date.today()


def test_diff_in_years_positive():
    dt = pendulum.date(2000, 1, 1)
    assert dt.diff(dt.add(years=1)).in_years() == 1


def test_diff_in_years_negative_with_sign():
    dt = pendulum.date(2000, 1, 1)
    assert dt.diff(dt.subtract(years=1), False).in_years() == -1


def test_diff_in_years_negative_no_sign():
    dt = pendulum.date(2000, 1, 1)
    assert dt.diff(dt.subtract(years=1)).in_years() == 1


def test_diff_in_years_vs_default_now(today):
    assert today.subtract(years=1).diff().in_years() == 1


def test_diff_in_years_ensure_is_truncated():
    dt = pendulum.date(2000, 1, 1)
    assert dt.diff(dt.add(years=1).add(months=7)).in_years() == 1


def test_diff_in_months_positive():
    dt = pendulum.date(2000, 1, 1)
    assert dt.diff(dt.add(years=1).add(months=1)).in_months() == 13


def test_diff_in_months_negative_with_sign():
    dt = pendulum.date(2000, 1, 1)

    assert dt.diff(dt.subtract(years=1).add(months=1), False).in_months() == -11


def test_diff_in_months_negative_no_sign():
    dt = pendulum.date(2000, 1, 1)
    assert dt.diff(dt.subtract(years=1).add(months=1)).in_months() == 11


def test_diff_in_months_vs_default_now(today):
    assert today.subtract(years=1).diff().in_months() == 12


def test_diff_in_months_ensure_is_truncated():
    dt = pendulum.date(2000, 1, 1)
    assert dt.diff(dt.add(months=1).add(days=16)).in_months() == 1


def test_diff_in_days_positive():
    dt = pendulum.date(2000, 1, 1)
    assert dt.diff(dt.add(years=1)).in_days() == 366


def test_diff_in_days_negative_with_sign():
    dt = pendulum.date(2000, 1, 1)
    assert dt.diff(dt.subtract(years=1), False).in_days() == -365


def test_diff_in_days_negative_no_sign():
    dt = pendulum.date(2000, 1, 1)
    assert dt.diff(dt.subtract(years=1)).in_days() == 365


def test_diff_in_days_vs_default_now(today):
    assert today.subtract(weeks=1).diff().in_days() == 7


def test_diff_in_weeks_positive():
    dt = pendulum.date(2000, 1, 1)
    assert dt.diff(dt.add(years=1)).in_weeks() == 52


def test_diff_in_weeks_negative_with_sign():
    dt = pendulum.date(2000, 1, 1)
    assert dt.diff(dt.subtract(years=1), False).in_weeks() == -52


def test_diff_in_weeks_negative_no_sign():
    dt = pendulum.date(2000, 1, 1)
    assert dt.diff(dt.subtract(years=1)).in_weeks() == 52


def test_diff_in_weeks_vs_default_now(today):
    assert today.subtract(weeks=1).diff().in_weeks() == 1


def test_diff_in_weeks_ensure_is_truncated():
    dt = pendulum.date(2000, 1, 1)
    assert dt.diff(dt.add(weeks=1).subtract(days=1)).in_weeks() == 0


def test_diff_for_humans_now_and_day(today):
    assert today.subtract(days=1).diff_for_humans() == "1 day ago"


def test_diff_for_humans_now_and_days(today):
    assert today.subtract(days=2).diff_for_humans() == "2 days ago"


def test_diff_for_humans_now_and_nearly_week(today):
    assert today.subtract(days=6).diff_for_humans() == "6 days ago"


def test_diff_for_humans_now_and_week(today):
    assert today.subtract(weeks=1).diff_for_humans() == "1 week ago"


def test_diff_for_humans_now_and_weeks(today):
    assert today.subtract(weeks=2).diff_for_humans() == "2 weeks ago"


def test_diff_for_humans_now_and_nearly_month(today):
    assert today.subtract(weeks=3).diff_for_humans() == "3 weeks ago"


def test_diff_for_humans_now_and_month():
    with pendulum.travel_to(pendulum.datetime(2016, 4, 1)):
        today = pendulum.today().date()

        assert today.subtract(weeks=4).diff_for_humans() == "4 weeks ago"
        assert today.subtract(months=1).diff_for_humans() == "1 month ago"

    with pendulum.travel_to(pendulum.datetime(2017, 3, 1)):
        today = pendulum.today().date()

        assert today.subtract(weeks=4).diff_for_humans() == "1 month ago"


def test_diff_for_humans_now_and_months(today):
    assert today.subtract(months=2).diff_for_humans() == "2 months ago"


def test_diff_for_humans_now_and_nearly_year(today):
    assert today.subtract(months=11).diff_for_humans() == "11 months ago"


def test_diff_for_humans_now_and_year(today):
    assert today.subtract(years=1).diff_for_humans() == "1 year ago"


def test_diff_for_humans_now_and_years(today):
    assert today.subtract(years=2).diff_for_humans() == "2 years ago"


def test_diff_for_humans_now_and_future_day(today):
    assert today.add(days=1).diff_for_humans() == "in 1 day"


def test_diff_for_humans_now_and_future_days(today):
    assert today.add(days=2).diff_for_humans() == "in 2 days"


def test_diff_for_humans_now_and_nearly_future_week(today):
    assert today.add(days=6).diff_for_humans() == "in 6 days"


def test_diff_for_humans_now_and_future_week(today):
    assert today.add(weeks=1).diff_for_humans() == "in 1 week"


def test_diff_for_humans_now_and_future_weeks(today):
    assert today.add(weeks=2).diff_for_humans() == "in 2 weeks"


def test_diff_for_humans_now_and_nearly_future_month(today):
    assert today.add(weeks=3).diff_for_humans() == "in 3 weeks"


def test_diff_for_humans_now_and_future_month():
    with pendulum.travel_to(pendulum.datetime(2016, 3, 1)):
        today = pendulum.today("UTC").date()

        assert today.add(weeks=4).diff_for_humans() == "in 4 weeks"
        assert today.add(months=1).diff_for_humans() == "in 1 month"

    with pendulum.travel_to(pendulum.datetime(2017, 3, 31)):
        today = pendulum.today("UTC").date()

        assert today.add(months=1).diff_for_humans() == "in 1 month"

    with pendulum.travel_to(pendulum.datetime(2017, 4, 30)):
        today = pendulum.today("UTC").date()

        assert today.add(months=1).diff_for_humans() == "in 1 month"

    with pendulum.travel_to(pendulum.datetime(2017, 1, 31)):
        today = pendulum.today("UTC").date()

        assert today.add(weeks=4).diff_for_humans() == "in 1 month"


def test_diff_for_humans_now_and_future_months(today):
    assert today.add(months=2).diff_for_humans() == "in 2 months"


def test_diff_for_humans_now_and_nearly_future_year(today):
    assert today.add(months=11).diff_for_humans() == "in 11 months"


def test_diff_for_humans_now_and_future_year(today):
    assert today.add(years=1).diff_for_humans() == "in 1 year"


def test_diff_for_humans_now_and_future_years(today):
    assert today.add(years=2).diff_for_humans() == "in 2 years"


def test_diff_for_humans_other_and_day(today):
    assert today.diff_for_humans(today.add(days=1)) == "1 day before"


def test_diff_for_humans_other_and_days(today):
    assert today.diff_for_humans(today.add(days=2)) == "2 days before"


def test_diff_for_humans_other_and_nearly_week(today):
    assert today.diff_for_humans(today.add(days=6)) == "6 days before"


def test_diff_for_humans_other_and_week(today):
    assert today.diff_for_humans(today.add(weeks=1)) == "1 week before"


def test_diff_for_humans_other_and_weeks(today):
    assert today.diff_for_humans(today.add(weeks=2)) == "2 weeks before"


def test_diff_for_humans_other_and_nearly_month(today):
    assert today.diff_for_humans(today.add(weeks=3)) == "3 weeks before"


def test_diff_for_humans_other_and_month():
    with pendulum.travel_to(pendulum.datetime(2016, 3, 1)):
        today = pendulum.today().date()

        assert today.diff_for_humans(today.add(weeks=4)) == "4 weeks before"
        assert today.diff_for_humans(today.add(months=1)) == "1 month before"

    with pendulum.travel_to(pendulum.datetime(2017, 3, 31)):
        today = pendulum.today().date()

        assert today.diff_for_humans(today.add(months=1)) == "1 month before"

    with pendulum.travel_to(pendulum.datetime(2017, 4, 30)):
        today = pendulum.today().date()

        assert today.diff_for_humans(today.add(months=1)) == "1 month before"

    with pendulum.travel_to(pendulum.datetime(2017, 1, 31)):
        today = pendulum.today().date()

        assert today.diff_for_humans(today.add(weeks=4)) == "1 month before"


def test_diff_for_humans_other_and_months(today):
    assert today.diff_for_humans(today.add(months=2)) == "2 months before"


def test_diff_for_humans_other_and_nearly_year(today):
    assert today.diff_for_humans(today.add(months=11)) == "11 months before"


def test_diff_for_humans_other_and_year(today):
    assert today.diff_for_humans(today.add(years=1)) == "1 year before"


def test_diff_for_humans_other_and_years(today):
    assert today.diff_for_humans(today.add(years=2)) == "2 years before"


def test_diff_for_humans_other_and_future_day(today):
    assert today.diff_for_humans(today.subtract(days=1)) == "1 day after"


def test_diff_for_humans_other_and_future_days(today):
    assert today.diff_for_humans(today.subtract(days=2)) == "2 days after"


def test_diff_for_humans_other_and_nearly_future_week(today):
    assert today.diff_for_humans(today.subtract(days=6)) == "6 days after"


def test_diff_for_humans_other_and_future_week(today):
    assert today.diff_for_humans(today.subtract(weeks=1)) == "1 week after"


def test_diff_for_humans_other_and_future_weeks(today):
    assert today.diff_for_humans(today.subtract(weeks=2)) == "2 weeks after"


def test_diff_for_humans_other_and_nearly_future_month(today):
    assert today.diff_for_humans(today.subtract(weeks=3)) == "3 weeks after"


def test_diff_for_humans_other_and_future_month():
    with pendulum.travel_to(pendulum.datetime(2016, 3, 1)):
        today = pendulum.today().date()

        assert today.diff_for_humans(today.subtract(weeks=4)) == "4 weeks after"
        assert today.diff_for_humans(today.subtract(months=1)) == "1 month after"

    with pendulum.travel_to(pendulum.datetime(2017, 2, 28)):
        today = pendulum.today().date()

        assert today.diff_for_humans(today.subtract(weeks=4)) == "1 month after"


def test_diff_for_humans_other_and_future_months(today):
    assert today.diff_for_humans(today.subtract(months=2)) == "2 months after"


def test_diff_for_humans_other_and_nearly_future_year(today):
    assert today.diff_for_humans(today.subtract(months=11)) == "11 months after"


def test_diff_for_humans_other_and_future_year(today):
    assert today.diff_for_humans(today.subtract(years=1)) == "1 year after"


def test_diff_for_humans_other_and_future_years(today):
    assert today.diff_for_humans(today.subtract(years=2)) == "2 years after"


def test_diff_for_humans_absolute_days(today):
    assert today.diff_for_humans(today.subtract(days=2), True) == "2 days"
    assert today.diff_for_humans(today.add(days=2), True) == "2 days"


def test_diff_for_humans_absolute_weeks(today):
    assert today.diff_for_humans(today.subtract(weeks=2), True) == "2 weeks"
    assert today.diff_for_humans(today.add(weeks=2), True) == "2 weeks"


def test_diff_for_humans_absolute_months(today):
    assert today.diff_for_humans(today.subtract(months=2), True) == "2 months"
    assert today.diff_for_humans(today.add(months=2), True) == "2 months"


def test_diff_for_humans_absolute_years(today):
    assert today.diff_for_humans(today.subtract(years=1), True) == "1 year"
    assert today.diff_for_humans(today.add(years=1), True) == "1 year"


def test_subtraction():
    d = pendulum.date(2016, 7, 5)
    future_dt = date(2016, 7, 6)
    future = d.add(days=1)

    assert (future - d).total_seconds() == 86400
    assert (future_dt - d).total_seconds() == 86400


# === tests/date/test_strings.py ===
from __future__ import annotations

import pendulum


def test_to_string():
    d = pendulum.Date(2016, 10, 16)
    assert str(d) == "2016-10-16"


def test_to_date_string():
    d = pendulum.Date(1975, 12, 25)
    assert d.to_date_string() == "1975-12-25"


def test_to_formatted_date_string():
    d = pendulum.Date(1975, 12, 25)
    assert d.to_formatted_date_string() == "Dec 25, 1975"


def test_repr():
    d = pendulum.Date(1975, 12, 25)

    assert repr(d) == "Date(1975, 12, 25)"
    assert d.__repr__() == "Date(1975, 12, 25)"


def test_format_with_locale():
    d = pendulum.Date(1975, 12, 25)
    expected = "jeudi 25e jour de décembre 1975"
    assert d.format("dddd Do [jour de] MMMM YYYY", locale="fr") == expected


def test_strftime():
    d = pendulum.Date(1975, 12, 25)
    assert d.strftime("%d") == "25"


def test_for_json():
    d = pendulum.Date(1975, 12, 25)
    assert d.for_json() == "1975-12-25"


def test_format():
    d = pendulum.Date(1975, 12, 25)
    assert f"{d}" == "1975-12-25"
    assert f"{d:YYYY}" == "1975"
    assert f"{d:%Y}" == "1975"


# === tests/date/test_fluent_setters.py ===
from __future__ import annotations

import pendulum

from tests.conftest import assert_date


def test_fluid_year_setter():
    d = pendulum.Date(2016, 10, 20)
    new = d.set(year=1995)

    assert_date(new, 1995, 10, 20)
    assert new.year == 1995


def test_fluid_month_setter():
    d = pendulum.Date(2016, 7, 2)
    new = d.set(month=11)

    assert new.month == 11
    assert d.month == 7


def test_fluid_day_setter():
    d = pendulum.Date(2016, 7, 2)
    new = d.set(day=9)

    assert new.day == 9
    assert d.day == 2


# === tests/date/test_sub.py ===
from __future__ import annotations

from datetime import datetime
from datetime import timedelta

import pytest

import pendulum

from tests.conftest import assert_date


def test_subtract_years_positive():
    assert pendulum.date(1975, 1, 1).subtract(years=1).year == 1974


def test_subtract_years_zero():
    assert pendulum.date(1975, 1, 1).subtract(years=0).year == 1975


def test_subtract_years_negative():
    assert pendulum.date(1975, 1, 1).subtract(years=-1).year == 1976


def test_subtract_months_positive():
    assert pendulum.date(1975, 1, 1).subtract(months=1).month == 12


def test_subtract_months_zero():
    assert pendulum.date(1975, 12, 1).subtract(months=0).month == 12


def test_subtract_months_negative():
    assert pendulum.date(1975, 11, 1).subtract(months=-1).month == 12


def test_subtract_days_positive():
    assert pendulum.Date(1975, 6, 1).subtract(days=1).day == 31


def test_subtract_days_zero():
    assert pendulum.Date(1975, 5, 31).subtract(days=0).day == 31


def test_subtract_days_negative():
    assert pendulum.Date(1975, 5, 30).subtract(days=-1).day == 31


def test_subtract_days_max():
    delta = pendulum.now() - pendulum.instance(datetime.min)
    assert pendulum.now().subtract(days=delta.days - 1).year == 1


def test_subtract_weeks_positive():
    assert pendulum.Date(1975, 5, 28).subtract(weeks=1).day == 21


def test_subtract_weeks_zero():
    assert pendulum.Date(1975, 5, 21).subtract(weeks=0).day == 21


def test_subtract_weeks_negative():
    assert pendulum.Date(1975, 5, 14).subtract(weeks=-1).day == 21


def test_subtract_timedelta():
    delta = timedelta(days=18)
    d = pendulum.date(2015, 3, 14)

    new = d - delta
    assert isinstance(new, pendulum.Date)
    assert_date(new, 2015, 2, 24)


def test_subtract_duration():
    delta = pendulum.duration(years=2, months=3, days=18)
    d = pendulum.date(2015, 3, 14)

    new = d - delta
    assert_date(new, 2012, 11, 26)


def test_addition_invalid_type():
    d = pendulum.date(2015, 3, 14)

    with pytest.raises(TypeError):
        d - "ab"

    with pytest.raises(TypeError):
        "ab" - d


# === tests/time/test_construct.py ===
from __future__ import annotations

import pendulum

from tests.conftest import assert_time


def test_init():
    t = pendulum.time(12, 34, 56, 123456)

    assert_time(t, 12, 34, 56, 123456)


def test_init_with_missing_values():
    t = pendulum.time(12, 34, 56)
    assert_time(t, 12, 34, 56, 0)

    t = pendulum.time(12, 34)
    assert_time(t, 12, 34, 0, 0)

    t = pendulum.time(12)
    assert_time(t, 12, 0, 0, 0)


# === tests/time/test_comparison.py ===
from __future__ import annotations

from datetime import time

import pendulum

from tests.conftest import assert_time


def test_equal_to_true():
    t1 = pendulum.time(1, 2, 3)
    t2 = pendulum.time(1, 2, 3)
    t3 = time(1, 2, 3)

    assert t1 == t2
    assert t1 == t3


def test_equal_to_false():
    t1 = pendulum.time(1, 2, 3)
    t2 = pendulum.time(1, 2, 4)
    t3 = time(1, 2, 4)

    assert t1 != t2
    assert t1 != t3


def test_not_equal_to_none():
    t1 = pendulum.time(1, 2, 3)

    assert t1 is not None


def test_greater_than_true():
    t1 = pendulum.time(1, 2, 3)
    t2 = pendulum.time(1, 2, 2)
    t3 = time(1, 2, 2)

    assert t1 > t2
    assert t1 > t3


def test_greater_than_false():
    t1 = pendulum.time(1, 2, 2)
    t2 = pendulum.time(1, 2, 3)
    t3 = time(1, 2, 3)

    assert not t1 > t2
    assert not t1 > t3


def test_greater_than_or_equal_true():
    t1 = pendulum.time(1, 2, 3)
    t2 = pendulum.time(1, 2, 2)
    t3 = time(1, 2, 2)

    assert t1 >= t2
    assert t1 >= t3


def test_greater_than_or_equal_true_equal():
    t1 = pendulum.time(1, 2, 3)
    t2 = pendulum.time(1, 2, 3)
    t3 = time(1, 2, 3)

    assert t1 >= t2
    assert t1 >= t3


def test_greater_than_or_equal_false():
    t1 = pendulum.time(1, 2, 2)
    t2 = pendulum.time(1, 2, 3)
    t3 = time(1, 2, 3)

    assert not t1 >= t2
    assert not t1 >= t3


def test_less_than_true():
    t1 = pendulum.time(1, 2, 2)
    t2 = pendulum.time(1, 2, 3)
    t3 = time(1, 2, 3)

    assert t1 < t2
    assert t1 < t3


def test_less_than_false():
    t1 = pendulum.time(1, 2, 3)
    t2 = pendulum.time(1, 2, 2)
    t3 = time(1, 2, 2)

    assert not t1 < t2
    assert not t1 < t3


def test_less_than_or_equal_true():
    t1 = pendulum.time(1, 2, 2)
    t2 = pendulum.time(1, 2, 3)
    t3 = time(1, 2, 3)

    assert t1 <= t2
    assert t1 <= t3


def test_less_than_or_equal_true_equal():
    t1 = pendulum.time(1, 2, 3)
    t2 = pendulum.time(1, 2, 3)
    t3 = time(1, 2, 3)

    assert t1 <= t2
    assert t1 <= t3


def test_less_than_or_equal_false():
    t1 = pendulum.time(1, 2, 3)
    t2 = pendulum.time(1, 2, 2)
    t3 = time(1, 2, 2)

    assert not t1 <= t2
    assert not t1 <= t3


def test_closest():
    instance = pendulum.time(12, 34, 56)
    t1 = pendulum.time(12, 34, 54)
    t2 = pendulum.time(12, 34, 59)
    closest = instance.closest(t1, t2)
    assert t1 == closest

    closest = instance.closest(t2, t1)
    assert t1 == closest


def test_closest_with_time():
    instance = pendulum.time(12, 34, 56)
    t1 = pendulum.time(12, 34, 54)
    t2 = pendulum.time(12, 34, 59)
    closest = instance.closest(t1, t2)

    assert_time(closest, 12, 34, 54)


def test_closest_with_equals():
    instance = pendulum.time(12, 34, 56)
    t1 = pendulum.time(12, 34, 56)
    t2 = pendulum.time(12, 34, 59)
    closest = instance.closest(t1, t2)
    assert t1 == closest


def test_farthest():
    instance = pendulum.time(12, 34, 56)
    t1 = pendulum.time(12, 34, 54)
    t2 = pendulum.time(12, 34, 59)
    farthest = instance.farthest(t1, t2)
    assert t2 == farthest

    farthest = instance.farthest(t2, t1)
    assert t2 == farthest


def test_farthest_with_time():
    instance = pendulum.time(12, 34, 56)
    t1 = pendulum.time(12, 34, 54)
    t2 = pendulum.time(12, 34, 59)
    farthest = instance.farthest(t1, t2)

    assert_time(farthest, 12, 34, 59)


def test_farthest_with_equals():
    instance = pendulum.time(12, 34, 56)
    t1 = pendulum.time(12, 34, 56)
    t2 = pendulum.time(12, 34, 59)

    farthest = instance.farthest(t1, t2)
    assert t2 == farthest


def test_comparison_to_unsupported():
    t1 = pendulum.now().time()

    assert t1 != "test"
    assert t1 not in ["test"]


# === tests/time/__init__.py ===


# === tests/time/test_add.py ===
from __future__ import annotations

from datetime import timedelta

import pytest

import pendulum


def test_add_hours_positive():
    assert pendulum.time(12, 34, 56).add(hours=1).hour == 13


def test_add_hours_zero():
    assert pendulum.time(12, 34, 56).add(hours=0).hour == 12


def test_add_hours_negative():
    assert pendulum.time(12, 34, 56).add(hours=-1).hour == 11


def test_add_minutes_positive():
    assert pendulum.time(12, 34, 56).add(minutes=1).minute == 35


def test_add_minutes_zero():
    assert pendulum.time(12, 34, 56).add(minutes=0).minute == 34


def test_add_minutes_negative():
    assert pendulum.time(12, 34, 56).add(minutes=-1).minute == 33


def test_add_seconds_positive():
    assert pendulum.time(12, 34, 56).add(seconds=1).second == 57


def test_add_seconds_zero():
    assert pendulum.time(12, 34, 56).add(seconds=0).second == 56


def test_add_seconds_negative():
    assert pendulum.time(12, 34, 56).add(seconds=-1).second == 55


def test_add_timedelta():
    delta = timedelta(seconds=45, microseconds=123456)
    d = pendulum.time(3, 12, 15, 654321)

    d = d.add_timedelta(delta)
    assert d.minute == 13
    assert d.second == 0
    assert d.microsecond == 777777

    d = pendulum.time(3, 12, 15, 654321)

    d = d + delta
    assert d.minute == 13
    assert d.second == 0
    assert d.microsecond == 777777


def test_add_timedelta_with_days():
    delta = timedelta(days=3, seconds=45, microseconds=123456)
    d = pendulum.time(3, 12, 15, 654321)

    with pytest.raises(TypeError):
        d.add_timedelta(delta)


def test_addition_invalid_type():
    d = pendulum.time(3, 12, 15, 654321)

    with pytest.raises(TypeError):
        d + 3

    with pytest.raises(TypeError):
        3 + d


# === tests/time/test_behavior.py ===
from __future__ import annotations

import pickle

from datetime import time

import pytest

import pendulum

from pendulum import Time


@pytest.fixture()
def p():
    return pendulum.Time(12, 34, 56, 123456, tzinfo=pendulum.timezone("Europe/Paris"))


@pytest.fixture()
def d():
    return time(12, 34, 56, 123456, tzinfo=pendulum.timezone("Europe/Paris"))


def test_hash(p, d):
    assert hash(d) == hash(p)
    dt1 = Time(12, 34, 57, 123456)

    assert hash(p) != hash(dt1)


def test_pickle():
    dt1 = Time(12, 34, 56, 123456)
    s = pickle.dumps(dt1)
    dt2 = pickle.loads(s)

    assert dt2 == dt1


def test_utcoffset(p, d):
    assert d.utcoffset() == p.utcoffset()


def test_dst(p, d):
    assert d.dst() == p.dst()


def test_tzname(p, d):
    assert d.tzname() == p.tzname()
    assert Time(12, 34, 56, 123456).tzname() == time(12, 34, 56, 123456).tzname()


# === tests/time/test_diff.py ===
from __future__ import annotations

import pendulum

from pendulum import Time


def test_diff_in_hours_positive():
    dt = Time(12, 34, 56)
    assert dt.diff(dt.add(hours=2).add(seconds=3672)).in_hours() == 3


def test_diff_in_hours_negative_with_sign():
    dt = Time(12, 34, 56)
    assert dt.diff(dt.subtract(hours=2).add(seconds=3600), False).in_hours() == -1


def test_diff_in_hours_negative_no_sign():
    dt = Time(12, 34, 56)
    assert dt.diff(dt.subtract(hours=2).add(seconds=3600)).in_hours() == 1


def test_diff_in_hours_vs_default_now():
    with pendulum.travel_to(pendulum.today().at(12, 34, 56)):
        now = pendulum.now().time()

        assert now.subtract(hours=2).diff().in_hours() == 2


def test_diff_in_hours_ensure_is_truncated():
    dt = Time(12, 34, 56)
    assert dt.diff(dt.add(hours=2).add(seconds=5401)).in_hours() == 3


def test_diff_in_minutes_positive():
    dt = Time(12, 34, 56)
    assert dt.diff(dt.add(hours=1).add(minutes=2)).in_minutes() == 62


def test_diff_in_minutes_positive_big():
    dt = Time(12, 34, 56)
    assert dt.diff(dt.add(hours=25).add(minutes=2)).in_minutes() == 62


def test_diff_in_minutes_negative_with_sign():
    dt = Time(12, 34, 56)
    assert dt.diff(dt.subtract(hours=1).add(minutes=2), False).in_minutes() == -58


def test_diff_in_minutes_negative_no_sign():
    dt = Time(12, 34, 56)
    assert dt.diff(dt.subtract(hours=1).add(minutes=2)).in_minutes() == 58


def test_diff_in_minutes_vs_default_now():
    with pendulum.travel_to(pendulum.today().at(12, 34, 56)):
        now = pendulum.now().time()

        assert now.subtract(hours=1).diff().in_minutes() == 60


def test_diff_in_minutes_ensure_is_truncated():
    dt = Time(12, 34, 56)
    assert dt.diff(dt.add(minutes=1).add(seconds=59)).in_minutes() == 1


def test_diff_in_seconds_positive():
    dt = Time(12, 34, 56)
    assert dt.diff(dt.add(minutes=1).add(seconds=2)).in_seconds() == 62


def test_diff_in_seconds_positive_big():
    dt = Time(12, 34, 56)
    assert dt.diff(dt.add(hours=2).add(seconds=2)).in_seconds() == 7202


def test_diff_in_seconds_negative_with_sign():
    dt = Time(12, 34, 56)
    assert dt.diff(dt.subtract(minutes=1).add(seconds=2), False).in_seconds() == -58


def test_diff_in_seconds_negative_no_sign():
    dt = Time(12, 34, 56)
    assert dt.diff(dt.subtract(minutes=1).add(seconds=2)).in_seconds() == 58


def test_diff_in_seconds_vs_default_now():
    with pendulum.travel_to(pendulum.today().at(12, 34, 56)):
        now = pendulum.now().time()

        assert now.subtract(hours=1).diff().in_seconds() == 3600


def test_diff_in_seconds_ensure_is_truncated():
    dt = Time(12, 34, 56)
    assert dt.diff(dt.add(seconds=1.9)).in_seconds() == 1


def test_diff_for_humans_now_and_second():
    with pendulum.travel_to(pendulum.today().at(12, 34, 56)):
        now = pendulum.now().time()

        assert now.diff_for_humans() == "a few seconds ago"


def test_diff_for_humans_now_and_seconds():
    with pendulum.travel_to(pendulum.today().at(12, 34, 56)):
        now = pendulum.now().time()

        assert now.subtract(seconds=2).diff_for_humans() == "a few seconds ago"


def test_diff_for_humans_now_and_nearly_minute():
    with pendulum.travel_to(pendulum.today().at(12, 34, 56)):
        now = pendulum.now().time()

        assert now.subtract(seconds=59).diff_for_humans() == "59 seconds ago"


def test_diff_for_humans_now_and_minute():
    with pendulum.travel_to(pendulum.today().at(12, 34, 56)):
        now = pendulum.now().time()

        assert now.subtract(minutes=1).diff_for_humans() == "1 minute ago"


def test_diff_for_humans_now_and_minutes():
    with pendulum.travel_to(pendulum.today().at(12, 34, 56)):
        now = pendulum.now().time()

        assert now.subtract(minutes=2).diff_for_humans() == "2 minutes ago"


def test_diff_for_humans_now_and_nearly_hour():
    with pendulum.travel_to(pendulum.today().at(12, 34, 56)):
        now = pendulum.now().time()

        assert now.subtract(minutes=59).diff_for_humans() == "59 minutes ago"


def test_diff_for_humans_now_and_hour():
    with pendulum.travel_to(pendulum.today().at(12, 34, 56)):
        now = pendulum.now().time()

        assert now.subtract(hours=1).diff_for_humans() == "1 hour ago"


def test_diff_for_humans_now_and_hours():
    with pendulum.travel_to(pendulum.today().at(12, 34, 56)):
        now = pendulum.now().time()

        assert now.subtract(hours=2).diff_for_humans() == "2 hours ago"


def test_diff_for_humans_now_and_future_second():
    with pendulum.travel_to(pendulum.today().at(12, 34, 56)):
        now = pendulum.now().time()

        assert now.add(seconds=1).diff_for_humans() == "in a few seconds"


def test_diff_for_humans_now_and_future_seconds():
    with pendulum.travel_to(pendulum.today().at(12, 34, 56)):
        now = pendulum.now().time()

        assert now.add(seconds=2).diff_for_humans() == "in a few seconds"


def test_diff_for_humans_now_and_nearly_future_minute():
    with pendulum.travel_to(pendulum.today().at(12, 34, 56)):
        now = pendulum.now().time()

        assert now.add(seconds=59).diff_for_humans() == "in 59 seconds"


def test_diff_for_humans_now_and_future_minute():
    with pendulum.travel_to(pendulum.today().at(12, 34, 56)):
        now = pendulum.now().time()

        assert now.add(minutes=1).diff_for_humans() == "in 1 minute"


def test_diff_for_humans_now_and_future_minutes():
    with pendulum.travel_to(pendulum.today().at(12, 34, 56)):
        now = pendulum.now().time()

        assert now.add(minutes=2).diff_for_humans() == "in 2 minutes"


def test_diff_for_humans_now_and_nearly_future_hour():
    with pendulum.travel_to(pendulum.today().at(12, 34, 56)):
        now = pendulum.now().time()

        assert now.add(minutes=59).diff_for_humans() == "in 59 minutes"


def test_diff_for_humans_now_and_future_hour():
    with pendulum.travel_to(pendulum.today().at(12, 34, 56)):
        now = pendulum.now().time()

        assert now.add(hours=1).diff_for_humans() == "in 1 hour"


def test_diff_for_humans_now_and_future_hours():
    with pendulum.travel_to(pendulum.today().at(12, 34, 56)):
        now = pendulum.now().time()

        assert now.add(hours=2).diff_for_humans() == "in 2 hours"


def test_diff_for_humans_other_and_second():
    with pendulum.travel_to(pendulum.today().at(12, 34, 56)):
        now = pendulum.now().time()

        assert now.diff_for_humans(now.add(seconds=1)) == "a few seconds before"


def test_diff_for_humans_other_and_seconds():
    with pendulum.travel_to(pendulum.today().at(12, 34, 56)):
        now = pendulum.now().time()

        assert now.diff_for_humans(now.add(seconds=2)) == "a few seconds before"


def test_diff_for_humans_other_and_nearly_minute():
    with pendulum.travel_to(pendulum.today().at(12, 34, 56)):
        now = pendulum.now().time()

        assert now.diff_for_humans(now.add(seconds=59)) == "59 seconds before"


def test_diff_for_humans_other_and_minute():
    with pendulum.travel_to(pendulum.today().at(12, 34, 56)):
        now = pendulum.now().time()

        assert now.diff_for_humans(now.add(minutes=1)) == "1 minute before"


def test_diff_for_humans_other_and_minutes():
    with pendulum.travel_to(pendulum.today().at(12, 34, 56)):
        now = pendulum.now().time()

        assert now.diff_for_humans(now.add(minutes=2)) == "2 minutes before"


def test_diff_for_humans_other_and_nearly_hour():
    with pendulum.travel_to(pendulum.today().at(12, 34, 56)):
        now = pendulum.now().time()

        assert now.diff_for_humans(now.add(minutes=59)) == "59 minutes before"


def test_diff_for_humans_other_and_hour():
    with pendulum.travel_to(pendulum.today().at(12, 34, 56)):
        now = pendulum.now().time()

        assert now.diff_for_humans(now.add(hours=1)) == "1 hour before"


def test_diff_for_humans_other_and_hours():
    with pendulum.travel_to(pendulum.today().at(12, 34, 56)):
        now = pendulum.now().time()

        assert now.diff_for_humans(now.add(hours=2)) == "2 hours before"


def test_diff_for_humans_other_and_future_second():
    with pendulum.travel_to(pendulum.today().at(12, 34, 56)):
        now = pendulum.now().time()

        assert now.diff_for_humans(now.subtract(seconds=1)) == "a few seconds after"


def test_diff_for_humans_other_and_future_seconds():
    with pendulum.travel_to(pendulum.today().at(12, 34, 56)):
        now = pendulum.now().time()

        assert now.diff_for_humans(now.subtract(seconds=2)) == "a few seconds after"


def test_diff_for_humans_other_and_nearly_future_minute():
    with pendulum.travel_to(pendulum.today().at(12, 34, 56)):
        now = pendulum.now().time()

        assert now.diff_for_humans(now.subtract(seconds=59)) == "59 seconds after"


def test_diff_for_humans_other_and_future_minute():
    with pendulum.travel_to(pendulum.today().at(12, 34, 56)):
        now = pendulum.now().time()

        assert now.diff_for_humans(now.subtract(minutes=1)) == "1 minute after"


def test_diff_for_humans_other_and_future_minutes():
    with pendulum.travel_to(pendulum.today().at(12, 34, 56)):
        now = pendulum.now().time()

        assert now.diff_for_humans(now.subtract(minutes=2)) == "2 minutes after"


def test_diff_for_humans_other_and_nearly_future_hour():
    with pendulum.travel_to(pendulum.today().at(12, 34, 56)):
        now = pendulum.now().time()

        assert now.diff_for_humans(now.subtract(minutes=59)) == "59 minutes after"


def test_diff_for_humans_other_and_future_hour():
    with pendulum.travel_to(pendulum.today().at(12, 34, 56)):
        now = pendulum.now().time()

        assert now.diff_for_humans(now.subtract(hours=1)) == "1 hour after"


def test_diff_for_humans_other_and_future_hours():
    with pendulum.travel_to(pendulum.today().at(12, 34, 56)):
        now = pendulum.now().time()

        assert now.diff_for_humans(now.subtract(hours=2)) == "2 hours after"


def test_diff_for_humans_absolute_seconds():
    with pendulum.travel_to(pendulum.today().at(12, 34, 56)):
        now = pendulum.now().time()

        assert now.diff_for_humans(now.subtract(seconds=59), True) == "59 seconds"
        now = pendulum.now().time()

        assert now.diff_for_humans(now.add(seconds=59), True) == "59 seconds"


def test_diff_for_humans_absolute_minutes():
    with pendulum.travel_to(pendulum.today().at(12, 34, 56)):
        now = pendulum.now().time()

        assert now.diff_for_humans(now.subtract(minutes=30), True) == "30 minutes"
        now = pendulum.now().time()

        assert now.diff_for_humans(now.add(minutes=30), True) == "30 minutes"


def test_diff_for_humans_absolute_hours():
    with pendulum.travel_to(pendulum.today().at(12, 34, 56)):
        now = pendulum.now().time()

        assert now.diff_for_humans(now.subtract(hours=3), True) == "3 hours"
        now = pendulum.now().time()

        assert now.diff_for_humans(now.add(hours=3), True) == "3 hours"


# === tests/time/test_strings.py ===
from __future__ import annotations

from pendulum import Time


def test_to_string():
    d = Time(1, 2, 3)
    assert str(d) == "01:02:03"
    d = Time(1, 2, 3, 123456)
    assert str(d) == "01:02:03.123456"


def test_repr():
    d = Time(1, 2, 3)
    assert repr(d) == "Time(1, 2, 3)"

    d = Time(1, 2, 3, 123456)
    assert repr(d) == "Time(1, 2, 3, 123456)"


def test_format_with_locale():
    d = Time(14, 15, 16)
    assert d.format("hh:mm:ss A", locale="fr") == "02:15:16 PM"


def test_strftime():
    d = Time(14, 15, 16)
    assert d.strftime("%H") == "14"


def test_for_json():
    d = Time(14, 15, 16)
    assert d.for_json() == "14:15:16"


def test_format():
    d = Time(14, 15, 16)
    assert f"{d}" == "14:15:16"
    assert f"{d:mm}" == "15"


# === tests/time/test_fluent_setters.py ===
from __future__ import annotations

from pendulum import Time
from tests.conftest import assert_time


def test_replace():
    t = Time(12, 34, 56, 123456)
    t = t.replace(1, 2, 3, 654321)

    assert isinstance(t, Time)
    assert_time(t, 1, 2, 3, 654321)


# === tests/time/test_sub.py ===
from __future__ import annotations

from datetime import time
from datetime import timedelta

import pytest
import pytz

import pendulum

from pendulum import Time
from tests.conftest import assert_duration


def test_sub_hours_positive():
    assert Time(0, 0, 0).subtract(hours=1).hour == 23


def test_sub_hours_zero():
    assert Time(0, 0, 0).subtract(hours=0).hour == 0


def test_sub_hours_negative():
    assert Time(0, 0, 0).subtract(hours=-1).hour == 1


def test_sub_minutes_positive():
    assert Time(0, 0, 0).subtract(minutes=1).minute == 59


def test_sub_minutes_zero():
    assert Time(0, 0, 0).subtract(minutes=0).minute == 0


def test_sub_minutes_negative():
    assert Time(0, 0, 0).subtract(minutes=-1).minute == 1


def test_sub_seconds_positive():
    assert Time(0, 0, 0).subtract(seconds=1).second == 59


def test_sub_seconds_zero():
    assert Time(0, 0, 0).subtract(seconds=0).second == 0


def test_sub_seconds_negative():
    assert Time(0, 0, 0).subtract(seconds=-1).second == 1


def test_subtract_timedelta():
    delta = timedelta(seconds=16, microseconds=654321)
    d = Time(3, 12, 15, 777777)

    d = d.subtract_timedelta(delta)
    assert d.minute == 11
    assert d.second == 59
    assert d.microsecond == 123456

    d = Time(3, 12, 15, 777777)

    d = d - delta
    assert d.minute == 11
    assert d.second == 59
    assert d.microsecond == 123456


def test_add_timedelta_with_days():
    delta = timedelta(days=3, seconds=45, microseconds=123456)
    d = Time(3, 12, 15, 654321)

    with pytest.raises(TypeError):
        d.subtract_timedelta(delta)


def test_subtract_invalid_type():
    d = Time(0, 0, 0)

    with pytest.raises(TypeError):
        d - "ab"

    with pytest.raises(TypeError):
        "ab" - d


def test_subtract_time():
    t = Time(12, 34, 56)
    t1 = Time(1, 1, 1)
    t2 = time(1, 1, 1)
    t3 = time(1, 1, 1, tzinfo=pytz.timezone("Europe/Paris"))

    diff = t - t1
    assert isinstance(diff, pendulum.Duration)
    assert_duration(diff, 0, hours=11, minutes=33, seconds=55)

    diff = t1 - t
    assert isinstance(diff, pendulum.Duration)
    assert_duration(diff, 0, hours=-11, minutes=-33, seconds=-55)

    diff = t - t2
    assert isinstance(diff, pendulum.Duration)
    assert_duration(diff, 0, hours=11, minutes=33, seconds=55)

    diff = t2 - t
    assert isinstance(diff, pendulum.Duration)
    assert_duration(diff, 0, hours=-11, minutes=-33, seconds=-55)

    with pytest.raises(TypeError):
        t - t3

    with pytest.raises(TypeError):
        t3 - t


# === tests/testing/__init__.py ===


# === tests/testing/test_time_travel.py ===
from __future__ import annotations

from time import sleep
from typing import TYPE_CHECKING

import pytest

import pendulum

from pendulum.utils._compat import PYPY


if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(autouse=True)
def setup() -> Generator[None, None, None]:
    pendulum.travel_back()

    yield

    pendulum.travel_back()


@pytest.mark.skipif(PYPY, reason="Time travelling not available on PyPy")
def test_travel() -> None:
    now = pendulum.now()

    pendulum.travel(minutes=5)

    assert pendulum.now().diff_for_humans(now) == "5 minutes after"


@pytest.mark.skipif(PYPY, reason="Time travelling not available on PyPy")
def test_travel_with_frozen_time() -> None:
    pendulum.travel(minutes=5, freeze=True)

    now = pendulum.now()

    sleep(0.01)

    assert now == pendulum.now()


@pytest.mark.skipif(PYPY, reason="Time travelling not available on PyPy")
def test_travel_to() -> None:
    dt = pendulum.datetime(2022, 1, 19, tz="local")

    pendulum.travel_to(dt)

    assert pendulum.now().date() == dt.date()


@pytest.mark.skipif(PYPY, reason="Time travelling not available on PyPy")
def test_freeze() -> None:
    pendulum.freeze()

    pendulum.travel(minutes=5)

    assert pendulum.now() == pendulum.now()

    pendulum.travel_back()

    pendulum.travel(minutes=5)

    now = pendulum.now()

    sleep(0.01)

    assert now != pendulum.now()

    pendulum.freeze()

    assert pendulum.now() == pendulum.now()

    pendulum.travel_back()

    with pendulum.freeze():
        assert pendulum.now() == pendulum.now()

    now = pendulum.now()

    sleep(0.01)

    assert now != pendulum.now()


# === tests/formatting/__init__.py ===


# === tests/formatting/test_formatter.py ===
from __future__ import annotations

import pytest

import pendulum

from pendulum.formatting import Formatter
from pendulum.locales.locale import Locale


@pytest.fixture(autouse=True)
def setup():
    Locale._cache["dummy"] = {}

    yield

    del Locale._cache["dummy"]


def test_year_tokens():
    d = pendulum.datetime(2009, 1, 14, 15, 25, 50, 123456)
    f = Formatter()

    assert f.format(d, "YYYY") == "2009"
    assert f.format(d, "YY") == "09"
    assert f.format(d, "Y") == "2009"


def test_quarter_tokens():
    f = Formatter()
    d = pendulum.datetime(1985, 1, 4)
    assert f.format(d, "Q") == "1"

    d = pendulum.datetime(2029, 8, 1)
    assert f.format(d, "Q") == "3"

    d = pendulum.datetime(1985, 1, 4)
    assert f.format(d, "Qo") == "1st"

    d = pendulum.datetime(2029, 8, 1)
    assert f.format(d, "Qo") == "3rd"

    d = pendulum.datetime(1985, 1, 4)
    assert f.format(d, "Qo", locale="fr") == "1er"

    d = pendulum.datetime(2029, 8, 1)
    assert f.format(d, "Qo", locale="fr") == "3e"


def test_month_tokens():
    f = Formatter()
    d = pendulum.datetime(2016, 3, 24)
    assert f.format(d, "MM") == "03"
    assert f.format(d, "M") == "3"

    assert f.format(d, "MMM") == "Mar"
    assert f.format(d, "MMMM") == "March"
    assert f.format(d, "Mo") == "3rd"

    assert f.format(d, "MMM", locale="fr") == "mars"
    assert f.format(d, "MMMM", locale="fr") == "mars"
    assert f.format(d, "Mo", locale="fr") == "3e"


def test_day_tokens():
    f = Formatter()
    d = pendulum.datetime(2016, 3, 7)
    assert f.format(d, "DD") == "07"
    assert f.format(d, "D") == "7"

    assert f.format(d, "Do") == "7th"
    assert f.format(d.first_of("month"), "Do") == "1st"

    assert f.format(d, "Do", locale="fr") == "7e"
    assert f.format(d.first_of("month"), "Do", locale="fr") == "1er"


def test_day_of_year():
    f = Formatter()
    d = pendulum.datetime(2016, 8, 28)
    assert f.format(d, "DDDD") == "241"
    assert f.format(d, "DDD") == "241"
    assert f.format(d.start_of("year"), "DDDD") == "001"
    assert f.format(d.start_of("year"), "DDD") == "1"

    assert f.format(d, "DDDo") == "241st"
    assert f.format(d.add(days=3), "DDDo") == "244th"

    assert f.format(d, "DDDo", locale="fr") == "241e"
    assert f.format(d.add(days=3), "DDDo", locale="fr") == "244e"


def test_week_of_year():
    f = Formatter()
    d = pendulum.datetime(2016, 8, 28)

    assert f.format(d, "wo") == "34th"


def test_day_of_week():
    f = Formatter()
    d = pendulum.datetime(2016, 8, 28)
    assert f.format(d, "d") == "0"

    assert f.format(d, "dd") == "Su"
    assert f.format(d, "ddd") == "Sun"
    assert f.format(d, "dddd") == "Sunday"

    assert f.format(d, "dd", locale="fr") == "di"
    assert f.format(d, "ddd", locale="fr") == "dim."
    assert f.format(d, "dddd", locale="fr") == "dimanche"

    assert f.format(d, "do") == "0th"


def test_localized_day_of_week():
    f = Formatter()
    d = pendulum.datetime(2016, 8, 28)
    assert f.format(d, "e") == "0"
    assert f.format(d, "e", locale="en-gb") == "6"
    assert f.format(d.add(days=2), "e") == "2"
    assert f.format(d.add(days=2), "e", locale="en-gb") == "1"

    assert f.format(d, "eo") == "1st"
    assert f.format(d, "eo", locale="en-gb") == "7th"
    assert f.format(d.add(days=2), "eo") == "3rd"
    assert f.format(d.add(days=2), "eo", locale="en-gb") == "2nd"


def test_day_of_iso_week():
    f = Formatter()
    d = pendulum.datetime(2016, 8, 28)
    assert f.format(d, "E") == "7"


def test_am_pm():
    f = Formatter()
    d = pendulum.datetime(2016, 8, 28, 23)
    assert f.format(d, "A") == "PM"
    assert f.format(d.set(hour=11), "A") == "AM"


def test_hour():
    f = Formatter()
    d = pendulum.datetime(2016, 8, 28, 7)
    assert f.format(d, "H") == "7"
    assert f.format(d, "HH") == "07"

    d = pendulum.datetime(2016, 8, 28, 0)
    assert f.format(d, "h") == "12"
    assert f.format(d, "hh") == "12"


def test_minute():
    f = Formatter()
    d = pendulum.datetime(2016, 8, 28, 7, 3)
    assert f.format(d, "m") == "3"
    assert f.format(d, "mm") == "03"


def test_second():
    f = Formatter()
    d = pendulum.datetime(2016, 8, 28, 7, 3, 6)
    assert f.format(d, "s") == "6"
    assert f.format(d, "ss") == "06"


def test_fractional_second():
    f = Formatter()
    d = pendulum.datetime(2016, 8, 28, 7, 3, 6, 123456)
    assert f.format(d, "S") == "1"
    assert f.format(d, "SS") == "12"
    assert f.format(d, "SSS") == "123"
    assert f.format(d, "SSSS") == "1234"
    assert f.format(d, "SSSSS") == "12345"
    assert f.format(d, "SSSSSS") == "123456"

    d = pendulum.datetime(2016, 8, 28, 7, 3, 6, 0)
    assert f.format(d, "S") == "0"
    assert f.format(d, "SS") == "00"
    assert f.format(d, "SSS") == "000"
    assert f.format(d, "SSSS") == "0000"
    assert f.format(d, "SSSSS") == "00000"
    assert f.format(d, "SSSSSS") == "000000"

    d = pendulum.datetime(2016, 8, 28, 7, 3, 6, 123)
    assert f.format(d, "S") == "0"
    assert f.format(d, "SS") == "00"
    assert f.format(d, "SSS") == "000"
    assert f.format(d, "SSSS") == "0001"
    assert f.format(d, "SSSSS") == "00012"
    assert f.format(d, "SSSSSS") == "000123"


def test_timezone():
    f = Formatter()
    d = pendulum.datetime(2016, 8, 28, 7, 3, 6, 123456, tz="Europe/Paris")
    assert f.format(d, "zz") == "CEST"
    assert f.format(d, "z") == "Europe/Paris"

    d = pendulum.datetime(2016, 1, 28, 7, 3, 6, 123456, tz="Europe/Paris")
    assert f.format(d, "zz") == "CET"
    assert f.format(d, "z") == "Europe/Paris"


def test_timezone_offset():
    f = Formatter()
    d = pendulum.datetime(2016, 8, 28, 7, 3, 6, 123456, tz="Europe/Paris")
    assert f.format(d, "ZZ") == "+0200"
    assert f.format(d, "Z") == "+02:00"

    d = pendulum.datetime(2016, 1, 28, 7, 3, 6, 123456, tz="Europe/Paris")
    assert f.format(d, "ZZ") == "+0100"
    assert f.format(d, "Z") == "+01:00"

    d = pendulum.datetime(2016, 1, 28, 7, 3, 6, 123456, tz="America/Guayaquil")
    assert f.format(d, "ZZ") == "-0500"
    assert f.format(d, "Z") == "-05:00"


def test_timestamp():
    f = Formatter()
    d = pendulum.datetime(1970, 1, 1)
    assert f.format(d, "X") == "0"
    assert f.format(d.add(days=1), "X") == "86400"


def test_timestamp_milliseconds():
    f = Formatter()
    d = pendulum.datetime(1970, 1, 1)
    assert f.format(d, "x") == "0"
    assert f.format(d.add(days=1), "x") == "86400000"
    assert f.format(d.add(days=1, microseconds=129123), "x") == "86400129"


def test_date_formats():
    f = Formatter()
    d = pendulum.datetime(2016, 8, 28, 7, 3, 6, 123456)
    assert f.format(d, "LT") == "7:03 AM"
    assert f.format(d, "LTS") == "7:03:06 AM"
    assert f.format(d, "L") == "08/28/2016"
    assert f.format(d, "LL") == "August 28, 2016"
    assert f.format(d, "LLL") == "August 28, 2016 7:03 AM"
    assert f.format(d, "LLLL") == "Sunday, August 28, 2016 7:03 AM"

    assert f.format(d, "LT", locale="fr") == "07:03"
    assert f.format(d, "LTS", locale="fr") == "07:03:06"
    assert f.format(d, "L", locale="fr") == "28/08/2016"
    assert f.format(d, "LL", locale="fr") == "28 août 2016"
    assert f.format(d, "LLL", locale="fr") == "28 août 2016 07:03"
    assert f.format(d, "LLLL", locale="fr") == "dimanche 28 août 2016 07:03"


def test_escape():
    f = Formatter()
    d = pendulum.datetime(2016, 8, 28)
    assert f.format(d, r"[YYYY] YYYY \[YYYY\]") == "YYYY 2016 [2016]"
    assert f.format(d, r"\D D \\D") == "D 28 \\28"


def test_date_formats_missing():
    f = Formatter()
    d = pendulum.datetime(2016, 8, 28, 7, 3, 6, 123456)

    assert f.format(d, "LT", locale="dummy") == "7:03 AM"
    assert f.format(d, "LTS", locale="dummy") == "7:03:06 AM"
    assert f.format(d, "L", locale="dummy") == "08/28/2016"
    assert f.format(d, "LL", locale="dummy") == "August 28, 2016"
    assert f.format(d, "LLL", locale="dummy") == "August 28, 2016 7:03 AM"
    assert f.format(d, "LLLL", locale="dummy") == "Sunday, August 28, 2016 7:03 AM"


def test_unknown_token():
    f = Formatter()
    d = pendulum.datetime(2016, 8, 28, 7, 3, 6, 123456)

    assert f.format(d, "J") == "J"


# === tests/benchmarks/__init__.py ===


# === tests/benchmarks/test_parse_8601.py ===
from __future__ import annotations

import pytest

from pendulum.parsing.iso8601 import parse_iso8601


@pytest.mark.benchmark(group="Parsing")
def test_parse_iso8601() -> None:
    # Date
    parse_iso8601("2016")
    parse_iso8601("2016-10")
    parse_iso8601("2016-10-06")
    parse_iso8601("20161006")

    # Time
    parse_iso8601("201610")

    # Datetime
    parse_iso8601("2016-10-06T12:34:56.123456")
    parse_iso8601("2016-10-06T12:34:56.123")
    parse_iso8601("2016-10-06T12:34:56.000123")
    parse_iso8601("2016-10-06T12")
    parse_iso8601("2016-10-06T123456")
    parse_iso8601("2016-10-06T123456.123456")
    parse_iso8601("20161006T123456.123456")
    parse_iso8601("20161006 123456.123456")

    # Datetime with offset
    parse_iso8601("2016-10-06T12:34:56.123456+05:30")
    parse_iso8601("2016-10-06T12:34:56.123456+0530")
    parse_iso8601("2016-10-06T12:34:56.123456-05:30")
    parse_iso8601("2016-10-06T12:34:56.123456-0530")
    parse_iso8601("2016-10-06T12:34:56.123456+05")
    parse_iso8601("2016-10-06T12:34:56.123456-05")
    parse_iso8601("20161006T123456,123456-05")
    parse_iso8601("2016-10-06T12:34:56.123456789+05:30")

    # Ordinal date
    parse_iso8601("2012-007")
    parse_iso8601("2012007")
    parse_iso8601("2017-079")

    # Week date
    parse_iso8601("2012-W05")
    parse_iso8601("2008-W39-6")
    parse_iso8601("2009-W53-7")
    parse_iso8601("2009-W01-1")

    # Week date wth time
    parse_iso8601("2008-W39-6T09")


# === tests/fixtures/__init__.py ===


# === tests/parsing/test_parsing.py ===
from __future__ import annotations

import datetime

import pytest

import pendulum

from pendulum.parsing import ParserError
from pendulum.parsing import parse


def test_y():
    text = "2016"

    parsed = parse(text)

    assert parsed.year == 2016
    assert parsed.month == 1
    assert parsed.day == 1
    assert parsed.hour == 0
    assert parsed.minute == 0
    assert parsed.second == 0
    assert parsed.microsecond == 0
    assert parsed.tzinfo is None


def test_ym():
    text = "2016-10"

    parsed = parse(text)

    assert parsed.year == 2016
    assert parsed.month == 10
    assert parsed.day == 1
    assert parsed.hour == 0
    assert parsed.minute == 0
    assert parsed.second == 0
    assert parsed.microsecond == 0
    assert parsed.tzinfo is None


def test_ymd():
    text = "2016-10-06"

    parsed = parse(text)

    assert parsed.year == 2016
    assert parsed.month == 10
    assert parsed.day == 6
    assert parsed.hour == 0
    assert parsed.minute == 0
    assert parsed.second == 0
    assert parsed.microsecond == 0
    assert parsed.tzinfo is None


def test_ymd_one_character():
    text = "2016-2-6"

    parsed = parse(text, strict=False)

    assert parsed.year == 2016
    assert parsed.month == 2
    assert parsed.day == 6
    assert parsed.hour == 0
    assert parsed.minute == 0
    assert parsed.second == 0
    assert parsed.microsecond == 0
    assert parsed.tzinfo is None


def test_ymd_hms():
    text = "2016-10-06 12:34:56"

    parsed = parse(text)

    assert parsed.year == 2016
    assert parsed.month == 10
    assert parsed.day == 6
    assert parsed.hour == 12
    assert parsed.minute == 34
    assert parsed.second == 56
    assert parsed.microsecond == 0
    assert parsed.tzinfo is None

    text = "2016-10-06 12:34:56.123456"

    parsed = parse(text)

    assert parsed.year == 2016
    assert parsed.month == 10
    assert parsed.day == 6
    assert parsed.hour == 12
    assert parsed.minute == 34
    assert parsed.second == 56
    assert parsed.microsecond == 123456
    assert parsed.tzinfo is None


def test_rfc_3339():
    text = "2016-10-06T12:34:56+05:30"

    parsed = parse(text)

    assert parsed.year == 2016
    assert parsed.month == 10
    assert parsed.day == 6
    assert parsed.hour == 12
    assert parsed.minute == 34
    assert parsed.second == 56
    assert parsed.microsecond == 0
    assert parsed.utcoffset().total_seconds() == 19800


def test_rfc_3339_extended():
    text = "2016-10-06T12:34:56.123456+05:30"

    parsed = parse(text)

    assert parsed.year == 2016
    assert parsed.month == 10
    assert parsed.day == 6
    assert parsed.hour == 12
    assert parsed.minute == 34
    assert parsed.second == 56
    assert parsed.microsecond == 123456
    assert parsed.utcoffset().total_seconds() == 19800

    text = "2016-10-06T12:34:56.000123+05:30"

    parsed = parse(text)

    assert parsed.year == 2016
    assert parsed.month == 10
    assert parsed.day == 6
    assert parsed.hour == 12
    assert parsed.minute == 34
    assert parsed.second == 56
    assert parsed.microsecond == 123
    assert parsed.utcoffset().total_seconds() == 19800


def test_rfc_3339_extended_nanoseconds():
    text = "2016-10-06T12:34:56.123456789+05:30"

    parsed = parse(text)

    assert parsed.year == 2016
    assert parsed.month == 10
    assert parsed.day == 6
    assert parsed.hour == 12
    assert parsed.minute == 34
    assert parsed.second == 56
    assert parsed.microsecond == 123456
    assert parsed.utcoffset().total_seconds() == 19800


def test_iso_8601_date():
    text = "2012"

    parsed = parse(text)

    assert parsed.year == 2012
    assert parsed.month == 1
    assert parsed.day == 1
    assert parsed.hour == 0
    assert parsed.minute == 0
    assert parsed.second == 0
    assert parsed.microsecond == 0
    assert parsed.tzinfo is None

    text = "2012-05-03"

    parsed = parse(text)

    assert parsed.year == 2012
    assert parsed.month == 5
    assert parsed.day == 3
    assert parsed.hour == 0
    assert parsed.minute == 0
    assert parsed.second == 0
    assert parsed.microsecond == 0
    assert parsed.tzinfo is None

    text = "20120503"

    parsed = parse(text)

    assert parsed.year == 2012
    assert parsed.month == 5
    assert parsed.day == 3
    assert parsed.hour == 0
    assert parsed.minute == 0
    assert parsed.second == 0
    assert parsed.microsecond == 0
    assert parsed.tzinfo is None

    text = "2012-05"

    parsed = parse(text)

    assert parsed.year == 2012
    assert parsed.month == 5
    assert parsed.day == 1
    assert parsed.hour == 0
    assert parsed.minute == 0
    assert parsed.second == 0
    assert parsed.microsecond == 0
    assert parsed.tzinfo is None


def test_iso8601_datetime():
    text = "2016-10-01T14"

    parsed = parse(text)

    assert parsed.year == 2016
    assert parsed.month == 10
    assert parsed.day == 1
    assert parsed.hour == 14
    assert parsed.minute == 0
    assert parsed.second == 0
    assert parsed.microsecond == 0
    assert parsed.tzinfo is None

    text = "2016-10-01T14:30"

    parsed = parse(text)

    assert parsed.year == 2016
    assert parsed.month == 10
    assert parsed.day == 1
    assert parsed.hour == 14
    assert parsed.minute == 30
    assert parsed.second == 0
    assert parsed.microsecond == 0
    assert parsed.tzinfo is None

    text = "20161001T14"

    parsed = parse(text)

    assert parsed.year == 2016
    assert parsed.month == 10
    assert parsed.day == 1
    assert parsed.hour == 14
    assert parsed.minute == 0
    assert parsed.second == 0
    assert parsed.microsecond == 0
    assert parsed.tzinfo is None

    text = "20161001T1430"

    parsed = parse(text)

    assert parsed.year == 2016
    assert parsed.month == 10
    assert parsed.day == 1
    assert parsed.hour == 14
    assert parsed.minute == 30
    assert parsed.second == 0
    assert parsed.microsecond == 0
    assert parsed.tzinfo is None

    text = "20161001T1430+0530"

    parsed = parse(text)

    assert parsed.year == 2016
    assert parsed.month == 10
    assert parsed.day == 1
    assert parsed.hour == 14
    assert parsed.minute == 30
    assert parsed.second == 0
    assert parsed.microsecond == 0
    assert parsed.utcoffset().total_seconds() == 19800

    text = "2008-09-03T20:56:35.450686+01"

    parsed = parse(text)

    assert parsed.year == 2008
    assert parsed.month == 9
    assert parsed.day == 3
    assert parsed.hour == 20
    assert parsed.minute == 56
    assert parsed.second == 35
    assert parsed.microsecond == 450686
    assert parsed.utcoffset().total_seconds() == 3600


def test_iso8601_week_number():
    text = "2012-W05"

    parsed = parse(text)

    assert parsed.year == 2012
    assert parsed.month == 1
    assert parsed.day == 30
    assert parsed.hour == 0
    assert parsed.minute == 0
    assert parsed.second == 0
    assert parsed.microsecond == 0
    assert parsed.tzinfo is None

    text = "2012W05"

    parsed = parse(text)

    assert parsed.year == 2012
    assert parsed.month == 1
    assert parsed.day == 30
    assert parsed.hour == 0
    assert parsed.minute == 0
    assert parsed.second == 0
    assert parsed.microsecond == 0
    assert parsed.tzinfo is None

    # Long Year
    text = "2015W53"

    parsed = parse(text)

    assert parsed.year == 2015
    assert parsed.month == 12
    assert parsed.day == 28
    assert parsed.hour == 0
    assert parsed.minute == 0
    assert parsed.second == 0
    assert parsed.microsecond == 0
    assert parsed.tzinfo is None

    text = "2012-W05-5"

    parsed = parse(text)

    assert parsed.year == 2012
    assert parsed.month == 2
    assert parsed.day == 3
    assert parsed.hour == 0
    assert parsed.minute == 0
    assert parsed.second == 0
    assert parsed.microsecond == 0
    assert parsed.tzinfo is None

    text = "2012W055"

    parsed = parse(text)

    assert parsed.year == 2012
    assert parsed.month == 2
    assert parsed.day == 3
    assert parsed.hour == 0
    assert parsed.minute == 0
    assert parsed.second == 0
    assert parsed.microsecond == 0
    assert parsed.tzinfo is None

    text = "2009-W53-7"
    parsed = parse(text)

    assert parsed.year == 2010
    assert parsed.month == 1
    assert parsed.day == 3
    assert parsed.hour == 0
    assert parsed.minute == 0
    assert parsed.second == 0
    assert parsed.microsecond == 0
    assert parsed.tzinfo is None

    text = "2009-W01-1"
    parsed = parse(text)

    assert parsed.year == 2008
    assert parsed.month == 12
    assert parsed.day == 29
    assert parsed.hour == 0
    assert parsed.minute == 0
    assert parsed.second == 0
    assert parsed.microsecond == 0
    assert parsed.tzinfo is None


def test_iso8601_week_number_with_time():
    text = "2012-W05T09"

    parsed = parse(text)

    assert parsed.year == 2012
    assert parsed.month == 1
    assert parsed.day == 30
    assert parsed.hour == 9
    assert parsed.minute == 0
    assert parsed.second == 0
    assert parsed.microsecond == 0
    assert parsed.tzinfo is None

    text = "2012W05T09"

    parsed = parse(text)

    assert parsed.year == 2012
    assert parsed.month == 1
    assert parsed.day == 30
    assert parsed.hour == 9
    assert parsed.minute == 0
    assert parsed.second == 0
    assert parsed.microsecond == 0
    assert parsed.tzinfo is None

    text = "2012-W05-5T09"

    parsed = parse(text)

    assert parsed.year == 2012
    assert parsed.month == 2
    assert parsed.day == 3
    assert parsed.hour == 9
    assert parsed.minute == 0
    assert parsed.second == 0
    assert parsed.microsecond == 0
    assert parsed.tzinfo is None

    text = "2012W055T09"

    parsed = parse(text)

    assert parsed.year == 2012
    assert parsed.month == 2
    assert parsed.day == 3
    assert parsed.hour == 9
    assert parsed.minute == 0
    assert parsed.second == 0
    assert parsed.microsecond == 0
    assert parsed.tzinfo is None


def test_iso8601_ordinal():
    text = "2012-007"

    parsed = parse(text)

    assert parsed.year == 2012
    assert parsed.month == 1
    assert parsed.day == 7
    assert parsed.hour == 0
    assert parsed.minute == 0
    assert parsed.second == 0
    assert parsed.microsecond == 0
    assert parsed.tzinfo is None

    text = "2012007"

    parsed = parse(text)

    assert parsed.year == 2012
    assert parsed.month == 1
    assert parsed.day == 7
    assert parsed.hour == 0
    assert parsed.minute == 0
    assert parsed.second == 0
    assert parsed.microsecond == 0
    assert parsed.tzinfo is None


def test_iso8601_time():
    now = pendulum.datetime(2015, 11, 12)

    text = "T201205"

    parsed = parse(text, now=now)

    assert parsed.year == 2015
    assert parsed.month == 11
    assert parsed.day == 12
    assert parsed.hour == 20
    assert parsed.minute == 12
    assert parsed.second == 5
    assert parsed.microsecond == 0
    assert parsed.tzinfo is None

    text = "20:12:05"

    parsed = parse(text, now=now)

    assert parsed.year == 2015
    assert parsed.month == 11
    assert parsed.day == 12
    assert parsed.hour == 20
    assert parsed.minute == 12
    assert parsed.second == 5
    assert parsed.microsecond == 0
    assert parsed.tzinfo is None

    text = "20:12:05.123456"

    parsed = parse(text, now=now)

    assert parsed.year == 2015
    assert parsed.month == 11
    assert parsed.day == 12
    assert parsed.hour == 20
    assert parsed.minute == 12
    assert parsed.second == 5
    assert parsed.microsecond == 123456
    assert parsed.tzinfo is None


def test_iso8601_ordinal_invalid():
    text = "2012-007-05"

    with pytest.raises(ParserError):
        parse(text)


def test_exact():
    text = "2012"

    parsed = parse(text, exact=True)

    assert isinstance(parsed, datetime.date)
    assert parsed.year == 2012
    assert parsed.month == 1
    assert parsed.day == 1

    text = "2012-03"

    parsed = parse(text, exact=True)

    assert isinstance(parsed, datetime.date)
    assert parsed.year == 2012
    assert parsed.month == 3
    assert parsed.day == 1

    text = "2012-03-13"

    parsed = parse(text, exact=True)

    assert isinstance(parsed, datetime.date)
    assert parsed.year == 2012
    assert parsed.month == 3
    assert parsed.day == 13

    text = "2012W055"

    parsed = parse(text, exact=True)

    assert isinstance(parsed, datetime.date)
    assert parsed.year == 2012
    assert parsed.month == 2
    assert parsed.day == 3

    text = "2012007"

    parsed = parse(text, exact=True)

    assert isinstance(parsed, datetime.date)
    assert parsed.year == 2012
    assert parsed.month == 1
    assert parsed.day == 7

    text = "20:12:05"

    parsed = parse(text, exact=True)

    assert isinstance(parsed, datetime.time)
    assert parsed.hour == 20
    assert parsed.minute == 12
    assert parsed.second == 5
    assert parsed.microsecond == 0


def test_edge_cases():
    text = "2013-11-1"

    parsed = parse(text, strict=False)
    assert parsed.year == 2013
    assert parsed.month == 11
    assert parsed.day == 1
    assert parsed.hour == 0
    assert parsed.minute == 0
    assert parsed.second == 0
    assert parsed.microsecond == 0
    assert parsed.tzinfo is None

    text = "10-01-01"

    parsed = parse(text, strict=False)
    assert parsed.year == 2010
    assert parsed.month == 1
    assert parsed.day == 1
    assert parsed.hour == 0
    assert parsed.minute == 0
    assert parsed.second == 0
    assert parsed.microsecond == 0
    assert parsed.tzinfo is None

    text = "31-01-01"

    parsed = parse(text, strict=False)
    assert parsed.year == 2031
    assert parsed.month == 1
    assert parsed.day == 1
    assert parsed.hour == 0
    assert parsed.minute == 0
    assert parsed.second == 0
    assert parsed.microsecond == 0
    assert parsed.tzinfo is None

    text = "32-01-01"

    parsed = parse(text, strict=False)
    assert parsed.year == 2032
    assert parsed.month == 1
    assert parsed.day == 1
    assert parsed.hour == 0
    assert parsed.minute == 0
    assert parsed.second == 0
    assert parsed.microsecond == 0
    assert parsed.tzinfo is None


def test_strict():
    text = "4 Aug 2015 - 11:20 PM"

    with pytest.raises(ParserError):
        parse(text)

    parsed = parse(text, strict=False)
    assert parsed.year == 2015
    assert parsed.month == 8
    assert parsed.day == 4
    assert parsed.hour == 23
    assert parsed.minute == 20
    assert parsed.second == 0
    assert parsed.microsecond == 0
    assert parsed.tzinfo is None


def test_invalid():
    text = "201610T"

    with pytest.raises(ParserError):
        parse(text)

    text = "2012-W54"

    with pytest.raises(ParserError):
        parse(text)

    text = "2012-W13-8"

    with pytest.raises(ParserError):
        parse(text)

    # W53 in normal year (not long)
    text = "2017W53"

    with pytest.raises(ParserError):
        parse(text)

    text = "/2012"

    with pytest.raises(ParserError):
        parse(text)

    text = "2012/"

    with pytest.raises(ParserError):
        parse(text)


def test_exif_edge_case():
    text = "2016:12:26 15:45:28"

    parsed = parse(text)

    assert parsed.year == 2016
    assert parsed.month == 12
    assert parsed.day == 26
    assert parsed.hour == 15
    assert parsed.minute == 45
    assert parsed.second == 28


# === tests/parsing/test_parsing_duration.py ===
from __future__ import annotations

import pytest

from pendulum.parsing import ParserError
from pendulum.parsing import parse


def test_parse_duration():
    text = "P2Y3M4DT5H6M7S"
    parsed = parse(text)

    assert parsed.years == 2
    assert parsed.months == 3
    assert parsed.weeks == 0
    assert parsed.remaining_days == 4
    assert parsed.hours == 5
    assert parsed.minutes == 6
    assert parsed.remaining_seconds == 7
    assert parsed.microseconds == 0

    text = "P1Y2M3DT4H5M6.5S"
    parsed = parse(text)

    assert parsed.years == 1
    assert parsed.months == 2
    assert parsed.weeks == 0
    assert parsed.remaining_days == 3
    assert parsed.hours == 4
    assert parsed.minutes == 5
    assert parsed.remaining_seconds == 6
    assert parsed.microseconds == 500000

    text = "P1Y2M3DT4H5M6,5S"
    parsed = parse(text)

    assert parsed.years == 1
    assert parsed.months == 2
    assert parsed.weeks == 0
    assert parsed.remaining_days == 3
    assert parsed.hours == 4
    assert parsed.minutes == 5
    assert parsed.remaining_seconds == 6
    assert parsed.microseconds == 500000

    text = "P1Y2M3D"
    parsed = parse(text)

    assert parsed.years == 1
    assert parsed.months == 2
    assert parsed.weeks == 0
    assert parsed.remaining_days == 3
    assert parsed.hours == 0
    assert parsed.minutes == 0
    assert parsed.remaining_seconds == 0
    assert parsed.microseconds == 0

    text = "P1Y2M3.5D"
    parsed = parse(text)

    assert parsed.years == 1
    assert parsed.months == 2
    assert parsed.weeks == 0
    assert parsed.remaining_days == 3
    assert parsed.hours == 12
    assert parsed.minutes == 0
    assert parsed.remaining_seconds == 0
    assert parsed.microseconds == 0

    text = "P1Y2M3,5D"
    parsed = parse(text)

    assert parsed.years == 1
    assert parsed.months == 2
    assert parsed.weeks == 0
    assert parsed.remaining_days == 3
    assert parsed.hours == 12
    assert parsed.minutes == 0
    assert parsed.remaining_seconds == 0
    assert parsed.microseconds == 0

    text = "PT4H54M6.5S"
    parsed = parse(text)

    assert parsed.years == 0
    assert parsed.months == 0
    assert parsed.weeks == 0
    assert parsed.remaining_days == 0
    assert parsed.hours == 4
    assert parsed.minutes == 54
    assert parsed.remaining_seconds == 6
    assert parsed.microseconds == 500000

    text = "PT4H54M6,5S"
    parsed = parse(text)

    assert parsed.years == 0
    assert parsed.months == 0
    assert parsed.weeks == 0
    assert parsed.remaining_days == 0
    assert parsed.hours == 4
    assert parsed.minutes == 54
    assert parsed.remaining_seconds == 6
    assert parsed.microseconds == 500000

    text = "P1Y"
    parsed = parse(text)

    assert parsed.years == 1
    assert parsed.months == 0
    assert parsed.weeks == 0
    assert parsed.remaining_days == 0
    assert parsed.hours == 0
    assert parsed.minutes == 0
    assert parsed.remaining_seconds == 0
    assert parsed.microseconds == 0

    text = "P1.5Y"
    with pytest.raises(ParserError):
        parse(text)

    text = "P1,5Y"
    with pytest.raises(ParserError):
        parse(text)

    text = "P1M"
    parsed = parse(text)

    assert parsed.years == 0
    assert parsed.months == 1
    assert parsed.weeks == 0
    assert parsed.remaining_days == 0
    assert parsed.hours == 0
    assert parsed.minutes == 0
    assert parsed.remaining_seconds == 0
    assert parsed.microseconds == 0

    text = "P1.5M"
    with pytest.raises(ParserError):
        parse(text)

    text = "P1,5M"
    with pytest.raises(ParserError):
        parse(text)

    text = "P1W"
    parsed = parse(text)

    assert parsed.years == 0
    assert parsed.months == 0
    assert parsed.weeks == 1
    assert parsed.remaining_days == 0
    assert parsed.hours == 0
    assert parsed.minutes == 0
    assert parsed.remaining_seconds == 0
    assert parsed.microseconds == 0

    text = "P1.5W"
    parsed = parse(text)

    assert parsed.years == 0
    assert parsed.months == 0
    assert parsed.weeks == 1
    assert parsed.remaining_days == 3
    assert parsed.hours == 12
    assert parsed.minutes == 0
    assert parsed.remaining_seconds == 0
    assert parsed.microseconds == 0

    text = "P1,5W"
    parsed = parse(text)

    assert parsed.years == 0
    assert parsed.months == 0
    assert parsed.weeks == 1
    assert parsed.remaining_days == 3
    assert parsed.hours == 12
    assert parsed.minutes == 0
    assert parsed.remaining_seconds == 0
    assert parsed.microseconds == 0

    text = "P1D"
    parsed = parse(text)

    assert parsed.years == 0
    assert parsed.months == 0
    assert parsed.weeks == 0
    assert parsed.remaining_days == 1
    assert parsed.hours == 0
    assert parsed.minutes == 0
    assert parsed.remaining_seconds == 0
    assert parsed.microseconds == 0

    text = "P1.5D"
    parsed = parse(text)

    assert parsed.years == 0
    assert parsed.months == 0
    assert parsed.weeks == 0
    assert parsed.remaining_days == 1
    assert parsed.hours == 12
    assert parsed.minutes == 0
    assert parsed.remaining_seconds == 0
    assert parsed.microseconds == 0

    text = "P1,5D"
    parsed = parse(text)

    assert parsed.years == 0
    assert parsed.months == 0
    assert parsed.weeks == 0
    assert parsed.remaining_days == 1
    assert parsed.hours == 12
    assert parsed.minutes == 0
    assert parsed.remaining_seconds == 0
    assert parsed.microseconds == 0

    text = "PT1H"
    parsed = parse(text)

    assert parsed.years == 0
    assert parsed.months == 0
    assert parsed.weeks == 0
    assert parsed.remaining_days == 0
    assert parsed.hours == 1
    assert parsed.minutes == 0
    assert parsed.remaining_seconds == 0
    assert parsed.microseconds == 0

    text = "PT1.5H"
    parsed = parse(text)

    assert parsed.years == 0
    assert parsed.months == 0
    assert parsed.weeks == 0
    assert parsed.remaining_days == 0
    assert parsed.hours == 1
    assert parsed.minutes == 30
    assert parsed.remaining_seconds == 0
    assert parsed.microseconds == 0

    text = "PT1,5H"
    parsed = parse(text)

    assert parsed.years == 0
    assert parsed.months == 0
    assert parsed.weeks == 0
    assert parsed.remaining_days == 0
    assert parsed.hours == 1
    assert parsed.minutes == 30
    assert parsed.remaining_seconds == 0
    assert parsed.microseconds == 0


def test_parse_duration_no_operator():
    with pytest.raises(ParserError):
        parse("2Y3M4DT5H6M7S")


def test_parse_duration_weeks_combined():
    with pytest.raises(ParserError):
        parse("P1Y2W")


def test_parse_duration_invalid_order():
    with pytest.raises(ParserError):
        parse("P1S")

    with pytest.raises(ParserError):
        parse("P1D1S")

    with pytest.raises(ParserError):
        parse("1Y2M3D1SPT1M")

    with pytest.raises(ParserError):
        parse("P1Y2M3D2MT1S")

    with pytest.raises(ParserError):
        parse("P2M3D1ST1Y1M")

    with pytest.raises(ParserError):
        parse("P1Y2M2MT3D1S")

    with pytest.raises(ParserError):
        parse("P1D1Y1M")

    with pytest.raises(ParserError):
        parse("PT1S1H")


def test_parse_duration_invalid():
    with pytest.raises(ParserError):
        parse("P1Dasdfasdf")


def test_parse_duration_fraction_only_allowed_on_last_component():
    with pytest.raises(ParserError):
        parse("P2Y3M4DT5.5H6M7S")


# === tests/parsing/__init__.py ===


# === tests/parsing/test_parse_iso8601.py ===
from __future__ import annotations

from datetime import date
from datetime import datetime
from datetime import time

import pytest

from pendulum.parsing import parse_iso8601


try:
    from pendulum._pendulum import FixedTimezone
except ImportError:
    from pendulum.tz.timezone import FixedTimezone


@pytest.mark.parametrize(
    ["text", "expected"],
    [
        ("2016-10", date(2016, 10, 1)),
        ("2016-10-06", date(2016, 10, 6)),
        # Ordinal date
        ("2012-007", date(2012, 1, 7)),
        ("2012007", date(2012, 1, 7)),
        ("2017-079", date(2017, 3, 20)),
        # Week date
        ("2012-W05", date(2012, 1, 30)),
        ("2008-W39-6", date(2008, 9, 27)),
        ("2009-W53-7", date(2010, 1, 3)),
        ("2009-W01-1", date(2008, 12, 29)),
        # Time
        ("12:34", time(12, 34, 0)),
        ("12:34:56", time(12, 34, 56)),
        ("12:34:56.123", time(12, 34, 56, 123000)),
        ("12:34:56.123456", time(12, 34, 56, 123456)),
        ("12:34+05:30", time(12, 34, 0, tzinfo=FixedTimezone(19800))),
        ("12:34:56+05:30", time(12, 34, 56, tzinfo=FixedTimezone(19800))),
        ("12:34:56.123+05:30", time(12, 34, 56, 123000, tzinfo=FixedTimezone(19800))),
        (
            "12:34:56.123456+05:30",
            time(12, 34, 56, 123456, tzinfo=FixedTimezone(19800)),
        ),
        # Datetime
        ("2016-10-06T12:34:56.123456", datetime(2016, 10, 6, 12, 34, 56, 123456)),
        ("2016-10-06T12:34:56.123", datetime(2016, 10, 6, 12, 34, 56, 123000)),
        ("2016-10-06T12:34:56.000123", datetime(2016, 10, 6, 12, 34, 56, 123)),
        ("20161006T12", datetime(2016, 10, 6, 12, 0, 0, 0)),
        ("20161006T123456", datetime(2016, 10, 6, 12, 34, 56, 0)),
        ("20161006T123456.123456", datetime(2016, 10, 6, 12, 34, 56, 123456)),
        ("20161006 123456.123456", datetime(2016, 10, 6, 12, 34, 56, 123456)),
        # Datetime with offset
        (
            "2016-10-06T12:34:56.123456+05:30",
            datetime(2016, 10, 6, 12, 34, 56, 123456, FixedTimezone(19800)),
        ),
        (
            "2016-10-06T12:34:56.123456+0530",
            datetime(2016, 10, 6, 12, 34, 56, 123456, FixedTimezone(19800)),
        ),
        (
            "2016-10-06T12:34:56.123456-05:30",
            datetime(2016, 10, 6, 12, 34, 56, 123456, FixedTimezone(-19800)),
        ),
        (
            "2016-10-06T12:34:56.123456-0530",
            datetime(2016, 10, 6, 12, 34, 56, 123456, FixedTimezone(-19800)),
        ),
        (
            "2016-10-06T12:34:56.123456+05",
            datetime(2016, 10, 6, 12, 34, 56, 123456, FixedTimezone(18000)),
        ),
        (
            "2016-10-06T12:34:56.123456-05",
            datetime(2016, 10, 6, 12, 34, 56, 123456, FixedTimezone(-18000)),
        ),
        (
            "20161006T123456,123456-05",
            datetime(2016, 10, 6, 12, 34, 56, 123456, FixedTimezone(-18000)),
        ),
        (
            "2016-10-06T12:34:56.123456789+05:30",
            datetime(2016, 10, 6, 12, 34, 56, 123456, FixedTimezone(+19800)),
        ),
        # Week date with time
        ("2008-W39-6T09", datetime(2008, 9, 27, 9, 0, 0, 0)),
    ],
)
def test_parse_iso8601(text: str, expected: date) -> None:
    assert parse_iso8601(text) == expected


def test_parse_ios8601_invalid():
    # Invalid month
    with pytest.raises(ValueError):
        parse_iso8601("20161306T123456")

    # Invalid day
    with pytest.raises(ValueError):
        parse_iso8601("20161033T123456")

    # Invalid day for month
    with pytest.raises(ValueError):
        parse_iso8601("20161131T123456")

    # Invalid hour
    with pytest.raises(ValueError):
        parse_iso8601("20161006T243456")

    # Invalid minute
    with pytest.raises(ValueError):
        parse_iso8601("20161006T126056")

    # Invalid second
    with pytest.raises(ValueError):
        parse_iso8601("20161006T123460")

    # Extraneous separator
    with pytest.raises(ValueError):
        parse_iso8601("20140203 04:05:.123456")
    with pytest.raises(ValueError):
        parse_iso8601("2009-05-19 14:")

    # Invalid ordinal
    with pytest.raises(ValueError):
        parse_iso8601("2009367")
    with pytest.raises(ValueError):
        parse_iso8601("2009-367")
    with pytest.raises(ValueError):
        parse_iso8601("2015-366")
    with pytest.raises(ValueError):
        parse_iso8601("2015-000")

    # Invalid date
    with pytest.raises(ValueError):
        parse_iso8601("2009-")

    # Invalid time
    with pytest.raises(ValueError):
        parse_iso8601("2009-05-19T14:3924")
    with pytest.raises(ValueError):
        parse_iso8601("2010-02-18T16.5:23.35:48")
    with pytest.raises(ValueError):
        parse_iso8601("2010-02-18T16:23.35:48.45")
    with pytest.raises(ValueError):
        parse_iso8601("2010-02-18T16:23.33.600")

    # Invalid offset
    with pytest.raises(ValueError):
        parse_iso8601("2009-05-19 14:39:22+063")
    with pytest.raises(ValueError):
        parse_iso8601("2009-05-19 14:39:22+06a00")
    with pytest.raises(ValueError):
        parse_iso8601("2009-05-19 14:39:22+0:6:00")

    # Missing time separator
    with pytest.raises(ValueError):
        parse_iso8601("2009-05-1914:39")

    # Invalid week date
    with pytest.raises(ValueError):
        parse_iso8601("2012-W63")
    with pytest.raises(ValueError):
        parse_iso8601("2012-W12-9")
    with pytest.raises(ValueError):
        parse_iso8601("2012W12-3")  # Missing separator
    with pytest.raises(ValueError):
        parse_iso8601("2012-W123")  # Missing separator


@pytest.mark.parametrize(
    ["text", "expected"],
    [
        ("P2Y3M4DT5H6M7S", (2, 3, 0, 4, 5, 6, 7, 0)),
        ("P1Y2M3DT4H5M6.5S", (1, 2, 0, 3, 4, 5, 6, 500_000)),
        ("P1Y2M3DT4H5M6,5S", (1, 2, 0, 3, 4, 5, 6, 500_000)),
        ("P1Y2M3D", (1, 2, 0, 3, 0, 0, 0, 0)),
        ("P1Y2M3.5D", (1, 2, 0, 3, 12, 0, 0, 0)),
        ("P1Y2M3,5D", (1, 2, 0, 3, 12, 0, 0, 0)),
        ("PT4H54M6.5S", (0, 0, 0, 0, 4, 54, 6, 500_000)),
        ("PT4H54M6,5S", (0, 0, 0, 0, 4, 54, 6, 500_000)),
        ("P1Y", (1, 0, 0, 0, 0, 0, 0, 0)),
        ("P1M", (0, 1, 0, 0, 0, 0, 0, 0)),
        ("P1W", (0, 0, 1, 0, 0, 0, 0, 0)),
        ("P1.5W", (0, 0, 1, 3, 12, 0, 0, 0)),
        ("P1,5W", (0, 0, 1, 3, 12, 0, 0, 0)),
        ("P1D", (0, 0, 0, 1, 0, 0, 0, 0)),
        ("P1.5D", (0, 0, 0, 1, 12, 0, 0, 0)),
        ("P1,5D", (0, 0, 0, 1, 12, 0, 0, 0)),
        ("PT1H", (0, 0, 0, 0, 1, 0, 0, 0)),
        ("PT1.5H", (0, 0, 0, 0, 1, 30, 0, 0)),
        ("PT1,5H", (0, 0, 0, 0, 1, 30, 0, 0)),
        ("P2Y30M4DT5H6M7S", (2, 30, 0, 4, 5, 6, 7, 0)),
    ],
)
def test_parse_ios8601_duration(
    text: str, expected: tuple[int, int, int, int, int, int, int, int]
) -> None:
    parsed = parse_iso8601(text)

    assert (
        parsed.years,
        parsed.months,
        parsed.weeks,
        parsed.remaining_days,
        parsed.hours,
        parsed.minutes,
        parsed.remaining_seconds,
        parsed.microseconds,
    ) == expected


# === tests/duration/test_construct.py ===
from __future__ import annotations

from datetime import timedelta

import pytest

import pendulum

from pendulum.duration import AbsoluteDuration
from tests.conftest import assert_duration


def test_defaults():
    pi = pendulum.duration()
    assert_duration(pi, 0, 0, 0, 0, 0, 0, 0)


def test_years():
    pi = pendulum.duration(years=2)
    assert_duration(pi, years=2, weeks=0)
    assert pi.days == 730
    assert pi.total_seconds() == 63072000


def test_months():
    pi = pendulum.duration(months=3)
    assert_duration(pi, months=3, weeks=0)
    assert pi.days == 90
    assert pi.total_seconds() == 7776000


def test_weeks():
    pi = pendulum.duration(days=365)
    assert_duration(pi, weeks=52)

    pi = pendulum.duration(days=13)
    assert_duration(pi, weeks=1)


def test_days():
    pi = pendulum.duration(days=6)
    assert_duration(pi, 0, 0, 0, 6, 0, 0, 0)

    pi = pendulum.duration(days=16)
    assert_duration(pi, 0, 0, 2, 2, 0, 0, 0)


def test_hours():
    pi = pendulum.duration(seconds=3600 * 3)
    assert_duration(pi, 0, 0, 0, 0, 3, 0, 0)


def test_minutes():
    pi = pendulum.duration(seconds=60 * 3)
    assert_duration(pi, 0, 0, 0, 0, 0, 3, 0)

    pi = pendulum.duration(seconds=60 * 3 + 12)
    assert_duration(pi, 0, 0, 0, 0, 0, 3, 12)


def test_all():
    pi = pendulum.duration(
        years=2, months=3, days=1177, seconds=7284, microseconds=1000000
    )
    assert_duration(pi, 2, 3, 168, 1, 2, 1, 25)
    assert pi.days == 1997
    assert pi.seconds == 7285


def test_absolute_interval():
    pi = AbsoluteDuration(days=-1177, seconds=-7284, microseconds=-1000001)
    assert_duration(pi, 0, 0, 168, 1, 2, 1, 25)
    assert pi.microseconds == 1
    assert pi.invert


def test_invert():
    pi = pendulum.duration(days=1177, seconds=7284, microseconds=1000000)
    assert not pi.invert

    pi = pendulum.duration(days=-1177, seconds=-7284, microseconds=-1000000)
    assert pi.invert


def test_as_timedelta():
    pi = pendulum.duration(seconds=3456.123456)
    assert_duration(pi, 0, 0, 0, 0, 0, 57, 36, 123456)
    delta = pi.as_timedelta()
    assert isinstance(delta, timedelta)
    assert delta.total_seconds() == 3456.123456
    assert delta.seconds == 3456


def test_float_years_and_months():
    with pytest.raises(ValueError):
        pendulum.duration(years=1.5)

    with pytest.raises(ValueError):
        pendulum.duration(months=1.5)


# === tests/duration/test_total_methods.py ===
from __future__ import annotations

import pendulum


def test_in_weeks():
    it = pendulum.duration(days=17)
    assert round(it.total_weeks(), 2) == 2.43


def test_in_days():
    it = pendulum.duration(days=3)
    assert it.total_days() == 3


def test_in_hours():
    it = pendulum.duration(days=3, minutes=72)
    assert it.total_hours() == 73.2


def test_in_minutes():
    it = pendulum.duration(minutes=6, seconds=72)
    assert it.total_minutes() == 7.2


def test_in_seconds():
    it = pendulum.duration(seconds=72, microseconds=123456)
    assert it.total_seconds() == 72.123456


# === tests/duration/test_in_methods.py ===
from __future__ import annotations

import pendulum


def test_in_weeks():
    it = pendulum.duration(days=17)
    assert it.in_weeks() == 2


def test_in_days():
    it = pendulum.duration(days=3)
    assert it.in_days() == 3


def test_in_hours():
    it = pendulum.duration(days=3, minutes=72)
    assert it.in_hours() == 73


def test_in_minutes():
    it = pendulum.duration(minutes=6, seconds=72)
    assert it.in_minutes() == 7


def test_in_seconds():
    it = pendulum.duration(seconds=72)
    assert it.in_seconds() == 72


# === tests/duration/__init__.py ===


# === tests/duration/test_behavior.py ===
from __future__ import annotations

import pickle

from copy import deepcopy
from datetime import timedelta

import pendulum

from tests.conftest import assert_duration


def test_pickle() -> None:
    it = pendulum.duration(days=3, seconds=2456, microseconds=123456)
    s = pickle.dumps(it)
    it2 = pickle.loads(s)

    assert it == it2


def test_comparison_to_timedelta() -> None:
    duration = pendulum.duration(days=3)

    assert duration < timedelta(days=4)


def test_deepcopy() -> None:
    duration = pendulum.duration(months=1)
    copied_duration = deepcopy(duration)

    assert copied_duration == duration
    assert_duration(copied_duration, months=1)


# === tests/duration/test_in_words.py ===
from __future__ import annotations

import pendulum


def test_week():
    assert pendulum.duration(days=364).in_words() == "52 weeks"
    assert pendulum.duration(days=7).in_words() == "1 week"


def test_week_to_string():
    assert str(pendulum.duration(days=364)) == "52 weeks"
    assert str(pendulum.duration(days=7)) == "1 week"


def test_weeks_and_day():
    assert pendulum.duration(days=365).in_words() == "52 weeks 1 day"


def test_all():
    pi = pendulum.duration(
        years=2, months=3, days=1177, seconds=7284, microseconds=1000000
    )

    expected = "2 years 3 months 168 weeks 1 day 2 hours 1 minute 25 seconds"
    assert pi.in_words() == expected


def test_in_french():
    pi = pendulum.duration(
        years=2, months=3, days=1177, seconds=7284, microseconds=1000000
    )

    expected = "2 ans 3 mois 168 semaines 1 jour 2 heures 1 minute 25 secondes"
    assert pi.in_words(locale="fr") == expected


def test_repr():
    pi = pendulum.duration(
        years=2, months=3, days=1177, seconds=7284, microseconds=1000000
    )

    expected = (
        "Duration(years=2, months=3, weeks=168, days=1, hours=2, minutes=1, seconds=25)"
    )
    assert repr(pi) == expected


def test_singular_negative_values():
    pi = pendulum.duration(days=-1)

    assert pi.in_words() == "-1 day"


def test_separator():
    pi = pendulum.duration(days=1177, seconds=7284, microseconds=1000000)

    expected = "168 weeks, 1 day, 2 hours, 1 minute, 25 seconds"
    assert pi.in_words(separator=", ") == expected


def test_subseconds():
    pi = pendulum.duration(microseconds=123456)

    assert pi.in_words() == "0.12 second"


def test_subseconds_with_seconds():
    pi = pendulum.duration(seconds=12, microseconds=123456)

    assert pi.in_words() == "12 seconds"


def test_duration_with_all_zero_values():
    pi = pendulum.duration()

    assert pi.in_words() == "0 microseconds"


# === tests/duration/test_arithmetic.py ===
from __future__ import annotations

import pendulum

from tests.conftest import assert_duration


def test_multiply():
    it = pendulum.duration(days=6, seconds=34, microseconds=522222)
    mul = it * 2

    assert isinstance(mul, pendulum.Duration)
    assert_duration(mul, 0, 0, 1, 5, 0, 1, 9, 44444)

    it = pendulum.duration(days=6, seconds=34, microseconds=522222)
    mul = 2 * it

    assert isinstance(mul, pendulum.Duration)
    assert_duration(mul, 0, 0, 1, 5, 0, 1, 9, 44444)

    it = pendulum.duration(
        years=2, months=3, weeks=4, days=6, seconds=34, microseconds=522222
    )
    mul = 2 * it

    assert isinstance(mul, pendulum.Duration)
    assert_duration(mul, 4, 6, 9, 5, 0, 1, 9, 44444)


def test_divide():
    it = pendulum.duration(days=2, seconds=34, microseconds=522222)
    mul = it / 2

    assert isinstance(mul, pendulum.Duration)
    assert_duration(mul, 0, 0, 0, 1, 0, 0, 17, 261111)

    it = pendulum.duration(days=2, seconds=35, microseconds=522222)
    mul = it / 2

    assert isinstance(mul, pendulum.Duration)
    assert_duration(mul, 0, 0, 0, 1, 0, 0, 17, 761111)

    it = pendulum.duration(days=2, seconds=35, microseconds=522222)
    mul = it / 1.1

    assert isinstance(mul, pendulum.Duration)
    assert_duration(mul, 0, 0, 0, 1, 19, 38, 43, 202020)

    it = pendulum.duration(years=2, months=4, days=2, seconds=35, microseconds=522222)
    mul = it / 2

    assert isinstance(mul, pendulum.Duration)
    assert_duration(mul, 1, 2, 0, 1, 0, 0, 17, 761111)

    it = pendulum.duration(years=2, months=4, days=2, seconds=35, microseconds=522222)
    mul = it / 2.0

    assert isinstance(mul, pendulum.Duration)
    assert_duration(mul, 1, 2, 0, 1, 0, 0, 17, 761111)


def test_floor_divide():
    it = pendulum.duration(days=2, seconds=34, microseconds=522222)
    mul = it // 2

    assert isinstance(mul, pendulum.Duration)
    assert_duration(mul, 0, 0, 0, 1, 0, 0, 17, 261111)

    it = pendulum.duration(days=2, seconds=35, microseconds=522222)
    mul = it // 3

    assert isinstance(mul, pendulum.Duration)
    assert_duration(mul, 0, 0, 0, 0, 16, 0, 11, 840740)

    it = pendulum.duration(years=2, months=4, days=2, seconds=34, microseconds=522222)
    mul = it // 2

    assert isinstance(mul, pendulum.Duration)
    assert_duration(mul, 1, 2, 0, 1, 0, 0, 17, 261111)

    it = pendulum.duration(years=2, months=4, days=2, seconds=35, microseconds=522222)
    mul = it // 3

    assert isinstance(mul, pendulum.Duration)
    assert_duration(mul, 0, 1, 0, 0, 16, 0, 11, 840740)


# === tests/duration/test_add_sub.py ===
from __future__ import annotations

from datetime import timedelta

import pendulum

from tests.conftest import assert_duration


def test_add_interval():
    p1 = pendulum.duration(days=23, seconds=32)
    p2 = pendulum.duration(days=12, seconds=30)

    p = p1 + p2
    assert_duration(p, 0, 0, 5, 0, 0, 1, 2)


def test_add_timedelta():
    p1 = pendulum.duration(days=23, seconds=32)
    p2 = timedelta(days=12, seconds=30)

    p = p1 + p2
    assert_duration(p, 0, 0, 5, 0, 0, 1, 2)


def test_add_unsupported():
    p = pendulum.duration(days=23, seconds=32)
    assert NotImplemented == p.__add__(5)


def test_sub_interval():
    p1 = pendulum.duration(days=23, seconds=32)
    p2 = pendulum.duration(days=12, seconds=28)

    p = p1 - p2
    assert_duration(p, 0, 0, 1, 4, 0, 0, 4)


def test_sub_timedelta():
    p1 = pendulum.duration(days=23, seconds=32)
    p2 = timedelta(days=12, seconds=28)

    p = p1 - p2
    assert_duration(p, 0, 0, 1, 4, 0, 0, 4)


def test_sub_unsupported():
    p = pendulum.duration(days=23, seconds=32)
    assert NotImplemented == p.__sub__(5)


def test_neg():
    p = pendulum.duration(days=23, seconds=32)
    assert_duration(-p, 0, 0, -3, -2, 0, 0, -32)


# === tests/tz/test_local_timezone.py ===
from __future__ import annotations

import os
import sys

import pytest

from pendulum.tz.local_timezone import _get_unix_timezone
from pendulum.tz.local_timezone import _get_windows_timezone


@pytest.mark.skipif(
    sys.platform == "win32", reason="Test only available for UNIX systems"
)
def test_unix_symlink():
    # A ZONE setting in the target path of a symbolic linked localtime,
    # f ex systemd distributions
    local_path = os.path.join(os.path.split(__file__)[0], "..")
    tz = _get_unix_timezone(_root=os.path.join(local_path, "fixtures", "tz", "symlink"))

    assert tz.name == "Europe/Paris"


@pytest.mark.skipif(
    sys.platform == "win32", reason="Test only available for UNIX systems"
)
def test_unix_clock():
    # A ZONE setting in the target path of a symbolic linked localtime,
    # f ex systemd distributions
    local_path = os.path.join(os.path.split(__file__)[0], "..")
    tz = _get_unix_timezone(_root=os.path.join(local_path, "fixtures", "tz", "clock"))

    assert tz.name == "Europe/Zurich"


@pytest.mark.skipif(sys.platform != "win32", reason="Test only available for Windows")
def test_windows_timezone():
    timezone = _get_windows_timezone()

    assert timezone is not None


@pytest.mark.skipif(
    sys.platform == "win32", reason="Test only available for UNIX systems"
)
def test_unix_etc_timezone_dir():
    # Should not fail if `/etc/timezone` is a folder
    local_path = os.path.join(os.path.split(__file__)[0], "..")
    root_path = os.path.join(local_path, "fixtures", "tz", "timezone_dir")
    tz = _get_unix_timezone(_root=root_path)

    assert tz.name == "Europe/Paris"


# === tests/tz/test_timezones.py ===
from __future__ import annotations

import pytest

import pendulum


def test_timezones():
    zones = pendulum.timezones()

    assert "America/Argentina/Buenos_Aires" in zones


@pytest.mark.parametrize("zone", list(pendulum.timezones()))
def test_timezones_are_loadable(zone):
    pendulum.timezone(zone)


# === tests/tz/__init__.py ===


# === tests/tz/test_timezone.py ===
from __future__ import annotations

import zoneinfo

from datetime import datetime
from datetime import timedelta

import pytest

import pendulum

from pendulum import timezone
from pendulum.tz import fixed_timezone
from pendulum.tz.exceptions import AmbiguousTime
from pendulum.tz.exceptions import NonExistingTime
from tests.conftest import assert_datetime


@pytest.fixture(autouse=True)
def setup():
    pendulum.tz._tz_cache = {}

    yield

    pendulum.tz._tz_cache = {}


def test_basic_convert():
    dt = datetime(2016, 6, 1, 12, 34, 56, 123456, fold=1)
    tz = timezone("Europe/Paris")
    dt = tz.convert(dt)

    assert dt.year == 2016
    assert dt.month == 6
    assert dt.day == 1
    assert dt.hour == 12
    assert dt.minute == 34
    assert dt.second == 56
    assert dt.microsecond == 123456
    assert dt.tzinfo.name == "Europe/Paris"
    assert dt.tzinfo.utcoffset(dt) == timedelta(seconds=7200)
    assert dt.tzinfo.dst(dt) == timedelta(seconds=3600)


def test_equality():
    assert timezone("Europe/Paris") == timezone("Europe/Paris")
    assert timezone("Europe/Paris") != timezone("Europe/Berlin")


def test_skipped_time_with_pre_rule():
    dt = datetime(2013, 3, 31, 2, 30, 45, 123456, fold=0)
    tz = timezone("Europe/Paris")
    dt = tz.convert(dt)

    assert dt.year == 2013
    assert dt.month == 3
    assert dt.day == 31
    assert dt.hour == 1
    assert dt.minute == 30
    assert dt.second == 45
    assert dt.microsecond == 123456
    assert dt.tzinfo.name == "Europe/Paris"
    assert dt.tzinfo.utcoffset(dt) == timedelta(seconds=3600)
    assert dt.tzinfo.dst(dt) == timedelta()


def test_skipped_time_with_post_rule():
    dt = datetime(2013, 3, 31, 2, 30, 45, 123456, fold=1)
    tz = timezone("Europe/Paris")
    dt = tz.convert(dt)

    assert dt.year == 2013
    assert dt.month == 3
    assert dt.day == 31
    assert dt.hour == 3
    assert dt.minute == 30
    assert dt.second == 45
    assert dt.microsecond == 123456
    assert dt.tzinfo.name == "Europe/Paris"
    assert dt.tzinfo.utcoffset(dt) == timedelta(seconds=7200)
    assert dt.tzinfo.dst(dt) == timedelta(seconds=3600)


def test_skipped_time_with_error():
    dt = datetime(2013, 3, 31, 2, 30, 45, 123456)
    tz = timezone("Europe/Paris")
    with pytest.raises(NonExistingTime):
        tz.convert(dt, raise_on_unknown_times=True)


def test_repeated_time():
    dt = datetime(2013, 10, 27, 2, 30, 45, 123456, fold=1)
    tz = timezone("Europe/Paris")
    dt = tz.convert(dt)

    assert dt.year == 2013
    assert dt.month == 10
    assert dt.day == 27
    assert dt.hour == 2
    assert dt.minute == 30
    assert dt.second == 45
    assert dt.microsecond == 123456
    assert dt.tzinfo.name == "Europe/Paris"
    assert dt.tzinfo.utcoffset(dt) == timedelta(seconds=3600)
    assert dt.tzinfo.dst(dt) == timedelta()


def test_repeated_time_pre_rule():
    dt = datetime(2013, 10, 27, 2, 30, 45, 123456, fold=0)
    tz = timezone("Europe/Paris")
    dt = tz.convert(dt)

    assert dt.year == 2013
    assert dt.month == 10
    assert dt.day == 27
    assert dt.hour == 2
    assert dt.minute == 30
    assert dt.second == 45
    assert dt.microsecond == 123456
    assert dt.tzinfo.name == "Europe/Paris"
    assert dt.tzinfo.utcoffset(dt) == timedelta(seconds=7200)
    assert dt.tzinfo.dst(dt) == timedelta(seconds=3600)


def test_repeated_time_with_error():
    dt = datetime(2013, 10, 27, 2, 30, 45, 123456)
    tz = timezone("Europe/Paris")
    with pytest.raises(AmbiguousTime):
        tz.convert(dt, raise_on_unknown_times=True)


def test_pendulum_create_basic():
    dt = pendulum.datetime(2016, 6, 1, 12, 34, 56, 123456, tz="Europe/Paris")

    assert_datetime(dt, 2016, 6, 1, 12, 34, 56, 123456)
    assert dt.timezone_name == "Europe/Paris"
    assert dt.offset == 7200
    assert dt.is_dst()


def test_pendulum_create_skipped():
    dt = pendulum.datetime(2013, 3, 31, 2, 30, 45, 123456, tz="Europe/Paris")

    assert isinstance(dt, pendulum.DateTime)
    assert_datetime(dt, 2013, 3, 31, 3, 30, 45, 123456)
    assert dt.timezone_name == "Europe/Paris"
    assert dt.tzinfo.utcoffset(dt) == timedelta(seconds=7200)
    assert dt.tzinfo.dst(dt) == timedelta(seconds=3600)


def test_pendulum_create_skipped_with_pre_rule():
    dt = pendulum.datetime(2013, 3, 31, 2, 30, 45, 123456, tz="Europe/Paris", fold=0)

    assert_datetime(dt, 2013, 3, 31, 1, 30, 45, 123456)
    assert dt.timezone_name == "Europe/Paris"
    assert dt.tzinfo.utcoffset(dt) == timedelta(seconds=3600)
    assert dt.tzinfo.dst(dt) == timedelta()


def test_pendulum_create_skipped_with_error():
    with pytest.raises(NonExistingTime):
        pendulum.datetime(
            2013,
            3,
            31,
            2,
            30,
            45,
            123456,
            tz="Europe/Paris",
            raise_on_unknown_times=True,
        )


def test_pendulum_create_repeated():
    dt = pendulum.datetime(2013, 10, 27, 2, 30, 45, 123456, tz="Europe/Paris")

    assert_datetime(dt, 2013, 10, 27, 2, 30, 45, 123456)
    assert dt.timezone_name == "Europe/Paris"
    assert dt.tzinfo.utcoffset(dt) == timedelta(seconds=3600)
    assert dt.tzinfo.dst(dt) == timedelta()


def test_pendulum_create_repeated_with_pre_rule():
    dt = pendulum.datetime(
        2013,
        10,
        27,
        2,
        30,
        45,
        123456,
        tz="Europe/Paris",
        fold=0,
    )

    assert_datetime(dt, 2013, 10, 27, 2, 30, 45, 123456)
    assert dt.timezone_name == "Europe/Paris"
    assert dt.tzinfo.utcoffset(dt) == timedelta(seconds=7200)
    assert dt.tzinfo.dst(dt) == timedelta(seconds=3600)


def test_pendulum_create_repeated_with_error():
    with pytest.raises(AmbiguousTime):
        pendulum.datetime(
            2013,
            10,
            27,
            2,
            30,
            45,
            123456,
            tz="Europe/Paris",
            raise_on_unknown_times=True,
        )


def test_convert_accept_pendulum_instance():
    dt = pendulum.datetime(2016, 8, 7, 12, 53, 54)
    tz = timezone("Europe/Paris")
    new = tz.convert(dt)

    assert isinstance(new, pendulum.DateTime)
    assert_datetime(new, 2016, 8, 7, 14, 53, 54)


def test_utcoffset():
    tz = pendulum.timezone("America/Guayaquil")
    utcoffset = tz.utcoffset(pendulum.now("UTC"))
    assert utcoffset == timedelta(0, -18000)


def test_utcoffset_pre_transition():
    tz = pendulum.timezone("America/Chicago")
    utcoffset = tz.utcoffset(datetime(1883, 11, 18))
    assert utcoffset == timedelta(days=-1, seconds=65364)


def test_dst():
    tz = pendulum.timezone("Europe/Amsterdam")
    dst = tz.dst(datetime(1940, 7, 1))
    native_tz = zoneinfo.ZoneInfo("Europe/Amsterdam")

    assert dst == native_tz.dst(datetime(1940, 7, 1))


def test_short_timezones_should_not_modify_time():
    tz = pendulum.timezone("EST")
    dt = tz.datetime(2017, 6, 15, 14, 0, 0)

    assert dt.year == 2017
    assert dt.month == 6
    assert dt.day == 15
    assert dt.hour == 14
    assert dt.minute == 0
    assert dt.second == 0

    tz = pendulum.timezone("HST")
    dt = tz.datetime(2017, 6, 15, 14, 0, 0)

    assert dt.year == 2017
    assert dt.month == 6
    assert dt.day == 15
    assert dt.hour == 14
    assert dt.minute == 0
    assert dt.second == 0


def test_after_last_transition():
    tz = pendulum.timezone("Europe/Paris")
    dt = tz.datetime(2135, 6, 15, 14, 0, 0)

    assert dt.year == 2135
    assert dt.month == 6
    assert dt.day == 15
    assert dt.hour == 14
    assert dt.minute == 0
    assert dt.second == 0
    assert dt.microsecond == 0


@pytest.mark.skip(
    reason=(
        "zoneinfo does not currently support POSIX transition"
        " rules to go beyond the last fixed transition."
    )
)
def test_on_last_transition():
    tz = pendulum.timezone("Europe/Paris")
    dt = pendulum.naive(2037, 10, 25, 2, 30)
    dt = tz.convert(dt, dst_rule=pendulum.POST_TRANSITION)

    assert dt.year == 2037
    assert dt.month == 10
    assert dt.day == 25
    assert dt.hour == 2
    assert dt.minute == 30
    assert dt.second == 0
    assert dt.microsecond == 0
    assert dt.utcoffset().total_seconds() == 3600

    dt = pendulum.naive(2037, 10, 25, 2, 30)
    dt = tz.convert(dt, dst_rule=pendulum.PRE_TRANSITION)

    assert dt.year == 2037
    assert dt.month == 10
    assert dt.day == 25
    assert dt.hour == 2
    assert dt.minute == 30
    assert dt.second == 0
    assert dt.microsecond == 0
    assert dt.utcoffset().total_seconds() == 7200


def test_convert_fold_attribute_is_honored():
    tz = pendulum.timezone("US/Eastern")
    dt = datetime(2014, 11, 2, 1, 30)

    new = tz.convert(dt)
    assert new.strftime("%z") == "-0400"

    new = tz.convert(dt.replace(fold=1))
    assert new.strftime("%z") == "-0500"


def test_utcoffset_fold_attribute_is_honored():
    tz = pendulum.timezone("US/Eastern")
    dt = datetime(2014, 11, 2, 1, 30)

    offset = tz.utcoffset(dt)

    assert offset.total_seconds() == -4 * 3600

    offset = tz.utcoffset(dt.replace(fold=1))

    assert offset.total_seconds() == -5 * 3600


def test_dst_fold_attribute_is_honored():
    tz = pendulum.timezone("US/Eastern")
    dt = datetime(2014, 11, 2, 1, 30)

    offset = tz.dst(dt)

    assert offset.total_seconds() == 3600

    offset = tz.dst(dt.replace(fold=1))

    assert offset.total_seconds() == 0


def test_tzname_fold_attribute_is_honored():
    tz = pendulum.timezone("US/Eastern")
    dt = datetime(2014, 11, 2, 1, 30)

    name = tz.tzname(dt)

    assert name == "EDT"

    name = tz.tzname(dt.replace(fold=1))

    assert name == "EST"


def test_constructor_fold_attribute_is_honored():
    tz = pendulum.timezone("US/Eastern")
    dt = datetime(2014, 11, 2, 1, 30, tzinfo=tz)

    assert dt.strftime("%z") == "-0400"

    dt = datetime(2014, 11, 2, 1, 30, tzinfo=tz, fold=1)

    assert dt.strftime("%z") == "-0500"


def test_datetime():
    tz = timezone("Europe/Paris")

    dt = tz.datetime(2013, 3, 24, 1, 30)
    assert dt.year == 2013
    assert dt.month == 3
    assert dt.day == 24
    assert dt.hour == 1
    assert dt.minute == 30
    assert dt.second == 0
    assert dt.microsecond == 0

    dt = tz.datetime(2013, 3, 31, 2, 30)
    assert dt.year == 2013
    assert dt.month == 3
    assert dt.day == 31
    assert dt.hour == 3
    assert dt.minute == 30
    assert dt.second == 0
    assert dt.microsecond == 0


def test_fixed_timezone():
    tz = fixed_timezone(19800)
    tz2 = fixed_timezone(18000)
    dt = datetime(2016, 11, 26, tzinfo=tz)

    assert tz2.utcoffset(dt).total_seconds() == 18000
    assert tz2.dst(dt) == timedelta()


def test_fixed_equality():
    assert fixed_timezone(19800) == fixed_timezone(19800)
    assert fixed_timezone(19800) != fixed_timezone(19801)


def test_just_before_last_transition():
    tz = pendulum.timezone("Asia/Shanghai")
    dt = datetime(1991, 4, 20, 1, 49, 8, fold=0)
    dt = tz.convert(dt)

    epoch = datetime(1970, 1, 1, tzinfo=timezone("UTC"))
    expected = (dt - epoch).total_seconds()
    assert expected == 672079748.0


@pytest.mark.skip(
    reason=(
        "zoneinfo does not currently support POSIX transition"
        " rules to go beyond the last fixed transition."
    )
)
def test_timezones_are_extended():
    tz = pendulum.timezone("Europe/Paris")
    dt = tz.convert(pendulum.naive(2134, 2, 13, 1))

    assert_datetime(dt, 2134, 2, 13, 1)
    assert dt.utcoffset().total_seconds() == 3600
    assert dt.dst() == timedelta()

    dt = tz.convert(pendulum.naive(2134, 3, 28, 2, 30))

    assert_datetime(dt, 2134, 3, 28, 3, 30)
    assert dt.utcoffset().total_seconds() == 7200
    assert dt.dst() == timedelta(seconds=3600)

    dt = tz.convert(pendulum.naive(2134, 7, 11, 2, 30))

    assert_datetime(dt, 2134, 7, 11, 2, 30)
    assert dt.utcoffset().total_seconds() == 7200
    assert dt.dst() == timedelta(seconds=3600)

    dt = tz.convert(pendulum.naive(2134, 10, 31, 2, 30, fold=0))

    assert_datetime(dt, 2134, 10, 31, 2, 30)
    assert dt.utcoffset().total_seconds() == 7200
    assert dt.dst() == timedelta(seconds=3600)

    dt = tz.convert(pendulum.naive(2134, 10, 31, 2, 30))

    assert_datetime(dt, 2134, 10, 31, 2, 30)
    assert dt.utcoffset().total_seconds() == 3600
    assert dt.dst() == timedelta()


def test_repr():
    tz = timezone("Europe/Paris")

    assert repr(tz) == "Timezone('Europe/Paris')"


# === tests/tz/test_helpers.py ===
from __future__ import annotations

import pytest

from pendulum import timezone
from pendulum.tz.exceptions import InvalidTimezone
from pendulum.tz.timezone import FixedTimezone
from pendulum.tz.timezone import Timezone


def test_timezone_with_name():
    tz = timezone("Europe/Paris")

    assert isinstance(tz, Timezone)
    assert tz.name == "Europe/Paris"


def test_timezone_with_invalid_name():
    with pytest.raises(InvalidTimezone):
        timezone("Invalid")


def test_timezone_with_offset():
    tz = timezone(-19800)

    assert isinstance(tz, FixedTimezone)
    assert tz.name == "-05:30"


# === tests/helpers/__init__.py ===


# === tests/helpers/test_local_time.py ===
from __future__ import annotations

import pendulum

from pendulum.helpers import local_time


def test_local_time_positive_integer():
    d = pendulum.datetime(2016, 8, 7, 12, 34, 56, 123456)

    t = local_time(d.int_timestamp, 0, d.microsecond)
    assert d.year == t[0]
    assert d.month == t[1]
    assert d.day == t[2]
    assert d.hour == t[3]
    assert d.minute == t[4]
    assert d.second == t[5]
    assert d.microsecond == t[6]


def test_local_time_negative_integer():
    d = pendulum.datetime(1951, 8, 7, 12, 34, 56, 123456)

    t = local_time(d.int_timestamp, 0, d.microsecond)
    assert d.year == t[0]
    assert d.month == t[1]
    assert d.day == t[2]
    assert d.hour == t[3]
    assert d.minute == t[4]
    assert d.second == t[5]
    assert d.microsecond == t[6]


# === tests/localization/test_id.py ===
from __future__ import annotations

import pendulum


locale = "id"


def test_diff_for_humans():
    with pendulum.travel_to(pendulum.datetime(2016, 8, 29), freeze=True):
        diff_for_humans()


def diff_for_humans():
    d = pendulum.now().subtract(seconds=1)
    assert d.diff_for_humans(locale=locale) == "beberapa detik yang lalu"

    d = pendulum.now().subtract(seconds=2)
    assert d.diff_for_humans(locale=locale) == "beberapa detik yang lalu"

    d = pendulum.now().subtract(seconds=21)
    assert d.diff_for_humans(locale=locale) == "21 detik yang lalu"

    d = pendulum.now().subtract(minutes=1)
    assert d.diff_for_humans(locale=locale) == "1 menit yang lalu"

    d = pendulum.now().subtract(minutes=2)
    assert d.diff_for_humans(locale=locale) == "2 menit yang lalu"

    d = pendulum.now().subtract(hours=1)
    assert d.diff_for_humans(locale=locale) == "1 jam yang lalu"

    d = pendulum.now().subtract(hours=2)
    assert d.diff_for_humans(locale=locale) == "2 jam yang lalu"

    d = pendulum.now().subtract(days=1)
    assert d.diff_for_humans(locale=locale) == "1 hari yang lalu"

    d = pendulum.now().subtract(days=2)
    assert d.diff_for_humans(locale=locale) == "2 hari yang lalu"

    d = pendulum.now().subtract(weeks=1)
    assert d.diff_for_humans(locale=locale) == "1 minggu yang lalu"

    d = pendulum.now().subtract(weeks=2)
    assert d.diff_for_humans(locale=locale) == "2 minggu yang lalu"

    d = pendulum.now().subtract(months=1)
    assert d.diff_for_humans(locale=locale) == "1 bulan yang lalu"

    d = pendulum.now().subtract(months=2)
    assert d.diff_for_humans(locale=locale) == "2 bulan yang lalu"

    d = pendulum.now().subtract(years=1)
    assert d.diff_for_humans(locale=locale) == "1 tahun yang lalu"

    d = pendulum.now().subtract(years=2)
    assert d.diff_for_humans(locale=locale) == "2 tahun yang lalu"

    d = pendulum.now().add(seconds=1)
    assert d.diff_for_humans(locale=locale) == "dalam beberapa detik"

    d = pendulum.now().add(seconds=1)
    d2 = pendulum.now()
    assert d.diff_for_humans(d2, locale=locale) == "beberapa detik kemudian"
    assert d2.diff_for_humans(d, locale=locale) == "beberapa detik yang lalu"

    assert d.diff_for_humans(d2, True, locale=locale) == "beberapa detik"
    assert d2.diff_for_humans(d.add(seconds=1), True, locale=locale) == "beberapa detik"


# === tests/localization/test_ja.py ===
from __future__ import annotations

import pendulum


locale = "ja"


def test_diff_for_humans():
    with pendulum.travel_to(pendulum.datetime(2016, 8, 29), freeze=True):
        diff_for_humans()


def diff_for_humans():
    d = pendulum.now().subtract(seconds=1)
    assert d.diff_for_humans(locale=locale) == "数秒 前に"

    d = pendulum.now().subtract(seconds=2)
    assert d.diff_for_humans(locale=locale) == "数秒 前に"

    d = pendulum.now().subtract(seconds=21)
    assert d.diff_for_humans(locale=locale) == "21 秒前"

    d = pendulum.now().subtract(minutes=1)
    assert d.diff_for_humans(locale=locale) == "1 分前"

    d = pendulum.now().subtract(minutes=2)
    assert d.diff_for_humans(locale=locale) == "2 分前"

    d = pendulum.now().subtract(hours=1)
    assert d.diff_for_humans(locale=locale) == "1 時間前"

    d = pendulum.now().subtract(hours=2)
    assert d.diff_for_humans(locale=locale) == "2 時間前"

    d = pendulum.now().subtract(days=1)
    assert d.diff_for_humans(locale=locale) == "1 日前"

    d = pendulum.now().subtract(days=2)
    assert d.diff_for_humans(locale=locale) == "2 日前"

    d = pendulum.now().subtract(weeks=1)
    assert d.diff_for_humans(locale=locale) == "1 週間前"

    d = pendulum.now().subtract(weeks=2)
    assert d.diff_for_humans(locale=locale) == "2 週間前"

    d = pendulum.now().subtract(months=1)
    assert d.diff_for_humans(locale=locale) == "1 か月前"

    d = pendulum.now().subtract(months=2)
    assert d.diff_for_humans(locale=locale) == "2 か月前"

    d = pendulum.now().subtract(years=1)
    assert d.diff_for_humans(locale=locale) == "1 年前"

    d = pendulum.now().subtract(years=2)
    assert d.diff_for_humans(locale=locale) == "2 年前"

    d = pendulum.now().add(seconds=1)
    assert d.diff_for_humans(locale=locale) == "今から 数秒"

    d = pendulum.now().add(seconds=1)
    d2 = pendulum.now()
    assert d.diff_for_humans(d2, locale=locale) == "数秒 後"
    assert d2.diff_for_humans(d, locale=locale) == "数秒 前"

    assert d.diff_for_humans(d2, True, locale=locale) == "数秒"
    assert d2.diff_for_humans(d.add(seconds=1), True, locale=locale) == "数秒"


# === tests/localization/test_he.py ===
from __future__ import annotations

import pendulum


locale = "he"


def test_diff_for_humans():
    with pendulum.travel_to(pendulum.datetime(2016, 8, 29), freeze=True):
        diff_for_humans()


def diff_for_humans():
    d = pendulum.now().subtract(seconds=1)
    assert d.diff_for_humans(locale=locale) == "לפני כמה שניות"

    d = pendulum.now().subtract(seconds=2)
    assert d.diff_for_humans(locale=locale) == "לפני כמה שניות"

    d = pendulum.now().subtract(minutes=1)
    assert d.diff_for_humans(locale=locale) == "לפני דקה"

    d = pendulum.now().subtract(minutes=2)
    assert d.diff_for_humans(locale=locale) == "לפני שתי דקות"

    d = pendulum.now().subtract(hours=1)
    assert d.diff_for_humans(locale=locale) == "לפני שעה"

    d = pendulum.now().subtract(hours=2)
    assert d.diff_for_humans(locale=locale) == "לפני שעתיים"

    d = pendulum.now().subtract(days=1)
    assert d.diff_for_humans(locale=locale) == "לפני יום 1"

    d = pendulum.now().subtract(days=2)
    assert d.diff_for_humans(locale=locale) == "לפני יומיים"

    d = pendulum.now().subtract(weeks=1)
    assert d.diff_for_humans(locale=locale) == "לפני שבוע"

    d = pendulum.now().subtract(weeks=2)
    assert d.diff_for_humans(locale=locale) == "לפני שבועיים"

    d = pendulum.now().subtract(months=1)
    assert d.diff_for_humans(locale=locale) == "לפני חודש"

    d = pendulum.now().subtract(months=2)
    assert d.diff_for_humans(locale=locale) == "לפני חודשיים"

    d = pendulum.now().subtract(years=1)
    assert d.diff_for_humans(locale=locale) == "לפני שנה"

    d = pendulum.now().subtract(years=2)
    assert d.diff_for_humans(locale=locale) == "לפני שנתיים"

    d = pendulum.now().add(seconds=1)
    assert d.diff_for_humans(locale=locale) == "תוך כמה שניות"

    d = pendulum.now().add(seconds=1)
    d2 = pendulum.now()
    assert d.diff_for_humans(d2, locale=locale) == "בעוד כמה שניות"
    assert d2.diff_for_humans(d, locale=locale) == "כמה שניות קודם"

    assert d.diff_for_humans(d2, True, locale=locale) == "כמה שניות"
    assert d2.diff_for_humans(d.add(seconds=1), True, locale=locale) == "כמה שניות"


# === tests/localization/test_bg.py ===
from __future__ import annotations

import pendulum


locale = "bg"


def test_diff_for_humans():
    with pendulum.travel_to(pendulum.datetime(2016, 8, 29), freeze=True):
        diff_for_humans()


def diff_for_humans():
    d = pendulum.now().subtract(seconds=1)
    assert d.diff_for_humans(locale=locale) == "преди 1 секунда"

    d = pendulum.now().subtract(seconds=2)
    assert d.diff_for_humans(locale=locale) == "преди 2 секунди"

    d = pendulum.now().subtract(seconds=5)
    assert d.diff_for_humans(locale=locale) == "преди 5 секунди"

    d = pendulum.now().subtract(seconds=21)
    assert d.diff_for_humans(locale=locale) == "преди 21 секунди"

    d = pendulum.now().subtract(minutes=1)
    assert d.diff_for_humans(locale=locale) == "преди 1 минута"

    d = pendulum.now().subtract(minutes=2)
    assert d.diff_for_humans(locale=locale) == "преди 2 минути"

    d = pendulum.now().subtract(minutes=5)
    assert d.diff_for_humans(locale=locale) == "преди 5 минути"

    d = pendulum.now().subtract(hours=1)
    assert d.diff_for_humans(locale=locale) == "преди 1 час"

    d = pendulum.now().subtract(hours=2)
    assert d.diff_for_humans(locale=locale) == "преди 2 часа"

    d = pendulum.now().subtract(hours=5)
    assert d.diff_for_humans(locale=locale) == "преди 5 часа"

    d = pendulum.now().subtract(days=1)
    assert d.diff_for_humans(locale=locale) == "преди 1 ден"

    d = pendulum.now().subtract(days=2)
    assert d.diff_for_humans(locale=locale) == "преди 2 дни"

    d = pendulum.now().subtract(days=5)
    assert d.diff_for_humans(locale=locale) == "преди 5 дни"

    d = pendulum.now().subtract(weeks=1)
    assert d.diff_for_humans(locale=locale) == "преди 1 седмица"

    d = pendulum.now().subtract(weeks=2)
    assert d.diff_for_humans(locale=locale) == "преди 2 седмици"

    d = pendulum.now().subtract(months=1)
    assert d.diff_for_humans(locale=locale) == "преди 1 месец"

    d = pendulum.now().subtract(months=2)
    assert d.diff_for_humans(locale=locale) == "преди 2 месеца"

    d = pendulum.now().subtract(months=5)
    assert d.diff_for_humans(locale=locale) == "преди 5 месеца"

    d = pendulum.now().subtract(years=1)
    assert d.diff_for_humans(locale=locale) == "преди 1 година"

    d = pendulum.now().subtract(years=2)
    assert d.diff_for_humans(locale=locale) == "преди 2 години"

    d = pendulum.now().subtract(years=5)
    assert d.diff_for_humans(locale=locale) == "преди 5 години"

    d = pendulum.now().add(seconds=1)
    assert d.diff_for_humans(locale=locale) == "след 1 секунда"

    d = pendulum.now().add(seconds=1)
    d2 = pendulum.now()
    assert d.diff_for_humans(d2, locale=locale) == "след 1 секунда"
    assert d2.diff_for_humans(d, locale=locale) == "преди 1 секунда"

    assert d.diff_for_humans(d2, True, locale=locale) == "1 секунда"
    assert d2.diff_for_humans(d.add(seconds=1), True, locale=locale) == "2 секунди"


# === tests/localization/test_ru.py ===
from __future__ import annotations

import pendulum


locale = "ru"


def test_diff_for_humans():
    with pendulum.travel_to(pendulum.datetime(2016, 8, 29), freeze=True):
        diff_for_humans()


def diff_for_humans():
    d = pendulum.now().subtract(seconds=1)
    assert d.diff_for_humans(locale=locale) == "1 секунду назад"

    d = pendulum.now().subtract(seconds=2)
    assert d.diff_for_humans(locale=locale) == "2 секунды назад"

    d = pendulum.now().subtract(seconds=5)
    assert d.diff_for_humans(locale=locale) == "5 секунд назад"

    d = pendulum.now().subtract(seconds=21)
    assert d.diff_for_humans(locale=locale) == "21 секунду назад"

    d = pendulum.now().subtract(minutes=1)
    assert d.diff_for_humans(locale=locale) == "1 минуту назад"

    d = pendulum.now().subtract(minutes=2)
    assert d.diff_for_humans(locale=locale) == "2 минуты назад"

    d = pendulum.now().subtract(minutes=5)
    assert d.diff_for_humans(locale=locale) == "5 минут назад"

    d = pendulum.now().subtract(hours=1)
    assert d.diff_for_humans(locale=locale) == "1 час назад"

    d = pendulum.now().subtract(hours=2)
    assert d.diff_for_humans(locale=locale) == "2 часа назад"

    d = pendulum.now().subtract(hours=5)
    assert d.diff_for_humans(locale=locale) == "5 часов назад"

    d = pendulum.now().subtract(days=1)
    assert d.diff_for_humans(locale=locale) == "1 день назад"

    d = pendulum.now().subtract(days=2)
    assert d.diff_for_humans(locale=locale) == "2 дня назад"

    d = pendulum.now().subtract(days=5)
    assert d.diff_for_humans(locale=locale) == "5 дней назад"

    d = pendulum.now().subtract(weeks=1)
    assert d.diff_for_humans(locale=locale) == "1 неделю назад"

    d = pendulum.now().subtract(weeks=2)
    assert d.diff_for_humans(locale=locale) == "2 недели назад"

    d = pendulum.now().subtract(months=1)
    assert d.diff_for_humans(locale=locale) == "1 месяц назад"

    d = pendulum.now().subtract(months=2)
    assert d.diff_for_humans(locale=locale) == "2 месяца назад"

    d = pendulum.now().subtract(months=5)
    assert d.diff_for_humans(locale=locale) == "5 месяцев назад"

    d = pendulum.now().subtract(years=1)
    assert d.diff_for_humans(locale=locale) == "1 год назад"

    d = pendulum.now().subtract(years=2)
    assert d.diff_for_humans(locale=locale) == "2 года назад"

    d = pendulum.now().subtract(years=5)
    assert d.diff_for_humans(locale=locale) == "5 лет назад"

    d = pendulum.now().add(seconds=1)
    assert d.diff_for_humans(locale=locale) == "через 1 секунду"

    d = pendulum.now().add(seconds=1)
    d2 = pendulum.now()
    assert d.diff_for_humans(d2, locale=locale) == "1 секунда после"
    assert d2.diff_for_humans(d, locale=locale) == "1 секунда до"

    assert d.diff_for_humans(d2, True, locale=locale) == "1 секунда"
    assert d2.diff_for_humans(d.add(seconds=1), True, locale=locale) == "2 секунды"


# === tests/localization/__init__.py ===


# === tests/localization/test_ko.py ===
from __future__ import annotations

import pendulum


locale = "ko"


def test_diff_for_humans():
    with pendulum.travel_to(pendulum.datetime(2016, 8, 29), freeze=True):
        diff_for_humans()


def diff_for_humans():
    d = pendulum.now().subtract(seconds=1)
    assert d.diff_for_humans(locale=locale) == "1초 전"

    d = pendulum.now().subtract(seconds=2)
    assert d.diff_for_humans(locale=locale) == "2초 전"

    d = pendulum.now().subtract(minutes=1)
    assert d.diff_for_humans(locale=locale) == "1분 전"

    d = pendulum.now().subtract(minutes=2)
    assert d.diff_for_humans(locale=locale) == "2분 전"

    d = pendulum.now().subtract(hours=1)
    assert d.diff_for_humans(locale=locale) == "1시간 전"

    d = pendulum.now().subtract(hours=2)
    assert d.diff_for_humans(locale=locale) == "2시간 전"

    d = pendulum.now().subtract(days=1)
    assert d.diff_for_humans(locale=locale) == "1일 전"

    d = pendulum.now().subtract(days=2)
    assert d.diff_for_humans(locale=locale) == "2일 전"

    d = pendulum.now().subtract(weeks=1)
    assert d.diff_for_humans(locale=locale) == "1주 전"

    d = pendulum.now().subtract(weeks=2)
    assert d.diff_for_humans(locale=locale) == "2주 전"

    d = pendulum.now().subtract(months=1)
    assert d.diff_for_humans(locale=locale) == "1개월 전"

    d = pendulum.now().subtract(months=2)
    assert d.diff_for_humans(locale=locale) == "2개월 전"

    d = pendulum.now().subtract(years=1)
    assert d.diff_for_humans(locale=locale) == "1년 전"

    d = pendulum.now().subtract(years=2)
    assert d.diff_for_humans(locale=locale) == "2년 전"

    d = pendulum.now().add(seconds=1)
    assert d.diff_for_humans(locale=locale) == "1초 후"

    d = pendulum.now().add(seconds=1)
    d2 = pendulum.now()
    assert d.diff_for_humans(d2, locale=locale) == "1초 후"
    assert d2.diff_for_humans(d, locale=locale) == "1초 전"

    assert d.diff_for_humans(d2, True, locale=locale) == "1초"
    assert d2.diff_for_humans(d.add(seconds=1), True, locale=locale) == "2초"


# === tests/localization/test_nn.py ===
from __future__ import annotations

import pendulum


locale = "nn"


def test_diff_for_humans():
    with pendulum.travel_to(pendulum.datetime(2016, 8, 29), freeze=True):
        diff_for_humans()


def diff_for_humans():
    d = pendulum.now().subtract(seconds=1)
    assert d.diff_for_humans(locale=locale) == "for 1 sekund sidan"

    d = pendulum.now().subtract(seconds=2)
    assert d.diff_for_humans(locale=locale) == "for 2 sekund sidan"

    d = pendulum.now().subtract(minutes=1)
    assert d.diff_for_humans(locale=locale) == "for 1 minutt sidan"

    d = pendulum.now().subtract(minutes=2)
    assert d.diff_for_humans(locale=locale) == "for 2 minutt sidan"

    d = pendulum.now().subtract(hours=1)
    assert d.diff_for_humans(locale=locale) == "for 1 time sidan"

    d = pendulum.now().subtract(hours=2)
    assert d.diff_for_humans(locale=locale) == "for 2 timar sidan"

    d = pendulum.now().subtract(days=1)
    assert d.diff_for_humans(locale=locale) == "for 1 dag sidan"

    d = pendulum.now().subtract(days=2)
    assert d.diff_for_humans(locale=locale) == "for 2 dagar sidan"

    d = pendulum.now().subtract(weeks=1)
    assert d.diff_for_humans(locale=locale) == "for 1 veke sidan"

    d = pendulum.now().subtract(weeks=2)
    assert d.diff_for_humans(locale=locale) == "for 2 veker sidan"

    d = pendulum.now().subtract(months=1)
    assert d.diff_for_humans(locale=locale) == "for 1 månad sidan"

    d = pendulum.now().subtract(months=2)
    assert d.diff_for_humans(locale=locale) == "for 2 månadar sidan"

    d = pendulum.now().subtract(years=1)
    assert d.diff_for_humans(locale=locale) == "for 1 år sidan"

    d = pendulum.now().subtract(years=2)
    assert d.diff_for_humans(locale=locale) == "for 2 år sidan"

    d = pendulum.now().add(seconds=1)
    assert d.diff_for_humans(locale=locale) == "om 1 sekund"

    d = pendulum.now().add(seconds=1)
    d2 = pendulum.now()
    assert d.diff_for_humans(d2, locale=locale) == "1 sekund etter"
    assert d2.diff_for_humans(d, locale=locale) == "1 sekund før"

    assert d.diff_for_humans(d2, True, locale=locale) == "1 sekund"
    assert d2.diff_for_humans(d.add(seconds=1), True, locale=locale) == "2 sekund"


def test_format():
    d = pendulum.datetime(2016, 8, 29, 7, 3, 6, 123456)
    assert d.format("dddd", locale=locale) == "måndag"
    assert d.format("ddd", locale=locale) == "mån."
    assert d.format("MMMM", locale=locale) == "august"
    assert d.format("MMM", locale=locale) == "aug."
    assert d.format("A", locale=locale) == "formiddag"
    assert d.format("Qo", locale=locale) == "3."
    assert d.format("Mo", locale=locale) == "8."
    assert d.format("Do", locale=locale) == "29."

    assert d.format("LT", locale=locale) == "07:03"
    assert d.format("LTS", locale=locale) == "07:03:06"
    assert d.format("L", locale=locale) == "29.08.2016"
    assert d.format("LL", locale=locale) == "29. august 2016"
    assert d.format("LLL", locale=locale) == "29. august 2016 07:03"
    assert d.format("LLLL", locale=locale) == "måndag 29. august 2016 07:03"


# === tests/localization/test_pl.py ===
from __future__ import annotations

import pendulum


locale = "pl"


def test_diff_for_humans():
    with pendulum.travel_to(pendulum.datetime(2016, 8, 29), freeze=True):
        diff_for_humans()


def diff_for_humans():
    d = pendulum.now().subtract(seconds=1)
    assert d.diff_for_humans(locale=locale) == "kilka sekund temu"

    d = pendulum.now().subtract(seconds=2)
    assert d.diff_for_humans(locale=locale) == "kilka sekund temu"

    d = pendulum.now().subtract(seconds=20)
    assert d.diff_for_humans(locale=locale) == "20 sekund temu"

    d = pendulum.now().subtract(minutes=1)
    assert d.diff_for_humans(locale=locale) == "1 minutę temu"

    d = pendulum.now().subtract(minutes=2)
    assert d.diff_for_humans(locale=locale) == "2 minuty temu"

    d = pendulum.now().subtract(minutes=5)
    assert d.diff_for_humans(locale=locale) == "5 minut temu"

    d = pendulum.now().subtract(hours=1)
    assert d.diff_for_humans(locale=locale) == "1 godzinę temu"

    d = pendulum.now().subtract(hours=2)
    assert d.diff_for_humans(locale=locale) == "2 godziny temu"

    d = pendulum.now().subtract(hours=5)
    assert d.diff_for_humans(locale=locale) == "5 godzin temu"

    d = pendulum.now().subtract(days=1)
    assert d.diff_for_humans(locale=locale) == "1 dzień temu"

    d = pendulum.now().subtract(days=2)
    assert d.diff_for_humans(locale=locale) == "2 dni temu"

    d = pendulum.now().subtract(weeks=1)
    assert d.diff_for_humans(locale=locale) == "1 tydzień temu"

    d = pendulum.now().subtract(weeks=2)
    assert d.diff_for_humans(locale=locale) == "2 tygodnie temu"

    d = pendulum.now().subtract(months=1)
    assert d.diff_for_humans(locale=locale) == "1 miesiąc temu"

    d = pendulum.now().subtract(months=2)
    assert d.diff_for_humans(locale=locale) == "2 miesiące temu"

    d = pendulum.now().subtract(months=5)
    assert d.diff_for_humans(locale=locale) == "5 miesięcy temu"

    d = pendulum.now().subtract(years=1)
    assert d.diff_for_humans(locale=locale) == "1 rok temu"

    d = pendulum.now().subtract(years=2)
    assert d.diff_for_humans(locale=locale) == "2 lata temu"

    d = pendulum.now().subtract(years=5)
    assert d.diff_for_humans(locale=locale) == "5 lat temu"

    d = pendulum.now().add(seconds=1)
    assert d.diff_for_humans(locale=locale) == "za kilka sekund"

    d = pendulum.now().add(seconds=1)
    d2 = pendulum.now()
    assert d.diff_for_humans(d2, locale=locale) == "kilka sekund po"
    assert d2.diff_for_humans(d, locale=locale) == "kilka sekund przed"

    assert d.diff_for_humans(d2, True, locale=locale) == "kilka sekund"
    assert d2.diff_for_humans(d.add(seconds=1), True, locale=locale) == "kilka sekund"

    d = pendulum.now().add(seconds=20)
    d2 = pendulum.now()
    assert d.diff_for_humans(d2, locale=locale) == "20 sekund po"
    assert d2.diff_for_humans(d, locale=locale) == "20 sekund przed"

    d = pendulum.now().add(seconds=10)
    d2 = pendulum.now()
    assert d.diff_for_humans(d2, True, locale=locale) == "kilka sekund"
    assert d2.diff_for_humans(d.add(seconds=1), True, locale=locale) == "11 sekund"


def test_format():
    d = pendulum.datetime(2016, 8, 29, 7, 3, 6, 123456)
    assert d.format("dddd", locale=locale) == "poniedziałek"
    assert d.format("ddd", locale=locale) == "pon."
    assert d.format("MMMM", locale=locale) == "sierpnia"
    assert d.format("MMM", locale=locale) == "sie"
    assert d.format("A", locale=locale) == "AM"
    assert d.format("Qo", locale=locale) == "3"
    assert d.format("Mo", locale=locale) == "8"
    assert d.format("Do", locale=locale) == "29"

    assert d.format("LT", locale=locale) == "07:03"
    assert d.format("LTS", locale=locale) == "07:03:06"
    assert d.format("L", locale=locale) == "29.08.2016"
    assert d.format("LL", locale=locale) == "29 sierpnia 2016"
    assert d.format("LLL", locale=locale) == "29 sierpnia 2016 07:03"
    assert d.format("LLLL", locale=locale) == "poniedziałek, 29 sierpnia 2016 07:03"


# === tests/localization/test_tr.py ===
from __future__ import annotations

import pendulum


locale = "tr"


def test_diff_for_humans():
    with pendulum.travel_to(pendulum.datetime(2016, 8, 29), freeze=True):
        diff_for_humans()


def diff_for_humans():
    d = pendulum.now().subtract(seconds=1)
    assert d.diff_for_humans(locale=locale) == "1 saniye önce"

    d = pendulum.now().subtract(seconds=2)
    assert d.diff_for_humans(locale=locale) == "2 saniye önce"

    d = pendulum.now().subtract(minutes=1)
    assert d.diff_for_humans(locale=locale) == "1 dakika önce"

    d = pendulum.now().subtract(minutes=2)
    assert d.diff_for_humans(locale=locale) == "2 dakika önce"

    d = pendulum.now().subtract(hours=1)
    assert d.diff_for_humans(locale=locale) == "1 saat önce"

    d = pendulum.now().subtract(hours=2)
    assert d.diff_for_humans(locale=locale) == "2 saat önce"

    d = pendulum.now().subtract(days=1)
    assert d.diff_for_humans(locale=locale) == "1 gün önce"

    d = pendulum.now().subtract(days=2)
    assert d.diff_for_humans(locale=locale) == "2 gün önce"

    d = pendulum.now().subtract(weeks=1)
    assert d.diff_for_humans(locale=locale) == "1 hafta önce"

    d = pendulum.now().subtract(weeks=2)
    assert d.diff_for_humans(locale=locale) == "2 hafta önce"

    d = pendulum.now().subtract(months=1)
    assert d.diff_for_humans(locale=locale) == "1 ay önce"

    d = pendulum.now().subtract(months=2)
    assert d.diff_for_humans(locale=locale) == "2 ay önce"

    d = pendulum.now().subtract(years=1)
    assert d.diff_for_humans(locale=locale) == "1 yıl önce"

    d = pendulum.now().subtract(years=2)
    assert d.diff_for_humans(locale=locale) == "2 yıl önce"

    d = pendulum.now().add(seconds=1)
    assert d.diff_for_humans(locale=locale) == "1 saniye sonra"

    d = pendulum.now().add(seconds=1)
    d2 = pendulum.now()
    assert d.diff_for_humans(d2, locale=locale) == "1 saniye sonra"
    assert d2.diff_for_humans(d, locale=locale) == "1 saniye önce"

    assert d.diff_for_humans(d2, True, locale=locale) == "1 saniye"
    assert d2.diff_for_humans(d.add(seconds=1), True, locale=locale) == "2 saniye"


# === tests/localization/test_fo.py ===
from __future__ import annotations

import pendulum


locale = "fo"


def test_diff_for_humans():
    with pendulum.travel_to(pendulum.datetime(2016, 8, 29), freeze=True):
        diff_for_humans()


def diff_for_humans():
    d = pendulum.now().subtract(seconds=1)
    assert d.diff_for_humans(locale=locale) == "1 sekund síðan"

    d = pendulum.now().subtract(seconds=2)
    assert d.diff_for_humans(locale=locale) == "2 sekund síðan"

    d = pendulum.now().subtract(minutes=1)
    assert d.diff_for_humans(locale=locale) == "1 minutt síðan"

    d = pendulum.now().subtract(minutes=2)
    assert d.diff_for_humans(locale=locale) == "2 minuttir síðan"

    d = pendulum.now().subtract(hours=1)
    assert d.diff_for_humans(locale=locale) == "1 tími síðan"

    d = pendulum.now().subtract(hours=2)
    assert d.diff_for_humans(locale=locale) == "2 tímar síðan"

    d = pendulum.now().subtract(days=1)
    assert d.diff_for_humans(locale=locale) == "1 dagur síðan"

    d = pendulum.now().subtract(days=2)
    assert d.diff_for_humans(locale=locale) == "2 dagar síðan"

    d = pendulum.now().subtract(weeks=1)
    assert d.diff_for_humans(locale=locale) == "1 vika síðan"

    d = pendulum.now().subtract(weeks=2)
    assert d.diff_for_humans(locale=locale) == "2 vikur síðan"

    d = pendulum.now().subtract(months=1)
    assert d.diff_for_humans(locale=locale) == "1 mánað síðan"

    d = pendulum.now().subtract(months=2)
    assert d.diff_for_humans(locale=locale) == "2 mánaðir síðan"

    d = pendulum.now().subtract(years=1)
    assert d.diff_for_humans(locale=locale) == "1 ár síðan"

    d = pendulum.now().subtract(years=2)
    assert d.diff_for_humans(locale=locale) == "2 ár síðan"

    d = pendulum.now().add(seconds=1)
    assert d.diff_for_humans(locale=locale) == "um 1 sekund"

    d = pendulum.now().add(seconds=1)
    d2 = pendulum.now()
    assert d.diff_for_humans(d2, locale=locale) == "1 sekund aftaná"
    assert d2.diff_for_humans(d, locale=locale) == "1 sekund áðrenn"

    assert d.diff_for_humans(d2, True, locale=locale) == "1 sekund"
    assert d2.diff_for_humans(d.add(seconds=1), True, locale=locale) == "2 sekundir"


# === tests/localization/test_es.py ===
from __future__ import annotations

import pendulum


locale = "es"


def test_diff_for_humans():
    with pendulum.travel_to(pendulum.datetime(2016, 8, 29), freeze=True):
        diff_for_humans()


def diff_for_humans():
    d = pendulum.now().subtract(seconds=1)
    assert d.diff_for_humans(locale=locale) == "hace unos segundos"

    d = pendulum.now().subtract(seconds=2)
    assert d.diff_for_humans(locale=locale) == "hace unos segundos"

    d = pendulum.now().subtract(minutes=1)
    assert d.diff_for_humans(locale=locale) == "hace 1 minuto"

    d = pendulum.now().subtract(minutes=2)
    assert d.diff_for_humans(locale=locale) == "hace 2 minutos"

    d = pendulum.now().subtract(hours=1)
    assert d.diff_for_humans(locale=locale) == "hace 1 hora"

    d = pendulum.now().subtract(hours=2)
    assert d.diff_for_humans(locale=locale) == "hace 2 horas"

    d = pendulum.now().subtract(days=1)
    assert d.diff_for_humans(locale=locale) == "hace 1 día"

    d = pendulum.now().subtract(days=2)
    assert d.diff_for_humans(locale=locale) == "hace 2 días"

    d = pendulum.now().subtract(weeks=1)
    assert d.diff_for_humans(locale=locale) == "hace 1 semana"

    d = pendulum.now().subtract(weeks=2)
    assert d.diff_for_humans(locale=locale) == "hace 2 semanas"

    d = pendulum.now().subtract(months=1)
    assert d.diff_for_humans(locale=locale) == "hace 1 mes"

    d = pendulum.now().subtract(months=2)
    assert d.diff_for_humans(locale=locale) == "hace 2 meses"

    d = pendulum.now().subtract(years=1)
    assert d.diff_for_humans(locale=locale) == "hace 1 año"

    d = pendulum.now().subtract(years=2)
    assert d.diff_for_humans(locale=locale) == "hace 2 años"

    d = pendulum.now().add(seconds=1)
    assert d.diff_for_humans(locale=locale) == "dentro de unos segundos"

    d = pendulum.now().add(seconds=1)
    d2 = pendulum.now()
    assert d.diff_for_humans(d2, locale=locale) == "unos segundos después"
    assert d2.diff_for_humans(d, locale=locale) == "unos segundos antes"

    assert d.diff_for_humans(d2, True, locale=locale) == "unos segundos"
    assert d2.diff_for_humans(d.add(seconds=1), True, locale=locale) == "unos segundos"


# === tests/localization/test_it.py ===
from __future__ import annotations

import pendulum


locale = "it"


def test_diff_for_humans():
    with pendulum.travel_to(pendulum.datetime(2016, 8, 29), freeze=True):
        diff_for_humans()


def diff_for_humans():
    d = pendulum.now().subtract(seconds=1)
    assert d.diff_for_humans(locale=locale) == "alcuni secondi fa"

    d = pendulum.now().subtract(seconds=2)
    assert d.diff_for_humans(locale=locale) == "alcuni secondi fa"

    d = pendulum.now().subtract(minutes=1)
    assert d.diff_for_humans(locale=locale) == "1 minuto fa"

    d = pendulum.now().subtract(minutes=2)
    assert d.diff_for_humans(locale=locale) == "2 minuti fa"

    d = pendulum.now().subtract(hours=1)
    assert d.diff_for_humans(locale=locale) == "1 ora fa"

    d = pendulum.now().subtract(hours=2)
    assert d.diff_for_humans(locale=locale) == "2 ore fa"

    d = pendulum.now().subtract(days=1)
    assert d.diff_for_humans(locale=locale) == "1 giorno fa"

    d = pendulum.now().subtract(days=2)
    assert d.diff_for_humans(locale=locale) == "2 giorni fa"

    d = pendulum.now().subtract(weeks=1)
    assert d.diff_for_humans(locale=locale) == "1 settimana fa"

    d = pendulum.now().subtract(weeks=2)
    assert d.diff_for_humans(locale=locale) == "2 settimane fa"

    d = pendulum.now().subtract(months=1)
    assert d.diff_for_humans(locale=locale) == "1 mese fa"

    d = pendulum.now().subtract(months=2)
    assert d.diff_for_humans(locale=locale) == "2 mesi fa"

    d = pendulum.now().subtract(years=1)
    assert d.diff_for_humans(locale=locale) == "1 anno fa"

    d = pendulum.now().subtract(years=2)
    assert d.diff_for_humans(locale=locale) == "2 anni fa"

    d = pendulum.now().add(seconds=1)
    assert d.diff_for_humans(locale=locale) == "in alcuni secondi"

    d = pendulum.now().add(seconds=1)
    d2 = pendulum.now()
    assert d.diff_for_humans(d2, locale=locale) == "alcuni secondi dopo"
    assert d2.diff_for_humans(d, locale=locale) == "alcuni secondi prima"

    assert d.diff_for_humans(d2, True, locale=locale) == "alcuni secondi"
    assert d2.diff_for_humans(d.add(seconds=1), True, locale=locale) == "alcuni secondi"


def test_format():
    d = pendulum.datetime(2016, 8, 28, 7, 3, 6, 123456)
    assert d.format("dddd", locale=locale) == "domenica"
    assert d.format("ddd", locale=locale) == "dom"
    assert d.format("MMMM", locale=locale) == "agosto"
    assert d.format("MMM", locale=locale) == "ago"
    assert d.format("A", locale=locale) == "AM"

    assert d.format("LT", locale=locale) == "7:03"
    assert d.format("LTS", locale=locale) == "7:03:06"
    assert d.format("L", locale=locale) == "28/08/2016"
    assert d.format("LL", locale=locale) == "28 agosto 2016"
    assert d.format("LLL", locale=locale) == "28 agosto 2016 alle 7:03"
    assert d.format("LLLL", locale=locale) == "domenica, 28 agosto 2016 alle 7:03"

    assert d.format("Do", locale=locale) == "28°"
    d = pendulum.datetime(2019, 1, 1, 7, 3, 6, 123456)
    assert d.format("Do", locale=locale) == "1°"


# === tests/localization/test_sk.py ===
from __future__ import annotations

import pendulum


locale = "sk"


def test_diff_for_humans():
    with pendulum.travel_to(pendulum.datetime(2016, 8, 29), freeze=True):
        diff_for_humans()


def diff_for_humans():
    d = pendulum.now().subtract(seconds=1)
    assert d.diff_for_humans(locale=locale) == "pred 1 sekundou"

    d = pendulum.now().add(seconds=1)
    assert d.diff_for_humans(locale=locale) == "o 1 sekundu"

    d = pendulum.now().add(seconds=2)
    assert d.diff_for_humans(locale=locale) == "o 2 sekundy"

    d = pendulum.now().add(seconds=5)
    assert d.diff_for_humans(locale=locale) == "o 5 sekúnd"

    d = pendulum.now().subtract(seconds=20)
    assert d.diff_for_humans(locale=locale) == "pred 20 sekundami"

    d = pendulum.now().subtract(minutes=1)
    assert d.diff_for_humans(locale=locale) == "pred 1 minútou"

    d = pendulum.now().subtract(minutes=2)
    assert d.diff_for_humans(locale=locale) == "pred 2 minútami"

    d = pendulum.now().add(minutes=5)
    assert d.diff_for_humans(locale=locale) == "o 5 minút"

    d = pendulum.now().subtract(hours=1)
    assert d.diff_for_humans(locale=locale) == "pred 1 hodinou"

    d = pendulum.now().subtract(hours=2)
    assert d.diff_for_humans(locale=locale) == "pred 2 hodinami"

    d = pendulum.now().subtract(hours=5)
    assert d.diff_for_humans(locale=locale) == "pred 5 hodinami"

    d = pendulum.now().subtract(days=1)
    assert d.diff_for_humans(locale=locale) == "pred 1 dňom"

    d = pendulum.now().subtract(days=2)
    assert d.diff_for_humans(locale=locale) == "pred 2 dňami"

    d = pendulum.now().subtract(weeks=1)
    assert d.diff_for_humans(locale=locale) == "pred 1 týždňom"

    d = pendulum.now().subtract(weeks=2)
    assert d.diff_for_humans(locale=locale) == "pred 2 týždňami"

    d = pendulum.now().subtract(months=1)
    assert d.diff_for_humans(locale=locale) == "pred 1 mesiacom"

    d = pendulum.now().subtract(months=2)
    assert d.diff_for_humans(locale=locale) == "pred 2 mesiacmi"

    d = pendulum.now().subtract(months=5)
    assert d.diff_for_humans(locale=locale) == "pred 5 mesiacmi"

    d = pendulum.now().subtract(years=1)
    assert d.diff_for_humans(locale=locale) == "pred 1 rokom"

    d = pendulum.now().subtract(years=2)
    assert d.diff_for_humans(locale=locale) == "pred 2 rokmi"

    d = pendulum.now().subtract(years=5)
    assert d.diff_for_humans(locale=locale) == "pred 5 rokmi"

    d = pendulum.now().add(seconds=1)
    d2 = pendulum.now()
    assert d.diff_for_humans(d2, locale=locale) == "1 sekunda po"
    assert d2.diff_for_humans(d, locale=locale) == "1 sekunda pred"

    assert d.diff_for_humans(d2, True, locale=locale) == "1 sekunda"
    assert d2.diff_for_humans(d.add(seconds=1), True, locale=locale) == "2 sekundy"

    d = pendulum.now().add(seconds=20)
    d2 = pendulum.now()
    assert d.diff_for_humans(d2, locale=locale) == "20 sekúnd po"
    assert d2.diff_for_humans(d, locale=locale) == "20 sekúnd pred"

    d = pendulum.now().add(seconds=10)
    d2 = pendulum.now()
    assert d.diff_for_humans(d2, True, locale=locale) == "10 sekúnd"
    assert d2.diff_for_humans(d.add(seconds=1), True, locale=locale) == "11 sekúnd"


def test_format():
    d = pendulum.datetime(2016, 8, 29, 7, 3, 6, 123456)
    assert d.format("dddd", locale=locale) == "pondelok"
    assert d.format("ddd", locale=locale) == "po"
    assert d.format("MMMM", locale=locale) == "augusta"
    assert d.format("MMM", locale=locale) == "aug"
    assert d.format("A", locale=locale) == "AM"
    assert d.format("Qo", locale=locale) == "3"
    assert d.format("Mo", locale=locale) == "8"
    assert d.format("Do", locale=locale) == "29"

    assert d.format("LT", locale=locale) == "07:03"
    assert d.format("LTS", locale=locale) == "07:03:06"
    assert d.format("L", locale=locale) == "29.08.2016"
    assert d.format("LL", locale=locale) == "29. augusta 2016"
    assert d.format("LLL", locale=locale) == "29. augusta 2016 07:03"
    assert d.format("LLLL", locale=locale) == "pondelok, 29. augusta 2016 07:03"


# === tests/localization/test_lt.py ===
from __future__ import annotations

import pendulum


locale = "lt"


def test_diff_for_humans():
    with pendulum.travel_to(pendulum.datetime(2016, 8, 29), freeze=True):
        diff_for_humans()


def diff_for_humans():
    d = pendulum.now().subtract(seconds=1)
    assert d.diff_for_humans(locale=locale) == "prieš 1 sekundę"

    d = pendulum.now().subtract(seconds=2)
    assert d.diff_for_humans(locale=locale) == "prieš 2 sekundes"

    d = pendulum.now().subtract(seconds=21)
    assert d.diff_for_humans(locale=locale) == "prieš 21 sekundę"

    d = pendulum.now().subtract(minutes=1)
    assert d.diff_for_humans(locale=locale) == "prieš 1 minutę"

    d = pendulum.now().subtract(minutes=2)
    assert d.diff_for_humans(locale=locale) == "prieš 2 minutes"

    d = pendulum.now().subtract(hours=1)
    assert d.diff_for_humans(locale=locale) == "prieš 1 valandą"

    d = pendulum.now().subtract(hours=2)
    assert d.diff_for_humans(locale=locale) == "prieš 2 valandas"

    d = pendulum.now().subtract(days=1)
    assert d.diff_for_humans(locale=locale) == "prieš 1 dieną"

    d = pendulum.now().subtract(days=2)
    assert d.diff_for_humans(locale=locale) == "prieš 2 dienas"

    d = pendulum.now().subtract(weeks=1)
    assert d.diff_for_humans(locale=locale) == "prieš 1 savaitę"

    d = pendulum.now().subtract(weeks=2)
    assert d.diff_for_humans(locale=locale) == "prieš 2 savaites"

    d = pendulum.now().subtract(months=1)
    assert d.diff_for_humans(locale=locale) == "prieš 1 mėnesį"

    d = pendulum.now().subtract(months=2)
    assert d.diff_for_humans(locale=locale) == "prieš 2 mėnesius"

    d = pendulum.now().subtract(years=1)
    assert d.diff_for_humans(locale=locale) == "prieš 1 metus"

    d = pendulum.now().subtract(years=2)
    assert d.diff_for_humans(locale=locale) == "prieš 2 metus"

    d = pendulum.now().add(seconds=1)
    assert d.diff_for_humans(locale=locale) == "po 1 sekundės"

    d = pendulum.now().add(seconds=1)
    d2 = pendulum.now()
    assert d.diff_for_humans(d2, locale=locale) == "po 1 sekundės"
    assert d2.diff_for_humans(d, locale=locale) == "1 sekundę nuo dabar"

    assert d.diff_for_humans(d2, True, locale=locale) == "1 sekundė"
    assert d2.diff_for_humans(d.add(seconds=1), True, locale=locale) == "2 sekundės"


# === tests/localization/test_sv.py ===
from __future__ import annotations

import pendulum


locale = "sv"


def test_diff_for_humans():
    with pendulum.travel_to(pendulum.datetime(2016, 8, 29), freeze=True):
        diff_for_humans()


def diff_for_humans():
    d = pendulum.now().subtract(seconds=1)
    assert d.diff_for_humans(locale=locale) == "för 1 sekund sedan"

    d = pendulum.now().subtract(seconds=2)
    assert d.diff_for_humans(locale=locale) == "för 2 sekunder sedan"

    d = pendulum.now().subtract(seconds=5)
    assert d.diff_for_humans(locale=locale) == "för 5 sekunder sedan"

    d = pendulum.now().subtract(seconds=21)
    assert d.diff_for_humans(locale=locale) == "för 21 sekunder sedan"

    d = pendulum.now().subtract(minutes=1)
    assert d.diff_for_humans(locale=locale) == "för 1 minut sedan"

    d = pendulum.now().subtract(minutes=2)
    assert d.diff_for_humans(locale=locale) == "för 2 minuter sedan"

    d = pendulum.now().subtract(minutes=5)
    assert d.diff_for_humans(locale=locale) == "för 5 minuter sedan"

    d = pendulum.now().subtract(hours=1)
    assert d.diff_for_humans(locale=locale) == "för 1 timme sedan"

    d = pendulum.now().subtract(hours=2)
    assert d.diff_for_humans(locale=locale) == "för 2 timmar sedan"

    d = pendulum.now().subtract(hours=5)
    assert d.diff_for_humans(locale=locale) == "för 5 timmar sedan"

    d = pendulum.now().subtract(days=1)
    assert d.diff_for_humans(locale=locale) == "för 1 dag sedan"

    d = pendulum.now().subtract(days=2)
    assert d.diff_for_humans(locale=locale) == "för 2 dagar sedan"

    d = pendulum.now().subtract(days=5)
    assert d.diff_for_humans(locale=locale) == "för 5 dagar sedan"

    d = pendulum.now().subtract(weeks=1)
    assert d.diff_for_humans(locale=locale) == "för 1 vecka sedan"

    d = pendulum.now().subtract(weeks=2)
    assert d.diff_for_humans(locale=locale) == "för 2 veckor sedan"

    d = pendulum.now().subtract(months=1)
    assert d.diff_for_humans(locale=locale) == "för 1 månad sedan"

    d = pendulum.now().subtract(months=2)
    assert d.diff_for_humans(locale=locale) == "för 2 månader sedan"

    d = pendulum.now().subtract(months=5)
    assert d.diff_for_humans(locale=locale) == "för 5 månader sedan"

    d = pendulum.now().subtract(years=1)
    assert d.diff_for_humans(locale=locale) == "för 1 år sedan"

    d = pendulum.now().subtract(years=2)
    assert d.diff_for_humans(locale=locale) == "för 2 år sedan"

    d = pendulum.now().subtract(years=5)
    assert d.diff_for_humans(locale=locale) == "för 5 år sedan"

    d = pendulum.now().add(seconds=1)
    assert d.diff_for_humans(locale=locale) == "om 1 sekund"

    d = pendulum.now().add(seconds=1)
    d2 = pendulum.now()
    assert d.diff_for_humans(d2, locale=locale) == "1 sekund efter"
    assert d2.diff_for_humans(d, locale=locale) == "1 sekund innan"

    assert d.diff_for_humans(d2, True, locale=locale) == "1 sekund"
    assert d2.diff_for_humans(d.add(seconds=1), True, locale=locale) == "2 sekunder"


# === tests/localization/test_nl.py ===
from __future__ import annotations

import pendulum


locale = "nl"


def test_diff_for_humans():
    with pendulum.travel_to(pendulum.datetime(2016, 8, 29), freeze=True):
        diff_for_humans()


def diff_for_humans():
    d = pendulum.now().subtract(seconds=1)
    assert d.diff_for_humans(locale=locale) == "enkele seconden geleden"

    d = pendulum.now().subtract(seconds=2)
    assert d.diff_for_humans(locale=locale) == "enkele seconden geleden"

    d = pendulum.now().subtract(seconds=22)
    assert d.diff_for_humans(locale=locale) == "22 seconden geleden"

    d = pendulum.now().subtract(minutes=1)
    assert d.diff_for_humans(locale=locale) == "1 minuut geleden"

    d = pendulum.now().subtract(minutes=2)
    assert d.diff_for_humans(locale=locale) == "2 minuten geleden"

    d = pendulum.now().subtract(hours=1)
    assert d.diff_for_humans(locale=locale) == "1 uur geleden"

    d = pendulum.now().subtract(hours=2)
    assert d.diff_for_humans(locale=locale) == "2 uur geleden"

    d = pendulum.now().subtract(days=1)
    assert d.diff_for_humans(locale=locale) == "1 dag geleden"

    d = pendulum.now().subtract(days=2)
    assert d.diff_for_humans(locale=locale) == "2 dagen geleden"

    d = pendulum.now().subtract(weeks=1)
    assert d.diff_for_humans(locale=locale) == "1 week geleden"

    d = pendulum.now().subtract(weeks=2)
    assert d.diff_for_humans(locale=locale) == "2 weken geleden"

    d = pendulum.now().subtract(months=1)
    assert d.diff_for_humans(locale=locale) == "1 maand geleden"

    d = pendulum.now().subtract(months=2)
    assert d.diff_for_humans(locale=locale) == "2 maanden geleden"

    d = pendulum.now().subtract(years=1)
    assert d.diff_for_humans(locale=locale) == "1 jaar geleden"

    d = pendulum.now().subtract(years=2)
    assert d.diff_for_humans(locale=locale) == "2 jaar geleden"

    d = pendulum.now().add(seconds=1)
    assert d.diff_for_humans(locale=locale) == "over enkele seconden"

    d = pendulum.now().add(weeks=1)
    assert d.diff_for_humans(locale=locale) == "over 1 week"

    d = pendulum.now().add(seconds=1)
    d2 = pendulum.now()
    assert d.diff_for_humans(d2, locale=locale) == "enkele seconden later"
    assert d2.diff_for_humans(d, locale=locale) == "enkele seconden eerder"

    assert d.diff_for_humans(d2, True, locale=locale) == "enkele seconden"
    assert (
        d2.diff_for_humans(d.add(seconds=1), True, locale=locale) == "enkele seconden"
    )


def test_format():
    d = pendulum.datetime(2016, 8, 28, 7, 3, 6, 123456)
    assert d.format("dddd", locale=locale) == "zondag"
    assert d.format("ddd", locale=locale) == "zo"
    assert d.format("MMMM", locale=locale) == "augustus"
    assert d.format("MMM", locale=locale) == "aug."
    assert d.format("A", locale=locale) == "a.m."
    assert d.format("Do", locale=locale) == "28e"


# === tests/localization/test_fr.py ===
from __future__ import annotations

import pendulum


locale = "fr"


def test_diff_for_humans():
    with pendulum.travel_to(pendulum.datetime(2016, 8, 29), freeze=True):
        diff_for_humans()


def diff_for_humans():
    d = pendulum.now().subtract(seconds=1)
    assert d.diff_for_humans(locale=locale) == "il y a quelques secondes"

    d = pendulum.now().subtract(seconds=2)
    assert d.diff_for_humans(locale=locale) == "il y a quelques secondes"

    d = pendulum.now().subtract(minutes=1)
    assert d.diff_for_humans(locale=locale) == "il y a 1 minute"

    d = pendulum.now().subtract(minutes=2)
    assert d.diff_for_humans(locale=locale) == "il y a 2 minutes"

    d = pendulum.now().subtract(hours=1)
    assert d.diff_for_humans(locale=locale) == "il y a 1 heure"

    d = pendulum.now().subtract(hours=2)
    assert d.diff_for_humans(locale=locale) == "il y a 2 heures"

    d = pendulum.now().subtract(days=1)
    assert d.diff_for_humans(locale=locale) == "il y a 1 jour"

    d = pendulum.now().subtract(days=2)
    assert d.diff_for_humans(locale=locale) == "il y a 2 jours"

    d = pendulum.now().subtract(weeks=1)
    assert d.diff_for_humans(locale=locale) == "il y a 1 semaine"

    d = pendulum.now().subtract(weeks=2)
    assert d.diff_for_humans(locale=locale) == "il y a 2 semaines"

    d = pendulum.now().subtract(months=1)
    assert d.diff_for_humans(locale=locale) == "il y a 1 mois"

    d = pendulum.now().subtract(months=2)
    assert d.diff_for_humans(locale=locale) == "il y a 2 mois"

    d = pendulum.now().subtract(years=1)
    assert d.diff_for_humans(locale=locale) == "il y a 1 an"

    d = pendulum.now().subtract(years=2)
    assert d.diff_for_humans(locale=locale) == "il y a 2 ans"

    d = pendulum.now().add(seconds=1)
    assert d.diff_for_humans(locale=locale) == "dans quelques secondes"

    d = pendulum.now().add(seconds=1)
    d2 = pendulum.now()
    assert d.diff_for_humans(d2, locale=locale) == "quelques secondes après"
    assert d2.diff_for_humans(d, locale=locale) == "quelques secondes avant"

    assert d.diff_for_humans(d2, True, locale=locale) == "quelques secondes"
    assert (
        d2.diff_for_humans(d.add(seconds=1), True, locale=locale) == "quelques secondes"
    )


def test_format():
    d = pendulum.datetime(2016, 8, 28, 7, 3, 6, 123456)
    assert d.format("dddd", locale=locale) == "dimanche"
    assert d.format("ddd", locale=locale) == "dim."
    assert d.format("MMMM", locale=locale) == "août"
    assert d.format("MMM", locale=locale) == "août"
    assert d.format("A", locale=locale) == "AM"
    assert d.format("Do", locale=locale) == "28e"

    assert d.format("LT", locale=locale) == "07:03"
    assert d.format("LTS", locale=locale) == "07:03:06"
    assert d.format("L", locale=locale) == "28/08/2016"
    assert d.format("LL", locale=locale) == "28 août 2016"
    assert d.format("LLL", locale=locale) == "28 août 2016 07:03"
    assert d.format("LLLL", locale=locale) == "dimanche 28 août 2016 07:03"


# === tests/localization/test_cs.py ===
from __future__ import annotations

import pendulum


locale = "cs"


def test_diff_for_humans():
    with pendulum.travel_to(pendulum.datetime(2016, 8, 29), freeze=True):
        diff_for_humans()


def diff_for_humans():
    d = pendulum.now().subtract(seconds=1)
    assert d.diff_for_humans(locale=locale) == "pár vteřin zpět"

    d = pendulum.now().subtract(seconds=2)
    assert d.diff_for_humans(locale=locale) == "pár vteřin zpět"

    d = pendulum.now().subtract(seconds=20)
    assert d.diff_for_humans(locale=locale) == "před 20 sekundami"

    d = pendulum.now().subtract(minutes=1)
    assert d.diff_for_humans(locale=locale) == "před 1 minutou"

    d = pendulum.now().subtract(minutes=2)
    assert d.diff_for_humans(locale=locale) == "před 2 minutami"

    d = pendulum.now().subtract(minutes=5)
    assert d.diff_for_humans(locale=locale) == "před 5 minutami"

    d = pendulum.now().subtract(hours=1)
    assert d.diff_for_humans(locale=locale) == "před 1 hodinou"

    d = pendulum.now().subtract(hours=2)
    assert d.diff_for_humans(locale=locale) == "před 2 hodinami"

    d = pendulum.now().subtract(hours=5)
    assert d.diff_for_humans(locale=locale) == "před 5 hodinami"

    d = pendulum.now().subtract(days=1)
    assert d.diff_for_humans(locale=locale) == "před 1 dnem"

    d = pendulum.now().subtract(days=2)
    assert d.diff_for_humans(locale=locale) == "před 2 dny"

    d = pendulum.now().subtract(weeks=1)
    assert d.diff_for_humans(locale=locale) == "před 1 týdnem"

    d = pendulum.now().subtract(weeks=2)
    assert d.diff_for_humans(locale=locale) == "před 2 týdny"

    d = pendulum.now().subtract(months=1)
    assert d.diff_for_humans(locale=locale) == "před 1 měsícem"

    d = pendulum.now().subtract(months=2)
    assert d.diff_for_humans(locale=locale) == "před 2 měsíci"

    d = pendulum.now().subtract(months=5)
    assert d.diff_for_humans(locale=locale) == "před 5 měsíci"

    d = pendulum.now().subtract(years=1)
    assert d.diff_for_humans(locale=locale) == "před 1 rokem"

    d = pendulum.now().subtract(years=2)
    assert d.diff_for_humans(locale=locale) == "před 2 lety"

    d = pendulum.now().subtract(years=5)
    assert d.diff_for_humans(locale=locale) == "před 5 lety"

    d = pendulum.now().add(seconds=1)
    assert d.diff_for_humans(locale=locale) == "za pár vteřin"

    d = pendulum.now().add(seconds=1)
    d2 = pendulum.now()
    assert d.diff_for_humans(d2, locale=locale) == "pár vteřin po"
    assert d2.diff_for_humans(d, locale=locale) == "pár vteřin zpět"

    assert d.diff_for_humans(d2, True, locale=locale) == "pár vteřin"
    assert d2.diff_for_humans(d.add(seconds=1), True, locale=locale) == "pár vteřin"

    d = pendulum.now().add(seconds=20)
    d2 = pendulum.now()
    assert d.diff_for_humans(d2, locale=locale) == "20 sekund po"
    assert d2.diff_for_humans(d, locale=locale) == "20 sekund zpět"

    d = pendulum.now().add(seconds=10)
    d2 = pendulum.now()
    assert d.diff_for_humans(d2, True, locale=locale) == "pár vteřin"
    assert d2.diff_for_humans(d.add(seconds=1), True, locale=locale) == "11 sekund"


def test_format():
    d = pendulum.datetime(2016, 8, 29, 7, 3, 6, 123456)
    assert d.format("dddd", locale=locale) == "pondělí"
    assert d.format("ddd", locale=locale) == "po"
    assert d.format("MMMM", locale=locale) == "srpna"
    assert d.format("MMM", locale=locale) == "srp"
    assert d.format("A", locale=locale) == "dop."
    assert d.format("Qo", locale=locale) == "3."
    assert d.format("Mo", locale=locale) == "8."
    assert d.format("Do", locale=locale) == "29."

    assert d.format("LT", locale=locale) == "7:03"
    assert d.format("LTS", locale=locale) == "7:03:06"
    assert d.format("L", locale=locale) == "29. 8. 2016"
    assert d.format("LL", locale=locale) == "29. srpna, 2016"
    assert d.format("LLL", locale=locale) == "29. srpna, 2016 7:03"
    assert d.format("LLLL", locale=locale) == "pondělí, 29. srpna, 2016 7:03"


# === tests/localization/test_fa.py ===
from __future__ import annotations

import pendulum


locale = "fa"


def test_diff_for_humans():
    with pendulum.travel_to(pendulum.datetime(2016, 8, 29), freeze=True):
        diff_for_humans()


def diff_for_humans():
    d = pendulum.now().subtract(seconds=1)
    assert d.diff_for_humans(locale=locale) == "1 ثانیه پیش"

    d = pendulum.now().subtract(seconds=2)
    assert d.diff_for_humans(locale=locale) == "2 ثانیه پیش"

    d = pendulum.now().subtract(minutes=1)
    assert d.diff_for_humans(locale=locale) == "1 دقیقه پیش"

    d = pendulum.now().subtract(minutes=2)
    assert d.diff_for_humans(locale=locale) == "2 دقیقه پیش"

    d = pendulum.now().subtract(hours=1)
    assert d.diff_for_humans(locale=locale) == "1 ساعت پیش"

    d = pendulum.now().subtract(hours=2)
    assert d.diff_for_humans(locale=locale) == "2 ساعت پیش"

    d = pendulum.now().subtract(days=1)
    assert d.diff_for_humans(locale=locale) == "1 روز پیش"

    d = pendulum.now().subtract(days=2)
    assert d.diff_for_humans(locale=locale) == "2 روز پیش"

    d = pendulum.now().subtract(weeks=1)
    assert d.diff_for_humans(locale=locale) == "1 هفته پیش"

    d = pendulum.now().subtract(weeks=2)
    assert d.diff_for_humans(locale=locale) == "2 هفته پیش"

    d = pendulum.now().subtract(months=1)
    assert d.diff_for_humans(locale=locale) == "1 ماه پیش"

    d = pendulum.now().subtract(months=2)
    assert d.diff_for_humans(locale=locale) == "2 ماه پیش"

    d = pendulum.now().subtract(years=1)
    assert d.diff_for_humans(locale=locale) == "1 سال پیش"

    d = pendulum.now().subtract(years=2)
    assert d.diff_for_humans(locale=locale) == "2 سال پیش"

    d = pendulum.now().add(seconds=1)
    assert d.diff_for_humans(locale=locale) == "1 ثانیه بعد"

    d = pendulum.now().add(seconds=1)
    d2 = pendulum.now()
    assert d.diff_for_humans(d2, locale=locale) == "1 ثانیه پس از"
    assert d2.diff_for_humans(d, locale=locale) == "1 ثانیه پیش از"

    assert d.diff_for_humans(d2, True, locale=locale) == "1 ثانیه"
    assert d2.diff_for_humans(d.add(seconds=1), True, locale=locale) == "2 ثانیه"


# === tests/localization/test_de.py ===
from __future__ import annotations

import pendulum


locale = "de"


def test_diff_for_humans():
    with pendulum.travel_to(pendulum.datetime(2016, 8, 29), freeze=True):
        diff_for_humans()


def diff_for_humans():
    d = pendulum.now().subtract(seconds=1)
    assert d.diff_for_humans(locale=locale) == "vor 1 Sekunde"

    d = pendulum.now().subtract(seconds=2)
    assert d.diff_for_humans(locale=locale) == "vor 2 Sekunden"

    d = pendulum.now().subtract(minutes=1)
    assert d.diff_for_humans(locale=locale) == "vor 1 Minute"

    d = pendulum.now().subtract(minutes=2)
    assert d.diff_for_humans(locale=locale) == "vor 2 Minuten"

    d = pendulum.now().subtract(hours=1)
    assert d.diff_for_humans(locale=locale) == "vor 1 Stunde"

    d = pendulum.now().subtract(hours=2)
    assert d.diff_for_humans(locale=locale) == "vor 2 Stunden"

    d = pendulum.now().subtract(days=1)
    assert d.diff_for_humans(locale=locale) == "vor 1 Tag"

    d = pendulum.now().subtract(days=2)
    assert d.diff_for_humans(locale=locale) == "vor 2 Tagen"

    d = pendulum.now().subtract(weeks=1)
    assert d.diff_for_humans(locale=locale) == "vor 1 Woche"

    d = pendulum.now().subtract(weeks=2)
    assert d.diff_for_humans(locale=locale) == "vor 2 Wochen"

    d = pendulum.now().subtract(months=1)
    assert d.diff_for_humans(locale=locale) == "vor 1 Monat"

    d = pendulum.now().subtract(months=2)
    assert d.diff_for_humans(locale=locale) == "vor 2 Monaten"

    d = pendulum.now().subtract(years=1)
    assert d.diff_for_humans(locale=locale) == "vor 1 Jahr"

    d = pendulum.now().subtract(years=2)
    assert d.diff_for_humans(locale=locale) == "vor 2 Jahren"

    d = pendulum.now().add(seconds=1)
    assert d.diff_for_humans(locale=locale) == "in 1 Sekunde"

    d = pendulum.now().add(seconds=1)
    d2 = pendulum.now()
    assert d.diff_for_humans(d2, locale=locale) == "1 Sekunde später"
    assert d2.diff_for_humans(d, locale=locale) == "1 Sekunde zuvor"

    assert d.diff_for_humans(d2, True, locale=locale) == "1 Sekunde"
    assert d2.diff_for_humans(d.add(seconds=1), True, locale=locale) == "2 Sekunden"


# === tests/localization/test_da.py ===
from __future__ import annotations

import pendulum


locale = "da"


def test_diff_for_humans():
    with pendulum.travel_to(pendulum.datetime(2016, 8, 29), freeze=True):
        diff_for_humans()


def diff_for_humans():
    d = pendulum.now().subtract(seconds=1)
    assert d.diff_for_humans(locale=locale) == "for 1 sekund siden"

    d = pendulum.now().subtract(seconds=2)
    assert d.diff_for_humans(locale=locale) == "for 2 sekunder siden"

    d = pendulum.now().subtract(minutes=1)
    assert d.diff_for_humans(locale=locale) == "for 1 minut siden"

    d = pendulum.now().subtract(minutes=2)
    assert d.diff_for_humans(locale=locale) == "for 2 minutter siden"

    d = pendulum.now().subtract(hours=1)
    assert d.diff_for_humans(locale=locale) == "for 1 time siden"

    d = pendulum.now().subtract(hours=2)
    assert d.diff_for_humans(locale=locale) == "for 2 timer siden"

    d = pendulum.now().subtract(days=1)
    assert d.diff_for_humans(locale=locale) == "for 1 dag siden"

    d = pendulum.now().subtract(days=2)
    assert d.diff_for_humans(locale=locale) == "for 2 dage siden"

    d = pendulum.now().subtract(weeks=1)
    assert d.diff_for_humans(locale=locale) == "for 1 uge siden"

    d = pendulum.now().subtract(weeks=2)
    assert d.diff_for_humans(locale=locale) == "for 2 uger siden"

    d = pendulum.now().subtract(months=1)
    assert d.diff_for_humans(locale=locale) == "for 1 måned siden"

    d = pendulum.now().subtract(months=2)
    assert d.diff_for_humans(locale=locale) == "for 2 måneder siden"

    d = pendulum.now().subtract(years=1)
    assert d.diff_for_humans(locale=locale) == "for 1 år siden"

    d = pendulum.now().subtract(years=2)
    assert d.diff_for_humans(locale=locale) == "for 2 år siden"

    d = pendulum.now().add(seconds=1)
    assert d.diff_for_humans(locale=locale) == "om 1 sekund"

    d = pendulum.now().add(seconds=1)
    d2 = pendulum.now()
    assert d.diff_for_humans(d2, locale=locale) == "1 sekund efter"
    assert d2.diff_for_humans(d, locale=locale) == "1 sekund før"

    assert d.diff_for_humans(d2, True, locale=locale) == "1 sekund"
    assert d2.diff_for_humans(d.add(seconds=1), True, locale=locale) == "2 sekunder"


# === tests/localization/test_nb.py ===
from __future__ import annotations

import pendulum


locale = "nb"


def test_diff_for_humans():
    with pendulum.travel_to(pendulum.datetime(2016, 8, 29), freeze=True):
        diff_for_humans()


def diff_for_humans():
    d = pendulum.now().subtract(seconds=1)
    assert d.diff_for_humans(locale=locale) == "for 1 sekund siden"

    d = pendulum.now().subtract(seconds=2)
    assert d.diff_for_humans(locale=locale) == "for 2 sekunder siden"

    d = pendulum.now().subtract(minutes=1)
    assert d.diff_for_humans(locale=locale) == "for 1 minutt siden"

    d = pendulum.now().subtract(minutes=2)
    assert d.diff_for_humans(locale=locale) == "for 2 minutter siden"

    d = pendulum.now().subtract(hours=1)
    assert d.diff_for_humans(locale=locale) == "for 1 time siden"

    d = pendulum.now().subtract(hours=2)
    assert d.diff_for_humans(locale=locale) == "for 2 timer siden"

    d = pendulum.now().subtract(days=1)
    assert d.diff_for_humans(locale=locale) == "for 1 dag siden"

    d = pendulum.now().subtract(days=2)
    assert d.diff_for_humans(locale=locale) == "for 2 dager siden"

    d = pendulum.now().subtract(weeks=1)
    assert d.diff_for_humans(locale=locale) == "for 1 uke siden"

    d = pendulum.now().subtract(weeks=2)
    assert d.diff_for_humans(locale=locale) == "for 2 uker siden"

    d = pendulum.now().subtract(months=1)
    assert d.diff_for_humans(locale=locale) == "for 1 måned siden"

    d = pendulum.now().subtract(months=2)
    assert d.diff_for_humans(locale=locale) == "for 2 måneder siden"

    d = pendulum.now().subtract(years=1)
    assert d.diff_for_humans(locale=locale) == "for 1 år siden"

    d = pendulum.now().subtract(years=2)
    assert d.diff_for_humans(locale=locale) == "for 2 år siden"

    d = pendulum.now().add(seconds=1)
    assert d.diff_for_humans(locale=locale) == "om 1 sekund"

    d = pendulum.now().add(seconds=1)
    d2 = pendulum.now()
    assert d.diff_for_humans(d2, locale=locale) == "1 sekund etter"
    assert d2.diff_for_humans(d, locale=locale) == "1 sekund før"

    assert d.diff_for_humans(d2, True, locale=locale) == "1 sekund"
    assert d2.diff_for_humans(d.add(seconds=1), True, locale=locale) == "2 sekunder"


def test_format():
    d = pendulum.datetime(2016, 8, 28, 7, 3, 6, 123456)
    assert d.format("dddd", locale=locale) == "søndag"
    assert d.format("ddd", locale=locale) == "søn."
    assert d.format("MMMM", locale=locale) == "august"
    assert d.format("MMM", locale=locale) == "aug."
    assert d.format("A", locale=locale) == "a.m."
    assert d.format("Qo", locale=locale) == "3."
    assert d.format("Mo", locale=locale) == "8."
    assert d.format("Do", locale=locale) == "28."

    assert d.format("LT", locale=locale) == "07:03"
    assert d.format("LTS", locale=locale) == "07:03:06"
    assert d.format("L", locale=locale) == "28.08.2016"
    assert d.format("LL", locale=locale) == "28. august 2016"
    assert d.format("LLL", locale=locale) == "28. august 2016 07:03"
    assert d.format("LLLL", locale=locale) == "søndag 28. august 2016 07:03"


# === src/pendulum/time.py ===
from __future__ import annotations

import datetime

from datetime import time
from datetime import timedelta
from typing import TYPE_CHECKING
from typing import Optional
from typing import cast
from typing import overload

import pendulum

from pendulum.constants import SECS_PER_HOUR
from pendulum.constants import SECS_PER_MIN
from pendulum.constants import USECS_PER_SEC
from pendulum.duration import AbsoluteDuration
from pendulum.duration import Duration
from pendulum.mixins.default import FormattableMixin
from pendulum.tz.timezone import UTC


if TYPE_CHECKING:
    from typing_extensions import Literal
    from typing_extensions import Self
    from typing_extensions import SupportsIndex

    from pendulum.tz.timezone import FixedTimezone
    from pendulum.tz.timezone import Timezone


class Time(FormattableMixin, time):
    """
    Represents a time instance as hour, minute, second, microsecond.
    """

    @classmethod
    def instance(
        cls, t: time, tz: str | Timezone | FixedTimezone | datetime.tzinfo | None = UTC
    ) -> Self:
        tz = t.tzinfo or tz

        if tz is not None:
            tz = pendulum._safe_timezone(tz)

        return cls(t.hour, t.minute, t.second, t.microsecond, tzinfo=tz, fold=t.fold)

    # String formatting
    def __repr__(self) -> str:
        us = ""
        if self.microsecond:
            us = f", {self.microsecond}"

        tzinfo = ""
        if self.tzinfo:
            tzinfo = f", tzinfo={self.tzinfo!r}"

        return (
            f"{self.__class__.__name__}"
            f"({self.hour}, {self.minute}, {self.second}{us}{tzinfo})"
        )

    # Comparisons

    def closest(self, dt1: Time | time, dt2: Time | time) -> Self:
        """
        Get the closest time from the instance.
        """
        dt1 = self.__class__(dt1.hour, dt1.minute, dt1.second, dt1.microsecond)
        dt2 = self.__class__(dt2.hour, dt2.minute, dt2.second, dt2.microsecond)

        if self.diff(dt1).in_seconds() < self.diff(dt2).in_seconds():
            return dt1

        return dt2

    def farthest(self, dt1: Time | time, dt2: Time | time) -> Self:
        """
        Get the farthest time from the instance.
        """
        dt1 = self.__class__(dt1.hour, dt1.minute, dt1.second, dt1.microsecond)
        dt2 = self.__class__(dt2.hour, dt2.minute, dt2.second, dt2.microsecond)

        if self.diff(dt1).in_seconds() > self.diff(dt2).in_seconds():
            return dt1

        return dt2

    # ADDITIONS AND SUBSTRACTIONS

    def add(
        self, hours: int = 0, minutes: int = 0, seconds: int = 0, microseconds: int = 0
    ) -> Time:
        """
        Add duration to the instance.

        :param hours: The number of hours
        :param minutes: The number of minutes
        :param seconds: The number of seconds
        :param microseconds: The number of microseconds
        """
        from pendulum.datetime import DateTime

        return (
            DateTime.EPOCH.at(self.hour, self.minute, self.second, self.microsecond)
            .add(
                hours=hours, minutes=minutes, seconds=seconds, microseconds=microseconds
            )
            .time()
        )

    def subtract(
        self, hours: int = 0, minutes: int = 0, seconds: int = 0, microseconds: int = 0
    ) -> Time:
        """
        Add duration to the instance.

        :param hours: The number of hours
        :type hours: int

        :param minutes: The number of minutes
        :type minutes: int

        :param seconds: The number of seconds
        :type seconds: int

        :param microseconds: The number of microseconds
        :type microseconds: int

        :rtype: Time
        """
        from pendulum.datetime import DateTime

        return (
            DateTime.EPOCH.at(self.hour, self.minute, self.second, self.microsecond)
            .subtract(
                hours=hours, minutes=minutes, seconds=seconds, microseconds=microseconds
            )
            .time()
        )

    def add_timedelta(self, delta: datetime.timedelta) -> Time:
        """
        Add timedelta duration to the instance.

        :param delta: The timedelta instance
        """
        if delta.days:
            raise TypeError("Cannot add timedelta with days to Time.")

        return self.add(seconds=delta.seconds, microseconds=delta.microseconds)

    def subtract_timedelta(self, delta: datetime.timedelta) -> Time:
        """
        Remove timedelta duration from the instance.

        :param delta: The timedelta instance
        """
        if delta.days:
            raise TypeError("Cannot subtract timedelta with days to Time.")

        return self.subtract(seconds=delta.seconds, microseconds=delta.microseconds)

    def __add__(self, other: datetime.timedelta) -> Time:
        if not isinstance(other, timedelta):
            return NotImplemented

        return self.add_timedelta(other)

    @overload
    def __sub__(self, other: time) -> pendulum.Duration: ...

    @overload
    def __sub__(self, other: datetime.timedelta) -> Time: ...

    def __sub__(self, other: time | datetime.timedelta) -> pendulum.Duration | Time:
        if not isinstance(other, (Time, time, timedelta)):
            return NotImplemented

        if isinstance(other, timedelta):
            return self.subtract_timedelta(other)

        if isinstance(other, time):
            if other.tzinfo is not None:
                raise TypeError("Cannot subtract aware times to or from Time.")

            other = self.__class__(
                other.hour, other.minute, other.second, other.microsecond
            )

        return other.diff(self, False)

    @overload
    def __rsub__(self, other: time) -> pendulum.Duration: ...

    @overload
    def __rsub__(self, other: datetime.timedelta) -> Time: ...

    def __rsub__(self, other: time | datetime.timedelta) -> pendulum.Duration | Time:
        if not isinstance(other, (Time, time)):
            return NotImplemented

        if isinstance(other, time):
            if other.tzinfo is not None:
                raise TypeError("Cannot subtract aware times to or from Time.")

            other = self.__class__(
                other.hour, other.minute, other.second, other.microsecond
            )

        return other.__sub__(self)

    # DIFFERENCES

    def diff(self, dt: time | None = None, abs: bool = True) -> Duration:
        """
        Returns the difference between two Time objects as an Duration.

        :param dt: The time to subtract from
        :param abs: Whether to return an absolute duration or not
        """
        if dt is None:
            dt = pendulum.now().time()
        else:
            dt = self.__class__(dt.hour, dt.minute, dt.second, dt.microsecond)

        us1 = (
            self.hour * SECS_PER_HOUR + self.minute * SECS_PER_MIN + self.second
        ) * USECS_PER_SEC

        us2 = (
            dt.hour * SECS_PER_HOUR + dt.minute * SECS_PER_MIN + dt.second
        ) * USECS_PER_SEC

        klass = Duration
        if abs:
            klass = AbsoluteDuration

        return klass(microseconds=us2 - us1)

    def diff_for_humans(
        self,
        other: time | None = None,
        absolute: bool = False,
        locale: str | None = None,
    ) -> str:
        """
        Get the difference in a human readable format in the current locale.

        :param dt: The time to subtract from
        :param absolute: removes time difference modifiers ago, after, etc
        :param locale: The locale to use for localization
        """
        is_now = other is None

        if is_now:
            other = pendulum.now().time()

        diff = self.diff(other)

        return pendulum.format_diff(diff, is_now, absolute, locale)

    # Compatibility methods

    def replace(
        self,
        hour: SupportsIndex | None = None,
        minute: SupportsIndex | None = None,
        second: SupportsIndex | None = None,
        microsecond: SupportsIndex | None = None,
        tzinfo: bool | datetime.tzinfo | Literal[True] | None = True,
        fold: int = 0,
    ) -> Self:
        if tzinfo is True:
            tzinfo = self.tzinfo

        hour = hour if hour is not None else self.hour
        minute = minute if minute is not None else self.minute
        second = second if second is not None else self.second
        microsecond = microsecond if microsecond is not None else self.microsecond

        t = super().replace(
            hour,
            minute,
            second,
            microsecond,
            tzinfo=cast("Optional[datetime.tzinfo]", tzinfo),
            fold=fold,
        )
        return self.__class__(
            t.hour, t.minute, t.second, t.microsecond, tzinfo=t.tzinfo
        )

    def __getnewargs__(self) -> tuple[Time]:
        return (self,)

    def _get_state(
        self, protocol: SupportsIndex = 3
    ) -> tuple[int, int, int, int, datetime.tzinfo | None]:
        tz = self.tzinfo

        return self.hour, self.minute, self.second, self.microsecond, tz

    def __reduce__(
        self,
    ) -> tuple[type[Time], tuple[int, int, int, int, datetime.tzinfo | None]]:
        return self.__reduce_ex__(2)

    def __reduce_ex__(
        self, protocol: SupportsIndex
    ) -> tuple[type[Time], tuple[int, int, int, int, datetime.tzinfo | None]]:
        return self.__class__, self._get_state(protocol)


Time.min = Time(0, 0, 0)
Time.max = Time(23, 59, 59, 999999)
Time.resolution = Duration(microseconds=1)


# === src/pendulum/interval.py ===
from __future__ import annotations

import operator

from datetime import date
from datetime import datetime
from datetime import timedelta
from typing import TYPE_CHECKING
from typing import Generic
from typing import TypeVar
from typing import cast
from typing import overload

import pendulum

from pendulum.constants import MONTHS_PER_YEAR
from pendulum.duration import Duration
from pendulum.helpers import precise_diff


if TYPE_CHECKING:
    from collections.abc import Iterator

    from typing_extensions import Self
    from typing_extensions import SupportsIndex

    from pendulum.helpers import PreciseDiff
    from pendulum.locales.locale import Locale


_T = TypeVar("_T", bound=date)


class Interval(Duration, Generic[_T]):
    """
    An interval of time between two datetimes.
    """

    def __new__(cls, start: _T, end: _T, absolute: bool = False) -> Self:
        if (isinstance(start, datetime) and not isinstance(end, datetime)) or (
            not isinstance(start, datetime) and isinstance(end, datetime)
        ):
            raise ValueError(
                "Both start and end of an Interval must have the same type"
            )

        if (
            isinstance(start, datetime)
            and isinstance(end, datetime)
            and (
                (start.tzinfo is None and end.tzinfo is not None)
                or (start.tzinfo is not None and end.tzinfo is None)
            )
        ):
            raise TypeError("can't compare offset-naive and offset-aware datetimes")

        if absolute and start > end:
            end, start = start, end

        _start = start
        _end = end
        if isinstance(start, pendulum.DateTime):
            _start = cast(
                "_T",
                datetime(
                    start.year,
                    start.month,
                    start.day,
                    start.hour,
                    start.minute,
                    start.second,
                    start.microsecond,
                    tzinfo=start.tzinfo,
                    fold=start.fold,
                ),
            )
        elif isinstance(start, pendulum.Date):
            _start = cast("_T", date(start.year, start.month, start.day))

        if isinstance(end, pendulum.DateTime):
            _end = cast(
                "_T",
                datetime(
                    end.year,
                    end.month,
                    end.day,
                    end.hour,
                    end.minute,
                    end.second,
                    end.microsecond,
                    tzinfo=end.tzinfo,
                    fold=end.fold,
                ),
            )
        elif isinstance(end, pendulum.Date):
            _end = cast("_T", date(end.year, end.month, end.day))

        # Fixing issues with datetime.__sub__()
        # not handling offsets if the tzinfo is the same
        if (
            isinstance(_start, datetime)
            and isinstance(_end, datetime)
            and _start.tzinfo is _end.tzinfo
        ):
            if _start.tzinfo is not None:
                offset = cast("timedelta", cast("datetime", start).utcoffset())
                _start = cast("_T", (_start - offset).replace(tzinfo=None))

            if isinstance(end, datetime) and _end.tzinfo is not None:
                offset = cast("timedelta", end.utcoffset())
                _end = cast("_T", (_end - offset).replace(tzinfo=None))

        delta: timedelta = _end - _start

        return super().__new__(cls, seconds=delta.total_seconds())

    def __init__(self, start: _T, end: _T, absolute: bool = False) -> None:
        super().__init__()

        _start: _T
        if not isinstance(start, pendulum.Date):
            if isinstance(start, datetime):
                start = cast("_T", pendulum.instance(start))
            else:
                start = cast("_T", pendulum.date(start.year, start.month, start.day))

            _start = start
        else:
            if isinstance(start, pendulum.DateTime):
                _start = cast(
                    "_T",
                    datetime(
                        start.year,
                        start.month,
                        start.day,
                        start.hour,
                        start.minute,
                        start.second,
                        start.microsecond,
                        tzinfo=start.tzinfo,
                    ),
                )
            else:
                _start = cast("_T", date(start.year, start.month, start.day))

        _end: _T
        if not isinstance(end, pendulum.Date):
            if isinstance(end, datetime):
                end = cast("_T", pendulum.instance(end))
            else:
                end = cast("_T", pendulum.date(end.year, end.month, end.day))

            _end = end
        else:
            if isinstance(end, pendulum.DateTime):
                _end = cast(
                    "_T",
                    datetime(
                        end.year,
                        end.month,
                        end.day,
                        end.hour,
                        end.minute,
                        end.second,
                        end.microsecond,
                        tzinfo=end.tzinfo,
                    ),
                )
            else:
                _end = cast("_T", date(end.year, end.month, end.day))

        self._invert = False
        if start > end:
            self._invert = True

            if absolute:
                end, start = start, end
                _end, _start = _start, _end

        self._absolute = absolute
        self._start: _T = start
        self._end: _T = end
        self._delta: PreciseDiff = precise_diff(_start, _end)

    @property
    def years(self) -> int:
        return self._delta.years

    @property
    def months(self) -> int:
        return self._delta.months

    @property
    def weeks(self) -> int:
        return abs(self._delta.days) // 7 * self._sign(self._delta.days)

    @property
    def days(self) -> int:
        return self._days

    @property
    def remaining_days(self) -> int:
        return abs(self._delta.days) % 7 * self._sign(self._days)

    @property
    def hours(self) -> int:
        return self._delta.hours

    @property
    def minutes(self) -> int:
        return self._delta.minutes

    @property
    def start(self) -> _T:
        return self._start

    @property
    def end(self) -> _T:
        return self._end

    def in_years(self) -> int:
        """
        Gives the duration of the Interval in full years.
        """
        return self.years

    def in_months(self) -> int:
        """
        Gives the duration of the Interval in full months.
        """
        return self.years * MONTHS_PER_YEAR + self.months

    def in_weeks(self) -> int:
        days = self.in_days()
        sign = 1

        if days < 0:
            sign = -1

        return sign * (abs(days) // 7)

    def in_days(self) -> int:
        return self._delta.total_days

    def in_words(self, locale: str | None = None, separator: str = " ") -> str:
        """
        Get the current interval in words in the current locale.

        Ex: 6 jours 23 heures 58 minutes

        :param locale: The locale to use. Defaults to current locale.
        :param separator: The separator to use between each unit
        """
        from pendulum.locales.locale import Locale

        intervals = [
            ("year", self.years),
            ("month", self.months),
            ("week", self.weeks),
            ("day", self.remaining_days),
            ("hour", self.hours),
            ("minute", self.minutes),
            ("second", self.remaining_seconds),
        ]
        loaded_locale: Locale = Locale.load(locale or pendulum.get_locale())
        parts = []
        for interval in intervals:
            unit, interval_count = interval
            if abs(interval_count) > 0:
                translation = loaded_locale.translation(
                    f"units.{unit}.{loaded_locale.plural(abs(interval_count))}"
                )
                parts.append(translation.format(interval_count))

        if not parts:
            count: str | int = 0
            if abs(self.microseconds) > 0:
                unit = f"units.second.{loaded_locale.plural(1)}"
                count = f"{abs(self.microseconds) / 1e6:.2f}"
            else:
                unit = f"units.microsecond.{loaded_locale.plural(0)}"

            translation = loaded_locale.translation(unit)
            parts.append(translation.format(count))

        return separator.join(parts)

    def range(self, unit: str, amount: int = 1) -> Iterator[_T]:
        method = "add"
        op = operator.le
        if not self._absolute and self.invert:
            method = "subtract"
            op = operator.ge

        start, end = self.start, self.end

        i = amount
        while op(start, end):
            yield start

            start = getattr(self.start, method)(**{unit: i})

            i += amount

    def as_duration(self) -> Duration:
        """
        Return the Interval as a Duration.
        """
        return Duration(seconds=self.total_seconds())

    def __iter__(self) -> Iterator[_T]:
        return self.range("days")

    def __contains__(self, item: _T) -> bool:
        return self.start <= item <= self.end

    def __add__(self, other: timedelta) -> Duration:  # type: ignore[override]
        return self.as_duration().__add__(other)

    __radd__ = __add__  # type: ignore[assignment]

    def __sub__(self, other: timedelta) -> Duration:  # type: ignore[override]
        return self.as_duration().__sub__(other)

    def __neg__(self) -> Self:
        return self.__class__(self.end, self.start, self._absolute)

    def __mul__(self, other: int | float) -> Duration:  # type: ignore[override]
        return self.as_duration().__mul__(other)

    __rmul__ = __mul__  # type: ignore[assignment]

    @overload  # type: ignore[override]
    def __floordiv__(self, other: timedelta) -> int: ...

    @overload
    def __floordiv__(self, other: int) -> Duration: ...

    def __floordiv__(self, other: int | timedelta) -> int | Duration:
        return self.as_duration().__floordiv__(other)

    __div__ = __floordiv__  # type: ignore[assignment]

    @overload  # type: ignore[override]
    def __truediv__(self, other: timedelta) -> float: ...

    @overload
    def __truediv__(self, other: float) -> Duration: ...

    def __truediv__(self, other: float | timedelta) -> Duration | float:
        return self.as_duration().__truediv__(other)

    def __mod__(self, other: timedelta) -> Duration:  # type: ignore[override]
        return self.as_duration().__mod__(other)

    def __divmod__(self, other: timedelta) -> tuple[int, Duration]:
        return self.as_duration().__divmod__(other)

    def __abs__(self) -> Self:
        return self.__class__(self.start, self.end, absolute=True)

    def __repr__(self) -> str:
        return f"<Interval [{self._start} -> {self._end}]>"

    def __str__(self) -> str:
        return self.__repr__()

    def _cmp(self, other: timedelta) -> int:
        # Only needed for PyPy
        assert isinstance(other, timedelta)

        if isinstance(other, Interval):
            other = other.as_timedelta()

        td = self.as_timedelta()

        return 0 if td == other else 1 if td > other else -1

    def _getstate(self, protocol: SupportsIndex = 3) -> tuple[_T, _T, bool]:
        start, end = self.start, self.end

        if self._invert and self._absolute:
            end, start = start, end

        return start, end, self._absolute

    def __reduce__(
        self,
    ) -> tuple[type[Self], tuple[_T, _T, bool]]:
        return self.__reduce_ex__(2)

    def __reduce_ex__(
        self, protocol: SupportsIndex
    ) -> tuple[type[Self], tuple[_T, _T, bool]]:
        return self.__class__, self._getstate(protocol)

    def __hash__(self) -> int:
        return hash((self.start, self.end, self._absolute))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Interval):
            return (self.start, self.end, self._absolute) == (
                other.start,
                other.end,
                other._absolute,
            )
        else:
            return self.as_duration() == other

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)


# === src/pendulum/constants.py ===
# The day constants
from __future__ import annotations


# Number of X in Y.
YEARS_PER_CENTURY = 100
YEARS_PER_DECADE = 10
MONTHS_PER_YEAR = 12
WEEKS_PER_YEAR = 52
DAYS_PER_WEEK = 7
HOURS_PER_DAY = 24
MINUTES_PER_HOUR = 60
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = MINUTES_PER_HOUR * SECONDS_PER_MINUTE
SECONDS_PER_DAY = HOURS_PER_DAY * SECONDS_PER_HOUR
US_PER_SECOND = 1000000

# Formats
ATOM = "YYYY-MM-DDTHH:mm:ssZ"
COOKIE = "dddd, DD-MMM-YYYY HH:mm:ss zz"
ISO8601 = "YYYY-MM-DDTHH:mm:ssZ"
ISO8601_EXTENDED = "YYYY-MM-DDTHH:mm:ss.SSSSSSZ"
RFC822 = "ddd, DD MMM YY HH:mm:ss ZZ"
RFC850 = "dddd, DD-MMM-YY HH:mm:ss zz"
RFC1036 = "ddd, DD MMM YY HH:mm:ss ZZ"
RFC1123 = "ddd, DD MMM YYYY HH:mm:ss ZZ"
RFC2822 = "ddd, DD MMM YYYY HH:mm:ss ZZ"
RFC3339 = ISO8601
RFC3339_EXTENDED = ISO8601_EXTENDED
RSS = "ddd, DD MMM YYYY HH:mm:ss ZZ"
W3C = ISO8601


EPOCH_YEAR = 1970

DAYS_PER_N_YEAR = 365
DAYS_PER_L_YEAR = 366

USECS_PER_SEC = 1000000

SECS_PER_MIN = 60
SECS_PER_HOUR = 60 * SECS_PER_MIN
SECS_PER_DAY = SECS_PER_HOUR * 24

# 400-year chunks always have 146097 days (20871 weeks).
SECS_PER_400_YEARS = 146097 * SECS_PER_DAY

# The number of seconds in an aligned 100-year chunk, for those that
# do not begin with a leap year and those that do respectively.
SECS_PER_100_YEARS = (
    (76 * DAYS_PER_N_YEAR + 24 * DAYS_PER_L_YEAR) * SECS_PER_DAY,
    (75 * DAYS_PER_N_YEAR + 25 * DAYS_PER_L_YEAR) * SECS_PER_DAY,
)

# The number of seconds in an aligned 4-year chunk, for those that
# do not begin with a leap year and those that do respectively.
SECS_PER_4_YEARS = (
    (4 * DAYS_PER_N_YEAR + 0 * DAYS_PER_L_YEAR) * SECS_PER_DAY,
    (3 * DAYS_PER_N_YEAR + 1 * DAYS_PER_L_YEAR) * SECS_PER_DAY,
)

# The number of seconds in non-leap and leap years respectively.
SECS_PER_YEAR = (DAYS_PER_N_YEAR * SECS_PER_DAY, DAYS_PER_L_YEAR * SECS_PER_DAY)

DAYS_PER_YEAR = (DAYS_PER_N_YEAR, DAYS_PER_L_YEAR)

# The month lengths in non-leap and leap years respectively.
DAYS_PER_MONTHS = (
    (-1, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31),
    (-1, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31),
)

# The day offsets of the beginning of each (1-based) month in non-leap
# and leap years respectively.
# For example, in a leap year there are 335 days before December.
MONTHS_OFFSETS = (
    (-1, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365),
    (-1, 0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366),
)

DAY_OF_WEEK_TABLE = (0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4)

TM_SUNDAY = 0
TM_MONDAY = 1
TM_TUESDAY = 2
TM_WEDNESDAY = 3
TM_THURSDAY = 4
TM_FRIDAY = 5
TM_SATURDAY = 6

TM_JANUARY = 0
TM_FEBRUARY = 1
TM_MARCH = 2
TM_APRIL = 3
TM_MAY = 4
TM_JUNE = 5
TM_JULY = 6
TM_AUGUST = 7
TM_SEPTEMBER = 8
TM_OCTOBER = 9
TM_NOVEMBER = 10
TM_DECEMBER = 11


# === src/pendulum/duration.py ===
from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING
from typing import cast
from typing import overload

import pendulum

from pendulum.constants import SECONDS_PER_DAY
from pendulum.constants import SECONDS_PER_HOUR
from pendulum.constants import SECONDS_PER_MINUTE
from pendulum.constants import US_PER_SECOND
from pendulum.utils._compat import PYPY


if TYPE_CHECKING:
    from typing_extensions import Self


def _divide_and_round(a: float, b: float) -> int:
    """divide a by b and round result to the nearest integer

    When the ratio is exactly half-way between two integers,
    the even integer is returned.
    """
    # Based on the reference implementation for divmod_near
    # in Objects/longobject.c.
    q, r = divmod(a, b)

    # The output of divmod() is either a float or an int,
    # but we always want it to be an int.
    q = int(q)

    # round up if either r / b > 0.5, or r / b == 0.5 and q is odd.
    # The expression r / b > 0.5 is equivalent to 2 * r > b if b is
    # positive, 2 * r < b if b negative.
    r *= 2
    greater_than_half = r > b if b > 0 else r < b
    if greater_than_half or (r == b and q % 2 == 1):
        q += 1

    return q


class Duration(timedelta):
    """
    Replacement for the standard timedelta class.

    Provides several improvements over the base class.
    """

    _total: float = 0
    _years: int = 0
    _months: int = 0
    _weeks: int = 0
    _days: int = 0
    _remaining_days: int = 0
    _seconds: int = 0
    _microseconds: int = 0

    _y = None
    _m = None
    _w = None
    _d = None
    _h = None
    _i = None
    _s = None
    _invert = None

    def __new__(
        cls,
        days: float = 0,
        seconds: float = 0,
        microseconds: float = 0,
        milliseconds: float = 0,
        minutes: float = 0,
        hours: float = 0,
        weeks: float = 0,
        years: float = 0,
        months: float = 0,
    ) -> Self:
        if not isinstance(years, int) or not isinstance(months, int):
            raise ValueError("Float year and months are not supported")

        self = timedelta.__new__(
            cls,
            days + years * 365 + months * 30,
            seconds,
            microseconds,
            milliseconds,
            minutes,
            hours,
            weeks,
        )

        # Intuitive normalization
        total = self.total_seconds() - (years * 365 + months * 30) * SECONDS_PER_DAY
        self._total = total

        m = 1
        if total < 0:
            m = -1

        self._microseconds = round(total % m * 1e6)
        self._seconds = abs(int(total)) % SECONDS_PER_DAY * m

        _days = abs(int(total)) // SECONDS_PER_DAY * m
        self._days = _days
        self._remaining_days = abs(_days) % 7 * m
        self._weeks = abs(_days) // 7 * m
        self._months = months
        self._years = years

        self._signature = {  # type: ignore[attr-defined]
            "years": years,
            "months": months,
            "weeks": weeks,
            "days": days,
            "hours": hours,
            "minutes": minutes,
            "seconds": seconds,
            "microseconds": microseconds + milliseconds * 1000,
        }

        return self

    def total_minutes(self) -> float:
        return self.total_seconds() / SECONDS_PER_MINUTE

    def total_hours(self) -> float:
        return self.total_seconds() / SECONDS_PER_HOUR

    def total_days(self) -> float:
        return self.total_seconds() / SECONDS_PER_DAY

    def total_weeks(self) -> float:
        return self.total_days() / 7

    if PYPY:

        def total_seconds(self) -> float:
            days = 0

            if hasattr(self, "_years"):
                days += self._years * 365

            if hasattr(self, "_months"):
                days += self._months * 30

            if hasattr(self, "_remaining_days"):
                days += self._weeks * 7 + self._remaining_days
            else:
                days += self._days

            return (
                (days * SECONDS_PER_DAY + self._seconds) * US_PER_SECOND
                + self._microseconds
            ) / US_PER_SECOND

    @property
    def years(self) -> int:
        return self._years

    @property
    def months(self) -> int:
        return self._months

    @property
    def weeks(self) -> int:
        return self._weeks

    if PYPY:

        @property
        def days(self) -> int:
            return self._years * 365 + self._months * 30 + self._days

    @property
    def remaining_days(self) -> int:
        return self._remaining_days

    @property
    def hours(self) -> int:
        if self._h is None:
            seconds = self._seconds
            self._h = 0
            if abs(seconds) >= 3600:
                self._h = (abs(seconds) // 3600 % 24) * self._sign(seconds)

        return self._h

    @property
    def minutes(self) -> int:
        if self._i is None:
            seconds = self._seconds
            self._i = 0
            if abs(seconds) >= 60:
                self._i = (abs(seconds) // 60 % 60) * self._sign(seconds)

        return self._i

    @property
    def seconds(self) -> int:
        return self._seconds

    @property
    def remaining_seconds(self) -> int:
        if self._s is None:
            self._s = self._seconds
            self._s = abs(self._s) % 60 * self._sign(self._s)

        return self._s

    @property
    def microseconds(self) -> int:
        return self._microseconds

    @property
    def invert(self) -> bool:
        if self._invert is None:
            self._invert = self.total_seconds() < 0

        return self._invert

    def in_weeks(self) -> int:
        return int(self.total_weeks())

    def in_days(self) -> int:
        return int(self.total_days())

    def in_hours(self) -> int:
        return int(self.total_hours())

    def in_minutes(self) -> int:
        return int(self.total_minutes())

    def in_seconds(self) -> int:
        return int(self.total_seconds())

    def in_words(self, locale: str | None = None, separator: str = " ") -> str:
        """
        Get the current interval in words in the current locale.

        Ex: 6 jours 23 heures 58 minutes

        :param locale: The locale to use. Defaults to current locale.
        :param separator: The separator to use between each unit
        """
        intervals = [
            ("year", self.years),
            ("month", self.months),
            ("week", self.weeks),
            ("day", self.remaining_days),
            ("hour", self.hours),
            ("minute", self.minutes),
            ("second", self.remaining_seconds),
        ]

        if locale is None:
            locale = pendulum.get_locale()

        loaded_locale = pendulum.locale(locale)

        parts = []
        for interval in intervals:
            unit, interval_count = interval
            if abs(interval_count) > 0:
                translation = loaded_locale.translation(
                    f"units.{unit}.{loaded_locale.plural(abs(interval_count))}"
                )
                parts.append(translation.format(interval_count))

        if not parts:
            count: int | str = 0
            if abs(self.microseconds) > 0:
                unit = f"units.second.{loaded_locale.plural(1)}"
                count = f"{abs(self.microseconds) / 1e6:.2f}"
            else:
                unit = f"units.microsecond.{loaded_locale.plural(0)}"
            translation = loaded_locale.translation(unit)
            parts.append(translation.format(count))

        return separator.join(parts)

    def _sign(self, value: float) -> int:
        if value < 0:
            return -1

        return 1

    def as_timedelta(self) -> timedelta:
        """
        Return the interval as a native timedelta.
        """
        return timedelta(seconds=self.total_seconds())

    def __str__(self) -> str:
        return self.in_words()

    def __repr__(self) -> str:
        rep = f"{self.__class__.__name__}("

        if self._years:
            rep += f"years={self._years}, "

        if self._months:
            rep += f"months={self._months}, "

        if self._weeks:
            rep += f"weeks={self._weeks}, "

        if self._days:
            rep += f"days={self._remaining_days}, "

        if self.hours:
            rep += f"hours={self.hours}, "

        if self.minutes:
            rep += f"minutes={self.minutes}, "

        if self.remaining_seconds:
            rep += f"seconds={self.remaining_seconds}, "

        if self.microseconds:
            rep += f"microseconds={self.microseconds}, "

        rep += ")"

        return rep.replace(", )", ")")

    def __add__(self, other: timedelta) -> Self:
        if isinstance(other, timedelta):
            return self.__class__(seconds=self.total_seconds() + other.total_seconds())

        return NotImplemented

    __radd__ = __add__

    def __sub__(self, other: timedelta) -> Self:
        if isinstance(other, timedelta):
            return self.__class__(seconds=self.total_seconds() - other.total_seconds())

        return NotImplemented

    def __neg__(self) -> Self:
        return self.__class__(
            years=-self._years,
            months=-self._months,
            weeks=-self._weeks,
            days=-self._remaining_days,
            seconds=-self._seconds,
            microseconds=-self._microseconds,
        )

    def _to_microseconds(self) -> int:
        return (self._days * (24 * 3600) + self._seconds) * 1000000 + self._microseconds

    def __mul__(self, other: int | float) -> Self:
        if isinstance(other, int):
            return self.__class__(
                years=self._years * other,
                months=self._months * other,
                seconds=self._total * other,
            )

        if isinstance(other, float):
            usec = self._to_microseconds()
            a, b = other.as_integer_ratio()

            return self.__class__(0, 0, _divide_and_round(usec * a, b))

        return NotImplemented

    __rmul__ = __mul__

    @overload
    def __floordiv__(self, other: timedelta) -> int: ...

    @overload
    def __floordiv__(self, other: int) -> Self: ...

    def __floordiv__(self, other: int | timedelta) -> int | Duration:
        if not isinstance(other, (int, timedelta)):
            return NotImplemented

        usec = self._to_microseconds()
        if isinstance(other, timedelta):
            return cast(
                "int",
                usec // other._to_microseconds(),  # type: ignore[attr-defined]
            )

        if isinstance(other, int):
            return self.__class__(
                0,
                0,
                usec // other,
                years=self._years // other,
                months=self._months // other,
            )

    @overload
    def __truediv__(self, other: timedelta) -> float: ...

    @overload
    def __truediv__(self, other: float) -> Self: ...

    def __truediv__(self, other: int | float | timedelta) -> Self | float:
        if not isinstance(other, (int, float, timedelta)):
            return NotImplemented

        usec = self._to_microseconds()
        if isinstance(other, timedelta):
            return cast(
                "float",
                usec / other._to_microseconds(),  # type: ignore[attr-defined]
            )

        if isinstance(other, int):
            return self.__class__(
                0,
                0,
                _divide_and_round(usec, other),
                years=_divide_and_round(self._years, other),
                months=_divide_and_round(self._months, other),
            )

        if isinstance(other, float):
            a, b = other.as_integer_ratio()

            return self.__class__(
                0,
                0,
                _divide_and_round(b * usec, a),
                years=_divide_and_round(self._years * b, a),
                months=_divide_and_round(self._months, other),
            )

    __div__ = __floordiv__

    def __mod__(self, other: timedelta) -> Self:
        if isinstance(other, timedelta):
            r = self._to_microseconds() % other._to_microseconds()  # type: ignore[attr-defined]

            return self.__class__(0, 0, r)

        return NotImplemented

    def __divmod__(self, other: timedelta) -> tuple[int, Duration]:
        if isinstance(other, timedelta):
            q, r = divmod(
                self._to_microseconds(),
                other._to_microseconds(),  # type: ignore[attr-defined]
            )

            return q, self.__class__(0, 0, r)

        return NotImplemented

    def __deepcopy__(self, _: dict[int, Self]) -> Self:
        return self.__class__(
            days=self.remaining_days,
            seconds=self.remaining_seconds,
            microseconds=self.microseconds,
            minutes=self.minutes,
            hours=self.hours,
            years=self.years,
            months=self.months,
        )


Duration.min = Duration(days=-999999999)
Duration.max = Duration(
    days=999999999, hours=23, minutes=59, seconds=59, microseconds=999999
)
Duration.resolution = Duration(microseconds=1)


class AbsoluteDuration(Duration):
    """
    Duration that expresses a time difference in absolute values.
    """

    def __new__(
        cls,
        days: float = 0,
        seconds: float = 0,
        microseconds: float = 0,
        milliseconds: float = 0,
        minutes: float = 0,
        hours: float = 0,
        weeks: float = 0,
        years: float = 0,
        months: float = 0,
    ) -> AbsoluteDuration:
        if not isinstance(years, int) or not isinstance(months, int):
            raise ValueError("Float year and months are not supported")

        self = timedelta.__new__(
            cls, days, seconds, microseconds, milliseconds, minutes, hours, weeks
        )

        # We need to compute the total_seconds() value
        # on a native timedelta object
        delta = timedelta(
            days, seconds, microseconds, milliseconds, minutes, hours, weeks
        )

        # Intuitive normalization
        self._total = delta.total_seconds()
        total = abs(self._total)

        self._microseconds = round(total % 1 * 1e6)
        days, self._seconds = divmod(int(total), SECONDS_PER_DAY)
        self._days = abs(days + years * 365 + months * 30)
        self._weeks, self._remaining_days = divmod(days, 7)
        self._months = abs(months)
        self._years = abs(years)

        return self

    def total_seconds(self) -> float:
        return abs(self._total)

    @property
    def invert(self) -> bool:
        if self._invert is None:
            self._invert = self._total < 0

        return self._invert


# === src/pendulum/__init__.py ===
from __future__ import annotations

import datetime as _datetime

from typing import Any
from typing import Union
from typing import cast
from typing import overload

from pendulum.constants import DAYS_PER_WEEK
from pendulum.constants import HOURS_PER_DAY
from pendulum.constants import MINUTES_PER_HOUR
from pendulum.constants import MONTHS_PER_YEAR
from pendulum.constants import SECONDS_PER_DAY
from pendulum.constants import SECONDS_PER_HOUR
from pendulum.constants import SECONDS_PER_MINUTE
from pendulum.constants import WEEKS_PER_YEAR
from pendulum.constants import YEARS_PER_CENTURY
from pendulum.constants import YEARS_PER_DECADE
from pendulum.date import Date
from pendulum.datetime import DateTime
from pendulum.day import WeekDay
from pendulum.duration import Duration
from pendulum.formatting import Formatter
from pendulum.helpers import format_diff
from pendulum.helpers import get_locale
from pendulum.helpers import locale
from pendulum.helpers import set_locale
from pendulum.helpers import week_ends_at
from pendulum.helpers import week_starts_at
from pendulum.interval import Interval
from pendulum.parser import parse as parse
from pendulum.testing.traveller import Traveller
from pendulum.time import Time
from pendulum.tz import UTC
from pendulum.tz import fixed_timezone
from pendulum.tz import local_timezone
from pendulum.tz import set_local_timezone
from pendulum.tz import test_local_timezone
from pendulum.tz import timezones
from pendulum.tz.timezone import FixedTimezone
from pendulum.tz.timezone import Timezone


MONDAY = WeekDay.MONDAY
TUESDAY = WeekDay.TUESDAY
WEDNESDAY = WeekDay.WEDNESDAY
THURSDAY = WeekDay.THURSDAY
FRIDAY = WeekDay.FRIDAY
SATURDAY = WeekDay.SATURDAY
SUNDAY = WeekDay.SUNDAY

_TEST_NOW: DateTime | None = None
_LOCALE = "en"
_WEEK_STARTS_AT: WeekDay = WeekDay.MONDAY
_WEEK_ENDS_AT: WeekDay = WeekDay.SUNDAY

_formatter = Formatter()


@overload
def timezone(name: int) -> FixedTimezone: ...


@overload
def timezone(name: str) -> Timezone: ...


@overload
def timezone(name: str | int) -> Timezone | FixedTimezone: ...


def timezone(name: str | int) -> Timezone | FixedTimezone:
    """
    Return a Timezone instance given its name.
    """
    if isinstance(name, int):
        return fixed_timezone(name)

    if name.lower() == "utc":
        return UTC

    return Timezone(name)


def _safe_timezone(
    obj: str | float | _datetime.tzinfo | Timezone | FixedTimezone | None,
    dt: _datetime.datetime | None = None,
) -> Timezone | FixedTimezone:
    """
    Creates a timezone instance
    from a string, Timezone, TimezoneInfo or integer offset.
    """
    if isinstance(obj, (Timezone, FixedTimezone)):
        return obj

    if obj is None or obj == "local":
        return local_timezone()

    if isinstance(obj, (int, float)):
        obj = int(obj * 60 * 60)
    elif isinstance(obj, _datetime.tzinfo):
        # zoneinfo
        if hasattr(obj, "key"):
            obj = obj.key
        # pytz
        elif hasattr(obj, "localize"):
            obj = obj.zone  # type: ignore[attr-defined]
        elif obj.tzname(None) == "UTC":
            return UTC
        else:
            offset = obj.utcoffset(dt)

            if offset is None:
                offset = _datetime.timedelta(0)

            obj = int(offset.total_seconds())

    obj = cast("Union[str, int]", obj)

    return timezone(obj)


# Public API
def datetime(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    second: int = 0,
    microsecond: int = 0,
    tz: str | float | Timezone | FixedTimezone | _datetime.tzinfo | None = UTC,
    fold: int = 1,
    raise_on_unknown_times: bool = False,
) -> DateTime:
    """
    Creates a new DateTime instance from a specific date and time.
    """
    return DateTime.create(
        year,
        month,
        day,
        hour=hour,
        minute=minute,
        second=second,
        microsecond=microsecond,
        tz=tz,
        fold=fold,
        raise_on_unknown_times=raise_on_unknown_times,
    )


def local(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    second: int = 0,
    microsecond: int = 0,
) -> DateTime:
    """
    Return a DateTime in the local timezone.
    """
    return datetime(
        year, month, day, hour, minute, second, microsecond, tz=local_timezone()
    )


def naive(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    second: int = 0,
    microsecond: int = 0,
    fold: int = 1,
) -> DateTime:
    """
    Return a naive DateTime.
    """
    return DateTime(year, month, day, hour, minute, second, microsecond, fold=fold)


def date(year: int, month: int, day: int) -> Date:
    """
    Create a new Date instance.
    """
    return Date(year, month, day)


def time(hour: int, minute: int = 0, second: int = 0, microsecond: int = 0) -> Time:
    """
    Create a new Time instance.
    """
    return Time(hour, minute, second, microsecond)


@overload
def instance(
    obj: _datetime.datetime,
    tz: str | Timezone | FixedTimezone | _datetime.tzinfo | None = UTC,
) -> DateTime: ...


@overload
def instance(
    obj: _datetime.date,
    tz: str | Timezone | FixedTimezone | _datetime.tzinfo | None = UTC,
) -> Date: ...


@overload
def instance(
    obj: _datetime.time,
    tz: str | Timezone | FixedTimezone | _datetime.tzinfo | None = UTC,
) -> Time: ...


def instance(
    obj: _datetime.datetime | _datetime.date | _datetime.time,
    tz: str | Timezone | FixedTimezone | _datetime.tzinfo | None = UTC,
) -> DateTime | Date | Time:
    """
    Create a DateTime/Date/Time instance from a datetime/date/time native one.
    """
    if isinstance(obj, (DateTime, Date, Time)):
        return obj

    if isinstance(obj, _datetime.date) and not isinstance(obj, _datetime.datetime):
        return date(obj.year, obj.month, obj.day)

    if isinstance(obj, _datetime.time):
        return Time.instance(obj, tz=tz)

    return DateTime.instance(obj, tz=tz)


def now(tz: str | Timezone | None = None) -> DateTime:
    """
    Get a DateTime instance for the current date and time.
    """
    return DateTime.now(tz)


def today(tz: str | Timezone = "local") -> DateTime:
    """
    Create a DateTime instance for today.
    """
    return now(tz).start_of("day")


def tomorrow(tz: str | Timezone = "local") -> DateTime:
    """
    Create a DateTime instance for tomorrow.
    """
    return today(tz).add(days=1)


def yesterday(tz: str | Timezone = "local") -> DateTime:
    """
    Create a DateTime instance for yesterday.
    """
    return today(tz).subtract(days=1)


def from_format(
    string: str,
    fmt: str,
    tz: str | Timezone = UTC,
    locale: str | None = None,
) -> DateTime:
    """
    Creates a DateTime instance from a specific format.
    """
    parts = _formatter.parse(string, fmt, now(tz=tz), locale=locale)
    if parts["tz"] is None:
        parts["tz"] = tz

    return datetime(**parts)


def from_timestamp(timestamp: int | float, tz: str | Timezone = UTC) -> DateTime:
    """
    Create a DateTime instance from a timestamp.
    """
    dt = _datetime.datetime.fromtimestamp(timestamp, tz=UTC)

    dt = datetime(
        dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond
    )

    if tz is not UTC or tz != "UTC":
        dt = dt.in_timezone(tz)

    return dt


def duration(
    days: float = 0,
    seconds: float = 0,
    microseconds: float = 0,
    milliseconds: float = 0,
    minutes: float = 0,
    hours: float = 0,
    weeks: float = 0,
    years: float = 0,
    months: float = 0,
) -> Duration:
    """
    Create a Duration instance.
    """
    return Duration(
        days=days,
        seconds=seconds,
        microseconds=microseconds,
        milliseconds=milliseconds,
        minutes=minutes,
        hours=hours,
        weeks=weeks,
        years=years,
        months=months,
    )


def interval(
    start: DateTime, end: DateTime, absolute: bool = False
) -> Interval[DateTime]:
    """
    Create an Interval instance.
    """
    return Interval(start, end, absolute=absolute)


# Testing

_traveller = Traveller(DateTime)

freeze = _traveller.freeze
travel = _traveller.travel
travel_to = _traveller.travel_to
travel_back = _traveller.travel_back


def __getattr__(name: str) -> Any:
    if name == "__version__":
        import importlib.metadata
        import warnings

        warnings.warn(
            "The '__version__' attribute is deprecated and will be removed in"
            " Pendulum 3.4. Use 'importlib.metadata.version(\"pendulum\")' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return importlib.metadata.version("pendulum")

    raise AttributeError(name)


__all__ = [
    "DAYS_PER_WEEK",
    "HOURS_PER_DAY",
    "MINUTES_PER_HOUR",
    "MONTHS_PER_YEAR",
    "SECONDS_PER_DAY",
    "SECONDS_PER_HOUR",
    "SECONDS_PER_MINUTE",
    "UTC",
    "WEEKS_PER_YEAR",
    "YEARS_PER_CENTURY",
    "YEARS_PER_DECADE",
    "Date",
    "DateTime",
    "Duration",
    "FixedTimezone",
    "Formatter",
    "Interval",
    "Time",
    "Timezone",
    "WeekDay",
    "date",
    "datetime",
    "duration",
    "format_diff",
    "freeze",
    "from_format",
    "from_timestamp",
    "get_locale",
    "instance",
    "interval",
    "local",
    "local_timezone",
    "locale",
    "naive",
    "now",
    "parse",
    "set_local_timezone",
    "set_locale",
    "test_local_timezone",
    "time",
    "timezone",
    "timezones",
    "today",
    "tomorrow",
    "travel",
    "travel_back",
    "travel_to",
    "week_ends_at",
    "week_starts_at",
    "yesterday",
]


# === src/pendulum/parser.py ===
from __future__ import annotations

import datetime
import os
import typing as t

import pendulum

from pendulum.duration import Duration
from pendulum.parsing import _Interval
from pendulum.parsing import parse as base_parse
from pendulum.tz.timezone import UTC


if t.TYPE_CHECKING:
    from pendulum.date import Date
    from pendulum.datetime import DateTime
    from pendulum.interval import Interval
    from pendulum.time import Time

with_extensions = os.getenv("PENDULUM_EXTENSIONS", "1") == "1"

try:
    if not with_extensions:
        raise ImportError()

    from pendulum._pendulum import Duration as RustDuration
except ImportError:
    RustDuration = None  # type: ignore[assignment,misc]


def parse(text: str, **options: t.Any) -> Date | Time | DateTime | Duration:
    # Use the mock now value if it exists
    options["now"] = options.get("now")

    return _parse(text, **options)


def _parse(
    text: str, **options: t.Any
) -> Date | DateTime | Time | Duration | Interval[DateTime]:
    """
    Parses a string with the given options.

    :param text: The string to parse.
    """
    # Handling special cases
    if text == "now":
        return pendulum.now(tz=options.get("tz", UTC))

    parsed = base_parse(text, **options)

    if isinstance(parsed, datetime.datetime):
        return pendulum.datetime(
            parsed.year,
            parsed.month,
            parsed.day,
            parsed.hour,
            parsed.minute,
            parsed.second,
            parsed.microsecond,
            tz=parsed.tzinfo or options.get("tz", UTC),
        )

    if isinstance(parsed, datetime.date):
        return pendulum.date(parsed.year, parsed.month, parsed.day)

    if isinstance(parsed, datetime.time):
        return pendulum.time(
            parsed.hour, parsed.minute, parsed.second, parsed.microsecond
        )

    if isinstance(parsed, _Interval):
        if parsed.duration is not None:
            duration = parsed.duration

            if parsed.start is not None:
                dt = pendulum.instance(parsed.start, tz=options.get("tz", UTC))

                return pendulum.interval(
                    dt,
                    dt.add(
                        years=duration.years,
                        months=duration.months,
                        weeks=duration.weeks,
                        days=duration.remaining_days,
                        hours=duration.hours,
                        minutes=duration.minutes,
                        seconds=duration.remaining_seconds,
                        microseconds=duration.microseconds,
                    ),
                )

            dt = pendulum.instance(
                t.cast("datetime.datetime", parsed.end), tz=options.get("tz", UTC)
            )

            return pendulum.interval(
                dt.subtract(
                    years=duration.years,
                    months=duration.months,
                    weeks=duration.weeks,
                    days=duration.remaining_days,
                    hours=duration.hours,
                    minutes=duration.minutes,
                    seconds=duration.remaining_seconds,
                    microseconds=duration.microseconds,
                ),
                dt,
            )

        return pendulum.interval(
            pendulum.instance(
                t.cast("datetime.datetime", parsed.start), tz=options.get("tz", UTC)
            ),
            pendulum.instance(
                t.cast("datetime.datetime", parsed.end), tz=options.get("tz", UTC)
            ),
        )

    if isinstance(parsed, Duration):
        return parsed

    if RustDuration is not None and isinstance(parsed, RustDuration):
        return pendulum.duration(
            years=parsed.years,
            months=parsed.months,
            weeks=parsed.weeks,
            days=parsed.days,
            hours=parsed.hours,
            minutes=parsed.minutes,
            seconds=parsed.seconds,
            microseconds=parsed.microseconds,
        )

    raise NotImplementedError


# === src/pendulum/day.py ===
from __future__ import annotations

from enum import IntEnum


class WeekDay(IntEnum):
    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6


# === src/pendulum/exceptions.py ===
from __future__ import annotations

from pendulum.parsing.exceptions import ParserError


class PendulumException(Exception):
    pass


__all__ = [
    "ParserError",
    "PendulumException",
]


# === src/pendulum/datetime.py ===
from __future__ import annotations

import calendar
import datetime
import traceback

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Optional
from typing import cast
from typing import overload

import pendulum

from pendulum.constants import ATOM
from pendulum.constants import COOKIE
from pendulum.constants import MINUTES_PER_HOUR
from pendulum.constants import MONTHS_PER_YEAR
from pendulum.constants import RFC822
from pendulum.constants import RFC850
from pendulum.constants import RFC1036
from pendulum.constants import RFC1123
from pendulum.constants import RFC2822
from pendulum.constants import RSS
from pendulum.constants import SECONDS_PER_DAY
from pendulum.constants import SECONDS_PER_MINUTE
from pendulum.constants import W3C
from pendulum.constants import YEARS_PER_CENTURY
from pendulum.constants import YEARS_PER_DECADE
from pendulum.date import Date
from pendulum.day import WeekDay
from pendulum.exceptions import PendulumException
from pendulum.helpers import add_duration
from pendulum.interval import Interval
from pendulum.time import Time
from pendulum.tz import UTC
from pendulum.tz import local_timezone
from pendulum.tz.timezone import FixedTimezone
from pendulum.tz.timezone import Timezone


if TYPE_CHECKING:
    from typing_extensions import Literal
    from typing_extensions import Self
    from typing_extensions import SupportsIndex


class DateTime(datetime.datetime, Date):
    EPOCH: ClassVar[DateTime]
    min: ClassVar[DateTime]
    max: ClassVar[DateTime]

    # Formats

    _FORMATS: ClassVar[dict[str, str | Callable[[datetime.datetime], str]]] = {
        "atom": ATOM,
        "cookie": COOKIE,
        "iso8601": lambda dt: dt.isoformat("T"),
        "rfc822": RFC822,
        "rfc850": RFC850,
        "rfc1036": RFC1036,
        "rfc1123": RFC1123,
        "rfc2822": RFC2822,
        "rfc3339": lambda dt: dt.isoformat("T"),
        "rss": RSS,
        "w3c": W3C,
    }

    _MODIFIERS_VALID_UNITS: ClassVar[list[str]] = [
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "decade",
        "century",
    ]

    _EPOCH: datetime.datetime = datetime.datetime(1970, 1, 1, tzinfo=UTC)

    @classmethod
    def create(
        cls,
        year: SupportsIndex,
        month: SupportsIndex,
        day: SupportsIndex,
        hour: SupportsIndex = 0,
        minute: SupportsIndex = 0,
        second: SupportsIndex = 0,
        microsecond: SupportsIndex = 0,
        tz: str | float | Timezone | FixedTimezone | None | datetime.tzinfo = UTC,
        fold: int = 1,
        raise_on_unknown_times: bool = False,
    ) -> Self:
        """
        Creates a new DateTime instance from a specific date and time.
        """
        if tz is not None:
            tz = pendulum._safe_timezone(tz)

        dt = datetime.datetime(
            year, month, day, hour, minute, second, microsecond, fold=fold
        )

        if tz is not None:
            dt = tz.convert(dt, raise_on_unknown_times=raise_on_unknown_times)

        return cls(
            dt.year,
            dt.month,
            dt.day,
            dt.hour,
            dt.minute,
            dt.second,
            dt.microsecond,
            tzinfo=dt.tzinfo,
            fold=dt.fold,
        )

    @classmethod
    def instance(
        cls,
        dt: datetime.datetime,
        tz: str | Timezone | FixedTimezone | datetime.tzinfo | None = UTC,
    ) -> Self:
        tz = dt.tzinfo or tz

        if tz is not None:
            tz = pendulum._safe_timezone(tz, dt=dt)

        return cls.create(
            dt.year,
            dt.month,
            dt.day,
            dt.hour,
            dt.minute,
            dt.second,
            dt.microsecond,
            tz=tz,
            fold=dt.fold,
        )

    @overload
    @classmethod
    def now(cls, tz: datetime.tzinfo | None = None) -> Self: ...

    @overload
    @classmethod
    def now(cls, tz: str | Timezone | FixedTimezone | None = None) -> Self: ...

    @classmethod
    def now(
        cls, tz: str | Timezone | FixedTimezone | datetime.tzinfo | None = None
    ) -> Self:
        """
        Get a DateTime instance for the current date and time.
        """
        if tz is None or tz == "local":
            dt = datetime.datetime.now(local_timezone())
        elif tz is UTC or tz == "UTC":
            dt = datetime.datetime.now(UTC)
        else:
            dt = datetime.datetime.now(UTC)
            tz = pendulum._safe_timezone(tz)
            dt = dt.astimezone(tz)

        return cls(
            dt.year,
            dt.month,
            dt.day,
            dt.hour,
            dt.minute,
            dt.second,
            dt.microsecond,
            tzinfo=dt.tzinfo,
            fold=dt.fold,
        )

    @classmethod
    def utcnow(cls) -> Self:
        """
        Get a DateTime instance for the current date and time in UTC.
        """
        return cls.now(UTC)

    @classmethod
    def today(cls) -> Self:
        return cls.now()

    @classmethod
    def strptime(cls, time: str, fmt: str) -> Self:
        return cls.instance(datetime.datetime.strptime(time, fmt))

    # Getters/Setters

    def set(
        self,
        year: int | None = None,
        month: int | None = None,
        day: int | None = None,
        hour: int | None = None,
        minute: int | None = None,
        second: int | None = None,
        microsecond: int | None = None,
        tz: str | float | Timezone | FixedTimezone | datetime.tzinfo | None = None,
    ) -> Self:
        if year is None:
            year = self.year
        if month is None:
            month = self.month
        if day is None:
            day = self.day
        if hour is None:
            hour = self.hour
        if minute is None:
            minute = self.minute
        if second is None:
            second = self.second
        if microsecond is None:
            microsecond = self.microsecond
        if tz is None:
            tz = self.tz

        return self.__class__.create(
            year, month, day, hour, minute, second, microsecond, tz=tz, fold=self.fold
        )

    @property
    def float_timestamp(self) -> float:
        return self.timestamp()

    @property
    def int_timestamp(self) -> int:
        # Workaround needed to avoid inaccuracy
        # for far into the future datetimes
        dt = datetime.datetime(
            self.year,
            self.month,
            self.day,
            self.hour,
            self.minute,
            self.second,
            self.microsecond,
            tzinfo=self.tzinfo,
            fold=self.fold,
        )

        delta = dt - self._EPOCH

        return delta.days * SECONDS_PER_DAY + delta.seconds

    @property
    def offset(self) -> int | None:
        return self.get_offset()

    @property
    def offset_hours(self) -> float | None:
        offset = self.get_offset()

        if offset is None:
            return None

        return offset / SECONDS_PER_MINUTE / MINUTES_PER_HOUR

    @property
    def timezone(self) -> Timezone | FixedTimezone | None:
        if not isinstance(self.tzinfo, (Timezone, FixedTimezone)):
            return None

        return self.tzinfo

    @property
    def tz(self) -> Timezone | FixedTimezone | None:
        return self.timezone

    @property
    def timezone_name(self) -> str | None:
        tz = self.timezone

        if tz is None:
            return None

        return tz.name

    @property
    def age(self) -> int:
        return self.date().diff(self.now(self.tz).date(), abs=False).in_years()

    def is_local(self) -> bool:
        return self.offset == self.in_timezone(pendulum.local_timezone()).offset

    def is_utc(self) -> bool:
        return self.offset == 0

    def is_dst(self) -> bool:
        return self.dst() != datetime.timedelta()

    def get_offset(self) -> int | None:
        utcoffset = self.utcoffset()
        if utcoffset is None:
            return None

        return int(utcoffset.total_seconds())

    def date(self) -> Date:
        return Date(self.year, self.month, self.day)

    def time(self) -> Time:
        return Time(self.hour, self.minute, self.second, self.microsecond)

    def naive(self) -> Self:
        """
        Return the DateTime without timezone information.
        """
        return self.__class__(
            self.year,
            self.month,
            self.day,
            self.hour,
            self.minute,
            self.second,
            self.microsecond,
        )

    def on(self, year: int, month: int, day: int) -> Self:
        """
        Returns a new instance with the current date set to a different date.
        """
        return self.set(year=int(year), month=int(month), day=int(day))

    def at(
        self, hour: int, minute: int = 0, second: int = 0, microsecond: int = 0
    ) -> Self:
        """
        Returns a new instance with the current time to a different time.
        """
        return self.set(
            hour=hour, minute=minute, second=second, microsecond=microsecond
        )

    def in_timezone(self, tz: str | Timezone | FixedTimezone) -> Self:
        """
        Set the instance's timezone from a string or object.
        """
        tz = pendulum._safe_timezone(tz)

        dt = self
        if not self.timezone:
            dt = dt.replace(fold=1)

        return tz.convert(dt)

    def in_tz(self, tz: str | Timezone | FixedTimezone) -> Self:
        """
        Set the instance's timezone from a string or object.
        """
        return self.in_timezone(tz)

    # STRING FORMATTING

    def to_time_string(self) -> str:
        """
        Format the instance as time.
        """
        return self.format("HH:mm:ss")

    def to_datetime_string(self) -> str:
        """
        Format the instance as date and time.
        """
        return self.format("YYYY-MM-DD HH:mm:ss")

    def to_day_datetime_string(self) -> str:
        """
        Format the instance as day, date and time (in english).
        """
        return self.format("ddd, MMM D, YYYY h:mm A", locale="en")

    def to_atom_string(self) -> str:
        """
        Format the instance as ATOM.
        """
        return self._to_string("atom")

    def to_cookie_string(self) -> str:
        """
        Format the instance as COOKIE.
        """
        return self._to_string("cookie", locale="en")

    def to_iso8601_string(self) -> str:
        """
        Format the instance as ISO 8601.
        """
        string = self._to_string("iso8601")

        if self.tz and self.tz.name == "UTC":
            string = string.replace("+00:00", "Z")

        return string

    def to_rfc822_string(self) -> str:
        """
        Format the instance as RFC 822.
        """
        return self._to_string("rfc822")

    def to_rfc850_string(self) -> str:
        """
        Format the instance as RFC 850.
        """
        return self._to_string("rfc850")

    def to_rfc1036_string(self) -> str:
        """
        Format the instance as RFC 1036.
        """
        return self._to_string("rfc1036")

    def to_rfc1123_string(self) -> str:
        """
        Format the instance as RFC 1123.
        """
        return self._to_string("rfc1123")

    def to_rfc2822_string(self) -> str:
        """
        Format the instance as RFC 2822.
        """
        return self._to_string("rfc2822")

    def to_rfc3339_string(self) -> str:
        """
        Format the instance as RFC 3339.
        """
        return self._to_string("rfc3339")

    def to_rss_string(self) -> str:
        """
        Format the instance as RSS.
        """
        return self._to_string("rss")

    def to_w3c_string(self) -> str:
        """
        Format the instance as W3C.
        """
        return self._to_string("w3c")

    def _to_string(self, fmt: str, locale: str | None = None) -> str:
        """
        Format the instance to a common string format.
        """
        if fmt not in self._FORMATS:
            raise ValueError(f"Format [{fmt}] is not supported")

        fmt_value = self._FORMATS[fmt]
        if callable(fmt_value):
            return fmt_value(self)

        return self.format(fmt_value, locale=locale)

    def __str__(self) -> str:
        return self.isoformat(" ")

    def __repr__(self) -> str:
        us = ""
        if self.microsecond:
            us = f", {self.microsecond}"

        repr_ = "{klass}({year}, {month}, {day}, {hour}, {minute}, {second}{us}"

        if self.tzinfo is not None:
            repr_ += ", tzinfo={tzinfo}"

        repr_ += ")"

        return repr_.format(
            klass=self.__class__.__name__,
            year=self.year,
            month=self.month,
            day=self.day,
            hour=self.hour,
            minute=self.minute,
            second=self.second,
            us=us,
            tzinfo=repr(self.tzinfo),
        )

    # Comparisons
    def closest(self, *dts: datetime.datetime) -> Self:  # type: ignore[override]
        """
        Get the closest date to the instance.
        """
        pdts = [self.instance(x) for x in dts]

        return min((abs(self - dt), dt) for dt in pdts)[1]

    def farthest(self, *dts: datetime.datetime) -> Self:  # type: ignore[override]
        """
        Get the farthest date from the instance.
        """
        pdts = [self.instance(x) for x in dts]

        return max((abs(self - dt), dt) for dt in pdts)[1]

    def is_future(self) -> bool:
        """
        Determines if the instance is in the future, ie. greater than now.
        """
        return self > self.now(self.timezone)

    def is_past(self) -> bool:
        """
        Determines if the instance is in the past, ie. less than now.
        """
        return self < self.now(self.timezone)

    def is_long_year(self) -> bool:
        """
        Determines if the instance is a long year

        See link `https://en.wikipedia.org/wiki/ISO_8601#Week_dates`_
        """
        return (
            DateTime.create(self.year, 12, 28, 0, 0, 0, tz=self.tz).isocalendar()[1]
            == 53
        )

    def is_same_day(self, dt: datetime.datetime) -> bool:  # type: ignore[override]
        """
        Checks if the passed in date is the same day
        as the instance current day.
        """
        dt = self.instance(dt)

        return self.to_date_string() == dt.to_date_string()

    def is_anniversary(  # type: ignore[override]
        self, dt: datetime.datetime | None = None
    ) -> bool:
        """
        Check if its the anniversary.
        Compares the date/month values of the two dates.
        """
        if dt is None:
            dt = self.now(self.tz)

        instance = self.instance(dt)

        return (self.month, self.day) == (instance.month, instance.day)

    # ADDITIONS AND SUBSTRACTIONS

    def add(
        self,
        years: int = 0,
        months: int = 0,
        weeks: int = 0,
        days: int = 0,
        hours: int = 0,
        minutes: int = 0,
        seconds: float = 0,
        microseconds: int = 0,
    ) -> Self:
        """
        Add a duration to the instance.

        If we're adding units of variable length (i.e., years, months),
        move forward from current time, otherwise move forward from utc, for accuracy
        when moving across DST boundaries.
        """
        units_of_variable_length = any([years, months, weeks, days])

        current_dt = datetime.datetime(
            self.year,
            self.month,
            self.day,
            self.hour,
            self.minute,
            self.second,
            self.microsecond,
        )
        if not units_of_variable_length:
            offset = self.utcoffset()
            if offset:
                current_dt = current_dt - offset

        dt = add_duration(
            current_dt,
            years=years,
            months=months,
            weeks=weeks,
            days=days,
            hours=hours,
            minutes=minutes,
            seconds=seconds,
            microseconds=microseconds,
        )

        if units_of_variable_length or self.tz is None:
            return self.__class__.create(
                dt.year,
                dt.month,
                dt.day,
                dt.hour,
                dt.minute,
                dt.second,
                dt.microsecond,
                tz=self.tz,
            )

        dt = datetime.datetime(
            dt.year,
            dt.month,
            dt.day,
            dt.hour,
            dt.minute,
            dt.second,
            dt.microsecond,
            tzinfo=UTC,
        )

        dt = self.tz.convert(dt)

        return self.__class__(
            dt.year,
            dt.month,
            dt.day,
            dt.hour,
            dt.minute,
            dt.second,
            dt.microsecond,
            tzinfo=self.tz,
            fold=dt.fold,
        )

    def subtract(
        self,
        years: int = 0,
        months: int = 0,
        weeks: int = 0,
        days: int = 0,
        hours: int = 0,
        minutes: int = 0,
        seconds: float = 0,
        microseconds: int = 0,
    ) -> Self:
        """
        Remove duration from the instance.
        """
        return self.add(
            years=-years,
            months=-months,
            weeks=-weeks,
            days=-days,
            hours=-hours,
            minutes=-minutes,
            seconds=-seconds,
            microseconds=-microseconds,
        )

    # Adding a final underscore to the method name
    # to avoid errors for PyPy which already defines
    # a _add_timedelta method
    def _add_timedelta_(self, delta: datetime.timedelta) -> Self:
        """
        Add timedelta duration to the instance.
        """
        if isinstance(delta, pendulum.Interval):
            return self.add(
                years=delta.years,
                months=delta.months,
                weeks=delta.weeks,
                days=delta.remaining_days,
                hours=delta.hours,
                minutes=delta.minutes,
                seconds=delta.remaining_seconds,
                microseconds=delta.microseconds,
            )
        elif isinstance(delta, pendulum.Duration):
            return self.add(**delta._signature)  # type: ignore[attr-defined]

        return self.add(seconds=delta.total_seconds())

    def _subtract_timedelta(self, delta: datetime.timedelta) -> Self:
        """
        Remove timedelta duration from the instance.
        """
        if isinstance(delta, pendulum.Duration):
            return self.subtract(
                years=delta.years, months=delta.months, seconds=delta._total
            )

        return self.subtract(seconds=delta.total_seconds())

    # DIFFERENCES

    def diff(  # type: ignore[override]
        self, dt: datetime.datetime | None = None, abs: bool = True
    ) -> Interval[datetime.datetime]:
        """
        Returns the difference between two DateTime objects represented as an Interval.
        """
        if dt is None:
            dt = self.now(self.tz)

        return Interval(self, dt, absolute=abs)

    def diff_for_humans(  # type: ignore[override]
        self,
        other: DateTime | None = None,
        absolute: bool = False,
        locale: str | None = None,
    ) -> str:
        """
        Get the difference in a human readable format in the current locale.

        When comparing a value in the past to default now:
        1 day ago
        5 months ago

        When comparing a value in the future to default now:
        1 day from now
        5 months from now

        When comparing a value in the past to another value:
        1 day before
        5 months before

        When comparing a value in the future to another value:
        1 day after
        5 months after
        """
        is_now = other is None

        if is_now:
            other = self.now()

        diff = self.diff(other)

        return pendulum.format_diff(diff, is_now, absolute, locale)

    # Modifiers
    def start_of(self, unit: str) -> Self:
        """
        Returns a copy of the instance with the time reset
        with the following rules:

        * second: microsecond set to 0
        * minute: second and microsecond set to 0
        * hour: minute, second and microsecond set to 0
        * day: time to 00:00:00
        * week: date to first day of the week and time to 00:00:00
        * month: date to first day of the month and time to 00:00:00
        * year: date to first day of the year and time to 00:00:00
        * decade: date to first day of the decade and time to 00:00:00
        * century: date to first day of century and time to 00:00:00
        """
        if unit not in self._MODIFIERS_VALID_UNITS:
            raise ValueError(f'Invalid unit "{unit}" for start_of()')

        return cast("Self", getattr(self, f"_start_of_{unit}")())

    def end_of(self, unit: str) -> Self:
        """
        Returns a copy of the instance with the time reset
        with the following rules:

        * second: microsecond set to 999999
        * minute: second set to 59 and microsecond set to 999999
        * hour: minute and second set to 59 and microsecond set to 999999
        * day: time to 23:59:59.999999
        * week: date to last day of the week and time to 23:59:59.999999
        * month: date to last day of the month and time to 23:59:59.999999
        * year: date to last day of the year and time to 23:59:59.999999
        * decade: date to last day of the decade and time to 23:59:59.999999
        * century: date to last day of century and time to 23:59:59.999999
        """
        if unit not in self._MODIFIERS_VALID_UNITS:
            raise ValueError(f'Invalid unit "{unit}" for end_of()')

        return cast("Self", getattr(self, f"_end_of_{unit}")())

    def _start_of_second(self) -> Self:
        """
        Reset microseconds to 0.
        """
        return self.set(microsecond=0)

    def _end_of_second(self) -> Self:
        """
        Set microseconds to 999999.
        """
        return self.set(microsecond=999999)

    def _start_of_minute(self) -> Self:
        """
        Reset seconds and microseconds to 0.
        """
        return self.set(second=0, microsecond=0)

    def _end_of_minute(self) -> Self:
        """
        Set seconds to 59 and microseconds to 999999.
        """
        return self.set(second=59, microsecond=999999)

    def _start_of_hour(self) -> Self:
        """
        Reset minutes, seconds and microseconds to 0.
        """
        return self.set(minute=0, second=0, microsecond=0)

    def _end_of_hour(self) -> Self:
        """
        Set minutes and seconds to 59 and microseconds to 999999.
        """
        return self.set(minute=59, second=59, microsecond=999999)

    def _start_of_day(self) -> Self:
        """
        Reset the time to 00:00:00.
        """
        return self.at(0, 0, 0, 0)

    def _end_of_day(self) -> Self:
        """
        Reset the time to 23:59:59.999999.
        """
        return self.at(23, 59, 59, 999999)

    def _start_of_month(self) -> Self:
        """
        Reset the date to the first day of the month and the time to 00:00:00.
        """
        return self.set(self.year, self.month, 1, 0, 0, 0, 0)

    def _end_of_month(self) -> Self:
        """
        Reset the date to the last day of the month
        and the time to 23:59:59.999999.
        """
        return self.set(self.year, self.month, self.days_in_month, 23, 59, 59, 999999)

    def _start_of_year(self) -> Self:
        """
        Reset the date to the first day of the year and the time to 00:00:00.
        """
        return self.set(self.year, 1, 1, 0, 0, 0, 0)

    def _end_of_year(self) -> Self:
        """
        Reset the date to the last day of the year
        and the time to 23:59:59.999999.
        """
        return self.set(self.year, 12, 31, 23, 59, 59, 999999)

    def _start_of_decade(self) -> Self:
        """
        Reset the date to the first day of the decade
        and the time to 00:00:00.
        """
        year = self.year - self.year % YEARS_PER_DECADE
        return self.set(year, 1, 1, 0, 0, 0, 0)

    def _end_of_decade(self) -> Self:
        """
        Reset the date to the last day of the decade
        and the time to 23:59:59.999999.
        """
        year = self.year - self.year % YEARS_PER_DECADE + YEARS_PER_DECADE - 1

        return self.set(year, 12, 31, 23, 59, 59, 999999)

    def _start_of_century(self) -> Self:
        """
        Reset the date to the first day of the century
        and the time to 00:00:00.
        """
        year = self.year - 1 - (self.year - 1) % YEARS_PER_CENTURY + 1

        return self.set(year, 1, 1, 0, 0, 0, 0)

    def _end_of_century(self) -> Self:
        """
        Reset the date to the last day of the century
        and the time to 23:59:59.999999.
        """
        year = self.year - 1 - (self.year - 1) % YEARS_PER_CENTURY + YEARS_PER_CENTURY

        return self.set(year, 12, 31, 23, 59, 59, 999999)

    def _start_of_week(self) -> Self:
        """
        Reset the date to the first day of the week
        and the time to 00:00:00.
        """
        dt = self

        if self.day_of_week != pendulum._WEEK_STARTS_AT:
            dt = self.previous(pendulum._WEEK_STARTS_AT)

        return dt.start_of("day")

    def _end_of_week(self) -> Self:
        """
        Reset the date to the last day of the week
        and the time to 23:59:59.
        """
        dt = self

        if self.day_of_week != pendulum._WEEK_ENDS_AT:
            dt = self.next(pendulum._WEEK_ENDS_AT)

        return dt.end_of("day")

    def next(self, day_of_week: WeekDay | None = None, keep_time: bool = False) -> Self:
        """
        Modify to the next occurrence of a given day of the week.
        If no day_of_week is provided, modify to the next occurrence
        of the current day of the week.  Use the supplied consts
        to indicate the desired day_of_week, ex. DateTime.MONDAY.
        """
        if day_of_week is None:
            day_of_week = self.day_of_week

        if day_of_week < WeekDay.MONDAY or day_of_week > WeekDay.SUNDAY:
            raise ValueError("Invalid day of week")

        dt = self if keep_time else self.start_of("day")

        dt = dt.add(days=1)
        while dt.day_of_week != day_of_week:
            dt = dt.add(days=1)

        return dt

    def previous(
        self, day_of_week: WeekDay | None = None, keep_time: bool = False
    ) -> Self:
        """
        Modify to the previous occurrence of a given day of the week.
        If no day_of_week is provided, modify to the previous occurrence
        of the current day of the week.  Use the supplied consts
        to indicate the desired day_of_week, ex. DateTime.MONDAY.
        """
        if day_of_week is None:
            day_of_week = self.day_of_week

        if day_of_week < WeekDay.MONDAY or day_of_week > WeekDay.SUNDAY:
            raise ValueError("Invalid day of week")

        dt = self if keep_time else self.start_of("day")

        dt = dt.subtract(days=1)
        while dt.day_of_week != day_of_week:
            dt = dt.subtract(days=1)

        return dt

    def first_of(self, unit: str, day_of_week: WeekDay | None = None) -> Self:
        """
        Returns an instance set to the first occurrence
        of a given day of the week in the current unit.
        If no day_of_week is provided, modify to the first day of the unit.
        Use the supplied consts to indicate the desired day_of_week,
        ex. DateTime.MONDAY.

        Supported units are month, quarter and year.
        """
        if unit not in ["month", "quarter", "year"]:
            raise ValueError(f'Invalid unit "{unit}" for first_of()')

        return cast("Self", getattr(self, f"_first_of_{unit}")(day_of_week))

    def last_of(self, unit: str, day_of_week: WeekDay | None = None) -> Self:
        """
        Returns an instance set to the last occurrence
        of a given day of the week in the current unit.
        If no day_of_week is provided, modify to the last day of the unit.
        Use the supplied consts to indicate the desired day_of_week,
        ex. DateTime.MONDAY.

        Supported units are month, quarter and year.
        """
        if unit not in ["month", "quarter", "year"]:
            raise ValueError(f'Invalid unit "{unit}" for first_of()')

        return cast("Self", getattr(self, f"_last_of_{unit}")(day_of_week))

    def nth_of(self, unit: str, nth: int, day_of_week: WeekDay) -> Self:
        """
        Returns a new instance set to the given occurrence
        of a given day of the week in the current unit.
        If the calculated occurrence is outside the scope of the current unit,
        then raise an error. Use the supplied consts
        to indicate the desired day_of_week, ex. DateTime.MONDAY.

        Supported units are month, quarter and year.
        """
        if unit not in ["month", "quarter", "year"]:
            raise ValueError(f'Invalid unit "{unit}" for first_of()')

        dt = cast("Optional[Self]", getattr(self, f"_nth_of_{unit}")(nth, day_of_week))
        if not dt:
            raise PendulumException(
                f"Unable to find occurrence {nth}"
                f" of {WeekDay(day_of_week).name.capitalize()} in {unit}"
            )

        return dt

    def _first_of_month(self, day_of_week: WeekDay | None = None) -> Self:
        """
        Modify to the first occurrence of a given day of the week
        in the current month. If no day_of_week is provided,
        modify to the first day of the month. Use the supplied consts
        to indicate the desired day_of_week, ex. DateTime.MONDAY.
        """
        dt = self.start_of("day")

        if day_of_week is None:
            return dt.set(day=1)

        month = calendar.monthcalendar(dt.year, dt.month)

        calendar_day = day_of_week

        if month[0][calendar_day] > 0:
            day_of_month = month[0][calendar_day]
        else:
            day_of_month = month[1][calendar_day]

        return dt.set(day=day_of_month)

    def _last_of_month(self, day_of_week: WeekDay | None = None) -> Self:
        """
        Modify to the last occurrence of a given day of the week
        in the current month. If no day_of_week is provided,
        modify to the last day of the month. Use the supplied consts
        to indicate the desired day_of_week, ex. DateTime.MONDAY.
        """
        dt = self.start_of("day")

        if day_of_week is None:
            return dt.set(day=self.days_in_month)

        month = calendar.monthcalendar(dt.year, dt.month)

        calendar_day = day_of_week

        if month[-1][calendar_day] > 0:
            day_of_month = month[-1][calendar_day]
        else:
            day_of_month = month[-2][calendar_day]

        return dt.set(day=day_of_month)

    def _nth_of_month(
        self, nth: int, day_of_week: WeekDay | None = None
    ) -> Self | None:
        """
        Modify to the given occurrence of a given day of the week
        in the current month. If the calculated occurrence is outside,
        the scope of the current month, then return False and no
        modifications are made. Use the supplied consts
        to indicate the desired day_of_week, ex. DateTime.MONDAY.
        """
        if nth == 1:
            return self.first_of("month", day_of_week)

        dt = self.first_of("month")
        check = dt.format("%Y-%M")
        for _ in range(nth - (1 if dt.day_of_week == day_of_week else 0)):
            dt = dt.next(day_of_week)

        if dt.format("%Y-%M") == check:
            return self.set(day=dt.day).start_of("day")

        return None

    def _first_of_quarter(self, day_of_week: WeekDay | None = None) -> Self:
        """
        Modify to the first occurrence of a given day of the week
        in the current quarter. If no day_of_week is provided,
        modify to the first day of the quarter. Use the supplied consts
        to indicate the desired day_of_week, ex. DateTime.MONDAY.
        """
        return self.on(self.year, self.quarter * 3 - 2, 1).first_of(
            "month", day_of_week
        )

    def _last_of_quarter(self, day_of_week: WeekDay | None = None) -> Self:
        """
        Modify to the last occurrence of a given day of the week
        in the current quarter. If no day_of_week is provided,
        modify to the last day of the quarter. Use the supplied consts
        to indicate the desired day_of_week, ex. DateTime.MONDAY.
        """
        return self.on(self.year, self.quarter * 3, 1).last_of("month", day_of_week)

    def _nth_of_quarter(
        self, nth: int, day_of_week: WeekDay | None = None
    ) -> Self | None:
        """
        Modify to the given occurrence of a given day of the week
        in the current quarter. If the calculated occurrence is outside,
        the scope of the current quarter, then return False and no
        modifications are made. Use the supplied consts
        to indicate the desired day_of_week, ex. DateTime.MONDAY.
        """
        if nth == 1:
            return self.first_of("quarter", day_of_week)

        dt = self.set(day=1, month=self.quarter * 3)
        last_month = dt.month
        year = dt.year
        dt = dt.first_of("quarter")
        for _ in range(nth - (1 if dt.day_of_week == day_of_week else 0)):
            dt = dt.next(day_of_week)

        if last_month < dt.month or year != dt.year:
            return None

        return self.on(self.year, dt.month, dt.day).start_of("day")

    def _first_of_year(self, day_of_week: WeekDay | None = None) -> Self:
        """
        Modify to the first occurrence of a given day of the week
        in the current year. If no day_of_week is provided,
        modify to the first day of the year. Use the supplied consts
        to indicate the desired day_of_week, ex. DateTime.MONDAY.
        """
        return self.set(month=1).first_of("month", day_of_week)

    def _last_of_year(self, day_of_week: WeekDay | None = None) -> Self:
        """
        Modify to the last occurrence of a given day of the week
        in the current year. If no day_of_week is provided,
        modify to the last day of the year. Use the supplied consts
        to indicate the desired day_of_week, ex. DateTime.MONDAY.
        """
        return self.set(month=MONTHS_PER_YEAR).last_of("month", day_of_week)

    def _nth_of_year(self, nth: int, day_of_week: WeekDay | None = None) -> Self | None:
        """
        Modify to the given occurrence of a given day of the week
        in the current year. If the calculated occurrence is outside,
        the scope of the current year, then return False and no
        modifications are made. Use the supplied consts
        to indicate the desired day_of_week, ex. DateTime.MONDAY.
        """
        if nth == 1:
            return self.first_of("year", day_of_week)

        dt = self.first_of("year")
        year = dt.year
        for _ in range(nth - (1 if dt.day_of_week == day_of_week else 0)):
            dt = dt.next(day_of_week)

        if year != dt.year:
            return None

        return self.on(self.year, dt.month, dt.day).start_of("day")

    def average(  # type: ignore[override]
        self, dt: datetime.datetime | None = None
    ) -> Self:
        """
        Modify the current instance to the average
        of a given instance (default now) and the current instance.
        """
        if dt is None:
            dt = self.now(self.tz)

        diff = self.diff(dt, False)
        return self.add(
            microseconds=(diff.in_seconds() * 1000000 + diff.microseconds) // 2
        )

    @overload  # type: ignore[override]
    def __sub__(self, other: datetime.timedelta) -> Self: ...

    @overload
    def __sub__(self, other: DateTime) -> Interval[datetime.datetime]: ...

    def __sub__(
        self, other: datetime.datetime | datetime.timedelta
    ) -> Self | Interval[datetime.datetime]:
        if isinstance(other, datetime.timedelta):
            return self._subtract_timedelta(other)

        if not isinstance(other, datetime.datetime):
            return NotImplemented

        if not isinstance(other, self.__class__):
            if other.tzinfo is None:
                other = pendulum.naive(
                    other.year,
                    other.month,
                    other.day,
                    other.hour,
                    other.minute,
                    other.second,
                    other.microsecond,
                )
            else:
                other = self.instance(other)

        return other.diff(self, False)

    def __rsub__(self, other: datetime.datetime) -> Interval[datetime.datetime]:
        if not isinstance(other, datetime.datetime):
            return NotImplemented

        if not isinstance(other, self.__class__):
            if other.tzinfo is None:
                other = pendulum.naive(
                    other.year,
                    other.month,
                    other.day,
                    other.hour,
                    other.minute,
                    other.second,
                    other.microsecond,
                )
            else:
                other = self.instance(other)

        return self.diff(other, False)

    def __add__(self, other: datetime.timedelta) -> Self:
        if not isinstance(other, datetime.timedelta):
            return NotImplemented

        caller = traceback.extract_stack(limit=2)[0].name
        if caller == "astimezone":
            return super().__add__(other)

        return self._add_timedelta_(other)

    def __radd__(self, other: datetime.timedelta) -> Self:
        return self.__add__(other)

    # Native methods override

    @classmethod
    def fromtimestamp(cls, t: float, tz: datetime.tzinfo | None = None) -> Self:
        tzinfo = pendulum._safe_timezone(tz)

        return cls.instance(datetime.datetime.fromtimestamp(t, tz=tzinfo), tz=tzinfo)

    @classmethod
    def utcfromtimestamp(cls, t: float) -> Self:
        return cls.instance(datetime.datetime.utcfromtimestamp(t), tz=None)

    @classmethod
    def fromordinal(cls, n: int) -> Self:
        return cls.instance(datetime.datetime.fromordinal(n), tz=None)

    @classmethod
    def combine(
        cls,
        date: datetime.date,
        time: datetime.time,
        tzinfo: datetime.tzinfo | None = None,
    ) -> Self:
        return cls.instance(datetime.datetime.combine(date, time), tz=tzinfo)

    def astimezone(self, tz: datetime.tzinfo | None = None) -> Self:
        dt = super().astimezone(tz)

        return self.__class__(
            dt.year,
            dt.month,
            dt.day,
            dt.hour,
            dt.minute,
            dt.second,
            dt.microsecond,
            fold=dt.fold,
            tzinfo=dt.tzinfo,
        )

    def replace(
        self,
        year: SupportsIndex | None = None,
        month: SupportsIndex | None = None,
        day: SupportsIndex | None = None,
        hour: SupportsIndex | None = None,
        minute: SupportsIndex | None = None,
        second: SupportsIndex | None = None,
        microsecond: SupportsIndex | None = None,
        tzinfo: bool | datetime.tzinfo | Literal[True] | None = True,
        fold: int | None = None,
    ) -> Self:
        if year is None:
            year = self.year
        if month is None:
            month = self.month
        if day is None:
            day = self.day
        if hour is None:
            hour = self.hour
        if minute is None:
            minute = self.minute
        if second is None:
            second = self.second
        if microsecond is None:
            microsecond = self.microsecond
        if tzinfo is True:
            tzinfo = self.tzinfo
        if fold is None:
            fold = self.fold

        if tzinfo is not None:
            tzinfo = pendulum._safe_timezone(tzinfo)

        return self.__class__.create(
            year,
            month,
            day,
            hour,
            minute,
            second,
            microsecond,
            tz=tzinfo,
            fold=fold,
        )

    def __getnewargs__(self) -> tuple[Self]:
        return (self,)

    def _getstate(
        self, protocol: SupportsIndex = 3
    ) -> tuple[int, int, int, int, int, int, int, datetime.tzinfo | None]:
        return (
            self.year,
            self.month,
            self.day,
            self.hour,
            self.minute,
            self.second,
            self.microsecond,
            self.tzinfo,
        )

    def __reduce__(
        self,
    ) -> tuple[
        type[Self],
        tuple[int, int, int, int, int, int, int, datetime.tzinfo | None],
    ]:
        return self.__reduce_ex__(2)

    def __reduce_ex__(
        self, protocol: SupportsIndex
    ) -> tuple[
        type[Self],
        tuple[int, int, int, int, int, int, int, datetime.tzinfo | None],
    ]:
        return self.__class__, self._getstate(protocol)

    def __deepcopy__(self, _: dict[int, Self]) -> Self:
        return self.__class__(
            self.year,
            self.month,
            self.day,
            self.hour,
            self.minute,
            self.second,
            self.microsecond,
            tzinfo=self.tz,
            fold=self.fold,
        )

    def _cmp(self, other: datetime.datetime, **kwargs: Any) -> int:
        # Fix for pypy which compares using this method
        # which would lead to infinite recursion if we didn't override
        dt = datetime.datetime(
            self.year,
            self.month,
            self.day,
            self.hour,
            self.minute,
            self.second,
            self.microsecond,
            tzinfo=self.tz,
            fold=self.fold,
        )

        return 0 if dt == other else 1 if dt > other else -1


DateTime.min = DateTime(1, 1, 1, 0, 0, tzinfo=UTC)
DateTime.max = DateTime(9999, 12, 31, 23, 59, 59, 999999, tzinfo=UTC)
DateTime.EPOCH = DateTime(1970, 1, 1, tzinfo=UTC)


# === src/pendulum/helpers.py ===
from __future__ import annotations

import os
import struct

from datetime import date
from datetime import datetime
from datetime import timedelta
from math import copysign
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import overload

import pendulum

from pendulum.constants import DAYS_PER_MONTHS
from pendulum.day import WeekDay
from pendulum.formatting.difference_formatter import DifferenceFormatter
from pendulum.locales.locale import Locale


if TYPE_CHECKING:
    # Prevent import cycles
    from pendulum.duration import Duration

with_extensions = os.getenv("PENDULUM_EXTENSIONS", "1") == "1"

_DT = TypeVar("_DT", bound=datetime)
_D = TypeVar("_D", bound=date)

try:
    if not with_extensions or struct.calcsize("P") == 4:
        raise ImportError()

    from pendulum._pendulum import PreciseDiff
    from pendulum._pendulum import days_in_year
    from pendulum._pendulum import is_leap
    from pendulum._pendulum import is_long_year
    from pendulum._pendulum import local_time
    from pendulum._pendulum import precise_diff
    from pendulum._pendulum import week_day
except ImportError:
    from pendulum._helpers import PreciseDiff  # type: ignore[assignment]
    from pendulum._helpers import days_in_year
    from pendulum._helpers import is_leap
    from pendulum._helpers import is_long_year
    from pendulum._helpers import local_time
    from pendulum._helpers import precise_diff  # type: ignore[assignment]
    from pendulum._helpers import week_day

difference_formatter = DifferenceFormatter()


@overload
def add_duration(
    dt: _DT,
    years: int = 0,
    months: int = 0,
    weeks: int = 0,
    days: int = 0,
    hours: int = 0,
    minutes: int = 0,
    seconds: float = 0,
    microseconds: int = 0,
) -> _DT: ...


@overload
def add_duration(
    dt: _D,
    years: int = 0,
    months: int = 0,
    weeks: int = 0,
    days: int = 0,
) -> _D:
    pass


def add_duration(
    dt: date | datetime,
    years: int = 0,
    months: int = 0,
    weeks: int = 0,
    days: int = 0,
    hours: int = 0,
    minutes: int = 0,
    seconds: float = 0,
    microseconds: int = 0,
) -> date | datetime:
    """
    Adds a duration to a date/datetime instance.
    """
    days += weeks * 7

    if (
        isinstance(dt, date)
        and not isinstance(dt, datetime)
        and any([hours, minutes, seconds, microseconds])
    ):
        raise RuntimeError("Time elements cannot be added to a date instance.")

    # Normalizing
    if abs(microseconds) > 999999:
        s = _sign(microseconds)
        div, mod = divmod(microseconds * s, 1000000)
        microseconds = mod * s
        seconds += div * s

    if abs(seconds) > 59:
        s = _sign(seconds)
        div, mod = divmod(seconds * s, 60)  # type: ignore[assignment]
        seconds = mod * s
        minutes += div * s

    if abs(minutes) > 59:
        s = _sign(minutes)
        div, mod = divmod(minutes * s, 60)
        minutes = mod * s
        hours += div * s

    if abs(hours) > 23:
        s = _sign(hours)
        div, mod = divmod(hours * s, 24)
        hours = mod * s
        days += div * s

    if abs(months) > 11:
        s = _sign(months)
        div, mod = divmod(months * s, 12)
        months = mod * s
        years += div * s

    year = dt.year + years
    month = dt.month

    if months:
        month += months
        if month > 12:
            year += 1
            month -= 12
        elif month < 1:
            year -= 1
            month += 12

    day = min(DAYS_PER_MONTHS[int(is_leap(year))][month], dt.day)

    dt = dt.replace(year=year, month=month, day=day)

    return dt + timedelta(
        days=days,
        hours=hours,
        minutes=minutes,
        seconds=seconds,
        microseconds=microseconds,
    )


def format_diff(
    diff: Duration,
    is_now: bool = True,
    absolute: bool = False,
    locale: str | None = None,
) -> str:
    if locale is None:
        locale = get_locale()

    return difference_formatter.format(diff, is_now, absolute, locale)


def _sign(x: float) -> int:
    return int(copysign(1, x))


# Global helpers


def locale(name: str) -> Locale:
    return Locale.load(name)


def set_locale(name: str) -> None:
    locale(name)

    pendulum._LOCALE = name


def get_locale() -> str:
    return pendulum._LOCALE


def week_starts_at(wday: WeekDay) -> None:
    if wday < WeekDay.MONDAY or wday > WeekDay.SUNDAY:
        raise ValueError("Invalid day of week")

    pendulum._WEEK_STARTS_AT = wday


def week_ends_at(wday: WeekDay) -> None:
    if wday < WeekDay.MONDAY or wday > WeekDay.SUNDAY:
        raise ValueError("Invalid day of week")

    pendulum._WEEK_ENDS_AT = wday


__all__ = [
    "PreciseDiff",
    "add_duration",
    "days_in_year",
    "format_diff",
    "get_locale",
    "is_leap",
    "is_long_year",
    "local_time",
    "locale",
    "precise_diff",
    "set_locale",
    "week_day",
    "week_ends_at",
    "week_starts_at",
]


# === src/pendulum/date.py ===
# The following is only needed because of Python 3.7
# mypy: no-warn-unused-ignores
from __future__ import annotations

import calendar
import math

from datetime import date
from datetime import datetime
from datetime import timedelta
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import NoReturn
from typing import cast
from typing import overload

import pendulum

from pendulum.constants import MONTHS_PER_YEAR
from pendulum.constants import YEARS_PER_CENTURY
from pendulum.constants import YEARS_PER_DECADE
from pendulum.day import WeekDay
from pendulum.exceptions import PendulumException
from pendulum.helpers import add_duration
from pendulum.interval import Interval
from pendulum.mixins.default import FormattableMixin


if TYPE_CHECKING:
    from typing_extensions import Self
    from typing_extensions import SupportsIndex


class Date(FormattableMixin, date):
    _MODIFIERS_VALID_UNITS: ClassVar[list[str]] = [
        "day",
        "week",
        "month",
        "year",
        "decade",
        "century",
    ]

    # Getters/Setters

    def set(
        self, year: int | None = None, month: int | None = None, day: int | None = None
    ) -> Self:
        return self.replace(year=year, month=month, day=day)

    @property
    def day_of_week(self) -> WeekDay:
        """
        Returns the day of the week (0-6).
        """
        return WeekDay(self.weekday())

    @property
    def day_of_year(self) -> int:
        """
        Returns the day of the year (1-366).
        """
        k = 1 if self.is_leap_year() else 2

        return (275 * self.month) // 9 - k * ((self.month + 9) // 12) + self.day - 30

    @property
    def week_of_year(self) -> int:
        return self.isocalendar()[1]

    @property
    def days_in_month(self) -> int:
        return calendar.monthrange(self.year, self.month)[1]

    @property
    def week_of_month(self) -> int:
        return math.ceil((self.day + self.first_of("month").isoweekday() - 1) / 7)

    @property
    def age(self) -> int:
        return self.diff(abs=False).in_years()

    @property
    def quarter(self) -> int:
        return math.ceil(self.month / 3)

    # String Formatting

    def to_date_string(self) -> str:
        """
        Format the instance as date.

        :rtype: str
        """
        return self.strftime("%Y-%m-%d")

    def to_formatted_date_string(self) -> str:
        """
        Format the instance as a readable date.

        :rtype: str
        """
        return self.strftime("%b %d, %Y")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.year}, {self.month}, {self.day})"

    # COMPARISONS

    def closest(self, dt1: date, dt2: date) -> Self:
        """
        Get the closest date from the instance.
        """
        dt1 = self.__class__(dt1.year, dt1.month, dt1.day)
        dt2 = self.__class__(dt2.year, dt2.month, dt2.day)

        if self.diff(dt1).in_seconds() < self.diff(dt2).in_seconds():
            return dt1

        return dt2

    def farthest(self, dt1: date, dt2: date) -> Self:
        """
        Get the farthest date from the instance.
        """
        dt1 = self.__class__(dt1.year, dt1.month, dt1.day)
        dt2 = self.__class__(dt2.year, dt2.month, dt2.day)

        if self.diff(dt1).in_seconds() > self.diff(dt2).in_seconds():
            return dt1

        return dt2

    def is_future(self) -> bool:
        """
        Determines if the instance is in the future, ie. greater than now.
        """
        return self > self.today()

    def is_past(self) -> bool:
        """
        Determines if the instance is in the past, ie. less than now.
        """
        return self < self.today()

    def is_leap_year(self) -> bool:
        """
        Determines if the instance is a leap year.
        """
        return calendar.isleap(self.year)

    def is_long_year(self) -> bool:
        """
        Determines if the instance is a long year

        See link `<https://en.wikipedia.org/wiki/ISO_8601#Week_dates>`_
        """
        return Date(self.year, 12, 28).isocalendar()[1] == 53

    def is_same_day(self, dt: date) -> bool:
        """
        Checks if the passed in date is the same day as the instance current day.
        """
        return self == dt

    def is_anniversary(self, dt: date | None = None) -> bool:
        """
        Check if it's the anniversary.

        Compares the date/month values of the two dates.
        """
        if dt is None:
            dt = self.__class__.today()

        instance = self.__class__(dt.year, dt.month, dt.day)

        return (self.month, self.day) == (instance.month, instance.day)

    # the additional method for checking if today is the anniversary day
    # the alias is provided to start using a new name and keep the backward
    # compatibility the old name can be completely replaced with the new in
    # one of the future versions
    is_birthday = is_anniversary

    # ADDITIONS AND SUBTRACTIONS

    def add(
        self, years: int = 0, months: int = 0, weeks: int = 0, days: int = 0
    ) -> Self:
        """
        Add duration to the instance.

        :param years: The number of years
        :param months: The number of months
        :param weeks: The number of weeks
        :param days: The number of days
        """
        dt = add_duration(
            date(self.year, self.month, self.day),
            years=years,
            months=months,
            weeks=weeks,
            days=days,
        )

        return self.__class__(dt.year, dt.month, dt.day)

    def subtract(
        self, years: int = 0, months: int = 0, weeks: int = 0, days: int = 0
    ) -> Self:
        """
        Remove duration from the instance.

        :param years: The number of years
        :param months: The number of months
        :param weeks: The number of weeks
        :param days: The number of days
        """
        return self.add(years=-years, months=-months, weeks=-weeks, days=-days)

    def _add_timedelta(self, delta: timedelta) -> Self:
        """
        Add timedelta duration to the instance.

        :param delta: The timedelta instance
        """
        if isinstance(delta, pendulum.Duration):
            return self.add(
                years=delta.years,
                months=delta.months,
                weeks=delta.weeks,
                days=delta.remaining_days,
            )

        return self.add(days=delta.days)

    def _subtract_timedelta(self, delta: timedelta) -> Self:
        """
        Remove timedelta duration from the instance.

        :param delta: The timedelta instance
        """
        if isinstance(delta, pendulum.Duration):
            return self.subtract(
                years=delta.years,
                months=delta.months,
                weeks=delta.weeks,
                days=delta.remaining_days,
            )

        return self.subtract(days=delta.days)

    def __add__(self, other: timedelta) -> Self:
        if not isinstance(other, timedelta):
            return NotImplemented

        return self._add_timedelta(other)

    @overload  # type: ignore[override]  # this is only needed because of Python 3.7
    def __sub__(self, __delta: timedelta) -> Self: ...

    @overload
    def __sub__(self, __dt: datetime) -> NoReturn: ...

    @overload
    def __sub__(self, __dt: Self) -> Interval[Date]: ...

    def __sub__(self, other: timedelta | date) -> Self | Interval[Date]:
        if isinstance(other, timedelta):
            return self._subtract_timedelta(other)

        if not isinstance(other, date):
            return NotImplemented

        dt = self.__class__(other.year, other.month, other.day)

        return dt.diff(self, False)

    # DIFFERENCES

    def diff(self, dt: date | None = None, abs: bool = True) -> Interval[Date]:
        """
        Returns the difference between two Date objects as an Interval.

        :param dt: The date to compare to (defaults to today)
        :param abs: Whether to return an absolute interval or not
        """
        if dt is None:
            dt = self.today()

        return Interval(self, Date(dt.year, dt.month, dt.day), absolute=abs)

    def diff_for_humans(
        self,
        other: date | None = None,
        absolute: bool = False,
        locale: str | None = None,
    ) -> str:
        """
        Get the difference in a human readable format in the current locale.

        When comparing a value in the past to default now:
        1 day ago
        5 months ago

        When comparing a value in the future to default now:
        1 day from now
        5 months from now

        When comparing a value in the past to another value:
        1 day before
        5 months before

        When comparing a value in the future to another value:
        1 day after
        5 months after

        :param other: The date to compare to (defaults to today)
        :param absolute: removes time difference modifiers ago, after, etc
        :param locale: The locale to use for localization
        """
        is_now = other is None

        if is_now:
            other = self.today()

        diff = self.diff(other)

        return pendulum.format_diff(diff, is_now, absolute, locale)

    # MODIFIERS

    def start_of(self, unit: str) -> Self:
        """
        Returns a copy of the instance with the time reset
        with the following rules:

        * day: time to 00:00:00
        * week: date to first day of the week and time to 00:00:00
        * month: date to first day of the month and time to 00:00:00
        * year: date to first day of the year and time to 00:00:00
        * decade: date to first day of the decade and time to 00:00:00
        * century: date to first day of century and time to 00:00:00

        :param unit: The unit to reset to
        """
        if unit not in self._MODIFIERS_VALID_UNITS:
            raise ValueError(f'Invalid unit "{unit}" for start_of()')

        return cast("Self", getattr(self, f"_start_of_{unit}")())

    def end_of(self, unit: str) -> Self:
        """
        Returns a copy of the instance with the time reset
        with the following rules:

        * week: date to last day of the week
        * month: date to last day of the month
        * year: date to last day of the year
        * decade: date to last day of the decade
        * century: date to last day of century

        :param unit: The unit to reset to
        """
        if unit not in self._MODIFIERS_VALID_UNITS:
            raise ValueError(f'Invalid unit "{unit}" for end_of()')

        return cast("Self", getattr(self, f"_end_of_{unit}")())

    def _start_of_day(self) -> Self:
        """
        Compatibility method.
        """
        return self

    def _end_of_day(self) -> Self:
        """
        Compatibility method
        """
        return self

    def _start_of_month(self) -> Self:
        """
        Reset the date to the first day of the month.
        """
        return self.set(self.year, self.month, 1)

    def _end_of_month(self) -> Self:
        """
        Reset the date to the last day of the month.
        """
        return self.set(self.year, self.month, self.days_in_month)

    def _start_of_year(self) -> Self:
        """
        Reset the date to the first day of the year.
        """
        return self.set(self.year, 1, 1)

    def _end_of_year(self) -> Self:
        """
        Reset the date to the last day of the year.
        """
        return self.set(self.year, 12, 31)

    def _start_of_decade(self) -> Self:
        """
        Reset the date to the first day of the decade.
        """
        year = self.year - self.year % YEARS_PER_DECADE

        return self.set(year, 1, 1)

    def _end_of_decade(self) -> Self:
        """
        Reset the date to the last day of the decade.
        """
        year = self.year - self.year % YEARS_PER_DECADE + YEARS_PER_DECADE - 1

        return self.set(year, 12, 31)

    def _start_of_century(self) -> Self:
        """
        Reset the date to the first day of the century.
        """
        year = self.year - 1 - (self.year - 1) % YEARS_PER_CENTURY + 1

        return self.set(year, 1, 1)

    def _end_of_century(self) -> Self:
        """
        Reset the date to the last day of the century.
        """
        year = self.year - 1 - (self.year - 1) % YEARS_PER_CENTURY + YEARS_PER_CENTURY

        return self.set(year, 12, 31)

    def _start_of_week(self) -> Self:
        """
        Reset the date to the first day of the week.
        """
        dt = self

        if self.day_of_week != pendulum._WEEK_STARTS_AT:
            dt = self.previous(pendulum._WEEK_STARTS_AT)

        return dt.start_of("day")

    def _end_of_week(self) -> Self:
        """
        Reset the date to the last day of the week.
        """
        dt = self

        if self.day_of_week != pendulum._WEEK_ENDS_AT:
            dt = self.next(pendulum._WEEK_ENDS_AT)

        return dt.end_of("day")

    def next(self, day_of_week: WeekDay | None = None) -> Self:
        """
        Modify to the next occurrence of a given day of the week.
        If no day_of_week is provided, modify to the next occurrence
        of the current day of the week.  Use the supplied consts
        to indicate the desired day_of_week, ex. pendulum.MONDAY.

        :param day_of_week: The next day of week to reset to.
        """
        if day_of_week is None:
            day_of_week = self.day_of_week

        if day_of_week < WeekDay.MONDAY or day_of_week > WeekDay.SUNDAY:
            raise ValueError("Invalid day of week")

        dt = self.add(days=1)
        while dt.day_of_week != day_of_week:
            dt = dt.add(days=1)

        return dt

    def previous(self, day_of_week: WeekDay | None = None) -> Self:
        """
        Modify to the previous occurrence of a given day of the week.
        If no day_of_week is provided, modify to the previous occurrence
        of the current day of the week.  Use the supplied consts
        to indicate the desired day_of_week, ex. pendulum.MONDAY.

        :param day_of_week: The previous day of week to reset to.
        """
        if day_of_week is None:
            day_of_week = self.day_of_week

        if day_of_week < WeekDay.MONDAY or day_of_week > WeekDay.SUNDAY:
            raise ValueError("Invalid day of week")

        dt = self.subtract(days=1)
        while dt.day_of_week != day_of_week:
            dt = dt.subtract(days=1)

        return dt

    def first_of(self, unit: str, day_of_week: WeekDay | None = None) -> Self:
        """
        Returns an instance set to the first occurrence
        of a given day of the week in the current unit.
        If no day_of_week is provided, modify to the first day of the unit.
        Use the supplied consts to indicate the desired day_of_week,
        ex. pendulum.MONDAY.

        Supported units are month, quarter and year.

        :param unit: The unit to use
        :param day_of_week: The day of week to reset to.
        """
        if unit not in ["month", "quarter", "year"]:
            raise ValueError(f'Invalid unit "{unit}" for first_of()')

        return cast("Self", getattr(self, f"_first_of_{unit}")(day_of_week))

    def last_of(self, unit: str, day_of_week: WeekDay | None = None) -> Self:
        """
        Returns an instance set to the last occurrence
        of a given day of the week in the current unit.
        If no day_of_week is provided, modify to the last day of the unit.
        Use the supplied consts to indicate the desired day_of_week,
        ex. pendulum.MONDAY.

        Supported units are month, quarter and year.

        :param unit: The unit to use
        :param day_of_week: The day of week to reset to.
        """
        if unit not in ["month", "quarter", "year"]:
            raise ValueError(f'Invalid unit "{unit}" for first_of()')

        return cast("Self", getattr(self, f"_last_of_{unit}")(day_of_week))

    def nth_of(self, unit: str, nth: int, day_of_week: WeekDay) -> Self:
        """
        Returns a new instance set to the given occurrence
        of a given day of the week in the current unit.
        If the calculated occurrence is outside the scope of the current unit,
        then raise an error. Use the supplied consts
        to indicate the desired day_of_week, ex. pendulum.MONDAY.

        Supported units are month, quarter and year.

        :param unit: The unit to use
        :param nth: The occurrence to use
        :param day_of_week: The day of week to set to.
        """
        if unit not in ["month", "quarter", "year"]:
            raise ValueError(f'Invalid unit "{unit}" for first_of()')

        dt = cast("Self", getattr(self, f"_nth_of_{unit}")(nth, day_of_week))
        if not dt:
            raise PendulumException(
                f"Unable to find occurrence {nth}"
                f" of {WeekDay(day_of_week).name.capitalize()} in {unit}"
            )

        return dt

    def _first_of_month(self, day_of_week: WeekDay) -> Self:
        """
        Modify to the first occurrence of a given day of the week
        in the current month. If no day_of_week is provided,
        modify to the first day of the month. Use the supplied consts
        to indicate the desired day_of_week, ex. pendulum.MONDAY.

        :param day_of_week: The day of week to set to.
        """
        dt = self

        if day_of_week is None:
            return dt.set(day=1)

        month = calendar.monthcalendar(dt.year, dt.month)

        calendar_day = day_of_week

        if month[0][calendar_day] > 0:
            day_of_month = month[0][calendar_day]
        else:
            day_of_month = month[1][calendar_day]

        return dt.set(day=day_of_month)

    def _last_of_month(self, day_of_week: WeekDay | None = None) -> Self:
        """
        Modify to the last occurrence of a given day of the week
        in the current month. If no day_of_week is provided,
        modify to the last day of the month. Use the supplied consts
        to indicate the desired day_of_week, ex. pendulum.MONDAY.

        :param day_of_week: The day of week to set to.
        """
        dt = self

        if day_of_week is None:
            return dt.set(day=self.days_in_month)

        month = calendar.monthcalendar(dt.year, dt.month)

        calendar_day = day_of_week

        if month[-1][calendar_day] > 0:
            day_of_month = month[-1][calendar_day]
        else:
            day_of_month = month[-2][calendar_day]

        return dt.set(day=day_of_month)

    def _nth_of_month(self, nth: int, day_of_week: WeekDay) -> Self | None:
        """
        Modify to the given occurrence of a given day of the week
        in the current month. If the calculated occurrence is outside,
        the scope of the current month, then return False and no
        modifications are made. Use the supplied consts
        to indicate the desired day_of_week, ex. pendulum.MONDAY.
        """
        if nth == 1:
            return self.first_of("month", day_of_week)

        dt = self.first_of("month")
        check = dt.format("YYYY-MM")
        for _ in range(nth - (1 if dt.day_of_week == day_of_week else 0)):
            dt = dt.next(day_of_week)

        if dt.format("YYYY-MM") == check:
            return self.set(day=dt.day)

        return None

    def _first_of_quarter(self, day_of_week: WeekDay | None = None) -> Self:
        """
        Modify to the first occurrence of a given day of the week
        in the current quarter. If no day_of_week is provided,
        modify to the first day of the quarter. Use the supplied consts
        to indicate the desired day_of_week, ex. pendulum.MONDAY.
        """
        return self.set(self.year, self.quarter * 3 - 2, 1).first_of(
            "month", day_of_week
        )

    def _last_of_quarter(self, day_of_week: WeekDay | None = None) -> Self:
        """
        Modify to the last occurrence of a given day of the week
        in the current quarter. If no day_of_week is provided,
        modify to the last day of the quarter. Use the supplied consts
        to indicate the desired day_of_week, ex. pendulum.MONDAY.
        """
        return self.set(self.year, self.quarter * 3, 1).last_of("month", day_of_week)

    def _nth_of_quarter(self, nth: int, day_of_week: WeekDay) -> Self | None:
        """
        Modify to the given occurrence of a given day of the week
        in the current quarter. If the calculated occurrence is outside,
        the scope of the current quarter, then return False and no
        modifications are made. Use the supplied consts
        to indicate the desired day_of_week, ex. pendulum.MONDAY.
        """
        if nth == 1:
            return self.first_of("quarter", day_of_week)

        dt = self.replace(self.year, self.quarter * 3, 1)
        last_month = dt.month
        year = dt.year
        dt = dt.first_of("quarter")
        for _ in range(nth - (1 if dt.day_of_week == day_of_week else 0)):
            dt = dt.next(day_of_week)

        if last_month < dt.month or year != dt.year:
            return None

        return self.set(self.year, dt.month, dt.day)

    def _first_of_year(self, day_of_week: WeekDay | None = None) -> Self:
        """
        Modify to the first occurrence of a given day of the week
        in the current year. If no day_of_week is provided,
        modify to the first day of the year. Use the supplied consts
        to indicate the desired day_of_week, ex. pendulum.MONDAY.
        """
        return self.set(month=1).first_of("month", day_of_week)

    def _last_of_year(self, day_of_week: WeekDay | None = None) -> Self:
        """
        Modify to the last occurrence of a given day of the week
        in the current year. If no day_of_week is provided,
        modify to the last day of the year. Use the supplied consts
        to indicate the desired day_of_week, ex. pendulum.MONDAY.
        """
        return self.set(month=MONTHS_PER_YEAR).last_of("month", day_of_week)

    def _nth_of_year(self, nth: int, day_of_week: WeekDay) -> Self | None:
        """
        Modify to the given occurrence of a given day of the week
        in the current year. If the calculated occurrence is outside,
        the scope of the current year, then return False and no
        modifications are made. Use the supplied consts
        to indicate the desired day_of_week, ex. pendulum.MONDAY.
        """
        if nth == 1:
            return self.first_of("year", day_of_week)

        dt = self.first_of("year")
        year = dt.year
        for _ in range(nth - (1 if dt.day_of_week == day_of_week else 0)):
            dt = dt.next(day_of_week)

        if year != dt.year:
            return None

        return self.set(self.year, dt.month, dt.day)

    def average(self, dt: date | None = None) -> Self:
        """
        Modify the current instance to the average
        of a given instance (default now) and the current instance.
        """
        if dt is None:
            dt = Date.today()

        return self.add(days=int(self.diff(dt, False).in_days() / 2))

    # Native methods override

    @classmethod
    def today(cls) -> Self:
        dt = date.today()

        return cls(dt.year, dt.month, dt.day)

    @classmethod
    def fromtimestamp(cls, t: float) -> Self:
        dt = super().fromtimestamp(t)

        return cls(dt.year, dt.month, dt.day)

    @classmethod
    def fromordinal(cls, n: int) -> Self:
        dt = super().fromordinal(n)

        return cls(dt.year, dt.month, dt.day)

    def replace(
        self,
        year: SupportsIndex | None = None,
        month: SupportsIndex | None = None,
        day: SupportsIndex | None = None,
    ) -> Self:
        year = year if year is not None else self.year
        month = month if month is not None else self.month
        day = day if day is not None else self.day

        return self.__class__(year, month, day)


# === src/pendulum/_helpers.py ===
from __future__ import annotations

import datetime
import math

from typing import TYPE_CHECKING
from typing import NamedTuple
from typing import cast

from pendulum.constants import DAY_OF_WEEK_TABLE
from pendulum.constants import DAYS_PER_L_YEAR
from pendulum.constants import DAYS_PER_MONTHS
from pendulum.constants import DAYS_PER_N_YEAR
from pendulum.constants import EPOCH_YEAR
from pendulum.constants import MONTHS_OFFSETS
from pendulum.constants import SECS_PER_4_YEARS
from pendulum.constants import SECS_PER_100_YEARS
from pendulum.constants import SECS_PER_400_YEARS
from pendulum.constants import SECS_PER_DAY
from pendulum.constants import SECS_PER_HOUR
from pendulum.constants import SECS_PER_MIN
from pendulum.constants import SECS_PER_YEAR
from pendulum.constants import TM_DECEMBER
from pendulum.constants import TM_JANUARY


if TYPE_CHECKING:
    import zoneinfo

    from pendulum.tz.timezone import Timezone


class PreciseDiff(NamedTuple):
    years: int
    months: int
    days: int
    hours: int
    minutes: int
    seconds: int
    microseconds: int
    total_days: int

    def __repr__(self) -> str:
        return (
            f"{self.years} years "
            f"{self.months} months "
            f"{self.days} days "
            f"{self.hours} hours "
            f"{self.minutes} minutes "
            f"{self.seconds} seconds "
            f"{self.microseconds} microseconds"
        )


def is_leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def is_long_year(year: int) -> bool:
    def p(y: int) -> int:
        return y + y // 4 - y // 100 + y // 400

    return p(year) % 7 == 4 or p(year - 1) % 7 == 3


def week_day(year: int, month: int, day: int) -> int:
    if month < 3:
        year -= 1

    w = (
        year
        + year // 4
        - year // 100
        + year // 400
        + DAY_OF_WEEK_TABLE[month - 1]
        + day
    ) % 7

    if not w:
        w = 7

    return w


def days_in_year(year: int) -> int:
    if is_leap(year):
        return DAYS_PER_L_YEAR

    return DAYS_PER_N_YEAR


def local_time(
    unix_time: int, utc_offset: int, microseconds: int
) -> tuple[int, int, int, int, int, int, int]:
    """
    Returns a UNIX time as a broken-down time
    for a particular transition type.
    """
    year = EPOCH_YEAR
    seconds = math.floor(unix_time)

    # Shift to a base year that is 400-year aligned.
    if seconds >= 0:
        seconds -= 10957 * SECS_PER_DAY
        year += 30  # == 2000
    else:
        seconds += (146097 - 10957) * SECS_PER_DAY
        year -= 370  # == 1600

    seconds += utc_offset

    # Handle years in chunks of 400/100/4/1
    year += 400 * (seconds // SECS_PER_400_YEARS)
    seconds %= SECS_PER_400_YEARS
    if seconds < 0:
        seconds += SECS_PER_400_YEARS
        year -= 400

    leap_year = 1  # 4-century aligned

    sec_per_100years = SECS_PER_100_YEARS[leap_year]
    while seconds >= sec_per_100years:
        seconds -= sec_per_100years
        year += 100
        leap_year = 0  # 1-century, non 4-century aligned
        sec_per_100years = SECS_PER_100_YEARS[leap_year]

    sec_per_4years = SECS_PER_4_YEARS[leap_year]
    while seconds >= sec_per_4years:
        seconds -= sec_per_4years
        year += 4
        leap_year = 1  # 4-year, non century aligned
        sec_per_4years = SECS_PER_4_YEARS[leap_year]

    sec_per_year = SECS_PER_YEAR[leap_year]
    while seconds >= sec_per_year:
        seconds -= sec_per_year
        year += 1
        leap_year = 0  # non 4-year aligned
        sec_per_year = SECS_PER_YEAR[leap_year]

    # Handle months and days
    month = TM_DECEMBER + 1
    day = seconds // SECS_PER_DAY + 1
    seconds %= SECS_PER_DAY
    while month != TM_JANUARY + 1:
        month_offset = MONTHS_OFFSETS[leap_year][month]
        if day > month_offset:
            day -= month_offset
            break

        month -= 1

    # Handle hours, minutes, seconds and microseconds
    hour, seconds = divmod(seconds, SECS_PER_HOUR)
    minute, second = divmod(seconds, SECS_PER_MIN)

    return year, month, day, hour, minute, second, microseconds


def precise_diff(
    d1: datetime.datetime | datetime.date, d2: datetime.datetime | datetime.date
) -> PreciseDiff:
    """
    Calculate a precise difference between two datetimes.

    :param d1: The first datetime
    :param d2: The second datetime
    """
    sign = 1

    if d1 == d2:
        return PreciseDiff(0, 0, 0, 0, 0, 0, 0, 0)

    tzinfo1: datetime.tzinfo | None = (
        d1.tzinfo if isinstance(d1, datetime.datetime) else None
    )
    tzinfo2: datetime.tzinfo | None = (
        d2.tzinfo if isinstance(d2, datetime.datetime) else None
    )

    if (tzinfo1 is None and tzinfo2 is not None) or (
        tzinfo2 is None and tzinfo1 is not None
    ):
        raise ValueError(
            "Comparison between naive and aware datetimes is not supported"
        )

    if d1 > d2:
        d1, d2 = d2, d1
        sign = -1

    d_diff = 0
    hour_diff = 0
    min_diff = 0
    sec_diff = 0
    mic_diff = 0
    total_days = _day_number(d2.year, d2.month, d2.day) - _day_number(
        d1.year, d1.month, d1.day
    )
    in_same_tz = False
    tz1 = None
    tz2 = None

    # Trying to figure out the timezone names
    # If we can't find them, we assume different timezones
    if tzinfo1 and tzinfo2:
        tz1 = _get_tzinfo_name(tzinfo1)
        tz2 = _get_tzinfo_name(tzinfo2)

        in_same_tz = tz1 == tz2 and tz1 is not None

    if isinstance(d2, datetime.datetime):
        if isinstance(d1, datetime.datetime):
            # If we are not in the same timezone
            # we need to adjust
            #
            # We also need to adjust if we do not
            # have variable-length units
            if not in_same_tz or total_days == 0:
                offset1 = d1.utcoffset()
                offset2 = d2.utcoffset()

                if offset1:
                    d1 = d1 - offset1

                if offset2:
                    d2 = d2 - offset2

            hour_diff = d2.hour - d1.hour
            min_diff = d2.minute - d1.minute
            sec_diff = d2.second - d1.second
            mic_diff = d2.microsecond - d1.microsecond
        else:
            hour_diff = d2.hour
            min_diff = d2.minute
            sec_diff = d2.second
            mic_diff = d2.microsecond

        if mic_diff < 0:
            mic_diff += 1000000
            sec_diff -= 1

        if sec_diff < 0:
            sec_diff += 60
            min_diff -= 1

        if min_diff < 0:
            min_diff += 60
            hour_diff -= 1

        if hour_diff < 0:
            hour_diff += 24
            d_diff -= 1

    y_diff = d2.year - d1.year
    m_diff = d2.month - d1.month
    d_diff += d2.day - d1.day

    if d_diff < 0:
        year = d2.year
        month = d2.month

        if month == 1:
            month = 12
            year -= 1
        else:
            month -= 1

        leap = int(is_leap(year))

        days_in_last_month = DAYS_PER_MONTHS[leap][month]
        days_in_month = DAYS_PER_MONTHS[int(is_leap(d2.year))][d2.month]

        if d_diff < days_in_month - days_in_last_month:
            # We don't have a full month, we calculate days
            if days_in_last_month < d1.day:
                d_diff += d1.day
            else:
                d_diff += days_in_last_month
        elif d_diff == days_in_month - days_in_last_month:
            # We have exactly a full month
            # We remove the days difference
            # and add one to the months difference
            d_diff = 0
            m_diff += 1
        else:
            # We have a full month
            d_diff += days_in_last_month

        m_diff -= 1

    if m_diff < 0:
        m_diff += 12
        y_diff -= 1

    return PreciseDiff(
        sign * y_diff,
        sign * m_diff,
        sign * d_diff,
        sign * hour_diff,
        sign * min_diff,
        sign * sec_diff,
        sign * mic_diff,
        sign * total_days,
    )


def _day_number(year: int, month: int, day: int) -> int:
    month = (month + 9) % 12
    year = year - month // 10

    return (
        365 * year
        + year // 4
        - year // 100
        + year // 400
        + (month * 306 + 5) // 10
        + (day - 1)
    )


def _get_tzinfo_name(tzinfo: datetime.tzinfo | None) -> str | None:
    if tzinfo is None:
        return None

    if hasattr(tzinfo, "key"):
        # zoneinfo timezone
        return cast("zoneinfo.ZoneInfo", tzinfo).key
    elif hasattr(tzinfo, "name"):
        # Pendulum timezone
        return cast("Timezone", tzinfo).name
    elif hasattr(tzinfo, "zone"):
        # pytz timezone
        return tzinfo.zone  # type: ignore[no-any-return]

    return None


# === src/pendulum/locales/__init__.py ===


# === src/pendulum/locales/locale.py ===
from __future__ import annotations

import re

from importlib import import_module, resources
from pathlib import Path
from typing import Any
from typing import ClassVar
from typing import Dict
from typing import cast


class Locale:
    """
    Represent a specific locale.
    """

    _cache: ClassVar[dict[str, Locale]] = {}

    def __init__(self, locale: str, data: Any) -> None:
        self._locale: str = locale
        self._data: Any = data
        self._key_cache: dict[str, str] = {}

    @classmethod
    def load(cls, locale: str | Locale) -> Locale:
        if isinstance(locale, Locale):
            return locale

        locale = cls.normalize_locale(locale)
        if locale in cls._cache:
            return cls._cache[locale]

        # Checking locale existence
        actual_locale = locale
        locale_path = cast(Path, resources.files(__package__).joinpath(actual_locale))
        while not locale_path.exists():
            if actual_locale == locale:
                raise ValueError(f"Locale [{locale}] does not exist.")

            actual_locale = actual_locale.split("_")[0]

        m = import_module(f"pendulum.locales.{actual_locale}.locale")

        cls._cache[locale] = cls(locale, m.locale)

        return cls._cache[locale]

    @classmethod
    def normalize_locale(cls, locale: str) -> str:
        m = re.fullmatch("([a-z]{2})[-_]([a-z]{2})", locale, re.I)
        if m:
            return f"{m.group(1).lower()}_{m.group(2).lower()}"
        else:
            return locale.lower()

    def get(self, key: str, default: Any | None = None) -> Any:
        if key in self._key_cache:
            return self._key_cache[key]

        parts = key.split(".")
        try:
            result = self._data[parts[0]]
            for part in parts[1:]:
                result = result[part]
        except KeyError:
            result = default

        self._key_cache[key] = result

        return self._key_cache[key]

    def translation(self, key: str) -> Any:
        return self.get(f"translations.{key}")

    def plural(self, number: int) -> str:
        return cast(str, self._data["plural"](number))

    def ordinal(self, number: int) -> str:
        return cast(str, self._data["ordinal"](number))

    def ordinalize(self, number: int) -> str:
        ordinal = self.get(f"custom.ordinal.{self.ordinal(number)}")

        if not ordinal:
            return f"{number}"

        return f"{number}{ordinal}"

    def match_translation(self, key: str, value: Any) -> dict[str, str] | None:
        translations = self.translation(key)
        if value not in translations.values():
            return None

        return cast(Dict[str, str], {v: k for k, v in translations.items()}[value])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self._locale}')"


# === src/pendulum/locales/sk/custom.py ===
"""
sk custom locale file.
"""
from __future__ import annotations


translations = {
    # Relative time
    "ago": "pred {}",
    "from_now": "o {}",
    "after": "{0} po",
    "before": "{0} pred",
    # Date formats
    "date_formats": {
        "LTS": "HH:mm:ss",
        "LT": "HH:mm",
        "LLLL": "dddd, D. MMMM YYYY HH:mm",
        "LLL": "D. MMMM YYYY HH:mm",
        "LL": "D. MMMM YYYY",
        "L": "DD.MM.YYYY",
    },
}


# === src/pendulum/locales/sk/__init__.py ===


# === src/pendulum/locales/sk/locale.py ===
from __future__ import annotations

from pendulum.locales.sk.custom import translations as custom_translations


"""
sk locale file.

It has been generated automatically and must not be modified directly.
"""


locale = {
    "plural": lambda n: "few"
    if ((n == n and (n >= 2 and n <= 4)) and (0 == 0 and (0 == 0)))
    else "many"
    if (not (0 == 0 and (0 == 0)))
    else "one"
    if ((n == n and (n == 1)) and (0 == 0 and (0 == 0)))
    else "other",
    "ordinal": lambda n: "other",
    "translations": {
        "days": {
            "abbreviated": {
                0: "po",
                1: "ut",
                2: "st",
                3: "št",
                4: "pi",
                5: "so",
                6: "ne",
            },
            "narrow": {
                0: "p",
                1: "u",
                2: "s",
                3: "š",
                4: "p",
                5: "s",
                6: "n",
            },
            "short": {
                0: "po",
                1: "ut",
                2: "st",
                3: "št",
                4: "pi",
                5: "so",
                6: "ne",
            },
            "wide": {
                0: "pondelok",
                1: "utorok",
                2: "streda",
                3: "štvrtok",
                4: "piatok",
                5: "sobota",
                6: "nedeľa",
            },
        },
        "months": {
            "abbreviated": {
                1: "jan",
                2: "feb",
                3: "mar",
                4: "apr",
                5: "máj",
                6: "jún",
                7: "júl",
                8: "aug",
                9: "sep",
                10: "okt",
                11: "nov",
                12: "dec",
            },
            "narrow": {
                1: "j",
                2: "f",
                3: "m",
                4: "a",
                5: "m",
                6: "j",
                7: "j",
                8: "a",
                9: "s",
                10: "o",
                11: "n",
                12: "d",
            },
            "wide": {
                1: "januára",
                2: "februára",
                3: "marca",
                4: "apríla",
                5: "mája",
                6: "júna",
                7: "júla",
                8: "augusta",
                9: "septembra",
                10: "októbra",
                11: "novembra",
                12: "decembra",
            },
        },
        "units": {
            "year": {
                "one": "{0} rok",
                "few": "{0} roky",
                "many": "{0} roka",
                "other": "{0} rokov",
            },
            "month": {
                "one": "{0} mesiac",
                "few": "{0} mesiace",
                "many": "{0} mesiaca",
                "other": "{0} mesiacov",
            },
            "week": {
                "one": "{0} týždeň",
                "few": "{0} týždne",
                "many": "{0} týždňa",
                "other": "{0} týždňov",
            },
            "day": {
                "one": "{0} deň",
                "few": "{0} dni",
                "many": "{0} dňa",
                "other": "{0} dní",
            },
            "hour": {
                "one": "{0} hodina",
                "few": "{0} hodiny",
                "many": "{0} hodiny",
                "other": "{0} hodín",
            },
            "minute": {
                "one": "{0} minúta",
                "few": "{0} minúty",
                "many": "{0} minúty",
                "other": "{0} minút",
            },
            "second": {
                "one": "{0} sekunda",
                "few": "{0} sekundy",
                "many": "{0} sekundy",
                "other": "{0} sekúnd",
            },
            "microsecond": {
                "one": "{0} mikrosekunda",
                "few": "{0} mikrosekundy",
                "many": "{0} mikrosekundy",
                "other": "{0} mikrosekúnd",
            },
        },
        "relative": {
            "year": {
                "future": {
                    "other": "o {0} rokov",
                    "one": "o {0} rok",
                    "few": "o {0} roky",
                    "many": "o {0} roka",
                },
                "past": {
                    "other": "pred {0} rokmi",
                    "one": "pred {0} rokom",
                    "few": "pred {0} rokmi",
                    "many": "pred {0} roka",
                },
            },
            "month": {
                "future": {
                    "other": "o {0} mesiacov",
                    "one": "o {0} mesiac",
                    "few": "o {0} mesiace",
                    "many": "o {0} mesiaca",
                },
                "past": {
                    "other": "pred {0} mesiacmi",
                    "one": "pred {0} mesiacom",
                    "few": "pred {0} mesiacmi",
                    "many": "pred {0} mesiaca",
                },
            },
            "week": {
                "future": {
                    "other": "o {0} týždňov",
                    "one": "o {0} týždeň",
                    "few": "o {0} týždne",
                    "many": "o {0} týždňa",
                },
                "past": {
                    "other": "pred {0} týždňami",
                    "one": "pred {0} týždňom",
                    "few": "pred {0} týždňami",
                    "many": "pred {0} týždňa",
                },
            },
            "day": {
                "future": {
                    "other": "o {0} dní",
                    "one": "o {0} deň",
                    "few": "o {0} dni",
                    "many": "o {0} dňa",
                },
                "past": {
                    "other": "pred {0} dňami",
                    "one": "pred {0} dňom",
                    "few": "pred {0} dňami",
                    "many": "pred {0} dňa",
                },
            },
            "hour": {
                "future": {
                    "other": "o {0} hodín",
                    "one": "o {0} hodinu",
                    "few": "o {0} hodiny",
                    "many": "o {0} hodiny",
                },
                "past": {
                    "other": "pred {0} hodinami",
                    "one": "pred {0} hodinou",
                    "few": "pred {0} hodinami",
                    "many": "pred {0} hodinou",
                },
            },
            "minute": {
                "future": {
                    "other": "o {0} minút",
                    "one": "o {0} minútu",
                    "few": "o {0} minúty",
                    "many": "o {0} minúty",
                },
                "past": {
                    "other": "pred {0} minútami",
                    "one": "pred {0} minútou",
                    "few": "pred {0} minútami",
                    "many": "pred {0} minúty",
                },
            },
            "second": {
                "future": {
                    "other": "o {0} sekúnd",
                    "one": "o {0} sekundu",
                    "few": "o {0} sekundy",
                    "many": "o {0} sekundy",
                },
                "past": {
                    "other": "pred {0} sekundami",
                    "one": "pred {0} sekundou",
                    "few": "pred {0} sekundami",
                    "many": "pred {0} sekundy",
                },
            },
        },
        "day_periods": {
            "midnight": "o polnoci",
            "am": "AM",
            "noon": "napoludnie",
            "pm": "PM",
            "morning1": "ráno",
            "morning2": "dopoludnia",
            "afternoon1": "popoludní",
            "evening1": "večer",
            "night1": "v noci",
        },
        "week_data": {
            "min_days": 1,
            "first_day": 0,
            "weekend_start": 5,
            "weekend_end": 6,
        },
    },
    "custom": custom_translations,
}


# === src/pendulum/locales/pl/custom.py ===
"""
pl custom locale file.
"""
from __future__ import annotations


translations = {
    "units": {"few_second": "kilka sekund"},
    # Relative time
    "ago": "{} temu",
    "from_now": "za {}",
    "after": "{0} po",
    "before": "{0} przed",
    # Date formats
    "date_formats": {
        "LTS": "HH:mm:ss",
        "LT": "HH:mm",
        "L": "DD.MM.YYYY",
        "LL": "D MMMM YYYY",
        "LLL": "D MMMM YYYY HH:mm",
        "LLLL": "dddd, D MMMM YYYY HH:mm",
    },
}


# === src/pendulum/locales/pl/__init__.py ===


# === src/pendulum/locales/pl/locale.py ===
from __future__ import annotations

from pendulum.locales.pl.custom import translations as custom_translations


"""
pl locale file.

It has been generated automatically and must not be modified directly.
"""


locale = {
    "plural": lambda n: "few"
    if (
        (
            (0 == 0 and (0 == 0))
            and ((n % 10) == (n % 10) and ((n % 10) >= 2 and (n % 10) <= 4))
        )
        and (not ((n % 100) == (n % 100) and ((n % 100) >= 12 and (n % 100) <= 14)))
    )
    else "many"
    if (
        (
            (
                ((0 == 0 and (0 == 0)) and (not (n == n and (n == 1))))
                and ((n % 10) == (n % 10) and ((n % 10) >= 0 and (n % 10) <= 1))
            )
            or (
                (0 == 0 and (0 == 0))
                and ((n % 10) == (n % 10) and ((n % 10) >= 5 and (n % 10) <= 9))
            )
        )
        or (
            (0 == 0 and (0 == 0))
            and ((n % 100) == (n % 100) and ((n % 100) >= 12 and (n % 100) <= 14))
        )
    )
    else "one"
    if ((n == n and (n == 1)) and (0 == 0 and (0 == 0)))
    else "other",
    "ordinal": lambda n: "other",
    "translations": {
        "days": {
            "abbreviated": {
                0: "pon.",
                1: "wt.",
                2: "śr.",
                3: "czw.",
                4: "pt.",
                5: "sob.",
                6: "niedz.",
            },
            "narrow": {0: "p", 1: "w", 2: "ś", 3: "c", 4: "p", 5: "s", 6: "n"},
            "short": {
                0: "pon",
                1: "wto",
                2: "śro",
                3: "czw",
                4: "pią",
                5: "sob",
                6: "nie",
            },
            "wide": {
                0: "poniedziałek",
                1: "wtorek",
                2: "środa",
                3: "czwartek",
                4: "piątek",
                5: "sobota",
                6: "niedziela",
            },
        },
        "months": {
            "abbreviated": {
                1: "sty",
                2: "lut",
                3: "mar",
                4: "kwi",
                5: "maj",
                6: "cze",
                7: "lip",
                8: "sie",
                9: "wrz",
                10: "paź",
                11: "lis",
                12: "gru",
            },
            "narrow": {
                1: "s",
                2: "l",
                3: "m",
                4: "k",
                5: "m",
                6: "c",
                7: "l",
                8: "s",
                9: "w",
                10: "p",
                11: "l",
                12: "g",
            },
            "wide": {
                1: "stycznia",
                2: "lutego",
                3: "marca",
                4: "kwietnia",
                5: "maja",
                6: "czerwca",
                7: "lipca",
                8: "sierpnia",
                9: "września",
                10: "października",
                11: "listopada",
                12: "grudnia",
            },
        },
        "units": {
            "year": {
                "one": "{0} rok",
                "few": "{0} lata",
                "many": "{0} lat",
                "other": "{0} roku",
            },
            "month": {
                "one": "{0} miesiąc",
                "few": "{0} miesiące",
                "many": "{0} miesięcy",
                "other": "{0} miesiąca",
            },
            "week": {
                "one": "{0} tydzień",
                "few": "{0} tygodnie",
                "many": "{0} tygodni",
                "other": "{0} tygodnia",
            },
            "day": {
                "one": "{0} dzień",
                "few": "{0} dni",
                "many": "{0} dni",
                "other": "{0} dnia",
            },
            "hour": {
                "one": "{0} godzina",
                "few": "{0} godziny",
                "many": "{0} godzin",
                "other": "{0} godziny",
            },
            "minute": {
                "one": "{0} minuta",
                "few": "{0} minuty",
                "many": "{0} minut",
                "other": "{0} minuty",
            },
            "second": {
                "one": "{0} sekunda",
                "few": "{0} sekundy",
                "many": "{0} sekund",
                "other": "{0} sekundy",
            },
            "microsecond": {
                "one": "{0} mikrosekunda",
                "few": "{0} mikrosekundy",
                "many": "{0} mikrosekund",
                "other": "{0} mikrosekundy",
            },
        },
        "relative": {
            "year": {
                "future": {
                    "other": "za {0} roku",
                    "one": "za {0} rok",
                    "few": "za {0} lata",
                    "many": "za {0} lat",
                },
                "past": {
                    "other": "{0} roku temu",
                    "one": "{0} rok temu",
                    "few": "{0} lata temu",
                    "many": "{0} lat temu",
                },
            },
            "month": {
                "future": {
                    "other": "za {0} miesiąca",
                    "one": "za {0} miesiąc",
                    "few": "za {0} miesiące",
                    "many": "za {0} miesięcy",
                },
                "past": {
                    "other": "{0} miesiąca temu",
                    "one": "{0} miesiąc temu",
                    "few": "{0} miesiące temu",
                    "many": "{0} miesięcy temu",
                },
            },
            "week": {
                "future": {
                    "other": "za {0} tygodnia",
                    "one": "za {0} tydzień",
                    "few": "za {0} tygodnie",
                    "many": "za {0} tygodni",
                },
                "past": {
                    "other": "{0} tygodnia temu",
                    "one": "{0} tydzień temu",
                    "few": "{0} tygodnie temu",
                    "many": "{0} tygodni temu",
                },
            },
            "day": {
                "future": {
                    "other": "za {0} dnia",
                    "one": "za {0} dzień",
                    "few": "za {0} dni",
                    "many": "za {0} dni",
                },
                "past": {
                    "other": "{0} dnia temu",
                    "one": "{0} dzień temu",
                    "few": "{0} dni temu",
                    "many": "{0} dni temu",
                },
            },
            "hour": {
                "future": {
                    "other": "za {0} godziny",
                    "one": "za {0} godzinę",
                    "few": "za {0} godziny",
                    "many": "za {0} godzin",
                },
                "past": {
                    "other": "{0} godziny temu",
                    "one": "{0} godzinę temu",
                    "few": "{0} godziny temu",
                    "many": "{0} godzin temu",
                },
            },
            "minute": {
                "future": {
                    "other": "za {0} minuty",
                    "one": "za {0} minutę",
                    "few": "za {0} minuty",
                    "many": "za {0} minut",
                },
                "past": {
                    "other": "{0} minuty temu",
                    "one": "{0} minutę temu",
                    "few": "{0} minuty temu",
                    "many": "{0} minut temu",
                },
            },
            "second": {
                "future": {
                    "other": "za {0} sekundy",
                    "one": "za {0} sekundę",
                    "few": "za {0} sekundy",
                    "many": "za {0} sekund",
                },
                "past": {
                    "other": "{0} sekundy temu",
                    "one": "{0} sekundę temu",
                    "few": "{0} sekundy temu",
                    "many": "{0} sekund temu",
                },
            },
        },
        "day_periods": {
            "midnight": "o północy",
            "am": "AM",
            "noon": "w południe",
            "pm": "PM",
            "morning1": "rano",
            "morning2": "przed południem",
            "afternoon1": "po południu",
            "evening1": "wieczorem",
            "night1": "w nocy",
        },
        "week_data": {
            "min_days": 1,
            "first_day": 0,
            "weekend_start": 5,
            "weekend_end": 6,
        },
    },
    "custom": custom_translations,
}


# === src/pendulum/locales/ua/custom.py ===
"""
ua custom locale file.
"""
from __future__ import annotations


translations = {
    "units": {"few_second": "кілька секунд"},
    # Relative time
    "ago": "{} тому",
    "from_now": "за {}",
    "after": "{0} посіля",
    "before": "{0} до",
    # Date formats
    "date_formats": {
        "LTS": "HH:mm:ss",
        "LT": "HH:mm",
        "L": "DD.MM.YYYY",
        "LL": "D MMMM YYYY р.",
        "LLL": "D MMMM YYYY р., HH:mm",
        "LLLL": "dddd, D MMMM YYYY р., HH:mm",
    },
}


# === src/pendulum/locales/ua/__init__.py ===


# === src/pendulum/locales/ua/locale.py ===
from __future__ import annotations

from pendulum.locales.ua.custom import translations as custom_translations


"""
ua locale file.

It has been generated automatically and must not be modified directly.
"""


locale = {
    "plural": lambda n: "few"
    if (
        (
            (0 == 0 and (0 == 0))
            and ((n % 10) == (n % 10) and ((n % 10) >= 2 and (n % 10) <= 4))
        )
        and (not ((n % 100) == (n % 100) and ((n % 100) >= 12 and (n % 100) <= 14)))
    )
    else "many"
    if (
        (
            ((0 == 0 and (0 == 0)) and ((n % 10) == (n % 10) and ((n % 10) == 0)))
            or (
                (0 == 0 and (0 == 0))
                and ((n % 10) == (n % 10) and ((n % 10) >= 5 and (n % 10) <= 9))
            )
        )
        or (
            (0 == 0 and (0 == 0))
            and ((n % 100) == (n % 100) and ((n % 100) >= 11 and (n % 100) <= 14))
        )
    )
    else "one"
    if (
        ((0 == 0 and (0 == 0)) and ((n % 10) == (n % 10) and ((n % 10) == 1)))
        and (not ((n % 100) == (n % 100) and ((n % 100) == 11)))
    )
    else "other",
    "ordinal": lambda n: "other",
    "translations": {
        "days": {
            "abbreviated": {
                0: "пн",
                1: "вт",
                2: "ср",
                3: "чт",
                4: "пт",
                5: "сб",
                6: "нд",
            },
            "narrow": {0: "пн", 1: "вт", 2: "ср", 3: "чт", 4: "пт", 5: "сб", 6: "нд"},
            "short": {0: "пн", 1: "вт", 2: "ср", 3: "чт", 4: "пт", 5: "сб", 6: "нд"},
            "wide": {
                0: "понеділок",
                1: "вівторок",
                2: "середа",
                3: "четвер",
                4: "п'ятниця",
                5: "субота",
                6: "неділя",
            },
        },
        "months": {
            "abbreviated": {
                1: "січ.",
                2: "лют.",
                3: "бер.",
                4: "квіт.",
                5: "трав.",
                6: "черв.",
                7: "лип.",
                8: "серп.",
                9: "вер.",
                10: "жовт.",
                11: "лист.",
                12: "груд.",
            },
            "narrow": {
                1: "С",
                2: "Л",
                3: "Б",
                4: "К",
                5: "Т",
                6: "Ч",
                7: "Л",
                8: "С",
                9: "В",
                10: "Ж",
                11: "Л",
                12: "Г",
            },
            "wide": {
                1: "січня",
                2: "лютого",
                3: "березня",
                4: "квітня",
                5: "травня",
                6: "червня",
                7: "липня",
                8: "серпня",
                9: "вересня",
                10: "жовтня",
                11: "листопада",
                12: "грудня",
            },
        },
        "units": {
            "year": {
                "one": "{0} рік",
                "few": "{0} роки",
                "many": "{0} років",
                "other": "{0} роки",
            },
            "month": {
                "one": "{0} місяць",
                "few": "{0} місяця",
                "many": "{0} місяців",
                "other": "{0} місяці",
            },
            "week": {
                "one": "{0} тиждень",
                "few": "{0} тижня",
                "many": "{0} тижнів",
                "other": "{0} тижні",
            },
            "day": {
                "one": "{0} день",
                "few": "{0} дні",
                "many": "{0} днів",
                "other": "{0} дні",
            },
            "hour": {
                "one": "{0} година",
                "few": "{0} години",
                "many": "{0} годин",
                "other": "{0} години",
            },
            "minute": {
                "one": "{0} хвилина",
                "few": "{0} хвилини",
                "many": "{0} хвилин",
                "other": "{0} хвилини",
            },
            "second": {
                "one": "{0} секунда",
                "few": "{0} секунди",
                "many": "{0} секунд",
                "other": "{0} секунди",
            },
            "microsecond": {
                "one": "{0} мікросекунда",
                "few": "{0} мікросекунди",
                "many": "{0} мікросекунд",
                "other": "{0} мікросекунд",
            },
        },
        "relative": {
            "year": {
                "future": {
                    "other": "за {0} роки",
                    "one": "за {0} рік",
                    "few": "за {0} роки",
                    "many": "за {0} років",
                },
                "past": {
                    "other": "{0} роки тому",
                    "one": "{0} рік тому",
                    "few": "{0} роки тому",
                    "many": "{0} років тому",
                },
            },
            "month": {
                "future": {
                    "other": "за {0} місяці",
                    "one": "за {0} місяць",
                    "few": "за {0} місяця",
                    "many": "за {0} місяців",
                },
                "past": {
                    "other": "{0} місяці тому",
                    "one": "{0} місяц тому",
                    "few": "{0} місяця тому",
                    "many": "{0} місяців тому",
                },
            },
            "week": {
                "future": {
                    "other": "за {0} тижні",
                    "one": "за {0} тиждень",
                    "few": "за {0} тижня",
                    "many": "за {0} тижднів",
                },
                "past": {
                    "other": "{0} тижні тому",
                    "one": "{0} тиждень тому",
                    "few": "{0} тижня тому",
                    "many": "{0} тижнів тому",
                },
            },
            "day": {
                "future": {
                    "other": "за {0} дні",
                    "one": "за {0} день",
                    "few": "за {0} дні",
                    "many": "за {0} днів",
                },
                "past": {
                    "other": "{0} дні тому",
                    "one": "{0} день тому",
                    "few": "{0} дні тому",
                    "many": "{0} днів тому",
                },
            },
            "hour": {
                "future": {
                    "other": "за {0} години",
                    "one": "за {0} година",
                    "few": "за {0} години",
                    "many": "за {0} годин",
                },
                "past": {
                    "other": "{0} години тому",
                    "one": "{0} година тому",
                    "few": "{0} години тому",
                    "many": "{0} годин тому",
                },
            },
            "minute": {
                "future": {
                    "other": "за {0} хвилини",
                    "one": "за {0} хвилина",
                    "few": "за {0} хвилини",
                    "many": "за {0} хвилин",
                },
                "past": {
                    "other": "{0} хвилини тому",
                    "one": "{0} хвилина тому",
                    "few": "{0} хвилини тому",
                    "many": "{0} хвилин тому",
                },
            },
            "second": {
                "future": {
                    "other": "за {0} секунди",
                    "one": "за {0} секунду",
                    "few": "за {0} секунди",
                    "many": "за {0} секунд",
                },
                "past": {
                    "other": "{0} секунди тому",
                    "one": "{0} секунду тому",
                    "few": "{0} секунди тому",
                    "many": "{0} секунд тому",
                },
            },
        },
        "day_periods": {
            "midnight": "опівночі",
            "am": "AM",
            "noon": "полудень",
            "pm": "PM",
            "morning1": "ранку",
            "morning2": "до півдня",
            "afternoon1": "дня",
            "afternoon2": "пополуднє",
            "evening1": "ввечері",
            "evening2": "увечері",
            "night1": "в ніч",
        },
        "week_data": {
            "min_days": 1,
            "first_day": 0,
            "weekend_start": 5,
            "weekend_end": 6,
        },
    },
    "custom": custom_translations,
}


# === src/pendulum/locales/sv/custom.py ===
"""
sv custom locale file.
"""
from __future__ import annotations


translations = {
    # Relative time
    "ago": "{} sedan",
    "from_now": "från nu {}",
    "after": "{0} efter",
    "before": "{0} innan",
    # Date formats
    "date_formats": {
        "LTS": "HH:mm:ss",
        "LT": "HH:mm",
        "L": "YYYY-MM-DD",
        "LL": "D MMMM YYYY",
        "LLL": "D MMMM YYYY, HH:mm",
        "LLLL": "dddd, D MMMM YYYY, HH:mm",
    },
}


# === src/pendulum/locales/sv/__init__.py ===


# === src/pendulum/locales/sv/locale.py ===
from __future__ import annotations

from pendulum.locales.sv.custom import translations as custom_translations


"""
sv locale file.

It has been generated automatically and must not be modified directly.
"""


locale = {
    "plural": lambda n: "one"
    if ((n == n and (n == 1)) and (0 == 0 and (0 == 0)))
    else "other",
    "ordinal": lambda n: "one"
    if (
        ((n % 10) == (n % 10) and (((n % 10) == 1) or ((n % 10) == 2)))
        and (not ((n % 100) == (n % 100) and (((n % 100) == 11) or ((n % 100) == 12))))
    )
    else "other",
    "translations": {
        "days": {
            "abbreviated": {
                0: "mån",
                1: "tis",
                2: "ons",
                3: "tors",
                4: "fre",
                5: "lör",
                6: "sön",
            },
            "narrow": {
                0: "M",
                1: "T",
                2: "O",
                3: "T",
                4: "F",
                5: "L",
                6: "S",
            },
            "short": {
                0: "må",
                1: "ti",
                2: "on",
                3: "to",
                4: "fr",
                5: "lö",
                6: "sö",
            },
            "wide": {
                0: "måndag",
                1: "tisdag",
                2: "onsdag",
                3: "torsdag",
                4: "fredag",
                5: "lördag",
                6: "söndag",
            },
        },
        "months": {
            "abbreviated": {
                1: "jan.",
                2: "feb.",
                3: "mars",
                4: "apr.",
                5: "maj",
                6: "juni",
                7: "juli",
                8: "aug.",
                9: "sep.",
                10: "okt.",
                11: "nov.",
                12: "dec.",
            },
            "narrow": {
                1: "J",
                2: "F",
                3: "M",
                4: "A",
                5: "M",
                6: "J",
                7: "J",
                8: "A",
                9: "S",
                10: "O",
                11: "N",
                12: "D",
            },
            "wide": {
                1: "januari",
                2: "februari",
                3: "mars",
                4: "april",
                5: "maj",
                6: "juni",
                7: "juli",
                8: "augusti",
                9: "september",
                10: "oktober",
                11: "november",
                12: "december",
            },
        },
        "units": {
            "year": {
                "one": "{0} år",
                "other": "{0} år",
            },
            "month": {
                "one": "{0} månad",
                "other": "{0} månader",
            },
            "week": {
                "one": "{0} vecka",
                "other": "{0} veckor",
            },
            "day": {
                "one": "{0} dygn",
                "other": "{0} dygn",
            },
            "hour": {
                "one": "{0} timme",
                "other": "{0} timmar",
            },
            "minute": {
                "one": "{0} minut",
                "other": "{0} minuter",
            },
            "second": {
                "one": "{0} sekund",
                "other": "{0} sekunder",
            },
            "microsecond": {
                "one": "{0} mikrosekund",
                "other": "{0} mikrosekunder",
            },
        },
        "relative": {
            "year": {
                "future": {
                    "other": "om {0} år",
                    "one": "om {0} år",
                },
                "past": {
                    "other": "för {0} år sedan",
                    "one": "för {0} år sedan",
                },
            },
            "month": {
                "future": {
                    "other": "om {0} månader",
                    "one": "om {0} månad",
                },
                "past": {
                    "other": "för {0} månader sedan",
                    "one": "för {0} månad sedan",
                },
            },
            "week": {
                "future": {
                    "other": "om {0} veckor",
                    "one": "om {0} vecka",
                },
                "past": {
                    "other": "för {0} veckor sedan",
                    "one": "för {0} vecka sedan",
                },
            },
            "day": {
                "future": {
                    "other": "om {0} dagar",
                    "one": "om {0} dag",
                },
                "past": {
                    "other": "för {0} dagar sedan",
                    "one": "för {0} dag sedan",
                },
            },
            "hour": {
                "future": {
                    "other": "om {0} timmar",
                    "one": "om {0} timme",
                },
                "past": {
                    "other": "för {0} timmar sedan",
                    "one": "för {0} timme sedan",
                },
            },
            "minute": {
                "future": {
                    "other": "om {0} minuter",
                    "one": "om {0} minut",
                },
                "past": {
                    "other": "för {0} minuter sedan",
                    "one": "för {0} minut sedan",
                },
            },
            "second": {
                "future": {
                    "other": "om {0} sekunder",
                    "one": "om {0} sekund",
                },
                "past": {
                    "other": "för {0} sekunder sedan",
                    "one": "för {0} sekund sedan",
                },
            },
        },
        "day_periods": {
            "midnight": "midnatt",
            "am": "fm",
            "pm": "em",
            "morning1": "på morgonen",
            "morning2": "på förmiddagen",
            "afternoon1": "på eftermiddagen",
            "evening1": "på kvällen",
            "night1": "på natten",
        },
        "week_data": {
            "min_days": 1,
            "first_day": 0,
            "weekend_start": 5,
            "weekend_end": 6,
        },
    },
    "custom": custom_translations,
}


# === src/pendulum/locales/he/custom.py ===
"""
he custom locale file.
"""
from __future__ import annotations


translations = {
    "units": {"few_second": "כמה שניות"},
    # Relative time
    "ago": "לפני {0}",
    "from_now": "תוך {0}",
    "after": "בעוד {0}",
    "before": "{0} קודם",
    # Ordinals
    "ordinal": {"other": "º"},
    # Date formats
    "date_formats": {
        "LTS": "H:mm:ss",
        "LT": "H:mm",
        "LLLL": "dddd, D [ב] MMMM [ב] YYYY H:mm",
        "LLL": "D [ב] MMMM [ב] YYYY H:mm",
        "LL": "D [ב] MMMM [ב] YYYY",
        "L": "DD/MM/YYYY",
    },
}


# === src/pendulum/locales/he/__init__.py ===


# === src/pendulum/locales/he/locale.py ===
from __future__ import annotations

from pendulum.locales.he.custom import translations as custom_translations


"""
he locale file.

It has been generated automatically and must not be modified directly.
"""


locale = {
    "plural": lambda n: "many"
    if (
        ((0 == 0 and (0 == 0)) and (not (n == n and (n >= 0 and n <= 10))))
        and ((n % 10) == (n % 10) and ((n % 10) == 0))
    )
    else "one"
    if ((n == n and (n == 1)) and (0 == 0 and (0 == 0)))
    else "two"
    if ((n == n and (n == 2)) and (0 == 0 and (0 == 0)))
    else "other",
    "ordinal": lambda n: "other",
    "translations": {
        "days": {
            "abbreviated": {
                0: "יום ב׳",
                1: "יום ג׳",
                2: "יום ד׳",
                3: "יום ה׳",
                4: "יום ו׳",
                5: "שבת",
                6: "יום א׳",
            },
            "narrow": {
                0: "ב׳",
                1: "ג׳",
                2: "ד׳",
                3: "ה׳",
                4: "ו׳",
                5: "ש׳",
                6: "א׳",
            },
            "short": {
                0: "ב׳",
                1: "ג׳",
                2: "ד׳",
                3: "ה׳",
                4: "ו׳",
                5: "ש׳",
                6: "א׳",
            },
            "wide": {
                0: "יום שני",
                1: "יום שלישי",
                2: "יום רביעי",
                3: "יום חמישי",
                4: "יום שישי",
                5: "יום שבת",
                6: "יום ראשון",
            },
        },
        "months": {
            "abbreviated": {
                1: "ינו׳",
                2: "פבר׳",
                3: "מרץ",
                4: "אפר׳",
                5: "מאי",
                6: "יוני",
                7: "יולי",
                8: "אוג׳",
                9: "ספט׳",
                10: "אוק׳",
                11: "נוב׳",
                12: "דצמ׳",
            },
            "narrow": {
                1: "1",
                2: "2",
                3: "3",
                4: "4",
                5: "5",
                6: "6",
                7: "7",
                8: "8",
                9: "9",
                10: "10",
                11: "11",
                12: "12",
            },
            "wide": {
                1: "ינואר",
                2: "פברואר",
                3: "מרץ",
                4: "אפריל",
                5: "מאי",
                6: "יוני",
                7: "יולי",
                8: "אוגוסט",
                9: "ספטמבר",
                10: "אוקטובר",
                11: "נובמבר",
                12: "דצמבר",
            },
        },
        "units": {
            "year": {
                "one": "שנה",
                "two": "שנתיים",
                "many": "{0} שנים",
                "other": "{0} שנים",
            },
            "month": {
                "one": "חודש",
                "two": "חודשיים",
                "many": "{0} חודשים",
                "other": "{0} חודשים",
            },
            "week": {
                "one": "שבוע",
                "two": "שבועיים",
                "many": "{0} שבועות",
                "other": "{0} שבועות",
            },
            "day": {
                "one": "יום {0}",
                "two": "יומיים",
                "many": "{0} יום",
                "other": "{0} ימים",
            },
            "hour": {
                "one": "שעה",
                "two": "שעתיים",
                "many": "{0} שעות",
                "other": "{0} שעות",
            },
            "minute": {
                "one": "דקה",
                "two": "שתי דקות",
                "many": "{0} דקות",
                "other": "{0} דקות",
            },
            "second": {
                "one": "שניה",
                "two": "שתי שניות",
                "many": "\u200f{0} שניות",
                "other": "{0} שניות",
            },
            "microsecond": {
                "one": "{0} מיליונית שנייה",
                "two": "{0} מיליוניות שנייה",
                "many": "{0} מיליוניות שנייה",
                "other": "{0} מיליוניות שנייה",
            },
        },
        "relative": {
            "year": {
                "future": {
                    "other": "בעוד {0} שנים",
                    "one": "בעוד שנה",
                    "two": "בעוד שנתיים",
                    "many": "בעוד {0} שנה",
                },
                "past": {
                    "other": "לפני {0} שנים",
                    "one": "לפני שנה",
                    "two": "לפני שנתיים",
                    "many": "לפני {0} שנה",
                },
            },
            "month": {
                "future": {
                    "other": "בעוד {0} חודשים",
                    "one": "בעוד חודש",
                    "two": "בעוד חודשיים",
                    "many": "בעוד {0} חודשים",
                },
                "past": {
                    "other": "לפני {0} חודשים",
                    "one": "לפני חודש",
                    "two": "לפני חודשיים",
                    "many": "לפני {0} חודשים",
                },
            },
            "week": {
                "future": {
                    "other": "בעוד {0} שבועות",
                    "one": "בעוד שבוע",
                    "two": "בעוד שבועיים",
                    "many": "בעוד {0} שבועות",
                },
                "past": {
                    "other": "לפני {0} שבועות",
                    "one": "לפני שבוע",
                    "two": "לפני שבועיים",
                    "many": "לפני {0} שבועות",
                },
            },
            "day": {
                "future": {
                    "other": "בעוד {0} ימים",
                    "one": "בעוד יום {0}",
                    "two": "בעוד יומיים",
                    "many": "בעוד {0} ימים",
                },
                "past": {
                    "other": "לפני {0} ימים",
                    "one": "לפני יום {0}",
                    "two": "לפני יומיים",
                    "many": "לפני {0} ימים",
                },
            },
            "hour": {
                "future": {
                    "other": "בעוד {0} שעות",
                    "one": "בעוד שעה",
                    "two": "בעוד שעתיים",
                    "many": "בעוד {0} שעות",
                },
                "past": {
                    "other": "לפני {0} שעות",
                    "one": "לפני שעה",
                    "two": "לפני שעתיים",
                    "many": "לפני {0} שעות",
                },
            },
            "minute": {
                "future": {
                    "other": "בעוד {0} דקות",
                    "one": "בעוד דקה",
                    "two": "בעוד שתי דקות",
                    "many": "בעוד {0} דקות",
                },
                "past": {
                    "other": "לפני {0} דקות",
                    "one": "לפני דקה",
                    "two": "לפני שתי דקות",
                    "many": "לפני {0} דקות",
                },
            },
            "second": {
                "future": {
                    "other": "בעוד {0} שניות",
                    "one": "בעוד שנייה",
                    "two": "בעוד שתי שניות",
                    "many": "בעוד {0} שניות",
                },
                "past": {
                    "other": "לפני {0} שניות",
                    "one": "לפני שנייה",
                    "two": "לפני שתי שניות",
                    "many": "לפני {0} שניות",
                },
            },
        },
        "day_periods": {
            "midnight": "חצות",
            "am": "לפנה״צ",
            "pm": "אחה״צ",
            "morning1": "בבוקר",
            "afternoon1": "בצהריים",
            "afternoon2": "אחר הצהריים",
            "evening1": "בערב",
            "night1": "בלילה",
            "night2": "לפנות בוקר",
        },
        "week_data": {
            "min_days": 1,
            "first_day": 0,
            "weekend_start": 5,
            "weekend_end": 6,
        },
    },
    "custom": custom_translations,
}


# === src/pendulum/locales/en_us/custom.py ===
"""
en-us custom locale file.
"""
from __future__ import annotations


translations = {
    "units": {"few_second": "a few seconds"},
    # Relative time
    "ago": "{} ago",
    "from_now": "in {}",
    "after": "{0} after",
    "before": "{0} before",
    # Ordinals
    "ordinal": {"one": "st", "two": "nd", "few": "rd", "other": "th"},
    # Date formats
    "date_formats": {
        "LTS": "h:mm:ss A",
        "LT": "h:mm A",
        "L": "MM/DD/YYYY",
        "LL": "MMMM D, YYYY",
        "LLL": "MMMM D, YYYY h:mm A",
        "LLLL": "dddd, MMMM D, YYYY h:mm A",
    },
}


# === src/pendulum/locales/en_us/__init__.py ===


# === src/pendulum/locales/en_us/locale.py ===
from __future__ import annotations

from pendulum.locales.en_us.custom import translations as custom_translations


"""
en-us locale file.

It has been generated automatically and must not be modified directly.
"""


locale = {
    "plural": lambda n: "one"
    if ((n == n and (n == 1)) and (0 == 0 and (0 == 0)))
    else "other",
    "ordinal": lambda n: "few"
    if (
        ((n % 10) == (n % 10) and ((n % 10) == 3))
        and (not ((n % 100) == (n % 100) and ((n % 100) == 13)))
    )
    else "one"
    if (
        ((n % 10) == (n % 10) and ((n % 10) == 1))
        and (not ((n % 100) == (n % 100) and ((n % 100) == 11)))
    )
    else "two"
    if (
        ((n % 10) == (n % 10) and ((n % 10) == 2))
        and (not ((n % 100) == (n % 100) and ((n % 100) == 12)))
    )
    else "other",
    "translations": {
        "days": {
            "abbreviated": {
                0: "Mon",
                1: "Tue",
                2: "Wed",
                3: "Thu",
                4: "Fri",
                5: "Sat",
                6: "Sun",
            },
            "narrow": {
                0: "M",
                1: "T",
                2: "W",
                3: "T",
                4: "F",
                5: "S",
                6: "S",
            },
            "short": {
                0: "Mo",
                1: "Tu",
                2: "We",
                3: "Th",
                4: "Fr",
                5: "Sa",
                6: "Su",
            },
            "wide": {
                0: "Monday",
                1: "Tuesday",
                2: "Wednesday",
                3: "Thursday",
                4: "Friday",
                5: "Saturday",
                6: "Sunday",
            },
        },
        "months": {
            "abbreviated": {
                1: "Jan",
                2: "Feb",
                3: "Mar",
                4: "Apr",
                5: "May",
                6: "Jun",
                7: "Jul",
                8: "Aug",
                9: "Sep",
                10: "Oct",
                11: "Nov",
                12: "Dec",
            },
            "narrow": {
                1: "J",
                2: "F",
                3: "M",
                4: "A",
                5: "M",
                6: "J",
                7: "J",
                8: "A",
                9: "S",
                10: "O",
                11: "N",
                12: "D",
            },
            "wide": {
                1: "January",
                2: "February",
                3: "March",
                4: "April",
                5: "May",
                6: "June",
                7: "July",
                8: "August",
                9: "September",
                10: "October",
                11: "November",
                12: "December",
            },
        },
        "units": {
            "year": {
                "one": "{0} year",
                "other": "{0} years",
            },
            "month": {
                "one": "{0} month",
                "other": "{0} months",
            },
            "week": {
                "one": "{0} week",
                "other": "{0} weeks",
            },
            "day": {
                "one": "{0} day",
                "other": "{0} days",
            },
            "hour": {
                "one": "{0} hour",
                "other": "{0} hours",
            },
            "minute": {
                "one": "{0} minute",
                "other": "{0} minutes",
            },
            "second": {
                "one": "{0} second",
                "other": "{0} seconds",
            },
            "microsecond": {
                "one": "{0} microsecond",
                "other": "{0} microseconds",
            },
        },
        "relative": {
            "year": {
                "future": {
                    "other": "in {0} years",
                    "one": "in {0} year",
                },
                "past": {
                    "other": "{0} years ago",
                    "one": "{0} year ago",
                },
            },
            "month": {
                "future": {
                    "other": "in {0} months",
                    "one": "in {0} month",
                },
                "past": {
                    "other": "{0} months ago",
                    "one": "{0} month ago",
                },
            },
            "week": {
                "future": {
                    "other": "in {0} weeks",
                    "one": "in {0} week",
                },
                "past": {
                    "other": "{0} weeks ago",
                    "one": "{0} week ago",
                },
            },
            "day": {
                "future": {
                    "other": "in {0} days",
                    "one": "in {0} day",
                },
                "past": {
                    "other": "{0} days ago",
                    "one": "{0} day ago",
                },
            },
            "hour": {
                "future": {
                    "other": "in {0} hours",
                    "one": "in {0} hour",
                },
                "past": {
                    "other": "{0} hours ago",
                    "one": "{0} hour ago",
                },
            },
            "minute": {
                "future": {
                    "other": "in {0} minutes",
                    "one": "in {0} minute",
                },
                "past": {
                    "other": "{0} minutes ago",
                    "one": "{0} minute ago",
                },
            },
            "second": {
                "future": {
                    "other": "in {0} seconds",
                    "one": "in {0} second",
                },
                "past": {
                    "other": "{0} seconds ago",
                    "one": "{0} second ago",
                },
            },
        },
        "day_periods": {
            "midnight": "midnight",
            "am": "AM",
            "noon": "noon",
            "pm": "PM",
            "morning1": "in the morning",
            "afternoon1": "in the afternoon",
            "evening1": "in the evening",
            "night1": "at night",
        },
        "week_data": {
            "min_days": 1,
            "first_day": 6,
            "weekend_start": 5,
            "weekend_end": 6,
        },
    },
    "custom": custom_translations,
}


# === src/pendulum/locales/nn/custom.py ===
"""
nn custom locale file.
"""
from __future__ import annotations


translations = {
    # Relative time
    "after": "{0} etter",
    "before": "{0} før",
    # Ordinals
    "ordinal": {"one": ".", "two": ".", "few": ".", "other": "."},
    # Date formats
    "date_formats": {
        "LTS": "HH:mm:ss",
        "LT": "HH:mm",
        "LLLL": "dddd Do MMMM YYYY HH:mm",
        "LLL": "Do MMMM YYYY HH:mm",
        "LL": "Do MMMM YYYY",
        "L": "DD.MM.YYYY",
    },
}


# === src/pendulum/locales/nn/__init__.py ===


# === src/pendulum/locales/nn/locale.py ===
from __future__ import annotations

from pendulum.locales.nn.custom import translations as custom_translations


"""
nn locale file.

It has been generated automatically and must not be modified directly.
"""


locale = {
    "plural": lambda n: "one" if (n == n and (n == 1)) else "other",
    "ordinal": lambda n: "other",
    "translations": {
        "days": {
            "abbreviated": {
                0: "mån.",
                1: "tys.",
                2: "ons.",
                3: "tor.",
                4: "fre.",
                5: "lau.",
                6: "søn.",
            },
            "narrow": {0: "M", 1: "T", 2: "O", 3: "T", 4: "F", 5: "L", 6: "S"},
            "short": {
                0: "må.",
                1: "ty.",
                2: "on.",
                3: "to.",
                4: "fr.",
                5: "la.",
                6: "sø.",
            },
            "wide": {
                0: "måndag",
                1: "tysdag",
                2: "onsdag",
                3: "torsdag",
                4: "fredag",
                5: "laurdag",
                6: "søndag",
            },
        },
        "months": {
            "abbreviated": {
                1: "jan.",
                2: "feb.",
                3: "mars",
                4: "apr.",
                5: "mai",
                6: "juni",
                7: "juli",
                8: "aug.",
                9: "sep.",
                10: "okt.",
                11: "nov.",
                12: "des.",
            },
            "narrow": {
                1: "J",
                2: "F",
                3: "M",
                4: "A",
                5: "M",
                6: "J",
                7: "J",
                8: "A",
                9: "S",
                10: "O",
                11: "N",
                12: "D",
            },
            "wide": {
                1: "januar",
                2: "februar",
                3: "mars",
                4: "april",
                5: "mai",
                6: "juni",
                7: "juli",
                8: "august",
                9: "september",
                10: "oktober",
                11: "november",
                12: "desember",
            },
        },
        "units": {
            "year": {"one": "{0} år", "other": "{0} år"},
            "month": {"one": "{0} månad", "other": "{0} månadar"},
            "week": {"one": "{0} veke", "other": "{0} veker"},
            "day": {"one": "{0} dag", "other": "{0} dagar"},
            "hour": {"one": "{0} time", "other": "{0} timar"},
            "minute": {"one": "{0} minutt", "other": "{0} minutt"},
            "second": {"one": "{0} sekund", "other": "{0} sekund"},
            "microsecond": {"one": "{0} mikrosekund", "other": "{0} mikrosekund"},
        },
        "relative": {
            "year": {
                "future": {"other": "om {0} år", "one": "om {0} år"},
                "past": {"other": "for {0} år sidan", "one": "for {0} år sidan"},
            },
            "month": {
                "future": {"other": "om {0} månadar", "one": "om {0} månad"},
                "past": {
                    "other": "for {0} månadar sidan",
                    "one": "for {0} månad sidan",
                },
            },
            "week": {
                "future": {"other": "om {0} veker", "one": "om {0} veke"},
                "past": {"other": "for {0} veker sidan", "one": "for {0} veke sidan"},
            },
            "day": {
                "future": {"other": "om {0} dagar", "one": "om {0} dag"},
                "past": {"other": "for {0} dagar sidan", "one": "for {0} dag sidan"},
            },
            "hour": {
                "future": {"other": "om {0} timar", "one": "om {0} time"},
                "past": {"other": "for {0} timar sidan", "one": "for {0} time sidan"},
            },
            "minute": {
                "future": {"other": "om {0} minutt", "one": "om {0} minutt"},
                "past": {
                    "other": "for {0} minutt sidan",
                    "one": "for {0} minutt sidan",
                },
            },
            "second": {
                "future": {"other": "om {0} sekund", "one": "om {0} sekund"},
                "past": {
                    "other": "for {0} sekund sidan",
                    "one": "for {0} sekund sidan",
                },
            },
        },
        "day_periods": {"am": "formiddag", "pm": "ettermiddag"},
        "week_data": {
            "min_days": 1,
            "first_day": 0,
            "weekend_start": 5,
            "weekend_end": 6,
        },
    },
    "custom": custom_translations,
}


# === src/pendulum/locales/da/custom.py ===
"""
da custom locale file.
"""
from __future__ import annotations


translations = {
    # Relative time
    "after": "{0} efter",
    "before": "{0} før",
    # Date formats
    "date_formats": {
        "LTS": "HH:mm:ss",
        "LT": "HH:mm",
        "LLLL": "dddd [d.] D. MMMM YYYY HH:mm",
        "LLL": "D. MMMM YYYY HH:mm",
        "LL": "D. MMMM YYYY",
        "L": "DD/MM/YYYY",
    },
}


# === src/pendulum/locales/da/__init__.py ===


# === src/pendulum/locales/da/locale.py ===
from __future__ import annotations

from pendulum.locales.da.custom import translations as custom_translations


"""
da locale file.

It has been generated automatically and must not be modified directly.
"""


locale = {
    "plural": lambda n: "one"
    if (
        (n == n and (n == 1))
        or ((not (0 == 0 and (0 == 0))) and (n == n and ((n == 0) or (n == 1))))
    )
    else "other",
    "ordinal": lambda n: "other",
    "translations": {
        "days": {
            "abbreviated": {
                0: "man.",
                1: "tir.",
                2: "ons.",
                3: "tor.",
                4: "fre.",
                5: "lør.",
                6: "søn.",
            },
            "narrow": {0: "M", 1: "T", 2: "O", 3: "T", 4: "F", 5: "L", 6: "S"},
            "short": {0: "ma", 1: "ti", 2: "on", 3: "to", 4: "fr", 5: "lø", 6: "sø"},
            "wide": {
                0: "mandag",
                1: "tirsdag",
                2: "onsdag",
                3: "torsdag",
                4: "fredag",
                5: "lørdag",
                6: "søndag",
            },
        },
        "months": {
            "abbreviated": {
                1: "jan.",
                2: "feb.",
                3: "mar.",
                4: "apr.",
                5: "maj",
                6: "jun.",
                7: "jul.",
                8: "aug.",
                9: "sep.",
                10: "okt.",
                11: "nov.",
                12: "dec.",
            },
            "narrow": {
                1: "J",
                2: "F",
                3: "M",
                4: "A",
                5: "M",
                6: "J",
                7: "J",
                8: "A",
                9: "S",
                10: "O",
                11: "N",
                12: "D",
            },
            "wide": {
                1: "januar",
                2: "februar",
                3: "marts",
                4: "april",
                5: "maj",
                6: "juni",
                7: "juli",
                8: "august",
                9: "september",
                10: "oktober",
                11: "november",
                12: "december",
            },
        },
        "units": {
            "year": {"one": "{0} år", "other": "{0} år"},
            "month": {"one": "{0} måned", "other": "{0} måneder"},
            "week": {"one": "{0} uge", "other": "{0} uger"},
            "day": {"one": "{0} dag", "other": "{0} dage"},
            "hour": {"one": "{0} time", "other": "{0} timer"},
            "minute": {"one": "{0} minut", "other": "{0} minutter"},
            "second": {"one": "{0} sekund", "other": "{0} sekunder"},
            "microsecond": {"one": "{0} mikrosekund", "other": "{0} mikrosekunder"},
        },
        "relative": {
            "year": {
                "future": {"other": "om {0} år", "one": "om {0} år"},
                "past": {"other": "for {0} år siden", "one": "for {0} år siden"},
            },
            "month": {
                "future": {"other": "om {0} måneder", "one": "om {0} måned"},
                "past": {
                    "other": "for {0} måneder siden",
                    "one": "for {0} måned siden",
                },
            },
            "week": {
                "future": {"other": "om {0} uger", "one": "om {0} uge"},
                "past": {"other": "for {0} uger siden", "one": "for {0} uge siden"},
            },
            "day": {
                "future": {"other": "om {0} dage", "one": "om {0} dag"},
                "past": {"other": "for {0} dage siden", "one": "for {0} dag siden"},
            },
            "hour": {
                "future": {"other": "om {0} timer", "one": "om {0} time"},
                "past": {"other": "for {0} timer siden", "one": "for {0} time siden"},
            },
            "minute": {
                "future": {"other": "om {0} minutter", "one": "om {0} minut"},
                "past": {
                    "other": "for {0} minutter siden",
                    "one": "for {0} minut siden",
                },
            },
            "second": {
                "future": {"other": "om {0} sekunder", "one": "om {0} sekund"},
                "past": {
                    "other": "for {0} sekunder siden",
                    "one": "for {0} sekund siden",
                },
            },
        },
        "day_periods": {
            "midnight": "midnat",
            "am": "AM",
            "pm": "PM",
            "morning1": "om morgenen",
            "morning2": "om formiddagen",
            "afternoon1": "om eftermiddagen",
            "evening1": "om aftenen",
            "night1": "om natten",
        },
        "week_data": {
            "min_days": 1,
            "first_day": 0,
            "weekend_start": 5,
            "weekend_end": 6,
        },
    },
    "custom": custom_translations,
}


# === src/pendulum/locales/pt_br/custom.py ===
"""
pt-br custom locale file.
"""
from __future__ import annotations


translations = {
    # Relative time
    "after": "após {0}",
    "before": "{0} atrás",
    # Date formats
    "date_formats": {
        "LTS": "HH:mm:ss",
        "LT": "HH:mm",
        "LLLL": "dddd, D [de] MMMM [de] YYYY [às] HH:mm",
        "LLL": "D [de] MMMM [de] YYYY [às] HH:mm",
        "LL": "D [de] MMMM [de] YYYY",
        "L": "DD/MM/YYYY",
    },
}


# === src/pendulum/locales/pt_br/__init__.py ===


# === src/pendulum/locales/pt_br/locale.py ===
from __future__ import annotations

from pendulum.locales.pt_br.custom import translations as custom_translations


"""
pt_br locale file.

It has been generated automatically and must not be modified directly.
"""


locale = {
    "plural": lambda n: "one"
    if ((n == n and (n >= 0 and n <= 2)) and (not (n == n and (n == 2))))
    else "other",
    "ordinal": lambda n: "other",
    "translations": {
        "days": {
            "abbreviated": {
                0: "seg",
                1: "ter",
                2: "qua",
                3: "qui",
                4: "sex",
                5: "sáb",
                6: "dom",
            },
            "narrow": {0: "S", 1: "T", 2: "Q", 3: "Q", 4: "S", 5: "S", 6: "D"},
            "short": {
                0: "seg",
                1: "ter",
                2: "qua",
                3: "qui",
                4: "sex",
                5: "sáb",
                6: "dom",
            },
            "wide": {
                0: "segunda-feira",
                1: "terça-feira",
                2: "quarta-feira",
                3: "quinta-feira",
                4: "sexta-feira",
                5: "sábado",
                6: "domingo",
            },
        },
        "months": {
            "abbreviated": {
                1: "jan",
                2: "fev",
                3: "mar",
                4: "abr",
                5: "mai",
                6: "jun",
                7: "jul",
                8: "ago",
                9: "set",
                10: "out",
                11: "nov",
                12: "dez",
            },
            "narrow": {
                1: "J",
                2: "F",
                3: "M",
                4: "A",
                5: "M",
                6: "J",
                7: "J",
                8: "A",
                9: "S",
                10: "O",
                11: "N",
                12: "D",
            },
            "wide": {
                1: "janeiro",
                2: "fevereiro",
                3: "março",
                4: "abril",
                5: "maio",
                6: "junho",
                7: "julho",
                8: "agosto",
                9: "setembro",
                10: "outubro",
                11: "novembro",
                12: "dezembro",
            },
        },
        "units": {
            "year": {"one": "{0} ano", "other": "{0} anos"},
            "month": {"one": "{0} mês", "other": "{0} meses"},
            "week": {"one": "{0} semana", "other": "{0} semanas"},
            "day": {"one": "{0} dia", "other": "{0} dias"},
            "hour": {"one": "{0} hora", "other": "{0} horas"},
            "minute": {"one": "{0} minuto", "other": "{0} minutos"},
            "second": {"one": "{0} segundo", "other": "{0} segundos"},
            "microsecond": {"one": "{0} microssegundo", "other": "{0} microssegundos"},
        },
        "relative": {
            "year": {
                "future": {"other": "em {0} anos", "one": "em {0} ano"},
                "past": {"other": "há {0} anos", "one": "há {0} ano"},
            },
            "month": {
                "future": {"other": "em {0} meses", "one": "em {0} mês"},
                "past": {"other": "há {0} meses", "one": "há {0} mês"},
            },
            "week": {
                "future": {"other": "em {0} semanas", "one": "em {0} semana"},
                "past": {"other": "há {0} semanas", "one": "há {0} semana"},
            },
            "day": {
                "future": {"other": "em {0} dias", "one": "em {0} dia"},
                "past": {"other": "há {0} dias", "one": "há {0} dia"},
            },
            "hour": {
                "future": {"other": "em {0} horas", "one": "em {0} hora"},
                "past": {"other": "há {0} horas", "one": "há {0} hora"},
            },
            "minute": {
                "future": {"other": "em {0} minutos", "one": "em {0} minuto"},
                "past": {"other": "há {0} minutos", "one": "há {0} minuto"},
            },
            "second": {
                "future": {"other": "em {0} segundos", "one": "em {0} segundo"},
                "past": {"other": "há {0} segundos", "one": "há {0} segundo"},
            },
        },
        "day_periods": {
            "midnight": "meia-noite",
            "am": "AM",
            "noon": "meio-dia",
            "pm": "PM",
            "morning1": "da manhã",
            "afternoon1": "da tarde",
            "evening1": "da noite",
            "night1": "da madrugada",
        },
        "week_data": {
            "min_days": 1,
            "first_day": 6,
            "weekend_start": 5,
            "weekend_end": 6,
        },
    },
    "custom": custom_translations,
}


# === src/pendulum/locales/ja/custom.py ===
"""
ja custom locale file.
"""
from __future__ import annotations


translations = {
    "units": {"few_second": "数秒"},
    # Relative time
    "ago": "{} 前に",
    "from_now": "今から {}",
    "after": "{0} 後",
    "before": "{0} 前",
    # Date formats
    "date_formats": {
        "LTS": "h:mm:ss A",
        "LT": "h:mm A",
        "L": "MM/DD/YYYY",
        "LL": "MMMM D, YYYY",
        "LLL": "MMMM D, YYYY h:mm A",
        "LLLL": "dddd, MMMM D, YYYY h:mm A",
    },
}


# === src/pendulum/locales/ja/__init__.py ===


# === src/pendulum/locales/ja/locale.py ===
from __future__ import annotations

from pendulum.locales.ja.custom import translations as custom_translations


"""
ja locale file.

It has been generated automatically and must not be modified directly.
"""


locale = {
    "plural": lambda n: "other",
    "ordinal": lambda n: "other",
    "translations": {
        "days": {
            "abbreviated": {
                0: "月",
                1: "火",
                2: "水",
                3: "木",
                4: "金",
                5: "土",
                6: "日",
            },
            "narrow": {
                0: "月",
                1: "火",
                2: "水",
                3: "木",
                4: "金",
                5: "土",
                6: "日",
            },
            "short": {
                0: "月",
                1: "火",
                2: "水",
                3: "木",
                4: "金",
                5: "土",
                6: "日",
            },
            "wide": {
                0: "月曜日",
                1: "火曜日",
                2: "水曜日",
                3: "木曜日",
                4: "金曜日",
                5: "土曜日",
                6: "日曜日",
            },
        },
        "months": {
            "abbreviated": {
                1: "1月",
                2: "2月",
                3: "3月",
                4: "4月",
                5: "5月",
                6: "6月",
                7: "7月",
                8: "8月",
                9: "9月",
                10: "10月",
                11: "11月",
                12: "12月",
            },
            "narrow": {
                1: "1",
                2: "2",
                3: "3",
                4: "4",
                5: "5",
                6: "6",
                7: "7",
                8: "8",
                9: "9",
                10: "10",
                11: "11",
                12: "12",
            },
            "wide": {
                1: "1月",
                2: "2月",
                3: "3月",
                4: "4月",
                5: "5月",
                6: "6月",
                7: "7月",
                8: "8月",
                9: "9月",
                10: "10月",
                11: "11月",
                12: "12月",
            },
        },
        "units": {
            "year": {
                "other": "{0} 年",
            },
            "month": {
                "other": "{0} か月",
            },
            "week": {
                "other": "{0} 週間",
            },
            "day": {
                "other": "{0} 日",
            },
            "hour": {
                "other": "{0} 時間",
            },
            "minute": {
                "other": "{0} 分",
            },
            "second": {
                "other": "{0} 秒",
            },
            "microsecond": {
                "other": "{0} マイクロ秒",
            },
        },
        "relative": {
            "year": {
                "future": {
                    "other": "{0} 年後",
                },
                "past": {
                    "other": "{0} 年前",
                },
            },
            "month": {
                "future": {
                    "other": "{0} か月後",
                },
                "past": {
                    "other": "{0} か月前",
                },
            },
            "week": {
                "future": {
                    "other": "{0} 週間後",
                },
                "past": {
                    "other": "{0} 週間前",
                },
            },
            "day": {
                "future": {
                    "other": "{0} 日後",
                },
                "past": {
                    "other": "{0} 日前",
                },
            },
            "hour": {
                "future": {
                    "other": "{0} 時間後",
                },
                "past": {
                    "other": "{0} 時間前",
                },
            },
            "minute": {
                "future": {
                    "other": "{0} 分後",
                },
                "past": {
                    "other": "{0} 分前",
                },
            },
            "second": {
                "future": {
                    "other": "{0} 秒後",
                },
                "past": {
                    "other": "{0} 秒前",
                },
            },
        },
        "day_periods": {
            "midnight": "真夜中",
            "am": "午前",
            "noon": "正午",
            "pm": "午後",
            "morning1": "朝",
            "afternoon1": "昼",
            "evening1": "夕方",
            "night1": "夜",
            "night2": "夜中",
        },
        "week_data": {
            "min_days": 1,
            "first_day": 0,
            "weekend_start": 5,
            "weekend_end": 6,
        },
    },
    "custom": custom_translations,
}


# === src/pendulum/locales/it/custom.py ===
"""
it custom locale file.
"""
from __future__ import annotations


translations = {
    "units": {"few_second": "alcuni secondi"},
    # Relative Time
    "ago": "{0} fa",
    "from_now": "in {0}",
    "after": "{0} dopo",
    "before": "{0} prima",
    # Ordinals
    "ordinal": {"other": "°"},
    # Date formats
    "date_formats": {
        "LTS": "H:mm:ss",
        "LT": "H:mm",
        "L": "DD/MM/YYYY",
        "LL": "D MMMM YYYY",
        "LLL": "D MMMM YYYY [alle] H:mm",
        "LLLL": "dddd, D MMMM YYYY [alle] H:mm",
    },
}


# === src/pendulum/locales/it/__init__.py ===


# === src/pendulum/locales/it/locale.py ===
from __future__ import annotations

from pendulum.locales.it.custom import translations as custom_translations


"""
it locale file.

It has been generated automatically and must not be modified directly.
"""


locale = {
    "plural": lambda n: "one"
    if ((n == n and (n == 1)) and (0 == 0 and (0 == 0)))
    else "other",
    "ordinal": lambda n: "many"
    if (n == n and ((n == 11) or (n == 8) or (n == 80) or (n == 800)))
    else "other",
    "translations": {
        "days": {
            "abbreviated": {
                0: "lun",
                1: "mar",
                2: "mer",
                3: "gio",
                4: "ven",
                5: "sab",
                6: "dom",
            },
            "narrow": {0: "L", 1: "M", 2: "M", 3: "G", 4: "V", 5: "S", 6: "D"},
            "short": {
                0: "lun",
                1: "mar",
                2: "mer",
                3: "gio",
                4: "ven",
                5: "sab",
                6: "dom",
            },
            "wide": {
                0: "lunedì",
                1: "martedì",
                2: "mercoledì",
                3: "giovedì",
                4: "venerdì",
                5: "sabato",
                6: "domenica",
            },
        },
        "months": {
            "abbreviated": {
                1: "gen",
                2: "feb",
                3: "mar",
                4: "apr",
                5: "mag",
                6: "giu",
                7: "lug",
                8: "ago",
                9: "set",
                10: "ott",
                11: "nov",
                12: "dic",
            },
            "narrow": {
                1: "G",
                2: "F",
                3: "M",
                4: "A",
                5: "M",
                6: "G",
                7: "L",
                8: "A",
                9: "S",
                10: "O",
                11: "N",
                12: "D",
            },
            "wide": {
                1: "gennaio",
                2: "febbraio",
                3: "marzo",
                4: "aprile",
                5: "maggio",
                6: "giugno",
                7: "luglio",
                8: "agosto",
                9: "settembre",
                10: "ottobre",
                11: "novembre",
                12: "dicembre",
            },
        },
        "units": {
            "year": {"one": "{0} anno", "other": "{0} anni"},
            "month": {"one": "{0} mese", "other": "{0} mesi"},
            "week": {"one": "{0} settimana", "other": "{0} settimane"},
            "day": {"one": "{0} giorno", "other": "{0} giorni"},
            "hour": {"one": "{0} ora", "other": "{0} ore"},
            "minute": {"one": "{0} minuto", "other": "{0} minuti"},
            "second": {"one": "{0} secondo", "other": "{0} secondi"},
            "microsecond": {"one": "{0} microsecondo", "other": "{0} microsecondi"},
        },
        "relative": {
            "year": {
                "future": {"other": "tra {0} anni", "one": "tra {0} anno"},
                "past": {"other": "{0} anni fa", "one": "{0} anno fa"},
            },
            "month": {
                "future": {"other": "tra {0} mesi", "one": "tra {0} mese"},
                "past": {"other": "{0} mesi fa", "one": "{0} mese fa"},
            },
            "week": {
                "future": {"other": "tra {0} settimane", "one": "tra {0} settimana"},
                "past": {"other": "{0} settimane fa", "one": "{0} settimana fa"},
            },
            "day": {
                "future": {"other": "tra {0} giorni", "one": "tra {0} giorno"},
                "past": {"other": "{0} giorni fa", "one": "{0} giorno fa"},
            },
            "hour": {
                "future": {"other": "tra {0} ore", "one": "tra {0} ora"},
                "past": {"other": "{0} ore fa", "one": "{0} ora fa"},
            },
            "minute": {
                "future": {"other": "tra {0} minuti", "one": "tra {0} minuto"},
                "past": {"other": "{0} minuti fa", "one": "{0} minuto fa"},
            },
            "second": {
                "future": {"other": "tra {0} secondi", "one": "tra {0} secondo"},
                "past": {"other": "{0} secondi fa", "one": "{0} secondo fa"},
            },
        },
        "day_periods": {
            "midnight": "mezzanotte",
            "am": "AM",
            "noon": "mezzogiorno",
            "pm": "PM",
            "morning1": "di mattina",
            "afternoon1": "del pomeriggio",
            "evening1": "di sera",
            "night1": "di notte",
        },
        "week_data": {
            "min_days": 1,
            "first_day": 0,
            "weekend_start": 5,
            "weekend_end": 6,
        },
    },
    "custom": custom_translations,
}


# === src/pendulum/locales/cs/custom.py ===
"""
cs custom locale file.
"""
from __future__ import annotations


translations = {
    "units": {"few_second": "pár vteřin"},
    # Relative time
    "ago": "{} zpět",
    "from_now": "za {}",
    "after": "{0} po",
    "before": "{0} zpět",
    # Ordinals
    "ordinal": {"one": ".", "two": ".", "few": ".", "other": "."},
    # Date formats
    "date_formats": {
        "LTS": "h:mm:ss",
        "LT": "h:mm",
        "L": "DD. M. YYYY",
        "LL": "D. MMMM, YYYY",
        "LLL": "D. MMMM, YYYY h:mm",
        "LLLL": "dddd, D. MMMM, YYYY h:mm",
    },
}


# === src/pendulum/locales/cs/__init__.py ===


# === src/pendulum/locales/cs/locale.py ===
from __future__ import annotations

from pendulum.locales.cs.custom import translations as custom_translations


"""
cs locale file.

It has been generated automatically and must not be modified directly.
"""


locale = {
    "plural": lambda n: "few"
    if ((n == n and (n >= 2 and n <= 4)) and (0 == 0 and (0 == 0)))
    else "many"
    if (not (0 == 0 and (0 == 0)))
    else "one"
    if ((n == n and (n == 1)) and (0 == 0 and (0 == 0)))
    else "other",
    "ordinal": lambda n: "other",
    "translations": {
        "days": {
            "abbreviated": {
                0: "po",
                1: "út",
                2: "st",
                3: "čt",
                4: "pá",
                5: "so",
                6: "ne",
            },
            "narrow": {
                0: "P",
                1: "Ú",
                2: "S",
                3: "Č",
                4: "P",
                5: "S",
                6: "N",
            },
            "short": {
                0: "po",
                1: "út",
                2: "st",
                3: "čt",
                4: "pá",
                5: "so",
                6: "ne",
            },
            "wide": {
                0: "pondělí",
                1: "úterý",
                2: "středa",
                3: "čtvrtek",
                4: "pátek",
                5: "sobota",
                6: "neděle",
            },
        },
        "months": {
            "abbreviated": {
                1: "led",
                2: "úno",
                3: "bře",
                4: "dub",
                5: "kvě",
                6: "čvn",
                7: "čvc",
                8: "srp",
                9: "zář",
                10: "říj",
                11: "lis",
                12: "pro",
            },
            "narrow": {
                1: "1",
                2: "2",
                3: "3",
                4: "4",
                5: "5",
                6: "6",
                7: "7",
                8: "8",
                9: "9",
                10: "10",
                11: "11",
                12: "12",
            },
            "wide": {
                1: "ledna",
                2: "února",
                3: "března",
                4: "dubna",
                5: "května",
                6: "června",
                7: "července",
                8: "srpna",
                9: "září",
                10: "října",
                11: "listopadu",
                12: "prosince",
            },
        },
        "units": {
            "year": {
                "one": "{0} rok",
                "few": "{0} roky",
                "many": "{0} roku",
                "other": "{0} let",
            },
            "month": {
                "one": "{0} měsíc",
                "few": "{0} měsíce",
                "many": "{0} měsíce",
                "other": "{0} měsíců",
            },
            "week": {
                "one": "{0} týden",
                "few": "{0} týdny",
                "many": "{0} týdne",
                "other": "{0} týdnů",
            },
            "day": {
                "one": "{0} den",
                "few": "{0} dny",
                "many": "{0} dne",
                "other": "{0} dní",
            },
            "hour": {
                "one": "{0} hodina",
                "few": "{0} hodiny",
                "many": "{0} hodiny",
                "other": "{0} hodin",
            },
            "minute": {
                "one": "{0} minuta",
                "few": "{0} minuty",
                "many": "{0} minuty",
                "other": "{0} minut",
            },
            "second": {
                "one": "{0} sekunda",
                "few": "{0} sekundy",
                "many": "{0} sekundy",
                "other": "{0} sekund",
            },
            "microsecond": {
                "one": "{0} mikrosekunda",
                "few": "{0} mikrosekundy",
                "many": "{0} mikrosekundy",
                "other": "{0} mikrosekund",
            },
        },
        "relative": {
            "year": {
                "future": {
                    "other": "za {0} let",
                    "one": "za {0} rok",
                    "few": "za {0} roky",
                    "many": "za {0} roku",
                },
                "past": {
                    "other": "před {0} lety",
                    "one": "před {0} rokem",
                    "few": "před {0} lety",
                    "many": "před {0} roku",
                },
            },
            "month": {
                "future": {
                    "other": "za {0} měsíců",
                    "one": "za {0} měsíc",
                    "few": "za {0} měsíce",
                    "many": "za {0} měsíce",
                },
                "past": {
                    "other": "před {0} měsíci",
                    "one": "před {0} měsícem",
                    "few": "před {0} měsíci",
                    "many": "před {0} měsíce",
                },
            },
            "week": {
                "future": {
                    "other": "za {0} týdnů",
                    "one": "za {0} týden",
                    "few": "za {0} týdny",
                    "many": "za {0} týdne",
                },
                "past": {
                    "other": "před {0} týdny",
                    "one": "před {0} týdnem",
                    "few": "před {0} týdny",
                    "many": "před {0} týdne",
                },
            },
            "day": {
                "future": {
                    "other": "za {0} dní",
                    "one": "za {0} den",
                    "few": "za {0} dny",
                    "many": "za {0} dne",
                },
                "past": {
                    "other": "před {0} dny",
                    "one": "před {0} dnem",
                    "few": "před {0} dny",
                    "many": "před {0} dne",
                },
            },
            "hour": {
                "future": {
                    "other": "za {0} hodin",
                    "one": "za {0} hodinu",
                    "few": "za {0} hodiny",
                    "many": "za {0} hodiny",
                },
                "past": {
                    "other": "před {0} hodinami",
                    "one": "před {0} hodinou",
                    "few": "před {0} hodinami",
                    "many": "před {0} hodiny",
                },
            },
            "minute": {
                "future": {
                    "other": "za {0} minut",
                    "one": "za {0} minutu",
                    "few": "za {0} minuty",
                    "many": "za {0} minuty",
                },
                "past": {
                    "other": "před {0} minutami",
                    "one": "před {0} minutou",
                    "few": "před {0} minutami",
                    "many": "před {0} minuty",
                },
            },
            "second": {
                "future": {
                    "other": "za {0} sekund",
                    "one": "za {0} sekundu",
                    "few": "za {0} sekundy",
                    "many": "za {0} sekundy",
                },
                "past": {
                    "other": "před {0} sekundami",
                    "one": "před {0} sekundou",
                    "few": "před {0} sekundami",
                    "many": "před {0} sekundy",
                },
            },
        },
        "day_periods": {
            "midnight": "půlnoc",
            "am": "dop.",
            "noon": "poledne",
            "pm": "odp.",
            "morning1": "ráno",
            "morning2": "dopoledne",
            "afternoon1": "odpoledne",
            "evening1": "večer",
            "night1": "v noci",
        },
        "week_data": {
            "min_days": 1,
            "first_day": 0,
            "weekend_start": 5,
            "weekend_end": 6,
        },
    },
    "custom": custom_translations,
}


# === src/pendulum/locales/ru/custom.py ===
"""
ru custom locale file.
"""
from __future__ import annotations


translations = {
    # Relative time
    "ago": "{} назад",
    "from_now": "через {}",
    "after": "{0} после",
    "before": "{0} до",
    # Date formats
    "date_formats": {
        "LTS": "HH:mm:ss",
        "LT": "HH:mm",
        "L": "DD.MM.YYYY",
        "LL": "D MMMM YYYY г.",
        "LLL": "D MMMM YYYY г., HH:mm",
        "LLLL": "dddd, D MMMM YYYY г., HH:mm",
    },
}


# === src/pendulum/locales/ru/__init__.py ===


# === src/pendulum/locales/ru/locale.py ===
from __future__ import annotations

from pendulum.locales.ru.custom import translations as custom_translations


"""
ru locale file.

It has been generated automatically and must not be modified directly.
"""


locale = {
    "plural": lambda n: "few"
    if (
        (
            (0 == 0 and (0 == 0))
            and ((n % 10) == (n % 10) and ((n % 10) >= 2 and (n % 10) <= 4))
        )
        and (not ((n % 100) == (n % 100) and ((n % 100) >= 12 and (n % 100) <= 14)))
    )
    else "many"
    if (
        (
            ((0 == 0 and (0 == 0)) and ((n % 10) == (n % 10) and ((n % 10) == 0)))
            or (
                (0 == 0 and (0 == 0))
                and ((n % 10) == (n % 10) and ((n % 10) >= 5 and (n % 10) <= 9))
            )
        )
        or (
            (0 == 0 and (0 == 0))
            and ((n % 100) == (n % 100) and ((n % 100) >= 11 and (n % 100) <= 14))
        )
    )
    else "one"
    if (
        ((0 == 0 and (0 == 0)) and ((n % 10) == (n % 10) and ((n % 10) == 1)))
        and (not ((n % 100) == (n % 100) and ((n % 100) == 11)))
    )
    else "other",
    "ordinal": lambda n: "other",
    "translations": {
        "days": {
            "abbreviated": {
                0: "пн",
                1: "вт",
                2: "ср",
                3: "чт",
                4: "пт",
                5: "сб",
                6: "вс",
            },
            "narrow": {0: "пн", 1: "вт", 2: "ср", 3: "чт", 4: "пт", 5: "сб", 6: "вс"},
            "short": {0: "пн", 1: "вт", 2: "ср", 3: "чт", 4: "пт", 5: "сб", 6: "вс"},
            "wide": {
                0: "понедельник",
                1: "вторник",
                2: "среда",
                3: "четверг",
                4: "пятница",
                5: "суббота",
                6: "воскресенье",
            },
        },
        "months": {
            "abbreviated": {
                1: "янв.",
                2: "февр.",
                3: "мар.",
                4: "апр.",
                5: "мая",
                6: "июн.",
                7: "июл.",
                8: "авг.",
                9: "сент.",
                10: "окт.",
                11: "нояб.",
                12: "дек.",
            },
            "narrow": {
                1: "Я",
                2: "Ф",
                3: "М",
                4: "А",
                5: "М",
                6: "И",
                7: "И",
                8: "А",
                9: "С",
                10: "О",
                11: "Н",
                12: "Д",
            },
            "wide": {
                1: "января",
                2: "февраля",
                3: "марта",
                4: "апреля",
                5: "мая",
                6: "июня",
                7: "июля",
                8: "августа",
                9: "сентября",
                10: "октября",
                11: "ноября",
                12: "декабря",
            },
        },
        "units": {
            "year": {
                "one": "{0} год",
                "few": "{0} года",
                "many": "{0} лет",
                "other": "{0} года",
            },
            "month": {
                "one": "{0} месяц",
                "few": "{0} месяца",
                "many": "{0} месяцев",
                "other": "{0} месяца",
            },
            "week": {
                "one": "{0} неделя",
                "few": "{0} недели",
                "many": "{0} недель",
                "other": "{0} недели",
            },
            "day": {
                "one": "{0} день",
                "few": "{0} дня",
                "many": "{0} дней",
                "other": "{0} дня",
            },
            "hour": {
                "one": "{0} час",
                "few": "{0} часа",
                "many": "{0} часов",
                "other": "{0} часа",
            },
            "minute": {
                "one": "{0} минута",
                "few": "{0} минуты",
                "many": "{0} минут",
                "other": "{0} минуты",
            },
            "second": {
                "one": "{0} секунда",
                "few": "{0} секунды",
                "many": "{0} секунд",
                "other": "{0} секунды",
            },
            "microsecond": {
                "one": "{0} микросекунда",
                "few": "{0} микросекунды",
                "many": "{0} микросекунд",
                "other": "{0} микросекунды",
            },
        },
        "relative": {
            "year": {
                "future": {
                    "other": "через {0} года",
                    "one": "через {0} год",
                    "few": "через {0} года",
                    "many": "через {0} лет",
                },
                "past": {
                    "other": "{0} года назад",
                    "one": "{0} год назад",
                    "few": "{0} года назад",
                    "many": "{0} лет назад",
                },
            },
            "month": {
                "future": {
                    "other": "через {0} месяца",
                    "one": "через {0} месяц",
                    "few": "через {0} месяца",
                    "many": "через {0} месяцев",
                },
                "past": {
                    "other": "{0} месяца назад",
                    "one": "{0} месяц назад",
                    "few": "{0} месяца назад",
                    "many": "{0} месяцев назад",
                },
            },
            "week": {
                "future": {
                    "other": "через {0} недели",
                    "one": "через {0} неделю",
                    "few": "через {0} недели",
                    "many": "через {0} недель",
                },
                "past": {
                    "other": "{0} недели назад",
                    "one": "{0} неделю назад",
                    "few": "{0} недели назад",
                    "many": "{0} недель назад",
                },
            },
            "day": {
                "future": {
                    "other": "через {0} дня",
                    "one": "через {0} день",
                    "few": "через {0} дня",
                    "many": "через {0} дней",
                },
                "past": {
                    "other": "{0} дня назад",
                    "one": "{0} день назад",
                    "few": "{0} дня назад",
                    "many": "{0} дней назад",
                },
            },
            "hour": {
                "future": {
                    "other": "через {0} часа",
                    "one": "через {0} час",
                    "few": "через {0} часа",
                    "many": "через {0} часов",
                },
                "past": {
                    "other": "{0} часа назад",
                    "one": "{0} час назад",
                    "few": "{0} часа назад",
                    "many": "{0} часов назад",
                },
            },
            "minute": {
                "future": {
                    "other": "через {0} минуты",
                    "one": "через {0} минуту",
                    "few": "через {0} минуты",
                    "many": "через {0} минут",
                },
                "past": {
                    "other": "{0} минуты назад",
                    "one": "{0} минуту назад",
                    "few": "{0} минуты назад",
                    "many": "{0} минут назад",
                },
            },
            "second": {
                "future": {
                    "other": "через {0} секунды",
                    "one": "через {0} секунду",
                    "few": "через {0} секунды",
                    "many": "через {0} секунд",
                },
                "past": {
                    "other": "{0} секунды назад",
                    "one": "{0} секунду назад",
                    "few": "{0} секунды назад",
                    "many": "{0} секунд назад",
                },
            },
        },
        "day_periods": {
            "midnight": "полночь",
            "am": "AM",
            "noon": "полдень",
            "pm": "PM",
            "morning1": "утра",
            "afternoon1": "дня",
            "evening1": "вечера",
            "night1": "ночи",
        },
        "week_data": {
            "min_days": 1,
            "first_day": 0,
            "weekend_start": 5,
            "weekend_end": 6,
        },
    },
    "custom": custom_translations,
}


# === src/pendulum/locales/zh/custom.py ===
"""
zh custom locale file.
"""
from __future__ import annotations


translations = {
    # Relative time
    "after": "{time}后",
    "before": "{time}前",
    # Date formats
    "date_formats": {
        "LTS": "Ah点m分s秒",
        "LT": "Ah点mm分",
        "LLLL": "YYYY年MMMD日ddddAh点mm分",
        "LLL": "YYYY年MMMD日Ah点mm分",
        "LL": "YYYY年MMMD日",
        "L": "YYYY-MM-DD",
    },
}


# === src/pendulum/locales/zh/__init__.py ===


# === src/pendulum/locales/zh/locale.py ===
from __future__ import annotations

from pendulum.locales.zh.custom import translations as custom_translations


"""
zh locale file.

It has been generated automatically and must not be modified directly.
"""


locale = {
    "plural": lambda n: "other",
    "ordinal": lambda n: "other",
    "translations": {
        "days": {
            "abbreviated": {
                0: "周一",
                1: "周二",
                2: "周三",
                3: "周四",
                4: "周五",
                5: "周六",
                6: "周日",
            },
            "narrow": {0: "一", 1: "二", 2: "三", 3: "四", 4: "五", 5: "六", 6: "日"},
            "short": {0: "周一", 1: "周二", 2: "周三", 3: "周四", 4: "周五", 5: "周六", 6: "周日"},
            "wide": {
                0: "星期一",
                1: "星期二",
                2: "星期三",
                3: "星期四",
                4: "星期五",
                5: "星期六",
                6: "星期日",
            },
        },
        "months": {
            "abbreviated": {
                1: "1月",
                2: "2月",
                3: "3月",
                4: "4月",
                5: "5月",
                6: "6月",
                7: "7月",
                8: "8月",
                9: "9月",
                10: "10月",
                11: "11月",
                12: "12月",
            },
            "narrow": {
                1: "1",
                2: "2",
                3: "3",
                4: "4",
                5: "5",
                6: "6",
                7: "7",
                8: "8",
                9: "9",
                10: "10",
                11: "11",
                12: "12",
            },
            "wide": {
                1: "一月",
                2: "二月",
                3: "三月",
                4: "四月",
                5: "五月",
                6: "六月",
                7: "七月",
                8: "八月",
                9: "九月",
                10: "十月",
                11: "十一月",
                12: "十二月",
            },
        },
        "units": {
            "year": {"other": "{0}年"},
            "month": {"other": "{0}个月"},
            "week": {"other": "{0}周"},
            "day": {"other": "{0}天"},
            "hour": {"other": "{0}小时"},
            "minute": {"other": "{0}分钟"},
            "second": {"other": "{0}秒钟"},
            "microsecond": {"other": "{0}微秒"},
        },
        "relative": {
            "year": {"future": {"other": "{0}年后"}, "past": {"other": "{0}年前"}},
            "month": {"future": {"other": "{0}个月后"}, "past": {"other": "{0}个月前"}},
            "week": {"future": {"other": "{0}周后"}, "past": {"other": "{0}周前"}},
            "day": {"future": {"other": "{0}天后"}, "past": {"other": "{0}天前"}},
            "hour": {"future": {"other": "{0}小时后"}, "past": {"other": "{0}小时前"}},
            "minute": {"future": {"other": "{0}分钟后"}, "past": {"other": "{0}分钟前"}},
            "second": {"future": {"other": "{0}秒钟后"}, "past": {"other": "{0}秒钟前"}},
        },
        "day_periods": {
            "midnight": "午夜",
            "am": "上午",
            "pm": "下午",
            "morning1": "清晨",
            "morning2": "上午",
            "afternoon1": "下午",
            "afternoon2": "下午",
            "evening1": "晚上",
            "night1": "凌晨",
        },
        "week_data": {
            "min_days": 1,
            "first_day": 0,
            "weekend_start": 5,
            "weekend_end": 6,
        },
    },
    "custom": custom_translations,
}


# === src/pendulum/locales/en_gb/custom.py ===
"""
en-gb custom locale file.
"""
from __future__ import annotations


translations = {
    "units": {"few_second": "a few seconds"},
    # Relative time
    "ago": "{} ago",
    "from_now": "in {}",
    "after": "{0} after",
    "before": "{0} before",
    # Ordinals
    "ordinal": {"one": "st", "two": "nd", "few": "rd", "other": "th"},
    # Date formats
    "date_formats": {
        "LTS": "HH:mm:ss",
        "LT": "HH:mm",
        "L": "DD/MM/YYYY",
        "LL": "D MMMM YYYY",
        "LLL": "D MMMM YYYY HH:mm",
        "LLLL": "dddd, D MMMM YYYY HH:mm",
    },
}


# === src/pendulum/locales/en_gb/__init__.py ===


# === src/pendulum/locales/en_gb/locale.py ===
from __future__ import annotations

from pendulum.locales.en_gb.custom import translations as custom_translations


"""
en-gb locale file.

It has been generated automatically and must not be modified directly.
"""


locale = {
    "plural": lambda n: "one"
    if ((n == n and (n == 1)) and (0 == 0 and (0 == 0)))
    else "other",
    "ordinal": lambda n: "few"
    if (
        ((n % 10) == (n % 10) and ((n % 10) == 3))
        and (not ((n % 100) == (n % 100) and ((n % 100) == 13)))
    )
    else "one"
    if (
        ((n % 10) == (n % 10) and ((n % 10) == 1))
        and (not ((n % 100) == (n % 100) and ((n % 100) == 11)))
    )
    else "two"
    if (
        ((n % 10) == (n % 10) and ((n % 10) == 2))
        and (not ((n % 100) == (n % 100) and ((n % 100) == 12)))
    )
    else "other",
    "translations": {
        "days": {
            "abbreviated": {
                0: "Mon",
                1: "Tue",
                2: "Wed",
                3: "Thu",
                4: "Fri",
                5: "Sat",
                6: "Sun",
            },
            "narrow": {
                0: "M",
                1: "T",
                2: "W",
                3: "T",
                4: "F",
                5: "S",
                6: "S",
            },
            "short": {
                0: "Mo",
                1: "Tu",
                2: "We",
                3: "Th",
                4: "Fr",
                5: "Sa",
                6: "Su",
            },
            "wide": {
                0: "Monday",
                1: "Tuesday",
                2: "Wednesday",
                3: "Thursday",
                4: "Friday",
                5: "Saturday",
                6: "Sunday",
            },
        },
        "months": {
            "abbreviated": {
                1: "Jan",
                2: "Feb",
                3: "Mar",
                4: "Apr",
                5: "May",
                6: "Jun",
                7: "Jul",
                8: "Aug",
                9: "Sept",
                10: "Oct",
                11: "Nov",
                12: "Dec",
            },
            "narrow": {
                1: "J",
                2: "F",
                3: "M",
                4: "A",
                5: "M",
                6: "J",
                7: "J",
                8: "A",
                9: "S",
                10: "O",
                11: "N",
                12: "D",
            },
            "wide": {
                1: "January",
                2: "February",
                3: "March",
                4: "April",
                5: "May",
                6: "June",
                7: "July",
                8: "August",
                9: "September",
                10: "October",
                11: "November",
                12: "December",
            },
        },
        "units": {
            "year": {
                "one": "{0} year",
                "other": "{0} years",
            },
            "month": {
                "one": "{0} month",
                "other": "{0} months",
            },
            "week": {
                "one": "{0} week",
                "other": "{0} weeks",
            },
            "day": {
                "one": "{0} day",
                "other": "{0} days",
            },
            "hour": {
                "one": "{0} hour",
                "other": "{0} hours",
            },
            "minute": {
                "one": "{0} minute",
                "other": "{0} minutes",
            },
            "second": {
                "one": "{0} second",
                "other": "{0} seconds",
            },
            "microsecond": {
                "one": "{0} microsecond",
                "other": "{0} microseconds",
            },
        },
        "relative": {
            "year": {
                "future": {
                    "other": "in {0} years",
                    "one": "in {0} year",
                },
                "past": {
                    "other": "{0} years ago",
                    "one": "{0} year ago",
                },
            },
            "month": {
                "future": {
                    "other": "in {0} months",
                    "one": "in {0} month",
                },
                "past": {
                    "other": "{0} months ago",
                    "one": "{0} month ago",
                },
            },
            "week": {
                "future": {
                    "other": "in {0} weeks",
                    "one": "in {0} week",
                },
                "past": {
                    "other": "{0} weeks ago",
                    "one": "{0} week ago",
                },
            },
            "day": {
                "future": {
                    "other": "in {0} days",
                    "one": "in {0} day",
                },
                "past": {
                    "other": "{0} days ago",
                    "one": "{0} day ago",
                },
            },
            "hour": {
                "future": {
                    "other": "in {0} hours",
                    "one": "in {0} hour",
                },
                "past": {
                    "other": "{0} hours ago",
                    "one": "{0} hour ago",
                },
            },
            "minute": {
                "future": {
                    "other": "in {0} minutes",
                    "one": "in {0} minute",
                },
                "past": {
                    "other": "{0} minutes ago",
                    "one": "{0} minute ago",
                },
            },
            "second": {
                "future": {
                    "other": "in {0} seconds",
                    "one": "in {0} second",
                },
                "past": {
                    "other": "{0} seconds ago",
                    "one": "{0} second ago",
                },
            },
        },
        "day_periods": {
            "midnight": "midnight",
            "am": "am",
            "noon": "noon",
            "pm": "pm",
            "morning1": "in the morning",
            "afternoon1": "in the afternoon",
            "evening1": "in the evening",
            "night1": "at night",
        },
        "week_data": {
            "min_days": 4,
            "first_day": 0,
            "weekend_start": 5,
            "weekend_end": 6,
        },
    },
    "custom": custom_translations,
}


# === src/pendulum/locales/nl/custom.py ===
"""
nl custom locale file.
"""
from __future__ import annotations


translations = {
    "units": {"few_second": "enkele seconden"},
    # Relative time
    "ago": "{} geleden",
    "from_now": "over {}",
    "after": "{0} later",
    "before": "{0} eerder",
    # Ordinals
    "ordinal": {"other": "e"},
    # Date formats
    "date_formats": {
        "L": "DD-MM-YYYY",
        "LL": "D MMMM YYYY",
        "LLL": "D MMMM YYYY HH:mm",
        "LLLL": "dddd D MMMM YYYY HH:mm",
        "LT": "HH:mm",
        "LTS": "HH:mm:ss",
    },
}


# === src/pendulum/locales/nl/__init__.py ===


# === src/pendulum/locales/nl/locale.py ===
from __future__ import annotations

from pendulum.locales.nl.custom import translations as custom_translations


"""
nl locale file.

It has been generated automatically and must not be modified directly.
"""


locale = {
    "plural": lambda n: "one"
    if ((n == n and (n == 1)) and (0 == 0 and (0 == 0)))
    else "other",
    "ordinal": lambda n: "other",
    "translations": {
        "days": {
            "abbreviated": {
                0: "ma",
                1: "di",
                2: "wo",
                3: "do",
                4: "vr",
                5: "za",
                6: "zo",
            },
            "narrow": {0: "M", 1: "D", 2: "W", 3: "D", 4: "V", 5: "Z", 6: "Z"},
            "short": {0: "ma", 1: "di", 2: "wo", 3: "do", 4: "vr", 5: "za", 6: "zo"},
            "wide": {
                0: "maandag",
                1: "dinsdag",
                2: "woensdag",
                3: "donderdag",
                4: "vrijdag",
                5: "zaterdag",
                6: "zondag",
            },
        },
        "months": {
            "abbreviated": {
                1: "jan.",
                2: "feb.",
                3: "mrt.",
                4: "apr.",
                5: "mei",
                6: "jun.",
                7: "jul.",
                8: "aug.",
                9: "sep.",
                10: "okt.",
                11: "nov.",
                12: "dec.",
            },
            "narrow": {
                1: "J",
                2: "F",
                3: "M",
                4: "A",
                5: "M",
                6: "J",
                7: "J",
                8: "A",
                9: "S",
                10: "O",
                11: "N",
                12: "D",
            },
            "wide": {
                1: "januari",
                2: "februari",
                3: "maart",
                4: "april",
                5: "mei",
                6: "juni",
                7: "juli",
                8: "augustus",
                9: "september",
                10: "oktober",
                11: "november",
                12: "december",
            },
        },
        "units": {
            "year": {"one": "{0} jaar", "other": "{0} jaar"},
            "month": {"one": "{0} maand", "other": "{0} maanden"},
            "week": {"one": "{0} week", "other": "{0} weken"},
            "day": {"one": "{0} dag", "other": "{0} dagen"},
            "hour": {"one": "{0} uur", "other": "{0} uur"},
            "minute": {"one": "{0} minuut", "other": "{0} minuten"},
            "second": {"one": "{0} seconde", "other": "{0} seconden"},
            "microsecond": {"one": "{0} microseconde", "other": "{0} microseconden"},
        },
        "relative": {
            "year": {
                "future": {"other": "over {0} jaar", "one": "over {0} jaar"},
                "past": {"other": "{0} jaar geleden", "one": "{0} jaar geleden"},
            },
            "month": {
                "future": {"other": "over {0} maanden", "one": "over {0} maand"},
                "past": {"other": "{0} maanden geleden", "one": "{0} maand geleden"},
            },
            "week": {
                "future": {"other": "over {0} weken", "one": "over {0} week"},
                "past": {"other": "{0} weken geleden", "one": "{0} week geleden"},
            },
            "day": {
                "future": {"other": "over {0} dagen", "one": "over {0} dag"},
                "past": {"other": "{0} dagen geleden", "one": "{0} dag geleden"},
            },
            "hour": {
                "future": {"other": "over {0} uur", "one": "over {0} uur"},
                "past": {"other": "{0} uur geleden", "one": "{0} uur geleden"},
            },
            "minute": {
                "future": {"other": "over {0} minuten", "one": "over {0} minuut"},
                "past": {"other": "{0} minuten geleden", "one": "{0} minuut geleden"},
            },
            "second": {
                "future": {"other": "over {0} seconden", "one": "over {0} seconde"},
                "past": {"other": "{0} seconden geleden", "one": "{0} seconde geleden"},
            },
        },
        "day_periods": {
            "midnight": "middernacht",
            "am": "a.m.",
            "pm": "p.m.",
            "morning1": "‘s ochtends",
            "afternoon1": "‘s middags",
            "evening1": "‘s avonds",
            "night1": "‘s nachts",
            "week_data": {
                "min_days": 1,
                "first_day": 0,
                "weekend_start": 5,
                "weekend_end": 6,
            },
        },
    },
    "custom": custom_translations,
}


# === src/pendulum/locales/bg/custom.py ===
"""
bg custom locale file.
"""
from __future__ import annotations


translations = {
    # Relative time
    "ago": "преди {}",
    "from_now": "след {}",
    "after": "след {0}",
    "before": "преди {0}",
    # Date formats
    "date_formats": {
        "LTS": "HH:mm:ss",
        "LT": "HH:mm",
        "L": "DD.MM.YYYY",
        "LL": "D MMMM YYYY г.",
        "LLL": "D MMMM YYYY г., HH:mm",
        "LLLL": "dddd, D MMMM YYYY г., HH:mm",
    },
}


# === src/pendulum/locales/bg/__init__.py ===


# === src/pendulum/locales/bg/locale.py ===
from .custom import translations as custom_translations


"""
bg locale file.

It has been generated automatically and must not be modified directly.
"""


locale = {
    'plural': lambda n: 'one' if (n == n and ((n == 1))) else 'other',
    'ordinal': lambda n: 'other',
    'translations': {
        'days': {
            'abbreviated': {
                0: 'пн',
                1: 'вт',
                2: 'ср',
                3: 'чт',
                4: 'пт',
                5: 'сб',
                6: 'нд',
            },
            'narrow': {
                0: 'п',
                1: 'в',
                2: 'с',
                3: 'ч',
                4: 'п',
                5: 'с',
                6: 'н',
            },
            'short': {
                0: 'пн',
                1: 'вт',
                2: 'ср',
                3: 'чт',
                4: 'пт',
                5: 'сб',
                6: 'нд',
            },
            'wide': {
                0: 'понеделник',
                1: 'вторник',
                2: 'сряда',
                3: 'четвъртък',
                4: 'петък',
                5: 'събота',
                6: 'неделя',
            },
        },
        'months': {
            'abbreviated': {
                1: 'яну',
                2: 'фев',
                3: 'март',
                4: 'апр',
                5: 'май',
                6: 'юни',
                7: 'юли',
                8: 'авг',
                9: 'сеп',
                10: 'окт',
                11: 'ное',
                12: 'дек',
            },
            'narrow': {
                1: 'я',
                2: 'ф',
                3: 'м',
                4: 'а',
                5: 'м',
                6: 'ю',
                7: 'ю',
                8: 'а',
                9: 'с',
                10: 'о',
                11: 'н',
                12: 'д',
            },
            'wide': {
                1: 'януари',
                2: 'февруари',
                3: 'март',
                4: 'април',
                5: 'май',
                6: 'юни',
                7: 'юли',
                8: 'август',
                9: 'септември',
                10: 'октомври',
                11: 'ноември',
                12: 'декември',
            },
        },
        'units': {
            'year': {
                'one': '{0} година',
                'other': '{0} години',
            },
            'month': {
                'one': '{0} месец',
                'other': '{0} месеца',
            },
            'week': {
                'one': '{0} седмица',
                'other': '{0} седмици',
            },
            'day': {
                'one': '{0} ден',
                'other': '{0} дни',
            },
            'hour': {
                'one': '{0} час',
                'other': '{0} часа',
            },
            'minute': {
                'one': '{0} минута',
                'other': '{0} минути',
            },
            'second': {
                'one': '{0} секунда',
                'other': '{0} секунди',
            },
            'microsecond': {
                'one': '{0} микросекунда',
                'other': '{0} микросекунди',
            },
        },
        'relative': {
            'year': {
                'future': {
                    'other': 'след {0} години',
                    'one': 'след {0} година',
                },
                'past': {
                    'other': 'преди {0} години',
                    'one': 'преди {0} година',
                },
            },
            'month': {
                'future': {
                    'other': 'след {0} месеца',
                    'one': 'след {0} месец',
                },
                'past': {
                    'other': 'преди {0} месеца',
                    'one': 'преди {0} месец',
                },
            },
            'week': {
                'future': {
                    'other': 'след {0} седмици',
                    'one': 'след {0} седмица',
                },
                'past': {
                    'other': 'преди {0} седмици',
                    'one': 'преди {0} седмица',
                },
            },
            'day': {
                'future': {
                    'other': 'след {0} дни',
                    'one': 'след {0} ден',
                },
                'past': {
                    'other': 'преди {0} дни',
                    'one': 'преди {0} ден',
                },
            },
            'hour': {
                'future': {
                    'other': 'след {0} часа',
                    'one': 'след {0} час',
                },
                'past': {
                    'other': 'преди {0} часа',
                    'one': 'преди {0} час',
                },
            },
            'minute': {
                'future': {
                    'other': 'след {0} минути',
                    'one': 'след {0} минута',
                },
                'past': {
                    'other': 'преди {0} минути',
                    'one': 'преди {0} минута',
                },
            },
            'second': {
                'future': {
                    'other': 'след {0} секунди',
                    'one': 'след {0} секунда',
                },
                'past': {
                    'other': 'преди {0} секунди',
                    'one': 'преди {0} секунда',
                },
            },
        },
        'day_periods': {
            'midnight': 'полунощ',
            'am': 'пр.об.',
            'pm': 'сл.об.',
            'morning1': 'сутринта',
            'morning2': 'на обяд',
            'afternoon1': 'следобед',
            'evening1': 'вечерта',
            'night1': 'през нощта',
        },
        'week_data': {
            'min_days': 1,
            'first_day': 0,
            'weekend_start': 5,
            'weekend_end': 6,
        },
    },
    'custom': custom_translations
}


# === src/pendulum/locales/nb/custom.py ===
"""
nn custom locale file.
"""
from __future__ import annotations


translations = {
    # Relative time
    "after": "{0} etter",
    "before": "{0} før",
    # Ordinals
    "ordinal": {"one": ".", "two": ".", "few": ".", "other": "."},
    # Date formats
    "date_formats": {
        "LTS": "HH:mm:ss",
        "LT": "HH:mm",
        "LLLL": "dddd Do MMMM YYYY HH:mm",
        "LLL": "Do MMMM YYYY HH:mm",
        "LL": "Do MMMM YYYY",
        "L": "DD.MM.YYYY",
    },
}


# === src/pendulum/locales/nb/__init__.py ===


# === src/pendulum/locales/nb/locale.py ===
from __future__ import annotations

from pendulum.locales.nb.custom import translations as custom_translations


"""
nb locale file.

It has been generated automatically and must not be modified directly.
"""


locale = {
    "plural": lambda n: "one" if (n == n and (n == 1)) else "other",
    "ordinal": lambda n: "other",
    "translations": {
        "days": {
            "abbreviated": {
                0: "man.",
                1: "tir.",
                2: "ons.",
                3: "tor.",
                4: "fre.",
                5: "lør.",
                6: "søn.",
            },
            "narrow": {0: "M", 1: "T", 2: "O", 3: "T", 4: "F", 5: "L", 6: "S"},
            "short": {
                0: "ma.",
                1: "ti.",
                2: "on.",
                3: "to.",
                4: "fr.",
                5: "lø.",
                6: "sø.",
            },
            "wide": {
                0: "mandag",
                1: "tirsdag",
                2: "onsdag",
                3: "torsdag",
                4: "fredag",
                5: "lørdag",
                6: "søndag",
            },
        },
        "months": {
            "abbreviated": {
                1: "jan.",
                2: "feb.",
                3: "mar.",
                4: "apr.",
                5: "mai",
                6: "jun.",
                7: "jul.",
                8: "aug.",
                9: "sep.",
                10: "okt.",
                11: "nov.",
                12: "des.",
            },
            "narrow": {
                1: "J",
                2: "F",
                3: "M",
                4: "A",
                5: "M",
                6: "J",
                7: "J",
                8: "A",
                9: "S",
                10: "O",
                11: "N",
                12: "D",
            },
            "wide": {
                1: "januar",
                2: "februar",
                3: "mars",
                4: "april",
                5: "mai",
                6: "juni",
                7: "juli",
                8: "august",
                9: "september",
                10: "oktober",
                11: "november",
                12: "desember",
            },
        },
        "units": {
            "year": {"one": "{0} år", "other": "{0} år"},
            "month": {"one": "{0} måned", "other": "{0} måneder"},
            "week": {"one": "{0} uke", "other": "{0} uker"},
            "day": {"one": "{0} dag", "other": "{0} dager"},
            "hour": {"one": "{0} time", "other": "{0} timer"},
            "minute": {"one": "{0} minutt", "other": "{0} minutter"},
            "second": {"one": "{0} sekund", "other": "{0} sekunder"},
            "microsecond": {"one": "{0} mikrosekund", "other": "{0} mikrosekunder"},
        },
        "relative": {
            "year": {
                "future": {"other": "om {0} år", "one": "om {0} år"},
                "past": {"other": "for {0} år siden", "one": "for {0} år siden"},
            },
            "month": {
                "future": {"other": "om {0} måneder", "one": "om {0} måned"},
                "past": {
                    "other": "for {0} måneder siden",
                    "one": "for {0} måned siden",
                },
            },
            "week": {
                "future": {"other": "om {0} uker", "one": "om {0} uke"},
                "past": {"other": "for {0} uker siden", "one": "for {0} uke siden"},
            },
            "day": {
                "future": {"other": "om {0} dager", "one": "om {0} dag"},
                "past": {"other": "for {0} dager siden", "one": "for {0} dag siden"},
            },
            "hour": {
                "future": {"other": "om {0} timer", "one": "om {0} time"},
                "past": {"other": "for {0} timer siden", "one": "for {0} time siden"},
            },
            "minute": {
                "future": {"other": "om {0} minutter", "one": "om {0} minutt"},
                "past": {
                    "other": "for {0} minutter siden",
                    "one": "for {0} minutt siden",
                },
            },
            "second": {
                "future": {"other": "om {0} sekunder", "one": "om {0} sekund"},
                "past": {
                    "other": "for {0} sekunder siden",
                    "one": "for {0} sekund siden",
                },
            },
        },
        "day_periods": {
            "midnight": "midnatt",
            "am": "a.m.",
            "pm": "p.m.",
            "morning1": "morgenen",
            "morning2": "formiddagen",
            "afternoon1": "ettermiddagen",
            "evening1": "kvelden",
            "night1": "natten",
        },
        "week_data": {
            "min_days": 1,
            "first_day": 0,
            "weekend_start": 5,
            "weekend_end": 6,
        },
    },
    "custom": custom_translations,
}


# === src/pendulum/locales/de/custom.py ===
"""
de custom locale file.
"""
from __future__ import annotations


translations = {
    # Relative time
    "after": "{0} später",
    "before": "{0} zuvor",
    "units_relative": {
        "year": {
            "future": {"one": "{0} Jahr", "other": "{0} Jahren"},
            "past": {"one": "{0} Jahr", "other": "{0} Jahren"},
        },
        "month": {
            "future": {"one": "{0} Monat", "other": "{0} Monaten"},
            "past": {"one": "{0} Monat", "other": "{0} Monaten"},
        },
        "week": {
            "future": {"one": "{0} Woche", "other": "{0} Wochen"},
            "past": {"one": "{0} Woche", "other": "{0} Wochen"},
        },
        "day": {
            "future": {"one": "{0} Tag", "other": "{0} Tagen"},
            "past": {"one": "{0} Tag", "other": "{0} Tagen"},
        },
    },
    # Date formats
    "date_formats": {
        "LTS": "HH:mm:ss",
        "LT": "HH:mm",
        "LLLL": "dddd, D. MMMM YYYY HH:mm",
        "LLL": "D. MMMM YYYY HH:mm",
        "LL": "D. MMMM YYYY",
        "L": "DD.MM.YYYY",
    },
}


# === src/pendulum/locales/de/__init__.py ===


# === src/pendulum/locales/de/locale.py ===
from __future__ import annotations

from pendulum.locales.de.custom import translations as custom_translations


"""
de locale file.

It has been generated automatically and must not be modified directly.
"""


locale = {
    "plural": lambda n: "one"
    if ((n == n and (n == 1)) and (0 == 0 and (0 == 0)))
    else "other",
    "ordinal": lambda n: "other",
    "translations": {
        "days": {
            "abbreviated": {
                0: "Mo.",
                1: "Di.",
                2: "Mi.",
                3: "Do.",
                4: "Fr.",
                5: "Sa.",
                6: "So.",
            },
            "narrow": {0: "M", 1: "D", 2: "M", 3: "D", 4: "F", 5: "S", 6: "S"},
            "short": {
                0: "Mo.",
                1: "Di.",
                2: "Mi.",
                3: "Do.",
                4: "Fr.",
                5: "Sa.",
                6: "So.",
            },
            "wide": {
                0: "Montag",
                1: "Dienstag",
                2: "Mittwoch",
                3: "Donnerstag",
                4: "Freitag",
                5: "Samstag",
                6: "Sonntag",
            },
        },
        "months": {
            "abbreviated": {
                1: "Jan.",
                2: "Feb.",
                3: "März",
                4: "Apr.",
                5: "Mai",
                6: "Juni",
                7: "Juli",
                8: "Aug.",
                9: "Sep.",
                10: "Okt.",
                11: "Nov.",
                12: "Dez.",
            },
            "narrow": {
                1: "J",
                2: "F",
                3: "M",
                4: "A",
                5: "M",
                6: "J",
                7: "J",
                8: "A",
                9: "S",
                10: "O",
                11: "N",
                12: "D",
            },
            "wide": {
                1: "Januar",
                2: "Februar",
                3: "März",
                4: "April",
                5: "Mai",
                6: "Juni",
                7: "Juli",
                8: "August",
                9: "September",
                10: "Oktober",
                11: "November",
                12: "Dezember",
            },
        },
        "units": {
            "year": {"one": "{0} Jahr", "other": "{0} Jahre"},
            "month": {"one": "{0} Monat", "other": "{0} Monate"},
            "week": {"one": "{0} Woche", "other": "{0} Wochen"},
            "day": {"one": "{0} Tag", "other": "{0} Tage"},
            "hour": {"one": "{0} Stunde", "other": "{0} Stunden"},
            "minute": {"one": "{0} Minute", "other": "{0} Minuten"},
            "second": {"one": "{0} Sekunde", "other": "{0} Sekunden"},
            "microsecond": {"one": "{0} Mikrosekunde", "other": "{0} Mikrosekunden"},
        },
        "relative": {
            "year": {
                "future": {"other": "in {0} Jahren", "one": "in {0} Jahr"},
                "past": {"other": "vor {0} Jahren", "one": "vor {0} Jahr"},
            },
            "month": {
                "future": {"other": "in {0} Monaten", "one": "in {0} Monat"},
                "past": {"other": "vor {0} Monaten", "one": "vor {0} Monat"},
            },
            "week": {
                "future": {"other": "in {0} Wochen", "one": "in {0} Woche"},
                "past": {"other": "vor {0} Wochen", "one": "vor {0} Woche"},
            },
            "day": {
                "future": {"other": "in {0} Tagen", "one": "in {0} Tag"},
                "past": {"other": "vor {0} Tagen", "one": "vor {0} Tag"},
            },
            "hour": {
                "future": {"other": "in {0} Stunden", "one": "in {0} Stunde"},
                "past": {"other": "vor {0} Stunden", "one": "vor {0} Stunde"},
            },
            "minute": {
                "future": {"other": "in {0} Minuten", "one": "in {0} Minute"},
                "past": {"other": "vor {0} Minuten", "one": "vor {0} Minute"},
            },
            "second": {
                "future": {"other": "in {0} Sekunden", "one": "in {0} Sekunde"},
                "past": {"other": "vor {0} Sekunden", "one": "vor {0} Sekunde"},
            },
        },
        "day_periods": {
            "midnight": "Mitternacht",
            "am": "vorm.",
            "pm": "nachm.",
            "morning1": "morgens",
            "morning2": "vormittags",
            "afternoon1": "mittags",
            "afternoon2": "nachmittags",
            "evening1": "abends",
            "night1": "nachts",
        },
        "week_data": {
            "min_days": 1,
            "first_day": 0,
            "weekend_start": 5,
            "weekend_end": 6,
        },
    },
    "custom": custom_translations,
}


# === src/pendulum/locales/ko/custom.py ===
"""
ko custom locale file.
"""
from __future__ import annotations


translations = {
    # Relative time
    "after": "{0} 후",
    "before": "{0} 전",
    # Date formats
    "date_formats": {
        "LTS": "A h시 m분 s초",
        "LT": "A h시 m분",
        "LLLL": "YYYY년 MMMM D일 dddd A h시 m분",
        "LLL": "YYYY년 MMMM D일 A h시 m분",
        "LL": "YYYY년 MMMM D일",
        "L": "YYYY.MM.DD",
    },
}


# === src/pendulum/locales/ko/__init__.py ===


# === src/pendulum/locales/ko/locale.py ===
from __future__ import annotations

from pendulum.locales.ko.custom import translations as custom_translations


"""
ko locale file.

It has been generated automatically and must not be modified directly.
"""


locale = {
    "plural": lambda n: "other",
    "ordinal": lambda n: "other",
    "translations": {
        "days": {
            "abbreviated": {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"},
            "narrow": {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"},
            "short": {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"},
            "wide": {
                0: "월요일",
                1: "화요일",
                2: "수요일",
                3: "목요일",
                4: "금요일",
                5: "토요일",
                6: "일요일",
            },
        },
        "months": {
            "abbreviated": {
                1: "1월",
                2: "2월",
                3: "3월",
                4: "4월",
                5: "5월",
                6: "6월",
                7: "7월",
                8: "8월",
                9: "9월",
                10: "10월",
                11: "11월",
                12: "12월",
            },
            "narrow": {
                1: "1월",
                2: "2월",
                3: "3월",
                4: "4월",
                5: "5월",
                6: "6월",
                7: "7월",
                8: "8월",
                9: "9월",
                10: "10월",
                11: "11월",
                12: "12월",
            },
            "wide": {
                1: "1월",
                2: "2월",
                3: "3월",
                4: "4월",
                5: "5월",
                6: "6월",
                7: "7월",
                8: "8월",
                9: "9월",
                10: "10월",
                11: "11월",
                12: "12월",
            },
        },
        "units": {
            "year": {"other": "{0}년"},
            "month": {"other": "{0}개월"},
            "week": {"other": "{0}주"},
            "day": {"other": "{0}일"},
            "hour": {"other": "{0}시간"},
            "minute": {"other": "{0}분"},
            "second": {"other": "{0}초"},
            "microsecond": {"other": "{0}마이크로초"},
        },
        "relative": {
            "year": {"future": {"other": "{0}년 후"}, "past": {"other": "{0}년 전"}},
            "month": {"future": {"other": "{0}개월 후"}, "past": {"other": "{0}개월 전"}},
            "week": {"future": {"other": "{0}주 후"}, "past": {"other": "{0}주 전"}},
            "day": {"future": {"other": "{0}일 후"}, "past": {"other": "{0}일 전"}},
            "hour": {"future": {"other": "{0}시간 후"}, "past": {"other": "{0}시간 전"}},
            "minute": {"future": {"other": "{0}분 후"}, "past": {"other": "{0}분 전"}},
            "second": {"future": {"other": "{0}초 후"}, "past": {"other": "{0}초 전"}},
        },
        "day_periods": {
            "midnight": "자정",
            "am": "오전",
            "noon": "정오",
            "pm": "오후",
            "morning1": "새벽",
            "morning2": "오전",
            "afternoon1": "오후",
            "evening1": "저녁",
            "night1": "밤",
        },
        "week_data": {
            "min_days": 1,
            "first_day": 0,
            "weekend_start": 5,
            "weekend_end": 6,
        },
    },
    "custom": custom_translations,
}


# === src/pendulum/locales/id/custom.py ===
"""
id custom locale file.
"""
from __future__ import annotations


translations = {
    "units": {"few_second": "beberapa detik"},
    "ago": "{} yang lalu",
    "from_now": "dalam {}",
    "after": "{0} kemudian",
    "before": "{0} yang lalu",
    "date_formats": {
        "LTS": "HH:mm:ss",
        "LT": "HH:mm",
        "LLLL": "dddd [d.] D. MMMM YYYY HH:mm",
        "LLL": "D. MMMM YYYY HH:mm",
        "LL": "D. MMMM YYYY",
        "L": "DD/MM/YYYY",
    },
}


# === src/pendulum/locales/id/__init__.py ===


# === src/pendulum/locales/id/locale.py ===
from __future__ import annotations

from pendulum.locales.id.custom import translations as custom_translations


"""
id locale file.

It has been generated automatically and must not be modified directly.
"""


locale = {
    "plural": lambda n: "other",
    "ordinal": lambda n: "other",
    "translations": {
        "days": {
            "abbreviated": {
                0: "Sen",
                1: "Sel",
                2: "Rab",
                3: "Kam",
                4: "Jum",
                5: "Sab",
                6: "Min",
            },
            "narrow": {0: "S", 1: "S", 2: "R", 3: "K", 4: "J", 5: "S", 6: "M"},
            "short": {
                0: "Sen",
                1: "Sel",
                2: "Rab",
                3: "Kam",
                4: "Jum",
                5: "Sab",
                6: "Min",
            },
            "wide": {
                0: "Senin",
                1: "Selasa",
                2: "Rabu",
                3: "Kamis",
                4: "Jumat",
                5: "Sabtu",
                6: "Minggu",
            },
        },
        "months": {
            "abbreviated": {
                1: "Jan",
                2: "Feb",
                3: "Mar",
                4: "Apr",
                5: "Mei",
                6: "Jun",
                7: "Jul",
                8: "Agt",
                9: "Sep",
                10: "Okt",
                11: "Nov",
                12: "Des",
            },
            "narrow": {
                1: "J",
                2: "F",
                3: "M",
                4: "A",
                5: "M",
                6: "J",
                7: "J",
                8: "A",
                9: "S",
                10: "O",
                11: "N",
                12: "D",
            },
            "wide": {
                1: "Januari",
                2: "Februari",
                3: "Maret",
                4: "April",
                5: "Mei",
                6: "Juni",
                7: "Juli",
                8: "Agustus",
                9: "September",
                10: "Oktober",
                11: "November",
                12: "Desember",
            },
        },
        "units": {
            "year": {"other": "{0} tahun"},
            "month": {"other": "{0} bulan"},
            "week": {"other": "{0} minggu"},
            "day": {"other": "{0} hari"},
            "hour": {"other": "{0} jam"},
            "minute": {"other": "{0} menit"},
            "second": {"other": "{0} detik"},
            "microsecond": {"other": "{0} mikrodetik"},
        },
        "relative": {
            "year": {
                "future": {"other": "dalam {0} tahun"},
                "past": {"other": "{0} tahun yang lalu"},
            },
            "month": {
                "future": {"other": "dalam {0} bulan"},
                "past": {"other": "{0} bulan yang lalu"},
            },
            "week": {
                "future": {"other": "dalam {0} minggu"},
                "past": {"other": "{0} minggu yang lalu"},
            },
            "day": {
                "future": {"other": "dalam {0} hari"},
                "past": {"other": "{0} hari yang lalu"},
            },
            "hour": {
                "future": {"other": "dalam {0} jam"},
                "past": {"other": "{0} jam yang lalu"},
            },
            "minute": {
                "future": {"other": "dalam {0} menit"},
                "past": {"other": "{0} menit yang lalu"},
            },
            "second": {
                "future": {"other": "dalam {0} detik"},
                "past": {"other": "{0} detik yang lalu"},
            },
        },
        "day_periods": {
            "midnight": "tengah malam",
            "am": "AM",
            "noon": "tengah hari",
            "pm": "PM",
            "morning1": "pagi",
            "afternoon1": "siang",
            "evening1": "sore",
            "night1": "malam",
        },
        "week_data": {
            "min_days": 1,
            "first_day": 0,
            "weekend_start": 5,
            "weekend_end": 6,
        },
    },
    "custom": custom_translations,
}


# === src/pendulum/locales/fr/custom.py ===
"""
fr custom locale file.
"""
from __future__ import annotations


translations = {
    "units": {"few_second": "quelques secondes"},
    # Relative Time
    "ago": "il y a {0}",
    "from_now": "dans {0}",
    "after": "{0} après",
    "before": "{0} avant",
    # Ordinals
    "ordinal": {"one": "er", "other": "e"},
    # Date formats
    "date_formats": {
        "LTS": "HH:mm:ss",
        "LT": "HH:mm",
        "LLLL": "dddd D MMMM YYYY HH:mm",
        "LLL": "D MMMM YYYY HH:mm",
        "LL": "D MMMM YYYY",
        "L": "DD/MM/YYYY",
    },
}


# === src/pendulum/locales/fr/__init__.py ===


# === src/pendulum/locales/fr/locale.py ===
from __future__ import annotations

from pendulum.locales.fr.custom import translations as custom_translations


"""
fr locale file.

It has been generated automatically and must not be modified directly.
"""


locale = {
    "plural": lambda n: "one" if (n == n and ((n == 0) or (n == 1))) else "other",
    "ordinal": lambda n: "one" if (n == n and (n == 1)) else "other",
    "translations": {
        "days": {
            "abbreviated": {
                0: "lun.",
                1: "mar.",
                2: "mer.",
                3: "jeu.",
                4: "ven.",
                5: "sam.",
                6: "dim.",
            },
            "narrow": {0: "L", 1: "M", 2: "M", 3: "J", 4: "V", 5: "S", 6: "D"},
            "short": {0: "lu", 1: "ma", 2: "me", 3: "je", 4: "ve", 5: "sa", 6: "di"},
            "wide": {
                0: "lundi",
                1: "mardi",
                2: "mercredi",
                3: "jeudi",
                4: "vendredi",
                5: "samedi",
                6: "dimanche",
            },
        },
        "months": {
            "abbreviated": {
                1: "janv.",
                2: "févr.",
                3: "mars",
                4: "avr.",
                5: "mai",
                6: "juin",
                7: "juil.",
                8: "août",
                9: "sept.",
                10: "oct.",
                11: "nov.",
                12: "déc.",
            },
            "narrow": {
                1: "J",
                2: "F",
                3: "M",
                4: "A",
                5: "M",
                6: "J",
                7: "J",
                8: "A",
                9: "S",
                10: "O",
                11: "N",
                12: "D",
            },
            "wide": {
                1: "janvier",
                2: "février",
                3: "mars",
                4: "avril",
                5: "mai",
                6: "juin",
                7: "juillet",
                8: "août",
                9: "septembre",
                10: "octobre",
                11: "novembre",
                12: "décembre",
            },
        },
        "units": {
            "year": {"one": "{0} an", "other": "{0} ans"},
            "month": {"one": "{0} mois", "other": "{0} mois"},
            "week": {"one": "{0} semaine", "other": "{0} semaines"},
            "day": {"one": "{0} jour", "other": "{0} jours"},
            "hour": {"one": "{0} heure", "other": "{0} heures"},
            "minute": {"one": "{0} minute", "other": "{0} minutes"},
            "second": {"one": "{0} seconde", "other": "{0} secondes"},
            "microsecond": {"one": "{0} microseconde", "other": "{0} microsecondes"},
        },
        "relative": {
            "year": {
                "future": {"other": "dans {0} ans", "one": "dans {0} an"},
                "past": {"other": "il y a {0} ans", "one": "il y a {0} an"},
            },
            "month": {
                "future": {"other": "dans {0} mois", "one": "dans {0} mois"},
                "past": {"other": "il y a {0} mois", "one": "il y a {0} mois"},
            },
            "week": {
                "future": {"other": "dans {0} semaines", "one": "dans {0} semaine"},
                "past": {"other": "il y a {0} semaines", "one": "il y a {0} semaine"},
            },
            "day": {
                "future": {"other": "dans {0} jours", "one": "dans {0} jour"},
                "past": {"other": "il y a {0} jours", "one": "il y a {0} jour"},
            },
            "hour": {
                "future": {"other": "dans {0} heures", "one": "dans {0} heure"},
                "past": {"other": "il y a {0} heures", "one": "il y a {0} heure"},
            },
            "minute": {
                "future": {"other": "dans {0} minutes", "one": "dans {0} minute"},
                "past": {"other": "il y a {0} minutes", "one": "il y a {0} minute"},
            },
            "second": {
                "future": {"other": "dans {0} secondes", "one": "dans {0} seconde"},
                "past": {"other": "il y a {0} secondes", "one": "il y a {0} seconde"},
            },
        },
        "day_periods": {
            "midnight": "minuit",
            "am": "AM",
            "noon": "midi",
            "pm": "PM",
            "morning1": "du matin",
            "afternoon1": "de l’après-midi",
            "evening1": "du soir",
            "night1": "de nuit",
        },
        "week_data": {
            "min_days": 1,
            "first_day": 0,
            "weekend_start": 5,
            "weekend_end": 6,
        },
    },
    "custom": custom_translations,
}


# === src/pendulum/locales/es/custom.py ===
"""
es custom locale file.
"""
from __future__ import annotations


translations = {
    "units": {"few_second": "unos segundos"},
    # Relative time
    "ago": "hace {0}",
    "from_now": "dentro de {0}",
    "after": "{0} después",
    "before": "{0} antes",
    # Ordinals
    "ordinal": {"other": "º"},
    # Date formats
    "date_formats": {
        "LTS": "H:mm:ss",
        "LT": "H:mm",
        "LLLL": "dddd, D [de] MMMM [de] YYYY H:mm",
        "LLL": "D [de] MMMM [de] YYYY H:mm",
        "LL": "D [de] MMMM [de] YYYY",
        "L": "DD/MM/YYYY",
    },
}


# === src/pendulum/locales/es/__init__.py ===


# === src/pendulum/locales/es/locale.py ===
from __future__ import annotations

from pendulum.locales.es.custom import translations as custom_translations


"""
es locale file.

It has been generated automatically and must not be modified directly.
"""


locale = {
    "plural": lambda n: "one" if (n == n and (n == 1)) else "other",
    "ordinal": lambda n: "other",
    "translations": {
        "days": {
            "abbreviated": {
                0: "lun.",
                1: "mar.",
                2: "mié.",
                3: "jue.",
                4: "vie.",
                5: "sáb.",
                6: "dom.",
            },
            "narrow": {0: "L", 1: "M", 2: "X", 3: "J", 4: "V", 5: "S", 6: "D"},
            "short": {0: "LU", 1: "MA", 2: "MI", 3: "JU", 4: "VI", 5: "SA", 6: "DO"},
            "wide": {
                0: "lunes",
                1: "martes",
                2: "miércoles",
                3: "jueves",
                4: "viernes",
                5: "sábado",
                6: "domingo",
            },
        },
        "months": {
            "abbreviated": {
                1: "ene.",
                2: "feb.",
                3: "mar.",
                4: "abr.",
                5: "may.",
                6: "jun.",
                7: "jul.",
                8: "ago.",
                9: "sept.",
                10: "oct.",
                11: "nov.",
                12: "dic.",
            },
            "narrow": {
                1: "E",
                2: "F",
                3: "M",
                4: "A",
                5: "M",
                6: "J",
                7: "J",
                8: "A",
                9: "S",
                10: "O",
                11: "N",
                12: "D",
            },
            "wide": {
                1: "enero",
                2: "febrero",
                3: "marzo",
                4: "abril",
                5: "mayo",
                6: "junio",
                7: "julio",
                8: "agosto",
                9: "septiembre",
                10: "octubre",
                11: "noviembre",
                12: "diciembre",
            },
        },
        "units": {
            "year": {"one": "{0} año", "other": "{0} años"},
            "month": {"one": "{0} mes", "other": "{0} meses"},
            "week": {"one": "{0} semana", "other": "{0} semanas"},
            "day": {"one": "{0} día", "other": "{0} días"},
            "hour": {"one": "{0} hora", "other": "{0} horas"},
            "minute": {"one": "{0} minuto", "other": "{0} minutos"},
            "second": {"one": "{0} segundo", "other": "{0} segundos"},
            "microsecond": {"one": "{0} microsegundo", "other": "{0} microsegundos"},
        },
        "relative": {
            "year": {
                "future": {"other": "dentro de {0} años", "one": "dentro de {0} año"},
                "past": {"other": "hace {0} años", "one": "hace {0} año"},
            },
            "month": {
                "future": {"other": "dentro de {0} meses", "one": "dentro de {0} mes"},
                "past": {"other": "hace {0} meses", "one": "hace {0} mes"},
            },
            "week": {
                "future": {
                    "other": "dentro de {0} semanas",
                    "one": "dentro de {0} semana",
                },
                "past": {"other": "hace {0} semanas", "one": "hace {0} semana"},
            },
            "day": {
                "future": {"other": "dentro de {0} días", "one": "dentro de {0} día"},
                "past": {"other": "hace {0} días", "one": "hace {0} día"},
            },
            "hour": {
                "future": {"other": "dentro de {0} horas", "one": "dentro de {0} hora"},
                "past": {"other": "hace {0} horas", "one": "hace {0} hora"},
            },
            "minute": {
                "future": {
                    "other": "dentro de {0} minutos",
                    "one": "dentro de {0} minuto",
                },
                "past": {"other": "hace {0} minutos", "one": "hace {0} minuto"},
            },
            "second": {
                "future": {
                    "other": "dentro de {0} segundos",
                    "one": "dentro de {0} segundo",
                },
                "past": {"other": "hace {0} segundos", "one": "hace {0} segundo"},
            },
        },
        "day_periods": {
            "am": "a. m.",
            "noon": "del mediodía",
            "pm": "p. m.",
            "morning1": "de la madrugada",
            "morning2": "de la mañana",
            "evening1": "de la tarde",
            "night1": "de la noche",
        },
        "week_data": {
            "min_days": 1,
            "first_day": 0,
            "weekend_start": 5,
            "weekend_end": 6,
        },
    },
    "custom": custom_translations,
}


# === src/pendulum/locales/en/custom.py ===
"""
en custom locale file.
"""
from __future__ import annotations


translations = {
    "units": {"few_second": "a few seconds"},
    # Relative time
    "ago": "{} ago",
    "from_now": "in {}",
    "after": "{0} after",
    "before": "{0} before",
    # Ordinals
    "ordinal": {"one": "st", "two": "nd", "few": "rd", "other": "th"},
    # Date formats
    "date_formats": {
        "LTS": "h:mm:ss A",
        "LT": "h:mm A",
        "L": "MM/DD/YYYY",
        "LL": "MMMM D, YYYY",
        "LLL": "MMMM D, YYYY h:mm A",
        "LLLL": "dddd, MMMM D, YYYY h:mm A",
    },
}


# === src/pendulum/locales/en/__init__.py ===


# === src/pendulum/locales/en/locale.py ===
from __future__ import annotations

from pendulum.locales.en.custom import translations as custom_translations


"""
en locale file.

It has been generated automatically and must not be modified directly.
"""


locale = {
    "plural": lambda n: "one"
    if ((n == n and (n == 1)) and (0 == 0 and (0 == 0)))
    else "other",
    "ordinal": lambda n: "few"
    if (
        ((n % 10) == (n % 10) and ((n % 10) == 3))
        and (not ((n % 100) == (n % 100) and ((n % 100) == 13)))
    )
    else "one"
    if (
        ((n % 10) == (n % 10) and ((n % 10) == 1))
        and (not ((n % 100) == (n % 100) and ((n % 100) == 11)))
    )
    else "two"
    if (
        ((n % 10) == (n % 10) and ((n % 10) == 2))
        and (not ((n % 100) == (n % 100) and ((n % 100) == 12)))
    )
    else "other",
    "translations": {
        "days": {
            "abbreviated": {
                0: "Mon",
                1: "Tue",
                2: "Wed",
                3: "Thu",
                4: "Fri",
                5: "Sat",
                6: "Sun",
            },
            "narrow": {0: "M", 1: "T", 2: "W", 3: "T", 4: "F", 5: "S", 6: "S"},
            "short": {0: "Mo", 1: "Tu", 2: "We", 3: "Th", 4: "Fr", 5: "Sa", 6: "Su"},
            "wide": {
                0: "Monday",
                1: "Tuesday",
                2: "Wednesday",
                3: "Thursday",
                4: "Friday",
                5: "Saturday",
                6: "Sunday",
            },
        },
        "months": {
            "abbreviated": {
                1: "Jan",
                2: "Feb",
                3: "Mar",
                4: "Apr",
                5: "May",
                6: "Jun",
                7: "Jul",
                8: "Aug",
                9: "Sep",
                10: "Oct",
                11: "Nov",
                12: "Dec",
            },
            "narrow": {
                1: "J",
                2: "F",
                3: "M",
                4: "A",
                5: "M",
                6: "J",
                7: "J",
                8: "A",
                9: "S",
                10: "O",
                11: "N",
                12: "D",
            },
            "wide": {
                1: "January",
                2: "February",
                3: "March",
                4: "April",
                5: "May",
                6: "June",
                7: "July",
                8: "August",
                9: "September",
                10: "October",
                11: "November",
                12: "December",
            },
        },
        "units": {
            "year": {"one": "{0} year", "other": "{0} years"},
            "month": {"one": "{0} month", "other": "{0} months"},
            "week": {"one": "{0} week", "other": "{0} weeks"},
            "day": {"one": "{0} day", "other": "{0} days"},
            "hour": {"one": "{0} hour", "other": "{0} hours"},
            "minute": {"one": "{0} minute", "other": "{0} minutes"},
            "second": {"one": "{0} second", "other": "{0} seconds"},
            "microsecond": {"one": "{0} microsecond", "other": "{0} microseconds"},
        },
        "relative": {
            "year": {
                "future": {"other": "in {0} years", "one": "in {0} year"},
                "past": {"other": "{0} years ago", "one": "{0} year ago"},
            },
            "month": {
                "future": {"other": "in {0} months", "one": "in {0} month"},
                "past": {"other": "{0} months ago", "one": "{0} month ago"},
            },
            "week": {
                "future": {"other": "in {0} weeks", "one": "in {0} week"},
                "past": {"other": "{0} weeks ago", "one": "{0} week ago"},
            },
            "day": {
                "future": {"other": "in {0} days", "one": "in {0} day"},
                "past": {"other": "{0} days ago", "one": "{0} day ago"},
            },
            "hour": {
                "future": {"other": "in {0} hours", "one": "in {0} hour"},
                "past": {"other": "{0} hours ago", "one": "{0} hour ago"},
            },
            "minute": {
                "future": {"other": "in {0} minutes", "one": "in {0} minute"},
                "past": {"other": "{0} minutes ago", "one": "{0} minute ago"},
            },
            "second": {
                "future": {"other": "in {0} seconds", "one": "in {0} second"},
                "past": {"other": "{0} seconds ago", "one": "{0} second ago"},
            },
        },
        "day_periods": {
            "midnight": "midnight",
            "am": "AM",
            "noon": "noon",
            "pm": "PM",
            "morning1": "in the morning",
            "afternoon1": "in the afternoon",
            "evening1": "in the evening",
            "night1": "at night",
        },
        "week_data": {
            "min_days": 1,
            "first_day": 6,
            "weekend_start": 5,
            "weekend_end": 6,
        },
    },
    "custom": custom_translations,
}


# === src/pendulum/locales/fa/custom.py ===
"""
fa custom locale file.
"""
from __future__ import annotations


translations = {
    # Relative time
    "after": "{0} پس از",
    "before": "{0} پیش از",
    # Date formats
    "date_formats": {
        "LTS": "HH:mm:ss",
        "LT": "HH:mm",
        "LLLL": "dddd, D MMMM YYYY HH:mm",
        "LLL": "D MMMM YYYY HH:mm",
        "LL": "D MMMM YYYY",
        "L": "DD/MM/YYYY",
    },
}


# === src/pendulum/locales/fa/__init__.py ===


# === src/pendulum/locales/fa/locale.py ===
from __future__ import annotations

from pendulum.locales.fa.custom import translations as custom_translations


"""
fa locale file.

It has been generated automatically and must not be modified directly.
"""


locale = {
    "plural": lambda n: "one"
    if ((n == n and (n == 0)) or (n == n and (n == 1)))
    else "other",
    "ordinal": lambda n: "other",
    "translations": {
        "days": {
            "abbreviated": {
                0: "دوشنبه",
                1: "سه\u200cشنبه",
                2: "چهارشنبه",
                3: "پنجشنبه",
                4: "جمعه",
                5: "شنبه",
                6: "یکشنبه",
            },
            "narrow": {0: "د", 1: "س", 2: "چ", 3: "پ", 4: "ج", 5: "ش", 6: "ی"},
            "short": {0: "۲ش", 1: "۳ش", 2: "۴ش", 3: "۵ش", 4: "ج", 5: "ش", 6: "۱ش"},
            "wide": {
                0: "دوشنبه",
                1: "سه\u200cشنبه",
                2: "چهارشنبه",
                3: "پنجشنبه",
                4: "جمعه",
                5: "شنبه",
                6: "یکشنبه",
            },
        },
        "months": {
            "abbreviated": {
                1: "ژانویهٔ",
                2: "فوریهٔ",
                3: "مارس",
                4: "آوریل",
                5: "مهٔ",
                6: "ژوئن",
                7: "ژوئیهٔ",
                8: "اوت",
                9: "سپتامبر",
                10: "اکتبر",
                11: "نوامبر",
                12: "دسامبر",
            },
            "narrow": {
                1: "ژ",
                2: "ف",
                3: "م",
                4: "آ",
                5: "م",
                6: "ژ",
                7: "ژ",
                8: "ا",
                9: "س",
                10: "ا",
                11: "ن",
                12: "د",
            },
            "wide": {
                1: "ژانویهٔ",
                2: "فوریهٔ",
                3: "مارس",
                4: "آوریل",
                5: "مهٔ",
                6: "ژوئن",
                7: "ژوئیهٔ",
                8: "اوت",
                9: "سپتامبر",
                10: "اکتبر",
                11: "نوامبر",
                12: "دسامبر",
            },
        },
        "units": {
            "year": {"one": "{0} سال", "other": "{0} سال"},
            "month": {"one": "{0} ماه", "other": "{0} ماه"},
            "week": {"one": "{0} هفته", "other": "{0} هفته"},
            "day": {"one": "{0} روز", "other": "{0} روز"},
            "hour": {"one": "{0} ساعت", "other": "{0} ساعت"},
            "minute": {"one": "{0} دقیقه", "other": "{0} دقیقه"},
            "second": {"one": "{0} ثانیه", "other": "{0} ثانیه"},
            "microsecond": {"one": "{0} میکروثانیه", "other": "{0} میکروثانیه"},
        },
        "relative": {
            "year": {
                "future": {"other": "{0} سال بعد", "one": "{0} سال بعد"},
                "past": {"other": "{0} سال پیش", "one": "{0} سال پیش"},
            },
            "month": {
                "future": {"other": "{0} ماه بعد", "one": "{0} ماه بعد"},
                "past": {"other": "{0} ماه پیش", "one": "{0} ماه پیش"},
            },
            "week": {
                "future": {"other": "{0} هفته بعد", "one": "{0} هفته بعد"},
                "past": {"other": "{0} هفته پیش", "one": "{0} هفته پیش"},
            },
            "day": {
                "future": {"other": "{0} روز بعد", "one": "{0} روز بعد"},
                "past": {"other": "{0} روز پیش", "one": "{0} روز پیش"},
            },
            "hour": {
                "future": {"other": "{0} ساعت بعد", "one": "{0} ساعت بعد"},
                "past": {"other": "{0} ساعت پیش", "one": "{0} ساعت پیش"},
            },
            "minute": {
                "future": {"other": "{0} دقیقه بعد", "one": "{0} دقیقه بعد"},
                "past": {"other": "{0} دقیقه پیش", "one": "{0} دقیقه پیش"},
            },
            "second": {
                "future": {"other": "{0} ثانیه بعد", "one": "{0} ثانیه بعد"},
                "past": {"other": "{0} ثانیه پیش", "one": "{0} ثانیه پیش"},
            },
        },
        "day_periods": {
            "midnight": "نیمه\u200cشب",
            "am": "قبل\u200cازظهر",
            "noon": "ظهر",
            "pm": "بعدازظهر",
            "morning1": "صبح",
            "afternoon1": "عصر",
            "evening1": "عصر",
            "night1": "شب",
        },
        "week_data": {
            "min_days": 1,
            "first_day": 0,
            "weekend_start": 5,
            "weekend_end": 6,
        },
    },
    "custom": custom_translations,
}


# === src/pendulum/locales/lt/custom.py ===
"""
lt custom locale file.
"""
from __future__ import annotations


translations = {
    # Relative time
    "units_relative": {
        "year": {
            "future": {
                "other": "{0} metų",
                "one": "{0} metų",
                "few": "{0} metų",
                "many": "{0} metų",
            },
            "past": {
                "other": "{0} metų",
                "one": "{0} metus",
                "few": "{0} metus",
                "many": "{0} metų",
            },
        },
        "month": {
            "future": {
                "other": "{0} mėnesių",
                "one": "{0} mėnesio",
                "few": "{0} mėnesių",
                "many": "{0} mėnesio",
            },
            "past": {
                "other": "{0} mėnesių",
                "one": "{0} mėnesį",
                "few": "{0} mėnesius",
                "many": "{0} mėnesio",
            },
        },
        "week": {
            "future": {
                "other": "{0} savaičių",
                "one": "{0} savaitės",
                "few": "{0} savaičių",
                "many": "{0} savaitės",
            },
            "past": {
                "other": "{0} savaičių",
                "one": "{0} savaitę",
                "few": "{0} savaites",
                "many": "{0} savaitės",
            },
        },
        "day": {
            "future": {
                "other": "{0} dienų",
                "one": "{0} dienos",
                "few": "{0} dienų",
                "many": "{0} dienos",
            },
            "past": {
                "other": "{0} dienų",
                "one": "{0} dieną",
                "few": "{0} dienas",
                "many": "{0} dienos",
            },
        },
        "hour": {
            "future": {
                "other": "{0} valandų",
                "one": "{0} valandos",
                "few": "{0} valandų",
                "many": "{0} valandos",
            },
            "past": {
                "other": "{0} valandų",
                "one": "{0} valandą",
                "few": "{0} valandas",
                "many": "{0} valandos",
            },
        },
        "minute": {
            "future": {
                "other": "{0} minučių",
                "one": "{0} minutės",
                "few": "{0} minučių",
                "many": "{0} minutės",
            },
            "past": {
                "other": "{0} minučių",
                "one": "{0} minutę",
                "few": "{0} minutes",
                "many": "{0} minutės",
            },
        },
        "second": {
            "future": {
                "other": "{0} sekundžių",
                "one": "{0} sekundės",
                "few": "{0} sekundžių",
                "many": "{0} sekundės",
            },
            "past": {
                "other": "{0} sekundžių",
                "one": "{0} sekundę",
                "few": "{0} sekundes",
                "many": "{0} sekundės",
            },
        },
    },
    "after": "po {0}",
    "before": "{0} nuo dabar",
    # Date formats
    "date_formats": {
        "LTS": "HH:mm:ss",
        "LT": "HH:mm",
        "LLLL": "YYYY [m.] MMMM D [d.], dddd, HH:mm [val.]",
        "LLL": "YYYY [m.] MMMM D [d.], HH:mm [val.]",
        "LL": "YYYY [m.] MMMM D [d.]",
        "L": "YYYY-MM-DD",
    },
}


# === src/pendulum/locales/lt/__init__.py ===


# === src/pendulum/locales/lt/locale.py ===
from __future__ import annotations

from pendulum.locales.lt.custom import translations as custom_translations


"""
lt locale file.

It has been generated automatically and must not be modified directly.
"""


locale = {
    "plural": lambda n: "few"
    if (
        ((n % 10) == (n % 10) and ((n % 10) >= 2 and (n % 10) <= 9))
        and (not ((n % 100) == (n % 100) and ((n % 100) >= 11 and (n % 100) <= 19)))
    )
    else "many"
    if (not (0 == 0 and (0 == 0)))
    else "one"
    if (
        ((n % 10) == (n % 10) and ((n % 10) == 1))
        and (not ((n % 100) == (n % 100) and ((n % 100) >= 11 and (n % 100) <= 19)))
    )
    else "other",
    "ordinal": lambda n: "other",
    "translations": {
        "days": {
            "abbreviated": {
                0: "pr",
                1: "an",
                2: "tr",
                3: "kt",
                4: "pn",
                5: "št",
                6: "sk",
            },
            "narrow": {0: "P", 1: "A", 2: "T", 3: "K", 4: "P", 5: "Š", 6: "S"},
            "short": {0: "Pr", 1: "An", 2: "Tr", 3: "Kt", 4: "Pn", 5: "Št", 6: "Sk"},
            "wide": {
                0: "pirmadienis",
                1: "antradienis",
                2: "trečiadienis",
                3: "ketvirtadienis",
                4: "penktadienis",
                5: "šeštadienis",
                6: "sekmadienis",
            },
        },
        "months": {
            "abbreviated": {
                1: "saus.",
                2: "vas.",
                3: "kov.",
                4: "bal.",
                5: "geg.",
                6: "birž.",
                7: "liep.",
                8: "rugp.",
                9: "rugs.",
                10: "spal.",
                11: "lapkr.",
                12: "gruod.",
            },
            "narrow": {
                1: "S",
                2: "V",
                3: "K",
                4: "B",
                5: "G",
                6: "B",
                7: "L",
                8: "R",
                9: "R",
                10: "S",
                11: "L",
                12: "G",
            },
            "wide": {
                1: "sausio",
                2: "vasario",
                3: "kovo",
                4: "balandžio",
                5: "gegužės",
                6: "birželio",
                7: "liepos",
                8: "rugpjūčio",
                9: "rugsėjo",
                10: "spalio",
                11: "lapkričio",
                12: "gruodžio",
            },
        },
        "units": {
            "year": {
                "one": "{0} metai",
                "few": "{0} metai",
                "many": "{0} metų",
                "other": "{0} metų",
            },
            "month": {
                "one": "{0} mėnuo",
                "few": "{0} mėnesiai",
                "many": "{0} mėnesio",
                "other": "{0} mėnesių",
            },
            "week": {
                "one": "{0} savaitė",
                "few": "{0} savaitės",
                "many": "{0} savaitės",
                "other": "{0} savaičių",
            },
            "day": {
                "one": "{0} diena",
                "few": "{0} dienos",
                "many": "{0} dienos",
                "other": "{0} dienų",
            },
            "hour": {
                "one": "{0} valanda",
                "few": "{0} valandos",
                "many": "{0} valandos",
                "other": "{0} valandų",
            },
            "minute": {
                "one": "{0} minutė",
                "few": "{0} minutės",
                "many": "{0} minutės",
                "other": "{0} minučių",
            },
            "second": {
                "one": "{0} sekundė",
                "few": "{0} sekundės",
                "many": "{0} sekundės",
                "other": "{0} sekundžių",
            },
            "microsecond": {
                "one": "{0} mikrosekundė",
                "few": "{0} mikrosekundės",
                "many": "{0} mikrosekundės",
                "other": "{0} mikrosekundžių",
            },
        },
        "relative": {
            "year": {
                "future": {
                    "other": "po {0} metų",
                    "one": "po {0} metų",
                    "few": "po {0} metų",
                    "many": "po {0} metų",
                },
                "past": {
                    "other": "prieš {0} metų",
                    "one": "prieš {0} metus",
                    "few": "prieš {0} metus",
                    "many": "prieš {0} metų",
                },
            },
            "month": {
                "future": {
                    "other": "po {0} mėnesių",
                    "one": "po {0} mėnesio",
                    "few": "po {0} mėnesių",
                    "many": "po {0} mėnesio",
                },
                "past": {
                    "other": "prieš {0} mėnesių",
                    "one": "prieš {0} mėnesį",
                    "few": "prieš {0} mėnesius",
                    "many": "prieš {0} mėnesio",
                },
            },
            "week": {
                "future": {
                    "other": "po {0} savaičių",
                    "one": "po {0} savaitės",
                    "few": "po {0} savaičių",
                    "many": "po {0} savaitės",
                },
                "past": {
                    "other": "prieš {0} savaičių",
                    "one": "prieš {0} savaitę",
                    "few": "prieš {0} savaites",
                    "many": "prieš {0} savaitės",
                },
            },
            "day": {
                "future": {
                    "other": "po {0} dienų",
                    "one": "po {0} dienos",
                    "few": "po {0} dienų",
                    "many": "po {0} dienos",
                },
                "past": {
                    "other": "prieš {0} dienų",
                    "one": "prieš {0} dieną",
                    "few": "prieš {0} dienas",
                    "many": "prieš {0} dienos",
                },
            },
            "hour": {
                "future": {
                    "other": "po {0} valandų",
                    "one": "po {0} valandos",
                    "few": "po {0} valandų",
                    "many": "po {0} valandos",
                },
                "past": {
                    "other": "prieš {0} valandų",
                    "one": "prieš {0} valandą",
                    "few": "prieš {0} valandas",
                    "many": "prieš {0} valandos",
                },
            },
            "minute": {
                "future": {
                    "other": "po {0} minučių",
                    "one": "po {0} minutės",
                    "few": "po {0} minučių",
                    "many": "po {0} minutės",
                },
                "past": {
                    "other": "prieš {0} minučių",
                    "one": "prieš {0} minutę",
                    "few": "prieš {0} minutes",
                    "many": "prieš {0} minutės",
                },
            },
            "second": {
                "future": {
                    "other": "po {0} sekundžių",
                    "one": "po {0} sekundės",
                    "few": "po {0} sekundžių",
                    "many": "po {0} sekundės",
                },
                "past": {
                    "other": "prieš {0} sekundžių",
                    "one": "prieš {0} sekundę",
                    "few": "prieš {0} sekundes",
                    "many": "prieš {0} sekundės",
                },
            },
        },
        "day_periods": {
            "midnight": "vidurnaktis",
            "am": "priešpiet",
            "noon": "perpiet",
            "pm": "popiet",
            "morning1": "rytas",
            "afternoon1": "popietė",
            "evening1": "vakaras",
            "night1": "naktis",
        },
        "week_data": {
            "min_days": 1,
            "first_day": 0,
            "weekend_start": 5,
            "weekend_end": 6,
        },
    },
    "custom": custom_translations,
}


# === src/pendulum/locales/fo/custom.py ===
"""
fo custom locale file.
"""
from __future__ import annotations


translations = {
    # Relative time
    "after": "{0} aftaná",
    "before": "{0} áðrenn",
    # Ordinals
    "ordinal": {"other": "."},
    # Date formats
    "date_formats": {
        "LTS": "HH:mm:ss",
        "LT": "HH:mm",
        "LLLL": "dddd D. MMMM, YYYY HH:mm",
        "LLL": "D MMMM YYYY HH:mm",
        "LL": "D MMMM YYYY",
        "L": "DD/MM/YYYY",
    },
}


# === src/pendulum/locales/fo/__init__.py ===


# === src/pendulum/locales/fo/locale.py ===
from __future__ import annotations

from pendulum.locales.fo.custom import translations as custom_translations


"""
fo locale file.

It has been generated automatically and must not be modified directly.
"""


locale = {
    "plural": lambda n: "one" if (n == n and (n == 1)) else "other",
    "ordinal": lambda n: "other",
    "translations": {
        "days": {
            "abbreviated": {
                0: "mán.",
                1: "týs.",
                2: "mik.",
                3: "hós.",
                4: "frí.",
                5: "ley.",
                6: "sun.",
            },
            "narrow": {0: "M", 1: "T", 2: "M", 3: "H", 4: "F", 5: "L", 6: "S"},
            "short": {
                0: "má.",
                1: "tý.",
                2: "mi.",
                3: "hó.",
                4: "fr.",
                5: "le.",
                6: "su.",
            },
            "wide": {
                0: "mánadagur",
                1: "týsdagur",
                2: "mikudagur",
                3: "hósdagur",
                4: "fríggjadagur",
                5: "leygardagur",
                6: "sunnudagur",
            },
        },
        "months": {
            "abbreviated": {
                1: "jan.",
                2: "feb.",
                3: "mar.",
                4: "apr.",
                5: "mai",
                6: "jun.",
                7: "jul.",
                8: "aug.",
                9: "sep.",
                10: "okt.",
                11: "nov.",
                12: "des.",
            },
            "narrow": {
                1: "J",
                2: "F",
                3: "M",
                4: "A",
                5: "M",
                6: "J",
                7: "J",
                8: "A",
                9: "S",
                10: "O",
                11: "N",
                12: "D",
            },
            "wide": {
                1: "januar",
                2: "februar",
                3: "mars",
                4: "apríl",
                5: "mai",
                6: "juni",
                7: "juli",
                8: "august",
                9: "september",
                10: "oktober",
                11: "november",
                12: "desember",
            },
        },
        "units": {
            "year": {"one": "{0} ár", "other": "{0} ár"},
            "month": {"one": "{0} mánaður", "other": "{0} mánaðir"},
            "week": {"one": "{0} vika", "other": "{0} vikur"},
            "day": {"one": "{0} dagur", "other": "{0} dagar"},
            "hour": {"one": "{0} tími", "other": "{0} tímar"},
            "minute": {"one": "{0} minuttur", "other": "{0} minuttir"},
            "second": {"one": "{0} sekund", "other": "{0} sekundir"},
            "microsecond": {"one": "{0} mikrosekund", "other": "{0} mikrosekundir"},
        },
        "relative": {
            "year": {
                "future": {"other": "um {0} ár", "one": "um {0} ár"},
                "past": {"other": "{0} ár síðan", "one": "{0} ár síðan"},
            },
            "month": {
                "future": {"other": "um {0} mánaðir", "one": "um {0} mánað"},
                "past": {"other": "{0} mánaðir síðan", "one": "{0} mánað síðan"},
            },
            "week": {
                "future": {"other": "um {0} vikur", "one": "um {0} viku"},
                "past": {"other": "{0} vikur síðan", "one": "{0} vika síðan"},
            },
            "day": {
                "future": {"other": "um {0} dagar", "one": "um {0} dag"},
                "past": {"other": "{0} dagar síðan", "one": "{0} dagur síðan"},
            },
            "hour": {
                "future": {"other": "um {0} tímar", "one": "um {0} tíma"},
                "past": {"other": "{0} tímar síðan", "one": "{0} tími síðan"},
            },
            "minute": {
                "future": {"other": "um {0} minuttir", "one": "um {0} minutt"},
                "past": {"other": "{0} minuttir síðan", "one": "{0} minutt síðan"},
            },
            "second": {
                "future": {"other": "um {0} sekund", "one": "um {0} sekund"},
                "past": {"other": "{0} sekund síðan", "one": "{0} sekund síðan"},
            },
        },
        "day_periods": {"am": "AM", "pm": "PM"},
        "week_data": {
            "min_days": 1,
            "first_day": 0,
            "weekend_start": 5,
            "weekend_end": 6,
        },
    },
    "custom": custom_translations,
}


# === src/pendulum/locales/tr/custom.py ===
"""
tr custom locale file.
"""
from __future__ import annotations


translations = {
    # Relative time
    "ago": "{} önce",
    "from_now": "{} içinde",
    "after": "{0} sonra",
    "before": "{0} önce",
    # Ordinals
    "ordinal": {"one": ".", "two": ".", "few": ".", "other": "."},
    # Date formats
    "date_formats": {
        "LTS": "h:mm:ss A",
        "LT": "h:mm A",
        "L": "MM/DD/YYYY",
        "LL": "MMMM D, YYYY",
        "LLL": "MMMM D, YYYY h:mm A",
        "LLLL": "dddd, MMMM D, YYYY h:mm A",
    },
}


# === src/pendulum/locales/tr/__init__.py ===


# === src/pendulum/locales/tr/locale.py ===
from __future__ import annotations

from pendulum.locales.tr.custom import translations as custom_translations


"""
tr locale file.

It has been generated automatically and must not be modified directly.
"""


locale = {
    "plural": lambda n: "one" if (n == n and (n == 1)) else "other",
    "ordinal": lambda n: "other",
    "translations": {
        "days": {
            "abbreviated": {
                0: "Pzt",
                1: "Sal",
                2: "Çar",
                3: "Per",
                4: "Cum",
                5: "Cmt",
                6: "Paz",
            },
            "narrow": {
                0: "P",
                1: "S",
                2: "Ç",
                3: "P",
                4: "C",
                5: "C",
                6: "P",
            },
            "short": {
                0: "Pt",
                1: "Sa",
                2: "Ça",
                3: "Pe",
                4: "Cu",
                5: "Ct",
                6: "Pa",
            },
            "wide": {
                0: "Pazartesi",
                1: "Salı",
                2: "Çarşamba",
                3: "Perşembe",
                4: "Cuma",
                5: "Cumartesi",
                6: "Pazar",
            },
        },
        "months": {
            "abbreviated": {
                1: "Oca",
                2: "Şub",
                3: "Mar",
                4: "Nis",
                5: "May",
                6: "Haz",
                7: "Tem",
                8: "Ağu",
                9: "Eyl",
                10: "Eki",
                11: "Kas",
                12: "Ara",
            },
            "narrow": {
                1: "O",
                2: "Ş",
                3: "M",
                4: "N",
                5: "M",
                6: "H",
                7: "T",
                8: "A",
                9: "E",
                10: "E",
                11: "K",
                12: "A",
            },
            "wide": {
                1: "Ocak",
                2: "Şubat",
                3: "Mart",
                4: "Nisan",
                5: "Mayıs",
                6: "Haziran",
                7: "Temmuz",
                8: "Ağustos",
                9: "Eylül",
                10: "Ekim",
                11: "Kasım",
                12: "Aralık",
            },
        },
        "units": {
            "year": {
                "one": "{0} yıl",
                "other": "{0} yıl",
            },
            "month": {
                "one": "{0} ay",
                "other": "{0} ay",
            },
            "week": {
                "one": "{0} hafta",
                "other": "{0} hafta",
            },
            "day": {
                "one": "{0} gün",
                "other": "{0} gün",
            },
            "hour": {
                "one": "{0} saat",
                "other": "{0} saat",
            },
            "minute": {
                "one": "{0} dakika",
                "other": "{0} dakika",
            },
            "second": {
                "one": "{0} saniye",
                "other": "{0} saniye",
            },
            "microsecond": {
                "one": "{0} mikrosaniye",
                "other": "{0} mikrosaniye",
            },
        },
        "relative": {
            "year": {
                "future": {
                    "other": "{0} yıl sonra",
                    "one": "{0} yıl sonra",
                },
                "past": {
                    "other": "{0} yıl önce",
                    "one": "{0} yıl önce",
                },
            },
            "month": {
                "future": {
                    "other": "{0} ay sonra",
                    "one": "{0} ay sonra",
                },
                "past": {
                    "other": "{0} ay önce",
                    "one": "{0} ay önce",
                },
            },
            "week": {
                "future": {
                    "other": "{0} hafta sonra",
                    "one": "{0} hafta sonra",
                },
                "past": {
                    "other": "{0} hafta önce",
                    "one": "{0} hafta önce",
                },
            },
            "day": {
                "future": {
                    "other": "{0} gün sonra",
                    "one": "{0} gün sonra",
                },
                "past": {
                    "other": "{0} gün önce",
                    "one": "{0} gün önce",
                },
            },
            "hour": {
                "future": {
                    "other": "{0} saat sonra",
                    "one": "{0} saat sonra",
                },
                "past": {
                    "other": "{0} saat önce",
                    "one": "{0} saat önce",
                },
            },
            "minute": {
                "future": {
                    "other": "{0} dakika sonra",
                    "one": "{0} dakika sonra",
                },
                "past": {
                    "other": "{0} dakika önce",
                    "one": "{0} dakika önce",
                },
            },
            "second": {
                "future": {
                    "other": "{0} saniye sonra",
                    "one": "{0} saniye sonra",
                },
                "past": {
                    "other": "{0} saniye önce",
                    "one": "{0} saniye önce",
                },
            },
        },
        "day_periods": {
            "midnight": "gece yarısı",
            "am": "ÖÖ",
            "noon": "öğle",
            "pm": "ÖS",
            "morning1": "sabah",
            "morning2": "öğleden önce",
            "afternoon1": "öğleden sonra",
            "afternoon2": "akşamüstü",
            "evening1": "akşam",
            "night1": "gece",
        },
        "week_data": {
            "min_days": 1,
            "first_day": 0,
            "weekend_start": 5,
            "weekend_end": 6,
        },
    },
    "custom": custom_translations,
}


# === src/pendulum/mixins/__init__.py ===


# === src/pendulum/mixins/default.py ===
from __future__ import annotations

from pendulum.formatting import Formatter


_formatter = Formatter()


class FormattableMixin:
    _formatter: Formatter = _formatter

    def format(self, fmt: str, locale: str | None = None) -> str:
        """
        Formats the instance using the given format.

        :param fmt: The format to use
        :param locale: The locale to use
        """
        return self._formatter.format(self, fmt, locale)

    def for_json(self) -> str:
        """
        Methods for automatic json serialization by simplejson.
        """
        return self.isoformat()

    def __format__(self, format_spec: str) -> str:
        if len(format_spec) > 0:
            if "%" in format_spec:
                return self.strftime(format_spec)

            return self.format(format_spec)

        return str(self)

    def __str__(self) -> str:
        return self.isoformat()


# === src/pendulum/utils/__init__.py ===


# === src/pendulum/utils/_compat.py ===
from __future__ import annotations

import sys


PYPY = hasattr(sys, "pypy_version_info")


# === src/pendulum/testing/__init__.py ===


# === src/pendulum/testing/traveller.py ===
from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

from pendulum.datetime import DateTime
from pendulum.utils._compat import PYPY


if TYPE_CHECKING:
    from types import TracebackType

    from typing_extensions import Self


class BaseTraveller:
    def __init__(self, datetime_class: type[DateTime] = DateTime) -> None:
        self._datetime_class: type[DateTime] = datetime_class

    def freeze(self) -> Self:
        raise self._not_implemented()

    def travel_back(self) -> Self:
        raise self._not_implemented()

    def travel(
        self,
        years: int = 0,
        months: int = 0,
        weeks: int = 0,
        days: int = 0,
        hours: int = 0,
        minutes: int = 0,
        seconds: int = 0,
        microseconds: int = 0,
    ) -> Self:
        raise self._not_implemented()

    def travel_to(self, dt: DateTime, *, freeze: bool = False) -> Self:
        raise self._not_implemented()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType,
    ) -> None: ...

    def _not_implemented(self) -> NotImplementedError:
        return NotImplementedError()


if not PYPY:
    try:
        import time_machine
    except ImportError:
        time_machine = None  # type: ignore[assignment]

    if time_machine is not None:

        class Traveller(BaseTraveller):
            def __init__(self, datetime_class: type[DateTime] = DateTime) -> None:
                super().__init__(datetime_class)

                self._started: bool = False
                self._traveller: time_machine.travel | None = None
                self._coordinates: time_machine.Coordinates | None = None

            def freeze(self) -> Self:
                if self._started:
                    cast("time_machine.Coordinates", self._coordinates).move_to(
                        self._datetime_class.now(), tick=False
                    )
                else:
                    self._start(freeze=True)

                return self

            def travel_back(self) -> Self:
                if not self._started:
                    return self

                cast("time_machine.travel", self._traveller).stop()
                self._coordinates = None
                self._traveller = None
                self._started = False

                return self

            def travel(
                self,
                years: int = 0,
                months: int = 0,
                weeks: int = 0,
                days: int = 0,
                hours: int = 0,
                minutes: int = 0,
                seconds: int = 0,
                microseconds: int = 0,
                *,
                freeze: bool = False,
            ) -> Self:
                self._start(freeze=freeze)

                cast("time_machine.Coordinates", self._coordinates).move_to(
                    self._datetime_class.now().add(
                        years=years,
                        months=months,
                        weeks=weeks,
                        days=days,
                        hours=hours,
                        minutes=minutes,
                        seconds=seconds,
                        microseconds=microseconds,
                    )
                )

                return self

            def travel_to(self, dt: DateTime, *, freeze: bool = False) -> Self:
                self._start(freeze=freeze)

                cast("time_machine.Coordinates", self._coordinates).move_to(dt)

                return self

            def _start(self, freeze: bool = False) -> None:
                if self._started:
                    return

                if not self._traveller:
                    self._traveller = time_machine.travel(
                        self._datetime_class.now(), tick=not freeze
                    )

                self._coordinates = self._traveller.start()

                self._started = True

            def __enter__(self) -> Self:
                self._start()

                return self

            def __exit__(
                self,
                exc_type: type[BaseException] | None,
                exc_val: BaseException | None,
                exc_tb: TracebackType,
            ) -> None:
                self.travel_back()

    else:

        class Traveller(BaseTraveller):  # type: ignore[no-redef]
            def _not_implemented(self) -> NotImplementedError:
                return NotImplementedError(
                    "Time travelling is an optional feature. "
                    'You can add it by installing Pendulum with the "test" extra.'
                )

else:

    class Traveller(BaseTraveller):  # type: ignore[no-redef]
        def _not_implemented(self) -> NotImplementedError:
            return NotImplementedError(
                "Time travelling is not supported on the PyPy Python implementation."
            )


# === src/pendulum/formatting/difference_formatter.py ===
from __future__ import annotations

import typing as t

from pendulum.locales.locale import Locale


if t.TYPE_CHECKING:
    from pendulum import Duration

DAYS_THRESHOLD_FOR_HALF_WEEK = 3
DAYS_THRESHOLD_FOR_HALF_MONTH = 15
MONTHS_THRESHOLD_FOR_HALF_YEAR = 6

HOURS_IN_NEARLY_A_DAY = 22
DAYS_IN_NEARLY_A_MONTH = 27
MONTHS_IN_NEARLY_A_YEAR = 11

DAYS_OF_WEEK = 7
SECONDS_OF_MINUTE = 60
FEW_SECONDS_MAX = 10

KEY_FUTURE = ".future"
KEY_PAST = ".past"
KEY_AFTER = ".after"
KEY_BEFORE = ".before"


class DifferenceFormatter:
    """
    Handles formatting differences in text.
    """

    def __init__(self, locale: str = "en") -> None:
        self._locale = Locale.load(locale)

    def format(
        self,
        diff: Duration,
        is_now: bool = True,
        absolute: bool = False,
        locale: str | Locale | None = None,
    ) -> str:
        """
        Formats a difference.

        :param diff: The difference to format
        :param is_now: Whether the difference includes now
        :param absolute: Whether it's an absolute difference or not
        :param locale: The locale to use
        """
        locale = self._locale if locale is None else Locale.load(locale)

        if diff.years > 0:
            unit = "year"
            count = diff.years

            if diff.months > MONTHS_THRESHOLD_FOR_HALF_YEAR:
                count += 1
        elif (diff.months == MONTHS_IN_NEARLY_A_YEAR) and (
            (diff.weeks * DAYS_OF_WEEK + diff.remaining_days)
            > DAYS_THRESHOLD_FOR_HALF_MONTH
        ):
            unit = "year"
            count = 1
        elif diff.months > 0:
            unit = "month"
            count = diff.months

            if (
                diff.weeks * DAYS_OF_WEEK + diff.remaining_days
            ) >= DAYS_IN_NEARLY_A_MONTH:
                count += 1
        elif diff.weeks > 0:
            unit = "week"
            count = diff.weeks

            if diff.remaining_days > DAYS_THRESHOLD_FOR_HALF_WEEK:
                count += 1
        elif diff.remaining_days > 0:
            unit = "day"
            count = diff.remaining_days

            if diff.hours >= HOURS_IN_NEARLY_A_DAY:
                count += 1
        elif diff.hours > 0:
            unit = "hour"
            count = diff.hours
        elif diff.minutes > 0:
            unit = "minute"
            count = diff.minutes
        elif FEW_SECONDS_MAX < diff.remaining_seconds < SECONDS_OF_MINUTE:
            unit = "second"
            count = diff.remaining_seconds
        else:
            # We check if the "a few seconds" unit exists
            time = locale.get("custom.units.few_second")
            if time is not None:
                if absolute:
                    return t.cast("str", time)

                key = "custom"
                is_future = diff.invert
                if is_now:
                    if is_future:
                        key += ".from_now"
                    else:
                        key += ".ago"
                else:
                    if is_future:
                        key += KEY_AFTER
                    else:
                        key += KEY_BEFORE

                return t.cast("str", locale.get(key).format(time))
            else:
                unit = "second"
                count = diff.remaining_seconds
        if count == 0:
            count = 1
        if absolute:
            key = f"translations.units.{unit}"
        else:
            is_future = diff.invert
            if is_now:
                # Relative to now, so we can use
                # the CLDR data
                key = f"translations.relative.{unit}"

                if is_future:
                    key += KEY_FUTURE
                else:
                    key += KEY_PAST
            else:
                # Absolute comparison
                # So we have to use the custom locale data

                # Checking for special pluralization rules
                key = "custom.units_relative"
                if is_future:
                    key += f".{unit}{KEY_FUTURE}"
                else:
                    key += f".{unit}{KEY_PAST}"

                trans = locale.get(key)
                if not trans:
                    # No special rule
                    key = f"translations.units.{unit}.{locale.plural(count)}"
                    time = locale.get(key).format(count)
                else:
                    time = trans[locale.plural(count)].format(count)

                key = "custom"
                if is_future:
                    key += KEY_AFTER
                else:
                    key += KEY_BEFORE

                return t.cast("str", locale.get(key).format(time))

        key += f".{locale.plural(count)}"

        return t.cast("str", locale.get(key).format(count))


# === src/pendulum/formatting/formatter.py ===
from __future__ import annotations

import datetime
import re

from re import Match
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import cast

import pendulum

from pendulum.locales.locale import Locale


if TYPE_CHECKING:
    from collections.abc import Sequence

    from pendulum import Timezone

_MATCH_1 = r"\d"
_MATCH_2 = r"\d\d"
_MATCH_3 = r"\d{3}"
_MATCH_4 = r"\d{4}"
_MATCH_6 = r"[+-]?\d{6}"
_MATCH_1_TO_2 = r"\d\d?"
_MATCH_1_TO_2_LEFT_PAD = r"[0-9 ]\d?"
_MATCH_1_TO_3 = r"\d{1,3}"
_MATCH_1_TO_4 = r"\d{1,4}"
_MATCH_1_TO_6 = r"[+-]?\d{1,6}"
_MATCH_3_TO_4 = r"\d{3}\d?"
_MATCH_5_TO_6 = r"\d{5}\d?"
_MATCH_UNSIGNED = r"\d+"
_MATCH_SIGNED = r"[+-]?\d+"
_MATCH_OFFSET = r"[Zz]|[+-]\d\d:?\d\d"
_MATCH_SHORT_OFFSET = r"[Zz]|[+-]\d\d(?::?\d\d)?"
_MATCH_TIMESTAMP = r"[+-]?\d+(\.\d{1,6})?"
_MATCH_WORD = (
    "(?i)[0-9]*"
    "['a-z\u00a0-\u05ff\u0700-\ud7ff\uf900-\ufdcf\ufdf0-\uffef]+"
    r"|[\u0600-\u06FF/]+(\s*?[\u0600-\u06FF]+){1,2}"
)
_MATCH_TIMEZONE = "[A-Za-z0-9-+]+(/[A-Za-z0-9-+_]+)?"


class Formatter:
    _TOKENS: str = (
        r"\[([^\[]*)\]|\\(.)|"
        "("
        "Mo|MM?M?M?"
        "|Do|DDDo|DD?D?D?|ddd?d?|do?|eo?"
        "|E{1,4}"
        "|w[o|w]?|W[o|W]?|Qo?"
        "|YYYY|YY|Y"
        "|gg(ggg?)?|GG(GGG?)?"
        "|a|A"
        "|hh?|HH?|kk?"
        "|mm?|ss?|S{1,9}"
        "|x|X"
        "|zz?|ZZ?"
        "|LTS|LT|LL?L?L?"
        ")"
    )

    _FORMAT_RE: re.Pattern[str] = re.compile(_TOKENS)

    _FROM_FORMAT_RE: re.Pattern[str] = re.compile(r"(?<!\\\[)" + _TOKENS + r"(?!\\\])")

    _LOCALIZABLE_TOKENS: ClassVar[
        dict[str, str | Callable[[Locale], Sequence[str]] | None]
    ] = {
        "Qo": None,
        "MMMM": "months.wide",
        "MMM": "months.abbreviated",
        "Mo": None,
        "DDDo": None,
        "Do": lambda locale: tuple(
            rf"\d+{o}" for o in locale.get("custom.ordinal").values()
        ),
        "dddd": "days.wide",
        "ddd": "days.abbreviated",
        "dd": "days.short",
        "do": None,
        "e": None,
        "eo": None,
        "Wo": None,
        "wo": None,
        "A": lambda locale: (
            locale.translation("day_periods.am"),
            locale.translation("day_periods.pm"),
        ),
        "a": lambda locale: (
            locale.translation("day_periods.am").lower(),
            locale.translation("day_periods.pm").lower(),
        ),
    }

    _TOKENS_RULES: ClassVar[dict[str, Callable[[pendulum.DateTime], str]]] = {
        # Year
        "YYYY": lambda dt: f"{dt.year:d}",
        "YY": lambda dt: f"{dt.year:d}"[2:],
        "Y": lambda dt: f"{dt.year:d}",
        # Quarter
        "Q": lambda dt: f"{dt.quarter:d}",
        # Month
        "MM": lambda dt: f"{dt.month:02d}",
        "M": lambda dt: f"{dt.month:d}",
        # Day
        "DD": lambda dt: f"{dt.day:02d}",
        "D": lambda dt: f"{dt.day:d}",
        # Day of Year
        "DDDD": lambda dt: f"{dt.day_of_year:03d}",
        "DDD": lambda dt: f"{dt.day_of_year:d}",
        # Day of Week
        "d": lambda dt: f"{(dt.day_of_week + 1) % 7:d}",
        # Day of ISO Week
        "E": lambda dt: f"{dt.isoweekday():d}",
        # Hour
        "HH": lambda dt: f"{dt.hour:02d}",
        "H": lambda dt: f"{dt.hour:d}",
        "hh": lambda dt: f"{dt.hour % 12 or 12:02d}",
        "h": lambda dt: f"{dt.hour % 12 or 12:d}",
        # Minute
        "mm": lambda dt: f"{dt.minute:02d}",
        "m": lambda dt: f"{dt.minute:d}",
        # Second
        "ss": lambda dt: f"{dt.second:02d}",
        "s": lambda dt: f"{dt.second:d}",
        # Fractional second
        "S": lambda dt: f"{dt.microsecond // 100000:01d}",
        "SS": lambda dt: f"{dt.microsecond // 10000:02d}",
        "SSS": lambda dt: f"{dt.microsecond // 1000:03d}",
        "SSSS": lambda dt: f"{dt.microsecond // 100:04d}",
        "SSSSS": lambda dt: f"{dt.microsecond // 10:05d}",
        "SSSSSS": lambda dt: f"{dt.microsecond:06d}",
        # Timestamp
        "X": lambda dt: f"{dt.int_timestamp:d}",
        "x": lambda dt: f"{dt.int_timestamp * 1000 + dt.microsecond // 1000:d}",
        # Timezone
        "zz": lambda dt: f"{dt.tzname() if dt.tzinfo is not None else ''}",
        "z": lambda dt: f"{dt.timezone_name or ''}",
    }

    _DATE_FORMATS: ClassVar[dict[str, str]] = {
        "LTS": "formats.time.full",
        "LT": "formats.time.short",
        "L": "formats.date.short",
        "LL": "formats.date.long",
        "LLL": "formats.datetime.long",
        "LLLL": "formats.datetime.full",
    }

    _DEFAULT_DATE_FORMATS: ClassVar[dict[str, str]] = {
        "LTS": "h:mm:ss A",
        "LT": "h:mm A",
        "L": "MM/DD/YYYY",
        "LL": "MMMM D, YYYY",
        "LLL": "MMMM D, YYYY h:mm A",
        "LLLL": "dddd, MMMM D, YYYY h:mm A",
    }

    _REGEX_TOKENS: ClassVar[dict[str, str | Sequence[str] | None]] = {
        "Y": _MATCH_SIGNED,
        "YY": (_MATCH_1_TO_2, _MATCH_2),
        "YYYY": (_MATCH_1_TO_4, _MATCH_4),
        "Q": _MATCH_1,
        "Qo": None,
        "M": _MATCH_1_TO_2,
        "MM": (_MATCH_1_TO_2, _MATCH_2),
        "MMM": _MATCH_WORD,
        "MMMM": _MATCH_WORD,
        "D": _MATCH_1_TO_2,
        "DD": (_MATCH_1_TO_2_LEFT_PAD, _MATCH_2),
        "DDD": _MATCH_1_TO_3,
        "DDDD": _MATCH_3,
        "dddd": _MATCH_WORD,
        "ddd": _MATCH_WORD,
        "dd": _MATCH_WORD,
        "d": _MATCH_1,
        "e": _MATCH_1,
        "E": _MATCH_1,
        "Do": None,
        "H": _MATCH_1_TO_2,
        "HH": (_MATCH_1_TO_2, _MATCH_2),
        "h": _MATCH_1_TO_2,
        "hh": (_MATCH_1_TO_2, _MATCH_2),
        "m": _MATCH_1_TO_2,
        "mm": (_MATCH_1_TO_2, _MATCH_2),
        "s": _MATCH_1_TO_2,
        "ss": (_MATCH_1_TO_2, _MATCH_2),
        "S": (_MATCH_1_TO_3, _MATCH_1),
        "SS": (_MATCH_1_TO_3, _MATCH_2),
        "SSS": (_MATCH_1_TO_3, _MATCH_3),
        "SSSS": _MATCH_UNSIGNED,
        "SSSSS": _MATCH_UNSIGNED,
        "SSSSSS": _MATCH_UNSIGNED,
        "x": _MATCH_SIGNED,
        "X": _MATCH_TIMESTAMP,
        "ZZ": _MATCH_SHORT_OFFSET,
        "Z": _MATCH_OFFSET,
        "z": _MATCH_TIMEZONE,
    }

    _PARSE_TOKENS: ClassVar[dict[str, Callable[[str], Any]]] = {
        "YYYY": lambda year: int(year),
        "YY": lambda year: int(year),
        "Q": lambda quarter: int(quarter),
        "MMMM": lambda month: month,
        "MMM": lambda month: month,
        "MM": lambda month: int(month),
        "M": lambda month: int(month),
        "DDDD": lambda day: int(day),
        "DDD": lambda day: int(day),
        "DD": lambda day: int(day),
        "D": lambda day: int(day),
        "dddd": lambda weekday: weekday,
        "ddd": lambda weekday: weekday,
        "dd": lambda weekday: weekday,
        "d": lambda weekday: int(weekday),
        "E": lambda weekday: int(weekday) - 1,
        "HH": lambda hour: int(hour),
        "H": lambda hour: int(hour),
        "hh": lambda hour: int(hour),
        "h": lambda hour: int(hour),
        "mm": lambda minute: int(minute),
        "m": lambda minute: int(minute),
        "ss": lambda second: int(second),
        "s": lambda second: int(second),
        "S": lambda us: int(us) * 100000,
        "SS": lambda us: int(us) * 10000,
        "SSS": lambda us: int(us) * 1000,
        "SSSS": lambda us: int(us) * 100,
        "SSSSS": lambda us: int(us) * 10,
        "SSSSSS": lambda us: int(us),
        "a": lambda meridiem: meridiem,
        "X": lambda ts: float(ts),
        "x": lambda ts: float(ts) / 1e3,
        "ZZ": str,
        "Z": str,
        "z": str,
    }

    def format(
        self, dt: pendulum.DateTime, fmt: str, locale: str | Locale | None = None
    ) -> str:
        """
        Formats a DateTime instance with a given format and locale.

        :param dt: The instance to format
        :param fmt: The format to use
        :param locale: The locale to use
        """
        loaded_locale: Locale = Locale.load(locale or pendulum.get_locale())

        result = self._FORMAT_RE.sub(
            lambda m: m.group(1)
            if m.group(1)
            else m.group(2)
            if m.group(2)
            else self._format_token(dt, m.group(3), loaded_locale),
            fmt,
        )

        return result

    def _format_token(self, dt: pendulum.DateTime, token: str, locale: Locale) -> str:
        """
        Formats a DateTime instance with a given token and locale.

        :param dt: The instance to format
        :param token: The token to use
        :param locale: The locale to use
        """
        if token in self._DATE_FORMATS:
            fmt = locale.get(f"custom.date_formats.{token}")
            if fmt is None:
                fmt = self._DEFAULT_DATE_FORMATS[token]

            return self.format(dt, fmt, locale)

        if token in self._LOCALIZABLE_TOKENS:
            return self._format_localizable_token(dt, token, locale)

        if token in self._TOKENS_RULES:
            return self._TOKENS_RULES[token](dt)

        # Timezone
        if token in ["ZZ", "Z"]:
            if dt.tzinfo is None:
                return ""

            separator = ":" if token == "Z" else ""
            offset = dt.utcoffset() or datetime.timedelta()
            minutes = offset.total_seconds() / 60

            sign = "+" if minutes >= 0 else "-"

            hour, minute = divmod(abs(int(minutes)), 60)

            return f"{sign}{hour:02d}{separator}{minute:02d}"

        return token

    def _format_localizable_token(
        self, dt: pendulum.DateTime, token: str, locale: Locale
    ) -> str:
        """
        Formats a DateTime instance
        with a given localizable token and locale.

        :param dt: The instance to format
        :param token: The token to use
        :param locale: The locale to use
        """
        if token == "MMM":
            return cast("str", locale.get("translations.months.abbreviated")[dt.month])
        elif token == "MMMM":
            return cast("str", locale.get("translations.months.wide")[dt.month])
        elif token == "dd":
            return cast("str", locale.get("translations.days.short")[dt.day_of_week])
        elif token == "ddd":
            return cast(
                "str",
                locale.get("translations.days.abbreviated")[dt.day_of_week],
            )
        elif token == "dddd":
            return cast("str", locale.get("translations.days.wide")[dt.day_of_week])
        elif token == "e":
            first_day = cast("int", locale.get("translations.week_data.first_day"))

            return str((dt.day_of_week % 7 - first_day) % 7)
        elif token == "Do":
            return locale.ordinalize(dt.day)
        elif token == "do":
            return locale.ordinalize((dt.day_of_week + 1) % 7)
        elif token == "Mo":
            return locale.ordinalize(dt.month)
        elif token == "Qo":
            return locale.ordinalize(dt.quarter)
        elif token == "wo":
            return locale.ordinalize(dt.week_of_year)
        elif token == "DDDo":
            return locale.ordinalize(dt.day_of_year)
        elif token == "eo":
            first_day = cast("int", locale.get("translations.week_data.first_day"))

            return locale.ordinalize((dt.day_of_week % 7 - first_day) % 7 + 1)
        elif token == "A":
            key = "translations.day_periods"
            if dt.hour >= 12:
                key += ".pm"
            else:
                key += ".am"

            return cast("str", locale.get(key))
        else:
            return token

    def parse(
        self,
        time: str,
        fmt: str,
        now: pendulum.DateTime,
        locale: str | None = None,
    ) -> dict[str, Any]:
        """
        Parses a time string matching a given format as a tuple.

        :param time: The timestring
        :param fmt: The format
        :param now: The datetime to use as "now"
        :param locale: The locale to use

        :return: The parsed elements
        """
        escaped_fmt = re.escape(fmt)

        if not self._FROM_FORMAT_RE.search(escaped_fmt):
            raise ValueError("The given time string does not match the given format")

        if not locale:
            locale = pendulum.get_locale()

        loaded_locale: Locale = Locale.load(locale)

        parsed = {
            "year": None,
            "month": None,
            "day": None,
            "hour": None,
            "minute": None,
            "second": None,
            "microsecond": None,
            "tz": None,
            "quarter": None,
            "day_of_week": None,
            "day_of_year": None,
            "meridiem": None,
            "timestamp": None,
        }

        pattern = self._FROM_FORMAT_RE.sub(
            lambda m: self._replace_tokens(m.group(0), loaded_locale), escaped_fmt
        )

        if not re.fullmatch(pattern, time):
            raise ValueError(f"String does not match format {fmt}")

        def _get_parsed_values(m: Match[str]) -> Any:
            return self._get_parsed_values(m, parsed, loaded_locale, now)

        re.sub(pattern, _get_parsed_values, time)

        return self._check_parsed(parsed, now)

    def _check_parsed(
        self, parsed: dict[str, Any], now: pendulum.DateTime
    ) -> dict[str, Any]:
        """
        Checks validity of parsed elements.

        :param parsed: The elements to parse.

        :return: The validated elements.
        """
        validated: dict[str, int | Timezone | None] = {
            "year": parsed["year"],
            "month": parsed["month"],
            "day": parsed["day"],
            "hour": parsed["hour"],
            "minute": parsed["minute"],
            "second": parsed["second"],
            "microsecond": parsed["microsecond"],
            "tz": None,
        }

        # If timestamp has been specified
        # we use it and don't go any further
        if parsed["timestamp"] is not None:
            str_us = str(parsed["timestamp"])
            if "." in str_us:
                microseconds = int(f"{str_us.split('.')[1].ljust(6, '0')}")
            else:
                microseconds = 0

            from pendulum.helpers import local_time

            time = local_time(parsed["timestamp"], 0, microseconds)
            validated["year"] = time[0]
            validated["month"] = time[1]
            validated["day"] = time[2]
            validated["hour"] = time[3]
            validated["minute"] = time[4]
            validated["second"] = time[5]
            validated["microsecond"] = time[6]

            return validated

        if parsed["quarter"] is not None:
            if validated["year"] is not None:
                dt = pendulum.datetime(cast("int", validated["year"]), 1, 1)
            else:
                dt = now

            dt = dt.start_of("year")

            while dt.quarter != parsed["quarter"]:
                dt = dt.add(months=3)

            validated["year"] = dt.year
            validated["month"] = dt.month
            validated["day"] = dt.day

        if validated["year"] is None:
            validated["year"] = now.year

        if parsed["day_of_year"] is not None:
            dt = cast(
                "pendulum.DateTime",
                pendulum.parse(f"{validated['year']}-{parsed['day_of_year']:>03d}"),
            )

            validated["month"] = dt.month
            validated["day"] = dt.day

        if parsed["day_of_week"] is not None:
            dt = pendulum.datetime(
                cast("int", validated["year"]),
                cast("int", validated["month"]) or now.month,
                cast("int", validated["day"]) or now.day,
            )
            dt = dt.start_of("week").subtract(days=1)
            dt = dt.next(parsed["day_of_week"])
            validated["year"] = dt.year
            validated["month"] = dt.month
            validated["day"] = dt.day

        # Meridiem
        if parsed["meridiem"] is not None:
            # If the time is greater than 13:00:00
            # This is not valid
            if validated["hour"] is None:
                raise ValueError("Invalid Date")

            t = (
                validated["hour"],
                validated["minute"],
                validated["second"],
                validated["microsecond"],
            )
            if t >= (13, 0, 0, 0):
                raise ValueError("Invalid date")

            pm = parsed["meridiem"] == "pm"
            validated["hour"] %= 12  # type: ignore[operator]
            if pm:
                validated["hour"] += 12  # type: ignore[operator]

        if validated["month"] is None:
            if parsed["year"] is not None:
                validated["month"] = parsed["month"] or 1
            else:
                validated["month"] = parsed["month"] or now.month

        if validated["day"] is None:
            if parsed["year"] is not None or parsed["month"] is not None:
                validated["day"] = parsed["day"] or 1
            else:
                validated["day"] = parsed["day"] or now.day

        for part in ["hour", "minute", "second", "microsecond"]:
            if validated[part] is None:
                validated[part] = 0

        validated["tz"] = parsed["tz"]

        return validated

    def _get_parsed_values(
        self,
        m: Match[str],
        parsed: dict[str, Any],
        locale: Locale,
        now: pendulum.DateTime,
    ) -> None:
        for token, index in m.re.groupindex.items():
            if token in self._LOCALIZABLE_TOKENS:
                self._get_parsed_locale_value(token, m.group(index), parsed, locale)
            else:
                self._get_parsed_value(token, m.group(index), parsed, now)

    def _get_parsed_value(
        self,
        token: str,
        value: str,
        parsed: dict[str, Any],
        now: pendulum.DateTime,
    ) -> None:
        parsed_token = self._PARSE_TOKENS[token](value)

        if "Y" in token:
            if token == "YY":
                if parsed_token <= 68:
                    parsed_token += 2000
                else:
                    parsed_token += 1900

            parsed["year"] = parsed_token
        elif token == "Q":
            parsed["quarter"] = parsed_token
        elif token in ["MM", "M"]:
            parsed["month"] = parsed_token
        elif token in ["DDDD", "DDD"]:
            parsed["day_of_year"] = parsed_token
        elif "D" in token:
            parsed["day"] = parsed_token
        elif "H" in token:
            parsed["hour"] = parsed_token
        elif token in ["hh", "h"]:
            if parsed_token > 12:
                raise ValueError("Invalid date")

            parsed["hour"] = parsed_token
        elif "m" in token:
            parsed["minute"] = parsed_token
        elif "s" in token:
            parsed["second"] = parsed_token
        elif "S" in token:
            parsed["microsecond"] = parsed_token
        elif token in ["d", "E"]:
            parsed["day_of_week"] = parsed_token
        elif token in ["X", "x"]:
            parsed["timestamp"] = parsed_token
        elif token in ["ZZ", "Z"]:
            negative = bool(value.startswith("-"))
            tz = value[1:]
            if ":" not in tz:
                if len(tz) == 2:
                    tz = f"{tz}00"

                off_hour = tz[0:2]
                off_minute = tz[2:4]
            else:
                off_hour, off_minute = tz.split(":")

            offset = ((int(off_hour) * 60) + int(off_minute)) * 60

            if negative:
                offset = -1 * offset

            parsed["tz"] = pendulum.timezone(offset)
        elif token == "z":
            # Full timezone
            if value not in pendulum.timezones():
                raise ValueError("Invalid date")

            parsed["tz"] = pendulum.timezone(value)

    def _get_parsed_locale_value(
        self, token: str, value: str, parsed: dict[str, Any], locale: Locale
    ) -> None:
        if token == "MMMM":
            unit = "month"
            match = "months.wide"
        elif token == "MMM":
            unit = "month"
            match = "months.abbreviated"
        elif token == "Do":
            parsed["day"] = int(cast("Match[str]", re.match(r"(\d+)", value)).group(1))

            return
        elif token == "dddd":
            unit = "day_of_week"
            match = "days.wide"
        elif token == "ddd":
            unit = "day_of_week"
            match = "days.abbreviated"
        elif token == "dd":
            unit = "day_of_week"
            match = "days.short"
        elif token in ["a", "A"]:
            valid_values = [
                locale.translation("day_periods.am"),
                locale.translation("day_periods.pm"),
            ]

            if token == "a":
                value = value.lower()
                valid_values = [x.lower() for x in valid_values]

            if value not in valid_values:
                raise ValueError("Invalid date")

            parsed["meridiem"] = ["am", "pm"][valid_values.index(value)]

            return
        else:
            raise ValueError(f'Invalid token "{token}"')

        parsed[unit] = locale.match_translation(match, value)
        if value is None:
            raise ValueError("Invalid date")

    def _replace_tokens(self, token: str, locale: Locale) -> str:
        if token.startswith("[") and token.endswith("]"):
            return token[1:-1]
        elif token.startswith("\\"):
            if len(token) == 2 and token[1] in {"[", "]"}:
                return ""

            return token
        elif token not in self._REGEX_TOKENS and token not in self._LOCALIZABLE_TOKENS:
            raise ValueError(f"Unsupported token: {token}")

        if token in self._LOCALIZABLE_TOKENS:
            values = self._LOCALIZABLE_TOKENS[token]
            if callable(values):
                candidates = values(locale)
            else:
                candidates = tuple(
                    locale.translation(
                        cast("str", self._LOCALIZABLE_TOKENS[token])
                    ).values()
                )
        else:
            candidates = cast("Sequence[str]", self._REGEX_TOKENS[token])

        if not candidates:
            raise ValueError(f"Unsupported token: {token}")

        if not isinstance(candidates, tuple):
            candidates = (cast("str", candidates),)

        pattern = f"(?P<{token}>{'|'.join(candidates)})"

        return pattern


# === src/pendulum/formatting/__init__.py ===
from __future__ import annotations

from pendulum.formatting.formatter import Formatter


__all__ = ["Formatter"]


# === src/pendulum/parsing/__init__.py ===
from __future__ import annotations

import contextlib
import copy
import os
import re

from datetime import date
from datetime import datetime
from datetime import time
from typing import Any
from typing import Optional
from typing import cast

from dateutil import parser

from pendulum.parsing.exceptions import ParserError


with_extensions = os.getenv("PENDULUM_EXTENSIONS", "1") == "1"

try:
    if not with_extensions:
        raise ImportError()

    from pendulum._pendulum import Duration
    from pendulum._pendulum import parse_iso8601
except ImportError:
    from pendulum.duration import Duration  # type: ignore[assignment]  # noqa: TC001
    from pendulum.parsing.iso8601 import parse_iso8601  # type: ignore[assignment]


COMMON = re.compile(
    # Date (optional)  # noqa: ERA001
    "^"
    "(?P<date>"
    "    (?P<classic>"  # Classic date (YYYY-MM-DD)
    r"        (?P<year>\d{4})"  # Year
    "        (?P<monthday>"
    r"            (?P<monthsep>[/:])?(?P<month>\d{2})"  # Month (optional)
    r"            ((?P<daysep>[/:])?(?P<day>\d{2}))"  # Day (optional)
    "        )?"
    "    )"
    ")?"
    # Time (optional)  # noqa: ERA001
    "(?P<time>"
    r"    (?P<timesep>\ )?"  # Separator (space)
    # HH:mm:ss (optional mm and ss)
    r"    (?P<hour>\d{1,2}):(?P<minute>\d{1,2})?(?::(?P<second>\d{1,2}))?"
    # Subsecond part (optional)
    "    (?P<subsecondsection>"
    "        (?:[.|,])"  # Subsecond separator (optional)
    r"        (?P<subsecond>\d{1,9})"  # Subsecond
    "    )?"
    ")?"
    "$",
    re.VERBOSE,
)

DEFAULT_OPTIONS = {
    "day_first": False,
    "year_first": True,
    "strict": True,
    "exact": False,
    "now": None,
}


def parse(text: str, **options: Any) -> datetime | date | time | _Interval | Duration:
    """
    Parses a string with the given options.

    :param text: The string to parse.
    """
    _options: dict[str, Any] = copy.copy(DEFAULT_OPTIONS)
    _options.update(options)

    return _normalize(_parse(text, **_options), **_options)


def _normalize(
    parsed: datetime | date | time | _Interval | Duration, **options: Any
) -> datetime | date | time | _Interval | Duration:
    """
    Normalizes the parsed element.

    :param parsed: The parsed elements.
    """
    if options.get("exact"):
        return parsed

    if isinstance(parsed, time):
        now = cast("Optional[datetime]", options["now"]) or datetime.now()

        return datetime(
            now.year,
            now.month,
            now.day,
            parsed.hour,
            parsed.minute,
            parsed.second,
            parsed.microsecond,
        )
    elif isinstance(parsed, date) and not isinstance(parsed, datetime):
        return datetime(parsed.year, parsed.month, parsed.day)

    return parsed


def _parse(text: str, **options: Any) -> datetime | date | time | _Interval | Duration:
    # Trying to parse ISO8601
    with contextlib.suppress(ValueError):
        return parse_iso8601(text)

    with contextlib.suppress(ValueError):
        return _parse_iso8601_interval(text)

    with contextlib.suppress(ParserError):
        return _parse_common(text, **options)

    # We couldn't parse the string
    # so we fallback on the dateutil parser
    # If not strict
    if options.get("strict", True):
        raise ParserError(f"Unable to parse string [{text}]")

    try:
        dt = parser.parse(
            text, dayfirst=options["day_first"], yearfirst=options["year_first"]
        )
    except ValueError:
        raise ParserError(f"Invalid date string: {text}")

    return dt


def _parse_common(text: str, **options: Any) -> datetime | date | time:
    """
    Tries to parse the string as a common datetime format.

    :param text: The string to parse.
    """
    m = COMMON.fullmatch(text)
    has_date = False
    year = 0
    month = 1
    day = 1

    if not m:
        raise ParserError("Invalid datetime string")

    if m.group("date"):
        # A date has been specified
        has_date = True

        year = int(m.group("year"))

        if not m.group("monthday"):
            # No month and day
            month = 1
            day = 1
        else:
            if options["day_first"]:
                month = int(m.group("day"))
                day = int(m.group("month"))
            else:
                month = int(m.group("month"))
                day = int(m.group("day"))

    if not m.group("time"):
        return date(year, month, day)

    # Grabbing hh:mm:ss
    hour = int(m.group("hour"))

    minute = int(m.group("minute"))

    second = int(m.group("second")) if m.group("second") else 0

    # Grabbing subseconds, if any
    microsecond = 0
    if m.group("subsecondsection"):
        # Limiting to 6 chars
        subsecond = m.group("subsecond")[:6]

        microsecond = int(f"{subsecond:0<6}")

    if has_date:
        return datetime(year, month, day, hour, minute, second, microsecond)

    return time(hour, minute, second, microsecond)


class _Interval:
    """
    Special class to handle ISO 8601 intervals
    """

    def __init__(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        duration: Duration | None = None,
    ) -> None:
        self.start = start
        self.end = end
        self.duration = duration


def _parse_iso8601_interval(text: str) -> _Interval:
    if "/" not in text:
        raise ParserError("Invalid interval")

    first, last = text.split("/")
    start = end = duration = None

    if first[:1] == "P":
        # duration/end
        duration = parse_iso8601(first)
        end = parse_iso8601(last)
    elif last[:1] == "P":
        # start/duration
        start = parse_iso8601(first)
        duration = parse_iso8601(last)
    else:
        # start/end
        start = parse_iso8601(first)
        end = parse_iso8601(last)

    return _Interval(
        cast("datetime", start), cast("datetime", end), cast("Duration", duration)
    )


__all__ = ["parse", "parse_iso8601"]


# === src/pendulum/parsing/iso8601.py ===
from __future__ import annotations

import datetime
import re

from typing import cast

from pendulum.constants import HOURS_PER_DAY
from pendulum.constants import MINUTES_PER_HOUR
from pendulum.constants import MONTHS_OFFSETS
from pendulum.constants import SECONDS_PER_MINUTE
from pendulum.duration import Duration
from pendulum.helpers import days_in_year
from pendulum.helpers import is_leap
from pendulum.helpers import is_long_year
from pendulum.helpers import week_day
from pendulum.parsing.exceptions import ParserError
from pendulum.tz.timezone import UTC
from pendulum.tz.timezone import FixedTimezone
from pendulum.tz.timezone import Timezone


ISO8601_DT = re.compile(
    # Date (optional)  # noqa: ERA001
    "^"
    "(?P<date>"
    "    (?P<classic>"  # Classic date (YYYY-MM-DD) or ordinal (YYYY-DDD)
    r"        (?P<year>\d{4})"  # Year
    "        (?P<monthday>"
    r"            (?P<monthsep>-)?(?P<month>\d{2})"  # Month (optional)
    r"            ((?P<daysep>-)?(?P<day>\d{1,2}))?"  # Day (optional)
    "        )?"
    "    )"
    "    |"
    "    (?P<isocalendar>"  # Calendar date (2016-W05 or 2016-W05-5)
    r"        (?P<isoyear>\d{4})"  # Year
    "        (?P<weeksep>-)?"  # Separator (optional)
    "        W"  # W separator
    r"        (?P<isoweek>\d{2})"  # Week number
    "        (?P<weekdaysep>-)?"  # Separator (optional)
    r"        (?P<isoweekday>\d)?"  # Weekday (optional)
    "    )"
    ")?"
    # Time (optional)  # noqa: ERA001
    "(?P<time>"
    r"    (?P<timesep>[T\ ])?"  # Separator (T or space)
    # HH:mm:ss (optional mm and ss)
    r"    (?P<hour>\d{1,2})(?P<minsep>:)?(?P<minute>\d{1,2})?(?P<secsep>:)?(?P<second>\d{1,2})?"
    # Subsecond part (optional)
    "    (?P<subsecondsection>"
    "        (?:[.,])"  # Subsecond separator (optional)
    r"        (?P<subsecond>\d{1,9})"  # Subsecond
    "    )?"
    # Timezone offset
    "    (?P<tz>"
    r"        (?:[-+])\d{2}:?(?:\d{2})?|Z"  # Offset (+HH:mm or +HHmm or +HH or Z)
    "    )?"
    ")?"
    "$",
    re.VERBOSE,
)

ISO8601_DURATION = re.compile(
    "^P"  # Duration P indicator
    # Years, months and days (optional)  # noqa: ERA001
    "(?P<w>"
    r"    (?P<weeks>\d+(?:[.,]\d+)?W)"
    ")?"
    "(?P<ymd>"
    r"    (?P<years>\d+(?:[.,]\d+)?Y)?"
    r"    (?P<months>\d+(?:[.,]\d+)?M)?"
    r"    (?P<days>\d+(?:[.,]\d+)?D)?"
    ")?"
    "(?P<hms>"
    "    (?P<timesep>T)"  # Separator (T)
    r"    (?P<hours>\d+(?:[.,]\d+)?H)?"
    r"    (?P<minutes>\d+(?:[.,]\d+)?M)?"
    r"    (?P<seconds>\d+(?:[.,]\d+)?S)?"
    ")?"
    "$",
    re.VERBOSE,
)


def parse_iso8601(
    text: str,
) -> datetime.datetime | datetime.date | datetime.time | Duration:
    """
    ISO 8601 compliant parser.

    :param text: The string to parse
    :type text: str

    :rtype: datetime.datetime or datetime.time or datetime.date
    """
    parsed = _parse_iso8601_duration(text)
    if parsed is not None:
        return parsed

    m = ISO8601_DT.fullmatch(text)
    if not m:
        raise ParserError("Invalid ISO 8601 string")

    ambiguous_date = False
    is_date = False
    is_time = False
    year = 0
    month = 1
    day = 1
    minute = 0
    second = 0
    microsecond = 0
    tzinfo: FixedTimezone | Timezone | None = None

    if m.group("date"):
        # A date has been specified
        is_date = True

        if m.group("isocalendar"):
            # We have a ISO 8601 string defined
            # by week number
            if (
                m.group("weeksep")
                and not m.group("weekdaysep")
                and m.group("isoweekday")
            ):
                raise ParserError(f"Invalid date string: {text}")

            if not m.group("weeksep") and m.group("weekdaysep"):
                raise ParserError(f"Invalid date string: {text}")

            try:
                date = _get_iso_8601_week(
                    m.group("isoyear"), m.group("isoweek"), m.group("isoweekday")
                )
            except ParserError:
                raise
            except ValueError:
                raise ParserError(f"Invalid date string: {text}")

            year = date["year"]
            month = date["month"]
            day = date["day"]
        else:
            # We have a classic date representation
            year = int(m.group("year"))

            if not m.group("monthday"):
                # No month and day
                month = 1
                day = 1
            else:
                if m.group("month") and m.group("day"):
                    # Month and day
                    if not m.group("daysep") and len(m.group("day")) == 1:
                        # Ordinal day
                        ordinal = int(m.group("month") + m.group("day"))
                        leap = is_leap(year)
                        months_offsets = MONTHS_OFFSETS[leap]

                        if ordinal > months_offsets[13]:
                            raise ParserError("Ordinal day is out of range")

                        for i in range(1, 14):
                            if ordinal <= months_offsets[i]:
                                day = ordinal - months_offsets[i - 1]
                                month = i - 1

                                break
                    else:
                        month = int(m.group("month"))
                        day = int(m.group("day"))
                else:
                    # Only month
                    if not m.group("monthsep"):
                        # The date looks like 201207
                        # which is invalid for a date
                        # But it might be a time in the form hhmmss
                        ambiguous_date = True

                    month = int(m.group("month"))
                    day = 1

    if not m.group("time"):
        # No time has been specified
        if ambiguous_date:
            # We can "safely" assume that the ambiguous date
            # was actually a time in the form hhmmss
            hhmmss = f"{year!s}{month!s:0>2}"

            return datetime.time(int(hhmmss[:2]), int(hhmmss[2:4]), int(hhmmss[4:]))

        return datetime.date(year, month, day)

    if ambiguous_date:
        raise ParserError(f"Invalid date string: {text}")

    if is_date and not m.group("timesep"):
        raise ParserError(f"Invalid date string: {text}")

    if not is_date:
        is_time = True

    # Grabbing hh:mm:ss
    hour = int(m.group("hour"))
    minsep = m.group("minsep")

    if m.group("minute"):
        minute = int(m.group("minute"))
    elif minsep:
        raise ParserError("Invalid ISO 8601 time part")

    secsep = m.group("secsep")
    if secsep and not minsep and m.group("minute"):
        # minute/second separator but no hour/minute separator
        raise ParserError("Invalid ISO 8601 time part")

    if m.group("second"):
        if not secsep and minsep:
            # No minute/second separator but hour/minute separator
            raise ParserError("Invalid ISO 8601 time part")

        second = int(m.group("second"))
    elif secsep:
        raise ParserError("Invalid ISO 8601 time part")

    # Grabbing subseconds, if any
    if m.group("subsecondsection"):
        # Limiting to 6 chars
        subsecond = m.group("subsecond")[:6]

        microsecond = int(f"{subsecond:0<6}")

    # Grabbing timezone, if any
    tz = m.group("tz")
    if tz:
        if tz == "Z":
            tzinfo = UTC
        else:
            negative = bool(tz.startswith("-"))
            tz = tz[1:]
            if ":" not in tz:
                if len(tz) == 2:
                    tz = f"{tz}00"

                off_hour = tz[0:2]
                off_minute = tz[2:4]
            else:
                off_hour, off_minute = tz.split(":")

            offset = ((int(off_hour) * 60) + int(off_minute)) * 60

            if negative:
                offset = -1 * offset

            tzinfo = FixedTimezone(offset)

    if is_time:
        return datetime.time(hour, minute, second, microsecond, tzinfo=tzinfo)

    return datetime.datetime(
        year, month, day, hour, minute, second, microsecond, tzinfo=tzinfo
    )


def _parse_iso8601_duration(text: str, **options: str) -> Duration | None:
    m = ISO8601_DURATION.fullmatch(text)
    if not m:
        return None

    years = 0
    months = 0
    weeks = 0
    days: int | float = 0
    hours: int | float = 0
    minutes: int | float = 0
    seconds: int | float = 0
    microseconds: int | float = 0
    fractional = False

    _days: str | float
    _hours: str | int | None
    _minutes: str | int | None
    _seconds: str | int | None
    if m.group("w"):
        # Weeks
        if m.group("ymd") or m.group("hms"):
            # Specifying anything more than weeks is not supported
            raise ParserError("Invalid duration string")

        _weeks = m.group("weeks")
        if not _weeks:
            raise ParserError("Invalid duration string")

        _weeks = _weeks.replace(",", ".").replace("W", "")
        if "." in _weeks:
            _weeks, portion = _weeks.split(".")
            weeks = int(_weeks)
            _days = int(portion) / 10 * 7
            days, hours = int(_days // 1), int(_days % 1 * HOURS_PER_DAY)
        else:
            weeks = int(_weeks)

    if m.group("ymd"):
        # Years, months and/or days
        _years = m.group("years")
        _months = m.group("months")
        _days = m.group("days")

        # Checking order
        years_start = m.start("years") if _years else -3
        months_start = m.start("months") if _months else years_start + 1
        days_start = m.start("days") if _days else months_start + 1

        # Check correct order
        if not (years_start < months_start < days_start):
            raise ParserError("Invalid duration")

        if _years:
            _years = _years.replace(",", ".").replace("Y", "")
            if "." in _years:
                raise ParserError("Float years in duration are not supported")
            else:
                years = int(_years)

        if _months:
            if fractional:
                raise ParserError("Invalid duration")

            _months = _months.replace(",", ".").replace("M", "")
            if "." in _months:
                raise ParserError("Float months in duration are not supported")
            else:
                months = int(_months)

        if _days:
            if fractional:
                raise ParserError("Invalid duration")

            _days = _days.replace(",", ".").replace("D", "")

            if "." in _days:
                fractional = True

                _days, _hours = _days.split(".")
                days = int(_days)
                hours = int(_hours) / 10 * HOURS_PER_DAY
            else:
                days = int(_days)

    if m.group("hms"):
        # Hours, minutes and/or seconds
        _hours = m.group("hours") or 0
        _minutes = m.group("minutes") or 0
        _seconds = m.group("seconds") or 0

        # Checking order
        hours_start = m.start("hours") if _hours else -3
        minutes_start = m.start("minutes") if _minutes else hours_start + 1
        seconds_start = m.start("seconds") if _seconds else minutes_start + 1

        # Check correct order
        if not (hours_start < minutes_start < seconds_start):
            raise ParserError("Invalid duration")

        if _hours:
            if fractional:
                raise ParserError("Invalid duration")

            _hours = cast("str", _hours).replace(",", ".").replace("H", "")

            if "." in _hours:
                fractional = True

                _hours, _mins = _hours.split(".")
                hours += int(_hours)
                minutes += int(_mins) / 10 * MINUTES_PER_HOUR
            else:
                hours += int(_hours)

        if _minutes:
            if fractional:
                raise ParserError("Invalid duration")

            _minutes = cast("str", _minutes).replace(",", ".").replace("M", "")

            if "." in _minutes:
                fractional = True

                _minutes, _secs = _minutes.split(".")
                minutes += int(_minutes)
                seconds += int(_secs) / 10 * SECONDS_PER_MINUTE
            else:
                minutes += int(_minutes)

        if _seconds:
            if fractional:
                raise ParserError("Invalid duration")

            _seconds = cast("str", _seconds).replace(",", ".").replace("S", "")

            if "." in _seconds:
                _seconds, _microseconds = _seconds.split(".")
                seconds += int(_seconds)
                microseconds += int(f"{_microseconds[:6]:0<6}")
            else:
                seconds += int(_seconds)

    return Duration(
        years=years,
        months=months,
        weeks=weeks,
        days=days,
        hours=hours,
        minutes=minutes,
        seconds=seconds,
        microseconds=microseconds,
    )


def _get_iso_8601_week(
    year: int | str, week: int | str, weekday: int | str
) -> dict[str, int]:
    weekday = 1 if not weekday else int(weekday)

    year = int(year)
    week = int(week)

    if week > 53 or (week > 52 and not is_long_year(year)):
        raise ParserError("Invalid week for week date")

    if weekday > 7:
        raise ParserError("Invalid weekday for week date")

    # We can't rely on strptime directly here since
    # it does not support ISO week date
    ordinal = week * 7 + weekday - (week_day(year, 1, 4) + 3)

    if ordinal < 1:
        # Previous year
        ordinal += days_in_year(year - 1)
        year -= 1

    if ordinal > days_in_year(year):
        # Next year
        ordinal -= days_in_year(year)
        year += 1

    fmt = "%Y-%j"
    string = f"{year}-{ordinal}"

    dt = datetime.datetime.strptime(string, fmt)

    return {"year": dt.year, "month": dt.month, "day": dt.day}


# === src/pendulum/parsing/exceptions/__init__.py ===
from __future__ import annotations


class ParserError(ValueError):
    pass


# === src/pendulum/tz/__init__.py ===
from __future__ import annotations

from functools import cache
from zoneinfo import available_timezones

from pendulum.tz.local_timezone import get_local_timezone
from pendulum.tz.local_timezone import set_local_timezone
from pendulum.tz.local_timezone import test_local_timezone
from pendulum.tz.timezone import UTC
from pendulum.tz.timezone import FixedTimezone
from pendulum.tz.timezone import Timezone


PRE_TRANSITION = "pre"
POST_TRANSITION = "post"
TRANSITION_ERROR = "error"

_tz_cache: dict[int, FixedTimezone] = {}


@cache
def timezones() -> set[str]:
    return available_timezones()


def fixed_timezone(offset: int) -> FixedTimezone:
    """
    Return a Timezone instance given its offset in seconds.
    """
    if offset in _tz_cache:
        return _tz_cache[offset]

    tz = FixedTimezone(offset)
    _tz_cache[offset] = tz

    return tz


def local_timezone() -> Timezone | FixedTimezone:
    """
    Return the local timezone.
    """
    return get_local_timezone()


__all__ = [
    "UTC",
    "FixedTimezone",
    "Timezone",
    "fixed_timezone",
    "get_local_timezone",
    "local_timezone",
    "set_local_timezone",
    "test_local_timezone",
    "timezones",
]


# === src/pendulum/tz/exceptions.py ===
from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from datetime import datetime


class TimezoneError(ValueError):
    pass


class InvalidTimezone(TimezoneError):
    pass


class NonExistingTime(TimezoneError):
    message = "The datetime {} does not exist."

    def __init__(self, dt: datetime) -> None:
        message = self.message.format(dt)

        super().__init__(message)


class AmbiguousTime(TimezoneError):
    message = "The datetime {} is ambiguous."

    def __init__(self, dt: datetime) -> None:
        message = self.message.format(dt)

        super().__init__(message)


# === src/pendulum/tz/timezone.py ===
# mypy: no-warn-redundant-casts
from __future__ import annotations

import datetime as _datetime
import zoneinfo

from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import cast

from pendulum.tz.exceptions import AmbiguousTime
from pendulum.tz.exceptions import InvalidTimezone
from pendulum.tz.exceptions import NonExistingTime


if TYPE_CHECKING:
    from typing_extensions import Self

POST_TRANSITION = "post"
PRE_TRANSITION = "pre"
TRANSITION_ERROR = "error"


_DT = TypeVar("_DT", bound=_datetime.datetime)


class PendulumTimezone(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def convert(self, dt: _DT, raise_on_unknown_times: bool = False) -> _DT:
        raise NotImplementedError

    @abstractmethod
    def datetime(
        self,
        year: int,
        month: int,
        day: int,
        hour: int = 0,
        minute: int = 0,
        second: int = 0,
        microsecond: int = 0,
    ) -> _datetime.datetime:
        raise NotImplementedError


class Timezone(zoneinfo.ZoneInfo, PendulumTimezone):
    """
    Represents a named timezone.

    The accepted names are those provided by the IANA time zone database.

    >>> from pendulum.tz.timezone import Timezone
    >>> tz = Timezone('Europe/Paris')
    """

    def __new__(cls, key: str) -> Self:
        try:
            return super().__new__(cls, key)  # type: ignore[call-arg]
        except zoneinfo.ZoneInfoNotFoundError:
            raise InvalidTimezone(key)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Timezone) and self.key == other.key

    @property
    def name(self) -> str:
        return self.key

    def convert(self, dt: _DT, raise_on_unknown_times: bool = False) -> _DT:
        """
        Converts a datetime in the current timezone.

        If the datetime is naive, it will be normalized.

        >>> from datetime import datetime
        >>> from pendulum import timezone
        >>> paris = timezone('Europe/Paris')
        >>> dt = datetime(2013, 3, 31, 2, 30, fold=1)
        >>> in_paris = paris.convert(dt)
        >>> in_paris.isoformat()
        '2013-03-31T03:30:00+02:00'

        If the datetime is aware, it will be properly converted.

        >>> new_york = timezone('America/New_York')
        >>> in_new_york = new_york.convert(in_paris)
        >>> in_new_york.isoformat()
        '2013-03-30T21:30:00-04:00'
        """

        if dt.tzinfo is None:
            # Technically, utcoffset() can return None, but none of the zone information
            # in tzdata sets _tti_before to None. This can be checked with the following
            # code:
            #
            # >>> import zoneinfo
            # >>> from zoneinfo._zoneinfo import ZoneInfo
            #
            # >>> for tzname in zoneinfo.available_timezones():
            # >>>     if ZoneInfo(tzname)._tti_before is None:
            # >>>         print(tzname)

            offset_before = cast(
                "_datetime.timedelta",
                (self.utcoffset(dt.replace(fold=0)) if dt.fold else self.utcoffset(dt)),
            )
            offset_after = cast(
                "_datetime.timedelta",
                (self.utcoffset(dt) if dt.fold else self.utcoffset(dt.replace(fold=1))),
            )

            if offset_after > offset_before:
                # Skipped time
                if raise_on_unknown_times:
                    raise NonExistingTime(dt)

                dt = cast(
                    "_DT",
                    dt
                    + (
                        (offset_after - offset_before)
                        if dt.fold
                        else (offset_before - offset_after)
                    ),
                )
            elif offset_before > offset_after and raise_on_unknown_times:
                # Repeated time
                raise AmbiguousTime(dt)

            return dt.replace(tzinfo=self)

        return cast("_DT", dt.astimezone(self))

    def datetime(
        self,
        year: int,
        month: int,
        day: int,
        hour: int = 0,
        minute: int = 0,
        second: int = 0,
        microsecond: int = 0,
    ) -> _datetime.datetime:
        """
        Return a normalized datetime for the current timezone.
        """
        return self.convert(
            _datetime.datetime(
                year, month, day, hour, minute, second, microsecond, fold=1
            )
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self.name}')"


class FixedTimezone(_datetime.tzinfo, PendulumTimezone):
    def __init__(self, offset: int, name: str | None = None) -> None:
        sign = "-" if offset < 0 else "+"

        minutes = offset / 60
        hour, minute = divmod(abs(int(minutes)), 60)

        if not name:
            name = f"{sign}{hour:02d}:{minute:02d}"

        self._name = name
        self._offset = offset
        self._utcoffset = _datetime.timedelta(seconds=offset)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FixedTimezone) and self._offset == other._offset

    @property
    def name(self) -> str:
        return self._name

    def convert(self, dt: _DT, raise_on_unknown_times: bool = False) -> _DT:
        if dt.tzinfo is None:
            return dt.__class__(
                dt.year,
                dt.month,
                dt.day,
                dt.hour,
                dt.minute,
                dt.second,
                dt.microsecond,
                tzinfo=self,
                fold=0,
            )

        return cast("_DT", dt.astimezone(self))

    def datetime(
        self,
        year: int,
        month: int,
        day: int,
        hour: int = 0,
        minute: int = 0,
        second: int = 0,
        microsecond: int = 0,
    ) -> _datetime.datetime:
        return self.convert(
            _datetime.datetime(
                year, month, day, hour, minute, second, microsecond, fold=1
            )
        )

    @property
    def offset(self) -> int:
        return self._offset

    def utcoffset(self, dt: _datetime.datetime | None) -> _datetime.timedelta:
        return self._utcoffset

    def dst(self, dt: _datetime.datetime | None) -> _datetime.timedelta:
        return _datetime.timedelta()

    def fromutc(self, dt: _datetime.datetime) -> _datetime.datetime:
        # Use the stdlib datetime's add method to avoid infinite recursion
        return (_datetime.datetime.__add__(dt, self._utcoffset)).replace(tzinfo=self)

    def tzname(self, dt: _datetime.datetime | None) -> str | None:
        return self._name

    def __getinitargs__(self) -> tuple[int, str]:
        return self._offset, self._name

    def __repr__(self) -> str:
        name = ""
        if self._name:
            name = f', name="{self._name}"'

        return f"{self.__class__.__name__}({self._offset}{name})"


UTC = Timezone("UTC")


# === src/pendulum/tz/local_timezone.py ===
from __future__ import annotations

import contextlib
import os
import re
import sys
import warnings

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from pendulum.tz.exceptions import InvalidTimezone
from pendulum.tz.timezone import UTC
from pendulum.tz.timezone import FixedTimezone
from pendulum.tz.timezone import Timezone


if TYPE_CHECKING:
    from collections.abc import Iterator


if sys.platform == "win32":
    import winreg

_mock_local_timezone = None
_local_timezone = None


def get_local_timezone() -> Timezone | FixedTimezone:
    global _local_timezone

    if _mock_local_timezone is not None:
        return _mock_local_timezone

    if _local_timezone is None:
        tz = _get_system_timezone()

        _local_timezone = tz

    return _local_timezone


def set_local_timezone(mock: str | Timezone | None = None) -> None:
    global _mock_local_timezone

    _mock_local_timezone = mock


@contextmanager
def test_local_timezone(mock: Timezone) -> Iterator[None]:
    set_local_timezone(mock)

    yield

    set_local_timezone()


def _get_system_timezone() -> Timezone:
    if sys.platform == "win32":
        return _get_windows_timezone()
    elif "darwin" in sys.platform:
        return _get_darwin_timezone()

    return _get_unix_timezone()


if sys.platform == "win32":

    def _get_windows_timezone() -> Timezone:
        from pendulum.tz.data.windows import windows_timezones

        # Windows is special. It has unique time zone names (in several
        # meanings of the word) available, but unfortunately, they can be
        # translated to the language of the operating system, so we need to
        # do a backwards lookup, by going through all time zones and see which
        # one matches.
        handle = winreg.ConnectRegistry(None, winreg.HKEY_LOCAL_MACHINE)

        tz_local_key_name = r"SYSTEM\CurrentControlSet\Control\TimeZoneInformation"
        localtz = winreg.OpenKey(handle, tz_local_key_name)

        timezone_info = {}
        size = winreg.QueryInfoKey(localtz)[1]
        for i in range(size):
            data = winreg.EnumValue(localtz, i)
            timezone_info[data[0]] = data[1]

        localtz.Close()

        if "TimeZoneKeyName" in timezone_info:
            # Windows 7 (and Vista?)

            # For some reason this returns a string with loads of NUL bytes at
            # least on some systems. I don't know if this is a bug somewhere, I
            # just work around it.
            tzkeyname = timezone_info["TimeZoneKeyName"].split("\x00", 1)[0]
        else:
            # Windows 2000 or XP

            # This is the localized name:
            tzwin = timezone_info["StandardName"]

            # Open the list of timezones to look up the real name:
            tz_key_name = r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Time Zones"
            tzkey = winreg.OpenKey(handle, tz_key_name)

            # Now, match this value to Time Zone information
            tzkeyname = None
            for i in range(winreg.QueryInfoKey(tzkey)[0]):
                subkey = winreg.EnumKey(tzkey, i)
                sub = winreg.OpenKey(tzkey, subkey)

                info = {}
                size = winreg.QueryInfoKey(sub)[1]
                for i in range(size):
                    data = winreg.EnumValue(sub, i)
                    info[data[0]] = data[1]

                sub.Close()
                with contextlib.suppress(KeyError):
                    # This timezone didn't have proper configuration.
                    # Ignore it.
                    if info["Std"] == tzwin:
                        tzkeyname = subkey
                        break

            tzkey.Close()
            handle.Close()

        if tzkeyname is None:
            raise LookupError("Can not find Windows timezone configuration")

        timezone = windows_timezones.get(tzkeyname)
        if timezone is None:
            # Nope, that didn't work. Try adding "Standard Time",
            # it seems to work a lot of times:
            timezone = windows_timezones.get(tzkeyname + " Standard Time")

        # Return what we have.
        if timezone is None:
            raise LookupError("Unable to find timezone " + tzkeyname)

        return Timezone(timezone)

else:

    def _get_windows_timezone() -> Timezone:
        raise NotImplementedError


def _get_darwin_timezone() -> Timezone:
    # link will be something like /usr/share/zoneinfo/America/Los_Angeles.
    link = os.readlink("/etc/localtime")
    tzname = link[link.rfind("zoneinfo/") + 9 :]

    return Timezone(tzname)


def _get_unix_timezone(_root: str = "/") -> Timezone:
    tzenv = os.environ.get("TZ")
    if tzenv:
        with contextlib.suppress(ValueError):
            return _tz_from_env(tzenv)

    # Now look for distribution specific configuration files
    # that contain the timezone name.
    tzpath = Path(_root) / "etc" / "timezone"
    if tzpath.is_file():
        tzfile_data = tzpath.read_bytes()
        # Issue #3 was that /etc/timezone was a zoneinfo file.
        # That's a misconfiguration, but we need to handle it gracefully:
        if not tzfile_data.startswith(b"TZif2"):
            etctz = tzfile_data.strip().decode()
            # Get rid of host definitions and comments:
            etctz, _, _ = etctz.partition(" ")
            etctz, _, _ = etctz.partition("#")
            return Timezone(etctz.replace(" ", "_"))

    # CentOS has a ZONE setting in /etc/sysconfig/clock,
    # OpenSUSE has a TIMEZONE setting in /etc/sysconfig/clock and
    # Gentoo has a TIMEZONE setting in /etc/conf.d/clock
    # We look through these files for a timezone:
    zone_re = re.compile(r'\s*(TIME)?ZONE\s*=\s*"([^"]+)?"')

    for filename in ("etc/sysconfig/clock", "etc/conf.d/clock"):
        tzpath = Path(_root) / filename
        if tzpath.is_file():
            data = tzpath.read_text().splitlines()
            for line in data:
                # Look for the ZONE= or TIMEZONE= setting.
                match = zone_re.match(line)
                if match:
                    etctz = match.group(2)
                    parts = list(reversed(etctz.replace(" ", "_").split(os.path.sep)))
                    tzpath_parts: list[str] = []
                    while parts:
                        tzpath_parts.insert(0, parts.pop(0))
                        with contextlib.suppress(InvalidTimezone):
                            return Timezone(os.path.sep.join(tzpath_parts))

    # systemd distributions use symlinks that include the zone name,
    # see manpage of localtime(5) and timedatectl(1)
    tzpath = Path(_root) / "etc" / "localtime"
    if tzpath.is_file() and tzpath.is_symlink():
        parts = [p.replace(" ", "_") for p in reversed(tzpath.resolve().parts)]
        tzpath_parts: list[str] = []  # type: ignore[no-redef]
        while parts:
            tzpath_parts.insert(0, parts.pop(0))
            with contextlib.suppress(InvalidTimezone):
                return Timezone(os.path.sep.join(tzpath_parts))

    # No explicit setting existed. Use localtime
    for filename in ("etc/localtime", "usr/local/etc/localtime"):
        tzpath = Path(_root) / filename
        if tzpath.is_file():
            with tzpath.open("rb") as f:
                return Timezone.from_file(f)

    warnings.warn(
        "Unable not find any timezone configuration, defaulting to UTC.", stacklevel=1
    )

    return UTC


def _tz_from_env(tzenv: str) -> Timezone:
    if tzenv[0] == ":":
        tzenv = tzenv[1:]

    # TZ specifies a file
    if os.path.isfile(tzenv):
        with open(tzenv, "rb") as f:
            return Timezone.from_file(f)

    # TZ specifies a zoneinfo zone.
    try:
        return Timezone(tzenv)
    except ValueError:
        raise


# === src/pendulum/tz/data/__init__.py ===


# === src/pendulum/tz/data/windows.py ===
from __future__ import annotations


windows_timezones = {
    "AUS Central Standard Time": "Australia/Darwin",
    "AUS Eastern Standard Time": "Australia/Sydney",
    "Afghanistan Standard Time": "Asia/Kabul",
    "Alaskan Standard Time": "America/Anchorage",
    "Aleutian Standard Time": "America/Adak",
    "Altai Standard Time": "Asia/Barnaul",
    "Arab Standard Time": "Asia/Riyadh",
    "Arabian Standard Time": "Asia/Dubai",
    "Arabic Standard Time": "Asia/Baghdad",
    "Argentina Standard Time": "America/Buenos_Aires",
    "Astrakhan Standard Time": "Europe/Astrakhan",
    "Atlantic Standard Time": "America/Halifax",
    "Aus Central W. Standard Time": "Australia/Eucla",
    "Azerbaijan Standard Time": "Asia/Baku",
    "Azores Standard Time": "Atlantic/Azores",
    "Bahia Standard Time": "America/Bahia",
    "Bangladesh Standard Time": "Asia/Dhaka",
    "Belarus Standard Time": "Europe/Minsk",
    "Bougainville Standard Time": "Pacific/Bougainville",
    "Canada Central Standard Time": "America/Regina",
    "Cape Verde Standard Time": "Atlantic/Cape_Verde",
    "Caucasus Standard Time": "Asia/Yerevan",
    "Cen. Australia Standard Time": "Australia/Adelaide",
    "Central America Standard Time": "America/Guatemala",
    "Central Asia Standard Time": "Asia/Almaty",
    "Central Brazilian Standard Time": "America/Cuiaba",
    "Central Europe Standard Time": "Europe/Budapest",
    "Central European Standard Time": "Europe/Warsaw",
    "Central Pacific Standard Time": "Pacific/Guadalcanal",
    "Central Standard Time": "America/Chicago",
    "Central Standard Time (Mexico)": "America/Mexico_City",
    "Chatham Islands Standard Time": "Pacific/Chatham",
    "China Standard Time": "Asia/Shanghai",
    "Cuba Standard Time": "America/Havana",
    "Dateline Standard Time": "Etc/GMT+12",
    "E. Africa Standard Time": "Africa/Nairobi",
    "E. Australia Standard Time": "Australia/Brisbane",
    "E. Europe Standard Time": "Europe/Chisinau",
    "E. South America Standard Time": "America/Sao_Paulo",
    "Easter Island Standard Time": "Pacific/Easter",
    "Eastern Standard Time": "America/New_York",
    "Eastern Standard Time (Mexico)": "America/Cancun",
    "Egypt Standard Time": "Africa/Cairo",
    "Ekaterinburg Standard Time": "Asia/Yekaterinburg",
    "FLE Standard Time": "Europe/Kyiv",
    "Fiji Standard Time": "Pacific/Fiji",
    "GMT Standard Time": "Europe/London",
    "GTB Standard Time": "Europe/Bucharest",
    "Georgian Standard Time": "Asia/Tbilisi",
    "Greenland Standard Time": "America/Godthab",
    "Greenwich Standard Time": "Atlantic/Reykjavik",
    "Haiti Standard Time": "America/Port-au-Prince",
    "Hawaiian Standard Time": "Pacific/Honolulu",
    "India Standard Time": "Asia/Calcutta",
    "Iran Standard Time": "Asia/Tehran",
    "Israel Standard Time": "Asia/Jerusalem",
    "Jordan Standard Time": "Asia/Amman",
    "Kaliningrad Standard Time": "Europe/Kaliningrad",
    "Korea Standard Time": "Asia/Seoul",
    "Libya Standard Time": "Africa/Tripoli",
    "Line Islands Standard Time": "Pacific/Kiritimati",
    "Lord Howe Standard Time": "Australia/Lord_Howe",
    "Magadan Standard Time": "Asia/Magadan",
    "Magallanes Standard Time": "America/Punta_Arenas",
    "Marquesas Standard Time": "Pacific/Marquesas",
    "Mauritius Standard Time": "Indian/Mauritius",
    "Middle East Standard Time": "Asia/Beirut",
    "Montevideo Standard Time": "America/Montevideo",
    "Morocco Standard Time": "Africa/Casablanca",
    "Mountain Standard Time": "America/Denver",
    "Mountain Standard Time (Mexico)": "America/Chihuahua",
    "Myanmar Standard Time": "Asia/Rangoon",
    "N. Central Asia Standard Time": "Asia/Novosibirsk",
    "Namibia Standard Time": "Africa/Windhoek",
    "Nepal Standard Time": "Asia/Katmandu",
    "New Zealand Standard Time": "Pacific/Auckland",
    "Newfoundland Standard Time": "America/St_Johns",
    "Norfolk Standard Time": "Pacific/Norfolk",
    "North Asia East Standard Time": "Asia/Irkutsk",
    "North Asia Standard Time": "Asia/Krasnoyarsk",
    "North Korea Standard Time": "Asia/Pyongyang",
    "Omsk Standard Time": "Asia/Omsk",
    "Pacific SA Standard Time": "America/Santiago",
    "Pacific Standard Time": "America/Los_Angeles",
    "Pacific Standard Time (Mexico)": "America/Tijuana",
    "Pakistan Standard Time": "Asia/Karachi",
    "Paraguay Standard Time": "America/Asuncion",
    "Romance Standard Time": "Europe/Paris",
    "Russia Time Zone 10": "Asia/Srednekolymsk",
    "Russia Time Zone 11": "Asia/Kamchatka",
    "Russia Time Zone 3": "Europe/Samara",
    "Russian Standard Time": "Europe/Moscow",
    "SA Eastern Standard Time": "America/Cayenne",
    "SA Pacific Standard Time": "America/Bogota",
    "SA Western Standard Time": "America/La_Paz",
    "SE Asia Standard Time": "Asia/Bangkok",
    "Saint Pierre Standard Time": "America/Miquelon",
    "Sakhalin Standard Time": "Asia/Sakhalin",
    "Samoa Standard Time": "Pacific/Apia",
    "Sao Tome Standard Time": "Africa/Sao_Tome",
    "Saratov Standard Time": "Europe/Saratov",
    "Singapore Standard Time": "Asia/Singapore",
    "South Africa Standard Time": "Africa/Johannesburg",
    "Sri Lanka Standard Time": "Asia/Colombo",
    "Sudan Standard Time": "Africa/Khartoum",
    "Syria Standard Time": "Asia/Damascus",
    "Taipei Standard Time": "Asia/Taipei",
    "Tasmania Standard Time": "Australia/Hobart",
    "Tocantins Standard Time": "America/Araguaina",
    "Tokyo Standard Time": "Asia/Tokyo",
    "Tomsk Standard Time": "Asia/Tomsk",
    "Tonga Standard Time": "Pacific/Tongatapu",
    "Transbaikal Standard Time": "Asia/Chita",
    "Turkey Standard Time": "Europe/Istanbul",
    "Turks And Caicos Standard Time": "America/Grand_Turk",
    "US Eastern Standard Time": "America/Indianapolis",
    "US Mountain Standard Time": "America/Phoenix",
    "UTC": "Etc/GMT",
    "UTC+12": "Etc/GMT-12",
    "UTC+13": "Etc/GMT-13",
    "UTC-02": "Etc/GMT+2",
    "UTC-08": "Etc/GMT+8",
    "UTC-09": "Etc/GMT+9",
    "UTC-11": "Etc/GMT+11",
    "Ulaanbaatar Standard Time": "Asia/Ulaanbaatar",
    "Venezuela Standard Time": "America/Caracas",
    "Vladivostok Standard Time": "Asia/Vladivostok",
    "W. Australia Standard Time": "Australia/Perth",
    "W. Central Africa Standard Time": "Africa/Lagos",
    "W. Europe Standard Time": "Europe/Berlin",
    "W. Mongolia Standard Time": "Asia/Hovd",
    "West Asia Standard Time": "Asia/Tashkent",
    "West Bank Standard Time": "Asia/Hebron",
    "West Pacific Standard Time": "Pacific/Port_Moresby",
    "Yakutsk Standard Time": "Asia/Yakutsk",
}
