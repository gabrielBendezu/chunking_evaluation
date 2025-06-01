# === tests/test_deserialization.py ===
# mypy: disable-error-code="arg-type"
import datetime as dt
import decimal
import ipaddress
import math
import uuid
from unittest.mock import patch

import pytest

from marshmallow import (
    EXCLUDE,
    INCLUDE,
    RAISE,
    Schema,
    fields,
    validate,
)
from marshmallow.exceptions import ValidationError
from tests.base import (
    ALL_FIELDS,
    DateEnum,
    GenderEnum,
    HairColorEnum,
    assert_date_equal,
    assert_time_equal,
    central,
    predicate,
)


class MockDateTimeOverflowError(dt.datetime):
    """Used to simulate the possible OverflowError of datetime.fromtimestamp"""

    def fromtimestamp(self, *args, **kwargs):  # type: ignore[override]
        raise OverflowError


class MockDateTimeOSError(dt.datetime):
    """Used to simulate the possible OSError of datetime.fromtimestamp"""

    def fromtimestamp(self, *args, **kwargs):  # type: ignore[override]
        raise OSError


class TestDeserializingNone:
    @pytest.mark.parametrize("FieldClass", ALL_FIELDS)
    def test_fields_allow_none_deserialize_to_none(self, FieldClass):
        field = FieldClass(allow_none=True)
        assert field.deserialize(None) is None

    # https://github.com/marshmallow-code/marshmallow/issues/111
    @pytest.mark.parametrize("FieldClass", ALL_FIELDS)
    def test_fields_dont_allow_none_by_default(self, FieldClass):
        field = FieldClass()
        with pytest.raises(ValidationError, match="Field may not be null."):
            field.deserialize(None)

    def test_allow_none_is_true_if_missing_is_true(self):
        field = fields.Raw(load_default=None)
        assert field.allow_none is True
        assert field.deserialize(None) is None

    def test_list_field_deserialize_none_to_none(self):
        field = fields.List(fields.String(allow_none=True), allow_none=True)
        assert field.deserialize(None) is None

    def test_tuple_field_deserialize_none_to_none(self):
        field = fields.Tuple([fields.String()], allow_none=True)
        assert field.deserialize(None) is None

    def test_list_of_nested_allow_none_deserialize_none_to_none(self):
        field = fields.List(fields.Nested(Schema(), allow_none=True))
        assert field.deserialize([None]) == [None]

    def test_list_of_nested_non_allow_none_deserialize_none_to_validation_error(self):
        field = fields.List(fields.Nested(Schema(), allow_none=False))
        with pytest.raises(ValidationError):
            field.deserialize([None])


class TestFieldDeserialization:
    def test_float_field_deserialization(self):
        field = fields.Float()
        assert math.isclose(field.deserialize("12.3"), 12.3)
        assert math.isclose(field.deserialize(12.3), 12.3)

    @pytest.mark.parametrize("in_val", ["bad", "", {}, True, False])
    def test_invalid_float_field_deserialization(self, in_val):
        field = fields.Float()
        with pytest.raises(ValidationError) as excinfo:
            field.deserialize(in_val)
        assert excinfo.value.args[0] == "Not a valid number."

    def test_float_field_overflow(self):
        field = fields.Float()
        with pytest.raises(ValidationError) as excinfo:
            field.deserialize(2**1024)
        assert excinfo.value.args[0] == "Number too large."

    def test_integer_field_deserialization(self):
        field = fields.Integer()
        assert field.deserialize("42") == 42
        with pytest.raises(ValidationError) as excinfo:
            field.deserialize("42.0")
        assert excinfo.value.args[0] == "Not a valid integer."
        with pytest.raises(ValidationError):
            field.deserialize("bad")
        assert excinfo.value.args[0] == "Not a valid integer."
        with pytest.raises(ValidationError):
            field.deserialize({})
        assert excinfo.value.args[0] == "Not a valid integer."

    def test_strict_integer_field_deserialization(self):
        field = fields.Integer(strict=True)
        assert field.deserialize(42) == 42
        with pytest.raises(ValidationError) as excinfo:
            field.deserialize(42.0)
        assert excinfo.value.args[0] == "Not a valid integer."
        with pytest.raises(ValidationError) as excinfo:
            field.deserialize(decimal.Decimal("42.0"))
        assert excinfo.value.args[0] == "Not a valid integer."
        with pytest.raises(ValidationError) as excinfo:
            field.deserialize("42")
        assert excinfo.value.args[0] == "Not a valid integer."

    def test_decimal_field_deserialization(self):
        m1 = 12
        m2 = "12.355"
        m3 = decimal.Decimal(1)
        m4 = 3.14
        m5 = "abc"
        m6 = [1, 2]

        field = fields.Decimal()
        assert isinstance(field.deserialize(m1), decimal.Decimal)
        assert field.deserialize(m1) == decimal.Decimal(12)
        assert isinstance(field.deserialize(m2), decimal.Decimal)
        assert field.deserialize(m2) == decimal.Decimal("12.355")
        assert isinstance(field.deserialize(m3), decimal.Decimal)
        assert field.deserialize(m3) == decimal.Decimal(1)
        assert isinstance(field.deserialize(m4), decimal.Decimal)
        assert field.deserialize(m4).as_tuple() == (0, (3, 1, 4), -2)
        with pytest.raises(ValidationError) as excinfo:
            field.deserialize(m5)
        assert excinfo.value.args[0] == "Not a valid number."
        with pytest.raises(ValidationError) as excinfo:
            field.deserialize(m6)
        assert excinfo.value.args[0] == "Not a valid number."

    def test_decimal_field_with_places(self):
        m1 = 12
        m2 = "12.355"
        m3 = decimal.Decimal(1)
        m4 = "abc"
        m5 = [1, 2]

        field = fields.Decimal(1)
        assert isinstance(field.deserialize(m1), decimal.Decimal)
        assert field.deserialize(m1) == decimal.Decimal(12)
        assert isinstance(field.deserialize(m2), decimal.Decimal)
        assert field.deserialize(m2) == decimal.Decimal("12.4")
        assert isinstance(field.deserialize(m3), decimal.Decimal)
        assert field.deserialize(m3) == decimal.Decimal(1)
        with pytest.raises(ValidationError) as excinfo:
            field.deserialize(m4)
        assert excinfo.value.args[0] == "Not a valid number."
        with pytest.raises(ValidationError) as excinfo:
            field.deserialize(m5)
        assert excinfo.value.args[0] == "Not a valid number."

    def test_decimal_field_with_places_and_rounding(self):
        m1 = 12
        m2 = "12.355"
        m3 = decimal.Decimal(1)
        m4 = "abc"
        m5 = [1, 2]

        field = fields.Decimal(1, decimal.ROUND_DOWN)
        assert isinstance(field.deserialize(m1), decimal.Decimal)
        assert field.deserialize(m1) == decimal.Decimal(12)
        assert isinstance(field.deserialize(m2), decimal.Decimal)
        assert field.deserialize(m2) == decimal.Decimal("12.3")
        assert isinstance(field.deserialize(m3), decimal.Decimal)
        assert field.deserialize(m3) == decimal.Decimal(1)
        with pytest.raises(ValidationError):
            field.deserialize(m4)
        with pytest.raises(ValidationError):
            field.deserialize(m5)

    def test_decimal_field_deserialization_string(self):
        m1 = 12
        m2 = "12.355"
        m3 = decimal.Decimal(1)
        m4 = "abc"
        m5 = [1, 2]

        field = fields.Decimal(as_string=True)
        assert isinstance(field.deserialize(m1), decimal.Decimal)
        assert field.deserialize(m1) == decimal.Decimal(12)
        assert isinstance(field.deserialize(m2), decimal.Decimal)
        assert field.deserialize(m2) == decimal.Decimal("12.355")
        assert isinstance(field.deserialize(m3), decimal.Decimal)
        assert field.deserialize(m3) == decimal.Decimal(1)
        with pytest.raises(ValidationError):
            field.deserialize(m4)
        with pytest.raises(ValidationError):
            field.deserialize(m5)

    def test_decimal_field_special_values(self):
        m1 = "-NaN"
        m2 = "NaN"
        m3 = "-sNaN"
        m4 = "sNaN"
        m5 = "-Infinity"
        m6 = "Infinity"
        m7 = "-0"

        field = fields.Decimal(places=2, allow_nan=True)

        m1d = field.deserialize(m1)
        assert isinstance(m1d, decimal.Decimal)
        assert m1d.is_qnan()
        assert not m1d.is_signed()

        m2d = field.deserialize(m2)
        assert isinstance(m2d, decimal.Decimal)
        assert m2d.is_qnan()
        assert not m2d.is_signed()

        m3d = field.deserialize(m3)
        assert isinstance(m3d, decimal.Decimal)
        assert m3d.is_qnan()
        assert not m3d.is_signed()

        m4d = field.deserialize(m4)
        assert isinstance(m4d, decimal.Decimal)
        assert m4d.is_qnan()
        assert not m4d.is_signed()

        m5d = field.deserialize(m5)
        assert isinstance(m5d, decimal.Decimal)
        assert m5d.is_infinite()
        assert m5d.is_signed()

        m6d = field.deserialize(m6)
        assert isinstance(m6d, decimal.Decimal)
        assert m6d.is_infinite()
        assert not m6d.is_signed()

        m7d = field.deserialize(m7)
        assert isinstance(m7d, decimal.Decimal)
        assert m7d.is_zero()
        assert m7d.is_signed()

    def test_decimal_field_special_values_not_permitted(self):
        m1 = "-NaN"
        m2 = "NaN"
        m3 = "-sNaN"
        m4 = "sNaN"
        m5 = "-Infinity"
        m6 = "Infinity"
        m7 = "-0"

        field = fields.Decimal(places=2)

        with pytest.raises(ValidationError) as excinfo:
            field.deserialize(m1)
        assert str(excinfo.value.args[0]) == (
            "Special numeric values (nan or infinity) are not permitted."
        )
        with pytest.raises(ValidationError):
            field.deserialize(m2)
        with pytest.raises(ValidationError):
            field.deserialize(m3)
        with pytest.raises(ValidationError):
            field.deserialize(m4)
        with pytest.raises(ValidationError):
            field.deserialize(m5)
        with pytest.raises(ValidationError):
            field.deserialize(m6)

        m7d = field.deserialize(m7)
        assert isinstance(m7d, decimal.Decimal)
        assert m7d.is_zero()
        assert m7d.is_signed()

    @pytest.mark.parametrize("allow_nan", (None, False, True))
    @pytest.mark.parametrize("value", ("nan", "-nan", "inf", "-inf"))
    def test_float_field_allow_nan(self, value, allow_nan):
        if allow_nan is None:
            # Test default case is False
            field = fields.Float()
        else:
            field = fields.Float(allow_nan=allow_nan)

        if allow_nan is True:
            res = field.deserialize(value)
            assert isinstance(res, float)
            if value.endswith("nan"):
                assert math.isnan(res)
            else:
                assert res == float(value)
        else:
            with pytest.raises(ValidationError) as excinfo:
                field.deserialize(value)
            assert str(excinfo.value.args[0]) == (
                "Special numeric values (nan or infinity) are not permitted."
            )

    def test_string_field_deserialization(self):
        field = fields.String()
        assert field.deserialize("foo") == "foo"
        assert field.deserialize(b"foo") == "foo"

        # https://github.com/marshmallow-code/marshmallow/issues/231
        with pytest.raises(ValidationError) as excinfo:
            field.deserialize(42)
        assert excinfo.value.args[0] == "Not a valid string."

        with pytest.raises(ValidationError):
            field.deserialize({})

    def test_boolean_field_deserialization(self):
        field = fields.Boolean()
        assert field.deserialize(True) is True
        assert field.deserialize(False) is False
        assert field.deserialize("True") is True
        assert field.deserialize("False") is False
        assert field.deserialize("true") is True
        assert field.deserialize("false") is False
        assert field.deserialize("1") is True
        assert field.deserialize("0") is False
        assert field.deserialize("on") is True
        assert field.deserialize("ON") is True
        assert field.deserialize("On") is True
        assert field.deserialize("off") is False
        assert field.deserialize("OFF") is False
        assert field.deserialize("Off") is False
        assert field.deserialize("y") is True
        assert field.deserialize("Y") is True
        assert field.deserialize("yes") is True
        assert field.deserialize("YES") is True
        assert field.deserialize("Yes") is True
        assert field.deserialize("n") is False
        assert field.deserialize("N") is False
        assert field.deserialize("no") is False
        assert field.deserialize("NO") is False
        assert field.deserialize("No") is False
        assert field.deserialize(1) is True
        assert field.deserialize(0) is False

        with pytest.raises(ValidationError) as excinfo:
            field.deserialize({})
        assert excinfo.value.args[0] == "Not a valid boolean."

        with pytest.raises(ValidationError) as excinfo:
            field.deserialize(42)

        with pytest.raises(ValidationError) as excinfo:
            field.deserialize("invalid-string")

    def test_boolean_field_deserialization_with_custom_truthy_values(self):
        class MyBoolean(fields.Boolean):
            truthy = {"yep"}

        field = MyBoolean()
        assert field.deserialize("yep") is True

        field2 = fields.Boolean(truthy=("yep",))
        assert field2.deserialize("yep") is True
        assert field2.deserialize(False) is False

    @pytest.mark.parametrize("in_val", ["notvalid", 123])
    def test_boolean_field_deserialization_with_custom_truthy_values_invalid(
        self, in_val
    ):
        class MyBoolean(fields.Boolean):
            truthy = {"yep"}

        field = MyBoolean()
        with pytest.raises(ValidationError) as excinfo:
            field.deserialize(in_val)
        expected_msg = "Not a valid boolean."
        assert str(excinfo.value.args[0]) == expected_msg

        field2 = fields.Boolean(truthy={"yep"})
        with pytest.raises(ValidationError) as excinfo:
            field2.deserialize(in_val)
        expected_msg = "Not a valid boolean."
        assert str(excinfo.value.args[0]) == expected_msg

        field3 = MyBoolean(error_messages={"invalid": "bad input"})
        with pytest.raises(ValidationError) as excinfo:
            field3.deserialize(in_val)
        assert str(excinfo.value.args[0]) == "bad input"

    def test_boolean_field_deserialization_with_empty_truthy(self):
        field = fields.Boolean(truthy=set())
        assert field.deserialize("yep") is True
        assert field.deserialize(True) is True
        assert field.deserialize(False) is False

    def test_boolean_field_deserialization_with_custom_falsy_values(self):
        field = fields.Boolean(falsy=("nope",))
        assert field.deserialize("nope") is False
        assert field.deserialize(True) is True

    def test_field_toggle_show_invalid_value_in_error_message(self):
        error_messages = {"invalid": "Not valid: {input}"}
        boolfield = fields.Boolean(error_messages=error_messages)
        with pytest.raises(ValidationError) as excinfo:
            boolfield.deserialize("notabool")
        assert str(excinfo.value.args[0]) == "Not valid: notabool"

        numfield = fields.Float(error_messages=error_messages)
        with pytest.raises(ValidationError) as excinfo:
            numfield.deserialize("notanum")
        assert str(excinfo.value.args[0]) == "Not valid: notanum"

        intfield = fields.Integer(error_messages=error_messages)
        with pytest.raises(ValidationError) as excinfo:
            intfield.deserialize("notanint")
        assert str(excinfo.value.args[0]) == "Not valid: notanint"

        date_error_messages = {"invalid": "Not a valid {obj_type}: {input}"}
        datefield = fields.DateTime(error_messages=date_error_messages)
        with pytest.raises(ValidationError) as excinfo:
            datefield.deserialize("notadate")
        assert str(excinfo.value.args[0]) == "Not a valid datetime: notadate"

    @pytest.mark.parametrize(
        "in_value",
        [
            "not-a-datetime",
            42,
            True,
            False,
            0,
            "",
            [],
            "2018",
            "2018-01",
            dt.datetime.now().strftime("%H:%M:%S %Y-%m-%d"),
            dt.datetime.now().strftime("%m-%d-%Y %H:%M:%S"),
        ],
    )
    def test_invalid_datetime_deserialization(self, in_value):
        field = fields.DateTime()
        with pytest.raises(ValidationError, match="Not a valid datetime."):
            field.deserialize(in_value)

    def test_custom_date_format_datetime_field_deserialization(self):
        # Datetime string with format "%H:%M:%S.%f %Y-%m-%d"
        datestring = "10:11:12.123456 2019-01-02"

        # Deserialization should fail when datestring is not of same format
        field = fields.DateTime(format="%d-%m-%Y %H:%M:%S")
        with pytest.raises(ValidationError, match="Not a valid datetime."):
            field.deserialize(datestring)

        field = fields.DateTime(format="%H:%M:%S.%f %Y-%m-%d")
        assert field.deserialize(datestring) == dt.datetime(
            2019, 1, 2, 10, 11, 12, 123456
        )

        field = fields.NaiveDateTime(format="%H:%M:%S.%f %Y-%m-%d")
        assert field.deserialize(datestring) == dt.datetime(
            2019, 1, 2, 10, 11, 12, 123456
        )

        field = fields.AwareDateTime(format="%H:%M:%S.%f %Y-%m-%d")
        with pytest.raises(ValidationError, match="Not a valid aware datetime."):
            field.deserialize(datestring)

    @pytest.mark.parametrize("fmt", ["rfc", "rfc822"])
    @pytest.mark.parametrize(
        ("value", "expected", "aware"),
        [
            (
                "Sun, 10 Nov 2013 01:23:45 -0000",
                dt.datetime(2013, 11, 10, 1, 23, 45),
                False,
            ),
            (
                "Sun, 10 Nov 2013 01:23:45 +0000",
                dt.datetime(2013, 11, 10, 1, 23, 45, tzinfo=dt.timezone.utc),
                True,
            ),
            (
                "Sun, 10 Nov 2013 01:23:45 -0600",
                dt.datetime(2013, 11, 10, 1, 23, 45, tzinfo=central),
                True,
            ),
        ],
    )
    def test_rfc_datetime_field_deserialization(self, fmt, value, expected, aware):
        field = fields.DateTime(format=fmt)
        assert field.deserialize(value) == expected
        field = fields.NaiveDateTime(format=fmt)
        if aware:
            with pytest.raises(ValidationError, match="Not a valid naive datetime."):
                field.deserialize(value)
        else:
            assert field.deserialize(value) == expected
        field = fields.AwareDateTime(format=fmt)
        if not aware:
            with pytest.raises(ValidationError, match="Not a valid aware datetime."):
                field.deserialize(value)
        else:
            assert field.deserialize(value) == expected

    @pytest.mark.parametrize("fmt", ["iso", "iso8601"])
    @pytest.mark.parametrize(
        ("value", "expected", "aware"),
        [
            ("2013-11-10T01:23:45", dt.datetime(2013, 11, 10, 1, 23, 45), False),
            (
                "2013-11-10T01:23:45+00:00",
                dt.datetime(2013, 11, 10, 1, 23, 45, tzinfo=dt.timezone.utc),
                True,
            ),
            (
                # Regression test for https://github.com/marshmallow-code/marshmallow/issues/1251
                "2013-11-10T01:23:45.123+00:00",
                dt.datetime(2013, 11, 10, 1, 23, 45, 123000, tzinfo=dt.timezone.utc),
                True,
            ),
            (
                "2013-11-10T01:23:45.123456+00:00",
                dt.datetime(2013, 11, 10, 1, 23, 45, 123456, tzinfo=dt.timezone.utc),
                True,
            ),
            (
                "2013-11-10T01:23:45-06:00",
                dt.datetime(2013, 11, 10, 1, 23, 45, tzinfo=central),
                True,
            ),
        ],
    )
    def test_iso_datetime_field_deserialization(self, fmt, value, expected, aware):
        field = fields.DateTime(format=fmt)
        assert field.deserialize(value) == expected
        field = fields.NaiveDateTime(format=fmt)
        if aware:
            with pytest.raises(ValidationError, match="Not a valid naive datetime."):
                field.deserialize(value)
        else:
            assert field.deserialize(value) == expected
        field = fields.AwareDateTime(format=fmt)
        if not aware:
            with pytest.raises(ValidationError, match="Not a valid aware datetime."):
                field.deserialize(value)
        else:
            assert field.deserialize(value) == expected

    @pytest.mark.parametrize(
        ("fmt", "value", "expected"),
        [
            ("timestamp", 1384043025, dt.datetime(2013, 11, 10, 0, 23, 45)),
            ("timestamp", "1384043025", dt.datetime(2013, 11, 10, 0, 23, 45)),
            ("timestamp", 1384043025.12, dt.datetime(2013, 11, 10, 0, 23, 45, 120000)),
            (
                "timestamp",
                1384043025.123456,
                dt.datetime(2013, 11, 10, 0, 23, 45, 123456),
            ),
            ("timestamp", 1, dt.datetime(1970, 1, 1, 0, 0, 1)),
            ("timestamp_ms", 1384043025000, dt.datetime(2013, 11, 10, 0, 23, 45)),
            ("timestamp_ms", 1000, dt.datetime(1970, 1, 1, 0, 0, 1)),
        ],
    )
    def test_timestamp_field_deserialization(self, fmt, value, expected):
        field = fields.DateTime(format=fmt)
        assert field.deserialize(value) == expected

        # By default, a datetime from a timestamp is never aware.
        field = fields.NaiveDateTime(format=fmt)
        assert field.deserialize(value) == expected

        field = fields.AwareDateTime(format=fmt)
        with pytest.raises(ValidationError, match="Not a valid aware datetime."):
            field.deserialize(value)

        # But it can be added by providing a default.
        field = fields.AwareDateTime(format=fmt, default_timezone=central)
        expected_aware = expected.replace(tzinfo=central)
        assert field.deserialize(value) == expected_aware

    @pytest.mark.parametrize("fmt", ["timestamp", "timestamp_ms"])
    @pytest.mark.parametrize(
        "in_value",
        ["", "!@#", -1],
    )
    def test_invalid_timestamp_field_deserialization(self, fmt, in_value):
        field = fields.DateTime(format=fmt)
        with pytest.raises(ValidationError, match="Not a valid datetime."):
            field.deserialize(in_value)

    # Regression test for https://github.com/marshmallow-code/marshmallow/pull/2102
    @pytest.mark.parametrize("fmt", ["timestamp", "timestamp_ms"])
    @pytest.mark.parametrize(
        "mock_fromtimestamp", [MockDateTimeOSError, MockDateTimeOverflowError]
    )
    def test_oversized_timestamp_field_deserialization(self, fmt, mock_fromtimestamp):
        with patch("datetime.datetime", mock_fromtimestamp):
            field = fields.DateTime(format=fmt)
            with pytest.raises(ValidationError, match="Not a valid datetime."):
                field.deserialize(99999999999999999)

    @pytest.mark.parametrize(
        ("fmt", "timezone", "value", "expected"),
        [
            ("iso", None, "2013-11-10T01:23:45", dt.datetime(2013, 11, 10, 1, 23, 45)),
            (
                "iso",
                dt.timezone.utc,
                "2013-11-10T01:23:45+00:00",
                dt.datetime(2013, 11, 10, 1, 23, 45),
            ),
            (
                "iso",
                central,
                "2013-11-10T01:23:45-03:00",
                dt.datetime(2013, 11, 9, 22, 23, 45),
            ),
            (
                "rfc",
                None,
                "Sun, 10 Nov 2013 01:23:45 -0000",
                dt.datetime(2013, 11, 10, 1, 23, 45),
            ),
            (
                "rfc",
                dt.timezone.utc,
                "Sun, 10 Nov 2013 01:23:45 +0000",
                dt.datetime(2013, 11, 10, 1, 23, 45),
            ),
            (
                "rfc",
                central,
                "Sun, 10 Nov 2013 01:23:45 -0300",
                dt.datetime(2013, 11, 9, 22, 23, 45),
            ),
        ],
    )
    def test_naive_datetime_with_timezone(self, fmt, timezone, value, expected):
        field = fields.NaiveDateTime(format=fmt, timezone=timezone)
        assert field.deserialize(value) == expected

    @pytest.mark.parametrize("timezone", (dt.timezone.utc, central))
    @pytest.mark.parametrize(
        ("fmt", "value"),
        [("iso", "2013-11-10T01:23:45"), ("rfc", "Sun, 10 Nov 2013 01:23:45")],
    )
    def test_aware_datetime_default_timezone(self, fmt, timezone, value):
        field = fields.AwareDateTime(format=fmt, default_timezone=timezone)
        assert field.deserialize(value) == dt.datetime(
            2013, 11, 10, 1, 23, 45, tzinfo=timezone
        )

    def test_time_field_deserialization(self):
        field = fields.Time()
        t = dt.time(1, 23, 45)
        t_formatted = t.isoformat()
        result = field.deserialize(t_formatted)
        assert isinstance(result, dt.time)
        assert_time_equal(result, t)
        # With microseconds
        t2 = dt.time(1, 23, 45, 6789)
        t2_formatted = t2.isoformat()
        result2 = field.deserialize(t2_formatted)
        assert_time_equal(result2, t2)

    @pytest.mark.parametrize("in_data", ["badvalue", "", [], 42])
    def test_invalid_time_field_deserialization(self, in_data):
        field = fields.Time()
        with pytest.raises(ValidationError) as excinfo:
            field.deserialize(in_data)
        assert excinfo.value.args[0] == "Not a valid time."

    def test_custom_time_format_time_field_deserialization(self):
        # Time string with format "%f.%S:%M:%H"
        timestring = "123456.12:11:10"

        # Deserialization should fail when timestring is not of same format
        field = fields.Time(format="%S:%M:%H")
        with pytest.raises(ValidationError, match="Not a valid time."):
            field.deserialize(timestring)

        field = fields.Time(format="%f.%S:%M:%H")
        assert field.deserialize(timestring) == dt.time(10, 11, 12, 123456)

    @pytest.mark.parametrize("fmt", ["iso", "iso8601", None])
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("01:23:45", dt.time(1, 23, 45)),
            ("01:23:45.123", dt.time(1, 23, 45, 123000)),
            ("01:23:45.123456", dt.time(1, 23, 45, 123456)),
            (
                "01:23:45+01:00",
                dt.time(1, 23, 45, tzinfo=dt.timezone(dt.timedelta(seconds=3600))),
            ),
        ],
    )
    def test_iso_time_field_deserialization(self, fmt, value, expected):
        if fmt is None:
            field = fields.Time()
        else:
            field = fields.Time(format=fmt)
        assert field.deserialize(value) == expected

    def test_invalid_timedelta_precision(self):
        with pytest.raises(ValueError, match="The precision must be one of: weeks,"):
            fields.TimeDelta("invalid")

    def test_timedelta_field_deserialization(self):
        field = fields.TimeDelta()
        result = field.deserialize("42")
        assert isinstance(result, dt.timedelta)
        assert result.days == 0
        assert result.seconds == 42
        assert result.microseconds == 0

        field = fields.TimeDelta()
        result = field.deserialize("42.9")
        assert isinstance(result, dt.timedelta)
        assert result.days == 0
        assert result.seconds == 42
        assert result.microseconds == 900000

        field = fields.TimeDelta(fields.TimeDelta.SECONDS)
        result = field.deserialize(100000)
        assert result.days == 1
        assert result.seconds == 13600
        assert result.microseconds == 0

        field = fields.TimeDelta(fields.TimeDelta.DAYS)
        result = field.deserialize("-42")
        assert isinstance(result, dt.timedelta)
        assert result.days == -42
        assert result.seconds == 0
        assert result.microseconds == 0

        field = fields.TimeDelta(fields.TimeDelta.MICROSECONDS)
        result = field.deserialize(10**6 + 1)
        assert isinstance(result, dt.timedelta)
        assert result.days == 0
        assert result.seconds == 1
        assert result.microseconds == 1

        field = fields.TimeDelta(fields.TimeDelta.MICROSECONDS)
        result = field.deserialize(86400 * 10**6 + 1)
        assert isinstance(result, dt.timedelta)
        assert result.days == 1
        assert result.seconds == 0
        assert result.microseconds == 1

        field = fields.TimeDelta()
        result = field.deserialize(12.9)
        assert isinstance(result, dt.timedelta)
        assert result.days == 0
        assert result.seconds == 12
        assert result.microseconds == 900000

        field = fields.TimeDelta(fields.TimeDelta.WEEKS)
        result = field.deserialize(1)
        assert isinstance(result, dt.timedelta)
        assert result.days == 7
        assert result.seconds == 0
        assert result.microseconds == 0

        field = fields.TimeDelta(fields.TimeDelta.HOURS)
        result = field.deserialize(25)
        assert isinstance(result, dt.timedelta)
        assert result.days == 1
        assert result.seconds == 3600
        assert result.microseconds == 0

        field = fields.TimeDelta(fields.TimeDelta.MINUTES)
        result = field.deserialize(1441)
        assert isinstance(result, dt.timedelta)
        assert result.days == 1
        assert result.seconds == 60
        assert result.microseconds == 0

        field = fields.TimeDelta(fields.TimeDelta.MILLISECONDS)
        result = field.deserialize(123456)
        assert isinstance(result, dt.timedelta)
        assert result.days == 0
        assert result.seconds == 123
        assert result.microseconds == 456000

        total_microseconds_value = 322.0
        field = fields.TimeDelta(fields.TimeDelta.MICROSECONDS)
        result = field.deserialize(total_microseconds_value)
        assert isinstance(result, dt.timedelta)
        unit_value = dt.timedelta(microseconds=1).total_seconds()
        assert math.isclose(
            result.total_seconds() / unit_value, total_microseconds_value
        )

        total_microseconds_value = 322.12345
        field = fields.TimeDelta(fields.TimeDelta.MICROSECONDS)
        result = field.deserialize(total_microseconds_value)
        assert isinstance(result, dt.timedelta)
        unit_value = dt.timedelta(microseconds=1).total_seconds()
        assert math.isclose(
            result.total_seconds() / unit_value, math.floor(total_microseconds_value)
        )

        total_milliseconds_value = 322.223
        field = fields.TimeDelta(fields.TimeDelta.MILLISECONDS)
        result = field.deserialize(total_milliseconds_value)
        assert isinstance(result, dt.timedelta)
        unit_value = dt.timedelta(milliseconds=1).total_seconds()
        assert math.isclose(
            result.total_seconds() / unit_value, total_milliseconds_value
        )

        total_seconds_value = 322.223
        field = fields.TimeDelta(fields.TimeDelta.SECONDS)
        result = field.deserialize(total_seconds_value)
        assert isinstance(result, dt.timedelta)
        assert math.isclose(result.total_seconds(), total_seconds_value)

        total_minutes_value = 322.223
        field = fields.TimeDelta(fields.TimeDelta.MINUTES)
        result = field.deserialize(total_minutes_value)
        assert isinstance(result, dt.timedelta)
        unit_value = dt.timedelta(minutes=1).total_seconds()
        assert math.isclose(result.total_seconds() / unit_value, total_minutes_value)

        total_hours_value = 322.223
        field = fields.TimeDelta(fields.TimeDelta.HOURS)
        result = field.deserialize(total_hours_value)
        assert isinstance(result, dt.timedelta)
        unit_value = dt.timedelta(hours=1).total_seconds()
        assert math.isclose(result.total_seconds() / unit_value, total_hours_value)

        total_days_value = 322.223
        field = fields.TimeDelta(fields.TimeDelta.DAYS)
        result = field.deserialize(total_days_value)
        assert isinstance(result, dt.timedelta)
        unit_value = dt.timedelta(days=1).total_seconds()
        assert math.isclose(result.total_seconds() / unit_value, total_days_value)

        total_weeks_value = 322.223
        field = fields.TimeDelta(fields.TimeDelta.WEEKS)
        result = field.deserialize(total_weeks_value)
        assert isinstance(result, dt.timedelta)
        unit_value = dt.timedelta(weeks=1).total_seconds()
        assert math.isclose(result.total_seconds() / unit_value, total_weeks_value)

    @pytest.mark.parametrize("in_value", ["", "badvalue", [], 9999999999])
    def test_invalid_timedelta_field_deserialization(self, in_value):
        field = fields.TimeDelta(fields.TimeDelta.DAYS)
        with pytest.raises(ValidationError) as excinfo:
            field.deserialize(in_value)
        assert excinfo.value.args[0] == "Not a valid period of time."

    @pytest.mark.parametrize("format", (None, "%Y-%m-%d"))
    def test_date_field_deserialization(self, format):  # noqa: A002
        field = fields.Date(format=format)
        d = dt.date(2014, 8, 21)
        iso_date = d.isoformat()
        result = field.deserialize(iso_date)
        assert type(result) is dt.date
        assert_date_equal(result, d)

    @pytest.mark.parametrize(
        "in_value", ["", 123, [], dt.date(2014, 8, 21).strftime("%d-%m-%Y")]
    )
    def test_invalid_date_field_deserialization(self, in_value):
        field = fields.Date()
        with pytest.raises(ValidationError) as excinfo:
            field.deserialize(in_value)
        msg = "Not a valid date."
        assert excinfo.value.args[0] == msg

    def test_dict_field_deserialization(self):
        data = {"foo": "bar"}
        field = fields.Dict()
        load = field.deserialize(data)
        assert load == {"foo": "bar"}
        # Check load is a distinct object
        load["foo"] = "baz"
        assert data["foo"] == "bar"
        with pytest.raises(ValidationError) as excinfo:
            field.deserialize("baddict")
        assert excinfo.value.args[0] == "Not a valid mapping type."

    def test_structured_dict_value_deserialization(self):
        field = fields.Dict(values=fields.List(fields.Str))
        assert field.deserialize({"foo": ["bar", "baz"]}) == {"foo": ["bar", "baz"]}
        with pytest.raises(ValidationError) as excinfo:
            field.deserialize({"foo": [1, 2], "bar": "baz", "ham": ["spam"]})
        assert excinfo.value.args[0] == {
            "foo": {"value": {0: ["Not a valid string."], 1: ["Not a valid string."]}},
            "bar": {"value": ["Not a valid list."]},
        }
        assert excinfo.value.valid_data == {"foo": [], "ham": ["spam"]}

    def test_structured_dict_key_deserialization(self):
        field = fields.Dict(keys=fields.Str)
        assert field.deserialize({"foo": "bar"}) == {"foo": "bar"}
        with pytest.raises(ValidationError) as excinfo:
            field.deserialize({1: "bar", "foo": "baz"})
        assert excinfo.value.args[0] == {1: {"key": ["Not a valid string."]}}
        assert excinfo.value.valid_data == {"foo": "baz"}

    def test_structured_dict_key_value_deserialization(self):
        field = fields.Dict(
            keys=fields.Str(
                validate=[validate.Email(), validate.Regexp(r".*@test\.com$")]
            ),
            values=fields.Decimal,
        )
        assert field.deserialize({"foo@test.com": 1}) == {
            "foo@test.com": decimal.Decimal(1)
        }
        with pytest.raises(ValidationError) as excinfo:
            field.deserialize({1: "bar"})
        assert excinfo.value.args[0] == {
            1: {"key": ["Not a valid string."], "value": ["Not a valid number."]}
        }
        with pytest.raises(ValidationError) as excinfo:
            field.deserialize({"foo@test.com": "bar"})
        assert excinfo.value.args[0] == {
            "foo@test.com": {"value": ["Not a valid number."]}
        }
        assert excinfo.value.valid_data == {}
        with pytest.raises(ValidationError) as excinfo:
            field.deserialize({1: 1})
        assert excinfo.value.args[0] == {1: {"key": ["Not a valid string."]}}
        assert excinfo.value.valid_data == {}
        with pytest.raises(ValidationError) as excinfo:
            field.deserialize({"foo": "bar"})
        assert excinfo.value.args[0] == {
            "foo": {
                "key": [
                    "Not a valid email address.",
                    "String does not match expected pattern.",
                ],
                "value": ["Not a valid number."],
            }
        }
        assert excinfo.value.valid_data == {}

    def test_url_field_deserialization(self):
        field = fields.Url()
        assert field.deserialize("https://duckduckgo.com") == "https://duckduckgo.com"
        with pytest.raises(ValidationError) as excinfo:
            field.deserialize("badurl")
        assert excinfo.value.args[0][0] == "Not a valid URL."
        # Relative URLS not allowed by default
        with pytest.raises(ValidationError) as excinfo:
            field.deserialize("/foo/bar")
        assert excinfo.value.args[0][0] == "Not a valid URL."

    # regression test for https://github.com/marshmallow-code/marshmallow/issues/1400
    def test_url_field_non_list_validators(self):
        field = fields.Url(validate=(validate.Length(min=16),))
        with pytest.raises(ValidationError, match="Shorter than minimum length"):
            field.deserialize("https://abc.def")

    def test_relative_url_field_deserialization(self):
        field = fields.Url(relative=True)
        assert field.deserialize("/foo/bar") == "/foo/bar"

    def test_url_field_schemes_argument(self):
        field = fields.URL()
        url = "ws://test.test"
        with pytest.raises(ValidationError):
            field.deserialize(url)
        field2 = fields.URL(schemes={"http", "https", "ws"})
        assert field2.deserialize(url) == url

    def test_email_field_deserialization(self):
        field = fields.Email()
        assert field.deserialize("foo@bar.com") == "foo@bar.com"
        with pytest.raises(ValidationError) as excinfo:
            field.deserialize("invalidemail")
        assert excinfo.value.args[0][0] == "Not a valid email address."

        field = fields.Email(validate=[validate.Length(min=12)])
        with pytest.raises(ValidationError) as excinfo:
            field.deserialize("foo@bar.com")
        assert excinfo.value.args[0][0] == "Shorter than minimum length 12."

    # regression test for https://github.com/marshmallow-code/marshmallow/issues/1400
    def test_email_field_non_list_validators(self):
        field = fields.Email(validate=(validate.Length(min=9),))
        with pytest.raises(ValidationError, match="Shorter than minimum length"):
            field.deserialize("a@bc.com")

    def test_function_field_deserialization_is_noop_by_default(self):
        field = fields.Function(lambda x: None)
        # Default is noop
        assert field.deserialize("foo") == "foo"
        assert field.deserialize(42) == 42

    def test_function_field_deserialization_with_callable(self):
        field = fields.Function(lambda x: None, deserialize=lambda val: val.upper())
        assert field.deserialize("foo") == "FOO"

    def test_function_field_passed_deserialize_only_is_load_only(self):
        field = fields.Function(deserialize=lambda val: val.upper())
        assert field.load_only is True

    def test_function_field_passed_deserialize_and_serialize_is_not_load_only(self):
        field = fields.Function(
            serialize=lambda val: val.lower(), deserialize=lambda val: val.upper()
        )
        assert field.load_only is False

    def test_uuid_field_deserialization(self):
        field = fields.UUID()
        uuid_str = str(uuid.uuid4())
        result = field.deserialize(uuid_str)
        assert isinstance(result, uuid.UUID)
        assert str(result) == uuid_str

        uuid4 = uuid.uuid4()
        result = field.deserialize(uuid4)
        assert isinstance(result, uuid.UUID)
        assert result == uuid4

        uuid_bytes = b"]\xc7wW\x132O\xf9\xa5\xbe\x13\x1f\x02\x18\xda\xbf"
        result = field.deserialize(uuid_bytes)
        assert isinstance(result, uuid.UUID)
        assert result.bytes == uuid_bytes

    @pytest.mark.parametrize("in_value", ["malformed", 123, [], b"tooshort"])
    def test_invalid_uuid_deserialization(self, in_value):
        field = fields.UUID()
        with pytest.raises(ValidationError) as excinfo:
            field.deserialize(in_value)

        assert excinfo.value.args[0] == "Not a valid UUID."

    def test_ip_field_deserialization(self):
        field = fields.IP()
        ipv4_str = "140.82.118.3"
        result = field.deserialize(ipv4_str)
        assert isinstance(result, ipaddress.IPv4Address)
        assert str(result) == ipv4_str

        ipv6_str = "2a00:1450:4001:824::200e"
        result = field.deserialize(ipv6_str)
        assert isinstance(result, ipaddress.IPv6Address)
        assert str(result) == ipv6_str

    @pytest.mark.parametrize(
        "in_value",
        ["malformed", 123, b"\x01\x02\03", "192.168", "192.168.0.1/24", "ff::aa:1::2"],
    )
    def test_invalid_ip_deserialization(self, in_value):
        field = fields.IP()
        with pytest.raises(ValidationError) as excinfo:
            field.deserialize(in_value)

        assert excinfo.value.args[0] == "Not a valid IP address."

    def test_ipv4_field_deserialization(self):
        field = fields.IPv4()
        ipv4_str = "140.82.118.3"
        result = field.deserialize(ipv4_str)
        assert isinstance(result, ipaddress.IPv4Address)
        assert str(result) == ipv4_str

    @pytest.mark.parametrize(
        "in_value",
        [
            "malformed",
            123,
            b"\x01\x02\03",
            "192.168",
            "192.168.0.1/24",
            "2a00:1450:4001:81d::200e",
        ],
    )
    def test_invalid_ipv4_deserialization(self, in_value):
        field = fields.IPv4()
        with pytest.raises(ValidationError) as excinfo:
            field.deserialize(in_value)

        assert excinfo.value.args[0] == "Not a valid IPv4 address."

    def test_ipv6_field_deserialization(self):
        field = fields.IPv6()
        ipv6_str = "2a00:1450:4001:824::200e"
        result = field.deserialize(ipv6_str)
        assert isinstance(result, ipaddress.IPv6Address)
        assert str(result) == ipv6_str

    def test_ipinterface_field_deserialization(self):
        field = fields.IPInterface()
        ipv4interface_str = "140.82.118.3/24"
        result = field.deserialize(ipv4interface_str)
        assert isinstance(result, ipaddress.IPv4Interface)
        assert str(result) == ipv4interface_str

        ipv6interface_str = "2a00:1450:4001:824::200e/128"
        result = field.deserialize(ipv6interface_str)
        assert isinstance(result, ipaddress.IPv6Interface)
        assert str(result) == ipv6interface_str

    @pytest.mark.parametrize(
        "in_value",
        [
            "malformed",
            123,
            b"\x01\x02\03",
            "192.168",
            "192.168.0.1/33",
            "ff::aa:1::2",
            "2a00:1450:4001:824::200e/129",
        ],
    )
    def test_invalid_ipinterface_deserialization(self, in_value):
        field = fields.IPInterface()
        with pytest.raises(ValidationError) as excinfo:
            field.deserialize(in_value)

        assert excinfo.value.args[0] == "Not a valid IP interface."

    def test_ipv4interface_field_deserialization(self):
        field = fields.IPv4Interface()
        ipv4interface_str = "140.82.118.3/24"
        result = field.deserialize(ipv4interface_str)
        assert isinstance(result, ipaddress.IPv4Interface)
        assert str(result) == ipv4interface_str

    @pytest.mark.parametrize(
        "in_value",
        [
            "malformed",
            123,
            b"\x01\x02\03",
            "192.168",
            "192.168.0.1/33",
            "2a00:1450:4001:81d::200e",
            "2a00:1450:4001:824::200e/129",
        ],
    )
    def test_invalid_ipv4interface_deserialization(self, in_value):
        field = fields.IPv4Interface()
        with pytest.raises(ValidationError) as excinfo:
            field.deserialize(in_value)

        assert excinfo.value.args[0] == "Not a valid IPv4 interface."

    def test_ipv6interface_field_deserialization(self):
        field = fields.IPv6Interface()
        ipv6interface_str = "2a00:1450:4001:824::200e/128"
        result = field.deserialize(ipv6interface_str)
        assert isinstance(result, ipaddress.IPv6Interface)
        assert str(result) == ipv6interface_str

    @pytest.mark.parametrize(
        "in_value",
        [
            "malformed",
            123,
            b"\x01\x02\03",
            "ff::aa:1::2",
            "192.168.0.1",
            "192.168.0.1/24",
            "2a00:1450:4001:824::200e/129",
        ],
    )
    def test_invalid_ipv6interface_deserialization(self, in_value):
        field = fields.IPv6Interface()
        with pytest.raises(ValidationError) as excinfo:
            field.deserialize(in_value)

        assert excinfo.value.args[0] == "Not a valid IPv6 interface."

    def test_enum_field_by_symbol_deserialization(self):
        field = fields.Enum(GenderEnum)
        assert field.deserialize("male") == GenderEnum.male

    def test_enum_field_by_symbol_invalid_value(self):
        field = fields.Enum(GenderEnum)
        with pytest.raises(
            ValidationError, match="Must be one of: male, female, non_binary."
        ):
            field.deserialize("dummy")

    def test_enum_field_by_symbol_not_string(self):
        field = fields.Enum(GenderEnum)
        with pytest.raises(ValidationError, match="Not a valid string."):
            field.deserialize(12)

    def test_enum_field_by_value_true_deserialization(self):
        field = fields.Enum(HairColorEnum, by_value=True)
        assert field.deserialize("black hair") == HairColorEnum.black
        field2 = fields.Enum(GenderEnum, by_value=True)
        assert field2.deserialize(1) == GenderEnum.male

    def test_enum_field_by_value_field_deserialization(self):
        field = fields.Enum(HairColorEnum, by_value=fields.String)
        assert field.deserialize("black hair") == HairColorEnum.black
        field2 = fields.Enum(GenderEnum, by_value=fields.Integer)
        assert field2.deserialize(1) == GenderEnum.male
        field3 = fields.Enum(DateEnum, by_value=fields.Date(format="%d/%m/%Y"))
        assert field3.deserialize("29/02/2004") == DateEnum.date_1

    def test_enum_field_by_value_true_invalid_value(self):
        field = fields.Enum(HairColorEnum, by_value=True)
        with pytest.raises(
            ValidationError,
            match="Must be one of: black hair, brown hair, blond hair, red hair.",
        ):
            field.deserialize("dummy")
        field2 = fields.Enum(GenderEnum, by_value=True)
        with pytest.raises(ValidationError, match="Must be one of: 1, 2, 3."):
            field2.deserialize(12)

    def test_enum_field_by_value_field_invalid_value(self):
        field = fields.Enum(HairColorEnum, by_value=fields.String)
        with pytest.raises(
            ValidationError,
            match="Must be one of: black hair, brown hair, blond hair, red hair.",
        ):
            field.deserialize("dummy")
        field2 = fields.Enum(GenderEnum, by_value=fields.Integer)
        with pytest.raises(ValidationError, match="Must be one of: 1, 2, 3."):
            field2.deserialize(12)
        field3 = fields.Enum(DateEnum, by_value=fields.Date(format="%d/%m/%Y"))
        with pytest.raises(
            ValidationError, match="Must be one of: 29/02/2004, 29/02/2008, 29/02/2012."
        ):
            field3.deserialize("28/02/2004")

    def test_enum_field_by_value_true_wrong_type(self):
        field = fields.Enum(HairColorEnum, by_value=True)
        with pytest.raises(
            ValidationError,
            match="Must be one of: black hair, brown hair, blond hair, red hair.",
        ):
            field.deserialize("dummy")
        field = fields.Enum(GenderEnum, by_value=True)
        with pytest.raises(ValidationError, match="Must be one of: 1, 2, 3."):
            field.deserialize(12)

    def test_enum_field_by_value_field_wrong_type(self):
        field = fields.Enum(HairColorEnum, by_value=fields.String)
        with pytest.raises(ValidationError, match="Not a valid string."):
            field.deserialize(12)
        field = fields.Enum(GenderEnum, by_value=fields.Integer)
        with pytest.raises(ValidationError, match="Not a valid integer."):
            field.deserialize("dummy")
        field = fields.Enum(DateEnum, by_value=fields.Date(format="%d/%m/%Y"))
        with pytest.raises(ValidationError, match="Not a valid date."):
            field.deserialize("30/02/2004")

    def test_deserialization_function_must_be_callable(self):
        with pytest.raises(TypeError):
            fields.Function(lambda x: None, deserialize="notvalid")

    def test_method_field_deserialization_is_noop_by_default(self):
        class MiniUserSchema(Schema):
            uppername = fields.Method("uppercase_name")

            def uppercase_name(self, obj):
                return obj.upper()

        s = MiniUserSchema()
        assert s.fields["uppername"].deserialize("steve") == "steve"

    def test_deserialization_method(self):
        class MiniUserSchema(Schema):
            uppername = fields.Method("uppercase_name", deserialize="lowercase_name")

            def uppercase_name(self, obj):
                return obj.name.upper()

            def lowercase_name(self, value):
                return value.lower()

        s = MiniUserSchema()
        assert s.fields["uppername"].deserialize("STEVE") == "steve"

    def test_deserialization_method_must_be_a_method(self):
        class BadSchema(Schema):
            uppername = fields.Method("uppercase_name", deserialize="lowercase_name")

        with pytest.raises(AttributeError):
            BadSchema()

    def test_method_field_deserialize_only(self):
        class MethodDeserializeOnly(Schema):
            name = fields.Method(deserialize="lowercase_name")

            def lowercase_name(self, value):
                return value.lower()

        assert MethodDeserializeOnly().load({"name": "ALEC"})["name"] == "alec"

    def test_datetime_list_field_deserialization(self):
        dtimes = dt.datetime.now(), dt.datetime.now(), dt.datetime.now(dt.timezone.utc)
        dstrings = [each.isoformat() for each in dtimes]
        field = fields.List(fields.DateTime())
        result = field.deserialize(dstrings)
        assert all(isinstance(each, dt.datetime) for each in result)
        for actual, expected in zip(result, dtimes):
            assert_date_equal(actual, expected)

    def test_list_field_deserialize_invalid_item(self):
        field = fields.List(fields.DateTime)
        with pytest.raises(ValidationError) as excinfo:
            field.deserialize(["badvalue"])
        assert excinfo.value.args[0] == {0: ["Not a valid datetime."]}

        field = fields.List(fields.Str())
        with pytest.raises(ValidationError) as excinfo:
            field.deserialize(["good", 42])
        assert excinfo.value.args[0] == {1: ["Not a valid string."]}

    def test_list_field_deserialize_multiple_invalid_items(self):
        field = fields.List(
            fields.Int(
                validate=validate.Range(10, 20, error="Value {input} not in range")
            )
        )
        with pytest.raises(ValidationError) as excinfo:
            field.deserialize([10, 5, 25])
        assert len(excinfo.value.args[0]) == 2
        assert excinfo.value.args[0][1] == ["Value 5 not in range"]
        assert excinfo.value.args[0][2] == ["Value 25 not in range"]

    @pytest.mark.parametrize("value", ["notalist", 42, {}])
    def test_list_field_deserialize_value_that_is_not_a_list(self, value):
        field = fields.List(fields.Str())
        with pytest.raises(ValidationError) as excinfo:
            field.deserialize(value)
        assert excinfo.value.args[0] == "Not a valid list."

    def test_datetime_int_tuple_field_deserialization(self):
        dtime = dt.datetime.now()
        data = dtime.isoformat(), 42
        field = fields.Tuple([fields.DateTime(), fields.Integer()])
        result = field.deserialize(data)

        assert isinstance(result, tuple)
        assert len(result) == 2
        for val, type_, true_val in zip(result, (dt.datetime, int), (dtime, 42)):
            assert isinstance(val, type_)
            assert val == true_val

    def test_tuple_field_deserialize_invalid_item(self):
        field = fields.Tuple([fields.DateTime])
        with pytest.raises(ValidationError) as excinfo:
            field.deserialize(["badvalue"])
        assert excinfo.value.args[0] == {0: ["Not a valid datetime."]}

        field = fields.Tuple([fields.Str(), fields.Integer()])
        with pytest.raises(ValidationError) as excinfo:
            field.deserialize(["good", "bad"])
        assert excinfo.value.args[0] == {1: ["Not a valid integer."]}

    def test_tuple_field_deserialize_multiple_invalid_items(self):
        validator = validate.Range(10, 20, error="Value {input} not in range")
        field = fields.Tuple(
            [
                fields.Int(validate=validator),
                fields.Int(validate=validator),
                fields.Int(validate=validator),
            ]
        )

        with pytest.raises(ValidationError) as excinfo:
            field.deserialize([10, 5, 25])
        assert len(excinfo.value.args[0]) == 2
        assert excinfo.value.args[0][1] == ["Value 5 not in range"]
        assert excinfo.value.args[0][2] == ["Value 25 not in range"]

    @pytest.mark.parametrize("value", ["notalist", 42, {}])
    def test_tuple_field_deserialize_value_that_is_not_a_collection(self, value):
        field = fields.Tuple([fields.Str()])
        with pytest.raises(ValidationError) as excinfo:
            field.deserialize(value)
        assert excinfo.value.args[0] == "Not a valid tuple."

    def test_tuple_field_deserialize_invalid_length(self):
        field = fields.Tuple([fields.Str(), fields.Str()])
        with pytest.raises(ValidationError) as excinfo:
            field.deserialize([42])
        assert excinfo.value.args[0] == "Length must be 2."

    def test_constant_field_deserialization(self):
        field = fields.Constant("something")
        assert field.deserialize("whatever") == "something"

    def test_constant_is_always_included_in_deserialized_data(self):
        class MySchema(Schema):
            foo = fields.Constant(42)

        sch = MySchema()
        assert sch.load({})["foo"] == 42
        assert sch.load({"foo": 24})["foo"] == 42

    def test_field_deserialization_with_user_validator_function(self):
        field = fields.String(validate=predicate(lambda s: s.lower() == "valid"))
        assert field.deserialize("Valid") == "Valid"
        with pytest.raises(ValidationError) as excinfo:
            field.deserialize("invalid")
        assert excinfo.value.args[0][0] == "Invalid value."
        assert type(excinfo.value) is ValidationError

    def test_field_deserialization_with_user_validator_that_raises_error_with_list(
        self,
    ):
        def validator(val):
            raise ValidationError(["err1", "err2"])

        class MySchema(Schema):
            foo = fields.Raw(validate=validator)

        errors = MySchema().validate({"foo": 42})
        assert errors["foo"] == ["err1", "err2"]

    def test_field_deserialization_with_validator_with_nonascii_input(self):
        def validate(val):
            raise ValidationError("oops")

        field = fields.String(validate=validate)
        with pytest.raises(ValidationError) as excinfo:
            field.deserialize("привет")
        assert type(excinfo.value) is ValidationError

    def test_field_deserialization_with_user_validators(self):
        validators_gen = (
            func
            for func in (
                predicate(lambda s: s.lower() == "valid"),
                predicate(lambda s: s.lower()[::-1] == "dilav"),
            )
        )

        m_colletion_type = [
            fields.String(
                validate=[
                    predicate(lambda s: s.lower() == "valid"),
                    predicate(lambda s: s.lower()[::-1] == "dilav"),
                ]
            ),
            fields.String(
                validate=(
                    predicate(lambda s: s.lower() == "valid"),
                    predicate(lambda s: s.lower()[::-1] == "dilav"),
                )
            ),
            fields.String(validate=validators_gen),
        ]

        for field in m_colletion_type:
            assert field.deserialize("Valid") == "Valid"
            with pytest.raises(ValidationError, match="Invalid value."):
                field.deserialize("invalid")

    @pytest.mark.parametrize(
        ("field", "value"),
        (
            pytest.param(fields.List(fields.String()), ["foo", "bar"], id="List"),
            pytest.param(
                fields.Tuple((fields.String(), fields.Integer())),
                ("foo", 42),
                id="Tuple",
            ),
            pytest.param(fields.String(), "valid", id="String"),
            pytest.param(fields.UUID(), uuid.uuid4(), id="UUID"),
            pytest.param(fields.Integer(), 42, id="Integer"),
            pytest.param(fields.Float(), 42.3, id="Float"),
            pytest.param(fields.Decimal(), decimal.Decimal("42.3"), id="Decimal"),
            pytest.param(fields.Boolean(), True, id="Boolean"),
            pytest.param(fields.DateTime(), dt.datetime(2014, 8, 21), id="DateTime"),
            pytest.param(fields.Time(), dt.time(10, 15), id="Time"),
            pytest.param(fields.Date(), dt.date(2014, 8, 21), id="Date"),
            pytest.param(fields.TimeDelta(), dt.timedelta(days=1), id="TimeDelta"),
            pytest.param(fields.Dict(), {"foo": "bar"}, id="Dict"),
            pytest.param(fields.Url(), "https://mallow.com", id="Url"),
            pytest.param(fields.Email(), "barbara37@example.net", id="Email"),
            pytest.param(fields.IP(), ipaddress.IPv4Address("67.60.134.65"), id="IP"),
            pytest.param(
                fields.IPv4(), ipaddress.IPv4Address("55.81.158.106"), id="IPv4"
            ),
            pytest.param(
                fields.IPv6(),
                ipaddress.IPv6Address("89f4:41b6:b97e:ad48:8480:1fda:a811:d1a5"),
                id="IPv6",
            ),
            pytest.param(fields.Enum(GenderEnum), GenderEnum.non_binary, id="Enum"),
        ),
    )
    def test_fields_accept_internal_types(self, field, value):
        assert field.deserialize(value) == value


# No custom deserialization behavior, so a dict is returned
class SimpleUserSchema(Schema):
    name = fields.String()
    age = fields.Float()


class Validator(Schema):
    email = fields.Email()
    colors = fields.Str(validate=validate.OneOf(["red", "blue"]))
    age = fields.Integer(validate=validate.Range(min=0, min_inclusive=False))


class Validators(Schema):
    email = fields.Email()
    colors = fields.Str(validate=validate.OneOf(["red", "blue"]))
    age = fields.Integer(validate=[validate.Range(1, 99)])


class TestSchemaDeserialization:
    def test_deserialize_to_dict(self):
        user_dict = {"name": "Monty", "age": "42.3"}
        result = SimpleUserSchema().load(user_dict)
        assert result["name"] == "Monty"
        assert math.isclose(result["age"], 42.3)

    def test_deserialize_with_missing_values(self):
        user_dict = {"name": "Monty"}
        result = SimpleUserSchema().load(user_dict)
        # 'age' is not included in result
        assert result == {"name": "Monty"}

    def test_deserialize_many(self):
        users_data = [{"name": "Mick", "age": "914"}, {"name": "Keith", "age": "8442"}]
        result = SimpleUserSchema(many=True).load(users_data)
        assert isinstance(result, list)
        user = result[0]
        assert user["age"] == int(users_data[0]["age"])

    def test_exclude(self):
        schema = SimpleUserSchema(exclude=("age",), unknown=EXCLUDE)
        result = schema.load({"name": "Monty", "age": 42})
        assert "name" in result
        assert "age" not in result

    def test_nested_single_deserialization_to_dict(self):
        class SimpleBlogSerializer(Schema):
            title = fields.String()
            author = fields.Nested(SimpleUserSchema, unknown=EXCLUDE)

        blog_dict = {
            "title": "Gimme Shelter",
            "author": {"name": "Mick", "age": "914", "email": "mick@stones.com"},
        }
        result = SimpleBlogSerializer().load(blog_dict)
        author = result["author"]
        assert author["name"] == "Mick"
        assert author["age"] == 914
        assert "email" not in author

    def test_nested_list_deserialization_to_dict(self):
        class SimpleBlogSerializer(Schema):
            title = fields.String()
            authors = fields.Nested(SimpleUserSchema, many=True)

        blog_dict = {
            "title": "Gimme Shelter",
            "authors": [
                {"name": "Mick", "age": "914"},
                {"name": "Keith", "age": "8442"},
            ],
        }
        result = SimpleBlogSerializer().load(blog_dict)
        assert isinstance(result["authors"], list)
        author = result["authors"][0]
        assert author["name"] == "Mick"
        assert author["age"] == 914

    def test_nested_single_none_not_allowed(self):
        class PetSchema(Schema):
            name = fields.Str()

        class OwnerSchema(Schema):
            pet = fields.Nested(PetSchema(), allow_none=False)

        sch = OwnerSchema()
        errors = sch.validate({"pet": None})
        assert "pet" in errors
        assert errors["pet"] == ["Field may not be null."]

    def test_nested_many_non_not_allowed(self):
        class PetSchema(Schema):
            name = fields.Str()

        class StoreSchema(Schema):
            pets = fields.Nested(PetSchema, allow_none=False, many=True)

        sch = StoreSchema()
        errors = sch.validate({"pets": None})
        assert "pets" in errors
        assert errors["pets"] == ["Field may not be null."]

    def test_nested_single_required_missing(self):
        class PetSchema(Schema):
            name = fields.Str()

        class OwnerSchema(Schema):
            pet = fields.Nested(PetSchema(), required=True)

        sch = OwnerSchema()
        errors = sch.validate({})
        assert "pet" in errors
        assert errors["pet"] == ["Missing data for required field."]

    def test_nested_many_required_missing(self):
        class PetSchema(Schema):
            name = fields.Str()

        class StoreSchema(Schema):
            pets = fields.Nested(PetSchema, required=True, many=True)

        sch = StoreSchema()
        errors = sch.validate({})
        assert "pets" in errors
        assert errors["pets"] == ["Missing data for required field."]

    def test_nested_only_basestring(self):
        class ANestedSchema(Schema):
            pk = fields.Str()

        class MainSchema(Schema):
            pk = fields.Str()
            child = fields.Pluck(ANestedSchema, "pk")

        sch = MainSchema()
        result = sch.load({"pk": "123", "child": "456"})
        assert result["child"]["pk"] == "456"

    def test_nested_only_basestring_with_list_data(self):
        class ANestedSchema(Schema):
            pk = fields.Str()

        class MainSchema(Schema):
            pk = fields.Str()
            children = fields.Pluck(ANestedSchema, "pk", many=True)

        sch = MainSchema()
        result = sch.load({"pk": "123", "children": ["456", "789"]})
        assert result["children"][0]["pk"] == "456"
        assert result["children"][1]["pk"] == "789"

    def test_nested_none_deserialization(self):
        class SimpleBlogSerializer(Schema):
            title = fields.String()
            author = fields.Nested(SimpleUserSchema, allow_none=True)

        blog_dict = {"title": "Gimme Shelter", "author": None}
        result = SimpleBlogSerializer().load(blog_dict)
        assert result["author"] is None
        assert result["title"] == blog_dict["title"]

    def test_deserialize_with_attribute_param(self):
        class AliasingUserSerializer(Schema):
            username = fields.Email(attribute="email")
            years = fields.Integer(attribute="age")

        data = {"username": "foo@bar.com", "years": "42"}
        result = AliasingUserSerializer().load(data)
        assert result["email"] == "foo@bar.com"
        assert result["age"] == 42

    # regression test for https://github.com/marshmallow-code/marshmallow/issues/450
    def test_deserialize_with_attribute_param_symmetry(self):
        class MySchema(Schema):
            foo = fields.Raw(attribute="bar.baz")

        schema = MySchema()
        dump_data = schema.dump({"bar": {"baz": 42}})
        assert dump_data == {"foo": 42}

        load_data = schema.load({"foo": 42})
        assert load_data == {"bar": {"baz": 42}}

    def test_deserialize_with_attribute_param_error_returns_field_name_not_attribute_name(
        self,
    ):
        class AliasingUserSerializer(Schema):
            username = fields.Email(attribute="email")
            years = fields.Integer(attribute="age")

        data = {"username": "foobar.com", "years": "42"}
        with pytest.raises(ValidationError) as excinfo:
            AliasingUserSerializer().load(data)
        errors = excinfo.value.messages
        assert errors["username"] == ["Not a valid email address."]

    def test_deserialize_with_attribute_param_error_returns_data_key_not_attribute_name(
        self,
    ):
        class AliasingUserSerializer(Schema):
            name = fields.String(data_key="Name")
            username = fields.Email(attribute="email", data_key="UserName")
            years = fields.Integer(attribute="age", data_key="Years")

        data = {"Name": "Mick", "UserName": "foobar.com", "Years": "abc"}
        with pytest.raises(ValidationError) as excinfo:
            AliasingUserSerializer().load(data)
        errors = excinfo.value.messages
        assert errors["UserName"] == ["Not a valid email address."]
        assert errors["Years"] == ["Not a valid integer."]

    def test_deserialize_with_data_key_param(self):
        class AliasingUserSerializer(Schema):
            name = fields.String(data_key="Name")
            username = fields.Email(attribute="email", data_key="UserName")
            years = fields.Integer(data_key="Years")

        data = {"Name": "Mick", "UserName": "foo@bar.com", "years": "42"}
        result = AliasingUserSerializer(unknown=EXCLUDE).load(data)
        assert result["name"] == "Mick"
        assert result["email"] == "foo@bar.com"
        assert "years" not in result

    def test_deserialize_with_data_key_as_empty_string(self):
        class MySchema(Schema):
            name = fields.Raw(data_key="")

        schema = MySchema()
        assert schema.load({"": "Grace"}) == {"name": "Grace"}

    def test_deserialize_with_dump_only_param(self):
        class AliasingUserSerializer(Schema):
            name = fields.String()
            years = fields.Integer(dump_only=True)
            size = fields.Integer(dump_only=True, load_only=True)
            nicknames = fields.List(fields.Str(), dump_only=True)

        data = {
            "name": "Mick",
            "years": "42",
            "size": "12",
            "nicknames": ["Your Majesty", "Brenda"],
        }
        result = AliasingUserSerializer(unknown=EXCLUDE).load(data)
        assert result["name"] == "Mick"
        assert "years" not in result
        assert "size" not in result
        assert "nicknames" not in result

    def test_deserialize_with_missing_param_value(self):
        bdate = dt.datetime(2017, 9, 29)

        class AliasingUserSerializer(Schema):
            name = fields.String()
            birthdate = fields.DateTime(load_default=bdate)

        data = {"name": "Mick"}
        result = AliasingUserSerializer().load(data)
        assert result["name"] == "Mick"
        assert result["birthdate"] == bdate

    def test_deserialize_with_missing_param_callable(self):
        bdate = dt.datetime(2017, 9, 29)

        class AliasingUserSerializer(Schema):
            name = fields.String()
            birthdate = fields.DateTime(load_default=lambda: bdate)

        data = {"name": "Mick"}
        result = AliasingUserSerializer().load(data)
        assert result["name"] == "Mick"
        assert result["birthdate"] == bdate

    def test_deserialize_with_missing_param_none(self):
        class AliasingUserSerializer(Schema):
            name = fields.String()
            years = fields.Integer(load_default=None, allow_none=True)

        data = {"name": "Mick"}
        result = AliasingUserSerializer().load(data)
        assert result["name"] == "Mick"
        assert result["years"] is None

    def test_deserialization_raises_with_errors(self):
        bad_data = {"email": "invalid-email", "colors": "burger", "age": -1}
        v = Validator()
        with pytest.raises(ValidationError) as excinfo:
            v.load(bad_data)
        errors = excinfo.value.messages
        assert "email" in errors
        assert "colors" in errors
        assert "age" in errors

    def test_deserialization_raises_with_errors_with_multiple_validators(self):
        bad_data = {"email": "invalid-email", "colors": "burger", "age": -1}
        v = Validators()
        with pytest.raises(ValidationError) as excinfo:
            v.load(bad_data)
        errors = excinfo.value.messages
        assert "email" in errors
        assert "colors" in errors
        assert "age" in errors

    def test_deserialization_many_raises_errors(self):
        bad_data = [
            {"email": "foo@bar.com", "colors": "red", "age": 18},
            {"email": "bad", "colors": "pizza", "age": -1},
        ]
        v = Validator(many=True)
        with pytest.raises(ValidationError):
            v.load(bad_data)

    def test_validation_errors_are_stored(self):
        def validate_field(val):
            raise ValidationError("Something went wrong")

        class MySchema(Schema):
            foo = fields.Raw(validate=validate_field)

        with pytest.raises(ValidationError) as excinfo:
            MySchema().load({"foo": 42})
        errors = excinfo.value.messages
        assert "Something went wrong" in errors["foo"]

    def test_multiple_errors_can_be_stored_for_a_field(self):
        def validate1(n):
            raise ValidationError("error one")

        def validate2(n):
            raise ValidationError("error two")

        class MySchema(Schema):
            foo = fields.Raw(required=True, validate=[validate1, validate2])

        with pytest.raises(ValidationError) as excinfo:
            MySchema().load({"foo": "bar"})
        errors = excinfo.value.messages

        assert type(errors["foo"]) is list
        assert len(errors["foo"]) == 2

    def test_multiple_errors_can_be_stored_for_an_email_field(self):
        def validate(val):
            raise ValidationError("Invalid value.")

        class MySchema(Schema):
            email = fields.Email(validate=[validate])

        with pytest.raises(ValidationError) as excinfo:
            MySchema().load({"email": "foo"})
        errors = excinfo.value.messages
        assert len(errors["email"]) == 2
        assert "Not a valid email address." in errors["email"][0]

    def test_multiple_errors_can_be_stored_for_a_url_field(self):
        def validator(val):
            raise ValidationError("Not a valid URL.")

        class MySchema(Schema):
            url = fields.Url(validate=[validator])

        with pytest.raises(ValidationError) as excinfo:
            MySchema().load({"url": "foo"})
        errors = excinfo.value.messages
        assert len(errors["url"]) == 2
        assert "Not a valid URL." in errors["url"][0]

    def test_required_value_only_passed_to_validators_if_provided(self):
        class MySchema(Schema):
            foo = fields.Raw(required=True, validate=lambda f: False)

        with pytest.raises(ValidationError) as excinfo:
            MySchema().load({})
        errors = excinfo.value.messages
        # required value missing
        assert len(errors["foo"]) == 1
        assert "Missing data for required field." in errors["foo"]

    @pytest.mark.parametrize("partial_schema", [True, False])
    def test_partial_deserialization(self, partial_schema):
        class MySchema(Schema):
            foo = fields.Raw(required=True)
            bar = fields.Raw(required=True)

        data = {"foo": 3}
        if partial_schema:
            result = MySchema(partial=True).load(data)
        else:
            result = MySchema().load(data, partial=True)

        assert result["foo"] == 3
        assert "bar" not in result

    def test_partial_fields_deserialization(self):
        class MySchema(Schema):
            foo = fields.Raw(required=True)
            bar = fields.Raw(required=True)
            baz = fields.Raw(required=True)

        with pytest.raises(ValidationError) as excinfo:
            MySchema().load({"foo": 3}, partial=tuple())
        data, errors = excinfo.value.valid_data, excinfo.value.messages
        assert data["foo"] == 3
        assert "bar" in errors
        assert "baz" in errors

        data = MySchema().load({"foo": 3}, partial=("bar", "baz"))
        assert isinstance(data, dict)
        assert data["foo"] == 3
        assert "bar" not in data
        assert "baz" not in data

        data = MySchema(partial=True).load({"foo": 3}, partial=("bar", "baz"))
        assert isinstance(data, dict)
        assert data["foo"] == 3
        assert "bar" not in data
        assert "baz" not in data

    def test_partial_fields_validation(self):
        class MySchema(Schema):
            foo = fields.Raw(required=True)
            bar = fields.Raw(required=True)
            baz = fields.Raw(required=True)

        errors = MySchema().validate({"foo": 3}, partial=tuple())
        assert "bar" in errors
        assert "baz" in errors

        errors = MySchema().validate({"foo": 3}, partial=("bar", "baz"))
        assert errors == {}

        errors = MySchema(partial=True).validate({"foo": 3}, partial=("bar", "baz"))
        assert errors == {}

    def test_unknown_fields_deserialization(self):
        class MySchema(Schema):
            foo = fields.Integer()

        data = MySchema(unknown=EXCLUDE).load({"foo": 3, "bar": 5})
        assert data["foo"] == 3
        assert "bar" not in data

        data = MySchema(unknown=INCLUDE).load({"foo": 3, "bar": 5}, unknown=EXCLUDE)
        assert data["foo"] == 3
        assert "bar" not in data

        data = MySchema(unknown=EXCLUDE).load({"foo": 3, "bar": 5}, unknown=INCLUDE)
        assert data["foo"] == 3
        assert data["bar"]

        data = MySchema(unknown=INCLUDE).load({"foo": 3, "bar": 5})
        assert data["foo"] == 3
        assert data["bar"]

        with pytest.raises(ValidationError, match="foo"):
            MySchema(unknown=INCLUDE).load({"foo": "asd", "bar": 5})

        data = MySchema(unknown=INCLUDE, many=True).load(
            [{"foo": 1}, {"foo": 3, "bar": 5}]
        )
        assert "foo" in data[1]
        assert "bar" in data[1]

        with pytest.raises(ValidationError) as excinfo:
            MySchema().load({"foo": 3, "bar": 5})
        err = excinfo.value
        assert "bar" in err.messages
        assert err.messages["bar"] == ["Unknown field."]

        with pytest.raises(ValidationError) as excinfo:
            MySchema(many=True).load([{"foo": "abc"}, {"foo": 3, "bar": 5}])
        err = excinfo.value
        assert 0 in err.messages
        assert "foo" in err.messages[0]
        assert err.messages[0]["foo"] == ["Not a valid integer."]
        assert 1 in err.messages
        assert "bar" in err.messages[1]
        assert err.messages[1]["bar"] == ["Unknown field."]

    def test_unknown_fields_deserialization_precedence(self):
        class MySchema(Schema):
            class Meta:
                unknown = INCLUDE

            foo = fields.Integer()

        data = MySchema().load({"foo": 3, "bar": 5})
        assert data["foo"] == 3
        assert data["bar"] == 5

        data = MySchema(unknown=EXCLUDE).load({"foo": 3, "bar": 5})
        assert data["foo"] == 3
        assert "bar" not in data

        data = MySchema().load({"foo": 3, "bar": 5}, unknown=EXCLUDE)
        assert data["foo"] == 3
        assert "bar" not in data

        with pytest.raises(ValidationError):
            MySchema(unknown=EXCLUDE).load({"foo": 3, "bar": 5}, unknown=RAISE)

    def test_unknown_fields_deserialization_with_data_key(self):
        class MySchema(Schema):
            foo = fields.Integer(data_key="Foo")

        data = MySchema().load({"Foo": 1})
        assert data["foo"] == 1
        assert "Foo" not in data

        data = MySchema(unknown=RAISE).load({"Foo": 1})
        assert data["foo"] == 1
        assert "Foo" not in data

        with pytest.raises(ValidationError):
            MySchema(unknown=RAISE).load({"foo": 1})

        data = MySchema(unknown=INCLUDE).load({"Foo": 1})
        assert data["foo"] == 1
        assert "Foo" not in data

    def test_unknown_fields_deserialization_with_index_errors_false(self):
        class MySchema(Schema):
            foo = fields.Integer()

            class Meta:
                unknown = RAISE
                index_errors = False

        with pytest.raises(ValidationError) as excinfo:
            MySchema(many=True).load([{"foo": "invalid"}, {"foo": 42, "bar": 24}])
        err = excinfo.value
        assert 1 not in err.messages
        assert "foo" in err.messages
        assert "bar" in err.messages
        assert err.messages["foo"] == ["Not a valid integer."]
        assert err.messages["bar"] == ["Unknown field."]

    def test_dump_only_fields_considered_unknown(self):
        class MySchema(Schema):
            foo = fields.Int(dump_only=True)

        with pytest.raises(ValidationError) as excinfo:
            MySchema().load({"foo": 42})
        err = excinfo.value
        assert "foo" in err.messages
        assert err.messages["foo"] == ["Unknown field."]

        # When unknown = INCLUDE, dump-only fields are included as unknown
        # without any validation.
        data = MySchema(unknown=INCLUDE).load({"foo": "LOL"})
        assert data["foo"] == "LOL"

    def test_unknown_fields_do_not_unpack_dotted_names(self):
        class MySchema(Schema):
            class Meta:
                unknown = INCLUDE

            foo = fields.Str()
            bar = fields.Str(data_key="bar.baz")

        # dotted names are still supported
        data = MySchema().load({"foo": "hi", "bar.baz": "okay"})
        assert data == {"foo": "hi", "bar": "okay"}

        # but extra keys included via unknown=INCLUDE are not transformed into nested dicts
        data = MySchema().load({"foo": "hi", "bar.baz": "okay", "alpha.beta": "woah!"})
        assert data == {"foo": "hi", "bar": "okay", "alpha.beta": "woah!"}


validators_gen = (
    func for func in [predicate(lambda x: x <= 24), predicate(lambda x: x >= 18)]
)

validators_gen_float = (
    func for func in [predicate(lambda f: f <= 4.1), predicate(lambda f: f >= 1.0)]
)

validators_gen_str = (
    func
    for func in [
        predicate(lambda n: len(n) == 3),
        predicate(lambda n: n[1].lower() == "o"),
    ]
)


class TestValidation:
    def test_integer_with_validator(self):
        field = fields.Integer(validate=validate.Range(18, 24))
        out = field.deserialize("20")
        assert out == 20
        with pytest.raises(ValidationError):
            field.deserialize(25)

    @pytest.mark.parametrize(
        "field",
        [
            fields.Integer(
                validate=[predicate(lambda x: x <= 24), predicate(lambda x: x >= 18)]
            ),
            fields.Integer(
                validate=(predicate(lambda x: x <= 24), predicate(lambda x: x >= 18))
            ),
            fields.Integer(validate=validators_gen),
        ],
    )
    def test_integer_with_validators(self, field):
        out = field.deserialize("20")
        assert out == 20
        with pytest.raises(ValidationError):
            field.deserialize(25)

    @pytest.mark.parametrize(
        "field",
        [
            fields.Float(
                validate=[predicate(lambda f: f <= 4.1), predicate(lambda f: f >= 1.0)]
            ),
            fields.Float(
                validate=(predicate(lambda f: f <= 4.1), predicate(lambda f: f >= 1.0))
            ),
            fields.Float(validate=validators_gen_float),
        ],
    )
    def test_float_with_validators(self, field):
        assert field.deserialize(3.14)
        with pytest.raises(ValidationError):
            field.deserialize(4.2)

    def test_string_validator(self):
        field = fields.String(validate=validate.Length(equal=3))
        assert field.deserialize("Joe") == "Joe"
        with pytest.raises(ValidationError):
            field.deserialize("joseph")

    def test_function_validator(self):
        field = fields.Function(
            lambda d: d.name.upper(), validate=validate.Length(equal=3)
        )
        assert field.deserialize("joe")
        with pytest.raises(ValidationError):
            field.deserialize("joseph")

    @pytest.mark.parametrize(
        "field",
        [
            fields.Function(
                lambda d: d.name.upper(),
                validate=[
                    validate.Length(equal=3),
                    predicate(lambda n: n[1].lower() == "o"),
                ],
            ),
            fields.Function(
                lambda d: d.name.upper(),
                validate=(
                    predicate(lambda n: len(n) == 3),
                    predicate(lambda n: n[1].lower() == "o"),
                ),
            ),
            fields.Function(lambda d: d.name.upper(), validate=validators_gen_str),
        ],
    )
    def test_function_validators(self, field):
        assert field.deserialize("joe")
        with pytest.raises(ValidationError):
            field.deserialize("joseph")

    def test_method_validator(self):
        class MethodSerializer(Schema):
            name = fields.Method(
                "get_name", deserialize="get_name", validate=validate.Length(equal=3)
            )

            def get_name(self, val):
                return val.upper()

        assert MethodSerializer().load({"name": "joe"})
        with pytest.raises(ValidationError, match="Length must be 3."):
            MethodSerializer().load({"name": "joseph"})

    # Regression test for https://github.com/marshmallow-code/marshmallow/issues/269
    def test_nested_data_is_stored_when_validation_fails(self):
        class SchemaA(Schema):
            x = fields.Integer()
            y = fields.Integer(validate=validate.Range(min=1))
            z = fields.Integer()

        class SchemaB(Schema):
            w = fields.Integer()
            n = fields.Nested(SchemaA)

        sch = SchemaB()

        with pytest.raises(ValidationError) as excinfo:
            sch.load({"w": 90, "n": {"x": 90, "y": 89, "z": None}})
        data, errors = excinfo.value.valid_data, excinfo.value.messages
        assert "z" in errors["n"]
        assert data == {"w": 90, "n": {"x": 90, "y": 89}}

        with pytest.raises(ValidationError) as excinfo:
            sch.load({"w": 90, "n": {"x": 90, "y": -1, "z": 180}})
        data, errors = excinfo.value.valid_data, excinfo.value.messages
        assert "y" in errors["n"]
        assert data == {"w": 90, "n": {"x": 90, "z": 180}}

    def test_nested_partial_load(self):
        class SchemaA(Schema):
            x = fields.Integer(required=True)
            y = fields.Integer()

        class SchemaB(Schema):
            z = fields.Nested(SchemaA)

        b_dict = {"z": {"y": 42}}
        # Partial loading shouldn't generate any errors.
        result = SchemaB().load(b_dict, partial=True)
        assert result["z"]["y"] == 42
        # Non partial loading should complain about missing values.
        with pytest.raises(ValidationError) as excinfo:
            SchemaB().load(b_dict)
        data, errors = excinfo.value.valid_data, excinfo.value.messages
        assert data["z"]["y"] == 42
        assert "z" in errors
        assert "x" in errors["z"]

    def test_deeply_nested_partial_load(self):
        class SchemaC(Schema):
            x = fields.Integer(required=True)
            y = fields.Integer()

        class SchemaB(Schema):
            c = fields.Nested(SchemaC)

        class SchemaA(Schema):
            b = fields.Nested(SchemaB)

        a_dict = {"b": {"c": {"y": 42}}}
        # Partial loading shouldn't generate any errors.
        result = SchemaA().load(a_dict, partial=True)
        assert result["b"]["c"]["y"] == 42
        # Non partial loading should complain about missing values.
        with pytest.raises(ValidationError) as excinfo:
            SchemaA().load(a_dict)
        data, errors = excinfo.value.valid_data, excinfo.value.messages
        assert data["b"]["c"]["y"] == 42
        assert "b" in errors
        assert "c" in errors["b"]
        assert "x" in errors["b"]["c"]

    def test_nested_partial_tuple(self):
        class SchemaA(Schema):
            x = fields.Integer(required=True)
            y = fields.Integer(required=True)

        class SchemaB(Schema):
            z = fields.Nested(SchemaA)

        b_dict = {"z": {"y": 42}}
        # If we ignore the missing z.x, z.y should still load.
        result = SchemaB().load(b_dict, partial=("z.x",))
        assert result["z"]["y"] == 42
        # If we ignore a missing z.y we should get a validation error.
        with pytest.raises(ValidationError):
            SchemaB().load(b_dict, partial=("z.y",))

    def test_nested_partial_default(self):
        class SchemaA(Schema):
            x = fields.Integer(required=True)
            y = fields.Integer(required=True)

        class SchemaB(Schema):
            z = fields.Nested(SchemaA(partial=("x",)))

        b_dict = {"z": {"y": 42}}
        # Nested partial args should be respected.
        result = SchemaB().load(b_dict)
        assert result["z"]["y"] == 42
        with pytest.raises(ValidationError):
            SchemaB().load({"z": {"x": 0}})


@pytest.mark.parametrize("FieldClass", ALL_FIELDS)
def test_required_field_failure(FieldClass):
    class RequireSchema(Schema):
        age = FieldClass(required=True)

    user_data = {"name": "Phil"}
    with pytest.raises(ValidationError) as excinfo:
        RequireSchema().load(user_data)
    errors = excinfo.value.messages
    assert "Missing data for required field." in errors["age"]


@pytest.mark.parametrize(
    "message",
    [
        "My custom required message",
        {"error": "something", "code": 400},
        ["first error", "second error"],
    ],
)
def test_required_message_can_be_changed(message):
    class RequireSchema(Schema):
        age = fields.Integer(required=True, error_messages={"required": message})

    user_data = {"name": "Phil"}
    with pytest.raises(ValidationError) as excinfo:
        RequireSchema().load(user_data)
    errors = excinfo.value.messages
    expected = [message] if isinstance(message, str) else message
    assert expected == errors["age"]


@pytest.mark.parametrize("unknown", (EXCLUDE, INCLUDE, RAISE))
@pytest.mark.parametrize("data", [True, False, 42, None, []])
def test_deserialize_raises_exception_if_input_type_is_incorrect(data, unknown):
    class MySchema(Schema):
        foo = fields.Raw()
        bar = fields.Raw()

    with pytest.raises(ValidationError, match="Invalid input type.") as excinfo:
        MySchema(unknown=unknown).load(data)
    exc = excinfo.value
    assert isinstance(exc.messages, dict)
    assert list(exc.messages.keys()) == ["_schema"]


# === tests/test_registry.py ===
import pytest

from marshmallow import Schema, class_registry, fields
from marshmallow.exceptions import RegistryError


def test_serializer_has_class_registry():
    class MySchema(Schema):
        pass

    class MySubSchema(Schema):
        pass

    assert "MySchema" in class_registry._registry
    assert "MySubSchema" in class_registry._registry

    # by fullpath
    assert "tests.test_registry.MySchema" in class_registry._registry
    assert "tests.test_registry.MySubSchema" in class_registry._registry


def test_register_class_meta_option():
    class UnregisteredSchema(Schema):
        class Meta:
            register = False

    class RegisteredSchema(Schema):
        class Meta:
            register = True

    class RegisteredOverrideSchema(UnregisteredSchema):
        class Meta:
            register = True

    class UnregisteredOverrideSchema(RegisteredSchema):
        class Meta:
            register = False

    assert "UnregisteredSchema" not in class_registry._registry
    assert "tests.test_registry.UnregisteredSchema" not in class_registry._registry

    assert "RegisteredSchema" in class_registry._registry
    assert "tests.test_registry.RegisteredSchema" in class_registry._registry

    assert "RegisteredOverrideSchema" in class_registry._registry
    assert "tests.test_registry.RegisteredOverrideSchema" in class_registry._registry

    assert "UnregisteredOverrideSchema" not in class_registry._registry
    assert (
        "tests.test_registry.UnregisteredOverrideSchema" not in class_registry._registry
    )


def test_serializer_class_registry_register_same_classname_different_module():
    reglen = len(class_registry._registry)

    type("MyTestRegSchema", (Schema,), {"__module__": "modA"})

    assert "MyTestRegSchema" in class_registry._registry
    result = class_registry._registry.get("MyTestRegSchema")
    assert isinstance(result, list)
    assert len(result) == 1
    assert "modA.MyTestRegSchema" in class_registry._registry
    # storing for classname and fullpath
    assert len(class_registry._registry) == reglen + 2

    type("MyTestRegSchema", (Schema,), {"__module__": "modB"})

    assert "MyTestRegSchema" in class_registry._registry
    # aggregating classes with same name from different modules
    result = class_registry._registry.get("MyTestRegSchema")
    assert isinstance(result, list)
    assert len(result) == 2
    assert "modB.MyTestRegSchema" in class_registry._registry
    # storing for same classname (+0) and different module (+1)
    assert len(class_registry._registry) == reglen + 2 + 1

    type("MyTestRegSchema", (Schema,), {"__module__": "modB"})

    assert "MyTestRegSchema" in class_registry._registry
    # only the class with matching module has been replaced
    result = class_registry._registry.get("MyTestRegSchema")
    assert isinstance(result, list)
    assert len(result) == 2
    assert "modB.MyTestRegSchema" in class_registry._registry
    # only the class with matching module has been replaced (+0)
    assert len(class_registry._registry) == reglen + 2 + 1


def test_serializer_class_registry_override_if_same_classname_same_module():
    reglen = len(class_registry._registry)

    type("MyTestReg2Schema", (Schema,), {"__module__": "SameModulePath"})

    assert "MyTestReg2Schema" in class_registry._registry
    result = class_registry._registry.get("MyTestReg2Schema")
    assert isinstance(result, list)
    assert len(result) == 1
    assert "SameModulePath.MyTestReg2Schema" in class_registry._registry
    result = class_registry._registry.get("SameModulePath.MyTestReg2Schema")
    assert isinstance(result, list)
    assert len(result) == 1
    # storing for classname and fullpath
    assert len(class_registry._registry) == reglen + 2

    type("MyTestReg2Schema", (Schema,), {"__module__": "SameModulePath"})

    assert "MyTestReg2Schema" in class_registry._registry
    # overriding same class name and same module
    result = class_registry._registry.get("MyTestReg2Schema")
    assert isinstance(result, list)
    assert len(result) == 1

    assert "SameModulePath.MyTestReg2Schema" in class_registry._registry
    # overriding same fullpath
    result = class_registry._registry.get("SameModulePath.MyTestReg2Schema")
    assert isinstance(result, list)
    assert len(result) == 1
    # overriding for same classname (+0) and different module (+0)
    assert len(class_registry._registry) == reglen + 2


class A:
    def __init__(self, _id, b=None):
        self.id = _id
        self.b = b


class B:
    def __init__(self, _id, a=None):
        self.id = _id
        self.a = a


class C:
    def __init__(self, _id, bs=None):
        self.id = _id
        self.bs = bs or []


class ASchema(Schema):
    id = fields.Integer()
    b = fields.Nested("tests.test_registry.BSchema", exclude=("a",))


class BSchema(Schema):
    id = fields.Integer()
    a = fields.Nested("tests.test_registry.ASchema")


class CSchema(Schema):
    id = fields.Integer()
    bs = fields.Nested("tests.test_registry.BSchema", many=True)


def test_two_way_nesting():
    a_obj = A(1)
    b_obj = B(2, a=a_obj)
    a_obj.b = b_obj

    a_serialized = ASchema().dump(a_obj)
    b_serialized = BSchema().dump(b_obj)
    assert a_serialized["b"]["id"] == b_obj.id
    assert b_serialized["a"]["id"] == a_obj.id


def test_nesting_with_class_name_many():
    c_obj = C(1, bs=[B(2), B(3), B(4)])

    c_serialized = CSchema().dump(c_obj)

    assert len(c_serialized["bs"]) == len(c_obj.bs)
    assert c_serialized["bs"][0]["id"] == c_obj.bs[0].id


def test_invalid_class_name_in_nested_field_raises_error(user):
    class MySchema(Schema):
        nf = fields.Nested("notfound")

    sch = MySchema()
    msg = "Class with name {!r} was not found".format("notfound")
    with pytest.raises(RegistryError, match=msg):
        sch.dump({"nf": None})


class FooSerializer(Schema):
    _id = fields.Integer()


def test_multiple_classes_with_same_name_raises_error():
    # Import a class with the same name
    from .foo_serializer import FooSerializer as FooSerializer1  # noqa: F401

    class MySchema(Schema):
        foo = fields.Nested("FooSerializer")

    # Using a nested field with the class name fails because there are
    # two defined classes with the same name
    sch = MySchema()
    msg = "Multiple classes with name {!r} were found.".format("FooSerializer")
    with pytest.raises(RegistryError, match=msg):
        sch.dump({"foo": {"_id": 1}})


def test_multiple_classes_with_all():
    # Import a class with the same name
    from .foo_serializer import FooSerializer as FooSerializer1  # noqa: F401

    classes = class_registry.get_class("FooSerializer", all=True)
    assert len(classes) == 2


def test_can_use_full_module_path_to_class():
    from .foo_serializer import FooSerializer as FooSerializer1  # noqa: F401

    # Using full paths is ok

    class Schema1(Schema):
        foo = fields.Nested("tests.foo_serializer.FooSerializer")

    sch = Schema1()

    # Note: The arguments here don't matter. What matters is that no
    # error is raised
    assert sch.dump({"foo": {"_id": 42}})

    class Schema2(Schema):
        foo = fields.Nested("tests.test_registry.FooSerializer")

    sch2 = Schema2()
    assert sch2.dump({"foo": {"_id": 42}})


# === tests/test_utils.py ===
from __future__ import annotations

import datetime as dt
from copy import copy, deepcopy
from typing import NamedTuple

import pytest

from marshmallow import Schema, fields, utils


def test_missing_singleton_copy():
    assert copy(utils.missing) is utils.missing
    assert deepcopy(utils.missing) is utils.missing


class PointNT(NamedTuple):
    x: int | None
    y: int | None


class PointClass:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class PointDict(dict):
    def __init__(self, x, y):
        super().__init__({"x": x})
        self.y = y


@pytest.mark.parametrize(
    "obj", [PointNT(24, 42), PointClass(24, 42), PointDict(24, 42), {"x": 24, "y": 42}]
)
def test_get_value_from_object(obj):
    assert utils.get_value(obj, "x") == 24
    assert utils.get_value(obj, "y") == 42


def test_get_value_from_namedtuple_with_default():
    p = PointNT(x=42, y=None)
    # Default is only returned if key is not found
    assert utils.get_value(p, "z", default=123) == 123
    # since 'y' is an attribute, None is returned instead of the default
    assert utils.get_value(p, "y", default=123) is None


class Triangle:
    def __init__(self, p1, p2, p3):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.points = [p1, p2, p3]


def test_get_value_for_nested_object():
    tri = Triangle(p1=PointClass(1, 2), p2=PointNT(3, 4), p3={"x": 5, "y": 6})
    assert utils.get_value(tri, "p1.x") == 1
    assert utils.get_value(tri, "p2.x") == 3
    assert utils.get_value(tri, "p3.x") == 5


# regression test for https://github.com/marshmallow-code/marshmallow/issues/62
def test_get_value_from_dict():
    d = dict(items=["foo", "bar"], keys=["baz", "quux"])
    assert utils.get_value(d, "items") == ["foo", "bar"]
    assert utils.get_value(d, "keys") == ["baz", "quux"]


def test_get_value():
    lst = [1, 2, 3]
    assert utils.get_value(lst, 1) == 2

    class MyInt(int):
        pass

    assert utils.get_value(lst, MyInt(1)) == 2


def test_set_value():
    d: dict[str, int | dict] = {}
    utils.set_value(d, "foo", 42)
    assert d == {"foo": 42}

    d = {}
    utils.set_value(d, "foo.bar", 42)
    assert d == {"foo": {"bar": 42}}

    d = {"foo": {}}
    utils.set_value(d, "foo.bar", 42)
    assert d == {"foo": {"bar": 42}}

    d = {"foo": 42}
    with pytest.raises(ValueError):
        utils.set_value(d, "foo.bar", 42)


def test_is_collection():
    assert utils.is_collection([1, "foo", {}]) is True
    assert utils.is_collection(("foo", 2.3)) is True
    assert utils.is_collection({"foo": "bar"}) is False


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1676386740, dt.datetime(2023, 2, 14, 14, 59, 00)),
        (1676386740.58, dt.datetime(2023, 2, 14, 14, 59, 00, 580000)),
    ],
)
def test_from_timestamp(value, expected):
    result = utils.from_timestamp(value)
    assert type(result) is dt.datetime
    assert result == expected


def test_from_timestamp_with_negative_value():
    value = -10
    with pytest.raises(ValueError, match=r"Not a valid POSIX timestamp"):
        utils.from_timestamp(value)


def test_from_timestamp_with_overflow_value():
    value = 9223372036854775
    with pytest.raises(ValueError, match="out of range"):
        utils.from_timestamp(value)


# Regression test for https://github.com/marshmallow-code/marshmallow/issues/540
def test_function_field_using_type_annotation():
    def get_split_words(value: str):
        return value.split(";")

    class MySchema(Schema):
        friends = fields.Function(deserialize=get_split_words)

    data = {"friends": "Clark;Alfred;Robin"}
    result = MySchema().load(data)
    assert result == {"friends": ["Clark", "Alfred", "Robin"]}


# === tests/conftest.py ===
"""Pytest fixtures that are available in all test modules."""

import pytest

from tests.base import Blog, User, UserSchema


@pytest.fixture
def user():
    return User(name="Monty", age=42.3, homepage="http://monty.python.org/")


@pytest.fixture
def blog(user):
    col1 = User(name="Mick", age=123)
    col2 = User(name="Keith", age=456)
    return Blog(
        "Monty's blog",
        user=user,
        categories=["humor", "violence"],
        collaborators=[col1, col2],
    )


@pytest.fixture
def serialized_user(user):
    return UserSchema().dump(user)


# === tests/test_decorators.py ===
import pytest

from marshmallow import (
    EXCLUDE,
    INCLUDE,
    RAISE,
    Schema,
    ValidationError,
    fields,
    post_dump,
    post_load,
    pre_dump,
    pre_load,
    validate,
    validates,
    validates_schema,
)
from tests.base import predicate


@pytest.mark.parametrize("partial_val", (True, False))
def test_decorated_processors(partial_val):
    class ExampleSchema(Schema):
        """Includes different ways to invoke decorators and set up methods"""

        TAG = "TAG"

        value = fields.Integer(as_string=True)

        # Implicit default raw, pre dump, static method.
        @pre_dump
        def increment_value(self, item, **kwargs):
            assert "many" in kwargs
            item["value"] += 1
            return item

        # Implicit default raw, post dump, class method.
        @post_dump
        def add_tag(self, item, **kwargs):
            assert "many" in kwargs
            item["value"] = self.TAG + item["value"]
            return item

        # Explicitly raw, post dump, instance method.
        @post_dump(pass_collection=True)
        def add_envelope(self, data, many, **kwargs):
            key = self.get_envelope_key(many)
            return {key: data}

        # Explicitly raw, pre load, instance method.
        @pre_load(pass_collection=True)
        def remove_envelope(self, data, many, partial, **kwargs):
            assert partial is partial_val
            key = self.get_envelope_key(many)
            return data[key]

        @staticmethod
        def get_envelope_key(many):
            return "data" if many else "datum"

        # Explicitly not raw, pre load, instance method.
        @pre_load(pass_collection=False)
        def remove_tag(self, item, partial, **kwargs):
            assert partial is partial_val
            assert "many" in kwargs
            item["value"] = item["value"][len(self.TAG) :]
            return item

        # Explicit default raw, post load, instance method.
        @post_load()
        def decrement_value(self, item, partial, **kwargs):
            assert partial is partial_val
            assert "many" in kwargs
            item["value"] -= 1
            return item

    schema = ExampleSchema(partial=partial_val)

    # Need to re-create these because the processors will modify in place.
    def make_item():
        return {"value": 3}

    def make_items():
        return [make_item(), {"value": 5}]

    item_dumped = schema.dump(make_item())
    assert item_dumped == {"datum": {"value": "TAG4"}}
    item_loaded = schema.load(item_dumped)
    assert item_loaded == make_item()

    items_dumped = schema.dump(make_items(), many=True)
    assert items_dumped == {"data": [{"value": "TAG4"}, {"value": "TAG6"}]}
    items_loaded = schema.load(items_dumped, many=True)
    assert items_loaded == make_items()


# Regression test for https://github.com/marshmallow-code/marshmallow/issues/347
@pytest.mark.parametrize("unknown", (EXCLUDE, INCLUDE, RAISE))
def test_decorated_processor_returning_none(unknown):
    class PostSchema(Schema):
        value = fields.Integer()

        @post_load
        def load_none(self, item, **kwargs):
            return None

        @post_dump
        def dump_none(self, item, **kwargs):
            return None

    class PreSchema(Schema):
        value = fields.Integer()

        @pre_load
        def load_none(self, item, **kwargs):
            return None

        @pre_dump
        def dump_none(self, item, **kwargs):
            return None

    schema = PostSchema(unknown=unknown)
    assert schema.dump({"value": 3}) is None
    assert schema.load({"value": 3}) is None
    pre_schema = PreSchema(unknown=unknown)
    assert pre_schema.dump({"value": 3}) == {}
    with pytest.raises(ValidationError) as excinfo:
        pre_schema.load({"value": 3})
    assert excinfo.value.messages == {"_schema": ["Invalid input type."]}


class TestPassOriginal:
    def test_pass_original_single(self):
        class MySchema(Schema):
            foo = fields.Raw()

            @post_load(pass_original=True)
            def post_load(self, data, original_data, **kwargs):
                ret = data.copy()
                ret["_post_load"] = original_data["sentinel"]
                return ret

            @post_dump(pass_original=True)
            def post_dump(self, data, obj, **kwargs):
                ret = data.copy()
                ret["_post_dump"] = obj["sentinel"]
                return ret

        schema = MySchema(unknown=EXCLUDE)
        datum = {"foo": 42, "sentinel": 24}
        item_loaded = schema.load(datum)
        assert item_loaded["foo"] == 42
        assert item_loaded["_post_load"] == 24

        item_dumped = schema.dump(datum)

        assert item_dumped["foo"] == 42
        assert item_dumped["_post_dump"] == 24

    def test_pass_original_many(self):
        class MySchema(Schema):
            foo = fields.Raw()

            @post_load(pass_collection=True, pass_original=True)
            def post_load(self, data, original, many, **kwargs):
                if many:
                    ret = []
                    for item, orig_item in zip(data, original):
                        item["_post_load"] = orig_item["sentinel"]
                        ret.append(item)
                else:
                    ret = data.copy()
                    ret["_post_load"] = original["sentinel"]
                return ret

            @post_dump(pass_collection=True, pass_original=True)
            def post_dump(self, data, original, many, **kwargs):
                if many:
                    ret = []
                    for item, orig_item in zip(data, original):
                        item["_post_dump"] = orig_item["sentinel"]
                        ret.append(item)
                else:
                    ret = data.copy()
                    ret["_post_dump"] = original["sentinel"]
                return ret

        schema = MySchema(unknown=EXCLUDE)
        data = [{"foo": 42, "sentinel": 24}, {"foo": 424, "sentinel": 242}]
        items_loaded = schema.load(data, many=True)
        assert items_loaded == [
            {"foo": 42, "_post_load": 24},
            {"foo": 424, "_post_load": 242},
        ]
        test_values = [e["_post_load"] for e in items_loaded]
        assert test_values == [24, 242]

        items_dumped = schema.dump(data, many=True)
        assert items_dumped == [
            {"foo": 42, "_post_dump": 24},
            {"foo": 424, "_post_dump": 242},
        ]

        # Also check load/dump of single item

        datum = {"foo": 42, "sentinel": 24}
        item_loaded = schema.load(datum, many=False)
        assert item_loaded == {"foo": 42, "_post_load": 24}

        item_dumped = schema.dump(datum, many=False)
        assert item_dumped == {"foo": 42, "_post_dump": 24}


def test_decorated_processor_inheritance():
    class ParentSchema(Schema):
        @post_dump
        def inherited(self, item, **kwargs):
            item["inherited"] = "inherited"
            return item

        @post_dump
        def overridden(self, item, **kwargs):
            item["overridden"] = "base"
            return item

        @post_dump
        def deleted(self, item, **kwargs):
            item["deleted"] = "retained"
            return item

    class ChildSchema(ParentSchema):
        @post_dump
        def overridden(self, item, **kwargs):
            item["overridden"] = "overridden"
            return item

        deleted = None

    parent_dumped = ParentSchema().dump({})
    assert parent_dumped == {
        "inherited": "inherited",
        "overridden": "base",
        "deleted": "retained",
    }

    child_dumped = ChildSchema().dump({})
    assert child_dumped == {"inherited": "inherited", "overridden": "overridden"}


class ValidatesSchema(Schema):
    foo = fields.Int()

    @validates("foo")
    def validate_foo(self, value, **kwargs):
        if value != 42:
            raise ValidationError("The answer to life the universe and everything.")


class TestValidatesDecorator:
    def test_validates(self):
        class VSchema(Schema):
            s = fields.String()

            @validates("s")
            def validate_string(self, data, **kwargs):
                raise ValidationError("nope")

        with pytest.raises(ValidationError) as excinfo:
            VSchema().load({"s": "bar"})

        assert excinfo.value.messages == {"s": ["nope"]}

    # Regression test for https://github.com/marshmallow-code/marshmallow/issues/350
    def test_validates_with_attribute(self):
        class S1(Schema):
            s = fields.String(attribute="string_name")

            @validates("s")
            def validate_string(self, data, **kwargs):
                raise ValidationError("nope")

        with pytest.raises(ValidationError) as excinfo:
            S1().load({"s": "foo"})
        assert excinfo.value.messages == {"s": ["nope"]}

        with pytest.raises(ValidationError):
            S1(many=True).load([{"s": "foo"}])

    def test_validates_decorator(self):
        schema = ValidatesSchema()

        errors = schema.validate({"foo": 41})
        assert "foo" in errors
        assert errors["foo"][0] == "The answer to life the universe and everything."

        errors = schema.validate({"foo": 42})
        assert errors == {}

        errors = schema.validate([{"foo": 42}, {"foo": 43}], many=True)
        assert "foo" in errors[1]
        assert len(errors[1]["foo"]) == 1
        assert errors[1]["foo"][0] == "The answer to life the universe and everything."

        errors = schema.validate([{"foo": 42}, {"foo": 42}], many=True)
        assert errors == {}

        errors = schema.validate({})
        assert errors == {}

        with pytest.raises(ValidationError) as excinfo:
            schema.load({"foo": 41})
        assert excinfo.value.messages
        result = excinfo.value.valid_data
        assert result == {}

        with pytest.raises(ValidationError) as excinfo:
            schema.load([{"foo": 42}, {"foo": 43}], many=True)
        error_messages = excinfo.value.messages
        result = excinfo.value.valid_data
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == {"foo": 42}
        assert result[1] == {}
        assert 1 in error_messages
        assert "foo" in error_messages[1]
        assert error_messages[1]["foo"] == [
            "The answer to life the universe and everything."
        ]

    def test_field_not_present(self):
        class BadSchema(ValidatesSchema):
            @validates("bar")
            def validate_bar(self, value, **kwargs):
                raise ValidationError("Never raised.")

        schema = BadSchema()

        with pytest.raises(ValueError, match='"bar" field does not exist.'):
            schema.validate({"foo": 42})

    def test_precedence(self):
        class Schema2(ValidatesSchema):
            foo = fields.Int(validate=predicate(lambda n: n != 42))
            bar = fields.Int(validate=validate.Equal(1))

            @validates("bar")
            def validate_bar(self, value, **kwargs):
                if value != 2:
                    raise ValidationError("Must be 2")

        schema = Schema2()

        errors = schema.validate({"foo": 42})
        assert "foo" in errors
        assert len(errors["foo"]) == 1
        assert "Invalid value." in errors["foo"][0]

        errors = schema.validate({"bar": 3})
        assert "bar" in errors
        assert len(errors["bar"]) == 1
        assert "Must be equal to 1." in errors["bar"][0]

        errors = schema.validate({"bar": 1})
        assert "bar" in errors
        assert len(errors["bar"]) == 1
        assert errors["bar"][0] == "Must be 2"

    # Regression test for https://github.com/marshmallow-code/marshmallow/issues/748
    def test_validates_with_data_key(self):
        class BadSchema(Schema):
            foo = fields.String(data_key="foo-name")

            @validates("foo")
            def validate_string(self, data, **kwargs):
                raise ValidationError("nope")

        schema = BadSchema()
        errors = schema.validate({"foo-name": "data"})
        assert "foo-name" in errors
        assert errors["foo-name"] == ["nope"]

        schema = BadSchema()
        errors = schema.validate(
            [{"foo-name": "data"}, {"foo-name": "data2"}], many=True
        )
        assert errors == {0: {"foo-name": ["nope"]}, 1: {"foo-name": ["nope"]}}

    def test_validates_accepts_multiple_fields(self):
        class BadSchema(Schema):
            foo = fields.String()
            bar = fields.String(data_key="Bar")

            @validates("foo", "bar")
            def validate_string(self, data: str, data_key: str):
                raise ValidationError(f"'{data}' is invalid for {data_key}.")

        schema = BadSchema()
        with pytest.raises(ValidationError) as excinfo:
            schema.load({"foo": "data", "Bar": "data2"})
        assert excinfo.value.messages == {
            "foo": ["'data' is invalid for foo."],
            "Bar": ["'data2' is invalid for Bar."],
        }


class TestValidatesSchemaDecorator:
    def test_validator_nested_many_invalid_data(self):
        class NestedSchema(Schema):
            foo = fields.Int(required=True)

        class MySchema(Schema):
            nested = fields.Nested(NestedSchema, required=True, many=True)

        schema = MySchema()
        errors = schema.validate({"nested": [1]})
        assert errors
        assert "nested" in errors
        assert 0 in errors["nested"]
        assert errors["nested"][0] == {"_schema": ["Invalid input type."]}

    def test_validator_nested_many_schema_error(self):
        class NestedSchema(Schema):
            foo = fields.Int(required=True)

            @validates_schema
            def validate_schema(self, data, **kwargs):
                raise ValidationError("This will never work.")

        class MySchema(Schema):
            nested = fields.Nested(NestedSchema, required=True, many=True)

        schema = MySchema()
        errors = schema.validate({"nested": [{"foo": 1}]})
        assert errors
        assert "nested" in errors
        assert 0 in errors["nested"]
        assert errors["nested"][0] == {"_schema": ["This will never work."]}

    def test_validator_nested_many_field_error(self):
        class NestedSchema(Schema):
            foo = fields.Int(required=True)

            @validates_schema
            def validate_schema(self, data, **kwargs):
                raise ValidationError("This will never work.", "foo")

        class MySchema(Schema):
            nested = fields.Nested(NestedSchema, required=True, many=True)

        schema = MySchema()
        errors = schema.validate({"nested": [{"foo": 1}]})
        assert errors
        assert "nested" in errors
        assert 0 in errors["nested"]
        assert errors["nested"][0] == {"foo": ["This will never work."]}

    @pytest.mark.parametrize("data", ([{"foo": 1, "bar": 2}],))
    @pytest.mark.parametrize(
        ("pass_collection", "expected_data", "expected_original_data"),
        (
            [True, [{"foo": 1}], [{"foo": 1, "bar": 2}]],
            [False, {"foo": 1}, {"foo": 1, "bar": 2}],
        ),
    )
    def test_validator_nested_many_pass_original_and_pass_collection(
        self, pass_collection, data, expected_data, expected_original_data
    ):
        class NestedSchema(Schema):
            foo = fields.Int(required=True)

            @validates_schema(pass_collection=pass_collection, pass_original=True)
            def validate_schema(self, data, original_data, many, **kwargs):
                assert data == expected_data
                assert original_data == expected_original_data
                assert many is True
                raise ValidationError("Method called")

        class MySchema(Schema):
            nested = fields.Nested(
                NestedSchema, required=True, many=True, unknown=EXCLUDE
            )

        schema = MySchema()
        errors = schema.validate({"nested": data})
        error = errors["nested"] if pass_collection else errors["nested"][0]
        assert error["_schema"][0] == "Method called"

    def test_decorated_validators(self):
        class MySchema(Schema):
            foo = fields.Int()
            bar = fields.Int()

            @validates_schema
            def validate_schema(self, data, **kwargs):
                if data["foo"] <= 3:
                    raise ValidationError("Must be greater than 3")

            @validates_schema(pass_collection=True)
            def validate_raw(self, data, many, **kwargs):
                if many:
                    assert type(data) is list
                    if len(data) < 2:
                        raise ValidationError("Must provide at least 2 items")

            @validates_schema
            def validate_bar(self, data, **kwargs):
                if "bar" in data and data["bar"] < 0:
                    raise ValidationError("bar must not be negative", "bar")

        schema = MySchema()
        errors = schema.validate({"foo": 3})
        assert "_schema" in errors
        assert errors["_schema"][0] == "Must be greater than 3"

        errors = schema.validate([{"foo": 4}], many=True)
        assert "_schema" in errors
        assert len(errors["_schema"]) == 1
        assert errors["_schema"][0] == "Must provide at least 2 items"

        errors = schema.validate({"foo": 4, "bar": -1})
        assert "bar" in errors
        assert len(errors["bar"]) == 1
        assert errors["bar"][0] == "bar must not be negative"

    def test_multiple_validators(self):
        class MySchema(Schema):
            foo = fields.Int()
            bar = fields.Int()

            @validates_schema
            def validate_schema(self, data, **kwargs):
                if data["foo"] <= 3:
                    raise ValidationError("Must be greater than 3")

            @validates_schema
            def validate_bar(self, data, **kwargs):
                if "bar" in data and data["bar"] < 0:
                    raise ValidationError("bar must not be negative")

        schema = MySchema()
        errors = schema.validate({"foo": 3, "bar": -1})
        assert type(errors) is dict
        assert "_schema" in errors
        assert len(errors["_schema"]) == 2
        assert "Must be greater than 3" in errors["_schema"]
        assert "bar must not be negative" in errors["_schema"]

        errors = schema.validate([{"foo": 3, "bar": -1}, {"foo": 3}], many=True)
        assert type(errors) is dict
        assert "_schema" in errors[0]
        assert len(errors[0]["_schema"]) == 2
        assert "Must be greater than 3" in errors[0]["_schema"]
        assert "bar must not be negative" in errors[0]["_schema"]
        assert len(errors[1]["_schema"]) == 1
        assert "Must be greater than 3" in errors[0]["_schema"]

    def test_multiple_validators_merge_dict_errors(self):
        class NestedSchema(Schema):
            foo = fields.Int()
            bar = fields.Int()

        class MySchema(Schema):
            nested = fields.Nested(NestedSchema)

            @validates_schema
            def validate_nested_foo(self, data, **kwargs):
                raise ValidationError({"nested": {"foo": ["Invalid foo"]}})

            @validates_schema
            def validate_nested_bar_1(self, data, **kwargs):
                raise ValidationError({"nested": {"bar": ["Invalid bar 1"]}})

            @validates_schema
            def validate_nested_bar_2(self, data, **kwargs):
                raise ValidationError({"nested": {"bar": ["Invalid bar 2"]}})

        with pytest.raises(ValidationError) as excinfo:
            MySchema().load({"nested": {"foo": 1, "bar": 2}})

        assert excinfo.value.messages == {
            "nested": {
                "foo": ["Invalid foo"],
                "bar": ["Invalid bar 1", "Invalid bar 2"],
            }
        }

    def test_passing_original_data(self):
        class MySchema(Schema):
            foo = fields.Int()
            bar = fields.Int()

            @validates_schema(pass_original=True)
            def validate_original(self, data, original_data, partial, **kwargs):
                if isinstance(original_data, dict) and isinstance(
                    original_data["foo"], str
                ):
                    raise ValidationError("foo cannot be a string")

            @validates_schema(pass_collection=True, pass_original=True)
            def validate_original_bar(self, data, original_data, many, **kwargs):
                def check(datum):
                    if isinstance(datum, dict) and isinstance(datum["bar"], str):
                        raise ValidationError("bar cannot be a string")

                if many:
                    for each in original_data:
                        check(each)
                else:
                    check(original_data)

        schema = MySchema()

        errors = schema.validate({"foo": "4", "bar": 12})
        assert errors["_schema"] == ["foo cannot be a string"]

        errors = schema.validate({"foo": 4, "bar": "42"})
        assert errors["_schema"] == ["bar cannot be a string"]

        errors = schema.validate([{"foo": 4, "bar": "42"}], many=True)
        assert errors["_schema"] == ["bar cannot be a string"]

    def test_allow_reporting_field_errors_in_schema_validator(self):
        class NestedSchema(Schema):
            baz = fields.Int(required=True)

        class MySchema(Schema):
            foo = fields.Int(required=True)
            bar = fields.Nested(NestedSchema, required=True)
            bam = fields.Int(required=True)

            @validates_schema(skip_on_field_errors=True)
            def consistency_validation(self, data, **kwargs):
                errors: dict[str, str | dict] = {}
                if data["bar"]["baz"] != data["foo"]:
                    errors["bar"] = {"baz": "Non-matching value"}
                if data["bam"] > data["foo"]:
                    errors["bam"] = "Value should be less than foo"
                if errors:
                    raise ValidationError(errors)

        schema = MySchema()
        errors = schema.validate({"foo": 2, "bar": {"baz": 5}, "bam": 6})
        assert errors["bar"]["baz"] == "Non-matching value"
        assert errors["bam"] == "Value should be less than foo"

    # https://github.com/marshmallow-code/marshmallow/issues/273
    def test_allow_arbitrary_field_names_in_error(self):
        class MySchema(Schema):
            @validates_schema
            def validator(self, data, **kwargs):
                raise ValidationError("Error message", "arbitrary_key")

        errors = MySchema().validate({})
        assert errors["arbitrary_key"] == ["Error message"]

    def test_skip_on_field_errors(self):
        class MySchema(Schema):
            foo = fields.Int(required=True, validate=validate.Equal(3))
            bar = fields.Int(required=True)

            @validates_schema(skip_on_field_errors=True)
            def validate_schema(self, data, **kwargs):
                if data["foo"] != data["bar"]:
                    raise ValidationError("Foo and bar must be equal.")

            @validates_schema(skip_on_field_errors=True, pass_collection=True)
            def validate_many(self, data, many, **kwargs):
                if many:
                    assert type(data) is list
                    if len(data) < 2:
                        raise ValidationError("Must provide at least 2 items")

        schema = MySchema()
        # check that schema errors still occur with no field errors
        errors = schema.validate({"foo": 3, "bar": 4})
        assert "_schema" in errors
        assert errors["_schema"][0] == "Foo and bar must be equal."

        errors = schema.validate([{"foo": 3, "bar": 3}], many=True)
        assert "_schema" in errors
        assert errors["_schema"][0] == "Must provide at least 2 items"

        # check that schema errors don't occur when field errors do
        errors = schema.validate({"foo": 3, "bar": "not an int"})
        assert "bar" in errors
        assert "_schema" not in errors

        errors = schema.validate({"foo": 2, "bar": 2})
        assert "foo" in errors
        assert "_schema" not in errors

        errors = schema.validate([{"foo": 3, "bar": "not an int"}], many=True)
        assert "bar" in errors[0]
        assert "_schema" not in errors

    # https://github.com/marshmallow-code/marshmallow/issues/2170
    def test_data_key_is_used_in_errors_dict(self):
        class MySchema(Schema):
            foo = fields.Int(data_key="fooKey")

            @validates("foo")
            def validate_foo(self, value, **kwargs):
                raise ValidationError("from validates")

            @validates_schema(skip_on_field_errors=False)
            def validate_schema(self, data, **kwargs):
                raise ValidationError("from validates_schema str", field_name="foo")

            @validates_schema(skip_on_field_errors=False)
            def validate_schema2(self, data, **kwargs):
                raise ValidationError({"fooKey": "from validates_schema dict"})

        with pytest.raises(ValidationError) as excinfo:
            MySchema().load({"fooKey": 42})
        exc = excinfo.value
        assert exc.messages == {
            "fooKey": [
                "from validates",
                "from validates_schema str",
                "from validates_schema dict",
            ]
        }


def test_decorator_error_handling():
    class ExampleSchema(Schema):
        foo = fields.Int()
        bar = fields.Int()

        @pre_load()
        def pre_load_error1(self, item, **kwargs):
            if item["foo"] != 0:
                return item
            errors = {"foo": ["preloadmsg1"], "bar": ["preloadmsg2", "preloadmsg3"]}
            raise ValidationError(errors)

        @pre_load()
        def pre_load_error2(self, item, **kwargs):
            if item["foo"] != 4:
                return item
            raise ValidationError("preloadmsg1", "foo")

        @pre_load()
        def pre_load_error3(self, item, **kwargs):
            if item["foo"] != 8:
                return item
            raise ValidationError("preloadmsg1")

        @post_load()
        def post_load_error1(self, item, **kwargs):
            if item["foo"] != 1:
                return item
            errors = {"foo": ["postloadmsg1"], "bar": ["postloadmsg2", "postloadmsg3"]}
            raise ValidationError(errors)

        @post_load()
        def post_load_error2(self, item, **kwargs):
            if item["foo"] != 5:
                return item
            raise ValidationError("postloadmsg1", "foo")

    def make_item(foo, bar):
        data = schema.load({"foo": foo, "bar": bar})
        assert data is not None
        return data

    schema = ExampleSchema()
    with pytest.raises(ValidationError) as excinfo:
        schema.load({"foo": 0, "bar": 1})
    errors = excinfo.value.messages
    assert "foo" in errors
    assert len(errors["foo"]) == 1
    assert errors["foo"][0] == "preloadmsg1"
    assert "bar" in errors
    assert len(errors["bar"]) == 2
    assert "preloadmsg2" in errors["bar"]
    assert "preloadmsg3" in errors["bar"]
    with pytest.raises(ValidationError) as excinfo:
        schema.load({"foo": 1, "bar": 1})
    errors = excinfo.value.messages
    assert "foo" in errors
    assert len(errors["foo"]) == 1
    assert errors["foo"][0] == "postloadmsg1"
    assert "bar" in errors
    assert len(errors["bar"]) == 2
    assert "postloadmsg2" in errors["bar"]
    assert "postloadmsg3" in errors["bar"]
    with pytest.raises(ValidationError) as excinfo:
        schema.load({"foo": 4, "bar": 1})
    errors = excinfo.value.messages
    assert len(errors) == 1
    assert "foo" in errors
    assert len(errors["foo"]) == 1
    assert errors["foo"][0] == "preloadmsg1"
    with pytest.raises(ValidationError) as excinfo:
        schema.load({"foo": 5, "bar": 1})
    errors = excinfo.value.messages
    assert len(errors) == 1
    assert "foo" in errors
    assert len(errors["foo"]) == 1
    assert errors["foo"][0] == "postloadmsg1"
    with pytest.raises(ValidationError) as excinfo:
        schema.load({"foo": 8, "bar": 1})
    errors = excinfo.value.messages
    assert len(errors) == 1
    assert "_schema" in errors
    assert len(errors["_schema"]) == 1
    assert errors["_schema"][0] == "preloadmsg1"


@pytest.mark.parametrize("decorator", [pre_load, post_load])
def test_decorator_error_handling_with_load(decorator):
    class ExampleSchema(Schema):
        @decorator
        def raise_value_error(self, item, **kwargs):
            raise ValidationError({"foo": "error"})

    schema = ExampleSchema()
    with pytest.raises(ValidationError) as exc:
        schema.load({})
    assert exc.value.messages == {"foo": "error"}
    schema.dump(object())


@pytest.mark.parametrize("decorator", [pre_load, post_load])
def test_decorator_error_handling_with_load_dict_error(decorator):
    class ExampleSchema(Schema):
        @decorator
        def raise_value_error(self, item, **kwargs):
            raise ValidationError({"foo": "error"}, "nested_field")

    schema = ExampleSchema()
    with pytest.raises(ValidationError) as exc:
        schema.load({})
    assert exc.value.messages == {"nested_field": {"foo": "error"}}
    schema.dump(object())


@pytest.mark.parametrize("decorator", [pre_dump, post_dump])
def test_decorator_error_handling_with_dump(decorator):
    class ExampleSchema(Schema):
        @decorator
        def raise_value_error(self, item, **kwargs):
            raise ValidationError({"foo": "error"})

    schema = ExampleSchema()
    with pytest.raises(ValidationError) as exc:
        schema.dump(object())
    assert exc.value.messages == {"foo": "error"}
    schema.load({})


class Nested:
    def __init__(self, foo):
        self.foo = foo


class Example:
    def __init__(self, nested):
        self.nested = nested


example = Example(nested=[Nested(x) for x in range(1)])


@pytest.mark.parametrize(
    ("data", "expected_data", "expected_original_data"),
    ([example, {"foo": 0}, example.nested[0]],),
)
def test_decorator_post_dump_with_nested_original_and_pass_collection(
    data, expected_data, expected_original_data
):
    class NestedSchema(Schema):
        foo = fields.Int(required=True)

        @post_dump(pass_collection=False, pass_original=True)
        def check_pass_original_when_pass_collection_false(
            self, data, original_data, **kwargs
        ):
            assert data == expected_data
            assert original_data == expected_original_data
            return data

        @post_dump(pass_collection=True, pass_original=True)
        def check_pass_original_when_pass_collection_true(
            self, data, original_data, many, **kwargs
        ):
            assert many is True
            assert data == [expected_data]
            assert original_data == [expected_original_data]
            return data

    class ExampleSchema(Schema):
        nested = fields.Nested(NestedSchema, required=True, many=True)

    schema = ExampleSchema()
    assert schema.dump(data) == {"nested": [{"foo": 0}]}


@pytest.mark.parametrize(
    ("data", "expected_data", "expected_original_data"),
    ([{"nested": [{"foo": 0}]}, {"foo": 0}, {"foo": 0}],),
)
def test_decorator_post_load_with_nested_original_and_pass_collection(
    data, expected_data, expected_original_data
):
    class NestedSchema(Schema):
        foo = fields.Int(required=True)

        @post_load(pass_collection=False, pass_original=True)
        def check_pass_original_when_pass_collection_false(
            self, data, original_data, **kwargs
        ):
            assert data == expected_data
            assert original_data == expected_original_data
            return data

        @post_load(pass_collection=True, pass_original=True)
        def check_pass_original_when_pass_collection_true(
            self, data, original_data, many, **kwargs
        ):
            assert many is True
            assert data == [expected_data]
            assert original_data == [expected_original_data]
            return data

    class ExampleSchema(Schema):
        nested = fields.Nested(NestedSchema, required=True, many=True)

    schema = ExampleSchema()
    assert schema.load(data) == data


@pytest.mark.parametrize("usage_location", ["meta", "init", "load"])
@pytest.mark.parametrize("unknown_val", (EXCLUDE, INCLUDE))
def test_load_processors_receive_unknown(usage_location, unknown_val):
    class ExampleSchema(Schema):
        foo = fields.Int()

        @validates_schema
        def check_unknown_validates(self, data, unknown, **kwargs):
            assert unknown == unknown_val

        @pre_load
        def check_unknown_pre(self, data, unknown, **kwargs):
            assert unknown == unknown_val
            return data

        @post_load
        def check_unknown_post(self, data, unknown, **kwargs):
            assert unknown == unknown_val
            return data

    if usage_location == "meta":

        class ExampleSchemaChild(ExampleSchema):
            class Meta:
                unknown = unknown_val

        ExampleSchemaChild().load({"foo": 42})
    if usage_location == "init":
        ExampleSchema(unknown=unknown_val).load({"foo": 42})
    else:
        ExampleSchema().load({"foo": 42}, unknown=unknown_val)


# https://github.com/marshmallow-code/marshmallow/issues/1755
def test_post_load_method_that_appends_to_data():
    class MySchema(Schema):
        foo = fields.Int()

        @post_load(pass_collection=True)
        def append_to_data(self, data, **kwargs):
            data.append({"foo": 42})
            return data

        @post_load(pass_collection=False, pass_original=True)
        def noop(self, data, original_data, **kwargs):
            if original_data is None:  # added item
                assert data == {"foo": 42}
            else:
                assert original_data == {"foo": 24}
                assert data == {"foo": 24}
            return data

    schema = MySchema(many=True)
    assert schema.load([{"foo": 24}]) == [{"foo": 24}, {"foo": 42}]


# === tests/test_options.py ===
import datetime as dt

from marshmallow import EXCLUDE, Schema, fields


class UserSchema(Schema):
    name = fields.String(allow_none=True)
    email = fields.Email(allow_none=True)
    age = fields.Integer()
    created = fields.DateTime()
    id = fields.Integer(allow_none=True)
    homepage = fields.Url()
    birthdate = fields.Date()


class ProfileSchema(Schema):
    user = fields.Nested(UserSchema)


class TestFieldOrdering:
    def test_declared_field_order_is_maintained_on_dump(self, user):
        ser = UserSchema()
        data = ser.dump(user)
        keys = list(data)
        assert keys == [
            "name",
            "email",
            "age",
            "created",
            "id",
            "homepage",
            "birthdate",
        ]

    def test_declared_field_order_is_maintained_on_load(self, serialized_user):
        schema = UserSchema(unknown=EXCLUDE)
        data = schema.load(serialized_user)
        keys = list(data)
        assert keys == [
            "name",
            "email",
            "age",
            "created",
            "id",
            "homepage",
            "birthdate",
        ]

    def test_nested_field_order_with_only_arg_is_maintained_on_dump(self, user):
        schema = ProfileSchema()
        data = schema.dump({"user": user})
        user_data = data["user"]
        keys = list(user_data)
        assert keys == [
            "name",
            "email",
            "age",
            "created",
            "id",
            "homepage",
            "birthdate",
        ]

    def test_nested_field_order_with_only_arg_is_maintained_on_load(self):
        schema = ProfileSchema()
        data = schema.load(
            {
                "user": {
                    "name": "Foo",
                    "email": "Foo@bar.com",
                    "age": 42,
                    "created": dt.datetime.now().isoformat(),
                    "id": 123,
                    "homepage": "http://foo.com",
                    "birthdate": dt.datetime.now().date().isoformat(),
                }
            }
        )
        user_data = data["user"]
        keys = list(user_data)
        assert keys == [
            "name",
            "email",
            "age",
            "created",
            "id",
            "homepage",
            "birthdate",
        ]

    def test_nested_field_order_with_exclude_arg_is_maintained(self, user):
        class HasNestedExclude(Schema):
            user = fields.Nested(UserSchema, exclude=("birthdate",))

        ser = HasNestedExclude()
        data = ser.dump({"user": user})
        user_data = data["user"]
        keys = list(user_data)
        assert keys == ["name", "email", "age", "created", "id", "homepage"]


class TestIncludeOption:
    class AddFieldsSchema(Schema):
        name = fields.Str()

        class Meta:
            include = {"from": fields.Str()}

    def test_fields_are_added(self):
        s = self.AddFieldsSchema()
        in_data = {"name": "Steve", "from": "Oskosh"}
        result = s.load({"name": "Steve", "from": "Oskosh"})
        assert result == in_data

    def test_included_fields_ordered_after_declared_fields(self):
        class AddFieldsOrdered(Schema):
            name = fields.Str()
            email = fields.Str()

            class Meta:
                include = {
                    "from": fields.Str(),
                    "in": fields.Str(),
                    "@at": fields.Str(),
                }

        s = AddFieldsOrdered()
        in_data = {
            "name": "Steve",
            "from": "Oskosh",
            "email": "steve@steve.steve",
            "in": "VA",
            "@at": "Charlottesville",
        }
        # declared fields, then "included" fields
        expected_fields = ["name", "email", "from", "in", "@at"]
        assert list(AddFieldsOrdered._declared_fields.keys()) == expected_fields

        result = s.load(in_data)
        assert list(result.keys()) == expected_fields

    def test_added_fields_are_inherited(self):
        class AddFieldsChild(self.AddFieldsSchema):  # type: ignore[name-defined]
            email = fields.Str()

        s = AddFieldsChild()
        assert "email" in s._declared_fields
        assert "from" in s._declared_fields
        assert isinstance(s._declared_fields["from"], fields.Str)


class TestManyOption:
    class ManySchema(Schema):
        foo = fields.Str()

        class Meta:
            many = True

    def test_many_by_default(self):
        test = self.ManySchema()
        assert test.load([{"foo": "bar"}]) == [{"foo": "bar"}]

    def test_explicit_single(self):
        test = self.ManySchema(many=False)
        assert test.load({"foo": "bar"}) == {"foo": "bar"}


# === tests/test_exceptions.py ===
import pytest

from marshmallow.exceptions import ValidationError


class TestValidationError:
    def test_stores_message_in_list(self):
        err = ValidationError("foo")
        assert err.messages == ["foo"]

    def test_can_pass_list_of_messages(self):
        err = ValidationError(["foo", "bar"])
        assert err.messages == ["foo", "bar"]

    def test_stores_dictionaries(self):
        messages = {"user": {"email": ["email is invalid"]}}
        err = ValidationError(messages)
        assert err.messages == messages

    def test_can_store_field_name(self):
        err = ValidationError("invalid email", field_name="email")
        assert err.field_name == "email"

    def test_str(self):
        err = ValidationError("invalid email")
        assert str(err) == "invalid email"

        err2 = ValidationError("invalid email", "email")
        assert str(err2) == "invalid email"

    def test_stores_dictionaries_in_messages_dict(self):
        messages = {"user": {"email": ["email is invalid"]}}
        err = ValidationError(messages)
        assert err.messages_dict == messages

    def test_messages_dict_type_error_on_badval(self):
        err = ValidationError("foo")
        with pytest.raises(TypeError) as excinfo:
            err.messages_dict  # noqa: B018
        assert "cannot access 'messages_dict' when 'messages' is of type list" in str(
            excinfo.value
        )


# === tests/__init__.py ===


# === tests/test_schema.py ===
import datetime as dt
import math
import random
from collections import OrderedDict
from typing import NamedTuple

import pytest
import simplejson as json

from marshmallow import (
    EXCLUDE,
    INCLUDE,
    RAISE,
    Schema,
    class_registry,
    fields,
    validate,
    validates,
    validates_schema,
)
from marshmallow.exceptions import (
    RegistryError,
    StringNotCollectionError,
    ValidationError,
)
from tests.base import (
    Blog,
    BlogOnlySchema,
    BlogSchema,
    BlogSchemaExclude,
    ExtendedUserSchema,
    User,
    UserExcludeSchema,
    UserFloatStringSchema,
    UserIntSchema,
    UserRelativeUrlSchema,
    UserSchema,
    mockjson,
)

random.seed(1)


def test_serializing_basic_object(user):
    s = UserSchema()
    data = s.dump(user)
    assert data["name"] == user.name
    assert math.isclose(data["age"], 42.3)
    assert data["registered"]


def test_serializer_dump(user):
    s = UserSchema()
    result = s.dump(user)
    assert result["name"] == user.name


def test_load_resets_errors():
    class MySchema(Schema):
        email = fields.Email()

    schema = MySchema()
    with pytest.raises(ValidationError) as excinfo:
        schema.load({"name": "Joe", "email": "notvalid"})
    errors = excinfo.value.messages
    assert len(errors["email"]) == 1
    assert "Not a valid email address." in errors["email"][0]
    with pytest.raises(ValidationError) as excinfo:
        schema.load({"name": "Joe", "email": "__invalid"})
    errors = excinfo.value.messages
    assert len(errors["email"]) == 1
    assert "Not a valid email address." in errors["email"][0]


def test_load_validation_error_stores_input_data_and_valid_data():
    def validator(val):
        raise ValidationError("oops")

    class MySchema(Schema):
        always_valid = fields.DateTime()
        always_invalid = fields.Raw(validate=[validator])

    schema = MySchema()
    input_data = {
        "always_valid": dt.datetime.now(dt.timezone.utc).isoformat(),
        "always_invalid": 24,
    }
    with pytest.raises(ValidationError) as excinfo:
        schema.load(input_data)
    err = excinfo.value
    # err.data is the raw input data
    assert err.data == input_data
    assert isinstance(err.valid_data, dict)
    assert "always_valid" in err.valid_data
    # err.valid_data contains valid, deserialized data
    assert isinstance(err.valid_data["always_valid"], dt.datetime)
    # excludes invalid data
    assert "always_invalid" not in err.valid_data


def test_load_resets_error_fields():
    class MySchema(Schema):
        email = fields.Email()
        name = fields.Str()

    schema = MySchema()
    with pytest.raises(ValidationError) as excinfo:
        schema.load({"name": "Joe", "email": "not-valid"})
    exc = excinfo.value
    assert isinstance(exc.messages, dict)
    assert len(exc.messages.keys()) == 1

    with pytest.raises(ValidationError) as excinfo:
        schema.load({"name": 12, "email": "mick@stones.com"})
    exc = excinfo.value


def test_errored_fields_do_not_appear_in_output():
    class MyField(fields.Field[int]):
        # Make sure validation fails during serialization
        def _serialize(self, value, attr, obj, **kwargs):
            raise ValidationError("oops")

    def validator(val):
        raise ValidationError("oops")

    class MySchema(Schema):
        foo = MyField(validate=validator)

    sch = MySchema()
    with pytest.raises(ValidationError) as excinfo:
        sch.load({"foo": 2})
    data, errors = excinfo.value.valid_data, excinfo.value.messages

    assert "foo" in errors
    assert isinstance(data, dict)
    assert "foo" not in data


def test_load_many_stores_error_indices():
    s = UserSchema()
    data = [
        {"name": "Mick", "email": "mick@stones.com"},
        {"name": "Keith", "email": "invalid-email", "homepage": "invalid-homepage"},
    ]
    with pytest.raises(ValidationError) as excinfo:
        s.load(data, many=True)
    errors = excinfo.value.messages
    assert 0 not in errors
    assert 1 in errors
    assert "email" in errors[1]
    assert "homepage" in errors[1]


def test_dump_many():
    s = UserSchema()
    u1, u2 = User("Mick"), User("Keith")
    data = s.dump([u1, u2], many=True)
    assert len(data) == 2
    assert data[0] == s.dump(u1)


def test_multiple_errors_can_be_stored_for_a_given_index():
    class MySchema(Schema):
        foo = fields.Str(validate=validate.Length(min=4))
        bar = fields.Int(validate=validate.Range(min=4))

    sch = MySchema()
    valid = {"foo": "loll", "bar": 42}
    invalid = {"foo": "lol", "bar": 3}
    errors = sch.validate([valid, invalid], many=True)

    assert 1 in errors
    assert len(errors[1]) == 2
    assert "foo" in errors[1]
    assert "bar" in errors[1]


def test_dump_returns_a_dict(user):
    s = UserSchema()
    result = s.dump(user)
    assert type(result) is dict


def test_dumps_returns_a_string(user):
    s = UserSchema()
    result = s.dumps(user)
    assert type(result) is str


def test_dumping_single_object_with_collection_schema(user):
    s = UserSchema(many=True)
    result = s.dump(user, many=False)
    assert type(result) is dict
    assert result == UserSchema().dump(user)


def test_loading_single_object_with_collection_schema():
    s = UserSchema(many=True)
    in_data = {"name": "Mick", "email": "mick@stones.com"}
    result = s.load(in_data, many=False)
    assert type(result) is User
    assert result.name == UserSchema().load(in_data).name


def test_dumps_many():
    s = UserSchema()
    u1, u2 = User("Mick"), User("Keith")
    json_result = s.dumps([u1, u2], many=True)
    data = json.loads(json_result)
    assert len(data) == 2
    assert data[0] == s.dump(u1)


def test_load_returns_an_object():
    s = UserSchema()
    result = s.load({"name": "Monty"})
    assert type(result) is User


def test_load_many():
    s = UserSchema()
    in_data = [{"name": "Mick"}, {"name": "Keith"}]
    result = s.load(in_data, many=True)
    assert type(result) is list
    assert type(result[0]) is User
    assert result[0].name == "Mick"


@pytest.mark.parametrize("val", (None, False, 1, 1.2, object(), [], set(), "lol"))
def test_load_invalid_input_type(val):
    class Sch(Schema):
        name = fields.Str()

    with pytest.raises(ValidationError) as e:
        Sch().load(val)
    assert e.value.messages == {"_schema": ["Invalid input type."]}
    assert e.value.valid_data == {}


# regression test for https://github.com/marshmallow-code/marshmallow/issues/906
@pytest.mark.parametrize("val", (None, False, 1, 1.2, object(), {}, {"1": 2}, "lol"))
def test_load_many_invalid_input_type(val):
    class Sch(Schema):
        name = fields.Str()

    with pytest.raises(ValidationError) as e:
        Sch(many=True).load(val)
    assert e.value.messages == {"_schema": ["Invalid input type."]}
    assert e.value.valid_data == []


@pytest.mark.parametrize("val", ([], tuple()))
def test_load_many_empty_collection(val):
    class Sch(Schema):
        name = fields.Str()

    assert Sch(many=True).load(val) == []


@pytest.mark.parametrize("val", (False, 1, 1.2, object(), {}, {"1": 2}, "lol"))
def test_load_many_in_nested_invalid_input_type(val):
    class Inner(Schema):
        name = fields.String()

    class Outer(Schema):
        list1 = fields.List(fields.Nested(Inner))
        list2 = fields.Nested(Inner, many=True)

    with pytest.raises(ValidationError) as e:
        Outer().load({"list1": val, "list2": val})
    # TODO: Error messages should be identical (#779)
    assert e.value.messages == {
        "list1": ["Not a valid list."],
        "list2": ["Invalid type."],
    }


@pytest.mark.parametrize("val", ([], tuple()))
def test_load_many_in_nested_empty_collection(val):
    class Inner(Schema):
        name = fields.String()

    class Outer(Schema):
        list1 = fields.List(fields.Nested(Inner))
        list2 = fields.Nested(Inner, many=True)

    assert Outer().load({"list1": val, "list2": val}) == {"list1": [], "list2": []}


def test_loads_returns_a_user():
    s = UserSchema()
    result = s.loads(json.dumps({"name": "Monty"}))
    assert type(result) is User


def test_loads_many():
    s = UserSchema()
    in_data = [{"name": "Mick"}, {"name": "Keith"}]
    in_json_data = json.dumps(in_data)
    result = s.loads(in_json_data, many=True)
    assert type(result) is list
    assert result[0].name == "Mick"


def test_loads_deserializes_from_json():
    user_dict = {"name": "Monty", "age": "42.3"}
    user_json = json.dumps(user_dict)
    result = UserSchema().loads(user_json)
    assert isinstance(result, User)
    assert result.name == "Monty"
    assert math.isclose(result.age, 42.3)


def test_serializing_none():
    class MySchema(Schema):
        id = fields.Str(dump_default="no-id")
        num = fields.Int()
        name = fields.Str()

    data = UserSchema().dump(None)
    assert data == {"id": "no-id"}


def test_default_many_symmetry():
    """The dump/load(s) methods should all default to the many value of the schema."""
    s_many = UserSchema(many=True, only=("name",))
    s_single = UserSchema(only=("name",))
    u1, u2 = User("King Arthur"), User("Sir Lancelot")
    s_single.load(s_single.dump(u1))
    s_single.loads(s_single.dumps(u1))
    s_many.load(s_many.dump([u1, u2]))
    s_many.loads(s_many.dumps([u1, u2]))


def test_on_bind_field_hook():
    class MySchema(Schema):
        foo = fields.Str()

        def on_bind_field(self, field_name, field_obj):
            assert field_obj.parent is self
            field_obj.metadata["fname"] = field_name

    schema = MySchema()
    assert schema.fields["foo"].metadata["fname"] == "foo"


def test_nested_on_bind_field_hook():
    class MySchema(Schema):
        class NestedSchema(Schema):
            bar = fields.Str()

            def on_bind_field(self, field_name, field_obj):
                assert field_obj.parent is self
                field_obj.metadata["fname"] = field_name

        foo = fields.Nested(NestedSchema)

    schema = MySchema()
    foo_field = schema.fields["foo"]
    assert isinstance(foo_field, fields.Nested)
    assert foo_field.schema.fields["bar"].metadata["fname"] == "bar"


class TestValidate:
    def test_validate_raises_with_errors_dict(self):
        s = UserSchema()
        errors = s.validate({"email": "bad-email", "name": "Valid Name"})
        assert type(errors) is dict
        assert "email" in errors
        assert "name" not in errors

        valid_data = {"name": "Valid Name", "email": "valid@email.com"}
        errors = s.validate(valid_data)
        assert errors == {}

    def test_validate_many(self):
        s = UserSchema(many=True)
        in_data = [
            {"name": "Valid Name", "email": "validemail@hotmail.com"},
            {"name": "Valid Name2", "email": "invalid"},
        ]
        errors = s.validate(in_data, many=True)
        assert 1 in errors
        assert "email" in errors[1]

    def test_validate_many_doesnt_store_index_if_index_errors_option_is_false(self):
        class NoIndex(Schema):
            email = fields.Email()

            class Meta:
                index_errors = False

        s = NoIndex()
        in_data = [
            {"name": "Valid Name", "email": "validemail@hotmail.com"},
            {"name": "Valid Name2", "email": "invalid"},
        ]
        errors = s.validate(in_data, many=True)
        assert 1 not in errors
        assert "email" in errors

    def test_validate(self):
        s = UserSchema()
        errors = s.validate({"email": "bad-email"})
        assert errors == {"email": ["Not a valid email address."]}

    def test_validate_required(self):
        class MySchema(Schema):
            foo = fields.Raw(required=True)

        s = MySchema()
        errors = s.validate({"bar": 42})
        assert "foo" in errors
        assert "required" in errors["foo"][0]


def test_fields_are_not_copies():
    s = UserSchema()
    s2 = UserSchema()
    assert s.fields is not s2.fields


def test_dumps_returns_json(user):
    ser = UserSchema()
    serialized = ser.dump(user)
    json_data = ser.dumps(user)
    assert type(json_data) is str
    expected = json.dumps(serialized)
    assert json_data == expected


def test_naive_datetime_field(user, serialized_user):
    expected = user.created.isoformat()
    assert serialized_user["created"] == expected


def test_datetime_formatted_field(user, serialized_user):
    result = serialized_user["created_formatted"]
    assert result == user.created.strftime("%Y-%m-%d")


def test_datetime_iso_field(user, serialized_user):
    assert serialized_user["created_iso"] == user.created.isoformat()


def test_tz_datetime_field(user, serialized_user):
    # Datetime is corrected back to GMT
    expected = user.updated.isoformat()
    assert serialized_user["updated"] == expected


def test_class_variable(serialized_user):
    assert serialized_user["species"] == "Homo sapiens"


def test_serialize_many():
    user1 = User(name="Mick", age=123)
    user2 = User(name="Keith", age=456)
    users = [user1, user2]
    serialized = UserSchema(many=True).dump(users)
    assert len(serialized) == 2
    assert serialized[0]["name"] == "Mick"
    assert serialized[1]["name"] == "Keith"


def test_inheriting_schema(user):
    sch = ExtendedUserSchema()
    result = sch.dump(user)
    assert result["name"] == user.name
    user.is_old = True
    result = sch.dump(user)
    assert result["is_old"] is True


def test_custom_field(serialized_user, user):
    assert serialized_user["uppername"] == user.name.upper()


def test_url_field(serialized_user, user):
    assert serialized_user["homepage"] == user.homepage


def test_relative_url_field():
    u = {"name": "John", "homepage": "/foo"}
    UserRelativeUrlSchema().load(u)


def test_stores_invalid_url_error():
    user = {"name": "Steve", "homepage": "www.foo.com"}
    with pytest.raises(ValidationError) as excinfo:
        UserSchema().load(user)
    errors = excinfo.value.messages
    assert "homepage" in errors
    expected = ["Not a valid URL."]
    assert errors["homepage"] == expected


def test_email_field():
    u = User("John", email="john@example.com")
    s = UserSchema().dump(u)
    assert s["email"] == "john@example.com"


def test_stored_invalid_email():
    u = {"name": "John", "email": "johnexample.com"}
    with pytest.raises(ValidationError) as excinfo:
        UserSchema().load(u)
    errors = excinfo.value.messages
    assert "email" in errors
    assert errors["email"][0] == "Not a valid email address."


def test_integer_field():
    u = User("John", age=42.3)
    serialized = UserIntSchema().dump(u)
    assert type(serialized["age"]) is int
    assert serialized["age"] == 42


def test_as_string():
    u = User("John", age=42.3)
    serialized = UserFloatStringSchema().dump(u)
    assert type(serialized["age"]) is str
    assert math.isclose(float(serialized["age"]), 42.3)


def test_method_field(serialized_user):
    assert serialized_user["is_old"] is False
    u = User("Joe", age=81)
    assert UserSchema().dump(u)["is_old"] is True


def test_function_field(serialized_user, user):
    assert serialized_user["lowername"] == user.name.lower()


def test_fields_must_be_declared_as_instances(user):
    with pytest.raises(
        TypeError, match='Field for "name" must be declared as a Field instance'
    ):

        class BadUserSchema(Schema):
            name = fields.String


# regression test
def test_bind_field_does_not_swallow_typeerror():
    class MySchema(Schema):
        name = fields.Str()

        def on_bind_field(self, field_name, field_obj):
            raise TypeError("boom")

    with pytest.raises(TypeError, match="boom"):
        MySchema()


def test_serializing_generator():
    users = [User("Foo"), User("Bar")]
    user_gen = (u for u in users)
    s = UserSchema(many=True).dump(user_gen)
    assert len(s) == 2
    assert s[0] == UserSchema().dump(users[0])


def test_serializing_empty_list_returns_empty_list():
    assert UserSchema(many=True).dump([]) == []


def test_serializing_dict():
    user = {
        "name": "foo",
        "email": "foo@bar.com",
        "age": 42,
        "various_data": {"foo": "bar"},
    }
    data = UserSchema().dump(user)
    assert data["name"] == "foo"
    assert data["age"] == 42
    assert data["various_data"] == {"foo": "bar"}


def test_exclude_in_init(user):
    s = UserSchema(exclude=("age", "homepage")).dump(user)
    assert "homepage" not in s
    assert "age" not in s
    assert "name" in s


def test_only_in_init(user):
    s = UserSchema(only=("name", "age")).dump(user)
    assert "homepage" not in s
    assert "name" in s
    assert "age" in s


def test_invalid_only_param(user):
    with pytest.raises(ValueError):
        UserSchema(only=("_invalid", "name")).dump(user)


def test_can_serialize_uuid(serialized_user, user):
    assert serialized_user["uid"] == str(user.uid)


def test_can_serialize_time(user, serialized_user):
    expected = user.time_registered.isoformat()[:15]
    assert serialized_user["time_registered"] == expected


def test_render_module():
    class UserJSONSchema(Schema):
        name = fields.String()

        class Meta:
            render_module = mockjson

    user = User("Joe")
    s = UserJSONSchema()
    result = s.dumps(user)
    assert result == mockjson.dumps("val")


def test_custom_error_message():
    class ErrorSchema(Schema):
        email = fields.Email(error_messages={"invalid": "Invalid email"})
        homepage = fields.Url(error_messages={"invalid": "Bad homepage."})
        balance = fields.Decimal(error_messages={"invalid": "Bad balance."})

    u = {"email": "joe.net", "homepage": "joe@example.com", "balance": "blah"}
    s = ErrorSchema()
    with pytest.raises(ValidationError) as excinfo:
        s.load(u)
    errors = excinfo.value.messages
    assert "Bad balance." in errors["balance"]
    assert "Bad homepage." in errors["homepage"]
    assert "Invalid email" in errors["email"]


def test_custom_unknown_error_message():
    custom_message = "custom error message."

    class ErrorSchema(Schema):
        error_messages = {"unknown": custom_message}
        name = fields.String()

    s = ErrorSchema()
    u = {"name": "Joe", "age": 13}
    with pytest.raises(ValidationError) as excinfo:
        s.load(u)
    errors = excinfo.value.messages
    assert custom_message in errors["age"]


def test_custom_type_error_message():
    custom_message = "custom error message."

    class ErrorSchema(Schema):
        error_messages = {"type": custom_message}
        name = fields.String()

    s = ErrorSchema()
    u = ["Joe"]
    with pytest.raises(ValidationError) as excinfo:
        s.load(u)  # type: ignore[arg-type]
    errors = excinfo.value.messages
    assert custom_message in errors["_schema"]


def test_custom_type_error_message_with_many():
    custom_message = "custom error message."

    class ErrorSchema(Schema):
        error_messages = {"type": custom_message}
        name = fields.String()

    s = ErrorSchema(many=True)
    u = {"name": "Joe"}
    with pytest.raises(ValidationError) as excinfo:
        s.load(u)
    errors = excinfo.value.messages
    assert custom_message in errors["_schema"]


def test_custom_error_messages_with_inheritance():
    parent_type_message = "parent type error message."
    parent_unknown_message = "parent unknown error message."
    child_type_message = "child type error message."

    class ParentSchema(Schema):
        error_messages = {
            "type": parent_type_message,
            "unknown": parent_unknown_message,
        }
        name = fields.String()

    class ChildSchema(ParentSchema):
        error_messages = {"type": child_type_message}

    unknown_user = {"name": "Eleven", "age": 12}

    parent_schema = ParentSchema()
    with pytest.raises(ValidationError) as excinfo:
        parent_schema.load(unknown_user)
    assert parent_unknown_message in excinfo.value.messages["age"]
    with pytest.raises(ValidationError) as excinfo:
        parent_schema.load(11)  # type: ignore[arg-type]
    assert parent_type_message in excinfo.value.messages["_schema"]

    child_schema = ChildSchema()
    with pytest.raises(ValidationError) as excinfo:
        child_schema.load(unknown_user)
    assert parent_unknown_message in excinfo.value.messages["age"]
    with pytest.raises(ValidationError) as excinfo:
        child_schema.load(11)  # type: ignore[arg-type]
    assert child_type_message in excinfo.value.messages["_schema"]


def test_load_errors_with_many():
    class ErrorSchema(Schema):
        email = fields.Email()

    data = [
        {"email": "bademail"},
        {"email": "goo@email.com"},
        {"email": "anotherbademail"},
    ]

    with pytest.raises(ValidationError) as excinfo:
        ErrorSchema(many=True).load(data)
    errors = excinfo.value.messages
    assert 0 in errors
    assert 2 in errors
    assert "Not a valid email address." in errors[0]["email"][0]
    assert "Not a valid email address." in errors[2]["email"][0]


def test_error_raised_if_fields_option_is_not_list():
    with pytest.raises(ValueError):

        class BadSchema(Schema):
            name = fields.String()

            class Meta:
                fields = "name"


def test_nested_custom_set_in_exclude_reusing_schema():
    class CustomSet:
        # This custom set is to allow the obj check in BaseSchema.__filter_fields
        # to pass, since it'll be a valid instance, and this class overrides
        # getitem method to allow the hasattr check to pass too, which will try
        # to access the first obj index and will simulate a IndexError throwing.
        # e.g. SqlAlchemy.Query is a valid use case for this 'obj'.

        def __getitem__(self, item):
            return [][item]

    class ChildSchema(Schema):
        foo = fields.Raw(required=True)
        bar = fields.Raw()

        class Meta:
            only = ("bar",)

    class ParentSchema(Schema):
        child = fields.Nested(ChildSchema, many=True, exclude=("foo",))

    sch = ParentSchema()
    obj = dict(child=CustomSet())
    sch.dumps(obj)
    data = dict(child=[{"bar": 1}])
    sch.load(data, partial=True)


def test_nested_only():
    class ChildSchema(Schema):
        foo = fields.Raw()
        bar = fields.Raw()
        baz = fields.Raw()

    class ParentSchema(Schema):
        bla = fields.Raw()
        bli = fields.Raw()
        blubb = fields.Nested(ChildSchema)

    sch = ParentSchema(only=("bla", "blubb.foo", "blubb.bar"))
    data = dict(bla=1, bli=2, blubb=dict(foo=42, bar=24, baz=242))
    result = sch.dump(data)
    assert "bla" in result
    assert "blubb" in result
    assert "bli" not in result
    child = result["blubb"]
    assert "foo" in child
    assert "bar" in child
    assert "baz" not in child


def test_nested_only_inheritance():
    class ChildSchema(Schema):
        foo = fields.Raw()
        bar = fields.Raw()
        baz = fields.Raw()

    class ParentSchema(Schema):
        bla = fields.Raw()
        bli = fields.Raw()
        blubb = fields.Nested(ChildSchema, only=("foo", "bar"))

    sch = ParentSchema(only=("blubb.foo", "blubb.baz"))
    data = dict(bla=1, bli=2, blubb=dict(foo=42, bar=24, baz=242))
    result = sch.dump(data)
    assert "bla" not in result
    assert "blubb" in result
    assert "bli" not in result
    child = result["blubb"]
    assert "foo" in child
    assert "bar" not in child
    assert "baz" not in child


def test_nested_only_empty_inheritance():
    class ChildSchema(Schema):
        foo = fields.Raw()
        bar = fields.Raw()
        baz = fields.Raw()

    class ParentSchema(Schema):
        bla = fields.Raw()
        bli = fields.Raw()
        blubb = fields.Nested(ChildSchema, only=("bar",))

    sch = ParentSchema(only=("blubb.foo",))
    data = dict(bla=1, bli=2, blubb=dict(foo=42, bar=24, baz=242))
    result = sch.dump(data)
    assert "bla" not in result
    assert "blubb" in result
    assert "bli" not in result
    child = result["blubb"]
    assert "foo" not in child
    assert "bar" not in child
    assert "baz" not in child


def test_nested_exclude():
    class ChildSchema(Schema):
        foo = fields.Raw()
        bar = fields.Raw()
        baz = fields.Raw()

    class ParentSchema(Schema):
        bla = fields.Raw()
        bli = fields.Raw()
        blubb = fields.Nested(ChildSchema)

    sch = ParentSchema(exclude=("bli", "blubb.baz"))
    data = dict(bla=1, bli=2, blubb=dict(foo=42, bar=24, baz=242))
    result = sch.dump(data)
    assert "bla" in result
    assert "blubb" in result
    assert "bli" not in result
    child = result["blubb"]
    assert "foo" in child
    assert "bar" in child
    assert "baz" not in child


def test_nested_exclude_inheritance():
    class ChildSchema(Schema):
        foo = fields.Raw()
        bar = fields.Raw()
        baz = fields.Raw()

    class ParentSchema(Schema):
        bla = fields.Raw()
        bli = fields.Raw()
        blubb = fields.Nested(ChildSchema, exclude=("baz",))

    sch = ParentSchema(exclude=("blubb.foo",))
    data = dict(bla=1, bli=2, blubb=dict(foo=42, bar=24, baz=242))
    result = sch.dump(data)
    assert "bla" in result
    assert "blubb" in result
    assert "bli" in result
    child = result["blubb"]
    assert "foo" not in child
    assert "bar" in child
    assert "baz" not in child


def test_nested_only_and_exclude():
    class ChildSchema(Schema):
        foo = fields.Raw()
        bar = fields.Raw()
        baz = fields.Raw()

    class ParentSchema(Schema):
        bla = fields.Raw()
        bli = fields.Raw()
        blubb = fields.Nested(ChildSchema)

    sch = ParentSchema(only=("bla", "blubb.foo", "blubb.bar"), exclude=("blubb.foo",))
    data = dict(bla=1, bli=2, blubb=dict(foo=42, bar=24, baz=242))
    result = sch.dump(data)
    assert "bla" in result
    assert "blubb" in result
    assert "bli" not in result
    child = result["blubb"]
    assert "foo" not in child
    assert "bar" in child
    assert "baz" not in child


def test_nested_only_then_exclude_inheritance():
    class ChildSchema(Schema):
        foo = fields.Raw()
        bar = fields.Raw()
        baz = fields.Raw()

    class ParentSchema(Schema):
        bla = fields.Raw()
        bli = fields.Raw()
        blubb = fields.Nested(ChildSchema, only=("foo", "bar"))

    sch = ParentSchema(exclude=("blubb.foo",))
    data = dict(bla=1, bli=2, blubb=dict(foo=42, bar=24, baz=242))
    result = sch.dump(data)
    assert "bla" in result
    assert "blubb" in result
    assert "bli" in result
    child = result["blubb"]
    assert "foo" not in child
    assert "bar" in child
    assert "baz" not in child


def test_nested_exclude_then_only_inheritance():
    class ChildSchema(Schema):
        foo = fields.Raw()
        bar = fields.Raw()
        baz = fields.Raw()

    class ParentSchema(Schema):
        bla = fields.Raw()
        bli = fields.Raw()
        blubb = fields.Nested(ChildSchema, exclude=("foo",))

    sch = ParentSchema(only=("blubb.bar",))
    data = dict(bla=1, bli=2, blubb=dict(foo=42, bar=24, baz=242))
    result = sch.dump(data)
    assert "bla" not in result
    assert "blubb" in result
    assert "bli" not in result
    child = result["blubb"]
    assert "foo" not in child
    assert "bar" in child
    assert "baz" not in child


def test_nested_exclude_and_only_inheritance():
    class ChildSchema(Schema):
        foo = fields.Raw()
        bar = fields.Raw()
        baz = fields.Raw()
        ban = fields.Raw()
        fuu = fields.Raw()

    class ParentSchema(Schema):
        bla = fields.Raw()
        bli = fields.Raw()
        blubb = fields.Nested(
            ChildSchema, only=("foo", "bar", "baz", "ban"), exclude=("foo",)
        )

    sch = ParentSchema(
        only=("blubb.foo", "blubb.bar", "blubb.baz"), exclude=("blubb.baz",)
    )
    data = dict(bla=1, bli=2, blubb=dict(foo=42, bar=24, baz=242))
    result = sch.dump(data)
    assert "bla" not in result
    assert "blubb" in result
    assert "bli" not in result
    child = result["blubb"]
    assert "foo" not in child
    assert "bar" in child
    assert "baz" not in child
    assert "ban" not in child
    assert "fuu" not in child


# https://github.com/marshmallow-code/marshmallow/issues/1160
def test_nested_instance_many():
    class BookSchema(Schema):
        id = fields.Int()
        title = fields.String()

    class UserSchema(Schema):
        id = fields.Int()
        name = fields.String()
        books = fields.Nested(BookSchema(many=True))

    books = [{"id": 1, "title": "First book"}, {"id": 2, "title": "Second book"}]
    user = {"id": 1, "name": "Peter", "books": books}

    user_dump = UserSchema().dump(user)
    assert user_dump["books"] == books

    user_load = UserSchema().load(user_dump)
    assert user_load == user


def test_nested_instance_only():
    class ArtistSchema(Schema):
        first = fields.Str()
        last = fields.Str()

    class AlbumSchema(Schema):
        title = fields.Str()
        artist = fields.Nested(ArtistSchema(), only=("last",))

    schema = AlbumSchema()
    album = {"title": "Hunky Dory", "artist": {"last": "Bowie"}}
    loaded = schema.load(album)
    assert loaded == album
    full_album = {"title": "Hunky Dory", "artist": {"first": "David", "last": "Bowie"}}
    assert schema.dump(full_album) == album


def test_nested_instance_exclude():
    class ArtistSchema(Schema):
        first = fields.Str()
        last = fields.Str()

    class AlbumSchema(Schema):
        title = fields.Str()
        artist = fields.Nested(ArtistSchema(), exclude=("first",))

    schema = AlbumSchema()
    album = {"title": "Hunky Dory", "artist": {"last": "Bowie"}}
    loaded = schema.load(album)
    assert loaded == album
    full_album = {"title": "Hunky Dory", "artist": {"first": "David", "last": "Bowie"}}
    assert schema.dump(full_album) == album


def test_meta_nested_exclude():
    class ChildSchema(Schema):
        foo = fields.Raw()
        bar = fields.Raw()
        baz = fields.Raw()

    class ParentSchema(Schema):
        bla = fields.Raw()
        bli = fields.Raw()
        blubb = fields.Nested(ChildSchema)

        class Meta:
            exclude = ("blubb.foo",)

    data = dict(bla=1, bli=2, blubb=dict(foo=42, bar=24, baz=242))

    sch = ParentSchema()
    result = sch.dump(data)
    assert "bla" in result
    assert "blubb" in result
    assert "bli" in result
    child = result["blubb"]
    assert "foo" not in child
    assert "bar" in child
    assert "baz" in child

    # Test fields with dot notations in Meta.exclude on multiple instantiations
    # Regression test for https://github.com/marshmallow-code/marshmallow/issues/1212
    sch = ParentSchema()
    result = sch.dump(data)
    assert "bla" in result
    assert "blubb" in result
    assert "bli" in result
    child = result["blubb"]
    assert "foo" not in child
    assert "bar" in child
    assert "baz" in child


def test_nested_custom_set_not_implementing_getitem():
    # This test checks that marshmallow can serialize implementations of
    # :mod:`collections.abc.MutableSequence`, with ``__getitem__`` arguments
    # that are not integers.

    class ListLikeParent:
        """
        Implements a list-like object that can get children using a
        non-integer key
        """

        def __init__(self, required_key, child):
            """
            :param required_key: The key to use in ``__getitem__`` in order
                to successfully get the ``child``
            :param child: The return value of the ``child`` if
            ``__getitem__`` succeeds
            """
            self.children = {required_key: child}

    class Child:
        """
        Implements an object with some attribute
        """

        def __init__(self, attribute: str):
            """
            :param attribute: The attribute to initialize
            """
            self.attribute = attribute

    class ChildSchema(Schema):
        """
        The marshmallow schema for the child
        """

        attribute = fields.Str()

    class ParentSchema(Schema):
        """
        The schema for the parent
        """

        children = fields.Nested(ChildSchema, many=True)

    attribute = "Foo"
    required_key = "key"
    child = Child(attribute)

    parent = ListLikeParent(required_key, child)

    ParentSchema().dump(parent)


def test_deeply_nested_only_and_exclude():
    class GrandChildSchema(Schema):
        goo = fields.Raw()
        gah = fields.Raw()
        bah = fields.Raw()

    class ChildSchema(Schema):
        foo = fields.Raw()
        bar = fields.Raw()
        flubb = fields.Nested(GrandChildSchema)

    class ParentSchema(Schema):
        bla = fields.Raw()
        bli = fields.Raw()
        blubb = fields.Nested(ChildSchema)

    sch = ParentSchema(
        only=("bla", "blubb.foo", "blubb.flubb.goo", "blubb.flubb.gah"),
        exclude=("blubb.flubb.goo",),
    )
    data = dict(bla=1, bli=2, blubb=dict(foo=3, bar=4, flubb=dict(goo=5, gah=6, bah=7)))
    result = sch.dump(data)
    assert "bla" in result
    assert "blubb" in result
    assert "bli" not in result
    child = result["blubb"]
    assert "foo" in child
    assert "flubb" in child
    assert "bar" not in child
    grand_child = child["flubb"]
    assert "gah" in grand_child
    assert "goo" not in grand_child
    assert "bah" not in grand_child


def test_nested_lambda():
    class ChildSchema(Schema):
        id = fields.Str()
        name = fields.Str()
        parent = fields.Nested(lambda: ParentSchema(only=("id",)), dump_only=True)
        siblings = fields.List(fields.Nested(lambda: ChildSchema(only=("id", "name"))))

    class ParentSchema(Schema):
        id = fields.Str()
        spouse = fields.Nested(lambda: ParentSchema(only=("id",)))
        children = fields.List(
            fields.Nested(lambda: ChildSchema(only=("id", "parent", "siblings")))
        )

    sch = ParentSchema()
    data_to_load = {
        "id": "p1",
        "spouse": {"id": "p2"},
        "children": [{"id": "c1", "siblings": [{"id": "c2", "name": "sis"}]}],
    }
    loaded = sch.load(data_to_load)
    assert loaded == data_to_load

    data_to_dump = dict(
        id="p2",
        spouse=dict(id="p2"),
        children=[
            dict(
                id="c1",
                name="bar",
                parent=dict(id="p2"),
                siblings=[dict(id="c2", name="sis")],
            )
        ],
    )
    dumped = sch.dump(data_to_dump)
    assert dumped == {
        "id": "p2",
        "spouse": {"id": "p2"},
        "children": [
            {
                "id": "c1",
                "parent": {"id": "p2"},
                "siblings": [{"id": "c2", "name": "sis"}],
            }
        ],
    }


@pytest.mark.parametrize("data_key", ("f1", "f5", None))
def test_data_key_collision(data_key):
    class MySchema(Schema):
        f1 = fields.Raw()
        f2 = fields.Raw(data_key=data_key)
        f3 = fields.Raw(data_key="f5")
        f4 = fields.Raw(data_key="f1", load_only=True)

    if data_key is None:
        MySchema()
    else:
        with pytest.raises(ValueError, match=data_key):
            MySchema()


@pytest.mark.parametrize("attribute", ("f1", "f5", None))
def test_attribute_collision(attribute):
    class MySchema(Schema):
        f1 = fields.Raw()
        f2 = fields.Raw(attribute=attribute)
        f3 = fields.Raw(attribute="f5")
        f4 = fields.Raw(attribute="f1", dump_only=True)

    if attribute is None:
        MySchema()
    else:
        with pytest.raises(ValueError, match=attribute):
            MySchema()


class TestDeeplyNestedLoadOnly:
    @pytest.fixture
    def schema(self):
        class GrandChildSchema(Schema):
            str_dump_only = fields.String()
            str_load_only = fields.String()
            str_regular = fields.String()

        class ChildSchema(Schema):
            str_dump_only = fields.String()
            str_load_only = fields.String()
            str_regular = fields.String()
            grand_child = fields.Nested(GrandChildSchema, unknown=EXCLUDE)

        class ParentSchema(Schema):
            str_dump_only = fields.String()
            str_load_only = fields.String()
            str_regular = fields.String()
            child = fields.Nested(ChildSchema, unknown=EXCLUDE)

        return ParentSchema(
            dump_only=(
                "str_dump_only",
                "child.str_dump_only",
                "child.grand_child.str_dump_only",
            ),
            load_only=(
                "str_load_only",
                "child.str_load_only",
                "child.grand_child.str_load_only",
            ),
        )

    @pytest.fixture
    def data(self):
        return dict(
            str_dump_only="Dump Only",
            str_load_only="Load Only",
            str_regular="Regular String",
            child=dict(
                str_dump_only="Dump Only",
                str_load_only="Load Only",
                str_regular="Regular String",
                grand_child=dict(
                    str_dump_only="Dump Only",
                    str_load_only="Load Only",
                    str_regular="Regular String",
                ),
            ),
        )

    def test_load_only(self, schema, data):
        result = schema.dump(data)
        assert "str_load_only" not in result
        assert "str_dump_only" in result
        assert "str_regular" in result
        child = result["child"]
        assert "str_load_only" not in child
        assert "str_dump_only" in child
        assert "str_regular" in child
        grand_child = child["grand_child"]
        assert "str_load_only" not in grand_child
        assert "str_dump_only" in grand_child
        assert "str_regular" in grand_child

    def test_dump_only(self, schema, data):
        result = schema.load(data, unknown=EXCLUDE)
        assert "str_dump_only" not in result
        assert "str_load_only" in result
        assert "str_regular" in result
        child = result["child"]
        assert "str_dump_only" not in child
        assert "str_load_only" in child
        assert "str_regular" in child
        grand_child = child["grand_child"]
        assert "str_dump_only" not in grand_child
        assert "str_load_only" in grand_child
        assert "str_regular" in grand_child


class TestDeeplyNestedListLoadOnly:
    @pytest.fixture
    def schema(self):
        class ChildSchema(Schema):
            str_dump_only = fields.String()
            str_load_only = fields.String()
            str_regular = fields.String()

        class ParentSchema(Schema):
            str_dump_only = fields.String()
            str_load_only = fields.String()
            str_regular = fields.String()
            child = fields.List(fields.Nested(ChildSchema, unknown=EXCLUDE))

        return ParentSchema(
            dump_only=("str_dump_only", "child.str_dump_only"),
            load_only=("str_load_only", "child.str_load_only"),
        )

    @pytest.fixture
    def data(self):
        return dict(
            str_dump_only="Dump Only",
            str_load_only="Load Only",
            str_regular="Regular String",
            child=[
                dict(
                    str_dump_only="Dump Only",
                    str_load_only="Load Only",
                    str_regular="Regular String",
                )
            ],
        )

    def test_load_only(self, schema, data):
        result = schema.dump(data)
        assert "str_load_only" not in result
        assert "str_dump_only" in result
        assert "str_regular" in result
        child = result["child"][0]
        assert "str_load_only" not in child
        assert "str_dump_only" in child
        assert "str_regular" in child

    def test_dump_only(self, schema, data):
        result = schema.load(data, unknown=EXCLUDE)
        assert "str_dump_only" not in result
        assert "str_load_only" in result
        assert "str_regular" in result
        child = result["child"][0]
        assert "str_dump_only" not in child
        assert "str_load_only" in child
        assert "str_regular" in child


def test_nested_constructor_only_and_exclude():
    class GrandChildSchema(Schema):
        goo = fields.Raw()
        gah = fields.Raw()
        bah = fields.Raw()

    class ChildSchema(Schema):
        foo = fields.Raw()
        bar = fields.Raw()
        flubb = fields.Nested(GrandChildSchema)

    class ParentSchema(Schema):
        bla = fields.Raw()
        bli = fields.Raw()
        blubb = fields.Nested(
            ChildSchema, only=("foo", "flubb.goo", "flubb.gah"), exclude=("flubb.goo",)
        )

    sch = ParentSchema(only=("bla", "blubb"))
    data = dict(bla=1, bli=2, blubb=dict(foo=3, bar=4, flubb=dict(goo=5, gah=6, bah=7)))
    result = sch.dump(data)
    assert "bla" in result
    assert "blubb" in result
    assert "bli" not in result
    child = result["blubb"]
    assert "foo" in child
    assert "flubb" in child
    assert "bar" not in child
    grand_child = child["flubb"]
    assert "gah" in grand_child
    assert "goo" not in grand_child
    assert "bah" not in grand_child


def test_only_and_exclude():
    class MySchema(Schema):
        foo = fields.Raw()
        bar = fields.Raw()
        baz = fields.Raw()

    sch = MySchema(only=("foo", "bar"), exclude=("bar",))
    data = dict(foo=42, bar=24, baz=242)
    result = sch.dump(data)
    assert "foo" in result
    assert "bar" not in result


def test_invalid_only_and_exclude_with_fields():
    class MySchema(Schema):
        foo = fields.Raw()

        class Meta:
            fields = ("bar", "baz")

    with pytest.raises(ValueError) as excinfo:
        MySchema(only=("foo", "par"), exclude=("ban",))

    assert "foo" in str(excinfo.value)
    assert "par" in str(excinfo.value)
    assert "ban" in str(excinfo.value)


def test_exclude_invalid_attribute():
    class MySchema(Schema):
        foo = fields.Raw()

    with pytest.raises(ValueError, match="'bar'"):
        MySchema(exclude=("bar",))


def test_only_bounded_by_fields():
    class MySchema(Schema):
        class Meta:
            fields = ("foo",)

    with pytest.raises(ValueError, match="'baz'"):
        MySchema(only=("baz",))


def test_only_bounded_by_additional():
    class MySchema(Schema):
        class Meta:
            additional = ("b",)

    with pytest.raises(ValueError):
        MySchema(only=("c",)).dump({"c": 3})


def test_only_empty():
    class MySchema(Schema):
        foo = fields.Raw()

    sch = MySchema(only=())
    assert "foo" not in sch.dump({"foo": "bar"})


@pytest.mark.parametrize("param", ("only", "exclude"))
def test_only_and_exclude_as_string(param):
    class MySchema(Schema):
        foo = fields.Raw()

    with pytest.raises(StringNotCollectionError):
        MySchema(**{param: "foo"})  # type: ignore[arg-type]


def test_nested_with_sets():
    class Inner(Schema):
        foo = fields.Raw()

    class Outer(Schema):
        inners = fields.Nested(Inner, many=True)

    sch = Outer()

    class Thing(NamedTuple):
        foo: int

    data = dict(inners={Thing(42), Thing(2)})
    result = sch.dump(data)
    assert len(result["inners"]) == 2


def test_meta_field_not_on_obj_raises_attribute_error(user):
    class BadUserSchema(Schema):
        class Meta:
            fields = ("name",)
            exclude = ("notfound",)

    with pytest.raises(ValueError, match="'notfound'"):
        BadUserSchema().dump(user)


def test_exclude_fields(user):
    s = UserExcludeSchema().dump(user)
    assert "created" not in s
    assert "updated" not in s
    assert "name" in s


def test_fields_option_must_be_list_or_tuple():
    with pytest.raises(ValueError):

        class BadFields(Schema):
            class Meta:
                fields = "name"


def test_exclude_option_must_be_list_or_tuple():
    with pytest.raises(ValueError):

        class BadExclude(Schema):
            class Meta:
                exclude = "name"


def test_datetimeformat_option(user):
    meta_fmt = "%Y-%m"
    field_fmt = "%m-%d"

    class DateTimeFormatSchema(Schema):
        created = fields.DateTime()
        updated = fields.DateTime(field_fmt)

        class Meta:
            datetimeformat = meta_fmt

    serialized = DateTimeFormatSchema().dump(user)
    assert serialized["created"] == user.created.strftime(meta_fmt)
    assert serialized["updated"] == user.updated.strftime(field_fmt)


def test_dateformat_option(user):
    fmt = "%Y-%m"
    field_fmt = "%m-%d"

    class DateFormatSchema(Schema):
        birthdate = fields.Date(field_fmt)
        activation_date = fields.Date()

        class Meta:
            dateformat = fmt

    serialized = DateFormatSchema().dump(user)
    assert serialized["birthdate"] == user.birthdate.strftime(field_fmt)
    assert serialized["activation_date"] == user.activation_date.strftime(fmt)


def test_timeformat_option(user):
    fmt = "%H:%M:%S"
    field_fmt = "%H:%M"

    class TimeFormatSchema(Schema):
        birthtime = fields.Time(field_fmt)
        time_registered = fields.Time()

        class Meta:
            timeformat = fmt

    serialized = TimeFormatSchema().dump(user)
    assert serialized["birthtime"] == user.birthtime.strftime(field_fmt)
    assert serialized["time_registered"] == user.time_registered.strftime(fmt)


def test_default_dateformat(user):
    class DateFormatSchema(Schema):
        created = fields.DateTime()
        updated = fields.DateTime(format="%m-%d")

    serialized = DateFormatSchema().dump(user)
    assert serialized["created"] == user.created.isoformat()
    assert serialized["updated"] == user.updated.strftime("%m-%d")


class CustomError(Exception):
    pass


class MySchema(Schema):
    name = fields.String()
    email = fields.Email()
    age = fields.Integer()

    def handle_error(self, error, data, *args, **kwargs):
        raise CustomError("Something bad happened")

    def test_load_with_custom_error_handler(self):
        in_data = {"email": "invalid"}

        class MySchema3(Schema):
            email = fields.Email()

            def handle_error(self, error, data, **kwargs):
                assert type(error) is ValidationError
                assert "email" in error.messages
                assert isinstance(error.messages, dict)
                assert list(error.messages.keys()) == ["email"]
                assert data == in_data
                raise CustomError("Something bad happened")

        with pytest.raises(CustomError):
            MySchema3().load(in_data)

    def test_load_with_custom_error_handler_and_partially_valid_data(self):
        in_data = {"email": "invalid", "url": "http://valid.com"}

        class MySchema(Schema):
            email = fields.Email()
            url = fields.URL()

            def handle_error(self, error, data, **kwargs):
                assert type(error) is ValidationError
                assert "email" in error.messages
                assert isinstance(error.messages, dict)
                assert list(error.messages.keys()) == ["email"]
                assert data == in_data
                raise CustomError("Something bad happened")

        with pytest.raises(CustomError):
            MySchema().load(in_data)

    def test_custom_error_handler_with_validates_decorator(self):
        in_data = {"num": -1}

        class MySchema(Schema):
            num = fields.Int()

            @validates("num")
            def validate_num(self, value):
                if value < 0:
                    raise ValidationError("Must be greater than 0.")

            def handle_error(self, error, data, **kwargs):
                assert type(error) is ValidationError
                assert "num" in error.messages
                assert isinstance(error.messages, dict)
                assert list(error.messages.keys()) == ["num"]
                assert data == in_data
                raise CustomError("Something bad happened")

        with pytest.raises(CustomError):
            MySchema().load(in_data)

    def test_custom_error_handler_with_validates_schema_decorator(self):
        in_data = {"num": -1}

        class MySchema(Schema):
            num = fields.Int()

            @validates_schema
            def validates_schema(self, data, **kwargs):
                raise ValidationError("Invalid schema!")

            def handle_error(self, error, data, **kwargs):
                assert type(error) is ValidationError
                assert isinstance(error.messages, dict)
                assert list(error.messages.keys()) == ["_schema"]
                assert data == in_data
                raise CustomError("Something bad happened")

        with pytest.raises(CustomError):
            MySchema().load(in_data)

    def test_validate_with_custom_error_handler(self):
        with pytest.raises(CustomError):
            MySchema().validate({"age": "notvalid", "email": "invalid"})


class TestFieldValidation:
    def test_errors_are_cleared_after_loading_collection(self):
        def always_fail(val):
            raise ValidationError("lol")

        class MySchema(Schema):
            foo = fields.Str(validate=always_fail)

        schema = MySchema()
        with pytest.raises(ValidationError) as excinfo:
            schema.load([{"foo": "bar"}, {"foo": "baz"}], many=True)
        errors = excinfo.value.messages
        assert len(errors[0]["foo"]) == 1
        assert len(errors[1]["foo"]) == 1
        with pytest.raises(ValidationError) as excinfo:
            schema.load({"foo": "bar"})
        errors = excinfo.value.messages
        assert len(errors["foo"]) == 1

    def test_raises_error_with_list(self):
        def validator(val):
            raise ValidationError(["err1", "err2"])

        class MySchema(Schema):
            foo = fields.Raw(validate=validator)

        s = MySchema()
        errors = s.validate({"foo": 42})
        assert errors["foo"] == ["err1", "err2"]

    # https://github.com/marshmallow-code/marshmallow/issues/110
    def test_raises_error_with_dict(self):
        def validator(val):
            raise ValidationError({"code": "invalid_foo"})

        class MySchema(Schema):
            foo = fields.Raw(validate=validator)

        s = MySchema()
        errors = s.validate({"foo": 42})
        assert errors["foo"] == [{"code": "invalid_foo"}]

    def test_ignored_if_not_in_only(self):
        class MySchema(Schema):
            a = fields.Raw()
            b = fields.Raw()

            @validates("a")
            def validate_a(self, val, **kwargs):
                raise ValidationError({"code": "invalid_a"})

            @validates("b")
            def validate_b(self, val, **kwargs):
                raise ValidationError({"code": "invalid_b"})

        s = MySchema(only=("b",))
        errors = s.validate({"b": "data"})
        assert errors == {"b": {"code": "invalid_b"}}


def test_schema_repr():
    class MySchema(Schema):
        name = fields.String()

    ser = MySchema(many=True)
    rep = repr(ser)
    assert "MySchema" in rep
    assert "many=True" in rep


class TestNestedSchema:
    @pytest.fixture
    def user(self):
        return User(name="Monty", age=81)

    @pytest.fixture
    def blog(self, user):
        col1 = User(name="Mick", age=123)
        col2 = User(name="Keith", age=456)
        return Blog(
            "Monty's blog",
            user=user,
            categories=["humor", "violence"],
            collaborators=[col1, col2],
        )

    # regression test for https://github.com/marshmallow-code/marshmallow/issues/64
    def test_nested_many_with_missing_attribute(self, user):
        class SimpleBlogSchema(Schema):
            title = fields.Str()
            wat = fields.Nested(UserSchema, many=True)

        blog = Blog("Simple blog", user=user, collaborators=None)
        schema = SimpleBlogSchema()
        result = schema.dump(blog)
        assert "wat" not in result

    def test_nested_with_attribute_none(self):
        class InnerSchema(Schema):
            bar = fields.Raw()

        class MySchema(Schema):
            foo = fields.Nested(InnerSchema)

        class MySchema2(Schema):
            foo = fields.Nested(InnerSchema)

        s = MySchema()
        result = s.dump({"foo": None})
        assert result["foo"] is None

        s2 = MySchema2()
        result2 = s2.dump({"foo": None})
        assert result2["foo"] is None

    def test_nested_field_does_not_validate_required(self):
        class BlogRequiredSchema(Schema):
            user = fields.Nested(UserSchema, required=True)

        b = Blog("Authorless blog", user=None)
        BlogRequiredSchema().dump(b)

    def test_nested_none(self):
        class BlogDefaultSchema(Schema):
            user = fields.Nested(UserSchema, dump_default=0)

        b = Blog("Just the default blog", user=None)
        data = BlogDefaultSchema().dump(b)
        assert data["user"] is None

    def test_nested(self, user, blog):
        blog_serializer = BlogSchema()
        serialized_blog = blog_serializer.dump(blog)
        user_serializer = UserSchema()
        serialized_user = user_serializer.dump(user)
        assert serialized_blog["user"] == serialized_user

        with pytest.raises(ValidationError, match="email"):
            BlogSchema().load(
                {"title": "Monty's blog", "user": {"name": "Monty", "email": "foo"}}
            )

    def test_nested_many_fields(self, blog):
        serialized_blog = BlogSchema().dump(blog)
        expected = [UserSchema().dump(col) for col in blog.collaborators]
        assert serialized_blog["collaborators"] == expected

    def test_nested_only(self, blog):
        col1 = User(name="Mick", age=123, id_="abc")
        col2 = User(name="Keith", age=456, id_="def")
        blog.collaborators = [col1, col2]
        serialized_blog = BlogOnlySchema().dump(blog)
        assert serialized_blog["collaborators"] == [{"id": col1.id}, {"id": col2.id}]

    def test_exclude(self, blog):
        serialized = BlogSchemaExclude().dump(blog)
        assert "uppername" not in serialized["user"]

    def test_list_field(self, blog):
        serialized = BlogSchema().dump(blog)
        assert serialized["categories"] == ["humor", "violence"]

    def test_nested_load_many(self):
        in_data = {
            "title": "Shine A Light",
            "collaborators": [
                {"name": "Mick", "email": "mick@stones.com"},
                {"name": "Keith", "email": "keith@stones.com"},
            ],
        }
        data = BlogSchema().load(in_data)
        collabs = data["collaborators"]
        assert len(collabs) == 2
        assert all(type(each) is User for each in collabs)
        assert collabs[0].name == in_data["collaborators"][0]["name"]

    def test_nested_errors(self):
        with pytest.raises(ValidationError) as excinfo:
            BlogSchema().load(
                {"title": "Monty's blog", "user": {"name": "Monty", "email": "foo"}}
            )
        errors = excinfo.value.messages
        assert "email" in errors["user"]
        assert len(errors["user"]["email"]) == 1
        assert "Not a valid email address." in errors["user"]["email"][0]
        # No problems with collaborators
        assert "collaborators" not in errors

    def test_nested_method_field(self, blog):
        data = BlogSchema().dump(blog)
        assert data["user"]["is_old"]
        assert data["collaborators"][0]["is_old"]

    def test_nested_function_field(self, blog, user):
        data = BlogSchema().dump(blog)
        assert data["user"]["lowername"] == user.name.lower()
        expected = blog.collaborators[0].name.lower()
        assert data["collaborators"][0]["lowername"] == expected

    def test_nested_fields_must_be_passed_a_serializer(self, blog):
        class BadNestedFieldSchema(BlogSchema):
            user = fields.Nested(fields.String)  # type: ignore[arg-type]

        with pytest.raises(ValueError):
            BadNestedFieldSchema().dump(blog)

    # regression test for https://github.com/marshmallow-code/marshmallow/issues/188
    def test_invalid_type_passed_to_nested_field(self):
        class InnerSchema(Schema):
            foo = fields.Raw()

        class MySchema(Schema):
            inner = fields.Nested(InnerSchema, many=True)

        sch = MySchema()

        sch.load({"inner": [{"foo": 42}]})

        with pytest.raises(ValidationError) as excinfo:
            sch.load({"inner": "invalid"})
        errors = excinfo.value.messages
        assert "inner" in errors
        assert errors["inner"] == ["Invalid type."]

        class OuterSchema(Schema):
            inner = fields.Nested(InnerSchema)

        schema = OuterSchema()
        with pytest.raises(ValidationError) as excinfo:
            schema.load({"inner": 1})
        errors = excinfo.value.messages
        assert errors["inner"]["_schema"] == ["Invalid input type."]

    # regression test for https://github.com/marshmallow-code/marshmallow/issues/298
    def test_all_errors_on_many_nested_field_with_validates_decorator(self):
        class Inner(Schema):
            req = fields.Raw(required=True)

        class Outer(Schema):
            inner = fields.Nested(Inner, many=True)

            @validates("inner")
            def validates_inner(self, data, **kwargs):
                raise ValidationError("not a chance")

        outer = Outer()
        with pytest.raises(ValidationError) as excinfo:
            outer.load({"inner": [{}]})
        errors = excinfo.value.messages
        assert "inner" in errors
        assert "_schema" in errors["inner"]

    @pytest.mark.parametrize("unknown", (None, RAISE, INCLUDE, EXCLUDE))
    def test_nested_unknown_validation(self, unknown):
        class ChildSchema(Schema):
            num = fields.Int()

        class ParentSchema(Schema):
            child = fields.Nested(ChildSchema, unknown=unknown)

        data = {"child": {"num": 1, "extra": 1}}
        if unknown is None or unknown == RAISE:
            with pytest.raises(ValidationError) as excinfo:
                ParentSchema().load(data)
            exc = excinfo.value
            assert exc.messages == {"child": {"extra": ["Unknown field."]}}
        else:
            output = {
                INCLUDE: {"child": {"num": 1, "extra": 1}},
                EXCLUDE: {"child": {"num": 1}},
            }[unknown]
            assert ParentSchema().load(data) == output


class TestPluckSchema:
    @pytest.mark.parametrize("user_schema", [UserSchema, UserSchema()])
    def test_pluck(self, user_schema, blog):
        class FlatBlogSchema(Schema):
            user = fields.Pluck(user_schema, "name")
            collaborators = fields.Pluck(user_schema, "name", many=True)

        s = FlatBlogSchema()
        data = s.dump(blog)
        assert data["user"] == blog.user.name
        for i, name in enumerate(data["collaborators"]):
            assert name == blog.collaborators[i].name

    def test_pluck_none(self, blog):
        class FlatBlogSchema(Schema):
            user = fields.Pluck(UserSchema, "name")
            collaborators = fields.Pluck(UserSchema, "name", many=True)

        col1 = User(name="Mick", age=123)
        col2 = User(name="Keith", age=456)
        blog = Blog(title="Unowned Blog", user=None, collaborators=[col1, col2])
        s = FlatBlogSchema()
        data = s.dump(blog)
        assert data["user"] == blog.user
        for i, name in enumerate(data["collaborators"]):
            assert name == blog.collaborators[i].name

    # Regression test for https://github.com/marshmallow-code/marshmallow/issues/800
    def test_pluck_with_data_key(self, blog):
        class UserSchema(Schema):
            name = fields.String(data_key="username")
            age = fields.Int()

        class FlatBlogSchema(Schema):
            user = fields.Pluck(UserSchema, "name")
            collaborators = fields.Pluck(UserSchema, "name", many=True)

        s = FlatBlogSchema()
        data = s.dump(blog)
        assert data["user"] == blog.user.name
        for i, name in enumerate(data["collaborators"]):
            assert name == blog.collaborators[i].name
        assert s.load(data) == {
            "user": {"name": "Monty"},
            "collaborators": [{"name": "Mick"}, {"name": "Keith"}],
        }


class TestSelfReference:
    @pytest.fixture
    def employer(self):
        return User(name="Joe", age=59)

    @pytest.fixture
    def user(self, employer):
        return User(name="Tom", employer=employer, age=28)

    def test_nesting_schema_by_passing_lambda(self, user, employer):
        class SelfReferencingSchema(Schema):
            name = fields.Str()
            age = fields.Int()
            employer = fields.Nested(
                lambda: SelfReferencingSchema(exclude=("employer",))
            )

        data = SelfReferencingSchema().dump(user)
        assert data["name"] == user.name
        assert data["employer"]["name"] == employer.name
        assert data["employer"]["age"] == employer.age

    def test_nesting_schema_by_passing_class_name(self, user, employer):
        class SelfReferencingSchema(Schema):
            name = fields.Str()
            age = fields.Int()
            employer = fields.Nested("SelfReferencingSchema", exclude=("employer",))

        data = SelfReferencingSchema().dump(user)
        assert data["name"] == user.name
        assert data["employer"]["name"] == employer.name
        assert data["employer"]["age"] == employer.age

    def test_nesting_within_itself_exclude(self, user, employer):
        class SelfSchema(Schema):
            name = fields.String()
            age = fields.Integer()
            employer = fields.Nested(lambda: SelfSchema(exclude=("employer",)))

        data = SelfSchema().dump(user)
        assert data["name"] == user.name
        assert data["age"] == user.age
        assert data["employer"]["name"] == employer.name
        assert data["employer"]["age"] == employer.age

    def test_nested_self_with_only_param(self, user, employer):
        class SelfSchema(Schema):
            name = fields.String()
            age = fields.Integer()
            employer = fields.Nested(lambda: SelfSchema(only=("name",)))

        data = SelfSchema().dump(user)
        assert data["employer"]["name"] == employer.name
        assert "age" not in data["employer"]

    def test_multiple_pluck_self_lambda(self, user):
        class MultipleSelfSchema(Schema):
            name = fields.String()
            emp = fields.Pluck(
                lambda: MultipleSelfSchema(), "name", attribute="employer"
            )
            rels = fields.Pluck(
                lambda: MultipleSelfSchema(), "name", many=True, attribute="relatives"
            )

        schema = MultipleSelfSchema()
        user.relatives = [User(name="Bar", age=12), User(name="Baz", age=34)]
        data = schema.dump(user)
        assert len(data["rels"]) == len(user.relatives)
        relative = data["rels"][0]
        assert relative == user.relatives[0].name

    def test_nested_self_many_lambda(self):
        class SelfManySchema(Schema):
            relatives = fields.Nested(lambda: SelfManySchema(), many=True)
            name = fields.String()
            age = fields.Integer()

        person = User(name="Foo")
        person.relatives = [User(name="Bar", age=12), User(name="Baz", age=34)]
        data = SelfManySchema().dump(person)
        assert data["name"] == person.name
        assert len(data["relatives"]) == len(person.relatives)
        assert data["relatives"][0]["name"] == person.relatives[0].name
        assert data["relatives"][0]["age"] == person.relatives[0].age

    def test_nested_self_list(self):
        class SelfListSchema(Schema):
            relatives = fields.List(fields.Nested(lambda: SelfListSchema()))
            name = fields.String()
            age = fields.Integer()

        person = User(name="Foo")
        person.relatives = [User(name="Bar", age=12), User(name="Baz", age=34)]
        data = SelfListSchema().dump(person)
        assert data["name"] == person.name
        assert len(data["relatives"]) == len(person.relatives)
        assert data["relatives"][0]["name"] == person.relatives[0].name
        assert data["relatives"][0]["age"] == person.relatives[0].age


class RequiredUserSchema(Schema):
    name = fields.Raw(required=True)


def test_serialization_with_required_field():
    user = User(name=None)
    RequiredUserSchema().dump(user)


def test_deserialization_with_required_field():
    with pytest.raises(ValidationError) as excinfo:
        RequiredUserSchema().load({})
    data, errors = excinfo.value.valid_data, excinfo.value.messages
    assert "name" in errors
    assert "Missing data for required field." in errors["name"]
    assert isinstance(data, dict)
    # field value should also not be in output data
    assert "name" not in data


def test_deserialization_with_required_field_and_custom_validator():
    def validator(val):
        if val.lower() not in {"red", "blue"}:
            raise ValidationError("Color must be red or blue")

    class ValidatingSchema(Schema):
        color = fields.String(
            required=True,
            validate=validator,
        )

    with pytest.raises(ValidationError) as excinfo:
        ValidatingSchema().load({"name": "foo"})
    errors = excinfo.value.messages
    assert errors
    assert "color" in errors
    assert "Missing data for required field." in errors["color"]

    with pytest.raises(ValidationError) as excinfo:
        ValidatingSchema().load({"color": "green"})
    errors = excinfo.value.messages
    assert "color" in errors
    assert "Color must be red or blue" in errors["color"]


def test_serializer_can_specify_nested_object_as_attribute(blog):
    class BlogUsernameSchema(Schema):
        author_name = fields.String(attribute="user.name")

    ser = BlogUsernameSchema()
    result = ser.dump(blog)
    assert result["author_name"] == blog.user.name


class TestFieldInheritance:
    def test_inherit_fields_from_schema_subclass(self):
        expected = {
            "field_a": fields.Integer(),
            "field_b": fields.Integer(),
        }

        class SerializerA(Schema):
            field_a = expected["field_a"]

        class SerializerB(SerializerA):
            field_b = expected["field_b"]

        assert SerializerB._declared_fields == expected

    def test_inherit_fields_from_non_schema_subclass(self):
        expected = {
            "field_a": fields.Integer(),
            "field_b": fields.Integer(),
        }

        class PlainBaseClass:
            field_a = expected["field_a"]

        class SerializerB1(Schema, PlainBaseClass):
            field_b = expected["field_b"]

        class SerializerB2(PlainBaseClass, Schema):
            field_b = expected["field_b"]

        assert SerializerB1._declared_fields == expected
        assert SerializerB2._declared_fields == expected

    def test_inheritance_follows_mro(self):
        expected = {
            "field_a": fields.String(),
            "field_b": fields.String(),
            "field_c": fields.String(),
            "field_d": fields.String(),
        }
        # Diamond inheritance graph
        # MRO: D -> B -> C -> A

        class SerializerA(Schema):
            field_a = expected["field_a"]

        class SerializerB(SerializerA):
            field_b = expected["field_b"]

        class SerializerC(SerializerA):
            field_c = expected["field_c"]

        class SerializerD(SerializerB, SerializerC):
            field_d = expected["field_d"]

        assert SerializerD._declared_fields == expected


def get_from_dict(schema, obj, key, default=None):
    return obj.get("_" + key, default)


class TestGetAttribute:
    def test_get_attribute_is_used(self):
        class UserDictSchema(Schema):
            name = fields.Str()
            email = fields.Email()

            def get_attribute(self, obj, attr, default):
                return get_from_dict(self, obj, attr, default)

        user_dict = {"_name": "joe", "_email": "joe@shmoe.com"}
        schema = UserDictSchema()
        result = schema.dump(user_dict)
        assert result["name"] == user_dict["_name"]
        assert result["email"] == user_dict["_email"]
        # can't serialize User object
        user = User(name="joe", email="joe@shmoe.com")
        with pytest.raises(AttributeError):
            schema.dump(user)

    def test_get_attribute_with_many(self):
        class UserDictSchema(Schema):
            name = fields.Str()
            email = fields.Email()

            def get_attribute(self, obj, attr, default):
                return get_from_dict(self, obj, attr, default)

        user_dicts = [
            {"_name": "joe", "_email": "joe@shmoe.com"},
            {"_name": "jane", "_email": "jane@shmane.com"},
        ]
        schema = UserDictSchema(many=True)
        results = schema.dump(user_dicts)
        for result, user_dict in zip(results, user_dicts):
            assert result["name"] == user_dict["_name"]
            assert result["email"] == user_dict["_email"]
        # can't serialize User object
        users = [
            User(name="joe", email="joe@shmoe.com"),
            User(name="jane", email="jane@shmane.com"),
        ]
        with pytest.raises(AttributeError):
            schema.dump(users)


class TestRequiredFields:
    class StringSchema(Schema):
        required_field = fields.Str(required=True)
        allow_none_field = fields.Str(allow_none=True)
        allow_none_required_field = fields.Str(required=True, allow_none=True)

    @pytest.fixture
    def string_schema(self):
        return self.StringSchema()

    @pytest.fixture
    def data(self):
        return dict(
            required_field="foo",
            allow_none_field="bar",
            allow_none_required_field="one",
        )

    def test_required_string_field_missing(self, string_schema, data):
        del data["required_field"]
        errors = string_schema.validate(data)
        assert errors["required_field"] == ["Missing data for required field."]

    def test_required_string_field_failure(self, string_schema, data):
        data["required_field"] = None
        errors = string_schema.validate(data)
        assert errors["required_field"] == ["Field may not be null."]

    def test_allow_none_param(self, string_schema, data):
        data["allow_none_field"] = None
        errors = string_schema.validate(data)
        assert errors == {}

        data["allow_none_required_field"] = None
        string_schema.validate(data)

        del data["allow_none_required_field"]
        errors = string_schema.validate(data)
        assert "allow_none_required_field" in errors

    def test_allow_none_custom_message(self, data):
        class MySchema(Schema):
            allow_none_field = fields.Raw(
                allow_none=False, error_messages={"null": "<custom>"}
            )

        schema = MySchema()
        errors = schema.validate({"allow_none_field": None})
        assert errors["allow_none_field"][0] == "<custom>"


class TestDefaults:
    class MySchema(Schema):
        int_no_default = fields.Int(allow_none=True)
        str_no_default = fields.Str(allow_none=True)
        list_no_default = fields.List(fields.Str, allow_none=True)
        nested_no_default = fields.Nested(UserSchema, many=True, allow_none=True)

        int_with_default = fields.Int(allow_none=True, dump_default=42)
        str_with_default = fields.Str(allow_none=True, dump_default="foo")

    @pytest.fixture
    def schema(self):
        return self.MySchema()

    @pytest.fixture
    def data(self):
        return dict(
            int_no_default=None,
            str_no_default=None,
            list_no_default=None,
            nested_no_default=None,
            int_with_default=None,
            str_with_default=None,
        )

    def test_missing_inputs_are_excluded_from_dump_output(self, schema, data):
        for key in [
            "int_no_default",
            "str_no_default",
            "list_no_default",
            "nested_no_default",
        ]:
            d = data.copy()
            del d[key]
            result = schema.dump(d)
            # the missing key is not in the serialized result
            assert key not in result
            # the rest of the keys are in the result
            assert all(k in result for k in d)

    def test_none_is_serialized_to_none(self, schema, data):
        errors = schema.validate(data)
        assert errors == {}
        result = schema.dump(data)
        for key in data:
            msg = f"result[{key!r}] should be None"
            assert result[key] is None, msg

    def test_default_and_value_missing(self, schema, data):
        del data["int_with_default"]
        del data["str_with_default"]
        result = schema.dump(data)
        assert result["int_with_default"] == 42
        assert result["str_with_default"] == "foo"

    def test_loading_none(self, schema, data):
        result = schema.load(data)
        for key in data:
            assert result[key] is None

    def test_missing_inputs_are_excluded_from_load_output(self, schema, data):
        for key in [
            "int_no_default",
            "str_no_default",
            "list_no_default",
            "nested_no_default",
        ]:
            d = data.copy()
            del d[key]
            result = schema.load(d)
            # the missing key is not in the deserialized result
            assert key not in result
            # the rest of the keys are in the result
            assert all(k in result for k in d)


class TestLoadOnly:
    class MySchema(Schema):
        class Meta:
            load_only = ("str_load_only",)
            dump_only = ("str_dump_only",)

        str_dump_only = fields.String()
        str_load_only = fields.String()
        str_regular = fields.String()

    @pytest.fixture
    def schema(self):
        return self.MySchema()

    @pytest.fixture
    def data(self):
        return dict(
            str_dump_only="Dump Only",
            str_load_only="Load Only",
            str_regular="Regular String",
        )

    def test_load_only(self, schema, data):
        result = schema.dump(data)
        assert "str_load_only" not in result
        assert "str_dump_only" in result
        assert "str_regular" in result

    def test_dump_only(self, schema, data):
        result = schema.load(data, unknown=EXCLUDE)
        assert "str_dump_only" not in result
        assert "str_load_only" in result
        assert "str_regular" in result

    # regression test for https://github.com/marshmallow-code/marshmallow/pull/765
    def test_url_field_requre_tld_false(self):
        class NoTldTestSchema(Schema):
            url = fields.Url(require_tld=False, schemes=["marshmallow"])

        schema = NoTldTestSchema()
        data_with_no_top_level_domain = {"url": "marshmallow://app/discounts"}
        result = schema.load(data_with_no_top_level_domain)
        assert result == data_with_no_top_level_domain


class TestFromDict:
    def test_generates_schema(self):
        MySchema = Schema.from_dict({"foo": fields.Str()})
        assert issubclass(MySchema, Schema)

    def test_name(self):
        MySchema = Schema.from_dict({"foo": fields.Str()})
        assert "GeneratedSchema" in repr(MySchema)
        SchemaWithName = Schema.from_dict(
            {"foo": fields.Int()}, name="MyGeneratedSchema"
        )
        assert "MyGeneratedSchema" in repr(SchemaWithName)

    def test_generated_schemas_are_not_registered(self):
        n_registry_entries = len(class_registry._registry)
        Schema.from_dict({"foo": fields.Str()})
        Schema.from_dict({"bar": fields.Str()}, name="MyGeneratedSchema")
        assert len(class_registry._registry) == n_registry_entries
        with pytest.raises(RegistryError):
            class_registry.get_class("GeneratedSchema")
        with pytest.raises(RegistryError):
            class_registry.get_class("MyGeneratedSchema")

    def test_meta_options_are_applied(self):
        class OrderedSchema(Schema):
            class Meta:
                load_only = ("bar",)

        OSchema = OrderedSchema.from_dict({"foo": fields.Int(), "bar": fields.Int()})
        dumped = OSchema().dump({"foo": 42, "bar": 24})
        assert "bar" not in dumped


def test_class_registry_returns_schema_type():
    class DefinitelyUniqueSchema(Schema):
        """
        Just a schema
        """

    SchemaClass = class_registry.get_class(DefinitelyUniqueSchema.__name__)
    assert SchemaClass is DefinitelyUniqueSchema


@pytest.mark.parametrize("dict_cls", (dict, OrderedDict))
def test_set_dict_class(dict_cls):
    """Demonstrate how to specify dict_class as class attribute"""

    class MySchema(Schema):
        dict_class = dict_cls
        foo = fields.String()

    result = MySchema().dump({"foo": "bar"})
    assert result == {"foo": "bar"}
    assert isinstance(result, dict_cls)


# === tests/test_fields.py ===
import pytest

from marshmallow import (
    EXCLUDE,
    INCLUDE,
    RAISE,
    Schema,
    ValidationError,
    fields,
    missing,
)
from marshmallow.exceptions import StringNotCollectionError
from marshmallow.orderedset import OrderedSet
from tests.base import ALL_FIELDS


@pytest.mark.parametrize(
    ("alias", "field"),
    [
        (fields.Int, fields.Integer),
        (fields.Str, fields.String),
        (fields.Bool, fields.Boolean),
        (fields.URL, fields.Url),
    ],
)
def test_field_aliases(alias, field):
    assert alias is field


class TestField:
    def test_repr(self):
        default = "œ∑´"  # noqa: RUF001
        field = fields.Raw(dump_default=default, attribute=None)
        assert repr(field) == (
            f"<fields.Raw(dump_default={default!r}, attribute=None, "
            "validate=None, required=False, "
            "load_only=False, dump_only=False, "
            f"load_default={missing}, allow_none=False, "
            f"error_messages={field.error_messages})>"
        )
        int_field = fields.Integer(validate=lambda x: True)
        assert "<fields.Integer" in repr(int_field)

    def test_error_raised_if_uncallable_validator_passed(self):
        with pytest.raises(ValueError, match="must be a callable"):
            fields.Raw(validate="notcallable")  # type: ignore[arg-type]

    def test_error_raised_if_missing_is_set_on_required_field(self):
        with pytest.raises(
            ValueError, match="'load_default' must not be set for required fields"
        ):
            fields.Raw(required=True, load_default=42)

    def test_custom_field_receives_attr_and_obj(self):
        class MyField(fields.Field[str]):
            def _deserialize(self, value, attr, data, **kwargs) -> str:
                assert attr == "name"
                assert data["foo"] == 42
                return str(value)

        class MySchema(Schema):
            name = MyField()

        result = MySchema(unknown=EXCLUDE).load({"name": "Monty", "foo": 42})
        assert result == {"name": "Monty"}

    def test_custom_field_receives_data_key_if_set(self):
        class MyField(fields.Field[str]):
            def _deserialize(self, value, attr, data, **kwargs):
                assert attr == "name"
                assert data["foo"] == 42
                return str(value)

        class MySchema(Schema):
            Name = MyField(data_key="name")

        result = MySchema(unknown=EXCLUDE).load({"name": "Monty", "foo": 42})
        assert result == {"Name": "Monty"}

    def test_custom_field_follows_data_key_if_set(self):
        class MyField(fields.Field[str]):
            def _serialize(self, value, attr, obj, **kwargs) -> str:
                assert attr == "name"
                assert obj["foo"] == 42
                return str(value)

        class MySchema(Schema):
            name = MyField(data_key="_NaMe")

        result = MySchema().dump({"name": "Monty", "foo": 42})
        assert result == {"_NaMe": "Monty"}


class TestParentAndName:
    class MySchema(Schema):
        foo = fields.Raw()
        bar = fields.List(fields.Str())
        baz = fields.Tuple((fields.Str(), fields.Int()))
        bax = fields.Dict(fields.Str(), fields.Int())

    @pytest.fixture
    def schema(self):
        return self.MySchema()

    def test_simple_field_parent_and_name(self, schema):
        assert schema.fields["foo"].parent == schema
        assert schema.fields["foo"].name == "foo"
        assert schema.fields["bar"].parent == schema
        assert schema.fields["bar"].name == "bar"

    # https://github.com/marshmallow-code/marshmallow/pull/572#issuecomment-275800288
    def test_unbound_field_root_returns_none(self):
        field = fields.Str()
        assert field.root is None

        inner_field = fields.Nested(self.MySchema())
        outer_field = fields.List(inner_field)

        assert outer_field.root is None
        assert inner_field.root is None

    def test_list_field_inner_parent_and_name(self, schema):
        assert schema.fields["bar"].inner.parent == schema.fields["bar"]
        assert schema.fields["bar"].inner.name == "bar"

    def test_tuple_field_inner_parent_and_name(self, schema):
        for field in schema.fields["baz"].tuple_fields:
            assert field.parent == schema.fields["baz"]
            assert field.name == "baz"

    def test_mapping_field_inner_parent_and_name(self, schema):
        assert schema.fields["bax"].value_field.parent == schema.fields["bax"]
        assert schema.fields["bax"].value_field.name == "bax"
        assert schema.fields["bax"].key_field.parent == schema.fields["bax"]
        assert schema.fields["bax"].key_field.name == "bax"

    def test_simple_field_root(self, schema):
        assert schema.fields["foo"].root == schema
        assert schema.fields["bar"].root == schema

    def test_list_field_inner_root(self, schema):
        assert schema.fields["bar"].inner.root == schema

    def test_tuple_field_inner_root(self, schema):
        for field in schema.fields["baz"].tuple_fields:
            assert field.root == schema

    def test_list_root_inheritance(self, schema):
        class OtherSchema(TestParentAndName.MySchema):
            pass

        schema2 = OtherSchema()

        bar_field = schema.fields["bar"]
        assert isinstance(bar_field, fields.List)
        assert bar_field.inner.root == schema

        bar_field2 = schema2.fields["bar"]
        assert isinstance(bar_field2, fields.List)
        assert bar_field2.inner.root == schema2

    def test_dict_root_inheritance(self):
        class MySchema(Schema):
            foo = fields.Dict(keys=fields.Str(), values=fields.Int())

        class OtherSchema(MySchema):
            pass

        schema = MySchema()
        schema2 = OtherSchema()

        foo_field = schema.fields["foo"]
        assert isinstance(foo_field, fields.Dict)
        assert isinstance(foo_field.key_field, fields.Str)
        assert isinstance(foo_field.value_field, fields.Int)
        assert foo_field.key_field.root == schema
        assert foo_field.value_field.root == schema

        foo_field2 = schema2.fields["foo"]
        assert isinstance(foo_field2, fields.Dict)
        assert isinstance(foo_field2.key_field, fields.Str)
        assert isinstance(foo_field2.value_field, fields.Int)
        assert foo_field2.key_field.root == schema2
        assert foo_field2.value_field.root == schema2

    # Regression test for https://github.com/marshmallow-code/marshmallow/issues/1357
    def test_datetime_list_inner_format(self, schema):
        class MySchema(Schema):
            foo = fields.List(fields.DateTime())
            bar = fields.Tuple((fields.DateTime(),))
            baz = fields.List(fields.Date())
            qux = fields.Tuple((fields.Date(),))

            class Meta:
                datetimeformat = "iso8601"
                dateformat = "iso8601"

        schema = MySchema()
        for field_name in ("foo", "baz"):
            assert schema.fields[field_name].inner.format == "iso8601"
        for field_name in ("bar", "qux"):
            assert schema.fields[field_name].tuple_fields[0].format == "iso8601"

    # Regression test for https://github.com/marshmallow-code/marshmallow/issues/1808
    def test_field_named_parent_has_root(self, schema):
        class MySchema(Schema):
            parent = fields.Raw()

        schema = MySchema()
        assert schema.fields["parent"].root == schema


class TestMetadata:
    @pytest.mark.parametrize("FieldClass", ALL_FIELDS)
    def test_extra_metadata_may_be_added_to_field(self, FieldClass):
        field = FieldClass(
            required=True,
            dump_default=None,
            validate=lambda v: True,
            metadata={"description": "foo", "widget": "select"},
        )
        assert field.metadata == {"description": "foo", "widget": "select"}


class TestErrorMessages:
    class MyField(fields.Field):
        default_error_messages = {"custom": "Custom error message."}

    error_messages = (
        ("required", "Missing data for required field."),
        ("null", "Field may not be null."),
        ("custom", "Custom error message."),
        ("validator_failed", "Invalid value."),
    )

    def test_default_error_messages_get_merged_with_parent_error_messages_cstm_msg(
        self,
    ):
        field = self.MyField()
        assert field.error_messages["custom"] == "Custom error message."
        assert "required" in field.error_messages

    def test_default_error_messages_get_merged_with_parent_error_messages(self):
        field = self.MyField(error_messages={"passed": "Passed error message"})
        assert field.error_messages["passed"] == "Passed error message"

    @pytest.mark.parametrize(("key", "message"), error_messages)
    def test_make_error(self, key, message):
        field = self.MyField()

        error = field.make_error(key)
        assert error.args[0] == message

    def test_make_error_key_doesnt_exist(self):
        with pytest.raises(AssertionError) as excinfo:
            self.MyField().make_error("doesntexist")
        assert "doesntexist" in excinfo.value.args[0]
        assert "MyField" in excinfo.value.args[0]


class TestNestedField:
    @pytest.mark.parametrize("param", ("only", "exclude"))
    def test_nested_only_and_exclude_as_string(self, param):
        with pytest.raises(StringNotCollectionError):
            fields.Nested(Schema, **{param: "foo"})  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "nested_value",
        [
            {"hello": fields.String()},
            lambda: {"hello": fields.String()},
        ],
    )
    def test_nested_instantiation_from_dict(self, nested_value):
        class MySchema(Schema):
            nested = fields.Nested(nested_value)

        schema = MySchema()

        ret = schema.load({"nested": {"hello": "world"}})
        assert ret == {"nested": {"hello": "world"}}

        with pytest.raises(ValidationError):
            schema.load({"nested": {"x": 1}})

    @pytest.mark.parametrize("schema_unknown", (EXCLUDE, INCLUDE, RAISE))
    @pytest.mark.parametrize("field_unknown", (None, EXCLUDE, INCLUDE, RAISE))
    def test_nested_unknown_override(self, schema_unknown, field_unknown):
        class NestedSchema(Schema):
            class Meta:
                unknown = schema_unknown

        class MySchema(Schema):
            nested = fields.Nested(NestedSchema, unknown=field_unknown)

        if field_unknown == EXCLUDE or (
            schema_unknown == EXCLUDE and not field_unknown
        ):
            assert MySchema().load({"nested": {"x": 1}}) == {"nested": {}}
        elif field_unknown == INCLUDE or (
            schema_unknown == INCLUDE and not field_unknown
        ):
            assert MySchema().load({"nested": {"x": 1}}) == {"nested": {"x": 1}}
        elif field_unknown == RAISE or (schema_unknown == RAISE and not field_unknown):
            with pytest.raises(ValidationError):
                MySchema().load({"nested": {"x": 1}})

    @pytest.mark.parametrize(
        ("param", "fields_list"), [("only", ["foo"]), ("exclude", ["bar"])]
    )
    def test_nested_schema_only_and_exclude(self, param, fields_list):
        class NestedSchema(Schema):
            # We mean to test the use of OrderedSet to specify it explicitly
            # even if it is default
            set_class = OrderedSet
            foo = fields.String()
            bar = fields.String()

        class MySchema(Schema):
            nested = fields.Nested(NestedSchema(), **{param: fields_list})

        assert MySchema().dump({"nested": {"foo": "baz", "bar": "bax"}}) == {
            "nested": {"foo": "baz"}
        }


class TestListNested:
    @pytest.mark.parametrize("param", ("only", "exclude", "dump_only", "load_only"))
    def test_list_nested_only_exclude_dump_only_load_only_propagated_to_nested(
        self, param
    ):
        class Child(Schema):
            name = fields.String()
            age = fields.Integer()

        class Family(Schema):
            children = fields.List(fields.Nested(Child))

        schema = Family(**{param: ["children.name"]})  # type: ignore[arg-type]
        children_field = schema.fields["children"]
        assert isinstance(children_field, fields.List)
        assert isinstance(children_field.inner, fields.Nested)
        assert getattr(children_field.inner.schema, param) == {"name"}

    @pytest.mark.parametrize(
        ("param", "expected_attribute", "expected_dump"),
        (
            ("only", {"name"}, {"children": [{"name": "Lily"}]}),
            ("exclude", {"name", "surname", "age"}, {"children": [{}]}),
        ),
    )
    def test_list_nested_class_only_and_exclude_merged_with_nested(
        self, param, expected_attribute, expected_dump
    ):
        class Child(Schema):
            name = fields.String()
            surname = fields.String()
            age = fields.Integer()

        class Family(Schema):
            children = fields.List(fields.Nested(Child, **{param: ("name", "surname")}))  # type: ignore[arg-type]

        schema = Family(**{param: ["children.name", "children.age"]})  # type: ignore[arg-type]
        children_field = schema.fields["children"]
        assert isinstance(children_field, fields.List)
        assert getattr(children_field.inner, param) == expected_attribute

        family = {"children": [{"name": "Lily", "surname": "Martinez", "age": 15}]}
        assert schema.dump(family) == expected_dump

    def test_list_nested_class_multiple_dumps(self):
        class Child(Schema):
            name = fields.String()
            surname = fields.String()
            age = fields.Integer()

        class Family(Schema):
            children = fields.List(fields.Nested(Child, only=("name", "age")))

        family = {"children": [{"name": "Lily", "surname": "Martinez", "age": 15}]}
        assert Family(only=("children.age",)).dump(family) == {
            "children": [{"age": 15}]
        }
        assert Family(only=("children.name",)).dump(family) == {
            "children": [{"name": "Lily"}]
        }

    @pytest.mark.parametrize(
        ("param", "expected_attribute", "expected_dump"),
        (
            ("only", {"name"}, {"children": [{"name": "Lily"}]}),
            ("exclude", {"name", "surname", "age"}, {"children": [{}]}),
        ),
    )
    def test_list_nested_instance_only_and_exclude_merged_with_nested(
        self, param, expected_attribute, expected_dump
    ):
        class Child(Schema):
            name = fields.String()
            surname = fields.String()
            age = fields.Integer()

        class Family(Schema):
            children = fields.List(fields.Nested(Child(**{param: ("name", "surname")})))  # type: ignore[arg-type]

        schema = Family(**{param: ["children.name", "children.age"]})  # type: ignore[arg-type]
        children_field = schema.fields["children"]
        assert isinstance(children_field, fields.List)
        assert isinstance(children_field.inner, fields.Nested)
        assert getattr(children_field.inner.schema, param) == expected_attribute

        family = {"children": [{"name": "Lily", "surname": "Martinez", "age": 15}]}
        assert schema.dump(family) == expected_dump

    def test_list_nested_instance_multiple_dumps(self):
        class Child(Schema):
            name = fields.String()
            surname = fields.String()
            age = fields.Integer()

        class Family(Schema):
            children = fields.List(fields.Nested(Child(only=("name", "age"))))

        family = {"children": [{"name": "Lily", "surname": "Martinez", "age": 15}]}
        assert Family(only=("children.age",)).dump(family) == {
            "children": [{"age": 15}]
        }
        assert Family(only=("children.name",)).dump(family) == {
            "children": [{"name": "Lily"}]
        }

    @pytest.mark.parametrize(
        ("param", "expected_attribute", "expected_dump"),
        (
            ("only", {"name"}, {"children": [{"name": "Lily"}]}),
            ("exclude", {"name", "surname", "age"}, {"children": [{}]}),
        ),
    )
    def test_list_nested_lambda_only_and_exclude_merged_with_nested(
        self, param, expected_attribute, expected_dump
    ):
        class Child(Schema):
            name = fields.String()
            surname = fields.String()
            age = fields.Integer()

        class Family(Schema):
            children = fields.List(
                fields.Nested(lambda: Child(**{param: ("name", "surname")}))  # type: ignore[arg-type]
            )

        schema = Family(**{param: ["children.name", "children.age"]})  # type: ignore[arg-type]
        children_field = schema.fields["children"]
        assert isinstance(children_field, fields.List)
        assert isinstance(children_field.inner, fields.Nested)
        assert getattr(children_field.inner.schema, param) == expected_attribute

        family = {"children": [{"name": "Lily", "surname": "Martinez", "age": 15}]}
        assert schema.dump(family) == expected_dump

    def test_list_nested_partial_propagated_to_nested(self):
        class Child(Schema):
            name = fields.String(required=True)
            age = fields.Integer(required=True)

        class Family(Schema):
            children = fields.List(fields.Nested(Child))

        payload = {"children": [{"name": "Lucette"}]}

        for val in (True, ("children.age",)):
            result = Family(partial=val).load(payload)
            assert result["children"][0]["name"] == "Lucette"
            result = Family().load(payload, partial=val)
            assert result["children"][0]["name"] == "Lucette"

        for val in (False, ("children.name",)):
            with pytest.raises(ValidationError) as excinfo:
                result = Family(partial=val).load(payload)
            assert excinfo.value.args[0] == {
                "children": {0: {"age": ["Missing data for required field."]}}
            }
            with pytest.raises(ValidationError) as excinfo:
                result = Family().load(payload, partial=val)
            assert excinfo.value.args[0] == {
                "children": {0: {"age": ["Missing data for required field."]}}
            }


class TestTupleNested:
    @pytest.mark.parametrize("param", ("dump_only", "load_only"))
    def test_tuple_nested_only_exclude_dump_only_load_only_propagated_to_nested(
        self, param
    ):
        class Child(Schema):
            name = fields.String()
            age = fields.Integer()

        class Family(Schema):
            children = fields.Tuple((fields.Nested(Child), fields.Nested(Child)))

        schema = Family(**{param: ["children.name"]})  # type: ignore[arg-type]
        children_field = schema.fields["children"]
        assert isinstance(children_field, fields.Tuple)
        field1, field2 = children_field.tuple_fields
        assert isinstance(field1, fields.Nested)
        assert isinstance(field2, fields.Nested)
        assert getattr(field1.schema, param) == {"name"}
        assert getattr(field2.schema, param) == {"name"}

    def test_tuple_nested_partial_propagated_to_nested(self):
        class Child(Schema):
            name = fields.String(required=True)
            age = fields.Integer(required=True)

        class Family(Schema):
            children = fields.Tuple((fields.Nested(Child),))

        payload = {"children": [{"name": "Lucette"}]}

        for val in (True, ("children.age",)):
            result = Family(partial=val).load(payload)
            assert result["children"][0]["name"] == "Lucette"
            result = Family().load(payload, partial=val)
            assert result["children"][0]["name"] == "Lucette"

        for val in (False, ("children.name",)):
            with pytest.raises(ValidationError) as excinfo:
                result = Family(partial=val).load(payload)
            assert excinfo.value.args[0] == {
                "children": {0: {"age": ["Missing data for required field."]}}
            }
            with pytest.raises(ValidationError) as excinfo:
                result = Family().load(payload, partial=val)
            assert excinfo.value.args[0] == {
                "children": {0: {"age": ["Missing data for required field."]}}
            }


class TestDictNested:
    @pytest.mark.parametrize("param", ("only", "exclude", "dump_only", "load_only"))
    def test_dict_nested_only_exclude_dump_only_load_only_propagated_to_nested(
        self, param
    ):
        class Child(Schema):
            name = fields.String()
            age = fields.Integer()

        class Family(Schema):
            children = fields.Dict(values=fields.Nested(Child))

        schema = Family(**{param: ["children.name"]})  # type: ignore[arg-type]
        children_field = schema.fields["children"]
        assert isinstance(children_field, fields.Dict)
        assert isinstance(children_field.value_field, fields.Nested)
        assert getattr(children_field.value_field.schema, param) == {"name"}

    @pytest.mark.parametrize(
        ("param", "expected"),
        (("only", {"name"}), ("exclude", {"name", "surname", "age"})),
    )
    def test_dict_nested_only_and_exclude_merged_with_nested(self, param, expected):
        class Child(Schema):
            name = fields.String()
            surname = fields.String()
            age = fields.Integer()

        class Family(Schema):
            children = fields.Dict(
                values=fields.Nested(Child, **{param: ("name", "surname")})  # type: ignore[arg-type]
            )

        schema = Family(**{param: ["children.name", "children.age"]})  # type: ignore[arg-type]
        children_field = schema.fields["children"]
        assert isinstance(children_field, fields.Dict)
        assert getattr(children_field.value_field, param) == expected

    def test_dict_nested_partial_propagated_to_nested(self):
        class Child(Schema):
            name = fields.String(required=True)
            age = fields.Integer(required=True)

        class Family(Schema):
            children = fields.Dict(values=fields.Nested(Child))

        payload = {"children": {"daughter": {"name": "Lucette"}}}

        for val in (True, ("children.age",)):
            result = Family(partial=val).load(payload)
            assert result["children"]["daughter"]["name"] == "Lucette"
            result = Family().load(payload, partial=val)
            assert result["children"]["daughter"]["name"] == "Lucette"

        for val in (False, ("children.name",)):
            with pytest.raises(ValidationError) as excinfo:
                result = Family(partial=val).load(payload)
            assert excinfo.value.args[0] == {
                "children": {
                    "daughter": {"value": {"age": ["Missing data for required field."]}}
                }
            }
            with pytest.raises(ValidationError) as excinfo:
                result = Family().load(payload, partial=val)
            assert excinfo.value.args[0] == {
                "children": {
                    "daughter": {"value": {"age": ["Missing data for required field."]}}
                }
            }


# === tests/foo_serializer.py ===
from marshmallow import Schema, fields


class FooSerializer(Schema):
    _id = fields.Integer()


# === tests/test_context.py ===
import typing

import pytest

from marshmallow import (
    Schema,
    fields,
    post_dump,
    post_load,
    pre_dump,
    pre_load,
    validates,
    validates_schema,
)
from marshmallow.exceptions import ValidationError
from marshmallow.experimental.context import Context
from tests.base import Blog, User


class UserContextSchema(Schema):
    is_owner = fields.Method("get_is_owner")
    is_collab = fields.Function(
        lambda user: user in Context[dict[str, typing.Any]].get()["blog"]
    )

    def get_is_owner(self, user):
        return Context.get()["blog"].user.name == user.name


class TestContext:
    def test_context_load_dump(self):
        class ContextField(fields.Integer):
            def _serialize(self, value, attr, obj, **kwargs):
                if (context := Context[dict].get(None)) is not None:
                    value *= context.get("factor", 1)
                return super()._serialize(value, attr, obj, **kwargs)

            def _deserialize(self, value, attr, data, **kwargs):
                val = super()._deserialize(value, attr, data, **kwargs)
                if (context := Context[dict].get(None)) is not None:
                    val *= context.get("factor", 1)
                return val

        class ContextSchema(Schema):
            ctx_fld = ContextField()

        ctx_schema = ContextSchema()

        assert ctx_schema.load({"ctx_fld": 1}) == {"ctx_fld": 1}
        assert ctx_schema.dump({"ctx_fld": 1}) == {"ctx_fld": 1}
        with Context({"factor": 2}):
            assert ctx_schema.load({"ctx_fld": 1}) == {"ctx_fld": 2}
            assert ctx_schema.dump({"ctx_fld": 1}) == {"ctx_fld": 2}

    def test_context_method(self):
        owner = User("Joe")
        blog = Blog(title="Joe Blog", user=owner)
        serializer = UserContextSchema()
        with Context({"blog": blog}):
            data = serializer.dump(owner)
            assert data["is_owner"] is True
            nonowner = User("Fred")
            data = serializer.dump(nonowner)
            assert data["is_owner"] is False

    def test_context_function(self):
        owner = User("Fred")
        blog = Blog("Killer Queen", user=owner)
        collab = User("Brian")
        blog.collaborators.append(collab)
        with Context({"blog": blog}):
            serializer = UserContextSchema()
            data = serializer.dump(collab)
            assert data["is_collab"] is True
            noncollab = User("Foo")
            data = serializer.dump(noncollab)
            assert data["is_collab"] is False

    def test_function_field_handles_bound_serializer(self):
        class SerializeA:
            def __call__(self, value):
                return "value"

        serialize = SerializeA()

        # only has a function field
        class UserFunctionContextSchema(Schema):
            is_collab = fields.Function(serialize)

        owner = User("Joe")
        serializer = UserFunctionContextSchema()
        data = serializer.dump(owner)
        assert data["is_collab"] == "value"

    def test_nested_fields_inherit_context(self):
        class InnerSchema(Schema):
            likes_bikes = fields.Function(lambda obj: "bikes" in Context.get()["info"])

        class CSchema(Schema):
            inner = fields.Nested(InnerSchema)

        ser = CSchema()
        with Context[dict]({"info": "i like bikes"}):
            obj: dict[str, dict] = {"inner": {}}
            result = ser.dump(obj)
            assert result["inner"]["likes_bikes"] is True

    # Regression test for https://github.com/marshmallow-code/marshmallow/issues/820
    def test_nested_list_fields_inherit_context(self):
        class InnerSchema(Schema):
            foo = fields.Raw()

            @validates("foo")
            def validate_foo(self, value, **kwargs):
                if "foo_context" not in Context[dict].get():
                    raise ValidationError("Missing context")

        class OuterSchema(Schema):
            bars = fields.List(fields.Nested(InnerSchema()))

        inner = InnerSchema()
        with Context({"foo_context": "foo"}):
            assert inner.load({"foo": 42})

        outer = OuterSchema()
        with Context({"foo_context": "foo"}):
            assert outer.load({"bars": [{"foo": 42}]})

    # Regression test for https://github.com/marshmallow-code/marshmallow/issues/820
    def test_nested_dict_fields_inherit_context(self):
        class InnerSchema(Schema):
            foo = fields.Raw()

            @validates("foo")
            def validate_foo(self, value, **kwargs):
                if "foo_context" not in Context[dict].get():
                    raise ValidationError("Missing context")

        class OuterSchema(Schema):
            bars = fields.Dict(values=fields.Nested(InnerSchema()))

        inner = InnerSchema()
        with Context({"foo_context": "foo"}):
            assert inner.load({"foo": 42})

        outer = OuterSchema()
        with Context({"foo_context": "foo"}):
            assert outer.load({"bars": {"test": {"foo": 42}}})

    # Regression test for https://github.com/marshmallow-code/marshmallow/issues/1404
    def test_nested_field_with_unpicklable_object_in_context(self):
        class Unpicklable:
            def __deepcopy__(self, _):
                raise NotImplementedError

        class InnerSchema(Schema):
            foo = fields.Raw()

        class OuterSchema(Schema):
            inner = fields.Nested(InnerSchema())

        outer = OuterSchema()
        obj = {"inner": {"foo": 42}}
        with Context({"unp": Unpicklable()}):
            assert outer.dump(obj)

    def test_function_field_passed_serialize_with_context(self, user):
        class Parent(Schema):
            pass

        field = fields.Function(
            serialize=lambda obj: obj.name.upper() + Context.get()["key"]
        )
        field.parent = Parent()
        with Context({"key": "BAR"}):
            assert field.serialize("key", user) == "MONTYBAR"

    def test_function_field_deserialization_with_context(self):
        class Parent(Schema):
            pass

        field = fields.Function(
            lambda x: None,
            deserialize=lambda val: val.upper() + Context.get()["key"],
        )
        field.parent = Parent()
        with Context({"key": "BAR"}):
            assert field.deserialize("foo") == "FOOBAR"

    def test_decorated_processors_with_context(self):
        NumDictContext = Context[dict[int, int]]

        class MySchema(Schema):
            f_1 = fields.Integer()
            f_2 = fields.Integer()
            f_3 = fields.Integer()
            f_4 = fields.Integer()

            @pre_dump
            def multiply_f_1(self, item, **kwargs):
                item["f_1"] *= NumDictContext.get()[1]
                return item

            @pre_load
            def multiply_f_2(self, data, **kwargs):
                data["f_2"] *= NumDictContext.get()[2]
                return data

            @post_dump
            def multiply_f_3(self, item, **kwargs):
                item["f_3"] *= NumDictContext.get()[3]
                return item

            @post_load
            def multiply_f_4(self, data, **kwargs):
                data["f_4"] *= NumDictContext.get()[4]
                return data

        schema = MySchema()

        with NumDictContext({1: 2, 2: 3, 3: 4, 4: 5}):
            assert schema.dump({"f_1": 1, "f_2": 1, "f_3": 1, "f_4": 1}) == {
                "f_1": 2,
                "f_2": 1,
                "f_3": 4,
                "f_4": 1,
            }
            assert schema.load({"f_1": 1, "f_2": 1, "f_3": 1, "f_4": 1}) == {
                "f_1": 1,
                "f_2": 3,
                "f_3": 1,
                "f_4": 5,
            }

    def test_validates_schema_with_context(self):
        class MySchema(Schema):
            f_1 = fields.Integer()
            f_2 = fields.Integer()

            @validates_schema
            def validate_schema(self, data, **kwargs):
                if data["f_2"] != data["f_1"] * Context.get():
                    raise ValidationError("Fail")

        schema = MySchema()

        with Context(2):
            schema.load({"f_1": 1, "f_2": 2})
            with pytest.raises(ValidationError) as excinfo:
                schema.load({"f_1": 1, "f_2": 3})
            assert excinfo.value.messages["_schema"] == ["Fail"]


# === tests/test_validate.py ===
"""Tests for marshmallow.validate"""

import re

import pytest

from marshmallow import ValidationError, validate


@pytest.mark.parametrize(
    "valid_url",
    [
        "http://example.org",
        "https://example.org",
        "ftp://example.org",
        "ftps://example.org",
        "http://example.co.jp",
        "http://www.example.com/a%C2%B1b",
        "http://www.example.com/~username/",
        "http://info.example.com/?fred",
        "http://xn--mgbh0fb.xn--kgbechtv/",
        "http://example.com/blue/red%3Fand+green",
        "http://www.example.com/?array%5Bkey%5D=value",
        "http://xn--rsum-bpad.example.org/",
        "http://123.45.67.8/",
        "http://123.45.67.8:8329/",
        "http://[2001:db8::ff00:42]:8329",
        "http://[2001::1]:8329",
        "http://www.example.com:8000/foo",
        "http://user@example.com",
        "http://user:pass@example.com",
        "http://:pass@example.com",
        "http://@example.com",
        "http://AZaz09-._~%2A!$&'()*+,;=:@example.com",
    ],
)
def test_url_absolute_valid(valid_url):
    validator = validate.URL(relative=False)
    assert validator(valid_url) == valid_url


@pytest.mark.parametrize(
    "invalid_url",
    [
        "http:///example.com/",
        "https:///example.com/",
        "https://example.org\\",
        "https://example.org\n",
        "ftp:///example.com/",
        "ftps:///example.com/",
        "http//example.org",
        "http:///",
        "http:/example.org",
        "foo://example.org",
        "../icons/logo.gif",
        "http://2001:db8::ff00:42:8329",
        "http://[192.168.1.1]:8329",
        "abc",
        "..",
        "/",
        " ",
        "",
        None,
        "http://user@pass@example.com",
        "http://@pass@example.com",
        "http://@@example.com",
        "http://^@example.com",
        "http://%0G@example.com",
        "http://%@example.com",
    ],
)
def test_url_absolute_invalid(invalid_url):
    validator = validate.URL(relative=False)
    with pytest.raises(ValidationError):
        validator(invalid_url)


@pytest.mark.parametrize(
    "valid_url",
    [
        "http://example.org",
        "http://123.45.67.8/",
        "http://example.com/foo/bar/../baz",
        "https://example.com/../icons/logo.gif",
        "http://example.com/./icons/logo.gif",
        "ftp://example.com/../../../../g",
        "http://example.com/g?y/./x",
        "/foo/bar",
        "/foo?bar",
        "/foo?bar#baz",
    ],
)
def test_url_relative_valid(valid_url):
    validator = validate.URL(relative=True)
    assert validator(valid_url) == valid_url


@pytest.mark.parametrize(
    "invalid_url",
    [
        "http//example.org",
        "http://example.org\n",
        "suppliers.html",
        "../icons/logo.gif",
        "icons/logo.gif",
        "../.../g",
        "...",
        "\\",
        " ",
        "",
        None,
    ],
)
def test_url_relative_invalid(invalid_url):
    validator = validate.URL(relative=True)
    with pytest.raises(ValidationError):
        validator(invalid_url)


@pytest.mark.parametrize(
    "valid_url",
    [
        "/foo/bar",
        "/foo?bar",
        "?bar",
        "/foo?bar#baz",
    ],
)
def test_url_relative_only_valid(valid_url):
    validator = validate.URL(relative=True, absolute=False)
    assert validator(valid_url) == valid_url


@pytest.mark.parametrize(
    "invalid_url",
    [
        "http//example.org",
        "http://example.org\n",
        "suppliers.html",
        "../icons/logo.gif",
        "icons/logo.gif",
        "../.../g",
        "...",
        "\\",
        " ",
        "",
        "http://example.org",
        "http://123.45.67.8/",
        "http://example.com/foo/bar/../baz",
        "https://example.com/../icons/logo.gif",
        "http://example.com/./icons/logo.gif",
        "ftp://example.com/../../../../g",
        "http://example.com/g?y/./x",
    ],
)
def test_url_relative_only_invalid(invalid_url):
    validator = validate.URL(relative=True, absolute=False)
    with pytest.raises(ValidationError):
        validator(invalid_url)


@pytest.mark.parametrize(
    "valid_url",
    [
        "http://example.org",
        "http://123.45.67.8/",
        "http://example",
        "http://example.",
        "http://example:80",
        "http://user.name:pass.word@example",
        "http://example/foo/bar",
    ],
)
def test_url_dont_require_tld_valid(valid_url):
    validator = validate.URL(require_tld=False)
    assert validator(valid_url) == valid_url


@pytest.mark.parametrize(
    "invalid_url",
    [
        "http//example",
        "http://example\n",
        "http://.example.org",
        "http:///foo/bar",
        "http:// /foo/bar",
        "",
        None,
    ],
)
def test_url_dont_require_tld_invalid(invalid_url):
    validator = validate.URL(require_tld=False)
    with pytest.raises(ValidationError):
        validator(invalid_url)


def test_url_custom_scheme():
    validator = validate.URL()
    # By default, ws not allowed
    url = "ws://test.test"
    with pytest.raises(ValidationError):
        validator(url)

    validator = validate.URL(schemes={"http", "https", "ws"})
    assert validator(url) == url


@pytest.mark.parametrize(
    "valid_url",
    (
        "file:///tmp/tmp1234",
        "file://localhost/tmp/tmp1234",
        "file:///C:/Users/test/file.txt",
        "file://localhost/C:/Program%20Files/file.exe",
        "file:///home/user/documents/test.pdf",
        "file:///tmp/test%20file.txt",
        "file:///",
        "file://localhost/",
    ),
)
def test_url_accepts_valid_file_urls(valid_url):
    validator = validate.URL(schemes={"file"})
    assert validator(valid_url) == valid_url


@pytest.mark.parametrize(
    "invalid_url",
    (
        "file://",
        "file:/tmp/file.txt",
        "file:tmp/file.txt",
        "file://hostname/path",
        "file:///tmp/test file.txt",
    ),
)
def test_url_rejects_invalid_file_urls(invalid_url):
    validator = validate.URL(schemes={"file"})
    with pytest.raises(ValidationError, match="Not a valid URL."):
        assert validator(invalid_url)


def test_url_relative_and_custom_schemes():
    validator = validate.URL(relative=True)
    # By default, ws not allowed
    url = "ws://test.test"
    with pytest.raises(ValidationError):
        validator(url)

    validator = validate.URL(relative=True, schemes={"http", "https", "ws"})
    assert validator(url) == url


def test_url_custom_message():
    validator = validate.URL(error="{input} ain't an URL")
    with pytest.raises(ValidationError, match="invalid ain't an URL"):
        validator("invalid")


def test_url_repr():
    assert repr(
        validate.URL(relative=False, error=None)
    ) == "<URL(relative=False, absolute=True, error={!r})>".format("Not a valid URL.")
    assert repr(
        validate.URL(relative=True, error="foo")
    ) == "<URL(relative=True, absolute=True, error={!r})>".format("foo")
    assert repr(
        validate.URL(relative=True, absolute=False, error="foo")
    ) == "<URL(relative=True, absolute=False, error={!r})>".format("foo")


def test_url_rejects_invalid_relative_usage():
    with pytest.raises(
        ValueError,
        match="URL validation cannot set both relative and absolute to False",
    ):
        validate.URL(relative=False, absolute=False)


@pytest.mark.parametrize(
    "valid_email",
    [
        "niceandsimple@example.com",
        "NiCeAnDsImPlE@eXaMpLe.CoM",
        "very.common@example.com",
        "a.little.lengthy.but.fine@a.iana-servers.net",
        "disposable.style.email.with+symbol@example.com",
        '"very.unusual.@.unusual.com"@example.com',
        "!#$%&'*+-/=?^_`{}|~@example.org",
        "niceandsimple@[64.233.160.0]",
        "niceandsimple@localhost",
        "josé@blah.com",
        "δοκ.ιμή@παράδειγμα.δοκιμή",
    ],
)
def test_email_valid(valid_email):
    validator = validate.Email()
    assert validator(valid_email) == valid_email


@pytest.mark.parametrize(
    "invalid_email",
    [
        "niceandsimple\n@example.com",
        "NiCeAnDsImPlE@eXaMpLe.CoM\n",
        'a"b(c)d,e:f;g<h>i[j\\k]l@example.com',
        'just"not"right@example.com',
        'this is"not\allowed@example.com',
        'this\\ still\\"not\\\\allowed@example.com',
        '"much.more unusual"@example.com',
        '"very.(),:;<>[]".VERY."very@\\ "very".unusual"@strange.example.com',
        '" "@example.org',
        "user@example",
        "@nouser.com",
        "example.com",
        "user",
        "",
        None,
    ],
)
def test_email_invalid(invalid_email):
    validator = validate.Email()
    with pytest.raises(ValidationError):
        validator(invalid_email)


def test_email_custom_message():
    validator = validate.Email(error="{input} is not an email addy.")
    with pytest.raises(ValidationError, match="invalid is not an email addy."):
        validator("invalid")


def test_email_repr():
    assert repr(validate.Email(error=None)) == "<Email(error={!r})>".format(
        "Not a valid email address."
    )
    assert repr(validate.Email(error="foo")) == "<Email(error={!r})>".format("foo")


def test_range_min():
    assert validate.Range(1, 2)(1) == 1
    assert validate.Range(0)(1) == 1
    assert validate.Range()(1) == 1
    assert validate.Range(min_inclusive=False, max_inclusive=False)(1) == 1
    assert validate.Range(1, 1)(1) == 1

    with pytest.raises(ValidationError, match="Must be greater than or equal to 2"):
        validate.Range(2, 3)(1)
    with pytest.raises(ValidationError, match="Must be greater than or equal to 2"):
        validate.Range(2)(1)
    with pytest.raises(ValidationError, match="Must be greater than 1"):
        validate.Range(1, 2, min_inclusive=False, max_inclusive=True, error=None)(1)
    with pytest.raises(ValidationError, match="less than 1"):
        validate.Range(1, 1, min_inclusive=True, max_inclusive=False, error=None)(1)


def test_range_max():
    assert validate.Range(1, 2)(2) == 2
    assert validate.Range(None, 2)(2) == 2
    assert validate.Range()(2) == 2
    assert validate.Range(min_inclusive=False, max_inclusive=False)(2) == 2
    assert validate.Range(2, 2)(2) == 2

    with pytest.raises(ValidationError, match="less than or equal to 1"):
        validate.Range(0, 1)(2)
    with pytest.raises(ValidationError, match="less than or equal to 1"):
        validate.Range(None, 1)(2)
    with pytest.raises(ValidationError, match="less than 2"):
        validate.Range(1, 2, min_inclusive=True, max_inclusive=False, error=None)(2)
    with pytest.raises(ValidationError, match="greater than 2"):
        validate.Range(2, 2, min_inclusive=False, max_inclusive=True, error=None)(2)


def test_range_custom_message():
    v = validate.Range(2, 3, error="{input} is not between {min} and {max}")
    with pytest.raises(ValidationError, match="1 is not between 2 and 3"):
        v(1)

    v = validate.Range(2, None, error="{input} is less than {min}")
    with pytest.raises(ValidationError, match="1 is less than 2"):
        v(1)

    v = validate.Range(None, 3, error="{input} is greater than {max}")
    with pytest.raises(ValidationError, match="4 is greater than 3"):
        v(4)


def test_range_repr():
    assert (
        repr(
            validate.Range(
                min=None, max=None, error=None, min_inclusive=True, max_inclusive=True
            )
        )
        == "<Range(min=None, max=None, min_inclusive=True, max_inclusive=True, error=None)>"
    )
    assert (
        repr(
            validate.Range(
                min=1, max=3, error="foo", min_inclusive=False, max_inclusive=False
            )
        )
        == "<Range(min=1, max=3, min_inclusive=False, max_inclusive=False, error={!r})>".format(
            "foo"
        )
    )


def test_length_min():
    assert validate.Length(3, 5)("foo") == "foo"
    assert validate.Length(3, 5)([1, 2, 3]) == [1, 2, 3]
    assert validate.Length(0)("a") == "a"
    assert validate.Length(0)([1]) == [1]
    assert validate.Length()("") == ""
    assert validate.Length()([]) == []
    assert validate.Length(1, 1)("a") == "a"
    assert validate.Length(1, 1)([1]) == [1]

    with pytest.raises(ValidationError):
        validate.Length(4, 5)("foo")
    with pytest.raises(ValidationError):
        validate.Length(4, 5)([1, 2, 3])
    with pytest.raises(ValidationError):
        validate.Length(5)("foo")
    with pytest.raises(ValidationError):
        validate.Length(5)([1, 2, 3])


def test_length_max():
    assert validate.Length(1, 3)("foo") == "foo"
    assert validate.Length(1, 3)([1, 2, 3]) == [1, 2, 3]
    assert validate.Length(None, 1)("a") == "a"
    assert validate.Length(None, 1)([1]) == [1]
    assert validate.Length()("") == ""
    assert validate.Length()([]) == []
    assert validate.Length(2, 2)("ab") == "ab"
    assert validate.Length(2, 2)([1, 2]) == [1, 2]

    with pytest.raises(ValidationError):
        validate.Length(1, 2)("foo")
    with pytest.raises(ValidationError):
        validate.Length(1, 2)([1, 2, 3])
    with pytest.raises(ValidationError):
        validate.Length(None, 2)("foo")
    with pytest.raises(ValidationError):
        validate.Length(None, 2)([1, 2, 3])


def test_length_equal():
    assert validate.Length(equal=3)("foo") == "foo"
    assert validate.Length(equal=3)([1, 2, 3]) == [1, 2, 3]
    assert validate.Length(equal=None)("") == ""
    assert validate.Length(equal=None)([]) == []

    with pytest.raises(ValidationError):
        validate.Length(equal=2)("foo")
    with pytest.raises(ValidationError):
        validate.Length(equal=2)([1, 2, 3])
    error_message = "The `equal` parameter was provided, maximum or minimum parameter must not be provided"
    with pytest.raises(ValueError, match=error_message):
        validate.Length(1, None, equal=3)("foo")
    with pytest.raises(ValueError, match=error_message):
        validate.Length(None, 5, equal=3)("foo")
    with pytest.raises(ValueError, match=error_message):
        validate.Length(1, 5, equal=3)("foo")


def test_length_custom_message():
    v = validate.Length(5, 6, error="{input} is not between {min} and {max}")
    with pytest.raises(ValidationError, match="foo is not between 5 and 6"):
        v("foo")

    v = validate.Length(5, None, error="{input} is shorter than {min}")
    with pytest.raises(ValidationError, match="foo is shorter than 5"):
        v("foo")

    v = validate.Length(None, 2, error="{input} is longer than {max}")
    with pytest.raises(ValidationError, match="foo is longer than 2"):
        v("foo")

    v = validate.Length(None, None, equal=4, error="{input} does not have {equal}")
    with pytest.raises(ValidationError, match="foo does not have 4"):
        v("foo")


def test_length_repr():
    assert (
        repr(validate.Length(min=None, max=None, error=None, equal=None))
        == "<Length(min=None, max=None, equal=None, error=None)>"
    )
    assert repr(
        validate.Length(min=1, max=3, error="foo", equal=None)
    ) == "<Length(min=1, max=3, equal=None, error={!r})>".format("foo")
    assert repr(
        validate.Length(min=None, max=None, error="foo", equal=5)
    ) == "<Length(min=None, max=None, equal=5, error={!r})>".format("foo")


def test_equal():
    assert validate.Equal("a")("a") == "a"
    assert validate.Equal(1)(1) == 1
    assert validate.Equal([1])([1]) == [1]

    with pytest.raises(ValidationError):
        validate.Equal("b")("a")
    with pytest.raises(ValidationError):
        validate.Equal(2)(1)
    with pytest.raises(ValidationError):
        validate.Equal([2])([1])


def test_equal_custom_message():
    v = validate.Equal("a", error="{input} is not equal to {other}.")
    with pytest.raises(ValidationError, match="b is not equal to a."):
        v("b")


def test_equal_repr():
    assert repr(
        validate.Equal(comparable=123, error=None)
    ) == "<Equal(comparable=123, error={!r})>".format("Must be equal to {other}.")
    assert repr(
        validate.Equal(comparable=123, error="foo")
    ) == "<Equal(comparable=123, error={!r})>".format("foo")


def test_regexp_str():
    assert validate.Regexp(r"a")("a") == "a"
    assert validate.Regexp(r"\w")("_") == "_"
    assert validate.Regexp(r"\s")(" ") == " "
    assert validate.Regexp(r"1")("1") == "1"
    assert validate.Regexp(r"[0-9]+")("1") == "1"
    assert validate.Regexp(r"a", re.IGNORECASE)("A") == "A"

    with pytest.raises(ValidationError):
        validate.Regexp(r"[0-9]+")("a")
    with pytest.raises(ValidationError):
        validate.Regexp(r"[a-z]+")("1")
    with pytest.raises(ValidationError):
        validate.Regexp(r"a")("A")


def test_regexp_compile():
    assert validate.Regexp(re.compile(r"a"))("a") == "a"
    assert validate.Regexp(re.compile(r"\w"))("_") == "_"
    assert validate.Regexp(re.compile(r"\s"))(" ") == " "
    assert validate.Regexp(re.compile(r"1"))("1") == "1"
    assert validate.Regexp(re.compile(r"[0-9]+"))("1") == "1"
    assert validate.Regexp(re.compile(r"a", re.IGNORECASE))("A") == "A"
    assert validate.Regexp(re.compile(r"a", re.IGNORECASE), re.IGNORECASE)("A") == "A"

    with pytest.raises(ValidationError):
        validate.Regexp(re.compile(r"[0-9]+"))("a")
    with pytest.raises(ValidationError):
        validate.Regexp(re.compile(r"[a-z]+"))("1")
    with pytest.raises(ValidationError):
        validate.Regexp(re.compile(r"a"))("A")
    with pytest.raises(ValidationError):
        validate.Regexp(re.compile(r"a"), re.IGNORECASE)("A")


def test_regexp_custom_message():
    rex = r"[0-9]+"
    v = validate.Regexp(rex, error="{input} does not match {regex}")
    with pytest.raises(ValidationError, match="a does not match"):
        v("a")


def test_regexp_repr():
    assert repr(
        validate.Regexp(regex="abc", flags=0, error=None)
    ) == "<Regexp(regex={!r}, error={!r})>".format(
        re.compile("abc"), "String does not match expected pattern."
    )
    assert repr(
        validate.Regexp(regex="abc", flags=re.IGNORECASE, error="foo")
    ) == "<Regexp(regex={!r}, error={!r})>".format(
        re.compile("abc", re.IGNORECASE), "foo"
    )


def test_predicate():
    class Dummy:
        def _true(self):
            return True

        def _false(self):
            return False

        def _list(self):
            return [1, 2, 3]

        def _empty(self):
            return []

        def _identity(self, arg):
            return arg

    d = Dummy()

    assert validate.Predicate("_true")(d) == d
    assert validate.Predicate("_list")(d) == d
    assert validate.Predicate("_identity", arg=True)(d) == d
    assert validate.Predicate("_identity", arg=1)(d) == d
    assert validate.Predicate("_identity", arg="abc")(d) == d

    with pytest.raises(ValidationError, match="Invalid input."):
        validate.Predicate("_false")(d)
    with pytest.raises(ValidationError):
        validate.Predicate("_empty")(d)
    with pytest.raises(ValidationError):
        validate.Predicate("_identity", arg=False)(d)
    with pytest.raises(ValidationError):
        validate.Predicate("_identity", arg=0)(d)
    with pytest.raises(ValidationError):
        validate.Predicate("_identity", arg="")(d)


def test_predicate_custom_message():
    class Dummy:
        def _false(self):
            return False

        def __str__(self):
            return "Dummy"

    d = Dummy()
    with pytest.raises(ValidationError, match="Dummy._false is invalid!"):
        validate.Predicate("_false", error="{input}.{method} is invalid!")(d)


def test_predicate_repr():
    assert repr(
        validate.Predicate(method="foo", error=None)
    ) == "<Predicate(method={!r}, kwargs={!r}, error={!r})>".format(
        "foo", {}, "Invalid input."
    )
    assert repr(
        validate.Predicate(method="foo", error="bar", zoo=1)
    ) == "<Predicate(method={!r}, kwargs={!r}, error={!r})>".format(
        "foo", {"zoo": 1}, "bar"
    )


def test_noneof():
    assert validate.NoneOf([1, 2, 3])(4) == 4
    assert validate.NoneOf("abc")("d") == "d"
    assert validate.NoneOf("")([]) == []
    assert validate.NoneOf([])("") == ""
    assert validate.NoneOf([])([]) == []
    assert validate.NoneOf([1, 2, 3])(None) is None

    with pytest.raises(ValidationError, match="Invalid input."):
        validate.NoneOf([1, 2, 3])(3)
    with pytest.raises(ValidationError):
        validate.NoneOf("abc")("c")
    with pytest.raises(ValidationError):
        validate.NoneOf([1, 2, None])(None)
    with pytest.raises(ValidationError):
        validate.NoneOf("")("")


def test_noneof_custom_message():
    with pytest.raises(ValidationError, match="<not valid>"):
        validate.NoneOf([1, 2], error="<not valid>")(1)

    none_of = validate.NoneOf([1, 2], error="{input} cannot be one of {values}")
    with pytest.raises(ValidationError, match="1 cannot be one of 1, 2"):
        none_of(1)


def test_noneof_repr():
    assert repr(
        validate.NoneOf(iterable=[1, 2, 3], error=None)
    ) == "<NoneOf(iterable=[1, 2, 3], error={!r})>".format("Invalid input.")
    assert repr(
        validate.NoneOf(iterable=[1, 2, 3], error="foo")
    ) == "<NoneOf(iterable=[1, 2, 3], error={!r})>".format("foo")


def test_oneof():
    assert validate.OneOf([1, 2, 3])(2) == 2
    assert validate.OneOf("abc")("b") == "b"
    assert validate.OneOf("")("") == ""
    assert validate.OneOf(dict(a=0, b=1))("a") == "a"
    assert validate.OneOf((1, 2, None))(None) is None

    with pytest.raises(ValidationError, match="Must be one of: 1, 2, 3."):
        validate.OneOf([1, 2, 3])(4)
    with pytest.raises(ValidationError):
        validate.OneOf("abc")("d")
    with pytest.raises(ValidationError):
        validate.OneOf((1, 2, 3))(None)
    with pytest.raises(ValidationError):
        validate.OneOf([])([])
    with pytest.raises(ValidationError):
        validate.OneOf(())(())
    with pytest.raises(ValidationError):
        validate.OneOf(dict(a=0, b=1))(0)
    with pytest.raises(ValidationError):
        validate.OneOf("123")(1)


def test_oneof_options():
    oneof = validate.OneOf([1, 2, 3], ["one", "two", "three"])
    expected = [("1", "one"), ("2", "two"), ("3", "three")]
    assert list(oneof.options()) == expected

    oneof = validate.OneOf([1, 2, 3], ["one", "two"])
    expected = [("1", "one"), ("2", "two"), ("3", "")]
    assert list(oneof.options()) == expected

    oneof = validate.OneOf([1, 2], ["one", "two", "three"])
    expected = [("1", "one"), ("2", "two"), ("", "three")]
    assert list(oneof.options()) == expected

    oneof = validate.OneOf([1, 2])
    expected = [("1", ""), ("2", "")]
    assert list(oneof.options()) == expected


def test_oneof_text():
    oneof = validate.OneOf([1, 2, 3], ["one", "two", "three"])
    assert oneof.choices_text == "1, 2, 3"
    assert oneof.labels_text == "one, two, three"

    oneof = validate.OneOf([1], ["one"])
    assert oneof.choices_text == "1"
    assert oneof.labels_text == "one"

    oneof = validate.OneOf(dict(a=0, b=1))
    assert ", ".join(sorted(oneof.choices_text.split(", "))) == "a, b"
    assert oneof.labels_text == ""


def test_oneof_custom_message():
    oneof = validate.OneOf([1, 2, 3], error="{input} is not one of {choices}")
    expected = "4 is not one of 1, 2, 3"
    with pytest.raises(ValidationError):
        oneof(4)
    assert expected in str(expected)

    oneof = validate.OneOf(
        [1, 2, 3], ["one", "two", "three"], error="{input} is not one of {labels}"
    )
    expected = "4 is not one of one, two, three"
    with pytest.raises(ValidationError):
        oneof(4)
    assert expected in str(expected)


def test_oneof_repr():
    assert repr(
        validate.OneOf(choices=[1, 2, 3], labels=None, error=None)
    ) == "<OneOf(choices=[1, 2, 3], labels=[], error={!r})>".format(
        "Must be one of: {choices}."
    )
    assert repr(
        validate.OneOf(choices=[1, 2, 3], labels=["a", "b", "c"], error="foo")
    ) == "<OneOf(choices=[1, 2, 3], labels={!r}, error={!r})>".format(
        ["a", "b", "c"], "foo"
    )


def test_containsonly_in_list():
    assert validate.ContainsOnly([])([]) == []
    assert validate.ContainsOnly([1, 2, 3])([1]) == [1]
    assert validate.ContainsOnly([1, 1, 2])([1, 1]) == [1, 1]
    assert validate.ContainsOnly([1, 2, 3])([1, 2]) == [1, 2]
    assert validate.ContainsOnly([1, 2, 3])([2, 1]) == [2, 1]
    assert validate.ContainsOnly([1, 2, 3])([1, 2, 3]) == [1, 2, 3]
    assert validate.ContainsOnly([1, 2, 3])([3, 1, 2]) == [3, 1, 2]
    assert validate.ContainsOnly([1, 2, 3])([2, 3, 1]) == [2, 3, 1]
    assert validate.ContainsOnly([1, 2, 3])([1, 2, 3, 1]) == [1, 2, 3, 1]
    assert validate.ContainsOnly([1, 2, 3])([]) == []

    with pytest.raises(
        ValidationError,
        match="One or more of the choices you made was not in: 1, 2, 3.",
    ):
        validate.ContainsOnly([1, 2, 3])([4])
    with pytest.raises(ValidationError):
        validate.ContainsOnly([])([1])


def test_contains_only_unhashable_types():
    assert validate.ContainsOnly([[1], [2], [3]])([[1]]) == [[1]]
    assert validate.ContainsOnly([[1], [1], [2]])([[1], [1]]) == [[1], [1]]
    assert validate.ContainsOnly([[1], [2], [3]])([[1], [2]]) == [[1], [2]]
    assert validate.ContainsOnly([[1], [2], [3]])([[2], [1]]) == [[2], [1]]
    assert validate.ContainsOnly([[1], [2], [3]])([[1], [2], [3]]) == [[1], [2], [3]]
    assert validate.ContainsOnly([[1], [2], [3]])([[3], [1], [2]]) == [[3], [1], [2]]
    assert validate.ContainsOnly([[1], [2], [3]])([[2], [3], [1]]) == [[2], [3], [1]]
    assert validate.ContainsOnly([[1], [2], [3]])([]) == []

    with pytest.raises(ValidationError):
        validate.ContainsOnly([[1], [2], [3]])([[4]])
    with pytest.raises(ValidationError):
        validate.ContainsOnly([])([1])


def test_containsonly_in_tuple():
    assert validate.ContainsOnly(())(()) == ()
    assert validate.ContainsOnly((1, 2, 3))((1,)) == (1,)
    assert validate.ContainsOnly((1, 1, 2))((1, 1)) == (1, 1)
    assert validate.ContainsOnly((1, 2, 3))((1, 2)) == (1, 2)
    assert validate.ContainsOnly((1, 2, 3))((2, 1)) == (2, 1)
    assert validate.ContainsOnly((1, 2, 3))((1, 2, 3)) == (1, 2, 3)
    assert validate.ContainsOnly((1, 2, 3))((3, 1, 2)) == (3, 1, 2)
    assert validate.ContainsOnly((1, 2, 3))((2, 3, 1)) == (2, 3, 1)
    assert validate.ContainsOnly((1, 2, 3))(()) == tuple()
    with pytest.raises(ValidationError):
        validate.ContainsOnly((1, 2, 3))((4,))
    with pytest.raises(ValidationError):
        validate.ContainsOnly(())((1,))


def test_contains_only_in_string():
    assert validate.ContainsOnly("")("") == ""
    assert validate.ContainsOnly("abc")("a") == "a"
    assert validate.ContainsOnly("aab")("aa") == "aa"
    assert validate.ContainsOnly("abc")("ab") == "ab"
    assert validate.ContainsOnly("abc")("ba") == "ba"
    assert validate.ContainsOnly("abc")("abc") == "abc"
    assert validate.ContainsOnly("abc")("cab") == "cab"
    assert validate.ContainsOnly("abc")("bca") == "bca"
    assert validate.ContainsOnly("abc")("") == ""

    with pytest.raises(ValidationError):
        validate.ContainsOnly("abc")("d")
    with pytest.raises(ValidationError):
        validate.ContainsOnly("")("a")


def test_containsonly_custom_message():
    containsonly = validate.ContainsOnly(
        [1, 2, 3], error="{input} is not one of {choices}"
    )
    expected = "4, 5 is not one of 1, 2, 3"
    with pytest.raises(ValidationError):
        containsonly([4, 5])
    assert expected in str(expected)

    containsonly = validate.ContainsOnly(
        [1, 2, 3], ["one", "two", "three"], error="{input} is not one of {labels}"
    )
    expected = "4, 5 is not one of one, two, three"
    with pytest.raises(ValidationError):
        containsonly([4, 5])
    assert expected in str(expected)


def test_containsonly_repr():
    assert repr(
        validate.ContainsOnly(choices=[1, 2, 3], labels=None, error=None)
    ) == "<ContainsOnly(choices=[1, 2, 3], labels=[], error={!r})>".format(
        "One or more of the choices you made was not in: {choices}."
    )
    assert repr(
        validate.ContainsOnly(choices=[1, 2, 3], labels=["a", "b", "c"], error="foo")
    ) == "<ContainsOnly(choices=[1, 2, 3], labels={!r}, error={!r})>".format(
        ["a", "b", "c"], "foo"
    )


def test_containsnoneof_error_message():
    with pytest.raises(
        ValidationError, match="One or more of the choices you made was in: 1"
    ):
        validate.ContainsNoneOf([1])([1])

    with pytest.raises(
        ValidationError, match="One or more of the choices you made was in: 1, 2, 3"
    ):
        validate.ContainsNoneOf([1, 2, 3])([1])

    with pytest.raises(
        ValidationError, match="One or more of the choices you made was in: one, two"
    ):
        validate.ContainsNoneOf(["one", "two"])(["one"])

    with pytest.raises(
        ValidationError, match="One or more of the choices you made was in: @, !, &, ?"
    ):
        validate.ContainsNoneOf("@!&?")("@")


def test_containsnoneof_in_list():
    assert validate.ContainsNoneOf([])([]) == []
    assert validate.ContainsNoneOf([])([1, 2, 3]) == [1, 2, 3]
    assert validate.ContainsNoneOf([4])([1, 2, 3]) == [1, 2, 3]
    assert validate.ContainsNoneOf([2])([1, 3, 4]) == [1, 3, 4]
    assert validate.ContainsNoneOf([1, 2, 3])([4]) == [4]
    assert validate.ContainsNoneOf([4])([1, 1, 1, 1]) == [1, 1, 1, 1]

    with pytest.raises(ValidationError):
        validate.ContainsNoneOf([1])([1, 2, 3])

    with pytest.raises(ValidationError):
        validate.ContainsNoneOf([1, 1, 1])([1, 2, 3])

    with pytest.raises(ValidationError):
        validate.ContainsNoneOf([1, 2])([1, 2])

    with pytest.raises(ValidationError):
        validate.ContainsNoneOf([1])([1, 1, 1, 1])


def test_containsnoneof_unhashable_types():
    assert validate.ContainsNoneOf([[1], [2], [3]])([]) == []
    assert validate.ContainsNoneOf([[1], [2], [3]])([[4]]) == [[4]]
    assert validate.ContainsNoneOf([[1], [2], [3]])([[4], [4]]) == [[4], [4]]
    assert validate.ContainsNoneOf([[1], [2], [3]])([[4], [5]]) == [[4], [5]]

    with pytest.raises(ValidationError):
        validate.ContainsNoneOf([[1], [2], [3]])([[1]])

    with pytest.raises(ValidationError):
        validate.ContainsNoneOf([[1], [2], [3]])([[1], [2]])

    with pytest.raises(ValidationError):
        validate.ContainsNoneOf([[1], [2], [3]])([[2], [1]])

    with pytest.raises(ValidationError):
        validate.ContainsNoneOf([[1], [2], [3]])([[1], [2], [3]])

    with pytest.raises(ValidationError):
        validate.ContainsNoneOf([[1], [2], [3]])([[3], [2], [1]])

    with pytest.raises(ValidationError):
        validate.ContainsNoneOf([[1], [2], [3]])([[2], [3], [1]])


def test_containsnoneof_in_tuple():
    assert validate.ContainsNoneOf(())(()) == ()
    assert validate.ContainsNoneOf(())((1, 2, 3)) == (1, 2, 3)
    assert validate.ContainsNoneOf((4,))((1, 2, 3)) == (1, 2, 3)
    assert validate.ContainsNoneOf((2,))((1, 3, 4)) == (1, 3, 4)
    assert validate.ContainsNoneOf((1, 2, 3))((4,)) == (4,)
    assert validate.ContainsNoneOf((4,))((1, 1, 1, 1)) == (1, 1, 1, 1)

    with pytest.raises(ValidationError):
        validate.ContainsNoneOf((1,))((1, 2, 3))

    with pytest.raises(ValidationError):
        validate.ContainsNoneOf((1, 1, 1))((1, 2, 3))

    with pytest.raises(ValidationError):
        validate.ContainsNoneOf((1, 2))((1, 2))

    with pytest.raises(ValidationError):
        validate.ContainsNoneOf((1,))((1, 1, 1, 1))


def test_containsnoneof_in_string():
    assert validate.ContainsNoneOf("")("") == ""
    assert validate.ContainsNoneOf("")("abc") == "abc"
    assert validate.ContainsNoneOf("d")("abc") == "abc"
    assert validate.ContainsNoneOf("b")("acd") == "acd"
    assert validate.ContainsNoneOf("abc")("d") == "d"
    assert validate.ContainsNoneOf("d")("aaaa") == "aaaa"

    with pytest.raises(ValidationError):
        validate.ContainsNoneOf("a")("abc")

    with pytest.raises(ValidationError):
        validate.ContainsNoneOf("aaa")("abc")

    with pytest.raises(ValidationError):
        validate.ContainsNoneOf("ab")("ab")

    with pytest.raises(ValidationError):
        validate.ContainsNoneOf("a")("aaaa")


def test_containsnoneof_custom_message():
    validator = validate.ContainsNoneOf(
        [1, 2, 3], error="{input} was in the banned list: {values}"
    )
    expected = "1 was in the banned list: 1, 2, 3"
    with pytest.raises(ValidationError, match=expected):
        validator([1])


def test_containsnoneof_mixing_types():
    with pytest.raises(ValidationError):
        validate.ContainsNoneOf("abc")(["a"])

    with pytest.raises(ValidationError):
        validate.ContainsNoneOf(["a", "b", "c"])("a")

    with pytest.raises(ValidationError):
        validate.ContainsNoneOf((1, 2, 3))([1])

    with pytest.raises(ValidationError):
        validate.ContainsNoneOf([1, 2, 3])((1,))


def is_even(value):
    if value % 2 != 0:
        raise ValidationError("Not an even value.")


def test_and():
    validator = validate.And(validate.Range(min=0), is_even)
    assert validator(2)
    with pytest.raises(ValidationError) as excinfo:
        validator(-1)
    errors = excinfo.value.messages
    assert errors == ["Must be greater than or equal to 0.", "Not an even value."]

    validator_with_composition = validate.And(validator, validate.Range(max=6))
    assert validator_with_composition(4)
    with pytest.raises(ValidationError) as excinfo:
        validator_with_composition(7)

    errors = excinfo.value.messages
    assert errors == ["Not an even value.", "Must be less than or equal to 6."]


# === tests/test_serialization.py ===
"""Tests for field serialization."""

import datetime as dt
import decimal
import ipaddress
import itertools
import math
import uuid
from collections import OrderedDict
from typing import NamedTuple

import pytest

from marshmallow import Schema, fields
from marshmallow import missing as missing_
from tests.base import ALL_FIELDS, DateEnum, GenderEnum, HairColorEnum, User, central


class Point(NamedTuple):
    x: int
    y: int


class DateTimeList:
    def __init__(self, dtimes):
        self.dtimes = dtimes


class IntegerList:
    def __init__(self, ints):
        self.ints = ints


class DateTimeIntegerTuple:
    def __init__(self, dtime_int):
        self.dtime_int = dtime_int


class TestFieldSerialization:
    @pytest.fixture
    def user(self):
        return User("Foo", email="foo@bar.com", age=42)

    def test_function_field_passed_func(self, user):
        field = fields.Function(lambda obj: obj.name.upper())
        assert field.serialize("key", user) == "FOO"

    def test_function_field_passed_serialize_only_is_dump_only(self, user):
        field = fields.Function(serialize=lambda obj: obj.name.upper())
        assert field.dump_only is True

    def test_function_field_passed_deserialize_and_serialize_is_not_dump_only(self):
        field = fields.Function(
            serialize=lambda val: val.lower(), deserialize=lambda val: val.upper()
        )
        assert field.dump_only is False

    def test_function_field_passed_serialize(self, user):
        field = fields.Function(serialize=lambda obj: obj.name.upper())
        assert field.serialize("key", user) == "FOO"

    # https://github.com/marshmallow-code/marshmallow/issues/395
    def test_function_field_does_not_swallow_attribute_error(self, user):
        def raise_error(obj):
            raise AttributeError

        field = fields.Function(serialize=raise_error)
        with pytest.raises(AttributeError):
            field.serialize("key", user)

    def test_serialize_with_load_only_param(self):
        class AliasingUserSerializer(Schema):
            name = fields.String()
            years = fields.Integer(load_only=True)
            size = fields.Integer(dump_only=True, load_only=True)
            nicknames = fields.List(fields.Str(), load_only=True)

        data = {
            "name": "Mick",
            "years": "42",
            "size": "12",
            "nicknames": ["Your Majesty", "Brenda"],
        }
        result = AliasingUserSerializer().dump(data)
        assert result["name"] == "Mick"
        assert "years" not in result
        assert "size" not in result
        assert "nicknames" not in result

    def test_function_field_load_only(self):
        field = fields.Function(deserialize=lambda obj: None)
        assert field.load_only

    def test_function_field_passed_uncallable_object(self):
        with pytest.raises(TypeError):
            fields.Function("uncallable")  # type: ignore[arg-type]

    def test_integer_field(self, user):
        field = fields.Integer()
        assert field.serialize("age", user) == 42

    def test_integer_as_string_field(self, user):
        field = fields.Integer(as_string=True)
        assert field.serialize("age", user) == "42"

    def test_integer_field_default(self, user):
        user.age = None
        field = fields.Integer(dump_default=0)
        assert field.serialize("age", user) is None
        # missing
        assert field.serialize("age", {}) == 0

    def test_integer_field_default_set_to_none(self, user):
        user.age = None
        field = fields.Integer(dump_default=None)
        assert field.serialize("age", user) is None

    def test_uuid_field(self, user):
        user.uuid1 = uuid.UUID("12345678123456781234567812345678")
        user.uuid2 = None

        field = fields.UUID()
        assert isinstance(field.serialize("uuid1", user), str)
        assert field.serialize("uuid1", user) == "12345678-1234-5678-1234-567812345678"
        assert field.serialize("uuid2", user) is None

    def test_ip_address_field(self, user):
        ipv4_string = "192.168.0.1"
        ipv6_string = "ffff::ffff"
        ipv6_exploded_string = ipaddress.ip_address("ffff::ffff").exploded

        user.ipv4 = ipaddress.ip_address(ipv4_string)
        user.ipv6 = ipaddress.ip_address(ipv6_string)
        user.empty_ip = None

        field_compressed = fields.IP()
        assert isinstance(field_compressed.serialize("ipv4", user), str)
        assert field_compressed.serialize("ipv4", user) == ipv4_string
        assert isinstance(field_compressed.serialize("ipv6", user), str)
        assert field_compressed.serialize("ipv6", user) == ipv6_string
        assert field_compressed.serialize("empty_ip", user) is None

        field_exploded = fields.IP(exploded=True)
        assert isinstance(field_exploded.serialize("ipv6", user), str)
        assert field_exploded.serialize("ipv6", user) == ipv6_exploded_string

    def test_ipv4_address_field(self, user):
        ipv4_string = "192.168.0.1"

        user.ipv4 = ipaddress.ip_address(ipv4_string)
        user.empty_ip = None

        field = fields.IPv4()
        assert isinstance(field.serialize("ipv4", user), str)
        assert field.serialize("ipv4", user) == ipv4_string
        assert field.serialize("empty_ip", user) is None

    def test_ipv6_address_field(self, user):
        ipv6_string = "ffff::ffff"
        ipv6_exploded_string = ipaddress.ip_address("ffff::ffff").exploded

        user.ipv6 = ipaddress.ip_address(ipv6_string)
        user.empty_ip = None

        field_compressed = fields.IPv6()
        assert isinstance(field_compressed.serialize("ipv6", user), str)
        assert field_compressed.serialize("ipv6", user) == ipv6_string
        assert field_compressed.serialize("empty_ip", user) is None

        field_exploded = fields.IPv6(exploded=True)
        assert isinstance(field_exploded.serialize("ipv6", user), str)
        assert field_exploded.serialize("ipv6", user) == ipv6_exploded_string

    def test_ip_interface_field(self, user):
        ipv4interface_string = "192.168.0.1/24"
        ipv6interface_string = "ffff::ffff/128"
        ipv6interface_exploded_string = ipaddress.ip_interface(
            "ffff::ffff/128"
        ).exploded

        user.ipv4interface = ipaddress.ip_interface(ipv4interface_string)
        user.ipv6interface = ipaddress.ip_interface(ipv6interface_string)
        user.empty_ipinterface = None

        field_compressed = fields.IPInterface()
        assert isinstance(field_compressed.serialize("ipv4interface", user), str)
        assert field_compressed.serialize("ipv4interface", user) == ipv4interface_string
        assert isinstance(field_compressed.serialize("ipv6interface", user), str)
        assert field_compressed.serialize("ipv6interface", user) == ipv6interface_string
        assert field_compressed.serialize("empty_ipinterface", user) is None

        field_exploded = fields.IPInterface(exploded=True)
        assert isinstance(field_exploded.serialize("ipv6interface", user), str)
        assert (
            field_exploded.serialize("ipv6interface", user)
            == ipv6interface_exploded_string
        )

    def test_ipv4_interface_field(self, user):
        ipv4interface_string = "192.168.0.1/24"

        user.ipv4interface = ipaddress.ip_interface(ipv4interface_string)
        user.empty_ipinterface = None

        field = fields.IPv4Interface()
        assert isinstance(field.serialize("ipv4interface", user), str)
        assert field.serialize("ipv4interface", user) == ipv4interface_string
        assert field.serialize("empty_ipinterface", user) is None

    def test_ipv6_interface_field(self, user):
        ipv6interface_string = "ffff::ffff/128"
        ipv6interface_exploded_string = ipaddress.ip_interface(
            "ffff::ffff/128"
        ).exploded

        user.ipv6interface = ipaddress.ip_interface(ipv6interface_string)
        user.empty_ipinterface = None

        field_compressed = fields.IPv6Interface()
        assert isinstance(field_compressed.serialize("ipv6interface", user), str)
        assert field_compressed.serialize("ipv6interface", user) == ipv6interface_string
        assert field_compressed.serialize("empty_ipinterface", user) is None

        field_exploded = fields.IPv6Interface(exploded=True)
        assert isinstance(field_exploded.serialize("ipv6interface", user), str)
        assert (
            field_exploded.serialize("ipv6interface", user)
            == ipv6interface_exploded_string
        )

    def test_enum_field_by_symbol_serialization(self, user):
        user.sex = GenderEnum.male
        field = fields.Enum(GenderEnum)
        assert field.serialize("sex", user) == "male"

    def test_enum_field_by_value_true_serialization(self, user):
        user.hair_color = HairColorEnum.black
        field = fields.Enum(HairColorEnum, by_value=True)
        assert field.serialize("hair_color", user) == "black hair"
        user.sex = GenderEnum.male
        field2 = fields.Enum(GenderEnum, by_value=True)
        assert field2.serialize("sex", user) == 1
        user.some_date = DateEnum.date_1

    def test_enum_field_by_value_field_serialization(self, user):
        user.hair_color = HairColorEnum.black
        field = fields.Enum(HairColorEnum, by_value=fields.String)
        assert field.serialize("hair_color", user) == "black hair"
        user.sex = GenderEnum.male
        field2 = fields.Enum(GenderEnum, by_value=fields.Integer)
        assert field2.serialize("sex", user) == 1
        user.some_date = DateEnum.date_1
        field3 = fields.Enum(DateEnum, by_value=fields.Date(format="%d/%m/%Y"))
        assert field3.serialize("some_date", user) == "29/02/2004"

    def test_decimal_field(self, user):
        user.m1 = 12
        user.m2 = "12.355"
        user.m3 = decimal.Decimal(1)
        user.m4 = None

        field = fields.Decimal()
        assert isinstance(field.serialize("m1", user), decimal.Decimal)
        assert field.serialize("m1", user) == decimal.Decimal(12)
        assert isinstance(field.serialize("m2", user), decimal.Decimal)
        assert field.serialize("m2", user) == decimal.Decimal("12.355")
        assert isinstance(field.serialize("m3", user), decimal.Decimal)
        assert field.serialize("m3", user) == decimal.Decimal(1)
        assert field.serialize("m4", user) is None

        field = fields.Decimal(1)
        assert isinstance(field.serialize("m1", user), decimal.Decimal)
        assert field.serialize("m1", user) == decimal.Decimal(12)
        assert isinstance(field.serialize("m2", user), decimal.Decimal)
        assert field.serialize("m2", user) == decimal.Decimal("12.4")
        assert isinstance(field.serialize("m3", user), decimal.Decimal)
        assert field.serialize("m3", user) == decimal.Decimal(1)
        assert field.serialize("m4", user) is None

        field = fields.Decimal(1, decimal.ROUND_DOWN)
        assert isinstance(field.serialize("m1", user), decimal.Decimal)
        assert field.serialize("m1", user) == decimal.Decimal(12)
        assert isinstance(field.serialize("m2", user), decimal.Decimal)
        assert field.serialize("m2", user) == decimal.Decimal("12.3")
        assert isinstance(field.serialize("m3", user), decimal.Decimal)
        assert field.serialize("m3", user) == decimal.Decimal(1)
        assert field.serialize("m4", user) is None

    def test_decimal_field_string(self, user):
        user.m1 = 12
        user.m2 = "12.355"
        user.m3 = decimal.Decimal(1)
        user.m4 = None

        field = fields.Decimal(as_string=True)
        assert isinstance(field.serialize("m1", user), str)
        assert field.serialize("m1", user) == "12"
        assert isinstance(field.serialize("m2", user), str)
        assert field.serialize("m2", user) == "12.355"
        assert isinstance(field.serialize("m3", user), str)
        assert field.serialize("m3", user) == "1"
        assert field.serialize("m4", user) is None

        field = fields.Decimal(1, as_string=True)
        assert isinstance(field.serialize("m1", user), str)
        assert field.serialize("m1", user) == "12.0"
        assert isinstance(field.serialize("m2", user), str)
        assert field.serialize("m2", user) == "12.4"
        assert isinstance(field.serialize("m3", user), str)
        assert field.serialize("m3", user) == "1.0"
        assert field.serialize("m4", user) is None

        field = fields.Decimal(1, decimal.ROUND_DOWN, as_string=True)
        assert isinstance(field.serialize("m1", user), str)
        assert field.serialize("m1", user) == "12.0"
        assert isinstance(field.serialize("m2", user), str)
        assert field.serialize("m2", user) == "12.3"
        assert isinstance(field.serialize("m3", user), str)
        assert field.serialize("m3", user) == "1.0"
        assert field.serialize("m4", user) is None

    def test_decimal_field_special_values(self, user):
        user.m1 = "-NaN"
        user.m2 = "NaN"
        user.m3 = "-sNaN"
        user.m4 = "sNaN"
        user.m5 = "-Infinity"
        user.m6 = "Infinity"
        user.m7 = "-0"

        field = fields.Decimal(places=2, allow_nan=True)

        m1s = field.serialize("m1", user)
        assert isinstance(m1s, decimal.Decimal)
        assert m1s.is_qnan()
        assert not m1s.is_signed()

        m2s = field.serialize("m2", user)
        assert isinstance(m2s, decimal.Decimal)
        assert m2s.is_qnan()
        assert not m2s.is_signed()

        m3s = field.serialize("m3", user)
        assert isinstance(m3s, decimal.Decimal)
        assert m3s.is_qnan()
        assert not m3s.is_signed()

        m4s = field.serialize("m4", user)
        assert isinstance(m4s, decimal.Decimal)
        assert m4s.is_qnan()
        assert not m4s.is_signed()

        m5s = field.serialize("m5", user)
        assert isinstance(m5s, decimal.Decimal)
        assert m5s.is_infinite()
        assert m5s.is_signed()

        m6s = field.serialize("m6", user)
        assert isinstance(m6s, decimal.Decimal)
        assert m6s.is_infinite()
        assert not m6s.is_signed()

        m7s = field.serialize("m7", user)
        assert isinstance(m7s, decimal.Decimal)
        assert m7s.is_zero()
        assert m7s.is_signed()

        field = fields.Decimal(as_string=True, allow_nan=True)

        m2s = field.serialize("m2", user)
        assert isinstance(m2s, str)
        assert m2s == user.m2

        m5s = field.serialize("m5", user)
        assert isinstance(m5s, str)
        assert m5s == user.m5

        m6s = field.serialize("m6", user)
        assert isinstance(m6s, str)
        assert m6s == user.m6

    def test_decimal_field_special_values_not_permitted(self, user):
        user.m7 = "-0"

        field = fields.Decimal(places=2)

        m7s = field.serialize("m7", user)
        assert isinstance(m7s, decimal.Decimal)
        assert m7s.is_zero()
        assert m7s.is_signed()

    def test_decimal_field_fixed_point_representation(self, user):
        """
        Test we get fixed-point string representation for a Decimal number that would normally
        output in engineering notation.
        """
        user.m1 = "0.00000000100000000"

        field = fields.Decimal()
        s = field.serialize("m1", user)
        assert isinstance(s, decimal.Decimal)
        assert s == decimal.Decimal("1.00000000E-9")

        field = fields.Decimal(as_string=True)
        s = field.serialize("m1", user)
        assert isinstance(s, str)
        assert s == user.m1

        field = fields.Decimal(as_string=True, places=2)
        s = field.serialize("m1", user)
        assert isinstance(s, str)
        assert s == "0.00"

    def test_email_field_serialize_none(self, user):
        user.email = None
        field = fields.Email()
        assert field.serialize("email", user) is None

    def test_dict_field_serialize_none(self, user):
        user.various_data = None
        field = fields.Dict()
        assert field.serialize("various_data", user) is None

    def test_dict_field_serialize(self, user):
        user.various_data = {"foo": "bar"}
        field = fields.Dict()
        dump = field.serialize("various_data", user)
        assert dump == {"foo": "bar"}
        # Check dump is a distinct object
        dump["foo"] = "baz"
        assert user.various_data["foo"] == "bar"

    def test_dict_field_serialize_ordereddict(self, user):
        user.various_data = OrderedDict([("foo", "bar"), ("bar", "baz")])
        field = fields.Dict()
        assert field.serialize("various_data", user) == OrderedDict(
            [("foo", "bar"), ("bar", "baz")]
        )

    def test_structured_dict_value_serialize(self, user):
        user.various_data = {"foo": decimal.Decimal("1")}
        field = fields.Dict(values=fields.Decimal)
        assert field.serialize("various_data", user) == {"foo": 1}

    def test_structured_dict_key_serialize(self, user):
        user.various_data = {1: "bar"}
        field = fields.Dict(keys=fields.Str)
        assert field.serialize("various_data", user) == {"1": "bar"}

    def test_structured_dict_key_value_serialize(self, user):
        user.various_data = {1: decimal.Decimal("1")}
        field = fields.Dict(keys=fields.Str, values=fields.Decimal)
        assert field.serialize("various_data", user) == {"1": 1}

    def test_url_field_serialize_none(self, user):
        user.homepage = None
        field = fields.Url()
        assert field.serialize("homepage", user) is None

    def test_method_field_with_method_missing(self):
        class BadSerializer(Schema):
            bad_field = fields.Method("invalid")

        with pytest.raises(AttributeError):
            BadSerializer()

    def test_method_field_passed_serialize_only_is_dump_only(self, user):
        field = fields.Method(serialize="method")
        assert field.dump_only is True
        assert field.load_only is False

    def test_method_field_passed_deserialize_only_is_load_only(self):
        field = fields.Method(deserialize="somemethod")
        assert field.load_only is True
        assert field.dump_only is False

    def test_method_field_with_uncallable_attribute(self):
        class BadSerializer(Schema):
            foo = "not callable"
            bad_field = fields.Method("foo")

        with pytest.raises(TypeError):
            BadSerializer()

    # https://github.com/marshmallow-code/marshmallow/issues/395
    def test_method_field_does_not_swallow_attribute_error(self):
        class MySchema(Schema):
            mfield = fields.Method("raise_error")

            def raise_error(self, obj):
                raise AttributeError

        with pytest.raises(AttributeError):
            MySchema().dump({})

    def test_method_with_no_serialize_is_missing(self):
        m = fields.Method()
        m.parent = Schema()

        assert m.serialize("", "", None) is missing_

    def test_serialize_with_data_key_param(self):
        class DumpToSchema(Schema):
            name = fields.String(data_key="NamE")
            years = fields.Integer(data_key="YearS")

        data = {"name": "Richard", "years": 11}
        result = DumpToSchema().dump(data)
        assert result == {"NamE": "Richard", "YearS": 11}

    def test_serialize_with_data_key_as_empty_string(self):
        class MySchema(Schema):
            name = fields.Raw(data_key="")

        schema = MySchema()
        assert schema.dump({"name": "Grace"}) == {"": "Grace"}

    def test_serialize_with_attribute_and_data_key_uses_data_key(self):
        class ConfusedDumpToAndAttributeSerializer(Schema):
            name = fields.String(data_key="FullName")
            username = fields.String(attribute="uname", data_key="UserName")
            years = fields.Integer(attribute="le_wild_age", data_key="Years")

        data = {"name": "Mick", "uname": "mick_the_awesome", "le_wild_age": 999}
        result = ConfusedDumpToAndAttributeSerializer().dump(data)

        assert result == {
            "FullName": "Mick",
            "UserName": "mick_the_awesome",
            "Years": 999,
        }

    @pytest.mark.parametrize("fmt", ["rfc", "rfc822"])
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (dt.datetime(2013, 11, 10, 1, 23, 45), "Sun, 10 Nov 2013 01:23:45 -0000"),
            (
                dt.datetime(2013, 11, 10, 1, 23, 45, tzinfo=dt.timezone.utc),
                "Sun, 10 Nov 2013 01:23:45 +0000",
            ),
            (
                dt.datetime(2013, 11, 10, 1, 23, 45, tzinfo=central),
                "Sun, 10 Nov 2013 01:23:45 -0600",
            ),
        ],
    )
    def test_datetime_field_rfc822(self, fmt, value, expected):
        field = fields.DateTime(format=fmt)
        assert field.serialize("d", {"d": value}) == expected

    @pytest.mark.parametrize(
        ("fmt", "value", "expected"),
        [
            ("timestamp", dt.datetime(1970, 1, 1), 0),
            ("timestamp", dt.datetime(2013, 11, 10, 0, 23, 45), 1384043025),
            (
                "timestamp",
                dt.datetime(2013, 11, 10, 0, 23, 45, tzinfo=dt.timezone.utc),
                1384043025,
            ),
            (
                "timestamp",
                dt.datetime(2013, 11, 10, 0, 23, 45, tzinfo=central),
                1384064625,
            ),
            ("timestamp_ms", dt.datetime(2013, 11, 10, 0, 23, 45), 1384043025000),
            (
                "timestamp_ms",
                dt.datetime(2013, 11, 10, 0, 23, 45, tzinfo=dt.timezone.utc),
                1384043025000,
            ),
            (
                "timestamp_ms",
                dt.datetime(2013, 11, 10, 0, 23, 45, tzinfo=central),
                1384064625000,
            ),
        ],
    )
    def test_datetime_field_timestamp(self, fmt, value, expected):
        field = fields.DateTime(format=fmt)
        assert field.serialize("d", {"d": value}) == expected

    @pytest.mark.parametrize("fmt", ["iso", "iso8601", None])
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (dt.datetime(2013, 11, 10, 1, 23, 45), "2013-11-10T01:23:45"),
            (
                dt.datetime(2013, 11, 10, 1, 23, 45, 123456, tzinfo=dt.timezone.utc),
                "2013-11-10T01:23:45.123456+00:00",
            ),
            (
                dt.datetime(2013, 11, 10, 1, 23, 45, tzinfo=dt.timezone.utc),
                "2013-11-10T01:23:45+00:00",
            ),
            (
                dt.datetime(2013, 11, 10, 1, 23, 45, tzinfo=central),
                "2013-11-10T01:23:45-06:00",
            ),
        ],
    )
    def test_datetime_field_iso8601(self, fmt, value, expected):
        if fmt is None:
            # Test default is ISO
            field = fields.DateTime()
        else:
            field = fields.DateTime(format=fmt)
        assert field.serialize("d", {"d": value}) == expected

    def test_datetime_field_format(self, user):
        datetimeformat = "%Y-%m-%d"
        field = fields.DateTime(format=datetimeformat)
        assert field.serialize("created", user) == user.created.strftime(datetimeformat)

    def test_string_field(self):
        field = fields.String()
        user = User(name=b"foo")
        assert field.serialize("name", user) == "foo"
        field = fields.String(allow_none=True)
        user.name = None
        assert field.serialize("name", user) is None

    def test_string_field_default_to_empty_string(self, user):
        field = fields.String(dump_default="")
        assert field.serialize("notfound", {}) == ""

    def test_time_field(self, user):
        field = fields.Time()
        expected = user.time_registered.isoformat()[:15]
        assert field.serialize("time_registered", user) == expected

        user.time_registered = None
        assert field.serialize("time_registered", user) is None

    @pytest.mark.parametrize("fmt", ["iso", "iso8601", None])
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (dt.time(1, 23, 45), "01:23:45"),
            (dt.time(1, 23, 45, 123000), "01:23:45.123000"),
            (dt.time(1, 23, 45, 123456), "01:23:45.123456"),
        ],
    )
    def test_time_field_iso8601(self, fmt, value, expected):
        if fmt is None:
            # Test default is ISO
            field = fields.Time()
        else:
            field = fields.Time(format=fmt)
        assert field.serialize("d", {"d": value}) == expected

    def test_time_field_format(self, user):
        fmt = "%H:%M:%S"
        field = fields.Time(format=fmt)
        assert field.serialize("birthtime", user) == user.birthtime.strftime(fmt)

    def test_date_field(self, user):
        field = fields.Date()
        assert field.serialize("birthdate", user) == user.birthdate.isoformat()

        user.birthdate = None
        assert field.serialize("birthdate", user) is None

    def test_timedelta_field(self, user):
        user.d1 = dt.timedelta(days=1, seconds=1, microseconds=1)
        user.d2 = dt.timedelta(days=0, seconds=86401, microseconds=1)
        user.d3 = dt.timedelta(days=0, seconds=0, microseconds=86401000001)
        user.d4 = dt.timedelta(days=0, seconds=0, microseconds=0)
        user.d5 = dt.timedelta(days=-1, seconds=0, microseconds=0)
        user.d6 = dt.timedelta(
            days=1,
            seconds=1,
            microseconds=1,
            milliseconds=1,
            minutes=1,
            hours=1,
            weeks=1,
        )

        field = fields.TimeDelta(fields.TimeDelta.DAYS)
        assert field.serialize("d1", user) == 1.0000115740856481
        field = fields.TimeDelta(fields.TimeDelta.SECONDS)
        assert field.serialize("d1", user) == 86401.000001
        field = fields.TimeDelta(fields.TimeDelta.MICROSECONDS)
        assert field.serialize("d1", user) == 86401000001
        field = fields.TimeDelta(fields.TimeDelta.HOURS)
        assert field.serialize("d1", user) == 24.000277778055555

        field = fields.TimeDelta(fields.TimeDelta.DAYS)
        assert field.serialize("d2", user) == 1.0000115740856481
        field = fields.TimeDelta(fields.TimeDelta.SECONDS)
        assert field.serialize("d2", user) == 86401.000001
        field = fields.TimeDelta(fields.TimeDelta.MICROSECONDS)
        assert field.serialize("d2", user) == 86401000001

        field = fields.TimeDelta(fields.TimeDelta.DAYS)
        assert field.serialize("d3", user) == 1.0000115740856481
        field = fields.TimeDelta(fields.TimeDelta.SECONDS)
        assert field.serialize("d3", user) == 86401.000001
        field = fields.TimeDelta(fields.TimeDelta.MICROSECONDS)
        assert field.serialize("d3", user) == 86401000001

        field = fields.TimeDelta(fields.TimeDelta.DAYS)
        assert field.serialize("d4", user) == 0
        field = fields.TimeDelta(fields.TimeDelta.SECONDS)
        assert field.serialize("d4", user) == 0
        field = fields.TimeDelta(fields.TimeDelta.MICROSECONDS)
        assert field.serialize("d4", user) == 0

        field = fields.TimeDelta(fields.TimeDelta.DAYS)
        assert field.serialize("d5", user) == -1
        field = fields.TimeDelta(fields.TimeDelta.SECONDS)
        assert field.serialize("d5", user) == -86400
        field = fields.TimeDelta(fields.TimeDelta.MICROSECONDS)
        assert field.serialize("d5", user) == -86400000000

        field = fields.TimeDelta(fields.TimeDelta.WEEKS)
        assert field.serialize("d6", user) == 1.1489103852529763
        field = fields.TimeDelta(fields.TimeDelta.DAYS)
        assert field.serialize("d6", user) == 8.042372696770833
        field = fields.TimeDelta(fields.TimeDelta.HOURS)
        assert field.serialize("d6", user) == 193.0169447225
        field = fields.TimeDelta(fields.TimeDelta.MINUTES)
        assert field.serialize("d6", user) == 11581.01668335
        field = fields.TimeDelta(fields.TimeDelta.SECONDS)
        assert field.serialize("d6", user) == 694861.001001
        field = fields.TimeDelta(fields.TimeDelta.MILLISECONDS)
        assert field.serialize("d6", user) == 694861001.001
        field = fields.TimeDelta(fields.TimeDelta.MICROSECONDS)
        assert field.serialize("d6", user) == 694861001001

        user.d7 = None
        assert field.serialize("d7", user) is None

        user.d8 = dt.timedelta(milliseconds=345)
        field = fields.TimeDelta(fields.TimeDelta.MILLISECONDS)
        assert field.serialize("d8", user) == 345

        user.d9 = dt.timedelta(milliseconds=1999)
        field = fields.TimeDelta(fields.TimeDelta.SECONDS)
        assert field.serialize("d9", user) == 1.999

        user.d10 = dt.timedelta(
            weeks=1,
            days=6,
            hours=2,
            minutes=5,
            seconds=51,
            milliseconds=10,
            microseconds=742,
        )

        field = fields.TimeDelta(fields.TimeDelta.MICROSECONDS)
        unit_value = dt.timedelta(microseconds=1).total_seconds()
        assert math.isclose(
            field.serialize("d10", user), user.d10.total_seconds() / unit_value
        )

        field = fields.TimeDelta(fields.TimeDelta.MILLISECONDS)
        unit_value = dt.timedelta(milliseconds=1).total_seconds()
        assert math.isclose(
            field.serialize("d10", user), user.d10.total_seconds() / unit_value
        )

        field = fields.TimeDelta(fields.TimeDelta.SECONDS)
        assert math.isclose(field.serialize("d10", user), user.d10.total_seconds())

        field = fields.TimeDelta(fields.TimeDelta.MINUTES)
        unit_value = dt.timedelta(minutes=1).total_seconds()
        assert math.isclose(
            field.serialize("d10", user), user.d10.total_seconds() / unit_value
        )

        field = fields.TimeDelta(fields.TimeDelta.HOURS)
        unit_value = dt.timedelta(hours=1).total_seconds()
        assert math.isclose(
            field.serialize("d10", user), user.d10.total_seconds() / unit_value
        )

        field = fields.TimeDelta(fields.TimeDelta.DAYS)
        unit_value = dt.timedelta(days=1).total_seconds()
        assert math.isclose(
            field.serialize("d10", user), user.d10.total_seconds() / unit_value
        )

        field = fields.TimeDelta(fields.TimeDelta.WEEKS)
        unit_value = dt.timedelta(weeks=1).total_seconds()
        assert math.isclose(
            field.serialize("d10", user), user.d10.total_seconds() / unit_value
        )

    def test_datetime_list_field(self):
        obj = DateTimeList([dt.datetime.now(dt.timezone.utc), dt.datetime.now()])
        field = fields.List(fields.DateTime)
        result = field.serialize("dtimes", obj)
        assert all(type(each) is str for each in result)

    def test_list_field_serialize_none_returns_none(self):
        obj = DateTimeList(None)
        field = fields.List(fields.DateTime)
        assert field.serialize("dtimes", obj) is None

    def test_list_field_work_with_generator_single_value(self):
        def custom_generator():
            yield dt.datetime.now(dt.timezone.utc)

        obj = DateTimeList(custom_generator())
        field = fields.List(fields.DateTime)
        result = field.serialize("dtimes", obj)
        assert len(result) == 1

    def test_list_field_work_with_generators_multiple_values(self):
        def custom_generator():
            yield from [dt.datetime.now(dt.timezone.utc), dt.datetime.now()]

        obj = DateTimeList(custom_generator())
        field = fields.List(fields.DateTime)
        result = field.serialize("dtimes", obj)
        assert len(result) == 2

    def test_list_field_work_with_generators_empty_generator_returns_none_for_every_non_returning_yield_statement(
        self,
    ):
        def custom_generator():
            yield
            yield

        obj = DateTimeList(custom_generator())
        field = fields.List(fields.DateTime, allow_none=True)
        result = field.serialize("dtimes", obj)
        assert len(result) == 2
        assert result[0] is None
        assert result[1] is None

    def test_list_field_work_with_set(self):
        custom_set = {1, 2, 3}
        obj = IntegerList(custom_set)
        field = fields.List(fields.Int)
        result = field.serialize("ints", obj)
        assert len(result) == 3
        assert 1 in result
        assert 2 in result
        assert 3 in result

    def test_list_field_work_with_custom_class_with_iterator_protocol(self):
        class IteratorSupportingClass:
            def __init__(self, iterable):
                self.iterable = iterable

            def __iter__(self):
                return iter(self.iterable)

        ints = IteratorSupportingClass([1, 2, 3])
        obj = IntegerList(ints)
        field = fields.List(fields.Int)
        result = field.serialize("ints", obj)
        assert len(result) == 3
        assert result[0] == 1
        assert result[1] == 2
        assert result[2] == 3

    def test_bad_list_field(self):
        class ASchema(Schema):
            id = fields.Int()

        with pytest.raises(ValueError):
            fields.List("string")  # type: ignore[arg-type]
        expected_msg = (
            "The list elements must be a subclass or instance of "
            "marshmallow.fields.Field"
        )
        with pytest.raises(ValueError, match=expected_msg):
            fields.List(ASchema)  # type: ignore[arg-type]

    def test_datetime_integer_tuple_field(self):
        obj = DateTimeIntegerTuple((dt.datetime.now(dt.timezone.utc), 42))
        field = fields.Tuple([fields.DateTime, fields.Integer])
        result = field.serialize("dtime_int", obj)
        assert type(result[0]) is str
        assert type(result[1]) is int

    def test_tuple_field_serialize_none_returns_none(self):
        obj = DateTimeIntegerTuple(None)
        field = fields.Tuple([fields.DateTime, fields.Integer])
        assert field.serialize("dtime_int", obj) is None

    def test_bad_tuple_field(self):
        class ASchema(Schema):
            id = fields.Int()

        with pytest.raises(ValueError):
            fields.Tuple(["string"])  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            fields.Tuple(fields.String)  # type: ignore[arg-type]
        expected_msg = (
            'Elements of "tuple_fields" must be subclasses or '
            "instances of marshmallow.fields.Field."
        )
        with pytest.raises(ValueError, match=expected_msg):
            fields.Tuple([ASchema])  # type: ignore[arg-type]

    def test_serialize_does_not_apply_validators(self, user):
        field = fields.Raw(validate=lambda x: False)
        # No validation error raised
        assert field.serialize("age", user) == user.age

    def test_constant_field_serialization(self, user):
        field = fields.Constant("something")
        assert field.serialize("whatever", user) == "something"

    def test_constant_is_always_included_in_serialized_data(self):
        class MySchema(Schema):
            foo = fields.Constant(42)

        sch = MySchema()
        assert sch.dump({"bar": 24})["foo"] == 42
        assert sch.dump({"foo": 24})["foo"] == 42

    def test_constant_field_serialize_when_omitted(self):
        class MiniUserSchema(Schema):
            name = fields.Constant("bill")

        s = MiniUserSchema()
        assert s.dump({})["name"] == "bill"

    @pytest.mark.parametrize("FieldClass", ALL_FIELDS)
    def test_all_fields_serialize_none_to_none(self, FieldClass):
        field = FieldClass(allow_none=True)
        res = field.serialize("foo", {"foo": None})
        assert res is None


class TestSchemaSerialization:
    def test_serialize_with_missing_param_value(self):
        class AliasingUserSerializer(Schema):
            name = fields.String()
            birthdate = fields.DateTime(dump_default=dt.datetime(2017, 9, 29))

        data = {"name": "Mick"}
        result = AliasingUserSerializer().dump(data)
        assert result["name"] == "Mick"
        assert result["birthdate"] == "2017-09-29T00:00:00"

    def test_serialize_with_missing_param_callable(self):
        class AliasingUserSerializer(Schema):
            name = fields.String()
            birthdate = fields.DateTime(dump_default=lambda: dt.datetime(2017, 9, 29))

        data = {"name": "Mick"}
        result = AliasingUserSerializer().dump(data)
        assert result["name"] == "Mick"
        assert result["birthdate"] == "2017-09-29T00:00:00"


def test_serializing_named_tuple():
    field = fields.Raw()

    p = Point(x=4, y=2)

    assert field.serialize("x", p) == 4


def test_serializing_named_tuple_with_meta():
    p = Point(x=4, y=2)

    class PointSerializer(Schema):
        x = fields.Int()
        y = fields.Int()

    serialized = PointSerializer().dump(p)
    assert serialized["x"] == 4
    assert serialized["y"] == 2


def test_serializing_slice():
    values = [{"value": value} for value in range(5)]
    sliced = itertools.islice(values, None)

    class ValueSchema(Schema):
        value = fields.Int()

    serialized = ValueSchema(many=True).dump(sliced)
    assert serialized == values


# https://github.com/marshmallow-code/marshmallow/issues/1163
def test_nested_field_many_serializing_generator():
    class MySchema(Schema):
        name = fields.Str()

    class OtherSchema(Schema):
        objects = fields.Nested(MySchema, many=True)

    def gen():
        yield {"name": "foo"}
        yield {"name": "bar"}

    obj = {"objects": gen()}
    data = OtherSchema().dump(obj)

    assert data.get("objects") == [{"name": "foo"}, {"name": "bar"}]


# === tests/test_error_store.py ===
from typing import NamedTuple

from marshmallow import missing
from marshmallow.error_store import merge_errors


def test_missing_is_falsy():
    assert bool(missing) is False


class CustomError(NamedTuple):
    code: int
    message: str


class TestMergeErrors:
    def test_merging_none_and_string(self):
        assert merge_errors(None, "error1") == "error1"

    def test_merging_none_and_custom_error(self):
        assert CustomError(123, "error1") == merge_errors(
            None, CustomError(123, "error1")
        )

    def test_merging_none_and_list(self):
        assert merge_errors(None, ["error1", "error2"]) == ["error1", "error2"]

    def test_merging_none_and_dict(self):
        assert merge_errors(None, {"field1": "error1"}) == {"field1": "error1"}

    def test_merging_string_and_none(self):
        assert merge_errors("error1", None) == "error1"

    def test_merging_custom_error_and_none(self):
        assert CustomError(123, "error1") == merge_errors(
            CustomError(123, "error1"), None
        )

    def test_merging_list_and_none(self):
        assert merge_errors(["error1", "error2"], None) == ["error1", "error2"]

    def test_merging_dict_and_none(self):
        assert merge_errors({"field1": "error1"}, None) == {"field1": "error1"}

    def test_merging_string_and_string(self):
        assert merge_errors("error1", "error2") == ["error1", "error2"]

    def test_merging_custom_error_and_string(self):
        assert [CustomError(123, "error1"), "error2"] == merge_errors(
            CustomError(123, "error1"), "error2"
        )

    def test_merging_string_and_custom_error(self):
        assert ["error1", CustomError(123, "error2")] == merge_errors(
            "error1", CustomError(123, "error2")
        )

    def test_merging_custom_error_and_custom_error(self):
        assert [CustomError(123, "error1"), CustomError(456, "error2")] == merge_errors(
            CustomError(123, "error1"), CustomError(456, "error2")
        )

    def test_merging_string_and_list(self):
        assert merge_errors("error1", ["error2"]) == ["error1", "error2"]

    def test_merging_string_and_dict(self):
        assert merge_errors("error1", {"field1": "error2"}) == {
            "_schema": "error1",
            "field1": "error2",
        }

    def test_merging_string_and_dict_with_schema_error(self):
        assert merge_errors("error1", {"_schema": "error2", "field1": "error3"}) == {
            "_schema": ["error1", "error2"],
            "field1": "error3",
        }

    def test_merging_custom_error_and_list(self):
        assert [CustomError(123, "error1"), "error2"] == merge_errors(
            CustomError(123, "error1"), ["error2"]
        )

    def test_merging_custom_error_and_dict(self):
        assert {
            "_schema": CustomError(123, "error1"),
            "field1": "error2",
        } == merge_errors(CustomError(123, "error1"), {"field1": "error2"})

    def test_merging_custom_error_and_dict_with_schema_error(self):
        assert {
            "_schema": [CustomError(123, "error1"), "error2"],
            "field1": "error3",
        } == merge_errors(
            CustomError(123, "error1"), {"_schema": "error2", "field1": "error3"}
        )

    def test_merging_list_and_string(self):
        assert merge_errors(["error1"], "error2") == ["error1", "error2"]

    def test_merging_list_and_custom_error(self):
        assert ["error1", CustomError(123, "error2")] == merge_errors(
            ["error1"], CustomError(123, "error2")
        )

    def test_merging_list_and_list(self):
        assert merge_errors(["error1"], ["error2"]) == ["error1", "error2"]

    def test_merging_list_and_dict(self):
        assert merge_errors(["error1"], {"field1": "error2"}) == {
            "_schema": ["error1"],
            "field1": "error2",
        }

    def test_merging_list_and_dict_with_schema_error(self):
        assert merge_errors(["error1"], {"_schema": "error2", "field1": "error3"}) == {
            "_schema": ["error1", "error2"],
            "field1": "error3",
        }

    def test_merging_dict_and_string(self):
        assert merge_errors({"field1": "error1"}, "error2") == {
            "_schema": "error2",
            "field1": "error1",
        }

    def test_merging_dict_and_custom_error(self):
        assert {
            "_schema": CustomError(123, "error2"),
            "field1": "error1",
        } == merge_errors({"field1": "error1"}, CustomError(123, "error2"))

    def test_merging_dict_and_list(self):
        assert merge_errors({"field1": "error1"}, ["error2"]) == {
            "_schema": ["error2"],
            "field1": "error1",
        }

    def test_merging_dict_and_dict(self):
        assert merge_errors(
            {"field1": "error1", "field2": "error2"},
            {"field2": "error3", "field3": "error4"},
        ) == {
            "field1": "error1",
            "field2": ["error2", "error3"],
            "field3": "error4",
        }

    def test_deep_merging_dicts(self):
        assert merge_errors(
            {"field1": {"field2": "error1"}}, {"field1": {"field2": "error2"}}
        ) == {"field1": {"field2": ["error1", "error2"]}}


# === tests/base.py ===
"""Test utilities and fixtures."""

import datetime as dt
import functools
import typing
import uuid
from enum import Enum, IntEnum
from zoneinfo import ZoneInfo

import simplejson

from marshmallow import Schema, fields, missing, post_load, validate
from marshmallow.exceptions import ValidationError

central = ZoneInfo("America/Chicago")


class GenderEnum(IntEnum):
    male = 1
    female = 2
    non_binary = 3


class HairColorEnum(Enum):
    black = "black hair"
    brown = "brown hair"
    blond = "blond hair"
    red = "red hair"


class DateEnum(Enum):
    date_1 = dt.date(2004, 2, 29)
    date_2 = dt.date(2008, 2, 29)
    date_3 = dt.date(2012, 2, 29)


ALL_FIELDS = [
    fields.String,
    fields.Integer,
    fields.Boolean,
    fields.Float,
    fields.DateTime,
    fields.Time,
    fields.Date,
    fields.TimeDelta,
    fields.Dict,
    fields.Url,
    fields.Email,
    fields.UUID,
    fields.Decimal,
    fields.IP,
    fields.IPv4,
    fields.IPv6,
    fields.IPInterface,
    fields.IPv4Interface,
    fields.IPv6Interface,
    functools.partial(fields.Enum, GenderEnum),
    functools.partial(fields.Enum, HairColorEnum, by_value=fields.String),
    functools.partial(fields.Enum, GenderEnum, by_value=fields.Integer),
]


##### Custom asserts #####


def assert_date_equal(d1: dt.date, d2: dt.date) -> None:
    assert d1.year == d2.year
    assert d1.month == d2.month
    assert d1.day == d2.day


def assert_time_equal(t1: dt.time, t2: dt.time) -> None:
    assert t1.hour == t2.hour
    assert t1.minute == t2.minute
    assert t1.second == t2.second
    assert t1.microsecond == t2.microsecond


##### Validation #####


def predicate(
    func: typing.Callable[[typing.Any], bool],
) -> typing.Callable[[typing.Any], None]:
    def validate(value: typing.Any) -> None:
        if func(value) is False:
            raise ValidationError("Invalid value.")

    return validate


##### Models #####


class User:
    SPECIES = "Homo sapiens"

    def __init__(
        self,
        name,
        *,
        age=0,
        id_=None,
        homepage=None,
        email=None,
        registered=True,
        time_registered=None,
        birthdate=None,
        birthtime=None,
        balance=100,
        sex=GenderEnum.male,
        hair_color=HairColorEnum.black,
        employer=None,
        various_data=None,
    ):
        self.name = name
        self.age = age
        # A naive datetime
        self.created = dt.datetime(2013, 11, 10, 14, 20, 58)
        # A TZ-aware datetime
        self.updated = dt.datetime(2013, 11, 10, 14, 20, 58, tzinfo=central)
        self.id = id_
        self.homepage = homepage
        self.email = email
        self.balance = balance
        self.registered = registered
        self.hair_colors = list(HairColorEnum.__members__)
        self.sex_choices = list(GenderEnum.__members__)
        self.finger_count = 10
        self.uid = uuid.uuid1()
        self.time_registered = time_registered or dt.time(1, 23, 45, 6789)
        self.birthdate = birthdate or dt.date(2013, 1, 23)
        self.birthtime = birthtime or dt.time(0, 1, 2, 3333)
        self.activation_date = dt.date(2013, 12, 11)
        self.sex = sex
        self.hair_color = hair_color
        self.employer = employer
        self.relatives = []
        self.various_data = various_data or {
            "pets": ["cat", "dog"],
            "address": "1600 Pennsylvania Ave\nWashington, DC 20006",
        }

    @property
    def since_created(self):
        return dt.datetime(2013, 11, 24) - self.created

    def __repr__(self):
        return f"<User {self.name}>"


class Blog:
    def __init__(self, title, user, collaborators=None, categories=None, id_=None):
        self.title = title
        self.user = user
        self.collaborators = collaborators or []  # List/tuple of users
        self.categories = categories
        self.id = id_

    def __contains__(self, item):
        return item.name in [each.name for each in self.collaborators]


class DummyModel:
    def __init__(self, foo):
        self.foo = foo

    def __eq__(self, other):
        return self.foo == other.foo

    def __str__(self):
        return f"bar {self.foo}"


###### Schemas #####


class Uppercased(fields.String):
    """Custom field formatting example."""

    def _serialize(self, value, attr, obj, **kwargs):
        if value:
            return value.upper()
        return None


def get_lowername(obj):
    if obj is None:
        return missing
    if isinstance(obj, dict):
        return obj.get("name", "").lower()
    return obj.name.lower()


class UserSchema(Schema):
    name = fields.String()
    age: fields.Field = fields.Float()
    created = fields.DateTime()
    created_formatted = fields.DateTime(
        format="%Y-%m-%d", attribute="created", dump_only=True
    )
    created_iso = fields.DateTime(format="iso", attribute="created", dump_only=True)
    updated = fields.DateTime()
    species = fields.String(attribute="SPECIES")
    id = fields.String(dump_default="no-id")
    uppername = Uppercased(attribute="name", dump_only=True)
    homepage = fields.Url()
    email = fields.Email()
    balance = fields.Decimal()
    is_old: fields.Field = fields.Method("get_is_old")
    lowername = fields.Function(get_lowername)
    registered = fields.Boolean()
    hair_colors = fields.List(fields.Raw)
    sex_choices = fields.List(fields.Raw)
    finger_count = fields.Integer()
    uid = fields.UUID()
    time_registered = fields.Time()
    birthdate = fields.Date()
    birthtime = fields.Time()
    activation_date = fields.Date()
    since_created = fields.TimeDelta()
    sex = fields.Str(validate=validate.OneOf(list(GenderEnum.__members__)))
    various_data = fields.Dict()

    class Meta:
        render_module = simplejson

    def get_is_old(self, obj):
        if obj is None:
            return missing
        if isinstance(obj, dict):
            age = obj.get("age", 0)
        else:
            age = obj.age
        try:
            return age > 80
        except TypeError as te:
            raise ValidationError(str(te)) from te

    @post_load
    def make_user(self, data, **kwargs):
        return User(**data)


class UserExcludeSchema(UserSchema):
    class Meta:
        exclude = ("created", "updated")


class UserIntSchema(UserSchema):
    age = fields.Integer()


class UserFloatStringSchema(UserSchema):
    age = fields.Float(as_string=True)


class ExtendedUserSchema(UserSchema):
    is_old = fields.Boolean()


class UserRelativeUrlSchema(UserSchema):
    homepage = fields.Url(relative=True)


class BlogSchema(Schema):
    title = fields.String()
    user = fields.Nested(UserSchema)
    collaborators = fields.List(fields.Nested(UserSchema()))
    categories = fields.List(fields.String)
    id = fields.String()


class BlogOnlySchema(Schema):
    title = fields.String()
    user = fields.Nested(UserSchema)
    collaborators = fields.List(fields.Nested(UserSchema(only=("id",))))


class BlogSchemaExclude(BlogSchema):
    user = fields.Nested(UserSchema, exclude=("uppername", "species"))


class BlogSchemaOnlyExclude(BlogSchema):
    user = fields.Nested(UserSchema, only=("name",), exclude=("name", "species"))


class mockjson:  # noqa: N801
    @staticmethod
    def dumps(val):
        return b"{'foo': 42}"

    @staticmethod
    def loads(val):
        return {"foo": 42}


# === tests/mypy_test_cases/test_class_registry.py ===
from marshmallow import class_registry

# Works without passing `all`
class_registry.get_class("MySchema")


# === tests/mypy_test_cases/test_schema.py ===
import json

from marshmallow import EXCLUDE, Schema
from marshmallow.fields import Integer, String


# Test that valid `Meta` class attributes pass type checking
class MySchema(Schema):
    foo = String()
    bar = Integer()

    class Meta(Schema.Meta):
        fields = ("foo", "bar")
        additional = ("baz", "qux")
        include = {
            "foo2": String(),
        }
        exclude = ("bar", "baz")
        many = True
        dateformat = "%Y-%m-%d"
        datetimeformat = "%Y-%m-%dT%H:%M:%S"
        timeformat = "%H:%M:%S"
        render_module = json
        ordered = False
        index_errors = True
        load_only = ("foo", "bar")
        dump_only = ("baz", "qux")
        unknown = EXCLUDE
        register = False


# === tests/mypy_test_cases/test_validation_error.py ===
from __future__ import annotations

import marshmallow as ma

# OK types for 'message'
ma.ValidationError("foo")
ma.ValidationError(["foo"])
ma.ValidationError({"foo": "bar"})

# non-OK types for 'message'
ma.ValidationError(0)  # type: ignore[arg-type]

# 'messages' is a dict|list
err = ma.ValidationError("foo")
a: dict | list = err.messages
# union type can't assign to non-union type
b: str = err.messages  # type: ignore[assignment]
c: dict = err.messages  # type: ignore[assignment]
# 'messages_dict' is a dict, so that it can assign to a dict
d: dict = err.messages_dict


# === docs/conf.py ===
import importlib.metadata

extensions = [
    "autodocsumm",
    "sphinx.ext.autodoc",
    "sphinx.ext.autodoc.typehints",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_issues",
    "sphinxext.opengraph",
]

primary_domain = "py"
default_role = "py:obj"

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

issues_github_path = "marshmallow-code/marshmallow"

source_suffix = ".rst"
master_doc = "index"

project = "marshmallow"
copyright = "Steven Loria and contributors"  # noqa: A001

version = release = importlib.metadata.version("marshmallow")

exclude_patterns = ["_build"]
# Ignore WARNING: more than one target found for cross-reference 'Schema': marshmallow.schema.Schema, marshmallow.Schema
suppress_warnings = ["ref.python"]

# THEME

html_theme = "furo"
html_theme_options = {
    "light_logo": "marshmallow-logo-with-title.png",
    "dark_logo": "marshmallow-logo-with-title-for-dark-theme.png",
    "source_repository": "https://github.com/marshmallow-code/marshmallow",
    "source_branch": "dev",
    "source_directory": "docs/",
    "sidebar_hide_name": True,
    "light_css_variables": {
        # Serif system font stack: https://systemfontstack.com/
        "font-stack": "Iowan Old Style, Apple Garamond, Baskerville, Times New Roman, Droid Serif, Times, Source Serif Pro, serif, Apple Color Emoji, Segoe UI Emoji, Segoe UI Symbol;",
    },
    "top_of_page_buttons": ["view"],
}
pygments_dark_style = "lightbulb"
html_favicon = "_static/favicon.ico"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_copy_source = False  # Don't copy source files to _build/sources
html_show_sourcelink = False  # Don't link to source files
ogp_image = "_static/marshmallow-logo-200.png"

# Strip the dollar prompt when copying code
# https://sphinx-copybutton.readthedocs.io/en/latest/use.html#strip-and-configure-input-prompts-for-code-cells
copybutton_prompt_text = "$ "

autodoc_default_options = {
    "exclude-members": "__new__",
    # Don't show signatures in the summary tables
    "autosummary-nosignatures": True,
    # Don't render summaries for classes within modules
    "autosummary-no-nesting": True,
}
# Only display type hints next to params but not within the signature
# to avoid the signature from getting too long
autodoc_typehints = "description"


# === examples/inflection_example.py ===
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "marshmallow",
# ]
# ///
from marshmallow import Schema, fields


def camelcase(s):
    parts = iter(s.split("_"))
    return next(parts) + "".join(i.title() for i in parts)


class CamelCaseSchema(Schema):
    """Schema that uses camel-case for its external representation
    and snake-case for its internal representation.
    """

    def on_bind_field(self, field_name, field_obj):
        field_obj.data_key = camelcase(field_obj.data_key or field_name)


# -----------------------------------------------------------------------------


class UserSchema(CamelCaseSchema):
    first_name = fields.Str(required=True)
    last_name = fields.Str(required=True)


schema = UserSchema()
loaded = schema.load({"firstName": "David", "lastName": "Bowie"})
print("Loaded data:")
print(loaded)
dumped = schema.dump(loaded)
print("Dumped data:")
print(dumped)


# === examples/package_json_example.py ===
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "marshmallow",
#     "packaging>=17.0",
# ]
# ///
import json
import sys
from pprint import pprint

from packaging import version

from marshmallow import INCLUDE, Schema, ValidationError, fields


class Version(fields.Field[version.Version]):
    """Version field that deserializes to a Version object."""

    def _deserialize(self, value, *args, **kwargs):
        try:
            return version.Version(value)
        except version.InvalidVersion as e:
            raise ValidationError("Not a valid version.") from e

    def _serialize(self, value, *args, **kwargs):
        return str(value)


class PackageSchema(Schema):
    name = fields.Str(required=True)
    version = Version(required=True)
    description = fields.Str(required=True)
    main = fields.Str(required=False)
    homepage = fields.URL(required=False)
    scripts = fields.Dict(keys=fields.Str(), values=fields.Str())
    license = fields.Str(required=True)
    dependencies = fields.Dict(keys=fields.Str(), values=fields.Str(), required=False)
    dev_dependencies = fields.Dict(
        keys=fields.Str(),
        values=fields.Str(),
        required=False,
        data_key="devDependencies",
    )

    class Meta:
        # Include unknown fields in the deserialized output
        unknown = INCLUDE


if __name__ == "__main__":
    pkg = json.load(sys.stdin)
    try:
        pprint(PackageSchema().load(pkg))
    except ValidationError as error:
        print("ERROR: package.json is invalid")
        pprint(error.messages)
        sys.exit(1)


# === examples/flask_example.py ===
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "flask",
#     "flask-sqlalchemy>=3.1.1",
#     "marshmallow",
#     "sqlalchemy>2.0",
# ]
# ///
from __future__ import annotations

import datetime

from flask import Flask, request
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from marshmallow import Schema, ValidationError, fields, pre_load

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:////tmp/quotes.db"


class Base(DeclarativeBase):
    pass


db = SQLAlchemy(app, model_class=Base)

##### MODELS #####


class Author(db.Model):  # type: ignore[name-defined]
    id: Mapped[int] = mapped_column(primary_key=True)
    first: Mapped[str]
    last: Mapped[str]


class Quote(db.Model):  # type: ignore[name-defined]
    id: Mapped[int] = mapped_column(primary_key=True)
    content: Mapped[str] = mapped_column(nullable=False)
    author_id: Mapped[int] = mapped_column(db.ForeignKey(Author.id))
    author: Mapped[Author] = relationship(backref=db.backref("quotes", lazy="dynamic"))
    posted_at: Mapped[datetime.datetime]


##### SCHEMAS #####


class AuthorSchema(Schema):
    id = fields.Int(dump_only=True)
    first = fields.Str()
    last = fields.Str()
    formatted_name = fields.Method("format_name", dump_only=True)

    def format_name(self, author):
        return f"{author.last}, {author.first}"


# Custom validator
def must_not_be_blank(data):
    if not data:
        raise ValidationError("Data not provided.")


class QuoteSchema(Schema):
    id = fields.Int(dump_only=True)
    author = fields.Nested(AuthorSchema, validate=must_not_be_blank)
    content = fields.Str(required=True, validate=must_not_be_blank)
    posted_at = fields.DateTime(dump_only=True)

    # Allow client to pass author's full name in request body
    # e.g. {"author': 'Tim Peters"} rather than {"first": "Tim", "last": "Peters"}
    @pre_load
    def process_author(self, data, **kwargs):
        author_name = data.get("author")
        if author_name:
            first, last = author_name.split(" ")
            author_dict = {"first": first, "last": last}
        else:
            author_dict = {}
        data["author"] = author_dict
        return data


author_schema = AuthorSchema()
authors_schema = AuthorSchema(many=True)
quote_schema = QuoteSchema()
quotes_schema = QuoteSchema(many=True, only=("id", "content"))

##### API #####


@app.route("/authors")
def get_authors():
    authors = Author.query.all()
    # Serialize the queryset
    result = authors_schema.dump(authors)
    return {"authors": result}


@app.route("/authors/<int:pk>")
def get_author(pk):
    try:
        author = Author.query.filter(Author.id == pk).one()
    except NoResultFound:
        return {"message": "Author could not be found."}, 400
    author_result = author_schema.dump(author)
    quotes_result = quotes_schema.dump(author.quotes.all())
    return {"author": author_result, "quotes": quotes_result}


@app.route("/quotes/", methods=["GET"])
def get_quotes():
    quotes = Quote.query.all()
    result = quotes_schema.dump(quotes, many=True)
    return {"quotes": result}


@app.route("/quotes/<int:pk>")
def get_quote(pk):
    try:
        quote = Quote.query.filter(Quote.id == pk).one()
    except NoResultFound:
        return {"message": "Quote could not be found."}, 400
    result = quote_schema.dump(quote)
    return {"quote": result}


@app.route("/quotes/", methods=["POST"])
def new_quote():
    json_data = request.get_json()
    if not json_data:
        return {"message": "No input data provided"}, 400
    # Validate and deserialize input
    try:
        data = quote_schema.load(json_data)
    except ValidationError as err:
        return err.messages, 422
    first, last = data["author"]["first"], data["author"]["last"]
    author = Author.query.filter_by(first=first, last=last).first()
    if author is None:
        # Create a new author
        author = Author(first=first, last=last)
        db.session.add(author)
    # Create new quote
    quote = Quote(
        content=data["content"],
        author=author,
        posted_at=datetime.datetime.now(datetime.UTC),
    )
    db.session.add(quote)
    db.session.commit()
    result = quote_schema.dump(Quote.query.get(quote.id))
    return {"message": "Created new quote.", "quote": result}


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5000)


# === performance/benchmark.py ===
"""Simple benchmark for marshmallow serialization of a moderately complex object.

Uses the `timeit` module to benchmark serializing an object through marshmallow.
"""

# ruff: noqa: A002, T201
import argparse
import cProfile
import datetime
import gc
import timeit

from marshmallow import Schema, ValidationError, fields, post_dump


# Custom validator
def must_not_be_blank(data):
    if not data:
        raise ValidationError("Data not provided.")


class AuthorSchema(Schema):
    id = fields.Int(dump_only=True)
    first = fields.Str()
    last = fields.Str()
    book_count = fields.Float()
    age = fields.Float()
    address = fields.Str()
    full_name = fields.Method("get_full_name")

    def get_full_name(self, author):
        return f"{author.last}, {author.first}"


class QuoteSchema(Schema):
    id = fields.Int(dump_only=True)
    author = fields.Nested(AuthorSchema, validate=must_not_be_blank)
    content = fields.Str(required=True, validate=must_not_be_blank)
    posted_at = fields.DateTime(dump_only=True)
    book_name = fields.Str()
    page_number = fields.Float()
    line_number = fields.Float()
    col_number = fields.Float()

    @post_dump
    def add_full_name(self, data, **kwargs):
        data["author_full"] = "{}, {}".format(
            data["author"]["last"], data["author"]["first"]
        )
        return data


class Author:
    def __init__(self, id, first, last, book_count, age, address):
        self.id = id
        self.first = first
        self.last = last
        self.book_count = book_count
        self.age = age
        self.address = address


class Quote:
    def __init__(
        self,
        id,
        author,
        content,
        posted_at,
        book_name,
        page_number,
        line_number,
        col_number,
    ):
        self.id = id
        self.author = author
        self.content = content
        self.posted_at = posted_at
        self.book_name = book_name
        self.page_number = page_number
        self.line_number = line_number
        self.col_number = col_number


def run_timeit(quotes, iterations, repeat, *, profile=False):
    quotes_schema = QuoteSchema(many=True)
    if profile:
        profile = cProfile.Profile()
        profile.enable()

    gc.collect()
    best = min(
        timeit.repeat(
            lambda: quotes_schema.dump(quotes),
            "gc.enable()",
            number=iterations,
            repeat=repeat,
        )
    )
    if profile:
        profile.disable()
        profile.dump_stats("marshmallow.pprof")

    return best * 1e6 / iterations / len(quotes)


def main():
    parser = argparse.ArgumentParser(description="Runs a benchmark of Marshmallow.")
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of iterations to run per test.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=5,
        help="Number of times to repeat the performance test.  The minimum will "
        "be used.",
    )
    parser.add_argument(
        "--object-count", type=int, default=20, help="Number of objects to dump."
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Whether or not to profile marshmallow while running the benchmark.",
    )
    args = parser.parse_args()

    quotes = [
        Quote(
            i,
            Author(i, "Foo", "Bar", 42, 66, "123 Fake St"),
            "Hello World",
            datetime.datetime(2019, 7, 4, tzinfo=datetime.timezone.utc),
            "The World",
            34,
            3,
            70,
        )
        for i in range(args.object_count)
    ]

    print(
        f"Benchmark Result: {run_timeit(quotes, args.iterations, args.repeat, profile=args.profile):.2f} usec/dump"
    )


if __name__ == "__main__":
    main()


# === src/marshmallow/fields.py ===
# ruff: noqa: F841, SLF001
from __future__ import annotations

import abc
import collections
import copy
import datetime as dt
import decimal
import email.utils
import ipaddress
import math
import numbers
import typing
import uuid
from collections.abc import Mapping as _Mapping
from enum import Enum as EnumType

try:
    from typing import Unpack
except ImportError:  # Remove when dropping Python 3.10
    from typing_extensions import Unpack

# Remove when dropping Python 3.10
try:
    from backports.datetime_fromisoformat import MonkeyPatch
except ImportError:
    pass
else:
    MonkeyPatch.patch_fromisoformat()

from marshmallow import class_registry, types, utils, validate
from marshmallow.constants import missing as missing_
from marshmallow.exceptions import (
    StringNotCollectionError,
    ValidationError,
    _FieldInstanceResolutionError,
)
from marshmallow.validate import And, Length

if typing.TYPE_CHECKING:
    from marshmallow.schema import Schema, SchemaMeta


__all__ = [
    "IP",
    "URL",
    "UUID",
    "AwareDateTime",
    "Bool",
    "Boolean",
    "Constant",
    "Date",
    "DateTime",
    "Decimal",
    "Dict",
    "Email",
    "Enum",
    "Field",
    "Float",
    "Function",
    "IPInterface",
    "IPv4",
    "IPv4Interface",
    "IPv6",
    "IPv6Interface",
    "Int",
    "Integer",
    "List",
    "Mapping",
    "Method",
    "NaiveDateTime",
    "Nested",
    "Number",
    "Pluck",
    "Raw",
    "Str",
    "String",
    "Time",
    "TimeDelta",
    "Tuple",
    "Url",
]

_InternalT = typing.TypeVar("_InternalT")


class _BaseFieldKwargs(typing.TypedDict, total=False):
    load_default: typing.Any
    dump_default: typing.Any
    data_key: str | None
    attribute: str | None
    validate: types.Validator | typing.Iterable[types.Validator] | None
    required: bool
    allow_none: bool | None
    load_only: bool
    dump_only: bool
    error_messages: dict[str, str] | None
    metadata: typing.Mapping[str, typing.Any] | None


def _resolve_field_instance(cls_or_instance: Field | type[Field]) -> Field:
    """Return a Field instance from a Field class or instance.

    :param cls_or_instance: Field class or instance.
    """
    if isinstance(cls_or_instance, type):
        if not issubclass(cls_or_instance, Field):
            raise _FieldInstanceResolutionError
        return cls_or_instance()
    if not isinstance(cls_or_instance, Field):
        raise _FieldInstanceResolutionError
    return cls_or_instance


class Field(typing.Generic[_InternalT]):
    """Base field from which all other fields inherit.
    This class should not be used directly within Schemas.

    :param dump_default: If set, this value will be used during serialization if the
        input value is missing. If not set, the field will be excluded from the
        serialized output if the input value is missing. May be a value or a callable.
    :param load_default: Default deserialization value for the field if the field is not
        found in the input data. May be a value or a callable.
    :param data_key: The name of the dict key in the external representation, i.e.
        the input of `load` and the output of `dump`.
        If `None`, the key will match the name of the field.
    :param attribute: The name of the key/attribute in the internal representation, i.e.
        the output of `load` and the input of `dump`.
        If `None`, the key/attribute will match the name of the field.
        Note: This should only be used for very specific use cases such as
        outputting multiple fields for a single attribute, or using keys/attributes
        that are invalid variable names, unsuitable for field names. In most cases,
        you should use ``data_key`` instead.
    :param validate: Validator or collection of validators that are called
        during deserialization. Validator takes a field's input value as
        its only parameter and returns a boolean.
        If it returns `False`, an :exc:`ValidationError` is raised.
    :param required: Raise a :exc:`ValidationError` if the field value
        is not supplied during deserialization.
    :param allow_none: Set this to `True` if `None` should be considered a valid value during
        validation/deserialization. If set to `False` (the default), `None` is considered invalid input.
        If ``load_default`` is explicitly set to `None` and ``allow_none`` is unset,
        `allow_none` is implicitly set to ``True``.
    :param load_only: If `True` skip this field during serialization, otherwise
        its value will be present in the serialized data.
    :param dump_only: If `True` skip this field during deserialization, otherwise
        its value will be present in the deserialized object. In the context of an
        HTTP API, this effectively marks the field as "read-only".
    :param error_messages: Overrides for `Field.default_error_messages`.
    :param metadata: Extra information to be stored as field metadata.

    .. versionchanged:: 3.0.0b8
        Add ``data_key`` parameter for the specifying the key in the input and
        output data. This parameter replaced both ``load_from`` and ``dump_to``.
    .. versionchanged:: 3.13.0
        Replace ``missing`` and ``default`` parameters with ``load_default`` and ``dump_default``.
    .. versionchanged:: 3.24.0
        `Field <marshmallow.fields.Field>` should no longer be used as a field within a `Schema <marshmallow.Schema>`.
        Use `Raw <marshmallow.fields.Raw>` or another `Field <marshmallow.fields.Field>` subclass instead.
    .. versionchanged:: 4.0.0
        Remove ``context`` property.
    """

    # Some fields, such as Method fields and Function fields, are not expected
    #  to exist as attributes on the objects to serialize. Set this to False
    #  for those fields
    _CHECK_ATTRIBUTE = True

    #: Default error messages for various kinds of errors. The keys in this dictionary
    #: are passed to `Field.make_error`. The values are error messages passed to
    #: :exc:`marshmallow.exceptions.ValidationError`.
    default_error_messages: dict[str, str] = {
        "required": "Missing data for required field.",
        "null": "Field may not be null.",
        "validator_failed": "Invalid value.",
    }

    def __init__(
        self,
        *,
        load_default: typing.Any = missing_,
        dump_default: typing.Any = missing_,
        data_key: str | None = None,
        attribute: str | None = None,
        validate: types.Validator | typing.Iterable[types.Validator] | None = None,
        required: bool = False,
        allow_none: bool | None = None,
        load_only: bool = False,
        dump_only: bool = False,
        error_messages: dict[str, str] | None = None,
        metadata: typing.Mapping[str, typing.Any] | None = None,
    ) -> None:
        self.dump_default = dump_default
        self.load_default = load_default

        self.attribute = attribute
        self.data_key = data_key
        self.validate = validate
        if validate is None:
            self.validators = []
        elif callable(validate):
            self.validators = [validate]
        elif utils.is_iterable_but_not_string(validate):
            self.validators = list(validate)
        else:
            raise ValueError(
                "The 'validate' parameter must be a callable "
                "or a collection of callables."
            )

        # If allow_none is None and load_default is None
        # None should be considered valid by default
        self.allow_none = load_default is None if allow_none is None else allow_none
        self.load_only = load_only
        self.dump_only = dump_only
        if required is True and load_default is not missing_:
            raise ValueError("'load_default' must not be set for required fields.")
        self.required = required

        metadata = metadata or {}
        self.metadata = metadata
        # Collect default error message from self and parent classes
        messages: dict[str, str] = {}
        for cls in reversed(self.__class__.__mro__):
            messages.update(getattr(cls, "default_error_messages", {}))
        messages.update(error_messages or {})
        self.error_messages = messages

        self.parent: Field | Schema | None = None
        self.name: str | None = None
        self.root: Schema | None = None

    def __repr__(self) -> str:
        return (
            f"<fields.{self.__class__.__name__}(dump_default={self.dump_default!r}, "
            f"attribute={self.attribute!r}, "
            f"validate={self.validate}, required={self.required}, "
            f"load_only={self.load_only}, dump_only={self.dump_only}, "
            f"load_default={self.load_default}, allow_none={self.allow_none}, "
            f"error_messages={self.error_messages})>"
        )

    def __deepcopy__(self, memo):
        return copy.copy(self)

    def get_value(
        self,
        obj: typing.Any,
        attr: str,
        accessor: (
            typing.Callable[[typing.Any, str, typing.Any], typing.Any] | None
        ) = None,
        default: typing.Any = missing_,
    ) -> _InternalT:
        """Return the value for a given key from an object.

        :param obj: The object to get the value from.
        :param attr: The attribute/key in `obj` to get the value from.
        :param accessor: A callable used to retrieve the value of `attr` from
            the object `obj`. Defaults to `marshmallow.utils.get_value`.
        """
        accessor_func = accessor or utils.get_value
        check_key = attr if self.attribute is None else self.attribute
        return accessor_func(obj, check_key, default)

    def _validate(self, value: typing.Any) -> None:
        """Perform validation on ``value``. Raise a :exc:`ValidationError` if validation
        does not succeed.
        """
        self._validate_all(value)

    @property
    def _validate_all(self) -> typing.Callable[[typing.Any], None]:
        return And(*self.validators)

    def make_error(self, key: str, **kwargs) -> ValidationError:
        """Helper method to make a `ValidationError` with an error message
        from ``self.error_messages``.
        """
        try:
            msg = self.error_messages[key]
        except KeyError as error:
            class_name = self.__class__.__name__
            message = (
                f"ValidationError raised by `{class_name}`, but error key `{key}` does "
                "not exist in the `error_messages` dictionary."
            )
            raise AssertionError(message) from error
        if isinstance(msg, (str, bytes)):
            msg = msg.format(**kwargs)
        return ValidationError(msg)

    def _validate_missing(self, value: typing.Any) -> None:
        """Validate missing values. Raise a :exc:`ValidationError` if
        `value` should be considered missing.
        """
        if value is missing_ and self.required:
            raise self.make_error("required")
        if value is None and not self.allow_none:
            raise self.make_error("null")

    def serialize(
        self,
        attr: str,
        obj: typing.Any,
        accessor: (
            typing.Callable[[typing.Any, str, typing.Any], typing.Any] | None
        ) = None,
        **kwargs,
    ):
        """Pulls the value for the given key from the object, applies the
        field's formatting and returns the result.

        :param attr: The attribute/key to get from the object.
        :param obj: The object to access the attribute/key from.
        :param accessor: Function used to access values from ``obj``.
        :param kwargs: Field-specific keyword arguments.
        """
        if self._CHECK_ATTRIBUTE:
            value = self.get_value(obj, attr, accessor=accessor)
            if value is missing_:
                default = self.dump_default
                value = default() if callable(default) else default
            if value is missing_:
                return value
        else:
            value = None
        return self._serialize(value, attr, obj, **kwargs)

    # If value is None, None may be returned
    @typing.overload
    def deserialize(
        self,
        value: None,
        attr: str | None = None,
        data: typing.Mapping[str, typing.Any] | None = None,
        **kwargs,
    ) -> None | _InternalT: ...

    # If value is not None, internal type is returned
    @typing.overload
    def deserialize(
        self,
        value: typing.Any,
        attr: str | None = None,
        data: typing.Mapping[str, typing.Any] | None = None,
        **kwargs,
    ) -> _InternalT: ...

    def deserialize(
        self,
        value: typing.Any,
        attr: str | None = None,
        data: typing.Mapping[str, typing.Any] | None = None,
        **kwargs,
    ) -> _InternalT | None:
        """Deserialize ``value``.

        :param value: The value to deserialize.
        :param attr: The attribute/key in `data` to deserialize.
        :param data: The raw input data passed to `Schema.load <marshmallow.Schema.load>`.
        :param kwargs: Field-specific keyword arguments.
        :raise ValidationError: If an invalid value is passed or if a required value
            is missing.
        """
        # Validate required fields, deserialize, then validate
        # deserialized value
        self._validate_missing(value)
        if value is missing_:
            _miss = self.load_default
            return _miss() if callable(_miss) else _miss
        if self.allow_none and value is None:
            return None
        output = self._deserialize(value, attr, data, **kwargs)
        self._validate(output)
        return output

    # Methods for concrete classes to override.

    def _bind_to_schema(self, field_name: str, parent: Schema | Field) -> None:
        """Update field with values from its parent schema. Called by
                `Schema._bind_field <marshmallow.Schema._bind_field>`.

        :param field_name: Field name set in schema.
        :param parent: Parent object.
        """
        self.parent = self.parent or parent
        self.name = self.name or field_name
        self.root = self.root or (
            self.parent.root if isinstance(self.parent, Field) else self.parent
        )

    def _serialize(
        self, value: _InternalT | None, attr: str | None, obj: typing.Any, **kwargs
    ) -> typing.Any:
        """Serializes ``value`` to a basic Python datatype. Noop by default.
        Concrete :class:`Field` classes should implement this method.

        Example: ::

            class TitleCase(Field):
                def _serialize(self, value, attr, obj, **kwargs):
                    if not value:
                        return ""
                    return str(value).title()

        :param value: The value to be serialized.
        :param attr: The attribute or key on the object to be serialized.
        :param obj: The object the value was pulled from.
        :param kwargs: Field-specific keyword arguments.
        :return: The serialized value
        """
        return value

    def _deserialize(
        self,
        value: typing.Any,
        attr: str | None,
        data: typing.Mapping[str, typing.Any] | None,
        **kwargs,
    ) -> _InternalT:
        """Deserialize value. Concrete :class:`Field` classes should implement this method.

        :param value: The value to be deserialized.
        :param attr: The attribute/key in `data` to be deserialized.
        :param data: The raw input data passed to the `Schema.load <marshmallow.Schema.load>`.
        :param kwargs: Field-specific keyword arguments.
        :raise ValidationError: In case of formatting or validation failure.
        :return: The deserialized value.

        .. versionchanged:: 3.0.0
            Added ``**kwargs`` to signature.
        """
        return value


class Raw(Field[typing.Any]):
    """Field that applies no formatting."""


class Nested(Field):
    """Allows you to nest a :class:`Schema <marshmallow.Schema>`
    inside a field.

    Examples: ::

        class ChildSchema(Schema):
            id = fields.Str()
            name = fields.Str()
            # Use lambda functions when you need two-way nesting or self-nesting
            parent = fields.Nested(lambda: ParentSchema(only=("id",)), dump_only=True)
            siblings = fields.List(
                fields.Nested(lambda: ChildSchema(only=("id", "name")))
            )


        class ParentSchema(Schema):
            id = fields.Str()
            children = fields.List(
                fields.Nested(ChildSchema(only=("id", "parent", "siblings")))
            )
            spouse = fields.Nested(lambda: ParentSchema(only=("id",)))

    When passing a `Schema <marshmallow.Schema>` instance as the first argument,
    the instance's ``exclude``, ``only``, and ``many`` attributes will be respected.

    Therefore, when passing the ``exclude``, ``only``, or ``many`` arguments to `fields.Nested`,
    you should pass a `Schema <marshmallow.Schema>` class (not an instance) as the first argument.

    ::

        # Yes
        author = fields.Nested(UserSchema, only=("id", "name"))

        # No
        author = fields.Nested(UserSchema(), only=("id", "name"))

    :param nested: `Schema <marshmallow.Schema>` instance, class, class name (string), dictionary, or callable that
        returns a `Schema <marshmallow.Schema>` or dictionary.
        Dictionaries are converted with `Schema.from_dict <marshmallow.Schema.from_dict>`.
    :param exclude: A list or tuple of fields to exclude.
    :param only: A list or tuple of fields to marshal. If `None`, all fields are marshalled.
        This parameter takes precedence over ``exclude``.
    :param many: Whether the field is a collection of objects.
    :param unknown: Whether to exclude, include, or raise an error for unknown
        fields in the data. Use `EXCLUDE`, `INCLUDE` or `RAISE`.
    :param kwargs: The same keyword arguments that :class:`Field` receives.
    """

    #: Default error messages.
    default_error_messages = {"type": "Invalid type."}

    def __init__(
        self,
        nested: (
            Schema
            | SchemaMeta
            | str
            | dict[str, Field]
            | typing.Callable[[], Schema | SchemaMeta | dict[str, Field]]
        ),
        *,
        only: types.StrSequenceOrSet | None = None,
        exclude: types.StrSequenceOrSet = (),
        many: bool = False,
        unknown: types.UnknownOption | None = None,
        **kwargs: Unpack[_BaseFieldKwargs],
    ):
        # Raise error if only or exclude is passed as string, not list of strings
        if only is not None and not utils.is_sequence_but_not_string(only):
            raise StringNotCollectionError('"only" should be a collection of strings.')
        if not utils.is_sequence_but_not_string(exclude):
            raise StringNotCollectionError(
                '"exclude" should be a collection of strings.'
            )
        self.nested = nested
        self.only = only
        self.exclude = exclude
        self.many = many
        self.unknown = unknown
        self._schema: Schema | None = None  # Cached Schema instance
        super().__init__(**kwargs)

    @property
    def schema(self) -> Schema:
        """The nested Schema object."""
        if not self._schema:
            if callable(self.nested) and not isinstance(self.nested, type):
                nested = self.nested()
            else:
                nested = typing.cast("Schema", self.nested)
            # defer the import of `marshmallow.schema` to avoid circular imports
            from marshmallow.schema import Schema

            if isinstance(nested, dict):
                nested = Schema.from_dict(nested)

            if isinstance(nested, Schema):
                self._schema = copy.copy(nested)
                # Respect only and exclude passed from parent and re-initialize fields
                set_class = typing.cast("type[set]", self._schema.set_class)
                if self.only is not None:
                    if self._schema.only is not None:
                        original = self._schema.only
                    else:  # only=None -> all fields
                        original = self._schema.fields.keys()
                    self._schema.only = set_class(self.only) & set_class(original)
                if self.exclude:
                    original = self._schema.exclude
                    self._schema.exclude = set_class(self.exclude) | set_class(original)
                self._schema._init_fields()
            else:
                if isinstance(nested, type) and issubclass(nested, Schema):
                    schema_class: type[Schema] = nested
                elif not isinstance(nested, (str, bytes)):
                    raise ValueError(
                        "`Nested` fields must be passed a "
                        f"`Schema`, not {nested.__class__}."
                    )
                else:
                    schema_class = class_registry.get_class(nested, all=False)
                self._schema = schema_class(
                    many=self.many,
                    only=self.only,
                    exclude=self.exclude,
                    load_only=self._nested_normalized_option("load_only"),
                    dump_only=self._nested_normalized_option("dump_only"),
                )
        return self._schema

    def _nested_normalized_option(self, option_name: str) -> list[str]:
        nested_field = f"{self.name}."
        return [
            field.split(nested_field, 1)[1]
            for field in getattr(self.root, option_name, set())
            if field.startswith(nested_field)
        ]

    def _serialize(self, nested_obj, attr, obj, **kwargs):
        # Load up the schema first. This allows a RegistryError to be raised
        # if an invalid schema name was passed
        schema = self.schema
        if nested_obj is None:
            return None
        many = schema.many or self.many
        return schema.dump(nested_obj, many=many)

    def _test_collection(self, value: typing.Any) -> None:
        many = self.schema.many or self.many
        if many and not utils.is_collection(value):
            raise self.make_error("type", input=value, type=value.__class__.__name__)

    def _load(
        self, value: typing.Any, partial: bool | types.StrSequenceOrSet | None = None
    ):
        try:
            valid_data = self.schema.load(value, unknown=self.unknown, partial=partial)
        except ValidationError as error:
            raise ValidationError(
                error.messages, valid_data=error.valid_data
            ) from error
        return valid_data

    def _deserialize(
        self,
        value: typing.Any,
        attr: str | None,
        data: typing.Mapping[str, typing.Any] | None,
        partial: bool | types.StrSequenceOrSet | None = None,
        **kwargs,
    ):
        """Same as :meth:`Field._deserialize` with additional ``partial`` argument.

        :param partial: For nested schemas, the ``partial``
            parameter passed to `marshmallow.Schema.load`.

        .. versionchanged:: 3.0.0
            Add ``partial`` parameter.
        """
        self._test_collection(value)
        return self._load(value, partial=partial)


class Pluck(Nested):
    """Allows you to replace nested data with one of the data's fields.

    Example: ::

        from marshmallow import Schema, fields


        class ArtistSchema(Schema):
            id = fields.Int()
            name = fields.Str()


        class AlbumSchema(Schema):
            artist = fields.Pluck(ArtistSchema, "id")


        in_data = {"artist": 42}
        loaded = AlbumSchema().load(in_data)  # => {'artist': {'id': 42}}
        dumped = AlbumSchema().dump(loaded)  # => {'artist': 42}

    :param nested: The Schema class or class name (string) to nest
    :param str field_name: The key to pluck a value from.
    :param kwargs: The same keyword arguments that :class:`Nested` receives.
    """

    def __init__(
        self,
        nested: Schema | SchemaMeta | str | typing.Callable[[], Schema],
        field_name: str,
        *,
        many: bool = False,
        unknown: types.UnknownOption | None = None,
        **kwargs: Unpack[_BaseFieldKwargs],
    ):
        super().__init__(
            nested, only=(field_name,), many=many, unknown=unknown, **kwargs
        )
        self.field_name = field_name

    @property
    def _field_data_key(self) -> str:
        only_field = self.schema.fields[self.field_name]
        return only_field.data_key or self.field_name

    def _serialize(self, nested_obj, attr, obj, **kwargs):
        ret = super()._serialize(nested_obj, attr, obj, **kwargs)
        if ret is None:
            return None
        if self.many:
            return utils.pluck(ret, key=self._field_data_key)
        return ret[self._field_data_key]

    def _deserialize(self, value, attr, data, partial=None, **kwargs):
        self._test_collection(value)
        if self.many:
            value = [{self._field_data_key: v} for v in value]
        else:
            value = {self._field_data_key: value}
        return self._load(value, partial=partial)


class List(Field[list[typing.Optional[_InternalT]]]):
    """A list field, composed with another `Field` class or
    instance.

    Example: ::

        numbers = fields.List(fields.Float())

    :param cls_or_instance: A field class or instance.
    :param kwargs: The same keyword arguments that :class:`Field` receives.

    .. versionchanged:: 3.0.0rc9
        Does not serialize scalar values to single-item lists.
    """

    #: Default error messages.
    default_error_messages = {"invalid": "Not a valid list."}

    def __init__(
        self,
        cls_or_instance: Field[_InternalT] | type[Field[_InternalT]],
        **kwargs: Unpack[_BaseFieldKwargs],
    ):
        super().__init__(**kwargs)
        try:
            self.inner: Field[_InternalT] = _resolve_field_instance(cls_or_instance)
        except _FieldInstanceResolutionError as error:
            raise ValueError(
                "The list elements must be a subclass or instance of "
                "marshmallow.fields.Field."
            ) from error
        if isinstance(self.inner, Nested):
            self.only = self.inner.only
            self.exclude = self.inner.exclude

    def _bind_to_schema(self, field_name: str, parent: Schema | Field) -> None:
        super()._bind_to_schema(field_name, parent)
        self.inner = copy.deepcopy(self.inner)
        self.inner._bind_to_schema(field_name, self)
        if isinstance(self.inner, Nested):
            self.inner.only = self.only
            self.inner.exclude = self.exclude

    def _serialize(self, value, attr, obj, **kwargs) -> list[_InternalT] | None:
        if value is None:
            return None
        return [self.inner._serialize(each, attr, obj, **kwargs) for each in value]

    def _deserialize(self, value, attr, data, **kwargs) -> list[_InternalT | None]:
        if not utils.is_collection(value):
            raise self.make_error("invalid")

        result = []
        errors = {}
        for idx, each in enumerate(value):
            try:
                result.append(self.inner.deserialize(each, **kwargs))
            except ValidationError as error:
                if error.valid_data is not None:
                    result.append(typing.cast("_InternalT", error.valid_data))
                errors.update({idx: error.messages})
        if errors:
            raise ValidationError(errors, valid_data=result)
        return result


class Tuple(Field[tuple]):
    """A tuple field, composed of a fixed number of other `Field` classes or
    instances

    Example: ::

        row = Tuple((fields.String(), fields.Integer(), fields.Float()))

    .. note::
        Because of the structured nature of `collections.namedtuple` and
        `typing.NamedTuple`, using a Schema within a Nested field for them is
        more appropriate than using a `Tuple` field.

    :param tuple_fields: An iterable of field classes or
        instances.
    :param kwargs: The same keyword arguments that :class:`Field` receives.

    .. versionadded:: 3.0.0rc4
    """

    #: Default error messages.
    default_error_messages = {"invalid": "Not a valid tuple."}

    def __init__(
        self,
        tuple_fields: typing.Iterable[Field] | typing.Iterable[type[Field]],
        **kwargs: Unpack[_BaseFieldKwargs],
    ):
        super().__init__(**kwargs)
        if not utils.is_collection(tuple_fields):
            raise ValueError(
                "tuple_fields must be an iterable of Field classes or instances."
            )

        try:
            self.tuple_fields = [
                _resolve_field_instance(cls_or_instance)
                for cls_or_instance in tuple_fields
            ]
        except _FieldInstanceResolutionError as error:
            raise ValueError(
                'Elements of "tuple_fields" must be subclasses or '
                "instances of marshmallow.fields.Field."
            ) from error

        self.validate_length = Length(equal=len(self.tuple_fields))

    def _bind_to_schema(self, field_name: str, parent: Schema | Field) -> None:
        super()._bind_to_schema(field_name, parent)
        new_tuple_fields = []
        for field in self.tuple_fields:
            new_field = copy.deepcopy(field)
            new_field._bind_to_schema(field_name, self)
            new_tuple_fields.append(new_field)

        self.tuple_fields = new_tuple_fields

    def _serialize(
        self, value: tuple | None, attr: str | None, obj: typing.Any, **kwargs
    ) -> tuple | None:
        if value is None:
            return None

        return tuple(
            field._serialize(each, attr, obj, **kwargs)
            for field, each in zip(self.tuple_fields, value)
        )

    def _deserialize(
        self,
        value: typing.Any,
        attr: str | None,
        data: typing.Mapping[str, typing.Any] | None,
        **kwargs,
    ) -> tuple:
        if not utils.is_sequence_but_not_string(value):
            raise self.make_error("invalid")

        self.validate_length(value)

        result = []
        errors = {}

        for idx, (field, each) in enumerate(zip(self.tuple_fields, value)):
            try:
                result.append(field.deserialize(each, **kwargs))
            except ValidationError as error:
                if error.valid_data is not None:
                    result.append(error.valid_data)
                errors.update({idx: error.messages})
        if errors:
            raise ValidationError(errors, valid_data=result)

        return tuple(result)


class String(Field[str]):
    """A string field.

    :param kwargs: The same keyword arguments that :class:`Field` receives.
    """

    #: Default error messages.
    default_error_messages = {
        "invalid": "Not a valid string.",
        "invalid_utf8": "Not a valid utf-8 string.",
    }

    def _serialize(self, value, attr, obj, **kwargs) -> str | None:
        if value is None:
            return None
        return utils.ensure_text_type(value)

    def _deserialize(self, value, attr, data, **kwargs) -> str:
        if not isinstance(value, (str, bytes)):
            raise self.make_error("invalid")
        try:
            return utils.ensure_text_type(value)
        except UnicodeDecodeError as error:
            raise self.make_error("invalid_utf8") from error


class UUID(Field[uuid.UUID]):
    """A UUID field."""

    #: Default error messages.
    default_error_messages = {"invalid_uuid": "Not a valid UUID."}

    def _validated(self, value) -> uuid.UUID:
        """Format the value or raise a :exc:`ValidationError` if an error occurs."""
        if isinstance(value, uuid.UUID):
            return value
        try:
            if isinstance(value, bytes) and len(value) == 16:
                return uuid.UUID(bytes=value)
            return uuid.UUID(value)
        except (ValueError, AttributeError, TypeError) as error:
            raise self.make_error("invalid_uuid") from error

    def _serialize(self, value, attr, obj, **kwargs) -> str | None:
        if value is None:
            return None
        return str(value)

    def _deserialize(self, value, attr, data, **kwargs) -> uuid.UUID:
        return self._validated(value)


_NumT = typing.TypeVar("_NumT")


class Number(Field[_NumT]):
    """Base class for number fields. This class should not be used within schemas.

    :param as_string: If `True`, format the serialized value as a string.
    :param kwargs: The same keyword arguments that :class:`Field` receives.

    .. versionchanged:: 3.24.0
        `Number <marshmallow.fields.Number>` should no longer be used as a field within a `Schema <marshmallow.Schema>`.
        Use `Integer <marshmallow.fields.Integer>`, `Float <marshmallow.fields.Float>`, or `Decimal <marshmallow.fields.Decimal>` instead.
    """

    num_type: type[_NumT]

    #: Default error messages.
    default_error_messages = {
        "invalid": "Not a valid number.",
        "too_large": "Number too large.",
    }

    def __init__(self, *, as_string: bool = False, **kwargs: Unpack[_BaseFieldKwargs]):
        self.as_string = as_string
        super().__init__(**kwargs)

    def _format_num(self, value) -> _NumT:
        """Return the number value for value, given this field's `num_type`."""
        return self.num_type(value)  # type: ignore[call-arg]

    def _validated(self, value: typing.Any) -> _NumT:
        """Format the value or raise a :exc:`ValidationError` if an error occurs."""
        # (value is True or value is False) is ~5x faster than isinstance(value, bool)
        if value is True or value is False:
            raise self.make_error("invalid", input=value)
        try:
            return self._format_num(value)
        except (TypeError, ValueError) as error:
            raise self.make_error("invalid", input=value) from error
        except OverflowError as error:
            raise self.make_error("too_large", input=value) from error

    def _to_string(self, value: _NumT) -> str:
        return str(value)

    def _serialize(self, value, attr, obj, **kwargs) -> str | _NumT | None:
        """Return a string if `self.as_string=True`, otherwise return this field's `num_type`."""
        if value is None:
            return None
        ret: _NumT = self._format_num(value)
        return self._to_string(ret) if self.as_string else ret

    def _deserialize(self, value, attr, data, **kwargs) -> _NumT:
        return self._validated(value)


class Integer(Number[int]):
    """An integer field.

    :param strict: If `True`, only integer types are valid.
        Otherwise, any value castable to `int` is valid.
    :param kwargs: The same keyword arguments that :class:`Number` receives.
    """

    num_type = int

    #: Default error messages.
    default_error_messages = {"invalid": "Not a valid integer."}

    def __init__(
        self,
        *,
        strict: bool = False,
        as_string: bool = False,
        **kwargs: Unpack[_BaseFieldKwargs],
    ):
        self.strict = strict
        super().__init__(as_string=as_string, **kwargs)

    # override Number
    def _validated(self, value: typing.Any) -> int:
        if self.strict and not isinstance(value, numbers.Integral):
            raise self.make_error("invalid", input=value)
        return super()._validated(value)


class Float(Number[float]):
    """A double as an IEEE-754 double precision string.

    :param allow_nan: If `True`, `NaN`, `Infinity` and `-Infinity` are allowed,
        even though they are illegal according to the JSON specification.
    :param as_string: If `True`, format the value as a string.
    :param kwargs: The same keyword arguments that :class:`Number` receives.
    """

    num_type = float

    #: Default error messages.
    default_error_messages = {
        "special": "Special numeric values (nan or infinity) are not permitted."
    }

    def __init__(
        self,
        *,
        allow_nan: bool = False,
        as_string: bool = False,
        **kwargs: Unpack[_BaseFieldKwargs],
    ):
        self.allow_nan = allow_nan
        super().__init__(as_string=as_string, **kwargs)

    def _validated(self, value: typing.Any) -> float:
        num = super()._validated(value)
        if self.allow_nan is False:
            if math.isnan(num) or num == float("inf") or num == float("-inf"):
                raise self.make_error("special")
        return num


class Decimal(Number[decimal.Decimal]):
    """A field that (de)serializes to the Python ``decimal.Decimal`` type.
    It's safe to use when dealing with money values, percentages, ratios
    or other numbers where precision is critical.

    .. warning::

        This field serializes to a `decimal.Decimal` object by default. If you need
        to render your data as JSON, keep in mind that the `json` module from the
        standard library does not encode `decimal.Decimal`. Therefore, you must use
        a JSON library that can handle decimals, such as `simplejson`, or serialize
        to a string by passing ``as_string=True``.

    .. warning::

        If a JSON `float` value is passed to this field for deserialization it will
        first be cast to its corresponding `string` value before being deserialized
        to a `decimal.Decimal` object. The default `__str__` implementation of the
        built-in Python `float` type may apply a destructive transformation upon
        its input data and therefore cannot be relied upon to preserve precision.
        To avoid this, you can instead pass a JSON `string` to be deserialized
        directly.

    :param places: How many decimal places to quantize the value. If `None`, does
        not quantize the value.
    :param rounding: How to round the value during quantize, for example
        `decimal.ROUND_UP`. If `None`, uses the rounding value from
        the current thread's context.
    :param allow_nan: If `True`, `NaN`, `Infinity` and `-Infinity` are allowed,
        even though they are illegal according to the JSON specification.
    :param as_string: If `True`, serialize to a string instead of a Python
        `decimal.Decimal` type.
    :param kwargs: The same keyword arguments that :class:`Number` receives.
    """

    num_type = decimal.Decimal

    #: Default error messages.
    default_error_messages = {
        "special": "Special numeric values (nan or infinity) are not permitted."
    }

    def __init__(
        self,
        places: int | None = None,
        rounding: str | None = None,
        *,
        allow_nan: bool = False,
        as_string: bool = False,
        **kwargs: Unpack[_BaseFieldKwargs],
    ):
        self.places = (
            decimal.Decimal((0, (1,), -places)) if places is not None else None
        )
        self.rounding = rounding
        self.allow_nan = allow_nan
        super().__init__(as_string=as_string, **kwargs)

    # override Number
    def _format_num(self, value):
        num = decimal.Decimal(str(value))
        if self.allow_nan:
            if num.is_nan():
                return decimal.Decimal("NaN")  # avoid sNaN, -sNaN and -NaN
        if self.places is not None and num.is_finite():
            num = num.quantize(self.places, rounding=self.rounding)
        return num

    # override Number
    def _validated(self, value: typing.Any) -> decimal.Decimal:
        try:
            num = super()._validated(value)
        except decimal.InvalidOperation as error:
            raise self.make_error("invalid") from error
        if not self.allow_nan and (num.is_nan() or num.is_infinite()):
            raise self.make_error("special")
        return num

    # override Number
    def _to_string(self, value: decimal.Decimal) -> str:
        return format(value, "f")


class Boolean(Field[bool]):
    """A boolean field.

    :param truthy: Values that will (de)serialize to `True`. If an empty
        set, any non-falsy value will deserialize to `True`. If `None`,
        `marshmallow.fields.Boolean.truthy` will be used.
    :param falsy: Values that will (de)serialize to `False`. If `None`,
        `marshmallow.fields.Boolean.falsy` will be used.
    :param kwargs: The same keyword arguments that :class:`Field` receives.
    """

    #: Default truthy values.
    truthy = {
        "t",
        "T",
        "true",
        "True",
        "TRUE",
        "on",
        "On",
        "ON",
        "y",
        "Y",
        "yes",
        "Yes",
        "YES",
        "1",
        1,
        # Equal to 1
        # True,
    }
    #: Default falsy values.
    falsy = {
        "f",
        "F",
        "false",
        "False",
        "FALSE",
        "off",
        "Off",
        "OFF",
        "n",
        "N",
        "no",
        "No",
        "NO",
        "0",
        0,
        # Equal to 0
        # 0.0,
        # False,
    }

    #: Default error messages.
    default_error_messages = {"invalid": "Not a valid boolean."}

    def __init__(
        self,
        *,
        truthy: typing.Iterable | None = None,
        falsy: typing.Iterable | None = None,
        **kwargs: Unpack[_BaseFieldKwargs],
    ):
        super().__init__(**kwargs)

        if truthy is not None:
            self.truthy = set(truthy)
        if falsy is not None:
            self.falsy = set(falsy)

    def _deserialize(
        self,
        value: typing.Any,
        attr: str | None,
        data: typing.Mapping[str, typing.Any] | None,
        **kwargs,
    ) -> bool:
        if not self.truthy:
            return bool(value)
        try:
            if value in self.truthy:
                return True
            if value in self.falsy:
                return False
        except TypeError as error:
            raise self.make_error("invalid", input=value) from error
        raise self.make_error("invalid", input=value)


_D = typing.TypeVar("_D", dt.datetime, dt.date, dt.time)


class _TemporalField(Field[_D], metaclass=abc.ABCMeta):
    """Base field for date and time related fields including common (de)serialization logic."""

    # Subclasses should define each of these class constants
    SERIALIZATION_FUNCS: dict[str, typing.Callable[[_D], str | float]]
    DESERIALIZATION_FUNCS: dict[str, typing.Callable[[str], _D]]
    DEFAULT_FORMAT: str
    OBJ_TYPE: str
    SCHEMA_OPTS_VAR_NAME: str

    default_error_messages = {
        "invalid": "Not a valid {obj_type}.",
        "invalid_awareness": "Not a valid {awareness} {obj_type}.",
        "format": '"{input}" cannot be formatted as a {obj_type}.',
    }

    def __init__(
        self,
        format: str | None = None,  # noqa: A002
        **kwargs: Unpack[_BaseFieldKwargs],
    ) -> None:
        super().__init__(**kwargs)
        # Allow this to be None. It may be set later in the ``_serialize``
        # or ``_deserialize`` methods. This allows a Schema to dynamically set the
        # format, e.g. from a Meta option
        self.format = format

    def _bind_to_schema(self, field_name, parent):
        super()._bind_to_schema(field_name, parent)
        self.format = (
            self.format
            or getattr(self.root.opts, self.SCHEMA_OPTS_VAR_NAME)
            or self.DEFAULT_FORMAT
        )

    def _serialize(self, value: _D | None, attr, obj, **kwargs) -> str | float | None:
        if value is None:
            return None
        data_format = self.format or self.DEFAULT_FORMAT
        format_func = self.SERIALIZATION_FUNCS.get(data_format)
        if format_func:
            return format_func(value)
        return value.strftime(data_format)

    def _deserialize(self, value, attr, data, **kwargs) -> _D:
        internal_type: type[_D] = getattr(dt, self.OBJ_TYPE)
        if isinstance(value, internal_type):
            return value
        data_format = self.format or self.DEFAULT_FORMAT
        func = self.DESERIALIZATION_FUNCS.get(data_format)
        try:
            if func:
                return func(value)
            return self._make_object_from_format(value, data_format)
        except (TypeError, AttributeError, ValueError) as error:
            raise self.make_error(
                "invalid", input=value, obj_type=self.OBJ_TYPE
            ) from error

    @staticmethod
    @abc.abstractmethod
    def _make_object_from_format(value: typing.Any, data_format: str) -> _D: ...


class DateTime(_TemporalField[dt.datetime]):
    """A formatted datetime string.

    Example: ``'2014-12-22T03:12:58.019077+00:00'``

    :param format: Either ``"rfc"`` (for RFC822), ``"iso"`` (for ISO8601),
        ``"timestamp"``, ``"timestamp_ms"`` (for a POSIX timestamp) or a date format string.
        If `None`, defaults to "iso".
    :param kwargs: The same keyword arguments that :class:`Field` receives.

    .. versionchanged:: 3.0.0rc9
        Does not modify timezone information on (de)serialization.
    .. versionchanged:: 3.19
        Add timestamp as a format.
    """

    SERIALIZATION_FUNCS: dict[str, typing.Callable[[dt.datetime], str | float]] = {
        "iso": dt.datetime.isoformat,
        "iso8601": dt.datetime.isoformat,
        "rfc": email.utils.format_datetime,
        "rfc822": email.utils.format_datetime,
        "timestamp": utils.timestamp,
        "timestamp_ms": utils.timestamp_ms,
    }

    DESERIALIZATION_FUNCS: dict[str, typing.Callable[[str], dt.datetime]] = {
        "iso": dt.datetime.fromisoformat,
        "iso8601": dt.datetime.fromisoformat,
        "rfc": email.utils.parsedate_to_datetime,
        "rfc822": email.utils.parsedate_to_datetime,
        "timestamp": utils.from_timestamp,
        "timestamp_ms": utils.from_timestamp_ms,
    }

    DEFAULT_FORMAT = "iso"

    OBJ_TYPE = "datetime"

    SCHEMA_OPTS_VAR_NAME = "datetimeformat"

    @staticmethod
    def _make_object_from_format(value, data_format) -> dt.datetime:
        return dt.datetime.strptime(value, data_format)


class NaiveDateTime(DateTime):
    """A formatted naive datetime string.

    :param format: See :class:`DateTime`.
    :param timezone: Used on deserialization. If `None`,
        aware datetimes are rejected. If not `None`, aware datetimes are
        converted to this timezone before their timezone information is
        removed.
    :param kwargs: The same keyword arguments that :class:`Field` receives.

    .. versionadded:: 3.0.0rc9
    """

    AWARENESS = "naive"

    def __init__(
        self,
        format: str | None = None,  # noqa: A002
        *,
        timezone: dt.timezone | None = None,
        **kwargs: Unpack[_BaseFieldKwargs],
    ) -> None:
        super().__init__(format=format, **kwargs)
        self.timezone = timezone

    def _deserialize(self, value, attr, data, **kwargs) -> dt.datetime:
        ret = super()._deserialize(value, attr, data, **kwargs)
        if utils.is_aware(ret):
            if self.timezone is None:
                raise self.make_error(
                    "invalid_awareness",
                    awareness=self.AWARENESS,
                    obj_type=self.OBJ_TYPE,
                )
            ret = ret.astimezone(self.timezone).replace(tzinfo=None)
        return ret


class AwareDateTime(DateTime):
    """A formatted aware datetime string.

    :param format: See :class:`DateTime`.
    :param default_timezone: Used on deserialization. If `None`, naive
        datetimes are rejected. If not `None`, naive datetimes are set this
        timezone.
    :param kwargs: The same keyword arguments that :class:`Field` receives.

    .. versionadded:: 3.0.0rc9
    """

    AWARENESS = "aware"

    def __init__(
        self,
        format: str | None = None,  # noqa: A002
        *,
        default_timezone: dt.tzinfo | None = None,
        **kwargs: Unpack[_BaseFieldKwargs],
    ) -> None:
        super().__init__(format=format, **kwargs)
        self.default_timezone = default_timezone

    def _deserialize(self, value, attr, data, **kwargs) -> dt.datetime:
        ret = super()._deserialize(value, attr, data, **kwargs)
        if not utils.is_aware(ret):
            if self.default_timezone is None:
                raise self.make_error(
                    "invalid_awareness",
                    awareness=self.AWARENESS,
                    obj_type=self.OBJ_TYPE,
                )
            ret = ret.replace(tzinfo=self.default_timezone)
        return ret


class Time(_TemporalField[dt.time]):
    """A formatted time string.

    Example: ``'03:12:58.019077'``

    :param format: Either ``"iso"`` (for ISO8601) or a date format string.
        If `None`, defaults to "iso".
    :param kwargs: The same keyword arguments that :class:`Field` receives.
    """

    SERIALIZATION_FUNCS = {
        "iso": dt.time.isoformat,
        "iso8601": dt.time.isoformat,
    }

    DESERIALIZATION_FUNCS = {
        "iso": dt.time.fromisoformat,
        "iso8601": dt.time.fromisoformat,
    }

    DEFAULT_FORMAT = "iso"

    OBJ_TYPE = "time"

    SCHEMA_OPTS_VAR_NAME = "timeformat"

    @staticmethod
    def _make_object_from_format(value, data_format):
        return dt.datetime.strptime(value, data_format).time()


class Date(_TemporalField[dt.date]):
    """ISO8601-formatted date string.

    :param format: Either ``"iso"`` (for ISO8601) or a date format string.
        If `None`, defaults to "iso".
    :param kwargs: The same keyword arguments that :class:`Field` receives.
    """

    #: Default error messages.
    default_error_messages = {
        "invalid": "Not a valid date.",
        "format": '"{input}" cannot be formatted as a date.',
    }

    SERIALIZATION_FUNCS = {
        "iso": dt.date.isoformat,
        "iso8601": dt.date.isoformat,
    }

    DESERIALIZATION_FUNCS = {
        "iso": dt.date.fromisoformat,
        "iso8601": dt.date.fromisoformat,
    }

    DEFAULT_FORMAT = "iso"

    OBJ_TYPE = "date"

    SCHEMA_OPTS_VAR_NAME = "dateformat"

    @staticmethod
    def _make_object_from_format(value, data_format):
        return dt.datetime.strptime(value, data_format).date()


class TimeDelta(Field[dt.timedelta]):
    """A field that (de)serializes a :class:`datetime.timedelta` object to a `float`.
    The `float` can represent any time unit that the :class:`datetime.timedelta` constructor
    supports.

    :param precision: The time unit used for (de)serialization. Must be one of 'weeks',
        'days', 'hours', 'minutes', 'seconds', 'milliseconds' or 'microseconds'.
    :param kwargs: The same keyword arguments that :class:`Field` receives.

    Float Caveats
    -------------
    Precision loss may occur when serializing a highly precise :class:`datetime.timedelta`
    object using a big ``precision`` unit due to floating point arithmetics.

    When necessary, the :class:`datetime.timedelta` constructor rounds `float` inputs
    to whole microseconds during initialization of the object. As a result, deserializing
    a `float` might be subject to rounding, regardless of `precision`. For example,
    ``TimeDelta().deserialize("1.1234567") == timedelta(seconds=1, microseconds=123457)``.

    .. versionchanged:: 3.17.0
        Allow serialization to `float` through use of a new `serialization_type` parameter.
        Defaults to `int` for backwards compatibility. Also affects deserialization.
    .. versionchanged:: 4.0.0
        Remove `serialization_type` parameter and always serialize to float.
        Value is cast to a `float` upon deserialization.
    """

    WEEKS = "weeks"
    DAYS = "days"
    HOURS = "hours"
    MINUTES = "minutes"
    SECONDS = "seconds"
    MILLISECONDS = "milliseconds"
    MICROSECONDS = "microseconds"

    # cache this mapping on class level for performance
    _unit_to_microseconds_mapping = {
        WEEKS: 1000000 * 60 * 60 * 24 * 7,
        DAYS: 1000000 * 60 * 60 * 24,
        HOURS: 1000000 * 60 * 60,
        MINUTES: 1000000 * 60,
        SECONDS: 1000000,
        MILLISECONDS: 1000,
        MICROSECONDS: 1,
    }

    #: Default error messages.
    default_error_messages = {
        "invalid": "Not a valid period of time.",
        "format": "{input!r} cannot be formatted as a timedelta.",
    }

    def __init__(
        self,
        precision: str = SECONDS,
        **kwargs: Unpack[_BaseFieldKwargs],
    ) -> None:
        precision = precision.lower()

        if precision not in self._unit_to_microseconds_mapping:
            units = ", ".join(self._unit_to_microseconds_mapping)
            msg = f"The precision must be one of: {units}."
            raise ValueError(msg)

        self.precision = precision
        super().__init__(**kwargs)

    def _serialize(self, value, attr, obj, **kwargs) -> float | None:
        if value is None:
            return None

        # limit float arithmetics to a single division to minimize precision loss
        microseconds: int = utils.timedelta_to_microseconds(value)
        microseconds_per_unit: int = self._unit_to_microseconds_mapping[self.precision]
        return microseconds / microseconds_per_unit

    def _deserialize(self, value, attr, data, **kwargs) -> dt.timedelta:
        if isinstance(value, dt.timedelta):
            return value
        try:
            value = float(value)
        except (TypeError, ValueError) as error:
            raise self.make_error("invalid") from error

        kwargs = {self.precision: value}

        try:
            return dt.timedelta(**kwargs)
        except OverflowError as error:
            raise self.make_error("invalid") from error


_MappingT = typing.TypeVar("_MappingT", bound=_Mapping)


class Mapping(Field[_MappingT]):
    """An abstract class for objects with key-value pairs. This class should not be used within schemas.

    :param keys: A field class or instance for dict keys.
    :param values: A field class or instance for dict values.
    :param kwargs: The same keyword arguments that :class:`Field` receives.

    .. note::
        When the structure of nested data is not known, you may omit the
        `keys` and `values` arguments to prevent content validation.

    .. versionadded:: 3.0.0rc4
    .. versionchanged:: 3.24.0
        `Mapping <marshmallow.fields.Mapping>` should no longer be used as a field within a `Schema <marshmallow.Schema>`.
        Use `Dict <marshmallow.fields.Dict>` instead.
    """

    mapping_type: type[_MappingT]

    #: Default error messages.
    default_error_messages = {"invalid": "Not a valid mapping type."}

    def __init__(
        self,
        keys: Field | type[Field] | None = None,
        values: Field | type[Field] | None = None,
        **kwargs: Unpack[_BaseFieldKwargs],
    ):
        super().__init__(**kwargs)
        if keys is None:
            self.key_field = None
        else:
            try:
                self.key_field = _resolve_field_instance(keys)
            except _FieldInstanceResolutionError as error:
                raise ValueError(
                    '"keys" must be a subclass or instance of marshmallow.fields.Field.'
                ) from error

        if values is None:
            self.value_field = None
        else:
            try:
                self.value_field = _resolve_field_instance(values)
            except _FieldInstanceResolutionError as error:
                raise ValueError(
                    '"values" must be a subclass or instance of '
                    "marshmallow.fields.Field."
                ) from error
            if isinstance(self.value_field, Nested):
                self.only = self.value_field.only
                self.exclude = self.value_field.exclude

    def _bind_to_schema(self, field_name, parent):
        super()._bind_to_schema(field_name, parent)
        if self.value_field:
            self.value_field = copy.deepcopy(self.value_field)
            self.value_field._bind_to_schema(field_name, self)
        if isinstance(self.value_field, Nested):
            self.value_field.only = self.only
            self.value_field.exclude = self.exclude
        if self.key_field:
            self.key_field = copy.deepcopy(self.key_field)
            self.key_field._bind_to_schema(field_name, self)

    def _serialize(self, value, attr, obj, **kwargs):
        if value is None:
            return None
        if not self.value_field and not self.key_field:
            return self.mapping_type(value)

        # Serialize keys
        if self.key_field is None:
            keys = {k: k for k in value}
        else:
            keys = {
                k: self.key_field._serialize(k, None, None, **kwargs) for k in value
            }

        # Serialize values
        result = self.mapping_type()
        if self.value_field is None:
            for k, v in value.items():
                if k in keys:
                    result[keys[k]] = v
        else:
            for k, v in value.items():
                result[keys[k]] = self.value_field._serialize(v, None, None, **kwargs)

        return result

    def _deserialize(self, value, attr, data, **kwargs):
        if not isinstance(value, _Mapping):
            raise self.make_error("invalid")
        if not self.value_field and not self.key_field:
            return self.mapping_type(value)

        errors = collections.defaultdict(dict)

        # Deserialize keys
        if self.key_field is None:
            keys = {k: k for k in value}
        else:
            keys = {}
            for key in value:
                try:
                    keys[key] = self.key_field.deserialize(key, **kwargs)
                except ValidationError as error:
                    errors[key]["key"] = error.messages

        # Deserialize values
        result = self.mapping_type()
        if self.value_field is None:
            for k, v in value.items():
                if k in keys:
                    result[keys[k]] = v
        else:
            for key, val in value.items():
                try:
                    deser_val = self.value_field.deserialize(val, **kwargs)
                except ValidationError as error:
                    errors[key]["value"] = error.messages
                    if error.valid_data is not None and key in keys:
                        result[keys[key]] = error.valid_data
                else:
                    if key in keys:
                        result[keys[key]] = deser_val

        if errors:
            raise ValidationError(errors, valid_data=result)

        return result


class Dict(Mapping[dict]):
    """A dict field. Supports dicts and dict-like objects

    Example: ::

        numbers = fields.Dict(keys=fields.Str(), values=fields.Float())

    :param kwargs: The same keyword arguments that :class:`Mapping` receives.

    .. versionadded:: 2.1.0
    """

    mapping_type = dict


class Url(String):
    """An URL field.

    :param default: Default value for the field if the attribute is not set.
    :param relative: Whether to allow relative URLs.
    :param absolute: Whether to allow absolute URLs.
    :param require_tld: Whether to reject non-FQDN hostnames.
    :param schemes: Valid schemes. By default, ``http``, ``https``,
        ``ftp``, and ``ftps`` are allowed.
    :param kwargs: The same keyword arguments that :class:`String` receives.
    """

    #: Default error messages.
    default_error_messages = {"invalid": "Not a valid URL."}

    def __init__(
        self,
        *,
        relative: bool = False,
        absolute: bool = True,
        schemes: types.StrSequenceOrSet | None = None,
        require_tld: bool = True,
        **kwargs: Unpack[_BaseFieldKwargs],
    ):
        super().__init__(**kwargs)

        self.relative = relative
        self.absolute = absolute
        self.require_tld = require_tld
        # Insert validation into self.validators so that multiple errors can be stored.
        validator = validate.URL(
            relative=self.relative,
            absolute=self.absolute,
            schemes=schemes,
            require_tld=self.require_tld,
            error=self.error_messages["invalid"],
        )
        self.validators.insert(0, validator)


class Email(String):
    """An email field.

    :param args: The same positional arguments that :class:`String` receives.
    :param kwargs: The same keyword arguments that :class:`String` receives.
    """

    #: Default error messages.
    default_error_messages = {"invalid": "Not a valid email address."}

    def __init__(self, **kwargs: Unpack[_BaseFieldKwargs]) -> None:
        super().__init__(**kwargs)
        # Insert validation into self.validators so that multiple errors can be stored.
        validator = validate.Email(error=self.error_messages["invalid"])
        self.validators.insert(0, validator)


class IP(Field[typing.Union[ipaddress.IPv4Address, ipaddress.IPv6Address]]):
    """A IP address field.

    :param exploded: If `True`, serialize ipv6 address in long form, ie. with groups
        consisting entirely of zeros included.

    .. versionadded:: 3.8.0
    """

    default_error_messages = {"invalid_ip": "Not a valid IP address."}

    DESERIALIZATION_CLASS: type | None = None

    def __init__(self, *, exploded: bool = False, **kwargs: Unpack[_BaseFieldKwargs]):
        super().__init__(**kwargs)
        self.exploded = exploded

    def _serialize(self, value, attr, obj, **kwargs) -> str | None:
        if value is None:
            return None
        if self.exploded:
            return value.exploded
        return value.compressed

    def _deserialize(
        self, value, attr, data, **kwargs
    ) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
        try:
            return (self.DESERIALIZATION_CLASS or ipaddress.ip_address)(
                utils.ensure_text_type(value)
            )
        except (ValueError, TypeError) as error:
            raise self.make_error("invalid_ip") from error


class IPv4(IP):
    """A IPv4 address field.

    .. versionadded:: 3.8.0
    """

    default_error_messages = {"invalid_ip": "Not a valid IPv4 address."}

    DESERIALIZATION_CLASS = ipaddress.IPv4Address


class IPv6(IP):
    """A IPv6 address field.

    .. versionadded:: 3.8.0
    """

    default_error_messages = {"invalid_ip": "Not a valid IPv6 address."}

    DESERIALIZATION_CLASS = ipaddress.IPv6Address


class IPInterface(
    Field[typing.Union[ipaddress.IPv4Interface, ipaddress.IPv6Interface]]
):
    """A IPInterface field.

    IP interface is the non-strict form of the IPNetwork type where arbitrary host
    addresses are always accepted.

    IPAddress and mask e.g. '192.168.0.2/24' or '192.168.0.2/255.255.255.0'

    see https://python.readthedocs.io/en/latest/library/ipaddress.html#interface-objects

    :param exploded: If `True`, serialize ipv6 interface in long form, ie. with groups
        consisting entirely of zeros included.
    """

    default_error_messages = {"invalid_ip_interface": "Not a valid IP interface."}

    DESERIALIZATION_CLASS: type | None = None

    def __init__(self, *, exploded: bool = False, **kwargs: Unpack[_BaseFieldKwargs]):
        super().__init__(**kwargs)
        self.exploded = exploded

    def _serialize(self, value, attr, obj, **kwargs) -> str | None:
        if value is None:
            return None
        if self.exploded:
            return value.exploded
        return value.compressed

    def _deserialize(
        self, value, attr, data, **kwargs
    ) -> ipaddress.IPv4Interface | ipaddress.IPv6Interface:
        try:
            return (self.DESERIALIZATION_CLASS or ipaddress.ip_interface)(
                utils.ensure_text_type(value)
            )
        except (ValueError, TypeError) as error:
            raise self.make_error("invalid_ip_interface") from error


class IPv4Interface(IPInterface):
    """A IPv4 Network Interface field."""

    default_error_messages = {"invalid_ip_interface": "Not a valid IPv4 interface."}

    DESERIALIZATION_CLASS = ipaddress.IPv4Interface


class IPv6Interface(IPInterface):
    """A IPv6 Network Interface field."""

    default_error_messages = {"invalid_ip_interface": "Not a valid IPv6 interface."}

    DESERIALIZATION_CLASS = ipaddress.IPv6Interface


_EnumT = typing.TypeVar("_EnumT", bound=EnumType)


class Enum(Field[_EnumT]):
    """An Enum field (de)serializing enum members by symbol (name) or by value.

    :param enum: Enum class
    :param by_value: Whether to (de)serialize by value or by name,
        or Field class or instance to use to (de)serialize by value. Defaults to False.

    If `by_value` is `False` (default), enum members are (de)serialized by symbol (name).
    If it is `True`, they are (de)serialized by value using `marshmallow.fields.Raw`.
    If it is a field instance or class, they are (de)serialized by value using this field.

    .. versionadded:: 3.18.0
    """

    default_error_messages = {
        "unknown": "Must be one of: {choices}.",
    }

    def __init__(
        self,
        enum: type[_EnumT],
        *,
        by_value: bool | Field | type[Field] = False,
        **kwargs: Unpack[_BaseFieldKwargs],
    ):
        super().__init__(**kwargs)
        self.enum = enum
        self.by_value = by_value

        # Serialization by name
        if by_value is False:
            self.field: Field = String()
            self.choices_text = ", ".join(
                str(self.field._serialize(m, None, None)) for m in enum.__members__
            )
        # Serialization by value
        else:
            if by_value is True:
                self.field = Raw()
            else:
                try:
                    self.field = _resolve_field_instance(by_value)
                except _FieldInstanceResolutionError as error:
                    raise ValueError(
                        '"by_value" must be either a bool or a subclass or instance of '
                        "marshmallow.fields.Field."
                    ) from error
            self.choices_text = ", ".join(
                str(self.field._serialize(m.value, None, None)) for m in enum
            )

    def _serialize(
        self, value: _EnumT | None, attr: str | None, obj: typing.Any, **kwargs
    ) -> typing.Any | None:
        if value is None:
            return None
        if self.by_value:
            val = value.value
        else:
            val = value.name
        return self.field._serialize(val, attr, obj, **kwargs)

    def _deserialize(self, value, attr, data, **kwargs) -> _EnumT:
        if isinstance(value, self.enum):
            return value
        val = self.field._deserialize(value, attr, data, **kwargs)
        if self.by_value:
            try:
                return self.enum(val)
            except ValueError as error:
                raise self.make_error("unknown", choices=self.choices_text) from error
        try:
            return getattr(self.enum, val)
        except AttributeError as error:
            raise self.make_error("unknown", choices=self.choices_text) from error


class Method(Field):
    """A field that takes the value returned by a `Schema <marshmallow.Schema>` method.

    :param serialize: The name of the Schema method from which
        to retrieve the value. The method must take an argument ``obj``
        (in addition to self) that is the object to be serialized.
    :param deserialize: Optional name of the Schema method for deserializing
        a value The method must take a single argument ``value``, which is the
        value to deserialize.

    .. versionchanged:: 3.0.0
        Removed ``method_name`` parameter.
    """

    _CHECK_ATTRIBUTE = False

    def __init__(
        self,
        serialize: str | None = None,
        deserialize: str | None = None,
        **kwargs: Unpack[_BaseFieldKwargs],  # FIXME: Omit dump_only and load_only
    ):
        # Set dump_only and load_only based on arguments
        kwargs["dump_only"] = bool(serialize) and not bool(deserialize)
        kwargs["load_only"] = bool(deserialize) and not bool(serialize)
        super().__init__(**kwargs)
        self.serialize_method_name = serialize
        self.deserialize_method_name = deserialize
        self._serialize_method = None
        self._deserialize_method = None

    def _bind_to_schema(self, field_name, parent):
        if self.serialize_method_name:
            self._serialize_method = utils.callable_or_raise(
                getattr(parent, self.serialize_method_name)
            )

        if self.deserialize_method_name:
            self._deserialize_method = utils.callable_or_raise(
                getattr(parent, self.deserialize_method_name)
            )

        super()._bind_to_schema(field_name, parent)

    def _serialize(self, value, attr, obj, **kwargs):
        if self._serialize_method is not None:
            return self._serialize_method(obj)
        return missing_

    def _deserialize(self, value, attr, data, **kwargs):
        if self._deserialize_method is not None:
            return self._deserialize_method(value)
        return value


class Function(Field):
    """A field that takes the value returned by a function.

    :param serialize: A callable from which to retrieve the value.
        The function must take a single argument ``obj`` which is the object
        to be serialized.
        If no callable is provided then the ```load_only``` flag will be set
        to True.
    :param deserialize: A callable from which to retrieve the value.
        The function must take a single argument ``value`` which is the value
        to be deserialized.
        If no callable is provided then ```value``` will be passed through
        unchanged.

    .. versionchanged:: 3.0.0a1
        Removed ``func`` parameter.

    .. versionchanged:: 4.0.0
        Don't pass context to serialization and deserialization functions.
    """

    _CHECK_ATTRIBUTE = False

    def __init__(
        self,
        serialize: (
            typing.Callable[[typing.Any], typing.Any]
            | typing.Callable[[typing.Any, dict], typing.Any]
            | None
        ) = None,
        deserialize: (
            typing.Callable[[typing.Any], typing.Any]
            | typing.Callable[[typing.Any, dict], typing.Any]
            | None
        ) = None,
        **kwargs: Unpack[_BaseFieldKwargs],  # FIXME: Omit dump_only and load_only
    ):
        # Set dump_only and load_only based on arguments
        kwargs["dump_only"] = bool(serialize) and not bool(deserialize)
        kwargs["load_only"] = bool(deserialize) and not bool(serialize)
        super().__init__(**kwargs)
        self.serialize_func = serialize and utils.callable_or_raise(serialize)
        self.deserialize_func = deserialize and utils.callable_or_raise(deserialize)

    def _serialize(self, value, attr, obj, **kwargs):
        return self.serialize_func(obj)

    def _deserialize(self, value, attr, data, **kwargs):
        if self.deserialize_func:
            return self.deserialize_func(value)
        return value


_ContantT = typing.TypeVar("_ContantT")


class Constant(Field[_ContantT]):
    """A field that (de)serializes to a preset constant.  If you only want the
    constant added for serialization or deserialization, you should use
    ``dump_only=True`` or ``load_only=True`` respectively.

    :param constant: The constant to return for the field attribute.
    """

    _CHECK_ATTRIBUTE = False

    def __init__(self, constant: _ContantT, **kwargs: Unpack[_BaseFieldKwargs]):
        super().__init__(**kwargs)
        self.constant = constant
        self.load_default = constant
        self.dump_default = constant

    def _serialize(self, value, *args, **kwargs) -> _ContantT:
        return self.constant

    def _deserialize(self, value, *args, **kwargs) -> _ContantT:
        return self.constant


# Aliases
URL = Url

Str = String
Bool = Boolean
Int = Integer


# === src/marshmallow/constants.py ===
import typing

EXCLUDE: typing.Final = "exclude"
INCLUDE: typing.Final = "include"
RAISE: typing.Final = "raise"


class _Missing:
    def __bool__(self):
        return False

    def __copy__(self):
        return self

    def __deepcopy__(self, _):
        return self

    def __repr__(self):
        return "<marshmallow.missing>"


missing: typing.Final = _Missing()


# === src/marshmallow/__init__.py ===
from marshmallow.constants import EXCLUDE, INCLUDE, RAISE, missing
from marshmallow.decorators import (
    post_dump,
    post_load,
    pre_dump,
    pre_load,
    validates,
    validates_schema,
)
from marshmallow.exceptions import ValidationError
from marshmallow.schema import Schema, SchemaOpts

from . import fields

__all__ = [
    "EXCLUDE",
    "INCLUDE",
    "RAISE",
    "Schema",
    "SchemaOpts",
    "ValidationError",
    "fields",
    "missing",
    "post_dump",
    "post_load",
    "pre_dump",
    "pre_load",
    "validates",
    "validates_schema",
]


# === src/marshmallow/types.py ===
"""Type aliases.

.. warning::

    This module is provisional. Types may be modified, added, and removed between minor releases.
"""

from __future__ import annotations

try:
    from typing import TypeAlias
except ImportError:  # Remove when dropping Python 3.9
    from typing_extensions import TypeAlias

import typing

#: A type that can be either a sequence of strings or a set of strings
StrSequenceOrSet: TypeAlias = typing.Union[
    typing.Sequence[str], typing.AbstractSet[str]
]

#: Type for validator functions
Validator: TypeAlias = typing.Callable[[typing.Any], typing.Any]

#: A valid option for the ``unknown`` schema option and argument
UnknownOption: TypeAlias = typing.Literal["exclude", "include", "raise"]


class SchemaValidator(typing.Protocol):
    def __call__(
        self,
        output: typing.Any,
        original_data: typing.Any = ...,
        *,
        partial: bool | StrSequenceOrSet | None = None,
        unknown: UnknownOption | None = None,
        many: bool = False,
    ) -> None: ...


class RenderModule(typing.Protocol):
    def dumps(
        self, obj: typing.Any, *args: typing.Any, **kwargs: typing.Any
    ) -> str: ...

    def loads(
        self, s: str | bytes | bytearray, *args: typing.Any, **kwargs: typing.Any
    ) -> typing.Any: ...


# === src/marshmallow/orderedset.py ===
# OrderedSet
# Copyright (c) 2009 Raymond Hettinger
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
#     The above copyright notice and this permission notice shall be
#     included in all copies or substantial portions of the Software.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
#     EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
#     OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#     NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
#     HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
#     WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#     FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
#     OTHER DEALINGS IN THE SOFTWARE.
from collections.abc import MutableSet


class OrderedSet(MutableSet):
    def __init__(self, iterable=None):
        self.end = end = []
        end += [None, end, end]  # sentinel node for doubly linked list
        self.map = {}  # key --> [key, prev, next]
        if iterable is not None:
            self |= iterable

    def __len__(self):
        return len(self.map)

    def __contains__(self, key):
        return key in self.map

    def add(self, key):
        if key not in self.map:
            end = self.end
            curr = end[1]
            curr[2] = end[1] = self.map[key] = [key, curr, end]

    def discard(self, key):
        if key in self.map:
            key, prev, next = self.map.pop(key)  # noqa: A001
            prev[2] = next
            next[1] = prev

    def __iter__(self):
        end = self.end
        curr = end[2]
        while curr is not end:
            yield curr[0]
            curr = curr[2]

    def __reversed__(self):
        end = self.end
        curr = end[1]
        while curr is not end:
            yield curr[0]
            curr = curr[1]

    def pop(self, last=True):
        if not self:
            raise KeyError("set is empty")
        key = self.end[1][0] if last else self.end[2][0]
        self.discard(key)
        return key

    def __repr__(self):
        if not self:
            return f"{self.__class__.__name__}()"
        return f"{self.__class__.__name__}({list(self)!r})"

    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        return set(self) == set(other)


if __name__ == "__main__":
    s = OrderedSet("abracadaba")
    t = OrderedSet("simsalabim")
    print(s | t)
    print(s & t)
    print(s - t)


# === src/marshmallow/validate.py ===
"""Validation classes for various types of data."""

from __future__ import annotations

import re
import typing
from abc import ABC, abstractmethod
from itertools import zip_longest
from operator import attrgetter

from marshmallow.exceptions import ValidationError

if typing.TYPE_CHECKING:
    from marshmallow import types

_T = typing.TypeVar("_T")


class Validator(ABC):
    """Abstract base class for validators.

    .. note::
        This class does not provide any validation behavior. It is only used to
        add a useful `__repr__` implementation for validators.
    """

    error: str | None = None

    def __repr__(self) -> str:
        args = self._repr_args()
        args = f"{args}, " if args else ""

        return f"<{self.__class__.__name__}({args}error={self.error!r})>"

    def _repr_args(self) -> str:
        """A string representation of the args passed to this validator. Used by
        `__repr__`.
        """
        return ""

    @abstractmethod
    def __call__(self, value: typing.Any) -> typing.Any: ...


class And(Validator):
    """Compose multiple validators and combine their error messages.

    Example: ::

        from marshmallow import validate, ValidationError


        def is_even(value):
            if value % 2 != 0:
                raise ValidationError("Not an even value.")


        validator = validate.And(validate.Range(min=0), is_even)
        validator(-1)
        # ValidationError: ['Must be greater than or equal to 0.', 'Not an even value.']

    :param validators: Validators to combine.
    """

    def __init__(self, *validators: types.Validator):
        self.validators = tuple(validators)

    def _repr_args(self) -> str:
        return f"validators={self.validators!r}"

    def __call__(self, value: typing.Any) -> typing.Any:
        errors: list[str | dict] = []
        kwargs: dict[str, typing.Any] = {}
        for validator in self.validators:
            try:
                validator(value)
            except ValidationError as err:
                kwargs.update(err.kwargs)
                if isinstance(err.messages, dict):
                    errors.append(err.messages)
                else:
                    errors.extend(err.messages)
        if errors:
            raise ValidationError(errors, **kwargs)
        return value


class URL(Validator):
    """Validate a URL.

    :param relative: Whether to allow relative URLs.
    :param absolute: Whether to allow absolute URLs.
    :param error: Error message to raise in case of a validation error.
        Can be interpolated with `{input}`.
    :param schemes: Valid schemes. By default, ``http``, ``https``,
        ``ftp``, and ``ftps`` are allowed.
    :param require_tld: Whether to reject non-FQDN hostnames.
    """

    class RegexMemoizer:
        def __init__(self):
            self._memoized = {}

        def _regex_generator(
            self, *, relative: bool, absolute: bool, require_tld: bool
        ) -> typing.Pattern:
            hostname_variants = [
                # a normal domain name, expressed in [A-Z0-9] chars with hyphens allowed only in the middle
                # note that the regex will be compiled with IGNORECASE, so these are upper and lowercase chars
                (
                    r"(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+"
                    r"(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)"
                ),
                # or the special string 'localhost'
                r"localhost",
                # or IPv4
                r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",
                # or IPv6
                r"\[[A-F0-9]*:[A-F0-9:]+\]",
            ]
            if not require_tld:
                # allow dotless hostnames
                hostname_variants.append(r"(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.?)")

            absolute_part = "".join(
                (
                    # scheme (e.g. 'https://', 'ftp://', etc)
                    # this is validated separately against allowed schemes, so in the regex
                    # we simply want to capture its existence
                    r"(?:[a-z0-9\.\-\+]*)://",
                    # userinfo, for URLs encoding authentication
                    # e.g. 'ftp://foo:bar@ftp.example.org/'
                    r"(?:(?:[a-z0-9\-._~!$&'()*+,;=:]|%[0-9a-f]{2})*@)?",
                    # netloc, the hostname/domain part of the URL plus the optional port
                    r"(?:",
                    "|".join(hostname_variants),
                    r")",
                    r"(?::\d+)?",
                )
            )
            relative_part = r"(?:/?|[/?]\S+)\Z"

            if relative:
                if absolute:
                    parts: tuple[str, ...] = (
                        r"^(",
                        absolute_part,
                        r")?",
                        relative_part,
                    )
                else:
                    parts = (r"^", relative_part)
            else:
                parts = (r"^", absolute_part, relative_part)

            return re.compile("".join(parts), re.IGNORECASE)

        def __call__(
            self, *, relative: bool, absolute: bool, require_tld: bool
        ) -> typing.Pattern:
            key = (relative, absolute, require_tld)
            if key not in self._memoized:
                self._memoized[key] = self._regex_generator(
                    relative=relative, absolute=absolute, require_tld=require_tld
                )

            return self._memoized[key]

    _regex = RegexMemoizer()

    default_message = "Not a valid URL."
    default_schemes = {"http", "https", "ftp", "ftps"}

    def __init__(
        self,
        *,
        relative: bool = False,
        absolute: bool = True,
        schemes: types.StrSequenceOrSet | None = None,
        require_tld: bool = True,
        error: str | None = None,
    ):
        if not relative and not absolute:
            raise ValueError(
                "URL validation cannot set both relative and absolute to False."
            )
        self.relative = relative
        self.absolute = absolute
        self.error: str = error or self.default_message
        self.schemes = schemes or self.default_schemes
        self.require_tld = require_tld

    def _repr_args(self) -> str:
        return f"relative={self.relative!r}, absolute={self.absolute!r}"

    def _format_error(self, value) -> str:
        return self.error.format(input=value)

    def __call__(self, value: str) -> str:
        message = self._format_error(value)
        if not value:
            raise ValidationError(message)

        # Check first if the scheme is valid
        scheme = None
        if "://" in value:
            scheme = value.split("://")[0].lower()
            if scheme not in self.schemes:
                raise ValidationError(message)

        regex = self._regex(
            relative=self.relative, absolute=self.absolute, require_tld=self.require_tld
        )

        # Hostname is optional for file URLS. If absent it means `localhost`.
        # Fill it in for the validation if needed
        if scheme == "file" and value.startswith("file:///"):
            matched = regex.search(value.replace("file:///", "file://localhost/", 1))
        else:
            matched = regex.search(value)

        if not matched:
            raise ValidationError(message)

        return value


class Email(Validator):
    """Validate an email address.

    :param error: Error message to raise in case of a validation error. Can be
        interpolated with `{input}`.
    """

    USER_REGEX = re.compile(
        r"(^[-!#$%&'*+/=?^`{}|~\w]+(\.[-!#$%&'*+/=?^`{}|~\w]+)*\Z"  # dot-atom
        # quoted-string
        r'|^"([\001-\010\013\014\016-\037!#-\[\]-\177]'
        r'|\\[\001-\011\013\014\016-\177])*"\Z)',
        re.IGNORECASE | re.UNICODE,
    )

    DOMAIN_REGEX = re.compile(
        # domain
        r"(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+"
        r"(?:[A-Z]{2,6}|[A-Z0-9-]{2,})\Z"
        # literal form, ipv4 address (SMTP 4.1.3)
        r"|^\[(25[0-5]|2[0-4]\d|[0-1]?\d?\d)"
        r"(\.(25[0-5]|2[0-4]\d|[0-1]?\d?\d)){3}\]\Z",
        re.IGNORECASE | re.UNICODE,
    )

    DOMAIN_WHITELIST = ("localhost",)

    default_message = "Not a valid email address."

    def __init__(self, *, error: str | None = None):
        self.error: str = error or self.default_message

    def _format_error(self, value: str) -> str:
        return self.error.format(input=value)

    def __call__(self, value: str) -> str:
        message = self._format_error(value)

        if not value or "@" not in value:
            raise ValidationError(message)

        user_part, domain_part = value.rsplit("@", 1)

        if not self.USER_REGEX.match(user_part):
            raise ValidationError(message)

        if domain_part not in self.DOMAIN_WHITELIST:
            if not self.DOMAIN_REGEX.match(domain_part):
                try:
                    domain_part = domain_part.encode("idna").decode("ascii")
                except UnicodeError:
                    pass
                else:
                    if self.DOMAIN_REGEX.match(domain_part):
                        return value
                raise ValidationError(message)

        return value


class Range(Validator):
    """Validator which succeeds if the value passed to it is within the specified
    range. If ``min`` is not specified, or is specified as `None`,
    no lower bound exists. If ``max`` is not specified, or is specified as `None`,
    no upper bound exists. The inclusivity of the bounds (if they exist) is configurable.
    If ``min_inclusive`` is not specified, or is specified as `True`, then
    the ``min`` bound is included in the range. If ``max_inclusive`` is not specified,
    or is specified as `True`, then the ``max`` bound is included in the range.

    :param min: The minimum value (lower bound). If not provided, minimum
        value will not be checked.
    :param max: The maximum value (upper bound). If not provided, maximum
        value will not be checked.
    :param min_inclusive: Whether the `min` bound is included in the range.
    :param max_inclusive: Whether the `max` bound is included in the range.
    :param error: Error message to raise in case of a validation error.
        Can be interpolated with `{input}`, `{min}` and `{max}`.
    """

    message_min = "Must be {min_op} {{min}}."
    message_max = "Must be {max_op} {{max}}."
    message_all = "Must be {min_op} {{min}} and {max_op} {{max}}."

    message_gte = "greater than or equal to"
    message_gt = "greater than"
    message_lte = "less than or equal to"
    message_lt = "less than"

    def __init__(
        self,
        min=None,  # noqa: A002
        max=None,  # noqa: A002
        *,
        min_inclusive: bool = True,
        max_inclusive: bool = True,
        error: str | None = None,
    ):
        self.min = min
        self.max = max
        self.error = error
        self.min_inclusive = min_inclusive
        self.max_inclusive = max_inclusive

        # interpolate messages based on bound inclusivity
        self.message_min = self.message_min.format(
            min_op=self.message_gte if self.min_inclusive else self.message_gt
        )
        self.message_max = self.message_max.format(
            max_op=self.message_lte if self.max_inclusive else self.message_lt
        )
        self.message_all = self.message_all.format(
            min_op=self.message_gte if self.min_inclusive else self.message_gt,
            max_op=self.message_lte if self.max_inclusive else self.message_lt,
        )

    def _repr_args(self) -> str:
        return f"min={self.min!r}, max={self.max!r}, min_inclusive={self.min_inclusive!r}, max_inclusive={self.max_inclusive!r}"

    def _format_error(self, value: _T, message: str) -> str:
        return (self.error or message).format(input=value, min=self.min, max=self.max)

    def __call__(self, value: _T) -> _T:
        if self.min is not None and (
            value < self.min if self.min_inclusive else value <= self.min
        ):
            message = self.message_min if self.max is None else self.message_all
            raise ValidationError(self._format_error(value, message))

        if self.max is not None and (
            value > self.max if self.max_inclusive else value >= self.max
        ):
            message = self.message_max if self.min is None else self.message_all
            raise ValidationError(self._format_error(value, message))

        return value


_SizedT = typing.TypeVar("_SizedT", bound=typing.Sized)


class Length(Validator):
    """Validator which succeeds if the value passed to it has a
    length between a minimum and maximum. Uses len(), so it
    can work for strings, lists, or anything with length.

    :param min: The minimum length. If not provided, minimum length
        will not be checked.
    :param max: The maximum length. If not provided, maximum length
        will not be checked.
    :param equal: The exact length. If provided, maximum and minimum
        length will not be checked.
    :param error: Error message to raise in case of a validation error.
        Can be interpolated with `{input}`, `{min}` and `{max}`.
    """

    message_min = "Shorter than minimum length {min}."
    message_max = "Longer than maximum length {max}."
    message_all = "Length must be between {min} and {max}."
    message_equal = "Length must be {equal}."

    def __init__(
        self,
        min: int | None = None,  # noqa: A002
        max: int | None = None,  # noqa: A002
        *,
        equal: int | None = None,
        error: str | None = None,
    ):
        if equal is not None and any([min, max]):
            raise ValueError(
                "The `equal` parameter was provided, maximum or "
                "minimum parameter must not be provided."
            )

        self.min = min
        self.max = max
        self.error = error
        self.equal = equal

    def _repr_args(self) -> str:
        return f"min={self.min!r}, max={self.max!r}, equal={self.equal!r}"

    def _format_error(self, value: _SizedT, message: str) -> str:
        return (self.error or message).format(
            input=value, min=self.min, max=self.max, equal=self.equal
        )

    def __call__(self, value: _SizedT) -> _SizedT:
        length = len(value)

        if self.equal is not None:
            if length != self.equal:
                raise ValidationError(self._format_error(value, self.message_equal))
            return value

        if self.min is not None and length < self.min:
            message = self.message_min if self.max is None else self.message_all
            raise ValidationError(self._format_error(value, message))

        if self.max is not None and length > self.max:
            message = self.message_max if self.min is None else self.message_all
            raise ValidationError(self._format_error(value, message))

        return value


class Equal(Validator):
    """Validator which succeeds if the ``value`` passed to it is
    equal to ``comparable``.

    :param comparable: The object to compare to.
    :param error: Error message to raise in case of a validation error.
        Can be interpolated with `{input}` and `{other}`.
    """

    default_message = "Must be equal to {other}."

    def __init__(self, comparable, *, error: str | None = None):
        self.comparable = comparable
        self.error: str = error or self.default_message

    def _repr_args(self) -> str:
        return f"comparable={self.comparable!r}"

    def _format_error(self, value: _T) -> str:
        return self.error.format(input=value, other=self.comparable)

    def __call__(self, value: _T) -> _T:
        if value != self.comparable:
            raise ValidationError(self._format_error(value))
        return value


class Regexp(Validator):
    """Validator which succeeds if the ``value`` matches ``regex``.

    .. note::

        Uses `re.match`, which searches for a match at the beginning of a string.

    :param regex: The regular expression string to use. Can also be a compiled
        regular expression pattern.
    :param flags: The regexp flags to use, for example re.IGNORECASE. Ignored
        if ``regex`` is not a string.
    :param error: Error message to raise in case of a validation error.
        Can be interpolated with `{input}` and `{regex}`.
    """

    default_message = "String does not match expected pattern."

    def __init__(
        self,
        regex: str | bytes | typing.Pattern,
        flags: int = 0,
        *,
        error: str | None = None,
    ):
        self.regex = (
            re.compile(regex, flags) if isinstance(regex, (str, bytes)) else regex
        )
        self.error: str = error or self.default_message

    def _repr_args(self) -> str:
        return f"regex={self.regex!r}"

    def _format_error(self, value: str | bytes) -> str:
        return self.error.format(input=value, regex=self.regex.pattern)

    @typing.overload
    def __call__(self, value: str) -> str: ...

    @typing.overload
    def __call__(self, value: bytes) -> bytes: ...

    def __call__(self, value):
        if self.regex.match(value) is None:
            raise ValidationError(self._format_error(value))

        return value


class Predicate(Validator):
    """Call the specified ``method`` of the ``value`` object. The
    validator succeeds if the invoked method returns an object that
    evaluates to True in a Boolean context. Any additional keyword
    argument will be passed to the method.

    :param method: The name of the method to invoke.
    :param error: Error message to raise in case of a validation error.
        Can be interpolated with `{input}` and `{method}`.
    :param kwargs: Additional keyword arguments to pass to the method.
    """

    default_message = "Invalid input."

    def __init__(self, method: str, *, error: str | None = None, **kwargs):
        self.method = method
        self.error: str = error or self.default_message
        self.kwargs = kwargs

    def _repr_args(self) -> str:
        return f"method={self.method!r}, kwargs={self.kwargs!r}"

    def _format_error(self, value: typing.Any) -> str:
        return self.error.format(input=value, method=self.method)

    def __call__(self, value: _T) -> _T:
        method = getattr(value, self.method)

        if not method(**self.kwargs):
            raise ValidationError(self._format_error(value))

        return value


class NoneOf(Validator):
    """Validator which fails if ``value`` is a member of ``iterable``.

    :param iterable: A sequence of invalid values.
    :param error: Error message to raise in case of a validation error. Can be
        interpolated using `{input}` and `{values}`.
    """

    default_message = "Invalid input."

    def __init__(self, iterable: typing.Iterable, *, error: str | None = None):
        self.iterable = iterable
        self.values_text = ", ".join(str(each) for each in self.iterable)
        self.error: str = error or self.default_message

    def _repr_args(self) -> str:
        return f"iterable={self.iterable!r}"

    def _format_error(self, value) -> str:
        return self.error.format(input=value, values=self.values_text)

    def __call__(self, value: typing.Any) -> typing.Any:
        try:
            if value in self.iterable:
                raise ValidationError(self._format_error(value))
        except TypeError:
            pass

        return value


class OneOf(Validator):
    """Validator which succeeds if ``value`` is a member of ``choices``.

    :param choices: A sequence of valid values.
    :param labels: Optional sequence of labels to pair with the choices.
    :param error: Error message to raise in case of a validation error. Can be
        interpolated with `{input}`, `{choices}` and `{labels}`.
    """

    default_message = "Must be one of: {choices}."

    def __init__(
        self,
        choices: typing.Iterable,
        labels: typing.Iterable[str] | None = None,
        *,
        error: str | None = None,
    ):
        self.choices = choices
        self.choices_text = ", ".join(str(choice) for choice in self.choices)
        self.labels = labels if labels is not None else []
        self.labels_text = ", ".join(str(label) for label in self.labels)
        self.error: str = error or self.default_message

    def _repr_args(self) -> str:
        return f"choices={self.choices!r}, labels={self.labels!r}"

    def _format_error(self, value) -> str:
        return self.error.format(
            input=value, choices=self.choices_text, labels=self.labels_text
        )

    def __call__(self, value: typing.Any) -> typing.Any:
        try:
            if value not in self.choices:
                raise ValidationError(self._format_error(value))
        except TypeError as error:
            raise ValidationError(self._format_error(value)) from error

        return value

    def options(
        self,
        valuegetter: str | typing.Callable[[typing.Any], typing.Any] = str,
    ) -> typing.Iterable[tuple[typing.Any, str]]:
        """Return a generator over the (value, label) pairs, where value
        is a string associated with each choice. This convenience method
        is useful to populate, for instance, a form select field.

        :param valuegetter: Can be a callable or a string. In the former case, it must
            be a one-argument callable which returns the value of a
            choice. In the latter case, the string specifies the name
            of an attribute of the choice objects. Defaults to `str()`
            or `str()`.
        """
        valuegetter = valuegetter if callable(valuegetter) else attrgetter(valuegetter)
        pairs = zip_longest(self.choices, self.labels, fillvalue="")

        return ((valuegetter(choice), label) for choice, label in pairs)


class ContainsOnly(OneOf):
    """Validator which succeeds if ``value`` is a sequence and each element
    in the sequence is also in the sequence passed as ``choices``. Empty input
    is considered valid.

    :param choices: Same as :class:`OneOf`.
    :param labels: Same as :class:`OneOf`.
    :param error: Same as :class:`OneOf`.

    .. versionchanged:: 3.0.0b2
        Duplicate values are considered valid.
    .. versionchanged:: 3.0.0b2
        Empty input is considered valid. Use `validate.Length(min=1) <marshmallow.validate.Length>`
        to validate against empty inputs.
    """

    default_message = "One or more of the choices you made was not in: {choices}."

    def _format_error(self, value) -> str:
        value_text = ", ".join(str(val) for val in value)
        return super()._format_error(value_text)

    def __call__(self, value: typing.Sequence[_T]) -> typing.Sequence[_T]:
        # We can't use set.issubset because does not handle unhashable types
        for val in value:
            if val not in self.choices:
                raise ValidationError(self._format_error(value))
        return value


class ContainsNoneOf(NoneOf):
    """Validator which fails if ``value`` is a sequence and any element
    in the sequence is a member of the sequence passed as ``iterable``. Empty input
    is considered valid.

    :param iterable: Same as :class:`NoneOf`.
    :param error: Same as :class:`NoneOf`.

    .. versionadded:: 3.6.0
    """

    default_message = "One or more of the choices you made was in: {values}."

    def _format_error(self, value) -> str:
        value_text = ", ".join(str(val) for val in value)
        return super()._format_error(value_text)

    def __call__(self, value: typing.Sequence[_T]) -> typing.Sequence[_T]:
        for val in value:
            if val in self.iterable:
                raise ValidationError(self._format_error(value))
        return value


# === src/marshmallow/class_registry.py ===
"""A registry of :class:`Schema <marshmallow.Schema>` classes. This allows for string
lookup of schemas, which may be used with
class:`fields.Nested <marshmallow.fields.Nested>`.

.. warning::

    This module is treated as private API.
    Users should not need to use this module directly.
"""
# ruff: noqa: ERA001

from __future__ import annotations

import typing

from marshmallow.exceptions import RegistryError

if typing.TYPE_CHECKING:
    from marshmallow import Schema

    SchemaType = type[Schema]

# {
#   <class_name>: <list of class objects>
#   <module_path_to_class>: <list of class objects>
# }
_registry = {}  # type: dict[str, list[SchemaType]]


def register(classname: str, cls: SchemaType) -> None:
    """Add a class to the registry of serializer classes. When a class is
    registered, an entry for both its classname and its full, module-qualified
    path are added to the registry.

    Example: ::

        class MyClass:
            pass


        register("MyClass", MyClass)
        # Registry:
        # {
        #   'MyClass': [path.to.MyClass],
        #   'path.to.MyClass': [path.to.MyClass],
        # }

    """
    # Module where the class is located
    module = cls.__module__
    # Full module path to the class
    # e.g. user.schemas.UserSchema
    fullpath = f"{module}.{classname}"
    # If the class is already registered; need to check if the entries are
    # in the same module as cls to avoid having multiple instances of the same
    # class in the registry
    if classname in _registry and not any(
        each.__module__ == module for each in _registry[classname]
    ):
        _registry[classname].append(cls)
    elif classname not in _registry:
        _registry[classname] = [cls]

    # Also register the full path
    if fullpath not in _registry:
        _registry.setdefault(fullpath, []).append(cls)
    else:
        # If fullpath does exist, replace existing entry
        _registry[fullpath] = [cls]


@typing.overload
def get_class(classname: str, *, all: typing.Literal[False] = ...) -> SchemaType: ...


@typing.overload
def get_class(
    classname: str, *, all: typing.Literal[True] = ...
) -> list[SchemaType]: ...


def get_class(classname: str, *, all: bool = False) -> list[SchemaType] | SchemaType:  # noqa: A002
    """Retrieve a class from the registry.

    :raises: `marshmallow.exceptions.RegistryError` if the class cannot be found
        or if there are multiple entries for the given class name.
    """
    try:
        classes = _registry[classname]
    except KeyError as error:
        raise RegistryError(
            f"Class with name {classname!r} was not found. You may need "
            "to import the class."
        ) from error
    if len(classes) > 1:
        if all:
            return _registry[classname]
        raise RegistryError(
            f"Multiple classes with name {classname!r} "
            "were found. Please use the full, "
            "module-qualified path."
        )
    return _registry[classname][0]


# === src/marshmallow/utils.py ===
"""Utility methods for marshmallow."""

# ruff: noqa: T201, T203
from __future__ import annotations

import datetime as dt
import inspect
import typing
from collections.abc import Mapping, Sequence

# Remove when we drop Python 3.9
try:
    from typing import TypeGuard
except ImportError:
    from typing_extensions import TypeGuard

from marshmallow.constants import missing


def is_generator(obj) -> TypeGuard[typing.Generator]:
    """Return True if ``obj`` is a generator"""
    return inspect.isgeneratorfunction(obj) or inspect.isgenerator(obj)


def is_iterable_but_not_string(obj) -> TypeGuard[typing.Iterable]:
    """Return True if ``obj`` is an iterable object that isn't a string."""
    return (hasattr(obj, "__iter__") and not hasattr(obj, "strip")) or is_generator(obj)


def is_sequence_but_not_string(obj) -> TypeGuard[Sequence]:
    """Return True if ``obj`` is a sequence that isn't a string."""
    return isinstance(obj, Sequence) and not isinstance(obj, (str, bytes))


def is_collection(obj) -> TypeGuard[typing.Iterable]:
    """Return True if ``obj`` is a collection type, e.g list, tuple, queryset."""
    return is_iterable_but_not_string(obj) and not isinstance(obj, Mapping)


# https://stackoverflow.com/a/27596917
def is_aware(datetime: dt.datetime) -> bool:
    return (
        datetime.tzinfo is not None and datetime.tzinfo.utcoffset(datetime) is not None
    )


def from_timestamp(value: typing.Any) -> dt.datetime:
    if value is True or value is False:
        raise ValueError("Not a valid POSIX timestamp")
    value = float(value)
    if value < 0:
        raise ValueError("Not a valid POSIX timestamp")

    # Load a timestamp with utc as timezone to prevent using system timezone.
    # Then set timezone to None, to let the Field handle adding timezone info.
    try:
        return dt.datetime.fromtimestamp(value, tz=dt.timezone.utc).replace(tzinfo=None)
    except OverflowError as exc:
        raise ValueError("Timestamp is too large") from exc
    except OSError as exc:
        raise ValueError("Error converting value to datetime") from exc


def from_timestamp_ms(value: typing.Any) -> dt.datetime:
    value = float(value)
    return from_timestamp(value / 1000)


def timestamp(
    value: dt.datetime,
) -> float:
    if not is_aware(value):
        # When a date is naive, use UTC as zone info to prevent using system timezone.
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.timestamp()


def timestamp_ms(value: dt.datetime) -> float:
    return timestamp(value) * 1000


def ensure_text_type(val: str | bytes) -> str:
    if isinstance(val, bytes):
        val = val.decode("utf-8")
    return str(val)


def pluck(dictlist: list[dict[str, typing.Any]], key: str):
    """Extracts a list of dictionary values from a list of dictionaries.
    ::

        >>> dlist = [{'id': 1, 'name': 'foo'}, {'id': 2, 'name': 'bar'}]
        >>> pluck(dlist, 'id')
        [1, 2]
    """
    return [d[key] for d in dictlist]


# Various utilities for pulling keyed values from objects


def get_value(obj, key: int | str, default=missing):
    """Helper for pulling a keyed value off various types of objects. Fields use
    this method by default to access attributes of the source object. For object `x`
    and attribute `i`, this method first tries to access `x[i]`, and then falls back to
    `x.i` if an exception is raised.

    .. warning::
        If an object `x` does not raise an exception when `x[i]` does not exist,
        `get_value` will never check the value `x.i`. Consider overriding
        `marshmallow.fields.Field.get_value` in this case.
    """
    if not isinstance(key, int) and "." in key:
        return _get_value_for_keys(obj, key.split("."), default)
    return _get_value_for_key(obj, key, default)


def _get_value_for_keys(obj, keys, default):
    if len(keys) == 1:
        return _get_value_for_key(obj, keys[0], default)
    return _get_value_for_keys(
        _get_value_for_key(obj, keys[0], default), keys[1:], default
    )


def _get_value_for_key(obj, key, default):
    if not hasattr(obj, "__getitem__"):
        return getattr(obj, key, default)

    try:
        return obj[key]
    except (KeyError, IndexError, TypeError, AttributeError):
        return getattr(obj, key, default)


def set_value(dct: dict[str, typing.Any], key: str, value: typing.Any):
    """Set a value in a dict. If `key` contains a '.', it is assumed
    be a path (i.e. dot-delimited string) to the value's location.

    ::

        >>> d = {}
        >>> set_value(d, 'foo.bar', 42)
        >>> d
        {'foo': {'bar': 42}}
    """
    if "." in key:
        head, rest = key.split(".", 1)
        target = dct.setdefault(head, {})
        if not isinstance(target, dict):
            raise ValueError(
                f"Cannot set {key} in {head} due to existing value: {target}"
            )
        set_value(target, rest, value)
    else:
        dct[key] = value


def callable_or_raise(obj):
    """Check that an object is callable, else raise a :exc:`TypeError`."""
    if not callable(obj):
        raise TypeError(f"Object {obj!r} is not callable.")
    return obj


def timedelta_to_microseconds(value: dt.timedelta) -> int:
    """Compute the total microseconds of a timedelta.

    https://github.com/python/cpython/blob/v3.13.1/Lib/_pydatetime.py#L805-L807
    """
    return (value.days * (24 * 3600) + value.seconds) * 1000000 + value.microseconds


# === src/marshmallow/error_store.py ===
"""Utilities for storing collections of error messages.

.. warning::

    This module is treated as private API.
    Users should not need to use this module directly.
"""

from marshmallow.exceptions import SCHEMA


class ErrorStore:
    def __init__(self):
        #: Dictionary of errors stored during serialization
        self.errors = {}

    def store_error(self, messages, field_name=SCHEMA, index=None):
        # field error  -> store/merge error messages under field name key
        # schema error -> if string or list, store/merge under _schema key
        #              -> if dict, store/merge with other top-level keys
        if field_name != SCHEMA or not isinstance(messages, dict):
            messages = {field_name: messages}
        if index is not None:
            messages = {index: messages}
        self.errors = merge_errors(self.errors, messages)


def merge_errors(errors1, errors2):  # noqa: PLR0911
    """Deeply merge two error messages.

    The format of ``errors1`` and ``errors2`` matches the ``message``
    parameter of :exc:`marshmallow.exceptions.ValidationError`.
    """
    if not errors1:
        return errors2
    if not errors2:
        return errors1
    if isinstance(errors1, list):
        if isinstance(errors2, list):
            return errors1 + errors2
        if isinstance(errors2, dict):
            return dict(errors2, **{SCHEMA: merge_errors(errors1, errors2.get(SCHEMA))})
        return [*errors1, errors2]
    if isinstance(errors1, dict):
        if isinstance(errors2, list):
            return dict(errors1, **{SCHEMA: merge_errors(errors1.get(SCHEMA), errors2)})
        if isinstance(errors2, dict):
            errors = dict(errors1)
            for key, val in errors2.items():
                if key in errors:
                    errors[key] = merge_errors(errors[key], val)
                else:
                    errors[key] = val
            return errors
        return dict(errors1, **{SCHEMA: merge_errors(errors1.get(SCHEMA), errors2)})
    if isinstance(errors2, list):
        return [errors1, *errors2]
    if isinstance(errors2, dict):
        return dict(errors2, **{SCHEMA: merge_errors(errors1, errors2.get(SCHEMA))})
    return [errors1, errors2]


# === src/marshmallow/exceptions.py ===
"""Exception classes for marshmallow-related errors."""

from __future__ import annotations

import typing

# Key used for schema-level validation errors
SCHEMA = "_schema"


class MarshmallowError(Exception):
    """Base class for all marshmallow-related errors."""


class ValidationError(MarshmallowError):
    """Raised when validation fails on a field or schema.

    Validators and custom fields should raise this exception.

    :param message: An error message, list of error messages, or dict of
        error messages. If a dict, the keys are subitems and the values are error messages.
    :param field_name: Field name to store the error on.
        If `None`, the error is stored as schema-level error.
    :param data: Raw input data.
    :param valid_data: Valid (de)serialized data.
    """

    def __init__(
        self,
        message: str | list | dict,
        field_name: str = SCHEMA,
        data: typing.Mapping[str, typing.Any]
        | typing.Iterable[typing.Mapping[str, typing.Any]]
        | None = None,
        valid_data: list[typing.Any] | dict[str, typing.Any] | None = None,
        **kwargs,
    ):
        self.messages = [message] if isinstance(message, (str, bytes)) else message
        self.field_name = field_name
        self.data = data
        self.valid_data = valid_data
        self.kwargs = kwargs
        super().__init__(message)

    def normalized_messages(self):
        if self.field_name == SCHEMA and isinstance(self.messages, dict):
            return self.messages
        return {self.field_name: self.messages}

    @property
    def messages_dict(self) -> dict[str, typing.Any]:
        if not isinstance(self.messages, dict):
            raise TypeError(
                "cannot access 'messages_dict' when 'messages' is of type "
                + type(self.messages).__name__
            )
        return self.messages


class RegistryError(NameError):
    """Raised when an invalid operation is performed on the serializer
    class registry.
    """


class StringNotCollectionError(MarshmallowError, TypeError):
    """Raised when a string is passed when a list of strings is expected."""


class _FieldInstanceResolutionError(MarshmallowError, TypeError):
    """Raised when an argument is passed to a field class that cannot be resolved to a Field instance."""


# === src/marshmallow/schema.py ===
"""The `Schema <marshmallow.Schema>` class, including its metaclass and options (`class Meta <marshmallow.Schema.Meta>`)."""

# ruff: noqa: SLF001
from __future__ import annotations

import copy
import datetime as dt
import decimal
import functools
import inspect
import json
import operator
import typing
import uuid
from abc import ABCMeta
from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import zip_longest

from marshmallow import class_registry, types
from marshmallow import fields as ma_fields
from marshmallow.constants import EXCLUDE, INCLUDE, RAISE, missing
from marshmallow.decorators import (
    POST_DUMP,
    POST_LOAD,
    PRE_DUMP,
    PRE_LOAD,
    VALIDATES,
    VALIDATES_SCHEMA,
)
from marshmallow.error_store import ErrorStore
from marshmallow.exceptions import SCHEMA, StringNotCollectionError, ValidationError
from marshmallow.orderedset import OrderedSet
from marshmallow.utils import (
    get_value,
    is_collection,
    is_sequence_but_not_string,
    set_value,
)

if typing.TYPE_CHECKING:
    from marshmallow.fields import Field


def _get_fields(attrs) -> list[tuple[str, Field]]:
    """Get fields from a class

    :param attrs: Mapping of class attributes
    """
    ret = []
    for field_name, field_value in attrs.items():
        if isinstance(field_value, type) and issubclass(field_value, ma_fields.Field):
            raise TypeError(
                f'Field for "{field_name}" must be declared as a '
                "Field instance, not a class. "
                f'Did you mean "fields.{field_value.__name__}()"?'
            )
        if isinstance(field_value, ma_fields.Field):
            ret.append((field_name, field_value))
    return ret


# This function allows Schemas to inherit from non-Schema classes and ensures
#   inheritance according to the MRO
def _get_fields_by_mro(klass: SchemaMeta):
    """Collect fields from a class, following its method resolution order. The
    class itself is excluded from the search; only its parents are checked. Get
    fields from ``_declared_fields`` if available, else use ``__dict__``.

    :param klass: Class whose fields to retrieve
    """
    mro = inspect.getmro(klass)
    # Combine fields from all parents
    # functools.reduce(operator.iadd, list_of_lists) is faster than sum(list_of_lists, [])
    # Loop over mro in reverse to maintain correct order of fields
    return functools.reduce(
        operator.iadd,
        (
            _get_fields(
                getattr(base, "_declared_fields", base.__dict__),
            )
            for base in mro[:0:-1]
        ),
        [],
    )


class SchemaMeta(ABCMeta):
    """Metaclass for the Schema class. Binds the declared fields to
    a ``_declared_fields`` attribute, which is a dictionary mapping attribute
    names to field objects. Also sets the ``opts`` class attribute, which is
    the Schema class's `class Meta <marshmallow.Schema.Meta>` options.
    """

    Meta: type
    opts: typing.Any
    OPTIONS_CLASS: type
    _declared_fields: dict[str, Field]

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        attrs: dict[str, typing.Any],
    ) -> SchemaMeta:
        meta = attrs.get("Meta")
        cls_fields = _get_fields(attrs)
        # Remove fields from list of class attributes to avoid shadowing
        # Schema attributes/methods in case of name conflict
        for field_name, _ in cls_fields:
            del attrs[field_name]
        klass = super().__new__(mcs, name, bases, attrs)
        inherited_fields = _get_fields_by_mro(klass)

        meta = klass.Meta
        # Set klass.opts in __new__ rather than __init__ so that it is accessible in
        # get_declared_fields
        klass.opts = klass.OPTIONS_CLASS(meta)
        # Add fields specified in the `include` class Meta option
        cls_fields += list(klass.opts.include.items())

        # Assign _declared_fields on class
        klass._declared_fields = mcs.get_declared_fields(
            klass=klass,
            cls_fields=cls_fields,
            inherited_fields=inherited_fields,
            dict_cls=dict,
        )
        return klass

    @classmethod
    def get_declared_fields(
        mcs,  # noqa: N804
        klass: SchemaMeta,
        cls_fields: list[tuple[str, Field]],
        inherited_fields: list[tuple[str, Field]],
        dict_cls: type[dict] = dict,
    ) -> dict[str, Field]:
        """Returns a dictionary of field_name => `Field` pairs declared on the class.
        This is exposed mainly so that plugins can add additional fields, e.g. fields
        computed from `class Meta <marshmallow.Schema.Meta>` options.

        :param klass: The class object.
        :param cls_fields: The fields declared on the class, including those added
            by the ``include`` `class Meta <marshmallow.Schema.Meta>` option.
        :param inherited_fields: Inherited fields.
        :param dict_cls: dict-like class to use for dict output Default to ``dict``.
        """
        return dict_cls(inherited_fields + cls_fields)

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        if name and cls.opts.register:
            class_registry.register(name, cls)
        cls._hooks = cls.resolve_hooks()

    def resolve_hooks(cls) -> dict[str, list[tuple[str, bool, dict]]]:
        """Add in the decorated processors

        By doing this after constructing the class, we let standard inheritance
        do all the hard work.
        """
        mro = inspect.getmro(cls)

        hooks: dict[str, list[tuple[str, bool, dict]]] = defaultdict(list)

        for attr_name in dir(cls):
            # Need to look up the actual descriptor, not whatever might be
            # bound to the class. This needs to come from the __dict__ of the
            # declaring class.
            for parent in mro:
                try:
                    attr = parent.__dict__[attr_name]
                except KeyError:
                    continue
                else:
                    break
            else:
                # In case we didn't find the attribute and didn't break above.
                # We should never hit this - it's just here for completeness
                # to exclude the possibility of attr being undefined.
                continue

            try:
                hook_config: dict[str, list[tuple[bool, dict]]] = (
                    attr.__marshmallow_hook__
                )
            except AttributeError:
                pass
            else:
                for tag, config in hook_config.items():
                    # Use name here so we can get the bound method later, in
                    # case the processor was a descriptor or something.
                    hooks[tag].extend(
                        (attr_name, many, kwargs) for many, kwargs in config
                    )

        return hooks


class SchemaOpts:
    """Defines defaults for `marshmallow.Schema.Meta`."""

    def __init__(self, meta: type):
        self.fields = getattr(meta, "fields", ())
        if not isinstance(self.fields, (list, tuple)):
            raise ValueError("`fields` option must be a list or tuple.")
        self.exclude = getattr(meta, "exclude", ())
        if not isinstance(self.exclude, (list, tuple)):
            raise ValueError("`exclude` must be a list or tuple.")
        self.dateformat = getattr(meta, "dateformat", None)
        self.datetimeformat = getattr(meta, "datetimeformat", None)
        self.timeformat = getattr(meta, "timeformat", None)
        self.render_module = getattr(meta, "render_module", json)
        self.index_errors = getattr(meta, "index_errors", True)
        self.include = getattr(meta, "include", {})
        self.load_only = getattr(meta, "load_only", ())
        self.dump_only = getattr(meta, "dump_only", ())
        self.unknown = getattr(meta, "unknown", RAISE)
        self.register = getattr(meta, "register", True)
        self.many = getattr(meta, "many", False)


class Schema(metaclass=SchemaMeta):
    """Base schema class with which to define schemas.

    Example usage:

    .. code-block:: python

        import datetime as dt
        from dataclasses import dataclass

        from marshmallow import Schema, fields


        @dataclass
        class Album:
            title: str
            release_date: dt.date


        class AlbumSchema(Schema):
            title = fields.Str()
            release_date = fields.Date()


        album = Album("Beggars Banquet", dt.date(1968, 12, 6))
        schema = AlbumSchema()
        data = schema.dump(album)
        data  # {'release_date': '1968-12-06', 'title': 'Beggars Banquet'}

    :param only: Whitelist of the declared fields to select when
        instantiating the Schema. If None, all fields are used. Nested fields
        can be represented with dot delimiters.
    :param exclude: Blacklist of the declared fields to exclude
        when instantiating the Schema. If a field appears in both `only` and
        `exclude`, it is not used. Nested fields can be represented with dot
        delimiters.
    :param many: Should be set to `True` if ``obj`` is a collection
        so that the object will be serialized to a list.
    :param load_only: Fields to skip during serialization (write-only fields)
    :param dump_only: Fields to skip during deserialization (read-only fields)
    :param partial: Whether to ignore missing fields and not require
        any fields declared. Propagates down to ``Nested`` fields as well. If
        its value is an iterable, only missing fields listed in that iterable
        will be ignored. Use dot delimiters to specify nested fields.
    :param unknown: Whether to exclude, include, or raise an error for unknown
        fields in the data. Use `EXCLUDE`, `INCLUDE` or `RAISE`.

    .. versionchanged:: 3.0.0
        Remove ``prefix`` parameter.

    .. versionchanged:: 4.0.0
        Remove ``context`` parameter.
    """

    TYPE_MAPPING: dict[type, type[Field]] = {
        str: ma_fields.String,
        bytes: ma_fields.String,
        dt.datetime: ma_fields.DateTime,
        float: ma_fields.Float,
        bool: ma_fields.Boolean,
        tuple: ma_fields.Raw,
        list: ma_fields.Raw,
        set: ma_fields.Raw,
        int: ma_fields.Integer,
        uuid.UUID: ma_fields.UUID,
        dt.time: ma_fields.Time,
        dt.date: ma_fields.Date,
        dt.timedelta: ma_fields.TimeDelta,
        decimal.Decimal: ma_fields.Decimal,
    }
    #: Overrides for default schema-level error messages
    error_messages: dict[str, str] = {}

    _default_error_messages: dict[str, str] = {
        "type": "Invalid input type.",
        "unknown": "Unknown field.",
    }

    OPTIONS_CLASS: type = SchemaOpts

    set_class = OrderedSet
    dict_class: type[dict] = dict
    """`dict` type to return when serializing."""

    # These get set by SchemaMeta
    opts: typing.Any
    _declared_fields: dict[str, Field] = {}
    _hooks: dict[str, list[tuple[str, bool, dict]]] = {}

    class Meta:
        """Options object for a Schema.

        Example usage: ::

            from marshmallow import Schema


            class MySchema(Schema):
                class Meta:
                    fields = ("id", "email", "date_created")
                    exclude = ("password", "secret_attribute")

        .. admonition:: A note on type checking

            Type checkers will only check the attributes of the `Meta <marshmallow.Schema.Meta>`
            class if you explicitly subclass `marshmallow.Schema.Meta`.

            .. code-block:: python

                from marshmallow import Schema


                class MySchema(Schema):
                    # Not checked by type checkers
                    class Meta:
                        additional = True


                class MySchema2(Schema):
                    # Type checkers will check attributes
                    class Meta(Schema.Opts):
                        additional = True  # Incompatible types in assignment

        .. versionremoved:: 3.0.0b7 Remove ``strict``.
        .. versionadded:: 3.0.0b12 Add `unknown`.
        .. versionchanged:: 3.0.0b17 Rename ``dateformat`` to `datetimeformat`.
        .. versionadded:: 3.9.0 Add `timeformat`.
        .. versionchanged:: 3.26.0 Deprecate ``ordered``. Field order is preserved by default.
        .. versionremoved:: 4.0.0 Remove ``ordered``.
        """

        fields: typing.ClassVar[tuple[str, ...] | list[str]]
        """Fields to include in the (de)serialized result"""
        additional: typing.ClassVar[tuple[str, ...] | list[str]]
        """Fields to include in addition to the explicitly declared fields.
        `additional <marshmallow.Schema.Meta.additional>` and `fields <marshmallow.Schema.Meta.fields>`
        are mutually-exclusive options.
        """
        include: typing.ClassVar[dict[str, Field]]
        """Dictionary of additional fields to include in the schema. It is
        usually better to define fields as class variables, but you may need to
        use this option, e.g., if your fields are Python keywords.
        """
        exclude: typing.ClassVar[tuple[str, ...] | list[str]]
        """Fields to exclude in the serialized result.
        Nested fields can be represented with dot delimiters.
        """
        many: typing.ClassVar[bool]
        """Whether data should be (de)serialized as a collection by default."""
        dateformat: typing.ClassVar[str]
        """Default format for `Date <marshmallow.fields.Date>` fields."""
        datetimeformat: typing.ClassVar[str]
        """Default format for `DateTime <marshmallow.fields.DateTime>` fields."""
        timeformat: typing.ClassVar[str]
        """Default format for `Time <marshmallow.fields.Time>` fields."""

        # FIXME: Use a more constrained type here.
        # ClassVar[RenderModule] doesn't work.
        render_module: typing.Any
        """ Module to use for `loads <marshmallow.Schema.loads>` and `dumps <marshmallow.Schema.dumps>`.
        Defaults to `json` from the standard library.
        """
        index_errors: typing.ClassVar[bool]
        """If `True`, errors dictionaries will include the index of invalid items in a collection."""
        load_only: typing.ClassVar[tuple[str, ...] | list[str]]
        """Fields to exclude from serialized results"""
        dump_only: typing.ClassVar[tuple[str, ...] | list[str]]
        """Fields to exclude from serialized results"""
        unknown: typing.ClassVar[types.UnknownOption]
        """Whether to exclude, include, or raise an error for unknown fields in the data.
        Use `EXCLUDE`, `INCLUDE` or `RAISE`.
        """
        register: typing.ClassVar[bool]
        """Whether to register the `Schema <marshmallow.Schema>` with marshmallow's internal
        class registry. Must be `True` if you intend to refer to this `Schema <marshmallow.Schema>`
        by class name in `Nested` fields. Only set this to `False` when memory
        usage is critical. Defaults to `True`.
        """

    def __init__(
        self,
        *,
        only: types.StrSequenceOrSet | None = None,
        exclude: types.StrSequenceOrSet = (),
        many: bool | None = None,
        load_only: types.StrSequenceOrSet = (),
        dump_only: types.StrSequenceOrSet = (),
        partial: bool | types.StrSequenceOrSet | None = None,
        unknown: types.UnknownOption | None = None,
    ):
        # Raise error if only or exclude is passed as string, not list of strings
        if only is not None and not is_collection(only):
            raise StringNotCollectionError('"only" should be a list of strings')
        if not is_collection(exclude):
            raise StringNotCollectionError('"exclude" should be a list of strings')
        # copy declared fields from metaclass
        self.declared_fields = copy.deepcopy(self._declared_fields)
        self.many = self.opts.many if many is None else many
        self.only = only
        self.exclude: set[typing.Any] | typing.MutableSet[typing.Any] = set(
            self.opts.exclude
        ) | set(exclude)
        self.load_only = set(load_only) or set(self.opts.load_only)
        self.dump_only = set(dump_only) or set(self.opts.dump_only)
        self.partial = partial
        self.unknown: types.UnknownOption = (
            self.opts.unknown if unknown is None else unknown
        )
        self._normalize_nested_options()
        #: Dictionary mapping field_names -> :class:`Field` objects
        self.fields: dict[str, Field] = {}
        self.load_fields: dict[str, Field] = {}
        self.dump_fields: dict[str, Field] = {}
        self._init_fields()
        messages = {}
        messages.update(self._default_error_messages)
        for cls in reversed(self.__class__.__mro__):
            messages.update(getattr(cls, "error_messages", {}))
        messages.update(self.error_messages or {})
        self.error_messages = messages

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(many={self.many})>"

    @classmethod
    def from_dict(
        cls,
        fields: dict[str, Field],
        *,
        name: str = "GeneratedSchema",
    ) -> type[Schema]:
        """Generate a `Schema <marshmallow.Schema>` class given a dictionary of fields.

        .. code-block:: python

            from marshmallow import Schema, fields

            PersonSchema = Schema.from_dict({"name": fields.Str()})
            print(PersonSchema().load({"name": "David"}))  # => {'name': 'David'}

        Generated schemas are not added to the class registry and therefore cannot
        be referred to by name in `Nested` fields.


        :param fields: Dictionary mapping field names to field instances.
        :param name: Optional name for the class, which will appear in
            the ``repr`` for the class.

        .. versionadded:: 3.0.0
        """
        Meta = type(
            "GeneratedMeta", (getattr(cls, "Meta", object),), {"register": False}
        )
        return type(name, (cls,), {**fields.copy(), "Meta": Meta})

    ##### Override-able methods #####

    def handle_error(
        self, error: ValidationError, data: typing.Any, *, many: bool, **kwargs
    ):
        """Custom error handler function for the schema.

        :param error: The `ValidationError` raised during (de)serialization.
        :param data: The original input data.
        :param many: Value of ``many`` on dump or load.
        :param partial: Value of ``partial`` on load.

        .. versionchanged:: 3.0.0rc9
            Receives `many` and `partial` (on deserialization) as keyword arguments.
        """

    def get_attribute(self, obj: typing.Any, attr: str, default: typing.Any):
        """Defines how to pull values from an object to serialize.

        .. versionchanged:: 3.0.0a1
            Changed position of ``obj`` and ``attr``.
        """
        return get_value(obj, attr, default)

    ##### Serialization/Deserialization API #####

    @staticmethod
    def _call_and_store(getter_func, data, *, field_name, error_store, index=None):
        """Call ``getter_func`` with ``data`` as its argument, and store any `ValidationErrors`.

        :param getter_func: Function for getting the serialized/deserialized
            value from ``data``.
        :param data: The data passed to ``getter_func``.
        :param field_name: Field name.
        :param index: Index of the item being validated, if validating a collection,
            otherwise `None`.
        """
        try:
            value = getter_func(data)
        except ValidationError as error:
            error_store.store_error(error.messages, field_name, index=index)
            # When a Nested field fails validation, the marshalled data is stored
            # on the ValidationError's valid_data attribute
            return error.valid_data or missing
        return value

    def _serialize(self, obj: typing.Any, *, many: bool = False):
        """Serialize ``obj``.

        :param obj: The object(s) to serialize.
        :param many: `True` if ``data`` should be serialized as a collection.
        :return: A dictionary of the serialized data
        """
        if many and obj is not None:
            return [self._serialize(d, many=False) for d in obj]
        ret = self.dict_class()
        for attr_name, field_obj in self.dump_fields.items():
            value = field_obj.serialize(attr_name, obj, accessor=self.get_attribute)
            if value is missing:
                continue
            key = field_obj.data_key if field_obj.data_key is not None else attr_name
            ret[key] = value
        return ret

    def dump(self, obj: typing.Any, *, many: bool | None = None):
        """Serialize an object to native Python data types according to this
        Schema's fields.

        :param obj: The object to serialize.
        :param many: Whether to serialize `obj` as a collection. If `None`, the value
            for `self.many` is used.
        :return: Serialized data

        .. versionchanged:: 3.0.0b7
            This method returns the serialized data rather than a ``(data, errors)`` duple.
            A :exc:`ValidationError <marshmallow.exceptions.ValidationError>` is raised
            if ``obj`` is invalid.
        .. versionchanged:: 3.0.0rc9
            Validation no longer occurs upon serialization.
        """
        many = self.many if many is None else bool(many)
        if self._hooks[PRE_DUMP]:
            processed_obj = self._invoke_dump_processors(
                PRE_DUMP, obj, many=many, original_data=obj
            )
        else:
            processed_obj = obj

        result = self._serialize(processed_obj, many=many)

        if self._hooks[POST_DUMP]:
            result = self._invoke_dump_processors(
                POST_DUMP, result, many=many, original_data=obj
            )

        return result

    def dumps(self, obj: typing.Any, *args, many: bool | None = None, **kwargs):
        """Same as :meth:`dump`, except return a JSON-encoded string.

        :param obj: The object to serialize.
        :param many: Whether to serialize `obj` as a collection. If `None`, the value
            for `self.many` is used.
        :return: A ``json`` string

        .. versionchanged:: 3.0.0b7
            This method returns the serialized data rather than a ``(data, errors)`` duple.
            A :exc:`ValidationError <marshmallow.exceptions.ValidationError>` is raised
            if ``obj`` is invalid.
        """
        serialized = self.dump(obj, many=many)
        return self.opts.render_module.dumps(serialized, *args, **kwargs)

    def _deserialize(
        self,
        data: Mapping[str, typing.Any] | Sequence[Mapping[str, typing.Any]],
        *,
        error_store: ErrorStore,
        many: bool = False,
        partial=None,
        unknown: types.UnknownOption = RAISE,
        index=None,
    ) -> typing.Any | list[typing.Any]:
        """Deserialize ``data``.

        :param data: The data to deserialize.
        :param error_store: Structure to store errors.
        :param many: `True` if ``data`` should be deserialized as a collection.
        :param partial: Whether to ignore missing fields and not require
            any fields declared. Propagates down to ``Nested`` fields as well. If
            its value is an iterable, only missing fields listed in that iterable
            will be ignored. Use dot delimiters to specify nested fields.
        :param unknown: Whether to exclude, include, or raise an error for unknown
            fields in the data. Use `EXCLUDE`, `INCLUDE` or `RAISE`.
        :param index: Index of the item being serialized (for storing errors) if
            serializing a collection, otherwise `None`.
        :return: The deserialized data as `dict_class` instance or list of `dict_class`
        instances if `many` is `True`.
        """
        index_errors = self.opts.index_errors
        index = index if index_errors else None
        if many:
            if not is_sequence_but_not_string(data):
                error_store.store_error([self.error_messages["type"]], index=index)
                ret_l = []
            else:
                ret_l = [
                    self._deserialize(
                        d,
                        error_store=error_store,
                        many=False,
                        partial=partial,
                        unknown=unknown,
                        index=idx,
                    )
                    for idx, d in enumerate(data)
                ]
            return ret_l
        ret_d = self.dict_class()
        # Check data is a dict
        if not isinstance(data, Mapping):
            error_store.store_error([self.error_messages["type"]], index=index)
        else:
            partial_is_collection = is_collection(partial)
            for attr_name, field_obj in self.load_fields.items():
                field_name = (
                    field_obj.data_key if field_obj.data_key is not None else attr_name
                )
                raw_value = data.get(field_name, missing)
                if raw_value is missing:
                    # Ignore missing field if we're allowed to.
                    if partial is True or (
                        partial_is_collection and attr_name in partial
                    ):
                        continue
                d_kwargs = {}
                # Allow partial loading of nested schemas.
                if partial_is_collection:
                    prefix = field_name + "."
                    len_prefix = len(prefix)
                    sub_partial = [
                        f[len_prefix:] for f in partial if f.startswith(prefix)
                    ]
                    d_kwargs["partial"] = sub_partial
                elif partial is not None:
                    d_kwargs["partial"] = partial

                def getter(
                    val, field_obj=field_obj, field_name=field_name, d_kwargs=d_kwargs
                ):
                    return field_obj.deserialize(
                        val,
                        field_name,
                        data,
                        **d_kwargs,
                    )

                value = self._call_and_store(
                    getter_func=getter,
                    data=raw_value,
                    field_name=field_name,
                    error_store=error_store,
                    index=index,
                )
                if value is not missing:
                    key = field_obj.attribute or attr_name
                    set_value(ret_d, key, value)
            if unknown != EXCLUDE:
                fields = {
                    field_obj.data_key if field_obj.data_key is not None else field_name
                    for field_name, field_obj in self.load_fields.items()
                }
                for key in set(data) - fields:
                    value = data[key]
                    if unknown == INCLUDE:
                        ret_d[key] = value
                    elif unknown == RAISE:
                        error_store.store_error(
                            [self.error_messages["unknown"]],
                            key,
                            (index if index_errors else None),
                        )
        return ret_d

    def load(
        self,
        data: Mapping[str, typing.Any] | Sequence[Mapping[str, typing.Any]],
        *,
        many: bool | None = None,
        partial: bool | types.StrSequenceOrSet | None = None,
        unknown: types.UnknownOption | None = None,
    ):
        """Deserialize a data structure to an object defined by this Schema's fields.

        :param data: The data to deserialize.
        :param many: Whether to deserialize `data` as a collection. If `None`, the
            value for `self.many` is used.
        :param partial: Whether to ignore missing fields and not require
            any fields declared. Propagates down to ``Nested`` fields as well. If
            its value is an iterable, only missing fields listed in that iterable
            will be ignored. Use dot delimiters to specify nested fields.
        :param unknown: Whether to exclude, include, or raise an error for unknown
            fields in the data. Use `EXCLUDE`, `INCLUDE` or `RAISE`.
            If `None`, the value for `self.unknown` is used.
        :return: Deserialized data

        .. versionchanged:: 3.0.0b7
            This method returns the deserialized data rather than a ``(data, errors)`` duple.
            A :exc:`ValidationError <marshmallow.exceptions.ValidationError>` is raised
            if invalid data are passed.
        """
        return self._do_load(
            data, many=many, partial=partial, unknown=unknown, postprocess=True
        )

    def loads(
        self,
        s: str | bytes | bytearray,
        /,
        *,
        many: bool | None = None,
        partial: bool | types.StrSequenceOrSet | None = None,
        unknown: types.UnknownOption | None = None,
        **kwargs,
    ):
        """Same as :meth:`load`, except it uses `marshmallow.Schema.Meta.render_module` to deserialize
        the passed string before passing data to :meth:`load`.

        :param s: A string of the data to deserialize.
        :param many: Whether to deserialize `obj` as a collection. If `None`, the
            value for `self.many` is used.
        :param partial: Whether to ignore missing fields and not require
            any fields declared. Propagates down to ``Nested`` fields as well. If
            its value is an iterable, only missing fields listed in that iterable
            will be ignored. Use dot delimiters to specify nested fields.
        :param unknown: Whether to exclude, include, or raise an error for unknown
            fields in the data. Use `EXCLUDE`, `INCLUDE` or `RAISE`.
            If `None`, the value for `self.unknown` is used.
        :return: Deserialized data

        .. versionchanged:: 3.0.0b7
            This method returns the deserialized data rather than a ``(data, errors)`` duple.
            A :exc:`ValidationError <marshmallow.exceptions.ValidationError>` is raised
            if invalid data are passed.
        .. versionchanged:: 4.0.0
            Rename ``json_module`` parameter to ``s``.
        """
        data = self.opts.render_module.loads(s, **kwargs)
        return self.load(data, many=many, partial=partial, unknown=unknown)

    def _run_validator(
        self,
        validator_func: types.SchemaValidator,
        output,
        *,
        original_data,
        error_store: ErrorStore,
        many: bool,
        partial: bool | types.StrSequenceOrSet | None,
        unknown: types.UnknownOption | None,
        pass_original: bool,
        index: int | None = None,
    ):
        try:
            if pass_original:  # Pass original, raw data (before unmarshalling)
                validator_func(
                    output, original_data, partial=partial, many=many, unknown=unknown
                )
            else:
                validator_func(output, partial=partial, many=many, unknown=unknown)
        except ValidationError as err:
            field_name = err.field_name
            data_key: str
            if field_name == SCHEMA:
                data_key = SCHEMA
            else:
                field_obj: Field | None = None
                try:
                    field_obj = self.fields[field_name]
                except KeyError:
                    if field_name in self.declared_fields:
                        field_obj = self.declared_fields[field_name]
                if field_obj:
                    data_key = (
                        field_obj.data_key
                        if field_obj.data_key is not None
                        else field_name
                    )
                else:
                    data_key = field_name
            error_store.store_error(err.messages, data_key, index=index)

    def validate(
        self,
        data: Mapping[str, typing.Any] | Sequence[Mapping[str, typing.Any]],
        *,
        many: bool | None = None,
        partial: bool | types.StrSequenceOrSet | None = None,
    ) -> dict[str, list[str]]:
        """Validate `data` against the schema, returning a dictionary of
        validation errors.

        :param data: The data to validate.
        :param many: Whether to validate `data` as a collection. If `None`, the
            value for `self.many` is used.
        :param partial: Whether to ignore missing fields and not require
            any fields declared. Propagates down to ``Nested`` fields as well. If
            its value is an iterable, only missing fields listed in that iterable
            will be ignored. Use dot delimiters to specify nested fields.
        :return: A dictionary of validation errors.
        """
        try:
            self._do_load(data, many=many, partial=partial, postprocess=False)
        except ValidationError as exc:
            return typing.cast("dict[str, list[str]]", exc.messages)
        return {}

    ##### Private Helpers #####

    def _do_load(
        self,
        data: (Mapping[str, typing.Any] | Sequence[Mapping[str, typing.Any]]),
        *,
        many: bool | None = None,
        partial: bool | types.StrSequenceOrSet | None = None,
        unknown: types.UnknownOption | None = None,
        postprocess: bool = True,
    ):
        """Deserialize `data`, returning the deserialized result.
        This method is private API.

        :param data: The data to deserialize.
        :param many: Whether to deserialize `data` as a collection. If `None`, the
            value for `self.many` is used.
        :param partial: Whether to validate required fields. If its
            value is an iterable, only fields listed in that iterable will be
            ignored will be allowed missing. If `True`, all fields will be allowed missing.
            If `None`, the value for `self.partial` is used.
        :param unknown: Whether to exclude, include, or raise an error for unknown
            fields in the data. Use `EXCLUDE`, `INCLUDE` or `RAISE`.
            If `None`, the value for `self.unknown` is used.
        :param postprocess: Whether to run post_load methods..
        :return: Deserialized data
        """
        error_store = ErrorStore()
        errors: dict[str, list[str]] = {}
        many = self.many if many is None else bool(many)
        unknown = self.unknown if unknown is None else unknown
        if partial is None:
            partial = self.partial
        # Run preprocessors
        if self._hooks[PRE_LOAD]:
            try:
                processed_data = self._invoke_load_processors(
                    PRE_LOAD,
                    data,
                    many=many,
                    original_data=data,
                    partial=partial,
                    unknown=unknown,
                )
            except ValidationError as err:
                errors = err.normalized_messages()
                result: list | dict | None = None
        else:
            processed_data = data
        if not errors:
            # Deserialize data
            result = self._deserialize(
                processed_data,
                error_store=error_store,
                many=many,
                partial=partial,
                unknown=unknown,
            )
            # Run field-level validation
            self._invoke_field_validators(
                error_store=error_store, data=result, many=many
            )
            # Run schema-level validation
            if self._hooks[VALIDATES_SCHEMA]:
                field_errors = bool(error_store.errors)
                self._invoke_schema_validators(
                    error_store=error_store,
                    pass_collection=True,
                    data=result,
                    original_data=data,
                    many=many,
                    partial=partial,
                    unknown=unknown,
                    field_errors=field_errors,
                )
                self._invoke_schema_validators(
                    error_store=error_store,
                    pass_collection=False,
                    data=result,
                    original_data=data,
                    many=many,
                    partial=partial,
                    unknown=unknown,
                    field_errors=field_errors,
                )
            errors = error_store.errors
            # Run post processors
            if not errors and postprocess and self._hooks[POST_LOAD]:
                try:
                    result = self._invoke_load_processors(
                        POST_LOAD,
                        result,
                        many=many,
                        original_data=data,
                        partial=partial,
                        unknown=unknown,
                    )
                except ValidationError as err:
                    errors = err.normalized_messages()
        if errors:
            exc = ValidationError(errors, data=data, valid_data=result)
            self.handle_error(exc, data, many=many, partial=partial)
            raise exc

        return result

    def _normalize_nested_options(self) -> None:
        """Apply then flatten nested schema options.
        This method is private API.
        """
        if self.only is not None:
            # Apply the only option to nested fields.
            self.__apply_nested_option("only", self.only, "intersection")
            # Remove the child field names from the only option.
            self.only = self.set_class([field.split(".", 1)[0] for field in self.only])
        if self.exclude:
            # Apply the exclude option to nested fields.
            self.__apply_nested_option("exclude", self.exclude, "union")
            # Remove the parent field names from the exclude option.
            self.exclude = self.set_class(
                [field for field in self.exclude if "." not in field]
            )

    def __apply_nested_option(self, option_name, field_names, set_operation) -> None:
        """Apply nested options to nested fields"""
        # Split nested field names on the first dot.
        nested_fields = [name.split(".", 1) for name in field_names if "." in name]
        # Partition the nested field names by parent field.
        nested_options = defaultdict(list)  # type: defaultdict
        for parent, nested_names in nested_fields:
            nested_options[parent].append(nested_names)
        # Apply the nested field options.
        for key, options in iter(nested_options.items()):
            new_options = self.set_class(options)
            original_options = getattr(self.declared_fields[key], option_name, ())
            if original_options:
                if set_operation == "union":
                    new_options |= self.set_class(original_options)
                if set_operation == "intersection":
                    new_options &= self.set_class(original_options)
            setattr(self.declared_fields[key], option_name, new_options)

    def _init_fields(self) -> None:
        """Update self.fields, self.load_fields, and self.dump_fields based on schema options.
        This method is private API.
        """
        if self.opts.fields:
            available_field_names = self.set_class(self.opts.fields)
        else:
            available_field_names = self.set_class(self.declared_fields.keys())

        invalid_fields = self.set_class()

        if self.only is not None:
            # Return only fields specified in only option
            field_names: typing.AbstractSet[typing.Any] = self.set_class(self.only)

            invalid_fields |= field_names - available_field_names
        else:
            field_names = available_field_names

        # If "exclude" option or param is specified, remove those fields.
        if self.exclude:
            # Note that this isn't available_field_names, since we want to
            # apply "only" for the actual calculation.
            field_names = field_names - self.exclude
            invalid_fields |= self.exclude - available_field_names

        if invalid_fields:
            message = f"Invalid fields for {self}: {invalid_fields}."
            raise ValueError(message)

        fields_dict = self.dict_class()
        for field_name in field_names:
            field_obj = self.declared_fields[field_name]
            self._bind_field(field_name, field_obj)
            fields_dict[field_name] = field_obj

        load_fields, dump_fields = self.dict_class(), self.dict_class()
        for field_name, field_obj in fields_dict.items():
            if not field_obj.dump_only:
                load_fields[field_name] = field_obj
            if not field_obj.load_only:
                dump_fields[field_name] = field_obj

        dump_data_keys = [
            field_obj.data_key if field_obj.data_key is not None else name
            for name, field_obj in dump_fields.items()
        ]
        if len(dump_data_keys) != len(set(dump_data_keys)):
            data_keys_duplicates = {
                x for x in dump_data_keys if dump_data_keys.count(x) > 1
            }
            raise ValueError(
                "The data_key argument for one or more fields collides "
                "with another field's name or data_key argument. "
                "Check the following field names and "
                f"data_key arguments: {list(data_keys_duplicates)}"
            )
        load_attributes = [obj.attribute or name for name, obj in load_fields.items()]
        if len(load_attributes) != len(set(load_attributes)):
            attributes_duplicates = {
                x for x in load_attributes if load_attributes.count(x) > 1
            }
            raise ValueError(
                "The attribute argument for one or more fields collides "
                "with another field's name or attribute argument. "
                "Check the following field names and "
                f"attribute arguments: {list(attributes_duplicates)}"
            )

        self.fields = fields_dict
        self.dump_fields = dump_fields
        self.load_fields = load_fields

    def on_bind_field(self, field_name: str, field_obj: Field) -> None:
        """Hook to modify a field when it is bound to the `Schema <marshmallow.Schema>`.

        No-op by default.
        """
        return

    def _bind_field(self, field_name: str, field_obj: Field) -> None:
        """Bind field to the schema, setting any necessary attributes on the
        field (e.g. parent and name).

        Also set field load_only and dump_only values if field_name was
        specified in `class Meta <marshmallow.Schema.Meta>`.
        """
        if field_name in self.load_only:
            field_obj.load_only = True
        if field_name in self.dump_only:
            field_obj.dump_only = True
        field_obj._bind_to_schema(field_name, self)
        self.on_bind_field(field_name, field_obj)

    def _invoke_dump_processors(
        self, tag: str, data, *, many: bool, original_data=None
    ):
        # The pass_collection post-dump processors may do things like add an envelope, so
        # invoke those after invoking the non-pass_collection processors which will expect
        # to get a list of items.
        data = self._invoke_processors(
            tag,
            pass_collection=False,
            data=data,
            many=many,
            original_data=original_data,
        )
        return self._invoke_processors(
            tag, pass_collection=True, data=data, many=many, original_data=original_data
        )

    def _invoke_load_processors(
        self,
        tag: str,
        data: Mapping[str, typing.Any] | Sequence[Mapping[str, typing.Any]],
        *,
        many: bool,
        original_data,
        partial: bool | types.StrSequenceOrSet | None,
        unknown: types.UnknownOption | None,
    ):
        # This has to invert the order of the dump processors, so run the pass_collection
        # processors first.
        data = self._invoke_processors(
            tag,
            pass_collection=True,
            data=data,
            many=many,
            original_data=original_data,
            partial=partial,
            unknown=unknown,
        )
        return self._invoke_processors(
            tag,
            pass_collection=False,
            data=data,
            many=many,
            original_data=original_data,
            partial=partial,
            unknown=unknown,
        )

    def _invoke_field_validators(self, *, error_store: ErrorStore, data, many: bool):
        for attr_name, _, validator_kwargs in self._hooks[VALIDATES]:
            validator = getattr(self, attr_name)

            field_names = validator_kwargs["field_names"]

            for field_name in field_names:
                try:
                    field_obj = self.fields[field_name]
                except KeyError as error:
                    if field_name in self.declared_fields:
                        continue
                    raise ValueError(f'"{field_name}" field does not exist.') from error

                data_key = (
                    field_obj.data_key if field_obj.data_key is not None else field_name
                )
                do_validate = functools.partial(validator, data_key=data_key)

                if many:
                    for idx, item in enumerate(data):
                        try:
                            value = item[field_obj.attribute or field_name]
                        except KeyError:
                            pass
                        else:
                            validated_value = self._call_and_store(
                                getter_func=do_validate,
                                data=value,
                                field_name=data_key,
                                error_store=error_store,
                                index=(idx if self.opts.index_errors else None),
                            )
                            if validated_value is missing:
                                item.pop(field_name, None)
                else:
                    try:
                        value = data[field_obj.attribute or field_name]
                    except KeyError:
                        pass
                    else:
                        validated_value = self._call_and_store(
                            getter_func=do_validate,
                            data=value,
                            field_name=data_key,
                            error_store=error_store,
                        )
                        if validated_value is missing:
                            data.pop(field_name, None)

    def _invoke_schema_validators(
        self,
        *,
        error_store: ErrorStore,
        pass_collection: bool,
        data,
        original_data,
        many: bool,
        partial: bool | types.StrSequenceOrSet | None,
        field_errors: bool = False,
        unknown: types.UnknownOption | None,
    ):
        for attr_name, hook_many, validator_kwargs in self._hooks[VALIDATES_SCHEMA]:
            if hook_many != pass_collection:
                continue
            validator = getattr(self, attr_name)
            if field_errors and validator_kwargs["skip_on_field_errors"]:
                continue
            pass_original = validator_kwargs.get("pass_original", False)

            if many and not pass_collection:
                for idx, (item, orig) in enumerate(zip(data, original_data)):
                    self._run_validator(
                        validator,
                        item,
                        original_data=orig,
                        error_store=error_store,
                        many=many,
                        partial=partial,
                        unknown=unknown,
                        index=idx,
                        pass_original=pass_original,
                    )
            else:
                self._run_validator(
                    validator,
                    data,
                    original_data=original_data,
                    error_store=error_store,
                    many=many,
                    pass_original=pass_original,
                    partial=partial,
                    unknown=unknown,
                )

    def _invoke_processors(
        self,
        tag: str,
        *,
        pass_collection: bool,
        data: Mapping[str, typing.Any] | Sequence[Mapping[str, typing.Any]],
        many: bool,
        original_data=None,
        **kwargs,
    ):
        for attr_name, hook_many, processor_kwargs in self._hooks[tag]:
            if hook_many != pass_collection:
                continue
            # This will be a bound method.
            processor = getattr(self, attr_name)
            pass_original = processor_kwargs.get("pass_original", False)

            if many and not pass_collection:
                if pass_original:
                    data = [
                        processor(item, original, many=many, **kwargs)
                        for item, original in zip_longest(data, original_data)
                    ]
                else:
                    data = [processor(item, many=many, **kwargs) for item in data]
            elif pass_original:
                data = processor(data, original_data, many=many, **kwargs)
            else:
                data = processor(data, many=many, **kwargs)
        return data


BaseSchema = Schema  # for backwards compatibility


# === src/marshmallow/decorators.py ===
"""Decorators for registering schema pre-processing and post-processing methods.
These should be imported from the top-level `marshmallow` module.

Methods decorated with
`pre_load <marshmallow.decorators.pre_load>`, `post_load <marshmallow.decorators.post_load>`,
`pre_dump <marshmallow.decorators.pre_dump>`, `post_dump <marshmallow.decorators.post_dump>`,
and `validates_schema <marshmallow.decorators.validates_schema>` receive
``many`` as a keyword argument. In addition, `pre_load <marshmallow.decorators.pre_load>`,
`post_load <marshmallow.decorators.post_load>`,
and `validates_schema <marshmallow.decorators.validates_schema>` receive
``partial``. If you don't need these arguments, add ``**kwargs`` to your method
signature.


Example: ::

    from marshmallow import (
        Schema,
        pre_load,
        pre_dump,
        post_load,
        validates_schema,
        validates,
        fields,
        ValidationError,
    )


    class UserSchema(Schema):
        email = fields.Str(required=True)
        age = fields.Integer(required=True)

        @post_load
        def lowerstrip_email(self, item, many, **kwargs):
            item["email"] = item["email"].lower().strip()
            return item

        @pre_load(pass_collection=True)
        def remove_envelope(self, data, many, **kwargs):
            namespace = "results" if many else "result"
            return data[namespace]

        @post_dump(pass_collection=True)
        def add_envelope(self, data, many, **kwargs):
            namespace = "results" if many else "result"
            return {namespace: data}

        @validates_schema
        def validate_email(self, data, **kwargs):
            if len(data["email"]) < 3:
                raise ValidationError("Email must be more than 3 characters", "email")

        @validates("age")
        def validate_age(self, data, **kwargs):
            if data < 14:
                raise ValidationError("Too young!")

.. note::
    These decorators only work with instance methods. Class and static
    methods are not supported.

.. warning::
    The invocation order of decorated methods of the same type is not guaranteed.
    If you need to guarantee order of different processing steps, you should put
    them in the same processing method.
"""

from __future__ import annotations

import functools
from collections import defaultdict
from typing import Any, Callable, cast

PRE_DUMP = "pre_dump"
POST_DUMP = "post_dump"
PRE_LOAD = "pre_load"
POST_LOAD = "post_load"
VALIDATES = "validates"
VALIDATES_SCHEMA = "validates_schema"


class MarshmallowHook:
    __marshmallow_hook__: dict[str, list[tuple[bool, Any]]] | None = None


def validates(*field_names: str) -> Callable[..., Any]:
    """Register a validator method for field(s).

    :param field_names: Names of the fields that the method validates.

    .. versionchanged:: 4.0.0 Accepts multiple field names as positional arguments.
    .. versionchanged:: 4.0.0 Decorated methods receive ``data_key`` as a keyword argument.
    """
    return set_hook(None, VALIDATES, field_names=field_names)


def validates_schema(
    fn: Callable[..., Any] | None = None,
    *,
    pass_collection: bool = False,
    pass_original: bool = False,
    skip_on_field_errors: bool = True,
) -> Callable[..., Any]:
    """Register a schema-level validator.

    By default it receives a single object at a time, transparently handling the ``many``
    argument passed to the `Schema <marshmallow.Schema>`'s :func:`~marshmallow.Schema.validate` call.
    If ``pass_collection=True``, the raw data (which may be a collection) is passed.

    If ``pass_original=True``, the original data (before unmarshalling) will be passed as
    an additional argument to the method.

    If ``skip_on_field_errors=True``, this validation method will be skipped whenever
    validation errors have been detected when validating fields.

    .. versionchanged:: 3.0.0b1 ``skip_on_field_errors`` defaults to `True`.
    .. versionchanged:: 3.0.0 ``partial`` and ``many`` are always passed as keyword arguments to
        the decorated method.
    .. versionchanged:: 4.0.0 ``unknown`` is passed as a keyword argument to the decorated method.
    .. versionchanged:: 4.0.0 ``pass_many`` is renamed to ``pass_collection``.
    .. versionchanged:: 4.0.0 ``pass_collection``, ``pass_original``, and ``skip_on_field_errors``
        are keyword-only arguments.
    """
    return set_hook(
        fn,
        VALIDATES_SCHEMA,
        many=pass_collection,
        pass_original=pass_original,
        skip_on_field_errors=skip_on_field_errors,
    )


def pre_dump(
    fn: Callable[..., Any] | None = None,
    *,
    pass_collection: bool = False,
) -> Callable[..., Any]:
    """Register a method to invoke before serializing an object. The method
    receives the object to be serialized and returns the processed object.

    By default it receives a single object at a time, transparently handling the ``many``
    argument passed to the `Schema <marshmallow.Schema>`'s :func:`~marshmallow.Schema.dump` call.
    If ``pass_collection=True``, the raw data (which may be a collection) is passed.

    .. versionchanged:: 3.0.0 ``many`` is always passed as a keyword arguments to the decorated method.
    .. versionchanged:: 4.0.0 ``pass_many`` is renamed to ``pass_collection``.
    .. versionchanged:: 4.0.0 ``pass_collection`` is a keyword-only argument.
    """
    return set_hook(fn, PRE_DUMP, many=pass_collection)


def post_dump(
    fn: Callable[..., Any] | None = None,
    *,
    pass_collection: bool = False,
    pass_original: bool = False,
) -> Callable[..., Any]:
    """Register a method to invoke after serializing an object. The method
    receives the serialized object and returns the processed object.

    By default it receives a single object at a time, transparently handling the ``many``
    argument passed to the `Schema <marshmallow.Schema>`'s :func:`~marshmallow.Schema.dump` call.
    If ``pass_collection=True``, the raw data (which may be a collection) is passed.

    If ``pass_original=True``, the original data (before serializing) will be passed as
    an additional argument to the method.

    .. versionchanged:: 3.0.0 ``many`` is always passed as a keyword arguments to the decorated method.
    .. versionchanged:: 4.0.0 ``pass_many`` is renamed to ``pass_collection``.
    .. versionchanged:: 4.0.0 ``pass_collection`` and ``pass_original`` are keyword-only arguments.
    """
    return set_hook(fn, POST_DUMP, many=pass_collection, pass_original=pass_original)


def pre_load(
    fn: Callable[..., Any] | None = None,
    *,
    pass_collection: bool = False,
) -> Callable[..., Any]:
    """Register a method to invoke before deserializing an object. The method
    receives the data to be deserialized and returns the processed data.

    By default it receives a single object at a time, transparently handling the ``many``
    argument passed to the `Schema <marshmallow.Schema>`'s :func:`~marshmallow.Schema.load` call.
    If ``pass_collection=True``, the raw data (which may be a collection) is passed.

    .. versionchanged:: 3.0.0 ``partial`` and ``many`` are always passed as keyword arguments to
        the decorated method.
    .. versionchanged:: 4.0.0 ``pass_many`` is renamed to ``pass_collection``.
    .. versionchanged:: 4.0.0 ``pass_collection`` is a keyword-only argument.
    .. versionchanged:: 4.0.0 ``unknown`` is passed as a keyword argument to the decorated method.
    """
    return set_hook(fn, PRE_LOAD, many=pass_collection)


def post_load(
    fn: Callable[..., Any] | None = None,
    *,
    pass_collection: bool = False,
    pass_original: bool = False,
) -> Callable[..., Any]:
    """Register a method to invoke after deserializing an object. The method
    receives the deserialized data and returns the processed data.

    By default it receives a single object at a time, transparently handling the ``many``
    argument passed to the `Schema <marshmallow.Schema>`'s :func:`~marshmallow.Schema.load` call.
    If ``pass_collection=True``, the raw data (which may be a collection) is passed.

    If ``pass_original=True``, the original data (before deserializing) will be passed as
    an additional argument to the method.

    .. versionchanged:: 3.0.0 ``partial`` and ``many`` are always passed as keyword arguments to
        the decorated method.
    .. versionchanged:: 4.0.0 ``pass_many`` is renamed to ``pass_collection``.
    .. versionchanged:: 4.0.0 ``pass_collection`` and ``pass_original`` are keyword-only arguments.
    .. versionchanged:: 4.0.0 ``unknown`` is passed as a keyword argument to the decorated method.
    """
    return set_hook(fn, POST_LOAD, many=pass_collection, pass_original=pass_original)


def set_hook(
    fn: Callable[..., Any] | None,
    tag: str,
    *,
    many: bool = False,
    **kwargs: Any,
) -> Callable[..., Any]:
    """Mark decorated function as a hook to be picked up later.
    You should not need to use this method directly.

    .. note::
        Currently only works with functions and instance methods. Class and
        static methods are not supported.

    :return: Decorated function if supplied, else this decorator with its args
        bound.
    """
    # Allow using this as either a decorator or a decorator factory.
    if fn is None:
        return functools.partial(set_hook, tag=tag, many=many, **kwargs)

    # Set a __marshmallow_hook__ attribute instead of wrapping in some class,
    # because I still want this to end up as a normal (unbound) method.
    function = cast("MarshmallowHook", fn)
    try:
        hook_config = function.__marshmallow_hook__
    except AttributeError:
        function.__marshmallow_hook__ = hook_config = defaultdict(list)
    # Also save the kwargs for the tagged function on
    # __marshmallow_hook__, keyed by <tag>
    if hook_config is not None:
        hook_config[tag].append((many, kwargs))

    return fn


# === src/marshmallow/experimental/__init__.py ===
"""Experimental features.

The features in this subpackage are experimental. Breaking changes may be
introduced in minor marshmallow versions.
"""


# === src/marshmallow/experimental/context.py ===
"""Helper API for setting serialization/deserialization context.

Example usage:

.. code-block:: python

    import typing

    from marshmallow import Schema, fields
    from marshmallow.experimental.context import Context


    class UserContext(typing.TypedDict):
        suffix: str


    UserSchemaContext = Context[UserContext]


    class UserSchema(Schema):
        name_suffixed = fields.Function(
            lambda user: user["name"] + UserSchemaContext.get()["suffix"]
        )


    with UserSchemaContext({"suffix": "bar"}):
        print(UserSchema().dump({"name": "foo"}))
        # {'name_suffixed': 'foobar'}
"""

from __future__ import annotations

import contextlib
import contextvars
import typing

try:
    from types import EllipsisType
except ImportError:  # Python<3.10
    EllipsisType = type(Ellipsis)  # type: ignore[misc]

_ContextT = typing.TypeVar("_ContextT")
_DefaultT = typing.TypeVar("_DefaultT")
_CURRENT_CONTEXT: contextvars.ContextVar = contextvars.ContextVar("context")


class Context(contextlib.AbstractContextManager, typing.Generic[_ContextT]):
    """Context manager for setting and retrieving context.

    :param context: The context to use within the context manager scope.
    """

    def __init__(self, context: _ContextT) -> None:
        self.context = context
        self.token: contextvars.Token | None = None

    def __enter__(self) -> Context[_ContextT]:
        self.token = _CURRENT_CONTEXT.set(self.context)
        return self

    def __exit__(self, *args, **kwargs) -> None:
        _CURRENT_CONTEXT.reset(typing.cast("contextvars.Token", self.token))

    @classmethod
    def get(cls, default: _DefaultT | EllipsisType = ...) -> _ContextT | _DefaultT:
        """Get the current context.

        :param default: Default value to return if no context is set.
            If not provided and no context is set, a :exc:`LookupError` is raised.
        """
        if default is not ...:
            return _CURRENT_CONTEXT.get(default)
        return _CURRENT_CONTEXT.get()
