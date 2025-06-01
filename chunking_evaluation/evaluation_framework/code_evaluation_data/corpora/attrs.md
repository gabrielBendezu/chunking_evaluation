# SPDX-License-Identifier: MIT

from datetime import timedelta

import pytest

from hypothesis import HealthCheck, settings

from attr._compat import PY_3_10_PLUS


@pytest.fixture(name="slots", params=(True, False))
def _slots(request):
    return request.param


@pytest.fixture(name="frozen", params=(True, False))
def _frozen(request):
    return request.param


def pytest_configure(config):
    # HealthCheck.too_slow causes more trouble than good -- especially in CIs.
    settings.register_profile(
        "patience",
        settings(
            suppress_health_check=[HealthCheck.too_slow],
            deadline=timedelta(milliseconds=400),
        ),
    )
    settings.load_profile("patience")


collect_ignore = []
if not PY_3_10_PLUS:
    collect_ignore.extend(["tests/test_pattern_matching.py"])


"""
Benchmark attrs using CodSpeed.
"""

from __future__ import annotations

import pytest

import attrs


pytestmark = pytest.mark.benchmark()

ROUNDS = 1_000


def test_create_simple_class():
    """
    Benchmark creating  a simple class without any extras.
    """
    for _ in range(ROUNDS):

        @attrs.define
        class LocalC:
            x: int
            y: str
            z: dict[str, int]


def test_create_frozen_class():
    """
    Benchmark creating a frozen class without any extras.
    """
    for _ in range(ROUNDS):

        @attrs.frozen
        class LocalC:
            x: int
            y: str
            z: dict[str, int]

        LocalC(1, "2", {})


def test_create_simple_class_make_class():
    """
    Benchmark creating a simple class using attrs.make_class().
    """
    for i in range(ROUNDS):
        LocalC = attrs.make_class(
            f"LocalC{i}",
            {
                "x": attrs.field(type=int),
                "y": attrs.field(type=str),
                "z": attrs.field(type=dict[str, int]),
            },
        )

        LocalC(1, "2", {})


@attrs.define
class C:
    x: int = 0
    y: str = "foo"
    z: dict[str, int] = attrs.Factory(dict)


def test_instantiate_no_defaults():
    """
    Benchmark instantiating a class without using any defaults.
    """
    for _ in range(ROUNDS):
        C(1, "2", {})


def test_instantiate_with_defaults():
    """
    Benchmark instantiating a class relying on defaults.
    """
    for _ in range(ROUNDS):
        C()


def test_eq_equal():
    """
    Benchmark comparing two equal instances for equality.
    """
    c1 = C()
    c2 = C()

    for _ in range(ROUNDS):
        c1 == c2


def test_eq_unequal():
    """
    Benchmark comparing two unequal instances for equality.
    """
    c1 = C()
    c2 = C(1, "bar", {"baz": 42})

    for _ in range(ROUNDS):
        c1 == c2


@attrs.frozen
class HashableC:
    x: int = 0
    y: str = "foo"
    z: tuple[str] = ("bar",)


def test_hash():
    """
    Benchmark hashing an instance.
    """
    c = HashableC()

    for _ in range(ROUNDS):
        hash(c)


# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import shutil
import subprocess

from pathlib import Path

import pytest


pytestmark = [
    pytest.mark.skipif(
        shutil.which("pyright") is None, reason="Requires pyright."
    ),
]


def parse_pyright_output(test_file: Path) -> set[tuple[str, str]]:
    pyright = subprocess.run(  # noqa: PLW1510
        ["pyright", "--outputjson", str(test_file)], capture_output=True
    )

    pyright_result = json.loads(pyright.stdout)

    # We use tuples instead of proper classes to get nicer diffs from pytest.
    return {
        (d["severity"], d["message"])
        for d in pyright_result["generalDiagnostics"]
    }


def test_pyright_baseline():
    """
    The typing.dataclass_transform decorator allows pyright to determine
    attrs decorated class types.
    """

    test_file = Path(__file__).parent / "dataclass_transform_example.py"

    diagnostics = parse_pyright_output(test_file)

    expected_diagnostics = {
        (
            "information",
            'Type of "Define.__init__" is "(self: Define, a: str, b: int) -> None"',
        ),
        (
            "information",
            'Type of "DefineConverter.__init__" is '
            '"(self: DefineConverter, with_converter: str | Buffer | '
            'SupportsInt | SupportsIndex | SupportsTrunc) -> None"',
        ),
        (
            "error",
            'Cannot assign to attribute "a" for class '
            '"Frozen"\n\xa0\xa0Attribute "a" is read-only',
        ),
        (
            "information",
            'Type of "d.a" is "Literal[\'new\']"',
        ),
        (
            "error",
            'Cannot assign to attribute "a" for class '
            '"FrozenDefine"\n\xa0\xa0Attribute "a" is read-only',
        ),
        (
            "information",
            'Type of "d2.a" is "Literal[\'new\']"',
        ),
        (
            "information",
            'Type of "af.__init__" is "(_a: int) -> None"',
        ),
    }

    assert expected_diagnostics == diagnostics


def test_pyright_attrsinstance_compat(tmp_path):
    """
    Test that `AttrsInstance` is compatible with Pyright.
    """
    test_pyright_attrsinstance_compat_path = (
        tmp_path / "test_pyright_attrsinstance_compat.py"
    )
    test_pyright_attrsinstance_compat_path.write_text(
        """\
import attrs

# We can assign any old object to `AttrsInstance`.
foo: attrs.AttrsInstance = object()

reveal_type(attrs.AttrsInstance)
"""
    )

    diagnostics = parse_pyright_output(test_pyright_attrsinstance_compat_path)
    expected_diagnostics = {
        (
            "information",
            'Type of "attrs.AttrsInstance" is "type[AttrsInstance]"',
        )
    }
    assert diagnostics == expected_diagnostics


# SPDX-License-Identifier: MIT

"""
Tests for `attr._funcs`.
"""

import re

from collections import OrderedDict
from typing import Generic, NamedTuple, TypeVar

import pytest

from hypothesis import assume, given
from hypothesis import strategies as st

import attr

from attr import asdict, assoc, astuple, evolve, fields, has
from attr._compat import Mapping, Sequence
from attr.exceptions import AttrsAttributeNotFoundError
from attr.validators import instance_of

from .strategies import nested_classes, simple_classes


MAPPING_TYPES = (dict, OrderedDict)
SEQUENCE_TYPES = (list, tuple)


@pytest.fixture(scope="session", name="C")
def _C():
    """
    Return a simple but fully featured attrs class with an x and a y attribute.
    """
    import attr

    @attr.s
    class C:
        x = attr.ib()
        y = attr.ib()

    return C


class TestAsDict:
    """
    Tests for `asdict`.
    """

    @given(st.sampled_from(MAPPING_TYPES))
    def test_shallow(self, C, dict_factory):
        """
        Shallow asdict returns correct dict.
        """
        assert {"x": 1, "y": 2} == asdict(
            C(x=1, y=2), False, dict_factory=dict_factory
        )

    @given(st.sampled_from(MAPPING_TYPES))
    def test_recurse(self, C, dict_class):
        """
        Deep asdict returns correct dict.
        """
        assert {"x": {"x": 1, "y": 2}, "y": {"x": 3, "y": 4}} == asdict(
            C(C(1, 2), C(3, 4)), dict_factory=dict_class
        )

    def test_nested_lists(self, C):
        """
        Test unstructuring deeply nested lists.
        """
        inner = C(1, 2)
        outer = C([[inner]], None)

        assert {"x": [[{"x": 1, "y": 2}]], "y": None} == asdict(outer)

    def test_nested_dicts(self, C):
        """
        Test unstructuring deeply nested dictionaries.
        """
        inner = C(1, 2)
        outer = C({1: {2: inner}}, None)

        assert {"x": {1: {2: {"x": 1, "y": 2}}}, "y": None} == asdict(outer)

    @given(nested_classes, st.sampled_from(MAPPING_TYPES))
    def test_recurse_property(self, cls, dict_class):
        """
        Property tests for recursive asdict.
        """
        obj = cls()
        obj_dict = asdict(obj, dict_factory=dict_class)

        def assert_proper_dict_class(obj, obj_dict):
            assert isinstance(obj_dict, dict_class)

            for field in fields(obj.__class__):
                field_val = getattr(obj, field.name)
                if has(field_val.__class__):
                    # This field holds a class, recurse the assertions.
                    assert_proper_dict_class(field_val, obj_dict[field.name])
                elif isinstance(field_val, Sequence):
                    dict_val = obj_dict[field.name]
                    for item, item_dict in zip(field_val, dict_val):
                        if has(item.__class__):
                            assert_proper_dict_class(item, item_dict)
                elif isinstance(field_val, Mapping):
                    # This field holds a dictionary.
                    assert isinstance(obj_dict[field.name], dict_class)

                    for key, val in field_val.items():
                        if has(val.__class__):
                            assert_proper_dict_class(
                                val, obj_dict[field.name][key]
                            )

        assert_proper_dict_class(obj, obj_dict)

    @given(st.sampled_from(MAPPING_TYPES))
    def test_filter(self, C, dict_factory):
        """
        Attributes that are supposed to be skipped are skipped.
        """
        assert {"x": {"x": 1}} == asdict(
            C(C(1, 2), C(3, 4)),
            filter=lambda a, v: a.name != "y",
            dict_factory=dict_factory,
        )

    @given(container=st.sampled_from(SEQUENCE_TYPES))
    def test_lists_tuples(self, container, C):
        """
        If recurse is True, also recurse into lists.
        """
        assert {
            "x": 1,
            "y": [{"x": 2, "y": 3}, {"x": 4, "y": 5}, "a"],
        } == asdict(C(1, container([C(2, 3), C(4, 5), "a"])))

    @given(container=st.sampled_from(SEQUENCE_TYPES))
    def test_lists_tuples_retain_type(self, container, C):
        """
        If recurse and retain_collection_types are True, also recurse
        into lists and do not convert them into list.
        """
        assert {
            "x": 1,
            "y": container([{"x": 2, "y": 3}, {"x": 4, "y": 5}, "a"]),
        } == asdict(
            C(1, container([C(2, 3), C(4, 5), "a"])),
            retain_collection_types=True,
        )

    @given(set_type=st.sampled_from((set, frozenset)))
    def test_sets_no_retain(self, C, set_type):
        """
        Set types are converted to lists if retain_collection_types=False.
        """
        d = asdict(
            C(1, set_type((1, 2, 3))),
            retain_collection_types=False,
            recurse=True,
        )

        assert {"x": 1, "y": [1, 2, 3]} == d

    @given(st.sampled_from(MAPPING_TYPES))
    def test_dicts(self, C, dict_factory):
        """
        If recurse is True, also recurse into dicts.
        """
        res = asdict(C(1, {"a": C(4, 5)}), dict_factory=dict_factory)

        assert {"x": 1, "y": {"a": {"x": 4, "y": 5}}} == res
        assert isinstance(res, dict_factory)

    @given(simple_classes(private_attrs=False), st.sampled_from(MAPPING_TYPES))
    def test_roundtrip(self, cls, dict_class):
        """
        Test dumping to dicts and back for Hypothesis-generated classes.

        Private attributes don't round-trip (the attribute name is different
        than the initializer argument).
        """
        instance = cls()
        dict_instance = asdict(instance, dict_factory=dict_class)

        assert isinstance(dict_instance, dict_class)

        roundtrip_instance = cls(**dict_instance)

        assert instance == roundtrip_instance

    @given(simple_classes())
    def test_asdict_preserve_order(self, cls):
        """
        Field order should be preserved when dumping to an ordered_dict.
        """
        instance = cls()
        dict_instance = asdict(instance, dict_factory=dict)

        assert [a.name for a in fields(cls)] == list(dict_instance.keys())

    def test_retain_keys_are_tuples(self):
        """
        retain_collect_types also retains keys.
        """

        @attr.s
        class A:
            a = attr.ib()

        instance = A({(1,): 1})

        assert {"a": {(1,): 1}} == attr.asdict(
            instance, retain_collection_types=True
        )

    def test_tuple_keys(self):
        """
        If a key is collection type, retain_collection_types is False,
         the key is serialized as a tuple.

        See #646
        """

        @attr.s
        class A:
            a = attr.ib()

        instance = A({(1,): 1})

        assert {"a": {(1,): 1}} == attr.asdict(instance)

    def test_named_tuple_retain_type(self):
        """
        Namedtuples can be serialized if retain_collection_types is True.

        See #1164
        """

        class Coordinates(NamedTuple):
            lat: float
            lon: float

        @attr.s
        class A:
            coords: Coordinates = attr.ib()

        instance = A(Coordinates(50.419019, 30.516225))

        assert {"coords": Coordinates(50.419019, 30.516225)} == attr.asdict(
            instance, retain_collection_types=True
        )

    def test_type_error_with_retain_type(self):
        """
        Serialization that fails with TypeError leaves the error through if
        they're not tuples.

        See #1164
        """

        message = "__new__() missing 1 required positional argument (asdict)"

        class Coordinates(list):
            def __init__(self, first, *rest):
                if isinstance(first, list):
                    raise TypeError(message)
                super().__init__([first, *rest])

        @attr.s
        class A:
            coords: Coordinates = attr.ib()

        instance = A(Coordinates(50.419019, 30.516225))

        with pytest.raises(TypeError, match=re.escape(message)):
            attr.asdict(instance, retain_collection_types=True)


class TestAsTuple:
    """
    Tests for `astuple`.
    """

    @given(st.sampled_from(SEQUENCE_TYPES))
    def test_shallow(self, C, tuple_factory):
        """
        Shallow astuple returns correct dict.
        """
        assert tuple_factory([1, 2]) == astuple(
            C(x=1, y=2), False, tuple_factory=tuple_factory
        )

    @given(st.sampled_from(SEQUENCE_TYPES))
    def test_recurse(self, C, tuple_factory):
        """
        Deep astuple returns correct tuple.
        """
        assert tuple_factory(
            [tuple_factory([1, 2]), tuple_factory([3, 4])]
        ) == astuple(C(C(1, 2), C(3, 4)), tuple_factory=tuple_factory)

    @given(nested_classes, st.sampled_from(SEQUENCE_TYPES))
    def test_recurse_property(self, cls, tuple_class):
        """
        Property tests for recursive astuple.
        """
        obj = cls()
        obj_tuple = astuple(obj, tuple_factory=tuple_class)

        def assert_proper_tuple_class(obj, obj_tuple):
            assert isinstance(obj_tuple, tuple_class)
            for index, field in enumerate(fields(obj.__class__)):
                field_val = getattr(obj, field.name)
                if has(field_val.__class__):
                    # This field holds a class, recurse the assertions.
                    assert_proper_tuple_class(field_val, obj_tuple[index])

        assert_proper_tuple_class(obj, obj_tuple)

    @given(nested_classes, st.sampled_from(SEQUENCE_TYPES))
    def test_recurse_retain(self, cls, tuple_class):
        """
        Property tests for asserting collection types are retained.
        """
        obj = cls()
        obj_tuple = astuple(
            obj, tuple_factory=tuple_class, retain_collection_types=True
        )

        def assert_proper_col_class(obj, obj_tuple):
            # Iterate over all attributes, and if they are lists or mappings
            # in the original, assert they are the same class in the dumped.
            for index, field in enumerate(fields(obj.__class__)):
                field_val = getattr(obj, field.name)
                if has(field_val.__class__):
                    # This field holds a class, recurse the assertions.
                    assert_proper_col_class(field_val, obj_tuple[index])
                elif isinstance(field_val, (list, tuple)):
                    # This field holds a sequence of something.
                    expected_type = type(obj_tuple[index])
                    assert type(field_val) is expected_type
                    for obj_e, obj_tuple_e in zip(field_val, obj_tuple[index]):
                        if has(obj_e.__class__):
                            assert_proper_col_class(obj_e, obj_tuple_e)
                elif isinstance(field_val, dict):
                    orig = field_val
                    tupled = obj_tuple[index]
                    assert type(orig) is type(tupled)
                    for obj_e, obj_tuple_e in zip(
                        orig.items(), tupled.items()
                    ):
                        if has(obj_e[0].__class__):  # Dict key
                            assert_proper_col_class(obj_e[0], obj_tuple_e[0])
                        if has(obj_e[1].__class__):  # Dict value
                            assert_proper_col_class(obj_e[1], obj_tuple_e[1])

        assert_proper_col_class(obj, obj_tuple)

    @given(st.sampled_from(SEQUENCE_TYPES))
    def test_filter(self, C, tuple_factory):
        """
        Attributes that are supposed to be skipped are skipped.
        """
        assert tuple_factory([tuple_factory([1])]) == astuple(
            C(C(1, 2), C(3, 4)),
            filter=lambda a, v: a.name != "y",
            tuple_factory=tuple_factory,
        )

    @given(container=st.sampled_from(SEQUENCE_TYPES))
    def test_lists_tuples(self, container, C):
        """
        If recurse is True, also recurse into lists.
        """
        assert (1, [(2, 3), (4, 5), "a"]) == astuple(
            C(1, container([C(2, 3), C(4, 5), "a"]))
        )

    @given(st.sampled_from(SEQUENCE_TYPES))
    def test_dicts(self, C, tuple_factory):
        """
        If recurse is True, also recurse into dicts.
        """
        res = astuple(C(1, {"a": C(4, 5)}), tuple_factory=tuple_factory)
        assert tuple_factory([1, {"a": tuple_factory([4, 5])}]) == res
        assert isinstance(res, tuple_factory)

    @given(container=st.sampled_from(SEQUENCE_TYPES))
    def test_lists_tuples_retain_type(self, container, C):
        """
        If recurse and retain_collection_types are True, also recurse
        into lists and do not convert them into list.
        """
        assert (1, container([(2, 3), (4, 5), "a"])) == astuple(
            C(1, container([C(2, 3), C(4, 5), "a"])),
            retain_collection_types=True,
        )

    @given(container=st.sampled_from(MAPPING_TYPES))
    def test_dicts_retain_type(self, container, C):
        """
        If recurse and retain_collection_types are True, also recurse
        into lists and do not convert them into list.
        """
        assert (1, container({"a": (4, 5)})) == astuple(
            C(1, container({"a": C(4, 5)})), retain_collection_types=True
        )

    @given(simple_classes(), st.sampled_from(SEQUENCE_TYPES))
    def test_roundtrip(self, cls, tuple_class):
        """
        Test dumping to tuple and back for Hypothesis-generated classes.
        """
        instance = cls()
        tuple_instance = astuple(instance, tuple_factory=tuple_class)

        assert isinstance(tuple_instance, tuple_class)

        roundtrip_instance = cls(*tuple_instance)

        assert instance == roundtrip_instance

    @given(set_type=st.sampled_from((set, frozenset)))
    def test_sets_no_retain(self, C, set_type):
        """
        Set types are converted to lists if retain_collection_types=False.
        """
        d = astuple(
            C(1, set_type((1, 2, 3))),
            retain_collection_types=False,
            recurse=True,
        )

        assert (1, [1, 2, 3]) == d

    def test_named_tuple_retain_type(self):
        """
        Namedtuples can be serialized if retain_collection_types is True.

        See #1164
        """

        class Coordinates(NamedTuple):
            lat: float
            lon: float

        @attr.s
        class A:
            coords: Coordinates = attr.ib()

        instance = A(Coordinates(50.419019, 30.516225))

        assert (Coordinates(50.419019, 30.516225),) == attr.astuple(
            instance, retain_collection_types=True
        )

    def test_type_error_with_retain_type(self):
        """
        Serialization that fails with TypeError leaves the error through if
        they're not tuples.

        See #1164
        """

        message = "__new__() missing 1 required positional argument (astuple)"

        class Coordinates(list):
            def __init__(self, first, *rest):
                if isinstance(first, list):
                    raise TypeError(message)
                super().__init__([first, *rest])

        @attr.s
        class A:
            coords: Coordinates = attr.ib()

        instance = A(Coordinates(50.419019, 30.516225))

        with pytest.raises(TypeError, match=re.escape(message)):
            attr.astuple(instance, retain_collection_types=True)


class TestHas:
    """
    Tests for `has`.
    """

    def test_positive(self, C):
        """
        Returns `True` on decorated classes.
        """
        assert has(C)

    def test_positive_empty(self):
        """
        Returns `True` on decorated classes even if there are no attributes.
        """

        @attr.s
        class D:
            pass

        assert has(D)

    def test_negative(self):
        """
        Returns `False` on non-decorated classes.
        """
        assert not has(object)

    def test_generics(self):
        """
        Works with generic classes.
        """
        T = TypeVar("T")

        @attr.define
        class A(Generic[T]):
            a: T

        assert has(A)

        assert has(A[str])
        # Verify twice, since there's caching going on.
        assert has(A[str])

    def test_generics_negative(self):
        """
        Returns `False` on non-decorated generic classes.
        """
        T = TypeVar("T")

        class A(Generic[T]):
            a: T

        assert not has(A)

        assert not has(A[str])
        # Verify twice, since there's caching going on.
        assert not has(A[str])


class TestAssoc:
    """
    Tests for `assoc`.
    """

    @given(slots=st.booleans(), frozen=st.booleans())
    def test_empty(self, slots, frozen):
        """
        Empty classes without changes get copied.
        """

        @attr.s(slots=slots, frozen=frozen)
        class C:
            pass

        i1 = C()
        i2 = assoc(i1)

        assert i1 is not i2
        assert i1 == i2

    @given(simple_classes())
    def test_no_changes(self, C):
        """
        No changes means a verbatim copy.
        """
        i1 = C()
        i2 = assoc(i1)

        assert i1 is not i2
        assert i1 == i2

    @given(simple_classes(), st.data())
    def test_change(self, C, data):
        """
        Changes work.
        """
        # Take the first attribute, and change it.
        assume(fields(C))  # Skip classes with no attributes.
        field_names = [a.name for a in fields(C)]
        original = C()
        chosen_names = data.draw(st.sets(st.sampled_from(field_names)))
        change_dict = {name: data.draw(st.integers()) for name in chosen_names}

        changed = assoc(original, **change_dict)

        for k, v in change_dict.items():
            assert getattr(changed, k) == v

    @given(simple_classes())
    def test_unknown(self, C):
        """
        Wanting to change an unknown attribute raises an
        AttrsAttributeNotFoundError.
        """
        # No generated class will have a four letter attribute.
        with pytest.raises(AttrsAttributeNotFoundError) as e:
            assoc(C(), aaaa=2)

        assert (f"aaaa is not an attrs attribute on {C!r}.",) == e.value.args

    def test_frozen(self):
        """
        Works on frozen classes.
        """

        @attr.s(frozen=True)
        class C:
            x = attr.ib()
            y = attr.ib()

        assert C(3, 2) == assoc(C(1, 2), x=3)


class TestEvolve:
    """
    Tests for `evolve`.
    """

    @given(slots=st.booleans(), frozen=st.booleans())
    def test_empty(self, slots, frozen):
        """
        Empty classes without changes get copied.
        """

        @attr.s(slots=slots, frozen=frozen)
        class C:
            pass

        i1 = C()
        i2 = evolve(i1)

        assert i1 is not i2
        assert i1 == i2

    @given(simple_classes())
    def test_no_changes(self, C):
        """
        No changes means a verbatim copy.
        """
        i1 = C()
        i2 = evolve(i1)

        assert i1 is not i2
        assert i1 == i2

    @given(simple_classes(), st.data())
    def test_change(self, C, data):
        """
        Changes work.
        """
        # Take the first attribute, and change it.
        assume(fields(C))  # Skip classes with no attributes.
        field_names = [a.name for a in fields(C)]
        original = C()
        chosen_names = data.draw(st.sets(st.sampled_from(field_names)))
        # We pay special attention to private attributes, they should behave
        # like in `__init__`.
        change_dict = {
            name.replace("_", ""): data.draw(st.integers())
            for name in chosen_names
        }
        changed = evolve(original, **change_dict)
        for name in chosen_names:
            assert getattr(changed, name) == change_dict[name.replace("_", "")]

    @given(simple_classes())
    def test_unknown(self, C):
        """
        Wanting to change an unknown attribute raises an
        AttrsAttributeNotFoundError.
        """
        # No generated class will have a four letter attribute.
        with pytest.raises(TypeError) as e:
            evolve(C(), aaaa=2)

        if hasattr(C, "__attrs_init__"):
            expected = (
                "__attrs_init__() got an unexpected keyword argument 'aaaa'"
            )
        else:
            expected = "__init__() got an unexpected keyword argument 'aaaa'"

        assert e.value.args[0].endswith(expected)

    def test_validator_failure(self):
        """
        TypeError isn't swallowed when validation fails within evolve.
        """

        @attr.s
        class C:
            a = attr.ib(validator=instance_of(int))

        with pytest.raises(TypeError) as e:
            evolve(C(a=1), a="some string")
        m = e.value.args[0]

        assert m.startswith("'a' must be <class 'int'>")

    def test_private(self):
        """
        evolve() acts as `__init__` with regards to private attributes.
        """

        @attr.s
        class C:
            _a = attr.ib()

        assert evolve(C(1), a=2)._a == 2

        with pytest.raises(TypeError):
            evolve(C(1), _a=2)

        with pytest.raises(TypeError):
            evolve(C(1), a=3, _a=2)

    def test_non_init_attrs(self):
        """
        evolve() handles `init=False` attributes.
        """

        @attr.s
        class C:
            a = attr.ib()
            b = attr.ib(init=False, default=0)

        assert evolve(C(1), a=2).a == 2

    def test_regression_attrs_classes(self):
        """
        evolve() can evolve fields that are instances of attrs classes.

        Regression test for #804
        """

        @attr.s
        class Cls1:
            param1 = attr.ib()

        @attr.s
        class Cls2:
            param2 = attr.ib()

        obj2a = Cls2(param2="a")
        obj2b = Cls2(param2="b")

        obj1a = Cls1(param1=obj2a)

        assert Cls1(param1=Cls2(param2="b")) == attr.evolve(
            obj1a, param1=obj2b
        )

    def test_dicts(self):
        """
        evolve() can replace an attrs class instance with a dict.

        See #806
        """

        @attr.s
        class Cls1:
            param1 = attr.ib()

        @attr.s
        class Cls2:
            param2 = attr.ib()

        obj2a = Cls2(param2="a")
        obj2b = {"foo": 42, "param2": 42}

        obj1a = Cls1(param1=obj2a)

        assert Cls1({"foo": 42, "param2": 42}) == attr.evolve(
            obj1a, param1=obj2b
        )

    def test_no_inst(self):
        """
        Missing inst argument raises a TypeError like Python would.
        """
        with pytest.raises(
            TypeError, match=r"evolve\(\) takes 1 positional argument"
        ):
            evolve(x=1)

    def test_too_many_pos_args(self):
        """
        More than one positional argument raises a TypeError like Python would.
        """
        with pytest.raises(
            TypeError,
            match=r"evolve\(\) takes 1 positional argument, but 2 were given",
        ):
            evolve(1, 2)

    def test_can_change_inst(self):
        """
        If the instance is passed by positional argument, a field named `inst`
        can be changed.
        """

        @attr.define
        class C:
            inst: int

        assert C(42) == evolve(C(23), inst=42)


from .utils import simple_class


class TestSimpleClass:
    """
    Tests for the testing helper function `make_class`.
    """

    def test_returns_class(self):
        """
        Returns a class object.
        """
        assert type is simple_class().__class__

    def test_returns_distinct_classes(self):
        """
        Each call returns a completely new class.
        """
        assert simple_class() is not simple_class()


# SPDX-License-Identifier: MIT

"""
Tests for `attr.converters`.
"""

import pickle

import pytest

import attr

from attr import Converter, Factory, attrib
from attr._compat import _AnnotationExtractor
from attr.converters import default_if_none, optional, pipe, to_bool


class TestConverter:
    @pytest.mark.parametrize("takes_self", [True, False])
    @pytest.mark.parametrize("takes_field", [True, False])
    def test_pickle(self, takes_self, takes_field):
        """
        Wrapped converters can be pickled.
        """
        c = Converter(int, takes_self=takes_self, takes_field=takes_field)

        new_c = pickle.loads(pickle.dumps(c))

        assert c == new_c
        assert takes_self == new_c.takes_self
        assert takes_field == new_c.takes_field
        assert c.__call__.__name__ == new_c.__call__.__name__

    @pytest.mark.parametrize(
        "scenario",
        [
            ((False, False), "__attr_converter_le_name(le_value)"),
            (
                (True, True),
                "__attr_converter_le_name(le_value, self, attr_dict['le_name'])",
            ),
            (
                (True, False),
                "__attr_converter_le_name(le_value, self)",
            ),
            (
                (False, True),
                "__attr_converter_le_name(le_value, attr_dict['le_name'])",
            ),
        ],
    )
    def test_fmt_converter_call(self, scenario):
        """
        _fmt_converter_call determines the arguments to the wrapped converter
        according to `takes_self` and `takes_field`.
        """
        (takes_self, takes_field), expect = scenario

        c = Converter(None, takes_self=takes_self, takes_field=takes_field)

        assert expect == c._fmt_converter_call("le_name", "le_value")

    def test_works_as_adapter(self):
        """
        Converter instances work as adapters and pass the correct arguments to
        the wrapped converter callable.
        """
        taken = None
        instance = object()
        field = object()

        def save_args(*args):
            nonlocal taken
            taken = args
            return args[0]

        Converter(save_args)(42, instance, field)

        assert (42,) == taken

        Converter(save_args, takes_self=True)(42, instance, field)

        assert (42, instance) == taken

        Converter(save_args, takes_field=True)(42, instance, field)

        assert (42, field) == taken

        Converter(save_args, takes_self=True, takes_field=True)(
            42, instance, field
        )

        assert (42, instance, field) == taken

    def test_annotations_if_last_in_pipe(self):
        """
        If the wrapped converter has annotations, they are copied to the
        Converter __call__.
        """

        def wrapped(_, __, ___) -> float:
            pass

        c = Converter(wrapped)

        assert float is c.__call__.__annotations__["return"]

        # Doesn't overwrite globally.

        c2 = Converter(int)

        assert float is c.__call__.__annotations__["return"]
        assert None is c2.__call__.__annotations__.get("return")

    def test_falsey_converter(self):
        """
        Passing a false-y instance still produces a valid converter.
        """

        class MyConv:
            def __bool__(self):
                return False

            def __call__(self, value):
                return value * 2

        @attr.s
        class C:
            a = attrib(converter=MyConv())

        c = C(21)
        assert 42 == c.a


class TestOptional:
    """
    Tests for `optional`.
    """

    def test_success_with_type(self):
        """
        Wrapped converter is used as usual if value is not None.
        """
        c = optional(int)

        assert c("42") == 42

    def test_success_with_none(self):
        """
        Nothing happens if None.
        """
        c = optional(int)

        assert c(None) is None

    def test_fail(self):
        """
        Propagates the underlying conversion error when conversion fails.
        """
        c = optional(int)

        with pytest.raises(ValueError):
            c("not_an_int")

    def test_converter_instance(self):
        """
        Works when passed a Converter instance as argument.
        """
        c = optional(Converter(to_bool))

        assert True is c("yes", None, None)


class TestDefaultIfNone:
    def test_missing_default(self):
        """
        Raises TypeError if neither default nor factory have been passed.
        """
        with pytest.raises(TypeError, match="Must pass either"):
            default_if_none()

    def test_too_many_defaults(self):
        """
        Raises TypeError if both default and factory are passed.
        """
        with pytest.raises(TypeError, match="but not both"):
            default_if_none(True, lambda: 42)

    def test_factory_takes_self(self):
        """
        Raises ValueError if passed Factory has takes_self=True.
        """
        with pytest.raises(ValueError, match="takes_self"):
            default_if_none(Factory(list, takes_self=True))

    @pytest.mark.parametrize("val", [1, 0, True, False, "foo", "", object()])
    def test_not_none(self, val):
        """
        If a non-None value is passed, it's handed down.
        """
        c = default_if_none("nope")

        assert val == c(val)

        c = default_if_none(factory=list)

        assert val == c(val)

    def test_none_value(self):
        """
        Default values are returned when a None is passed.
        """
        c = default_if_none(42)

        assert 42 == c(None)

    def test_none_factory(self):
        """
        Factories are used if None is passed.
        """
        c = default_if_none(factory=list)

        assert [] == c(None)

        c = default_if_none(default=Factory(list))

        assert [] == c(None)


class TestPipe:
    def test_success(self):
        """
        Succeeds if all wrapped converters succeed.
        """
        c = pipe(str, Converter(to_bool), bool)

        assert (
            True
            is c.converter("True", None, None)
            is c.converter(True, None, None)
        )

    def test_fail(self):
        """
        Fails if any wrapped converter fails.
        """
        c = pipe(str, to_bool)

        # First wrapped converter fails:
        with pytest.raises(ValueError):
            c(33)

        # Last wrapped converter fails:
        with pytest.raises(ValueError):
            c("33")

    def test_sugar(self):
        """
        `pipe(c1, c2, c3)` and `[c1, c2, c3]` are equivalent.
        """

        @attr.s
        class C:
            a1 = attrib(default="True", converter=pipe(str, to_bool, bool))
            a2 = attrib(default=True, converter=[str, to_bool, bool])

        c = C()
        assert True is c.a1 is c.a2

    def test_empty(self):
        """
        Empty pipe returns same value.
        """
        o = object()

        assert o is pipe()(o)

    def test_wrapped_annotation(self):
        """
        The return type of the wrapped converter is copied into its __call__
        and ultimately into pipe's wrapped converter.
        """

        def last(value) -> bool:
            return bool(value)

        @attr.s
        class C:
            x = attr.ib(converter=[Converter(int), Converter(last)])

        i = C(5)

        assert True is i.x
        assert (
            bool
            is _AnnotationExtractor(
                attr.fields(C).x.converter.__call__
            ).get_return_type()
        )


class TestOptionalPipe:
    def test_optional(self):
        """
        Nothing happens if None.
        """
        c = optional(pipe(str, Converter(to_bool), bool))

        assert None is c.converter(None, None, None)

    def test_pipe(self):
        """
        A value is given, run it through all wrapped converters.
        """
        c = optional(pipe(str, Converter(to_bool), bool))

        assert (
            True
            is c.converter("True", None, None)
            is c.converter(True, None, None)
        )

    def test_instance(self):
        """
        Should work when set as an attrib.
        """

        @attr.s
        class C:
            x = attrib(
                converter=optional(pipe(str, Converter(to_bool), bool)),
                default=None,
            )

        c1 = C()

        assert None is c1.x

        c2 = C("True")

        assert True is c2.x


class TestToBool:
    def test_unhashable(self):
        """
        Fails if value is unhashable.
        """
        with pytest.raises(ValueError, match="Cannot convert value to bool"):
            to_bool([])

    def test_truthy(self):
        """
        Fails if truthy values are incorrectly converted.
        """
        assert to_bool("t")
        assert to_bool("yes")
        assert to_bool("on")

    def test_falsy(self):
        """
        Fails if falsy values are incorrectly converted.
        """
        assert not to_bool("f")
        assert not to_bool("no")
        assert not to_bool("off")


# SPDX-License-Identifier: MIT


from attr import *  # noqa: F403


# This is imported by test_import::test_from_attr_import_star; this must
# be done indirectly because importing * is only allowed on module level,
# so can't be done inside a test.


# SPDX-License-Identifier: MIT


from importlib import metadata

import pytest

import attr
import attrs


@pytest.fixture(name="mod", params=(attr, attrs))
def _mod(request):
    return request.param


class TestLegacyMetadataHack:
    def test_version(self, mod, recwarn):
        """
        __version__ returns the correct version and doesn't warn.
        """
        assert metadata.version("attrs") == mod.__version__

        assert [] == recwarn.list

    def test_does_not_exist(self, mod):
        """
        Asking for unsupported dunders raises an AttributeError.
        """
        with pytest.raises(
            AttributeError,
            match=f"module {mod.__name__} has no attribute __yolo__",
        ):
            mod.__yolo__

    def test_version_info(self, recwarn, mod):
        """
        ___version_info__ is not deprecated, therefore doesn't raise a warning
        and parses correctly.
        """
        assert isinstance(mod.__version_info__, attr.VersionInfo)
        assert [] == recwarn.list


# SPDX-License-Identifier: MIT

"""
Testing strategies for Hypothesis-based tests.
"""

import functools
import keyword
import string

from collections import OrderedDict

from hypothesis import strategies as st

import attr

from .utils import make_class


optional_bool = st.one_of(st.none(), st.booleans())


def gen_attr_names():
    """
    Generate names for attributes, 'a'...'z', then 'aa'...'zz'.

    ~702 different attribute names should be enough in practice.

    Some short strings (such as 'as') are keywords, so we skip them.
    """
    lc = string.ascii_lowercase
    yield from lc
    for outer in lc:
        for inner in lc:
            res = outer + inner
            if keyword.iskeyword(res):
                continue
            yield outer + inner


def maybe_underscore_prefix(source):
    """
    A generator to sometimes prepend an underscore.
    """
    to_underscore = False
    for val in source:
        yield val if not to_underscore else "_" + val
        to_underscore = not to_underscore


@st.composite
def _create_hyp_nested_strategy(draw, simple_class_strategy):
    """
    Create a recursive attrs class.

    Given a strategy for building (simpler) classes, create and return
    a strategy for building classes that have as an attribute: either just
    the simpler class, a list of simpler classes, a tuple of simpler classes,
    an ordered dict or a dict mapping the string "cls" to a simpler class.
    """
    cls = draw(simple_class_strategy)
    factories = [
        cls,
        lambda: [cls()],
        lambda: (cls(),),
        lambda: {"cls": cls()},
        lambda: OrderedDict([("cls", cls())]),
    ]
    factory = draw(st.sampled_from(factories))
    attrs = [*draw(list_of_attrs), attr.ib(default=attr.Factory(factory))]
    return make_class("HypClass", dict(zip(gen_attr_names(), attrs)))


bare_attrs = st.builds(attr.ib, default=st.none())
int_attrs = st.integers().map(lambda i: attr.ib(default=i))
str_attrs = st.text().map(lambda s: attr.ib(default=s))
float_attrs = st.floats(allow_nan=False).map(lambda f: attr.ib(default=f))
dict_attrs = st.dictionaries(keys=st.text(), values=st.integers()).map(
    lambda d: attr.ib(default=d)
)

simple_attrs_without_metadata = (
    bare_attrs | int_attrs | str_attrs | float_attrs | dict_attrs
)


@st.composite
def simple_attrs_with_metadata(draw):
    """
    Create a simple attribute with arbitrary metadata.
    """
    c_attr = draw(simple_attrs)
    keys = st.booleans() | st.binary() | st.integers() | st.text()
    vals = st.booleans() | st.binary() | st.integers() | st.text()
    metadata = draw(
        st.dictionaries(keys=keys, values=vals, min_size=1, max_size=3)
    )

    return attr.ib(
        default=c_attr._default,
        validator=c_attr._validator,
        repr=c_attr.repr,
        eq=c_attr.eq,
        order=c_attr.order,
        hash=c_attr.hash,
        init=c_attr.init,
        metadata=metadata,
        type=None,
        converter=c_attr.converter,
    )


simple_attrs = simple_attrs_without_metadata | simple_attrs_with_metadata()


# Python functions support up to 255 arguments.
list_of_attrs = st.lists(simple_attrs, max_size=3)


@st.composite
def simple_classes(
    draw,
    slots=None,
    frozen=None,
    weakref_slot=None,
    private_attrs=None,
    cached_property=None,
):
    """
    A strategy that generates classes with default non-attr attributes.

    For example, this strategy might generate a class such as:

    @attr.s(slots=True, frozen=True, weakref_slot=True)
    class HypClass:
        a = attr.ib(default=1)
        _b = attr.ib(default=None)
        c = attr.ib(default='text')
        _d = attr.ib(default=1.0)
        c = attr.ib(default={'t': 1})

    By default, all combinations of slots, frozen, and weakref_slot classes
    will be generated.  If `slots=True` is passed in, only slotted classes will
    be generated, and if `slots=False` is passed in, no slotted classes will be
    generated. The same applies to `frozen` and `weakref_slot`.

    By default, some attributes will be private (those prefixed with an
    underscore). If `private_attrs=True` is passed in, all attributes will be
    private, and if `private_attrs=False`, no attributes will be private.
    """
    attrs = draw(list_of_attrs)
    frozen_flag = draw(st.booleans())
    slots_flag = draw(st.booleans())
    weakref_flag = draw(st.booleans())

    if private_attrs is None:
        attr_names = maybe_underscore_prefix(gen_attr_names())
    elif private_attrs is True:
        attr_names = ("_" + n for n in gen_attr_names())
    elif private_attrs is False:
        attr_names = gen_attr_names()

    cls_dict = dict(zip(attr_names, attrs))
    pre_init_flag = draw(st.booleans())
    post_init_flag = draw(st.booleans())
    init_flag = draw(st.booleans())
    cached_property_flag = draw(st.booleans())

    if pre_init_flag:

        def pre_init(self):
            pass

        cls_dict["__attrs_pre_init__"] = pre_init

    if post_init_flag:

        def post_init(self):
            pass

        cls_dict["__attrs_post_init__"] = post_init

    if not init_flag:

        def init(self, *args, **kwargs):
            self.__attrs_init__(*args, **kwargs)

        cls_dict["__init__"] = init

    bases = (object,)
    if cached_property or (cached_property is None and cached_property_flag):

        class BaseWithCachedProperty:
            @functools.cached_property
            def _cached_property(self) -> int:
                return 1

        bases = (BaseWithCachedProperty,)

    return make_class(
        "HypClass",
        cls_dict,
        bases=bases,
        slots=slots_flag if slots is None else slots,
        frozen=frozen_flag if frozen is None else frozen,
        weakref_slot=weakref_flag if weakref_slot is None else weakref_slot,
        init=init_flag,
    )


# st.recursive works by taking a base strategy (in this case, simple_classes)
# and a special function.  This function receives a strategy, and returns
# another strategy (building on top of the base strategy).
nested_classes = st.recursive(
    simple_classes(), _create_hyp_nested_strategy, max_leaves=3
)


# SPDX-License-Identifier: MIT

"""
End-to-end tests.
"""

import copy
import inspect
import pickle

from copy import deepcopy

import pytest

from hypothesis import given
from hypothesis.strategies import booleans

import attr

from attr._compat import PY_3_13_PLUS
from attr._make import NOTHING, Attribute
from attr.exceptions import FrozenInstanceError


@attr.s
class C1:
    x = attr.ib(validator=attr.validators.instance_of(int))
    y = attr.ib()


@attr.s(slots=True)
class C1Slots:
    x = attr.ib(validator=attr.validators.instance_of(int))
    y = attr.ib()


foo = None


@attr.s()
class C2:
    x = attr.ib(default=foo)
    y = attr.ib(default=attr.Factory(list))


@attr.s(slots=True)
class C2Slots:
    x = attr.ib(default=foo)
    y = attr.ib(default=attr.Factory(list))


@attr.s
class Base:
    x = attr.ib()

    def meth(self):
        return self.x


@attr.s(slots=True)
class BaseSlots:
    x = attr.ib()

    def meth(self):
        return self.x


@attr.s
class Sub(Base):
    y = attr.ib()


@attr.s(slots=True)
class SubSlots(BaseSlots):
    y = attr.ib()


@attr.s(frozen=True, slots=True)
class Frozen:
    x = attr.ib()


@attr.s
class SubFrozen(Frozen):
    y = attr.ib()


@attr.s(frozen=True, slots=False)
class FrozenNoSlots:
    x = attr.ib()


class Meta(type):
    pass


@attr.s
class WithMeta(metaclass=Meta):
    pass


@attr.s(slots=True)
class WithMetaSlots(metaclass=Meta):
    pass


FromMakeClass = attr.make_class("FromMakeClass", ["x"])


class TestFunctional:
    """
    Functional tests.
    """

    @pytest.mark.parametrize("cls", [C2, C2Slots])
    def test_fields(self, cls):
        """
        `attr.fields` works.
        """
        assert (
            Attribute(
                name="x",
                alias="x",
                default=foo,
                validator=None,
                repr=True,
                cmp=None,
                eq=True,
                order=True,
                hash=None,
                init=True,
                inherited=False,
            ),
            Attribute(
                name="y",
                alias="y",
                default=attr.Factory(list),
                validator=None,
                repr=True,
                cmp=None,
                eq=True,
                order=True,
                hash=None,
                init=True,
                inherited=False,
            ),
        ) == attr.fields(cls)

    @pytest.mark.parametrize("cls", [C1, C1Slots])
    def test_asdict(self, cls):
        """
        `attr.asdict` works.
        """
        assert {"x": 1, "y": 2} == attr.asdict(cls(x=1, y=2))

    @pytest.mark.parametrize("cls", [C1, C1Slots])
    def test_validator(self, cls):
        """
        `instance_of` raises `TypeError` on type mismatch.
        """
        with pytest.raises(TypeError) as e:
            cls("1", 2)

        # Using C1 explicitly, since slotted classes don't support this.
        assert (
            "'x' must be <class 'int'> (got '1' that is a <class 'str'>).",
            attr.fields(C1).x,
            int,
            "1",
        ) == e.value.args

    @given(booleans())
    def test_renaming(self, slots):
        """
        Private members are renamed but only in `__init__`.
        """

        @attr.s(slots=slots)
        class C3:
            _x = attr.ib()

        assert "C3(_x=1)" == repr(C3(x=1))

    @given(booleans(), booleans())
    def test_programmatic(self, slots, frozen):
        """
        `attr.make_class` works.
        """
        PC = attr.make_class("PC", ["a", "b"], slots=slots, frozen=frozen)

        assert (
            Attribute(
                name="a",
                alias="a",
                default=NOTHING,
                validator=None,
                repr=True,
                cmp=None,
                eq=True,
                order=True,
                hash=None,
                init=True,
                inherited=False,
            ),
            Attribute(
                name="b",
                alias="b",
                default=NOTHING,
                validator=None,
                repr=True,
                cmp=None,
                eq=True,
                order=True,
                hash=None,
                init=True,
                inherited=False,
            ),
        ) == attr.fields(PC)

    @pytest.mark.parametrize("cls", [Sub, SubSlots])
    def test_subclassing_with_extra_attrs(self, cls):
        """
        Subclassing (where the subclass has extra attrs) does what you'd hope
        for.
        """
        obj = object()
        i = cls(x=obj, y=2)
        assert i.x is i.meth() is obj
        assert i.y == 2
        if cls is Sub:
            assert f"Sub(x={obj}, y=2)" == repr(i)
        else:
            assert f"SubSlots(x={obj}, y=2)" == repr(i)

    @pytest.mark.parametrize("base", [Base, BaseSlots])
    def test_subclass_without_extra_attrs(self, base):
        """
        Subclassing (where the subclass does not have extra attrs) still
        behaves the same as a subclass with extra attrs.
        """

        class Sub2(base):
            pass

        obj = object()
        i = Sub2(x=obj)
        assert i.x is i.meth() is obj
        assert f"Sub2(x={obj})" == repr(i)

    @pytest.mark.parametrize(
        "frozen_class",
        [
            Frozen,  # has slots=True
            attr.make_class("FrozenToo", ["x"], slots=False, frozen=True),
        ],
    )
    def test_frozen_instance(self, frozen_class):
        """
        Frozen instances can't be modified (easily).
        """
        frozen = frozen_class(1)

        with pytest.raises(FrozenInstanceError) as e:
            frozen.x = 2

        with pytest.raises(FrozenInstanceError) as e:
            del frozen.x

        assert e.value.args[0] == "can't set attribute"
        assert 1 == frozen.x

    @pytest.mark.parametrize(
        "cls",
        [
            C1,
            C1Slots,
            C2,
            C2Slots,
            Base,
            BaseSlots,
            Sub,
            SubSlots,
            Frozen,
            FrozenNoSlots,
            FromMakeClass,
        ],
    )
    @pytest.mark.parametrize("protocol", range(2, pickle.HIGHEST_PROTOCOL + 1))
    def test_pickle_attributes(self, cls, protocol):
        """
        Pickling/un-pickling of Attribute instances works.
        """
        for attribute in attr.fields(cls):
            assert attribute == pickle.loads(pickle.dumps(attribute, protocol))

    @pytest.mark.parametrize(
        "cls",
        [
            C1,
            C1Slots,
            C2,
            C2Slots,
            Base,
            BaseSlots,
            Sub,
            SubSlots,
            Frozen,
            FrozenNoSlots,
            FromMakeClass,
        ],
    )
    @pytest.mark.parametrize("protocol", range(2, pickle.HIGHEST_PROTOCOL + 1))
    def test_pickle_object(self, cls, protocol):
        """
        Pickle object serialization works on all kinds of attrs classes.
        """
        obj = cls(123, 456) if len(attr.fields(cls)) == 2 else cls(123)

        assert repr(obj) == repr(pickle.loads(pickle.dumps(obj, protocol)))

    def test_subclassing_frozen_gives_frozen(self):
        """
        The frozen-ness of classes is inherited.  Subclasses of frozen classes
        are also frozen and can be instantiated.
        """
        i = SubFrozen("foo", "bar")

        assert i.x == "foo"
        assert i.y == "bar"

        with pytest.raises(FrozenInstanceError):
            i.x = "baz"

    @pytest.mark.parametrize("cls", [WithMeta, WithMetaSlots])
    def test_metaclass_preserved(self, cls):
        """
        Metaclass data is preserved.
        """
        assert Meta is type(cls)

    def test_default_decorator(self):
        """
        Default decorator sets the default and the respective method gets
        called.
        """

        @attr.s
        class C:
            x = attr.ib(default=1)
            y = attr.ib()

            @y.default
            def compute(self):
                return self.x + 1

        assert C(1, 2) == C()

    @pytest.mark.parametrize("weakref_slot", [True, False])
    def test_attrib_overwrite(self, slots, frozen, weakref_slot):
        """
        Subclasses can overwrite attributes of their base class.
        """

        @attr.s(slots=slots, frozen=frozen, weakref_slot=weakref_slot)
        class SubOverwrite(Base):
            x = attr.ib(default=attr.Factory(list))

        assert SubOverwrite([]) == SubOverwrite()

    def test_dict_patch_class(self):
        """
        dict-classes are never replaced.
        """

        class C:
            x = attr.ib()

        C_new = attr.s(C)

        assert C_new is C

    def test_hash_by_id(self):
        """
        With dict classes, hashing by ID is active for hash=False.  This is
        incorrect behavior but we have to retain it for
        backwards-compatibility.
        """

        @attr.s(unsafe_hash=False)
        class HashByIDBackwardCompat:
            x = attr.ib()

        assert hash(HashByIDBackwardCompat(1)) != hash(
            HashByIDBackwardCompat(1)
        )

        @attr.s(unsafe_hash=False, eq=False)
        class HashByID:
            x = attr.ib()

        assert hash(HashByID(1)) != hash(HashByID(1))

        @attr.s(unsafe_hash=True)
        class HashByValues:
            x = attr.ib()

        assert hash(HashByValues(1)) == hash(HashByValues(1))

    def test_handles_different_defaults(self):
        """
        Unhashable defaults + subclassing values work.
        """

        @attr.s
        class Unhashable:
            pass

        @attr.s
        class C:
            x = attr.ib(default=Unhashable())

        @attr.s
        class D(C):
            pass

    def test_unsafe_hash_false_eq_false(self, slots):
        """
        unsafe_hash=False and eq=False make a class hashable by ID.
        """

        @attr.s(unsafe_hash=False, eq=False, slots=slots)
        class C:
            pass

        assert hash(C()) != hash(C())

    def test_hash_deprecated(self):
        """
        Using the hash argument is deprecated.
        """

    def test_eq_false(self, slots):
        """
        eq=False makes a class hashable by ID.
        """

        @attr.s(eq=False, slots=slots)
        class C:
            pass

        # Ensure both objects live long enough such that their ids/hashes
        # can't be recycled. Thanks to Ask Hjorth Larsen for pointing that
        # out.
        c1 = C()
        c2 = C()

        assert hash(c1) != hash(c2)

    def test_overwrite_base(self):
        """
        Base classes can overwrite each other and the attributes are added
        in the order they are defined.
        """

        @attr.s
        class C:
            c = attr.ib(default=100)
            x = attr.ib(default=1)
            b = attr.ib(default=23)

        @attr.s
        class D(C):
            a = attr.ib(default=42)
            x = attr.ib(default=2)
            d = attr.ib(default=3.14)

        @attr.s
        class E(D):
            y = attr.ib(default=3)
            z = attr.ib(default=4)

        assert "E(c=100, b=23, a=42, x=2, d=3.14, y=3, z=4)" == repr(E())

    @pytest.mark.parametrize("base_slots", [True, False])
    @pytest.mark.parametrize("sub_slots", [True, False])
    @pytest.mark.parametrize("base_frozen", [True, False])
    @pytest.mark.parametrize("sub_frozen", [True, False])
    @pytest.mark.parametrize("base_weakref_slot", [True, False])
    @pytest.mark.parametrize("sub_weakref_slot", [True, False])
    @pytest.mark.parametrize("base_converter", [True, False])
    @pytest.mark.parametrize("sub_converter", [True, False])
    def test_frozen_slots_combo(
        self,
        base_slots,
        sub_slots,
        base_frozen,
        sub_frozen,
        base_weakref_slot,
        sub_weakref_slot,
        base_converter,
        sub_converter,
    ):
        """
        A class with a single attribute, inheriting from another class
        with a single attribute.
        """

        @attr.s(
            frozen=base_frozen,
            slots=base_slots,
            weakref_slot=base_weakref_slot,
        )
        class Base:
            a = attr.ib(converter=int if base_converter else None)

        @attr.s(
            frozen=sub_frozen, slots=sub_slots, weakref_slot=sub_weakref_slot
        )
        class Sub(Base):
            b = attr.ib(converter=int if sub_converter else None)

        i = Sub("1", "2")

        assert i.a == (1 if base_converter else "1")
        assert i.b == (2 if sub_converter else "2")

        if base_frozen or sub_frozen:
            with pytest.raises(FrozenInstanceError):
                i.a = "2"

            with pytest.raises(FrozenInstanceError):
                i.b = "3"

    def test_tuple_class_aliasing(self):
        """
        itemgetter and property are legal attribute names.
        """

        @attr.s
        class C:
            property = attr.ib()
            itemgetter = attr.ib()
            x = attr.ib()

        assert "property" == attr.fields(C).property.name
        assert "itemgetter" == attr.fields(C).itemgetter.name
        assert "x" == attr.fields(C).x.name

    def test_auto_exc(self, slots, frozen):
        """
        Classes with auto_exc=True have a Exception-style __str__, compare and
        hash by id, and store the fields additionally in self.args.
        """

        @attr.s(auto_exc=True, slots=slots, frozen=frozen)
        class FooError(Exception):
            x = attr.ib()
            y = attr.ib(init=False, default=42)
            z = attr.ib(init=False)
            a = attr.ib()

        FooErrorMade = attr.make_class(
            "FooErrorMade",
            bases=(Exception,),
            attrs={
                "x": attr.ib(),
                "y": attr.ib(init=False, default=42),
                "z": attr.ib(init=False),
                "a": attr.ib(),
            },
            auto_exc=True,
            slots=slots,
            frozen=frozen,
        )

        assert FooError(1, "foo") != FooError(1, "foo")
        assert FooErrorMade(1, "foo") != FooErrorMade(1, "foo")

        for cls in (FooError, FooErrorMade):
            with pytest.raises(cls) as ei1:
                raise cls(1, "foo")

            with pytest.raises(cls) as ei2:
                raise cls(1, "foo")

            e1 = ei1.value
            e2 = ei2.value

            assert e1 is e1
            assert e1 == e1
            assert e2 == e2
            assert e1 != e2
            assert "(1, 'foo')" == str(e1) == str(e2)
            assert (1, "foo") == e1.args == e2.args

            hash(e1) == hash(e1)
            hash(e2) == hash(e2)

            if not frozen:
                deepcopy(e1)
                deepcopy(e2)

    def test_auto_exc_one_attrib(self, slots, frozen):
        """
        Having one attribute works with auto_exc=True.

        Easy to get wrong with tuple literals.
        """

        @attr.s(auto_exc=True, slots=slots, frozen=frozen)
        class FooError(Exception):
            x = attr.ib()

        FooError(1)

    def test_eq_only(self, slots, frozen):
        """
        Classes with order=False cannot be ordered.
        """

        @attr.s(eq=True, order=False, slots=slots, frozen=frozen)
        class C:
            x = attr.ib()

        possible_errors = (
            "unorderable types: C() < C()",
            "'<' not supported between instances of 'C' and 'C'",
            "unorderable types: C < C",  # old PyPy 3
        )

        with pytest.raises(TypeError) as ei:
            C(5) < C(6)

        assert ei.value.args[0] in possible_errors

    @pytest.mark.parametrize("cmp", [True, False])
    def test_attrib_cmp_shortcut(self, slots, cmp):
        """
        Setting cmp on `attr.ib`s sets both eq and order.
        """

        @attr.s(slots=slots)
        class C:
            x = attr.ib(cmp=cmp)

        assert cmp is attr.fields(C).x.eq
        assert cmp is attr.fields(C).x.order

    def test_no_setattr_if_validate_without_validators(self, slots):
        """
        If a class has on_setattr=attr.setters.validate (former default in NG
        APIs) but sets no validators, don't use the (slower) setattr in
        __init__.

        Regression test for #816.
        """

        @attr.s(on_setattr=attr.setters.validate, slots=slots)
        class C:
            x = attr.ib()

        @attr.s(on_setattr=attr.setters.validate, slots=slots)
        class D(C):
            y = attr.ib()

        src = inspect.getsource(D.__init__)

        assert "setattr" not in src
        assert "self.x = x" in src
        assert "self.y = y" in src
        assert object.__setattr__ == D.__setattr__

    def test_no_setattr_if_convert_without_converters(self, slots):
        """
        If a class has on_setattr=attr.setters.convert but sets no validators,
        don't use the (slower) setattr in __init__.
        """

        @attr.s(on_setattr=attr.setters.convert, slots=slots)
        class C:
            x = attr.ib()

        @attr.s(on_setattr=attr.setters.convert, slots=slots)
        class D(C):
            y = attr.ib()

        src = inspect.getsource(D.__init__)

        assert "setattr" not in src
        assert "self.x = x" in src
        assert "self.y = y" in src
        assert object.__setattr__ == D.__setattr__

    def test_no_setattr_with_ng_defaults(self, slots):
        """
        If a class has the NG default on_setattr=[convert, validate] but sets
        no validators or converters, don't use the (slower) setattr in
        __init__.
        """

        @attr.define(slots=slots)
        class C:
            x = attr.ib()

        src = inspect.getsource(C.__init__)

        assert "setattr" not in src
        assert "self.x = x" in src
        assert object.__setattr__ == C.__setattr__

        @attr.define(slots=slots)
        class D(C):
            y = attr.ib()

        src = inspect.getsource(D.__init__)

        assert "setattr" not in src
        assert "self.x = x" in src
        assert "self.y = y" in src
        assert object.__setattr__ == D.__setattr__

    def test_on_setattr_detect_inherited_validators(self):
        """
        _make_init detects the presence of a validator even if the field is
        inherited.
        """

        @attr.s(on_setattr=attr.setters.validate)
        class C:
            x = attr.ib(validator=42)

        @attr.s(on_setattr=attr.setters.validate)
        class D(C):
            y = attr.ib()

        src = inspect.getsource(D.__init__)

        assert "_setattr = _cached_setattr_get(self)" in src
        assert "_setattr('x', x)" in src
        assert "_setattr('y', y)" in src
        assert object.__setattr__ != D.__setattr__

    def test_unsafe_hash(self, slots):
        """
        attr.s(unsafe_hash=True) makes a class hashable.
        """

        @attr.s(slots=slots, unsafe_hash=True)
        class Hashable:
            pass

        assert hash(Hashable())

    def test_init_subclass(self, slots):
        """
        __attrs_init_subclass__ is called on subclasses.
        """
        REGISTRY = []

        @attr.s(slots=slots)
        class Base:
            @classmethod
            def __attrs_init_subclass__(cls):
                REGISTRY.append(cls)

        @attr.s(slots=slots)
        class ToRegister(Base):
            pass

        assert [ToRegister] == REGISTRY


@pytest.mark.skipif(not PY_3_13_PLUS, reason="requires Python 3.13+")
class TestReplace:
    def test_replaces(self):
        """
        copy.replace() is added by default and works like `attrs.evolve`.
        """
        inst = C1(1, 2)

        assert C1(1, 42) == copy.replace(inst, y=42)
        assert C1(42, 2) == copy.replace(inst, x=42)

    def test_already_has_one(self):
        """
        If the object already has a __replace__, it's left alone.
        """
        sentinel = object()

        @attr.s
        class C:
            x = attr.ib()

            __replace__ = sentinel

        assert sentinel == C.__replace__

    def test_invalid_field_name(self):
        """
        Invalid field names raise a TypeError.

        This is consistent with dataclasses.
        """
        inst = C1(1, 2)

        with pytest.raises(TypeError):
            copy.replace(inst, z=42)


# SPDX-License-Identifier: MIT


# SPDX-License-Identifier: MIT

"""
Tests for compatibility against other Python modules.
"""

import pytest

from hypothesis import given

from .strategies import simple_classes


cloudpickle = pytest.importorskip("cloudpickle")


class TestCloudpickleCompat:
    """
    Tests for compatibility with ``cloudpickle``.
    """

    @given(simple_classes(cached_property=False))
    def test_repr(self, cls):
        """
        attrs instances can be pickled and un-pickled with cloudpickle.
        """
        inst = cls()
        # Exact values aren't a concern so long as neither direction
        # raises an exception.
        pkl = cloudpickle.dumps(inst)
        cloudpickle.loads(pkl)


# SPDX-License-Identifier: MIT


import pickle

import pytest

import attr

from attr import setters
from attr.exceptions import FrozenAttributeError
from attr.validators import instance_of, matches_re


@attr.s(frozen=True)
class Frozen:
    x = attr.ib()


@attr.s
class WithOnSetAttrHook:
    x = attr.ib(on_setattr=lambda *args: None)


class TestSetAttr:
    def test_change(self):
        """
        The return value of a hook overwrites the value. But they are not run
        on __init__.
        """

        def hook(*a, **kw):
            return "hooked!"

        @attr.s
        class Hooked:
            x = attr.ib(on_setattr=hook)
            y = attr.ib()

        h = Hooked("x", "y")

        assert "x" == h.x
        assert "y" == h.y

        h.x = "xxx"
        h.y = "yyy"

        assert "yyy" == h.y
        assert "hooked!" == h.x

    def test_frozen_attribute(self):
        """
        Frozen attributes raise FrozenAttributeError, others are not affected.
        """

        @attr.s
        class PartiallyFrozen:
            x = attr.ib(on_setattr=setters.frozen)
            y = attr.ib()

        pf = PartiallyFrozen("x", "y")

        pf.y = "yyy"

        assert "yyy" == pf.y

        with pytest.raises(FrozenAttributeError):
            pf.x = "xxx"

        assert "x" == pf.x

    @pytest.mark.parametrize(
        "on_setattr",
        [setters.validate, [setters.validate], setters.pipe(setters.validate)],
    )
    def test_validator(self, on_setattr):
        """
        Validators are run and they don't alter the value.
        """

        @attr.s(on_setattr=on_setattr)
        class ValidatedAttribute:
            x = attr.ib()
            y = attr.ib(validator=[instance_of(str), matches_re("foo.*qux")])

        va = ValidatedAttribute(42, "foobarqux")

        with pytest.raises(TypeError) as ei:
            va.y = 42

        assert "foobarqux" == va.y

        assert ei.value.args[0].startswith("'y' must be <")

        with pytest.raises(ValueError) as ei:
            va.y = "quxbarfoo"

        assert ei.value.args[0].startswith("'y' must match regex '")

        assert "foobarqux" == va.y

        va.y = "foobazqux"

        assert "foobazqux" == va.y

    def test_pipe(self):
        """
        Multiple hooks are possible, in that case the last return value is
        used. They can be supplied using the pipe functions or by passing a
        list to on_setattr.
        """
        taken = None

        def takes_all(val, instance, attrib):
            nonlocal taken
            taken = val, instance, attrib

            return val

        s = [setters.convert, lambda _, __, nv: nv + 1]

        @attr.s
        class Piped:
            x1 = attr.ib(
                converter=[
                    attr.Converter(
                        takes_all, takes_field=True, takes_self=True
                    ),
                    int,
                ],
                on_setattr=setters.pipe(*s),
            )
            x2 = attr.ib(converter=int, on_setattr=s)

        p = Piped("41", "22")

        assert ("41", p) == taken[:-1]
        assert "x1" == taken[-1].name

        assert 41 == p.x1
        assert 22 == p.x2

        p.x1 = "41"
        p.x2 = "22"

        assert 42 == p.x1
        assert 23 == p.x2

    def test_make_class(self):
        """
        on_setattr of make_class gets forwarded.
        """
        C = attr.make_class("C", {"x": attr.ib()}, on_setattr=setters.frozen)

        c = C(1)

        with pytest.raises(FrozenAttributeError):
            c.x = 2

    def test_no_validator_no_converter(self):
        """
        validate and convert tolerate missing validators and converters.
        """

        @attr.s(on_setattr=[setters.convert, setters.validate])
        class C:
            x = attr.ib()

        c = C(1)

        c.x = 2

        assert 2 == c.x

    def test_validate_respects_run_validators_config(self):
        """
        If run validators is off, validate doesn't run them.
        """

        @attr.s(on_setattr=setters.validate)
        class C:
            x = attr.ib(validator=attr.validators.instance_of(int))

        c = C(1)

        attr.set_run_validators(False)

        c.x = "1"

        assert "1" == c.x

        attr.set_run_validators(True)

        with pytest.raises(TypeError) as ei:
            c.x = "1"

        assert ei.value.args[0].startswith("'x' must be <")

    def test_frozen_on_setattr_class_is_caught(self):
        """
        @attr.s(on_setattr=X, frozen=True) raises an ValueError.
        """
        with pytest.raises(ValueError) as ei:

            @attr.s(frozen=True, on_setattr=setters.validate)
            class C:
                x = attr.ib()

        assert "Frozen classes can't use on_setattr." == ei.value.args[0]

    def test_frozen_on_setattr_attribute_is_caught(self):
        """
        attr.ib(on_setattr=X) on a frozen class raises an ValueError.
        """

        with pytest.raises(ValueError) as ei:

            @attr.s(frozen=True)
            class C:
                x = attr.ib(on_setattr=setters.validate)

        assert "Frozen classes can't use on_setattr." == ei.value.args[0]

    def test_setattr_reset_if_no_custom_setattr(self, slots):
        """
        If a class with an active setattr is subclassed and no new setattr
        is generated, the __setattr__ is set to object.__setattr__.

        We do the double test because of Python 2.
        """

        def boom(*args):
            pytest.fail("Must not be called.")

        @attr.s
        class Hooked:
            x = attr.ib(on_setattr=boom)

        @attr.s(slots=slots)
        class NoHook(WithOnSetAttrHook):
            x = attr.ib()

        assert NoHook.__setattr__ == object.__setattr__
        assert 1 == NoHook(1).x
        assert Hooked.__attrs_own_setattr__
        assert not NoHook.__attrs_own_setattr__
        assert WithOnSetAttrHook.__attrs_own_setattr__

    def test_setattr_inherited_do_not_reset(self, slots):
        """
        If we inherit a __setattr__ that has been written by the user, we must
        not reset it unless necessary.
        """

        class A:
            """
            Not an attrs class on purpose to prevent accidental resets that
            would render the asserts meaningless.
            """

            def __setattr__(self, *args):
                pass

        @attr.s(slots=slots)
        class B(A):
            pass

        assert B.__setattr__ == A.__setattr__

        @attr.s(slots=slots)
        class C(B):
            pass

        assert C.__setattr__ == A.__setattr__

    def test_pickling_retains_attrs_own(self, slots):
        """
        Pickling/Unpickling does not lose ownership information about
        __setattr__.
        """
        i = WithOnSetAttrHook(1)

        assert True is i.__attrs_own_setattr__

        i2 = pickle.loads(pickle.dumps(i))

        assert True is i2.__attrs_own_setattr__

        WOSAH = pickle.loads(pickle.dumps(WithOnSetAttrHook))

        assert True is WOSAH.__attrs_own_setattr__

    def test_slotted_class_can_have_custom_setattr(self):
        """
        A slotted class can define a custom setattr and it doesn't get
        overwritten.

        Regression test for #680.
        """

        @attr.s(slots=True)
        class A:
            def __setattr__(self, key, value):
                raise SystemError

        with pytest.raises(SystemError):
            A().x = 1

    @pytest.mark.xfail(raises=attr.exceptions.FrozenAttributeError)
    def test_slotted_confused(self):
        """
        If we have a in-between non-attrs class, setattr reset detection
        should still work, but currently doesn't.

        It works with dict classes because we can look the finished class and
        patch it.  With slotted classes we have to deduce it ourselves.
        """

        @attr.s(slots=True)
        class A:
            x = attr.ib(on_setattr=setters.frozen)

        class B(A):
            pass

        @attr.s(slots=True)
        class C(B):
            x = attr.ib()

        C(1).x = 2

    def test_setattr_auto_detect_if_no_custom_setattr(self, slots):
        """
        It's possible to remove the on_setattr hook from an attribute and
        therefore write a custom __setattr__.
        """
        assert 1 == WithOnSetAttrHook(1).x

        @attr.s(auto_detect=True, slots=slots)
        class RemoveNeedForOurSetAttr(WithOnSetAttrHook):
            x = attr.ib()

            def __setattr__(self, name, val):
                object.__setattr__(self, name, val * 2)

        i = RemoveNeedForOurSetAttr(1)

        assert not RemoveNeedForOurSetAttr.__attrs_own_setattr__
        assert 2 == i.x

    def test_setattr_restore_respects_auto_detect(self, slots):
        """
        If __setattr__ should be restored but the user supplied its own and
        set auto_detect, leave is alone.
        """

        @attr.s(auto_detect=True, slots=slots)
        class CustomSetAttr:
            def __setattr__(self, _, __):
                pass

        assert CustomSetAttr.__setattr__ != object.__setattr__

    def test_setattr_auto_detect_frozen(self, slots):
        """
        frozen=True together with a detected custom __setattr__ are rejected.
        """
        with pytest.raises(
            ValueError, match="Can't freeze a class with a custom __setattr__."
        ):

            @attr.s(auto_detect=True, slots=slots, frozen=True)
            class CustomSetAttr(Frozen):
                def __setattr__(self, _, __):
                    pass

    def test_setattr_auto_detect_on_setattr(self, slots):
        """
        on_setattr attributes together with a detected custom __setattr__ are
        rejected.
        """
        with pytest.raises(
            ValueError,
            match="Can't combine custom __setattr__ with on_setattr hooks.",
        ):

            @attr.s(auto_detect=True, slots=slots)
            class HookAndCustomSetAttr:
                x = attr.ib(on_setattr=lambda *args: None)

                def __setattr__(self, _, __):
                    pass

    @pytest.mark.parametrize("a_slots", [True, False])
    @pytest.mark.parametrize("b_slots", [True, False])
    @pytest.mark.parametrize("c_slots", [True, False])
    def test_setattr_inherited_do_not_reset_intermediate(
        self, a_slots, b_slots, c_slots
    ):
        """
        A user-provided intermediate __setattr__ is not reset to
        object.__setattr__.

        This only can work with auto_detect activated, such that attrs can know
        that there is a user-provided __setattr__.
        """

        @attr.s(slots=a_slots)
        class A:
            x = attr.ib(on_setattr=setters.frozen)

        @attr.s(slots=b_slots, auto_detect=True)
        class B(A):
            x = attr.ib(on_setattr=setters.NO_OP)

            def __setattr__(self, key, value):
                raise SystemError

        @attr.s(slots=c_slots)
        class C(B):
            pass

        assert getattr(A, "__attrs_own_setattr__", False) is True
        assert getattr(B, "__attrs_own_setattr__", False) is False
        assert getattr(C, "__attrs_own_setattr__", False) is False

        with pytest.raises(SystemError):
            C(1).x = 3

    def test_docstring(self):
        """
        Generated __setattr__ has a useful docstring.
        """
        assert (
            "Method generated by attrs for class WithOnSetAttrHook."
            == WithOnSetAttrHook.__setattr__.__doc__
        )

    def test_setattr_converter_piped(self):
        """
        If a converter is used, it is piped through the on_setattr hooks.

        Regression test for https://github.com/python-attrs/attrs/issues/1327
        """

        @attr.define  # converter on setattr is implied in NG
        class C:
            x = attr.field(converter=[int])

        c = C("1")
        c.x = "2"

        assert 2 == c.x


# SPDX-License-Identifier: MIT

"""
Tests for `attr._make`.
"""

import copy
import functools
import gc
import inspect
import itertools
import sys
import unicodedata

from operator import attrgetter
from typing import Generic, TypeVar

import pytest

from hypothesis import assume, given
from hypothesis.strategies import booleans, integers, lists, sampled_from, text

import attr

from attr import _config
from attr._compat import PY_3_10_PLUS, PY_3_14_PLUS
from attr._make import (
    Attribute,
    Factory,
    _AndValidator,
    _Attributes,
    _ClassBuilder,
    _CountingAttr,
    _determine_attrib_eq_order,
    _determine_attrs_eq_order,
    _determine_whether_to_implement,
    _transform_attrs,
    and_,
    fields,
    fields_dict,
    make_class,
    validate,
)
from attr.exceptions import DefaultAlreadySetError, NotAnAttrsClassError

from .strategies import (
    gen_attr_names,
    list_of_attrs,
    optional_bool,
    simple_attrs,
    simple_attrs_with_metadata,
    simple_attrs_without_metadata,
    simple_classes,
)
from .utils import simple_attr


attrs_st = simple_attrs.map(lambda c: Attribute.from_counting_attr("name", c))


@pytest.fixture(name="with_and_without_validation", params=[True, False])
def _with_and_without_validation(request):
    """
    Run tests with and without validation enabled.
    """
    attr.validators.set_disabled(request.param)

    try:
        yield
    finally:
        attr.validators.set_disabled(False)


class TestCountingAttr:
    """
    Tests for `attr`.
    """

    def test_returns_Attr(self):
        """
        Returns an instance of _CountingAttr.
        """
        a = attr.ib()

        assert isinstance(a, _CountingAttr)

    def test_validators_lists_to_wrapped_tuples(self):
        """
        If a list is passed as validator, it's just converted to a tuple.
        """

        def v1(_, __):
            pass

        def v2(_, __):
            pass

        a = attr.ib(validator=[v1, v2])

        assert _AndValidator((v1, v2)) == a._validator

    def test_validator_decorator_single(self):
        """
        If _CountingAttr.validator is used as a decorator and there is no
        decorator set, the decorated method is used as the validator.
        """
        a = attr.ib()

        @a.validator
        def v():
            pass

        assert v == a._validator

    @pytest.mark.parametrize(
        "wrap", [lambda v: v, lambda v: [v], lambda v: and_(v)]
    )
    def test_validator_decorator(self, wrap):
        """
        If _CountingAttr.validator is used as a decorator and there is already
        a decorator set, the decorators are composed using `and_`.
        """

        def v(_, __):
            pass

        a = attr.ib(validator=wrap(v))

        @a.validator
        def v2(self, _, __):
            pass

        assert _AndValidator((v, v2)) == a._validator

    def test_default_decorator_already_set(self):
        """
        Raise DefaultAlreadySetError if the decorator is used after a default
        has been set.
        """
        a = attr.ib(default=42)

        with pytest.raises(DefaultAlreadySetError):

            @a.default
            def f(self):
                pass

    def test_default_decorator_sets(self):
        """
        Decorator wraps the method in a Factory with pass_self=True and sets
        the default.
        """
        a = attr.ib()

        @a.default
        def f(self):
            pass

        assert Factory(f, True) == a._default


def make_tc():
    class TransformC:
        z = attr.ib()
        y = attr.ib()
        x = attr.ib()
        a = 42

    return TransformC


class TestTransformAttrs:
    """
    Tests for `_transform_attrs`.
    """

    def test_no_modifications(self):
        """
        Does not attach __attrs_attrs__ to the class.
        """
        C = make_tc()
        _transform_attrs(C, None, False, False, True, None)

        assert None is getattr(C, "__attrs_attrs__", None)

    def test_normal(self):
        """
        Transforms every `_CountingAttr` and leaves others (a) be.
        """
        C = make_tc()
        attrs, _, _ = _transform_attrs(C, None, False, False, True, None)

        assert ["z", "y", "x"] == [a.name for a in attrs]

    def test_empty(self):
        """
        No attributes works as expected.
        """

        @attr.s
        class C:
            pass

        assert _Attributes((), [], {}) == _transform_attrs(
            C, None, False, False, True, None
        )

    def test_transforms_to_attribute(self):
        """
        All `_CountingAttr`s are transformed into `Attribute`s.
        """
        C = make_tc()
        attrs, base_attrs, _ = _transform_attrs(
            C, None, False, False, True, None
        )

        assert [] == base_attrs
        assert 3 == len(attrs)
        assert all(isinstance(a, Attribute) for a in attrs)

    def test_conflicting_defaults(self):
        """
        Raises `ValueError` if attributes with defaults are followed by
        mandatory attributes.
        """

        class C:
            x = attr.ib(default=None)
            y = attr.ib()

        with pytest.raises(ValueError) as e:
            _transform_attrs(C, None, False, False, True, None)
        assert (
            "No mandatory attributes allowed after an attribute with a "
            "default value or factory.  Attribute in question: Attribute"
            "(name='y', default=NOTHING, validator=None, repr=True, "
            "eq=True, eq_key=None, order=True, order_key=None, "
            "hash=None, init=True, "
            "metadata=mappingproxy({}), type=None, converter=None, "
            "kw_only=False, inherited=False, on_setattr=None, alias=None)",
        ) == e.value.args

    def test_kw_only(self):
        """
        Converts all attributes, including base class' attributes, if `kw_only`
        is provided. Therefore, `kw_only` allows attributes with defaults to
        precede mandatory attributes.

        Updates in the subclass *don't* affect the base class attributes.
        """

        @attr.s
        class B:
            b = attr.ib()

        for b_a in B.__attrs_attrs__:
            assert b_a.kw_only is False

        class C(B):
            x = attr.ib(default=None)
            y = attr.ib()

        attrs, base_attrs, _ = _transform_attrs(
            C, None, False, True, True, None
        )

        assert len(attrs) == 3
        assert len(base_attrs) == 1

        for a in attrs:
            assert a.kw_only is True

        for b_a in B.__attrs_attrs__:
            assert b_a.kw_only is False

    def test_these(self):
        """
        If these is passed, use it and ignore body and base classes.
        """

        class Base:
            z = attr.ib()

        class C(Base):
            y = attr.ib()

        attrs, base_attrs, _ = _transform_attrs(
            C, {"x": attr.ib()}, False, False, True, None
        )

        assert [] == base_attrs
        assert (simple_attr("x"),) == attrs

    def test_these_leave_body(self):
        """
        If these is passed, no attributes are removed from the body.
        """

        @attr.s(init=False, these={"x": attr.ib()})
        class C:
            x = 5

        assert 5 == C().x
        assert "C(x=5)" == repr(C())

    def test_these_ordered(self):
        """
        If these is passed ordered attrs, their order respect instead of the
        counter.
        """
        b = attr.ib(default=2)
        a = attr.ib(default=1)

        @attr.s(these={"a": a, "b": b})
        class C:
            pass

        assert "C(a=1, b=2)" == repr(C())

    def test_multiple_inheritance_old(self):
        """
        Old multiple inheritance attribute collection behavior is retained.

        See #285
        """

        @attr.s
        class A:
            a1 = attr.ib(default="a1")
            a2 = attr.ib(default="a2")

        @attr.s
        class B(A):
            b1 = attr.ib(default="b1")
            b2 = attr.ib(default="b2")

        @attr.s
        class C(B, A):
            c1 = attr.ib(default="c1")
            c2 = attr.ib(default="c2")

        @attr.s
        class D(A):
            d1 = attr.ib(default="d1")
            d2 = attr.ib(default="d2")

        @attr.s
        class E(C, D):
            e1 = attr.ib(default="e1")
            e2 = attr.ib(default="e2")

        assert (
            "E(a1='a1', a2='a2', b1='b1', b2='b2', c1='c1', c2='c2', d1='d1', "
            "d2='d2', e1='e1', e2='e2')"
        ) == repr(E())

    def test_overwrite_proper_mro(self):
        """
        The proper MRO path works single overwrites too.
        """

        @attr.s(collect_by_mro=True)
        class C:
            x = attr.ib(default=1)

        @attr.s(collect_by_mro=True)
        class D(C):
            x = attr.ib(default=2)

        assert "D(x=2)" == repr(D())

    def test_multiple_inheritance_proper_mro(self):
        """
        Attributes are collected according to the MRO.

        See #428
        """

        @attr.s
        class A:
            a1 = attr.ib(default="a1")
            a2 = attr.ib(default="a2")

        @attr.s
        class B(A):
            b1 = attr.ib(default="b1")
            b2 = attr.ib(default="b2")

        @attr.s
        class C(B, A):
            c1 = attr.ib(default="c1")
            c2 = attr.ib(default="c2")

        @attr.s
        class D(A):
            d1 = attr.ib(default="d1")
            d2 = attr.ib(default="d2")

        @attr.s(collect_by_mro=True)
        class E(C, D):
            e1 = attr.ib(default="e1")
            e2 = attr.ib(default="e2")

        assert (
            "E(a1='a1', a2='a2', d1='d1', d2='d2', b1='b1', b2='b2', c1='c1', "
            "c2='c2', e1='e1', e2='e2')"
        ) == repr(E())

    def test_mro(self):
        """
        Attributes and methods are looked up the same way.

        See #428
        """

        @attr.s(collect_by_mro=True)
        class A:
            x = attr.ib(10)

            def xx(self):
                return 10

        @attr.s(collect_by_mro=True)
        class B(A):
            y = attr.ib(20)

        @attr.s(collect_by_mro=True)
        class C(A):
            x = attr.ib(50)

            def xx(self):
                return 50

        @attr.s(collect_by_mro=True)
        class D(B, C):
            pass

        d = D()

        assert d.x == d.xx()

    def test_inherited(self):
        """
        Inherited Attributes have `.inherited` True, otherwise False.
        """

        @attr.s
        class A:
            a = attr.ib()

        @attr.s
        class B(A):
            b = attr.ib()

        @attr.s
        class C(B):
            a = attr.ib()
            c = attr.ib()

        f = attr.fields

        assert False is f(A).a.inherited

        assert True is f(B).a.inherited
        assert False is f(B).b.inherited

        assert False is f(C).a.inherited
        assert True is f(C).b.inherited
        assert False is f(C).c.inherited


class TestAttributes:
    """
    Tests for the `attrs`/`attr.s` class decorator.
    """

    def test_sets_attrs(self):
        """
        Sets the `__attrs_attrs__` class attribute with a list of `Attribute`s.
        """

        @attr.s
        class C:
            x = attr.ib()

        assert "x" == C.__attrs_attrs__[0].name
        assert all(isinstance(a, Attribute) for a in C.__attrs_attrs__)

    def test_empty(self):
        """
        No attributes, no problems.
        """

        @attr.s
        class C3:
            pass

        assert "C3()" == repr(C3())
        assert C3() == C3()

    @given(attr=attrs_st, attr_name=sampled_from(Attribute.__slots__))
    def test_immutable(self, attr, attr_name):
        """
        Attribute instances are immutable.
        """
        with pytest.raises(AttributeError):
            setattr(attr, attr_name, 1)

    @pytest.mark.parametrize(
        "method_name", ["__repr__", "__eq__", "__hash__", "__init__"]
    )
    def test_adds_all_by_default(self, method_name):
        """
        If no further arguments are supplied, all add_XXX functions except
        add_hash are applied.  __hash__ is set to None.
        """
        # Set the method name to a sentinel and check whether it has been
        # overwritten afterwards.
        sentinel = object()

        class C:
            x = attr.ib()

        setattr(C, method_name, sentinel)

        C = attr.s(C)
        meth = getattr(C, method_name)

        assert sentinel != meth
        if method_name == "__hash__":
            assert meth is None

    @pytest.mark.parametrize(
        ("arg_name", "method_name"),
        [
            ("repr", "__repr__"),
            ("eq", "__eq__"),
            ("order", "__le__"),
            ("unsafe_hash", "__hash__"),
            ("init", "__init__"),
        ],
    )
    def test_respects_add_arguments(self, arg_name, method_name):
        """
        If a certain `XXX` is `False`, `__XXX__` is not added to the class.
        """
        # Set the method name to a sentinel and check whether it has been
        # overwritten afterwards.
        sentinel = object()

        am_args = {
            "repr": True,
            "eq": True,
            "order": True,
            "unsafe_hash": True,
            "init": True,
        }
        am_args[arg_name] = False
        if arg_name == "eq":
            am_args["order"] = False

        class C:
            x = attr.ib()

        setattr(C, method_name, sentinel)

        C = attr.s(**am_args)(C)

        assert sentinel == getattr(C, method_name)

    @pytest.mark.parametrize("init", [True, False])
    def test_respects_init_attrs_init(self, init):
        """
        If init=False, adds __attrs_init__ to the class.
        Otherwise, it does not.
        """

        class C:
            x = attr.ib()

        C = attr.s(init=init)(C)
        assert hasattr(C, "__attrs_init__") != init

    @given(slots_outer=booleans(), slots_inner=booleans())
    def test_repr_qualname(self, slots_outer, slots_inner):
        """
        The name in repr is the __qualname__.
        """

        @attr.s(slots=slots_outer)
        class C:
            @attr.s(slots=slots_inner)
            class D:
                pass

        assert "C.D()" == repr(C.D())
        assert "GC.D()" == repr(GC.D())

    @given(slots_outer=booleans(), slots_inner=booleans())
    def test_repr_fake_qualname(self, slots_outer, slots_inner):
        """
        Setting repr_ns overrides a potentially guessed namespace.
        """

        with pytest.deprecated_call(match="The `repr_ns` argument"):

            @attr.s(slots=slots_outer)
            class C:
                @attr.s(repr_ns="C", slots=slots_inner)
                class D:
                    pass

        assert "C.D()" == repr(C.D())

    @given(slots_outer=booleans(), slots_inner=booleans())
    def test_name_not_overridden(self, slots_outer, slots_inner):
        """
        __name__ is different from __qualname__.
        """

        @attr.s(slots=slots_outer)
        class C:
            @attr.s(slots=slots_inner)
            class D:
                pass

        assert C.D.__name__ == "D"
        assert C.D.__qualname__ == C.__qualname__ + ".D"

    @pytest.mark.usefixtures("with_and_without_validation")
    def test_pre_init(self):
        """
        Verify that __attrs_pre_init__ gets called if defined.
        """

        @attr.s
        class C:
            def __attrs_pre_init__(self2):
                self2.z = 30

        c = C()

        assert 30 == getattr(c, "z", None)

    @pytest.mark.usefixtures("with_and_without_validation")
    def test_pre_init_args(self):
        """
        Verify that __attrs_pre_init__ gets called with extra args if defined.
        """

        @attr.s
        class C:
            x = attr.ib()

            def __attrs_pre_init__(self2, x):
                self2.z = x + 1

        c = C(x=10)

        assert 11 == getattr(c, "z", None)

    @pytest.mark.usefixtures("with_and_without_validation")
    def test_pre_init_kwargs(self):
        """
        Verify that __attrs_pre_init__ gets called with extra args and kwargs
        if defined.
        """

        @attr.s
        class C:
            x = attr.ib()
            y = attr.field(kw_only=True)

            def __attrs_pre_init__(self2, x, y):
                self2.z = x + y + 1

        c = C(10, y=11)

        assert 22 == getattr(c, "z", None)

    @pytest.mark.usefixtures("with_and_without_validation")
    def test_pre_init_kwargs_only(self):
        """
        Verify that __attrs_pre_init__ gets called with extra kwargs only if
        defined.
        """

        @attr.s
        class C:
            y = attr.field(kw_only=True)

            def __attrs_pre_init__(self2, y):
                self2.z = y + 1

        c = C(y=11)

        assert 12 == getattr(c, "z", None)

    @pytest.mark.usefixtures("with_and_without_validation")
    def test_pre_init_kw_only_work_with_defaults(self):
        """
        Default values together with kw_only don't break __attrs__pre_init__.
        """
        val = None

        @attr.define
        class KWOnlyAndDefault:
            kw_and_default: int = attr.field(kw_only=True, default=3)

            def __attrs_pre_init__(self, *, kw_and_default):
                nonlocal val
                val = kw_and_default

        inst = KWOnlyAndDefault()

        assert 3 == val == inst.kw_and_default

    @pytest.mark.usefixtures("with_and_without_validation")
    def test_post_init(self):
        """
        Verify that __attrs_post_init__ gets called if defined.
        """

        @attr.s
        class C:
            x = attr.ib()
            y = attr.ib()

            def __attrs_post_init__(self2):
                self2.z = self2.x + self2.y

        c = C(x=10, y=20)

        assert 30 == getattr(c, "z", None)

    @pytest.mark.usefixtures("with_and_without_validation")
    def test_pre_post_init_order(self):
        """
        Verify that __attrs_post_init__ gets called if defined.
        """

        @attr.s
        class C:
            x = attr.ib()

            def __attrs_pre_init__(self2):
                self2.z = 30

            def __attrs_post_init__(self2):
                self2.z += self2.x

        c = C(x=10)

        assert 40 == getattr(c, "z", None)

    def test_types(self):
        """
        Sets the `Attribute.type` attr from type argument.
        """

        @attr.s
        class C:
            x = attr.ib(type=int)
            y = attr.ib(type=str)
            z = attr.ib()

        assert int is fields(C).x.type
        assert str is fields(C).y.type
        assert None is fields(C).z.type

    def test_clean_class(self, slots):
        """
        Attribute definitions do not appear on the class body after @attr.s.
        """

        @attr.s(slots=slots)
        class C:
            x = attr.ib()

        x = getattr(C, "x", None)

        assert not isinstance(x, _CountingAttr)

    def test_factory_sugar(self):
        """
        Passing factory=f is syntactic sugar for passing default=Factory(f).
        """

        @attr.s
        class C:
            x = attr.ib(factory=list)

        assert Factory(list) == attr.fields(C).x.default

    def test_sugar_factory_mutex(self):
        """
        Passing both default and factory raises ValueError.
        """
        with pytest.raises(ValueError, match="mutually exclusive"):

            @attr.s
            class C:
                x = attr.ib(factory=list, default=Factory(list))

    def test_sugar_callable(self):
        """
        Factory has to be a callable to prevent people from passing Factory
        into it.
        """
        with pytest.raises(ValueError, match="must be a callable"):

            @attr.s
            class C:
                x = attr.ib(factory=Factory(list))

    def test_inherited_does_not_affect_hashing_and_equality(self):
        """
        Whether or not an Attribute has been inherited doesn't affect how it's
        hashed and compared.
        """

        @attr.s
        class BaseClass:
            x = attr.ib()

        @attr.s
        class SubClass(BaseClass):
            pass

        ba = attr.fields(BaseClass)[0]
        sa = attr.fields(SubClass)[0]

        assert ba == sa
        assert hash(ba) == hash(sa)


class TestKeywordOnlyAttributes:
    """
    Tests for keyword-only attributes.
    """

    def test_adds_keyword_only_arguments(self):
        """
        Attributes can be added as keyword-only.
        """

        @attr.s
        class C:
            a = attr.ib()
            b = attr.ib(default=2, kw_only=True)
            c = attr.ib(kw_only=True)
            d = attr.ib(default=attr.Factory(lambda: 4), kw_only=True)

        c = C(1, c=3)

        assert c.a == 1
        assert c.b == 2
        assert c.c == 3
        assert c.d == 4

    def test_ignores_kw_only_when_init_is_false(self):
        """
        Specifying ``kw_only=True`` when ``init=False`` is essentially a no-op.
        """

        @attr.s
        class C:
            x = attr.ib(init=False, default=0, kw_only=True)
            y = attr.ib()

        c = C(1)

        assert c.x == 0
        assert c.y == 1

    def test_keyword_only_attributes_presence(self):
        """
        Raises `TypeError` when keyword-only arguments are
        not specified.
        """

        @attr.s
        class C:
            x = attr.ib(kw_only=True)

        with pytest.raises(TypeError) as e:
            C()

        assert (
            "missing 1 required keyword-only argument: 'x'"
        ) in e.value.args[0]

    def test_keyword_only_attributes_unexpected(self):
        """
        Raises `TypeError` when unexpected keyword argument passed.
        """

        @attr.s
        class C:
            x = attr.ib(kw_only=True)

        with pytest.raises(TypeError) as e:
            C(x=5, y=10)

        assert "got an unexpected keyword argument 'y'" in e.value.args[0]

    def test_keyword_only_attributes_can_come_in_any_order(self):
        """
        Mandatory vs non-mandatory attr order only matters when they are part
        of the __init__ signature and when they aren't kw_only (which are
        moved to the end and can be mandatory or non-mandatory in any order,
        as they will be specified as keyword args anyway).
        """

        @attr.s
        class C:
            a = attr.ib(kw_only=True)
            b = attr.ib(kw_only=True, default="b")
            c = attr.ib(kw_only=True)
            d = attr.ib()
            e = attr.ib(default="e")
            f = attr.ib(kw_only=True)
            g = attr.ib(kw_only=True, default="g")
            h = attr.ib(kw_only=True)
            i = attr.ib(init=False)

        c = C("d", a="a", c="c", f="f", h="h")

        assert c.a == "a"
        assert c.b == "b"
        assert c.c == "c"
        assert c.d == "d"
        assert c.e == "e"
        assert c.f == "f"
        assert c.g == "g"
        assert c.h == "h"

    def test_keyword_only_attributes_allow_subclassing(self):
        """
        Subclass can define keyword-only attributed without defaults,
        when the base class has attributes with defaults.
        """

        @attr.s
        class Base:
            x = attr.ib(default=0)

        @attr.s
        class C(Base):
            y = attr.ib(kw_only=True)

        c = C(y=1)

        assert c.x == 0
        assert c.y == 1

    def test_keyword_only_class_level(self):
        """
        `kw_only` can be provided at the attr.s level, converting all
        attributes to `kw_only.`
        """

        @attr.s(kw_only=True)
        class C:
            x = attr.ib()
            y = attr.ib(kw_only=True)

        with pytest.raises(TypeError):
            C(0, y=1)

        c = C(x=0, y=1)

        assert c.x == 0
        assert c.y == 1

    def test_keyword_only_class_level_subclassing(self):
        """
        Subclass `kw_only` propagates to attrs inherited from the base,
        allowing non-default following default.
        """

        @attr.s
        class Base:
            x = attr.ib(default=0)

        @attr.s(kw_only=True)
        class C(Base):
            y = attr.ib()

        with pytest.raises(TypeError):
            C(1)

        c = C(x=0, y=1)

        assert c.x == 0
        assert c.y == 1

    def test_init_false_attribute_after_keyword_attribute(self):
        """
        A positional attribute cannot follow a `kw_only` attribute,
        but an `init=False` attribute can because it won't appear
        in `__init__`
        """

        @attr.s
        class KwArgBeforeInitFalse:
            kwarg = attr.ib(kw_only=True)
            non_init_function_default = attr.ib(init=False)
            non_init_keyword_default = attr.ib(
                init=False, default="default-by-keyword"
            )

            @non_init_function_default.default
            def _init_to_init(self):
                return self.kwarg + "b"

        c = KwArgBeforeInitFalse(kwarg="a")

        assert c.kwarg == "a"
        assert c.non_init_function_default == "ab"
        assert c.non_init_keyword_default == "default-by-keyword"

    def test_init_false_attribute_after_keyword_attribute_with_inheritance(
        self,
    ):
        """
        A positional attribute cannot follow a `kw_only` attribute,
        but an `init=False` attribute can because it won't appear
        in `__init__`. This test checks that we allow this
        even when the `kw_only` attribute appears in a parent class
        """

        @attr.s
        class KwArgBeforeInitFalseParent:
            kwarg = attr.ib(kw_only=True)

        @attr.s
        class KwArgBeforeInitFalseChild(KwArgBeforeInitFalseParent):
            non_init_function_default = attr.ib(init=False)
            non_init_keyword_default = attr.ib(
                init=False, default="default-by-keyword"
            )

            @non_init_function_default.default
            def _init_to_init(self):
                return self.kwarg + "b"

        c = KwArgBeforeInitFalseChild(kwarg="a")

        assert c.kwarg == "a"
        assert c.non_init_function_default == "ab"
        assert c.non_init_keyword_default == "default-by-keyword"


@attr.s
class GC:
    @attr.s
    class D:
        pass


class TestMakeClass:
    """
    Tests for `make_class`.
    """

    @pytest.mark.parametrize("ls", [list, tuple])
    def test_simple(self, ls):
        """
        Passing a list of strings creates attributes with default args.
        """
        C1 = make_class("C1", ls(["a", "b"]))

        @attr.s
        class C2:
            a = attr.ib()
            b = attr.ib()

        assert C1.__attrs_attrs__ == C2.__attrs_attrs__

    def test_dict(self):
        """
        Passing a dict of name: _CountingAttr creates an equivalent class.
        """
        C1 = make_class(
            "C1", {"a": attr.ib(default=42), "b": attr.ib(default=None)}
        )

        @attr.s
        class C2:
            a = attr.ib(default=42)
            b = attr.ib(default=None)

        assert C1.__attrs_attrs__ == C2.__attrs_attrs__

    def test_attr_args(self):
        """
        attributes_arguments are passed to attributes
        """
        C = make_class("C", ["x"], repr=False)

        assert repr(C(1)).startswith("<tests.test_make.C object at 0x")

    def test_normalized_unicode_attr_args(self):
        """
        Unicode identifiers are valid in Python.
        """
        clsname = "ü"

        assert clsname == unicodedata.normalize("NFKC", clsname)

        attrname = "ß"

        assert attrname == unicodedata.normalize("NFKC", attrname)

        C = make_class(clsname, [attrname], repr=False)

        assert repr(C(1)).startswith("<tests.test_make.ü object at 0x")

        kwargs = {"ß": 1}
        c = C(**kwargs)

        assert 1 == c.ß

    def test_unnormalized_unicode_attr_args(self):
        """
        Unicode identifiers are normalized to NFKC form in Python.
        """

        clsname = "Ŀ"

        assert clsname != unicodedata.normalize("NFKC", clsname)

        attrname = "ㅁ"

        assert attrname != unicodedata.normalize("NFKC", attrname)

        C = make_class(clsname, [attrname], repr=False)
        assert repr(C(1)).startswith("<tests.test_make.L· object at 0x")

        kwargs = {unicodedata.normalize("NFKC", attrname): 1}
        c = C(**kwargs)

        assert 1 == c.ㅁ

    def test_catches_wrong_attrs_type(self):
        """
        Raise `TypeError` if an invalid type for attrs is passed.
        """
        with pytest.raises(TypeError) as e:
            make_class("C", object())

        assert ("attrs argument must be a dict or a list.",) == e.value.args

    def test_bases(self):
        """
        Parameter bases default to (object,) and subclasses correctly
        """

        class D:
            pass

        cls = make_class("C", {})

        assert cls.__mro__[-1] is object

        cls = make_class("C", {}, bases=(D,))

        assert D in cls.__mro__
        assert isinstance(cls(), D)

    def test_additional_class_body(self):
        """
        Additional class_body is added to newly created class.
        """

        def echo_func(cls, *args):
            return args

        cls = make_class("C", {}, class_body={"echo": classmethod(echo_func)})

        assert ("a", "b") == cls.echo("a", "b")

    def test_clean_class(self, slots):
        """
        Attribute definitions do not appear on the class body.
        """
        C = make_class("C", ["x"], slots=slots)

        x = getattr(C, "x", None)

        assert not isinstance(x, _CountingAttr)

    def test_missing_sys_getframe(self, monkeypatch):
        """
        `make_class()` does not fail when `sys._getframe()` is not available.
        """
        monkeypatch.delattr(sys, "_getframe")
        C = make_class("C", ["x"])

        assert 1 == len(C.__attrs_attrs__)

    def test_make_class_ordered(self):
        """
        If `make_class()` is passed ordered attrs, their order is respected
        instead of the counter.
        """
        b = attr.ib(default=2)
        a = attr.ib(default=1)

        C = attr.make_class("C", {"a": a, "b": b})

        assert "C(a=1, b=2)" == repr(C())

    def test_generic_dynamic_class(self):
        """
        make_class can create generic dynamic classes.

        https://github.com/python-attrs/attrs/issues/756
        https://bugs.python.org/issue33188
        """
        from types import new_class
        from typing import Generic, TypeVar

        MyTypeVar = TypeVar("MyTypeVar")
        MyParent = new_class("MyParent", (Generic[MyTypeVar],), {})

        attr.make_class("test", {"id": attr.ib(type=str)}, (MyParent[int],))

    def test_annotations(self):
        """
        make_class fills the __annotations__ dict for attributes with a known
        type.
        """
        a = attr.ib(type=bool)
        b = attr.ib(
            type=None
        )  # Won't be added to ann. b/c of unfavorable default
        c = attr.ib()

        C = attr.make_class("C", {"a": a, "b": b, "c": c})
        C = attr.resolve_types(C)

        assert {"a": bool} == C.__annotations__

    def test_annotations_resolve(self):
        """
        resolve_types() resolves the annotations added by make_class().
        """
        a = attr.ib(type="bool")

        C = attr.make_class("C", {"a": a})
        C = attr.resolve_types(C)

        assert attr.fields(C).a.type is bool
        assert {"a": "bool"} == C.__annotations__


class TestFields:
    """
    Tests for `fields`.
    """

    @given(simple_classes())
    def test_instance(self, C):
        """
        Raises `TypeError` on non-classes.
        """
        with pytest.raises(TypeError) as e:
            fields(C())

        assert "Passed object must be a class." == e.value.args[0]

    def test_handler_non_attrs_class(self):
        """
        Raises `ValueError` if passed a non-*attrs* instance.
        """
        with pytest.raises(NotAnAttrsClassError) as e:
            fields(object)

        assert (
            f"{object!r} is not an attrs-decorated class."
        ) == e.value.args[0]

    def test_handler_non_attrs_generic_class(self):
        """
        Raises `ValueError` if passed a non-*attrs* generic class.
        """
        T = TypeVar("T")

        class B(Generic[T]):
            pass

        with pytest.raises(NotAnAttrsClassError) as e:
            fields(B[str])

        assert (
            f"{B[str]!r} is not an attrs-decorated class."
        ) == e.value.args[0]

    @given(simple_classes())
    def test_fields(self, C):
        """
        Returns a list of `Attribute`a.
        """
        assert all(isinstance(a, Attribute) for a in fields(C))

    @given(simple_classes())
    def test_fields_properties(self, C):
        """
        Fields returns a tuple with properties.
        """
        for attribute in fields(C):
            assert getattr(fields(C), attribute.name) is attribute

    def test_generics(self):
        """
        Fields work with generic classes.
        """
        T = TypeVar("T")

        @attr.define
        class A(Generic[T]):
            a: T

        assert len(fields(A)) == 1
        assert fields(A).a.name == "a"
        assert fields(A).a.default is attr.NOTHING

        assert len(fields(A[str])) == 1
        assert fields(A[str]).a.name == "a"
        assert fields(A[str]).a.default is attr.NOTHING


class TestFieldsDict:
    """
    Tests for `fields_dict`.
    """

    @given(simple_classes())
    def test_instance(self, C):
        """
        Raises `TypeError` on non-classes.
        """
        with pytest.raises(TypeError) as e:
            fields_dict(C())

        assert "Passed object must be a class." == e.value.args[0]

    def test_handler_non_attrs_class(self):
        """
        Raises `ValueError` if passed a non-*attrs* instance.
        """
        with pytest.raises(NotAnAttrsClassError) as e:
            fields_dict(object)

        assert (
            f"{object!r} is not an attrs-decorated class."
        ) == e.value.args[0]

    @given(simple_classes())
    def test_fields_dict(self, C):
        """
        Returns an ordered dict of ``{attribute_name: Attribute}``.
        """
        d = fields_dict(C)

        assert isinstance(d, dict)
        assert list(fields(C)) == list(d.values())
        assert [a.name for a in fields(C)] == list(d)


class TestConverter:
    """
    Tests for attribute conversion.
    """

    def test_converter(self):
        """
        Return value of converter is used as the attribute's value.
        """
        C = make_class(
            "C", {"x": attr.ib(converter=lambda v: v + 1), "y": attr.ib()}
        )
        c = C(1, 2)

        assert c.x == 2
        assert c.y == 2

    def test_converter_wrapped_takes_self(self):
        """
        When wrapped and passed `takes_self`, the converter receives the
        instance that's being initializes -- and the return value is used as
        the field's value.
        """

        def converter_with_self(v, self_):
            return v * self_.y

        @attr.define
        class C:
            x: int = attr.field(
                converter=attr.Converter(converter_with_self, takes_self=True)
            )
            y = 42

        assert 84 == C(2).x

    def test_converter_wrapped_takes_field(self):
        """
        When wrapped and passed `takes_field`, the converter receives the field
        definition -- and the return value is used as the field's value.
        """

        def converter_with_field(v, field):
            assert isinstance(field, attr.Attribute)
            return v * field.metadata["x"]

        @attr.define
        class C:
            x: int = attr.field(
                converter=attr.Converter(
                    converter_with_field, takes_field=True
                ),
                metadata={"x": 42},
            )

        assert 84 == C(2).x

    @given(integers(), booleans())
    def test_convert_property(self, val, init):
        """
        Property tests for attributes using converter.
        """
        C = make_class(
            "C",
            {
                "y": attr.ib(),
                "x": attr.ib(
                    init=init, default=val, converter=lambda v: v + 1
                ),
            },
        )
        c = C(2)

        assert c.x == val + 1
        assert c.y == 2

    @given(integers(), booleans())
    def test_converter_factory_property(self, val, init):
        """
        Property tests for attributes with converter, and a factory default.
        """
        C = make_class(
            "C",
            {
                "y": attr.ib(),
                "x": attr.ib(
                    init=init,
                    default=Factory(lambda: val),
                    converter=lambda v: v + 1,
                ),
            },
        )
        c = C(2)

        assert c.x == val + 1
        assert c.y == 2

    def test_convert_before_validate(self):
        """
        Validation happens after conversion.
        """

        def validator(inst, attr, val):
            raise RuntimeError("foo")

        C = make_class(
            "C",
            {
                "x": attr.ib(validator=validator, converter=lambda v: 1 / 0),
                "y": attr.ib(),
            },
        )
        with pytest.raises(ZeroDivisionError):
            C(1, 2)

    def test_frozen(self):
        """
        Converters circumvent immutability.
        """
        C = make_class(
            "C", {"x": attr.ib(converter=lambda v: int(v))}, frozen=True
        )
        C("1")


class TestValidate:
    """
    Tests for `validate`.
    """

    def test_success(self):
        """
        If the validator succeeds, nothing gets raised.
        """
        C = make_class(
            "C", {"x": attr.ib(validator=lambda *a: None), "y": attr.ib()}
        )
        validate(C(1, 2))

    def test_propagates(self):
        """
        The exception of the validator is handed through.
        """

        def raiser(_, __, value):
            if value == 42:
                raise FloatingPointError

        C = make_class("C", {"x": attr.ib(validator=raiser)})
        i = C(1)
        i.x = 42

        with pytest.raises(FloatingPointError):
            validate(i)

    def test_run_validators(self):
        """
        Setting `_run_validators` to False prevents validators from running.
        """
        _config._run_validators = False
        obj = object()

        def raiser(_, __, ___):
            raise Exception(obj)

        C = make_class("C", {"x": attr.ib(validator=raiser)})
        c = C(1)
        validate(c)
        assert 1 == c.x
        _config._run_validators = True

        with pytest.raises(Exception):
            validate(c)

        with pytest.raises(Exception) as e:
            C(1)
        assert (obj,) == e.value.args

    def test_multiple_validators(self):
        """
        If a list is passed as a validator, all of its items are treated as one
        and must pass.
        """

        def v1(_, __, value):
            if value == 23:
                raise TypeError("omg")

        def v2(_, __, value):
            if value == 42:
                raise ValueError("omg")

        C = make_class("C", {"x": attr.ib(validator=[v1, v2])})

        validate(C(1))

        with pytest.raises(TypeError) as e:
            C(23)

        assert "omg" == e.value.args[0]

        with pytest.raises(ValueError) as e:
            C(42)

        assert "omg" == e.value.args[0]

    def test_multiple_empty(self):
        """
        Empty list/tuple for validator is the same as None.
        """
        C1 = make_class("C", {"x": attr.ib(validator=[])})
        C2 = make_class("C", {"x": attr.ib(validator=None)})

        assert inspect.getsource(C1.__init__) == inspect.getsource(C2.__init__)


# Hypothesis seems to cache values, so the lists of attributes come out
# unsorted.
sorted_lists_of_attrs = list_of_attrs.map(
    lambda ln: sorted(ln, key=attrgetter("counter"))
)


class TestMetadata:
    """
    Tests for metadata handling.
    """

    @given(sorted_lists_of_attrs)
    def test_metadata_present(self, list_of_attrs):
        """
        Assert dictionaries are copied and present.
        """
        C = make_class("C", dict(zip(gen_attr_names(), list_of_attrs)))

        for hyp_attr, class_attr in zip(list_of_attrs, fields(C)):
            if hyp_attr.metadata is None:
                # The default is a singleton empty dict.
                assert class_attr.metadata is not None
                assert len(class_attr.metadata) == 0
            else:
                assert hyp_attr.metadata == class_attr.metadata

                # Once more, just to assert getting items and iteration.
                for k in class_attr.metadata:
                    assert hyp_attr.metadata[k] == class_attr.metadata[k]
                    assert hyp_attr.metadata.get(k) == class_attr.metadata.get(
                        k
                    )

    @given(simple_classes(), text())
    def test_metadata_immutability(self, C, string):
        """
        The metadata dict should be best-effort immutable.
        """
        for a in fields(C):
            with pytest.raises(TypeError):
                a.metadata[string] = string
            with pytest.raises(AttributeError):
                a.metadata.update({string: string})
            with pytest.raises(AttributeError):
                a.metadata.clear()
            with pytest.raises(AttributeError):
                a.metadata.setdefault(string, string)

            for k in a.metadata:
                # For some reason, MappingProxyType throws an IndexError for
                # deletes on a large integer key.
                with pytest.raises((TypeError, IndexError)):
                    del a.metadata[k]
                with pytest.raises(AttributeError):
                    a.metadata.pop(k)
            with pytest.raises(AttributeError):
                a.metadata.popitem()

    @given(lists(simple_attrs_without_metadata, min_size=2, max_size=5))
    def test_empty_metadata_singleton(self, list_of_attrs):
        """
        All empty metadata attributes share the same empty metadata dict.
        """
        C = make_class("C", dict(zip(gen_attr_names(), list_of_attrs)))
        for a in fields(C)[1:]:
            assert a.metadata is fields(C)[0].metadata

    @given(lists(simple_attrs_without_metadata, min_size=2, max_size=5))
    def test_empty_countingattr_metadata_independent(self, list_of_attrs):
        """
        All empty metadata attributes are independent before ``@attr.s``.
        """
        for x, y in itertools.combinations(list_of_attrs, 2):
            assert x.metadata is not y.metadata

    @given(lists(simple_attrs_with_metadata(), min_size=2, max_size=5))
    def test_not_none_metadata(self, list_of_attrs):
        """
        Non-empty metadata attributes exist as fields after ``@attr.s``.
        """
        C = make_class("C", dict(zip(gen_attr_names(), list_of_attrs)))

        assert len(fields(C)) > 0

        for cls_a, raw_a in zip(fields(C), list_of_attrs):
            assert cls_a.metadata != {}
            assert cls_a.metadata == raw_a.metadata

    def test_metadata(self):
        """
        If metadata that is not None is passed, it is used.

        This is necessary for coverage because the previous test is
        hypothesis-based.
        """
        md = {}
        a = attr.ib(metadata=md)

        assert md is a.metadata


class TestClassBuilder:
    """
    Tests for `_ClassBuilder`.
    """

    def test_repr_str(self):
        """
        Trying to add a `__str__` without having a `__repr__` raises a
        ValueError.
        """
        with pytest.raises(ValueError) as ei:
            make_class("C", {}, repr=False, str=True)

        assert (
            "__str__ can only be generated if a __repr__ exists.",
        ) == ei.value.args

    def test_repr(self):
        """
        repr of builder itself makes sense.
        """

        class C:
            pass

        b = _ClassBuilder(
            C,
            None,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            None,
            False,
            None,
        )

        assert "<_ClassBuilder(cls=C)>" == repr(b)

    def test_returns_self(self):
        """
        All methods return the builder for chaining.
        """

        class C:
            x = attr.ib()

        b = _ClassBuilder(
            C,
            None,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            None,
            False,
            None,
        )

        cls = (
            b.add_eq()
            .add_order()
            .add_hash()
            .add_init()
            .add_attrs_init()
            .add_repr("ns")
            .add_str()
            .build_class()
        )

        assert "ns.C(x=1)" == repr(cls(1))

    @pytest.mark.parametrize(
        "meth_name",
        [
            "__init__",
            "__hash__",
            "__repr__",
            "__str__",
            "__eq__",
            "__lt__",
            "__le__",
            "__gt__",
            "__ge__",
        ],
    )
    def test_attaches_meta_dunders(self, meth_name):
        """
        Generated methods have correct __module__, __name__, and __qualname__
        attributes.
        """

        @attr.s(unsafe_hash=True, str=True)
        class C:
            def organic(self):
                pass

        @attr.s(unsafe_hash=True, str=True)
        class D:
            pass

        meth_C = getattr(C, meth_name)
        meth_D = getattr(D, meth_name)

        assert meth_name == meth_C.__name__ == meth_D.__name__
        assert C.organic.__module__ == meth_C.__module__ == meth_D.__module__
        # This is assertion that would fail if a single __ne__ instance
        # was reused across multiple _make_eq calls.
        organic_prefix = C.organic.__qualname__.rsplit(".", 1)[0]
        assert organic_prefix + "." + meth_name == meth_C.__qualname__

    def test_handles_missing_meta_on_class(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """
        If the class hasn't a __module__ or __qualname__, the method hasn't
        either.
        """

        class C:
            pass

        orig_hasattr = __builtins__["hasattr"]

        def our_hasattr(obj, name, /) -> bool:
            if name in ("__module__", "__qualname__"):
                return False
            return orig_hasattr(obj, name)

        monkeypatch.setitem(
            _ClassBuilder.__init__.__globals__["__builtins__"],
            "hasattr",
            our_hasattr,
        )

        b = _ClassBuilder(
            C,
            these=None,
            slots=False,
            frozen=False,
            weakref_slot=True,
            getstate_setstate=False,
            auto_attribs=False,
            is_exc=False,
            kw_only=False,
            cache_hash=False,
            collect_by_mro=True,
            on_setattr=None,
            has_custom_setattr=False,
            field_transformer=None,
        )

        def fake_meth(self):
            pass

        fake_meth.__module__ = "42"
        fake_meth.__qualname__ = "23"

        b._cls = {}  # No module and qualname

        rv = b._add_method_dunders(fake_meth)

        assert "42" == rv.__module__ == fake_meth.__module__
        assert "23" == rv.__qualname__ == fake_meth.__qualname__

    def test_weakref_setstate(self):
        """
        __weakref__ is not set on in setstate because it's not writable in
        slotted classes.
        """

        @attr.s(slots=True)
        class C:
            __weakref__ = attr.ib(
                init=False, hash=False, repr=False, eq=False, order=False
            )

        assert C() == copy.deepcopy(C())

    def test_no_references_to_original(self):
        """
        When subclassing a slotted class, there are no stray references to the
        original class.
        """

        @attr.s(slots=True)
        class C:
            pass

        @attr.s(slots=True)
        class C2(C):
            pass

        # The original C2 is in a reference cycle, so force a collect:
        gc.collect()

        assert [C2] == C.__subclasses__()

    @pytest.mark.xfail(PY_3_14_PLUS, reason="Currently broken on nightly.")
    def test_no_references_to_original_when_using_cached_property(self):
        """
        When subclassing a slotted class and using cached property, there are
        no stray references to the original class.
        """

        @attr.s(slots=True)
        class C:
            pass

        @attr.s(slots=True)
        class C2(C):
            @functools.cached_property
            def value(self) -> int:
                return 0

        # The original C2 is in a reference cycle, so force a collect:
        gc.collect()

        assert [C2] == C.__subclasses__()

    def _get_copy_kwargs(include_slots=True):
        """
        Generate a list of compatible attr.s arguments for the `copy` tests.
        """
        options = ["frozen", "unsafe_hash", "cache_hash"]

        if include_slots:
            options.extend(["slots", "weakref_slot"])

        out_kwargs = []
        for args in itertools.product([True, False], repeat=len(options)):
            kwargs = dict(zip(options, args))

            kwargs["unsafe_hash"] = kwargs["unsafe_hash"] or None

            if kwargs["cache_hash"] and not (
                kwargs["frozen"] or kwargs["unsafe_hash"]
            ):
                continue

            out_kwargs.append(kwargs)

        return out_kwargs

    @pytest.mark.parametrize("kwargs", _get_copy_kwargs())
    def test_copy(self, kwargs):
        """
        Ensure that an attrs class can be copied successfully.
        """

        @attr.s(eq=True, **kwargs)
        class C:
            x = attr.ib()

        a = C(1)
        b = copy.deepcopy(a)

        assert a == b

    @pytest.mark.parametrize("kwargs", _get_copy_kwargs(include_slots=False))
    def test_copy_custom_setstate(self, kwargs):
        """
        Ensure that non-slots classes respect a custom __setstate__.
        """

        @attr.s(eq=True, **kwargs)
        class C:
            x = attr.ib()

            def __getstate__(self):
                return self.__dict__

            def __setstate__(self, state):
                state["x"] *= 5
                self.__dict__.update(state)

        expected = C(25)
        actual = copy.copy(C(5))

        assert actual == expected


class TestInitAlias:
    """
    Tests for Attribute alias handling.
    """

    def test_default_and_specify(self):
        """
        alias is present on the Attributes returned from attr.fields.

        If left unspecified, it defaults to standard private-attribute
        handling.  If specified, it passes through the explicit alias.
        """

        # alias is None by default on _CountingAttr
        default_counting = attr.ib()
        assert default_counting.alias is None

        override_counting = attr.ib(alias="specified")
        assert override_counting.alias == "specified"

        @attr.s
        class Cases:
            public_default = attr.ib()
            _private_default = attr.ib()
            __dunder_default__ = attr.ib()

            public_override = attr.ib(alias="public")
            _private_override = attr.ib(alias="_private")
            __dunder_override__ = attr.ib(alias="__dunder__")

        cases = attr.fields_dict(Cases)

        # Default applies private-name mangling logic
        assert cases["public_default"].name == "public_default"
        assert cases["public_default"].alias == "public_default"

        assert cases["_private_default"].name == "_private_default"
        assert cases["_private_default"].alias == "private_default"

        assert cases["__dunder_default__"].name == "__dunder_default__"
        assert cases["__dunder_default__"].alias == "dunder_default__"

        # Override is passed through
        assert cases["public_override"].name == "public_override"
        assert cases["public_override"].alias == "public"

        assert cases["_private_override"].name == "_private_override"
        assert cases["_private_override"].alias == "_private"

        assert cases["__dunder_override__"].name == "__dunder_override__"
        assert cases["__dunder_override__"].alias == "__dunder__"

        # And aliases are applied to the __init__ signature
        example = Cases(
            public_default=1,
            private_default=2,
            dunder_default__=3,
            public=4,
            _private=5,
            __dunder__=6,
        )

        assert example.public_default == 1
        assert example._private_default == 2
        assert example.__dunder_default__ == 3
        assert example.public_override == 4
        assert example._private_override == 5
        assert example.__dunder_override__ == 6

    def test_evolve(self):
        """
        attr.evolve uses Attribute.alias to determine parameter names.
        """

        @attr.s
        class EvolveCase:
            _override = attr.ib(alias="_override")
            __mangled = attr.ib()
            __dunder__ = attr.ib()

        org = EvolveCase(1, 2, 3)

        # Previous behavior of evolve as broken for double-underscore
        # passthrough, and would raise here due to mis-mapping the __dunder__
        # alias
        assert attr.evolve(org) == org

        # evolve uses the alias to match __init__ signature
        assert attr.evolve(
            org,
            _override=0,
        ) == EvolveCase(0, 2, 3)

        # and properly passes through dunders and mangles
        assert attr.evolve(
            org,
            EvolveCase__mangled=4,
            dunder__=5,
        ) == EvolveCase(1, 4, 5)


class TestMakeOrder:
    """
    Tests for _make_order().
    """

    def test_subclasses_cannot_be_compared(self):
        """
        Calling comparison methods on subclasses raises a TypeError.

        We use the actual operation so we get an error raised.
        """

        @attr.s
        class A:
            a = attr.ib()

        @attr.s
        class B(A):
            pass

        a = A(42)
        b = B(42)

        assert a <= a
        assert a >= a
        assert not a < a
        assert not a > a

        assert (
            NotImplemented
            == a.__lt__(b)
            == a.__le__(b)
            == a.__gt__(b)
            == a.__ge__(b)
        )

        with pytest.raises(TypeError):
            a <= b

        with pytest.raises(TypeError):
            a >= b

        with pytest.raises(TypeError):
            a < b

        with pytest.raises(TypeError):
            a > b


class TestDetermineAttrsEqOrder:
    def test_default(self):
        """
        If all are set to None, set both eq and order to the passed default.
        """
        assert (42, 42) == _determine_attrs_eq_order(None, None, None, 42)

    @pytest.mark.parametrize("eq", [True, False])
    def test_order_mirrors_eq_by_default(self, eq):
        """
        If order is None, it mirrors eq.
        """
        assert (eq, eq) == _determine_attrs_eq_order(None, eq, None, True)

    def test_order_without_eq(self):
        """
        eq=False, order=True raises a meaningful ValueError.
        """
        with pytest.raises(
            ValueError, match="`order` can only be True if `eq` is True too."
        ):
            _determine_attrs_eq_order(None, False, True, True)

    @given(cmp=booleans(), eq=optional_bool, order=optional_bool)
    def test_mix(self, cmp, eq, order):
        """
        If cmp is not None, eq and order must be None and vice versa.
        """
        assume(eq is not None or order is not None)

        with pytest.raises(
            ValueError, match="Don't mix `cmp` with `eq' and `order`."
        ):
            _determine_attrs_eq_order(cmp, eq, order, True)


class TestDetermineAttribEqOrder:
    def test_default(self):
        """
        If all are set to None, set both eq and order to the passed default.
        """
        assert (42, None, 42, None) == _determine_attrib_eq_order(
            None, None, None, 42
        )

    def test_eq_callable_order_boolean(self):
        """
        eq=callable or order=callable need to transformed into eq/eq_key
        or order/order_key.
        """
        assert (True, str.lower, False, None) == _determine_attrib_eq_order(
            None, str.lower, False, True
        )

    def test_eq_callable_order_callable(self):
        """
        eq=callable or order=callable need to transformed into eq/eq_key
        or order/order_key.
        """
        assert (True, str.lower, True, abs) == _determine_attrib_eq_order(
            None, str.lower, abs, True
        )

    def test_eq_boolean_order_callable(self):
        """
        eq=callable or order=callable need to transformed into eq/eq_key
        or order/order_key.
        """
        assert (True, None, True, str.lower) == _determine_attrib_eq_order(
            None, True, str.lower, True
        )

    @pytest.mark.parametrize("eq", [True, False])
    def test_order_mirrors_eq_by_default(self, eq):
        """
        If order is None, it mirrors eq.
        """
        assert (eq, None, eq, None) == _determine_attrib_eq_order(
            None, eq, None, True
        )

    def test_order_without_eq(self):
        """
        eq=False, order=True raises a meaningful ValueError.
        """
        with pytest.raises(
            ValueError, match="`order` can only be True if `eq` is True too."
        ):
            _determine_attrib_eq_order(None, False, True, True)

    @given(cmp=booleans(), eq=optional_bool, order=optional_bool)
    def test_mix(self, cmp, eq, order):
        """
        If cmp is not None, eq and order must be None and vice versa.
        """
        assume(eq is not None or order is not None)

        with pytest.raises(
            ValueError, match="Don't mix `cmp` with `eq' and `order`."
        ):
            _determine_attrib_eq_order(cmp, eq, order, True)


class TestDocs:
    @pytest.mark.parametrize(
        "meth_name",
        [
            "__init__",
            "__repr__",
            "__eq__",
            "__lt__",
            "__le__",
            "__gt__",
            "__ge__",
        ],
    )
    def test_docs(self, meth_name):
        """
        Tests the presence and correctness of the documentation
        for the generated methods
        """

        @attr.s
        class A:
            pass

        if hasattr(A, "__qualname__"):
            method = getattr(A, meth_name)
            expected = f"Method generated by attrs for class {A.__qualname__}."
            assert expected == method.__doc__


class BareC:
    pass


class BareSlottedC:
    __slots__ = ()


class TestAutoDetect:
    @pytest.mark.parametrize("C", [BareC, BareSlottedC])
    def test_determine_detects_non_presence_correctly(self, C):
        """
        On an empty class, nothing should be detected.
        """
        assert True is _determine_whether_to_implement(
            C, None, True, ("__init__",)
        )
        assert True is _determine_whether_to_implement(
            C, None, True, ("__repr__",)
        )
        assert True is _determine_whether_to_implement(
            C, None, True, ("__eq__", "__ne__")
        )
        assert True is _determine_whether_to_implement(
            C, None, True, ("__le__", "__lt__", "__ge__", "__gt__")
        )

    def test_make_all_by_default(self, slots, frozen):
        """
        If nothing is there to be detected, imply init=True, repr=True,
        unsafe_hash=None, eq=True, order=True.
        """

        @attr.s(auto_detect=True, slots=slots, frozen=frozen)
        class C:
            x = attr.ib()

        i = C(1)
        o = object()

        assert i.__init__ is not o.__init__
        assert i.__repr__ is not o.__repr__
        assert i.__eq__ is not o.__eq__
        assert i.__ne__ is not o.__ne__
        assert i.__le__ is not o.__le__
        assert i.__lt__ is not o.__lt__
        assert i.__ge__ is not o.__ge__
        assert i.__gt__ is not o.__gt__

    def test_detect_auto_init(self, slots, frozen):
        """
        If auto_detect=True and an __init__ exists, don't write one.
        """

        @attr.s(auto_detect=True, slots=slots, frozen=frozen)
        class CI:
            x = attr.ib()

            def __init__(self):
                object.__setattr__(self, "x", 42)

        assert 42 == CI().x

    def test_detect_auto_repr(self, slots, frozen):
        """
        If auto_detect=True and an __repr__ exists, don't write one.
        """

        @attr.s(auto_detect=True, slots=slots, frozen=frozen)
        class C:
            x = attr.ib()

            def __repr__(self):
                return "hi"

        assert "hi" == repr(C(42))

    def test_hash_uses_eq(self, slots, frozen):
        """
        If eq is passed in, then __hash__ should use the eq callable
        to generate the hash code.
        """

        @attr.s(slots=slots, frozen=frozen, unsafe_hash=True)
        class C:
            x = attr.ib(eq=str)

        @attr.s(slots=slots, frozen=frozen, unsafe_hash=True)
        class D:
            x = attr.ib()

        # These hashes should be the same because 1 is turned into
        # string before hashing.
        assert hash(C("1")) == hash(C(1))
        assert hash(D("1")) != hash(D(1))

    def test_detect_auto_hash(self, slots, frozen):
        """
        If auto_detect=True and an __hash__ exists, don't write one.
        """

        @attr.s(auto_detect=True, slots=slots, frozen=frozen)
        class C:
            x = attr.ib()

            def __hash__(self):
                return 0xC0FFEE

        assert 0xC0FFEE == hash(C(42))

    def test_detect_auto_eq(self, slots, frozen):
        """
        If auto_detect=True and an __eq__ or an __ne__, exist, don't write one.
        """

        @attr.s(auto_detect=True, slots=slots, frozen=frozen)
        class C:
            x = attr.ib()

            def __eq__(self, o):
                raise ValueError("worked")

        with pytest.raises(ValueError, match="worked"):
            C(1) == C(1)

        @attr.s(auto_detect=True, slots=slots, frozen=frozen)
        class D:
            x = attr.ib()

            def __ne__(self, o):
                raise ValueError("worked")

        with pytest.raises(ValueError, match="worked"):
            D(1) != D(1)

    def test_detect_auto_order(self, slots, frozen):
        """
        If auto_detect=True and an __ge__, __gt__, __le__, or and __lt__ exist,
        don't write one.

        It's surprisingly difficult to test this programmatically, so we do it
        by hand.
        """

        def assert_not_set(cls, ex, meth_name):
            __tracebackhide__ = True

            a = getattr(cls, meth_name)
            if meth_name == ex:
                assert a == 42
            else:
                assert a is getattr(object, meth_name)

        def assert_none_set(cls, ex):
            __tracebackhide__ = True

            for m in ("le", "lt", "ge", "gt"):
                assert_not_set(cls, ex, "__" + m + "__")

        @attr.s(auto_detect=True, slots=slots, frozen=frozen)
        class LE:
            __le__ = 42

        @attr.s(auto_detect=True, slots=slots, frozen=frozen)
        class LT:
            __lt__ = 42

        @attr.s(auto_detect=True, slots=slots, frozen=frozen)
        class GE:
            __ge__ = 42

        @attr.s(auto_detect=True, slots=slots, frozen=frozen)
        class GT:
            __gt__ = 42

        assert_none_set(LE, "__le__")
        assert_none_set(LT, "__lt__")
        assert_none_set(GE, "__ge__")
        assert_none_set(GT, "__gt__")

    def test_override_init(self, slots, frozen):
        """
        If init=True is passed, ignore __init__.
        """

        @attr.s(init=True, auto_detect=True, slots=slots, frozen=frozen)
        class C:
            x = attr.ib()

            def __init__(self):
                pytest.fail("should not be called")

        assert C(1) == C(1)

    def test_override_repr(self, slots, frozen):
        """
        If repr=True is passed, ignore __repr__.
        """

        @attr.s(repr=True, auto_detect=True, slots=slots, frozen=frozen)
        class C:
            x = attr.ib()

            def __repr__(self):
                pytest.fail("should not be called")

        assert "C(x=1)" == repr(C(1))

    def test_override_hash(self, slots, frozen):
        """
        If unsafe_hash=True is passed, ignore __hash__.
        """

        @attr.s(unsafe_hash=True, auto_detect=True, slots=slots, frozen=frozen)
        class C:
            x = attr.ib()

            def __hash__(self):
                pytest.fail("should not be called")

        assert hash(C(1))

    def test_override_eq(self, slots, frozen):
        """
        If eq=True is passed, ignore __eq__ and __ne__.
        """

        @attr.s(eq=True, auto_detect=True, slots=slots, frozen=frozen)
        class C:
            x = attr.ib()

            def __eq__(self, o):
                pytest.fail("should not be called")

            def __ne__(self, o):
                pytest.fail("should not be called")

        assert C(1) == C(1)

    @pytest.mark.parametrize(
        ("eq", "order", "cmp"),
        [
            (True, None, None),
            (True, True, None),
            (None, True, None),
            (None, None, True),
        ],
    )
    def test_override_order(self, slots, frozen, eq, order, cmp):
        """
        If order=True is passed, ignore __le__, __lt__, __gt__, __ge__.

        eq=True and cmp=True both imply order=True so test it too.
        """

        def meth(self, o):
            pytest.fail("should not be called")

        @attr.s(
            cmp=cmp,
            order=order,
            eq=eq,
            auto_detect=True,
            slots=slots,
            frozen=frozen,
        )
        class C:
            x = attr.ib()
            __le__ = __lt__ = __gt__ = __ge__ = meth

        assert C(1) < C(2)
        assert C(1) <= C(2)
        assert C(2) > C(1)
        assert C(2) >= C(1)

    @pytest.mark.parametrize("first", [True, False])
    def test_total_ordering(self, slots, first):
        """
        functools.total_ordering works as expected if an order method and an eq
        method are detected.

        Ensure the order doesn't matter.
        """

        class C:
            x = attr.ib()
            own_eq_called = attr.ib(default=False)
            own_le_called = attr.ib(default=False)

            def __eq__(self, o):
                self.own_eq_called = True
                return self.x == o.x

            def __le__(self, o):
                self.own_le_called = True
                return self.x <= o.x

        if first:
            C = functools.total_ordering(
                attr.s(auto_detect=True, slots=slots)(C)
            )
        else:
            C = attr.s(auto_detect=True, slots=slots)(
                functools.total_ordering(C)
            )

        c1, c2 = C(1), C(2)

        assert c1 < c2
        assert c1.own_le_called

        c1, c2 = C(1), C(2)

        assert c2 > c1
        assert c2.own_le_called

        c1, c2 = C(1), C(2)

        assert c2 != c1
        assert c1 == c1

        assert c1.own_eq_called

    def test_detects_setstate_getstate(self, slots):
        """
        __getstate__ and __setstate__ are not overwritten if either is present.
        """

        @attr.s(slots=slots, auto_detect=True)
        class C:
            def __getstate__(self):
                return ("hi",)

        assert getattr(object, "__setstate__", None) is getattr(
            C, "__setstate__", None
        )

        @attr.s(slots=slots, auto_detect=True)
        class C:
            called = attr.ib(False)

            def __setstate__(self, state):
                self.called = True

        i = C()

        assert False is i.called

        i.__setstate__(())

        assert True is i.called
        assert getattr(object, "__getstate__", None) is getattr(
            C, "__getstate__", None
        )

    @pytest.mark.skipif(PY_3_10_PLUS, reason="Pre-3.10 only.")
    def test_match_args_pre_310(self):
        """
        __match_args__ is not created on Python versions older than 3.10.
        """

        @attr.s
        class C:
            a = attr.ib()

        assert None is getattr(C, "__match_args__", None)


@pytest.mark.skipif(
    not PY_3_10_PLUS, reason="Structural pattern matching is 3.10+"
)
class TestMatchArgs:
    """
    Tests for match_args and __match_args__ generation.
    """

    def test_match_args(self):
        """
        __match_args__ is created by default on Python 3.10.
        """

        @attr.define
        class C:
            a = attr.field()

        assert ("a",) == C.__match_args__

    def test_explicit_match_args(self):
        """
        A custom __match_args__ set is not overwritten.
        """

        ma = ()

        @attr.define
        class C:
            a = attr.field()
            __match_args__ = ma

        assert C(42).__match_args__ is ma

    @pytest.mark.parametrize("match_args", [True, False])
    def test_match_args_attr_set(self, match_args):
        """
        __match_args__ is set depending on match_args.
        """

        @attr.define(match_args=match_args)
        class C:
            a = attr.field()

        if match_args:
            assert hasattr(C, "__match_args__")
        else:
            assert not hasattr(C, "__match_args__")

    def test_match_args_kw_only(self):
        """
        kw_only classes don't generate __match_args__.
        kw_only fields are not included in __match_args__.
        """

        @attr.define
        class C:
            a = attr.field(kw_only=True)
            b = attr.field()

        assert C.__match_args__ == ("b",)

        @attr.define(kw_only=True)
        class C:
            a = attr.field()
            b = attr.field()

        assert C.__match_args__ == ()

    def test_match_args_argument(self):
        """
        match_args being False with inheritance.
        """

        @attr.define(match_args=False)
        class X:
            a = attr.field()

        assert "__match_args__" not in X.__dict__

        @attr.define(match_args=False)
        class Y:
            a = attr.field()
            __match_args__ = ("b",)

        assert Y.__match_args__ == ("b",)

        @attr.define(match_args=False)
        class Z(Y):
            z = attr.field()

        assert Z.__match_args__ == ("b",)

        @attr.define
        class A:
            a = attr.field()
            z = attr.field()

        @attr.define(match_args=False)
        class B(A):
            b = attr.field()

        assert B.__match_args__ == ("a", "z")

    def test_make_class(self):
        """
        match_args generation with make_class.
        """

        C1 = make_class("C1", ["a", "b"])
        assert ("a", "b") == C1.__match_args__

        C1 = make_class("C1", ["a", "b"], match_args=False)
        assert not hasattr(C1, "__match_args__")

        C1 = make_class("C1", ["a", "b"], kw_only=True)
        assert () == C1.__match_args__

        C1 = make_class("C1", {"a": attr.ib(kw_only=True), "b": attr.ib()})
        assert ("b",) == C1.__match_args__


# SPDX-License-Identifier: MIT

"""
Tests for `attr.validators`.
"""

import re

import pytest

import attr

from attr import _config, fields, has
from attr import validators as validator_module
from attr.validators import (
    _subclass_of,
    and_,
    deep_iterable,
    deep_mapping,
    ge,
    gt,
    in_,
    instance_of,
    is_callable,
    le,
    lt,
    matches_re,
    max_len,
    min_len,
    not_,
    optional,
    or_,
)

from .utils import simple_attr


class TestDisableValidators:
    @pytest.fixture(autouse=True)
    def _reset_default(self):
        """
        Make sure validators are always enabled after a test.
        """
        yield
        _config._run_validators = True

    def test_default(self):
        """
        Run validators by default.
        """
        assert _config._run_validators is True

    @pytest.mark.parametrize(
        ("value", "expected"), [(True, False), (False, True)]
    )
    def test_set_validators_disabled(self, value, expected):
        """
        Sets `_run_validators`.
        """
        validator_module.set_disabled(value)

        assert _config._run_validators is expected

    @pytest.mark.parametrize(
        ("value", "expected"), [(True, False), (False, True)]
    )
    def test_disabled(self, value, expected):
        """
        Returns `_run_validators`.
        """
        _config._run_validators = value

        assert validator_module.get_disabled() is expected

    def test_disabled_ctx(self):
        """
        The `disabled` context manager disables running validators,
        but only within its context.
        """
        assert _config._run_validators is True

        with validator_module.disabled():
            assert _config._run_validators is False

        assert _config._run_validators is True

    def test_disabled_ctx_with_errors(self):
        """
        Running validators is re-enabled even if an error is raised.
        """
        assert _config._run_validators is True

        with pytest.raises(ValueError), validator_module.disabled():
            assert _config._run_validators is False

            raise ValueError("haha!")

        assert _config._run_validators is True


class TestInstanceOf:
    """
    Tests for `instance_of`.
    """

    def test_in_all(self):
        """
        Verify that this validator is in ``__all__``.
        """
        assert instance_of.__name__ in validator_module.__all__

    def test_success(self):
        """
        Nothing happens if types match.
        """
        v = instance_of(int)
        v(None, simple_attr("test"), 42)

    def test_subclass(self):
        """
        Subclasses are accepted too.
        """
        v = instance_of(int)
        # yep, bools are a subclass of int :(
        v(None, simple_attr("test"), True)

    def test_fail(self):
        """
        Raises `TypeError` on wrong types.
        """
        v = instance_of(int)
        a = simple_attr("test")
        with pytest.raises(TypeError) as e:
            v(None, a, "42")
        assert (
            "'test' must be <class 'int'> (got '42' that is a <class 'str'>).",
            a,
            int,
            "42",
        ) == e.value.args

    def test_repr(self):
        """
        Returned validator has a useful `__repr__`.
        """
        v = instance_of(int)
        assert ("<instance_of validator for type <class 'int'>>") == repr(v)


class TestMatchesRe:
    """
    Tests for `matches_re`.
    """

    def test_in_all(self):
        """
        validator is in ``__all__``.
        """
        assert matches_re.__name__ in validator_module.__all__

    def test_match(self):
        """
        Silent on matches, raises ValueError on mismatches.
        """

        @attr.s
        class ReTester:
            str_match = attr.ib(validator=matches_re("a|ab"))

        ReTester("ab")  # shouldn't raise exceptions
        with pytest.raises(TypeError):
            ReTester(1)
        with pytest.raises(ValueError):
            ReTester("1")
        with pytest.raises(ValueError):
            ReTester("a1")

    def test_flags(self):
        """
        Flags are propagated to the match function.
        """

        @attr.s
        class MatchTester:
            val = attr.ib(validator=matches_re("a", re.IGNORECASE, re.match))

        MatchTester("A1")  # test flags and using re.match

    def test_precompiled_pattern(self):
        """
        Pre-compiled patterns are accepted.
        """
        pattern = re.compile("a")

        @attr.s
        class RePatternTester:
            val = attr.ib(validator=matches_re(pattern))

        RePatternTester("a")

    def test_precompiled_pattern_no_flags(self):
        """
        A pre-compiled pattern cannot be combined with a 'flags' argument.
        """
        pattern = re.compile("")

        with pytest.raises(
            TypeError, match="can only be used with a string pattern"
        ):
            matches_re(pattern, flags=re.IGNORECASE)

    def test_different_func(self):
        """
        Changing the match functions works.
        """

        @attr.s
        class SearchTester:
            val = attr.ib(validator=matches_re("a", 0, re.search))

        SearchTester("bab")  # re.search will match

    def test_catches_invalid_func(self):
        """
        Invalid match functions are caught.
        """
        with pytest.raises(ValueError) as ei:
            matches_re("a", 0, lambda: None)

        assert (
            "'func' must be one of None, fullmatch, match, search."
            == ei.value.args[0]
        )

    @pytest.mark.parametrize(
        "func", [None, getattr(re, "fullmatch", None), re.match, re.search]
    )
    def test_accepts_all_valid_func(self, func):
        """
        Every valid match function is accepted.
        """
        matches_re("a", func=func)

    def test_repr(self):
        """
        __repr__ is meaningful.
        """
        assert repr(matches_re("a")).startswith(
            "<matches_re validator for pattern"
        )


def always_pass(_, __, ___):
    """
    Toy validator that always passes.
    """


def always_fail(_, __, ___):
    """
    Toy validator that always fails.
    """
    0 / 0


class TestAnd:
    def test_in_all(self):
        """
        Verify that this validator is in ``__all__``.
        """
        assert and_.__name__ in validator_module.__all__

    def test_success(self):
        """
        Succeeds if all wrapped validators succeed.
        """
        v = and_(instance_of(int), always_pass)

        v(None, simple_attr("test"), 42)

    def test_fail(self):
        """
        Fails if any wrapped validator fails.
        """
        v = and_(instance_of(int), always_fail)

        with pytest.raises(ZeroDivisionError):
            v(None, simple_attr("test"), 42)

    def test_sugar(self):
        """
        `and_(v1, v2, v3)` and `[v1, v2, v3]` are equivalent.
        """

        @attr.s
        class C:
            a1 = attr.ib("a1", validator=and_(instance_of(int)))
            a2 = attr.ib("a2", validator=[instance_of(int)])

        assert C.__attrs_attrs__[0].validator == C.__attrs_attrs__[1].validator


@pytest.mark.parametrize(
    "validator",
    [
        instance_of(int),
        [always_pass, instance_of(int)],
        (always_pass, instance_of(int)),
    ],
)
class TestOptional:
    """
    Tests for `optional`.
    """

    def test_in_all(self, validator):
        """
        Verify that this validator is in ``__all__``.
        """
        assert optional.__name__ in validator_module.__all__

    def test_success(self, validator):
        """
        Nothing happens if validator succeeds.
        """
        v = optional(validator)
        v(None, simple_attr("test"), 42)

    def test_success_with_none(self, validator):
        """
        Nothing happens if None.
        """
        v = optional(validator)
        v(None, simple_attr("test"), None)

    def test_fail(self, validator):
        """
        Raises `TypeError` on wrong types.
        """
        v = optional(validator)
        a = simple_attr("test")
        with pytest.raises(TypeError) as e:
            v(None, a, "42")
        assert (
            "'test' must be <class 'int'> (got '42' that is a <class 'str'>).",
            a,
            int,
            "42",
        ) == e.value.args

    def test_repr(self, validator):
        """
        Returned validator has a useful `__repr__`.
        """
        v = optional(validator)

        if isinstance(validator, list):
            repr_s = (
                f"<optional validator for _AndValidator(_validators=[{always_pass!r}, "
                "<instance_of validator for type <class 'int'>>]) or None>"
            )
        elif isinstance(validator, tuple):
            repr_s = (
                f"<optional validator for _AndValidator(_validators=({always_pass!r}, "
                "<instance_of validator for type <class 'int'>>)) or None>"
            )
        else:
            repr_s = (
                "<optional validator for <instance_of validator for type "
                "<class 'int'>> or None>"
            )

        assert repr_s == repr(v)


class TestIn_:
    """
    Tests for `in_`.
    """

    def test_in_all(self):
        """
        Verify that this validator is in ``__all__``.
        """
        assert in_.__name__ in validator_module.__all__

    def test_success_with_value(self):
        """
        If the value is in our options, nothing happens.
        """
        v = in_([1, 2, 3])
        a = simple_attr("test")

        v(1, a, 3)

    def test_fail(self):
        """
        Raise ValueError if the value is outside our options.
        """
        v = in_([1, 2, 3])
        a = simple_attr("test")

        with pytest.raises(ValueError) as e:
            v(None, a, None)

        assert (
            "'test' must be in [1, 2, 3] (got None)",
            a,
            [1, 2, 3],
            None,
        ) == e.value.args

    def test_fail_with_string(self):
        """
        Raise ValueError if the value is outside our options when the
        options are specified as a string and the value is not a string.
        """
        v = in_("abc")
        a = simple_attr("test")
        with pytest.raises(ValueError) as e:
            v(None, a, None)
        assert (
            "'test' must be in 'abc' (got None)",
            a,
            "abc",
            None,
        ) == e.value.args

    def test_repr(self):
        """
        Returned validator has a useful `__repr__`.
        """
        v = in_([3, 4, 5])
        assert ("<in_ validator with options [3, 4, 5]>") == repr(v)

    def test_is_hashable(self):
        """
        `in_` is hashable, so fields using it can be used with the include and
        exclude filters.
        """

        @attr.s
        class C:
            x: int = attr.ib(validator=attr.validators.in_({1, 2}))

        i = C(2)

        attr.asdict(i, filter=attr.filters.include(lambda val: True))
        attr.asdict(i, filter=attr.filters.exclude(lambda val: True))


@pytest.fixture(
    name="member_validator",
    params=(
        instance_of(int),
        [always_pass, instance_of(int)],
        (always_pass, instance_of(int)),
    ),
    scope="module",
)
def _member_validator(request):
    """
    Provides sample `member_validator`s for some tests in `TestDeepIterable`
    """
    return request.param


class TestDeepIterable:
    """
    Tests for `deep_iterable`.
    """

    def test_in_all(self):
        """
        Verify that this validator is in ``__all__``.
        """
        assert deep_iterable.__name__ in validator_module.__all__

    def test_success_member_only(self, member_validator):
        """
        If the member validator succeeds and the iterable validator is not set,
        nothing happens.
        """
        v = deep_iterable(member_validator)
        a = simple_attr("test")
        v(None, a, [42])

    def test_success_member_and_iterable(self, member_validator):
        """
        If both the member and iterable validators succeed, nothing happens.
        """
        iterable_validator = instance_of(list)
        v = deep_iterable(member_validator, iterable_validator)
        a = simple_attr("test")
        v(None, a, [42])

    @pytest.mark.parametrize(
        ("member_validator", "iterable_validator"),
        [
            (instance_of(int), 42),
            (42, instance_of(list)),
            (42, 42),
            (42, None),
            ([instance_of(int), 42], 42),
            ([42, instance_of(int)], 42),
        ],
    )
    def test_noncallable_validators(
        self, member_validator, iterable_validator
    ):
        """
        Raise `TypeError` if any validators are not callable.
        """
        with pytest.raises(TypeError) as e:
            deep_iterable(member_validator, iterable_validator)
        value = 42
        message = (
            f"must be callable (got {value} that is a {value.__class__})."
        )

        assert message in e.value.args[0]
        assert value == e.value.args[1]
        assert message in e.value.msg
        assert value == e.value.value

    def test_fail_invalid_member(self, member_validator):
        """
        Raise member validator error if an invalid member is found.
        """
        v = deep_iterable(member_validator)
        a = simple_attr("test")
        with pytest.raises(TypeError):
            v(None, a, [42, "42"])

    def test_fail_invalid_iterable(self, member_validator):
        """
        Raise iterable validator error if an invalid iterable is found.
        """
        member_validator = instance_of(int)
        iterable_validator = instance_of(tuple)
        v = deep_iterable(member_validator, iterable_validator)
        a = simple_attr("test")
        with pytest.raises(TypeError):
            v(None, a, [42])

    def test_fail_invalid_member_and_iterable(self, member_validator):
        """
        Raise iterable validator error if both the iterable
        and a member are invalid.
        """
        iterable_validator = instance_of(tuple)
        v = deep_iterable(member_validator, iterable_validator)
        a = simple_attr("test")
        with pytest.raises(TypeError):
            v(None, a, [42, "42"])

    def test_repr_member_only(self):
        """
        Returned validator has a useful `__repr__`
        when only member validator is set.
        """
        member_validator = instance_of(int)
        member_repr = "<instance_of validator for type <class 'int'>>"
        v = deep_iterable(member_validator)
        expected_repr = (
            f"<deep_iterable validator for iterables of {member_repr}>"
        )
        assert expected_repr == repr(v)

    def test_repr_member_only_sequence(self):
        """
        Returned validator has a useful `__repr__`
        when only member validator is set and the member validator is a list of
        validators
        """
        member_validator = [always_pass, instance_of(int)]
        member_repr = (
            f"_AndValidator(_validators=({always_pass!r}, "
            "<instance_of validator for type <class 'int'>>))"
        )
        v = deep_iterable(member_validator)
        expected_repr = (
            f"<deep_iterable validator for iterables of {member_repr}>"
        )
        assert expected_repr == repr(v)

    def test_repr_member_and_iterable(self):
        """
        Returned validator has a useful `__repr__` when both member
        and iterable validators are set.
        """
        member_validator = instance_of(int)
        member_repr = "<instance_of validator for type <class 'int'>>"
        iterable_validator = instance_of(list)
        iterable_repr = "<instance_of validator for type <class 'list'>>"
        v = deep_iterable(member_validator, iterable_validator)
        expected_repr = (
            "<deep_iterable validator for"
            f" {iterable_repr} iterables of {member_repr}>"
        )
        assert expected_repr == repr(v)

    def test_repr_sequence_member_and_iterable(self):
        """
        Returned validator has a useful `__repr__` when both member
        and iterable validators are set and the member validator is a list of
        validators
        """
        member_validator = [always_pass, instance_of(int)]
        member_repr = (
            f"_AndValidator(_validators=({always_pass!r}, "
            "<instance_of validator for type <class 'int'>>))"
        )
        iterable_validator = instance_of(list)
        iterable_repr = "<instance_of validator for type <class 'list'>>"
        v = deep_iterable(member_validator, iterable_validator)
        expected_repr = (
            "<deep_iterable validator for"
            f" {iterable_repr} iterables of {member_repr}>"
        )

        assert expected_repr == repr(v)


class TestDeepMapping:
    """
    Tests for `deep_mapping`.
    """

    def test_in_all(self):
        """
        Verify that this validator is in ``__all__``.
        """
        assert deep_mapping.__name__ in validator_module.__all__

    def test_success(self):
        """
        If both the key and value validators succeed, nothing happens.
        """
        key_validator = instance_of(str)
        value_validator = instance_of(int)
        v = deep_mapping(key_validator, value_validator)
        a = simple_attr("test")
        v(None, a, {"a": 6, "b": 7})

    @pytest.mark.parametrize(
        ("key_validator", "value_validator", "mapping_validator"),
        [
            (42, instance_of(int), None),
            (instance_of(str), 42, None),
            (instance_of(str), instance_of(int), 42),
            (42, 42, None),
            (42, 42, 42),
        ],
    )
    def test_noncallable_validators(
        self, key_validator, value_validator, mapping_validator
    ):
        """
        Raise `TypeError` if any validators are not callable.
        """
        with pytest.raises(TypeError) as e:
            deep_mapping(key_validator, value_validator, mapping_validator)

        value = 42
        message = (
            f"must be callable (got {value} that is a {value.__class__})."
        )

        assert message in e.value.args[0]
        assert value == e.value.args[1]
        assert message in e.value.msg
        assert value == e.value.value

    def test_fail_invalid_mapping(self):
        """
        Raise `TypeError` if mapping validator fails.
        """
        key_validator = instance_of(str)
        value_validator = instance_of(int)
        mapping_validator = instance_of(dict)
        v = deep_mapping(key_validator, value_validator, mapping_validator)
        a = simple_attr("test")
        with pytest.raises(TypeError):
            v(None, a, None)

    def test_fail_invalid_key(self):
        """
        Raise key validator error if an invalid key is found.
        """
        key_validator = instance_of(str)
        value_validator = instance_of(int)
        v = deep_mapping(key_validator, value_validator)
        a = simple_attr("test")
        with pytest.raises(TypeError):
            v(None, a, {"a": 6, 42: 7})

    def test_fail_invalid_member(self):
        """
        Raise key validator error if an invalid member value is found.
        """
        key_validator = instance_of(str)
        value_validator = instance_of(int)
        v = deep_mapping(key_validator, value_validator)
        a = simple_attr("test")
        with pytest.raises(TypeError):
            v(None, a, {"a": "6", "b": 7})

    def test_repr(self):
        """
        Returned validator has a useful `__repr__`.
        """
        key_validator = instance_of(str)
        key_repr = "<instance_of validator for type <class 'str'>>"
        value_validator = instance_of(int)
        value_repr = "<instance_of validator for type <class 'int'>>"
        v = deep_mapping(key_validator, value_validator)
        expected_repr = (
            "<deep_mapping validator for objects mapping "
            f"{key_repr} to {value_repr}>"
        )
        assert expected_repr == repr(v)


class TestIsCallable:
    """
    Tests for `is_callable`.
    """

    def test_in_all(self):
        """
        Verify that this validator is in ``__all__``.
        """
        assert is_callable.__name__ in validator_module.__all__

    def test_success(self):
        """
        If the value is callable, nothing happens.
        """
        v = is_callable()
        a = simple_attr("test")
        v(None, a, isinstance)

    def test_fail(self):
        """
        Raise TypeError if the value is not callable.
        """
        v = is_callable()
        a = simple_attr("test")
        with pytest.raises(TypeError) as e:
            v(None, a, None)

        value = None
        message = "'test' must be callable (got {value} that is a {type_})."
        expected_message = message.format(value=value, type_=value.__class__)

        assert expected_message == e.value.args[0]
        assert value == e.value.args[1]
        assert expected_message == e.value.msg
        assert value == e.value.value

    def test_repr(self):
        """
        Returned validator has a useful `__repr__`.
        """
        v = is_callable()
        assert "<is_callable validator>" == repr(v)

    def test_exception_repr(self):
        """
        Verify that NotCallableError exception has a useful `__str__`.
        """
        from attr.exceptions import NotCallableError

        instance = NotCallableError(msg="Some Message", value=42)
        assert "Some Message" == str(instance)


def test_hashability():
    """
    Validator classes are hashable.
    """
    for obj_name in dir(validator_module):
        obj = getattr(validator_module, obj_name)
        if not has(obj):
            continue
        hash_func = getattr(obj, "__hash__", None)
        assert hash_func is not None
        assert hash_func is not object.__hash__


class TestLtLeGeGt:
    """
    Tests for `Lt, Le, Ge, Gt`.
    """

    BOUND = 4

    def test_in_all(self):
        """
        validator is in ``__all__``.
        """
        assert all(
            f.__name__ in validator_module.__all__ for f in [lt, le, ge, gt]
        )

    @pytest.mark.parametrize("v", [lt, le, ge, gt])
    def test_retrieve_bound(self, v):
        """
        The configured bound for the comparison can be extracted from the
        Attribute.
        """

        @attr.s
        class Tester:
            value = attr.ib(validator=v(self.BOUND))

        assert fields(Tester).value.validator.bound == self.BOUND

    @pytest.mark.parametrize(
        ("v", "value"),
        [
            (lt, 3),
            (le, 3),
            (le, 4),
            (ge, 4),
            (ge, 5),
            (gt, 5),
        ],
    )
    def test_check_valid(self, v, value):
        """Silent if value {op} bound."""

        @attr.s
        class Tester:
            value = attr.ib(validator=v(self.BOUND))

        Tester(value)  # shouldn't raise exceptions

    @pytest.mark.parametrize(
        ("v", "value"),
        [
            (lt, 4),
            (le, 5),
            (ge, 3),
            (gt, 4),
        ],
    )
    def test_check_invalid(self, v, value):
        """Raise ValueError if value {op} bound."""

        @attr.s
        class Tester:
            value = attr.ib(validator=v(self.BOUND))

        with pytest.raises(ValueError):
            Tester(value)

    @pytest.mark.parametrize("v", [lt, le, ge, gt])
    def test_repr(self, v):
        """
        __repr__ is meaningful.
        """
        nv = v(23)
        assert repr(nv) == f"<Validator for x {nv.compare_op} {23}>"


class TestMaxLen:
    """
    Tests for `max_len`.
    """

    MAX_LENGTH = 4

    def test_in_all(self):
        """
        validator is in ``__all__``.
        """
        assert max_len.__name__ in validator_module.__all__

    def test_retrieve_max_len(self):
        """
        The configured max. length can be extracted from the Attribute
        """

        @attr.s
        class Tester:
            value = attr.ib(validator=max_len(self.MAX_LENGTH))

        assert fields(Tester).value.validator.max_length == self.MAX_LENGTH

    @pytest.mark.parametrize(
        "value",
        [
            "",
            "foo",
            "spam",
            [],
            list(range(MAX_LENGTH)),
            {"spam": 3, "eggs": 4},
        ],
    )
    def test_check_valid(self, value):
        """
        Silent if len(value) <= max_len.
        Values can be strings and other iterables.
        """

        @attr.s
        class Tester:
            value = attr.ib(validator=max_len(self.MAX_LENGTH))

        Tester(value)  # shouldn't raise exceptions

    @pytest.mark.parametrize(
        "value",
        [
            "bacon",
            list(range(6)),
        ],
    )
    def test_check_invalid(self, value):
        """
        Raise ValueError if len(value) > max_len.
        """

        @attr.s
        class Tester:
            value = attr.ib(validator=max_len(self.MAX_LENGTH))

        with pytest.raises(ValueError):
            Tester(value)

    def test_repr(self):
        """
        __repr__ is meaningful.
        """
        assert repr(max_len(23)) == "<max_len validator for 23>"


class TestMinLen:
    """
    Tests for `min_len`.
    """

    MIN_LENGTH = 2

    def test_in_all(self):
        """
        validator is in ``__all__``.
        """
        assert min_len.__name__ in validator_module.__all__

    def test_retrieve_min_len(self):
        """
        The configured min. length can be extracted from the Attribute
        """

        @attr.s
        class Tester:
            value = attr.ib(validator=min_len(self.MIN_LENGTH))

        assert fields(Tester).value.validator.min_length == self.MIN_LENGTH

    @pytest.mark.parametrize(
        "value",
        [
            "foo",
            "spam",
            list(range(MIN_LENGTH)),
            {"spam": 3, "eggs": 4},
        ],
    )
    def test_check_valid(self, value):
        """
        Silent if len(value) => min_len.
        Values can be strings and other iterables.
        """

        @attr.s
        class Tester:
            value = attr.ib(validator=min_len(self.MIN_LENGTH))

        Tester(value)  # shouldn't raise exceptions

    @pytest.mark.parametrize(
        "value",
        [
            "",
            list(range(1)),
        ],
    )
    def test_check_invalid(self, value):
        """
        Raise ValueError if len(value) < min_len.
        """

        @attr.s
        class Tester:
            value = attr.ib(validator=min_len(self.MIN_LENGTH))

        with pytest.raises(ValueError):
            Tester(value)

    def test_repr(self):
        """
        __repr__ is meaningful.
        """
        assert repr(min_len(23)) == "<min_len validator for 23>"


class TestSubclassOf:
    """
    Tests for `_subclass_of`.
    """

    def test_success(self):
        """
        Nothing happens if classes match.
        """
        v = _subclass_of(int)
        v(None, simple_attr("test"), int)

    def test_subclass(self):
        """
        Subclasses are accepted too.
        """
        v = _subclass_of(int)
        # yep, bools are a subclass of int :(
        v(None, simple_attr("test"), bool)

    def test_fail(self):
        """
        Raises `TypeError` on wrong types.
        """
        v = _subclass_of(int)
        a = simple_attr("test")
        with pytest.raises(TypeError) as e:
            v(None, a, str)
        assert (
            "'test' must be a subclass of <class 'int'> (got <class 'str'>).",
            a,
            int,
            str,
        ) == e.value.args

    def test_repr(self):
        """
        Returned validator has a useful `__repr__`.
        """
        v = _subclass_of(int)
        assert ("<subclass_of validator for type <class 'int'>>") == repr(v)


class TestNot_:
    """
    Tests for `not_`.
    """

    DEFAULT_EXC_TYPES = (ValueError, TypeError)

    def test_not_all(self):
        """
        The validator is in ``__all__``.
        """
        assert not_.__name__ in validator_module.__all__

    def test_repr(self):
        """
        Returned validator has a useful `__repr__`.
        """
        wrapped = in_([3, 4, 5])

        v = not_(wrapped)

        assert (
            f"<not_ validator wrapping {wrapped!r}, capturing {v.exc_types!r}>"
        ) == repr(v)

    def test_success_because_fails(self):
        """
        If the wrapped validator fails, we're happy.
        """

        def always_fails(inst, attr, value):
            raise ValueError("always fails")

        v = not_(always_fails)
        a = simple_attr("test")
        input_value = 3

        v(1, a, input_value)

    def test_fails_because_success(self):
        """
        If the wrapped validator doesn't fail, not_ should fail.
        """

        def always_passes(inst, attr, value):
            pass

        v = not_(always_passes)
        a = simple_attr("test")
        input_value = 3

        with pytest.raises(ValueError) as e:
            v(1, a, input_value)

        assert (
            (
                f"not_ validator child '{always_passes!r}' did not raise a captured error"
            ),
            a,
            always_passes,
            input_value,
            self.DEFAULT_EXC_TYPES,
        ) == e.value.args

    def test_composable_with_in_pass(self):
        """
        Check something is ``not in`` something else.
        """
        v = not_(in_("abc"))
        a = simple_attr("test")
        input_value = "d"

        v(None, a, input_value)

    def test_composable_with_in_fail(self):
        """
        Check something is ``not in`` something else, but it is, so fail.
        """
        wrapped = in_("abc")
        v = not_(wrapped)
        a = simple_attr("test")
        input_value = "b"

        with pytest.raises(ValueError) as e:
            v(None, a, input_value)

        assert (
            (
                "not_ validator child '{!r}' did not raise a captured error"
            ).format(in_("abc")),
            a,
            wrapped,
            input_value,
            self.DEFAULT_EXC_TYPES,
        ) == e.value.args

    def test_composable_with_matches_re_pass(self):
        """
        Check something does not match a regex.
        """
        v = not_(matches_re("[a-z]{3}"))
        a = simple_attr("test")
        input_value = "spam"

        v(None, a, input_value)

    def test_composable_with_matches_re_fail(self):
        """
        Check something does not match a regex, but it does, so fail.
        """
        wrapped = matches_re("[a-z]{3}")
        v = not_(wrapped)
        a = simple_attr("test")
        input_value = "egg"

        with pytest.raises(ValueError) as e:
            v(None, a, input_value)

        assert (
            (
                f"not_ validator child '{wrapped!r}' did not raise a captured error"
            ),
            a,
            wrapped,
            input_value,
            self.DEFAULT_EXC_TYPES,
        ) == e.value.args

    def test_composable_with_instance_of_pass(self):
        """
        Check something is not a type. This validator raises a TypeError,
        rather than a ValueError like the others.
        """
        v = not_(instance_of((int, float)))
        a = simple_attr("test")

        v(None, a, "spam")

    def test_composable_with_instance_of_fail(self):
        """
        Check something is not a type, but it is, so fail.
        """
        wrapped = instance_of((int, float))
        v = not_(wrapped)
        a = simple_attr("test")
        input_value = 2.718281828

        with pytest.raises(ValueError) as e:
            v(None, a, input_value)

        assert (
            (
                f"not_ validator child '{instance_of((int, float))!r}' did not raise a captured error"
            ),
            a,
            wrapped,
            input_value,
            self.DEFAULT_EXC_TYPES,
        ) == e.value.args

    def test_custom_capture_match(self):
        """
        Match a custom exception provided to `not_`
        """
        v = not_(in_("abc"), exc_types=ValueError)
        a = simple_attr("test")

        v(None, a, "d")

    def test_custom_capture_miss(self):
        """
        If the exception doesn't match, the underlying raise comes through
        """

        class MyError(Exception):
            """:("""

        wrapped = in_("abc")
        v = not_(wrapped, exc_types=MyError)
        a = simple_attr("test")
        input_value = "d"

        with pytest.raises(ValueError) as e:
            v(None, a, input_value)

        # get the underlying exception to compare
        with pytest.raises(Exception) as e_from_wrapped:
            wrapped(None, a, input_value)
        assert e_from_wrapped.value.args == e.value.args

    def test_custom_msg(self):
        """
        If provided, use the custom message in the raised error
        """
        custom_msg = "custom message!"
        wrapped = in_("abc")
        v = not_(wrapped, msg=custom_msg)
        a = simple_attr("test")
        input_value = "a"

        with pytest.raises(ValueError) as e:
            v(None, a, input_value)

        assert (
            custom_msg,
            a,
            wrapped,
            input_value,
            self.DEFAULT_EXC_TYPES,
        ) == e.value.args

    def test_bad_exception_args(self):
        """
        Malformed exception arguments
        """
        wrapped = in_("abc")

        with pytest.raises(TypeError) as e:
            not_(wrapped, exc_types=(str, int))

        assert (
            "'exc_types' must be a subclass of <class 'Exception'> "
            "(got <class 'str'>)."
        ) == e.value.args[0]


class TestOr:
    def test_in_all(self):
        """
        Verify that this validator is in ``__all__``.
        """
        assert or_.__name__ in validator_module.__all__

    def test_success(self):
        """
        Succeeds if at least one of wrapped validators succeed.
        """
        v = or_(instance_of(str), always_pass)

        v(None, simple_attr("test"), 42)

    def test_fail(self):
        """
        Fails if all wrapped validators fail.
        """
        v = or_(instance_of(str), always_fail)

        with pytest.raises(ValueError):
            v(None, simple_attr("test"), 42)

    def test_repr(self):
        """
        Returned validator has a useful `__repr__`.
        """
        v = or_(instance_of(int), instance_of(str))
        assert (
            "<or validator wrapping (<instance_of validator for type "
            "<class 'int'>>, <instance_of validator for type <class 'str'>>)>"
        ) == repr(v)


# SPDX-License-Identifier: MIT


class TestImportStar:
    def test_from_attr_import_star(self):
        """
        import * from attr
        """
        # attr_import_star contains `from attr import *`, which cannot
        # be done here because *-imports are only allowed on module level.
        from . import attr_import_star  # noqa: F401


# SPDX-License-Identifier: MIT

from __future__ import annotations

from datetime import datetime

import pytest

import attr


class TestTransformHook:
    """
    Tests for `attrs(tranform_value_serializer=func)`
    """

    def test_hook_applied(self):
        """
        The transform hook is applied to all attributes.  Types can be missing,
        explicitly set, or annotated.
        """
        results = []

        def hook(cls, attribs):
            attr.resolve_types(cls, attribs=attribs)
            results[:] = [(a.name, a.type) for a in attribs]
            return attribs

        @attr.s(field_transformer=hook)
        class C:
            x = attr.ib()
            y = attr.ib(type=int)
            z: float = attr.ib()

        assert [("x", None), ("y", int), ("z", float)] == results

    def test_hook_applied_auto_attrib(self):
        """
        The transform hook is applied to all attributes and type annotations
        are detected.
        """
        results = []

        def hook(cls, attribs):
            attr.resolve_types(cls, attribs=attribs)
            results[:] = [(a.name, a.type) for a in attribs]
            return attribs

        @attr.s(auto_attribs=True, field_transformer=hook)
        class C:
            x: int
            y: str = attr.ib()

        assert [("x", int), ("y", str)] == results

    def test_hook_applied_modify_attrib(self):
        """
        The transform hook can modify attributes.
        """

        def hook(cls, attribs):
            attr.resolve_types(cls, attribs=attribs)
            return [a.evolve(converter=a.type) for a in attribs]

        @attr.s(auto_attribs=True, field_transformer=hook)
        class C:
            x: int = attr.ib(converter=int)
            y: float

        c = C(x="3", y="3.14")

        assert C(x=3, y=3.14) == c

    def test_hook_remove_field(self):
        """
        It is possible to remove fields via the hook.
        """

        def hook(cls, attribs):
            attr.resolve_types(cls, attribs=attribs)
            return [a for a in attribs if a.type is not int]

        @attr.s(auto_attribs=True, field_transformer=hook)
        class C:
            x: int
            y: float

        assert {"y": 2.7} == attr.asdict(C(2.7))

    def test_hook_add_field(self):
        """
        It is possible to add fields via the hook.
        """

        def hook(cls, attribs):
            a1 = attribs[0]
            a2 = a1.evolve(name="new")
            return [a1, a2]

        @attr.s(auto_attribs=True, field_transformer=hook)
        class C:
            x: int

        assert {"x": 1, "new": 2} == attr.asdict(C(1, 2))

    def test_hook_override_alias(self):
        """
        It is possible to set field alias via hook
        """

        def use_dataclass_names(cls, attribs):
            return [a.evolve(alias=a.name) for a in attribs]

        @attr.s(auto_attribs=True, field_transformer=use_dataclass_names)
        class NameCase:
            public: int
            _private: int
            __dunder__: int

        assert NameCase(public=1, _private=2, __dunder__=3) == NameCase(
            1, 2, 3
        )

    def test_hook_reorder_fields(self):
        """
        It is possible to reorder fields via the hook.
        """

        def hook(cls, attribs):
            return sorted(attribs, key=lambda x: x.metadata["field_order"])

        @attr.s(field_transformer=hook)
        class C:
            x: int = attr.ib(metadata={"field_order": 1})
            y: int = attr.ib(metadata={"field_order": 0})

        assert {"x": 0, "y": 1} == attr.asdict(C(1, 0))

    def test_hook_reorder_fields_before_order_check(self):
        """
        It is possible to reorder fields via the hook before order-based errors are raised.

        Regression test for #1147.
        """

        def hook(cls, attribs):
            return sorted(attribs, key=lambda x: x.metadata["field_order"])

        @attr.s(field_transformer=hook)
        class C:
            x: int = attr.ib(metadata={"field_order": 1}, default=0)
            y: int = attr.ib(metadata={"field_order": 0})

        assert {"x": 0, "y": 1} == attr.asdict(C(1))

    def test_hook_conflicting_defaults_after_reorder(self):
        """
        Raises `ValueError` if attributes with defaults are followed by
        mandatory attributes after the hook reorders fields.

        Regression test for #1147.
        """

        def hook(cls, attribs):
            return sorted(attribs, key=lambda x: x.metadata["field_order"])

        with pytest.raises(ValueError) as e:

            @attr.s(field_transformer=hook)
            class C:
                x: int = attr.ib(metadata={"field_order": 1})
                y: int = attr.ib(metadata={"field_order": 0}, default=0)

        assert (
            "No mandatory attributes allowed after an attribute with a "
            "default value or factory.  Attribute in question: Attribute"
            "(name='x', default=NOTHING, validator=None, repr=True, "
            "eq=True, eq_key=None, order=True, order_key=None, "
            "hash=None, init=True, "
            "metadata=mappingproxy({'field_order': 1}), type='int', converter=None, "
            "kw_only=False, inherited=False, on_setattr=None, alias=None)",
        ) == e.value.args

    def test_hook_with_inheritance(self):
        """
        The hook receives all fields from base classes.
        """

        def hook(cls, attribs):
            assert ["x", "y"] == [a.name for a in attribs]
            # Remove Base' "x"
            return attribs[1:]

        @attr.s(auto_attribs=True)
        class Base:
            x: int

        @attr.s(auto_attribs=True, field_transformer=hook)
        class Sub(Base):
            y: int

        assert {"y": 2} == attr.asdict(Sub(2))

    def test_attrs_attrclass(self):
        """
        The list of attrs returned by a field_transformer is converted to
        "AttrsClass" again.

        Regression test for #821.
        """

        @attr.s(auto_attribs=True, field_transformer=lambda c, a: list(a))
        class C:
            x: int

        fields_type = type(attr.fields(C))
        assert "CAttributes" == fields_type.__name__
        assert issubclass(fields_type, tuple)

    def test_hook_generator(self):
        """
        field_transfromers can be a generators.

        Regression test for #1416.
        """

        def hook(cls, attribs):
            yield from attribs

        @attr.s(auto_attribs=True, field_transformer=hook)
        class Base:
            x: int

        assert ["x"] == [a.name for a in attr.fields(Base)]


class TestAsDictHook:
    def test_asdict(self):
        """
        asdict() calls the hooks in attrs classes and in other datastructures
        like lists or dicts.
        """

        def hook(inst, a, v):
            if isinstance(v, datetime):
                return v.isoformat()
            return v

        @attr.dataclass
        class Child:
            x: datetime
            y: list[datetime]

        @attr.dataclass
        class Parent:
            a: Child
            b: list[Child]
            c: dict[str, Child]
            d: dict[str, datetime]

        inst = Parent(
            a=Child(1, [datetime(2020, 7, 1)]),
            b=[Child(2, [datetime(2020, 7, 2)])],
            c={"spam": Child(3, [datetime(2020, 7, 3)])},
            d={"eggs": datetime(2020, 7, 4)},
        )

        result = attr.asdict(inst, value_serializer=hook)
        assert {
            "a": {"x": 1, "y": ["2020-07-01T00:00:00"]},
            "b": [{"x": 2, "y": ["2020-07-02T00:00:00"]}],
            "c": {"spam": {"x": 3, "y": ["2020-07-03T00:00:00"]}},
            "d": {"eggs": "2020-07-04T00:00:00"},
        } == result

    def test_asdict_calls(self):
        """
        The correct instances and attribute names are passed to the hook.
        """
        calls = []

        def hook(inst, a, v):
            calls.append((inst, a.name if a else a, v))
            return v

        @attr.dataclass
        class Child:
            x: int

        @attr.dataclass
        class Parent:
            a: Child
            b: list[Child]
            c: dict[str, Child]

        inst = Parent(a=Child(1), b=[Child(2)], c={"spam": Child(3)})

        attr.asdict(inst, value_serializer=hook)
        assert [
            (inst, "a", inst.a),
            (inst.a, "x", inst.a.x),
            (inst, "b", inst.b),
            (inst.b[0], "x", inst.b[0].x),
            (inst, "c", inst.c),
            (None, None, "spam"),
            (inst.c["spam"], "x", inst.c["spam"].x),
        ] == calls


# SPDX-License-Identifier: MIT

"""
Tests for PEP-526 type annotations.
"""

import sys
import types
import typing

import pytest

import attr
import attrs

from attr._compat import PY_3_14_PLUS
from attr._make import _is_class_var
from attr.exceptions import UnannotatedAttributeError


def assert_init_annotations(cls, **annotations):
    """
    Assert cls.__init__ has the correct annotations.
    """
    __tracebackhide__ = True

    annotations["return"] = type(None)

    assert annotations == typing.get_type_hints(cls.__init__)


class TestAnnotations:
    """
    Tests for types derived from variable annotations (PEP-526).
    """

    def test_basic_annotations(self):
        """
        Sets the `Attribute.type` attr from basic type annotations.
        """

        @attr.resolve_types
        @attr.s
        class C:
            x: int = attr.ib()
            y = attr.ib(type=str)
            z = attr.ib()

        assert int is attr.fields(C).x.type
        assert str is attr.fields(C).y.type
        assert None is attr.fields(C).z.type
        assert_init_annotations(C, x=int, y=str)

    def test_catches_basic_type_conflict(self):
        """
        Raises ValueError if type is specified both ways.
        """
        with pytest.raises(ValueError) as e:

            @attr.s
            class C:
                x: int = attr.ib(type=int)

        assert (
            "Type annotation and type argument cannot both be present for 'x'.",
        ) == e.value.args

    def test_typing_annotations(self):
        """
        Sets the `Attribute.type` attr from typing annotations.
        """

        @attr.resolve_types
        @attr.s
        class C:
            x: typing.List[int] = attr.ib()
            y = attr.ib(type=typing.Optional[str])

        assert typing.List[int] is attr.fields(C).x.type
        assert typing.Optional[str] is attr.fields(C).y.type
        assert_init_annotations(C, x=typing.List[int], y=typing.Optional[str])

    def test_only_attrs_annotations_collected(self):
        """
        Annotations that aren't set to an attr.ib are ignored.
        """

        @attr.resolve_types
        @attr.s
        class C:
            x: typing.List[int] = attr.ib()
            y: int

        assert 1 == len(attr.fields(C))
        assert_init_annotations(C, x=typing.List[int])

    @pytest.mark.skipif(
        sys.version_info[:2] < (3, 11),
        reason="Incompatible behavior on older Pythons",
    )
    def test_auto_attribs(self, slots):
        """
        If *auto_attribs* is True, bare annotations are collected too.
        Defaults work and class variables are ignored.
        """

        @attr.s(auto_attribs=True, slots=slots)
        class C:
            cls_var: typing.ClassVar[int] = 23
            a: int
            x: typing.List[int] = attrs.Factory(list)
            y: int = 2
            z: int = attr.ib(default=3)
            foo: typing.Any = None

        i = C(42)
        assert "C(a=42, x=[], y=2, z=3, foo=None)" == repr(i)

        attr_names = {a.name for a in C.__attrs_attrs__}
        assert "a" in attr_names  # just double check that the set works
        assert "cls_var" not in attr_names

        attr.resolve_types(C)

        assert int is attr.fields(C).a.type

        assert attr.Factory(list) == attr.fields(C).x.default
        assert typing.List[int] is attr.fields(C).x.type

        assert int is attr.fields(C).y.type
        assert 2 == attr.fields(C).y.default

        assert int is attr.fields(C).z.type

        assert typing.Any == attr.fields(C).foo.type

        # Class body is clean.
        if slots is False:
            with pytest.raises(AttributeError):
                C.y

            assert 2 == i.y
        else:
            assert isinstance(C.y, types.MemberDescriptorType)

            i.y = 23
            assert 23 == i.y

        assert_init_annotations(
            C,
            a=int,
            x=typing.List[int],
            y=int,
            z=int,
            foo=typing.Any,
        )

    def test_auto_attribs_unannotated(self, slots):
        """
        Unannotated `attr.ib`s raise an error.
        """
        with pytest.raises(UnannotatedAttributeError) as e:

            @attr.s(slots=slots, auto_attribs=True)
            class C:
                v = attr.ib()
                x: int
                y = attr.ib()
                z: str

        assert (
            "The following `attr.ib`s lack a type annotation: v, y.",
        ) == e.value.args

    def test_auto_attribs_subclassing(self, slots):
        """
        Attributes from base classes are inherited, it doesn't matter if the
        subclass has annotations or not.

        Ref #291
        """

        @attr.resolve_types
        @attr.s(slots=slots, auto_attribs=True)
        class A:
            a: int = 1

        @attr.resolve_types
        @attr.s(slots=slots, auto_attribs=True)
        class B(A):
            b: int = 2

        @attr.resolve_types
        @attr.s(slots=slots, auto_attribs=True)
        class C(A):
            pass

        assert "B(a=1, b=2)" == repr(B())
        assert "C(a=1)" == repr(C())
        assert_init_annotations(A, a=int)
        assert_init_annotations(B, a=int, b=int)
        assert_init_annotations(C, a=int)

    def test_converter_annotations(self):
        """
        An unannotated attribute with an annotated converter gets its
        annotation from the converter.
        """

        def int2str(x: int) -> str:
            return str(x)

        @attr.s
        class A:
            a = attr.ib(converter=int2str)

        assert_init_annotations(A, a=int)

        def int2str_(x: int, y: str = ""):
            return str(x)

        @attr.s
        class A:
            a = attr.ib(converter=int2str_)

        assert_init_annotations(A, a=int)

    def test_converter_attrib_annotations(self):
        """
        If a converter is provided, an explicit type annotation has no
        effect on an attribute's type annotation.
        """

        def int2str(x: int) -> str:
            return str(x)

        @attr.s
        class A:
            a: str = attr.ib(converter=int2str)
            b = attr.ib(converter=int2str, type=str)

        assert_init_annotations(A, a=int, b=int)

    def test_non_introspectable_converter(self):
        """
        A non-introspectable converter doesn't cause a crash.
        """

        @attr.s
        class A:
            a = attr.ib(converter=print)

    def test_nullary_converter(self):
        """
        A converter with no arguments doesn't cause a crash.
        """

        def noop():
            pass

        @attr.s
        class A:
            a = attr.ib(converter=noop)

        assert A.__init__.__annotations__ == {"return": None}

    def test_pipe(self):
        """
        pipe() uses the input annotation of its first argument and the
        output annotation of its last argument.
        """

        def int2str(x: int) -> str:
            return str(x)

        def strlen(y: str) -> int:
            return len(y)

        def identity(z):
            return z

        assert attr.converters.pipe(int2str).__annotations__ == {
            "val": int,
            "return": str,
        }
        assert attr.converters.pipe(int2str, strlen).__annotations__ == {
            "val": int,
            "return": int,
        }
        assert attr.converters.pipe(identity, strlen).__annotations__ == {
            "return": int
        }
        assert attr.converters.pipe(int2str, identity).__annotations__ == {
            "val": int
        }

        def int2str_(x: int, y: int = 0) -> str:
            return str(x)

        assert attr.converters.pipe(int2str_).__annotations__ == {
            "val": int,
            "return": str,
        }

    def test_pipe_empty(self):
        """
        pipe() with no converters is annotated like the identity.
        """

        p = attr.converters.pipe()

        assert "val" in p.__annotations__

        t = p.__annotations__["val"]

        assert isinstance(t, typing.TypeVar)
        assert p.__annotations__ == {"val": t, "return": t}

    def test_pipe_non_introspectable(self):
        """
        pipe() doesn't crash when passed a non-introspectable converter.
        """

        assert attr.converters.pipe(print).__annotations__ == {}

    def test_pipe_nullary(self):
        """
        pipe() doesn't crash when passed a nullary converter.
        """

        def noop():
            pass

        assert attr.converters.pipe(noop).__annotations__ == {}

    def test_optional(self):
        """
        optional() uses the annotations of the converter it wraps.
        """

        def int2str(x: int) -> str:
            return str(x)

        def int_identity(x: int):
            return x

        def strify(x) -> str:
            return str(x)

        def identity(x):
            return x

        assert attr.converters.optional(int2str).__annotations__ == {
            "val": typing.Optional[int],
            "return": typing.Optional[str],
        }
        assert attr.converters.optional(int_identity).__annotations__ == {
            "val": typing.Optional[int]
        }
        assert attr.converters.optional(strify).__annotations__ == {
            "return": typing.Optional[str]
        }
        assert attr.converters.optional(identity).__annotations__ == {}

        def int2str_(x: int, y: int = 0) -> str:
            return str(x)

        assert attr.converters.optional(int2str_).__annotations__ == {
            "val": typing.Optional[int],
            "return": typing.Optional[str],
        }

    def test_optional_non_introspectable(self):
        """
        optional() doesn't crash when passed a non-introspectable
        converter.
        """

        assert attr.converters.optional(print).__annotations__ == {}

    def test_optional_nullary(self):
        """
        optional() doesn't crash when passed a nullary converter.
        """

        def noop():
            pass

        assert attr.converters.optional(noop).__annotations__ == {}

    @pytest.mark.skipif(
        sys.version_info[:2] < (3, 11),
        reason="Incompatible behavior on older Pythons",
    )
    def test_annotations_strings(self, slots):
        """
        String annotations are passed into __init__ as is.

        The strings keep changing between releases.
        """
        import typing as t

        from typing import ClassVar

        @attr.s(auto_attribs=True, slots=slots)
        class C:
            cls_var1: "typing.ClassVar[int]" = 23
            cls_var2: "ClassVar[int]" = 23
            cls_var3: "t.ClassVar[int]" = 23
            a: "int"
            x: "typing.List[int]" = attrs.Factory(list)
            y: "int" = 2
            z: "int" = attr.ib(default=3)
            foo: "typing.Any" = None

        attr.resolve_types(C, locals(), globals())

        assert_init_annotations(
            C,
            a=int,
            x=typing.List[int],
            y=int,
            z=int,
            foo=typing.Any,
        )

    def test_typing_extensions_classvar(self, slots):
        """
        If ClassVar is coming from typing_extensions, it is recognized too.
        """

        @attr.s(auto_attribs=True, slots=slots)
        class C:
            cls_var: "typing_extensions.ClassVar" = 23  # noqa: F821

        assert_init_annotations(C)

    def test_keyword_only_auto_attribs(self):
        """
        `kw_only` propagates to attributes defined via `auto_attribs`.
        """

        @attr.s(auto_attribs=True, kw_only=True)
        class C:
            x: int
            y: int

        with pytest.raises(TypeError):
            C(0, 1)

        with pytest.raises(TypeError):
            C(x=0)

        c = C(x=0, y=1)

        assert c.x == 0
        assert c.y == 1

    def test_base_class_variable(self):
        """
        Base class' class variables can be overridden with an attribute
        without resorting to using an explicit `attr.ib()`.
        """

        class Base:
            x: int = 42

        @attr.s(auto_attribs=True)
        class C(Base):
            x: int

        assert 1 == C(1).x

    def test_removes_none_too(self):
        """
        Regression test for #523: make sure defaults that are set to None are
        removed too.
        """

        @attr.s(auto_attribs=True)
        class C:
            x: int = 42
            y: typing.Any = None

        with pytest.raises(AttributeError):
            C.x

        with pytest.raises(AttributeError):
            C.y

    def test_non_comparable_defaults(self):
        """
        Regression test for #585: objects that are not directly comparable
        (for example numpy arrays) would cause a crash when used as
        default values of an attrs auto-attrib class.
        """

        class NonComparable:
            def __eq__(self, other):
                raise ValueError

        @attr.s(auto_attribs=True)
        class C:
            x: typing.Any = NonComparable()  # noqa: RUF009

    def test_basic_resolve(self):
        """
        Resolve the `Attribute.type` attr from basic type annotations.
        Unannotated types are ignored.
        """

        @attr.s
        class C:
            x: "int" = attr.ib()
            y = attr.ib(type=str)
            z = attr.ib()

        attr.resolve_types(C)

        assert int is attr.fields(C).x.type
        assert str is attr.fields(C).y.type
        assert None is attr.fields(C).z.type

    @pytest.mark.skipif(
        sys.version_info[:2] < (3, 9),
        reason="Incompatible behavior on older Pythons",
    )
    def test_extra_resolve(self):
        """
        `get_type_hints` returns extra type hints.
        """
        from typing import Annotated

        globals = {"Annotated": Annotated}

        @attr.define
        class C:
            x: 'Annotated[float, "test"]'

        attr.resolve_types(C, globals)

        assert Annotated[float, "test"] is attr.fields(C).x.type

        @attr.define
        class D:
            x: 'Annotated[float, "test"]'

        attr.resolve_types(D, globals, include_extras=False)

        assert float is attr.fields(D).x.type

    def test_resolve_types_auto_attrib(self, slots):
        """
        Types can be resolved even when strings are involved.
        """

        @attr.s(slots=slots, auto_attribs=True)
        class A:
            a: typing.List[int]
            b: typing.List["int"]
            c: "typing.List[int]"

        # Note: I don't have to pass globals and locals here because
        # int is a builtin and will be available in any scope.
        attr.resolve_types(A)

        assert typing.List[int] == attr.fields(A).a.type
        assert typing.List[int] == attr.fields(A).b.type
        assert typing.List[int] == attr.fields(A).c.type

    def test_resolve_types_decorator(self, slots):
        """
        Types can be resolved using it as a decorator.
        """

        @attr.resolve_types
        @attr.s(slots=slots, auto_attribs=True)
        class A:
            a: typing.List[int]
            b: typing.List["int"]
            c: "typing.List[int]"

        assert typing.List[int] == attr.fields(A).a.type
        assert typing.List[int] == attr.fields(A).b.type
        assert typing.List[int] == attr.fields(A).c.type

    def test_self_reference(self, slots):
        """
        References to self class using quotes can be resolved.
        """
        if PY_3_14_PLUS and not slots:
            pytest.xfail("References are changing a lot in 3.14.")

        @attr.s(slots=slots, auto_attribs=True)
        class A:
            a: "A"
            b: typing.Optional["A"]  # will resolve below -- noqa: F821

        attr.resolve_types(A, globals(), locals())

        assert A == attr.fields(A).a.type
        assert typing.Optional[A] == attr.fields(A).b.type

    def test_forward_reference(self, slots):
        """
        Forward references can be resolved.
        """
        if PY_3_14_PLUS and not slots:
            pytest.xfail("Forward references are changing a lot in 3.14.")

        @attr.s(slots=slots, auto_attribs=True)
        class A:
            a: typing.List["B"]  # will resolve below -- noqa: F821

        @attr.s(slots=slots, auto_attribs=True)
        class B:
            a: A

        attr.resolve_types(A, globals(), locals())
        attr.resolve_types(B, globals(), locals())

        assert typing.List[B] == attr.fields(A).a.type
        assert A == attr.fields(B).a.type

        assert typing.List[B] == attr.fields(A).a.type
        assert A == attr.fields(B).a.type

    def test_init_type_hints(self):
        """
        Forward references in __init__ can be automatically resolved.
        """

        @attr.s
        class C:
            x = attr.ib(type="typing.List[int]")

        assert_init_annotations(C, x=typing.List[int])

    def test_init_type_hints_fake_module(self):
        """
        If you somehow set the __module__ to something that doesn't exist
        you'll lose __init__ resolution.
        """

        class C:
            x = attr.ib(type="typing.List[int]")

        C.__module__ = "totally fake"
        C = attr.s(C)

        with pytest.raises(NameError):
            typing.get_type_hints(C.__init__)

    def test_inheritance(self):
        """
        Subclasses can be resolved after the parent is resolved.
        """

        @attr.define()
        class A:
            n: "int"

        @attr.define()
        class B(A):
            pass

        attr.resolve_types(A)
        attr.resolve_types(B)

        assert int is attr.fields(A).n.type
        assert int is attr.fields(B).n.type

    def test_resolve_twice(self):
        """
        You can call resolve_types as many times as you like.
        This test is here mostly for coverage.
        """

        @attr.define()
        class A:
            n: "int"

        attr.resolve_types(A)

        assert int is attr.fields(A).n.type

        attr.resolve_types(A)

        assert int is attr.fields(A).n.type


@pytest.mark.parametrize(
    "annot",
    [
        typing.ClassVar,
        "typing.ClassVar",
        "'typing.ClassVar[dict]'",
        "t.ClassVar[int]",
    ],
)
def test_is_class_var(annot):
    """
    ClassVars are detected, even if they're a string or quoted.
    """
    assert _is_class_var(annot)


# SPDX-License-Identifier: MIT

"""
Common helper functions for tests.
"""

from attr import Attribute
from attr._make import NOTHING, _default_init_alias_for, make_class


def simple_class(
    eq=False,
    order=False,
    repr=False,
    unsafe_hash=False,
    str=False,
    slots=False,
    frozen=False,
    cache_hash=False,
):
    """
    Return a new simple class.
    """
    return make_class(
        "C",
        ["a", "b"],
        eq=eq or order,
        order=order,
        repr=repr,
        unsafe_hash=unsafe_hash,
        init=True,
        slots=slots,
        str=str,
        frozen=frozen,
        cache_hash=cache_hash,
    )


def simple_attr(
    name,
    default=NOTHING,
    validator=None,
    repr=True,
    eq=True,
    hash=None,
    init=True,
    converter=None,
    kw_only=False,
    inherited=False,
):
    """
    Return an attribute with a name and no other bells and whistles.
    """
    return Attribute(
        name=name,
        default=default,
        validator=validator,
        repr=repr,
        cmp=None,
        eq=eq,
        hash=hash,
        init=init,
        converter=converter,
        kw_only=kw_only,
        inherited=inherited,
        alias=_default_init_alias_for(name),
    )


# SPDX-License-Identifier: MIT

import types

from typing import Protocol

import pytest

import attr


@pytest.fixture(name="mp")
def _mp():
    return types.MappingProxyType({"x": 42, "y": "foo"})


class TestMetadataProxy:
    """
    Ensure properties of metadata proxy independently of hypothesis strategies.
    """

    def test_repr(self, mp):
        """
        repr makes sense and is consistent across Python versions.
        """
        assert any(
            [
                "mappingproxy({'x': 42, 'y': 'foo'})" == repr(mp),
                "mappingproxy({'y': 'foo', 'x': 42})" == repr(mp),
            ]
        )

    def test_immutable(self, mp):
        """
        All mutating methods raise errors.
        """
        with pytest.raises(TypeError, match="not support item assignment"):
            mp["z"] = 23

        with pytest.raises(TypeError, match="not support item deletion"):
            del mp["x"]

        with pytest.raises(AttributeError, match="no attribute 'update'"):
            mp.update({})

        with pytest.raises(AttributeError, match="no attribute 'clear'"):
            mp.clear()

        with pytest.raises(AttributeError, match="no attribute 'pop'"):
            mp.pop("x")

        with pytest.raises(AttributeError, match="no attribute 'popitem'"):
            mp.popitem()

        with pytest.raises(AttributeError, match="no attribute 'setdefault'"):
            mp.setdefault("x")


def test_attrsinstance_subclass_protocol():
    """
    It's possible to subclass AttrsInstance and Protocol at once.
    """

    class Foo(attr.AttrsInstance, Protocol):
        def attribute(self) -> int: ...


# SPDX-License-Identifier: MIT

"""
Tests for `attr._config`.
"""

import pytest

from attr import _config


class TestConfig:
    def test_default(self):
        """
        Run validators by default.
        """
        assert True is _config._run_validators

    def test_set_run_validators(self):
        """
        Sets `_run_validators`.
        """
        _config.set_run_validators(False)
        assert False is _config._run_validators
        _config.set_run_validators(True)
        assert True is _config._run_validators

    def test_get_run_validators(self):
        """
        Returns `_run_validators`.
        """
        _config._run_validators = False
        assert _config._run_validators is _config.get_run_validators()
        _config._run_validators = True
        assert _config._run_validators is _config.get_run_validators()

    def test_wrong_type(self):
        """
        Passing anything else than a boolean raises TypeError.
        """
        with pytest.raises(TypeError) as e:
            _config.set_run_validators("False")
        assert "'run' must be bool." == e.value.args[0]


# SPDX-License-Identifier: MIT

"""
Unit tests for slots-related functionality.
"""

import functools
import pickle
import weakref

from unittest import mock

import pytest

import attr
import attrs

from attr._compat import PY_3_14_PLUS, PYPY


# Pympler doesn't work on PyPy.
try:
    from pympler.asizeof import asizeof

    has_pympler = True
except BaseException:  # Won't be an import error.  # noqa: BLE001
    has_pympler = False


@attr.s
class C1:
    x = attr.ib(validator=attr.validators.instance_of(int))
    y = attr.ib()

    def method(self):
        return self.x

    @classmethod
    def classmethod(cls):
        return "clsmethod"

    @staticmethod
    def staticmethod():
        return "staticmethod"

    def my_class(self):
        return __class__

    def my_super(self):
        """Just to test out the no-arg super."""
        return super().__repr__()


@attr.s(slots=True, unsafe_hash=True)
class C1Slots:
    x = attr.ib(validator=attr.validators.instance_of(int))
    y = attr.ib()

    def method(self):
        return self.x

    @classmethod
    def classmethod(cls):
        return "clsmethod"

    @staticmethod
    def staticmethod():
        return "staticmethod"

    def my_class(self):
        return __class__

    def my_super(self):
        """Just to test out the no-arg super."""
        return super().__repr__()


def test_slots_being_used():
    """
    The class is really using __slots__.
    """
    non_slot_instance = C1(x=1, y="test")
    slot_instance = C1Slots(x=1, y="test")

    assert "__dict__" not in dir(slot_instance)
    assert "__slots__" in dir(slot_instance)

    assert "__dict__" in dir(non_slot_instance)
    assert "__slots__" not in dir(non_slot_instance)

    assert {"__weakref__", "x", "y"} == set(slot_instance.__slots__)

    if has_pympler:
        assert asizeof(slot_instance) < asizeof(non_slot_instance)

    non_slot_instance.t = "test"
    with pytest.raises(AttributeError):
        slot_instance.t = "test"

    assert 1 == non_slot_instance.method()
    assert 1 == slot_instance.method()

    assert attr.fields(C1Slots) == attr.fields(C1)
    assert attr.asdict(slot_instance) == attr.asdict(non_slot_instance)


def test_basic_attr_funcs():
    """
    Comparison, `__eq__`, `__hash__`, `__repr__`, `attrs.asdict` work.
    """
    a = C1Slots(x=1, y=2)
    b = C1Slots(x=1, y=3)
    a_ = C1Slots(x=1, y=2)

    # Comparison.
    assert b > a

    assert a_ == a

    # Hashing.
    hash(b)  # Just to assert it doesn't raise.

    # Repr.
    assert "C1Slots(x=1, y=2)" == repr(a)

    assert {"x": 1, "y": 2} == attr.asdict(a)


def test_inheritance_from_nonslots():
    """
    Inheritance from a non-slotted class works.

    Note that a slotted class inheriting from an ordinary class loses most of
    the benefits of slotted classes, but it should still work.
    """

    @attr.s(slots=True, unsafe_hash=True)
    class C2Slots(C1):
        z = attr.ib()

    c2 = C2Slots(x=1, y=2, z="test")

    assert 1 == c2.x
    assert 2 == c2.y
    assert "test" == c2.z

    c2.t = "test"  # This will work, using the base class.

    assert "test" == c2.t

    assert 1 == c2.method()
    assert "clsmethod" == c2.classmethod()
    assert "staticmethod" == c2.staticmethod()

    assert {"z"} == set(C2Slots.__slots__)

    c3 = C2Slots(x=1, y=3, z="test")

    assert c3 > c2

    c2_ = C2Slots(x=1, y=2, z="test")

    assert c2 == c2_

    assert "C2Slots(x=1, y=2, z='test')" == repr(c2)

    hash(c2)  # Just to assert it doesn't raise.

    assert {"x": 1, "y": 2, "z": "test"} == attr.asdict(c2)


def test_nonslots_these():
    """
    Enhancing a dict class using 'these' works.

    This will actually *replace* the class with another one, using slots.
    """

    class SimpleOrdinaryClass:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z

        def method(self):
            return self.x

        @classmethod
        def classmethod(cls):
            return "clsmethod"

        @staticmethod
        def staticmethod():
            return "staticmethod"

    C2Slots = attr.s(
        these={"x": attr.ib(), "y": attr.ib(), "z": attr.ib()},
        init=False,
        slots=True,
        unsafe_hash=True,
    )(SimpleOrdinaryClass)

    c2 = C2Slots(x=1, y=2, z="test")
    assert 1 == c2.x
    assert 2 == c2.y
    assert "test" == c2.z
    with pytest.raises(AttributeError):
        c2.t = "test"  # We have slots now.

    assert 1 == c2.method()
    assert "clsmethod" == c2.classmethod()
    assert "staticmethod" == c2.staticmethod()

    assert {"__weakref__", "x", "y", "z"} == set(C2Slots.__slots__)

    c3 = C2Slots(x=1, y=3, z="test")
    assert c3 > c2
    c2_ = C2Slots(x=1, y=2, z="test")
    assert c2 == c2_

    assert "SimpleOrdinaryClass(x=1, y=2, z='test')" == repr(c2)

    hash(c2)  # Just to assert it doesn't raise.

    assert {"x": 1, "y": 2, "z": "test"} == attr.asdict(c2)


def test_inheritance_from_slots():
    """
    Inheriting from an attrs slotted class works.
    """

    @attr.s(slots=True, unsafe_hash=True)
    class C2Slots(C1Slots):
        z = attr.ib()

    @attr.s(slots=True, unsafe_hash=True)
    class C2(C1):
        z = attr.ib()

    c2 = C2Slots(x=1, y=2, z="test")
    assert 1 == c2.x
    assert 2 == c2.y
    assert "test" == c2.z

    assert {"z"} == set(C2Slots.__slots__)

    assert 1 == c2.method()
    assert "clsmethod" == c2.classmethod()
    assert "staticmethod" == c2.staticmethod()

    with pytest.raises(AttributeError):
        c2.t = "test"

    non_slot_instance = C2(x=1, y=2, z="test")
    if has_pympler:
        assert asizeof(c2) < asizeof(non_slot_instance)

    c3 = C2Slots(x=1, y=3, z="test")
    assert c3 > c2
    c2_ = C2Slots(x=1, y=2, z="test")
    assert c2 == c2_

    assert "C2Slots(x=1, y=2, z='test')" == repr(c2)

    hash(c2)  # Just to assert it doesn't raise.

    assert {"x": 1, "y": 2, "z": "test"} == attr.asdict(c2)


def test_inheritance_from_slots_with_attribute_override():
    """
    Inheriting from a slotted class doesn't re-create existing slots
    """

    class HasXSlot:
        __slots__ = ("x",)

    @attr.s(slots=True, unsafe_hash=True)
    class C2Slots(C1Slots):
        # y re-defined here but it shouldn't get a slot
        y = attr.ib()
        z = attr.ib()

    @attr.s(slots=True, unsafe_hash=True)
    class NonAttrsChild(HasXSlot):
        # Parent class has slot for "x" already, so we skip it
        x = attr.ib()
        y = attr.ib()
        z = attr.ib()

    c2 = C2Slots(1, 2, "test")
    assert 1 == c2.x
    assert 2 == c2.y
    assert "test" == c2.z

    assert {"z"} == set(C2Slots.__slots__)

    na = NonAttrsChild(1, 2, "test")
    assert 1 == na.x
    assert 2 == na.y
    assert "test" == na.z

    assert {"__weakref__", "y", "z"} == set(NonAttrsChild.__slots__)


def test_inherited_slot_reuses_slot_descriptor():
    """
    We reuse slot descriptor for an attr.ib defined in a slotted attr.s
    """

    class HasXSlot:
        __slots__ = ("x",)

    class OverridesX(HasXSlot):
        @property
        def x(self):
            return None

    @attr.s(slots=True)
    class Child(OverridesX):
        x = attr.ib()

    assert Child.x is not OverridesX.x
    assert Child.x is HasXSlot.x

    c = Child(1)
    assert 1 == c.x
    assert set() == set(Child.__slots__)

    ox = OverridesX()
    assert ox.x is None


def test_bare_inheritance_from_slots():
    """
    Inheriting from a bare attrs slotted class works.
    """

    @attr.s(
        init=False,
        eq=False,
        order=False,
        unsafe_hash=False,
        repr=False,
        slots=True,
    )
    class C1BareSlots:
        x = attr.ib(validator=attr.validators.instance_of(int))
        y = attr.ib()

        def method(self):
            return self.x

        @classmethod
        def classmethod(cls):
            return "clsmethod"

        @staticmethod
        def staticmethod():
            return "staticmethod"

    @attr.s(init=False, eq=False, order=False, unsafe_hash=False, repr=False)
    class C1Bare:
        x = attr.ib(validator=attr.validators.instance_of(int))
        y = attr.ib()

        def method(self):
            return self.x

        @classmethod
        def classmethod(cls):
            return "clsmethod"

        @staticmethod
        def staticmethod():
            return "staticmethod"

    @attr.s(slots=True, unsafe_hash=True)
    class C2Slots(C1BareSlots):
        z = attr.ib()

    @attr.s(slots=True, unsafe_hash=True)
    class C2(C1Bare):
        z = attr.ib()

    c2 = C2Slots(x=1, y=2, z="test")
    assert 1 == c2.x
    assert 2 == c2.y
    assert "test" == c2.z

    assert 1 == c2.method()
    assert "clsmethod" == c2.classmethod()
    assert "staticmethod" == c2.staticmethod()

    with pytest.raises(AttributeError):
        c2.t = "test"

    non_slot_instance = C2(x=1, y=2, z="test")
    if has_pympler:
        assert asizeof(c2) < asizeof(non_slot_instance)

    c3 = C2Slots(x=1, y=3, z="test")
    assert c3 > c2
    c2_ = C2Slots(x=1, y=2, z="test")
    assert c2 == c2_

    assert "C2Slots(x=1, y=2, z='test')" == repr(c2)

    hash(c2)  # Just to assert it doesn't raise.

    assert {"x": 1, "y": 2, "z": "test"} == attr.asdict(c2)


class TestClosureCellRewriting:
    def test_closure_cell_rewriting(self):
        """
        Slotted classes support proper closure cell rewriting.

        This affects features like `__class__` and the no-arg super().
        """
        non_slot_instance = C1(x=1, y="test")
        slot_instance = C1Slots(x=1, y="test")

        assert non_slot_instance.my_class() is C1
        assert slot_instance.my_class() is C1Slots

        # Just assert they return something, and not an exception.
        assert non_slot_instance.my_super()
        assert slot_instance.my_super()

    def test_inheritance(self):
        """
        Slotted classes support proper closure cell rewriting when inheriting.

        This affects features like `__class__` and the no-arg super().
        """

        @attr.s
        class C2(C1):
            def my_subclass(self):
                return __class__

        @attr.s
        class C2Slots(C1Slots):
            def my_subclass(self):
                return __class__

        non_slot_instance = C2(x=1, y="test")
        slot_instance = C2Slots(x=1, y="test")

        assert non_slot_instance.my_class() is C1
        assert slot_instance.my_class() is C1Slots

        # Just assert they return something, and not an exception.
        assert non_slot_instance.my_super()
        assert slot_instance.my_super()

        assert non_slot_instance.my_subclass() is C2
        assert slot_instance.my_subclass() is C2Slots

    def test_cls_static(self, slots):
        """
        Slotted classes support proper closure cell rewriting for class- and
        static methods.
        """
        # Python can reuse closure cells, so we create new classes just for
        # this test.

        @attr.s(slots=slots)
        class C:
            @classmethod
            def clsmethod(cls):
                return __class__

        assert C.clsmethod() is C

        @attr.s(slots=slots)
        class D:
            @staticmethod
            def statmethod():
                return __class__

        assert D.statmethod() is D


@pytest.mark.skipif(PYPY, reason="__slots__ only block weakref on CPython")
def test_not_weakrefable():
    """
    Instance is not weak-referenceable when `weakref_slot=False` in CPython.
    """

    @attr.s(slots=True, weakref_slot=False)
    class C:
        pass

    c = C()

    with pytest.raises(TypeError):
        weakref.ref(c)


@pytest.mark.skipif(
    not PYPY, reason="slots without weakref_slot should only work on PyPy"
)
def test_implicitly_weakrefable():
    """
    Instance is weak-referenceable even when `weakref_slot=False` in PyPy.
    """

    @attr.s(slots=True, weakref_slot=False)
    class C:
        pass

    c = C()
    w = weakref.ref(c)

    assert c is w()


def test_weakrefable():
    """
    Instance is weak-referenceable when `weakref_slot=True`.
    """

    @attr.s(slots=True, weakref_slot=True)
    class C:
        pass

    c = C()
    w = weakref.ref(c)

    assert c is w()


def test_weakref_does_not_add_a_field():
    """
    `weakref_slot=True` does not add a field to the class.
    """

    @attr.s(slots=True, weakref_slot=True)
    class C:
        field = attr.ib()

    assert [f.name for f in attr.fields(C)] == ["field"]


def tests_weakref_does_not_add_when_inheriting_with_weakref():
    """
    `weakref_slot=True` does not add a new __weakref__ slot when inheriting
    one.
    """

    @attr.s(slots=True, weakref_slot=True)
    class C:
        pass

    @attr.s(slots=True, weakref_slot=True)
    class D(C):
        pass

    d = D()
    w = weakref.ref(d)

    assert d is w()


def tests_weakref_does_not_add_with_weakref_attribute():
    """
    `weakref_slot=True` does not add a new __weakref__ slot when an attribute
    of that name exists.
    """

    @attr.s(slots=True, weakref_slot=True)
    class C:
        __weakref__ = attr.ib(
            init=False, hash=False, repr=False, eq=False, order=False
        )

    c = C()
    w = weakref.ref(c)

    assert c is w()


def test_slots_empty_cell():
    """
    Tests that no `ValueError: Cell is empty` exception is raised when
    closure cells are present with no contents in a `slots=True` class.
    (issue https://github.com/python-attrs/attrs/issues/589)

    If a method mentions `__class__` or uses the no-arg `super()`, the compiler
    will bake a reference to the class in the method itself as
    `method.__closure__`. Since `attrs` replaces the class with a clone,
    `_ClassBuilder._create_slots_class(self)` will rewrite these references so
    it keeps working. This method was not properly covering the edge case where
    the closure cell was empty, we fixed it and this is the non-regression
    test.
    """

    @attr.s(slots=True)
    class C:
        field = attr.ib()

        def f(self, a):
            super(C, self).__init__()  # noqa: UP008

    C(field=1)


@attr.s(getstate_setstate=True)
class C2:
    x = attr.ib()


@attr.s(slots=True, getstate_setstate=True)
class C2Slots:
    x = attr.ib()


class TestPickle:
    @pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL))
    def test_pickleable_by_default(self, protocol):
        """
        If nothing else is passed, slotted classes can be pickled and
        unpickled with all supported protocols.
        """
        i1 = C1Slots(1, 2)
        i2 = pickle.loads(pickle.dumps(i1, protocol))

        assert i1 == i2
        assert i1 is not i2

    def test_no_getstate_setstate_for_dict_classes(self):
        """
        As long as getstate_setstate is None, nothing is done to dict
        classes.
        """
        assert getattr(object, "__getstate__", None) is getattr(
            C1, "__getstate__", None
        )
        assert getattr(object, "__setstate__", None) is getattr(
            C1, "__setstate__", None
        )

    def test_no_getstate_setstate_if_option_false(self):
        """
        Don't add getstate/setstate if getstate_setstate is False.
        """

        @attr.s(slots=True, getstate_setstate=False)
        class C:
            x = attr.ib()

        assert getattr(object, "__getstate__", None) is getattr(
            C, "__getstate__", None
        )
        assert getattr(object, "__setstate__", None) is getattr(
            C, "__setstate__", None
        )

    @pytest.mark.parametrize("cls", [C2(1), C2Slots(1)])
    def test_getstate_set_state_force_true(self, cls):
        """
        If getstate_setstate is True, add them unconditionally.
        """
        assert None is not getattr(cls, "__getstate__", None)
        assert None is not getattr(cls, "__setstate__", None)


def test_slots_super_property_get():
    """
    Both `super()` and `super(self.__class__, self)` work.
    """

    @attr.s(slots=True)
    class A:
        x = attr.ib()

        @property
        def f(self):
            return self.x

    @attr.s(slots=True)
    class B(A):
        @property
        def f(self):
            return super().f ** 2

    @attr.s(slots=True)
    class C(A):
        @property
        def f(self):
            return super(C, self).f ** 2  # noqa: UP008

    assert B(11).f == 121
    assert B(17).f == 289
    assert C(11).f == 121
    assert C(17).f == 289


def test_slots_super_property_get_shortcut():
    """
    The `super()` shortcut is allowed.
    """

    @attr.s(slots=True)
    class A:
        x = attr.ib()

        @property
        def f(self):
            return self.x

    @attr.s(slots=True)
    class B(A):
        @property
        def f(self):
            return super().f ** 2

    assert B(11).f == 121
    assert B(17).f == 289


def test_slots_cached_property_allows_call():
    """
    cached_property in slotted class allows call.
    """

    @attr.s(slots=True)
    class A:
        x = attr.ib()

        @functools.cached_property
        def f(self):
            return self.x

    assert A(11).f == 11


def test_slots_cached_property_class_does_not_have__dict__():
    """
    slotted class with cached property has no __dict__ attribute.
    """

    @attr.s(slots=True)
    class A:
        x = attr.ib()

        @functools.cached_property
        def f(self):
            return self.x

    assert set(A.__slots__) == {"x", "f", "__weakref__"}
    assert "__dict__" not in dir(A)


def test_slots_cached_property_works_on_frozen_isntances():
    """
    Infers type of cached property.
    """

    @attrs.frozen(slots=True)
    class A:
        x: int

        @functools.cached_property
        def f(self) -> int:
            return self.x

    assert A(x=1).f == 1


@pytest.mark.xfail(
    PY_3_14_PLUS, reason="3.14 returns weird annotation for cached_properies"
)
def test_slots_cached_property_infers_type():
    """
    Infers type of cached property.
    """

    @attrs.frozen(slots=True)
    class A:
        x: int

        @functools.cached_property
        def f(self) -> int:
            return self.x

    assert A.__annotations__ == {"x": int, "f": int}


def test_slots_cached_property_with_empty_getattr_raises_attribute_error_of_requested():
    """
    Ensures error information is not lost.
    """

    @attr.s(slots=True)
    class A:
        x = attr.ib()

        @functools.cached_property
        def f(self):
            return self.x

    a = A(1)
    with pytest.raises(
        AttributeError, match="'A' object has no attribute 'z'"
    ):
        a.z


def test_slots_cached_property_raising_attributeerror():
    """
    Ensures AttributeError raised by a property is preserved by __getattr__()
    implementation.

    Regression test for issue https://github.com/python-attrs/attrs/issues/1230
    """

    @attr.s(slots=True)
    class A:
        x = attr.ib()

        @functools.cached_property
        def f(self):
            return self.p

        @property
        def p(self):
            raise AttributeError("I am a property")

        @functools.cached_property
        def g(self):
            return self.q

        @property
        def q(self):
            return 2

    a = A(1)
    with pytest.raises(AttributeError, match=r"^I am a property$"):
        a.p
    with pytest.raises(AttributeError, match=r"^I am a property$"):
        a.f

    assert a.g == 2
    assert a.q == 2


def test_slots_cached_property_with_getattr_calls_getattr_for_missing_attributes():
    """
    Ensure __getattr__ implementation is maintained for non cached_properties.
    """

    @attr.s(slots=True)
    class A:
        x = attr.ib()

        @functools.cached_property
        def f(self):
            return self.x

        def __getattr__(self, item):
            return item

    a = A(1)
    assert a.f == 1
    assert a.z == "z"


def test_slots_getattr_in_superclass__is_called_for_missing_attributes_when_cached_property_present():
    """
    Ensure __getattr__ implementation is maintained in subclass.
    """

    @attr.s(slots=True)
    class A:
        x = attr.ib()

        def __getattr__(self, item):
            return item

    @attr.s(slots=True)
    class B(A):
        @functools.cached_property
        def f(self):
            return self.x

    b = B(1)
    assert b.f == 1
    assert b.z == "z"


def test_slots_getattr_in_subclass_gets_superclass_cached_property():
    """
    Ensure super() in __getattr__ is not broken through cached_property re-write.
    """

    @attr.s(slots=True)
    class A:
        x = attr.ib()

        @functools.cached_property
        def f(self):
            return self.x

        def __getattr__(self, item):
            return item

    @attr.s(slots=True)
    class B(A):
        @functools.cached_property
        def g(self):
            return self.x

        def __getattr__(self, item):
            return super().__getattr__(item)

    b = B(1)
    assert b.f == 1
    assert b.z == "z"


def test_slots_sub_class_with_independent_cached_properties_both_work():
    """
    Subclassing shouldn't break cached properties.
    """

    @attr.s(slots=True)
    class A:
        x = attr.ib()

        @functools.cached_property
        def f(self):
            return self.x

    @attr.s(slots=True)
    class B(A):
        @functools.cached_property
        def g(self):
            return self.x * 2

    assert B(1).f == 1
    assert B(1).g == 2


def test_slots_with_multiple_cached_property_subclasses_works():
    """
    Multiple sub-classes shouldn't break cached properties.
    """

    @attr.s(slots=True)
    class A:
        x = attr.ib(kw_only=True)

        @functools.cached_property
        def f(self):
            return self.x

    @attr.s(slots=False)
    class B:
        @functools.cached_property
        def g(self):
            return self.x * 2

        def __getattr__(self, item):
            if hasattr(super(), "__getattr__"):
                return super().__getattr__(item)
            return item

    @attr.s(slots=True)
    class AB(A, B):
        pass

    ab = AB(x=1)

    assert ab.f == 1
    assert ab.g == 2
    assert ab.h == "h"


def test_slotted_cached_property_can_access_super():
    """
    Multiple sub-classes shouldn't break cached properties.
    """

    @attr.s(slots=True)
    class A:
        x = attr.ib(kw_only=True)

    @attr.s(slots=True)
    class B(A):
        @functools.cached_property
        def f(self):
            return super().x * 2

    assert B(x=1).f == 2


def test_slots_sub_class_avoids_duplicated_slots():
    """
    Duplicating the slots is a waste of memory.
    """

    @attr.s(slots=True)
    class A:
        x = attr.ib()

        @functools.cached_property
        def f(self):
            return self.x

    @attr.s(slots=True)
    class B(A):
        @functools.cached_property
        def f(self):
            return self.x * 2

    assert B(1).f == 2
    assert B.__slots__ == ()


def test_slots_sub_class_with_actual_slot():
    """
    A sub-class can have an explicit attrs field that replaces a cached property.
    """

    @attr.s(slots=True)
    class A:  # slots : (x, f)
        x = attr.ib()

        @functools.cached_property
        def f(self):
            return self.x

    @attr.s(slots=True)
    class B(A):
        f: int = attr.ib()

    assert B(1, 2).f == 2
    assert B.__slots__ == ()


def test_slots_cached_property_is_not_called_at_construction():
    """
    A cached property function should only be called at property access point.
    """
    call_count = 0

    @attr.s(slots=True)
    class A:
        x = attr.ib()

        @functools.cached_property
        def f(self):
            nonlocal call_count
            call_count += 1
            return self.x

    A(1)
    assert call_count == 0


def test_slots_cached_property_repeat_call_only_once():
    """
    A cached property function should be called only once, on repeated attribute access.
    """
    call_count = 0

    @attr.s(slots=True)
    class A:
        x = attr.ib()

        @functools.cached_property
        def f(self):
            nonlocal call_count
            call_count += 1
            return self.x

    obj = A(1)
    obj.f
    obj.f
    assert call_count == 1


def test_slots_cached_property_called_independent_across_instances():
    """
    A cached property value should be specific to the given instance.
    """

    @attr.s(slots=True)
    class A:
        x = attr.ib()

        @functools.cached_property
        def f(self):
            return self.x

    obj_1 = A(1)
    obj_2 = A(2)

    assert obj_1.f == 1
    assert obj_2.f == 2


def test_slots_cached_properties_work_independently():
    """
    Multiple cached properties should work independently.
    """

    @attr.s(slots=True)
    class A:
        x = attr.ib()

        @functools.cached_property
        def f_1(self):
            return self.x

        @functools.cached_property
        def f_2(self):
            return self.x * 2

    obj = A(1)

    assert obj.f_1 == 1
    assert obj.f_2 == 2


@attr.s(slots=True)
class A:
    x = attr.ib()
    b = attr.ib()
    c = attr.ib()


def test_slots_unpickle_after_attr_removed():
    """
    We don't assign attributes we don't have anymore if the class has
    removed it.
    """
    a = A(1, 2, 3)
    a_pickled = pickle.dumps(a)
    a_unpickled = pickle.loads(a_pickled)
    assert a_unpickled == a

    @attr.s(slots=True)
    class NEW_A:
        x = attr.ib()
        c = attr.ib()

    with mock.patch(f"{__name__}.A", NEW_A):
        new_a = pickle.loads(a_pickled)

        assert new_a.x == 1
        assert new_a.c == 3
        assert not hasattr(new_a, "b")


def test_slots_unpickle_after_attr_added(frozen):
    """
    We don't assign attribute we haven't had before if the class has one added.
    """
    a = A(1, 2, 3)
    a_pickled = pickle.dumps(a)
    a_unpickled = pickle.loads(a_pickled)

    assert a_unpickled == a

    @attr.s(slots=True, frozen=frozen)
    class NEW_A:
        x = attr.ib()
        b = attr.ib()
        d = attr.ib()
        c = attr.ib()

    with mock.patch(f"{__name__}.A", NEW_A):
        new_a = pickle.loads(a_pickled)

        assert new_a.x == 1
        assert new_a.b == 2
        assert new_a.c == 3
        assert not hasattr(new_a, "d")


def test_slots_unpickle_is_backward_compatible(frozen):
    """
    Ensure object pickled before v22.2.0 can still be unpickled.
    """
    a = A(1, 2, 3)

    a_pickled = (
        b"\x80\x04\x95&\x00\x00\x00\x00\x00\x00\x00\x8c\x10"
        + a.__module__.encode()
        + b"\x94\x8c\x01A\x94\x93\x94)\x81\x94K\x01K\x02K\x03\x87\x94b."
    )

    a_unpickled = pickle.loads(a_pickled)

    assert a_unpickled == a


# SPDX-License-Identifier: MIT

import abc
import inspect

import pytest

import attrs

from attr._compat import PY_3_10_PLUS, PY_3_12_PLUS


@pytest.mark.skipif(
    not PY_3_10_PLUS, reason="abc.update_abstractmethods is 3.10+"
)
class TestUpdateAbstractMethods:
    def test_abc_implementation(self, slots):
        """
        If an attrs class implements an abstract method, it stops being
        abstract.
        """

        class Ordered(abc.ABC):
            @abc.abstractmethod
            def __lt__(self, other):
                pass

            @abc.abstractmethod
            def __le__(self, other):
                pass

        @attrs.define(order=True, slots=slots)
        class Concrete(Ordered):
            x: int

        assert not inspect.isabstract(Concrete)
        assert Concrete(2) > Concrete(1)

    def test_remain_abstract(self, slots):
        """
        If an attrs class inherits from an abstract class but doesn't implement
        abstract methods, it remains abstract.
        """

        class A(abc.ABC):
            @abc.abstractmethod
            def foo(self):
                pass

        @attrs.define(slots=slots)
        class StillAbstract(A):
            pass

        assert inspect.isabstract(StillAbstract)
        expected_exception_message = (
            "^Can't instantiate abstract class StillAbstract without an "
            "implementation for abstract method 'foo'$"
            if PY_3_12_PLUS
            else "class StillAbstract with abstract method foo"
        )
        with pytest.raises(TypeError, match=expected_exception_message):
            StillAbstract()


# SPDX-License-Identifier: MIT

"""
Tests for dunder methods from `attrib._make`.
"""

import copy
import inspect
import pickle

import pytest

from hypothesis import given
from hypothesis.strategies import booleans

import attr

from attr._make import (
    NOTHING,
    Factory,
    _add_repr,
    _compile_and_eval,
    _make_init_script,
    fields,
    make_class,
)
from attr.validators import instance_of

from .utils import simple_attr, simple_class


EqC = simple_class(eq=True)
EqCSlots = simple_class(eq=True, slots=True)
OrderC = simple_class(order=True)
OrderCSlots = simple_class(order=True, slots=True)
ReprC = simple_class(repr=True)
ReprCSlots = simple_class(repr=True, slots=True)


@attr.s(eq=True)
class EqCallableC:
    a = attr.ib(eq=str.lower, order=False)
    b = attr.ib(eq=True)


@attr.s(eq=True, slots=True)
class EqCallableCSlots:
    a = attr.ib(eq=str.lower, order=False)
    b = attr.ib(eq=True)


@attr.s(order=True)
class OrderCallableC:
    a = attr.ib(eq=True, order=str.lower)
    b = attr.ib(order=True)


@attr.s(order=True, slots=True)
class OrderCallableCSlots:
    a = attr.ib(eq=True, order=str.lower)
    b = attr.ib(order=True)


# HashC is hashable by explicit definition while HashCSlots is hashable
# implicitly.  The "Cached" versions are the same, except with hash code
# caching enabled
HashC = simple_class(unsafe_hash=True)
HashCSlots = simple_class(unsafe_hash=None, eq=True, frozen=True, slots=True)
HashCCached = simple_class(unsafe_hash=True, cache_hash=True)
HashCSlotsCached = simple_class(
    unsafe_hash=None, eq=True, frozen=True, slots=True, cache_hash=True
)
# the cached hash code is stored slightly differently in this case
# so it needs to be tested separately
HashCFrozenNotSlotsCached = simple_class(
    frozen=True, slots=False, unsafe_hash=True, cache_hash=True
)


def _add_init(cls, frozen):
    """
    Add a __init__ method to *cls*.  If *frozen* is True, make it immutable.

    This function used to be part of _make.  It wasn't used anymore however
    the tests for it are still useful to test the behavior of _make_init.
    """
    has_pre_init = bool(getattr(cls, "__attrs_pre_init__", False))

    script, globs, annots = _make_init_script(
        cls,
        cls.__attrs_attrs__,
        has_pre_init,
        (
            len(inspect.signature(cls.__attrs_pre_init__).parameters) > 1
            if has_pre_init
            else False
        ),
        getattr(cls, "__attrs_post_init__", False),
        frozen,
        "__slots__" in cls.__dict__,
        cache_hash=False,
        base_attr_map={},
        is_exc=False,
        cls_on_setattr=None,
        attrs_init=False,
    )
    _compile_and_eval(script, globs, filename="__init__")
    cls.__init__ = globs["__init__"]
    cls.__init__.__annotations__ = annots
    return cls


class InitC:
    __attrs_attrs__ = [simple_attr("a"), simple_attr("b")]


InitC = _add_init(InitC, False)


class TestEqOrder:
    """
    Tests for eq and order related methods.
    """

    @given(booleans())
    def test_eq_ignore_attrib(self, slots):
        """
        If `eq` is False for an attribute, ignore that attribute.
        """
        C = make_class(
            "C", {"a": attr.ib(eq=False), "b": attr.ib()}, slots=slots
        )

        assert C(1, 2) == C(2, 2)

    @pytest.mark.parametrize("cls", [EqC, EqCSlots])
    def test_equal(self, cls):
        """
        Equal objects are detected as equal.
        """
        assert cls(1, 2) == cls(1, 2)
        assert not (cls(1, 2) != cls(1, 2))

    @pytest.mark.parametrize("cls", [EqCallableC, EqCallableCSlots])
    def test_equal_callable(self, cls):
        """
        Equal objects are detected as equal.
        """
        assert cls("Test", 1) == cls("test", 1)
        assert cls("Test", 1) != cls("test", 2)
        assert not (cls("Test", 1) != cls("test", 1))
        assert not (cls("Test", 1) == cls("test", 2))

    @pytest.mark.parametrize("cls", [EqC, EqCSlots])
    def test_unequal_same_class(self, cls):
        """
        Unequal objects of correct type are detected as unequal.
        """
        assert cls(1, 2) != cls(2, 1)
        assert not (cls(1, 2) == cls(2, 1))

    @pytest.mark.parametrize("cls", [EqCallableC, EqCallableCSlots])
    def test_unequal_same_class_callable(self, cls):
        """
        Unequal objects of correct type are detected as unequal.
        """
        assert cls("Test", 1) != cls("foo", 2)
        assert not (cls("Test", 1) == cls("foo", 2))

    @pytest.mark.parametrize(
        "cls", [EqC, EqCSlots, EqCallableC, EqCallableCSlots]
    )
    def test_unequal_different_class(self, cls):
        """
        Unequal objects of different type are detected even if their attributes
        match.
        """

        class NotEqC:
            a = 1
            b = 2

        assert cls(1, 2) != NotEqC()
        assert not (cls(1, 2) == NotEqC())

    @pytest.mark.parametrize("cls", [OrderC, OrderCSlots])
    def test_lt(self, cls):
        """
        __lt__ compares objects as tuples of attribute values.
        """
        for a, b in [
            ((1, 2), (2, 1)),
            ((1, 2), (1, 3)),
            (("a", "b"), ("b", "a")),
        ]:
            assert cls(*a) < cls(*b)

    @pytest.mark.parametrize("cls", [OrderCallableC, OrderCallableCSlots])
    def test_lt_callable(self, cls):
        """
        __lt__ compares objects as tuples of attribute values.
        """
        # Note: "A" < "a"
        for a, b in [
            (("test1", 1), ("Test1", 2)),
            (("test0", 1), ("Test1", 1)),
        ]:
            assert cls(*a) < cls(*b)

    @pytest.mark.parametrize(
        "cls", [OrderC, OrderCSlots, OrderCallableC, OrderCallableCSlots]
    )
    def test_lt_unordable(self, cls):
        """
        __lt__ returns NotImplemented if classes differ.
        """
        assert NotImplemented == (cls(1, 2).__lt__(42))

    @pytest.mark.parametrize("cls", [OrderC, OrderCSlots])
    def test_le(self, cls):
        """
        __le__ compares objects as tuples of attribute values.
        """
        for a, b in [
            ((1, 2), (2, 1)),
            ((1, 2), (1, 3)),
            ((1, 1), (1, 1)),
            (("a", "b"), ("b", "a")),
            (("a", "b"), ("a", "b")),
        ]:
            assert cls(*a) <= cls(*b)

    @pytest.mark.parametrize("cls", [OrderCallableC, OrderCallableCSlots])
    def test_le_callable(self, cls):
        """
        __le__ compares objects as tuples of attribute values.
        """
        # Note: "A" < "a"
        for a, b in [
            (("test1", 1), ("Test1", 1)),
            (("test1", 1), ("Test1", 2)),
            (("test0", 1), ("Test1", 1)),
            (("test0", 2), ("Test1", 1)),
        ]:
            assert cls(*a) <= cls(*b)

    @pytest.mark.parametrize(
        "cls", [OrderC, OrderCSlots, OrderCallableC, OrderCallableCSlots]
    )
    def test_le_unordable(self, cls):
        """
        __le__ returns NotImplemented if classes differ.
        """
        assert NotImplemented == (cls(1, 2).__le__(42))

    @pytest.mark.parametrize("cls", [OrderC, OrderCSlots])
    def test_gt(self, cls):
        """
        __gt__ compares objects as tuples of attribute values.
        """
        for a, b in [
            ((2, 1), (1, 2)),
            ((1, 3), (1, 2)),
            (("b", "a"), ("a", "b")),
        ]:
            assert cls(*a) > cls(*b)

    @pytest.mark.parametrize("cls", [OrderCallableC, OrderCallableCSlots])
    def test_gt_callable(self, cls):
        """
        __gt__ compares objects as tuples of attribute values.
        """
        # Note: "A" < "a"
        for a, b in [
            (("Test1", 2), ("test1", 1)),
            (("Test1", 1), ("test0", 1)),
        ]:
            assert cls(*a) > cls(*b)

    @pytest.mark.parametrize(
        "cls", [OrderC, OrderCSlots, OrderCallableC, OrderCallableCSlots]
    )
    def test_gt_unordable(self, cls):
        """
        __gt__ returns NotImplemented if classes differ.
        """
        assert NotImplemented == (cls(1, 2).__gt__(42))

    @pytest.mark.parametrize("cls", [OrderC, OrderCSlots])
    def test_ge(self, cls):
        """
        __ge__ compares objects as tuples of attribute values.
        """
        for a, b in [
            ((2, 1), (1, 2)),
            ((1, 3), (1, 2)),
            ((1, 1), (1, 1)),
            (("b", "a"), ("a", "b")),
            (("a", "b"), ("a", "b")),
        ]:
            assert cls(*a) >= cls(*b)

    @pytest.mark.parametrize("cls", [OrderCallableC, OrderCallableCSlots])
    def test_ge_callable(self, cls):
        """
        __ge__ compares objects as tuples of attribute values.
        """
        # Note: "A" < "a"
        for a, b in [
            (("Test1", 1), ("test1", 1)),
            (("Test1", 2), ("test1", 1)),
            (("Test1", 1), ("test0", 1)),
            (("Test1", 1), ("test0", 2)),
        ]:
            assert cls(*a) >= cls(*b)

    @pytest.mark.parametrize(
        "cls", [OrderC, OrderCSlots, OrderCallableC, OrderCallableCSlots]
    )
    def test_ge_unordable(self, cls):
        """
        __ge__ returns NotImplemented if classes differ.
        """
        assert NotImplemented == (cls(1, 2).__ge__(42))


class TestAddRepr:
    """
    Tests for `_add_repr`.
    """

    def test_repr(self, slots):
        """
        If `repr` is False, ignore that attribute.
        """
        C = make_class(
            "C", {"a": attr.ib(repr=False), "b": attr.ib()}, slots=slots
        )

        assert "C(b=2)" == repr(C(1, 2))

    @pytest.mark.parametrize("cls", [ReprC, ReprCSlots])
    def test_repr_works(self, cls):
        """
        repr returns a sensible value.
        """
        assert "C(a=1, b=2)" == repr(cls(1, 2))

    def test_custom_repr_works(self):
        """
        repr returns a sensible value for attributes with a custom repr
        callable.
        """

        def custom_repr(value):
            return "foo:" + str(value)

        @attr.s
        class C:
            a = attr.ib(repr=custom_repr)

        assert "C(a=foo:1)" == repr(C(1))

    def test_infinite_recursion(self):
        """
        In the presence of a cyclic graph, repr will emit an ellipsis and not
        raise an exception.
        """

        @attr.s
        class Cycle:
            value = attr.ib(default=7)
            cycle = attr.ib(default=None)

        cycle = Cycle()
        cycle.cycle = cycle
        assert "Cycle(value=7, cycle=...)" == repr(cycle)

    def test_infinite_recursion_long_cycle(self):
        """
        A cyclic graph can pass through other non-attrs objects, and repr will
        still emit an ellipsis and not raise an exception.
        """

        @attr.s
        class LongCycle:
            value = attr.ib(default=14)
            cycle = attr.ib(default=None)

        cycle = LongCycle()
        # Ensure that the reference cycle passes through a non-attrs object.
        # This demonstrates the need for a thread-local "global" ID tracker.
        cycle.cycle = {"cycle": [cycle]}
        assert "LongCycle(value=14, cycle={'cycle': [...]})" == repr(cycle)

    def test_underscores(self):
        """
        repr does not strip underscores.
        """

        class C:
            __attrs_attrs__ = [simple_attr("_x")]

        C = _add_repr(C)
        i = C()
        i._x = 42

        assert "C(_x=42)" == repr(i)

    def test_repr_uninitialized_member(self):
        """
        repr signals unset attributes
        """
        C = make_class("C", {"a": attr.ib(init=False)})

        assert "C(a=NOTHING)" == repr(C())

    @given(add_str=booleans(), slots=booleans())
    def test_str(self, add_str, slots):
        """
        If str is True, it returns the same as repr.

        This only makes sense when subclassing a class with an poor __str__
        (like Exceptions).
        """

        @attr.s(str=add_str, slots=slots)
        class Error(Exception):
            x = attr.ib()

        e = Error(42)

        assert (str(e) == repr(e)) is add_str

    def test_str_no_repr(self):
        """
        Raises a ValueError if repr=False and str=True.
        """
        with pytest.raises(ValueError) as e:
            simple_class(repr=False, str=True)

        assert (
            "__str__ can only be generated if a __repr__ exists."
        ) == e.value.args[0]


# these are for use in TestAddHash.test_cache_hash_serialization
# they need to be out here so they can be un-pickled
@attr.attrs(unsafe_hash=True, cache_hash=False)
class HashCacheSerializationTestUncached:
    foo_value = attr.ib()


@attr.attrs(unsafe_hash=True, cache_hash=True)
class HashCacheSerializationTestCached:
    foo_value = attr.ib()


@attr.attrs(slots=True, unsafe_hash=True, cache_hash=True)
class HashCacheSerializationTestCachedSlots:
    foo_value = attr.ib()


class IncrementingHasher:
    def __init__(self):
        self.hash_value = 100

    def __hash__(self):
        rv = self.hash_value
        self.hash_value += 1
        return rv


class TestAddHash:
    """
    Tests for `_add_hash`.
    """

    def test_enforces_type(self):
        """
        The `hash` argument to both attrs and attrib must be None, True, or
        False.
        """
        exc_args = ("Invalid value for hash.  Must be True, False, or None.",)

        with pytest.raises(TypeError) as e:
            make_class("C", {}, unsafe_hash=1)

        assert exc_args == e.value.args

        with pytest.raises(TypeError) as e:
            make_class("C", {"a": attr.ib(hash=1)})

        assert exc_args == e.value.args

    def test_enforce_no_cache_hash_without_hash(self):
        """
        Ensure exception is thrown if caching the hash code is requested
        but attrs is not requested to generate `__hash__`.
        """
        exc_args = (
            "Invalid value for cache_hash.  To use hash caching,"
            " hashing must be either explicitly or implicitly "
            "enabled.",
        )
        with pytest.raises(TypeError) as e:
            make_class("C", {}, unsafe_hash=False, cache_hash=True)
        assert exc_args == e.value.args

        # unhashable case
        with pytest.raises(TypeError) as e:
            make_class(
                "C",
                {},
                unsafe_hash=None,
                eq=True,
                frozen=False,
                cache_hash=True,
            )
        assert exc_args == e.value.args

    def test_enforce_no_cached_hash_without_init(self):
        """
        Ensure exception is thrown if caching the hash code is requested
        but attrs is not requested to generate `__init__`.
        """
        exc_args = (
            "Invalid value for cache_hash.  To use hash caching,"
            " init must be True.",
        )
        with pytest.raises(TypeError) as e:
            make_class("C", {}, init=False, unsafe_hash=True, cache_hash=True)
        assert exc_args == e.value.args

    @given(booleans(), booleans())
    def test_hash_attribute(self, slots, cache_hash):
        """
        If `hash` is False on an attribute, ignore that attribute.
        """
        C = make_class(
            "C",
            {"a": attr.ib(hash=False), "b": attr.ib()},
            slots=slots,
            unsafe_hash=True,
            cache_hash=cache_hash,
        )

        assert hash(C(1, 2)) == hash(C(2, 2))

    @given(booleans())
    def test_hash_attribute_mirrors_eq(self, eq):
        """
        If `hash` is None, the hash generation mirrors `eq`.
        """
        C = make_class("C", {"a": attr.ib(eq=eq)}, eq=True, frozen=True)

        if eq:
            assert C(1) != C(2)
            assert hash(C(1)) != hash(C(2))
            assert hash(C(1)) == hash(C(1))
        else:
            assert C(1) == C(2)
            assert hash(C(1)) == hash(C(2))

    @given(booleans())
    def test_hash_mirrors_eq(self, eq):
        """
        If `hash` is None, the hash generation mirrors `eq`.
        """
        C = make_class("C", {"a": attr.ib()}, eq=eq, frozen=True)

        i = C(1)

        assert i == i
        assert hash(i) == hash(i)

        if eq:
            assert C(1) == C(1)
            assert hash(C(1)) == hash(C(1))
        else:
            assert C(1) != C(1)
            assert hash(C(1)) != hash(C(1))

    @pytest.mark.parametrize(
        "cls",
        [
            HashC,
            HashCSlots,
            HashCCached,
            HashCSlotsCached,
            HashCFrozenNotSlotsCached,
        ],
    )
    def test_hash_works(self, cls):
        """
        __hash__ returns different hashes for different values.
        """
        a = cls(1, 2)
        b = cls(1, 1)
        assert hash(a) != hash(b)
        # perform the test again to test the pre-cached path through
        # __hash__ for the cached-hash versions
        assert hash(a) != hash(b)

    def test_hash_default(self):
        """
        Classes are not hashable by default.
        """
        C = make_class("C", {})

        with pytest.raises(TypeError) as e:
            hash(C())

        assert e.value.args[0] in (
            "'C' objects are unhashable",  # PyPy
            "unhashable type: 'C'",  # CPython
        )

    def test_cache_hashing(self):
        """
        Ensure that hash computation if cached if and only if requested
        """

        class HashCounter:
            """
            A class for testing which counts how many times its hash
            has been requested
            """

            def __init__(self):
                self.times_hash_called = 0

            def __hash__(self):
                self.times_hash_called += 1
                return 12345

        Uncached = make_class(
            "Uncached",
            {"hash_counter": attr.ib(factory=HashCounter)},
            unsafe_hash=True,
            cache_hash=False,
        )
        Cached = make_class(
            "Cached",
            {"hash_counter": attr.ib(factory=HashCounter)},
            unsafe_hash=True,
            cache_hash=True,
        )

        uncached_instance = Uncached()
        cached_instance = Cached()

        hash(uncached_instance)
        hash(uncached_instance)
        hash(cached_instance)
        hash(cached_instance)

        assert 2 == uncached_instance.hash_counter.times_hash_called
        assert 1 == cached_instance.hash_counter.times_hash_called

    @pytest.mark.parametrize("cache_hash", [True, False])
    def test_copy_hash_cleared(self, cache_hash, frozen, slots):
        """
        Test that the default hash is recalculated after a copy operation.
        """

        kwargs = {"frozen": frozen, "slots": slots, "cache_hash": cache_hash}

        # Give it an explicit hash if we don't have an implicit one
        if not frozen:
            kwargs["unsafe_hash"] = True

        @attr.s(**kwargs)
        class C:
            x = attr.ib()

        a = C(IncrementingHasher())
        # Ensure that any hash cache would be calculated before copy
        orig_hash = hash(a)
        b = copy.deepcopy(a)

        if kwargs["cache_hash"]:
            # For cache_hash classes, this call is cached
            assert orig_hash == hash(a)

        assert orig_hash != hash(b)

    @pytest.mark.parametrize(
        ("klass", "cached"),
        [
            (HashCacheSerializationTestUncached, False),
            (HashCacheSerializationTestCached, True),
            (HashCacheSerializationTestCachedSlots, True),
        ],
    )
    def test_cache_hash_serialization_hash_cleared(self, klass, cached):
        """
        Tests that the hash cache is cleared on deserialization to fix
        https://github.com/python-attrs/attrs/issues/482 .

        This test is intended to guard against a stale hash code surviving
        across serialization (which may cause problems when the hash value
        is different in different interpreters).
        """

        obj = klass(IncrementingHasher())
        original_hash = hash(obj)
        obj_rt = self._roundtrip_pickle(obj)

        if cached:
            assert original_hash == hash(obj)

        assert original_hash != hash(obj_rt)

    def test_copy_two_arg_reduce(self, frozen):
        """
        If __getstate__ returns None, the tuple returned by object.__reduce__
        won't contain the state dictionary; this test ensures that the custom
        __reduce__ generated when cache_hash=True works in that case.
        """

        @attr.s(frozen=frozen, cache_hash=True, unsafe_hash=True)
        class C:
            x = attr.ib()

            def __getstate__(self):
                return None

        # By the nature of this test it doesn't really create an object that's
        # in a valid state - it basically does the equivalent of
        # `object.__new__(C)`, so it doesn't make much sense to assert anything
        # about the result of the copy. This test will just check that it
        # doesn't raise an *error*.
        copy.deepcopy(C(1))

    def _roundtrip_pickle(self, obj):
        pickle_str = pickle.dumps(obj)
        return pickle.loads(pickle_str)


class TestAddInit:
    """
    Tests for `_add_init`.
    """

    @given(booleans(), booleans())
    def test_init(self, slots, frozen):
        """
        If `init` is False, ignore that attribute.
        """
        C = make_class(
            "C",
            {"a": attr.ib(init=False), "b": attr.ib()},
            slots=slots,
            frozen=frozen,
        )
        with pytest.raises(TypeError) as e:
            C(a=1, b=2)

        assert e.value.args[0].endswith(
            "__init__() got an unexpected keyword argument 'a'"
        )

    @given(booleans(), booleans())
    def test_no_init_default(self, slots, frozen):
        """
        If `init` is False but a Factory is specified, don't allow passing that
        argument but initialize it anyway.
        """
        C = make_class(
            "C",
            {
                "_a": attr.ib(init=False, default=42),
                "_b": attr.ib(init=False, default=Factory(list)),
                "c": attr.ib(),
            },
            slots=slots,
            frozen=frozen,
        )
        with pytest.raises(TypeError):
            C(a=1, c=2)
        with pytest.raises(TypeError):
            C(b=1, c=2)

        i = C(23)
        assert (42, [], 23) == (i._a, i._b, i.c)

    @given(booleans(), booleans())
    def test_no_init_order(self, slots, frozen):
        """
        If an attribute is `init=False`, it's legal to come after a mandatory
        attribute.
        """
        make_class(
            "C",
            {"a": attr.ib(default=Factory(list)), "b": attr.ib(init=False)},
            slots=slots,
            frozen=frozen,
        )

    def test_sets_attributes(self):
        """
        The attributes are initialized using the passed keywords.
        """
        obj = InitC(a=1, b=2)
        assert 1 == obj.a
        assert 2 == obj.b

    def test_default(self):
        """
        If a default value is present, it's used as fallback.
        """

        class C:
            __attrs_attrs__ = [
                simple_attr(name="a", default=2),
                simple_attr(name="b", default="hallo"),
                simple_attr(name="c", default=None),
            ]

        C = _add_init(C, False)
        i = C()
        assert 2 == i.a
        assert "hallo" == i.b
        assert None is i.c

    def test_factory(self):
        """
        If a default factory is present, it's used as fallback.
        """

        class D:
            pass

        class C:
            __attrs_attrs__ = [
                simple_attr(name="a", default=Factory(list)),
                simple_attr(name="b", default=Factory(D)),
            ]

        C = _add_init(C, False)
        i = C()

        assert [] == i.a
        assert isinstance(i.b, D)

    def test_factory_takes_self(self):
        """
        If takes_self on factories is True, self is passed.
        """
        C = make_class(
            "C",
            {
                "x": attr.ib(
                    default=Factory((lambda self: self), takes_self=True)
                )
            },
        )

        i = C()

        assert i is i.x

    def test_factory_hashable(self):
        """
        Factory is hashable.
        """
        assert hash(Factory(None, False)) == hash(Factory(None, False))

    def test_validator(self):
        """
        If a validator is passed, call it with the preliminary instance, the
        Attribute, and the argument.
        """

        class VException(Exception):
            pass

        def raiser(*args):
            raise VException(*args)

        C = make_class("C", {"a": attr.ib("a", validator=raiser)})
        with pytest.raises(VException) as e:
            C(42)

        assert (fields(C).a, 42) == e.value.args[1:]
        assert isinstance(e.value.args[0], C)

    def test_validator_slots(self):
        """
        If a validator is passed, call it with the preliminary instance, the
        Attribute, and the argument.
        """

        class VException(Exception):
            pass

        def raiser(*args):
            raise VException(*args)

        C = make_class("C", {"a": attr.ib("a", validator=raiser)}, slots=True)
        with pytest.raises(VException) as e:
            C(42)

        assert (fields(C)[0], 42) == e.value.args[1:]
        assert isinstance(e.value.args[0], C)

    @given(booleans())
    def test_validator_others(self, slots):
        """
        Does not interfere when setting non-attrs attributes.
        """
        C = make_class(
            "C", {"a": attr.ib("a", validator=instance_of(int))}, slots=slots
        )
        i = C(1)

        assert 1 == i.a

        if not slots:
            i.b = "foo"
            assert "foo" == i.b
        else:
            with pytest.raises(AttributeError):
                i.b = "foo"

    def test_underscores(self):
        """
        The argument names in `__init__` are without leading and trailing
        underscores.
        """

        class C:
            __attrs_attrs__ = [simple_attr("_private")]

        C = _add_init(C, False)
        i = C(private=42)
        assert 42 == i._private


class TestNothing:
    """
    Tests for `NOTHING`.
    """

    def test_copy(self):
        """
        __copy__ returns the same object.
        """
        n = NOTHING
        assert n is copy.copy(n)

    def test_deepcopy(self):
        """
        __deepcopy__ returns the same object.
        """
        n = NOTHING
        assert n is copy.deepcopy(n)

    def test_eq(self):
        """
        All instances are equal.
        """
        assert NOTHING == NOTHING == NOTHING
        assert not (NOTHING != NOTHING)
        assert 1 != NOTHING

    def test_false(self):
        """
        NOTHING evaluates as falsey.
        """
        assert not NOTHING
        assert False is bool(NOTHING)


@attr.s(unsafe_hash=True, order=True)
class C:
    pass


# Store this class so that we recreate it.
OriginalC = C


@attr.s(unsafe_hash=True, order=True)
class C:
    pass


CopyC = C


@attr.s(unsafe_hash=True, order=True)
class C:
    """
    A different class, to generate different methods.
    """

    a = attr.ib()


class TestFilenames:
    def test_filenames(self):
        """
        The created dunder methods have a "consistent" filename.
        """
        assert (
            OriginalC.__init__.__code__.co_filename
            == "<attrs generated methods tests.test_dunders.C>"
        )
        assert (
            OriginalC.__eq__.__code__.co_filename
            == "<attrs generated methods tests.test_dunders.C>"
        )
        assert (
            OriginalC.__hash__.__code__.co_filename
            == "<attrs generated methods tests.test_dunders.C>"
        )
        assert (
            CopyC.__init__.__code__.co_filename
            == "<attrs generated methods tests.test_dunders.C>"
        )
        assert (
            CopyC.__eq__.__code__.co_filename
            == "<attrs generated methods tests.test_dunders.C>"
        )
        assert (
            CopyC.__hash__.__code__.co_filename
            == "<attrs generated methods tests.test_dunders.C>"
        )
        assert (
            C.__init__.__code__.co_filename
            == "<attrs generated methods tests.test_dunders.C-1>"
        )
        assert (
            C.__eq__.__code__.co_filename
            == "<attrs generated methods tests.test_dunders.C-1>"
        )
        assert (
            C.__hash__.__code__.co_filename
            == "<attrs generated methods tests.test_dunders.C-1>"
        )


# SPDX-License-Identifier: MIT


import pytest

from attr import VersionInfo


@pytest.fixture(name="vi")
def fixture_vi():
    return VersionInfo(19, 2, 0, "final")


class TestVersionInfo:
    def test_from_string_no_releaselevel(self, vi):
        """
        If there is no suffix, the releaselevel becomes "final" by default.
        """
        assert vi == VersionInfo._from_version_string("19.2.0")

    def test_suffix_is_preserved(self):
        """
        If there is a suffix, it's preserved.
        """
        assert (
            "dev0"
            == VersionInfo._from_version_string("19.2.0.dev0").releaselevel
        )

    @pytest.mark.parametrize("other", [(), (19, 2, 0, "final", "garbage")])
    def test_wrong_len(self, vi, other):
        """
        Comparing with a tuple that has the wrong length raises an error.
        """
        assert vi != other

        with pytest.raises(TypeError):
            vi < other

    @pytest.mark.parametrize("other", [[19, 2, 0, "final"]])
    def test_wrong_type(self, vi, other):
        """
        Only compare to other VersionInfos or tuples.
        """
        assert vi != other

    def test_order(self, vi):
        """
        Ordering works as expected.
        """
        assert vi < (20,)
        assert vi < (19, 2, 1)
        assert vi > (0,)
        assert vi <= (19, 2)
        assert vi >= (19, 2)
        assert vi > (19, 2, 0, "dev0")
        assert vi < (19, 2, 0, "post1")


# SPDX-License-Identifier: MIT

"""
Tests for `attr.filters`.
"""

import pytest

import attr

from attr import fields
from attr.filters import _split_what, exclude, include


@attr.s
class C:
    a = attr.ib()
    b = attr.ib()


class TestSplitWhat:
    """
    Tests for `_split_what`.
    """

    def test_splits(self):
        """
        Splits correctly.
        """
        assert (
            frozenset((int, str)),
            frozenset(("abcd", "123")),
            frozenset((fields(C).a,)),
        ) == _split_what((str, "123", fields(C).a, int, "abcd"))


class TestInclude:
    """
    Tests for `include`.
    """

    @pytest.mark.parametrize(
        ("incl", "value"),
        [
            ((int,), 42),
            ((str,), "hello"),
            ((str, fields(C).a), 42),
            ((str, fields(C).b), "hello"),
            (("a",), 42),
            (("a",), "hello"),
            (("a", str), 42),
            (("a", fields(C).b), "hello"),
        ],
    )
    def test_allow(self, incl, value):
        """
        Return True if a class or attribute is included.
        """
        i = include(*incl)
        assert i(fields(C).a, value) is True

    @pytest.mark.parametrize(
        ("incl", "value"),
        [
            ((str,), 42),
            ((int,), "hello"),
            ((str, fields(C).b), 42),
            ((int, fields(C).b), "hello"),
            (("b",), 42),
            (("b",), "hello"),
            (("b", str), 42),
            (("b", fields(C).b), "hello"),
        ],
    )
    def test_drop_class(self, incl, value):
        """
        Return False on non-included classes and attributes.
        """
        i = include(*incl)
        assert i(fields(C).a, value) is False


class TestExclude:
    """
    Tests for `exclude`.
    """

    @pytest.mark.parametrize(
        ("excl", "value"),
        [
            ((str,), 42),
            ((int,), "hello"),
            ((str, fields(C).b), 42),
            ((int, fields(C).b), "hello"),
            (("b",), 42),
            (("b",), "hello"),
            (("b", str), 42),
            (("b", fields(C).b), "hello"),
        ],
    )
    def test_allow(self, excl, value):
        """
        Return True if class or attribute is not excluded.
        """
        e = exclude(*excl)
        assert e(fields(C).a, value) is True

    @pytest.mark.parametrize(
        ("excl", "value"),
        [
            ((int,), 42),
            ((str,), "hello"),
            ((str, fields(C).a), 42),
            ((str, fields(C).b), "hello"),
            (("a",), 42),
            (("a",), "hello"),
            (("a", str), 42),
            (("a", fields(C).b), "hello"),
        ],
    )
    def test_drop_class(self, excl, value):
        """
        Return True on non-excluded classes and attributes.
        """
        e = exclude(*excl)
        assert e(fields(C).a, value) is False


# SPDX-License-Identifier: MIT

import pytest

import attr


class TestPatternMatching:
    """
    Pattern matching syntax test cases.
    """

    @pytest.mark.parametrize("dec", [attr.s, attr.define, attr.frozen])
    def test_simple_match_case(self, dec):
        """
        Simple match case statement works as expected with all class
        decorators.
        """

        @dec
        class C:
            a = attr.ib()

        assert ("a",) == C.__match_args__

        matched = False
        c = C(a=1)
        match c:
            case C(a):
                matched = True

        assert matched
        assert 1 == a

    def test_explicit_match_args(self):
        """
        Does not overwrite a manually set empty __match_args__.
        """

        ma = ()

        @attr.define
        class C:
            a = attr.field()
            __match_args__ = ma

        c = C(a=1)

        msg = r"C\(\) accepts 0 positional sub-patterns \(1 given\)"
        with pytest.raises(TypeError, match=msg):
            match c:
                case C(_):
                    pass

    def test_match_args_kw_only(self):
        """
        kw_only classes don't generate __match_args__.
        kw_only fields are not included in __match_args__.
        """

        @attr.define
        class C:
            a = attr.field(kw_only=True)
            b = attr.field()

        assert ("b",) == C.__match_args__

        c = C(a=1, b=1)
        msg = r"C\(\) accepts 1 positional sub-pattern \(2 given\)"
        with pytest.raises(TypeError, match=msg):
            match c:
                case C(a, b):
                    pass

        found = False
        match c:
            case C(b, a=a):
                found = True

        assert found

        @attr.define(kw_only=True)
        class C:
            a = attr.field()
            b = attr.field()

        c = C(a=1, b=1)
        msg = r"C\(\) accepts 0 positional sub-patterns \(2 given\)"
        with pytest.raises(TypeError, match=msg):
            match c:
                case C(a, b):
                    pass

        found = False
        match c:
            case C(a=a, b=b):
                found = True

        assert found
        assert (1, 1) == (a, b)


# SPDX-License-Identifier: MIT

"""
Integration tests for next-generation APIs.
"""

import re

from contextlib import contextmanager
from functools import partial

import pytest

import attr as _attr  # don't use it by accident
import attrs

from attr._compat import PY_3_11_PLUS


@attrs.define
class C:
    x: str
    y: int


class TestNextGen:
    def test_simple(self):
        """
        Instantiation works.
        """
        C("1", 2)

    def test_field_type(self):
        """
        Make class with attrs.field and type parameter.
        """
        classFields = {"testint": attrs.field(type=int)}

        A = attrs.make_class("A", classFields)

        assert int is attrs.fields(A).testint.type

    def test_no_slots(self):
        """
        slots can be deactivated.
        """

        @attrs.define(slots=False)
        class NoSlots:
            x: int

        ns = NoSlots(1)

        assert {"x": 1} == ns.__dict__

    def test_validates(self):
        """
        Validators at __init__ and __setattr__ work.
        """

        @attrs.define
        class Validated:
            x: int = attrs.field(validator=attrs.validators.instance_of(int))

        v = Validated(1)

        with pytest.raises(TypeError):
            Validated(None)

        with pytest.raises(TypeError):
            v.x = "1"

    def test_no_order(self):
        """
        Order is off by default but can be added.
        """
        with pytest.raises(TypeError):
            C("1", 2) < C("2", 3)

        @attrs.define(order=True)
        class Ordered:
            x: int

        assert Ordered(1) < Ordered(2)

    def test_override_auto_attribs_true(self):
        """
        Don't guess if auto_attrib is set explicitly.

        Having an unannotated attrs.ib/attrs.field fails.
        """
        with pytest.raises(attrs.exceptions.UnannotatedAttributeError):

            @attrs.define(auto_attribs=True)
            class ThisFails:
                x = attrs.field()
                y: int

    def test_override_auto_attribs_false(self):
        """
        Don't guess if auto_attrib is set explicitly.

        Annotated fields that don't carry an attrs.ib are ignored.
        """

        @attrs.define(auto_attribs=False)
        class NoFields:
            x: int
            y: int

        assert NoFields() == NoFields()

    def test_auto_attribs_detect(self):
        """
        define correctly detects if a class lacks type annotations.
        """

        @attrs.define
        class OldSchool:
            x = attrs.field()

        assert OldSchool(1) == OldSchool(1)

        # Test with maybe_cls = None
        @attrs.define()
        class OldSchool2:
            x = attrs.field()

        assert OldSchool2(1) == OldSchool2(1)

    def test_auto_attribs_detect_fields_and_annotations(self):
        """
        define infers auto_attribs=True if fields have type annotations
        """

        @attrs.define
        class NewSchool:
            x: int
            y: list = attrs.field()

            @y.validator
            def _validate_y(self, attribute, value):
                if value < 0:
                    raise ValueError("y must be positive")

        assert NewSchool(1, 1) == NewSchool(1, 1)
        with pytest.raises(ValueError):
            NewSchool(1, -1)
        assert list(attrs.fields_dict(NewSchool).keys()) == ["x", "y"]

    def test_auto_attribs_partially_annotated(self):
        """
        define infers auto_attribs=True if any type annotations are found
        """

        @attrs.define
        class NewSchool:
            x: int
            y: list
            z = 10

        # fields are defined for any annotated attributes
        assert NewSchool(1, []) == NewSchool(1, [])
        assert list(attrs.fields_dict(NewSchool).keys()) == ["x", "y"]

        # while the unannotated attributes are left as class vars
        assert NewSchool.z == 10
        assert "z" in NewSchool.__dict__

    def test_auto_attribs_detect_annotations(self):
        """
        define correctly detects if a class has type annotations.
        """

        @attrs.define
        class NewSchool:
            x: int

        assert NewSchool(1) == NewSchool(1)

        # Test with maybe_cls = None
        @attrs.define()
        class NewSchool2:
            x: int

        assert NewSchool2(1) == NewSchool2(1)

    def test_exception(self):
        """
        Exceptions are detected and correctly handled.
        """

        @attrs.define
        class E(Exception):
            msg: str
            other: int

        with pytest.raises(E) as ei:
            raise E("yolo", 42)

        e = ei.value

        assert ("yolo", 42) == e.args
        assert "yolo" == e.msg
        assert 42 == e.other

    def test_frozen(self):
        """
        attrs.frozen freezes classes.
        """

        @attrs.frozen
        class F:
            x: str

        f = F(1)

        with pytest.raises(attrs.exceptions.FrozenInstanceError):
            f.x = 2

    def test_auto_detect_eq(self):
        """
        auto_detect=True works for eq.

        Regression test for #670.
        """

        @attrs.define
        class C:
            def __eq__(self, o):
                raise ValueError

        with pytest.raises(ValueError):
            C() == C()

    def test_subclass_frozen(self):
        """
        It's possible to subclass an `attrs.frozen` class and the frozen-ness
        is inherited.
        """

        @attrs.frozen
        class A:
            a: int

        @attrs.frozen
        class B(A):
            b: int

        @attrs.define(on_setattr=attrs.setters.NO_OP)
        class C(B):
            c: int

        assert B(1, 2) == B(1, 2)
        assert C(1, 2, 3) == C(1, 2, 3)

        with pytest.raises(attrs.exceptions.FrozenInstanceError):
            A(1).a = 1

        with pytest.raises(attrs.exceptions.FrozenInstanceError):
            B(1, 2).a = 1

        with pytest.raises(attrs.exceptions.FrozenInstanceError):
            B(1, 2).b = 2

        with pytest.raises(attrs.exceptions.FrozenInstanceError):
            C(1, 2, 3).c = 3

    def test_catches_frozen_on_setattr(self):
        """
        Passing frozen=True and on_setattr hooks is caught, even if the
        immutability is inherited.
        """

        @attrs.define(frozen=True)
        class A:
            pass

        with pytest.raises(
            ValueError, match="Frozen classes can't use on_setattr."
        ):

            @attrs.define(frozen=True, on_setattr=attrs.setters.validate)
            class B:
                pass

        with pytest.raises(
            ValueError,
            match=re.escape(
                "Frozen classes can't use on_setattr "
                "(frozen-ness was inherited)."
            ),
        ):

            @attrs.define(on_setattr=attrs.setters.validate)
            class C(A):
                pass

    @pytest.mark.parametrize(
        "decorator",
        [
            partial(_attr.s, frozen=True, slots=True, auto_exc=True),
            attrs.frozen,
            attrs.define,
            attrs.mutable,
        ],
    )
    def test_discard_context(self, decorator):
        """
        raise from None works.

        Regression test for #703.
        """

        @decorator
        class MyException(Exception):
            x: str = attrs.field()

        with pytest.raises(MyException) as ei:
            try:
                raise ValueError
            except ValueError:
                raise MyException("foo") from None

        assert "foo" == ei.value.x
        assert ei.value.__cause__ is None

    @pytest.mark.parametrize(
        "decorator",
        [
            partial(_attr.s, frozen=True, slots=True, auto_exc=True),
            attrs.frozen,
            attrs.define,
            attrs.mutable,
        ],
    )
    def test_setting_exception_mutable_attributes(self, decorator):
        """
        contextlib.contextlib (re-)sets __traceback__ on raised exceptions.

        Ensure that works, as well as if done explicitly
        """

        @decorator
        class MyException(Exception):
            pass

        @contextmanager
        def do_nothing():
            yield

        with do_nothing(), pytest.raises(MyException) as ei:
            raise MyException

        assert isinstance(ei.value, MyException)

        # this should not raise an exception either
        ei.value.__traceback__ = ei.value.__traceback__
        ei.value.__cause__ = ValueError("cause")
        ei.value.__context__ = TypeError("context")
        ei.value.__suppress_context__ = True
        ei.value.__suppress_context__ = False
        ei.value.__notes__ = []
        del ei.value.__notes__

        if PY_3_11_PLUS:
            ei.value.add_note("note")
            del ei.value.__notes__

    def test_converts_and_validates_by_default(self):
        """
        If no on_setattr is set, assume setters.convert, setters.validate.
        """

        @attrs.define
        class C:
            x: int = attrs.field(converter=int)

            @x.validator
            def _v(self, _, value):
                if value < 10:
                    raise ValueError("must be >=10")

        inst = C(10)

        # Converts
        inst.x = "11"

        assert 11 == inst.x

        # Validates
        with pytest.raises(ValueError, match="must be >=10"):
            inst.x = "9"

    def test_mro_ng(self):
        """
        Attributes and methods are looked up the same way in NG by default.

        See #428
        """

        @attrs.define
        class A:
            x: int = 10

            def xx(self):
                return 10

        @attrs.define
        class B(A):
            y: int = 20

        @attrs.define
        class C(A):
            x: int = 50

            def xx(self):
                return 50

        @attrs.define
        class D(B, C):
            pass

        d = D()

        assert d.x == d.xx()


class TestAsTuple:
    def test_smoke(self):
        """
        `attrs.astuple` only changes defaults, so we just call it and compare.
        """
        inst = C("foo", 42)

        assert attrs.astuple(inst) == _attr.astuple(inst)


class TestAsDict:
    def test_smoke(self):
        """
        `attrs.asdict` only changes defaults, so we just call it and compare.
        """
        inst = C("foo", {(1,): 42})

        assert attrs.asdict(inst) == _attr.asdict(
            inst, retain_collection_types=True
        )


class TestImports:
    """
    Verify our re-imports and mirroring works.
    """

    def test_converters(self):
        """
        Importing from attrs.converters works.
        """
        from attrs.converters import optional

        assert optional is _attr.converters.optional

    def test_exceptions(self):
        """
        Importing from attrs.exceptions works.
        """
        from attrs.exceptions import FrozenError

        assert FrozenError is _attr.exceptions.FrozenError

    def test_filters(self):
        """
        Importing from attrs.filters works.
        """
        from attrs.filters import include

        assert include is _attr.filters.include

    def test_setters(self):
        """
        Importing from attrs.setters works.
        """
        from attrs.setters import pipe

        assert pipe is _attr.setters.pipe

    def test_validators(self):
        """
        Importing from attrs.validators works.
        """
        from attrs.validators import and_

        assert and_ is _attr.validators.and_


# SPDX-License-Identifier: MIT

"""
Tests for methods from `attrib._cmp`.
"""

import pytest

from attr._cmp import cmp_using
from attr._compat import PY_3_13_PLUS


# Test parameters.
EqCSameType = cmp_using(eq=lambda a, b: a == b, class_name="EqCSameType")
PartialOrderCSameType = cmp_using(
    eq=lambda a, b: a == b,
    lt=lambda a, b: a < b,
    class_name="PartialOrderCSameType",
)
FullOrderCSameType = cmp_using(
    eq=lambda a, b: a == b,
    lt=lambda a, b: a < b,
    le=lambda a, b: a <= b,
    gt=lambda a, b: a > b,
    ge=lambda a, b: a >= b,
    class_name="FullOrderCSameType",
)

EqCAnyType = cmp_using(
    eq=lambda a, b: a == b, require_same_type=False, class_name="EqCAnyType"
)
PartialOrderCAnyType = cmp_using(
    eq=lambda a, b: a == b,
    lt=lambda a, b: a < b,
    require_same_type=False,
    class_name="PartialOrderCAnyType",
)


eq_data = [
    (EqCSameType, True),
    (EqCAnyType, False),
]

order_data = [
    (PartialOrderCSameType, True),
    (PartialOrderCAnyType, False),
    (FullOrderCSameType, True),
]

eq_ids = [c[0].__name__ for c in eq_data]
order_ids = [c[0].__name__ for c in order_data]

cmp_data = eq_data + order_data
cmp_ids = eq_ids + order_ids

# Compiler strips indents from docstrings in Python 3.13+
indent = "" if PY_3_13_PLUS else " " * 8


class TestEqOrder:
    """
    Tests for eq and order related methods.
    """

    #########
    # eq
    #########
    @pytest.mark.parametrize(
        ("cls", "requires_same_type"), cmp_data, ids=cmp_ids
    )
    def test_equal_same_type(self, cls, requires_same_type):
        """
        Equal objects are detected as equal.
        """
        assert cls(1) == cls(1)
        assert not (cls(1) != cls(1))

    @pytest.mark.parametrize(
        ("cls", "requires_same_type"), cmp_data, ids=cmp_ids
    )
    def test_unequal_same_type(self, cls, requires_same_type):
        """
        Unequal objects of correct type are detected as unequal.
        """
        assert cls(1) != cls(2)
        assert not (cls(1) == cls(2))

    @pytest.mark.parametrize(
        ("cls", "requires_same_type"), cmp_data, ids=cmp_ids
    )
    def test_equal_different_type(self, cls, requires_same_type):
        """
        Equal values of different types are detected appropriately.
        """
        assert (cls(1) == cls(1.0)) == (not requires_same_type)
        assert not (cls(1) != cls(1.0)) == (not requires_same_type)

    #########
    # lt
    #########
    @pytest.mark.parametrize(
        ("cls", "requires_same_type"), eq_data, ids=eq_ids
    )
    def test_lt_unorderable(self, cls, requires_same_type):
        """
        TypeError is raised if class does not implement __lt__.
        """
        with pytest.raises(TypeError):
            cls(1) < cls(2)

    @pytest.mark.parametrize(
        ("cls", "requires_same_type"), order_data, ids=order_ids
    )
    def test_lt_same_type(self, cls, requires_same_type):
        """
        Less-than objects are detected appropriately.
        """
        assert cls(1) < cls(2)
        assert not (cls(2) < cls(1))

    @pytest.mark.parametrize(
        ("cls", "requires_same_type"), order_data, ids=order_ids
    )
    def test_not_lt_same_type(self, cls, requires_same_type):
        """
        Not less-than objects are detected appropriately.
        """
        assert cls(2) >= cls(1)
        assert not (cls(1) >= cls(2))

    @pytest.mark.parametrize(
        ("cls", "requires_same_type"), order_data, ids=order_ids
    )
    def test_lt_different_type(self, cls, requires_same_type):
        """
        Less-than values of different types are detected appropriately.
        """
        if requires_same_type:
            # Unlike __eq__, NotImplemented will cause an exception to be
            # raised from __lt__.
            with pytest.raises(TypeError):
                cls(1) < cls(2.0)
        else:
            assert cls(1) < cls(2.0)
            assert not (cls(2) < cls(1.0))

    #########
    # le
    #########
    @pytest.mark.parametrize(
        ("cls", "requires_same_type"), eq_data, ids=eq_ids
    )
    def test_le_unorderable(self, cls, requires_same_type):
        """
        TypeError is raised if class does not implement __le__.
        """
        with pytest.raises(TypeError):
            cls(1) <= cls(2)

    @pytest.mark.parametrize(
        ("cls", "requires_same_type"), order_data, ids=order_ids
    )
    def test_le_same_type(self, cls, requires_same_type):
        """
        Less-than-or-equal objects are detected appropriately.
        """
        assert cls(1) <= cls(1)
        assert cls(1) <= cls(2)
        assert not (cls(2) <= cls(1))

    @pytest.mark.parametrize(
        ("cls", "requires_same_type"), order_data, ids=order_ids
    )
    def test_not_le_same_type(self, cls, requires_same_type):
        """
        Not less-than-or-equal objects are detected appropriately.
        """
        assert cls(2) > cls(1)
        assert not (cls(1) > cls(1))
        assert not (cls(1) > cls(2))

    @pytest.mark.parametrize(
        ("cls", "requires_same_type"), order_data, ids=order_ids
    )
    def test_le_different_type(self, cls, requires_same_type):
        """
        Less-than-or-equal values of diff. types are detected appropriately.
        """
        if requires_same_type:
            # Unlike __eq__, NotImplemented will cause an exception to be
            # raised from __le__.
            with pytest.raises(TypeError):
                cls(1) <= cls(2.0)
        else:
            assert cls(1) <= cls(2.0)
            assert cls(1) <= cls(1.0)
            assert not (cls(2) <= cls(1.0))

    #########
    # gt
    #########
    @pytest.mark.parametrize(
        ("cls", "requires_same_type"), eq_data, ids=eq_ids
    )
    def test_gt_unorderable(self, cls, requires_same_type):
        """
        TypeError is raised if class does not implement __gt__.
        """
        with pytest.raises(TypeError):
            cls(2) > cls(1)

    @pytest.mark.parametrize(
        ("cls", "requires_same_type"), order_data, ids=order_ids
    )
    def test_gt_same_type(self, cls, requires_same_type):
        """
        Greater-than objects are detected appropriately.
        """
        assert cls(2) > cls(1)
        assert not (cls(1) > cls(2))

    @pytest.mark.parametrize(
        ("cls", "requires_same_type"), order_data, ids=order_ids
    )
    def test_not_gt_same_type(self, cls, requires_same_type):
        """
        Not greater-than objects are detected appropriately.
        """
        assert cls(1) <= cls(2)
        assert not (cls(2) <= cls(1))

    @pytest.mark.parametrize(
        ("cls", "requires_same_type"), order_data, ids=order_ids
    )
    def test_gt_different_type(self, cls, requires_same_type):
        """
        Greater-than values of different types are detected appropriately.
        """
        if requires_same_type:
            # Unlike __eq__, NotImplemented will cause an exception to be
            # raised from __gt__.
            with pytest.raises(TypeError):
                cls(2) > cls(1.0)
        else:
            assert cls(2) > cls(1.0)
            assert not (cls(1) > cls(2.0))

    #########
    # ge
    #########
    @pytest.mark.parametrize(
        ("cls", "requires_same_type"), eq_data, ids=eq_ids
    )
    def test_ge_unorderable(self, cls, requires_same_type):
        """
        TypeError is raised if class does not implement __ge__.
        """
        with pytest.raises(TypeError):
            cls(2) >= cls(1)

    @pytest.mark.parametrize(
        ("cls", "requires_same_type"), order_data, ids=order_ids
    )
    def test_ge_same_type(self, cls, requires_same_type):
        """
        Greater-than-or-equal objects are detected appropriately.
        """
        assert cls(1) >= cls(1)
        assert cls(2) >= cls(1)
        assert not (cls(1) >= cls(2))

    @pytest.mark.parametrize(
        ("cls", "requires_same_type"), order_data, ids=order_ids
    )
    def test_not_ge_same_type(self, cls, requires_same_type):
        """
        Not greater-than-or-equal objects are detected appropriately.
        """
        assert cls(1) < cls(2)
        assert not (cls(1) < cls(1))
        assert not (cls(2) < cls(1))

    @pytest.mark.parametrize(
        ("cls", "requires_same_type"), order_data, ids=order_ids
    )
    def test_ge_different_type(self, cls, requires_same_type):
        """
        Greater-than-or-equal values of diff. types are detected appropriately.
        """
        if requires_same_type:
            # Unlike __eq__, NotImplemented will cause an exception to be
            # raised from __ge__.
            with pytest.raises(TypeError):
                cls(2) >= cls(1.0)
        else:
            assert cls(2) >= cls(2.0)
            assert cls(2) >= cls(1.0)
            assert not (cls(1) >= cls(2.0))


class TestDundersUnnamedClass:
    """
    Tests for dunder attributes of unnamed classes.
    """

    cls = cmp_using(eq=lambda a, b: a == b)

    def test_class(self):
        """
        Class name and qualified name should be well behaved.
        """
        assert self.cls.__name__ == "Comparable"
        assert self.cls.__qualname__ == "Comparable"

    def test_eq(self):
        """
        __eq__ docstring and qualified name should be well behaved.
        """
        method = self.cls.__eq__
        assert method.__doc__.strip() == "Return a == b.  Computed by attrs."
        assert method.__name__ == "__eq__"

    def test_ne(self):
        """
        __ne__ docstring and qualified name should be well behaved.
        """
        method = self.cls.__ne__
        assert method.__doc__.strip() == (
            "Check equality and either forward a NotImplemented or\n"
            f"{'' if PY_3_13_PLUS else ' ' * 4}return the result negated."
        )
        assert method.__name__ == "__ne__"


class TestTotalOrderingException:
    """
    Test for exceptions related to total ordering.
    """

    def test_eq_must_specified(self):
        """
        `total_ordering` requires `__eq__` to be specified.
        """
        with pytest.raises(ValueError) as ei:
            cmp_using(lt=lambda a, b: a < b)

        assert ei.value.args[0] == (
            "eq must be define is order to complete ordering from "
            "lt, le, gt, ge."
        )


class TestNotImplementedIsPropagated:
    """
    Test related to functions that return NotImplemented.
    """

    def test_not_implemented_is_propagated(self):
        """
        If the comparison function returns NotImplemented,
        the dunder method should too.
        """
        C = cmp_using(eq=lambda a, b: NotImplemented if a == 1 else a == b)

        assert C(2) == C(2)
        assert C(1) != C(1)


class TestDundersPartialOrdering:
    """
    Tests for dunder attributes of classes with partial ordering.
    """

    cls = PartialOrderCSameType

    def test_class(self):
        """
        Class name and qualified name should be well behaved.
        """
        assert self.cls.__name__ == "PartialOrderCSameType"
        assert self.cls.__qualname__ == "PartialOrderCSameType"

    def test_eq(self):
        """
        __eq__ docstring and qualified name should be well behaved.
        """
        method = self.cls.__eq__
        assert method.__doc__.strip() == "Return a == b.  Computed by attrs."
        assert method.__name__ == "__eq__"

    def test_ne(self):
        """
        __ne__ docstring and qualified name should be well behaved.
        """
        method = self.cls.__ne__
        assert method.__doc__.strip() == (
            "Check equality and either forward a NotImplemented or\n"
            f"{'' if PY_3_13_PLUS else ' ' * 4}return the result negated."
        )
        assert method.__name__ == "__ne__"

    def test_lt(self):
        """
        __lt__ docstring and qualified name should be well behaved.
        """
        method = self.cls.__lt__
        assert method.__doc__.strip() == "Return a < b.  Computed by attrs."
        assert method.__name__ == "__lt__"

    def test_le(self):
        """
        __le__ docstring and qualified name should be well behaved.
        """
        method = self.cls.__le__
        assert method.__doc__.strip().startswith(
            "Return a <= b.  Computed by @total_ordering from"
        )
        assert method.__name__ == "__le__"

    def test_gt(self):
        """
        __gt__ docstring and qualified name should be well behaved.
        """
        method = self.cls.__gt__
        assert method.__doc__.strip().startswith(
            "Return a > b.  Computed by @total_ordering from"
        )
        assert method.__name__ == "__gt__"

    def test_ge(self):
        """
        __ge__ docstring and qualified name should be well behaved.
        """
        method = self.cls.__ge__
        assert method.__doc__.strip().startswith(
            "Return a >= b.  Computed by @total_ordering from"
        )
        assert method.__name__ == "__ge__"


class TestDundersFullOrdering:
    """
    Tests for dunder attributes of classes with full ordering.
    """

    cls = FullOrderCSameType

    def test_class(self):
        """
        Class name and qualified name should be well behaved.
        """
        assert self.cls.__name__ == "FullOrderCSameType"
        assert self.cls.__qualname__ == "FullOrderCSameType"

    def test_eq(self):
        """
        __eq__ docstring and qualified name should be well behaved.
        """
        method = self.cls.__eq__
        assert method.__doc__.strip() == "Return a == b.  Computed by attrs."
        assert method.__name__ == "__eq__"

    def test_ne(self):
        """
        __ne__ docstring and qualified name should be well behaved.
        """
        method = self.cls.__ne__
        assert method.__doc__.strip() == (
            "Check equality and either forward a NotImplemented or\n"
            f"{'' if PY_3_13_PLUS else ' ' * 4}return the result negated."
        )
        assert method.__name__ == "__ne__"

    def test_lt(self):
        """
        __lt__ docstring and qualified name should be well behaved.
        """
        method = self.cls.__lt__
        assert method.__doc__.strip() == "Return a < b.  Computed by attrs."
        assert method.__name__ == "__lt__"

    def test_le(self):
        """
        __le__ docstring and qualified name should be well behaved.
        """
        method = self.cls.__le__
        assert method.__doc__.strip() == "Return a <= b.  Computed by attrs."
        assert method.__name__ == "__le__"

    def test_gt(self):
        """
        __gt__ docstring and qualified name should be well behaved.
        """
        method = self.cls.__gt__
        assert method.__doc__.strip() == "Return a > b.  Computed by attrs."
        assert method.__name__ == "__gt__"

    def test_ge(self):
        """
        __ge__ docstring and qualified name should be well behaved.
        """
        method = self.cls.__ge__
        assert method.__doc__.strip() == "Return a >= b.  Computed by attrs."
        assert method.__name__ == "__ge__"


# SPDX-License-Identifier: MIT

import attr
import attrs


@attr.define()
class Define:
    a: str
    b: int


reveal_type(Define.__init__)  # noqa: F821


@attr.define()
class DefineConverter:
    with_converter: int = attr.field(converter=int)


reveal_type(DefineConverter.__init__)  # noqa: F821

DefineConverter(with_converter=b"42")


@attr.frozen()
class Frozen:
    a: str


d = Frozen("a")
d.a = "new"

reveal_type(d.a)  # noqa: F821


@attr.define(frozen=True)
class FrozenDefine:
    a: str


d2 = FrozenDefine("a")
d2.a = "new"

reveal_type(d2.a)  # noqa: F821


# Field-aliasing works
@attrs.define
class AliasedField:
    _a: int = attrs.field(alias="_a")


af = AliasedField(42)

reveal_type(af.__init__)  # noqa: F821


# unsafe_hash is accepted
@attrs.define(unsafe_hash=True)
class Hashable:
    pass


# SPDX-License-Identifier: MIT

"""
Tests for `__init_subclass__` related functionality.
"""

import attr


def test_init_subclass_vanilla(slots):
    """
    `super().__init_subclass__` can be used if the subclass is not an attrs
    class both with dict and slotted classes.
    """

    @attr.s(slots=slots)
    class Base:
        def __init_subclass__(cls, param, **kw):
            super().__init_subclass__(**kw)
            cls.param = param

    class Vanilla(Base, param="foo"):
        pass

    assert "foo" == Vanilla().param


def test_init_subclass_attrs():
    """
    `__init_subclass__` works with attrs classes as long as slots=False.
    """

    @attr.s(slots=False)
    class Base:
        def __init_subclass__(cls, param, **kw):
            super().__init_subclass__(**kw)
            cls.param = param

    @attr.s
    class Attrs(Base, param="foo"):
        pass

    assert "foo" == Attrs().param


def test_init_subclass_slots_workaround():
    """
    `__init_subclass__` works with modern APIs if care is taken around classes
    existing twice.
    """
    subs = {}

    @attr.define
    class Base:
        def __init_subclass__(cls):
            subs[cls.__qualname__] = cls

    @attr.define
    class Sub1(Base):
        x: int

    @attr.define
    class Sub2(Base):
        y: int

    assert (Sub1, Sub2) == tuple(subs.values())


# SPDX-License-Identifier: MIT

from __future__ import annotations

import re

from typing import Any, Dict, List, Tuple

import attr
import attrs


# Typing via "type" Argument ---


@attr.s
class C:
    a = attr.ib(type=int)


c = C(1)
C(a=1)


@attr.s
class D:
    x = attr.ib(type=List[int])


@attr.s
class E:
    y = attr.ib(type="List[int]")


@attr.s
class F:
    z = attr.ib(type=Any)


# Typing via Annotations ---


@attr.s
class CC:
    a: int = attr.ib()


cc = CC(1)
CC(a=1)


@attr.s
class DD:
    x: list[int] = attr.ib()


@attr.s
class EE:
    y: "list[int]" = attr.ib()


@attr.s
class FF:
    z: Any = attr.ib()


@attrs.define
class FFF:
    z: int


FFF(1)


# Inheritance --


@attr.s
class GG(DD):
    y: str = attr.ib()


GG(x=[1], y="foo")


@attr.s
class HH(DD, EE):
    z: float = attr.ib()


HH(x=[1], y=[], z=1.1)


# same class
c == cc


# Exceptions
@attr.s(auto_exc=True)
class Error(Exception):
    x: int = attr.ib()


try:
    raise Error(1)
except Error as e:
    e.x
    e.args
    str(e)


@attrs.define
class Error2(Exception):
    x: int


try:
    raise Error2(1)
except Error as e:
    e.x
    e.args
    str(e)

# Field aliases


@attrs.define
class AliasExample:
    without_alias: int
    _with_alias: int = attr.ib(alias="_with_alias")


attr.fields(AliasExample).without_alias.alias
attr.fields(AliasExample)._with_alias.alias


# Converters


@attr.s
class ConvCOptional:
    x: int | None = attr.ib(converter=attr.converters.optional(int))


ConvCOptional(1)
ConvCOptional(None)


# XXX: Fails with E: Unsupported converter, only named functions, types and lambdas are currently supported  [misc]
# See https://github.com/python/mypy/issues/15736
#
# @attr.s
# class ConvCPipe:
#     x: str = attr.ib(converter=attr.converters.pipe(int, str))
#
#
# ConvCPipe(3.4)
# ConvCPipe("09")
#
#
# @attr.s
# class ConvCDefaultIfNone:
#     x: int = attr.ib(converter=attr.converters.default_if_none(42))
#
#
# ConvCDefaultIfNone(1)
# ConvCDefaultIfNone(None)


@attr.s
class ConvCToBool:
    x: int = attr.ib(converter=attr.converters.to_bool)


ConvCToBool(1)
ConvCToBool(True)
ConvCToBool("on")
ConvCToBool("yes")
ConvCToBool(0)
ConvCToBool(False)
ConvCToBool("n")


# Validators
@attr.s
class Validated:
    a = attr.ib(
        type=List[C],
        validator=attr.validators.deep_iterable(
            attr.validators.instance_of(C), attr.validators.instance_of(list)
        ),
    )
    aa = attr.ib(
        type=Tuple[C],
        validator=attr.validators.deep_iterable(
            attr.validators.instance_of(C), attr.validators.instance_of(tuple)
        ),
    )
    b = attr.ib(
        type=List[C],
        validator=attr.validators.deep_iterable(
            attr.validators.instance_of(C)
        ),
    )
    c = attr.ib(
        type=Dict[C, D],
        validator=attr.validators.deep_mapping(
            attr.validators.instance_of(C),
            attr.validators.instance_of(D),
            attr.validators.instance_of(dict),
        ),
    )
    d = attr.ib(
        type=Dict[C, D],
        validator=attr.validators.deep_mapping(
            attr.validators.instance_of(C), attr.validators.instance_of(D)
        ),
    )
    e: str = attr.ib(validator=attr.validators.matches_re(re.compile(r"foo")))
    f: str = attr.ib(
        validator=attr.validators.matches_re(r"foo", flags=42, func=re.search)
    )

    # Test different forms of instance_of
    g: int = attr.ib(validator=attr.validators.instance_of(int))
    h: int = attr.ib(validator=attr.validators.instance_of((int,)))
    j: int | str = attr.ib(validator=attr.validators.instance_of((int, str)))
    k: int | str | C = attr.ib(
        validator=attrs.validators.instance_of((int, C, str))
    )
    kk: int | str | C = attr.ib(
        validator=attrs.validators.instance_of(int | C | str)
    )

    l: Any = attr.ib(
        validator=attr.validators.not_(attr.validators.in_("abc"))
    )
    m: Any = attr.ib(
        validator=attr.validators.not_(
            attr.validators.in_("abc"), exc_types=ValueError
        )
    )
    n: Any = attr.ib(
        validator=attr.validators.not_(
            attr.validators.in_("abc"), exc_types=(ValueError,)
        )
    )
    o: Any = attr.ib(
        validator=attr.validators.not_(attr.validators.in_("abc"), msg="spam")
    )
    p: Any = attr.ib(
        validator=attr.validators.not_(attr.validators.in_("abc"), msg=None)
    )
    q: Any = attr.ib(
        validator=attrs.validators.optional(attrs.validators.instance_of(C))
    )
    r: Any = attr.ib(
        validator=attrs.validators.optional([attrs.validators.instance_of(C)])
    )
    s: Any = attr.ib(
        validator=attrs.validators.optional((attrs.validators.instance_of(C),))
    )


@attr.define
class Validated2:
    num: int = attr.field(validator=attr.validators.ge(0))


@attrs.define
class Validated3:
    num: int = attrs.field(validator=attrs.validators.ge(0))


with attr.validators.disabled():
    Validated2(num=-1)

with attrs.validators.disabled():
    Validated3(num=-1)

try:
    attr.validators.set_disabled(True)
    Validated2(num=-1)
finally:
    attr.validators.set_disabled(False)


# Custom repr()
@attr.s
class WithCustomRepr:
    a: int = attr.ib(repr=True)
    b: str = attr.ib(repr=False)
    c: str = attr.ib(repr=lambda value: "c is for cookie")
    d: bool = attr.ib(repr=str)


@attrs.define
class WithCustomRepr2:
    a: int = attrs.field(repr=True)
    b: str = attrs.field(repr=False)
    c: str = attrs.field(repr=lambda value: "c is for cookie")
    d: bool = attrs.field(repr=str)


# Check some of our own types
@attr.s(eq=True, order=False)
class OrderFlags:
    a: int = attr.ib(eq=False, order=False)
    b: int = attr.ib(eq=True, order=True)


# on_setattr hooks
@attr.s(on_setattr=attr.setters.validate)
class ValidatedSetter:
    a: int
    b: str = attr.ib(on_setattr=attr.setters.NO_OP)
    c: bool = attr.ib(on_setattr=attr.setters.frozen)
    d: int = attr.ib(on_setattr=[attr.setters.convert, attr.setters.validate])
    e: bool = attr.ib(
        on_setattr=attr.setters.pipe(
            attr.setters.convert, attr.setters.validate
        )
    )


@attrs.define(on_setattr=attr.setters.validate)
class ValidatedSetter2:
    a: int
    b: str = attrs.field(on_setattr=attrs.setters.NO_OP)
    c: bool = attrs.field(on_setattr=attrs.setters.frozen)
    d: int = attrs.field(
        on_setattr=[attrs.setters.convert, attrs.setters.validate]
    )
    e: bool = attrs.field(
        on_setattr=attrs.setters.pipe(
            attrs.setters.convert, attrs.setters.validate
        )
    )


# field_transformer
def ft_hook(cls: type, attribs: list[attr.Attribute]) -> list[attr.Attribute]:
    return attribs


# field_transformer
def ft_hook2(
    cls: type, attribs: list[attrs.Attribute]
) -> list[attrs.Attribute]:
    return attribs


@attr.s(field_transformer=ft_hook)
class TransformedAttrs:
    x: int


@attrs.define(field_transformer=ft_hook2)
class TransformedAttrs2:
    x: int


# Auto-detect
@attr.s(auto_detect=True)
class AutoDetect:
    x: int

    def __init__(self, x: int):
        self.x = x


# Provisional APIs
@attr.define(order=True)
class NGClass:
    x: int = attr.field(default=42)


ngc = NGClass(1)


@attr.mutable(slots=False)
class NGClass2:
    x: int


ngc2 = NGClass2(1)


@attr.frozen(str=True)
class NGFrozen:
    x: int


ngf = NGFrozen(1)

attr.fields(NGFrozen).x.evolve(eq=False)
a = attr.fields(NGFrozen).x
a.evolve(repr=False)


attrs.fields(NGFrozen).x.evolve(eq=False)
a = attrs.fields(NGFrozen).x
a.evolve(repr=False)


@attr.s(collect_by_mro=True)
class MRO:
    pass


@attr.s
class FactoryTest:
    a: list[int] = attr.ib(default=attr.Factory(list))
    b: list[Any] = attr.ib(default=attr.Factory(list, False))
    c: list[int] = attr.ib(default=attr.Factory((lambda s: s.a), True))


@attrs.define
class FactoryTest2:
    a: list[int] = attrs.field(default=attrs.Factory(list))
    b: list[Any] = attrs.field(default=attrs.Factory(list, False))
    c: list[int] = attrs.field(default=attrs.Factory((lambda s: s.a), True))


attrs.asdict(FactoryTest2())
attr.asdict(FactoryTest(), tuple_keys=True)


# Check match_args stub
@attr.s(match_args=False)
class MatchArgs:
    a: int = attr.ib()
    b: int = attr.ib()


attr.asdict(FactoryTest())
attr.asdict(FactoryTest(), retain_collection_types=False)


# Check match_args stub
@attrs.define(match_args=False)
class MatchArgs2:
    a: int
    b: int


# NG versions of asdict/astuple
attrs.asdict(MatchArgs2(1, 2))
attrs.astuple(MatchArgs2(1, 2))


def accessing_from_attr() -> None:
    """
    Use a function to keep the ns clean.
    """
    attr.converters.optional
    attr.exceptions.FrozenError
    attr.filters.include
    attr.filters.exclude
    attr.setters.frozen
    attr.validators.and_
    attr.cmp_using


def accessing_from_attrs() -> None:
    """
    Use a function to keep the ns clean.
    """
    attrs.converters.optional
    attrs.exceptions.FrozenError
    attrs.filters.include
    attrs.filters.exclude
    attrs.setters.frozen
    attrs.validators.and_
    attrs.cmp_using


foo = object
if attrs.has(foo) or attr.has(foo):
    foo.__attrs_attrs__


@attrs.define(unsafe_hash=True)
class Hashable:
    pass


def test(cls: type) -> None:
    if attr.has(cls):
        attr.resolve_types(cls)


# SPDX-License-Identifier: MIT

import os

from importlib import metadata
from pathlib import Path


# Set canonical URL from the Read the Docs Domain
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")

# Tell Jinja2 templates the build is running on Read the Docs
if os.environ.get("READTHEDOCS", "") == "True":
    html_context = {"READTHEDOCS": True}


# -- Path setup -----------------------------------------------------------

PROJECT_ROOT_DIR = Path(__file__).parents[1].resolve()


# -- General configuration ------------------------------------------------

doctest_global_setup = """
from attr import define, frozen, field, validators, Factory
"""

linkcheck_ignore = [
    # Fastly blocks this.
    "https://pypi.org/project/attr/#history",
    # We run into GitHub's rate limits.
    r"https://github.com/.*/(issues|pull)/\d+",
    # Rate limits and the latest tag is missing anyways on release.
    "https://github.com/python-attrs/attrs/tree/.*",
]

linkcheck_anchors_ignore_for_url = [
    r"^https?://(www\.)?github\.com/.*",
]

# In nitpick mode (-n), still ignore any of the following "broken" references
# to non-types.
nitpick_ignore = [
    ("py:class", "Any value"),
    ("py:class", "callable"),
    ("py:class", "callables"),
    ("py:class", "tuple of types"),
]

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "notfound.extension",
    "sphinxcontrib.towncrier",
]

myst_enable_extensions = [
    "colon_fence",
    "smartquotes",
    "deflist",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix of source filenames.
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "attrs"
author = "Hynek Schlawack"
copyright = f"2015, {author}"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.

# The full version, including alpha/beta/rc tags.
release = metadata.version("attrs")
if "dev" in release:
    release = version = "UNRELEASED"
else:
    # The short X.Y version.
    version = release.rsplit(".", 1)[0]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build"]

# The reST default role (used for this markup: `text`) to use for all
# documents.
default_role = "any"

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = True

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = "furo"
html_theme_options = {
    "sidebar_hide_name": True,
    "light_logo": "attrs_logo.svg",
    "dark_logo": "attrs_logo_white.svg",
    "top_of_page_buttons": [],
    "light_css_variables": {
        "font-stack": "Inter,sans-serif",
        "font-stack--monospace": "BerkeleyMono, MonoLisa, ui-monospace, "
        "SFMono-Regular, Menlo, Consolas, Liberation Mono, monospace",
    },
}
html_css_files = ["custom.css"]


# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
# html_favicon = None

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# If false, no module index is generated.
html_domain_indices = True

# If false, no index is generated.
html_use_index = True

# If true, the index is split into individual pages for each letter.
html_split_index = False

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
# html_use_opensearch = ''

# Output file base name for HTML help builder.
htmlhelp_basename = "attrsdoc"

# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [("index", "attrs", "attrs Documentation", ["Hynek Schlawack"], 1)]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        "index",
        "attrs",
        "attrs Documentation",
        "Hynek Schlawack",
        "attrs",
        "Python Classes Without Boilerplate",
        "Miscellaneous",
    )
]

epub_description = "Python Classes Without Boilerplate"

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

# Allow non-local URIs so we can have images in CHANGELOG etc.
suppress_warnings = ["image.nonlocal_uri"]


# -- Options for sphinxcontrib.towncrier extension ------------------------

towncrier_draft_autoversion_mode = "draft"
towncrier_draft_include_empty = True
towncrier_draft_working_directory = PROJECT_ROOT_DIR
towncrier_draft_config_path = "pyproject.toml"


# SPDX-License-Identifier: MIT

from attr.setters import *  # noqa: F403


# SPDX-License-Identifier: MIT

from attr.validators import *  # noqa: F403


# SPDX-License-Identifier: MIT

from attr import (
    NOTHING,
    Attribute,
    AttrsInstance,
    Converter,
    Factory,
    NothingType,
    _make_getattr,
    assoc,
    cmp_using,
    define,
    evolve,
    field,
    fields,
    fields_dict,
    frozen,
    has,
    make_class,
    mutable,
    resolve_types,
    validate,
)
from attr._next_gen import asdict, astuple

from . import converters, exceptions, filters, setters, validators


__all__ = [
    "NOTHING",
    "Attribute",
    "AttrsInstance",
    "Converter",
    "Factory",
    "NothingType",
    "__author__",
    "__copyright__",
    "__description__",
    "__doc__",
    "__email__",
    "__license__",
    "__title__",
    "__url__",
    "__version__",
    "__version_info__",
    "asdict",
    "assoc",
    "astuple",
    "cmp_using",
    "converters",
    "define",
    "evolve",
    "exceptions",
    "field",
    "fields",
    "fields_dict",
    "filters",
    "frozen",
    "has",
    "make_class",
    "mutable",
    "resolve_types",
    "setters",
    "validate",
    "validators",
]

__getattr__ = _make_getattr(__name__)


# SPDX-License-Identifier: MIT

from attr.exceptions import *  # noqa: F403


# SPDX-License-Identifier: MIT

from attr.converters import *  # noqa: F403


# SPDX-License-Identifier: MIT

from attr.filters import *  # noqa: F403


# SPDX-License-Identifier: MIT

"""
Commonly used hooks for on_setattr.
"""

from . import _config
from .exceptions import FrozenAttributeError


def pipe(*setters):
    """
    Run all *setters* and return the return value of the last one.

    .. versionadded:: 20.1.0
    """

    def wrapped_pipe(instance, attrib, new_value):
        rv = new_value

        for setter in setters:
            rv = setter(instance, attrib, rv)

        return rv

    return wrapped_pipe


def frozen(_, __, ___):
    """
    Prevent an attribute to be modified.

    .. versionadded:: 20.1.0
    """
    raise FrozenAttributeError


def validate(instance, attrib, new_value):
    """
    Run *attrib*'s validator on *new_value* if it has one.

    .. versionadded:: 20.1.0
    """
    if _config._run_validators is False:
        return new_value

    v = attrib.validator
    if not v:
        return new_value

    v(instance, attrib, new_value)

    return new_value


def convert(instance, attrib, new_value):
    """
    Run *attrib*'s converter -- if it has one -- on *new_value* and return the
    result.

    .. versionadded:: 20.1.0
    """
    c = attrib.converter
    if c:
        # This can be removed once we drop 3.8 and use attrs.Converter instead.
        from ._make import Converter

        if not isinstance(c, Converter):
            return c(new_value)

        return c(new_value, instance, attrib)

    return new_value


# Sentinel for disabling class-wide *on_setattr* hooks for certain attributes.
# Sphinx's autodata stopped working, so the docstring is inlined in the API
# docs.
NO_OP = object()


# SPDX-License-Identifier: MIT

"""
Commonly useful validators.
"""

import operator
import re

from contextlib import contextmanager
from re import Pattern

from ._config import get_run_validators, set_run_validators
from ._make import _AndValidator, and_, attrib, attrs
from .converters import default_if_none
from .exceptions import NotCallableError


__all__ = [
    "and_",
    "deep_iterable",
    "deep_mapping",
    "disabled",
    "ge",
    "get_disabled",
    "gt",
    "in_",
    "instance_of",
    "is_callable",
    "le",
    "lt",
    "matches_re",
    "max_len",
    "min_len",
    "not_",
    "optional",
    "or_",
    "set_disabled",
]


def set_disabled(disabled):
    """
    Globally disable or enable running validators.

    By default, they are run.

    Args:
        disabled (bool): If `True`, disable running all validators.

    .. warning::

        This function is not thread-safe!

    .. versionadded:: 21.3.0
    """
    set_run_validators(not disabled)


def get_disabled():
    """
    Return a bool indicating whether validators are currently disabled or not.

    Returns:
        bool:`True` if validators are currently disabled.

    .. versionadded:: 21.3.0
    """
    return not get_run_validators()


@contextmanager
def disabled():
    """
    Context manager that disables running validators within its context.

    .. warning::

        This context manager is not thread-safe!

    .. versionadded:: 21.3.0
    """
    set_run_validators(False)
    try:
        yield
    finally:
        set_run_validators(True)


@attrs(repr=False, slots=True, unsafe_hash=True)
class _InstanceOfValidator:
    type = attrib()

    def __call__(self, inst, attr, value):
        """
        We use a callable class to be able to change the ``__repr__``.
        """
        if not isinstance(value, self.type):
            msg = f"'{attr.name}' must be {self.type!r} (got {value!r} that is a {value.__class__!r})."
            raise TypeError(
                msg,
                attr,
                self.type,
                value,
            )

    def __repr__(self):
        return f"<instance_of validator for type {self.type!r}>"


def instance_of(type):
    """
    A validator that raises a `TypeError` if the initializer is called with a
    wrong type for this particular attribute (checks are performed using
    `isinstance` therefore it's also valid to pass a tuple of types).

    Args:
        type (type | tuple[type]): The type to check for.

    Raises:
        TypeError:
            With a human readable error message, the attribute (of type
            `attrs.Attribute`), the expected type, and the value it got.
    """
    return _InstanceOfValidator(type)


@attrs(repr=False, frozen=True, slots=True)
class _MatchesReValidator:
    pattern = attrib()
    match_func = attrib()

    def __call__(self, inst, attr, value):
        """
        We use a callable class to be able to change the ``__repr__``.
        """
        if not self.match_func(value):
            msg = f"'{attr.name}' must match regex {self.pattern.pattern!r} ({value!r} doesn't)"
            raise ValueError(
                msg,
                attr,
                self.pattern,
                value,
            )

    def __repr__(self):
        return f"<matches_re validator for pattern {self.pattern!r}>"


def matches_re(regex, flags=0, func=None):
    r"""
    A validator that raises `ValueError` if the initializer is called with a
    string that doesn't match *regex*.

    Args:
        regex (str, re.Pattern):
            A regex string or precompiled pattern to match against

        flags (int):
            Flags that will be passed to the underlying re function (default 0)

        func (typing.Callable):
            Which underlying `re` function to call. Valid options are
            `re.fullmatch`, `re.search`, and `re.match`; the default `None`
            means `re.fullmatch`. For performance reasons, the pattern is
            always precompiled using `re.compile`.

    .. versionadded:: 19.2.0
    .. versionchanged:: 21.3.0 *regex* can be a pre-compiled pattern.
    """
    valid_funcs = (re.fullmatch, None, re.search, re.match)
    if func not in valid_funcs:
        msg = "'func' must be one of {}.".format(
            ", ".join(
                sorted((e and e.__name__) or "None" for e in set(valid_funcs))
            )
        )
        raise ValueError(msg)

    if isinstance(regex, Pattern):
        if flags:
            msg = "'flags' can only be used with a string pattern; pass flags to re.compile() instead"
            raise TypeError(msg)
        pattern = regex
    else:
        pattern = re.compile(regex, flags)

    if func is re.match:
        match_func = pattern.match
    elif func is re.search:
        match_func = pattern.search
    else:
        match_func = pattern.fullmatch

    return _MatchesReValidator(pattern, match_func)


@attrs(repr=False, slots=True, unsafe_hash=True)
class _OptionalValidator:
    validator = attrib()

    def __call__(self, inst, attr, value):
        if value is None:
            return

        self.validator(inst, attr, value)

    def __repr__(self):
        return f"<optional validator for {self.validator!r} or None>"


def optional(validator):
    """
    A validator that makes an attribute optional.  An optional attribute is one
    which can be set to `None` in addition to satisfying the requirements of
    the sub-validator.

    Args:
        validator
            (typing.Callable | tuple[typing.Callable] | list[typing.Callable]):
            A validator (or validators) that is used for non-`None` values.

    .. versionadded:: 15.1.0
    .. versionchanged:: 17.1.0 *validator* can be a list of validators.
    .. versionchanged:: 23.1.0 *validator* can also be a tuple of validators.
    """
    if isinstance(validator, (list, tuple)):
        return _OptionalValidator(_AndValidator(validator))

    return _OptionalValidator(validator)


@attrs(repr=False, slots=True, unsafe_hash=True)
class _InValidator:
    options = attrib()
    _original_options = attrib(hash=False)

    def __call__(self, inst, attr, value):
        try:
            in_options = value in self.options
        except TypeError:  # e.g. `1 in "abc"`
            in_options = False

        if not in_options:
            msg = f"'{attr.name}' must be in {self._original_options!r} (got {value!r})"
            raise ValueError(
                msg,
                attr,
                self._original_options,
                value,
            )

    def __repr__(self):
        return f"<in_ validator with options {self._original_options!r}>"


def in_(options):
    """
    A validator that raises a `ValueError` if the initializer is called with a
    value that does not belong in the *options* provided.

    The check is performed using ``value in options``, so *options* has to
    support that operation.

    To keep the validator hashable, dicts, lists, and sets are transparently
    transformed into a `tuple`.

    Args:
        options: Allowed options.

    Raises:
        ValueError:
            With a human readable error message, the attribute (of type
            `attrs.Attribute`), the expected options, and the value it got.

    .. versionadded:: 17.1.0
    .. versionchanged:: 22.1.0
       The ValueError was incomplete until now and only contained the human
       readable error message. Now it contains all the information that has
       been promised since 17.1.0.
    .. versionchanged:: 24.1.0
       *options* that are a list, dict, or a set are now transformed into a
       tuple to keep the validator hashable.
    """
    repr_options = options
    if isinstance(options, (list, dict, set)):
        options = tuple(options)

    return _InValidator(options, repr_options)


@attrs(repr=False, slots=False, unsafe_hash=True)
class _IsCallableValidator:
    def __call__(self, inst, attr, value):
        """
        We use a callable class to be able to change the ``__repr__``.
        """
        if not callable(value):
            message = (
                "'{name}' must be callable "
                "(got {value!r} that is a {actual!r})."
            )
            raise NotCallableError(
                msg=message.format(
                    name=attr.name, value=value, actual=value.__class__
                ),
                value=value,
            )

    def __repr__(self):
        return "<is_callable validator>"


def is_callable():
    """
    A validator that raises a `attrs.exceptions.NotCallableError` if the
    initializer is called with a value for this particular attribute that is
    not callable.

    .. versionadded:: 19.1.0

    Raises:
        attrs.exceptions.NotCallableError:
            With a human readable error message containing the attribute
            (`attrs.Attribute`) name, and the value it got.
    """
    return _IsCallableValidator()


@attrs(repr=False, slots=True, unsafe_hash=True)
class _DeepIterable:
    member_validator = attrib(validator=is_callable())
    iterable_validator = attrib(
        default=None, validator=optional(is_callable())
    )

    def __call__(self, inst, attr, value):
        """
        We use a callable class to be able to change the ``__repr__``.
        """
        if self.iterable_validator is not None:
            self.iterable_validator(inst, attr, value)

        for member in value:
            self.member_validator(inst, attr, member)

    def __repr__(self):
        iterable_identifier = (
            ""
            if self.iterable_validator is None
            else f" {self.iterable_validator!r}"
        )
        return (
            f"<deep_iterable validator for{iterable_identifier}"
            f" iterables of {self.member_validator!r}>"
        )


def deep_iterable(member_validator, iterable_validator=None):
    """
    A validator that performs deep validation of an iterable.

    Args:
        member_validator: Validator to apply to iterable members.

        iterable_validator:
            Validator to apply to iterable itself (optional).

    Raises
        TypeError: if any sub-validators fail

    .. versionadded:: 19.1.0
    """
    if isinstance(member_validator, (list, tuple)):
        member_validator = and_(*member_validator)
    return _DeepIterable(member_validator, iterable_validator)


@attrs(repr=False, slots=True, unsafe_hash=True)
class _DeepMapping:
    key_validator = attrib(validator=is_callable())
    value_validator = attrib(validator=is_callable())
    mapping_validator = attrib(default=None, validator=optional(is_callable()))

    def __call__(self, inst, attr, value):
        """
        We use a callable class to be able to change the ``__repr__``.
        """
        if self.mapping_validator is not None:
            self.mapping_validator(inst, attr, value)

        for key in value:
            self.key_validator(inst, attr, key)
            self.value_validator(inst, attr, value[key])

    def __repr__(self):
        return f"<deep_mapping validator for objects mapping {self.key_validator!r} to {self.value_validator!r}>"


def deep_mapping(key_validator, value_validator, mapping_validator=None):
    """
    A validator that performs deep validation of a dictionary.

    Args:
        key_validator: Validator to apply to dictionary keys.

        value_validator: Validator to apply to dictionary values.

        mapping_validator:
            Validator to apply to top-level mapping attribute (optional).

    .. versionadded:: 19.1.0

    Raises:
        TypeError: if any sub-validators fail
    """
    return _DeepMapping(key_validator, value_validator, mapping_validator)


@attrs(repr=False, frozen=True, slots=True)
class _NumberValidator:
    bound = attrib()
    compare_op = attrib()
    compare_func = attrib()

    def __call__(self, inst, attr, value):
        """
        We use a callable class to be able to change the ``__repr__``.
        """
        if not self.compare_func(value, self.bound):
            msg = f"'{attr.name}' must be {self.compare_op} {self.bound}: {value}"
            raise ValueError(msg)

    def __repr__(self):
        return f"<Validator for x {self.compare_op} {self.bound}>"


def lt(val):
    """
    A validator that raises `ValueError` if the initializer is called with a
    number larger or equal to *val*.

    The validator uses `operator.lt` to compare the values.

    Args:
        val: Exclusive upper bound for values.

    .. versionadded:: 21.3.0
    """
    return _NumberValidator(val, "<", operator.lt)


def le(val):
    """
    A validator that raises `ValueError` if the initializer is called with a
    number greater than *val*.

    The validator uses `operator.le` to compare the values.

    Args:
        val: Inclusive upper bound for values.

    .. versionadded:: 21.3.0
    """
    return _NumberValidator(val, "<=", operator.le)


def ge(val):
    """
    A validator that raises `ValueError` if the initializer is called with a
    number smaller than *val*.

    The validator uses `operator.ge` to compare the values.

    Args:
        val: Inclusive lower bound for values

    .. versionadded:: 21.3.0
    """
    return _NumberValidator(val, ">=", operator.ge)


def gt(val):
    """
    A validator that raises `ValueError` if the initializer is called with a
    number smaller or equal to *val*.

    The validator uses `operator.gt` to compare the values.

    Args:
       val: Exclusive lower bound for values

    .. versionadded:: 21.3.0
    """
    return _NumberValidator(val, ">", operator.gt)


@attrs(repr=False, frozen=True, slots=True)
class _MaxLengthValidator:
    max_length = attrib()

    def __call__(self, inst, attr, value):
        """
        We use a callable class to be able to change the ``__repr__``.
        """
        if len(value) > self.max_length:
            msg = f"Length of '{attr.name}' must be <= {self.max_length}: {len(value)}"
            raise ValueError(msg)

    def __repr__(self):
        return f"<max_len validator for {self.max_length}>"


def max_len(length):
    """
    A validator that raises `ValueError` if the initializer is called
    with a string or iterable that is longer than *length*.

    Args:
        length (int): Maximum length of the string or iterable

    .. versionadded:: 21.3.0
    """
    return _MaxLengthValidator(length)


@attrs(repr=False, frozen=True, slots=True)
class _MinLengthValidator:
    min_length = attrib()

    def __call__(self, inst, attr, value):
        """
        We use a callable class to be able to change the ``__repr__``.
        """
        if len(value) < self.min_length:
            msg = f"Length of '{attr.name}' must be >= {self.min_length}: {len(value)}"
            raise ValueError(msg)

    def __repr__(self):
        return f"<min_len validator for {self.min_length}>"


def min_len(length):
    """
    A validator that raises `ValueError` if the initializer is called
    with a string or iterable that is shorter than *length*.

    Args:
        length (int): Minimum length of the string or iterable

    .. versionadded:: 22.1.0
    """
    return _MinLengthValidator(length)


@attrs(repr=False, slots=True, unsafe_hash=True)
class _SubclassOfValidator:
    type = attrib()

    def __call__(self, inst, attr, value):
        """
        We use a callable class to be able to change the ``__repr__``.
        """
        if not issubclass(value, self.type):
            msg = f"'{attr.name}' must be a subclass of {self.type!r} (got {value!r})."
            raise TypeError(
                msg,
                attr,
                self.type,
                value,
            )

    def __repr__(self):
        return f"<subclass_of validator for type {self.type!r}>"


def _subclass_of(type):
    """
    A validator that raises a `TypeError` if the initializer is called with a
    wrong type for this particular attribute (checks are performed using
    `issubclass` therefore it's also valid to pass a tuple of types).

    Args:
        type (type | tuple[type, ...]): The type(s) to check for.

    Raises:
        TypeError:
            With a human readable error message, the attribute (of type
            `attrs.Attribute`), the expected type, and the value it got.
    """
    return _SubclassOfValidator(type)


@attrs(repr=False, slots=True, unsafe_hash=True)
class _NotValidator:
    validator = attrib()
    msg = attrib(
        converter=default_if_none(
            "not_ validator child '{validator!r}' "
            "did not raise a captured error"
        )
    )
    exc_types = attrib(
        validator=deep_iterable(
            member_validator=_subclass_of(Exception),
            iterable_validator=instance_of(tuple),
        ),
    )

    def __call__(self, inst, attr, value):
        try:
            self.validator(inst, attr, value)
        except self.exc_types:
            pass  # suppress error to invert validity
        else:
            raise ValueError(
                self.msg.format(
                    validator=self.validator,
                    exc_types=self.exc_types,
                ),
                attr,
                self.validator,
                value,
                self.exc_types,
            )

    def __repr__(self):
        return f"<not_ validator wrapping {self.validator!r}, capturing {self.exc_types!r}>"


def not_(validator, *, msg=None, exc_types=(ValueError, TypeError)):
    """
    A validator that wraps and logically 'inverts' the validator passed to it.
    It will raise a `ValueError` if the provided validator *doesn't* raise a
    `ValueError` or `TypeError` (by default), and will suppress the exception
    if the provided validator *does*.

    Intended to be used with existing validators to compose logic without
    needing to create inverted variants, for example, ``not_(in_(...))``.

    Args:
        validator: A validator to be logically inverted.

        msg (str):
            Message to raise if validator fails. Formatted with keys
            ``exc_types`` and ``validator``.

        exc_types (tuple[type, ...]):
            Exception type(s) to capture. Other types raised by child
            validators will not be intercepted and pass through.

    Raises:
        ValueError:
            With a human readable error message, the attribute (of type
            `attrs.Attribute`), the validator that failed to raise an
            exception, the value it got, and the expected exception types.

    .. versionadded:: 22.2.0
    """
    try:
        exc_types = tuple(exc_types)
    except TypeError:
        exc_types = (exc_types,)
    return _NotValidator(validator, msg, exc_types)


@attrs(repr=False, slots=True, unsafe_hash=True)
class _OrValidator:
    validators = attrib()

    def __call__(self, inst, attr, value):
        for v in self.validators:
            try:
                v(inst, attr, value)
            except Exception:  # noqa: BLE001, PERF203, S112
                continue
            else:
                return

        msg = f"None of {self.validators!r} satisfied for value {value!r}"
        raise ValueError(msg)

    def __repr__(self):
        return f"<or validator wrapping {self.validators!r}>"


def or_(*validators):
    """
    A validator that composes multiple validators into one.

    When called on a value, it runs all wrapped validators until one of them is
    satisfied.

    Args:
        validators (~collections.abc.Iterable[typing.Callable]):
            Arbitrary number of validators.

    Raises:
        ValueError:
            If no validator is satisfied. Raised with a human-readable error
            message listing all the wrapped validators and the value that
            failed all of them.

    .. versionadded:: 24.1.0
    """
    vals = []
    for v in validators:
        vals.extend(v.validators if isinstance(v, _OrValidator) else [v])

    return _OrValidator(tuple(vals))


# SPDX-License-Identifier: MIT

from __future__ import annotations

import abc
import contextlib
import copy
import enum
import inspect
import itertools
import linecache
import sys
import types
import unicodedata

from collections.abc import Callable, Mapping
from functools import cached_property
from typing import Any, NamedTuple, TypeVar

# We need to import _compat itself in addition to the _compat members to avoid
# having the thread-local in the globals here.
from . import _compat, _config, setters
from ._compat import (
    PY_3_10_PLUS,
    PY_3_11_PLUS,
    PY_3_13_PLUS,
    _AnnotationExtractor,
    _get_annotations,
    get_generic_base,
)
from .exceptions import (
    DefaultAlreadySetError,
    FrozenInstanceError,
    NotAnAttrsClassError,
    UnannotatedAttributeError,
)


# This is used at least twice, so cache it here.
_OBJ_SETATTR = object.__setattr__
_INIT_FACTORY_PAT = "__attr_factory_%s"
_CLASSVAR_PREFIXES = (
    "typing.ClassVar",
    "t.ClassVar",
    "ClassVar",
    "typing_extensions.ClassVar",
)
# we don't use a double-underscore prefix because that triggers
# name mangling when trying to create a slot for the field
# (when slots=True)
_HASH_CACHE_FIELD = "_attrs_cached_hash"

_EMPTY_METADATA_SINGLETON = types.MappingProxyType({})

# Unique object for unequivocal getattr() defaults.
_SENTINEL = object()

_DEFAULT_ON_SETATTR = setters.pipe(setters.convert, setters.validate)


class _Nothing(enum.Enum):
    """
    Sentinel to indicate the lack of a value when `None` is ambiguous.

    If extending attrs, you can use ``typing.Literal[NOTHING]`` to show
    that a value may be ``NOTHING``.

    .. versionchanged:: 21.1.0 ``bool(NOTHING)`` is now False.
    .. versionchanged:: 22.2.0 ``NOTHING`` is now an ``enum.Enum`` variant.
    """

    NOTHING = enum.auto()

    def __repr__(self):
        return "NOTHING"

    def __bool__(self):
        return False


NOTHING = _Nothing.NOTHING
"""
Sentinel to indicate the lack of a value when `None` is ambiguous.

When using in 3rd party code, use `attrs.NothingType` for type annotations.
"""


class _CacheHashWrapper(int):
    """
    An integer subclass that pickles / copies as None

    This is used for non-slots classes with ``cache_hash=True``, to avoid
    serializing a potentially (even likely) invalid hash value. Since `None`
    is the default value for uncalculated hashes, whenever this is copied,
    the copy's value for the hash should automatically reset.

    See GH #613 for more details.
    """

    def __reduce__(self, _none_constructor=type(None), _args=()):  # noqa: B008
        return _none_constructor, _args


def attrib(
    default=NOTHING,
    validator=None,
    repr=True,
    cmp=None,
    hash=None,
    init=True,
    metadata=None,
    type=None,
    converter=None,
    factory=None,
    kw_only=False,
    eq=None,
    order=None,
    on_setattr=None,
    alias=None,
):
    """
    Create a new field / attribute on a class.

    Identical to `attrs.field`, except it's not keyword-only.

    Consider using `attrs.field` in new code (``attr.ib`` will *never* go away,
    though).

    ..  warning::

        Does **nothing** unless the class is also decorated with
        `attr.s` (or similar)!


    .. versionadded:: 15.2.0 *convert*
    .. versionadded:: 16.3.0 *metadata*
    .. versionchanged:: 17.1.0 *validator* can be a ``list`` now.
    .. versionchanged:: 17.1.0
       *hash* is `None` and therefore mirrors *eq* by default.
    .. versionadded:: 17.3.0 *type*
    .. deprecated:: 17.4.0 *convert*
    .. versionadded:: 17.4.0
       *converter* as a replacement for the deprecated *convert* to achieve
       consistency with other noun-based arguments.
    .. versionadded:: 18.1.0
       ``factory=f`` is syntactic sugar for ``default=attr.Factory(f)``.
    .. versionadded:: 18.2.0 *kw_only*
    .. versionchanged:: 19.2.0 *convert* keyword argument removed.
    .. versionchanged:: 19.2.0 *repr* also accepts a custom callable.
    .. deprecated:: 19.2.0 *cmp* Removal on or after 2021-06-01.
    .. versionadded:: 19.2.0 *eq* and *order*
    .. versionadded:: 20.1.0 *on_setattr*
    .. versionchanged:: 20.3.0 *kw_only* backported to Python 2
    .. versionchanged:: 21.1.0
       *eq*, *order*, and *cmp* also accept a custom callable
    .. versionchanged:: 21.1.0 *cmp* undeprecated
    .. versionadded:: 22.2.0 *alias*
    """
    eq, eq_key, order, order_key = _determine_attrib_eq_order(
        cmp, eq, order, True
    )

    if hash is not None and hash is not True and hash is not False:
        msg = "Invalid value for hash.  Must be True, False, or None."
        raise TypeError(msg)

    if factory is not None:
        if default is not NOTHING:
            msg = (
                "The `default` and `factory` arguments are mutually exclusive."
            )
            raise ValueError(msg)
        if not callable(factory):
            msg = "The `factory` argument must be a callable."
            raise ValueError(msg)
        default = Factory(factory)

    if metadata is None:
        metadata = {}

    # Apply syntactic sugar by auto-wrapping.
    if isinstance(on_setattr, (list, tuple)):
        on_setattr = setters.pipe(*on_setattr)

    if validator and isinstance(validator, (list, tuple)):
        validator = and_(*validator)

    if converter and isinstance(converter, (list, tuple)):
        converter = pipe(*converter)

    return _CountingAttr(
        default=default,
        validator=validator,
        repr=repr,
        cmp=None,
        hash=hash,
        init=init,
        converter=converter,
        metadata=metadata,
        type=type,
        kw_only=kw_only,
        eq=eq,
        eq_key=eq_key,
        order=order,
        order_key=order_key,
        on_setattr=on_setattr,
        alias=alias,
    )


def _compile_and_eval(
    script: str,
    globs: dict[str, Any] | None,
    locs: Mapping[str, object] | None = None,
    filename: str = "",
) -> None:
    """
    Evaluate the script with the given global (globs) and local (locs)
    variables.
    """
    bytecode = compile(script, filename, "exec")
    eval(bytecode, globs, locs)


def _linecache_and_compile(
    script: str,
    filename: str,
    globs: dict[str, Any] | None,
    locals: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """
    Cache the script with _linecache_, compile it and return the _locals_.
    """

    locs = {} if locals is None else locals

    # In order of debuggers like PDB being able to step through the code,
    # we add a fake linecache entry.
    count = 1
    base_filename = filename
    while True:
        linecache_tuple = (
            len(script),
            None,
            script.splitlines(True),
            filename,
        )
        old_val = linecache.cache.setdefault(filename, linecache_tuple)
        if old_val == linecache_tuple:
            break

        filename = f"{base_filename[:-1]}-{count}>"
        count += 1

    _compile_and_eval(script, globs, locs, filename)

    return locs


def _make_attr_tuple_class(cls_name: str, attr_names: list[str]) -> type:
    """
    Create a tuple subclass to hold `Attribute`s for an `attrs` class.

    The subclass is a bare tuple with properties for names.

    class MyClassAttributes(tuple):
        __slots__ = ()
        x = property(itemgetter(0))
    """
    attr_class_name = f"{cls_name}Attributes"
    body = {}
    for i, attr_name in enumerate(attr_names):

        def getter(self, i=i):
            return self[i]

        body[attr_name] = property(getter)
    return type(attr_class_name, (tuple,), body)


# Tuple class for extracted attributes from a class definition.
# `base_attrs` is a subset of `attrs`.
class _Attributes(NamedTuple):
    attrs: type
    base_attrs: list[Attribute]
    base_attrs_map: dict[str, type]


def _is_class_var(annot):
    """
    Check whether *annot* is a typing.ClassVar.

    The string comparison hack is used to avoid evaluating all string
    annotations which would put attrs-based classes at a performance
    disadvantage compared to plain old classes.
    """
    annot = str(annot)

    # Annotation can be quoted.
    if annot.startswith(("'", '"')) and annot.endswith(("'", '"')):
        annot = annot[1:-1]

    return annot.startswith(_CLASSVAR_PREFIXES)


def _has_own_attribute(cls, attrib_name):
    """
    Check whether *cls* defines *attrib_name* (and doesn't just inherit it).
    """
    return attrib_name in cls.__dict__


def _collect_base_attrs(
    cls, taken_attr_names
) -> tuple[list[Attribute], dict[str, type]]:
    """
    Collect attr.ibs from base classes of *cls*, except *taken_attr_names*.
    """
    base_attrs = []
    base_attr_map = {}  # A dictionary of base attrs to their classes.

    # Traverse the MRO and collect attributes.
    for base_cls in reversed(cls.__mro__[1:-1]):
        for a in getattr(base_cls, "__attrs_attrs__", []):
            if a.inherited or a.name in taken_attr_names:
                continue

            a = a.evolve(inherited=True)  # noqa: PLW2901
            base_attrs.append(a)
            base_attr_map[a.name] = base_cls

    # For each name, only keep the freshest definition i.e. the furthest at the
    # back.  base_attr_map is fine because it gets overwritten with every new
    # instance.
    filtered = []
    seen = set()
    for a in reversed(base_attrs):
        if a.name in seen:
            continue
        filtered.insert(0, a)
        seen.add(a.name)

    return filtered, base_attr_map


def _collect_base_attrs_broken(cls, taken_attr_names):
    """
    Collect attr.ibs from base classes of *cls*, except *taken_attr_names*.

    N.B. *taken_attr_names* will be mutated.

    Adhere to the old incorrect behavior.

    Notably it collects from the front and considers inherited attributes which
    leads to the buggy behavior reported in #428.
    """
    base_attrs = []
    base_attr_map = {}  # A dictionary of base attrs to their classes.

    # Traverse the MRO and collect attributes.
    for base_cls in cls.__mro__[1:-1]:
        for a in getattr(base_cls, "__attrs_attrs__", []):
            if a.name in taken_attr_names:
                continue

            a = a.evolve(inherited=True)  # noqa: PLW2901
            taken_attr_names.add(a.name)
            base_attrs.append(a)
            base_attr_map[a.name] = base_cls

    return base_attrs, base_attr_map


def _transform_attrs(
    cls, these, auto_attribs, kw_only, collect_by_mro, field_transformer
) -> _Attributes:
    """
    Transform all `_CountingAttr`s on a class into `Attribute`s.

    If *these* is passed, use that and don't look for them on the class.

    If *collect_by_mro* is True, collect them in the correct MRO order,
    otherwise use the old -- incorrect -- order.  See #428.

    Return an `_Attributes`.
    """
    cd = cls.__dict__
    anns = _get_annotations(cls)

    if these is not None:
        ca_list = list(these.items())
    elif auto_attribs is True:
        ca_names = {
            name
            for name, attr in cd.items()
            if attr.__class__ is _CountingAttr
        }
        ca_list = []
        annot_names = set()
        for attr_name, type in anns.items():
            if _is_class_var(type):
                continue
            annot_names.add(attr_name)
            a = cd.get(attr_name, NOTHING)

            if a.__class__ is not _CountingAttr:
                a = attrib(a)
            ca_list.append((attr_name, a))

        unannotated = ca_names - annot_names
        if unannotated:
            raise UnannotatedAttributeError(
                "The following `attr.ib`s lack a type annotation: "
                + ", ".join(
                    sorted(unannotated, key=lambda n: cd.get(n).counter)
                )
                + "."
            )
    else:
        ca_list = sorted(
            (
                (name, attr)
                for name, attr in cd.items()
                if attr.__class__ is _CountingAttr
            ),
            key=lambda e: e[1].counter,
        )

    fca = Attribute.from_counting_attr
    own_attrs = [
        fca(attr_name, ca, anns.get(attr_name)) for attr_name, ca in ca_list
    ]

    if collect_by_mro:
        base_attrs, base_attr_map = _collect_base_attrs(
            cls, {a.name for a in own_attrs}
        )
    else:
        base_attrs, base_attr_map = _collect_base_attrs_broken(
            cls, {a.name for a in own_attrs}
        )

    if kw_only:
        own_attrs = [a.evolve(kw_only=True) for a in own_attrs]
        base_attrs = [a.evolve(kw_only=True) for a in base_attrs]

    attrs = base_attrs + own_attrs

    if field_transformer is not None:
        attrs = tuple(field_transformer(cls, attrs))

    # Check attr order after executing the field_transformer.
    # Mandatory vs non-mandatory attr order only matters when they are part of
    # the __init__ signature and when they aren't kw_only (which are moved to
    # the end and can be mandatory or non-mandatory in any order, as they will
    # be specified as keyword args anyway). Check the order of those attrs:
    had_default = False
    for a in (a for a in attrs if a.init is not False and a.kw_only is False):
        if had_default is True and a.default is NOTHING:
            msg = f"No mandatory attributes allowed after an attribute with a default value or factory.  Attribute in question: {a!r}"
            raise ValueError(msg)

        if had_default is False and a.default is not NOTHING:
            had_default = True

    # Resolve default field alias after executing field_transformer.
    # This allows field_transformer to differentiate between explicit vs
    # default aliases and supply their own defaults.
    for a in attrs:
        if not a.alias:
            # Evolve is very slow, so we hold our nose and do it dirty.
            _OBJ_SETATTR.__get__(a)("alias", _default_init_alias_for(a.name))

    # Create AttrsClass *after* applying the field_transformer since it may
    # add or remove attributes!
    attr_names = [a.name for a in attrs]
    AttrsClass = _make_attr_tuple_class(cls.__name__, attr_names)

    return _Attributes(AttrsClass(attrs), base_attrs, base_attr_map)


def _make_cached_property_getattr(cached_properties, original_getattr, cls):
    lines = [
        # Wrapped to get `__class__` into closure cell for super()
        # (It will be replaced with the newly constructed class after construction).
        "def wrapper(_cls):",
        "    __class__ = _cls",
        "    def __getattr__(self, item, cached_properties=cached_properties, original_getattr=original_getattr, _cached_setattr_get=_cached_setattr_get):",
        "         func = cached_properties.get(item)",
        "         if func is not None:",
        "              result = func(self)",
        "              _setter = _cached_setattr_get(self)",
        "              _setter(item, result)",
        "              return result",
    ]
    if original_getattr is not None:
        lines.append(
            "         return original_getattr(self, item)",
        )
    else:
        lines.extend(
            [
                "         try:",
                "             return super().__getattribute__(item)",
                "         except AttributeError:",
                "             if not hasattr(super(), '__getattr__'):",
                "                 raise",
                "             return super().__getattr__(item)",
                "         original_error = f\"'{self.__class__.__name__}' object has no attribute '{item}'\"",
                "         raise AttributeError(original_error)",
            ]
        )

    lines.extend(
        [
            "    return __getattr__",
            "__getattr__ = wrapper(_cls)",
        ]
    )

    unique_filename = _generate_unique_filename(cls, "getattr")

    glob = {
        "cached_properties": cached_properties,
        "_cached_setattr_get": _OBJ_SETATTR.__get__,
        "original_getattr": original_getattr,
    }

    return _linecache_and_compile(
        "\n".join(lines), unique_filename, glob, locals={"_cls": cls}
    )["__getattr__"]


def _frozen_setattrs(self, name, value):
    """
    Attached to frozen classes as __setattr__.
    """
    if isinstance(self, BaseException) and name in (
        "__cause__",
        "__context__",
        "__traceback__",
        "__suppress_context__",
        "__notes__",
    ):
        BaseException.__setattr__(self, name, value)
        return

    raise FrozenInstanceError


def _frozen_delattrs(self, name):
    """
    Attached to frozen classes as __delattr__.
    """
    if isinstance(self, BaseException) and name in ("__notes__",):
        BaseException.__delattr__(self, name)
        return

    raise FrozenInstanceError


def evolve(*args, **changes):
    """
    Create a new instance, based on the first positional argument with
    *changes* applied.

    .. tip::

       On Python 3.13 and later, you can also use `copy.replace` instead.

    Args:

        inst:
            Instance of a class with *attrs* attributes. *inst* must be passed
            as a positional argument.

        changes:
            Keyword changes in the new copy.

    Returns:
        A copy of inst with *changes* incorporated.

    Raises:
        TypeError:
            If *attr_name* couldn't be found in the class ``__init__``.

        attrs.exceptions.NotAnAttrsClassError:
            If *cls* is not an *attrs* class.

    .. versionadded:: 17.1.0
    .. deprecated:: 23.1.0
       It is now deprecated to pass the instance using the keyword argument
       *inst*. It will raise a warning until at least April 2024, after which
       it will become an error. Always pass the instance as a positional
       argument.
    .. versionchanged:: 24.1.0
       *inst* can't be passed as a keyword argument anymore.
    """
    try:
        (inst,) = args
    except ValueError:
        msg = (
            f"evolve() takes 1 positional argument, but {len(args)} were given"
        )
        raise TypeError(msg) from None

    cls = inst.__class__
    attrs = fields(cls)
    for a in attrs:
        if not a.init:
            continue
        attr_name = a.name  # To deal with private attributes.
        init_name = a.alias
        if init_name not in changes:
            changes[init_name] = getattr(inst, attr_name)

    return cls(**changes)


class _ClassBuilder:
    """
    Iteratively build *one* class.
    """

    __slots__ = (
        "_add_method_dunders",
        "_attr_names",
        "_attrs",
        "_base_attr_map",
        "_base_names",
        "_cache_hash",
        "_cls",
        "_cls_dict",
        "_delete_attribs",
        "_frozen",
        "_has_custom_setattr",
        "_has_post_init",
        "_has_pre_init",
        "_is_exc",
        "_on_setattr",
        "_pre_init_has_args",
        "_repr_added",
        "_script_snippets",
        "_slots",
        "_weakref_slot",
        "_wrote_own_setattr",
    )

    def __init__(
        self,
        cls: type,
        these,
        slots,
        frozen,
        weakref_slot,
        getstate_setstate,
        auto_attribs,
        kw_only,
        cache_hash,
        is_exc,
        collect_by_mro,
        on_setattr,
        has_custom_setattr,
        field_transformer,
    ):
        attrs, base_attrs, base_map = _transform_attrs(
            cls,
            these,
            auto_attribs,
            kw_only,
            collect_by_mro,
            field_transformer,
        )

        self._cls = cls
        self._cls_dict = dict(cls.__dict__) if slots else {}
        self._attrs = attrs
        self._base_names = {a.name for a in base_attrs}
        self._base_attr_map = base_map
        self._attr_names = tuple(a.name for a in attrs)
        self._slots = slots
        self._frozen = frozen
        self._weakref_slot = weakref_slot
        self._cache_hash = cache_hash
        self._has_pre_init = bool(getattr(cls, "__attrs_pre_init__", False))
        self._pre_init_has_args = False
        if self._has_pre_init:
            # Check if the pre init method has more arguments than just `self`
            # We want to pass arguments if pre init expects arguments
            pre_init_func = cls.__attrs_pre_init__
            pre_init_signature = inspect.signature(pre_init_func)
            self._pre_init_has_args = len(pre_init_signature.parameters) > 1
        self._has_post_init = bool(getattr(cls, "__attrs_post_init__", False))
        self._delete_attribs = not bool(these)
        self._is_exc = is_exc
        self._on_setattr = on_setattr

        self._has_custom_setattr = has_custom_setattr
        self._wrote_own_setattr = False

        self._cls_dict["__attrs_attrs__"] = self._attrs

        if frozen:
            self._cls_dict["__setattr__"] = _frozen_setattrs
            self._cls_dict["__delattr__"] = _frozen_delattrs

            self._wrote_own_setattr = True
        elif on_setattr in (
            _DEFAULT_ON_SETATTR,
            setters.validate,
            setters.convert,
        ):
            has_validator = has_converter = False
            for a in attrs:
                if a.validator is not None:
                    has_validator = True
                if a.converter is not None:
                    has_converter = True

                if has_validator and has_converter:
                    break
            if (
                (
                    on_setattr == _DEFAULT_ON_SETATTR
                    and not (has_validator or has_converter)
                )
                or (on_setattr == setters.validate and not has_validator)
                or (on_setattr == setters.convert and not has_converter)
            ):
                # If class-level on_setattr is set to convert + validate, but
                # there's no field to convert or validate, pretend like there's
                # no on_setattr.
                self._on_setattr = None

        if getstate_setstate:
            (
                self._cls_dict["__getstate__"],
                self._cls_dict["__setstate__"],
            ) = self._make_getstate_setstate()

        # tuples of script, globs, hook
        self._script_snippets: list[
            tuple[str, dict, Callable[[dict, dict], Any]]
        ] = []
        self._repr_added = False

        # We want to only do this check once; in 99.9% of cases these
        # exist.
        if not hasattr(self._cls, "__module__") or not hasattr(
            self._cls, "__qualname__"
        ):
            self._add_method_dunders = self._add_method_dunders_safe
        else:
            self._add_method_dunders = self._add_method_dunders_unsafe

    def __repr__(self):
        return f"<_ClassBuilder(cls={self._cls.__name__})>"

    def _eval_snippets(self) -> None:
        """
        Evaluate any registered snippets in one go.
        """
        script = "\n".join([snippet[0] for snippet in self._script_snippets])
        globs = {}
        for _, snippet_globs, _ in self._script_snippets:
            globs.update(snippet_globs)

        locs = _linecache_and_compile(
            script,
            _generate_unique_filename(self._cls, "methods"),
            globs,
        )

        for _, _, hook in self._script_snippets:
            hook(self._cls_dict, locs)

    def build_class(self):
        """
        Finalize class based on the accumulated configuration.

        Builder cannot be used after calling this method.
        """
        self._eval_snippets()
        if self._slots is True:
            cls = self._create_slots_class()
        else:
            cls = self._patch_original_class()
            if PY_3_10_PLUS:
                cls = abc.update_abstractmethods(cls)

        # The method gets only called if it's not inherited from a base class.
        # _has_own_attribute does NOT work properly for classmethods.
        if (
            getattr(cls, "__attrs_init_subclass__", None)
            and "__attrs_init_subclass__" not in cls.__dict__
        ):
            cls.__attrs_init_subclass__()

        return cls

    def _patch_original_class(self):
        """
        Apply accumulated methods and return the class.
        """
        cls = self._cls
        base_names = self._base_names

        # Clean class of attribute definitions (`attr.ib()`s).
        if self._delete_attribs:
            for name in self._attr_names:
                if (
                    name not in base_names
                    and getattr(cls, name, _SENTINEL) is not _SENTINEL
                ):
                    # An AttributeError can happen if a base class defines a
                    # class variable and we want to set an attribute with the
                    # same name by using only a type annotation.
                    with contextlib.suppress(AttributeError):
                        delattr(cls, name)

        # Attach our dunder methods.
        for name, value in self._cls_dict.items():
            setattr(cls, name, value)

        # If we've inherited an attrs __setattr__ and don't write our own,
        # reset it to object's.
        if not self._wrote_own_setattr and getattr(
            cls, "__attrs_own_setattr__", False
        ):
            cls.__attrs_own_setattr__ = False

            if not self._has_custom_setattr:
                cls.__setattr__ = _OBJ_SETATTR

        return cls

    def _create_slots_class(self):
        """
        Build and return a new class with a `__slots__` attribute.
        """
        cd = {
            k: v
            for k, v in self._cls_dict.items()
            if k not in (*tuple(self._attr_names), "__dict__", "__weakref__")
        }

        # If our class doesn't have its own implementation of __setattr__
        # (either from the user or by us), check the bases, if one of them has
        # an attrs-made __setattr__, that needs to be reset. We don't walk the
        # MRO because we only care about our immediate base classes.
        # XXX: This can be confused by subclassing a slotted attrs class with
        # XXX: a non-attrs class and subclass the resulting class with an attrs
        # XXX: class.  See `test_slotted_confused` for details.  For now that's
        # XXX: OK with us.
        if not self._wrote_own_setattr:
            cd["__attrs_own_setattr__"] = False

            if not self._has_custom_setattr:
                for base_cls in self._cls.__bases__:
                    if base_cls.__dict__.get("__attrs_own_setattr__", False):
                        cd["__setattr__"] = _OBJ_SETATTR
                        break

        # Traverse the MRO to collect existing slots
        # and check for an existing __weakref__.
        existing_slots = {}
        weakref_inherited = False
        for base_cls in self._cls.__mro__[1:-1]:
            if base_cls.__dict__.get("__weakref__", None) is not None:
                weakref_inherited = True
            existing_slots.update(
                {
                    name: getattr(base_cls, name)
                    for name in getattr(base_cls, "__slots__", [])
                }
            )

        base_names = set(self._base_names)

        names = self._attr_names
        if (
            self._weakref_slot
            and "__weakref__" not in getattr(self._cls, "__slots__", ())
            and "__weakref__" not in names
            and not weakref_inherited
        ):
            names += ("__weakref__",)

        cached_properties = {
            name: cached_prop.func
            for name, cached_prop in cd.items()
            if isinstance(cached_prop, cached_property)
        }

        # Collect methods with a `__class__` reference that are shadowed in the new class.
        # To know to update them.
        additional_closure_functions_to_update = []
        if cached_properties:
            class_annotations = _get_annotations(self._cls)
            for name, func in cached_properties.items():
                # Add cached properties to names for slotting.
                names += (name,)
                # Clear out function from class to avoid clashing.
                del cd[name]
                additional_closure_functions_to_update.append(func)
                annotation = inspect.signature(func).return_annotation
                if annotation is not inspect.Parameter.empty:
                    class_annotations[name] = annotation

            original_getattr = cd.get("__getattr__")
            if original_getattr is not None:
                additional_closure_functions_to_update.append(original_getattr)

            cd["__getattr__"] = _make_cached_property_getattr(
                cached_properties, original_getattr, self._cls
            )

        # We only add the names of attributes that aren't inherited.
        # Setting __slots__ to inherited attributes wastes memory.
        slot_names = [name for name in names if name not in base_names]

        # There are slots for attributes from current class
        # that are defined in parent classes.
        # As their descriptors may be overridden by a child class,
        # we collect them here and update the class dict
        reused_slots = {
            slot: slot_descriptor
            for slot, slot_descriptor in existing_slots.items()
            if slot in slot_names
        }
        slot_names = [name for name in slot_names if name not in reused_slots]
        cd.update(reused_slots)
        if self._cache_hash:
            slot_names.append(_HASH_CACHE_FIELD)

        cd["__slots__"] = tuple(slot_names)

        cd["__qualname__"] = self._cls.__qualname__

        # Create new class based on old class and our methods.
        cls = type(self._cls)(self._cls.__name__, self._cls.__bases__, cd)

        # The following is a fix for
        # <https://github.com/python-attrs/attrs/issues/102>.
        # If a method mentions `__class__` or uses the no-arg super(), the
        # compiler will bake a reference to the class in the method itself
        # as `method.__closure__`.  Since we replace the class with a
        # clone, we rewrite these references so it keeps working.
        for item in itertools.chain(
            cls.__dict__.values(), additional_closure_functions_to_update
        ):
            if isinstance(item, (classmethod, staticmethod)):
                # Class- and staticmethods hide their functions inside.
                # These might need to be rewritten as well.
                closure_cells = getattr(item.__func__, "__closure__", None)
            elif isinstance(item, property):
                # Workaround for property `super()` shortcut (PY3-only).
                # There is no universal way for other descriptors.
                closure_cells = getattr(item.fget, "__closure__", None)
            else:
                closure_cells = getattr(item, "__closure__", None)

            if not closure_cells:  # Catch None or the empty list.
                continue
            for cell in closure_cells:
                try:
                    match = cell.cell_contents is self._cls
                except ValueError:  # noqa: PERF203
                    # ValueError: Cell is empty
                    pass
                else:
                    if match:
                        cell.cell_contents = cls
        return cls

    def add_repr(self, ns):
        script, globs = _make_repr_script(self._attrs, ns)

        def _attach_repr(cls_dict, globs):
            cls_dict["__repr__"] = self._add_method_dunders(globs["__repr__"])

        self._script_snippets.append((script, globs, _attach_repr))
        self._repr_added = True
        return self

    def add_str(self):
        if not self._repr_added:
            msg = "__str__ can only be generated if a __repr__ exists."
            raise ValueError(msg)

        def __str__(self):
            return self.__repr__()

        self._cls_dict["__str__"] = self._add_method_dunders(__str__)
        return self

    def _make_getstate_setstate(self):
        """
        Create custom __setstate__ and __getstate__ methods.
        """
        # __weakref__ is not writable.
        state_attr_names = tuple(
            an for an in self._attr_names if an != "__weakref__"
        )

        def slots_getstate(self):
            """
            Automatically created by attrs.
            """
            return {name: getattr(self, name) for name in state_attr_names}

        hash_caching_enabled = self._cache_hash

        def slots_setstate(self, state):
            """
            Automatically created by attrs.
            """
            __bound_setattr = _OBJ_SETATTR.__get__(self)
            if isinstance(state, tuple):
                # Backward compatibility with attrs instances pickled with
                # attrs versions before v22.2.0 which stored tuples.
                for name, value in zip(state_attr_names, state):
                    __bound_setattr(name, value)
            else:
                for name in state_attr_names:
                    if name in state:
                        __bound_setattr(name, state[name])

            # The hash code cache is not included when the object is
            # serialized, but it still needs to be initialized to None to
            # indicate that the first call to __hash__ should be a cache
            # miss.
            if hash_caching_enabled:
                __bound_setattr(_HASH_CACHE_FIELD, None)

        return slots_getstate, slots_setstate

    def make_unhashable(self):
        self._cls_dict["__hash__"] = None
        return self

    def add_hash(self):
        script, globs = _make_hash_script(
            self._cls,
            self._attrs,
            frozen=self._frozen,
            cache_hash=self._cache_hash,
        )

        def attach_hash(cls_dict: dict, locs: dict) -> None:
            cls_dict["__hash__"] = self._add_method_dunders(locs["__hash__"])

        self._script_snippets.append((script, globs, attach_hash))

        return self

    def add_init(self):
        script, globs, annotations = _make_init_script(
            self._cls,
            self._attrs,
            self._has_pre_init,
            self._pre_init_has_args,
            self._has_post_init,
            self._frozen,
            self._slots,
            self._cache_hash,
            self._base_attr_map,
            self._is_exc,
            self._on_setattr,
            attrs_init=False,
        )

        def _attach_init(cls_dict, globs):
            init = globs["__init__"]
            init.__annotations__ = annotations
            cls_dict["__init__"] = self._add_method_dunders(init)

        self._script_snippets.append((script, globs, _attach_init))

        return self

    def add_replace(self):
        self._cls_dict["__replace__"] = self._add_method_dunders(
            lambda self, **changes: evolve(self, **changes)
        )
        return self

    def add_match_args(self):
        self._cls_dict["__match_args__"] = tuple(
            field.name
            for field in self._attrs
            if field.init and not field.kw_only
        )

    def add_attrs_init(self):
        script, globs, annotations = _make_init_script(
            self._cls,
            self._attrs,
            self._has_pre_init,
            self._pre_init_has_args,
            self._has_post_init,
            self._frozen,
            self._slots,
            self._cache_hash,
            self._base_attr_map,
            self._is_exc,
            self._on_setattr,
            attrs_init=True,
        )

        def _attach_attrs_init(cls_dict, globs):
            init = globs["__attrs_init__"]
            init.__annotations__ = annotations
            cls_dict["__attrs_init__"] = self._add_method_dunders(init)

        self._script_snippets.append((script, globs, _attach_attrs_init))

        return self

    def add_eq(self):
        cd = self._cls_dict

        script, globs = _make_eq_script(self._attrs)

        def _attach_eq(cls_dict, globs):
            cls_dict["__eq__"] = self._add_method_dunders(globs["__eq__"])

        self._script_snippets.append((script, globs, _attach_eq))

        cd["__ne__"] = __ne__

        return self

    def add_order(self):
        cd = self._cls_dict

        cd["__lt__"], cd["__le__"], cd["__gt__"], cd["__ge__"] = (
            self._add_method_dunders(meth)
            for meth in _make_order(self._cls, self._attrs)
        )

        return self

    def add_setattr(self):
        sa_attrs = {}
        for a in self._attrs:
            on_setattr = a.on_setattr or self._on_setattr
            if on_setattr and on_setattr is not setters.NO_OP:
                sa_attrs[a.name] = a, on_setattr

        if not sa_attrs:
            return self

        if self._has_custom_setattr:
            # We need to write a __setattr__ but there already is one!
            msg = "Can't combine custom __setattr__ with on_setattr hooks."
            raise ValueError(msg)

        # docstring comes from _add_method_dunders
        def __setattr__(self, name, val):
            try:
                a, hook = sa_attrs[name]
            except KeyError:
                nval = val
            else:
                nval = hook(self, a, val)

            _OBJ_SETATTR(self, name, nval)

        self._cls_dict["__attrs_own_setattr__"] = True
        self._cls_dict["__setattr__"] = self._add_method_dunders(__setattr__)
        self._wrote_own_setattr = True

        return self

    def _add_method_dunders_unsafe(self, method: Callable) -> Callable:
        """
        Add __module__ and __qualname__ to a *method*.
        """
        method.__module__ = self._cls.__module__

        method.__qualname__ = f"{self._cls.__qualname__}.{method.__name__}"

        method.__doc__ = (
            f"Method generated by attrs for class {self._cls.__qualname__}."
        )

        return method

    def _add_method_dunders_safe(self, method: Callable) -> Callable:
        """
        Add __module__ and __qualname__ to a *method* if possible.
        """
        with contextlib.suppress(AttributeError):
            method.__module__ = self._cls.__module__

        with contextlib.suppress(AttributeError):
            method.__qualname__ = f"{self._cls.__qualname__}.{method.__name__}"

        with contextlib.suppress(AttributeError):
            method.__doc__ = f"Method generated by attrs for class {self._cls.__qualname__}."

        return method


def _determine_attrs_eq_order(cmp, eq, order, default_eq):
    """
    Validate the combination of *cmp*, *eq*, and *order*. Derive the effective
    values of eq and order.  If *eq* is None, set it to *default_eq*.
    """
    if cmp is not None and any((eq is not None, order is not None)):
        msg = "Don't mix `cmp` with `eq' and `order`."
        raise ValueError(msg)

    # cmp takes precedence due to bw-compatibility.
    if cmp is not None:
        return cmp, cmp

    # If left None, equality is set to the specified default and ordering
    # mirrors equality.
    if eq is None:
        eq = default_eq

    if order is None:
        order = eq

    if eq is False and order is True:
        msg = "`order` can only be True if `eq` is True too."
        raise ValueError(msg)

    return eq, order


def _determine_attrib_eq_order(cmp, eq, order, default_eq):
    """
    Validate the combination of *cmp*, *eq*, and *order*. Derive the effective
    values of eq and order.  If *eq* is None, set it to *default_eq*.
    """
    if cmp is not None and any((eq is not None, order is not None)):
        msg = "Don't mix `cmp` with `eq' and `order`."
        raise ValueError(msg)

    def decide_callable_or_boolean(value):
        """
        Decide whether a key function is used.
        """
        if callable(value):
            value, key = True, value
        else:
            key = None
        return value, key

    # cmp takes precedence due to bw-compatibility.
    if cmp is not None:
        cmp, cmp_key = decide_callable_or_boolean(cmp)
        return cmp, cmp_key, cmp, cmp_key

    # If left None, equality is set to the specified default and ordering
    # mirrors equality.
    if eq is None:
        eq, eq_key = default_eq, None
    else:
        eq, eq_key = decide_callable_or_boolean(eq)

    if order is None:
        order, order_key = eq, eq_key
    else:
        order, order_key = decide_callable_or_boolean(order)

    if eq is False and order is True:
        msg = "`order` can only be True if `eq` is True too."
        raise ValueError(msg)

    return eq, eq_key, order, order_key


def _determine_whether_to_implement(
    cls, flag, auto_detect, dunders, default=True
):
    """
    Check whether we should implement a set of methods for *cls*.

    *flag* is the argument passed into @attr.s like 'init', *auto_detect* the
    same as passed into @attr.s and *dunders* is a tuple of attribute names
    whose presence signal that the user has implemented it themselves.

    Return *default* if no reason for either for or against is found.
    """
    if flag is True or flag is False:
        return flag

    if flag is None and auto_detect is False:
        return default

    # Logically, flag is None and auto_detect is True here.
    for dunder in dunders:
        if _has_own_attribute(cls, dunder):
            return False

    return default


def attrs(
    maybe_cls=None,
    these=None,
    repr_ns=None,
    repr=None,
    cmp=None,
    hash=None,
    init=None,
    slots=False,
    frozen=False,
    weakref_slot=True,
    str=False,
    auto_attribs=False,
    kw_only=False,
    cache_hash=False,
    auto_exc=False,
    eq=None,
    order=None,
    auto_detect=False,
    collect_by_mro=False,
    getstate_setstate=None,
    on_setattr=None,
    field_transformer=None,
    match_args=True,
    unsafe_hash=None,
):
    r"""
    A class decorator that adds :term:`dunder methods` according to the
    specified attributes using `attr.ib` or the *these* argument.

    Consider using `attrs.define` / `attrs.frozen` in new code (``attr.s`` will
    *never* go away, though).

    Args:
        repr_ns (str):
            When using nested classes, there was no way in Python 2 to
            automatically detect that.  This argument allows to set a custom
            name for a more meaningful ``repr`` output.  This argument is
            pointless in Python 3 and is therefore deprecated.

    .. caution::
        Refer to `attrs.define` for the rest of the parameters, but note that they
        can have different defaults.

        Notably, leaving *on_setattr* as `None` will **not** add any hooks.

    .. versionadded:: 16.0.0 *slots*
    .. versionadded:: 16.1.0 *frozen*
    .. versionadded:: 16.3.0 *str*
    .. versionadded:: 16.3.0 Support for ``__attrs_post_init__``.
    .. versionchanged:: 17.1.0
       *hash* supports `None` as value which is also the default now.
    .. versionadded:: 17.3.0 *auto_attribs*
    .. versionchanged:: 18.1.0
       If *these* is passed, no attributes are deleted from the class body.
    .. versionchanged:: 18.1.0 If *these* is ordered, the order is retained.
    .. versionadded:: 18.2.0 *weakref_slot*
    .. deprecated:: 18.2.0
       ``__lt__``, ``__le__``, ``__gt__``, and ``__ge__`` now raise a
       `DeprecationWarning` if the classes compared are subclasses of
       each other. ``__eq`` and ``__ne__`` never tried to compared subclasses
       to each other.
    .. versionchanged:: 19.2.0
       ``__lt__``, ``__le__``, ``__gt__``, and ``__ge__`` now do not consider
       subclasses comparable anymore.
    .. versionadded:: 18.2.0 *kw_only*
    .. versionadded:: 18.2.0 *cache_hash*
    .. versionadded:: 19.1.0 *auto_exc*
    .. deprecated:: 19.2.0 *cmp* Removal on or after 2021-06-01.
    .. versionadded:: 19.2.0 *eq* and *order*
    .. versionadded:: 20.1.0 *auto_detect*
    .. versionadded:: 20.1.0 *collect_by_mro*
    .. versionadded:: 20.1.0 *getstate_setstate*
    .. versionadded:: 20.1.0 *on_setattr*
    .. versionadded:: 20.3.0 *field_transformer*
    .. versionchanged:: 21.1.0
       ``init=False`` injects ``__attrs_init__``
    .. versionchanged:: 21.1.0 Support for ``__attrs_pre_init__``
    .. versionchanged:: 21.1.0 *cmp* undeprecated
    .. versionadded:: 21.3.0 *match_args*
    .. versionadded:: 22.2.0
       *unsafe_hash* as an alias for *hash* (for :pep:`681` compliance).
    .. deprecated:: 24.1.0 *repr_ns*
    .. versionchanged:: 24.1.0
       Instances are not compared as tuples of attributes anymore, but using a
       big ``and`` condition. This is faster and has more correct behavior for
       uncomparable values like `math.nan`.
    .. versionadded:: 24.1.0
       If a class has an *inherited* classmethod called
       ``__attrs_init_subclass__``, it is executed after the class is created.
    .. deprecated:: 24.1.0 *hash* is deprecated in favor of *unsafe_hash*.
    """
    if repr_ns is not None:
        import warnings

        warnings.warn(
            DeprecationWarning(
                "The `repr_ns` argument is deprecated and will be removed in or after August 2025."
            ),
            stacklevel=2,
        )

    eq_, order_ = _determine_attrs_eq_order(cmp, eq, order, None)

    #  unsafe_hash takes precedence due to PEP 681.
    if unsafe_hash is not None:
        hash = unsafe_hash

    if isinstance(on_setattr, (list, tuple)):
        on_setattr = setters.pipe(*on_setattr)

    def wrap(cls):
        is_frozen = frozen or _has_frozen_base_class(cls)
        is_exc = auto_exc is True and issubclass(cls, BaseException)
        has_own_setattr = auto_detect and _has_own_attribute(
            cls, "__setattr__"
        )

        if has_own_setattr and is_frozen:
            msg = "Can't freeze a class with a custom __setattr__."
            raise ValueError(msg)

        builder = _ClassBuilder(
            cls,
            these,
            slots,
            is_frozen,
            weakref_slot,
            _determine_whether_to_implement(
                cls,
                getstate_setstate,
                auto_detect,
                ("__getstate__", "__setstate__"),
                default=slots,
            ),
            auto_attribs,
            kw_only,
            cache_hash,
            is_exc,
            collect_by_mro,
            on_setattr,
            has_own_setattr,
            field_transformer,
        )

        if _determine_whether_to_implement(
            cls, repr, auto_detect, ("__repr__",)
        ):
            builder.add_repr(repr_ns)

        if str is True:
            builder.add_str()

        eq = _determine_whether_to_implement(
            cls, eq_, auto_detect, ("__eq__", "__ne__")
        )
        if not is_exc and eq is True:
            builder.add_eq()
        if not is_exc and _determine_whether_to_implement(
            cls, order_, auto_detect, ("__lt__", "__le__", "__gt__", "__ge__")
        ):
            builder.add_order()

        if not frozen:
            builder.add_setattr()

        nonlocal hash
        if (
            hash is None
            and auto_detect is True
            and _has_own_attribute(cls, "__hash__")
        ):
            hash = False

        if hash is not True and hash is not False and hash is not None:
            # Can't use `hash in` because 1 == True for example.
            msg = "Invalid value for hash.  Must be True, False, or None."
            raise TypeError(msg)

        if hash is False or (hash is None and eq is False) or is_exc:
            # Don't do anything. Should fall back to __object__'s __hash__
            # which is by id.
            if cache_hash:
                msg = "Invalid value for cache_hash.  To use hash caching, hashing must be either explicitly or implicitly enabled."
                raise TypeError(msg)
        elif hash is True or (
            hash is None and eq is True and is_frozen is True
        ):
            # Build a __hash__ if told so, or if it's safe.
            builder.add_hash()
        else:
            # Raise TypeError on attempts to hash.
            if cache_hash:
                msg = "Invalid value for cache_hash.  To use hash caching, hashing must be either explicitly or implicitly enabled."
                raise TypeError(msg)
            builder.make_unhashable()

        if _determine_whether_to_implement(
            cls, init, auto_detect, ("__init__",)
        ):
            builder.add_init()
        else:
            builder.add_attrs_init()
            if cache_hash:
                msg = "Invalid value for cache_hash.  To use hash caching, init must be True."
                raise TypeError(msg)

        if PY_3_13_PLUS and not _has_own_attribute(cls, "__replace__"):
            builder.add_replace()

        if (
            PY_3_10_PLUS
            and match_args
            and not _has_own_attribute(cls, "__match_args__")
        ):
            builder.add_match_args()

        return builder.build_class()

    # maybe_cls's type depends on the usage of the decorator.  It's a class
    # if it's used as `@attrs` but `None` if used as `@attrs()`.
    if maybe_cls is None:
        return wrap

    return wrap(maybe_cls)


_attrs = attrs
"""
Internal alias so we can use it in functions that take an argument called
*attrs*.
"""


def _has_frozen_base_class(cls):
    """
    Check whether *cls* has a frozen ancestor by looking at its
    __setattr__.
    """
    return cls.__setattr__ is _frozen_setattrs


def _generate_unique_filename(cls: type, func_name: str) -> str:
    """
    Create a "filename" suitable for a function being generated.
    """
    return (
        f"<attrs generated {func_name} {cls.__module__}."
        f"{getattr(cls, '__qualname__', cls.__name__)}>"
    )


def _make_hash_script(
    cls: type, attrs: list[Attribute], frozen: bool, cache_hash: bool
) -> tuple[str, dict]:
    attrs = tuple(
        a for a in attrs if a.hash is True or (a.hash is None and a.eq is True)
    )

    tab = "        "

    type_hash = hash(_generate_unique_filename(cls, "hash"))
    # If eq is custom generated, we need to include the functions in globs
    globs = {}

    hash_def = "def __hash__(self"
    hash_func = "hash(("
    closing_braces = "))"
    if not cache_hash:
        hash_def += "):"
    else:
        hash_def += ", *"

        hash_def += ", _cache_wrapper=__import__('attr._make')._make._CacheHashWrapper):"
        hash_func = "_cache_wrapper(" + hash_func
        closing_braces += ")"

    method_lines = [hash_def]

    def append_hash_computation_lines(prefix, indent):
        """
        Generate the code for actually computing the hash code.
        Below this will either be returned directly or used to compute
        a value which is then cached, depending on the value of cache_hash
        """

        method_lines.extend(
            [
                indent + prefix + hash_func,
                indent + f"        {type_hash},",
            ]
        )

        for a in attrs:
            if a.eq_key:
                cmp_name = f"_{a.name}_key"
                globs[cmp_name] = a.eq_key
                method_lines.append(
                    indent + f"        {cmp_name}(self.{a.name}),"
                )
            else:
                method_lines.append(indent + f"        self.{a.name},")

        method_lines.append(indent + "    " + closing_braces)

    if cache_hash:
        method_lines.append(tab + f"if self.{_HASH_CACHE_FIELD} is None:")
        if frozen:
            append_hash_computation_lines(
                f"object.__setattr__(self, '{_HASH_CACHE_FIELD}', ", tab * 2
            )
            method_lines.append(tab * 2 + ")")  # close __setattr__
        else:
            append_hash_computation_lines(
                f"self.{_HASH_CACHE_FIELD} = ", tab * 2
            )
        method_lines.append(tab + f"return self.{_HASH_CACHE_FIELD}")
    else:
        append_hash_computation_lines("return ", tab)

    script = "\n".join(method_lines)
    return script, globs


def _add_hash(cls: type, attrs: list[Attribute]):
    """
    Add a hash method to *cls*.
    """
    script, globs = _make_hash_script(
        cls, attrs, frozen=False, cache_hash=False
    )
    _compile_and_eval(
        script, globs, filename=_generate_unique_filename(cls, "__hash__")
    )
    cls.__hash__ = globs["__hash__"]
    return cls


def __ne__(self, other):
    """
    Check equality and either forward a NotImplemented or
    return the result negated.
    """
    result = self.__eq__(other)
    if result is NotImplemented:
        return NotImplemented

    return not result


def _make_eq_script(attrs: list) -> tuple[str, dict]:
    """
    Create __eq__ method for *cls* with *attrs*.
    """
    attrs = [a for a in attrs if a.eq]

    lines = [
        "def __eq__(self, other):",
        "    if other.__class__ is not self.__class__:",
        "        return NotImplemented",
    ]

    globs = {}
    if attrs:
        lines.append("    return  (")
        for a in attrs:
            if a.eq_key:
                cmp_name = f"_{a.name}_key"
                # Add the key function to the global namespace
                # of the evaluated function.
                globs[cmp_name] = a.eq_key
                lines.append(
                    f"        {cmp_name}(self.{a.name}) == {cmp_name}(other.{a.name})"
                )
            else:
                lines.append(f"        self.{a.name} == other.{a.name}")
            if a is not attrs[-1]:
                lines[-1] = f"{lines[-1]} and"
        lines.append("    )")
    else:
        lines.append("    return True")

    script = "\n".join(lines)

    return script, globs


def _make_order(cls, attrs):
    """
    Create ordering methods for *cls* with *attrs*.
    """
    attrs = [a for a in attrs if a.order]

    def attrs_to_tuple(obj):
        """
        Save us some typing.
        """
        return tuple(
            key(value) if key else value
            for value, key in (
                (getattr(obj, a.name), a.order_key) for a in attrs
            )
        )

    def __lt__(self, other):
        """
        Automatically created by attrs.
        """
        if other.__class__ is self.__class__:
            return attrs_to_tuple(self) < attrs_to_tuple(other)

        return NotImplemented

    def __le__(self, other):
        """
        Automatically created by attrs.
        """
        if other.__class__ is self.__class__:
            return attrs_to_tuple(self) <= attrs_to_tuple(other)

        return NotImplemented

    def __gt__(self, other):
        """
        Automatically created by attrs.
        """
        if other.__class__ is self.__class__:
            return attrs_to_tuple(self) > attrs_to_tuple(other)

        return NotImplemented

    def __ge__(self, other):
        """
        Automatically created by attrs.
        """
        if other.__class__ is self.__class__:
            return attrs_to_tuple(self) >= attrs_to_tuple(other)

        return NotImplemented

    return __lt__, __le__, __gt__, __ge__


def _add_eq(cls, attrs=None):
    """
    Add equality methods to *cls* with *attrs*.
    """
    if attrs is None:
        attrs = cls.__attrs_attrs__

    script, globs = _make_eq_script(attrs)
    _compile_and_eval(
        script, globs, filename=_generate_unique_filename(cls, "__eq__")
    )
    cls.__eq__ = globs["__eq__"]
    cls.__ne__ = __ne__

    return cls


def _make_repr_script(attrs, ns) -> tuple[str, dict]:
    """
    Create the source and globs for a __repr__ and return it.
    """
    # Figure out which attributes to include, and which function to use to
    # format them. The a.repr value can be either bool or a custom
    # callable.
    attr_names_with_reprs = tuple(
        (a.name, (repr if a.repr is True else a.repr), a.init)
        for a in attrs
        if a.repr is not False
    )
    globs = {
        name + "_repr": r for name, r, _ in attr_names_with_reprs if r != repr
    }
    globs["_compat"] = _compat
    globs["AttributeError"] = AttributeError
    globs["NOTHING"] = NOTHING
    attribute_fragments = []
    for name, r, i in attr_names_with_reprs:
        accessor = (
            "self." + name if i else 'getattr(self, "' + name + '", NOTHING)'
        )
        fragment = (
            "%s={%s!r}" % (name, accessor)
            if r == repr
            else "%s={%s_repr(%s)}" % (name, name, accessor)
        )
        attribute_fragments.append(fragment)
    repr_fragment = ", ".join(attribute_fragments)

    if ns is None:
        cls_name_fragment = '{self.__class__.__qualname__.rsplit(">.", 1)[-1]}'
    else:
        cls_name_fragment = ns + ".{self.__class__.__name__}"

    lines = [
        "def __repr__(self):",
        "  try:",
        "    already_repring = _compat.repr_context.already_repring",
        "  except AttributeError:",
        "    already_repring = {id(self),}",
        "    _compat.repr_context.already_repring = already_repring",
        "  else:",
        "    if id(self) in already_repring:",
        "      return '...'",
        "    else:",
        "      already_repring.add(id(self))",
        "  try:",
        f"    return f'{cls_name_fragment}({repr_fragment})'",
        "  finally:",
        "    already_repring.remove(id(self))",
    ]

    return "\n".join(lines), globs


def _add_repr(cls, ns=None, attrs=None):
    """
    Add a repr method to *cls*.
    """
    if attrs is None:
        attrs = cls.__attrs_attrs__

    script, globs = _make_repr_script(attrs, ns)
    _compile_and_eval(
        script, globs, filename=_generate_unique_filename(cls, "__repr__")
    )
    cls.__repr__ = globs["__repr__"]
    return cls


def fields(cls):
    """
    Return the tuple of *attrs* attributes for a class.

    The tuple also allows accessing the fields by their names (see below for
    examples).

    Args:
        cls (type): Class to introspect.

    Raises:
        TypeError: If *cls* is not a class.

        attrs.exceptions.NotAnAttrsClassError:
            If *cls* is not an *attrs* class.

    Returns:
        tuple (with name accessors) of `attrs.Attribute`

    .. versionchanged:: 16.2.0 Returned tuple allows accessing the fields
       by name.
    .. versionchanged:: 23.1.0 Add support for generic classes.
    """
    generic_base = get_generic_base(cls)

    if generic_base is None and not isinstance(cls, type):
        msg = "Passed object must be a class."
        raise TypeError(msg)

    attrs = getattr(cls, "__attrs_attrs__", None)

    if attrs is None:
        if generic_base is not None:
            attrs = getattr(generic_base, "__attrs_attrs__", None)
            if attrs is not None:
                # Even though this is global state, stick it on here to speed
                # it up. We rely on `cls` being cached for this to be
                # efficient.
                cls.__attrs_attrs__ = attrs
                return attrs
        msg = f"{cls!r} is not an attrs-decorated class."
        raise NotAnAttrsClassError(msg)

    return attrs


def fields_dict(cls):
    """
    Return an ordered dictionary of *attrs* attributes for a class, whose keys
    are the attribute names.

    Args:
        cls (type): Class to introspect.

    Raises:
        TypeError: If *cls* is not a class.

        attrs.exceptions.NotAnAttrsClassError:
            If *cls* is not an *attrs* class.

    Returns:
        dict[str, attrs.Attribute]: Dict of attribute name to definition

    .. versionadded:: 18.1.0
    """
    if not isinstance(cls, type):
        msg = "Passed object must be a class."
        raise TypeError(msg)
    attrs = getattr(cls, "__attrs_attrs__", None)
    if attrs is None:
        msg = f"{cls!r} is not an attrs-decorated class."
        raise NotAnAttrsClassError(msg)
    return {a.name: a for a in attrs}


def validate(inst):
    """
    Validate all attributes on *inst* that have a validator.

    Leaves all exceptions through.

    Args:
        inst: Instance of a class with *attrs* attributes.
    """
    if _config._run_validators is False:
        return

    for a in fields(inst.__class__):
        v = a.validator
        if v is not None:
            v(inst, a, getattr(inst, a.name))


def _is_slot_attr(a_name, base_attr_map):
    """
    Check if the attribute name comes from a slot class.
    """
    cls = base_attr_map.get(a_name)
    return cls and "__slots__" in cls.__dict__


def _make_init_script(
    cls,
    attrs,
    pre_init,
    pre_init_has_args,
    post_init,
    frozen,
    slots,
    cache_hash,
    base_attr_map,
    is_exc,
    cls_on_setattr,
    attrs_init,
) -> tuple[str, dict, dict]:
    has_cls_on_setattr = (
        cls_on_setattr is not None and cls_on_setattr is not setters.NO_OP
    )

    if frozen and has_cls_on_setattr:
        msg = "Frozen classes can't use on_setattr."
        raise ValueError(msg)

    needs_cached_setattr = cache_hash or frozen
    filtered_attrs = []
    attr_dict = {}
    for a in attrs:
        if not a.init and a.default is NOTHING:
            continue

        filtered_attrs.append(a)
        attr_dict[a.name] = a

        if a.on_setattr is not None:
            if frozen is True:
                msg = "Frozen classes can't use on_setattr."
                raise ValueError(msg)

            needs_cached_setattr = True
        elif has_cls_on_setattr and a.on_setattr is not setters.NO_OP:
            needs_cached_setattr = True

    script, globs, annotations = _attrs_to_init_script(
        filtered_attrs,
        frozen,
        slots,
        pre_init,
        pre_init_has_args,
        post_init,
        cache_hash,
        base_attr_map,
        is_exc,
        needs_cached_setattr,
        has_cls_on_setattr,
        "__attrs_init__" if attrs_init else "__init__",
    )
    if cls.__module__ in sys.modules:
        # This makes typing.get_type_hints(CLS.__init__) resolve string types.
        globs.update(sys.modules[cls.__module__].__dict__)

    globs.update({"NOTHING": NOTHING, "attr_dict": attr_dict})

    if needs_cached_setattr:
        # Save the lookup overhead in __init__ if we need to circumvent
        # setattr hooks.
        globs["_cached_setattr_get"] = _OBJ_SETATTR.__get__

    return script, globs, annotations


def _setattr(attr_name: str, value_var: str, has_on_setattr: bool) -> str:
    """
    Use the cached object.setattr to set *attr_name* to *value_var*.
    """
    return f"_setattr('{attr_name}', {value_var})"


def _setattr_with_converter(
    attr_name: str, value_var: str, has_on_setattr: bool, converter: Converter
) -> str:
    """
    Use the cached object.setattr to set *attr_name* to *value_var*, but run
    its converter first.
    """
    return f"_setattr('{attr_name}', {converter._fmt_converter_call(attr_name, value_var)})"


def _assign(attr_name: str, value: str, has_on_setattr: bool) -> str:
    """
    Unless *attr_name* has an on_setattr hook, use normal assignment. Otherwise
    relegate to _setattr.
    """
    if has_on_setattr:
        return _setattr(attr_name, value, True)

    return f"self.{attr_name} = {value}"


def _assign_with_converter(
    attr_name: str, value_var: str, has_on_setattr: bool, converter: Converter
) -> str:
    """
    Unless *attr_name* has an on_setattr hook, use normal assignment after
    conversion. Otherwise relegate to _setattr_with_converter.
    """
    if has_on_setattr:
        return _setattr_with_converter(attr_name, value_var, True, converter)

    return f"self.{attr_name} = {converter._fmt_converter_call(attr_name, value_var)}"


def _determine_setters(
    frozen: bool, slots: bool, base_attr_map: dict[str, type]
):
    """
    Determine the correct setter functions based on whether a class is frozen
    and/or slotted.
    """
    if frozen is True:
        if slots is True:
            return (), _setattr, _setattr_with_converter

        # Dict frozen classes assign directly to __dict__.
        # But only if the attribute doesn't come from an ancestor slot
        # class.
        # Note _inst_dict will be used again below if cache_hash is True

        def fmt_setter(
            attr_name: str, value_var: str, has_on_setattr: bool
        ) -> str:
            if _is_slot_attr(attr_name, base_attr_map):
                return _setattr(attr_name, value_var, has_on_setattr)

            return f"_inst_dict['{attr_name}'] = {value_var}"

        def fmt_setter_with_converter(
            attr_name: str,
            value_var: str,
            has_on_setattr: bool,
            converter: Converter,
        ) -> str:
            if has_on_setattr or _is_slot_attr(attr_name, base_attr_map):
                return _setattr_with_converter(
                    attr_name, value_var, has_on_setattr, converter
                )

            return f"_inst_dict['{attr_name}'] = {converter._fmt_converter_call(attr_name, value_var)}"

        return (
            ("_inst_dict = self.__dict__",),
            fmt_setter,
            fmt_setter_with_converter,
        )

    # Not frozen -- we can just assign directly.
    return (), _assign, _assign_with_converter


def _attrs_to_init_script(
    attrs: list[Attribute],
    is_frozen: bool,
    is_slotted: bool,
    call_pre_init: bool,
    pre_init_has_args: bool,
    call_post_init: bool,
    does_cache_hash: bool,
    base_attr_map: dict[str, type],
    is_exc: bool,
    needs_cached_setattr: bool,
    has_cls_on_setattr: bool,
    method_name: str,
) -> tuple[str, dict, dict]:
    """
    Return a script of an initializer for *attrs*, a dict of globals, and
    annotations for the initializer.

    The globals are required by the generated script.
    """
    lines = ["self.__attrs_pre_init__()"] if call_pre_init else []

    if needs_cached_setattr:
        lines.append(
            # Circumvent the __setattr__ descriptor to save one lookup per
            # assignment. Note _setattr will be used again below if
            # does_cache_hash is True.
            "_setattr = _cached_setattr_get(self)"
        )

    extra_lines, fmt_setter, fmt_setter_with_converter = _determine_setters(
        is_frozen, is_slotted, base_attr_map
    )
    lines.extend(extra_lines)

    args = []
    kw_only_args = []
    attrs_to_validate = []

    # This is a dictionary of names to validator and converter callables.
    # Injecting this into __init__ globals lets us avoid lookups.
    names_for_globals = {}
    annotations = {"return": None}

    for a in attrs:
        if a.validator:
            attrs_to_validate.append(a)

        attr_name = a.name
        has_on_setattr = a.on_setattr is not None or (
            a.on_setattr is not setters.NO_OP and has_cls_on_setattr
        )
        # a.alias is set to maybe-mangled attr_name in _ClassBuilder if not
        # explicitly provided
        arg_name = a.alias

        has_factory = isinstance(a.default, Factory)
        maybe_self = "self" if has_factory and a.default.takes_self else ""

        if a.converter is not None and not isinstance(a.converter, Converter):
            converter = Converter(a.converter)
        else:
            converter = a.converter

        if a.init is False:
            if has_factory:
                init_factory_name = _INIT_FACTORY_PAT % (a.name,)
                if converter is not None:
                    lines.append(
                        fmt_setter_with_converter(
                            attr_name,
                            init_factory_name + f"({maybe_self})",
                            has_on_setattr,
                            converter,
                        )
                    )
                    names_for_globals[converter._get_global_name(a.name)] = (
                        converter.converter
                    )
                else:
                    lines.append(
                        fmt_setter(
                            attr_name,
                            init_factory_name + f"({maybe_self})",
                            has_on_setattr,
                        )
                    )
                names_for_globals[init_factory_name] = a.default.factory
            elif converter is not None:
                lines.append(
                    fmt_setter_with_converter(
                        attr_name,
                        f"attr_dict['{attr_name}'].default",
                        has_on_setattr,
                        converter,
                    )
                )
                names_for_globals[converter._get_global_name(a.name)] = (
                    converter.converter
                )
            else:
                lines.append(
                    fmt_setter(
                        attr_name,
                        f"attr_dict['{attr_name}'].default",
                        has_on_setattr,
                    )
                )
        elif a.default is not NOTHING and not has_factory:
            arg = f"{arg_name}=attr_dict['{attr_name}'].default"
            if a.kw_only:
                kw_only_args.append(arg)
            else:
                args.append(arg)

            if converter is not None:
                lines.append(
                    fmt_setter_with_converter(
                        attr_name, arg_name, has_on_setattr, converter
                    )
                )
                names_for_globals[converter._get_global_name(a.name)] = (
                    converter.converter
                )
            else:
                lines.append(fmt_setter(attr_name, arg_name, has_on_setattr))

        elif has_factory:
            arg = f"{arg_name}=NOTHING"
            if a.kw_only:
                kw_only_args.append(arg)
            else:
                args.append(arg)
            lines.append(f"if {arg_name} is not NOTHING:")

            init_factory_name = _INIT_FACTORY_PAT % (a.name,)
            if converter is not None:
                lines.append(
                    "    "
                    + fmt_setter_with_converter(
                        attr_name, arg_name, has_on_setattr, converter
                    )
                )
                lines.append("else:")
                lines.append(
                    "    "
                    + fmt_setter_with_converter(
                        attr_name,
                        init_factory_name + "(" + maybe_self + ")",
                        has_on_setattr,
                        converter,
                    )
                )
                names_for_globals[converter._get_global_name(a.name)] = (
                    converter.converter
                )
            else:
                lines.append(
                    "    " + fmt_setter(attr_name, arg_name, has_on_setattr)
                )
                lines.append("else:")
                lines.append(
                    "    "
                    + fmt_setter(
                        attr_name,
                        init_factory_name + "(" + maybe_self + ")",
                        has_on_setattr,
                    )
                )
            names_for_globals[init_factory_name] = a.default.factory
        else:
            if a.kw_only:
                kw_only_args.append(arg_name)
            else:
                args.append(arg_name)

            if converter is not None:
                lines.append(
                    fmt_setter_with_converter(
                        attr_name, arg_name, has_on_setattr, converter
                    )
                )
                names_for_globals[converter._get_global_name(a.name)] = (
                    converter.converter
                )
            else:
                lines.append(fmt_setter(attr_name, arg_name, has_on_setattr))

        if a.init is True:
            if a.type is not None and converter is None:
                annotations[arg_name] = a.type
            elif converter is not None and converter._first_param_type:
                # Use the type from the converter if present.
                annotations[arg_name] = converter._first_param_type

    if attrs_to_validate:  # we can skip this if there are no validators.
        names_for_globals["_config"] = _config
        lines.append("if _config._run_validators is True:")
        for a in attrs_to_validate:
            val_name = "__attr_validator_" + a.name
            attr_name = "__attr_" + a.name
            lines.append(f"    {val_name}(self, {attr_name}, self.{a.name})")
            names_for_globals[val_name] = a.validator
            names_for_globals[attr_name] = a

    if call_post_init:
        lines.append("self.__attrs_post_init__()")

    # Because this is set only after __attrs_post_init__ is called, a crash
    # will result if post-init tries to access the hash code.  This seemed
    # preferable to setting this beforehand, in which case alteration to field
    # values during post-init combined with post-init accessing the hash code
    # would result in silent bugs.
    if does_cache_hash:
        if is_frozen:
            if is_slotted:
                init_hash_cache = f"_setattr('{_HASH_CACHE_FIELD}', None)"
            else:
                init_hash_cache = f"_inst_dict['{_HASH_CACHE_FIELD}'] = None"
        else:
            init_hash_cache = f"self.{_HASH_CACHE_FIELD} = None"
        lines.append(init_hash_cache)

    # For exceptions we rely on BaseException.__init__ for proper
    # initialization.
    if is_exc:
        vals = ",".join(f"self.{a.name}" for a in attrs if a.init)

        lines.append(f"BaseException.__init__(self, {vals})")

    args = ", ".join(args)
    pre_init_args = args
    if kw_only_args:
        # leading comma & kw_only args
        args += f"{', ' if args else ''}*, {', '.join(kw_only_args)}"
        pre_init_kw_only_args = ", ".join(
            [
                f"{kw_arg_name}={kw_arg_name}"
                # We need to remove the defaults from the kw_only_args.
                for kw_arg_name in (kwa.split("=")[0] for kwa in kw_only_args)
            ]
        )
        pre_init_args += ", " if pre_init_args else ""
        pre_init_args += pre_init_kw_only_args

    if call_pre_init and pre_init_has_args:
        # If pre init method has arguments, pass same arguments as `__init__`.
        lines[0] = f"self.__attrs_pre_init__({pre_init_args})"

    # Python <3.12 doesn't allow backslashes in f-strings.
    NL = "\n    "
    return (
        f"""def {method_name}(self, {args}):
    {NL.join(lines) if lines else "pass"}
""",
        names_for_globals,
        annotations,
    )


def _default_init_alias_for(name: str) -> str:
    """
    The default __init__ parameter name for a field.

    This performs private-name adjustment via leading-unscore stripping,
    and is the default value of Attribute.alias if not provided.
    """

    return name.lstrip("_")


class Attribute:
    """
    *Read-only* representation of an attribute.

    .. warning::

       You should never instantiate this class yourself.

    The class has *all* arguments of `attr.ib` (except for ``factory`` which is
    only syntactic sugar for ``default=Factory(...)`` plus the following:

    - ``name`` (`str`): The name of the attribute.
    - ``alias`` (`str`): The __init__ parameter name of the attribute, after
      any explicit overrides and default private-attribute-name handling.
    - ``inherited`` (`bool`): Whether or not that attribute has been inherited
      from a base class.
    - ``eq_key`` and ``order_key`` (`typing.Callable` or `None`): The
      callables that are used for comparing and ordering objects by this
      attribute, respectively. These are set by passing a callable to
      `attr.ib`'s ``eq``, ``order``, or ``cmp`` arguments. See also
      :ref:`comparison customization <custom-comparison>`.

    Instances of this class are frequently used for introspection purposes
    like:

    - `fields` returns a tuple of them.
    - Validators get them passed as the first argument.
    - The :ref:`field transformer <transform-fields>` hook receives a list of
      them.
    - The ``alias`` property exposes the __init__ parameter name of the field,
      with any overrides and default private-attribute handling applied.


    .. versionadded:: 20.1.0 *inherited*
    .. versionadded:: 20.1.0 *on_setattr*
    .. versionchanged:: 20.2.0 *inherited* is not taken into account for
        equality checks and hashing anymore.
    .. versionadded:: 21.1.0 *eq_key* and *order_key*
    .. versionadded:: 22.2.0 *alias*

    For the full version history of the fields, see `attr.ib`.
    """

    # These slots must NOT be reordered because we use them later for
    # instantiation.
    __slots__ = (  # noqa: RUF023
        "name",
        "default",
        "validator",
        "repr",
        "eq",
        "eq_key",
        "order",
        "order_key",
        "hash",
        "init",
        "metadata",
        "type",
        "converter",
        "kw_only",
        "inherited",
        "on_setattr",
        "alias",
    )

    def __init__(
        self,
        name,
        default,
        validator,
        repr,
        cmp,  # XXX: unused, remove along with other cmp code.
        hash,
        init,
        inherited,
        metadata=None,
        type=None,
        converter=None,
        kw_only=False,
        eq=None,
        eq_key=None,
        order=None,
        order_key=None,
        on_setattr=None,
        alias=None,
    ):
        eq, eq_key, order, order_key = _determine_attrib_eq_order(
            cmp, eq_key or eq, order_key or order, True
        )

        # Cache this descriptor here to speed things up later.
        bound_setattr = _OBJ_SETATTR.__get__(self)

        # Despite the big red warning, people *do* instantiate `Attribute`
        # themselves.
        bound_setattr("name", name)
        bound_setattr("default", default)
        bound_setattr("validator", validator)
        bound_setattr("repr", repr)
        bound_setattr("eq", eq)
        bound_setattr("eq_key", eq_key)
        bound_setattr("order", order)
        bound_setattr("order_key", order_key)
        bound_setattr("hash", hash)
        bound_setattr("init", init)
        bound_setattr("converter", converter)
        bound_setattr(
            "metadata",
            (
                types.MappingProxyType(dict(metadata))  # Shallow copy
                if metadata
                else _EMPTY_METADATA_SINGLETON
            ),
        )
        bound_setattr("type", type)
        bound_setattr("kw_only", kw_only)
        bound_setattr("inherited", inherited)
        bound_setattr("on_setattr", on_setattr)
        bound_setattr("alias", alias)

    def __setattr__(self, name, value):
        raise FrozenInstanceError

    @classmethod
    def from_counting_attr(cls, name: str, ca: _CountingAttr, type=None):
        # type holds the annotated value. deal with conflicts:
        if type is None:
            type = ca.type
        elif ca.type is not None:
            msg = f"Type annotation and type argument cannot both be present for '{name}'."
            raise ValueError(msg)
        return cls(
            name,
            ca._default,
            ca._validator,
            ca.repr,
            None,
            ca.hash,
            ca.init,
            False,
            ca.metadata,
            type,
            ca.converter,
            ca.kw_only,
            ca.eq,
            ca.eq_key,
            ca.order,
            ca.order_key,
            ca.on_setattr,
            ca.alias,
        )

    # Don't use attrs.evolve since fields(Attribute) doesn't work
    def evolve(self, **changes):
        """
        Copy *self* and apply *changes*.

        This works similarly to `attrs.evolve` but that function does not work
        with :class:`attrs.Attribute`.

        It is mainly meant to be used for `transform-fields`.

        .. versionadded:: 20.3.0
        """
        new = copy.copy(self)

        new._setattrs(changes.items())

        return new

    # Don't use _add_pickle since fields(Attribute) doesn't work
    def __getstate__(self):
        """
        Play nice with pickle.
        """
        return tuple(
            getattr(self, name) if name != "metadata" else dict(self.metadata)
            for name in self.__slots__
        )

    def __setstate__(self, state):
        """
        Play nice with pickle.
        """
        self._setattrs(zip(self.__slots__, state))

    def _setattrs(self, name_values_pairs):
        bound_setattr = _OBJ_SETATTR.__get__(self)
        for name, value in name_values_pairs:
            if name != "metadata":
                bound_setattr(name, value)
            else:
                bound_setattr(
                    name,
                    (
                        types.MappingProxyType(dict(value))
                        if value
                        else _EMPTY_METADATA_SINGLETON
                    ),
                )


_a = [
    Attribute(
        name=name,
        default=NOTHING,
        validator=None,
        repr=True,
        cmp=None,
        eq=True,
        order=False,
        hash=(name != "metadata"),
        init=True,
        inherited=False,
        alias=_default_init_alias_for(name),
    )
    for name in Attribute.__slots__
]

Attribute = _add_hash(
    _add_eq(
        _add_repr(Attribute, attrs=_a),
        attrs=[a for a in _a if a.name != "inherited"],
    ),
    attrs=[a for a in _a if a.hash and a.name != "inherited"],
)


class _CountingAttr:
    """
    Intermediate representation of attributes that uses a counter to preserve
    the order in which the attributes have been defined.

    *Internal* data structure of the attrs library.  Running into is most
    likely the result of a bug like a forgotten `@attr.s` decorator.
    """

    __slots__ = (
        "_default",
        "_validator",
        "alias",
        "converter",
        "counter",
        "eq",
        "eq_key",
        "hash",
        "init",
        "kw_only",
        "metadata",
        "on_setattr",
        "order",
        "order_key",
        "repr",
        "type",
    )
    __attrs_attrs__ = (
        *tuple(
            Attribute(
                name=name,
                alias=_default_init_alias_for(name),
                default=NOTHING,
                validator=None,
                repr=True,
                cmp=None,
                hash=True,
                init=True,
                kw_only=False,
                eq=True,
                eq_key=None,
                order=False,
                order_key=None,
                inherited=False,
                on_setattr=None,
            )
            for name in (
                "counter",
                "_default",
                "repr",
                "eq",
                "order",
                "hash",
                "init",
                "on_setattr",
                "alias",
            )
        ),
        Attribute(
            name="metadata",
            alias="metadata",
            default=None,
            validator=None,
            repr=True,
            cmp=None,
            hash=False,
            init=True,
            kw_only=False,
            eq=True,
            eq_key=None,
            order=False,
            order_key=None,
            inherited=False,
            on_setattr=None,
        ),
    )
    cls_counter = 0

    def __init__(
        self,
        default,
        validator,
        repr,
        cmp,
        hash,
        init,
        converter,
        metadata,
        type,
        kw_only,
        eq,
        eq_key,
        order,
        order_key,
        on_setattr,
        alias,
    ):
        _CountingAttr.cls_counter += 1
        self.counter = _CountingAttr.cls_counter
        self._default = default
        self._validator = validator
        self.converter = converter
        self.repr = repr
        self.eq = eq
        self.eq_key = eq_key
        self.order = order
        self.order_key = order_key
        self.hash = hash
        self.init = init
        self.metadata = metadata
        self.type = type
        self.kw_only = kw_only
        self.on_setattr = on_setattr
        self.alias = alias

    def validator(self, meth):
        """
        Decorator that adds *meth* to the list of validators.

        Returns *meth* unchanged.

        .. versionadded:: 17.1.0
        """
        if self._validator is None:
            self._validator = meth
        else:
            self._validator = and_(self._validator, meth)
        return meth

    def default(self, meth):
        """
        Decorator that allows to set the default for an attribute.

        Returns *meth* unchanged.

        Raises:
            DefaultAlreadySetError: If default has been set before.

        .. versionadded:: 17.1.0
        """
        if self._default is not NOTHING:
            raise DefaultAlreadySetError

        self._default = Factory(meth, takes_self=True)

        return meth


_CountingAttr = _add_eq(_add_repr(_CountingAttr))


class Factory:
    """
    Stores a factory callable.

    If passed as the default value to `attrs.field`, the factory is used to
    generate a new value.

    Args:
        factory (typing.Callable):
            A callable that takes either none or exactly one mandatory
            positional argument depending on *takes_self*.

        takes_self (bool):
            Pass the partially initialized instance that is being initialized
            as a positional argument.

    .. versionadded:: 17.1.0  *takes_self*
    """

    __slots__ = ("factory", "takes_self")

    def __init__(self, factory, takes_self=False):
        self.factory = factory
        self.takes_self = takes_self

    def __getstate__(self):
        """
        Play nice with pickle.
        """
        return tuple(getattr(self, name) for name in self.__slots__)

    def __setstate__(self, state):
        """
        Play nice with pickle.
        """
        for name, value in zip(self.__slots__, state):
            setattr(self, name, value)


_f = [
    Attribute(
        name=name,
        default=NOTHING,
        validator=None,
        repr=True,
        cmp=None,
        eq=True,
        order=False,
        hash=True,
        init=True,
        inherited=False,
    )
    for name in Factory.__slots__
]

Factory = _add_hash(_add_eq(_add_repr(Factory, attrs=_f), attrs=_f), attrs=_f)


class Converter:
    """
    Stores a converter callable.

    Allows for the wrapped converter to take additional arguments. The
    arguments are passed in the order they are documented.

    Args:
        converter (Callable): A callable that converts the passed value.

        takes_self (bool):
            Pass the partially initialized instance that is being initialized
            as a positional argument. (default: `False`)

        takes_field (bool):
            Pass the field definition (an :class:`Attribute`) into the
            converter as a positional argument. (default: `False`)

    .. versionadded:: 24.1.0
    """

    __slots__ = (
        "__call__",
        "_first_param_type",
        "_global_name",
        "converter",
        "takes_field",
        "takes_self",
    )

    def __init__(self, converter, *, takes_self=False, takes_field=False):
        self.converter = converter
        self.takes_self = takes_self
        self.takes_field = takes_field

        ex = _AnnotationExtractor(converter)
        self._first_param_type = ex.get_first_param_type()

        if not (self.takes_self or self.takes_field):
            self.__call__ = lambda value, _, __: self.converter(value)
        elif self.takes_self and not self.takes_field:
            self.__call__ = lambda value, instance, __: self.converter(
                value, instance
            )
        elif not self.takes_self and self.takes_field:
            self.__call__ = lambda value, __, field: self.converter(
                value, field
            )
        else:
            self.__call__ = lambda value, instance, field: self.converter(
                value, instance, field
            )

        rt = ex.get_return_type()
        if rt is not None:
            self.__call__.__annotations__["return"] = rt

    @staticmethod
    def _get_global_name(attr_name: str) -> str:
        """
        Return the name that a converter for an attribute name *attr_name*
        would have.
        """
        return f"__attr_converter_{attr_name}"

    def _fmt_converter_call(self, attr_name: str, value_var: str) -> str:
        """
        Return a string that calls the converter for an attribute name
        *attr_name* and the value in variable named *value_var* according to
        `self.takes_self` and `self.takes_field`.
        """
        if not (self.takes_self or self.takes_field):
            return f"{self._get_global_name(attr_name)}({value_var})"

        if self.takes_self and self.takes_field:
            return f"{self._get_global_name(attr_name)}({value_var}, self, attr_dict['{attr_name}'])"

        if self.takes_self:
            return f"{self._get_global_name(attr_name)}({value_var}, self)"

        return f"{self._get_global_name(attr_name)}({value_var}, attr_dict['{attr_name}'])"

    def __getstate__(self):
        """
        Return a dict containing only converter and takes_self -- the rest gets
        computed when loading.
        """
        return {
            "converter": self.converter,
            "takes_self": self.takes_self,
            "takes_field": self.takes_field,
        }

    def __setstate__(self, state):
        """
        Load instance from state.
        """
        self.__init__(**state)


_f = [
    Attribute(
        name=name,
        default=NOTHING,
        validator=None,
        repr=True,
        cmp=None,
        eq=True,
        order=False,
        hash=True,
        init=True,
        inherited=False,
    )
    for name in ("converter", "takes_self", "takes_field")
]

Converter = _add_hash(
    _add_eq(_add_repr(Converter, attrs=_f), attrs=_f), attrs=_f
)


def make_class(
    name, attrs, bases=(object,), class_body=None, **attributes_arguments
):
    r"""
    A quick way to create a new class called *name* with *attrs*.

    .. note::

        ``make_class()`` is a thin wrapper around `attr.s`, not `attrs.define`
        which means that it doesn't come with some of the improved defaults.

        For example, if you want the same ``on_setattr`` behavior as in
        `attrs.define`, you have to pass the hooks yourself: ``make_class(...,
        on_setattr=setters.pipe(setters.convert, setters.validate)``

    .. warning::

        It is *your* duty to ensure that the class name and the attribute names
        are valid identifiers. ``make_class()`` will *not* validate them for
        you.

    Args:
        name (str): The name for the new class.

        attrs (list | dict):
            A list of names or a dictionary of mappings of names to `attr.ib`\
            s / `attrs.field`\ s.

            The order is deduced from the order of the names or attributes
            inside *attrs*.  Otherwise the order of the definition of the
            attributes is used.

        bases (tuple[type, ...]): Classes that the new class will subclass.

        class_body (dict):
            An optional dictionary of class attributes for the new class.

        attributes_arguments: Passed unmodified to `attr.s`.

    Returns:
        type: A new class with *attrs*.

    .. versionadded:: 17.1.0 *bases*
    .. versionchanged:: 18.1.0 If *attrs* is ordered, the order is retained.
    .. versionchanged:: 23.2.0 *class_body*
    .. versionchanged:: 25.2.0 Class names can now be unicode.
    """
    # Class identifiers are converted into the normal form NFKC while parsing
    name = unicodedata.normalize("NFKC", name)

    if isinstance(attrs, dict):
        cls_dict = attrs
    elif isinstance(attrs, (list, tuple)):
        cls_dict = {a: attrib() for a in attrs}
    else:
        msg = "attrs argument must be a dict or a list."
        raise TypeError(msg)

    pre_init = cls_dict.pop("__attrs_pre_init__", None)
    post_init = cls_dict.pop("__attrs_post_init__", None)
    user_init = cls_dict.pop("__init__", None)

    body = {}
    if class_body is not None:
        body.update(class_body)
    if pre_init is not None:
        body["__attrs_pre_init__"] = pre_init
    if post_init is not None:
        body["__attrs_post_init__"] = post_init
    if user_init is not None:
        body["__init__"] = user_init

    type_ = types.new_class(name, bases, {}, lambda ns: ns.update(body))

    # For pickling to work, the __module__ variable needs to be set to the
    # frame where the class is created.  Bypass this step in environments where
    # sys._getframe is not defined (Jython for example) or sys._getframe is not
    # defined for arguments greater than 0 (IronPython).
    with contextlib.suppress(AttributeError, ValueError):
        type_.__module__ = sys._getframe(1).f_globals.get(
            "__name__", "__main__"
        )

    # We do it here for proper warnings with meaningful stacklevel.
    cmp = attributes_arguments.pop("cmp", None)
    (
        attributes_arguments["eq"],
        attributes_arguments["order"],
    ) = _determine_attrs_eq_order(
        cmp,
        attributes_arguments.get("eq"),
        attributes_arguments.get("order"),
        True,
    )

    cls = _attrs(these=cls_dict, **attributes_arguments)(type_)
    # Only add type annotations now or "_attrs()" will complain:
    cls.__annotations__ = {
        k: v.type for k, v in cls_dict.items() if v.type is not None
    }
    return cls


# These are required by within this module so we define them here and merely
# import into .validators / .converters.


@attrs(slots=True, unsafe_hash=True)
class _AndValidator:
    """
    Compose many validators to a single one.
    """

    _validators = attrib()

    def __call__(self, inst, attr, value):
        for v in self._validators:
            v(inst, attr, value)


def and_(*validators):
    """
    A validator that composes multiple validators into one.

    When called on a value, it runs all wrapped validators.

    Args:
        validators (~collections.abc.Iterable[typing.Callable]):
            Arbitrary number of validators.

    .. versionadded:: 17.1.0
    """
    vals = []
    for validator in validators:
        vals.extend(
            validator._validators
            if isinstance(validator, _AndValidator)
            else [validator]
        )

    return _AndValidator(tuple(vals))


def pipe(*converters):
    """
    A converter that composes multiple converters into one.

    When called on a value, it runs all wrapped converters, returning the
    *last* value.

    Type annotations will be inferred from the wrapped converters', if they
    have any.

        converters (~collections.abc.Iterable[typing.Callable]):
            Arbitrary number of converters.

    .. versionadded:: 20.1.0
    """

    return_instance = any(isinstance(c, Converter) for c in converters)

    if return_instance:

        def pipe_converter(val, inst, field):
            for c in converters:
                val = (
                    c(val, inst, field) if isinstance(c, Converter) else c(val)
                )

            return val

    else:

        def pipe_converter(val):
            for c in converters:
                val = c(val)

            return val

    if not converters:
        # If the converter list is empty, pipe_converter is the identity.
        A = TypeVar("A")
        pipe_converter.__annotations__.update({"val": A, "return": A})
    else:
        # Get parameter type from first converter.
        t = _AnnotationExtractor(converters[0]).get_first_param_type()
        if t:
            pipe_converter.__annotations__["val"] = t

        last = converters[-1]
        if not PY_3_11_PLUS and isinstance(last, Converter):
            last = last.__call__

        # Get return type from last converter.
        rt = _AnnotationExtractor(last).get_return_type()
        if rt:
            pipe_converter.__annotations__["return"] = rt

    if return_instance:
        return Converter(pipe_converter, takes_self=True, takes_field=True)
    return pipe_converter


# SPDX-License-Identifier: MIT


import functools
import types

from ._make import __ne__


_operation_names = {"eq": "==", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}


def cmp_using(
    eq=None,
    lt=None,
    le=None,
    gt=None,
    ge=None,
    require_same_type=True,
    class_name="Comparable",
):
    """
    Create a class that can be passed into `attrs.field`'s ``eq``, ``order``,
    and ``cmp`` arguments to customize field comparison.

    The resulting class will have a full set of ordering methods if at least
    one of ``{lt, le, gt, ge}`` and ``eq``  are provided.

    Args:
        eq (typing.Callable | None):
            Callable used to evaluate equality of two objects.

        lt (typing.Callable | None):
            Callable used to evaluate whether one object is less than another
            object.

        le (typing.Callable | None):
            Callable used to evaluate whether one object is less than or equal
            to another object.

        gt (typing.Callable | None):
            Callable used to evaluate whether one object is greater than
            another object.

        ge (typing.Callable | None):
            Callable used to evaluate whether one object is greater than or
            equal to another object.

        require_same_type (bool):
            When `True`, equality and ordering methods will return
            `NotImplemented` if objects are not of the same type.

        class_name (str | None): Name of class. Defaults to "Comparable".

    See `comparison` for more details.

    .. versionadded:: 21.1.0
    """

    body = {
        "__slots__": ["value"],
        "__init__": _make_init(),
        "_requirements": [],
        "_is_comparable_to": _is_comparable_to,
    }

    # Add operations.
    num_order_functions = 0
    has_eq_function = False

    if eq is not None:
        has_eq_function = True
        body["__eq__"] = _make_operator("eq", eq)
        body["__ne__"] = __ne__

    if lt is not None:
        num_order_functions += 1
        body["__lt__"] = _make_operator("lt", lt)

    if le is not None:
        num_order_functions += 1
        body["__le__"] = _make_operator("le", le)

    if gt is not None:
        num_order_functions += 1
        body["__gt__"] = _make_operator("gt", gt)

    if ge is not None:
        num_order_functions += 1
        body["__ge__"] = _make_operator("ge", ge)

    type_ = types.new_class(
        class_name, (object,), {}, lambda ns: ns.update(body)
    )

    # Add same type requirement.
    if require_same_type:
        type_._requirements.append(_check_same_type)

    # Add total ordering if at least one operation was defined.
    if 0 < num_order_functions < 4:
        if not has_eq_function:
            # functools.total_ordering requires __eq__ to be defined,
            # so raise early error here to keep a nice stack.
            msg = "eq must be define is order to complete ordering from lt, le, gt, ge."
            raise ValueError(msg)
        type_ = functools.total_ordering(type_)

    return type_


def _make_init():
    """
    Create __init__ method.
    """

    def __init__(self, value):
        """
        Initialize object with *value*.
        """
        self.value = value

    return __init__


def _make_operator(name, func):
    """
    Create operator method.
    """

    def method(self, other):
        if not self._is_comparable_to(other):
            return NotImplemented

        result = func(self.value, other.value)
        if result is NotImplemented:
            return NotImplemented

        return result

    method.__name__ = f"__{name}__"
    method.__doc__ = (
        f"Return a {_operation_names[name]} b.  Computed by attrs."
    )

    return method


def _is_comparable_to(self, other):
    """
    Check whether `other` is comparable to `self`.
    """
    return all(func(self, other) for func in self._requirements)


def _check_same_type(self, other):
    """
    Return True if *self* and *other* are of the same type, False otherwise.
    """
    return other.value.__class__ is self.value.__class__


# SPDX-License-Identifier: MIT

"""
Classes Without Boilerplate
"""

from functools import partial
from typing import Callable, Literal, Protocol

from . import converters, exceptions, filters, setters, validators
from ._cmp import cmp_using
from ._config import get_run_validators, set_run_validators
from ._funcs import asdict, assoc, astuple, has, resolve_types
from ._make import (
    NOTHING,
    Attribute,
    Converter,
    Factory,
    _Nothing,
    attrib,
    attrs,
    evolve,
    fields,
    fields_dict,
    make_class,
    validate,
)
from ._next_gen import define, field, frozen, mutable
from ._version_info import VersionInfo


s = attributes = attrs
ib = attr = attrib
dataclass = partial(attrs, auto_attribs=True)  # happy Easter ;)


class AttrsInstance(Protocol):
    pass


NothingType = Literal[_Nothing.NOTHING]

__all__ = [
    "NOTHING",
    "Attribute",
    "AttrsInstance",
    "Converter",
    "Factory",
    "NothingType",
    "asdict",
    "assoc",
    "astuple",
    "attr",
    "attrib",
    "attributes",
    "attrs",
    "cmp_using",
    "converters",
    "define",
    "evolve",
    "exceptions",
    "field",
    "fields",
    "fields_dict",
    "filters",
    "frozen",
    "get_run_validators",
    "has",
    "ib",
    "make_class",
    "mutable",
    "resolve_types",
    "s",
    "set_run_validators",
    "setters",
    "validate",
    "validators",
]


def _make_getattr(mod_name: str) -> Callable:
    """
    Create a metadata proxy for packaging information that uses *mod_name* in
    its warnings and errors.
    """

    def __getattr__(name: str) -> str:
        if name not in ("__version__", "__version_info__"):
            msg = f"module {mod_name} has no attribute {name}"
            raise AttributeError(msg)

        from importlib.metadata import metadata

        meta = metadata("attrs")

        if name == "__version_info__":
            return VersionInfo._from_version_string(meta["version"])

        return meta["version"]

    return __getattr__


__getattr__ = _make_getattr(__name__)


# SPDX-License-Identifier: MIT

"""
These are keyword-only APIs that call `attr.s` and `attr.ib` with different
default values.
"""

from functools import partial

from . import setters
from ._funcs import asdict as _asdict
from ._funcs import astuple as _astuple
from ._make import (
    _DEFAULT_ON_SETATTR,
    NOTHING,
    _frozen_setattrs,
    attrib,
    attrs,
)
from .exceptions import UnannotatedAttributeError


def define(
    maybe_cls=None,
    *,
    these=None,
    repr=None,
    unsafe_hash=None,
    hash=None,
    init=None,
    slots=True,
    frozen=False,
    weakref_slot=True,
    str=False,
    auto_attribs=None,
    kw_only=False,
    cache_hash=False,
    auto_exc=True,
    eq=None,
    order=False,
    auto_detect=True,
    getstate_setstate=None,
    on_setattr=None,
    field_transformer=None,
    match_args=True,
):
    r"""
    A class decorator that adds :term:`dunder methods` according to
    :term:`fields <field>` specified using :doc:`type annotations <types>`,
    `field()` calls, or the *these* argument.

    Since *attrs* patches or replaces an existing class, you cannot use
    `object.__init_subclass__` with *attrs* classes, because it runs too early.
    As a replacement, you can define ``__attrs_init_subclass__`` on your class.
    It will be called by *attrs* classes that subclass it after they're
    created. See also :ref:`init-subclass`.

    Args:
        slots (bool):
            Create a :term:`slotted class <slotted classes>` that's more
            memory-efficient. Slotted classes are generally superior to the
            default dict classes, but have some gotchas you should know about,
            so we encourage you to read the :term:`glossary entry <slotted
            classes>`.

        auto_detect (bool):
            Instead of setting the *init*, *repr*, *eq*, and *hash* arguments
            explicitly, assume they are set to True **unless any** of the
            involved methods for one of the arguments is implemented in the
            *current* class (meaning, it is *not* inherited from some base
            class).

            So, for example by implementing ``__eq__`` on a class yourself,
            *attrs* will deduce ``eq=False`` and will create *neither*
            ``__eq__`` *nor* ``__ne__`` (but Python classes come with a
            sensible ``__ne__`` by default, so it *should* be enough to only
            implement ``__eq__`` in most cases).

            Passing True or False` to *init*, *repr*, *eq*, or *hash*
            overrides whatever *auto_detect* would determine.

        auto_exc (bool):
            If the class subclasses `BaseException` (which implicitly includes
            any subclass of any exception), the following happens to behave
            like a well-behaved Python exception class:

            - the values for *eq*, *order*, and *hash* are ignored and the
              instances compare and hash by the instance's ids [#]_ ,
            - all attributes that are either passed into ``__init__`` or have a
              default value are additionally available as a tuple in the
              ``args`` attribute,
            - the value of *str* is ignored leaving ``__str__`` to base
              classes.

            .. [#]
               Note that *attrs* will *not* remove existing implementations of
               ``__hash__`` or the equality methods. It just won't add own
               ones.

        on_setattr (~typing.Callable | list[~typing.Callable] | None | ~typing.Literal[attrs.setters.NO_OP]):
            A callable that is run whenever the user attempts to set an
            attribute (either by assignment like ``i.x = 42`` or by using
            `setattr` like ``setattr(i, "x", 42)``). It receives the same
            arguments as validators: the instance, the attribute that is being
            modified, and the new value.

            If no exception is raised, the attribute is set to the return value
            of the callable.

            If a list of callables is passed, they're automatically wrapped in
            an `attrs.setters.pipe`.

            If left None, the default behavior is to run converters and
            validators whenever an attribute is set.

        init (bool):
            Create a ``__init__`` method that initializes the *attrs*
            attributes. Leading underscores are stripped for the argument name,
            unless an alias is set on the attribute.

            .. seealso::
                `init` shows advanced ways to customize the generated
                ``__init__`` method, including executing code before and after.

        repr(bool):
            Create a ``__repr__`` method with a human readable representation
            of *attrs* attributes.

        str (bool):
            Create a ``__str__`` method that is identical to ``__repr__``. This
            is usually not necessary except for `Exception`\ s.

        eq (bool | None):
            If True or None (default), add ``__eq__`` and ``__ne__`` methods
            that check two instances for equality.

            .. seealso::
                `comparison` describes how to customize the comparison behavior
                going as far comparing NumPy arrays.

        order (bool | None):
            If True, add ``__lt__``, ``__le__``, ``__gt__``, and ``__ge__``
            methods that behave like *eq* above and allow instances to be
            ordered.

            They compare the instances as if they were tuples of their *attrs*
            attributes if and only if the types of both classes are
            *identical*.

            If `None` mirror value of *eq*.

            .. seealso:: `comparison`

        unsafe_hash (bool | None):
            If None (default), the ``__hash__`` method is generated according
            how *eq* and *frozen* are set.

            1. If *both* are True, *attrs* will generate a ``__hash__`` for
               you.
            2. If *eq* is True and *frozen* is False, ``__hash__`` will be set
               to None, marking it unhashable (which it is).
            3. If *eq* is False, ``__hash__`` will be left untouched meaning
               the ``__hash__`` method of the base class will be used. If the
               base class is `object`, this means it will fall back to id-based
               hashing.

            Although not recommended, you can decide for yourself and force
            *attrs* to create one (for example, if the class is immutable even
            though you didn't freeze it programmatically) by passing True or
            not.  Both of these cases are rather special and should be used
            carefully.

            .. seealso::

                - Our documentation on `hashing`,
                - Python's documentation on `object.__hash__`,
                - and the `GitHub issue that led to the default \ behavior
                  <https://github.com/python-attrs/attrs/issues/136>`_ for more
                  details.

        hash (bool | None):
            Deprecated alias for *unsafe_hash*. *unsafe_hash* takes precedence.

        cache_hash (bool):
            Ensure that the object's hash code is computed only once and stored
            on the object.  If this is set to True, hashing must be either
            explicitly or implicitly enabled for this class.  If the hash code
            is cached, avoid any reassignments of fields involved in hash code
            computation or mutations of the objects those fields point to after
            object creation.  If such changes occur, the behavior of the
            object's hash code is undefined.

        frozen (bool):
            Make instances immutable after initialization.  If someone attempts
            to modify a frozen instance, `attrs.exceptions.FrozenInstanceError`
            is raised.

            .. note::

                1. This is achieved by installing a custom ``__setattr__``
                   method on your class, so you can't implement your own.

                2. True immutability is impossible in Python.

                3. This *does* have a minor a runtime performance `impact
                   <how-frozen>` when initializing new instances.  In other
                   words: ``__init__`` is slightly slower with ``frozen=True``.

                4. If a class is frozen, you cannot modify ``self`` in
                   ``__attrs_post_init__`` or a self-written ``__init__``. You
                   can circumvent that limitation by using
                   ``object.__setattr__(self, "attribute_name", value)``.

                5. Subclasses of a frozen class are frozen too.

        kw_only (bool):
            Make all attributes keyword-only in the generated ``__init__`` (if
            *init* is False, this parameter is ignored).

        weakref_slot (bool):
            Make instances weak-referenceable.  This has no effect unless
            *slots* is True.

        field_transformer (~typing.Callable | None):
            A function that is called with the original class object and all
            fields right before *attrs* finalizes the class.  You can use this,
            for example, to automatically add converters or validators to
            fields based on their types.

            .. seealso:: `transform-fields`

        match_args (bool):
            If True (default), set ``__match_args__`` on the class to support
            :pep:`634` (*Structural Pattern Matching*). It is a tuple of all
            non-keyword-only ``__init__`` parameter names on Python 3.10 and
            later. Ignored on older Python versions.

        collect_by_mro (bool):
            If True, *attrs* collects attributes from base classes correctly
            according to the `method resolution order
            <https://docs.python.org/3/howto/mro.html>`_. If False, *attrs*
            will mimic the (wrong) behavior of `dataclasses` and :pep:`681`.

            See also `issue #428
            <https://github.com/python-attrs/attrs/issues/428>`_.

        getstate_setstate (bool | None):
            .. note::

                This is usually only interesting for slotted classes and you
                should probably just set *auto_detect* to True.

            If True, ``__getstate__`` and ``__setstate__`` are generated and
            attached to the class. This is necessary for slotted classes to be
            pickleable. If left None, it's True by default for slotted classes
            and False for dict classes.

            If *auto_detect* is True, and *getstate_setstate* is left None, and
            **either** ``__getstate__`` or ``__setstate__`` is detected
            directly on the class (meaning: not inherited), it is set to False
            (this is usually what you want).

        auto_attribs (bool | None):
            If True, look at type annotations to determine which attributes to
            use, like `dataclasses`. If False, it will only look for explicit
            :func:`field` class attributes, like classic *attrs*.

            If left None, it will guess:

            1. If any attributes are annotated and no unannotated
               `attrs.field`\ s are found, it assumes *auto_attribs=True*.
            2. Otherwise it assumes *auto_attribs=False* and tries to collect
               `attrs.field`\ s.

            If *attrs* decides to look at type annotations, **all** fields
            **must** be annotated. If *attrs* encounters a field that is set to
            a :func:`field` / `attr.ib` but lacks a type annotation, an
            `attrs.exceptions.UnannotatedAttributeError` is raised.  Use
            ``field_name: typing.Any = field(...)`` if you don't want to set a
            type.

            .. warning::

                For features that use the attribute name to create decorators
                (for example, :ref:`validators <validators>`), you still *must*
                assign :func:`field` / `attr.ib` to them. Otherwise Python will
                either not find the name or try to use the default value to
                call, for example, ``validator`` on it.

            Attributes annotated as `typing.ClassVar`, and attributes that are
            neither annotated nor set to an `field()` are **ignored**.

        these (dict[str, object]):
            A dictionary of name to the (private) return value of `field()`
            mappings. This is useful to avoid the definition of your attributes
            within the class body because you can't (for example, if you want
            to add ``__repr__`` methods to Django models) or don't want to.

            If *these* is not `None`, *attrs* will *not* search the class body
            for attributes and will *not* remove any attributes from it.

            The order is deduced from the order of the attributes inside
            *these*.

            Arguably, this is a rather obscure feature.

    .. versionadded:: 20.1.0
    .. versionchanged:: 21.3.0 Converters are also run ``on_setattr``.
    .. versionadded:: 22.2.0
       *unsafe_hash* as an alias for *hash* (for :pep:`681` compliance).
    .. versionchanged:: 24.1.0
       Instances are not compared as tuples of attributes anymore, but using a
       big ``and`` condition. This is faster and has more correct behavior for
       uncomparable values like `math.nan`.
    .. versionadded:: 24.1.0
       If a class has an *inherited* classmethod called
       ``__attrs_init_subclass__``, it is executed after the class is created.
    .. deprecated:: 24.1.0 *hash* is deprecated in favor of *unsafe_hash*.
    .. versionadded:: 24.3.0
       Unless already present, a ``__replace__`` method is automatically
       created for `copy.replace` (Python 3.13+ only).

    .. note::

        The main differences to the classic `attr.s` are:

        - Automatically detect whether or not *auto_attribs* should be `True`
          (c.f. *auto_attribs* parameter).
        - Converters and validators run when attributes are set by default --
          if *frozen* is `False`.
        - *slots=True*

          Usually, this has only upsides and few visible effects in everyday
          programming. But it *can* lead to some surprising behaviors, so
          please make sure to read :term:`slotted classes`.

        - *auto_exc=True*
        - *auto_detect=True*
        - *order=False*
        - Some options that were only relevant on Python 2 or were kept around
          for backwards-compatibility have been removed.

    """

    def do_it(cls, auto_attribs):
        return attrs(
            maybe_cls=cls,
            these=these,
            repr=repr,
            hash=hash,
            unsafe_hash=unsafe_hash,
            init=init,
            slots=slots,
            frozen=frozen,
            weakref_slot=weakref_slot,
            str=str,
            auto_attribs=auto_attribs,
            kw_only=kw_only,
            cache_hash=cache_hash,
            auto_exc=auto_exc,
            eq=eq,
            order=order,
            auto_detect=auto_detect,
            collect_by_mro=True,
            getstate_setstate=getstate_setstate,
            on_setattr=on_setattr,
            field_transformer=field_transformer,
            match_args=match_args,
        )

    def wrap(cls):
        """
        Making this a wrapper ensures this code runs during class creation.

        We also ensure that frozen-ness of classes is inherited.
        """
        nonlocal frozen, on_setattr

        had_on_setattr = on_setattr not in (None, setters.NO_OP)

        # By default, mutable classes convert & validate on setattr.
        if frozen is False and on_setattr is None:
            on_setattr = _DEFAULT_ON_SETATTR

        # However, if we subclass a frozen class, we inherit the immutability
        # and disable on_setattr.
        for base_cls in cls.__bases__:
            if base_cls.__setattr__ is _frozen_setattrs:
                if had_on_setattr:
                    msg = "Frozen classes can't use on_setattr (frozen-ness was inherited)."
                    raise ValueError(msg)

                on_setattr = setters.NO_OP
                break

        if auto_attribs is not None:
            return do_it(cls, auto_attribs)

        try:
            return do_it(cls, True)
        except UnannotatedAttributeError:
            return do_it(cls, False)

    # maybe_cls's type depends on the usage of the decorator.  It's a class
    # if it's used as `@attrs` but `None` if used as `@attrs()`.
    if maybe_cls is None:
        return wrap

    return wrap(maybe_cls)


mutable = define
frozen = partial(define, frozen=True, on_setattr=None)


def field(
    *,
    default=NOTHING,
    validator=None,
    repr=True,
    hash=None,
    init=True,
    metadata=None,
    type=None,
    converter=None,
    factory=None,
    kw_only=False,
    eq=None,
    order=None,
    on_setattr=None,
    alias=None,
):
    """
    Create a new :term:`field` / :term:`attribute` on a class.

    ..  warning::

        Does **nothing** unless the class is also decorated with
        `attrs.define` (or similar)!

    Args:
        default:
            A value that is used if an *attrs*-generated ``__init__`` is used
            and no value is passed while instantiating or the attribute is
            excluded using ``init=False``.

            If the value is an instance of `attrs.Factory`, its callable will
            be used to construct a new value (useful for mutable data types
            like lists or dicts).

            If a default is not set (or set manually to `attrs.NOTHING`), a
            value *must* be supplied when instantiating; otherwise a
            `TypeError` will be raised.

            .. seealso:: `defaults`

        factory (~typing.Callable):
            Syntactic sugar for ``default=attr.Factory(factory)``.

        validator (~typing.Callable | list[~typing.Callable]):
            Callable that is called by *attrs*-generated ``__init__`` methods
            after the instance has been initialized.  They receive the
            initialized instance, the :func:`~attrs.Attribute`, and the passed
            value.

            The return value is *not* inspected so the validator has to throw
            an exception itself.

            If a `list` is passed, its items are treated as validators and must
            all pass.

            Validators can be globally disabled and re-enabled using
            `attrs.validators.get_disabled` / `attrs.validators.set_disabled`.

            The validator can also be set using decorator notation as shown
            below.

            .. seealso:: :ref:`validators`

        repr (bool | ~typing.Callable):
            Include this attribute in the generated ``__repr__`` method. If
            True, include the attribute; if False, omit it. By default, the
            built-in ``repr()`` function is used. To override how the attribute
            value is formatted, pass a ``callable`` that takes a single value
            and returns a string. Note that the resulting string is used as-is,
            which means it will be used directly *instead* of calling
            ``repr()`` (the default).

        eq (bool | ~typing.Callable):
            If True (default), include this attribute in the generated
            ``__eq__`` and ``__ne__`` methods that check two instances for
            equality. To override how the attribute value is compared, pass a
            callable that takes a single value and returns the value to be
            compared.

            .. seealso:: `comparison`

        order (bool | ~typing.Callable):
            If True (default), include this attributes in the generated
            ``__lt__``, ``__le__``, ``__gt__`` and ``__ge__`` methods. To
            override how the attribute value is ordered, pass a callable that
            takes a single value and returns the value to be ordered.

            .. seealso:: `comparison`

        hash (bool | None):
            Include this attribute in the generated ``__hash__`` method.  If
            None (default), mirror *eq*'s value.  This is the correct behavior
            according the Python spec.  Setting this value to anything else
            than None is *discouraged*.

            .. seealso:: `hashing`

        init (bool):
            Include this attribute in the generated ``__init__`` method.

            It is possible to set this to False and set a default value. In
            that case this attributed is unconditionally initialized with the
            specified default value or factory.

            .. seealso:: `init`

        converter (typing.Callable | Converter):
            A callable that is called by *attrs*-generated ``__init__`` methods
            to convert attribute's value to the desired format.

            If a vanilla callable is passed, it is given the passed-in value as
            the only positional argument. It is possible to receive additional
            arguments by wrapping the callable in a `Converter`.

            Either way, the returned value will be used as the new value of the
            attribute.  The value is converted before being passed to the
            validator, if any.

            .. seealso:: :ref:`converters`

        metadata (dict | None):
            An arbitrary mapping, to be used by third-party code.

            .. seealso:: `extending-metadata`.

        type (type):
            The type of the attribute. Nowadays, the preferred method to
            specify the type is using a variable annotation (see :pep:`526`).
            This argument is provided for backwards-compatibility and for usage
            with `make_class`. Regardless of the approach used, the type will
            be stored on ``Attribute.type``.

            Please note that *attrs* doesn't do anything with this metadata by
            itself. You can use it as part of your own code or for `static type
            checking <types>`.

        kw_only (bool):
            Make this attribute keyword-only in the generated ``__init__`` (if
            ``init`` is False, this parameter is ignored).

        on_setattr (~typing.Callable | list[~typing.Callable] | None | ~typing.Literal[attrs.setters.NO_OP]):
            Allows to overwrite the *on_setattr* setting from `attr.s`. If left
            None, the *on_setattr* value from `attr.s` is used. Set to
            `attrs.setters.NO_OP` to run **no** `setattr` hooks for this
            attribute -- regardless of the setting in `define()`.

        alias (str | None):
            Override this attribute's parameter name in the generated
            ``__init__`` method. If left None, default to ``name`` stripped
            of leading underscores. See `private-attributes`.

    .. versionadded:: 20.1.0
    .. versionchanged:: 21.1.0
       *eq*, *order*, and *cmp* also accept a custom callable
    .. versionadded:: 22.2.0 *alias*
    .. versionadded:: 23.1.0
       The *type* parameter has been re-added; mostly for `attrs.make_class`.
       Please note that type checkers ignore this metadata.

    .. seealso::

       `attr.ib`
    """
    return attrib(
        default=default,
        validator=validator,
        repr=repr,
        hash=hash,
        init=init,
        metadata=metadata,
        type=type,
        converter=converter,
        factory=factory,
        kw_only=kw_only,
        eq=eq,
        order=order,
        on_setattr=on_setattr,
        alias=alias,
    )


def asdict(inst, *, recurse=True, filter=None, value_serializer=None):
    """
    Same as `attr.asdict`, except that collections types are always retained
    and dict is always used as *dict_factory*.

    .. versionadded:: 21.3.0
    """
    return _asdict(
        inst=inst,
        recurse=recurse,
        filter=filter,
        value_serializer=value_serializer,
        retain_collection_types=True,
    )


def astuple(inst, *, recurse=True, filter=None):
    """
    Same as `attr.astuple`, except that collections types are always retained
    and `tuple` is always used as the *tuple_factory*.

    .. versionadded:: 21.3.0
    """
    return _astuple(
        inst=inst, recurse=recurse, filter=filter, retain_collection_types=True
    )


# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import ClassVar


class FrozenError(AttributeError):
    """
    A frozen/immutable instance or attribute have been attempted to be
    modified.

    It mirrors the behavior of ``namedtuples`` by using the same error message
    and subclassing `AttributeError`.

    .. versionadded:: 20.1.0
    """

    msg = "can't set attribute"
    args: ClassVar[tuple[str]] = [msg]


class FrozenInstanceError(FrozenError):
    """
    A frozen instance has been attempted to be modified.

    .. versionadded:: 16.1.0
    """


class FrozenAttributeError(FrozenError):
    """
    A frozen attribute has been attempted to be modified.

    .. versionadded:: 20.1.0
    """


class AttrsAttributeNotFoundError(ValueError):
    """
    An *attrs* function couldn't find an attribute that the user asked for.

    .. versionadded:: 16.2.0
    """


class NotAnAttrsClassError(ValueError):
    """
    A non-*attrs* class has been passed into an *attrs* function.

    .. versionadded:: 16.2.0
    """


class DefaultAlreadySetError(RuntimeError):
    """
    A default has been set when defining the field and is attempted to be reset
    using the decorator.

    .. versionadded:: 17.1.0
    """


class UnannotatedAttributeError(RuntimeError):
    """
    A class with ``auto_attribs=True`` has a field without a type annotation.

    .. versionadded:: 17.3.0
    """


class PythonTooOldError(RuntimeError):
    """
    It was attempted to use an *attrs* feature that requires a newer Python
    version.

    .. versionadded:: 18.2.0
    """


class NotCallableError(TypeError):
    """
    A field requiring a callable has been set with a value that is not
    callable.

    .. versionadded:: 19.2.0
    """

    def __init__(self, msg, value):
        super(TypeError, self).__init__(msg, value)
        self.msg = msg
        self.value = value

    def __str__(self):
        return str(self.msg)


# SPDX-License-Identifier: MIT


from functools import total_ordering

from ._funcs import astuple
from ._make import attrib, attrs


@total_ordering
@attrs(eq=False, order=False, slots=True, frozen=True)
class VersionInfo:
    """
    A version object that can be compared to tuple of length 1--4:

    >>> attr.VersionInfo(19, 1, 0, "final")  <= (19, 2)
    True
    >>> attr.VersionInfo(19, 1, 0, "final") < (19, 1, 1)
    True
    >>> vi = attr.VersionInfo(19, 2, 0, "final")
    >>> vi < (19, 1, 1)
    False
    >>> vi < (19,)
    False
    >>> vi == (19, 2,)
    True
    >>> vi == (19, 2, 1)
    False

    .. versionadded:: 19.2
    """

    year = attrib(type=int)
    minor = attrib(type=int)
    micro = attrib(type=int)
    releaselevel = attrib(type=str)

    @classmethod
    def _from_version_string(cls, s):
        """
        Parse *s* and return a _VersionInfo.
        """
        v = s.split(".")
        if len(v) == 3:
            v.append("final")

        return cls(
            year=int(v[0]), minor=int(v[1]), micro=int(v[2]), releaselevel=v[3]
        )

    def _ensure_tuple(self, other):
        """
        Ensure *other* is a tuple of a valid length.

        Returns a possibly transformed *other* and ourselves as a tuple of
        the same length as *other*.
        """

        if self.__class__ is other.__class__:
            other = astuple(other)

        if not isinstance(other, tuple):
            raise NotImplementedError

        if not (1 <= len(other) <= 4):
            raise NotImplementedError

        return astuple(self)[: len(other)], other

    def __eq__(self, other):
        try:
            us, them = self._ensure_tuple(other)
        except NotImplementedError:
            return NotImplemented

        return us == them

    def __lt__(self, other):
        try:
            us, them = self._ensure_tuple(other)
        except NotImplementedError:
            return NotImplemented

        # Since alphabetically "dev0" < "final" < "post1" < "post2", we don't
        # have to do anything special with releaselevel for now.
        return us < them


# SPDX-License-Identifier: MIT

"""
Commonly useful converters.
"""

import typing

from ._compat import _AnnotationExtractor
from ._make import NOTHING, Converter, Factory, pipe


__all__ = [
    "default_if_none",
    "optional",
    "pipe",
    "to_bool",
]


def optional(converter):
    """
    A converter that allows an attribute to be optional. An optional attribute
    is one which can be set to `None`.

    Type annotations will be inferred from the wrapped converter's, if it has
    any.

    Args:
        converter (typing.Callable):
            the converter that is used for non-`None` values.

    .. versionadded:: 17.1.0
    """

    if isinstance(converter, Converter):

        def optional_converter(val, inst, field):
            if val is None:
                return None
            return converter(val, inst, field)

    else:

        def optional_converter(val):
            if val is None:
                return None
            return converter(val)

    xtr = _AnnotationExtractor(converter)

    t = xtr.get_first_param_type()
    if t:
        optional_converter.__annotations__["val"] = typing.Optional[t]

    rt = xtr.get_return_type()
    if rt:
        optional_converter.__annotations__["return"] = typing.Optional[rt]

    if isinstance(converter, Converter):
        return Converter(optional_converter, takes_self=True, takes_field=True)

    return optional_converter


def default_if_none(default=NOTHING, factory=None):
    """
    A converter that allows to replace `None` values by *default* or the result
    of *factory*.

    Args:
        default:
            Value to be used if `None` is passed. Passing an instance of
            `attrs.Factory` is supported, however the ``takes_self`` option is
            *not*.

        factory (typing.Callable):
            A callable that takes no parameters whose result is used if `None`
            is passed.

    Raises:
        TypeError: If **neither** *default* or *factory* is passed.

        TypeError: If **both** *default* and *factory* are passed.

        ValueError:
            If an instance of `attrs.Factory` is passed with
            ``takes_self=True``.

    .. versionadded:: 18.2.0
    """
    if default is NOTHING and factory is None:
        msg = "Must pass either `default` or `factory`."
        raise TypeError(msg)

    if default is not NOTHING and factory is not None:
        msg = "Must pass either `default` or `factory` but not both."
        raise TypeError(msg)

    if factory is not None:
        default = Factory(factory)

    if isinstance(default, Factory):
        if default.takes_self:
            msg = "`takes_self` is not supported by default_if_none."
            raise ValueError(msg)

        def default_if_none_converter(val):
            if val is not None:
                return val

            return default.factory()

    else:

        def default_if_none_converter(val):
            if val is not None:
                return val

            return default

    return default_if_none_converter


def to_bool(val):
    """
    Convert "boolean" strings (for example, from environment variables) to real
    booleans.

    Values mapping to `True`:

    - ``True``
    - ``"true"`` / ``"t"``
    - ``"yes"`` / ``"y"``
    - ``"on"``
    - ``"1"``
    - ``1``

    Values mapping to `False`:

    - ``False``
    - ``"false"`` / ``"f"``
    - ``"no"`` / ``"n"``
    - ``"off"``
    - ``"0"``
    - ``0``

    Raises:
        ValueError: For any other value.

    .. versionadded:: 21.3.0
    """
    if isinstance(val, str):
        val = val.lower()

    if val in (True, "true", "t", "yes", "y", "on", "1", 1):
        return True
    if val in (False, "false", "f", "no", "n", "off", "0", 0):
        return False

    msg = f"Cannot convert value to bool: {val!r}"
    raise ValueError(msg)


# SPDX-License-Identifier: MIT

import inspect
import platform
import sys
import threading

from collections.abc import Mapping, Sequence  # noqa: F401
from typing import _GenericAlias


PYPY = platform.python_implementation() == "PyPy"
PY_3_9_PLUS = sys.version_info[:2] >= (3, 9)
PY_3_10_PLUS = sys.version_info[:2] >= (3, 10)
PY_3_11_PLUS = sys.version_info[:2] >= (3, 11)
PY_3_12_PLUS = sys.version_info[:2] >= (3, 12)
PY_3_13_PLUS = sys.version_info[:2] >= (3, 13)
PY_3_14_PLUS = sys.version_info[:2] >= (3, 14)


if PY_3_14_PLUS:  # pragma: no cover
    import annotationlib

    _get_annotations = annotationlib.get_annotations

else:

    def _get_annotations(cls):
        """
        Get annotations for *cls*.
        """
        return cls.__dict__.get("__annotations__", {})


class _AnnotationExtractor:
    """
    Extract type annotations from a callable, returning None whenever there
    is none.
    """

    __slots__ = ["sig"]

    def __init__(self, callable):
        try:
            self.sig = inspect.signature(callable)
        except (ValueError, TypeError):  # inspect failed
            self.sig = None

    def get_first_param_type(self):
        """
        Return the type annotation of the first argument if it's not empty.
        """
        if not self.sig:
            return None

        params = list(self.sig.parameters.values())
        if params and params[0].annotation is not inspect.Parameter.empty:
            return params[0].annotation

        return None

    def get_return_type(self):
        """
        Return the return type if it's not empty.
        """
        if (
            self.sig
            and self.sig.return_annotation is not inspect.Signature.empty
        ):
            return self.sig.return_annotation

        return None


# Thread-local global to track attrs instances which are already being repr'd.
# This is needed because there is no other (thread-safe) way to pass info
# about the instances that are already being repr'd through the call stack
# in order to ensure we don't perform infinite recursion.
#
# For instance, if an instance contains a dict which contains that instance,
# we need to know that we're already repr'ing the outside instance from within
# the dict's repr() call.
#
# This lives here rather than in _make.py so that the functions in _make.py
# don't have a direct reference to the thread-local in their globals dict.
# If they have such a reference, it breaks cloudpickle.
repr_context = threading.local()


def get_generic_base(cl):
    """If this is a generic class (A[str]), return the generic base for it."""
    if cl.__class__ is _GenericAlias:
        return cl.__origin__
    return None


# SPDX-License-Identifier: MIT

__all__ = ["get_run_validators", "set_run_validators"]

_run_validators = True


def set_run_validators(run):
    """
    Set whether or not validators are run.  By default, they are run.

    .. deprecated:: 21.3.0 It will not be removed, but it also will not be
        moved to new ``attrs`` namespace. Use `attrs.validators.set_disabled()`
        instead.
    """
    if not isinstance(run, bool):
        msg = "'run' must be bool."
        raise TypeError(msg)
    global _run_validators
    _run_validators = run


def get_run_validators():
    """
    Return whether or not validators are run.

    .. deprecated:: 21.3.0 It will not be removed, but it also will not be
        moved to new ``attrs`` namespace. Use `attrs.validators.get_disabled()`
        instead.
    """
    return _run_validators


# SPDX-License-Identifier: MIT


import copy

from ._compat import PY_3_9_PLUS, get_generic_base
from ._make import _OBJ_SETATTR, NOTHING, fields
from .exceptions import AttrsAttributeNotFoundError


def asdict(
    inst,
    recurse=True,
    filter=None,
    dict_factory=dict,
    retain_collection_types=False,
    value_serializer=None,
):
    """
    Return the *attrs* attribute values of *inst* as a dict.

    Optionally recurse into other *attrs*-decorated classes.

    Args:
        inst: Instance of an *attrs*-decorated class.

        recurse (bool): Recurse into classes that are also *attrs*-decorated.

        filter (~typing.Callable):
            A callable whose return code determines whether an attribute or
            element is included (`True`) or dropped (`False`).  Is called with
            the `attrs.Attribute` as the first argument and the value as the
            second argument.

        dict_factory (~typing.Callable):
            A callable to produce dictionaries from.  For example, to produce
            ordered dictionaries instead of normal Python dictionaries, pass in
            ``collections.OrderedDict``.

        retain_collection_types (bool):
            Do not convert to `list` when encountering an attribute whose type
            is `tuple` or `set`.  Only meaningful if *recurse* is `True`.

        value_serializer (typing.Callable | None):
            A hook that is called for every attribute or dict key/value.  It
            receives the current instance, field and value and must return the
            (updated) value.  The hook is run *after* the optional *filter* has
            been applied.

    Returns:
        Return type of *dict_factory*.

    Raises:
        attrs.exceptions.NotAnAttrsClassError:
            If *cls* is not an *attrs* class.

    ..  versionadded:: 16.0.0 *dict_factory*
    ..  versionadded:: 16.1.0 *retain_collection_types*
    ..  versionadded:: 20.3.0 *value_serializer*
    ..  versionadded:: 21.3.0
        If a dict has a collection for a key, it is serialized as a tuple.
    """
    attrs = fields(inst.__class__)
    rv = dict_factory()
    for a in attrs:
        v = getattr(inst, a.name)
        if filter is not None and not filter(a, v):
            continue

        if value_serializer is not None:
            v = value_serializer(inst, a, v)

        if recurse is True:
            if has(v.__class__):
                rv[a.name] = asdict(
                    v,
                    recurse=True,
                    filter=filter,
                    dict_factory=dict_factory,
                    retain_collection_types=retain_collection_types,
                    value_serializer=value_serializer,
                )
            elif isinstance(v, (tuple, list, set, frozenset)):
                cf = v.__class__ if retain_collection_types is True else list
                items = [
                    _asdict_anything(
                        i,
                        is_key=False,
                        filter=filter,
                        dict_factory=dict_factory,
                        retain_collection_types=retain_collection_types,
                        value_serializer=value_serializer,
                    )
                    for i in v
                ]
                try:
                    rv[a.name] = cf(items)
                except TypeError:
                    if not issubclass(cf, tuple):
                        raise
                    # Workaround for TypeError: cf.__new__() missing 1 required
                    # positional argument (which appears, for a namedturle)
                    rv[a.name] = cf(*items)
            elif isinstance(v, dict):
                df = dict_factory
                rv[a.name] = df(
                    (
                        _asdict_anything(
                            kk,
                            is_key=True,
                            filter=filter,
                            dict_factory=df,
                            retain_collection_types=retain_collection_types,
                            value_serializer=value_serializer,
                        ),
                        _asdict_anything(
                            vv,
                            is_key=False,
                            filter=filter,
                            dict_factory=df,
                            retain_collection_types=retain_collection_types,
                            value_serializer=value_serializer,
                        ),
                    )
                    for kk, vv in v.items()
                )
            else:
                rv[a.name] = v
        else:
            rv[a.name] = v
    return rv


def _asdict_anything(
    val,
    is_key,
    filter,
    dict_factory,
    retain_collection_types,
    value_serializer,
):
    """
    ``asdict`` only works on attrs instances, this works on anything.
    """
    if getattr(val.__class__, "__attrs_attrs__", None) is not None:
        # Attrs class.
        rv = asdict(
            val,
            recurse=True,
            filter=filter,
            dict_factory=dict_factory,
            retain_collection_types=retain_collection_types,
            value_serializer=value_serializer,
        )
    elif isinstance(val, (tuple, list, set, frozenset)):
        if retain_collection_types is True:
            cf = val.__class__
        elif is_key:
            cf = tuple
        else:
            cf = list

        rv = cf(
            [
                _asdict_anything(
                    i,
                    is_key=False,
                    filter=filter,
                    dict_factory=dict_factory,
                    retain_collection_types=retain_collection_types,
                    value_serializer=value_serializer,
                )
                for i in val
            ]
        )
    elif isinstance(val, dict):
        df = dict_factory
        rv = df(
            (
                _asdict_anything(
                    kk,
                    is_key=True,
                    filter=filter,
                    dict_factory=df,
                    retain_collection_types=retain_collection_types,
                    value_serializer=value_serializer,
                ),
                _asdict_anything(
                    vv,
                    is_key=False,
                    filter=filter,
                    dict_factory=df,
                    retain_collection_types=retain_collection_types,
                    value_serializer=value_serializer,
                ),
            )
            for kk, vv in val.items()
        )
    else:
        rv = val
        if value_serializer is not None:
            rv = value_serializer(None, None, rv)

    return rv


def astuple(
    inst,
    recurse=True,
    filter=None,
    tuple_factory=tuple,
    retain_collection_types=False,
):
    """
    Return the *attrs* attribute values of *inst* as a tuple.

    Optionally recurse into other *attrs*-decorated classes.

    Args:
        inst: Instance of an *attrs*-decorated class.

        recurse (bool):
            Recurse into classes that are also *attrs*-decorated.

        filter (~typing.Callable):
            A callable whose return code determines whether an attribute or
            element is included (`True`) or dropped (`False`).  Is called with
            the `attrs.Attribute` as the first argument and the value as the
            second argument.

        tuple_factory (~typing.Callable):
            A callable to produce tuples from. For example, to produce lists
            instead of tuples.

        retain_collection_types (bool):
            Do not convert to `list` or `dict` when encountering an attribute
            which type is `tuple`, `dict` or `set`. Only meaningful if
            *recurse* is `True`.

    Returns:
        Return type of *tuple_factory*

    Raises:
        attrs.exceptions.NotAnAttrsClassError:
            If *cls* is not an *attrs* class.

    ..  versionadded:: 16.2.0
    """
    attrs = fields(inst.__class__)
    rv = []
    retain = retain_collection_types  # Very long. :/
    for a in attrs:
        v = getattr(inst, a.name)
        if filter is not None and not filter(a, v):
            continue
        if recurse is True:
            if has(v.__class__):
                rv.append(
                    astuple(
                        v,
                        recurse=True,
                        filter=filter,
                        tuple_factory=tuple_factory,
                        retain_collection_types=retain,
                    )
                )
            elif isinstance(v, (tuple, list, set, frozenset)):
                cf = v.__class__ if retain is True else list
                items = [
                    (
                        astuple(
                            j,
                            recurse=True,
                            filter=filter,
                            tuple_factory=tuple_factory,
                            retain_collection_types=retain,
                        )
                        if has(j.__class__)
                        else j
                    )
                    for j in v
                ]
                try:
                    rv.append(cf(items))
                except TypeError:
                    if not issubclass(cf, tuple):
                        raise
                    # Workaround for TypeError: cf.__new__() missing 1 required
                    # positional argument (which appears, for a namedturle)
                    rv.append(cf(*items))
            elif isinstance(v, dict):
                df = v.__class__ if retain is True else dict
                rv.append(
                    df(
                        (
                            (
                                astuple(
                                    kk,
                                    tuple_factory=tuple_factory,
                                    retain_collection_types=retain,
                                )
                                if has(kk.__class__)
                                else kk
                            ),
                            (
                                astuple(
                                    vv,
                                    tuple_factory=tuple_factory,
                                    retain_collection_types=retain,
                                )
                                if has(vv.__class__)
                                else vv
                            ),
                        )
                        for kk, vv in v.items()
                    )
                )
            else:
                rv.append(v)
        else:
            rv.append(v)

    return rv if tuple_factory is list else tuple_factory(rv)


def has(cls):
    """
    Check whether *cls* is a class with *attrs* attributes.

    Args:
        cls (type): Class to introspect.

    Raises:
        TypeError: If *cls* is not a class.

    Returns:
        bool:
    """
    attrs = getattr(cls, "__attrs_attrs__", None)
    if attrs is not None:
        return True

    # No attrs, maybe it's a specialized generic (A[str])?
    generic_base = get_generic_base(cls)
    if generic_base is not None:
        generic_attrs = getattr(generic_base, "__attrs_attrs__", None)
        if generic_attrs is not None:
            # Stick it on here for speed next time.
            cls.__attrs_attrs__ = generic_attrs
        return generic_attrs is not None
    return False


def assoc(inst, **changes):
    """
    Copy *inst* and apply *changes*.

    This is different from `evolve` that applies the changes to the arguments
    that create the new instance.

    `evolve`'s behavior is preferable, but there are `edge cases`_ where it
    doesn't work. Therefore `assoc` is deprecated, but will not be removed.

    .. _`edge cases`: https://github.com/python-attrs/attrs/issues/251

    Args:
        inst: Instance of a class with *attrs* attributes.

        changes: Keyword changes in the new copy.

    Returns:
        A copy of inst with *changes* incorporated.

    Raises:
        attrs.exceptions.AttrsAttributeNotFoundError:
            If *attr_name* couldn't be found on *cls*.

        attrs.exceptions.NotAnAttrsClassError:
            If *cls* is not an *attrs* class.

    ..  deprecated:: 17.1.0
        Use `attrs.evolve` instead if you can. This function will not be
        removed du to the slightly different approach compared to
        `attrs.evolve`, though.
    """
    new = copy.copy(inst)
    attrs = fields(inst.__class__)
    for k, v in changes.items():
        a = getattr(attrs, k, NOTHING)
        if a is NOTHING:
            msg = f"{k} is not an attrs attribute on {new.__class__}."
            raise AttrsAttributeNotFoundError(msg)
        _OBJ_SETATTR(new, k, v)
    return new


def resolve_types(
    cls, globalns=None, localns=None, attribs=None, include_extras=True
):
    """
    Resolve any strings and forward annotations in type annotations.

    This is only required if you need concrete types in :class:`Attribute`'s
    *type* field. In other words, you don't need to resolve your types if you
    only use them for static type checking.

    With no arguments, names will be looked up in the module in which the class
    was created. If this is not what you want, for example, if the name only
    exists inside a method, you may pass *globalns* or *localns* to specify
    other dictionaries in which to look up these names. See the docs of
    `typing.get_type_hints` for more details.

    Args:
        cls (type): Class to resolve.

        globalns (dict | None): Dictionary containing global variables.

        localns (dict | None): Dictionary containing local variables.

        attribs (list | None):
            List of attribs for the given class. This is necessary when calling
            from inside a ``field_transformer`` since *cls* is not an *attrs*
            class yet.

        include_extras (bool):
            Resolve more accurately, if possible. Pass ``include_extras`` to
            ``typing.get_hints``, if supported by the typing module. On
            supported Python versions (3.9+), this resolves the types more
            accurately.

    Raises:
        TypeError: If *cls* is not a class.

        attrs.exceptions.NotAnAttrsClassError:
            If *cls* is not an *attrs* class and you didn't pass any attribs.

        NameError: If types cannot be resolved because of missing variables.

    Returns:
        *cls* so you can use this function also as a class decorator. Please
        note that you have to apply it **after** `attrs.define`. That means the
        decorator has to come in the line **before** `attrs.define`.

    ..  versionadded:: 20.1.0
    ..  versionadded:: 21.1.0 *attribs*
    ..  versionadded:: 23.1.0 *include_extras*
    """
    # Since calling get_type_hints is expensive we cache whether we've
    # done it already.
    if getattr(cls, "__attrs_types_resolved__", None) != cls:
        import typing

        kwargs = {"globalns": globalns, "localns": localns}

        if PY_3_9_PLUS:
            kwargs["include_extras"] = include_extras

        hints = typing.get_type_hints(cls, **kwargs)
        for field in fields(cls) if attribs is None else attribs:
            if field.name in hints:
                # Since fields have been frozen we must work around it.
                _OBJ_SETATTR(field, "type", hints[field.name])
        # We store the class we resolved so that subclasses know they haven't
        # been resolved.
        cls.__attrs_types_resolved__ = cls

    # Return the class so you can use it as a decorator too.
    return cls


# SPDX-License-Identifier: MIT

"""
Commonly useful filters for `attrs.asdict` and `attrs.astuple`.
"""

from ._make import Attribute


def _split_what(what):
    """
    Returns a tuple of `frozenset`s of classes and attributes.
    """
    return (
        frozenset(cls for cls in what if isinstance(cls, type)),
        frozenset(cls for cls in what if isinstance(cls, str)),
        frozenset(cls for cls in what if isinstance(cls, Attribute)),
    )


def include(*what):
    """
    Create a filter that only allows *what*.

    Args:
        what (list[type, str, attrs.Attribute]):
            What to include. Can be a type, a name, or an attribute.

    Returns:
        Callable:
            A callable that can be passed to `attrs.asdict`'s and
            `attrs.astuple`'s *filter* argument.

    .. versionchanged:: 23.1.0 Accept strings with field names.
    """
    cls, names, attrs = _split_what(what)

    def include_(attribute, value):
        return (
            value.__class__ in cls
            or attribute.name in names
            or attribute in attrs
        )

    return include_


def exclude(*what):
    """
    Create a filter that does **not** allow *what*.

    Args:
        what (list[type, str, attrs.Attribute]):
            What to exclude. Can be a type, a name, or an attribute.

    Returns:
        Callable:
            A callable that can be passed to `attrs.asdict`'s and
            `attrs.astuple`'s *filter* argument.

    .. versionchanged:: 23.3.0 Accept field name string as input argument
    """
    cls, names, attrs = _split_what(what)

    def exclude_(attribute, value):
        return not (
            value.__class__ in cls
            or attribute.name in names
            or attribute in attrs
        )

    return exclude_
