import unittest

import numpy as np
from typing import Any, Optional, Union
from collections.abc import Mapping, Sequence, Set
from collections import defaultdict

import numpy.testing

from ionsim.api import Unit, ionsim_api
from ionsim import ureg


def assert_array_equal(arr0, arr1):
    numpy.testing.assert_array_equal(arr0.magnitude, arr1.magnitude)
    assert arr0.units == arr1.units


class ApiUnitConverter(unittest.TestCase):

    def test_require_return_annotation(self):
        """Test that we require a return annotation."""

        with self.assertRaises(ValueError):
            @ionsim_api
            def foo(x):
                pass

    def test_convert_tuple_argument(self):
        """Test converting a tuple to a pint.Quantity."""

        x_inner = None

        @ionsim_api
        def foo(x: Unit[float, 'm']) -> Unit[float, 'm']:
            nonlocal x_inner
            x_inner = x
            return x

        exp = ureg.Quantity(5, 'm')
        act = foo((5, 'm'))
        self.assertEqual(exp, act)
        self.assertEqual(x_inner, 5)

        exp = ureg.Quantity(np.array([5, 6, 7]), 'm')
        act = foo((np.array([5, 6, 7]), 'm'))
        assert_array_equal(exp, act)
        numpy.testing.assert_array_equal(np.array([5, 6, 7]), x_inner)

    def test_convert_list_argument(self):
        """Test converting a list to a pint.Quantity."""

        x_inner = None

        @ionsim_api
        def foo(x: Unit[float, 'm']) -> Unit[float, 'm']:
            nonlocal x_inner
            x_inner = x
            return x

        exp = ureg.Quantity(5, 'm')
        act = foo([5, 'm'])
        self.assertEqual(exp, act)
        self.assertEqual(x_inner, 5)

        exp = ureg.Quantity(np.array([5, 6, 7]), 'm')
        act = foo([np.array([5, 6, 7]), 'm'])
        assert_array_equal(exp, act)
        numpy.testing.assert_array_equal(np.array([5, 6, 7]), x_inner)

    def test_convert_string_argument(self):
        """Test converting a string to a pint.Quantity."""

        x_inner = None

        @ionsim_api
        def foo(x: Unit[float, 'm']) -> Unit[float, 'm']:
            nonlocal x_inner
            x_inner = x
            return x

        exp = ureg.Quantity(5, 'm')
        act = foo('5 m')
        self.assertEqual(exp, act)
        self.assertEqual(x_inner, 5)

    def test_pass_through_quantity_argument(self):
        """Test that a pint.Quantity is passed through unchanged."""

        x_inner = None

        @ionsim_api
        def foo(x: Unit[float, 'm']) -> Unit[float, 'm']:
            nonlocal x_inner
            x_inner = x
            return x

        exp = ureg.Quantity(5, 'm')
        act = foo(ureg.Quantity(5, 'm'))
        self.assertEqual(exp, act)
        self.assertEqual(x_inner, 5)

        exp = ureg.Quantity(np.array([5, 6, 7]), 'm')
        act = foo(ureg.Quantity(np.array([5, 6, 7]), 'm'))
        assert_array_equal(exp, act)
        numpy.testing.assert_array_equal(np.array([5, 6, 7]), x_inner)

    def test_output_not_converted(self):
        """Test that no conversion happens on the output."""

        x_inner = None

        @ionsim_api
        def foo(x: Unit[float, 'm']) -> Unit[float, 'cm']:
            nonlocal x_inner
            x_inner = x
            return x

        exp = ureg.Quantity(10_000, 'cm')  # NOT 1_000_000 cm
        act = foo((10, 'km'))
        self.assertEqual(exp, act)
        self.assertEqual(10000, x_inner)

    def test_convert_compatible_units(self):
        """Test that compatible units are converted on input."""

        x_inner = None

        @ionsim_api
        def foo(x: Unit[float, 'm']) -> Unit[float, 'm']:
            nonlocal x_inner
            x_inner = x
            return x

        exp = ureg.Quantity(5000, 'm')
        act = foo(ureg.Quantity(5, 'km'))
        self.assertEqual(exp, act)
        self.assertEqual(5000, x_inner)

        exp = ureg.Quantity(np.array([5000, 6000, 7000]), 'm')
        act = foo(ureg.Quantity(np.array([5, 6, 7]), 'km'))
        assert_array_equal(exp, act)
        numpy.testing.assert_array_equal(np.array([5000, 6000, 7000]), x_inner)

    def test_reject_scalar(self):
        """Test that a scalar is rejected with an error."""

        @ionsim_api
        def foo(x: Unit[float, 'm']) -> Unit[float, 'm']:
            return x

        with self.assertRaises(ValueError):
            foo(5)

    def test_reject_ndarray(self):
        """Test that a bare ndarray is rejected with an error."""

        @ionsim_api
        def foo(x: Unit[float, 'm']) -> Unit[float, 'm']:
            return x

        with self.assertRaises(ValueError):
            foo(np.array([1,2,3,4]))

    def test_unannotated_are_unchanged(self):
        """Test that any value may be passed through to an unannotated parameter."""

        x_inner = None

        @ionsim_api
        def foo(x) -> Any:
            nonlocal x_inner
            x_inner = x
            return x

        self.assertEqual(foo(5), 5)
        self.assertEqual(5, x_inner)

    def test_class_decorator(self):
        """Test that the methods of a decorated class are all decorated."""

        x_inner = None
        y_inner = None
        z_inner = None

        @ionsim_api
        class Foo:
            def bar(self, x: Unit[float, 'm']) -> Unit[float, 'm']:
                nonlocal x_inner
                x_inner = x
                return x
            @staticmethod
            def baz(y: Unit[float, 's']) -> Unit[float, 's']:
                nonlocal y_inner
                y_inner = y
                return y
            @classmethod
            def qux(cls, z: Unit[float, 'J']) -> Unit[float, 'J']:
                nonlocal z_inner
                z_inner = z
                return z

        f = Foo()

        exp = ureg.Quantity(10, 'm')
        act = f.bar((10, 'm'))
        self.assertEqual(exp, act)
        self.assertEqual(10, x_inner)

        exp = ureg.Quantity(.002, 's')
        act = f.baz((2, 'ms'))
        self.assertEqual(exp, act)
        self.assertEqual(.002, y_inner)

        exp = ureg.Quantity(3, 'J')
        act = Foo.qux((3, 'J'))
        self.assertEqual(exp, act)
        self.assertEqual(3, z_inner)

    def test_unit_or_none(self):
        """Test that a unit may be passed in or None."""

        x_inner = None

        @ionsim_api
        def foo(x: Unit[float, 'm'] | None) -> Unit[float, 'm'] | None:
            nonlocal x_inner
            x_inner = x
            return x

        exp = ureg.Quantity(10, 'm')
        act = foo((10, 'm'))
        self.assertEqual(exp, act)
        self.assertEqual(10, x_inner)

        exp = None
        act = foo(None)
        self.assertEqual(exp, act)
        self.assertEqual(None, x_inner)

        with self.assertRaises(ValueError):
            foo(5)

    def test_optional_unit(self):
        """Test an Optional unit."""

        x_inner = None

        @ionsim_api
        def foo(x: Optional[Unit[float, 'm']]) -> Optional[Unit[float, 'm']]:
            nonlocal x_inner
            x_inner = x
            return x

        exp = ureg.Quantity(10, 'm')
        act = foo((10, 'm'))
        self.assertEqual(exp, act)
        self.assertEqual(10, x_inner)

        exp = None
        act = foo(None)
        self.assertEqual(exp, act)
        self.assertEqual(None, x_inner)

        with self.assertRaises(ValueError):
            foo(5)

    def test_union_unit(self):
        """Test a union of a unit and other type annotation."""

        x_inner = None

        @ionsim_api
        def foo(x: Union[Unit[float, 'm'], bool]) -> Union[Unit[float, 'm'], bool]:
            nonlocal x_inner
            x_inner = x
            return x

        exp = ureg.Quantity(10, 'm')
        act = foo((10, 'm'))
        self.assertEqual(exp, act)
        self.assertEqual(10, x_inner)

        exp = True
        act = foo(True)
        self.assertEqual(exp, act)
        self.assertEqual(True, x_inner)

        with self.assertRaises(ValueError):
            foo(5)

    def test_unit_or_any(self):
        """Test that a Unit may be passed in or any other value, but
        if the object can be converted to a Quantity it is."""

        x_inner = None

        @ionsim_api
        def foo(x: Unit[float, 'm'] | Any) -> Unit[float, 'm']:
            nonlocal x_inner
            x_inner = x
            return x

        exp = ureg.Quantity(10, 'm')
        act = foo((10, 'm'))
        self.assertEqual(exp, act)
        self.assertEqual(10, x_inner)

        exp = ureg.Quantity(5, 'm')
        act = foo(5)
        self.assertEqual(exp, act)
        self.assertEqual(5, x_inner)

    def test_units_in_list(self):
        """Test a list of units."""

        x_inner = None

        @ionsim_api
        def foo(x: list[Unit[float, 'm']]) -> list[Unit[float, 'm']]:
            nonlocal x_inner
            x_inner = x
            return x

        exp = [ureg.Quantity(5, 'm'), ureg.Quantity(10, 'm')]
        act = foo([(5, 'm'), (10, 'm')])
        self.assertEqual(exp, act)
        self.assertEqual([5, 10], x_inner)

    def test_reject_invalid_in_list(self):
        """Reject unitless quantities embedded in a list."""

        @ionsim_api
        def foo(x: list[Unit[float, 'm']]) -> list[Unit[float, 'm']]:
            return x

        with self.assertRaises(ValueError):
            foo([42, 5])

    def test_sequence_list_equivalence(self):
        """Test that the Sequence annotation is equivalent to list."""

        x_inner = None

        @ionsim_api
        def foo(x: Sequence[Unit[float, 'm']]) -> Sequence[Unit[float, 'm']]:
            nonlocal x_inner
            x_inner = x
            return x

        exp = [ureg.Quantity(5, 'm'), ureg.Quantity(10, 'm')]
        act = foo([(5, 'm'), (10, 'm')])
        self.assertEqual(exp, act)
        self.assertEqual([5, 10], x_inner)

    def test_bare_list_ignored(self):
        """Test that a bare list annotation is ignored."""

        x_inner = None

        @ionsim_api
        def foo(x: list) -> Any:
            nonlocal x_inner
            x_inner = x
            return x

        exp = [1, 2, 'a']
        act = foo([1, 2, 'a'])
        self.assertEqual(exp, act)
        self.assertEqual([1, 2, 'a'], x_inner)

    def test_units_in_tuple(self):
        """Test a tuple of units."""

        x_inner=  None

        @ionsim_api
        def foo(x: tuple[Unit[float, 'm'], int, Unit[float, 's']]) -> tuple[Unit[float, 'm'], int, Unit[float, 's']]:
            nonlocal x_inner
            x_inner = x
            return x

        exp = (ureg.Quantity(5, 'm'), 15, ureg.Quantity(10, 's'))
        act = foo([(5, 'm'), 15, (10, 's')])
        self.assertEqual(exp, act)
        self.assertEqual((5, 15, 10), x_inner)

    def test_bare_tuple_ignored(self):
        """Test that a bare tuple annotation is ignored."""

        x_inner = None

        @ionsim_api
        def foo(x: tuple) -> tuple:
            nonlocal x_inner
            x_inner = x
            return x

        exp = (1, 2, 'a')
        act = foo((1, 2, 'a'))
        self.assertEqual(exp, act)
        self.assertEqual((1, 2, 'a'), x_inner)

    def test_unit_keys_in_dict(self):
        """Test using units as keys in a dict."""

        x_inner = None

        @ionsim_api
        def foo(x: dict[Unit[float, 'm'], int]) -> dict[Unit[float, 'm'], int]:
            nonlocal x_inner
            x_inner = x
            return x

        exp = {ureg.Quantity(5, 'm'): 42}
        act = foo({(5, 'm'): 42})
        self.assertEqual(exp, act)
        self.assertEqual({5: 42}, x_inner)

    def test_unit_values_in_dict(self):
        """Test using units as values in a dict."""

        x_inner = None

        @ionsim_api
        def foo(x: dict[int, Unit[float, 'm']]) -> dict[int, Unit[float, 'm']]:
            nonlocal x_inner
            x_inner = x
            return x

        exp = {42: ureg.Quantity(5, 'm')}
        act = foo({42: (5, 'm')})
        self.assertEqual(exp, act)
        self.assertEqual({42: 5}, x_inner)

    def test_reject_invalid_in_dict(self):
        """Reject unitless quantities embedded in a dict."""

        @ionsim_api
        def foo(x: dict[int, Unit[float, 'm']]) -> dict[int, Unit[float, 'm']]:
            return x

        with self.assertRaises(ValueError):
            foo({42: 5})

    def test_mapping_equivalent_to_dict(self):
        """Test that Mapping is treated like dict."""

        x_inner = None

        @ionsim_api
        def foo(x: Mapping[Unit[float, 's'], Unit[float, 'm']]) -> Mapping[Unit[float, 's'], Unit[float, 'm']]:
            nonlocal x_inner
            x_inner = x
            return x

        exp = {ureg.Quantity(0.5, 's'): ureg.Quantity(100, 'm')}
        act = foo({(0.5, 's'): (100, 'm')})
        self.assertEqual(exp, act)
        self.assertEqual({0.5: 100}, x_inner)

    def test_units_in_defaultdict(self):
        """Test that if a key or value is converted in a defaultdict that remains a defaultdict."""

        x_inner = None

        @ionsim_api
        def foo(x: Mapping[int, Unit[float, 'm']]) -> Mapping[int, Unit[float, 'm']]:
            nonlocal x_inner
            x_inner = x
            return x

        exp = defaultdict(int, [(42, ureg.Quantity(5, 'm'))])
        act = foo(defaultdict(int, {42: (5, 'm')}))
        self.assertEqual(exp, act)
        self.assertEqual(defaultdict(int, [(42, 5)]), x_inner)

        self.assertEqual(act[100], 0)

    def test_bare_dict_ignored(self):
        """Test that a bare dict annotation is ignored."""

        x_inner = None

        @ionsim_api
        def foo(x: dict) -> dict:
            nonlocal x_inner
            x_inner = x
            return x

        exp = {1: 5, 2: 'b', 'a': 9}
        act = foo({1: 5, 2: 'b', 'a': 9})
        self.assertEqual(exp, act)
        self.assertEqual(exp, x_inner)

    def test_units_in_set(self):
        """Test units in a set."""

        x_inner = None

        @ionsim_api
        def foo(x: set[Unit[float, 'm']]) -> set[Unit[float, 'm']]:
            nonlocal x_inner
            x_inner = x
            return x

        exp = {ureg.Quantity(5, 'm')}
        act = foo({(5, 'm')})
        self.assertEqual(exp, act)
        self.assertEqual({5}, x_inner)

    def test_reject_invalid_in_set(self):
        """Reject unitless quantities embedded in a set."""

        @ionsim_api
        def foo(x: set[Unit[float, 'm']]) -> set[Unit[float, 'm']]:
            return x

        with self.assertRaises(ValueError):
            foo({5})

    def test_set_abc_equivalence(self):
        """Test that set and Set are equivalent as annotations."""

        x_inner = None

        @ionsim_api
        def foo(x: Set[Unit[float, 'm']]) -> Set[Unit[float, 'm']]:
            nonlocal x_inner
            x_inner = x
            return x

        exp = {ureg.Quantity(5, 'm')}
        act = foo({(5, 'm')})
        self.assertEqual(exp, act)
        self.assertEqual({5}, x_inner)

    def test_frozenset_preserved(self):
        """Test that units in a frozenset preserve the frozenset."""

        x_inner = None

        @ionsim_api
        def foo(x: Set[Unit[float, 'm']]) -> Set[Unit[float, 'm']]:
            nonlocal x_inner
            x_inner = x
            return x

        exp = frozenset({ureg.Quantity(5, 'm'), ureg.Quantity(20, 'm')})
        act = foo(frozenset({(5, 'm'), (20, 'm')}))
        self.assertEqual(exp, act)
        self.assertEqual(frozenset({5, 20}), x_inner)

    def test_bare_set_ignored(self):
        """Test that a bare set annotation is ignored."""

        x_inner = None

        @ionsim_api
        def foo(x: set) -> set:
            nonlocal x_inner
            x_inner = x
            return x

        exp = {1, 3, 'a'}
        act = foo({1, 3, 'a'})
        self.assertEqual(exp, act)
        self.assertEqual(exp, x_inner)

    def test_nested_units(self):
        """Test units nested several layers in an annotation."""

        x_inner = None

        @ionsim_api
        def foo(x: dict[Unit[float, 'm'], list[Unit[float, 's']]]) -> dict[Unit[float, 'm'], list[Unit[float, 's']]]:
            nonlocal x_inner
            x_inner = x
            return x

        exp = {ureg.Quantity(42, 'm'): [ureg.Quantity(1, 's'), ureg.Quantity(2, 's')]}
        act = foo({(42, 'm'): [(1, 's'), (2, 's')]})
        self.assertEqual(exp, act)
        self.assertEqual({42: [1, 2]}, x_inner)

    def test_multiple_annotations(self):
        """Test matching multiple annotations."""

        x_inner = None

        @ionsim_api
        def foo(x: Unit[float, 'J'] | Unit[float, 'Hz']) -> float:
            nonlocal x_inner
            x_inner = x
            return x

        exp = 20
        act = foo((20, 'J'))
        self.assertEqual(exp, act)
        self.assertEqual(20, x_inner)

        exp = 1000
        act = foo((1000, 'Hz'))
        self.assertEqual(exp, act)
        self.assertEqual(1000, x_inner)

        with self.assertRaises(ValueError):
            foo((20, 'm'))

    def test_multiple_annotations_compatible(self):
        """Test matching multiple annotations where none match but one is compatible."""

        x_inner = None

        @ionsim_api
        def foo(x: Unit[float, 'J'] | Unit[float, 'Hz']) -> float:
            nonlocal x_inner
            x_inner = x
            return x

        exp = 1000
        act = foo((1, 'kHz'))
        self.assertEqual(exp, act)
        self.assertEqual(1000, x_inner)

    def test_multiple_annotations_fail_on_ambiguous(self):
        """Test with multiple annotations we fail more than one are compatible."""

        @ionsim_api
        def foo(x: Unit[float, 'm'] | Unit[float, 'km']) -> Unit[float, 'm'] | Unit[float, 'km']:
            return x

        with self.assertRaises(ValueError):
            foo((30, 'm'))

        with self.assertRaises(ValueError):
            foo((30, 'cm'))

        with self.assertRaises(ValueError):
            foo((30, 'km'))

    def test_union_over_complex_types(self):
        """Test a union of annotations where none at the top level is a Unit."""

        x_inner = None

        @ionsim_api
        def foo(x: list[Unit[float, 'm']] | list[Unit[float, 's']]) -> list[Unit[float, 'm']]:
            nonlocal x_inner
            x_inner = x
            return x

        exp = [ureg.Quantity(10, 'm'), ureg.Quantity(20, 'm')]
        act = foo([(10, 'm'), (20, 'm')])
        self.assertEqual(exp, act)
        self.assertEqual([10, 20], x_inner)

        exp = [ureg.Quantity(1, 'm'), ureg.Quantity(2, 'm')]
        act = foo([(1, 's'), (2, 's')])
        self.assertEqual(exp, act)
        self.assertEqual([1, 2], x_inner)

        with self.assertRaises(ValueError):
            foo([(1, 'J')])

    def test_pipe_union_equivalence(self):
        """Test that the pipe operator is handled equivalently to the Union type."""

        x_inner = None

        @ionsim_api
        def foo(x: Union[Unit[float, 'J'], Unit[float, 'Hz']]) -> float:
            nonlocal x_inner
            x_inner = x
            return x

        exp = 20
        act = foo((20, 'J'))
        self.assertEqual(exp, act)
        self.assertEqual(20, x_inner)

        exp = 1000
        act = foo((1000, 'Hz'))
        self.assertEqual(exp, act)
        self.assertEqual(1000, x_inner)

        with self.assertRaises(ValueError):
            foo((20, 'm'))

    def test_convert_hertz_to_radians_per_sec(self):
        """Test converting radians per second to Hz divides by 2*pi."""

        x_inner = None

        @ionsim_api
        def foo(x: Unit[float, 'Hz']) -> Unit[float, 'Hz']:
            nonlocal x_inner
            x_inner = x
            return x

        exp = ureg.Quantity(2, 'Hz')
        act = foo((4*np.pi, 'rad/s'))
        self.assertAlmostEqual(exp, act)
        self.assertAlmostEqual(2, x_inner)

        exp = ureg.Quantity(2, 'Hz')
        act = foo((2, 'Hz'))
        self.assertEqual(exp, act)
        self.assertEqual(2, x_inner)

    def test_convert_radians_per_sec_to_hertz(self):
        """Test converting Hz to radians per second multiplies by 2*pi."""

        x_inner = None

        @ionsim_api
        def foo(x: Unit[float, 'rad/s']) -> Unit[float, 'rad/s']:
            nonlocal x_inner
            x_inner = x
            return x

        exp = ureg.Quantity(4*np.pi, 'rad/s')
        act = foo((2, 'Hz'))
        self.assertAlmostEqual(exp, act)
        self.assertAlmostEqual(4*np.pi, x_inner)

        exp = ureg.Quantity(4*np.pi, 'rad/s')
        act = foo((4*np.pi, 'rad/s'))
        self.assertEqual(exp, act)
        self.assertEqual(4*np.pi, x_inner)

    def test_convert_mhz_to_radians_per_sec(self):
        """Test converting MHz to rad/s."""

        x_inner = None

        @ionsim_api
        def foo(x: Unit[float, 'Hz']) -> Unit[float, 'Hz']:
            nonlocal x_inner
            x_inner = x
            return x

        exp = ureg.Quantity(2, 'MHz')
        act = foo((4*np.pi * 1000000, 'rad/s'))
        self.assertAlmostEqual(exp, act)
        self.assertAlmostEqual(2000000, x_inner)

        exp = ureg.Quantity(2, 'MHz')
        act = foo((2000000, 'Hz'))
        self.assertEqual(exp, act)
        self.assertEqual(2000000, x_inner)


    def test_convert_radians_per_sec_to_mhz(self):
        """Test converting rad/s to MHz."""

        x_inner = None

        @ionsim_api
        def foo(x: Unit[float, 'rad/s']) -> Unit[float, 'rad/s']:
            nonlocal x_inner
            x_inner = x
            return x

        exp = ureg.Quantity(4*np.pi * 1000000, 'rad/s')
        act = foo((2, 'MHz'))
        self.assertAlmostEqual(exp, act)
        self.assertAlmostEqual(4*np.pi * 1000000, x_inner)

        exp = ureg.Quantity(4*np.pi, 'rad/s')
        act = foo((4*np.pi, 'rad/s'))
        self.assertEqual(exp, act)
        self.assertEqual(4*np.pi, x_inner)
