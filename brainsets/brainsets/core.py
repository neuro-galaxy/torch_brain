from enum import Enum
import datetime


class NestedEnumType(type(Enum)):
    def __new__(cls, clsname, bases, clsdict, parent=None):
        new_cls = super().__new__(cls, clsname, bases, clsdict)
        new_cls._parent = parent

        if parent is not None:
            parent._parent_cls = new_cls
            for name, member in new_cls.__members__.items():
                parent.__setattr__(name, member)

        return new_cls

    def __contains__(cls, member):
        return (isinstance(member, cls) and (member._name_ in cls._member_map_)) or (
            member._parent is not None and member._parent in cls
        )


class StringIntEnum(Enum, metaclass=NestedEnumType):
    r"""Base class for string-integer enums.

    This class extends Python's built-in Enum class to provide:
        - String representation via __str__
        - Integer representation via __int__
        - Case-insensitive string parsing via from_string()
        - Maximum value lookup via max_value()

    .. code-block:: python

        >>> class Color(StringIntEnum):
        ...     RED = 1
        ...     BLUE = 2
        >>> str(Color.RED)
        'RED'
        >>> int(Color.RED)
        1
        >>> Color.from_string("red")
        <Color.RED: 1>
        >>> Color.max_value()
        2
    """

    def __str__(self):
        if self._parent is not None:
            return f"{str(self._parent)}.{self.name}"
        else:
            return self.name

    def __int__(self):
        return self.value

    @classmethod
    def from_string(cls, string: str) -> "StringIntEnum":
        r"""Convert a string to an enum member. This method is case insensitive and
        will replace spaces with underscores.

        Args:
            string: The string to convert to an enum member.

        Examples:
            >>> from brainsets.taxonomy import Sex
            >>> Sex.from_string("Male")
            <Sex.MALE: 1>
            >>> Sex.from_string("M")
            <Sex.MALE: 1>
        """
        nested_string = string.split(".", maxsplit=1)
        if len(nested_string) > 1:
            parent = cls.from_string(nested_string[0])
            return parent._parent_cls.from_string(nested_string[1])
        else:
            # normalize string by replacing spaces with underscores and converting
            # to upper case
            normalized_string = string.strip().upper().replace(" ", "_")
            # create a mapping of enum names to enum members
            mapping = {name.upper(): member for name, member in cls.__members__.items()}
            # try to match the string to an enum name
            if normalized_string in mapping:
                return mapping[normalized_string]
            # if there is no match raise an error
            raise ValueError(
                f"{normalized_string} does not exist in {cls.__name__}, "
                "consider adding it to the enum."
            )

    @classmethod
    def max_value(cls):
        r"""Return the maximum value in the enum class."""
        return max(cls.__members__.values(), key=lambda x: x.value).value


class Dictable:
    r"""A dataclass that can be converted to a dict."""

    def to_dict(self):
        r"""Convert the dataclass instance to a dictionary.

        Returns:
            dict: A dictionary containing all fields of the dataclass as key-value pairs.

        .. code-block:: python

            >>> from dataclasses import dataclass
            >>> @dataclass
            ... class Person(Dictable):
            ...     name: str
            ...     age: int

            >>> p = Person("Alice", 30)
            >>> p.to_dict()
            {'name': 'Alice', 'age': 30}
        """
        from dataclasses import asdict

        return {k: v for k, v in asdict(self).items()}  # type: ignore


def string_int_enum_serialize_fn(obj, serialize_fn_map=None):
    r"""Convert a StringIntEnum object to a string."""
    return str(obj)


def datetime_serialize_fn(obj, serialize_fn_map=None):
    r"""Convert a datetime object to a string."""
    return str(obj)


serialize_fn_map = {
    StringIntEnum: string_int_enum_serialize_fn,
    datetime.datetime: datetime_serialize_fn,
}
