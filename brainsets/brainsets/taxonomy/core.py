from enum import Enum
from dataclasses import asdict


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
    """Enum where the value is a string, but can be cast to an int."""

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
            >>> from kirby.taxonomy import Sex
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
        r"""Return the maximum value of the enum."""
        return max(cls.__members__.values(), key=lambda x: x.value).value


class Dictable:
    """A dataclass that can be converted to a dict."""

    def to_dict(self):
        """__dict__ doesn't play well with torch.load"""
        return {k: v for k, v in asdict(self).items()}  # type: ignore
