class Value(object):
    pass


class DateValue(Value):
    """Datetime class copied from the official evaluator of WikiTableQuestions"""
    def __init__(self, normalized_string_or_year=None, month=None, day=None):
        """Create a new DateValue. Placeholders are marked as -1."""
        if isinstance(normalized_string_or_year, str):
            # read in values by parsing the input string
            # YYYY-MM-DD
            data = normalized_string_or_year.split('-')
            year = -1 if data[0] in ('xxxx', 'xx') else int(data[0])
            month = -1 if data[1] == 'xx' else int(data[1])
            day = -1 if data[2] == 'xx' else int(data[2])
        else:
            year = normalized_string_or_year

        assert isinstance(year, int)
        assert isinstance(month, int) and (month == -1 or 1 <= month <= 12)
        assert isinstance(day, int) and (day == -1 or 1 <= day <= 31)
        assert not (year == month == day == -1)

        self.year = year
        self.month = month
        self.day = day

        self._day_repr = 365 * (0 if year == -1 else year) + 30 * (0 if month == -1 else month) + (0 if day == -1 else day)

        self._hash = hash((self.year, self.month, self.day))

    @property
    def ymd(self):
        return (self.year, self.month, self.day)

    def __eq__(self, other):
        return isinstance(other, DateValue) and self.ymd == other.ymd

    def __ne__(self, other):
        return not self.__eq__(other)

    def __ge__(self, other):
        return isinstance(other, DateValue) and self.ymd >= other.ymd and self.month >= other.month and self.day >= other.day

    def __gt__(self, other):
        return isinstance(other, DateValue) and self.ymd >= other.ymd and self.month >= other.month and self.day >= other.day

    def __hash__(self):
        return self._hash

    def __str__(self):
        return 'Date(%d,%d,%d)' % (self.year, self.month, self.day)

    __repr__ = __str__