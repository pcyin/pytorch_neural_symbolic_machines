from collections import namedtuple

Example = namedtuple('Example', ['question', 'table'])


class Column(object):
    def __init__(self, name, type, sample_value=None, **kwargs):
        self.name = name
        self.type = type
        self.sample_value = sample_value

        self.fields = []
        for key, val in kwargs.items():
            self.fields.append(key)
            setattr(self, key, val)

    def to_dict(self):
        data = {
            'name': self.name,
            'type': self.type,
            'sample_value': self.sample_value,
        }

        for key in self.fields:
            data[key] = getattr(self, key)

        return data


class Table(object):
    def __init__(self, id, header, data=None):
        self.id = id
        self.header = header
        self.data = data
