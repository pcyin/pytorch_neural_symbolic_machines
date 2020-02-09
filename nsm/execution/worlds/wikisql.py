import collections

from nsm.execution.executor_factory import TableExecutor
from nsm.execution.type_system import get_simple_type_hierarchy


class WikiSQLExecutor(TableExecutor):

    def __init__(self, table_info, use_filter_str_contain=True, use_filter_str_equal=False):
        super(TableExecutor, self).__init__(table_info)
        self.n_rows = len(table_info['row_ents'])
        self.use_filter_str_equal = use_filter_str_equal
        self.use_filter_str_contain = use_filter_str_contain

    def hop(self, entities, prop, keep_dup=True):
        """Get the property of a list of entities."""
        # Note this changes keep_dup=True as default, which is
        # different from WikiTableQuestions experiments.
        if keep_dup:
            result = []
        else:
            result = set()
        for ent in entities:
            try:
                if keep_dup:
                    result += self.kg[ent][prop]
                else:
                    result = result.union(self.kg[ent][prop])
            except KeyError:
                continue
        return list(result)

    def get_api(self):
        """Get the functions, constants and type hierarchy."""
        func_dict = collections.OrderedDict()

        def hop_return_type_fn(arg1_type, arg2_type):
            if arg2_type == 'num_property':
                return 'num_list'
            elif arg2_type == 'string_property':
                return 'string_list'
            elif arg2_type == 'datetime_property':
                return 'datetime_list'
            elif arg2_type == 'entity_property':
                return 'entity_list'
            else:
                raise ValueError('Unknown type {}'.format(arg2_type))

        func_dict['hop'] = dict(
            name='hop',
            args=[{'types': ['entity_list']},
                  {'types': ['property']}],
            return_type=hop_return_type_fn,
            autocomplete=self.autocomplete_hop,
            type='primitive_function',
            value=self.hop)

        if self.use_filter_str_equal:
            # Allow equal to work on every type.
            func_dict['filter_eq'] = dict(
                name='filter_eq',
                args=[{'types': ['entity_list']},
                      {'types': ['entity_list']},
                      {'types': ['property']}],
                return_type='entity_list',
                autocomplete=self.autocomplete_filter_equal,
                type='primitive_function',
                value=self.filter_equal)
        else:
            # Only use filter equal for number and date and
            # entities. Use filter_str_contain for string values.
            func_dict['filter_eq'] = dict(
                name='filter_eq',
                args=[{'types': ['entity_list']},
                      {'types': ['ordered_list']},
                      {'types': ['ordered_property']}],
                return_type='entity_list',
                autocomplete=self.autocomplete_filter_equal,
                type='primitive_function',
                value=self.filter_equal)

        if self.use_filter_str_contain:
            func_dict['filter_str_contain_any'] = dict(
                name='filter_str_contain_any',
                args=[{'types': ['entity_list']},
                      {'types': ['string_list']},
                      {'types': ['string_property']}],
                return_type='entity_list',
                autocomplete=self.autocomplete_filter_str_contain_any,
                type='primitive_function',
                value=self.filter_str_contain_any)

        func_dict['filter_greater'] = dict(
            name='filter_greater',
            args=[{'types': ['entity_list']},
                  {'types': ['ordered_list']},
                  {'types': ['ordered_property']}],
            return_type='entity_list',
            autocomplete=self.return_all_tokens,
            type='primitive_function',
            value=self.filter_greater)

        func_dict['filter_less'] = dict(
            name='filter_less',
            args=[{'types': ['entity_list']},
                  {'types': ['ordered_list']},
                  {'types': ['ordered_property']}],
            return_type='entity_list',
            autocomplete=self.return_all_tokens,
            type='primitive_function',
            value=self.filter_less)

        func_dict['count'] = dict(
            name='count',
            args=[{'types': ['entity_list']}],
            return_type='num',
            autocomplete=self.return_all_tokens,
            type='primitive_function',
            value=self.count)

        # aggregation functions.
        for k, f in zip(['maximum', 'minimum'],
                        [self.maximum, self.minimum]):
            func_dict[k] = dict(
                name=k,
                args=[{'types': ['entity_list']},
                      {'types': ['ordered_property']}],
                return_type='ordered_list',
                autocomplete=self.autocomplete_aggregation,
                type='primitive_function',
                value=f)

        func_dict['average'] = dict(
            name='average',
            args=[{'types': ['entity_list']},
                  {'types': ['num_property']}],
            return_type='num',
            autocomplete=self.autocomplete_aggregation,
            type='primitive_function',
            value=self.average)

        func_dict['sum'] = dict(
            name='sum',
            args=[{'types': ['entity_list']},
                  {'types': ['num_property']}],
            return_type='num',
            autocomplete=self.autocomplete_aggregation,
            type='primitive_function',
            value=self.sum)

        constant_dict = collections.OrderedDict()

        for p in self.props:
            if p in self.num_props:
                tp = 'num_property'
            elif p in self.datetime_props:
                tp = 'datetime_property'
            elif p.split('-')[-1] == 'entity':
                tp = 'entity_property'
            else:
                tp = 'string_property'

            constant_dict[p] = dict(
                value=p, type=tp, name=p)

        type_hierarchy = get_simple_type_hierarchy()
        return dict(type_hierarchy=type_hierarchy,
                    func_dict=func_dict,
                    constant_dict=constant_dict)