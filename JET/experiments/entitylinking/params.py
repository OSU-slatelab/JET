class EntityComboMethods:
    Sum = 'sum'
    Multiply = 'multiply'

class LLParams:
    def __init__(self, ctx_vocab_size=0, ctx_dim=0,
            entity_vocab_size=0, entity_dim=0,
            secondary_entity_vocab_size=0, secondary_entity_dim=0,
            window_size=0, max_num_entities=0, max_mention_size=0,
            entity_combo_method=EntityComboMethods.Sum,
            using_mention=False):
        self._ctx_vocab_size = ctx_vocab_size
        self._ctx_dim = ctx_dim
        self._entity_vocab_size = entity_vocab_size
        self._entity_dim = entity_dim
        self._secondary_entity_vocab_size = secondary_entity_vocab_size
        self._secondary_entity_dim = secondary_entity_dim
        self._window_size = window_size
        self._max_num_entities = max_num_entities
        self._max_mention_size = max_mention_size
        self._entity_combination = entity_combo_method
        self._using_mention = using_mention
