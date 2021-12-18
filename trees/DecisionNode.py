import numpy as np
from typing import Callable, Any, Iterable, Union


class DecisionNode:
    def __init__(self, n_feature: int, decision_rule: Callable[[Any], bool], decision_repr: str, yes_result = None, no_result = None):
        self._n_feature = n_feature
        self._rule = decision_rule
        self._rule_repr = decision_repr
        self._yes_result = yes_result
        self._no_result = no_result
        self._yes_child: 'DecisionNode' = None
        self._no_child: 'DecisionNode' = None
        self._ancestors: set['DecisionNode'] = set()

    def decide(self, x: np.array):
        #print("\nProbe:", x, self._n_feature, x[self._n_feature], self._rule_repr)
        if self._rule(x[self._n_feature]):
            #print("YES, ", end='')
            if self._yes_child is None:
                #print("RESULT")
                return self._yes_result
            else:
                #print("CHILD")
                return self._yes_child.decide(x)
        else:
            #print("NO, ", end='')
            if self._no_child is None:
                #print("RESULT")
                return self._no_result
            else:
                #print("CHILD")
                return self._no_child.decide(x)

    def add_ancestor(self, other: 'DecisionNode'):
        self._ancestors.add(other)

    def set_yes_child(self, child: 'DecisionNode'):
        if (child not in self.ancestors) and (child != self):
            self._yes_child = child
            child.add_ancestor(self)

    def set_no_child(self, child: 'DecisionNode'):
        if (child not in self.ancestors) and (child != self):
            self._no_child = child
            child.add_ancestor(self)

    def set_yes_result(self, result):
        self._yes_result = result

    def set_no_result(self, result):
        self._no_result = result

    @property
    def ancestors(self) -> set['DecisionNode']:
        return self._ancestors.copy()

    def as_dict(self):
        if self._yes_child is not None:
            yes = self._yes_child.as_dict()
        else:
            yes = self._yes_result

        if self._no_child is not None:
            no = self._no_child.as_dict()
        else:
            no = self._no_result

        result = {
            self.__repr__(): (yes, no)
        }

        return result

    def __iter__(self) -> Iterable[tuple['DecisionNode', Union[Any, 'DecisionNode'], Union[Any, 'DecisionNode']]]:
        yes = self._yes_result if self._yes_result is not None else self._yes_child
        no = self._no_result if self._no_result is not None else self._no_child
        yield self, yes, no

        if self._yes_result is None and self._yes_child is not None:
            yield from yes

        if self._no_result is None and self._no_child is not None:
            yield from no

    def copy(self) -> 'DecisionNode':
        new = DecisionNode(self._n_feature, self._rule, self._rule_repr)
        new._yes_child = self._yes_child if self._yes_child is not None else None
        new._no_child = self._no_child if self._no_child is not None else None
        new._yes_result = self._yes_result
        new._no_result = self._no_result

        return new

    def __hash__(self):
        return hash((id(self), str(type(self))))

    def __eq__(self, other: 'DecisionNode'):
        return hash(self) == hash(other)

    def __repr__(self) -> str:
        return f"[{self._rule_repr}]".replace("x", f"x{self._n_feature}")

    def __str__(self) -> str:
        yes = self._yes_result if self._yes_result is not None else self._yes_child
        no = self._no_child if self._no_result is not None else self._no_child
        return f"[{self._n_feature}:{self._rule_repr}]: yes -> {yes}, no-> {no}"
