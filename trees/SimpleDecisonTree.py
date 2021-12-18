import numpy as np
import pydot
from .DecisionNode import DecisionNode


class SimpleDecisionTree:
    def __init__(self, feature_split_intervals: int = 100, accuracy: float = 0.7):
        self._n_samples: int = 0
        self._accuracy = accuracy
        self._n_features: int = 0
        self._feature_split_intervals: int = feature_split_intervals
        self._n_classes: np.array = None
        self._feature_intervals: np.array = None
        self._feature_interval_class_amount: np.array = None
        self._feature_average_class_amount: np.array = None
        self._x: np.array = None
        self._y: np.array = None
        self._decision_tree: DecisionNode = DecisionNode(0, lambda x: False, "start_node")

    def fit(self, x: np.array, y: np.array) -> None:
        self._x = np.copy(x)
        self._y = np.copy(y)
        self._n_samples, self._n_features = x.shape
        self._n_classes = len(np.unique(y))
        self._feature_intervals = np.array([[[0, 0]]*self._feature_split_intervals for f in range(self._n_features)])

        self._generate_features_class_amount()
        self._generate_decision_tree()

    def predict(self, x: np.array):
        return self._decision_tree.decide(x)

    def make_dot(self, file="tree.png"):
        data = self.as_dict()
        graph = pydot.Dot()

        counter = 0
        for this, yes, no in self:
            counter += 1
            this_name = str(hash(this)) if isinstance(this, DecisionNode) else f"{counter}_{this}"
            yes_name = str(hash(yes)) if isinstance(yes, DecisionNode) else f"{counter}_{yes}"
            no_name = str(hash(no)) if isinstance(no, DecisionNode) else f"{counter}_{no}"
            this_node = pydot.Node(this_name, label=this.__repr__())
            yes_node = pydot.Node(yes_name, label=yes.__repr__())
            no_node = pydot.Node(no_name, label=no.__repr__())
            graph.add_node(this_node)
            graph.add_node(yes_node)
            graph.add_node(no_node)
            graph.add_edge(pydot.Edge(this_node, yes_node, label="yes"))
            graph.add_edge(pydot.Edge(this_node, no_node, label="no"))

        graph.write(file, format="png")

    def as_dict(self):
        return self._decision_tree.as_dict()


    def _generate_features_class_amount(self):
        self._feature_interval_class_amount = np.array([[[0]*self._n_classes]*self._feature_split_intervals]*self._n_features)
        self._feature_average_class_amount = np.array([0]*self._n_features)
        for f in range(self._n_features):
            interval_start = np.min(self._x[:, f])
            interval_end = np.max(self._x[:, f])
            interval_step = (interval_end-interval_start)/self._feature_split_intervals
            for i in range(self._feature_split_intervals):
                i_start = interval_start+i*interval_step
                i_end = interval_start+(i+1)*interval_step
                self._feature_intervals[f][i][0] = i_start
                self._feature_intervals[f][i][1] = i_end
                i_samples = np.where(np.all([self._x[:, f] >= i_start, self._x[:, f] <= i_end], axis=0))[0]
                i_classes = self._y[i_samples]
                for c in i_classes:
                    self._feature_interval_class_amount[f][i][c] = len(np.where(i_classes == c))
            feature_interval_classes_amount = np.unique(np.where(self._feature_interval_class_amount[f]), return_counts=True)[1]
            self._feature_average_class_amount[f] = np.mean(feature_interval_classes_amount)

    def _generate_decision_tree(self):
        def lambda_between(istart, iend):
            return lambda x: istart <= x <= iend
        def lambda_less(i):
            return lambda x: x <= i
        def lambda_bigger(i):
            return lambda x: i <= x

        sorted_feature_indexes = sorted(enumerate(self._feature_average_class_amount), key=lambda x: x[1])
        node = self._decision_tree
        lost_node: DecisionNode = None
        backup_nodes: list[DecisionNode] = []
        for f, class_amount in sorted_feature_indexes:
            prev_i_start = 9999999999999999
            prev_i_end = -99999999999999
            for i, classes in enumerate(self._feature_interval_class_amount[f]):
                total_samples = np.sum(classes)
                max_class_index = np.argmax(classes)
                max_class_samples = classes[max_class_index]

                if lost_node is not None:
                    node = lost_node
                    lost_node = None

                i_start = min(prev_i_start, self._feature_intervals[f][i][0])
                i_end = max(prev_i_end, self._feature_intervals[f][i][1])
                if i == 0:
                    new_node = DecisionNode(f, lambda_less(i_end), f"x<={i_end}", yes_result=max_class_index)
                elif i >= len(self._feature_interval_class_amount[f])-1:
                    new_node = DecisionNode(f, lambda_bigger(i_start), f"{i_start}<=x", yes_result=max_class_index)
                else:
                    new_node = DecisionNode(f, lambda_between(i_start, i_end), f"{i_start}<=x<={i_end}", yes_result=max_class_index)

                if total_samples > 0 and max_class_samples/total_samples >= self._accuracy:
                    prev_i_start = self._feature_intervals[f][i][0]
                    prev_i_end = self._feature_intervals[f][i][1]
                    node.set_no_child(new_node)
                    node = new_node
                    for bn in backup_nodes:
                        bn.set_no_child(new_node.copy())
                    backup_nodes = []
                else:
                    last = min(len(sorted_feature_indexes)-1, 3)
                    for try_f, _ in sorted_feature_indexes[:last]:
                        if try_f == f:
                            continue
                        try_f_interval_class_amount = self._feature_interval_class_amount[try_f]
                        try_f_our_clas_present_index = enumerate(try_f_interval_class_amount[:, max_class_index])
                        try_f_our_class_sorted = sorted(try_f_our_clas_present_index, key=lambda x: x[1], reverse=True)
                        wanted_intervals_amount = int(len(try_f_our_class_sorted)*(1-self._accuracy))+1
                        additional_intervals = try_f_our_class_sorted[:wanted_intervals_amount]
                        if list(filter(lambda x: x[1] > 0, additional_intervals)):
                            node.set_no_child(new_node)
                            node = new_node
                            for bn in backup_nodes:
                                bn.set_no_child(new_node.copy())
                            backup_nodes = []
                            lost_node = new_node
                        for try_i, try_amount in additional_intervals:
                            if try_amount > 0:
                                node.set_yes_result(None)
                                i_start = self._feature_intervals[try_f][try_i][0]
                                i_end = self._feature_intervals[try_f][try_i][1]
                                try_node = DecisionNode(try_f, lambda_between(i_start, i_end), f"{i_start}<=x<={i_end}", yes_result=max_class_index)
                                node.set_yes_child(try_node)
                                node = try_node
                                backup_nodes.append(try_node)

    def __iter__(self):
        yield from self._decision_tree
