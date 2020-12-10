import pandas as pd
import numpy as np
import re


''' Default functions for plain operations '''


def plain_rational(args):
    assert len(args) == 2, "Rational operation only works for two arguments"
    return "(" + args[0] + "*1.0/" + args[1] + ")"


def plain_plus(args):
    expr = args[0]
    for arg in args[1:]:
        expr += " + " + arg
    return "(" + expr + ")"


def plain_times(args):
    expr = args[0]
    for arg in args[1:]:
        expr += " * " + arg
    return "(" + expr + ")"


def plain_power(args):
    assert len(args) == 2, "Power operation only works for two arguments"
    return "(" + "pow(" + args[0] + ", " + args[1] + ")" + ")"


''' Class for parsing mathematica code into a tree of operations '''


class FlowEquationParser:
    # Default properties
    constants = ["Pi"]
    operations = ["Plus", "Times", "Rational", "Power"]
    associative_operations = ["Plus", "Times"]
    plain_operations = {"Plus": plain_plus, "Times": plain_times, "Rational": plain_rational, "Power": plain_power}
    
    # Further static variables
    time_variable = None
    coupling_appendix = None
    couplings = []

    num_inter_expr = 0

    def __init__(self, **kwargs):
        self.expression_kind = kwargs.pop("expression_kind")

        if self.expression_kind == "flow_equation":
            self.lhs, self.expression_to_evaluate = kwargs.pop("equality_content").split(",", maxsplit=1)
            self.expression_to_evaluate = self.expression_to_evaluate[:-1]
            self.lhs = self.lhs[1:]
        elif self.expression_kind == "jacobian":
            self.expression_to_evaluate = kwargs.pop("jacobian_element")

        self.max_num_of_terms = kwargs.pop("max_num_of_terms", 2)
        self.max_num_of_term_operations = kwargs.pop("max_num_of_term_operations", 2)

        self.id = 0
        self.operation_tree = {'id': [], 'operation': [], 'value': [], 'child_ids': [], 'depth': []}
        self.operation_tree_dataframe = None

    ''' Operations on the left hand side of the flow equation '''

    # Stores the time variable and the couplings in the corresponding static variables
    def extract_time_variable_and_couplings(self):
        variable_indices = [(m.start(0), m.end(0)) for m in re.finditer("\[[^]]{1,10}]", self.lhs)]

        n_derivate_indices = variable_indices[0]
        assert self.lhs[n_derivate_indices[0] + 1:n_derivate_indices[1] - 1] == '1', \
            "Not 1-th derivative in flow equation makes no sense"

        coupling_indices = variable_indices[1]
        coupling = self.lhs[coupling_indices[0] + 1:coupling_indices[1] - 1]
        if coupling not in FlowEquationParser.couplings:
            FlowEquationParser.couplings.append(coupling)

        time_indices = variable_indices[2]
        time_variable = self.lhs[time_indices[0] + 1:time_indices[1] - 1]
        if FlowEquationParser.time_variable is None:
            FlowEquationParser.time_variable = time_variable
            FlowEquationParser.coupling_appendix = "[" + FlowEquationParser.time_variable + "]"
        assert FlowEquationParser.time_variable == time_variable, "Time variable has changed"

    ''' Generate a tree from the right hand side of the equation where all operations are listed separately '''

    # Main function for the tree generation
    def generated_binary_tree(self):
        # Initialize
        self.id = 0
        self.operation_tree = {'id': [], 'operation': [], 'value': [], 'child_ids': [], 'depth': []}

        left_brace, right_brace = FlowEquationParser.find_actual_left_and_right_brace(expr=self.expression_to_evaluate)
        if left_brace == -1:
            # Theres is no need to evaluate binary operations
            self.add_binary_operation(0, None, self.expression_to_evaluate, None, 0)
        else:
            # Fill binary operations
            self.get_scoped_expression(self.expression_to_evaluate)

        # Transform binary tree to pandas DataFrame
        self.operation_tree_dataframe = pd.DataFrame(self.operation_tree)
        self.operation_tree_dataframe = self.operation_tree_dataframe.set_index("id")
        self.operation_tree_dataframe = self.operation_tree_dataframe.assign(eff_term_occur=None)
        self.operation_tree_dataframe = self.operation_tree_dataframe.assign(eff_terms=None)
        self.operation_tree_dataframe = self.operation_tree_dataframe.assign(type=None)

        # Check for correct generation
        self.check_for_tree_consistency()

    # Main recursive function for the tree generation
    def get_scoped_expression(self, expr, depth=1):
        operation, expr = expr.split("[", 1)
        child_ids = []
        while expr != "":
            left_brace, right_brace = FlowEquationParser.find_actual_left_and_right_brace(expr=expr)

            # Sub operation has been found -> e.g. for [Rational; Rational; ,Rational
            if (right_brace > left_brace) and (left_brace != -1) and \
                    (expr[:left_brace] in FlowEquationParser.operations or
                     expr[1:left_brace] in FlowEquationParser.operations or
                     expr[2:left_brace] in FlowEquationParser.operations):
                # print("Operation", operation)
                expr, child_id = self.get_scoped_expression(expr, depth + 1)
                child_ids.append(child_id)
                # print("Remaining expression", expr, depth)
            else:
                next_seperator = expr[1:].find(",")
                # Equation needs to be finalized before splitting
                if right_brace < next_seperator or next_seperator == -1:
                    child_expression = expr[:right_brace + 1]
                    expr = expr[right_brace + 1:]
                # Regular child is produced
                else:
                    seperate_expressions = expr.split(",", 1)
                    child_expression, expr = seperate_expressions

                _, child_right_brace = FlowEquationParser.find_actual_left_and_right_brace(child_expression)

                # Equation needs to be finalized
                if child_right_brace == right_brace:
                    # Equation only needs to be finalized -> no further childs are present
                    if len(child_expression[:-1]) > 0:
                        self.add_binary_operation(self.id, None, child_expression[:-1], None, depth)
                        child_ids.append(self.id)
                        self.id += 1

                    self.add_binary_operation(self.id, operation, None, child_ids, depth - 1)
                    self.id += 1

                    remaining_expression = expr
                    return remaining_expression, self.id - 1
                # Regular child is appended
                else:
                    self.add_binary_operation(self.id, None, child_expression, None, depth)
                    child_ids.append(self.id)
                    self.id += 1

        # Finalize on operation level
        self.add_binary_operation(self.id, operation, None, child_ids, depth)
        self.id += 1
        remaining_expression = expr[right_brace + 1:]
        return remaining_expression, self.id - 1

    # Helper function for finding the first left and right bracket whereby coupling expressions, i.e. a[k] are ignored
    @staticmethod
    def find_actual_left_and_right_brace(expr):
        coupling_positions = np.array(
            [(m.start(0), m.end(0)) for m in re.finditer("\[" + FlowEquationParser.time_variable + "]", expr)]
        ).transpose()

        if len(coupling_positions) > 0:
            start_ids = coupling_positions[0]
            end_ids = coupling_positions[1] - 1

            left_offset = 0
            while left_offset - 1 in start_ids or left_offset == 0:
                left_brace = expr[left_offset:].find("[")
                if left_brace == -1:
                    break
                left_offset += left_brace + 1
            if left_brace != -1:
                left_brace = left_offset - 1

            right_offset = 0
            while right_offset - 1 in end_ids or right_offset == 0:
                right_brace = expr[right_offset:].find("]")
                if right_brace == -1:
                    break
                right_offset += right_brace + 1
            if right_brace != -1:
                right_brace = right_offset - 1
        else:
            left_brace = expr.find("[")
            right_brace = expr.find("]")

        return left_brace, right_brace

    # Helper function for appending a new row to the tree
    def add_binary_operation(self, id, operation, value, child_ids, depth):
        if operation is not None:
            operation = operation.strip()
        if value is not None:
            value = value.strip()
            if "Pi" in value:
                value = value.replace("Pi", "M_PI")

        self.operation_tree["id"].append(id)
        self.operation_tree["operation"].append(operation)
        self.operation_tree["value"].append(value)
        self.operation_tree["child_ids"].append(child_ids)
        self.operation_tree["depth"].append(depth)

        # self.operation_tree_dataframe = pd.DataFrame(self.operation_tree)
        # self.operation_tree_dataframe = self.operation_tree_dataframe.set_index("id")

    ''' Contract coupling free operations two a single expression and determine when an intermediate storage
    of results is necessary '''

    # Main function for the contractions
    def contract_operations(self):
        # Contract operations
        tree_root_index = self.operation_tree_dataframe.query("depth == 0").iloc[0].name
        self.count_and_contract_actual_terms_in_operations(tree_index=tree_root_index)

        # Drop contracted rows:
        indices_to_drop = self.operation_tree_dataframe.query("value != value and operation != operation").index
        self.operation_tree_dataframe = self.operation_tree_dataframe.drop(indices_to_drop)
        print("Contraction dropped", len(indices_to_drop), "rows")

        # Update depth column and check for consistency
        tree_root_index = self.operation_tree_dataframe.query("depth == 0").iloc[0].name
        self.update_depth_factor(tree_root_index, 0)
        self.check_for_tree_consistency()

    # Main recursive function for the contraction
    def count_and_contract_actual_terms_in_operations(self, tree_index):
        # 'effective_terms' contains dependencies on couplings and intermediate expressions
        # -> for the computation of the node one has to iterate over these expressions 
        effective_terms = []
        current_entry = self.operation_tree_dataframe.loc[tree_index]

        # Found leaf of the tree
        if current_entry.value is not None:
            # Cut coupling appendix ([k]) for a reasonable comparison with couplings
            val = current_entry.value[:-len(FlowEquationParser.coupling_appendix)]

            # Check if a coupling or a numeric constant has been found
            if val in FlowEquationParser.couplings:
                self.operation_tree_dataframe.at[tree_index, "type"] = "coupling"
                return [val]
            else:
                self.operation_tree_dataframe.at[tree_index, "type"] = "numeric constant"
                self.operation_tree_dataframe.at[tree_index, "value"] = FlowEquationParser.strip_mathematica_precision_appendix(self.operation_tree_dataframe.at[tree_index, "value"])
                return []
        else:
            # Collect dependencies on couplings and intermediate expressions (terms) from branches of the current node
            term_free_branches = []  # contains branches that depend only on numeric or constant expressions
            for child_id in current_entry.child_ids:
                # Recursive call
                actual_terms = self.count_and_contract_actual_terms_in_operations(tree_index=child_id)
                effective_terms += actual_terms
                if len(actual_terms) == 0:
                    term_free_branches.append(child_id)

            # Sub tree is free of couplings -> Can be contracted to a constant expression
            if len(effective_terms) == 0:
                self.contract_to_constant_expression(tree_index=tree_index, current_entry=current_entry)
            # Sub tree is partly free of couplings -> Can be partly contracted to a constant expression
            elif len(term_free_branches) >= 2:
                new_child_ids = self.partly_contract_to_constant_expression(
                    tree_index=tree_index, current_entry=current_entry, term_free_branches=term_free_branches)
                self.operation_tree_dataframe.at[tree_index, "child_ids"] = new_child_ids
                current_entry = self.operation_tree_dataframe.loc[tree_index] # To take care of changes

            # Operations of branches contain to many dependencies on couplings and expressions
            # -> Convert branches with coupling operations into intermediate expressions
            if len(set(effective_terms)) > self.max_num_of_terms or \
                    len(effective_terms) > self.max_num_of_term_operations:
                effective_terms = self.mark_branches_as_intermediate_expressions(current_entry=current_entry)

            # The operation itself depends on two many terms
            # -> Split node if possible (depends on operation) into sub trees
            if (len(set(effective_terms)) > self.max_num_of_terms or
                len(effective_terms) > self.max_num_of_term_operations) and \
                    current_entry.operation in FlowEquationParser.associative_operations:
                # Split node until conditions on effective terms are fulfilled
                while len(set(effective_terms)) > self.max_num_of_terms or \
                        len(effective_terms) > self.max_num_of_term_operations:
                    current_entry, effective_terms = self.split_node_into_subtrees(tree_index=tree_index,
                                                                                   current_entry=current_entry)

            # Add information about effective terms to tree
            self.operation_tree_dataframe.at[tree_index, "eff_term_occur"] = len(effective_terms)
            self.operation_tree_dataframe.at[tree_index, "eff_terms"] = set(effective_terms)
            return effective_terms

    # Helper function for contracting branches and converting current entry to a constant expression
    def contract_to_constant_expression(self, tree_index, current_entry):
        operation, child_ids = current_entry[["operation", "child_ids"]].values
        self.operation_tree_dataframe.at[tree_index, "value"] = self.contract_numeric_operation(
            operation=operation, child_ids=child_ids)
        self.operation_tree_dataframe.at[tree_index, "child_ids"] = None
        self.operation_tree_dataframe.at[tree_index, "operation"] = None
        self.operation_tree_dataframe.at[tree_index, "type"] = "constant expression"

    # Helper function for partly contracting branches and adapting current entry to new contracted constant epxression
    def partly_contract_to_constant_expression(self, tree_index, current_entry, term_free_branches):
        # Generate a constant expression from term free branches
        new_value = self.contract_numeric_operation(
            operation=current_entry["operation"],
            child_ids=term_free_branches
        )
        # Add constant expression as new child to tree
        self.operation_tree_dataframe = self.operation_tree_dataframe.append(
            {"child_ids": None, "depth": current_entry.depth + 1, "operation": None, "value": new_value,
             "eff_term_occur": 0, "eff_terms": set(), "type": "constant expression"}, ignore_index=True)
        # Adapt child ids of current entry
        new_child_index = self.operation_tree_dataframe.iloc[-1].name
        new_child_ids = [item for item in current_entry.child_ids if item not in term_free_branches] + [new_child_index]
        return new_child_ids

    # Helper function for contracting branches
    def contract_numeric_operation(self, operation, child_ids):
        args = []
        for child_id in child_ids:
            arg = FlowEquationParser.strip_mathematica_precision_appendix(self.operation_tree_dataframe.loc[child_id].value)
            args.append(arg)
            self.operation_tree_dataframe.at[child_id, "value"] = None
        return FlowEquationParser.plain_operations[operation](args)

    # Helper function for marking branches as intermediate expressions.
    # Returns the adpated effective terms of the current entry
    def mark_branches_as_intermediate_expressions(self, current_entry):
        effective_terms = []
        # Iterate over childs and mark childs with operations as intermediate results
        for child_id in current_entry.child_ids:
            if self.operation_tree_dataframe.loc[child_id, "operation"] is not None:
                self.operation_tree_dataframe.at[child_id, "type"] = "intermediate expression"
                self.operation_tree_dataframe.at[child_id, "value"] = "inter_expr_" + str(
                    FlowEquationParser.num_inter_expr)
                effective_terms.append("inter_expr_" + str(FlowEquationParser.num_inter_expr))
                FlowEquationParser.num_inter_expr += 1
            elif self.operation_tree_dataframe.loc[child_id, "type"] == "coupling":
                effective_terms.append(
                    self.operation_tree_dataframe.loc[child_id, "value"][:-len(FlowEquationParser.coupling_appendix)])
        return effective_terms

    # Helper function for splitting the operation into a subtrees.
    # Returns the adapted effective terms of current entry
    def split_node_into_subtrees(self, tree_index, current_entry):
        effective_terms = []
        overall_new_child_indices = []  # List for collecting new child indices
        # Important: This for loop iterates over the original child ids of the current considered entry
        previous_child_index = 0  # Stores the first index of the effective remaining child ids

        # Iterate over childs and generate new child if the conditions on the effective term is no longer fulfilled
        for idx, child_id in enumerate(current_entry.child_ids):
            # Collect effective terms
            if self.operation_tree_dataframe.loc[child_id, "type"] == "intermediate expression":
                effective_terms.append(self.operation_tree_dataframe.loc[child_id, "value"])
            elif self.operation_tree_dataframe.loc[child_id, "type"] == "coupling":
                effective_terms.append(
                    self.operation_tree_dataframe.loc[child_id, "value"][:-len(FlowEquationParser.coupling_appendix)])

            # Split node if necessary
            if len(set(effective_terms)) == self.max_num_of_terms or \
                    len(effective_terms) == self.max_num_of_term_operations:
                # Extract the latest child ids
                sub_node_child_ids = current_entry.child_ids[previous_child_index:idx + 1]
                # Add new child
                self.operation_tree_dataframe = self.operation_tree_dataframe.append(
                    {"child_ids": sub_node_child_ids, "depth": current_entry.depth + 1,
                     "operation": current_entry.operation,
                     "value": "inter_expr_" + str(FlowEquationParser.num_inter_expr),
                     "eff_term_occur": len(effective_terms),
                     "eff_terms": set(effective_terms),
                     "type": "intermediate expression"},
                    ignore_index=True
                )
                FlowEquationParser.num_inter_expr += 1

                # Store new child index
                overall_new_child_indices.append(self.operation_tree_dataframe.iloc[-1].name)

                # Reset effective terms and set previous child index
                effective_terms = []
                previous_child_index = idx + 1

        # Update child indices of current entry
        self.operation_tree_dataframe.at[tree_index, "child_ids"] = current_entry.child_ids[
                                                                    previous_child_index:] + overall_new_child_indices

        # Update current entry and effective terms
        current_entry = self.operation_tree_dataframe.loc[tree_index]
        effective_terms = []
        for idx, child_id in enumerate(current_entry.child_ids):
            if self.operation_tree_dataframe.loc[child_id, "type"] == "intermediate expression":
                effective_terms.append(self.operation_tree_dataframe.loc[child_id, "value"])
            elif self.operation_tree_dataframe.loc[child_id, "type"] == "coupling":
                effective_terms.append(
                    self.operation_tree_dataframe.loc[child_id, "value"][:-len(FlowEquationParser.coupling_appendix)])

        return current_entry, effective_terms

    ''' Further helper functions '''

    # Updates the depth factors within the tree
    def update_depth_factor(self, tree_index, depth):
        self.operation_tree_dataframe.at[tree_index, "depth"] = depth
        if self.operation_tree_dataframe.loc[tree_index, "child_ids"] is not None:
            for child_id in self.operation_tree_dataframe.loc[tree_index, "child_ids"]:
                self.update_depth_factor(child_id, depth + 1)

    # Checks if the child ids within tree comply with the ids of the tree
    def check_for_tree_consistency(self):
        ids = []
        for elem in self.operation_tree_dataframe.child_ids.values:
            if elem is not None:
                ids += elem
        ids = np.sort(ids)
        if not np.array_equal(ids, self.operation_tree_dataframe.query("depth != 0").index.values):
            assert False, "An error occurred during the generation of the binary operation tree"

    def get_operation_tree(self):
        return self.operation_tree_dataframe

    @staticmethod
    def strip_mathematica_precision_appendix(arg):
        if arg.find(".`") != -1:
            return arg[:-2]
        elif arg.find("`") != -1:
            return arg[:-1]
        else:
            return arg

    @staticmethod
    def get_time_variable_and_couplings():
        return FlowEquationParser.time_variable, FlowEquationParser.couplings

    @staticmethod
    def clear_static_variables():
        FlowEquationParser.time_variable = None
        FlowEquationParser.coupling_appendix = None
        FlowEquationParser.couplings = []
