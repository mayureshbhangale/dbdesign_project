import itertools
import logging
import pickle
from random import randint
from time import perf_counter

import networkx as nx
import numpy as np
from spn.algorithms.splitting.RDC import rdc_cca, rdc_transformer
from spn.structure.Base import Context

from aqp_spn.aqp_spn import AQPSPN
from data_preparation.join_data_preparation import JoinDataPreparator
from ensemble_compilation.physical_db import DBConnection
from ensemble_compilation.spn_ensemble import SPNEnsemble
from ensemble_creation.naive import RATIO_MIN_INSTANCE_SLICE
from ensemble_creation.utils import create_random_join

logger = logging.getLogger(__name__)


def generate_candidate_solution(pairwise_max_rdc, table_index_dict, prep, max_budget, schema, max_no_relationships,
                                 rdc_threshold):
    spn_relationships_list = set()
    all_merged_tables = set()
    learning_costs = 0

    for relationship_obj in schema.relationships:
        relationship_list = [relationship_obj.identifier]
        merged_tables = [relationship_obj.start, relationship_obj.end]

        if candidate_rdc_sum_means(pairwise_max_rdc, table_index_dict,
                                   [(relationship_list, merged_tables)]) > rdc_threshold:
            all_merged_tables.update(merged_tables)
            spn_relationships_list.add((frozenset(relationship_list), frozenset(merged_tables)))

    for table in set([table.table_name for table in schema.tables]).difference(all_merged_tables):
        spn_relationships_list.add((frozenset(), frozenset([table])))

    rejected_candidates = 0
    while rejected_candidates < 5:
        no_joins = randint(2, max_no_relationships)
        relationship_list, merged_tables = create_random_join(schema, no_joins)
        current_costs = learning_cost(prep, [relationship_list])

        if (frozenset(relationship_list), frozenset(merged_tables)) in spn_relationships_list:
            rejected_candidates += 1
            continue

        if candidate_rdc_sum_means(pairwise_max_rdc, table_index_dict,
                                   [(relationship_list, merged_tables)]) <= rdc_threshold:
            rejected_candidates += 1
            continue

        if learning_costs + current_costs > max_budget:
            break

        all_merged_tables.update(merged_tables)
        learning_costs += current_costs
        spn_relationships_list.add((frozenset(relationship_list), frozenset(merged_tables)))

    return frozenset(spn_relationships_list), learning_costs


def learning_cost(prep, spn_relationships_list, single_table=None):
    if single_table is not None:
        assert spn_relationships_list is None, "Specify either single table or relationship list"
        return prep.column_number(single_table=single_table) ** 2

    learning_cost_estimate = sum(
        [prep.column_number(relationship_list=relationship_list) ** 2 for relationship_list in spn_relationships_list])

    return learning_cost_estimate


def candidate_rdc_sum_means(pairwise_max_rdc, table_index_dict, ensemble_candidate):
    rdc_mean_sum = 0
    for relationship_list, merged_tables in ensemble_candidate:
        if len(relationship_list) == 0:
            continue
        included_table_idxs = [table_index_dict[table] for table in merged_tables]
        rdc_vals = [pairwise_max_rdc[(min(left_idx, right_idx), max(left_idx, right_idx))] for left_idx, right_idx
                    in itertools.combinations(included_table_idxs, 2)]
        rdc_mean_sum += sum(rdc_vals) / len(rdc_vals)

    return rdc_mean_sum


def candidate_evaluation(schema, meta_data_path, sample_size, spn_sample_size, max_table_data, ensemble_path,
                         physical_db_name, postsampling_factor, ensemble_budget_factor, max_no_joins, rdc_learn,
                         pairwise_rdc_path, rdc_threshold=0.15, random_solutions=10000, bloom_filters=False,
                         incremental_learning_rate=0, incremental_condition=None):

    assert incremental_learning_rate == 0 or incremental_condition is None
    prep = JoinDataPreparator(meta_data_path + "/meta_data_sampled.pkl", schema, max_table_data=max_table_data)

    table_index_dict = {table.table_name: i for i, table in enumerate(schema.tables)}
    inverse_table_index_dict = {table_index_dict[k]: k for k in table_index_dict.keys()}
    G = nx.Graph()
    for relationship in schema.relationships:
        start_idx = table_index_dict[relationship.start]
        end_idx = table_index_dict[relationship.end]
        G.add_edge(start_idx, end_idx, relationship=relationship)

    all_pairs = dict(nx.all_pairs_shortest_path(G))
    all_pair_list = []
    for left_idx, right_idx_dict in all_pairs.items():
        for right_idx, shortest_path_list in right_idx_dict.items():
            if left_idx >= right_idx:
                continue
            all_pair_list.append((left_idx, right_idx, shortest_path_list,))

    rdc_attribute_dict = dict()
    all_pair_list.sort(key=lambda x: len(x[2]))
    pairwise_max_rdc = {comb: 0 for comb in itertools.combinations(range(len(table_index_dict)), 2)}
    for left_idx, right_idx, shortest_path_list in all_pair_list:
        left_table = inverse_table_index_dict[left_idx]
        right_table = inverse_table_index_dict[right_idx]
        relationship_list = [G[shortest_path_list[i]][shortest_path_list[i + 1]]['relationship'].identifier
                             for i in range(len(shortest_path_list) - 1)]
        df_samples, meta_types, _ = prep.generate_join_sample(
            relationship_list=relationship_list,
            min_start_table_size=1, sample_rate=1.0,
            drop_redundant_columns=True,
            max_intermediate_size=sample_size * postsampling_factor[0])
        rdc_value = max_rdc(schema, left_table, right_table, df_samples, meta_types, rdc_attribute_dict)
        if rdc_value > rdc_threshold:
            pairwise_max_rdc[(left_idx, right_idx)] = rdc_value

    with open(pairwise_rdc_path, 'wb') as f:
        pickle.dump(rdc_attribute_dict, f, pickle.HIGHEST_PROTOCOL)

    budget = ensemble_budget_factor * sum(
        [prep.column_number(single_table=table.table_name) ** 2 for table in schema.tables])

    ensemble_candidates = set([generate_candidate_solution(pairwise_max_rdc, table_index_dict, prep, budget, schema,
                                                           max_no_joins, rdc_threshold) for i in
                               range(random_solutions)])

    candidates = [(ensemble_candidate[0], ensemble_candidate[1],
                   candidate_rdc_sum_means(pairwise_max_rdc, table_index_dict, ensemble_candidate[0])) for
                  ensemble_candidate in ensemble_candidates]
    candidates.sort(key=lambda x: (-x[2], x[1]))
    optimal_candidate = list(candidates[0][0])

    optimal_candidate.sort(key=lambda x: -len(x[1]))
    spn_ensemble = SPNEnsemble(schema)
    prep = JoinDataPreparator(meta_data_path + "/meta_data_sampled.pkl", schema, max_table_data=max_table_data)
    if physical_db_name is not None:
        db_connection = DBConnection(db=physical_db_name)

    for relationship_list, merged_tables in optimal_candidate:
        df_samples, df_inc_samples, meta_types, null_values, full_join_est = prep.generate_n_samples_with_incremental_part(
            spn_sample_size[len(merged_tables) - 1],
            relationship_list=list(relationship_list),
            post_sampling_factor=postsampling_factor[len(merged_tables) - 1],
            incremental_learning_rate=incremental_learning_rate,
            incremental_condition=incremental_condition)

        if len(relationship_list) > 0:
            aqp_spn = AQPSPN(meta_types, null_values, full_join_est, schema,
                             list(relationship_list), full_sample_size=len(df_samples),
                             column_names=list(df_samples.columns), table_meta_data=prep.table_meta_data)
        else:
            aqp_spn = AQPSPN(meta_types, null_values, full_join_est, schema,
                             [], full_sample_size=len(df_samples), table_set={list(merged_tables)[0]},
                             column_names=list(df_samples.columns), table_meta_data=prep.table_meta_data)

        min_instance_slice = RATIO_MIN_INSTANCE_SLICE * min(spn_sample_size[len(merged_tables) - 1], len(df_samples))
        aqp_spn.learn(df_samples.values, min_instances_slice=min_instance_slice, bloom_filters=bloom_filters,
                      rdc_threshold=rdc_learn)
        if incremental_learning_rate > 0 or incremental_condition is not None:
            aqp_spn.learn_incremental(df_inc_samples.values)
        spn_ensemble.add_spn(aqp_spn)

    spn_ensemble.save(ensemble_path)

def max_rdc(schema, left_table, right_table, df_samples, meta_types, rdc_attribute_dict,
            max_sampling_threshold_rows=10000, k=10, s=1 / 6, non_linearity=np.sin, n_jobs=-2, debug=True):
    # only keep columns of left or right table
    irrelevant_cols = []
    relevant_meta_types = []
    for i, column in enumerate(df_samples.columns):
        not_of_left_or_right = not (column.startswith(left_table + '.') or column.startswith(right_table + '.'))
        is_nn_attribute = (column == left_table + '.' + schema.table_dictionary[left_table].table_nn_attribute) or \
                          (column == right_table + '.' + schema.table_dictionary[right_table].table_nn_attribute)
        is_multiplier = False
        is_fk_field = False
        for relationship_obj in schema.relationships:  # [relationship_obj_list[0], relationship_obj_list[-1]]
            if relationship_obj.end + '.' + relationship_obj.end_attr == column or \
                    relationship_obj.start + '.' + relationship_obj.start_attr == column:
                is_fk_field = True
                break

            if relationship_obj.end + '.' + relationship_obj.multiplier_attribute_name_nn == column or \
                    relationship_obj.end + '.' + relationship_obj.multiplier_attribute_name == column:
                is_multiplier = True
                break
        is_uninformative = False

        if not_of_left_or_right or is_nn_attribute or is_multiplier or is_fk_field or is_uninformative:
            irrelevant_cols.append(column)
        else:
            relevant_meta_types.append(meta_types[i])

    df_samples.drop(columns=irrelevant_cols, inplace=True)

    left_column_names = [(i, column) for i, column in enumerate(df_samples.columns) if
                         column.startswith(left_table + '.')]
    right_column_names = [(i, column) for i, column in enumerate(df_samples.columns) if
                          column.startswith(right_table + '.')]
    left_columns = [i for i, column in left_column_names]
    right_columns = [i for i, column in right_column_names]

    data = df_samples.values
    # sample if necessary
    if data.shape[0] > max_sampling_threshold_rows:
        data = data[np.random.randint(data.shape[0], size=max_sampling_threshold_rows), :]

    n_features = data.shape[1]
    assert n_features == len(relevant_meta_types)

    ds_context = Context(meta_types=relevant_meta_types)
    ds_context.add_domains(data)

    rdc_features = rdc_transformer(
        data, relevant_meta_types, ds_context.domains, k=k, s=s, non_linearity=non_linearity, return_matrix=False
    )
    pairwise_comparisons = [(i, j) for i in left_columns for j in right_columns]

    from joblib import Parallel, delayed

    rdc_vals = Parallel(n_jobs=n_jobs, max_nbytes=1024, backend="threading")(
        delayed(rdc_cca)((i, j, rdc_features)) for i, j in pairwise_comparisons
    )

    for (i, j), rdc in zip(pairwise_comparisons, rdc_vals):
        if np.isnan(rdc):
            rdc = 0
        if debug:
            logger.debug(f"{df_samples.columns[i]}, {df_samples.columns[j]}: {rdc}")

    pairwise_comparisons = [(column_left, column_right) for i, column_left in left_column_names for j, column_right in
                            right_column_names]
    for (column_left, column_right), rdc in zip(pairwise_comparisons, rdc_vals):
        rdc_attribute_dict[(column_left, column_right)] = rdc
        rdc_attribute_dict[(column_right, column_left)] = rdc

    return max(rdc_vals)


