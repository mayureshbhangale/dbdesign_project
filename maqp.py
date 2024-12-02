import argparse
import logging
import os
import shutil
import time
import subprocess

import numpy as np

from rspn.code_generation.generate_code import generate_ensemble_code
from data_preparation.join_data_preparation import prepare_sample_hdf
from data_preparation.prepare_single_tables import prepare_all_tables
from ensemble_compilation.spn_ensemble import read_ensemble
from ensemble_creation.naive import create_naive_all_split_ensemble, naive_every_relationship_ensemble
from ensemble_creation.rdc_based import candidate_evaluation
from aqp_spn.util import initialutils


np.random.seed(1)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--result', help='Run the script to load required utils', action='store_true')
    
    
    parser.add_argument('--dataset', default='fligths', help='Which dataset to be used')

    
    # generate hdf
    parser.add_argument('--generate_hdf', help='Prepare hdf5 files for single tables', action='store_true')
    parser.add_argument('--generate_sampled_hdfs', help='Prepare hdf5 files for single tables', action='store_true')
    parser.add_argument('--csv_seperator', default='|')
    parser.add_argument('--csv_path', default='../flight-benchmarks')
    parser.add_argument('--hdf_path', default='../flights-benchmarks/gen_hdf')
    parser.add_argument('--max_rows_per_hdf_file', type=int, default=20000000)
    parser.add_argument('--hdf_sample_size', type=int, default=1000000)

    # generate ensembles
    parser.add_argument('--generate_ensemble', help='Trains SPNs on schema', action='store_true')
    parser.add_argument('--ensemble_strategy', default='single')
    parser.add_argument('--ensemble_path', default='../flights-benchmarks/spn_ensembles')
    parser.add_argument('--pairwise_rdc_path', default=None)
    parser.add_argument('--samples_rdc_ensemble_tests', type=int, default=10000)
    parser.add_argument('--samples_per_spn', help="How many samples to use for joins with n tables",
                        nargs='+', type=int, default=[10000000, 10000000, 2000000, 2000000])
    parser.add_argument('--post_sampling_factor', nargs='+', type=int, default=[30, 30, 2, 1])
    parser.add_argument('--rdc_threshold', help='If RDC value is smaller independence is assumed', type=float,
                        default=0.3)
    parser.add_argument('--bloom_filters', help='Generates Bloom filters for grouping', action='store_true')
    parser.add_argument('--ensemble_budget_factor', type=int, default=5)
    parser.add_argument('--ensemble_max_no_joins', type=int, default=3)
    parser.add_argument('--incremental_learning_rate', type=int, default=0)
    parser.add_argument('--incremental_condition', type=str, default=None)

    # generate code
    parser.add_argument('--code_generation', help='Generates code for trained SPNs for faster Inference',
                        action='store_true')
    parser.add_argument('--use_generated_code', action='store_true')

    # ground truth
    parser.add_argument('--aqp_ground_truth', help='Computes ground truth for AQP', action='store_true')
    parser.add_argument('--cardinalities_ground_truth', help='Computes ground truth for Cardinalities',
                        action='store_true')

    # evaluation
    parser.add_argument('--evaluate_cardinalities', help='Evaluates SPN ensemble to compute cardinalities',
                        action='store_true')
    parser.add_argument('--rdc_spn_selection', help='Uses pairwise rdc values to for the SPN compilation',
                        action='store_true')
    parser.add_argument('--evaluate_cardinalities_scale', help='Evaluates SPN ensemble to compute cardinalities',
                        action='store_true')
    parser.add_argument('--evaluate_aqp_queries', help='Evaluates SPN ensemble for AQP', action='store_true')
    parser.add_argument('--against_ground_truth', help='Computes ground truth for AQP', action='store_true')
    parser.add_argument('--evaluate_confidence_intervals',
                        help='Evaluates SPN ensemble and compares stds with true stds', action='store_true')
    parser.add_argument('--confidence_upsampling_factor', type=int, default=300)
    parser.add_argument('--confidence_sample_size', type=int, default=10000000)
    parser.add_argument('--ensemble_location', nargs='+',
                        default=['../flights-benchmarks/spn_ensembles/ensemble_single_ssb-500gb_10000000.pkl',
                                 '../flights-benchmarks/spn_ensembles/ensemble_relationships_ssb-500gb_10000000.pkl'])
    parser.add_argument('--query_file_location', default='../flights-benchmarks/sql/cardinality_queries.sql')
    parser.add_argument('--ground_truth_file_location',
                        default='./benchmarks/ssb/sql/cardinality_true_cardinalities_100GB.csv')
    parser.add_argument('--database_name', default=None)
    parser.add_argument('--target_path', default='../flights-benchmarks/results')
    parser.add_argument('--raw_folder', default='../flights-benchmarks/results')
    parser.add_argument('--confidence_intervals', help='Compute confidence intervals', action='store_true')
    parser.add_argument('--max_variants', help='How many spn compilations should be computed for the cardinality '
                                               'estimation. Seeting this parameter to 1 means greedy strategy.',
                        type=int, default=1)
    parser.add_argument('--no_exploit_overlapping', action='store_true')
    parser.add_argument('--no_merge_indicator_exp', action='store_true')

    # evaluation of spn ensembles in folder
    parser.add_argument('--hdf_build_path', default='')

    # log level
    parser.add_argument('--log_level', type=int, default=logging.DEBUG)

    args = parser.parse_args()
    args.exploit_overlapping = not args.no_exploit_overlapping

    args.merge_indicator_exp = not args.no_merge_indicator_exp

    # Setup logging
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler("logs/{}_{}.log".format(args.dataset, time.strftime("%Y%m%d-%H%M%S"))),
            logging.StreamHandler()
        ])
    logger = logging.getLogger(__name__)

    # Generate schema
    table_csv_path = args.csv_path.format(args.dataset)
    # Generate HDF files
    if args.generate_hdf:
        logger.info(f"Generating HDF files in {args.hdf_path}")
        if os.path.exists(args.hdf_path):
            shutil.rmtree(args.hdf_path)
        os.makedirs(args.hdf_path)
        prepare_all_tables(schema, args.hdf_path, csv_seperator=args.csv_seperator,
                           max_table_data=args.max_rows_per_hdf_file)
        logger.info(f"HDF files generated at {args.hdf_path}")

    # Generate sampled HDF files
    if args.generate_sampled_hdfs:
        logger.info(f"Generating sampled HDF files in {args.hdf_path}")
        logger.info(f"Schema tables: {[table.table_name for table in schema.tables]}")
        prepare_sample_hdf(schema, args.hdf_path, args.max_rows_per_hdf_file, args.hdf_sample_size)

    # Generate ensemble
    if args.generate_ensemble:
        if not os.path.exists(args.ensemble_path):
            os.makedirs(args.ensemble_path)

        if args.ensemble_strategy == 'single':
            create_naive_all_split_ensemble(schema, args.hdf_path, args.samples_per_spn[0], args.ensemble_path,
                                            args.dataset, args.bloom_filters, args.rdc_threshold,
                                            args.max_rows_per_hdf_file, args.post_sampling_factor)
        elif args.ensemble_strategy == 'relationship':
            naive_every_relationship_ensemble(schema, args.hdf_path, args.samples_per_spn[1], args.ensemble_path,
                                              args.dataset, args.bloom_filters, args.rdc_threshold,
                                              args.max_rows_per_hdf_file, args.post_sampling_factor)
        elif args.ensemble_strategy == 'rdc_based':
            candidate_evaluation(schema, args.hdf_path, args.samples_rdc_ensemble_tests, args.samples_per_spn,
                                 args.max_rows_per_hdf_file, args.ensemble_path, args.database_name,
                                 args.post_sampling_factor, args.ensemble_budget_factor, args.ensemble_max_no_joins,
                                 args.rdc_threshold, args.pairwise_rdc_path,
                                 incremental_learning_rate=args.incremental_learning_rate,
                                 incremental_condition=args.incremental_condition)
        else:
            raise NotImplementedError

    if args.result:
        
    
        initialutils.utils() 
 
    
    if args.code_generation:
        spn_ensemble = read_ensemble(args.ensemble_path, build_reverse_dict=True)
        generate_ensemble_code(spn_ensemble, floating_data_type='float', ensemble_path=args.ensemble_path)

    
    if args.evaluate_cardinalities:
        from evaluation.cardinality_evaluation import evaluate_cardinalities
        evaluate_cardinalities(args.ensemble_path, args.database_name, args.query_file_location, args.target_path,
                               schema, True, args.pairwise_rdc_path, use_generated_code=args.use_generated_code)
        
    
 
    if args.evaluate_aqp_queries:
        from evaluation.aqp_evaluation import evaluate_aqp_queries

        evaluate_aqp_queries(args.ensemble_location, args.query_file_location, args.target_path, schema,
                             args.ground_truth_file_location, args.rdc_spn_selection, args.pairwise_rdc_path,
                             max_variants=args.max_variants,
                             merge_indicator_exp=args.merge_indicator_exp,
                             exploit_overlapping=args.exploit_overlapping, min_sample_ratio=0, debug=True,
                             show_confidence_intervals=args.confidence_intervals)

    

   

