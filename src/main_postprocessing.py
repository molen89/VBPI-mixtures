import argparse
import os
import utils_postprocessing
import networkx as nx
import numpy as np
import json
import logging

parser = argparse.ArgumentParser()

######### Data arguments
parser.add_argument('--dataset', required=True, help=' DS1 | DS2 | DS3 | DS4 | DS5 | DS6 | DS7 | DS8 ')

######### Model arguments
parser.add_argument('--pathGoldenRunDir', type=str, default='./results/', help=' path to dir with golden runs')
parser.add_argument('--loggerName', type=str, default='postprocessing.log', help=' namn of the logging file')

parser.add_argument('--trimLargest', type=int, default=4096, help=' top largest to save')
parser.add_argument('--trimAttrVBPI', type=str, default="frq", help='frq | lnl | logq')
parser.add_argument('--trimAttrMCMC', type=str, default="frq", help='frq | lnl | logq')
parser.add_argument('--trimCS', type=float, default=0.95, help='credible set')
parser.add_argument('--goldenLimit', default=False, action='store_true', help=' use the golden run as limit of how for vbpi')

parser.add_argument('--maxCluster', type=int, default=8, help=' amount of clusters that are maximum allowed')
parser.add_argument('--n_proc', type=int, default=2, help=' amount of processes allowed')
parser.add_argument('--maxDistance', type=int, default=1, help=' max distance allowed when creating graph')
parser.add_argument('--distanceType', type=str, default="rspr", help=' rf | rspr')
parser.add_argument('--rsprPath', type=str, default="/home/morningstar/Documents/github/rspr/rspr", help=' path to rspr bin')
#n_proc

parser.add_argument('--nexus2unique', default=False, action='store_true', help=' convert nexus to unique')
parser.add_argument('--test', default=False, action='store_true', help=' calculate metrics between goldenrun/vbpi')
parser.add_argument('--graphGoldenRun', default=False, action='store_true', help=' create graph for goldenrun')
parser.add_argument('--graphVBPI', default=False, action='store_true', help=' create graph for VBPI')
parser.add_argument('--graphMixVBPI', default=False, action='store_true', help=' create graph for VBPI')
parser.add_argument('--graphTemplate', default=False, action='store_true', help=' create graph for VBPI')
parser.add_argument('--vbpiUtFromMrbayes', default=False, action='store_true', help=' is the vbpi-ut from sbn support')

# parser.add_argument('--graphClusters', default=False, action='store_true', help=' run clustering algorithm')


parser.add_argument('--resultPath', type=str, default="results/", help=' path to results folder')

parser.add_argument('--dev', default=False, action='store_true', help=' dev')

def main(args):
    print(f"Starting main with args: {args}")

    args.result_folder = os.path.join(args.resultPath, args.dataset)

    LOG_DATEFMT = '%Y-%m-%d %H:%M:%S'
    LOG_FORMAT = ('[%(levelname)s/%(name)s:%(lineno)d] %(asctime)s ' +
                  '(%(processName)s/%(threadName)s)> %(message)s')
    logging.basicConfig(filename=os.path.join(args.result_folder, args.loggerName), filemode="w", level=logging.DEBUG, format=LOG_FORMAT, datefmt=LOG_DATEFMT)
    logging.info("\n---------------------------------\n Starting new run \n---------------------------------")

    if args.nexus2unique:
        logging.info("Converting nexus to newick")
        utils_postprocessing.nexus_2_unique_trees(os.path.join(args.pathGoldenRunDir, args.dataset))

    ## test
    if args.test:
        logging.info("\n---------------------------------\n Starting testing \n---------------------------------")
        logging.info("Calculating metrics between golden run and VBPI")
        goldenrun = utils_postprocessing.UT(os.path.join(args.pathGoldenRunDir, args.dataset))

        if args.vbpiUtFromMrbayes:
            vbpirun = utils_postprocessing.UT()
            vbpirun.load_unique_trees(args.result_folder, postfix="_mix")
            # vbpirun.rebase_ut()

            utils_postprocessing.get_metrics_from_support(goldenrun.unique_trees, vbpirun.unique_trees)
        else:
            vbpirun = utils_postprocessing.UT(args.result_folder)
            utils_postprocessing.get_metrics(goldenrun.unique_trees, vbpirun.unique_trees)


    if args.graphGoldenRun:
        logging.info("\n---------------------------------\n Graphing golden \n---------------------------------")
        golden_run_path = os.path.join(args.pathGoldenRunDir, args.dataset)
        goldenrun = utils_postprocessing.UT(path=golden_run_path)
        goldenrun_unique_trees_trimmed = utils_postprocessing.trim_unique_trees(goldenrun.unique_trees,
                                                                      nlargest=args.trimLargest,
                                                                      trim_attr=args.trimAttrMCMC,
                                                                      credible_set=args.trimCS)
        G = utils_postprocessing.unique_trees_2_graph_cluster(goldenrun_unique_trees_trimmed,
                                                           max_distance=args.maxDistance,
                                                           rspr_path=args.rsprPath, attr=args.trimAttrMCMC,
                                                           max_cluster=args.maxCluster,
                                                           n_proc=args.n_proc)
        G = nx.convert_node_labels_to_integers(G)
        utils_postprocessing.save_graph(G, golden_run_path)


    if args.graphVBPI:
        logging.info("\n---------------------------------\n Graphing vbpi \n---------------------------------")
        if goldenrun and args.goldenLimit:
            if not 'goldenrun_unique_trees_trimmed' in globals():
                logging.info("Trimming golden to find limit")
                goldenrun_unique_trees_trimmed = utils_postprocessing.trim_unique_trees(goldenrun.unique_trees,
                                                                              nlargest=args.trimLargest,
                                                                              trim_attr=args.trimAttrMCMC,
                                                                              credible_set=args.trimCS)
            args.trimLargest = len(goldenrun_unique_trees_trimmed)
            args.trimCS = 1.0

        logging.info("Creating graph of VBPI samples with clusters")
        vbpi_run = utils_postprocessing.UT(args.result_folder)
        unique_trees = utils_postprocessing.trim_unique_trees(vbpi_run.unique_trees, nlargest=args.trimLargest, trim_attr=args.trimAttrVBPI, credible_set=args.trimCS)
        G = utils_postprocessing.unique_trees_2_graph_cluster(unique_trees,
                                                              max_distance=args.maxDistance,
                                                              rspr_path=args.rsprPath, attr=args.trimAttrVBPI,
                                                              max_cluster=args.maxCluster, n_proc=args.n_proc)
        G = nx.convert_node_labels_to_integers(G)
        utils_postprocessing.save_graph(G, args.result_folder)

    if args.graphTemplate:
        path = args.result_folder + f'/unique_trees_mix.json'
        logging.info(f"Loading unique trees: {path}")
        with open(path) as json_unique_trees:
            unique_trees_template = json.load(json_unique_trees)
        utils_postprocessing.support_2_graph(unique_trees_template, args.rsprPath, args.result_folder, args.n_proc)


    if args.graphMixVBPI:
        # get template
        G = utils_postprocessing.load_graph(args.result_folder, postfix="_mix_template", trimmed=True)
        path = args.result_folder + f'/unique_trees_mix.json'
        logging.info(f"Loading unique trees: {path}")
        with open(path) as json_unique_trees:
            unique_trees_mix = json.load(json_unique_trees)
        logging.info(f"Finished loading unique trees, len={len(unique_trees_mix)}")
        utils_postprocessing.unique_trees_mix_2_graph(G, unique_trees_mix, args.result_folder)


    logging.info("Finished")
    logging.shutdown()

import sys
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)