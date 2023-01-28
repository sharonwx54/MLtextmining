from src.link_analysis import *


if __name__ == '__main__':
    # loading matrices and probability distributions
    doc_topic_mtx = read_doc_topic_distri_from_file(DOC_TOPIC_PATH)
    trans_mtx, zero_outlink_idx = read_create_transition_mtx_from_file(TRANSITION_PATH)
    query_distri, query_index_map = read_item_topic_distri_from_file(QTSPR_DISTRI_PATH)
    user_distri, user_index_map = read_item_topic_distri_from_file(PTSPR_DISTRI_PATH)

    # loading all search scores into one table
    ir_full_df = read_and_combine_search_results(INDRI_PATH)
    # get the document num
    doc_size = trans_mtx.shape[0]
    # initialize r vector by assigning each cell the same value, sum up to one
    # technically the initial r could be any format
    init_r_vec = np.vstack([1.0/doc_size]*doc_size)

    # RUN GPR to get global PageRank scores
    GPR_vec = global_PR(init_r_vec, trans_mtx, zero_outlink_idx, DAMPENING_FACTOR)
    # RUN QTSPR to get a matrix which combine the TSPR scores for each user-query into
    # one big matrix of n x q, in the order by user-query index
    QTSPR_mtx = full_score_TSPR(
        doc_topic_mtx, query_distri, trans_mtx, zero_outlink_idx, DAMPENING_FACTOR, BETA)
    # RUN PTSPR to get a matrix which combine the TSPR scores for each user-query into
    # one big matrix of n x q, in the order by user-query index
    PTSPR_mtx = full_score_TSPR(
        doc_topic_mtx, user_distri, trans_mtx, zero_outlink_idx, DAMPENING_FACTOR, BETA)

    # write pageRank scores into txt for each PR methods - comment out
    """generate_sample_txt(GPR_vec, QTSPR_mtx, query_index_map, PTSPR_mtx, user_index_map)"""
    # For each of the weighting method, run the full GPR and XTSPR on all user-query pairs
    # Then save the final results in txt files with corresponding name
    for w in COMBINE_METHOD_WT_MAP.keys():
        run_GPR_into_file(GPR_vec, ir_full_df, w)
        run_TSPR_into_file("QTSPR", QTSPR_mtx, query_index_map, ir_full_df, w)
        run_TSPR_into_file("PTSPR", PTSPR_mtx, user_index_map, ir_full_df, w)


