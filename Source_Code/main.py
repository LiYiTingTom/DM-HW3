import utils
import time
import algos
import config

if __name__ == '__main__':
    # Arguments parsing.
    s = time.time()
    args = utils.parse_argurments()
    in_file = args.in_file
    iteration = args.iter

    # Get graph object.
    graph = utils.Graph.init(in_file)
    print('read file from:', in_file)

    # HITS
    hits = algos.HITS(
        graph=graph,
        iteration=iteration)
    auth_list, hub_list = hits.get_auth_hub_list()
    print(f'HITS:\n\tAuthority:{auth_list}\n\tHub:      {hub_list}')

    # PageRank
    pr = algos.PageRank(
        graph=graph,
        damping_factor=args.damp_fac,
        iteration=iteration)
    pr_list = pr.get_pr_arr()
    print('\nPR:\n\tPageRank:', pr_list)

    # SimRank
    sim = algos.SimMatrix(
        graph=graph,
        decay_fac=args.decay_fac)

    sr = algos.SimRank(
        graph=graph,
        iteration=iteration,
        sim=sim)

    sim_mat = sr.get_sim_matrix()
    print('\nSimRank:\n', sim_mat)

    # to files.
    utils.to_file(
        file_name=in_file.split('/')[-1].split('.')[0],
        record_txt=[
            ('_HITS_authority.txt', auth_list),
            ('_HITS_hub.txt', hub_list),
            ('_PageRank.txt', pr_list),
            ('_SimRank.txt', sim_mat, '\n'),
        ])
    print(f'\n{time.time() - s: .3f}')
