import copy
import numpy as np
from typing import Optional, Tuple
import abc
import utils


class Algorithm(abc.ABC):
    """The Algorithm ABC.

    Parameters
    ----------
    graph : utils.Graph
        The graph object.
    iteration : Optional[int], optional
        The iteration times, by default 100
    """

    def __init__(self,
                 graph: utils.Graph,
                 iteration: Optional[int] = 100):

        self.graph = graph
        self.nodes = graph.nodes.values()
        self.iteration = int(iteration)

    @abc.abstractmethod
    def iterate(self) -> None:
        """The iterated function for each iteration."""
        pass


class PageRank(Algorithm):
    """Page Rank Algorithm.

    Parameters
    ----------
    graph : utils.Graph
        The graph object
    iteration : Optional[int], optional
        The iteration times, by default 100
    damping_factor : Optional[float], optional
        The factor of damping, by default .15
    """

    def __init__(self,
                 graph: utils.Graph,
                 iteration: Optional[int] = 100,
                 damping_factor: Optional[float] = .15):

        super().__init__(graph, iteration)
        self.d = damping_factor

    def iterate(self) -> None:
        for _ in range(self.iteration):
            for node in self.nodes:
                node.set_pr(self.d, len(self.nodes))
            self.norm_pr()

    def get_pr_arr(self):
        """Get the page rank(pr) array.

        Returns
        -------
        np.array
            The page rank array.
        """
        self.iterate()
        return np.round(np.asarray([node.pr for node in self.nodes]), 3)

    def norm_pr(self):
        """Normalize the page rank value."""
        pr_sum = sum(node.pr for node in self.nodes)

        for node in self.nodes:
            node.pr /= pr_sum


class HITS(Algorithm):
    """HITS Algorithm

    Parameters
    ----------
    graph : utils.Graph
        The graph object
    iteration : Optional[int], optional
        The iteration times, by default 100
    """

    def __init__(self,
                 graph: utils.Graph,
                 iteration: Optional[int] = 100):
        super().__init__(graph, iteration)

    def iterate(self) -> None:
        for _ in range(self.iteration):
            nodes = self.nodes
            for node in nodes:
                node.set_auth()
            for node in nodes:
                node.set_hub()
            self.norm_auth_hub()

    def get_auth_hub_list(self) -> Tuple[np.array]:
        """Get the authority and hub array.

        Returns
        -------
        Tuple[np.array]
            The authority array and hub array.
        """
        self.iterate()
        auth_list = np.asarray([node.auth for node in self.nodes])
        hub_list = np.asarray([node.hub for node in self.nodes])

        return np.round(auth_list, 3), np.round(hub_list, 3)

    def norm_auth_hub(self):
        """Normalized the auth and hub value."""
        auth_sum = sum(node.auth for node in self.nodes)
        hub_sum = sum(node.hub for node in self.nodes)

        for node in self.nodes:
            node.auth /= auth_sum
            node.hub /= hub_sum


class SimMatrix:
    """SimMatrix object.

    Parameters
    ----------
    graph : utils.Graph
        The graph object.
    decay_fac : float
        The factor of decay.
    """

    def __init__(self,
                 graph: utils.Graph,
                 decay_fac: float):
        self.decay_fac = decay_fac
        self.node_map = dict(zip(graph.nodes.keys(), range(len(graph))))
        self.sim = np.eye(len(graph))
        self.sim__ = np.zeros(shape=[len(graph), len(graph)])

    def replace_sim(self) -> None:
        """change the original sim matrix to updated sim matrix."""
        self.sim = copy.deepcopy(self.sim__)

    def compute_sr(self,
                   n1: utils.Node,
                   n2: utils.Node) -> float:
        """Compute the Sim Rank.

        Parameters
        ----------
        n1 : utils.Node
            The first node.
        n2 : utils.Node
            The second node.

        Returns
        -------
        float
            The Sim Rank.
        """
        # n1 and n2 are the same node.
        if n1.tag == n2.tag:
            return 1.0

        # get there in neighbors.
        tags1 = n1.from_dict.keys()
        tags2 = n2.from_dict.keys()

        # check the in neighbors is empty or not.
        if not tags1 or not tags2:
            return 0.0

        # compute the sum of sim rank.
        sr_sum = 0
        for t1 in tags1:
            for t2 in tags2:
                sr_sum += self.sim[self.node_map[t1]][self.node_map[t2]]

        # compute the scalered sim rank.
        return (self.decay_fac / (len(tags1) * len(tags2))) * sr_sum

    def update_sim(self,
                   n1: utils.Node,
                   n2: utils.Node,
                   value: float) -> None:
        """update the sim matrix.

        Parameters
        ----------
        n1 : utils.Node
            The first node.
        n2 : utils.Node
            The second node.
        value : float
            The target value.
        """
        self.sim__[self.node_map[n1.tag]][self.node_map[n2.tag]] = value


class SimRank(Algorithm):
    """Sim Rank Algorithm.

    Parameters
    ----------
    graph : utils.Graph
        The graph object
    iteration : Optional[int], optional
        The iteration times, by default 100
    sim : Optional[SimMatrix], optional
        The SimMatrix object, by default None
    """

    def __init__(self, graph: utils.Graph,
                 iteration: Optional[int] = 100,
                 sim: Optional[SimMatrix] = None):

        super().__init__(graph, iteration)
        self.sim = sim

    def iterate(self):
        for _ in range(self.iteration):
            for n1 in self.nodes:
                for n2 in self.nodes:
                    sim_rank = self.sim.compute_sr(n1, n2)
                    self.sim.update_sim(n1, n2, sim_rank)
            self.sim.replace_sim()

    def get_sim_matrix(self):
        """Get the sim matrix after sim rank iteration.

        Returns
        -------
        np.array
            The Sim Rank Matrix.
        """
        self.iterate()
        return np.round(np.asarray(self.sim.sim__), 3)
