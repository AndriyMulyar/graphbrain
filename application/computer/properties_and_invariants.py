# -*- coding: UTF-8 -*-
from sage.all import *

def add_to_lists(graph, *L):
    """
    Adds the specified graph to the arbitrary number of lists given as the second through last argument
    Use this function to build the lists of graphs
    """
    for list in L:
            list.append(graph)

#GRAPH INVARIANTS
all_invariants = []

efficient_invariants = []
intractable_invariants = []
theorem_invariants = []
broken_invariants = []

"""
    Last version of graphs packaged checked: Sage 8.2
    sage: sage.misc.banner.version_dict()['major'] < 8 or (sage.misc.banner.version_dict()['major'] == 8 and sage.misc.banner.version_dict()['minor'] <= 2)
    True
"""
sage_efficient_invariants = [Graph.number_of_loops, Graph.density, Graph.order, Graph.size, Graph.average_degree,
Graph.triangles_count, Graph.szeged_index, Graph.radius, Graph.diameter, Graph.girth, Graph.wiener_index,
Graph.average_distance, Graph.connected_components_number, Graph.maximum_average_degree, Graph.lovasz_theta,
Graph.spanning_trees_count, Graph.odd_girth, Graph.clustering_average, Graph.cluster_transitivity]

sage_intractable_invariants = [Graph.chromatic_number, Graph.chromatic_index, Graph.treewidth,
Graph.clique_number, Graph.pathwidth, Graph.fractional_chromatic_index, Graph.edge_connectivity,
Graph.vertex_connectivity, Graph.genus, Graph.crossing_number]

for i in sage_efficient_invariants:
    add_to_lists(i, efficient_invariants, all_invariants)
for i in sage_intractable_invariants:
    add_to_lists(i, intractable_invariants, all_invariants)

def distinct_degrees(g):
    """
    returns the number of distinct degrees of a graph
        sage: distinct_degrees(p4)
        2
        sage: distinct_degrees(k4)
        1
    """
    return len(set(g.degree()))
add_to_lists(distinct_degrees, efficient_invariants, all_invariants)



def max_common_neighbors(g):
    """
    Returns the maximum number of common neighbors of any pair of distinct vertices in g.

        sage: max_common_neighbors(p4)
        1
        sage: max_common_neighbors(k4)
        2
    """
    max = 0
    V = g.vertices()
    n = g.order()
    for i in range(n):
        for j in range(n):
            if i < j:
                temp = len(common_neighbors(g, V[i], V[j]))
                if temp > max:
                    max = temp
    return max
add_to_lists(max_common_neighbors, efficient_invariants, all_invariants)

def min_common_neighbors(g):
    """
    Returns the minimum number of common neighbors of any pair of distinct vertices in g,
    which is necessarily 0 for disconnected graphs.

        sage: min_common_neighbors(p4)
        0
        sage: min_common_neighbors(k4)
        2
    """
    n = g.order()
    min = n
    V = g.vertices()
    for i in range(n):
        for j in range(n):
            if i < j:
                temp = len(common_neighbors(g, V[i], V[j]))
                #if temp == 0:
                    #print "i={}, j={}".format(i,j)
                if temp < min:
                    min = temp
    return min
add_to_lists(min_common_neighbors, efficient_invariants, all_invariants)

def mean_common_neighbors(g):
    """
    Returns the average number of common neighbors of any pair of distinct vertices in g.
        sage: mean_common_neighbors(p4)
        1/3
        sage: mean_common_neighbors(k4)
        2
    """
    V = g.vertices()
    n = g.order()
    sum = 0
    for i in range(n):
        for j in range(n):
            if i < j:
                sum += len(common_neighbors(g, V[i], V[j]))
    return 2*sum/(n*(n-1))
add_to_lists(mean_common_neighbors, efficient_invariants, all_invariants)

def min_degree(g):
    """
    Returns the minimum of all degrees of the graph g.

        sage: min_degree(graphs.CompleteGraph(5))
        4
        sage: min_degree(graphs.CycleGraph(5))
        2
        sage: min_degree(graphs.StarGraph(5))
        1
        sage: min_degree(graphs.CompleteBipartiteGraph(3,5))
        3
    """
    return min(g.degree())
add_to_lists(min_degree, efficient_invariants, all_invariants)

def max_degree(g):
    """
    Returns the maximum of all degrees of the graph g.

        sage: max_degree(graphs.CompleteGraph(5))
        4
        sage: max_degree(graphs.CycleGraph(5))
        2
        sage: max_degree(graphs.StarGraph(5))
        5
        sage: max_degree(graphs.CompleteBipartiteGraph(3,5))
        5
    """
    return max(g.degree())
add_to_lists(max_degree, efficient_invariants, all_invariants)

def median_degree(g):
    """
    Return the median of the list of vertex degrees.

        sage: median_degree(p4)
        3/2
        sage: median_degree(p3)
        1
    """
    return median(g.degree())
add_to_lists(median_degree, efficient_invariants, all_invariants)

def inverse_degree(g):
    """
    Return the sum of the reciprocals of the non-zero degrees.

    Return 0 if the graph has no edges.

        sage: inverse_degree(p4)
        3
        sage: inverse_degree(graphs.CompleteGraph(1))
        0
    """
    if g.size() == 0:
        return 0
    return sum([(1.0/d) for d in g.degree() if d!= 0])
add_to_lists(inverse_degree, efficient_invariants, all_invariants)

def eulerian_faces(g):
    """
    Returns 2 - order + size, which is the number of faces if the graph is planar,
    a consequence of Euler's Formula.

        sage: eulerian_faces(graphs.CycleGraph(5))
        2
        sage: eulerian_faces(graphs.DodecahedralGraph())
        12
    """
    n = g.order()
    m = g.size()
    return 2 - n + m
add_to_lists(eulerian_faces, efficient_invariants, all_invariants)

def barrus_q(g):
    """
    If the degrees sequence is in non-increasing order, with index starting at 1,
    barrus_q = max(k:d_k >= k)

    Defined by M. Barrus in "Havel-Hakimi Residues of Unigraphs", 2012

        sage: barrus_q(graphs.CompleteGraph(5))
        4
        sage: barrus_q(graphs.StarGraph(3))
        1
    """
    Degrees = g.degree()
    Degrees.sort()
    Degrees.reverse()
    return max(k for k in range(g.order()) if Degrees[k] >= (k+1)) + 1
add_to_lists(barrus_q, efficient_invariants, all_invariants)

def barrus_bound(g):
    """
    Returns n - barrus q

    Defined in: Barrus, Michael D. "Havel–Hakimi residues of unigraphs." Information Processing Letters 112.1 (2012): 44-48.

        sage: barrus_bound(k4)
        1
        sage: barrus_bound(graphs.OctahedralGraph())
        2
    """
    return g.order() - barrus_q(g)
add_to_lists(barrus_bound, efficient_invariants, all_invariants)

def matching_number(g):
    """
    Returns the matching number of the graph g, i.e., the size of a maximum
    matching.

    A matching is a set of independent edges.

    See: https://en.wikipedia.org/wiki/Matching_(graph_theory)

        sage: matching_number(graphs.CompleteGraph(5))
        2
        sage: matching_number(graphs.CycleGraph(5))
        2
        sage: matching_number(graphs.StarGraph(5))
        1
        sage: matching_number(graphs.CompleteBipartiteGraph(3,5))
        3
    """
    return int(g.matching(value_only=True, use_edge_labels=False))
add_to_lists(matching_number, efficient_invariants, all_invariants)

def residue(g):
    """
    If the Havel-Hakimi process is iterated until a sequence of 0s is returned,
    residue is defined to be the number of zeros of this sequence.

    See: Favaron, Odile, Maryvonne Mahéo, and J‐F. Saclé. "On the residue of a graph." Journal of Graph Theory 15.1 (1991): 39-64.

        sage: residue(k4)
        1
        sage: residue(p4)
        2
    """
    seq = g.degree_sequence()

    while seq[0] > 0:
        d = seq.pop(0)
        seq[:d] = [k-1 for k in seq[:d]]
        seq.sort(reverse=True)

    return len(seq)
add_to_lists(residue, efficient_invariants, all_invariants)

def annihilation_number(g):
    """
    Given the degree sequence in non-degreasing order, with indices starting at 1, the annihilation number is the largest index k so the sum of the first k degrees is no more than the sum of the remaining degrees

    See: Larson, Craig E., and Ryan Pepper. "Graphs with equal independence and annihilation numbers." the electronic journal of combinatorics 18.1 (2011): 180.

        sage: annihilation_number(c4)
        2
        sage: annihilation_number(p5)
        3
    """
    seq = sorted(g.degree())

    a = 0
    while sum(seq[:a+1]) <= sum(seq[a+1:]):
        a += 1

    return a
add_to_lists(annihilation_number, efficient_invariants, all_invariants)

def fractional_alpha(g):
    """
    This is the optimal solution of the linear programming relaxation of the integer programming formulation of independence number (alpha).

    See: Nemhauser, George L., and Leslie Earl Trotter. "Vertex packings: structural properties and algorithms." Mathematical Programming 8.1 (1975): 232-248.

        sage: fractional_alpha(k3)
        1.5
        sage: fractional_alpha(p5)
        3.0
    """
    if len(g.vertices()) == 0:
        return 0
    p = MixedIntegerLinearProgram(maximization=True)
    x = p.new_variable(nonnegative=True)
    p.set_objective(sum(x[v] for v in g.vertices()))

    for v in g.vertices():
        p.add_constraint(x[v], max=1)

    for (u,v) in g.edge_iterator(labels=False):
        p.add_constraint(x[u] + x[v], max=1)

    return p.solve()
add_to_lists(fractional_alpha, efficient_invariants, all_invariants)

def fractional_covering(g):
    """
    This is the optimal solution of the linear programming relaxation of the integer programming formulation of covering number.

    For ILP formulation see: https://en.wikipedia.org/wiki/Vertex_cover

        sage: fractional_covering(k3)
        1.5
        sage: fractional_covering(p5)
        2.0
    """
    if len(g.vertices()) == 0:
        return 0
    p = MixedIntegerLinearProgram(maximization=False)
    x = p.new_variable(nonnegative=True)
    p.set_objective(sum(x[v] for v in g.vertices()))

    for v in g.vertices():
        p.add_constraint(x[v], max=1)

    for (u,v) in g.edge_iterator(labels=False):
        p.add_constraint(x[u] + x[v], min=1)

    return p.solve()
add_to_lists(fractional_covering, efficient_invariants, all_invariants)

def cvetkovic(g):
    """
    This in the minimum of the number of nonnegative and nonpositive eigenvalues of the adjacency matrix.

    Cvetkovic's theorem says that this number is an upper bound for the independence number of a graph.

    See: Cvetković, Dragoš M., Michael Doob, and Horst Sachs. Spectra of graphs: theory and application. Vol. 87. Academic Pr, 1980.

        sage: cvetkovic(p5)
        3
        sage: cvetkovic(graphs.PetersenGraph())
        4
    """
    eigenvalues = g.spectrum()
    positive = 0
    negative = 0
    zero = 0
    for e in eigenvalues:
        if e > 0:
            positive += 1
        elif e < 0:
            negative += 1
        else:
            zero += 1

    return zero + min([positive, negative])
add_to_lists(cvetkovic, efficient_invariants, all_invariants)

def cycle_space_dimension(g):
    """
    Returns the dimension of the cycle space (also called the circuit rank).

    See: https://en.wikipedia.org/wiki/Cycle_space
    And: https://en.wikipedia.org/wiki/Circuit_rank

        sage: cycle_space_dimension(k3)
        1
        sage: cycle_space_dimension(c4c4)
        2
        sage: cycle_space_dimension(glasses_5_5)
        2
    """
    return g.size()-g.order()+g.connected_components_number()
add_to_lists(cycle_space_dimension, efficient_invariants, all_invariants)

def card_center(g):
    return len(g.center())
add_to_lists(card_center, efficient_invariants, all_invariants)

def card_periphery(g):
    return len(g.periphery())
add_to_lists(card_periphery, efficient_invariants, all_invariants)

def max_eigenvalue(g):
    return max(g.adjacency_matrix().change_ring(RDF).eigenvalues())
add_to_lists(max_eigenvalue, efficient_invariants, all_invariants)

def min_eigenvalue(g):
    return min(g.adjacency_matrix().change_ring(RDF).eigenvalues())
add_to_lists(min_eigenvalue, efficient_invariants, all_invariants)

def resistance_distance_matrix(g):
    L = g.laplacian_matrix()
    n = g.order()
    J = ones_matrix(n,n)
    temp = L+(1.0/n)*J
    X = temp.inverse()
    R = (1.0)*ones_matrix(n,n)
    for i in range(n):
        for j in range(n):
            R[i,j] = X[i,i] + X[j,j] - 2*X[i,j]
    return R

def kirchhoff_index(g):
    R = resistance_distance_matrix(g)
    return .5*sum(sum(R))
add_to_lists(kirchhoff_index, efficient_invariants, all_invariants)

def largest_singular_value(g):
    A = matrix(RDF,g.adjacency_matrix(sparse=False))
    svd = A.SVD()
    sigma = svd[1]
    return sigma[0,0]
add_to_lists(largest_singular_value, efficient_invariants, all_invariants)

def card_max_cut(g):
    return g.max_cut(value_only=True)
add_to_lists(card_max_cut, intractable_invariants, all_invariants)

def welsh_powell(g):
    """
    for degrees d_1 >= ... >= d_n
    returns the maximum over all indices i of of the min(i,d_i + 1)

    sage: welsh_powell(k5) = 4
    """
    n= g.order()
    D = g.degree()
    D.sort(reverse=True)
    mx = 0
    for i in range(n):
        temp = min({i,D[i]})
        if temp > mx:
            mx = temp
    return mx + 1
add_to_lists(welsh_powell, efficient_invariants, all_invariants)

#outputs upper bound from Brooks Theorem: returns Delta + 1 for complete and odd cycles
def brooks(g):
    Delta = max(g.degree())
    delta = min(g.degree())
    n = g.order()
    if is_complete(g):
        return Delta + 1
    elif n%2 == 1 and g.is_connected() and Delta == 2 and delta == 2: #same as if all degrees are 2
        return Delta + 1
    else:
        return Delta
add_to_lists(brooks, efficient_invariants, all_invariants)

#wilf's upper bound for chromatic number
def wilf(g):
    return max_eigenvalue(g) + 1
add_to_lists(wilf, efficient_invariants, all_invariants)

#a measure of irregularity
def different_degrees(g):
    return len(set(g.degree()))
add_to_lists(different_degrees, efficient_invariants, all_invariants)

def szekeres_wilf(g):
    """
    Returns 1+ max of the minimum degrees for all subgraphs
    Its an upper bound for chromatic number

    sage: szekeres_wilf(graphs.CompleteGraph(5))
    5
    """
    #removes a vertex, if possible, of degree <= i
    def remove_vertex_of_degree(gc,i):
        Dc = gc.degree()
        V = gc.vertices()
        #print "Dc is %s, V is %s" %(Dc,V)
        mind = min(Dc)
        #print "mind is %s" %mind
        if mind <= i:

            ind = Dc.index(mind)
            #print "ind is %s, vertex is %s" %(ind,V[ind])
            return gc.delete_vertex(V[ind])
        else:
            return gc
    D = g.degree()
    delta = min(D)
    Delta = max(D)
    for i in range(delta,Delta+1):
        gc = copy(g)
        value = g.order() + 1
        while gc.size() > 0 and gc.order() < value:
            #gc.show()
            value = gc.order()
            remove_vertex_of_degree(gc,i)
        if gc.size() == 0:
            return i + 1
add_to_lists(szekeres_wilf, efficient_invariants, all_invariants)

def average_vertex_temperature(g):
     D = g.degree()
     n = g.order()
     return sum([D[i]/(n-D[i]+0.0) for i in range(n)])/n
add_to_lists(average_vertex_temperature, efficient_invariants, all_invariants)

def sum_temperatures(g):
     D = g.degree()
     n = g.order()
     return sum([D[i]/(n-D[i]+0.0) for i in range(n)])
add_to_lists(sum_temperatures, efficient_invariants, all_invariants)

def randic(g):
     D = g.degree()
     V = g.vertices()
     if min(D) == 0:
          return oo
     sum = 0
     for e in g.edges():
         v = e[0]
         i = V.index(v)
         w = e[1]
         j = V.index(w)
         sum += 1.0/(D[i]*D[j])**0.5
     return sum
add_to_lists(randic, efficient_invariants, all_invariants)

#a very good lower bound for alpha
def max_even_minus_even_horizontal(g):
    """
    finds def max_even_minus_even_horizontal for each component and adds them up.
    """
    mx_even=0
    Gcomps=g.connected_components_subgraphs()

    while Gcomps != []:
            H=Gcomps.pop()
            temp=max_even_minus_even_horizontal_component(H)
            mx_even+=temp
            #print "temp = {}, mx_even = {}".format(temp,mx_even)

    return mx_even
add_to_lists(max_even_minus_even_horizontal, efficient_invariants, theorem_invariants, all_invariants)

def max_even_minus_even_horizontal_component(g):
    """
    for each vertex v, find the number of vertices at even distance from v,
    and substract the number of edges induced by these vertices.
    this number is a lower bound for independence number.
    take the max. returns 0 if graph is not connected
    """
    if g.is_connected()==False:
        return 0

    distances = g.distance_all_pairs()
    mx=0
    for v in g.vertices():
        Even=[]
        for w in g.vertices():
            if distances[v][w]%2==0:
                Even.append(w)

        #print len(Even), len(g.subgraph(Even).edges())
        l=len(Even)-len(g.subgraph(Even).edges())
        if l>mx:
            mx=l
    return mx

def laplacian_energy(g):
     L = g.laplacian_matrix().change_ring(RDF).eigenvalues()
     Ls = [1/lam**2 for lam in L if lam > 0]
     return 1 + sum(Ls)
add_to_lists(laplacian_energy, efficient_invariants, all_invariants)

def laplacian_energy_like(g):
    """
    Returns the sum of the square roots of the laplacian eigenvalues

    Liu, Jianping, and Bolian Liu. "A Laplacian-energy-like invariant of a graph." MATCH-COMMUNICATIONS IN MATHEMATICAL AND IN COMPUTER CHEMISTRY 59.2 (2008): 355-372.
    """
    return sum([sqrt(x) for x in g.spectrum(laplacian = True)])
add_to_lists(laplacian_energy_like, efficient_invariants, all_invariants)

#sum of the positive eigenvalues of a graph
def gutman_energy(g):
     L = g.adjacency_matrix().change_ring(RDF).eigenvalues()
     Ls = [lam for lam in L if lam > 0]
     return sum(Ls)
add_to_lists(gutman_energy, efficient_invariants, all_invariants)

#the second smallest eigenvalue of the Laplacian matrix of a graph, also called the "algebraic connectivity" - the smallest should be 0
def fiedler(g):
     L = g.laplacian_matrix().change_ring(RDF).eigenvalues()
     L.sort()
     return L[1]
add_to_lists(fiedler, broken_invariants, all_invariants)

def degree_variance(g):
     mu = mean(g.degree())
     s = sum((x-mu)**2 for x in g.degree())
     return s/g.order()
add_to_lists(degree_variance, efficient_invariants, all_invariants)

def graph_rank(g):
    return g.adjacency_matrix().rank()
add_to_lists(graph_rank, efficient_invariants, all_invariants)

def card_positive_eigenvalues(g):
    return len([lam for lam in g.adjacency_matrix().eigenvalues() if lam > 0])
add_to_lists(card_positive_eigenvalues, efficient_invariants, all_invariants)

def card_zero_eigenvalues(g):
    return len([lam for lam in g.adjacency_matrix().eigenvalues() if lam == 0])
add_to_lists(card_zero_eigenvalues, efficient_invariants, all_invariants)

def card_negative_eigenvalues(g):
    return len([lam for lam in g.adjacency_matrix().eigenvalues() if lam < 0])
add_to_lists(card_negative_eigenvalues, efficient_invariants, all_invariants)

def card_cut_vertices(g):
    return len((g.blocks_and_cut_vertices())[1])
add_to_lists(card_cut_vertices, efficient_invariants, all_invariants)

def card_connectors(g):
    return g.order() - card_cut_vertices(g)
add_to_lists(card_connectors, efficient_invariants, all_invariants)

#return number of leafs or pendants
def card_pendants(g):
    return sum([x for x in g.degree() if x == 1])
add_to_lists(card_pendants, efficient_invariants, all_invariants)

#returns number of bridges in graph
def card_bridges(g):
    gs = g.strong_orientation()
    bridges = []
    for scc in gs.strongly_connected_components():
        bridges.extend(gs.edge_boundary(scc))
    return len(bridges)
add_to_lists(card_bridges, efficient_invariants, all_invariants)

#upper bound for the domination number
def alon_spencer(g):
    delta = min(g.degree())
    n = g.order()
    return n*((1+log(delta + 1.0)/(delta + 1)))
add_to_lists(alon_spencer, efficient_invariants, all_invariants)

#lower bound for residue and, hence, independence number
def caro_wei(g):
    return sum([1.0/(d + 1) for d in g.degree()])
add_to_lists(caro_wei, efficient_invariants, all_invariants)

#equals 2*size, the 1st theorem of graph theory
def degree_sum(g):
    return sum(g.degree())
add_to_lists(degree_sum, efficient_invariants, all_invariants)

#smallest sum of degrees of non-adjacent degrees, invariant in ore condition for hamiltonicity
#default for complete graph?
def sigma_dist2(g):
    if g.size() == g.order()*(g.order()-1)/2:
        return g.order()
    Dist = g.distance_all_pairs()
    return min(g.degree(v) + g.degree(w) for v in g for w in g if Dist[v][w] > 1)
add_to_lists(sigma_dist2, efficient_invariants, all_invariants)

#cardinality of the automorphism group of the graph
def order_automorphism_group(g):
    return g.automorphism_group(return_group=False, order=True)
add_to_lists(order_automorphism_group, efficient_invariants, all_invariants)

#in sufficient condition for graphs where vizing's independence theorem holds
def brinkmann_steffen(g):
    E = g.edges()
    if len(E) == 0:
        return 0
    Dist = g.distance_all_pairs()
    return min(g.degree(v) + g.degree(w) for v in g for w in g if Dist[v][w] == 1)
add_to_lists(brinkmann_steffen, efficient_invariants, all_invariants)

def alpha_critical_optimum(g, alpha_critical_names):

    n = g.order()
    V = g.vertices()
    #g.show()

    alpha_critical_graph_names = []

    #get alpha_critical graphs with order <= n
    for name in alpha_critical_names:
        h = Graph(name)
        if h.order() <= n:
            alpha_critical_graph_names.append(h.graph6_string())

    #print alpha_critical_graphs

    LP = MixedIntegerLinearProgram(maximization=True)
    b = LP.new_variable(nonnegative=True)

    # We define the objective
    LP.set_objective(sum([b[v] for v in g]))

    # For any edge, we define a constraint
    for (u,v) in g.edges(labels=None):
        LP.add_constraint(b[u]+b[v],max=1)
        #LP.add_constraint(b[u]+b[v],min=1)

    #search through all subsets of g with order >= 3
    #and look for *any* subgraph isomorphic to an alpha critical graph
    #for any match we define a constraint

    i = 3
    while i <= n:
        SS = Subsets(Set(V),i)
        for S in SS:
            L = [g6 for g6 in alpha_critical_graph_names if Graph(g6).order() == i]
            #print L
            for g6 in L:
                h = Graph(g6)
                if g.subgraph(S).subgraph_search(h, induced=False):

                    #print S
                    #add constraint
                    alpha = independence_number(h)
                    #print h.graph6_string(), alpha
                    LP.add_constraint(sum([b[j] for j in S]), max = alpha, name = h.graph6_string())
        i = i + 1

    #for c in LP.constraints():
        #print c

    # The .solve() functions returns the objective value
    LP.solve()

    #return LP

    b_sol = LP.get_values(b)
    return b_sol, sum(b_sol.values())


###several invariants and auxiliary functions related to the Independence Decomposition Theorem

#finds all vertices with weight 1 in some max weighted stable set with wieghts in {0,1,1/2}
#had problem that LP solver has small numerical errors, fixed with kludgy if condition
def find_stable_ones_vertices(g):
    F = []
    alpha_f = fractional_alpha(g)
    for v in g.vertices():
        gc = copy(g)
        gc.delete_vertices(closed_neighborhood(gc, v))
        alpha_f_prime = fractional_alpha(gc)
        if abs(alpha_f - alpha_f_prime - 1) < .01:
            F.append(v)
    return F

def find_max_critical_independent_set(g):
    S = find_stable_ones_vertices(g)
    H = g.subgraph(S)
    return H.independent_set()

def critical_independence_number(g):
    return len(find_max_critical_independent_set(g))
add_to_lists(critical_independence_number, efficient_invariants, all_invariants)

def card_independence_irreducible_part(g):
    return len(find_independence_irreducible_part(g))
add_to_lists(card_independence_irreducible_part, efficient_invariants, all_invariants)

def find_independence_irreducible_part(g):
    X = find_KE_part(g)
    SX = Set(X)
    Vertices = Set(g.vertices())
    return list(Vertices.difference(SX))

#returns KE part guaranteed by Independence Decomposition Theorem
def find_KE_part(g):
    return closed_neighborhood(g, find_max_critical_independent_set(g))

def card_KE_part(g):
    return len(find_KE_part(g))
add_to_lists(card_KE_part, efficient_invariants, all_invariants)

def find_independence_irreducible_subgraph(g):
    return g.subgraph(find_independence_irreducible_part(g))

def find_KE_subgraph(g):
    return g.subgraph(find_KE_part(g))


#make invariant from property
def make_invariant_from_property(property, name=None):
    """
    This function takes a property as an argument and returns an invariant
    whose value is 1 if the object has the property, else 0
    Optionally a name for the new property can be provided as a second argument.
    """
    def boolean_valued_invariant(g):
        if property(g):
            return 1
        else:
            return 0

    if name is not None:
        boolean_valued_invariant.__name__ = name
    elif hasattr(property, '__name__'):
        boolean_valued_invariant.__name__ = '{}_value'.format(property.__name__)
    else:
        raise ValueError('Please provide a name for the new function')

    return boolean_valued_invariant

# defined by R. Pepper in an unpublished paper on graph irregularity
def geometric_length_of_degree_sequence(g):
    return sqrt(sum(d**2 for d in g.degree()))
add_to_lists(geometric_length_of_degree_sequence, efficient_invariants, all_invariants)

# Two Stability Theta Bound
# For graphs with alpha <= 2,
# lovasz_theta <= 2^(2/3)*n^(1/3)
# The Sandwich Theorem by Knuth p. 47
def two_stability_theta_bound(g):
    return 2**(2/3)*g.order()**(1/3)
add_to_lists(two_stability_theta_bound, efficient_invariants, all_invariants)

# Lovasz Theta over Root N
# The Sandwich Theorem by Knuth p. 45
def lovasz_theta_over_root_n(g):
    return g.lovasz_theta()/sqrt(g.order())
add_to_lists(lovasz_theta_over_root_n, efficient_invariants, all_invariants)

# Theta * Theta-Complement
# The Sandwich Theorem by Knuth, p. 27
def theta_theta_complement(g):
    return g.lovasz_theta() * g.complement().lovasz_theta()
add_to_lists(theta_theta_complement, efficient_invariants, all_invariants)

# Depth = Order - Residue
# This is the number of steps it takes for Havel-Hakimi to terminate
def depth(g):
    return g.order()-residue(g)
add_to_lists(depth, efficient_invariants, all_invariants)

# Lovasz Theta of the complement of the given graph
def lovasz_theta_complement(g):
    return g.complement().lovasz_theta()
add_to_lists(lovasz_theta_complement, efficient_invariants, all_invariants)

# N over lovasz_theta_complement
# This is a lower bound for lovasz theta
# The Sandwich Theorem by Knuth, p. 27
def n_over_lovasz_theta_complement(g):
    return g.order()/lovasz_theta_complement(g)
add_to_lists(n_over_lovasz_theta_complement, efficient_invariants, all_invariants)

# The number of vertices at even distance from v and return the max over all vertices
def max_even(g):
    from sage.graphs.distances_all_pairs import distances_all_pairs
    D = distances_all_pairs(g)
    evens_list = []
    for u in D:
        evens = 0
        for v in D[u]:
            if D[u][v] % 2 == 0:
                evens += 1
        evens_list.append(evens)
    return max(evens_list)
add_to_lists(max_even, efficient_invariants, all_invariants)

# The number of vertices at even distance from v and return the min over all vertices
def min_even(g):
    from sage.graphs.distances_all_pairs import distances_all_pairs
    D = distances_all_pairs(g)
    evens_list = []
    for u in D:
        evens = 0
        for v in D[u]:
            if D[u][v] % 2 == 0:
                evens += 1
        evens_list.append(evens)
    return min(evens_list)
add_to_lists(min_even, efficient_invariants, all_invariants)

# The number of vertices at odd distance from v and return the max over all vertices
def max_odd(g):
    from sage.graphs.distances_all_pairs import distances_all_pairs
    D = distances_all_pairs(g)
    odds_list = []
    for u in D:
        odds = 0
        for v in D[u]:
            if D[u][v] % 2 != 0:
                odds += 1
        odds_list.append(odds)
    return max(odds_list)
add_to_lists(max_odd, efficient_invariants, all_invariants)

# The number of vertices at odd distance from v and return the min over all vertices
def min_odd(g):
    from sage.graphs.distances_all_pairs import distances_all_pairs
    D = distances_all_pairs(g)
    odds_list = []
    for u in D:
        odds = 0
        for v in D[u]:
            if D[u][v] % 2 != 0:
                odds += 1
        odds_list.append(odds)
    return min(odds_list)
add_to_lists(min_odd, efficient_invariants, all_invariants)

#returns sum of distances between *distinct* vertices, return infinity is graph is not connected
def transmission(g):
    if not g.is_connected():
        return Infinity
    if g.is_tree() and max(g.degree()) == 2:
        summation = 0
        for i in range(1,g.order()):
            summation += (i*(i+1))/2
        return summation * 2
    else:
        V = g.vertices()
        D = g.distance_all_pairs()
        return sum([D[v][w] for v in V for w in V if v != w])
add_to_lists(transmission, efficient_invariants, all_invariants)

def harmonic_index(g):
    sum = 0
    for (u,v) in g.edges(labels = false):
        sum += (2 / (g.degree(u) + g.degree(v)))
    return sum
add_to_lists(harmonic_index, efficient_invariants, all_invariants)

def bavelas_index(g):
    """
    returns sum over all edges (v,w) of (distance from v to all other vertices)/(distance from w to all other vertices)
    computes each edge twice (once with v computation in numerator, once with w computation in numerator)

        sage: bavelas_index(p6)
        5086/495
        sage: bavelas_index(k4)
        12
    """
    D = g.distance_all_pairs()

    def s_aux(v):
        """
        computes sum of distances from v to all other vertices
        """
        sum = 0
        for w in g.vertices():
            sum += D[v][w]
        return sum

    sum_final = 0

    for edge in g.edges(labels=false):
        v = edge[0]
        w = edge[1]
        sum_final += (s_aux(w) / s_aux(v)) + (s_aux(v) / s_aux(w))

    return sum_final
add_to_lists(bavelas_index, efficient_invariants, all_invariants)

#a solution of the invariant interpolation problem for upper bound of chromatic number for c8chords
#all upper bounds in theory have value at least 3 for c8chords
#returns 2 for bipartite graphs, order for non-bipartite
def bipartite_chromatic(g):
    if g.is_bipartite():
        return 2
    else:
        return g.order()
add_to_lists(bipartite_chromatic, efficient_invariants, all_invariants)

def beauchamp_index(g):
    """
    Defined on page 597 of Sabidussi, Gert. "The centrality index of a graph." Psychometrika 31.4 (1966): 581-603.

    sage: beauchamp_index(c4)
    1
    sage: beauchamp_index(p5)
    137/210
    sage: beauchamp_index(graphs.PetersenGraph())
    2/3
    """

    D = g.distance_all_pairs()

    def s_aux(v): #computes sum of distances from v to all other vertices
        sum = 0
        for w in g.vertices():
            sum += D[v][w]
        return sum

    sum_final = 0

    for v in g.vertices():
        sum_final += 1/s_aux(v)
    return sum_final

add_to_lists(beauchamp_index, efficient_invariants, all_invariants)

def subcubic_tr(g):
    """
    Returns the maximum number of vertex disjoint triangles of the graph

    Harant, Jochen, et al. "The independence number in graphs of maximum degree three." Discrete Mathematics 308.23 (2008): 5829-5833.
    """
    return len(form_triangles_graph(g).connected_components())
add_to_lists(subcubic_tr, efficient_invariants, all_invariants)

def edge_clustering_centrality(g, edge = None):
    """
    Returns edge clustering centrality for all edges in a list, or a single centrality for the given edge
    Utility to be used with min, avg, max invariants
    INPUT: g - a graph
           edge - (default: None) An edge in g. If given, will compute centrality for given edge, otherwise all edges. See Graph.has_Edge for acceptable input.
    From:
    An Application of Edge Clustering Centrality to Brain Connectivity by Joy Lind, Frank Garcea, Bradford Mahon, Roger Vargas, Darren A. Narayan

    TESTS:
        sage: edge_clustering_centrality(graphs.CompleteGraph(5))
        [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
        sage: edge_clustering_centrality(graphs.CompleteBipartiteGraph(3,4))
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        sage: edge_clustering_centrality(graphs.PetersenGraph())
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        sage: edge_clustering_centrality(graphs.BullGraph())
        [3, 3, 3, 2, 2]
    """
    if edge is None:
        edge_clusering_centralities = []
        for e in g.edges(labels = False):
            edge_clusering_centralities.append(len(set(g.neighbors(e[0])) & set(g.neighbors(e[1]))) + 2) # +2 for the two vertices in e
        return edge_clusering_centralities
    else:
        return len(set(g.neighbors(edge[0])) & set(g.neighbors(edge[1]))) + 2 # +2 for the two vertices in e

def max_edge_clustering_centrality(g):
    """
        sage: max_edge_clustering_centrality(p3)
        2
        sage: max_edge_clustering_centrality(paw)
        3
    """
    return max(edge_clustering_centrality(g))
add_to_lists(max_edge_clustering_centrality, efficient_invariants, all_invariants)

def min_edge_clustering_centrality(g):
    """
        sage: min_edge_clustering_centrality(p3)
        2
        sage: min_edge_clustering_centrality(paw)
        2
    """
    return min(edge_clustering_centrality(g))
add_to_lists(min_edge_clustering_centrality, efficient_invariants, all_invariants)

def mean_edge_clustering_centrality(g):
    """
        sage: mean_edge_clustering_centrality(p3)
        2
        sage: mean_edge_clustering_centrality(paw)
        11/4
    """
    centralities = edge_clustering_centrality(g)
    return sum(centralities) / len(centralities)
add_to_lists(mean_edge_clustering_centrality, efficient_invariants, all_invariants)

def local_density(g, vertex = None):
    """
    Returns local density for all vertices as a list, or a single local density for the given vertex
    INPUT: g - a graph
           vertex - (default: None) A vertex in g. If given, it will compute local density for just that vertex, otherwise for all of them

    Pavlopoulos, Georgios A., et al. "Using graph theory to analyze biological networks." BioData mining 4.1 (2011): 10.
    """
    if vertex == None:
        densities = []
        for v in g.vertices():
            densities.append(g.subgraph(g[v] + [v]).density())
        return densities
    return g.subgraph(g[vertex] + [vertex]).density()

def min_local_density(g):
    """
        sage: min_local_density(p3)
        2/3
        sage: min_local_density(paw)
        2/3
    """
    return min(local_density(g))
add_to_lists(min_local_density, efficient_invariants, all_invariants)

def max_local_density(g):
    """
        sage: max_local_density(p3)
        1
        sage: max_local_density(paw)
        1
    """
    return max(local_density(g))
add_to_lists(max_local_density, efficient_invariants, all_invariants)

def mean_local_density(g):
    """
        sage: mean_local_density(p3)
        8/9
        sage: mean_local_density(paw)
        11/12
    """
    densities = local_density(g)
    return sum(densities) / len(densities)
add_to_lists(mean_local_density, efficient_invariants, all_invariants)

def card_simple_blocks(g):
    """
    returns the number of blocks with order 2

        sage: card_simple_blocks(k10)
        0
        sage: card_simple_blocks(paw)
        1
        sage: card_simple_blocks(kite_with_tail)
        1
    """
    blocks = g.blocks_and_cut_vertices()[0]
    count = 0
    for block in blocks:
        if len(block) == 2:
            count += 1
    return count
add_to_lists(card_simple_blocks, efficient_invariants, all_invariants)

# Block of more than 2 vertices
def card_complex_blocks(g):
    """
    returns the number of blocks with order 2

        sage: card_complex_blocks(k10)
        1
        sage: card_complex_blocks(paw)
        1
        sage: card_complex_blocks(kite_with_tail)
        1
    """
    blocks = g.blocks_and_cut_vertices()[0]
    count = 0
    for block in blocks:
        if len(block) > 2:
            count += 1
    return count
add_to_lists(card_complex_blocks, efficient_invariants, all_invariants)

# Block is a clique and more than 2 vertices
def card_complex_cliques(g):
    """
    returns the number of blocks with order 2

        sage: card_complex_cliques(k10)
        1
        sage: card_complex_cliques(paw)
        1
        sage: card_complex_cliques(kite_with_tail)
        0
    """
    blocks = g.blocks_and_cut_vertices()[0]
    count = 0
    for block in blocks:
        h = g.subgraph(block)
        if h.is_clique() and h.order() > 2:
            count += 1
    return count
add_to_lists(card_complex_cliques, efficient_invariants, all_invariants)

def max_minus_min_degrees(g):
    return max_degree(g) - min_degree(g)
add_to_lists(max_minus_min_degrees, efficient_invariants, all_invariants)

def randic_irregularity(g):
    return order(g)/2 - randic(g)
add_to_lists(randic_irregularity, efficient_invariants, all_invariants)

def degree_variance(g):
    avg_degree = g.average_degree()
    return 1/order(g) * sum([d**2 - avg_degree for d in [g.degree(v) for v in g.vertices()]])
add_to_lists(degree_variance, efficient_invariants, all_invariants)

def sum_edges_degree_difference(g):
    return sum([abs(g.degree(e[0]) - g.degree(e[1])) for e in g.edges()])
add_to_lists(sum_edges_degree_difference, efficient_invariants, all_invariants)

def one_over_size_sedd(g):
    return 1/g.size() * sum_edges_degree_difference(g)
add_to_lists(one_over_size_sedd, efficient_invariants, all_invariants)

def largest_eigenvalue_minus_avg_degree(g):
    return max_eigenvalue(g) - g.average_degree()
add_to_lists(largest_eigenvalue_minus_avg_degree, efficient_invariants, all_invariants)

def min_betweenness_centrality(g):
    centralities = g.centrality_betweenness(exact=True)
    return centralities[min(centralities)]
add_to_lists(min_betweenness_centrality, efficient_invariants, all_invariants)

def max_betweenness_centrality(g):
    centralities = g.centrality_betweenness(exact=True)
    return centralities[max(centralities)]
add_to_lists(max_betweenness_centrality, efficient_invariants, all_invariants)

def mean_betweenness_centrality(g):
    centralities = g.centrality_betweenness(exact=True)
    return sum([centralities[vertex] for vertex in g.vertices()]) / g.order()
add_to_lists(mean_betweenness_centrality, efficient_invariants, all_invariants)

def min_centrality_closeness(g):
    centralities = g.centrality_closeness()
    return centralities[min(centralities)]
add_to_lists(min_centrality_closeness, efficient_invariants, all_invariants)

def max_centrality_closeness(g):
    centralities = g.centrality_closeness()
    return centralities[max(centralities)]
add_to_lists(max_centrality_closeness, efficient_invariants, all_invariants)

def mean_centrality_closeness(g):
    centralities = g.centrality_closeness()
    return sum([centralities[vertex] for vertex in g.vertices()]) / g.order()
add_to_lists(mean_centrality_closeness, efficient_invariants, all_invariants)

def min_centrality_degree(g):
    centralities = g.centrality_degree()
    return centralities[min(centralities)]
add_to_lists(min_centrality_degree, efficient_invariants, all_invariants)

def max_centrality_degree(g):
    centralities = g.centrality_degree()
    return centralities[max(centralities)]
add_to_lists(max_centrality_degree, efficient_invariants, all_invariants)

def mean_centrality_degree(g):
    centralities = g.centrality_degree()
    return sum([centralities[vertex] for vertex in g.vertices()]) / g.order()
add_to_lists(mean_centrality_degree, efficient_invariants, all_invariants)

def homo_lumo_gap(g):
    order = g.order()
    if order % 2 != 0:
        return 0
    eigenvalues = g.spectrum()
    # Minus 1 accounts for the 0 indexing of a list
    return eigenvalues[floor((order+1)/2) - 1] - eigenvalues[ceil((order+1)/2) - 1]
add_to_lists(homo_lumo_gap, efficient_invariants, all_invariants)

def homo_lumo_index(g):
    order = g.order()
    eigenvalues = g.adjacency_matrix(sparse=False).change_ring(RDF).eigenvalues(algorithm="symmetric")
    if order%2 == 0:
        # Minus 1 accounts for the 0 indexing of a list
        return max(abs(eigenvalues[floor((order+1)/2) - 1]), abs(eigenvalues[ceil((order+1)/2) - 1]))
    else:
        return eigenvalues[floor(order/2)]
add_to_lists(homo_lumo_index, efficient_invariants, all_invariants)

def neighborhood_union_nonadjacent(g):
    # Define that for copmlete graphs (i.e. nothing to minimize over later), return n, which is trivial upper bound.
    all_dist = g.distance_all_pairs()
    nonadj = [(v,w) for v in g for w in g if all_dist[v][w] > 1]
    if not nonadj:
        return g.order()
    else:
        return min( len(union(g.neighbors(v), g.neighbors(w))) for (v,w) in nonadj)
add_to_lists(neighborhood_union_nonadjacent, efficient_invariants, all_invariants)

def neighborhood_union_dist2(g):
    # Define that for graphs with no dist 2 (i.e. nothing to minimize over later), return n, which is trivial upper bound.
    all_dist = g.distance_all_pairs()
    dist2 = [(v,w) for v in g for w in g if all_dist[v][w] == 2]
    if not dist2:
        return g.order()
    else:
        return min( len(union(g.neighbors(v), g.neighbors(w))) for (v, w) in dist2)
add_to_lists(neighborhood_union_dist2, efficient_invariants, all_invariants)

def simplical_vertices(g):
    """
    The number of simplical vertices in g.
    v is simplical if the induced nieghborhood is a clique.
    """
    return sum( is_simplical_vertex(g,v) for v in g.vertices() )
add_to_lists(simplical_vertices, efficient_invariants, all_invariants)

def first_zagreb_index(g):
    """
    The sume of squares of the degrees
    """
    return sum(g.degree(v)**2 for v in g.vertices())
add_to_lists(first_zagreb_index, efficient_invariants, all_invariants)

def degree_two_vertices(g):
    """
    The number of degree 2 vertices
    """
    return len([deg for deg in g.degree() if deg == 2])
add_to_lists(degree_two_vertices, efficient_invariants, all_invariants)

def degree_order_minus_one_vertices(g):
    """
    The number of vertices with degree = n-1
    """
    return len([deg for deg in g.degree() if deg == g.order() - 1])
add_to_lists(degree_order_minus_one_vertices, efficient_invariants, all_invariants)

def maximum_degree_vertices(g):
    """
    The number of vertices with degree equal to the maximum degree
    """
    return len([deg for deg in g.degree() if deg == max_degree(g)])
add_to_lists(maximum_degree_vertices, efficient_invariants, all_invariants)

def minimum_degree_vertices(g):
    """
    The number of vertices with degree equal to the minimum degree
    """
    return len([deg for deg in g.degree() if deg == min_degree(g)])
add_to_lists(minimum_degree_vertices, efficient_invariants, all_invariants)

def second_zagreb_index(g):
    """
    The sum over all edges (v,w) of the product of degrees(v)*degree(w)
    """
    return sum(g.degree(v)*g.degree(w) for (v,w) in g.edge_iterator(labels=False))
add_to_lists(second_zagreb_index, efficient_invariants, all_invariants)

# Damir Vukicevic, Qiuli Li, Jelena Sedlar, and Tomislav Doslic, Lanzhou Index. MATCH Commun. Math. Comput. Chem., 80: 863-876, 2018.
def lanzhou_index(g):
    """
    The sum over all vertices v of products of the co-degree of v (deg(v) in the complement of g) times the square of deg(v).

    sage: lanzhou_index(graphs.CompleteGraph(10))
    0
    sage: lanzhou_index(graphs.CompleteBipartiteGraph(5,5))
    1000
    """
    n = g.order()
    return sum( ((n-1) - g.degree(v)) * (g.degree(v) ** 2) for v in g.vertices() )
add_to_lists(lanzhou_index, efficient_invariants, all_invariants)

def friendship_number(g):
    """
    The friendship number of a graph is the number of pairs of vertices that have a unique common neighbour.

    sage: friendship_number(graphs.FriendshipGraph(3))
    21
    sage: friendship_number(graphs.CompleteGraph(7))
    0
    """
    from itertools import combinations
    return sum((1 if len(common_neighbors(g, u, v))==1 else 0) for (u,v) in combinations(g.vertices(), 2))
add_to_lists(friendship_number, efficient_invariants, all_invariants)

#####
# INTRACTABLE INVATIANTS
#####
def domination_number(g):
    """
    Returns the domination number of the graph g, i.e., the size of a maximum
    dominating set.

    A complete graph is dominated by any of its vertices::

        sage: domination_number(graphs.CompleteGraph(5))
        1

    A star graph is dominated by its central vertex::

        sage: domination_number(graphs.StarGraph(5))
        1

    The domination number of a cycle of length n is the ceil of n/3.

        sage: domination_number(graphs.CycleGraph(5))
        2
    """
    return g.dominating_set(value_only=True)
add_to_lists(domination_number, intractable_invariants, all_invariants)

def independence_number(g):
    return g.independent_set(value_only=True)
add_to_lists(independence_number, intractable_invariants, all_invariants)

def clique_covering_number(g):
    # Finding the chromatic number of the complement of a fullerene
    # is extremely slow, even when using MILP as the algorithm.
    # Therefore we check to see if the graph is triangle-free.
    # If it is, then the clique covering number is equal to the
    # number of vertices minus the size of a maximum matching.
    if g.is_triangle_free():
        return g.order() - matching_number(g)
    gc = g.complement()
    return gc.chromatic_number(algorithm="MILP")
add_to_lists(clique_covering_number, intractable_invariants, all_invariants)

def n_over_alpha(g):
    n = g.order() + 0.0
    return n/independence_number(g)
add_to_lists(n_over_alpha, intractable_invariants, all_invariants)

def independent_dominating_set_number(g):
    return g.dominating_set(value_only=True, independent=True)
add_to_lists(independent_dominating_set_number, intractable_invariants, all_invariants)

# Clearly intractable
# alpha / order
def independence_ratio(g):
    return independence_number(g)/(g.order()+0.0)
add_to_lists(independence_ratio, intractable_invariants, all_invariants)

def min_degree_of_max_ind_set(g):
    """
    Returns the minimum degree of any vertex that is a part of any maximum indepdendent set

    sage: min_degree_of_max_ind_set(c4)
    2
    sage: min_degree_of_max_ind_set(graphs.PetersenGraph())
    3
    """

    low_degree = g.order()
    list_of_vertices = []

    UnionSet = Set({})
    IndSets = find_all_max_ind_sets(g)

    for s in IndSets:
        UnionSet = UnionSet.union(Set(s))

    list_of_vertices = list(UnionSet)

    for v in list_of_vertices:
        if g.degree(v) < low_degree:
            low_degree = g.degree(v)

    return low_degree
add_to_lists(min_degree_of_max_ind_set, intractable_invariants, all_invariants)

def bipartite_number(g):
    """
    Defined as the largest number of vertices that induces a bipartite subgraph

    sage: bipartite_number(graphs.PetersenGraph())
    7
    sage: bipartite_number(c4)
    4
    sage: bipartite_number(graphs.CompleteGraph(3))
    2
    """
    if g.is_bipartite():
        return g.order()
    return len(max_bipartite_set(g, [], g.vertices()))
add_to_lists(bipartite_number, intractable_invariants, all_invariants)

# Needs Enhancement
def edge_bipartite_number(g):
    """
    Defined as the largest number of edges in an induced bipartite subgraph

        sage: edge_bipartite_number(graphs.CompleteGraph(5))
        1
        sage: edge_bipartite_number(graphs.CompleteBipartiteGraph(5, 5))
        25
        sage: edge_bipartite_number(graphs.ButterflyGraph())
        2
    """
    return g.subgraph(max_bipartite_set(g, [], g.vertices())).size()
add_to_lists(edge_bipartite_number, intractable_invariants, all_invariants)

def cheeger_constant(g):
    """
    Defined at https://en.wikipedia.org/wiki/Cheeger_constant_(graph_theory)

    sage: cheeger_constant(graphs.PathGraph(2))
    1
    sage: cheeger_constant(graphs.CompleteGraph(5))
    3
    sage: cheeger_constant(paw)
    1
    """
    n = g.order()
    upper = floor(n/2)

    v = g.vertices()
    SetV = Set(v)

    temp = g.order()
    best = n

    for i in range(1, upper+1):
        for s in SetV.subsets(i):
            count = 0
            for u in s:
                for w in SetV.difference(s):
                    for e in g.edges(labels=false):
                        if Set([u,w]) == Set(e):
                            count += 1
            temp = count/i
            if temp < best:
                best = temp
    return best
add_to_lists(cheeger_constant, intractable_invariants, all_invariants)

def tr(g):
    """
    Returns the maximum number of vertex disjoint triangles of the graph

    Harant, Jochen, et al. "The independence number in graphs of maximum degree three." Discrete Mathematics 308.23 (2008): 5829-5833.
    """
    if is_subcubic(g):
        return subcubic_tr(g)
    return independence_number(form_triangles_graph(g))
add_to_lists(tr, intractable_invariants, all_invariants)

def total_domination_number(g):
    return g.dominating_set(total=True, value_only=True)
add_to_lists(total_domination_number, intractable_invariants, all_invariants)

# A graph G is t-tough for real t if for every integer k>1, G cannot be split into k connected components by removal of fewer than tk vertices
# Returns Infinity if g is complete
# Inefficient to calculate
def toughness(g):
    """
    Tests:
        sage: toughness(graphs.PathGraph(3))
        0.5
        sage: toughness(graphs.CompleteGraph(5))
        +Infinity
        sage: toughness(graphs.PetersenGraph())
        1.3333333333333333
    """
    order = g.order()
    t = Infinity
    for x in Subsets(g.vertices()):
        if x and len(x) != order: # Proper, non-empty subset
            H = copy(g)
            H.delete_vertices(x)
            k = H.connected_components_number()
            if k > 1:
                t = min(float(len(x)) / k, t)
    return t
add_to_lists(toughness, intractable_invariants, all_invariants)

# Sigma_k = min( sum( degrees(v_i) : every k-element independent set v_1,..,v_k ) )
# Inefficient to calculate
def sigma_k(g,k):
    """
    Tests:
        sage: sigma_k(graphs.CompleteGraph(5), 1)
        4
        sage: sigma_k(graphs.PathGraph(4), 2)
        2
    """
    sigma = Infinity
    for S in Subsets(g.vertices(), k):
        if g.is_independent_set(S):
            sigma = min(sigma, sum([g.degree(x) for x in S]) )
    return sigma

def sigma_2(g):
    return sigma_k(g,2)
def sigma_3(g):
    return sigma_k(g,3)
add_to_lists(sigma_2, intractable_invariants, all_invariants)
add_to_lists(sigma_3, intractable_invariants, all_invariants)

def homogenous_number(g):
    """
    Equals the larger of the independence number or the clique number
    """
    return max(independence_number(g), g.clique_number())
add_to_lists(homogenous_number, intractable_invariants, all_invariants)

def edge_domination_number(g):
    """
    The minimum size of a set of edges S such that every edge not in S is incident to an edge in S
    """
    return domination_number(g.line_graph())
add_to_lists(edge_domination_number, intractable_invariants, all_invariants)

def circumference(g):
    """
    Returns length of longest cycle in g

    If acyclic, throws a ValueError. Some define this to be 0; we leave it up to the user.
    """
    lengths = cycle_lengths(g)
    if not lengths:
        raise ValueError("Graph is acyclic. Circumference undefined")
    else:
        return max(lengths)
add_to_lists(circumference, intractable_invariants, all_invariants)

def tree_number(g):
    """
    The order of a maximum-size induced subgraph that's a tree in g

    See Erdös, Paul, Michael Saks, and Vera T. Sós. "Maximum induced trees in graphs." Journal of Combinatorial Theory, Series B 41.1 (1986): 61-79.
    """
    return max_induced_tree(g).order()
add_to_lists(tree_number, intractable_invariants, all_invariants)

def forest_number(g):
    """
    The order of a maximum-size induced subgraph of g that's a forest
    """
    return max_induced_forest(g).order()
add_to_lists(forest_number, intractable_invariants, all_invariants)

def minimum_maximal_matching_size(g):
    """
    The minimum number of edges k s.t. there exists a matching of size k which is not extendable
    """
    if(g.size() == 0):
        return 0

    matchings_old = []
    matchings = [[e] for e in g.edges()]
    while True:
        matchings_old = matchings
        matchings = []
        for matching in matchings_old:
            extendable = False
            for e in (edge for edge in g.edges() if edge not in matching):
                possible_matching = matching + [e]
                if is_matching(possible_matching):
                    matchings.append(possible_matching)
                    extendable = True
            if not extendable:
                return len(matching)
add_to_lists(minimum_maximal_matching_size, intractable_invariants, all_invariants)

def hamiltonian_index(g):
    """
    Returns i, where L^i(g) = L(L(L(...L(g)))) is the first line graph iterate of g such that L^i(g) is Hamiltonian

    If g is Hamiltonian, then h(G) = 0.
    Raises ValueError if g is disconnected or if g is a simple path, since h(g) is undefined for either.

    Defined in: Chartrand, Gary. "On hamiltonian line-graphs." Transactions of the American Mathematical Society 134.3 (1968): 559-566.

    sage: hamiltonian_index(graphs.CycleGraph(5))
    0
    sage: hamiltonian_index(graphs.PetersenGraph())
    1
    sage: hamiltonian_index(graphs.TadpoleGraph(4, 3))
    3
    """
    if not g.is_connected():
        raise ValueError("The input graph g is not connected. The Hamiltonian index is only defined for connected graphs.")
    if g.is_isomorphic(graphs.PathGraph(g.order())):
        raise ValueError("The input graph g is a simple path. The Hamiltonian index is not defined for path graphs.")
    line_graph_i = g
    for index in xrange(0, (g.order() - 3) + 1): # [Chartrand, 68] proved index is upper bounded by n - 3.
        if line_graph_i.is_hamiltonian():
            return index
        line_graph_i = line_graph_i.line_graph()
add_to_lists(hamiltonian_index, intractable_invariants, all_invariants)


#FAST ENOUGH (tested for graphs on 140921): lovasz_theta, clique_covering_number, all efficiently_computable
#SLOW but FIXED for SpecialGraphs

#############################################################################
# End of invariants section                                                 #
#############################################################################
# GRAPH PROPERTIES

def has_star_center(g):
    """
    Evalutes whether graph ``g`` has a vertex adjacent to all others.

    EXAMPLES:

        sage: has_star_center(flower_with_3_petals)
        True

        sage: has_star_center(c4)
        False

    Edge cases ::

        sage: has_star_center(Graph(1))
        True

        sage: has_star_center(Graph(0))
        False
    """
    return (g.order() - 1) in g.degree()

def is_complement_of_chordal(g):
    """
    Evaluates whether graph ``g`` is a complement of a chordal graph.

    A chordal graph is one in which all cycles of four or more vertices have a
    chord, which is an edge that is not part of the cycle but connects two
    vertices of the cycle.

    EXAMPLES:

        sage: is_complement_of_chordal(p4)
        True

        sage: is_complement_of_chordal(Graph(4))
        True

        sage: is_complement_of_chordal(p5)
        False

    Any graph without a 4-or-more cycle is vacuously chordal. ::

        sage: is_complement_of_chordal(graphs.CompleteGraph(4))
        True

        sage: is_complement_of_chordal(Graph(3))
        True

        sage: is_complement_of_chordal(Graph(0))
        True
    """
    return g.complement().is_chordal()

def pairs_have_unique_common_neighbor(g):
    """
    Evalaute if each pair of vertices in ``g`` has exactly one common neighbor.

    Also known as the friendship property.
    By the Friendship Theorem, the only connected graphs with the friendship
    property are flowers.

    EXAMPLES:

        sage: pairs_have_unique_common_neighbor(flower(5))
        True

        sage: pairs_have_unique_common_neighbor(k3)
        True

        sage: pairs_have_unique_common_neighbor(k4)
        False

        sage: pairs_have_unique_common_neighbor(graphs.CompleteGraph(2))
        False

    Vacuous cases ::

        sage: pairs_have_unique_common_neighbor(Graph(1))
        True

        sage: pairs_have_unique_common_neighbor(Graph(0))
        True
    """
    from itertools import combinations
    for (u,v) in combinations(g.vertices(), 2):
        if len(common_neighbors(g, u, v)) != 1:
            return False
    return True

def is_distance_transitive(g):
    """
    Evaluates if graph ``g`` is distance transitive.

    A graph is distance transitive if all a,b,u,v satisfy that
    dist(a,b) = dist(u,v) implies there's an automorphism with a->u and b->v.

    EXAMPLES:

        sage: is_distance_transitive(graphs.CompleteGraph(4))
        True

        sage: is_distance_transitive(graphs.PetersenGraph())
        True

        sage: is_distance_transitive(Graph(3))
        True

        sage: is_distance_transitive(graphs.ShrikhandeGraph())
        False

    This method accepts disconnected graphs. ::

        sage: is_distance_transitive(graphs.CompleteGraph(3).disjoint_union(graphs.CompleteGraph(3)))
        True

        sage: is_distance_transitive(graphs.CompleteGraph(2).disjoint_union(Graph(2)))
        False

    Vacuous cases ::

        sage: is_distance_transitive(Graph(0))
        True

        sage: is_distance_transitive(Graph(1))
        True

        sage: is_distance_transitive(Graph(2))
        True

    ... WARNING ::

        This method calls, via the automorphism group, the Gap package. This
        package behaves badly with most threading or multiprocessing tools.
    """
    from itertools import combinations
    dist_dict = g.distance_all_pairs()
    auto_group = g.automorphism_group()

    for d in g.distances_distribution():
        sameDistPairs = []
        for (u,v) in combinations(g.vertices(), 2):
            # By default, no entry if disconnected. We substitute +Infinity.
            if dist_dict[u].get(v, +Infinity) == d:
                sameDistPairs.append(Set([u,v]))
        if len(sameDistPairs) >= 2:
            if len(sameDistPairs) != len(auto_group.orbit(sameDistPairs[0], action = "OnSets")):
                return False
    return True

def is_dirac(g):
    """
    Evaluates if graph ``g`` has order at least 3 and min. degree at least n/2.

    See Dirac's Theorem: If graph is_dirac, then it is hamiltonian.

    EXAMPLES:

        sage: is_dirac(graphs.CompleteGraph(6))
        True

        sage: is_dirac(graphs.CompleteGraph(3))
        True

        sage: is_dirac(graphs.CompleteGraph(2))
        False

        sage: is_dirac(graphs.CycleGraph(5))
        False
    """
    n = g.order()
    return n > 2 and min(g.degree()) >= n/2

def is_ore(g):
    """
    Evaluate if deg(v)+deg(w)>=n for all non-adjacent pairs v,w in graph ``g``.

    See Ore's Theorem: If graph is_ore, then it is hamiltonian.

    EXAMPLES:

        sage: is_ore(graphs.CompleteGraph(5))
        True

        sage: is_ore(graphs.CompleteGraph(2))
        True

        sage: is_ore(dart)
        False

        sage: is_ore(Graph(2))
        False

        sage: is_ore(graphs.CompleteGraph(2).disjoint_union(Graph(1)))
        False

    Vacous cases ::

        sage: is_ore(Graph(0))
        True

        sage: is_ore(Graph(1))
        True
    """
    A = g.adjacency_matrix()
    n = g.order()
    D = g.degree()
    for i in xrange(n):
        for j in xrange(i):
            if A[i][j]==0:
                if D[i] + D[j] < n:
                    return False
    return True

def is_haggkvist_nicoghossian(g):
    """
    Evaluates if g is 2-connected and min degree >= (n + vertex_connectivity)/3.

    INPUT:

    - ``g`` -- graph

    EXAMPLES:

        sage: is_haggkvist_nicoghossian(graphs.CompleteGraph(3))
        True

        sage: is_haggkvist_nicoghossian(graphs.CompleteGraph(5))
        True

        sage: is_haggkvist_nicoghossian(graphs.CycleGraph(5))
        False

        sage: is_haggkvist_nicoghossian(graphs.CompleteBipartiteGraph(4,3)
        False

        sage: is_haggkvist_nicoghossian(Graph(1))
        False

        sage: is_haggkvist_nicoghossian(graphs.CompleteGraph(2))
        False

    REFERENCES:

    Theorem: If a graph ``is_haggkvist_nicoghossian``, then it is Hamiltonian.

    .. [HN1981]     \R. Häggkvist and G. Nicoghossian, "A remark on Hamiltonian
                    cycles". Journal of Combinatorial Theory, Series B, 30(1):
                    118--120, 1981.
    """
    k = g.vertex_connectivity()
    return k >= 2 and min(g.degree()) >= (1.0/3) * (g.order() + k)

def is_genghua_fan(g):
    """
    Evaluates if graph ``g`` satisfies a condition for Hamiltonicity by G. Fan.

    OUTPUT:

    Returns ``True`` if ``g`` is 2-connected and satisfies that
    `dist(u,v)=2` implies `\max(deg(u), deg(v)) \geq n/2` for all
    vertices `u,v`.
    Returns ``False`` otherwise.

    EXAMPLES:

        sage: is_genghua_fan(graphs.DiamondGraph())
        True

        sage: is_genghua_fan(graphs.CycleGraph(4))
        False

        sage: is_genghua_fan(graphs.ButterflyGraph())
        False

        sage: is_genghua_fan(Graph(1))
        False

    REFERENCES:

    Theorem: If a graph ``is_genghua_fan``, then it is Hamiltonian.

    .. [Fan1984]    Geng-Hua Fan, "New sufficient conditions for cycles in
                    graphs". Journal of Combinatorial Theory, Series B, 37(3):
                    221--227, 1984.
    """
    if not is_two_connected(g):
        return False
    D = g.degree()
    Dist = g.distance_all_pairs()
    V = g.vertices()
    n = g.order()
    for i in xrange(n):
        for j in xrange(i):
            if Dist[V[i]][V[j]] == 2 and max(D[i], D[j]) < n / 2.0:
                return False
    return True

def is_planar_transitive(g):
    """
    Evaluates whether graph ``g`` is planar and is vertex-transitive.

    EXAMPLES:

        sage: is_planar_transitive(graphs.HexahedralGraph())
        True

        sage: is_planar_transitive(graphs.CompleteGraph(2))
        True

        sage: is_planar_transitive(graphs.FranklinGraph())
        False

        sage: is_planar_transitive(graphs.BullGraph())
        False

    Vacuous cases ::

        sage: is_planar_transitive(Graph(1))
        True

    Sage defines `Graph(0).is_vertex_transitive() == False``. ::

        sage: is_planar_transitive(Graph(0))
        False
    """
    return g.is_planar() and g.is_vertex_transitive()

def is_generalized_dirac(g):
    """
    Test if ``graph`` g meets condition in a generalization of Dirac's Theorem.

    OUTPUT:

    Returns ``True`` if g is 2-connected and for all non-adjacent u,v,
    the cardinality of the union of neighborhood(u) and neighborhood(v)
    is `>= (2n-1)/3`.

    EXAMPLES:

        sage: is_generalized_dirac(graphs.HouseGraph())
        True

        sage: is_generalized_dirac(graphs.PathGraph(5))
        False

        sage: is_generalized_dirac(graphs.DiamondGraph())
        False

        sage: is_generalized_dirac(Graph(1))
        False

    REFERENCES:

    Theorem: If graph g is_generalized_dirac, then it is Hamiltonian.

    .. [FGJS1989]   \R.J. Faudree, Ronald Gould, Michael Jacobson, and
                    R.H. Schelp, "Neighborhood unions and hamiltonian
                    properties in graphs". Journal of Combinatorial
                    Theory, Series B, 47(1): 1--9, 1989.
    """
    from itertools import combinations

    if not is_two_connected(g):
        return False
    for (u,v) in combinations(g.vertices(), 2):
        if not g.has_edge(u,v):
            if len(neighbors_set(u, v)) < (2.0 * g.order() - 1) / 3:
                return False
    return True

def is_van_den_heuvel(g):
    """
    Evaluates if g meets an eigenvalue condition related to Hamiltonicity.

    INPUT:

    - ``g`` -- graph

    OUTPUT:

    Let ``g`` be of order `n`.
    Let `A_H` denote the adjacency matrix of a graph `H`, and `D_H` denote
    the matrix with the degrees of the vertices of `H` on the diagonal.
    Define `Q_H = D_H + A_H` and `L_H = D_H - A_H` (i.e. the Laplacian).
    Finally, let `C` be the cycle graph on `n` vertices.

    Returns ``True`` if the `i`-th eigenvalue of `L_C` is at most the `i`-th
    eigenvalue of `L_g` and the `i`-th eigenvalue of `Q_C` is at most the
    `i`-th eigenvalue of `Q_g for all `i`.

    EXAMPLES:

        sage: is_van_den_heuvel(graphs.CycleGraph(5))
        True

        sage: is_van_den_heuvel(graphs.PetersenGraph())
        False

    REFERENCES:

    Theorem: If a graph is Hamiltonian, then it ``is_van_den_heuvel``.

    .. [Heu1995]    \J.van den Heuvel, "Hamilton cycles and eigenvalues of
                    graphs". Linear Algebra and its Applications, 226--228:
                    723--730, 1995.

    TESTS::

        sage: is_van_den_heuvel(Graph(0))
        False

        sage: is_van_den_heuvel(Graph(1))
        True
    """
    cycle_n = graphs.CycleGraph(g.order())

    cycle_laplac_eigen = sorted(cycle_n.laplacian_matrix().eigenvalues())
    g_laplac_eigen = sorted(g.laplacian_matrix().eigenvalues())
    for cycle_lambda_i, g_lambda_i in zip(cycle_laplac_eigen, g_laplac_eigen):
        if cycle_lambda_i > g_lambda_i:
            return False

    def Q(g):
        A = g.adjacency_matrix(sparse=False)
        D = matrix(g.order(), sparse=False)
        row_sums = [sum(r) for r in A.rows()]
        for i in xrange(A.nrows()):
            D[i,i] = row_sums[i]
        return D + A
    cycle_q_matrix = sorted(Q(cycle_n).eigenvalues())
    g_q_matrix = sorted(Q(g).eigenvalues())
    for cycle_q_lambda_i, g_q_lambda_i in zip(cycle_q_matrix, g_q_matrix):
        if cycle_q_lambda_i > g_q_lambda_i:
            return False

    return True

def is_two_connected(g):
    """
    Evaluates whether graph ``g`` is 2-connected.

    A 2-connected graph is a connected graph on at least 3 vertices such that
    the removal of any single vertex still gives a connected graph.
    Follows convention that complete graph `K_n` is `n-1`-connected.

    Almost equivalent to ``Graph.is_biconnected()``. We prefer our name. AND,
    while that method defines that ``graphs.CompleteGraph(2)`` is biconnected,
    we follow the convention that `K_n` is `n-1`-connected, so `K_2` is
    only 1-connected.

    EXAMPLES:

        sage: is_two_connected(graphs.CycleGraph(5))
        True

        sage: is_two_connected(graphs.CompleteGraph(3))
        True

        sage: is_two_connected(graphs.PathGraph(5))
        False

        sage: is_two_connected(graphs.CompleteGraph(2))
        False

        sage: is_two_connected(Graph(3))
        False

    Edge cases ::

        sage: is_two_connected(Graph(0))
        False

        sage: is_two_connected(Graph(1))
        False
    """
    if g.is_isomorphic(graphs.CompleteGraph(2)):
        return False
    return g.is_biconnected()

def is_three_connected(g):
    """
    Evaluates whether graph ``g`` is 3-connected.

    A 3-connected graph is a connected graph on at least 4 vertices such that
    the removal of any two vertices still gives a connected graph.
    Follows convention that complete graph `K_n` is `n-1`-connected.

    EXAMPLES:

        sage: is_three_connected(graphs.PetersenGraph())
        True

        sage: is_three_connected(graphs.CompleteGraph(4))
        True

        sage: is_three_connected(graphs.CycleGraph(5))
        False

        sage: is_three_connected(graphs.PathGraph(5))
        False

        sage: is_three_connected(graphs.CompleteGraph(3))
        False

        sage: is_three_connected(graphs.CompleteGraph(2))
        False

        sage: is_three_connected(Graph(4))
        False

    Edge cases ::

        sage: is_three_connected(Graph(0))
        False

        sage: is_three_connected(Graph(1))
        False

    .. WARNING::

        Implementation requires Sage 8.2+.
    """
    return g.vertex_connectivity(k = 3)

def is_four_connected(g):
    """
    Evaluates whether ``g`` is 4-connected.

    A 4-connected graph is a connected graph on at least 5 vertices such that
    the removal of any three vertices still gives a connected graph.
    Follows convention that complete graph `K_n` is `n-1`-connected.

    EXAMPLES:


        sage: is_four_connected(graphs.CompleteGraph(5))
        True

        sage: is_four_connected(graphs.PathGraph(5))
        False

        sage: is_four_connected(Graph(5))
        False

        sage: is_four_connected(graphs.CompleteGraph(4))
        False

    Edge cases ::

        sage: is_four_connected(Graph(0))
        False

        sage: is_four_connected(Graph(1))
        False

    .. WARNING::

        Implementation requires Sage 8.2+.
    """
    return g.vertex_connectivity(k = 4)

def is_lindquester(g):
    """
    Test if graph ``g`` meets a neighborhood union condition for Hamiltonicity.

    OUTPUT:

    Let ``g`` be of order `n`.

    Returns ``True`` if ``g`` is 2-connected and for all vertices `u,v`,
    `dist(u,v) = 2` implies that the cardinality of the union of
    neighborhood(`u`) and neighborhood(`v`) is `\geq (2n-1)/3`.
    Returns ``False`` otherwise.

    EXAMPLES:

        sage: is_lindquester(graphs.HouseGraph())
        True

        sage: is_lindquester(graphs.OctahedralGraph())
        True

        sage: is_lindquester(graphs.PathGraph(3))
        False

        sage: is_lindquester(graphs.DiamondGraph())
        False

    REFERENCES:

    Theorem: If a graph ``is_lindquester``, then it is Hamiltonian.

    .. [Lin1989]    \T.E. Lindquester, "The effects of distance and
                    neighborhood union conditions on hamiltonian properties
                    in graphs". Journal of Graph Theory, 13(3): 335-352,
                    1989.
    """
    if not is_two_connected(g):
        return False
    D = g.distance_all_pairs()
    n = g.order()
    V = g.vertices()
    for i in range(n):
        for j in range(i):
            if D[V[i]][V[j]] == 2:
                if len(neighbors_set(g,[V[i],V[j]])) < (2*n-1)/3.0:
                    return False
    return True

def is_complete(g):
    """
    Tests whether ``g`` is a complete graph.

    OUTPUT:

    Returns ``True`` if ``g`` is a complete graph; returns ``False`` otherwise.
    A complete graph is one where every vertex is connected to every others
    vertex.

    EXAMPLES:

        sage: is_complete(graphs.CompleteGraph(1))
        True

        sage: is_complete(graphs.CycleGraph(3))
        True

        sage: is_complete(graphs.CompleteGraph(6))
        True

        sage: is_complete(Graph(0))
        True

        sage: is_complete(graphs.PathGraph(5))
        False

        sage: is_complete(graphs.CycleGraph(4))
        False
    """
    n = g.order()
    e = g.size()
    if not g.has_multiple_edges():
        return e == n*(n-1)/2
    else:
        D = g.distance_all_pairs()
        for i in range(n):
            for j in range(i):
                if D[V[i]][V[j]] != 1:
                    return False
    return True

def has_c4(g):
    """
    Tests whether graph ``g`` contains Cycle_4 as an *induced* subgraph.

    EXAMPLES:

        sage: has_c4(graphs.CycleGraph(4))
        True

        sage: has_c4(graphs.HouseGraph())
        True

        sage: has_c4(graphs.CycleGraph(5))
        False

        sage: has_c4(graphs.DiamondGraph())
        False
    """
    return g.subgraph_search(c4, induced=True) is not None

def is_c4_free(g):
    """
    Tests whether graph ``g`` does not contain Cycle_4 as an *induced* subgraph.

    EXAMPLES:

        sage: is_c4_free(graphs.CycleGraph(4))
        False

        sage: is_c4_free(graphs.HouseGraph())
        False

        sage: is_c4_free(graphs.CycleGraph(5))
        True

        sage: is_c4_free(graphs.DiamondGraph())
        True
    """
    return not has_c4(g)

def has_paw(g):
    """
    Tests whether graph ``g`` contains a Paw as an *induced* subgraph.

    OUTPUT:

    Define a Paw to be a 4-vertex graph formed by a triangle and a pendant.
    Returns ``True`` if ``g`` contains a Paw as an induced subgraph.
    Returns ``False`` otherwise.

    EXAMPLES:

        sage: has_paw(paw)
        True

        sage: has_paw(graphs.BullGraph())
        True

        sage: has_paw(graphs.ClawGraph())
        False

        sage: has_paw(graphs.DiamondGraph())
        False
    """
    return g.subgraph_search(paw, induced=True) is not None

def is_paw_free(g):
    """
    Tests whether graph ``g`` does not contain a Paw as an *induced* subgraph.

    OUTPUT:

    Define a Paw to be a 4-vertex graph formed by a triangle and a pendant.
    Returns ``False`` if ``g`` contains a Paw as an induced subgraph.
    Returns ``True`` otherwise.

    EXAMPLES:

        sage: is_paw_free(paw)
        False

        sage: is_paw_free(graphs.BullGraph())
        False

        sage: is_paw_free(graphs.ClawGraph())
        True

        sage: is_paw_free(graphs.DiamondGraph())
        True
    """
    return not has_paw(g)

def has_dart(g):
    """
    Tests whether graph ``g`` contains a Dart as an *induced* subgraph.

    OUTPUT:

    Define a Dart to be a 5-vertex graph formed by ``graphs.DiamondGraph()``
    with and a pendant added to one of the degree-3 vertices.
    Returns ``True`` if ``g`` contains a Dart as an induced subgraph.
    Returns ``False`` otherwise.

    EXAMPLES:

        sage: has_dart(dart)
        True

        sage: has_dart(umbrella_4)
        True

        sage: has_dart(graphs.DiamondGraph())
        False

        sage: has_dart(bridge)
        False
    """
    return g.subgraph_search(dart, induced=True) is not None

def is_dart_free(g):
    """
    Tests whether graph ``g`` does not contain a Dart as an *induced* subgraph.

    OUTPUT:

    Define a Dart to be a 5-vertex graph formed by ``graphs.DiamondGraph()``
    with and a pendant added to one of the degree-3 vertices.
    Returns ``False`` if ``g`` contains a Dart as an induced subgraph.
    Returns ``True`` otherwise.

    EXAMPLES:

        sage: is_dart_free(dart)
        False

        sage: is_dart_free(umbrella_4)
        False

        sage: is_dart_free(graphs.DiamondGraph())
        True

        sage: is_dart_free(bridge)
        True
    """
    return not has_dart(g)

def is_p4_free(g):
    """
    Equivalent to is a cograph - https://en.wikipedia.org/wiki/Cograph
    """
    return not has_p4(g)

def has_p4(g):
    """
    Tests whether graph ``g`` contains a Path_4 as an *induced* subgraph.

    Might also be known as "is not a cograph".

    EXAMPLES:

        sage: has_p4(graphs.PathGraph(4))
        True

        sage: has_p4(graphs.CycleGraph(5))
        True

        sage: has_p4(graphs.CycleGraph(4))
        False

        sage: has_p4(graphs.CompleteGraph(5))
        False
    """
    return g.subgraph_search(p4, induced=True) is not None

def has_kite(g):
    """
    Tests whether graph ``g`` contains a Kite as an *induced* subgraph.

    A Kite is a 5-vertex graph formed by a ``graphs.DiamondGraph()`` with a
    pendant attached to one of the degree-2 vertices.

    EXAMPLES:

        sage: has_kite(kite_with_tail)
        True

        sage: has_kite(graphs.KrackhardtKiteGraph())
        True

        sage: has_kite(graphs.DiamondGraph())
        False

        sage: has_kite(bridge)
        False
    """
    return g.subgraph_search(kite_with_tail, induced=True) is not None

def is_kite_free(g):
    """
    Tests whether graph ``g`` does not contain a Kite as an *induced* subgraph.

    A Kite is a 5-vertex graph formed by a ``graphs.DiamondGraph()`` with a
    pendant attached to one of the degree-2 vertices.

    EXAMPLES:

        sage: is_kite_free(kite_with_tail)
        False

        sage: is_kite_free(graphs.KrackhardtKiteGraph())
        False

        sage: is_kite_free(graphs.DiamondGraph())
        True

        sage: is_kite_free(bridge)
        True
    """
    return not has_kite(g)

def has_claw(g):
    """
    Tests whether graph ``g`` contains a Claw as an *induced* subgraph.

    A Claw is a 4-vertex graph with one central vertex and 3 pendants.
    This is encoded as ``graphs.ClawGraph()``.

    EXAMPLES:

        sage: has_claw(graphs.ClawGraph())
        True

        sage: has_claw(graphs.PetersenGraph())
        True

        sage: has_claw(graphs.BullGraph())
        False

        sage: has_claw(graphs.HouseGraph())
        False
    """
    return g.subgraph_search(graphs.ClawGraph(), induced=True) is not None

def is_claw_free(g):
    """
    Tests whether graph ``g`` does not contain a Claw as an *induced* subgraph.

    A Claw is a 4-vertex graph with one central vertex and 3 pendants.
    This is encoded as ``graphs.ClawGraph()``.

    EXAMPLES:

        sage: is_claw_free(graphs.ClawGraph())
        False

        sage: is_claw_free(graphs.PetersenGraph())
        False

        sage: is_claw_free(graphs.BullGraph())
        True

        sage: is_claw_free(graphs.HouseGraph())
        True
    """
    return not has_claw(g)

def has_H(g):
    """
    Tests whether graph ``g`` contains an H graph as an *induced* subgraph.

    An H graph may also be known as a double fork. It is a 6-vertex graph
    formed by two Path_3s with their midpoints joined by a bridge.

    EXAMPLES:

        sage: has_H(double_fork)
        True

        sage: has_H(graphs.PetersenGraph())
        True

        sage: has_H(ce71) # double_fork with extra edge
        False

        sage: has_H(graphs.BullGraph())
        False
    """
    return g.subgraph_search(double_fork, induced=True) is not None

def is_H_free(g):
    """
    Tests if graph ``g`` does not contain a H graph as an *induced* subgraph.

    An H graph may also be known as a double fork. It is a 6-vertex graph
    formed by two Path_3s with their midpoints joined by a bridge.

    EXAMPLES:

        sage: is_H_free(double_fork)
        False

        sage: is_H_free(graphs.PetersenGraph())
        False

        sage: is_H_free(ce71) # double_fork with extra edge
        True

        sage: is_H_free(graphs.BullGraph())
        True
    """
    return not has_H(g)

def has_fork(g):
    """
    Tests if graph ``g`` contains a Fork graph as an *induced* subgraph.

    A Fork graph may also be known as a Star_1_1_3. It is a 6-vertex graph
    formed by a Path_4 with two pendants connected to one end.
    It is stored as `star_1_1_3`.

    EXAMPLES:

        sage: has_fork(star_1_1_3)
        True

        sage: has_fork(graphs.PetersenGraph())
        True

        sage: has_fork(graphs.LollipopGraph(3, 2))
        False

        sage: has_fork(graphs.HouseGraph())
        False

        sage: has_fork(graphs.ClawGraph())
        False
    """
    return g.subgraph_search(star_1_1_3, induced=True) is not None

def is_fork_free(g):
    """
    Tests if graph ``g`` does not contain Fork graph as an *induced* subgraph.

    A Fork graph may also be known as a Star_1_1_3. It is a 6-vertex graph
    formed by a Path_4 with two pendants connected to one end.
    It is stored as `star_1_1_3`.

    EXAMPLES:

        sage: is_fork_free(star_1_1_3)
        False

        sage: is_fork_free(graphs.PetersenGraph())
        False

        sage: is_fork_free(graphs.LollipopGraph(3, 2))
        True

        sage: is_fork_free(graphs.HouseGraph())
        True

        sage: is_fork_free(graphs.ClawGraph())
        True
    """
    return not has_fork(g)

def has_k4(g):
    """
    Tests if graph ``g`` contains a `K_4` as an *induced* subgraph.

    `K_4` is the complete graph on 4 vertices.

    EXAMPLES:

        sage: has_k4(graphs.CompleteGraph(4))
        True

        sage: has_k4(graphs.CompleteGraph(5))
        True

        sage: has_k4(graphs.CompleteGraph(3))
        False

        sage: has_k4(graphs.PetersenGraph())
        False
    """
    return g.subgraph_search(alpha_critical_easy[2], induced=True) is not None

def is_k4_free(g):
    """
    Tests if graph ``g`` does not contain a `K_4` as an *induced* subgraph.

    `K_4` is the complete graph on 4 vertices.

    EXAMPLES:

        sage: is_k4_free(graphs.CompleteGraph(4))
        False

        sage: is_k4_free(graphs.CompleteGraph(5))
        False

        sage: is_k4_free(graphs.CompleteGraph(3))
        True

        sage: is_k4_free(graphs.PetersenGraph())
        True
    """
    return not has_k4(g)

def is_double_clique(g):
    """
    Tests if graph ``g`` can be partitioned into 2 sets which induce cliques.

    EXAMPLE:

        sage: is_double_clique(p4)
        True

        sage: is_double_clique(graphs.ButterflyGraph())
        True

        sage: is_double_clique(graphs.CompleteBipartiteGraph(3,4))
        False

        sage: is_double_clique(graphs.ClawGraph())
        False

        sage: is_double_clique(Graph(3))
        False

    Edge cases ::

        sage: is_double_clique(Graph(0))
        True

        sage: is_double_clique(Graph(1))
        True

        sage: is_double_clique(Graph(2))
        True
    """
    gc = g.complement()
    return gc.is_bipartite()

def has_radius_equal_diameter(g):
    """
    Evaluates whether the radius of graph ``g`` equals its diameter.

    Recall the radius of a graph is the minimum eccentricity over all vertices,
    or the minimum over all longest distances from a vertex to any other vertex.
    Diameter is the maximum eccentricity over all vertices.
    Both radius and diamter are defined to be `+Infinity` for disconnected
    graphs.

    Both radius and diameter are undefined for the empty graph.

    EXAMPLES:

        sage: has_radius_equal_diameter(Graph(4))
        True

        sage: has_radius_equal_diameter(graphs.HouseGraph())
        True

        sage: has_radius_equal_diameter(Graph(1))
        True

        sage: has_radius_equal_diameter(graphs.ClawGraph())
        False

        sage: has_radius_equal_diameter(graphs.BullGraph())
        False
    """
    return g.radius() == g.diameter()

def has_residue_equals_alpha(g):
    """
    Evaluate whether the residue of graph ``g`` equals its independence number.

    The independence number is the cardinality of the largest independent set
    of vertices in ``g``.
    The residue of a graph ``g`` with degrees `d_1 \geq d_2 \geq ... \geq d_n`
    is found iteratively. First, remove `d_1` from consideration and subtract
    `d_1` from the following `d_1` number of elements. Sort. Repeat this
    process for `d_2,d_3, ...` until only 0s remain. The number of elements,
    i.e. the number of 0s, is the residue of ``g``.

    Residue is undefined on the empty graph.

    EXAMPLES:

        sage: has_residue_equals_alpha(graphs.HouseGraph())
        True

        sage: has_residue_equals_alpha(graphs.ClawGraph())
        True

        sage: has_residue_equals_alpha(graphs.CompleteGraph(4))
        True

        sage: has_residue_equals_alpha(Graph(1))
        True

        sage: has_residue_equals_alpha(graphs.PetersenGraph())
        False

        sage: has_residue_equals_alpha(graphs.PathGraph(5))
        False
    """
    return residue(g) == independence_number(g)

def is_not_forest(g):
    """
    Evaluates if graph ``g`` is not a forest.

    A forest is a disjoint union of trees. Equivalently, a forest is any acylic
    graph, which may or may not be connected.

    EXAMPLES:
        sage: is_not_forest(graphs.BalancedTree(2,3))
        False

        sage: is_not_forest(graphs.BalancedTree(2,3).disjoint_union(graphs.BalancedTree(3,3)))
        False

        sage: is_not_forest(graphs.CycleGraph(5))
        True

        sage: is_not_forest(graphs.HouseGraph())
        True

    Edge cases ::

        sage: is_not_forest(Graph(1))
        False

        sage: is_not_forest(Graph(0))
        False
    """
    return not g.is_forest()

def has_empty_KE_part(g):
    """
    Evaluates whether graph ``g`` has an empty Konig-Egervary subgraph.

    A Konig-Egervary graph satisfies
        independence number + matching number = order.
    By [Lar2011]_, every graph contains a unique induced subgraph which is a
    Konig-Egervary graph.

    EXAMPLES:

        sage: has_empty_KE_part(graphs.PetersenGraph())
        True

        sage: has_empty_KE_part(graphs.CycleGraph(5))
        True

        sage: has_empty_KE_part(graphs.CompleteBipartiteGraph(3,4))
        False

        sage: has_empty_KE_part(graphs.CycleGraph(6))
        False

    Edge cases ::

        sage: has_empty_KE_part(Graph(1))
        False

        sage: has_empty_KE_part(Graph(0))
        True

    ALGORITHM:

    This function is implemented using the Maximum Critical Independent
    Set (MCIS) algorithm of [DL2013]_ and applying a Theorem of [Lar2011]_.

    Define that an independent set `I` is a critical independent set if
    `|I|−|N(I)| \geq |J|−|N(J)|` for any independent set J. Define that a
    maximum critical independent set is a critical independent set of maximum
    cardinality.

    By a Theorem of [Lar2011]_, for every maximum critical independent set `J`
    of `G`, the unique Konig-Egervary inducing set `X` is `X = J \cup N(J)`,
    where `N(J)` is the neighborhood of `J`.
    Therefore, the ``g`` has an empty Konig-Egervary induced subgraph if and
    only if the MCIS `J = \emptyset`.

    Next, we describe the MCIS algorithm.
    Let `B(G) = K_2 \ times G`, i.e. `B(G)` is the bipartite double cover
    of `G`. Let `v' \in B(G)` denote the new "twin" of vertex `v \in G`.
    Let `a` be the independence number of `B(G)`.
    For each vertex `v` in `B(G)`, calculate
        `t := independence number(B(G) - \{v,v'\} - N(\{v,v'\})) + 2`.
    If `t = a`, then `v` is in the MCIS.
        Since we only care about whether the MCIS is empty, if `t = a`,
        we return ``False`` and terminate.

    Finally, use the Gallai identities to show matching

    Finally, we apply the Konig-Egervary Theorem (1931) that for all bipartite
    graphs, matching number = vertex cover number. We substitute this into
    one of the Gallai identities, that
        independence number + covering number = order,
    yielding,
        independence number = order - matching number.
    Since matching number is efficient to compute, our final algorithm is
    in fact efficient.

    REFERENCES:

    .. [DL2013]     \Ermelinda DeLaVina and Craig Larson, "A parallel ALGORITHM
                    for computing the critical independence number and related
                    sets". ARS Mathematica Contemporanea 6: 237--245, 2013.

    .. [Lar2011]    \C.E. Larson, "Critical Independent Sets and an
                    Independence Decomposition Theorem". European Journal of
                    Combinatorics 32: 294--300, 2011.
    """
    b = bipartite_double_cover(g)
    alpha = b.order() - b.matching(value_only=True)
    for v in g.vertices():
        test = b.copy()
        test.delete_vertices(closed_neighborhood(b,[(v,0), (v,1)]))
        alpha_test = test.order() - test.matching(value_only=True) + 2
        if alpha_test == alpha:
            return False
    return True

def is_class1(g):
    """
    Evaluates whether the chomatic index of graph ``g`` equals its max degree.

    Let `D` be the maximum degree. By Vizing's Thoerem, all graphs can be
    edge-colored in either `D` or `D+1` colors. The case of `D` colors is
    called "class 1".

    Max degree is undefined for the empty graph.

    EXAMPLES:

        sage: is_class1(graphs.CompleteGraph(4))
        True

        sage: is_class1(graphs.WindmillGraph(4,3))
        True

        sage: is_class1(Graph(1))
        True

        sage: is_class1(graphs.CompleteGraph(3))
        False

        sage: is_class1(graphs.PetersenGraph())
        False
    """
    return g.chromatic_index() == max(g.degree())

def is_class2(g):
    """
    Evaluates whether the chomatic index of graph ``g`` equals max degree + 1.

    Let `D` be the maximum degree. By Vizing's Thoerem, all graphs can be
    edge-colored in either `D` or `D+1` colors. The case of `D+1` colors is
    called "class 2".

    Max degree is undefined for the empty graph.

    EXAMPLES:

        sage: is_class2(graphs.CompleteGraph(4))
        False

        sage: is_class2(graphs.WindmillGraph(4,3))
        False

        sage: is_class2(Graph(1))
        False

        sage: is_class2(graphs.CompleteGraph(3))
        True

        sage: is_class2(graphs.PetersenGraph())
        True
    """
    return not(g.chromatic_index() == max(g.degree()))

def is_cubic(g):
    """
    Evalutes whether graph ``g`` is cubic, i.e. is 3-regular.

    EXAMPLES:

        sage: is_cubic(graphs.CompleteGraph(4))
        True

        sage: is_cubic(graphs.PetersenGraph())
        True

        sage: is_cubic(graphs.CompleteGraph(3))
        False

        sage: is_cubic(graphs.HouseGraph())
        False
    """
    D = g.degree()
    return min(D) == 3 and max(D) == 3

def is_anti_tutte(g):
    """
    Evalutes if graph ``g`` is connected and indep. number <= diameter + girth.

    This property is satisfied by many Hamiltonian graphs, but notably not by
    the Tutte graph ``graphs.TutteGraph()``.

    Diameter is undefined for the empty graph.

    EXAMPLES:

        sage: is_anti_tutte(graphs.CompleteBipartiteGraph(4, 5))
        True

        sage: is_anti_tutte(graphs.PetersenGraph())
        True

        sage: is_anti_tutte(Graph(1))

        sage: is_anti_tutte(graphs.TutteGraph())
        False

        sage: is_anti_tutte(graphs.TutteCoxeterGraph())
        False
    """
    if not g.is_connected():
        return False
    return independence_number(g) <= g.diameter() + g.girth()

def is_anti_tutte2(g):
    """
    Tests if graph ``g`` has indep. number <= domination number + radius - 1.

    ``g`` must also be connected.
    This property is satisfied by many Hamiltonian graphs, but notably not by
    the Tutte graph ``graphs.TutteGraph()``.

    Radius is undefined for the empty graph.

    EXAMPLES:

        sage: is_anti_tutte2(graphs.CompleteGraph(5))
        True

        sage: is_anti_tutte2(graphs.PetersenGraph())
        True

        sage: is_anti_tutte2(graphs.TutteGraph())
        False

        sage: is_anti_tutte2(graphs.TutteCoxeterGraph())
        False

        sage: is_anti_tutte2(Graph(1))
        False
    """
    if not g.is_connected():
        return False
    return independence_number(g) <=  domination_number(g) + g.radius()- 1

def diameter_equals_twice_radius(g):
    """
    Evaluates whether the diameter of graph ``g`` is equal to twice its radius.

    Diameter and radius are undefined for the empty graph.

    EXAMPLES:

        sage: has_radius_equal_diameter(graphs.ClawGraph())
        True

        sage: has_radius_equal_diameter(graphs.KrackhardtKiteGraph())
        True

        sage: diameter_equals_twice_radius(graphs.HouseGraph())
        False

        sage: has_radius_equal_diameter(graphs.BullGraph())
        False

    The radius and diameter of ``Graph(1)`` are both 1. ::

        sage: diameter_equals_twice_radius(Graph(1))
        True

    Disconnected graphs have both diameter and radius equal infinity.

        sage: diameter_equals_twice_radius(Graph(4))
        True
    """
    return g.diameter() == 2*g.radius()

def diameter_equals_two(g):
    """
    Evaluates whether the diameter of graph ``g`` equals 2.

    Diameter is undefined for the empty graph.

    EXAMPLES:

        sage: diameter_equals_two(graphs.ClawGraph())
        True

        sage: diameter_equals_two(graphs.HouseGraph())
        True

        sage: diameter_equals_two(graphs.KrackhardtKiteGraph())
        False

        sage: diameter_equals_two(graphs.BullGraph())
        False

    Disconnected graphs have diameter equals infinity.

        sage: diameter_equals_two(Graph(4))
        False
    """
    return g.diameter() == 2

def has_lovasz_theta_equals_alpha(g):
    """
    Tests if the Lovasz number of graph ``g`` equals its independence number.

    Examples:

        sage: has_lovasz_theta_equals_alpha(graphs.CompleteGraph(12))
        True

        sage: has_lovasz_theta_equals_alpha(double_fork)
        True

        sage: has_lovasz_theta_equals_alpha(graphs.PetersenGraph())
        True

        sage: has_lovasz_theta_equals_alpha(graphs.ClebschGraph())
        False

        sage: has_lovasz_theta_equals_alpha(graphs.CycleGraph(24))
        False

    True for all graphs with no edges ::

        sage: has_lovasz_theta_equals_alpha(Graph(12))
        True

    Edge cases ::

        sage: has_lovasz_theta_equals_alpha(Graph(0))
        True

        # Broken. Issue #584
        sage: has_lovasz_theta_equals_alpha(Graph(1)) # doctest: +SKIP
        True
    """
    return g.lovasz_theta() == independence_number(g)

def has_lovasz_theta_equals_cc(g):
    """
    Test if the Lovasz number of graph ``g`` equals its clique covering number.

    Examples:

        sage: has_lovasz_theta_equals_cc(graphs.CompleteGraph(12))
        True

        sage: has_lovasz_theta_equals_cc(double_fork)
        True

        sage: has_lovasz_theta_equals_cc(graphs.PetersenGraph())
        True

        sage: has_lovasz_theta_equals_cc(Graph(12))
        True

        sage: has_lovasz_theta_equals_cc(graphs.ClebschGraph())
        False

        has_lovasz_theta_equals_alpha(graphs.BuckyBall())
        False

    Edge cases ::

        sage: has_lovasz_theta_equals_cc(Graph(0))
        True

        # Broken. Issue #584
        sage: has_lovasz_theta_equals_cc(Graph(1)) # doctest: +SKIP
        True
    """
    return g.lovasz_theta() == clique_covering_number(g)

def is_chvatal_erdos(g):
    """
    Evaluates whether graph ``g`` meets a Hamiltonicity condition of [CV1972]_.

    OUTPUT:

    Returns ``True`` if the independence number of ``g`` is less than or equal
    to the vertex connectivity of ``g``.
    Returns ``False`` otherwise.

    EXAMPLES:

        sage: is_chvatal_erdos(graphs.CompleteGraph(5))
        True

        sage: is_chvatal_erdos(graphs.CycleGraph(5))
        True

        sage: is_chvatal_erdos(graphs.CompleteGraph(2))
        True

        sage: is_chvatal_erdos(graphs.PetersenGraph())
        False

        sage: is_chvatal_erdos(graphs.ClawGraph())
        False

        sage: is_chvatal_erdos(graphs.DodecahedralGraph())
        False

    Edge cases ::

        sage: is_chvatal_erdos(Graph(1))
        False

        sage: is_chvatal_erdos(Graph(0))
        True

    REFERENCES:

    Theorem: If a graph ``is_chvatal_erdos``, then it is Hamiltonian.

    .. [CV1972]     \V. Chvatal and P. Erdos, "A note on hamiltonian cycles".
                    Discrete Mathematics, 2(2): 111--113, 1972.
    """
    return independence_number(g) <= g.vertex_connectivity()

def matching_covered(g):
    """
    Skipping because broken. See Issue #585.
    """
    g = g.copy()
    nu = matching_number(g)
    E = g.edges()
    for e in E:
        g.delete_edge(e)
        nu2 = matching_number(g)
        if nu != nu2:
            return False
        g.add_edge(e)
    return True

def radius_greater_than_center(g):
    """
    Test if connected graph ``g`` has radius greater than num. of center verts.

    If ``g`` is not connected, returns ``False``.
    Radius is undefined for the empty graph.

    EXAMPLES:

        sage: radius_greater_than_center(graphs.TutteGraph())
        True

        sage: radius_greater_than_center(graphs.KrackhardtKiteGraph())
        True

        sage: radius_greater_than_center(graphs.SousselierGraph())
        True

        sage: radius_greater_than_center(graphs.PetersenGraph())
        False

        sage: radius_greater_than_center(graphs.DiamondGraph())
        False

        sage: radius_greater_than_center(Graph(1))
        False
    """
    return g.is_connected() and g.radius() > card_center(g)

def avg_distance_greater_than_girth(g):
    """
    Tests if graph ``g`` is connected and avg. distance greater than the girth.

    Average distance is undefined for 1- and 0- vertex graphs.

    EXAMPLES:

        sage: avg_distance_greater_than_girth(graphs.TutteGraph())
        True

        sage: avg_distance_greater_than_girth(graphs.HarborthGraph())
        True

        sage: avg_distance_greater_than_girth(graphs.HortonGraph())
        True

        sage: avg_distance_greater_than_girth(graphs.BullGraph())
        False

        sage: avg_distance_greater_than_girth(Graph("NC`@A?_C?@_JA??___W"))
        False

        sage: avg_distance_greater_than_girth(Graph(2))
        False

    Acyclic graphs have girth equals infinity. ::

        sage: avg_distance_greater_than_girth(graphs.CompleteGraph(2))
        False
    """
    return g.is_connected() and g.average_distance() > g.girth()

def chi_equals_min_theory(g):
    """
    Evaluate if chromatic num. of graph ``g`` equals min. of some upper bounds.

    Some known upper bounds on the chromatic number Chi (`\chi`) include
    our invariants `[brooks, wilf, welsh_powell, szekeres_wilf]`.
    Returns ``True`` if the actual chromatic number of ``g`` equals the minimum
    of / "the best of" these known upper bounds.

    Some of these invariants are undefined on the empty graph.

    EXAMPLES:

        sage: chi_equals_min_theory(Graph(1))
        True

        sage: chi_equals_min_theory(graphs.PetersenGraph())
        True

        sage: chi_equals_min_theory(double_fork)
        True

        sage: chi_equals_min_theory(Graph(3))
        False

        chi_equals_min_theory(graphs.CompleteBipartiteGraph(3,5))
        False

        chi_equals_min_theory(graphs.IcosahedralGraph())
        False
    """
    chromatic_upper_theory = [brooks, wilf, welsh_powell, szekeres_wilf]
    min_theory = min([f(g) for f in chromatic_upper_theory])
    return min_theory == g.chromatic_number()

def is_heliotropic_plant(g):
    """
    Evaluates whether graph ``g`` is a heliotropic plant. BROKEN

    BROKEN: code should be nonnegative eigen, not just positive eigen.
    See Issue #586

    A graph is heliotropic iff the independence number equals the number of
    nonnegative eigenvalues.

    See [BDF1995]_ for a definition and some related conjectures, where
    [BDF1995]_ builds on the conjecturing work of Siemion Fajtlowicz.

    EXAMPLES:

    REFERENCES:

    .. [BDF1995]    Tony Brewster, Michael J.Dinneen, and Vance Faber, "A
                    computational attack on the conjectures of Graffiti: New
                    counterexamples and proofs". Discrete Mathematics,
                    147(1--3): 35--55, 1995.
    """
    return (independence_number(g) == card_positive_eigenvalues(g))

def is_geotropic_plant(g):
    """
    Evaluates whether graph ``g`` is a heliotropic plant. BROKEN

    BROKEN: code should be nonpositive eigen, not just negative eigen.
    See Issue #586

    A graph is geotropic iff the independence number equals the number of
    nonnegative eigenvalues.

    See [BDF1995]_ for a definition and some related conjectures, where
    [BDF1995]_ builds on the conjecturing work of Siemion Fajtlowicz.

    EXAMPLES:

    REFERENCES:

    .. [BDF1995]    Tony Brewster, Michael J.Dinneen, and Vance Faber, "A
                    computational attack on the conjectures of Graffiti: New
                    counterexamples and proofs". Discrete Mathematics,
                    147(1--3): 35--55, 1995.
    """
    return (independence_number(g) == card_negative_eigenvalues(g))

def is_traceable(g):
    """
    Evaluates whether graph ``g`` is traceable.

    A graph ``g`` is traceable iff there exists a Hamiltonian path, i.e. a path
    which visits all vertices in ``g`` once.
    This is different from ``is_hamiltonian``, since that checks if there
    exists a Hamiltonian *cycle*, i.e. a path which then connects backs to
    its starting point.

    EXAMPLES:

        sage: is_traceable(graphs.CompleteGraph(5))
        True

        sage: is_traceable(graphs.PathGraph(5))
        True

        sage: is_traceable(graphs.PetersenGraph())
        True

        sage: is_traceable(graphs.CompleteGraphs(2))
        True

        sage: is_traceable(Graph(3))
        False

        sage: is_traceable(graphs.ClawGraph())
        False

        sage: is_traceable(graphs.ButterflyGraph())
        False

    Edge cases ::

        sage: is_traceable(Graph(0))
        False

        sage: is_traceable(Graph(1))
        False

    ALGORITHM:

    A graph `G` is traceable iff the join `G'` of `G` with a single new vertex
    `v` is Hamiltonian, where join means to connect every vertex of `G` to the
    new vertex `v`.
    Why? Suppose there exists a Hamiltonian path between `u` and `w` in `G`.
    Then, in `G'`, make a cycle from `v` to `u` to `w` and back to `v`.
    For the reverse direction, just note that the additional vertex `v` cannot
    "help" since Hamiltonian paths can only visit any vertex once.
    """
    gadd = g.join(Graph(1),labels="integers")
    return gadd.is_hamiltonian()

def has_residue_equals_two(g):
    """
    Evaluates whether the residue of graph ``g`` equals 2.

    The residue of a graph ``g`` with degrees `d_1 \geq d_2 \geq ... \geq d_n`
    is found iteratively. First, remove `d_1` from consideration and subtract
    `d_1` from the following `d_1` number of elements. Sort. Repeat this
    process for `d_2,d_3, ...` until only 0s remain. The number of elements,
    i.e. the number of 0s, is the residue of ``g``.

    Residue is undefined on the empty graph.

    EXAMPLES:

        sage: has_residue_equals_two(graphs.ButterflyGraph())
        True

        sage: has_residue_equals_two(graphs.IcosahedralGraph())
        True

        sage: has_residue_equals_two(graphs.OctahedralGraph())
        True

        sage: has_residue_equals_two(Graph(1))
        False

        sage: has_residue_equals_two(graphs.BullGraph())
        False

        sage: has_residue_equals_two(graphs.BrinkmannGraph())
        False
    """
    return residue(g) == 2

def is_chordal_or_not_perfect(g):
    """
    Evaluates if graph ``g`` is either chordal or not perfect.

    There is a known theorem that every chordal graph is perfect.

    OUTPUT:

    Returns ``True`` iff ``g`` is chordal OR (inclusive or) ``g`` is not
    perfect.

    EXAMPLES:

        sage: is_chordal_or_not_perfect(graphs.DiamondGraph())
        True

        sage: is_chordal_or_not_perfect(graphs.CycleGraph(5))
        True

        sage: is_chordal_or_not_perfect(graphs.LollipopGraph(5,3))
        True

        sage: is_chordal_or_not_perfect(graphs.CycleGraph(4))
        False

        sage: is_chordal_or_not_perfect(graphs.HexahedralGraph())
        False

    Vacuously chordal cases ::

        sage: is_chordal_or_not_perfect(Graph(0))
        True

        sage: is_chordal_or_not_perfect(Graph(1))
        True

        sage: is_complement_of_chordal(Graph(4))
        True
    """
    if g.is_chordal():
        return true
    else:
        return not g.is_perfect()

def has_alpha_residue_equal_two(g):
    """
    Tests if both the residue and independence number of graphs ``g`` equal 2.

    The residue of a graph ``g`` with degrees `d_1 \geq d_2 \geq ... \geq d_n`
    is found iteratively. First, remove `d_1` from consideration and subtract
    `d_1` from the following `d_1` number of elements. Sort. Repeat this
    process for `d_2,d_3, ...` until only 0s remain. The number of elements,
    i.e. the number of 0s, is the residue of ``g``.

    Residue is undefined on the empty graph.

    EXAMPLES:

        sage: has_alpha_residue_equal_two(graphs.DiamondGraph())
        True

        sage: has_alpha_residue_equal_two(Graph(2))
        True

        sage: has_alpha_residue_equal_two(graphs.OctahedralGraph())
        True

        sage: has_alpha_residue_equal_two(graphs.BullGraph())
        False

        sage: has_alpha_residue_equal_two(graphs.BidiakisCube())
        False

        sage: has_alpha_residue_equal_two(Graph(3))
        False

        sage: has_alpha_residue_equal_two(Graph(1))
        False
    """
    if residue(g) != 2:
        return false
    else:
        return independence_number(g) == 2

def alpha_leq_order_over_two(g):
    """
    Tests if the independence number of graph ``g`` is at most half its order.

    EXAMPLES:

        sage: alpha_leq_order_over_two(graphs.ButterflyGraph())
        True

        sage: alpha_leq_order_over_two(graphs.DiamondGraph())
        True

        sage: alpha_leq_order_over_two(graphs.CoxeterGraph())
        True

        sage: alpha_leq_order_over_two(Graph(4))
        False

        sage: alpha_leq_order_over_two(graphs.BullGraph())
        False

    Edge cases ::

        sage: alpha_leq_order_over_two(Graph(0))
        True

        sage: alpha_leq_order_over_two(Graph(1))
        False
    """
    return (2*independence_number(g) <= g.order())

def order_leq_twice_max_degree(g):
    """
    Tests if the order of graph ``g`` is at most twice the max of its degrees.

    Undefined for the empty graph.

    EXAMPLES:

        sage: order_leq_twice_max_degree(graphs.BullGraph())
        True

        sage: order_leq_twice_max_degree(graphs.ThomsenGraph())
        True

        sage: order_leq_twice_max_degree(graphs.CycleGraph(4))
        True

        sage: order_leq_twice_max_degree(graphs.BidiakisCube())
        False

        sage: order_leq_twice_max_degree(graphs.CycleGraph(5))
        False

        sage: order_leq_twice_max_degree(Graph(1))
        False
    """
    return (g.order() <= 2*max(g.degree()))

def is_chromatic_index_critical(g):
    """
    Evaluates whether graph ``g`` is chromatic index critical.

    Let `\chi(G)` denote the chromatic index of a graph `G`.
    Then `G` is chromatic index critical if `\chi(G-e) < \chi(G)` (strictly
    less than) for all `e \in G` AND if (by definition) `G` is class 2.

    See [FW1977]_ for a more extended definition and discussion.

    We initially found it surprising that `G` is required to be class 2; for
    example, the Star Graph is a class 1 graph which satisfies the rest of
    the definition. We have found articles which equivalently define critical
    graphs as class 2 graphs which become class 1 when any edge is removed.
    Perhaps this latter definition inspired the one we state above?

    Max degree is undefined on the empty graph, so ``is_class`` is also
    undefined. Therefore this property is undefined on the empty graph.

    EXAMPLES:

        sage: is_chromatic_index_critical(Graph('Djk'))
        True

        sage: is_chromatic_index_critical(graphs.CompleteGraph(3))
        True

        sage: is_chromatic_index_critical(graphs.CycleGraph(5))
        True

        sage: is_chromatic_index_critical(graphs.CompleteGraph(5))
        False

        sage: is_chromatic_index_critical(graphs.PetersenGraph())
        False

        sage: is_chromatic_index_critical(graphs.FlowerSnark())
        False

    Non-trivially disconnected graphs ::

        sage: is_chromatic_index_critical(graphs.CycleGraph(4).disjoint_union(graphs.CompleteGraph(4)))
        False

    Class 1 graphs ::

        sage: is_chromatic_index_critical(Graph(1))
        False

        sage: is_chromatic_index_critical(graphs.CompleteGraph(4))
        False

        sage: is_chromatic_index_critical(graphs.CompleteGraph(2))
        False

        sage: is_chromatic_index_critical(graphs.StarGraph(4))
        False

    ALGORITHM:

    This function uses a series of tricks to reduce the number of cases that
    need to be considered, before finally checking in the obvious way.

    First, if a graph has more than 1 non-trivial connected component, then
    return ``False``. This is because in a graph with multiple such components,
    removing any edges from the smaller component cannot affect the chromatic
    index.

    Second, check if the graph is class 2. If not, stop and return ``False``.

    Finally, identify isomorphic edges using the line graph and its orbits.
    We then need only check the non-equivalent edges to see that they reduce
    the chromatic index when deleted.

    REFERENCES:

    .. [FW1977]     \S. Fiorini and R.J. Wilson, "Edge-colourings of graphs".
                    Pitman Publishing, London, UK, 1977.
    """
    component_sizes = g.connected_components_sizes()
    if len(component_sizes) > 1:
        if component_sizes[1] > 1:
            return False

    if chi == max_degree(g):
        return False

    lg = g.line_graph()
    equiv_lines = lg.automorphism_group(return_group=False, orbits=true)
    equiv_lines_representatives = [orb[0] for orb in equiv_lines]

    gc = g.copy()
    for e in equiv_lines_representatives:
        gc.delete_edge(e)
        chi_prime = gc.chromatic_index()
        if chi_prime == chi:
            return False
        gc.add_edge(e)
    return True

#alpha(g-e) > alpha(g) for *every* edge g
def is_alpha_critical(g):
    #if not g.is_connected():
        #return False
    alpha = independence_number(g)
    for e in g.edges():
        gc = copy(g)
        gc.delete_edge(e)
        alpha_prime = independence_number(gc)
        if alpha_prime <= alpha:
            return False
    return True

#graph is KE if matching number + independence number = n, test does *not* compute alpha
def is_KE(g):
    return g.order() == len(find_KE_part(g))

#graph is KE if matching number + independence number = n, test comoutes alpha
#def is_KE(g):
#    return (g.matching(value_only = True) + independence_number(g) == g.order())

#possibly faster version of is_KE (not currently in invariants)
#def is_KE2(g):
#    return (independence_number(g) == critical_independence_number(g))

def is_independence_irreducible(g):
    return g.order() == card_independence_irreducible_part(g)


def is_factor_critical(g):
    """
    a graph is factor-critical if order is odd and removal of any vertex gives graph with perfect matching
        is_factor_critical(graphs.PathGraph(3))
        False
        sage: is_factor_critical(graphs.CycleGraph(5))
        True
    """
    if g.order() % 2 == 0:
        return False
    for v in g.vertices():
        gc = copy(g)
        gc.delete_vertex(v)
        if not gc.has_perfect_matching:
            return False
    return True

#returns a list of (necessarily non-adjacent) vertices that have the same neighbors as v if a pair exists or None
def find_twins_of_vertex(g,v):
    L = []
    V = g.vertices()
    D = g.distance_all_pairs()
    for i in range(g.order()):
        w = V[i]
        if  D[v][w] == 2 and g.neighbors(v) == g.neighbors(w):
                L.append(w)
    return L

def has_twin(g):
    t = find_twin(g)
    if t == None:
        return False
    else:
        return True

def is_twin_free(g):
    return not has_twin(g)

#returns twin pair (v,w) if one exists, else returns None
#can't be adjacent
def find_twin(g):

    V = g.vertices()
    for v in V:
        Nv = set(g.neighbors(v))
        for w in V:
            Nw = set(g.neighbors(w))
            if v not in Nw and Nv == Nw:
                return (v,w)
    return None

def find_neighbor_twins(g):
    V = g.vertices()
    for v in V:
        Nv = g.neighbors(v)
        for w in Nv:
            if set(closed_neighborhood(g,v)) == set(closed_neighborhood(g,w)):
                return (v,w)
    return None

#given graph g and subset S, looks for any neighbor twin of any vertex in T
#if result = T, then no twins, else the result is maximal, but not necessarily unique
def find_neighbor_twin(g, T):
    gT = g.subgraph(T)
    for v in T:
        condition = False
        Nv = set(g.neighbors(v))
        #print "v = {}, Nv = {}".format(v,Nv)
        NvT = set(gT.neighbors(v))
        for w in Nv:
            NwT = set(g.neighbors(w)).intersection(set(T))
            if w not in T and NvT.issubset(NwT):
                T.append(w)
                condition = True
                #print "TWINS: v = {}, w = {}, sp3 = {}".format(v,w,sp3)
                break
        if condition == True:
            break

#if result = T, then no twins, else the result is maximal, but not necessarily unique
def iterative_neighbor_twins(g, T):
    T2 = copy(T)
    find_neighbor_twin(g, T)
    while T2 != T:
        T2 = copy(T)
        find_neighbor_twin(g, T)
    return T


#can't compute membership in this class directly. instead testing isomorhism for 400 known class0 graphs
def is_pebbling_class0(g):
    for hkey in class0graphs_dict:
        h = Graph(class0graphs_dict[hkey])
        if g.is_isomorphic(h):
            return True
    return False

def girth_greater_than_2log(g):
    return bool(g.girth() > 2*log(g.order(),2))

def szekeres_wilf_equals_chromatic_number(g):
    return szekeres_wilf(g) == g.chromatic_number()

def has_Havel_Hakimi_property(g, v):
    """
    This function returns whether the vertex v in the graph g has the Havel-Hakimi
    property as defined in [1]. A vertex has the Havel-Hakimi property if it has
    maximum degree and the minimum degree of its neighbours is at least the maximum
    degree of its non-neigbours.

    [1] Graphs with the strong Havel-Hakimi property, M. Barrus, G. Molnar, Graphs
        and Combinatorics, 2016, http://dx.doi.org/10.1007/s00373-015-1674-7

    Every vertex in a regular graph has the Havel-Hakimi property::

        sage: P = graphs.PetersenGraph()
        sage: for v in range(10):
        ....:     has_Havel_Hakimi_property(P,v)
        True
        True
        True
        True
        True
        True
        True
        True
        True
        True
        sage: has_Havel_Hakimi_property(Graph([[0,1,2,3],lambda x,y: False]),0)
        True
        sage: has_Havel_Hakimi_property(graphs.CompleteGraph(5),0)
        True
    """
    if max_degree(g) > g.degree(v): return False

    #handle the case where the graph is an independent set
    if len(g.neighbors(v)) == 0: return True

    #handle the case where v is adjacent with all vertices
    if len(g.neighbors(v)) == len(g.vertices()) - 1: return True

    return (min(g.degree(nv) for nv in g.neighbors(v)) >=
        max(g.degree(nnv) for nnv in g.vertices() if nnv != v and nnv not in g.neighbors(v)))

def has_strong_Havel_Hakimi_property(g):
    """
    This function returns whether the graph g has the strong Havel-Hakimi property
    as defined in [1]. A graph has the strong Havel-Hakimi property if in every
    induced subgraph H of G, every vertex of maximum degree has the Havel-Hakimi
    property.

    [1] Graphs with the strong Havel-Hakimi property, M. Barrus, G. Molnar, Graphs
        and Combinatorics, 2016, http://dx.doi.org/10.1007/s00373-015-1674-7

    The graph obtained by connecting two cycles of length 3 by a single edge has
    the strong Havel-Hakimi property::

        sage: has_strong_Havel_Hakimi_property(Graph('E{CW'))
        True
    """
    for S in Subsets(g.vertices()):
        if len(S)>2:
            H = g.subgraph(S)
            Delta = max_degree(H)
            if any(not has_Havel_Hakimi_property(H, v) for v in S if H.degree(v) == Delta):
                return False
    return True

# Graph is subcubic is each vertex is at most degree 3
def is_subcubic(g):
    return max_degree(g) <= 3

# Max and min degree varies by at most 1
def is_quasi_regular(g):
    if max_degree(g) - min_degree(g) < 2:
        return true
    return false

# g is bad if a block is isomorphic to k3, c5, k4*, c5*
def is_bad(g):
    blocks = g.blocks_and_cut_vertices()[0]
    # To make a subgraph of g from the ith block
    for i in blocks:
        h = g.subgraph(i)
        boolean = h.is_isomorphic(alpha_critical_easy[1]) or h.is_isomorphic(alpha_critical_easy[4]) or h.is_isomorphic(alpha_critical_easy[5]) or h.is_isomorphic(alpha_critical_easy[21])
        if boolean == True:
            return True
    return False

# Graph g is complement_hamiltonian if the complement of the graph is hamiltonian.
def is_complement_hamiltonian(g):
    return g.complement().is_hamiltonian()

# A graph is unicyclic if it is connected and has order == size
# Equivalently, graph is connected and has exactly one cycle
def is_unicyclic(g):
    """
    Tests:
        sage: is_unicyclic(graphs.BullGraph())
        True
        sage: is_unicyclic(graphs.ButterflyGraph())
        False
    """
    return g.is_connected() and g.order() == g.size()

def is_k_tough(g,k):
    return toughness(g) >= k # In invariants
def is_1_tough(g):
    return is_k_tough(g, 1)
def is_2_tough(g):
    return is_k_tough(g, 2)

# True if graph has at least two hamiltonian cycles. The cycles may share some edges.
def has_two_ham_cycles(gIn):
    g = gIn.copy()
    g.relabel()
    try:
        ham1 = g.hamiltonian_cycle()
    except EmptySetError:
        return False

    for e in ham1.edges():
        h = copy(g)
        h.delete_edge(e)
        if h.is_hamiltonian():
            return true
    return false

def has_simplical_vertex(g):
    """
    v is a simplical vertex if induced neighborhood is a clique.
    """
    for v in g.vertices():
        if is_simplical_vertex(g, v):
            return true
    return false

def has_exactly_two_simplical_vertices(g):
    """
    v is a simplical vertex if induced neighborhood is a clique.
    """
    return simplical_vertices(g) == 2

def is_two_tree(g):
    """
    Define k-tree recursively:
        - Complete Graph on (k+1)-vertices is a k-tree
        - A k-tree on n+1 vertices is constructed by selecting some k-tree on n vertices and
            adding a degree k vertex such that its open neighborhood is a clique.
    """
    if(g.is_isomorphic(graphs.CompleteGraph(3))):
        return True

    # We can just recurse from any degree-2 vertex; no need to test is_two_tree(g-w) if is_two_tree(g-v) returns False.
    # Intuition why: if neighborhood of a degree-2 v is not a triangle, it won't become one if we remove w (so clique check OK),
    # and, removing a degree-2 vertex of one triangle cannot destroy another triangle (so recursion OK).
    degree_two_vertices = (v for v in g.vertices() if g.degree(v) == 2)
    try:
        v = next(degree_two_vertices)
    except StopIteration: # Empty list. No degree 2 vertices.
        return False

    if not g.has_edge(g.neighbors(v)): # Clique
        return false
    g2 = g.copy()
    g2.delete_vertex(v)
    return is_two_tree(g2)

def is_two_path(g):
    """
    Graph g is a two_path if it is a two_tree and has exactly 2 simplical vertices
    """
    return has_exactly_two_simplical_vertices(g) and is_two_tree(g)

def is_prism_hamiltonian(g):
    """
    A graph G is prism hamiltonian if G x K2 (cartesian product) is hamiltonian
    """
    return g.cartesian_product(graphs.CompleteGraph(2)).is_hamiltonian()

# Bauer, Douglas, et al. "Long cycles in graphs with large degree sums." Discrete Mathematics 79.1 (1990): 59-70.
def is_bauer(g):
    """
    True if g is 2_tough and sigma_3 >= order
    """
    return is_2_tough(g) and sigma_k(g, 3) >= g.order()

# Jung, H. A. "On maximal circuits in finite graphs." Annals of Discrete Mathematics. Vol. 3. Elsevier, 1978. 129-144.
def is_jung(g):
    """
    True if graph has n >= 11, if graph is 1-tough, and sigma_2 >= n - 4.
    See functions toughness(g) and sigma_2(g) for more details.
    """
    return g.order() >= 11 and is_1_tough(g) and sigma_2(g) >= g.order() - 4

# Bela Bollobas and Andrew Thomason, Weakly Pancyclic Graphs. Journal of Combinatorial Theory 77: 121--137, 1999.
def is_weakly_pancyclic(g):
    """
    True if g contains cycles of every length k from k = girth to k = circumfrence

    Returns False if g is acyclic (in which case girth = circumfrence = +Infinity).

    sage: is_weakly_pancyclic(graphs.CompleteGraph(6))
    True
    sage: is_weakly_pancyclic(graphs.PetersenGraph())
    False
    """
    lengths = cycle_lengths(g)
    if not lengths: # acyclic
        raise ValueError("Graph is acyclic. Property undefined.")
    else:
        return lengths == set(range(min(lengths),max(lengths)+1))

def is_pancyclic(g):
    """
    True if g contains cycles of every length from 3 to g.order()

    sage: is_pancyclic(graphs.OctahedralGraph())
    True
    sage: is_pancyclic(graphs.CycleGraph(10))
    False
    """
    lengths = cycle_lengths(g)
    return lengths == set(range(3, g.order()+1))

def has_two_walk(g):
    """
    A two-walk is a closed walk that visits every vertex and visits no vertex more than twice.

    Two-walk is a generalization of Hamiltonian cycles. If a graph is Hamiltonian, then it has a two-walk.

    sage: has_two_walk(c4c4)
    True
    sage: has_two_walk(graphs.WindmillGraph(3,3))
    False
    """
    for init_vertex in g.vertices():
        path_stack = [[init_vertex]]
        while path_stack:
            path = path_stack.pop()
            for neighbor in g.neighbors(path[-1]):
                if neighbor == path[0] and all(v in path for v in g.vertices()):
                    return True
                elif path.count(neighbor) < 2:
                    path_stack.append(path + [neighbor])
    return False

def is_claw_free_paw_free(g):
    return is_claw_free(g) and is_paw_free(g)

def has_bull(g):
    """
    True if g has an induced subgraph isomorphic to graphs.BullGraph()
    """
    return g.subgraph_search(graphs.BullGraph(), induced = True) != None

def is_bull_free(g):
    """
    True if g does not have an induced subgraph isomoprhic to graphs.BullGraph()
    """
    return not has_bull(g)

def is_claw_free_bull_free(g):
    return is_claw_free(g) and is_bull_free(g)

def has_F(g):
    """
    Let F be a triangle with 3 pendants. True if g has an induced F.
    """
    F = graphs.CycleGraph(3)
    F.add_vertices([3,4,5])
    F.add_edges([(0,3), [1,4], [2,5]])
    return g.subgraph_search(F, induced = True) != None

def is_F_free(g):
    """
    Let F be a triangle with 3 pendants. True if g has no induced F.
    """
    return not has_F(g)

# Ronald Gould, Updating the Hamiltonian problem — a survey. Journal of Graph Theory 15.2: 121-157, 1991.
def is_oberly_sumner(g):
    """
    g is_oberly_sumner if order >= 3, is_two_connected, is_claw_free, AND is_F_free
    """
    return g.order() >= 3 and is_two_connected(g) and is_claw_free(g) and is_F_free(g)
def is_oberly_sumner_bull(g):
    """
    True if g is 2-connected, claw-free, and bull-free
    """
    return is_two_connected(g) and is_claw_free_bull_free(g)
def is_oberly_sumner_p4(g):
    """
    True if g is 2-connected, claw-free, and p4-free
    """
    return is_two_connected(g) and is_claw_free(g) and is_p4_free(g)

# Ronald Gould, Updating the Hamiltonian problem — a survey. Journal of Graph Theory 15.2: 121-157, 1991.
def is_matthews_sumner(g):
    """
    True if g is 2-connected, claw-free, and minimum-degree >= (order-1) / 3
    """
    return is_two_connected(g) and is_claw_free(g) and min_degree(g) >= (g.order() - 1) / 3
def is_broersma_veldman_gould(g):
    """
    True if g is 2-connected, claw-free, and diameter <= 2
    """
    return is_two_connected(g) and is_claw_free(g) and g.diameter() <= 2

def chvatals_condition(g):
    """
    True if g.order()>=3 and given increasing degrees d_1,..,d_n, for all i, i>=n/2 or d_i>i or d_{n-i}>=n-1

    This condition is based on Thm 1 of
    Chvátal, Václav. "On Hamilton's ideals." Journal of Combinatorial Theory, Series B 12.2 (1972): 163-168.

    [Chvatal, 72] also showed this condition is sufficient to imply g is Hamiltonian.
    """
    if g.order() < 3:
        return False
    degrees = g.degree()
    degrees.sort()
    n = g.order()
    return all(degrees[i] > i or i >= n/2 or degrees[n-i] >= n-i for i in range(0, len(degrees)))

def is_matching(g):
    """
    Returns True if this graph is the disjoint union of complete graphs of order 2.

    Tests:
        sage: is_matching(graphs.CompleteGraph(2))
        True
        sage: is_matching(graphs.PathGraph(4))
        False
        sage: is_matching(graphs.CompleteGraph(2).disjoint_union(graphs.CompleteGraph(2)))
        True
    """
    return min(g.degree())==1 and max(g.degree())==1

def has_odd_order(g):
    """
    True if the number of vertices in g is odd

    sage: has_odd_order(Graph(5))
    True
    sage: has_odd_order(Graph(2))
    False
    """
    return g.order() % 2 == 1

def has_even_order(g):
    """
    True if the number of vertices in g is even

    sage: has_even_order(Graph(5))
    False
    sage: has_even_order(Graph(2))
    True
    """
    return g.order() % 2 == 0

def is_maximal_triangle_free(g):
    """
    Evaluates whether graphs ``g`` is a maximal triangle-free graph

    Maximal triangle-free means that adding any edge to ``g`` will create a
    triangle.
    If ``g`` is not triangle-free, then returns ``False``.

    EXAMPLES:

        sage: is_maximal_triangle_free(graphs.CompleteGraph(2))
        True

        sage: is_maximal_triangle_free(graphs.CycleGraph(5))
        True

        sage: is_maximal_triangle_free(Graph('Esa?'))
        True

        sage: is_maximal_triangle_free(Graph('KsaCCA?_C?O?'))
        True

        sage: is_maximal_triangle_free(graphs.PathGraph(5))
        False

        sage: is_maximal_triangle_free(Graph('LQY]?cYE_sBOE_'))
        False

        sage: is_maximal_triangle_free(graphs.HouseGraph())
        False

    Edge cases ::

        sage: is_maximal_triangle_free(Graph(0))
        False

        sage: is_maximal_triangle_free(Graph(1))
        False

        sage: is_maximal_triangle_free(Graph(3))
        False
    """
    if not g.is_triangle_free():
        return False
    g_comp = g.complement()
    g_copy = g.copy()
    for e in g_comp.edges():
        g_copy.add_edge(e)
        if g.is_triangle_free():
            return False
        g_copy.delete_edge(e)
    return True

def is_locally_two_connected(g):
    """

    ALGORITHM:

    We modify the algorithm from our ``localise`` factory method to stop at
    subgraphs of 2 vertices, since ``is_two_connected`` is undefined on smaller
    subgraphs.
    """
    return all((f(g.subgraph(g.neighbors(v))) if len(g.neighbors(v)) >= 2
                                              else True) for v in g.vertices())

######################################################################################################################
#Below are some factory methods which create properties based on invariants or other properties

def has_equal_invariants(invar1, invar2, name=None):
    """
    This function takes two invariants as an argument and returns the property that these invariants are equal.
    Optionally a name for the new function can be provided as a third argument.
    """
    def equality_checker(g):
        return invar1(g) == invar2(g)

    if name is not None:
        equality_checker.__name__ = name
    elif hasattr(invar1, '__name__') and hasattr(invar2, '__name__'):
        equality_checker.__name__ = 'has_{}_equals_{}'.format(invar1.__name__, invar2.__name__)
    else:
        raise ValueError('Please provide a name for the new function')

    return equality_checker

"""
    sage: has_alpha_equals_clique_covering(graphs.CycleGraph(5))
    False
"""
has_alpha_equals_clique_covering = has_equal_invariants(independence_number, clique_covering_number, name="has_alpha_equals_clique_covering")


def has_invariant_equal_to(invar, value, name=None, documentation=None):
    """
    This function takes an invariant and a value as arguments and returns the property
    that the invariant value for a graph is equal to the provided value.

    Optionally a name and documentation for the new function can be provided.

    sage: order_is_five = has_invariant_equal_to(Graph.order, 5)
    sage: order_is_five(graphs.CycleGraph(5))
    True
    sage: order_is_five(graphs.CycleGraph(6))
    False
    """
    def equality_checker(g):
        return invar(g) == value

    if name is not None:
        equality_checker.__name__ = name
    elif hasattr(invar, '__name__'):
        equality_checker.__name__ = 'has_{}_equal_to_{}'.format(invar.__name__, value)
    else:
        raise ValueError('Please provide a name for the new function')

    equality_checker.__doc__ = documentation

    return equality_checker

def has_leq_invariants(invar1, invar2, name=None):
    """
    This function takes two invariants as an argument and returns the property that the first invariant is
    less than or equal to the second invariant.
    Optionally a name for the new function can be provided as a third argument.
    """
    def comparator(g):
        return invar1(g) <= invar2(g)

    if name is not None:
        comparator.__name__ = name
    elif hasattr(invar1, '__name__') and hasattr(invar2, '__name__'):
        comparator.__name__ = 'has_{}_leq_{}'.format(invar1.__name__, invar2.__name__)
    else:
        raise ValueError('Please provide a name for the new function')

    return comparator

#add all properties derived from pairs of invariants
invariant_relation_properties = [has_leq_invariants(f,g) for f in all_invariants for g in all_invariants if f != g]


def localise(f, name=None, documentation=None):
    """
    This function takes a property (i.e., a function taking only a graph as an argument) and
    returns the local variant of that property. The local variant is True if the property is
    True for the neighbourhood of each vertex and False otherwise.
    """
    #create a local version of f
    def localised_function(g):
        return all((f(g.subgraph(g.neighbors(v))) if g.neighbors(v) else True) for v in g.vertices())

    #we set a nice name for the new function
    if name is not None:
        localised_function.__name__ = name
    elif hasattr(f, '__name__'):
        if f.__name__.startswith('is_'):
            localised_function.__name__ = 'is_locally' + f.__name__[2:]
        elif f.__name__.startswith('has_'):
            localised_function.__name__ = 'has_locally' + f.__name__[2:]
        else:
            localised_function.__name__ = 'localised_' + f.__name__
    else:
        raise ValueError('Please provide a name for the new function')

    localised_function.__doc__ = documentation

    return localised_function

is_locally_dirac = localise(is_dirac)
is_locally_bipartite = localise(Graph.is_bipartite)
is_locally_planar = localise(Graph.is_planar, documentation="True if the open neighborhood of each vertex v is planar")
"""
Tests:
    sage: is_locally_unicyclic(graphs.OctahedralGraph())
    True
    sage: is_locally_unicyclic(graphs.BullGraph())
    False
"""
is_locally_unicyclic = localise(is_unicyclic, documentation="""A graph is locally unicyclic if all its local subgraphs are unicyclic.

Tests:
    sage: is_locally_unicyclic(graphs.OctahedralGraph())
    True
    sage: is_locally_unicyclic(graphs.BullGraph())
    False
""")
is_locally_connected = localise(Graph.is_connected, documentation="True if the neighborhood of every vertex is connected (stronger than claw-free)")
"""
sage: is_local_matching(graphs.CompleteGraph(3))
True
sage: is_local_matching(graphs.CompleteGraph(4))
False
sage: is_local_matching(graphs.FriendshipGraph(5))
True
"""
is_local_matching = localise(is_matching, name="is_local_matching", documentation="""True if the neighborhood of each vertex consists of independent edges.

Tests:
    sage: is_local_matching(graphs.CompleteGraph(3))
    True
    sage: is_local_matching(graphs.CompleteGraph(4))
    False
    sage: is_local_matching(graphs.FriendshipGraph(5))
    True
""")

######################################################################################################################

efficiently_computable_properties = [Graph.is_regular, Graph.is_planar,
Graph.is_forest, Graph.is_eulerian, Graph.is_connected, Graph.is_clique,
Graph.is_circular_planar, Graph.is_chordal, Graph.is_bipartite,
Graph.is_cartesian_product,Graph.is_distance_regular,  Graph.is_even_hole_free,
Graph.is_gallai_tree, Graph.is_line_graph, Graph.is_overfull, Graph.is_perfect,
Graph.is_split, Graph.is_strongly_regular, Graph.is_triangle_free,
Graph.is_weakly_chordal, is_dirac, is_ore,
is_generalized_dirac, is_van_den_heuvel, is_two_connected, is_three_connected,
is_lindquester, is_claw_free, Graph.has_perfect_matching, has_radius_equal_diameter,
is_not_forest, is_genghua_fan, is_cubic, diameter_equals_twice_radius,
is_locally_connected, matching_covered, is_locally_dirac,
is_locally_bipartite, is_locally_two_connected, Graph.is_interval, has_paw,
is_paw_free, has_p4, is_p4_free, has_dart, is_dart_free, has_kite, is_kite_free,
has_H, is_H_free, has_residue_equals_two, order_leq_twice_max_degree,
alpha_leq_order_over_two, is_factor_critical, is_independence_irreducible,
has_twin, is_twin_free, diameter_equals_two, girth_greater_than_2log, Graph.is_cycle,
pairs_have_unique_common_neighbor, has_star_center, is_complement_of_chordal,
has_c4, is_c4_free, is_subcubic, is_quasi_regular, is_bad, has_k4, is_k4_free,
is_distance_transitive, is_unicyclic, is_locally_unicyclic, has_simplical_vertex,
has_exactly_two_simplical_vertices, is_two_tree, is_locally_planar,
is_four_connected, is_claw_free_paw_free, has_bull, is_bull_free,
is_claw_free_bull_free, has_F, is_F_free, is_oberly_sumner, is_oberly_sumner_bull,
is_oberly_sumner_p4, is_matthews_sumner, chvatals_condition, is_matching, is_local_matching,
has_odd_order, has_even_order, Graph.is_circulant, Graph.has_loops,
Graph.is_asteroidal_triple_free, Graph.is_block_graph, Graph.is_cactus,
Graph.is_cograph, Graph.is_long_antihole_free, Graph.is_long_hole_free, Graph.is_partial_cube,
Graph.is_polyhedral, Graph.is_prime, Graph.is_tree, Graph.is_apex, Graph.is_arc_transitive,
Graph.is_self_complementary, is_double_clique, has_fork, is_fork_free,
has_empty_KE_part]

intractable_properties = [Graph.is_hamiltonian, Graph.is_vertex_transitive,
Graph.is_edge_transitive, has_residue_equals_alpha, Graph.is_odd_hole_free,
Graph.is_semi_symmetric, is_planar_transitive, is_class1,
is_class2, is_anti_tutte, is_anti_tutte2, has_lovasz_theta_equals_cc,
has_lovasz_theta_equals_alpha, is_chvatal_erdos, is_heliotropic_plant,
is_geotropic_plant, is_traceable, is_chordal_or_not_perfect,
has_alpha_residue_equal_two, is_complement_hamiltonian, is_1_tough, is_2_tough,
has_two_ham_cycles, is_two_path, is_prism_hamiltonian, is_bauer, is_jung,
is_weakly_pancyclic, is_pancyclic, has_two_walk, has_alpha_equals_clique_covering,
Graph.is_transitively_reduced, Graph.is_half_transitive, Graph.is_line_graph,
is_haggkvist_nicoghossian, is_chromatic_index_critical]

removed_properties = [is_pebbling_class0]

"""
    Last version of graphs packaged checked: Sage 8.2
    This means checked for new functions, and for any errors/changes in old functions!
    sage: sage.misc.banner.version_dict()['major'] < 8 or (sage.misc.banner.version_dict()['major'] == 8 and sage.misc.banner.version_dict()['minor'] <= 2)
    True

    Skip Graph.is_circumscribable() and Graph.is_inscribable() because they
        throw errors for the vast majority of our graphs.
    Skip Graph.is_biconnected() in favor of our is_two_connected(), because we
        prefer our name, and because we disagree with their definition on K2.
        We define that K2 is NOT 2-connected, it is n-1 = 1 connected.
    Implementation of Graph.is_line_graph() is intractable, despite a theoretically efficient algorithm existing.
"""
sage_properties = [Graph.is_hamiltonian, Graph.is_eulerian, Graph.is_planar,
Graph.is_circular_planar, Graph.is_regular, Graph.is_chordal, Graph.is_circulant,
Graph.is_interval, Graph.is_gallai_tree, Graph.is_clique, Graph.is_cycle,
Graph.is_transitively_reduced, Graph.is_self_complementary, Graph.is_connected,
Graph.has_loops, Graph.is_asteroidal_triple_free, Graph.is_bipartite,
Graph.is_block_graph, Graph.is_cactus, Graph.is_cartesian_product,
Graph.is_cograph, Graph.is_distance_regular, Graph.is_edge_transitive, Graph.is_even_hole_free,
Graph.is_forest, Graph.is_half_transitive, Graph.is_line_graph,
Graph.is_long_antihole_free, Graph.is_long_hole_free, Graph.is_odd_hole_free,
Graph.is_overfull, Graph.is_partial_cube, Graph.is_polyhedral, Graph.is_prime,
Graph.is_semi_symmetric, Graph.is_split, Graph.is_strongly_regular, Graph.is_tree,
Graph.is_triangle_free, Graph.is_weakly_chordal, Graph.has_perfect_matching, Graph.is_apex,
Graph.is_arc_transitive]

#speed notes
#FAST ENOUGH (tested for graphs on 140921): is_hamiltonian, is_vertex_transitive,
#    is_edge_transitive, has_residue_equals_alpha, is_odd_hole_free, is_semi_symmetric,
#    is_line_graph, is_line_graph, is_anti_tutte, is_planar_transitive
#SLOW but FIXED for SpecialGraphs: is_class1, is_class2

properties = efficiently_computable_properties + intractable_properties
properties_plus = efficiently_computable_properties + intractable_properties + invariant_relation_properties


invariants_from_properties = [make_invariant_from_property(property) for property in properties]
invariants_plus = all_invariants + invariants_from_properties

# weakly_chordal = weakly chordal, i.e., the graph and its complement have no induced cycle of length at least 5


#############################################################################
# End of properties section                                                 #
#############################################################################