"""
A service queue that attempts to compute property and invariant values for polled graphs
1) Polls the API for a graph that has not yet been processed
2) Computes all properties and invariants of the graph with a cut-off computation time.
3) Inserts graph back in
"""
from sage.all import *
from sage.graphs.graph_input import from_graph6
from properties_and_invariants import properties_plus, invariants_plus
import time, requests, json, timeout_decorator



@timeout_decorator.timeout(60, use_signals=False)
def compute_value(callback, graph):
    """
    Set 60 seconds timeout for computations, don't use signals so this function can be multi-processed
    """
    return callback(graph)



if __name__ == '__main__':
    r = requests.post("http://206.189.196.27/api/graph/poll", data={'reset_queue':True}) #reset queue of graphs in need of computation
    if r.status_code != 200:
        raise RuntimeError('Could not reset queue on start up')

    while True:
        time.sleep(.1)
        r = requests.get("http://206.189.196.27/api/graph/poll")
        if r.status_code != 200:
            continue
        graph_entity = json.loads(r.content)

        #retrieve the polled graph
        r = requests.post("http://206.189.196.27/api/graph/", data={'graph6': graph_entity['id']})
        property_request = requests.get("http://206.189.196.27/api/property/")
        invariant_request = requests.get("http://206.189.196.27/api/invariant/")

        database_properties = json.loads(property_request.content)['properties']
        database_invariants = json.loads(invariant_request.content)['invariants']
        computed_properties = json.loads(r.content)['properties'].keys()
        computed_invariants = json.loads(r.content)['invariants'].keys()

        needed_properties = set(database_properties).difference(set(computed_properties))
        needed_invariants = set(database_invariants).difference(set(computed_invariants))

        # if not needed_properties and not needed_invariants: # no computation needs to be done
        #     continue

        computation_results = {'id': int(graph_entity['id'])}
        computation_results['properties'] = []
        computation_results['invariants'] = []

        graph = Graph()
        from_graph6(graph, str(graph_entity['graph6']))


        for property_func in properties_plus:
            if property_func.__name__ in needed_properties:
                try:
                    computation_results['properties'].append( (property_func.__name__, str(compute_value(property_func, graph))) )
                except BaseException:
                    continue
        for invariant_func in invariants_plus:
            if invariant_func.__name__ in needed_invariants:
                try:
                    computation_results['invariants'].append((invariant_func.__name__, str(compute_value(invariant_func, graph))))
                except BaseException:
                    continue
        try:
            r = requests.put('http://206.189.196.27/api/graph/poll', data=json.dumps(computation_results))
        except BaseException:
            continue #TODO possible would be better to mark as un-computed in DB
        print(r.content)




