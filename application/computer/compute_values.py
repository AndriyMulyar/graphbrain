"""
A service queue that attempts to compute property and invariant values for polled graphs
1) Polls the API for a graph that has not yet been processed
2) Computes all properties and invariants of the graph with a cut-off computation time.
3) Inserts graph back in
"""
from sage.all import *
from properties_and_invariants import properties_plus, invariants_plus
import time, requests, json






if __name__ == '__main__':
    r = requests.post("http://api:8000/api/graph/", data={'reset_queue':True}) #reset queue of graphs in need of computation
    if r.status_code != 200:
        raise RuntimeError('Could not reset queue on start up')

    while True:
        time.sleep(5)
        r = requests.get("http://api:8000/api/graph/")
        if r.status_code != 200:
            continue
        graph = json.loads(r.content)

        #retrieve the polled graph
        r = requests.get("http://api:8000/api/graph/"+str(graph['id']))
        property_request = requests.get("http://api:8000/api/property/")
        invariant_request = requests.get("http://api:8000/api/invariant/")

        database_properties = json.loads(property_request.content)['properties']
        database_invariants = json.loads(invariant_request.content)['invariants']
        computed_properties = json.loads(r.content)['properties'].keys()
        computed_invariants = json.loads(r.content)['invariants'].keys()

        needed_properties = set(database_properties).difference(set(computed_properties))
        needed_invariants = set(database_invariants).difference(set(computed_invariants))



        print(needed_properties)



